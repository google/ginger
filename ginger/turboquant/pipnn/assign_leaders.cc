// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Fused GEMM + top-K leader assignment for assigning points to their
// nearest leaders in a pre-transposed dataset. Specialized to K=1, 3, 10 for
// the partitioning step in PiPNN.
//
// Leverages RunQueryLoop from gemm_loop-inl.h to drive the GEMM loop.
// Decodes queries on-the-fly and tracks top-K scores in registers (small K) or
// stack-allocated buffers.

#include "ginger/turboquant/pipnn/assign_leaders.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "ginger/turboquant/pipnn/topk_result.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/pipnn/assign_leaders.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/decode_queries-inl.h"
#include "ginger/turboquant/many_to_many/gemm_loop-inl.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/pipnn/topk_tracker-inl.h"
#include "ginger/turboquant/simd/vec_group-inl.h"

namespace ginger {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

static constexpr size_t kQueriesPerBatch = 6;

// Processes queries in micro-batches (kQueriesPerBatch=6) with fused top-K
// tracking. Uses `if constexpr` on K to select the state representation:
//   K == 1 or K == 3: in-register (unrolled via VEC_GROUP).
//   K == 10:          spill to memory.
template <size_t K>
std::vector<FixedTopKResult<K>> RunAssignLeadersLoop(
    const TurboQuantDatasetSoA& qp, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& db) {
  static_assert(K == 1 || K == 3 || K == 10);
  using I32 = hn::ScalableTag<int32_t>;
  using F32 = hn::ScalableTag<float>;
  const I32 di;
  const F32 df;
  const size_t NF = hn::Lanes(df);
  const size_t num_points = db.num_points;
  const bool is_l2 =
      qp.metric() == TurboQuantDistanceMetric::kSquaredL2Distance;
  const size_t decoded_stride = DecodedQueryStride(qp);
  const hn::Vec<F32> inf_v = hn::Set(df, std::numeric_limits<float>::max());
  const size_t num_queries = query_subset.size();

  // Register-resident top-K state.
  // top-3 spread across 3 different registers. tracked per lane.
  VEC_GROUP_DEFINE(df, scores0);
  VEC_GROUP_DEFINE(di, indices0);
  VEC_GROUP_DEFINE(df, scores1);
  VEC_GROUP_DEFINE(di, indices1);
  VEC_GROUP_DEFINE(df, scores2);
  VEC_GROUP_DEFINE(di, indices2);

  // Buffer-based top-K state (only materialized when K > 3).
  // Also tracked per lane.
  constexpr size_t kBufSize =
      (K > 3) ? kQueriesPerBatch * K * HWY_MAX_BYTES / sizeof(float) : 1;
  HWY_ALIGN float top_score_buf[kBufSize];
  HWY_ALIGN int32_t top_idx_buf[kBufSize];

  std::vector<FixedTopKResult<K>> results(num_queries);

  // Small reuse buffers for kQueriesPerBatch queries.
  hwy::AlignedVector<int8_t> decoded_buf(kQueriesPerBatch * decoded_stride);
  float norms_buf[kQueriesPerBatch];
  float sqr_norms_buf[kQueriesPerBatch];
  int32_t byte_sums_buf[kQueriesPerBatch];

  auto panel_ptr = [panel_data = db.panel_data.data(),
                    panel_bytes = db.panel_bytes](size_t p) -> const uint8_t* {
    return panel_data + p * panel_bytes;
  };

  RunQueryLoop<kQueriesPerBatch>(
      num_queries, is_l2, db.num_panels, db.total_tiles,
      db.norm_scaling_factors.data(), is_l2 ? db.sqr_norms.data() : nullptr,
      /*lower_triangle=*/false, panel_ptr,
      // init_batch_fn
      [&](size_t q_start, size_t num_active, QueryBatch<kQueriesPerBatch>& qb) {
        const DecodedQueryView dq_view = DecodeQueriesAndMetadata(
            qp, query_subset.subspan(q_start, num_active),
            absl::MakeSpan(decoded_buf),
            absl::MakeSpan(byte_sums_buf, num_active),
            absl::MakeSpan(norms_buf, num_active),
            absl::MakeSpan(sqr_norms_buf, num_active));
        internal::PrepareQueryBatch(dq_view, 0, num_active, qb);

        if constexpr (K == 1 || K == 3) {
          ForIndex<kQueriesPerBatch>([&](auto q) HWY_ATTR {
            VEC_GROUP_ACCESSOR(scores0)(q) = inf_v;
            VEC_GROUP_ACCESSOR(indices0)(q) = hn::Set(di, -1);
            if constexpr (K >= 3) {
              VEC_GROUP_ACCESSOR(scores1)(q) = inf_v;
              VEC_GROUP_ACCESSOR(indices1)(q) = hn::Set(di, -1);
              VEC_GROUP_ACCESSOR(scores2)(q) = inf_v;
              VEC_GROUP_ACCESSOR(indices2)(q) = hn::Set(di, -1);
            }
          });
        } else {
          std::fill_n(top_score_buf, kQueriesPerBatch * K * NF,
                      std::numeric_limits<float>::max());
          std::fill_n(top_idx_buf, kQueriesPerBatch * K * NF, int32_t{-1});
        }
      },
      // score_consumer_fn
      [&](auto scores_accessor, size_t /*q_start*/, size_t num_active,
          size_t panel_base) HWY_ATTR {
        ForIndex<kQueriesPerBatch>([&](auto qi_v) HWY_ATTR {
          constexpr size_t q = decltype(qi_v)::value;
          if (q >= num_active) return;
          hn::Vec<F32> score = scores_accessor(qi_v);
          if constexpr (K == 1) {
            UpdateTopK1InRegisters(score, panel_base, num_points,
                                   VEC_GROUP_ACCESSOR(scores0)(qi_v),
                                   VEC_GROUP_ACCESSOR(indices0)(qi_v));
          } else if constexpr (K == 3) {
            UpdateTopK3InRegisters(score, panel_base, num_points,
                                   VEC_GROUP_ACCESSOR(scores0)(qi_v),
                                   VEC_GROUP_ACCESSOR(scores1)(qi_v),
                                   VEC_GROUP_ACCESSOR(scores2)(qi_v),
                                   VEC_GROUP_ACCESSOR(indices0)(qi_v),
                                   VEC_GROUP_ACCESSOR(indices1)(qi_v),
                                   VEC_GROUP_ACCESSOR(indices2)(qi_v));
          } else {
            UpdateTopK<K>(score, panel_base, num_points,
                          top_score_buf + q * K * NF, top_idx_buf + q * K * NF);
          }
        });
      },
      // reduce_batch_fn
      [&](size_t q_start, const QueryBatch<kQueriesPerBatch>& qb) {
        ForIndex<kQueriesPerBatch>([&](auto qi_v) HWY_ATTR {
          constexpr size_t q = decltype(qi_v)::value;
          if (q >= qb.num_active_queries) return;
          if constexpr (K == 1) {
            HorizontalReduceTopK1FromRegisters(
                VEC_GROUP_ACCESSOR(scores0)(qi_v),
                VEC_GROUP_ACCESSOR(indices0)(qi_v), results[q_start + q]);
          } else if constexpr (K == 3) {
            HorizontalReduceTopK3FromRegisters(
                VEC_GROUP_ACCESSOR(scores0)(qi_v),
                VEC_GROUP_ACCESSOR(scores1)(qi_v),
                VEC_GROUP_ACCESSOR(scores2)(qi_v),
                VEC_GROUP_ACCESSOR(indices0)(qi_v),
                VEC_GROUP_ACCESSOR(indices1)(qi_v),
                VEC_GROUP_ACCESSOR(indices2)(qi_v), results[q_start + q]);
          } else {
            ReduceTopK<K>(top_score_buf + q * K * NF, top_idx_buf + q * K * NF,
                          results[q_start + q]);
          }
        });
      });
  return results;
}

// Entry-point implementations (decode + dispatch)
std::vector<FixedTopKResult<1>> TopK1Impl(
    const TurboQuantDatasetSoA& qp, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& db) {
  return RunAssignLeadersLoop<1>(qp, query_subset, db);
}

std::vector<FixedTopKResult<3>> TopK3Impl(
    const TurboQuantDatasetSoA& qp, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& db) {
  return RunAssignLeadersLoop<3>(qp, query_subset, db);
}

std::vector<FixedTopKResult<10>> TopK10Impl(
    const TurboQuantDatasetSoA& qp, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& db) {
  return RunAssignLeadersLoop<10>(qp, query_subset, db);
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace ginger {

HWY_EXPORT(TopK1Impl);
HWY_EXPORT(TopK3Impl);
HWY_EXPORT(TopK10Impl);

std::vector<FixedTopKResult<1>> AssignToLeadersTopK1(
    const TurboQuantDatasetSoA& quantized_points, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& leaders) {
  return HWY_DYNAMIC_DISPATCH(TopK1Impl)(quantized_points, query_subset,
                                         leaders);
}

std::vector<FixedTopKResult<3>> AssignToLeadersTopK3(
    const TurboQuantDatasetSoA& quantized_points, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& leaders) {
  return HWY_DYNAMIC_DISPATCH(TopK3Impl)(quantized_points, query_subset,
                                         leaders);
}

std::vector<FixedTopKResult<10>> AssignToLeadersTopK10(
    const TurboQuantDatasetSoA& quantized_points, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& leaders) {
  return HWY_DYNAMIC_DISPATCH(TopK10Impl)(quantized_points, query_subset,
                                          leaders);
}

}  // namespace ginger

#endif  // HWY_ONCE
