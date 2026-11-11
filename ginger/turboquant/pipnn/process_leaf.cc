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

// Fused GEMM + symmetric Top-2 nearest-neighbor selection for a single
// partition leaf.  All N points in the leaf act as both queries and database
// points; the lower-triangle optimization computes only N*(N-1)/2 distances.
// Notably, we do not perform cache blocking since the leaf size is small.
//
// Outline:
//   1. Decode & transpose: decode quantized vectors, transpose into
//      SIMD-panel layout (tiles of 4 bytes × points_per_panel lanes).
//   2. Initialize workspace: pad norms/sqr_norms, allocate SOA Top-2
//      trackers for both query-side (register-resident) and db-side (memory).
//   3. Outer loop over query batches (kMq queries at a time):
//      a. Load query pointers, norms, byte-sum corrections.
//      b. Inner loop over panels (lower triangle: panel_base >= query_start):
//         i.   GEMM: MicroKernelAccumulate produces kMq int32 accumulators.
//         ii.  ScoreAndScatter: resolve accumulators → float scores, mask
//              self-distances / OOB / diagonal-skip lanes to +inf, then
//              update *both* query-side and db-side Top-2 trackers with the
//              same score (symmetric scatter — each pair computed once).
//         iii. Spill db-side trackers back to memory.
//      c. Cross-lane reduce query-side SIMD trackers to scalar Top2Results.
//   4. Merge db-side SOA trackers into the final per-point Top2Results.

#include "ginger/turboquant/pipnn/process_leaf.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/query/query.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/pipnn/process_leaf.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/accumulate-inl.h"
#include "ginger/turboquant/many_to_many/decode_db-inl.h"
#include "ginger/turboquant/many_to_many/decode_queries-inl.h"
#include "ginger/turboquant/many_to_many/gemm_loop-inl.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/pipnn/scatter_epilogue-inl.h"

namespace ginger {

namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

// Allocate and zero-initialize workspace buffers.  Padding beyond
// num_points uses inert values: norm=0 (contributes nothing to scores),
// sqr_norm=+inf (never selected as a neighbor), tracker scores=+inf.
void InitializeWorkspace(const TurboQuantDatasetSoA& points,
                         absl::Span<const int> indices,
                         ProcessLeafWorkspace* workspace) {
  using I32 = hn::ScalableTag<int32_t>;
  const I32 di32;
  const size_t points_per_panel = hn::Lanes(di32);
  const size_t num_points = indices.size();
  const bool is_l2 =
      points.metric() == TurboQuantDistanceMetric::kSquaredL2Distance;
  const size_t num_panels = hwy::DivCeil(num_points, points_per_panel);
  const size_t padded_num_points = num_panels * points_per_panel;  // NOMUTANTS

  // Pad to full-panel width so SIMD panel loads need no bounds check.
  workspace->norms.assign(padded_num_points, 0.0f);
  workspace->sqr_norms.assign(padded_num_points,
                              std::numeric_limits<float>::max());
  for (size_t i = 0; i < num_points; ++i) {
    workspace->norms[i] = points.norm_scaling_factor(indices[i]);
    if (is_l2) {
      workspace->sqr_norms[i] = points.unquantized_squared_norm(indices[i]);
    }
  }

  // DB-side SOA Top-2 trackers (Load per panel, update across kMq queries,
  // Store back).
  workspace->top2_results.assign(num_points, Top2Result{});

  workspace->db_min1_s.assign(padded_num_points,
                              std::numeric_limits<float>::max());
  workspace->db_min2_s.assign(padded_num_points,
                              std::numeric_limits<float>::max());
  workspace->db_min1_idx_s.assign(padded_num_points, int32_t{-1});
  workspace->db_min2_idx_s.assign(padded_num_points, int32_t{-1});
}

}  // namespace

void ProcessLeafTop2Impl(const TurboQuantDatasetSoA& points,
                         absl::Span<const int> indices,
                         ProcessLeafWorkspace* workspace) {
  constexpr size_t kQueriesPerBatch = 6;
  using I32 = hn::ScalableTag<int32_t>;
  using F32 = hn::ScalableTag<float>;
  const I32 di32;
  const F32 df;
  const hn::ScalableTag<int8_t> di8;

  const size_t N = hn::Lanes(di8);
  const size_t points_per_panel = hn::Lanes(di32);
  const size_t num_points = indices.size();
  const size_t decoded_stride = DecodedQueryStride(points);
  const size_t bytes_per_point = points.num_bytes_per_datapoint();
  const size_t num_chunks = hwy::DivCeil(bytes_per_point, N);
  const size_t tiles = num_chunks * (N / 2);
  const bool is_l2 =
      points.metric() == TurboQuantDistanceMetric::kSquaredL2Distance;

  // Step 1: Decode database vectors into SIMD-friendly panel layout using
  // DecodeDbPanel.
  const size_t num_panels = hwy::DivCeil(num_points, points_per_panel);
  const size_t panel_bytes = DbPanelBytes(points);
  workspace->panel_data.resize(num_panels * panel_bytes);
  const IndexSubset index_subset = IndexSubset::Gathered(indices);
  for (size_t p = 0; p < num_panels; ++p) {
    const size_t base = p * points_per_panel;
    const size_t count =
        std::min(points_per_panel, num_points - base);  // NOMUTANTS
    DecodeDbPanel(points, index_subset.subspan(base, count),
                  absl::MakeSpan(workspace->panel_data.data() + p * panel_bytes,
                                 panel_bytes));
  }

  InitializeWorkspace(points, indices, workspace);

  // lane_idx = [0, 1, ..., Lanes-1]: reused for index construction and masking.
  const hn::Vec<I32> lane_idx = hn::Iota(di32, 0);
  const hn::Vec<F32> inf_v = hn::Set(df, std::numeric_limits<float>::max());

  // Small reuse buffers for kMq queries.
  workspace->decoded_buf.resize(kQueriesPerBatch * decoded_stride);
  float norms_buf[kQueriesPerBatch];
  float sqr_norms_buf[kQueriesPerBatch];
  int32_t byte_sums_buf[kQueriesPerBatch];

  auto panel_ptr = [panel_data = workspace->panel_data.data(),
                    panel_bytes](size_t p) -> const uint8_t* {
    return panel_data + p * panel_bytes;
  };

  // Query-side SIMD Top-2 trackers — kept in registers across all panels.
  VEC_GROUP_DEFINE(df, min1_v);
  VEC_GROUP_DEFINE(df, min2_v);
  VEC_GROUP_DEFINE(di32, min1_idx);
  VEC_GROUP_DEFINE(di32, min2_idx);

  // Step 3: Outer loop — process kMq queries per iteration using RunQueryLoop.
  // Lower-triangle: only visit panels whose base >= query_start.
  RunQueryLoop<kQueriesPerBatch>(
      num_points, is_l2, num_panels, tiles, workspace->norms.data(),
      is_l2 ? workspace->sqr_norms.data() : nullptr, /*lower_triangle=*/true,
      panel_ptr,
      // init_batch: decode kMq queries into reuse buffers and reset top-2
      // state.
      [&](size_t query_start, size_t num_active,
          QueryBatch<kQueriesPerBatch>& qb) {
        const DecodedQueryView dq_view = DecodeQueriesAndMetadata(
            points, index_subset.subspan(query_start, num_active),
            absl::MakeSpan(workspace->decoded_buf),
            absl::MakeSpan(byte_sums_buf, num_active),
            absl::MakeSpan(norms_buf, num_active),
            absl::MakeSpan(sqr_norms_buf, num_active));
        internal::PrepareQueryBatch(dq_view, 0, num_active, qb);

        ForIndex<kQueriesPerBatch>([&](auto q_idx) {
          VEC_GROUP_ACCESSOR(min1_v)(q_idx) = inf_v;
          VEC_GROUP_ACCESSOR(min2_v)(q_idx) = inf_v;
          VEC_GROUP_ACCESSOR(min1_idx)(q_idx) = hn::Set(di32, -1);
          VEC_GROUP_ACCESSOR(min2_idx)(q_idx) = hn::Set(di32, -1);
        });
      },
      // score_consumer: scatter score to query-side and db-side trackers.
      [&](auto scores_accessor, size_t query_start, size_t num_active,
          size_t panel_base) HWY_ATTR {
        hn::Vec<F32> db_min1_scores =
            hn::Load(df, workspace->db_min1_s.data() + panel_base);
        hn::Vec<F32> db_min2_scores =
            hn::Load(df, workspace->db_min2_s.data() + panel_base);
        hn::Vec<I32> db_min1_indices =
            hn::Load(di32, workspace->db_min1_idx_s.data() + panel_base);
        hn::Vec<I32> db_min2_indices =
            hn::Load(di32, workspace->db_min2_idx_s.data() + panel_base);

        ScatterBatch<kQueriesPerBatch>(
            df, di32, scores_accessor, query_start, num_active, panel_base,
            num_points, lane_idx, inf_v, VEC_GROUP_ACCESSOR(min1_v),
            VEC_GROUP_ACCESSOR(min2_v), VEC_GROUP_ACCESSOR(min1_idx),
            VEC_GROUP_ACCESSOR(min2_idx), db_min1_scores, db_min2_scores,
            db_min1_indices, db_min2_indices);

        hn::Store(db_min1_scores, df, workspace->db_min1_s.data() + panel_base);
        hn::Store(db_min2_scores, df, workspace->db_min2_s.data() + panel_base);
        hn::Store(db_min1_indices, di32,
                  workspace->db_min1_idx_s.data() + panel_base);
        hn::Store(db_min2_indices, di32,
                  workspace->db_min2_idx_s.data() + panel_base);
      },
      // reduce_batch: cross-lane reduce query-side SIMD trackers → scalar.
      [&](size_t query_start, const QueryBatch<kQueriesPerBatch>& qb) {
        ForIndex<kQueriesPerBatch>([&](auto q_idx) {
          constexpr size_t q = decltype(q_idx)::value;
          if (q < qb.num_active_queries) {
            const size_t query_idx = query_start + q;
            auto& r = workspace->top2_results[query_idx];
            ReduceTop2ToScalar(df, di32, VEC_GROUP_ACCESSOR(min1_v)(q_idx),
                               VEC_GROUP_ACCESSOR(min2_v)(q_idx),
                               VEC_GROUP_ACCESSOR(min1_idx)(q_idx),
                               VEC_GROUP_ACCESSOR(min2_idx)(q_idx), r);
          }
        });
      });

  // Step 4: Merge db-side SOA trackers into final per-point Top2Results.
  for (size_t i = 0; i < num_points; ++i) {
    auto& r = workspace->top2_results[i];
    if (workspace->db_min1_s[i] < std::numeric_limits<float>::max()) {
      r.Update(workspace->db_min1_s[i], workspace->db_min1_idx_s[i]);
    }
    if (workspace->db_min2_s[i] < std::numeric_limits<float>::max()) {
      r.Update(workspace->db_min2_s[i], workspace->db_min2_idx_s[i]);
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace ginger

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace ginger {

HWY_EXPORT(ProcessLeafTop2Impl);

// For very small leaves the full ManyToMany machinery doesn't work.
// For each point i, create a query from its quantized data and call
// UnsafeComputeDistances to score it against all other points.
void ProcessLeafSmallN(const TurboQuantCodec& quantizer,
                       const TurboQuantDatasetSoA& points,
                       absl::Span<const int> indices,
                       ProcessLeafWorkspace* workspace) {
  const size_t n = indices.size();
  workspace->top2_results.assign(n, Top2Result{});
  if (n <= 1) {
    return;
  }

  // Collect the other-point indices for each query.
  std::vector<int32_t> other_indices;
  other_indices.reserve(n - 1);
  std::vector<float> scores(n - 1);  // NOMUTANTS

  for (size_t i = 0; i < n; ++i) {
    auto query_or = TurboQuantQuery::CreateFromQuantizedDataPoint(
        quantizer, points.DataPointView(indices[i]));
    ABSL_CHECK_OK(query_or);

    // Score against all other points in the leaf.
    other_indices.clear();
    for (size_t j = 0; j < n; ++j) {
      if (j != i) {
        other_indices.push_back(indices[j]);
      }
    }
    scores.resize(other_indices.size());
    points.UnsafeComputeDistances(
        *query_or, CreateScoringAccessor(absl::MakeSpan(other_indices),
                                         absl::MakeSpan(scores)));

    for (size_t k = 0; k < other_indices.size(); ++k) {
      const int32_t local_idx = (k < i) ? k : k + 1;
      workspace->top2_results[i].Update(scores[k], local_idx);
    }
  }
}

void ProcessLeafTop2(const TurboQuantCodec& quantizer,
                     const TurboQuantDatasetSoA& quantized_points,
                     absl::Span<const int> indices,
                     ProcessLeafWorkspace& workspace) {
  // For very small leaves use the scalar fallback — the SIMD lower-triangle
  // loop requires at least one full kMq batch and one full panel, which is
  // wasteful for n <= 3 (at most 3 pairs to compute).
  if (indices.size() <= 3) {
    ProcessLeafSmallN(quantizer, quantized_points, indices, &workspace);
    return;
  }
  HWY_DYNAMIC_DISPATCH(ProcessLeafTop2Impl)(quantized_points, indices,
                                            &workspace);
}

}  // namespace ginger

#endif  // HWY_ONCE
