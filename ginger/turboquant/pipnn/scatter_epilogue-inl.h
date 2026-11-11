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

// Symmetric scatter epilogue for ProcessLeaf.
//
// After each GEMM panel, run this epilogue:
//   1. Masks self-distances, OOB padding, and lower-triangle skip lanes.
//   2. Updates both query-side and db-side SIMD Top-2 trackers with the
//      same score (symmetric scatter — halves GEMM work).
//
// Each SIMD lane independently tracks its own (min1, min2) pair. After all
// panels, a scalar cross-lane reduction merges per-lane results.
//
// Include after foreach_target.h, highway.h. Toggle guard handles
// per-target re-inclusion.

#ifndef GINGER_TURBOQUANT_PIPNN_SCATTER_EPILOGUE_INL_H_
#define GINGER_TURBOQUANT_PIPNN_SCATTER_EPILOGUE_INL_H_
#endif  // GINGER_TURBOQUANT_PIPNN_SCATTER_EPILOGUE_INL_H_

// clang-format off
#if defined(GINGER_TURBOQUANT_PIPNN_SCATTER_EPILOGUE_TOGGLE) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef GINGER_TURBOQUANT_PIPNN_SCATTER_EPILOGUE_TOGGLE
#undef GINGER_TURBOQUANT_PIPNN_SCATTER_EPILOGUE_TOGGLE
#else
#define GINGER_TURBOQUANT_PIPNN_SCATTER_EPILOGUE_TOGGLE
#endif
// clang-format on

#include <cstddef>
#include <cstdint>
#include <limits>

#include "ginger/turboquant/pipnn/process_leaf.h"
#include "ginger/turboquant/simd/vec_group-inl.h"
#include "hwy/highway.h"

namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Masks invalid lanes to +inf: self-distance (query in this panel), OOB
// padding lanes, and lower-triangle skip lanes. Scalar branch checks avoid
// unnecessary mask construction when the condition is false for the panel.
template <class Float, class Int32>
HWY_INLINE void MaskInvalidLanes(Float df, Int32 di32, hn::Vec<Float>& scores,
                                 size_t query_index, size_t base,
                                 size_t num_points,
                                 const hn::Vec<Int32>& base_indices,
                                 const hn::Vec<Float>& inf_v) {
  const size_t points_per_panel = hn::Lanes(di32);
  if (base <= query_index || base + points_per_panel > num_points) {
    // Lower-triangle & self-distance: mask all database points d <=
    // query_index.
    if (base <= query_index) {
      const hn::Mask<Float> d_le_q = hn::RebindMask(
          df, hn::Le(base_indices,
                     hn::Set(di32, static_cast<int32_t>(query_index))));
      scores = hn::IfThenElse(d_le_q, inf_v, scores);
    }
    // Out-of-bounds: mask trailing padding lanes beyond num_points.
    if (base + points_per_panel > num_points) {
      const hn::Mask<Float> oob = hn::RebindMask(
          df, hn::Not(hn::Lt(base_indices,
                             hn::Set(di32, static_cast<int32_t>(num_points)))));
      scores = hn::IfThenElse(oob, inf_v, scores);
    }
  }
}

// Branchless per-lane Top-2 insertion. Each lane independently maintains
// (min1, min2) + indices. Uses a two-mask cascade:
//   lt1 = (score < min1)       — new global best
//   lt2 = !lt1 && (score < min2) — exclusively second-best
// Updates min2 before min1 so the old min1 can be demoted into min2.
// RebindMask converts float masks to int32 for index updates.
template <class Float, class Int32>
HWY_INLINE void UpdateTop2Trackers(Float /*df*/, Int32 di32,
                                   hn::Vec<Float> scores,
                                   const hn::Vec<Int32>& indices,
                                   hn::Vec<Float>& min1_v,
                                   hn::Vec<Float>& min2_v,
                                   hn::Vec<Int32>& min1_idx_v,
                                   hn::Vec<Int32>& min2_idx_v) {
  const hn::Mask<Float> lt1 = hn::Lt(scores, min1_v);
  const hn::Mask<Float> lt2 = hn::AndNot(lt1, hn::Lt(scores, min2_v));
  const hn::Mask<Int32> lt1_i = hn::RebindMask(di32, lt1);
  const hn::Mask<Int32> lt2_i = hn::RebindMask(di32, lt2);
  min2_v = hn::IfThenElse(lt1, min1_v, hn::IfThenElse(lt2, scores, min2_v));
  min2_idx_v = hn::IfThenElse(lt1_i, min1_idx_v,
                              hn::IfThenElse(lt2_i, indices, min2_idx_v));
  min1_v = hn::IfThenElse(lt1, scores, min1_v);
  min1_idx_v = hn::IfThenElse(lt1_i, indices, min1_idx_v);
}

// Symmetric scatter. Masks invalids, then feeds the *same* score into:
//   - query-side tracker (indices = base + lane_idx, i.e. db-point IDs)
//   - db-side tracker   (index  = broadcast query_index)
// This dual update is the symmetric scatter trick.
template <class Float, class Int32>
HWY_INLINE void ScatterSingleQuery(
    Float df, Int32 di32, hn::Vec<Float> score, size_t query_index, size_t base,
    size_t num_points, const hn::Vec<Int32>& base_indices,
    const hn::Vec<Float>& inf_v, hn::Vec<Float>& q_min1, hn::Vec<Float>& q_min2,
    hn::Vec<Int32>& q_min1_idx, hn::Vec<Int32>& q_min2_idx,
    hn::Vec<Float>& db_min1, hn::Vec<Float>& db_min2,
    hn::Vec<Int32>& db_min1_idx, hn::Vec<Int32>& db_min2_idx) {
  MaskInvalidLanes(df, di32, score, query_index, base, num_points, base_indices,
                   inf_v);

  UpdateTop2Trackers(df, di32, score, base_indices, q_min1, q_min2, q_min1_idx,
                     q_min2_idx);

  const hn::Vec<Int32> qi_v = hn::Set(di32, static_cast<int32_t>(query_index));
  UpdateTop2Trackers(df, di32, score, qi_v, db_min1, db_min2, db_min1_idx,
                     db_min2_idx);
}

// Scatters a batch of kMq queries against a panel. Computes base_indices once
// per panel and forwards VEC_GROUP accessors to ScatterSingleQuery.
template <size_t kMq, class Float, class Int32, class ScoresAccessor,
          class Min1Accessor, class Min2Accessor, class Min1IdxAccessor,
          class Min2IdxAccessor>
HWY_INLINE void ScatterBatch(
    Float df, Int32 di32, ScoresAccessor scores_accessor, size_t query_start,
    size_t num_active, size_t panel_base, size_t num_points,
    const hn::Vec<Int32>& lane_idx, const hn::Vec<Float>& inf_v,
    Min1Accessor q_min1_acc, Min2Accessor q_min2_acc,
    Min1IdxAccessor q_min1_idx_acc, Min2IdxAccessor q_min2_idx_acc,
    hn::Vec<Float>& db_min1, hn::Vec<Float>& db_min2,
    hn::Vec<Int32>& db_min1_idx, hn::Vec<Int32>& db_min2_idx) {
  const hn::Vec<Int32> base_indices =
      hn::Add(hn::Set(di32, static_cast<int32_t>(panel_base)), lane_idx);
  ForIndex<kMq>([&](auto q_idx) {
    constexpr size_t q = decltype(q_idx)::value;
    if (q < num_active) {
      const size_t query_index = query_start + q;
      ScatterSingleQuery(df, di32, scores_accessor(q_idx), query_index,
                         panel_base, num_points, base_indices, inf_v,
                         q_min1_acc(q_idx), q_min2_acc(q_idx),
                         q_min1_idx_acc(q_idx), q_min2_idx_acc(q_idx), db_min1,
                         db_min2, db_min1_idx, db_min2_idx);
    }
  });
}

// Cross-lane reduction: extract global top-2 from NF per-lane (min1, min2)
// pairs and merge into a scalar Top2Result.
template <class Float, class Int32>
HWY_INLINE void ReduceTop2ToScalar(Float df, Int32 di32,
                                   const hn::Vec<Float>& min1_v,
                                   const hn::Vec<Float>& min2_v,
                                   const hn::Vec<Int32>& min1_idx_v,
                                   const hn::Vec<Int32>& min2_idx_v,
                                   Top2Result& r) {
  const size_t NF = hn::Lanes(df);
  const float inf = std::numeric_limits<float>::max();
  HWY_ALIGN float s1[HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN float s2[HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t i1[HWY_MAX_BYTES / sizeof(int32_t)];
  HWY_ALIGN int32_t i2[HWY_MAX_BYTES / sizeof(int32_t)];
  hn::Store(min1_v, df, s1);
  hn::Store(min2_v, df, s2);
  hn::Store(min1_idx_v, di32, i1);
  hn::Store(min2_idx_v, di32, i2);
  for (size_t j = 0; j < NF; ++j) {
    if (s1[j] < inf) r.Update(s1[j], i1[j]);
    if (s2[j] < inf) r.Update(s2[j], i2[j]);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace ginger

// clang-format off
#endif  // GINGER_TURBOQUANT_PIPNN_SCATTER_EPILOGUE_TOGGLE // NOLINT
// clang-format on
