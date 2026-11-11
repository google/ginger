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

// SIMD lane-parallel top-K tracking.
//
// Provides helpers for fused GEMM + selection kernels where scores are
// immediately consumed by per-query SIMD min-trackers, avoiding
// materialization of a full score matrix.
//
// Two state representations are supported:
//   1. Buffer-based (UpdateTopK / ReduceTopK):
//      K * hn::Lanes(hn::ScalableTag<float>()) elements in aligned memory;
//      requires Load/Store each panel. Safe for any K.
//   2. Register-resident (UpdateTopK1InRegisters / UpdateTopK3InRegisters):
//      Vec pairs passed directly as references. No memory traffic during
//      the panel loop. Supported for K=1 and K=3.
//
// Include after foreach_target.h, highway.h. Self-contained: sets up
// HWY_BEFORE_NAMESPACE()/HWY_AFTER_NAMESPACE() internally.

#ifndef GINGER_TURBOQUANT_PIPNN_TOPK_TRACKER_INL_H_
#define GINGER_TURBOQUANT_PIPNN_TOPK_TRACKER_INL_H_
#include <cstddef>
#include <cstdint>
#include <limits>
#endif  // GINGER_TURBOQUANT_PIPNN_TOPK_TRACKER_INL_H_

// clang-format off
#if defined(GINGER_TURBOQUANT_PIPNN_TOPK_TRACKER_TOGGLE) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef GINGER_TURBOQUANT_PIPNN_TOPK_TRACKER_TOGGLE
#undef GINGER_TURBOQUANT_PIPNN_TOPK_TRACKER_TOGGLE
#else
#define GINGER_TURBOQUANT_PIPNN_TOPK_TRACKER_TOGGLE
#endif
// clang-format on

#include "ginger/turboquant/pipnn/topk_result.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using Float = hn::ScalableTag<float>;
using Int = hn::ScalableTag<int32_t>;

// Mask padding lane scores (for datapoint indices >= num_points) to +inf and
// return the index vector (Iota(panel_base)).
HWY_INLINE hn::Vec<Int> MaskPaddingLanes(hn::Vec<Float>& score,
                                         int32_t panel_base,
                                         int32_t num_points) {
  const Float df;
  const Int di;
  const hn::Vec<Int> panel_indices = hn::Iota(di, panel_base);
  if (panel_base + hn::Lanes(di) > num_points) {
    const hn::Mask<Int> valid = hn::Lt(panel_indices, hn::Set(di, num_points));
    score = hn::IfThenElse(hn::RebindMask(df, valid), score,
                           hn::Set(df, std::numeric_limits<float>::max()));
  }
  return panel_indices;
}

// Update per-lane top-K state stored in buffers with a new score vector.
// Layout: K contiguous vectors of NF floats each, where NF = hn::Lanes(Float).
// The k-th vector (top_score_buf[k*NF .. (k+1)*NF-1]) holds the (k+1)-th best
// score for each of the NF parallel lanes. This layout enables shared vector
// operations across all NF lanes simultaneously. `panel_base` is the starting
// global index of the current panel of database points being processed.
template <size_t K>
HWY_INLINE void UpdateTopK(hn::Vec<Float> score, size_t panel_base,
                           size_t num_points, float* HWY_RESTRICT top_score_buf,
                           int32_t* HWY_RESTRICT top_idx_buf) {
  const Float df;
  const Int di;
  const size_t NF = hn::Lanes(df);
  const hn::Vec<Int> panel_indices =
      MaskPaddingLanes(score, panel_base, num_points);
  if (hn::AllTrue(df,
                  hn::Ge(score, hn::Load(df, top_score_buf + (K - 1) * NF)))) {
    return;
  }
  hn::Vec<Float> displaced_score = score;
  hn::Vec<Int> displaced_idx = panel_indices;
  // Try to insert the new scores first at top-1. Then propagate the worse score
  // to the next iteration and repeat.
  for (size_t k = 0; k < K; ++k) {
    hn::Vec<Float> slot_score = hn::Load(df, top_score_buf + k * NF);
    hn::Vec<Int> slot_idx = hn::Load(di, top_idx_buf + k * NF);
    const hn::Mask<Float> is_better = hn::Lt(displaced_score, slot_score);
    const hn::Mask<Int> is_better_i = hn::RebindMask(di, is_better);
    const hn::Vec<Float> new_score =
        hn::IfThenElse(is_better, displaced_score, slot_score);
    const hn::Vec<Int> new_idx =
        hn::IfThenElse(is_better_i, displaced_idx, slot_idx);
    displaced_score = hn::IfThenElse(is_better, slot_score, displaced_score);
    displaced_idx = hn::IfThenElse(is_better_i, slot_idx, displaced_idx);
    hn::Store(new_score, df, top_score_buf + k * NF);
    hn::Store(new_idx, di, top_idx_buf + k * NF);
  }
}

// Cross-lane reduction: extract global top-K from K×NF buffer lanes.
template <size_t K>
HWY_INLINE void ReduceTopK(const float* HWY_RESTRICT top_score_buf,
                           const int32_t* HWY_RESTRICT top_idx_buf,
                           FixedTopKResult<K>& result) {
  const Float df;
  const size_t NF = hn::Lanes(df);
  // TODO: Investigate whether partial sort is faster
  // here.
  for (size_t j = 0; j < K * NF; ++j) {
    if (top_score_buf[j] < std::numeric_limits<float>::max()) {
      result.Insert(top_score_buf[j], top_idx_buf[j]);
    }
  }
}

// Register-resident top-K (K=1 and K=3)

// K=1 update: single best (score, index) pair per lane.
// `panel_base` is the starting global index of the current panel of database
// points being processed.
HWY_INLINE void UpdateTopK1InRegisters(hn::Vec<Float> score, size_t panel_base,
                                       size_t num_points,
                                       hn::Vec<Float>& best_score,
                                       hn::Vec<Int>& best_idx) {
  const Int di;
  const hn::Vec<Int> panel_indices =
      MaskPaddingLanes(score, panel_base, num_points);
  const hn::Mask<Float> is_better = hn::Lt(score, best_score);
  best_score = hn::IfThenElse(is_better, score, best_score);
  best_idx =
      hn::IfThenElse(hn::RebindMask(di, is_better), panel_indices, best_idx);
}

// K=3 update: displacement sort across 3 (score, index) pairs per lane.
// `panel_base` is the starting global index of the current panel of database
// points being processed.
// Precondition/Invariant: Maintains score0 <= score1 <= score2 per lane across
// calls.
HWY_INLINE void UpdateTopK3InRegisters(
    hn::Vec<Float> score, size_t panel_base, size_t num_points,
    hn::Vec<Float>& score0, hn::Vec<Float>& score1, hn::Vec<Float>& score2,
    hn::Vec<Int>& idx0, hn::Vec<Int>& idx1, hn::Vec<Int>& idx2) {
  const Int di;
  hn::Vec<Int> panel_indices = MaskPaddingLanes(score, panel_base, num_points);

  // Position 0.
  hn::Mask<Float> is_better = hn::Lt(score, score0);
  hn::Mask<Int> is_better_i = hn::RebindMask(di, is_better);
  hn::Vec<Float> evicted_score = hn::IfThenElse(is_better, score0, score);
  hn::Vec<Int> evicted_idx = hn::IfThenElse(is_better_i, idx0, panel_indices);
  score0 = hn::IfThenElse(is_better, score, score0);
  idx0 = hn::IfThenElse(is_better_i, panel_indices, idx0);

  // Position 1.
  is_better = hn::Lt(evicted_score, score1);
  is_better_i = hn::RebindMask(di, is_better);
  hn::Vec<Float> evicted_score2 =
      hn::IfThenElse(is_better, score1, evicted_score);
  hn::Vec<Int> evicted_idx2 = hn::IfThenElse(is_better_i, idx1, evicted_idx);
  score1 = hn::IfThenElse(is_better, evicted_score, score1);
  idx1 = hn::IfThenElse(is_better_i, evicted_idx, idx1);

  // Position 2 (last — evicted value is discarded).
  is_better = hn::Lt(evicted_score2, score2);
  is_better_i = hn::RebindMask(di, is_better);
  score2 = hn::IfThenElse(is_better, evicted_score2, score2);
  idx2 = hn::IfThenElse(is_better_i, evicted_idx2, idx2);
}

// K=1 reduction: extract the single best (score, index) across all lanes.
HWY_INLINE void HorizontalReduceTopK1FromRegisters(hn::Vec<Float> score_vec,
                                                   hn::Vec<Int> idx_vec,
                                                   FixedTopKResult<1>& result) {
  const Float df;
  const hn::Vec<Float> min_vec = hn::MinOfLanes(df, score_vec);
  const float best = hn::GetLane(min_vec);
  if (best >= std::numeric_limits<float>::max()) return;
  const hn::Mask<Float> eq = hn::Eq(score_vec, min_vec);
  const intptr_t lane = hn::FindKnownFirstTrue(df, eq);
  result.Set(0, best, hn::ExtractLane(idx_vec, lane));
}

// K=3 reduction: iteratively extract the 3 best (score, index) pairs.
// Scores are taken by value so we can mask out winners without mutating
// the caller's register state.
//
// Precondition: UpdateTopK3InRegisters maintains score0 <= score1 <= score2 per
// lane. We exploit this by always finding the global minimum in
// score0, then blending the next-best values up (score1 -> score0, score2 ->
// score1, inf -> score2) at the winning lane. Output result is guaranteed to
// be sorted ascending (result.scores[0] <= result.scores[1] <=
// result.scores[2]).
HWY_INLINE void HorizontalReduceTopK3FromRegisters(
    hn::Vec<Float> score0, hn::Vec<Float> score1, hn::Vec<Float> score2,
    hn::Vec<Int> idx0, hn::Vec<Int> idx1, hn::Vec<Int> idx2,
    FixedTopKResult<3>& result) {
  const Float df;
  const Int di;
  const hn::Vec<Float> inf_v = hn::Set(df, std::numeric_limits<float>::max());
  for (size_t k = 0; k < 3; ++k) {
    const hn::Vec<Float> min_score_vec = hn::MinOfLanes(df, score0);
    const float min_score = hn::GetLane(min_score_vec);
    if (min_score >= std::numeric_limits<float>::max()) break;

    const hn::Mask<Float> eq = hn::Eq(score0, min_score_vec);
    const intptr_t lane = hn::FindKnownFirstTrue(df, eq);
    result.Set(k, min_score, hn::ExtractLane(idx0, lane));

    // Mask the winning lane and shift next-best values up.
    const hn::Mask<Float> first_eq = hn::And(eq, hn::FirstN(df, lane + 1));
    const hn::Mask<Int> first_eq_i = hn::RebindMask(di, first_eq);

    score0 = hn::IfThenElse(first_eq, score1, score0);
    score1 = hn::IfThenElse(first_eq, score2, score1);
    score2 = hn::IfThenElse(first_eq, inf_v, score2);

    idx0 = hn::IfThenElse(first_eq_i, idx1, idx0);
    idx1 = hn::IfThenElse(first_eq_i, idx2, idx1);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

// clang-format off
#endif  // GINGER_TURBOQUANT_PIPNN_TOPK_TRACKER_TOGGLE // NOLINT
// clang-format on
