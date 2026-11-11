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

#include "ginger/turboquant/many_to_many/many_to_many_topk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/many_to_many/many_to_many_topk.cc"
// clang-format on
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/many_to_many_turboquant.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "hwy/cache_control.h"  // IWYU pragma: keep
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

#ifndef MANY_TO_MANY_TOPK_CC_HELPERS_
#define MANY_TO_MANY_TOPK_CC_HELPERS_

namespace ginger {

using ::ginger::many_to_many_topk_internal::TopKState;

// Decodes an order-preserving uint32_t back to a float (inverse of
// FloatToOrderedInt). Restores positive floats by clearing the high bit, and
// negative floats by bitwise NOT. Reverse operation of FloatToOrderedInt (see
// below).
inline float OrderedIntToFloatScalar(uint32_t u) {
  if (u & 0x80000000) {
    u &= ~0x80000000;
  } else {
    u = ~u;
  }
  return absl::bit_cast<float>(u);
}

}  // namespace ginger

#endif  // MANY_TO_MANY_TOPK_CC_HELPERS_

HWY_BEFORE_NAMESPACE();

namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

float ReduceMinSpan(const float* HWY_RESTRICT data, size_t n) {
  const hn::ScalableTag<float> f32;
  const size_t N = hn::Lanes(f32);
  const auto inf = hn::Set(f32, std::numeric_limits<float>::max());
  size_t i = 0;

  auto min0 = inf, min1 = inf, min2 = inf, min3 = inf;

  for (; i + 4 * N <= n; i += 4 * N) {
    min0 = hn::Min(min0, hn::LoadU(f32, data + i + 0 * N));
    min1 = hn::Min(min1, hn::LoadU(f32, data + i + 1 * N));
    min2 = hn::Min(min2, hn::LoadU(f32, data + i + 2 * N));
    min3 = hn::Min(min3, hn::LoadU(f32, data + i + 3 * N));
  }
  for (; i + 2 * N <= n; i += 2 * N) {
    min0 = hn::Min(min0, hn::LoadU(f32, data + i + 0 * N));
    min1 = hn::Min(min1, hn::LoadU(f32, data + i + 1 * N));
  }
  if (i + N <= n) {
    min0 = hn::Min(min0, hn::LoadU(f32, data + i));
    i += N;
  }

  float result =
      hn::ReduceMin(f32, hn::Min(hn::Min(min0, min1), hn::Min(min2, min3)));
  for (; i < n; ++i) {
    result = HWY_MIN(result, data[i]);
  }
  return result;
}

size_t FindNeedle(const float* HWY_RESTRICT data, size_t n, float needle) {
  const hn::ScalableTag<float> f32;
  const size_t N = hn::Lanes(f32);
  const auto needle_v = hn::Set(f32, needle);
  size_t i = 0;

  for (; i + 4 * N <= n; i += 4 * N) {
    auto eq0 = hn::Eq(hn::LoadU(f32, data + i + 0 * N), needle_v);
    auto eq1 = hn::Eq(hn::LoadU(f32, data + i + 1 * N), needle_v);
    auto eq2 = hn::Eq(hn::LoadU(f32, data + i + 2 * N), needle_v);
    auto eq3 = hn::Eq(hn::LoadU(f32, data + i + 3 * N), needle_v);
    auto any_eq = hn::Or(hn::Or(eq0, eq1), hn::Or(eq2, eq3));

    if (HWY_UNLIKELY(!hn::AllFalse(f32, any_eq))) {
      if (intptr_t pos = hn::FindFirstTrue(f32, eq0); pos >= 0)
        return i + 0 * N + pos;
      if (intptr_t pos = hn::FindFirstTrue(f32, eq1); pos >= 0)
        return i + 1 * N + pos;
      if (intptr_t pos = hn::FindFirstTrue(f32, eq2); pos >= 0)
        return i + 2 * N + pos;
      if (intptr_t pos = hn::FindFirstTrue(f32, eq3); pos >= 0)
        return i + 3 * N + pos;
    }
  }

  for (; i + 2 * N <= n; i += 2 * N) {
    auto eq0 = hn::Eq(hn::LoadU(f32, data + i + 0 * N), needle_v);
    auto eq1 = hn::Eq(hn::LoadU(f32, data + i + 1 * N), needle_v);
    auto any_eq = hn::Or(eq0, eq1);

    if (HWY_UNLIKELY(!hn::AllFalse(f32, any_eq))) {
      if (intptr_t pos = hn::FindFirstTrue(f32, eq0); pos >= 0)
        return i + 0 * N + pos;
      if (intptr_t pos = hn::FindFirstTrue(f32, eq1); pos >= 0)
        return i + 1 * N + pos;
    }
  }

  for (; i + 1 * N <= n; i += 1 * N) {
    auto eq0 = hn::Eq(hn::LoadU(f32, data + i), needle_v);
    if (HWY_UNLIKELY(!hn::AllFalse(f32, eq0))) {
      return i + hn::FindFirstTrue(f32, eq0);
    }
  }

  for (; i < n; ++i) {
    if (data[i] == needle) return i;
  }
  return n;
}

void ArgMinNeedle(const float* HWY_RESTRICT data, size_t n, float& best_dist,
                  uint32_t& best_idx, size_t first_db_idx) {
  const float span_min = ReduceMinSpan(data, n);
  if (span_min < best_dist) {
    best_dist = span_min;
    best_idx =
        static_cast<uint32_t>(first_db_idx + FindNeedle(data, n, span_min));
  }
}

void ManyToManyTurboQuantTop1Impl(const TurboQuantDatasetSoA& queries,
                                  const TurboQuantDatasetSoA& database,
                                  IndexSubset query_subset,
                                  IndexSubset db_subset,
                                  absl::Span<TopKResult> results,
                                  ManyToManyTurboQuantWorkspace& workspace) {
  std::fill(results.begin(), results.end(),
            TopKResult{.db_idx = std::numeric_limits<uint32_t>::max(),
                       .distance = std::numeric_limits<float>::max()});
  auto callback = [&](size_t query_idx, absl::Span<const float> scores,
                      size_t first_db_idx) {
    ArgMinNeedle(scores.data(), scores.size(), results[query_idx].distance,
                 results[query_idx].db_idx, first_db_idx);
  };
  ManyToManyTurboQuant(queries, database, query_subset, db_subset, callback,
                       workspace);
}

void ManyToManyTurboQuantTop1PreTransposedImpl(
    const TurboQuantDatasetSoA& queries,
    const PreTransposedTurboQuantDataset& database, IndexSubset query_subset,
    absl::Span<TopKResult> results, ManyToManyTurboQuantWorkspace& workspace) {
  std::fill(results.begin(), results.end(),
            TopKResult{.db_idx = std::numeric_limits<uint32_t>::max(),
                       .distance = std::numeric_limits<float>::max()});
  auto callback = [&](size_t query_idx, absl::Span<const float> scores,
                      size_t first_db_idx) {
    ArgMinNeedle(scores.data(), scores.size(), results[query_idx].distance,
                 results[query_idx].db_idx, first_db_idx);
  };
  ManyToManyTurboQuant(queries, database, query_subset, callback, workspace);
}

// Converts a vector of floats to a
// vector of order-preserving 32-bit unsigned integers.
//
// Bit-level operations for order preservation:
// 1. Reinterpret floats as uint32_t (`scores_v`).
// 2. Extract the high (sign) bit to create an execution mask `is_neg`.
// 3. Conditional flipping:
//    - Negative floats (sign=1) have reversed sort behavior (bigger magnitude
//      means more negative). Applying bitwise NOT (`hn::Not`) reverses both
//      the magnitude sorting and drops the sign bit to 0.
//    - Positive floats (sign=0) already sort naturally by magnitude, but must
//      sit higher than negatives. A bitwise OR with the sign bit flips the MSB
//      to 1 without disturbing internal sort rank.
template <class D, class V>
inline auto FloatToOrderedInt(D, V scores_v) {
  const hn::Rebind<uint32_t, D> u32;
  const auto sign_bit = hn::Set(u32, 0x80000000);
  auto u = hn::BitCast(u32, scores_v);
  auto sign = hn::And(u, sign_bit);
  auto is_neg = hn::Eq(sign, sign_bit);
  return hn::IfThenElse(is_neg, hn::Not(u), hn::Or(u, sign_bit));
}

// Inverse of `FloatToOrderedInt`: converts order-preserving uint32_t lanes back
// to IEEE-754 float bit patterns (as uint32_t lanes). If the high bit is set,
// the value came from a positive float (clear the high bit); otherwise from a
// negative float (bitwise NOT restores the original).
template <class D, class V>
inline auto OrderedIntToFloat(D d, V ordered_v) {
  const auto sign_bit = hn::Set(d, 0x80000000);
  auto is_pos = hn::Eq(hn::And(ordered_v, sign_bit), sign_bit);
  return hn::IfThenElse(is_pos, hn::AndNot(sign_bit, ordered_v),
                        hn::Not(ordered_v));
}

// Packs floating-point scores and database indices into 64-bit items suitable
// for sorting. Each 64-bit item is laid out as:
//   [63..32] order-preserving integer score (via FloatToOrderedInt)
//   [31.. 0] absolute database index
//
// Sorting these uint64_t values in ascending order sorts by score first, then
// breaks ties by database index. The two SIMD lanes of f32/u32 are promoted
// to u64, shifted, and OR'd together.
template <class V1, class V2>
inline void StorePackedScores(V1 scores, V2 indices, uint64_t* dst,
                              size_t count) {
  const hn::ScalableTag<float> f32;
  const hn::ScalableTag<uint64_t> u64;

  auto ord = FloatToOrderedInt(f32, scores);
  auto plo = hn::Or(hn::ShiftLeft<32>(hn::PromoteLowerTo(u64, ord)),
                    hn::PromoteLowerTo(u64, indices));
  auto phi = hn::Or(hn::ShiftLeft<32>(hn::PromoteUpperTo(u64, ord)),
                    hn::PromoteUpperTo(u64, indices));

  const size_t n64 = hn::Lanes(u64);
  if (count >= 2 * n64) {
    hn::StoreU(plo, u64, dst);
    hn::StoreU(phi, u64, dst + n64);
  } else if (count > n64) {
    hn::StoreU(plo, u64, dst);
    hn::StoreN(phi, u64, dst + n64, count - n64);
  } else {
    hn::StoreN(plo, u64, dst, count);
  }
}

// Prunes a query's candidate buffer down to k elements using Highway's
// VQSelect (SIMD quickselect). After selection, the k-th best score is
// decoded back to a float and stored as the new pruning threshold so that
// subsequent scores worse than this can be skipped cheaply.
inline void PruneCandidates(TopKState& state, size_t query_idx) {
  const size_t start = query_idx * state.buffer_capacity;
  const size_t size = state.sizes[query_idx];
  const size_t k = state.k;
  // NOMUTANTS -- `size < k` is equivalent (selecting k of k keeps all).
  if (size <= k) return;
  hwy::K32V32* items =
      reinterpret_cast<hwy::K32V32*>(state.items.data() + start);
  // NOMUTANTS -- +1 pivot mutation is equivalent (leaves threshold looser,
  // elements intact). -1 pivot (k-2) is NOT equivalent and is covered by
  // tests (TopKCoverageTest, TightPivotMaskDrop), but is suppressed here
  // since it modifies the exact same source line.
  hwy::VQSelect(items, size, k - 1, hwy::SortAscending());
  const uint32_t u_score = items[k - 1].key;
  state.thresholds[query_idx] = OrderedIntToFloatScalar(u_score);
  state.sizes[query_idx] = k;
}

// Fast path: appends all incoming scores to a query's buffer without any
// threshold filtering. Used when no threshold has been established yet (i.e.,
// fewer than k candidates have been collected) and the buffer has enough
// remaining capacity for the entire batch. Avoids the per-element comparison
// and compress overhead of the filtered path.
inline void FillCandidatesUnconditionally(TopKState& state, size_t start,
                                          size_t& size, size_t num_scores,
                                          const float* data,
                                          size_t first_db_idx) {
  const hn::ScalableTag<float> f32;
  const hn::ScalableTag<uint32_t> u32;
  const size_t N = hn::Lanes(u32);
  uint64_t* dst = state.items.data() + start + size;
  auto base_idx = hn::Set(u32, static_cast<uint32_t>(first_db_idx));
  auto iota = hn::Iota(u32, 0);
  size_t j = 0;
  for (; j + N <= num_scores; j += N) {
    auto s = hn::LoadU(f32, data + j);
    auto idx = hn::Add(base_idx, iota);
    StorePackedScores(s, idx, dst + j, N);
    base_idx = hn::Add(base_idx, hn::Set(u32, static_cast<uint32_t>(N)));
  }
  if (j < num_scores) {
    const size_t rem = num_scores - j;
    auto s = hn::LoadN(f32, data + j, rem);
    auto idx = hn::Add(base_idx, iota);
    StorePackedScores(s, idx, dst + j, rem);
  }
  size += num_scores;
}

// Core TopK accumulation loop. Wraps any ManyToMany variant with a callback
// that feeds per-query score batches into buffers, prunes on-the-fly with an
// approximate threshold and prunes the buffer periodically.
template <typename F>
void ExecuteTopK(TopKState& state, F&& many_to_many_func) {
  many_to_many_func([&](size_t query_idx, absl::Span<const float> query_scores,
                        size_t first_db_idx) {
    size_t start = query_idx * state.buffer_capacity;
    size_t& size = state.sizes[query_idx];
    size_t num_scores = query_scores.size();
    const float* data = query_scores.data();

    const hn::ScalableTag<float> f32;
    const hn::ScalableTag<uint32_t> u32;
    const size_t N = hn::Lanes(f32);

    // Fast path: no threshold yet and buffer has room for the whole batch.
    // Unlike the filtered path, this needs no N-slot headroom, since
    // FillCandidatesUnconditionally writes exactly `num_scores` items.
    if (state.thresholds[query_idx] == std::numeric_limits<float>::max() &&
        size + num_scores <= state.buffer_capacity) {
      return FillCandidatesUnconditionally(state, start, size, num_scores, data,
                                           first_db_idx);
    }

    // Ensure at least N free slots before entering the filtered loop.
    if (size + N > state.buffer_capacity) {
      PruneCandidates(state, query_idx);
    }

    auto thresh_v = hn::Set(f32, state.thresholds[query_idx]);
    size_t i = 0;
    for (; i + N <= num_scores; i += N) {
      const auto s = hn::LoadU(f32, data + i);
      // Keep only scores strictly below the current threshold.
      const auto mask = hn::Lt(s, thresh_v);

      if (hn::AllFalse(f32, mask)) {
        continue;
      }

      // Compress qualifying scores and their indices into contiguous lanes.
      const auto base = hn::Set(u32, static_cast<uint32_t>(first_db_idx + i));
      const auto idx_v = hn::Add(base, hn::Iota(u32, 0));

      const auto comp_s = hn::Compress(s, mask);
      const auto comp_idx = hn::Compress(idx_v, hn::RebindMask(u32, mask));
      const size_t num = hn::CountTrue(f32, mask);
      StorePackedScores(comp_s, comp_idx, state.items.data() + start + size,
                        num);
      size += num;

      // If the buffer is nearly full, prune and tighten the threshold.
      if (size + N > state.buffer_capacity) {
        PruneCandidates(state, query_idx);
        thresh_v = hn::Set(f32, state.thresholds[query_idx]);
      }
    }
    if (i < num_scores) {
      const size_t remaining = num_scores - i;
      const auto s = hn::LoadN(f32, data + i, remaining);
      auto mask = hn::Lt(s, thresh_v);
      mask = hn::And(mask, hn::FirstN(f32, remaining));

      if (!hn::AllFalse(f32, mask)) {
        const auto base = hn::Set(u32, static_cast<uint32_t>(first_db_idx + i));
        const auto idx_v = hn::Add(base, hn::Iota(u32, 0));

        const auto comp_s = hn::Compress(s, mask);
        const auto comp_idx = hn::Compress(idx_v, hn::RebindMask(u32, mask));
        const size_t num = hn::CountTrue(f32, mask);
        StorePackedScores(comp_s, comp_idx, state.items.data() + start + size,
                          num);
        size += num;

        if (size + N > state.buffer_capacity) {
          PruneCandidates(state, query_idx);
          thresh_v = hn::Set(f32, state.thresholds[query_idx]);
        }
      }
    }
  });
}

void ManyToManyTurboQuantTopKImpl(const TurboQuantDatasetSoA& queries,
                                  const TurboQuantDatasetSoA& database,
                                  IndexSubset query_subset,
                                  IndexSubset db_subset, TopKState& state) {
  ExecuteTopK(state, [&](auto&& callback) {
    ManyToManyTurboQuant(queries, database, query_subset, db_subset, callback,
                         state.workspace);
  });
}

void ManyToManyTurboQuantTopKPreTransposedImpl(
    const TurboQuantDatasetSoA& queries,
    const PreTransposedTurboQuantDataset& database, IndexSubset query_subset,
    TopKState& state) {
  ExecuteTopK(state, [&](auto&& callback) {
    ManyToManyTurboQuant(queries, database, query_subset, callback,
                         state.workspace);
  });
}

// Partially sorts the first k candidates from `src` (ascending by packed
// score) and decodes them into TopKResult output.
//
// The internal representation packs each candidate as a uint64_t:
//   [63..32] order-preserving integer score
//   [31.. 0] database index
//
// Since TopKResult is {uint32_t db_idx, float distance} (8 bytes total, same
// as uint64_t), we reinterpret the sorted buffer as interleaved uint32_t
// pairs: even lanes hold indices, odd lanes hold encoded scores.
//
// The SIMD decode loop applies OrderedIntToFloat to all lanes simultaneously,
// then uses OddEven to merge corrected float scores (from odd lanes) with
// the original indices (from even lanes), producing valid TopKResult structs
// in a single pass.
void FinalizeTopKBlocks(uint64_t* HWY_RESTRICT src,
                        TopKResult* HWY_RESTRICT out, size_t size, size_t k) {
  size_t m = std::min(size, k);

  // Partial sort: places the m smallest elements (by ascending packed score)
  // into src[0..m), in sorted order.
  hwy::K32V32* src_kv = reinterpret_cast<hwy::K32V32*>(src);
  hwy::VQPartialSort(src_kv, size, m, hwy::SortAscending());

  // Decode loop: process 2*m uint32_t lanes (m uint64_t items) and convert
  // odd (score) lanes from order-preserving integers back to IEEE floats.
  const hn::ScalableTag<uint32_t> u32;
  const size_t N = hn::Lanes(u32);

  size_t j = 0;
  for (; j + N <= 2 * m; j += N) {
    const auto v = hn::LoadU(u32, reinterpret_cast<const uint32_t*>(src) + j);
    const auto corrected_distances = OrderedIntToFloat(u32, v);

    // Blend: take corrected float scores from odd lanes (the score halves)
    // and keep original values in even lanes (the index halves).
    const auto fused_distances_and_indices =
        hn::OddEven(corrected_distances, v);

    hn::StoreU(fused_distances_and_indices, u32,
               reinterpret_cast<uint32_t*>(out) + j);
  }
  if (j < 2 * m) {
    const size_t rem = 2 * m - j;
    const auto v =
        hn::LoadN(u32, reinterpret_cast<const uint32_t*>(src) + j, rem);
    const auto corrected_distances = OrderedIntToFloat(u32, v);

    const auto fused_distances_and_indices =
        hn::OddEven(corrected_distances, v);

    hn::StoreN(fused_distances_and_indices, u32,
               reinterpret_cast<uint32_t*>(out) + j, rem);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace ginger {

// Gives the functions below access to the internals of `ManyToManyTopKState`.
struct ManyToManyTopKStateAccess {
  static TopKState& Get(ManyToManyTopKState& state) { return state.state_; }
};

HWY_EXPORT(FinalizeTopKBlocks);
HWY_EXPORT(ManyToManyTurboQuantTop1Impl);
HWY_EXPORT(ManyToManyTurboQuantTop1PreTransposedImpl);
HWY_EXPORT(ManyToManyTurboQuantTopKImpl);
HWY_EXPORT(ManyToManyTurboQuantTopKPreTransposedImpl);

namespace {

// Returns the first `num_queries * k` elements of `state.results`, growing it
// if needed.
absl::Span<TopKResult> PrepareResults(size_t num_queries, int k,
                                      TopKState& state) {
  const size_t num_results = num_queries * static_cast<size_t>(std::max(k, 0));
  if (state.results.size() < num_results) {
    state.results.resize(num_results);
  }
  return absl::MakeSpan(state.results.data(), num_results);
}

// Returns whether to use the `k == 1` fast path, which ignores thresholds.
bool UseTop1(int k, float initial_threshold,
             absl::Span<const float> initial_thresholds) {
  return k == 1 && initial_threshold == std::numeric_limits<float>::max() &&
         initial_thresholds.empty();
}

// Prepares `state` for a run over `num_queries` queries and `num_db` database
// points that keeps up to `k > 0` candidates per query. The initial per-query
// thresholds are `initial_thresholds` if non-empty, and `initial_threshold`
// otherwise. Reuses existing allocations when they are large enough.
void ResetManyToManyTopKState(size_t num_queries, int k, size_t num_db,
                              float initial_threshold,
                              absl::Span<const float> initial_thresholds,
                              TopKState& state) {
  state.num_queries = num_queries;
  state.k = k;
  state.buffer_capacity =
      std::min<size_t>(std::max<size_t>(4 * k + 128, 1024), num_db);

  const size_t items_size = num_queries * state.buffer_capacity;
  if (state.items.size() < items_size) {
    state.items.resize(items_size);
  }
  if (state.sizes.size() < num_queries) {
    state.sizes.resize(num_queries);
  }
  if (state.thresholds.size() < num_queries) {
    state.thresholds.resize(num_queries);
  }
  std::fill_n(state.sizes.begin(), num_queries, 0);
  if (initial_thresholds.empty()) {
    std::fill_n(state.thresholds.begin(), num_queries, initial_threshold);
  } else {
    std::copy(initial_thresholds.begin(), initial_thresholds.end(),
              state.thresholds.begin());
  }
}

// Sorts and decodes the best `state.k` candidates of each query into
// `results`, which holds `state.k` results per query.
void FinalizeManyToManyTopK(TopKState& state, absl::Span<TopKResult> results) {
  for (size_t q = 0; q < state.num_queries; ++q) {
    const size_t size = state.sizes[q];
    uint64_t* src = state.items.data() + q * state.buffer_capacity;
    const size_t m = std::min(size, state.k);

    if (m > 0) {
      HWY_DYNAMIC_DISPATCH(FinalizeTopKBlocks)(
          src, results.data() + q * state.k, size, m);
    }

    // Pad remaining slots with sentinel values so callers always get exactly k
    // results per query, even when the database has fewer than k points.
    for (size_t i = m; i < state.k; ++i) {
      results[q * state.k + i] = {
          .db_idx = std::numeric_limits<uint32_t>::max(),
          .distance = std::numeric_limits<float>::max()};
    }
  }
}

// Shared implementation of the `ManyToManyTurboQuantTopK` overloads: dispatches
// between the Top-1 fast path and general Top-K, writing `k` results per query
// into `state.results` and returning them. The initial thresholds are
// `initial_thresholds` if non-empty, and `initial_threshold` otherwise.
absl::Span<const TopKResult> ManyToManyTurboQuantTop1OrTopK(
    const TurboQuantDatasetSoA& queries, const TurboQuantDatasetSoA& database,
    IndexSubset query_subset, IndexSubset db_subset, int k,
    float initial_threshold, absl::Span<const float> initial_thresholds,
    TopKState& state) {
  if (!initial_thresholds.empty()) {
    ABSL_CHECK_EQ(initial_thresholds.size(), query_subset.size());
  }
  const absl::Span<TopKResult> results =
      PrepareResults(query_subset.size(), k, state);
  if (results.empty()) return results;
  if (UseTop1(k, initial_threshold, initial_thresholds)) {
    HWY_DYNAMIC_DISPATCH(ManyToManyTurboQuantTop1Impl)(
        queries, database, query_subset, db_subset, results, state.workspace);
    return results;
  }
  ResetManyToManyTopKState(query_subset.size(), k, db_subset.size(),
                           initial_threshold, initial_thresholds, state);
  HWY_DYNAMIC_DISPATCH(ManyToManyTurboQuantTopKImpl)(
      queries, database, query_subset, db_subset, state);
  FinalizeManyToManyTopK(state, results);
  return results;
}

// As above, for a pretransposed database.
absl::Span<const TopKResult> ManyToManyTurboQuantTop1OrTopK(
    const TurboQuantDatasetSoA& queries,
    const PreTransposedTurboQuantDataset& database, IndexSubset query_subset,
    int k, float initial_threshold, absl::Span<const float> initial_thresholds,
    TopKState& state) {
  if (!initial_thresholds.empty()) {
    ABSL_CHECK_EQ(initial_thresholds.size(), query_subset.size());
  }
  const absl::Span<TopKResult> results =
      PrepareResults(query_subset.size(), k, state);
  if (results.empty()) return results;
  if (UseTop1(k, initial_threshold, initial_thresholds)) {
    HWY_DYNAMIC_DISPATCH(ManyToManyTurboQuantTop1PreTransposedImpl)(
        queries, database, query_subset, results, state.workspace);
    return results;
  }
  ResetManyToManyTopKState(query_subset.size(), k, database.num_points,
                           initial_threshold, initial_thresholds, state);
  HWY_DYNAMIC_DISPATCH(ManyToManyTurboQuantTopKPreTransposedImpl)(
      queries, database, query_subset, state);
  FinalizeManyToManyTopK(state, results);
  return results;
}

}  // namespace

std::vector<TopKResult> ManyToManyTurboQuantTopK(
    const TurboQuantDatasetSoA& queries, const TurboQuantDatasetSoA& database,
    IndexSubset query_subset, IndexSubset db_subset, int k,
    float initial_threshold) {
  TopKState state;
  ManyToManyTurboQuantTop1OrTopK(queries, database, query_subset, db_subset, k,
                                 initial_threshold, /*initial_thresholds=*/{},
                                 state);
  // `state` is fresh, so `state.results` holds exactly the computed results.
  return std::move(state.results);
}

std::vector<TopKResult> ManyToManyTurboQuantTopK(
    const TurboQuantDatasetSoA& queries,
    const PreTransposedTurboQuantDataset& database, IndexSubset query_subset,
    int k, float initial_threshold) {
  TopKState state;
  ManyToManyTurboQuantTop1OrTopK(queries, database, query_subset, k,
                                 initial_threshold, /*initial_thresholds=*/{},
                                 state);
  // `state` is fresh, so `state.results` holds exactly the computed results.
  return std::move(state.results);
}

absl::Span<const TopKResult> ManyToManyTurboQuantTopK(
    const TurboQuantDatasetSoA& queries, const TurboQuantDatasetSoA& database,
    IndexSubset query_subset, IndexSubset db_subset, int k,
    ManyToManyTopKState& state, absl::Span<const float> initial_thresholds) {
  return ManyToManyTurboQuantTop1OrTopK(
      queries, database, query_subset, db_subset, k,
      std::numeric_limits<float>::max(), initial_thresholds,
      ManyToManyTopKStateAccess::Get(state));
}

absl::Span<const TopKResult> ManyToManyTurboQuantTopK(
    const TurboQuantDatasetSoA& queries,
    const PreTransposedTurboQuantDataset& database, IndexSubset query_subset,
    int k, ManyToManyTopKState& state,
    absl::Span<const float> initial_thresholds) {
  return ManyToManyTurboQuantTop1OrTopK(
      queries, database, query_subset, k, std::numeric_limits<float>::max(),
      initial_thresholds, ManyToManyTopKStateAccess::Get(state));
}

}  // namespace ginger

#endif
