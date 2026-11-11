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

// SIMD-accelerated RobustPrune for TurboQuant datasets.
//
// Algorithm Overview:
// RobustPrune greedily retains candidates c in ascending distance to the
// source node. A candidate c is pruned by an existing survivor s if:
//   - Squared L2:  dist(s, c) < dist(source_node, c) / alpha
//                  (equivalently, alpha * dist(s, c) < dist(source_node, c))
//   - Dot Product distance: dist(s, c) < dist(source_node, c) * alpha
//                  (for non-positive distances where smaller is closer)
// Naive distance evaluation incurs redundant decoding and memory traffic. This
// implementation eliminates them.
//
// Candidates are processed in batches (kBatchSize = 4) and decoded into 8-bit
// integers once. The candidates are evaluated against already decoded
// survivors. Pruned candidates are dropped from the batch immediately to skip
// remaining survivor comparisons. Those that survive all checks are retained
// and used to prune subsequent candidates.

#include "ginger/turboquant/pipnn/robust_prune.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "hwy/contrib/sort/vqsort.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ginger/turboquant/pipnn/robust_prune.cc"

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/decode_queries-inl.h"
#include "ginger/turboquant/simd/vec_group-inl.h"

namespace ginger {

namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

using ::ginger::GingerNodeId;
using ::ginger::TurboQuantDatasetSoA;
using ::ginger::HWY_NAMESPACE::ForIndex;
using ::ginger::turboquant_codebooks::kTurboQuantCentroidsInt8;

// Number of candidates decoded and scored together per batch.
constexpr size_t kBatchSize = 4;

inline uint32_t FloatToSortableUint32(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if (bits == 0x80000000u) bits = 0;  // Canonicalize -0.0f to +0.0f
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

inline float SortableUint32ToFloat(uint32_t bits) {
  const uint32_t raw = (bits & 0x80000000u) ? (bits & 0x7FFFFFFFu) : ~bits;
  float value;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

inline uint64_t PackSortKey(float distance, GingerNodeId node_id) {
  return (static_cast<uint64_t>(FloatToSortableUint32(distance)) << 32) |
         static_cast<uint32_t>(node_id);
}

inline void UnpackSortKey(uint64_t key, GingerNodeId* node_id,
                          float* distance) {
  *node_id = static_cast<GingerNodeId>(static_cast<uint32_t>(key));
  *distance = SortableUint32ToFloat(static_cast<uint32_t>(key >> 32));
}

// Decodes all num_chunks chunks of num_rows packed 4-bit rows into signed
// centroid bytes (chunked even/odd layout) and fuses each row's byte sum
// (the bias-correction input) into the same sweep.
HWY_INLINE void PredecodeCandidateRows(
    const uint8_t* const* HWY_RESTRICT packed, size_t num_rows,
    size_t num_chunks, hn::Mask<hn::ScalableTag<int8_t>> last_even_mask,
    hn::Mask<hn::ScalableTag<int8_t>> last_odd_mask, int8_t* HWY_RESTRICT out,
    size_t decoded_stride, int32_t* HWY_RESTRICT batch_byte_sums) {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t lanes = hn::Lanes(u8);
  const auto nib_mask = hn::Set(u8, uint8_t{0x0F});
  const auto centroids = hn::LoadDup128(i8, kTurboQuantCentroidsInt8.data());
  const auto ones = hn::Set(u8, uint8_t{1});
  for (size_t k = 0; k < num_rows; ++k) {
    int8_t* HWY_RESTRICT dst = out + k * decoded_stride;
    auto sum = hn::Zero(i32);
    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
      const auto nibbles = hn::LoadU(u8, packed[k] + chunk * lanes);
      hn::Vec<decltype(i8)> q_even, q_odd;
      DequantizeNibbles<decltype(i8), decltype(u8)>(i8, nib_mask, centroids,
                                                    nibbles, q_even, q_odd);
      if (chunk + 1 == num_chunks) {
        q_even = hn::IfThenElseZero(last_even_mask, q_even);
        q_odd = hn::IfThenElseZero(last_odd_mask, q_odd);
      }
      hn::StoreU(q_even, i8, dst + chunk * 2 * lanes);
      hn::StoreU(q_odd, i8, dst + chunk * 2 * lanes + lanes);
      sum = hn::SumOfMulQuadAccumulate(i32, ones, q_even, sum);
      sum = hn::SumOfMulQuadAccumulate(i32, ones, q_odd, sum);
    }
    batch_byte_sums[k] = hn::ReduceSum(i32, sum);
  }
}

// Computes integer dot products between one survivor's decoded unsigned uint8
// row and kBatchSize candidates' decoded signed int8 rows over num_chunks
// chunks. Pure load+VNNI: no nibble decode.
template <size_t kBatchSize>
HWY_INLINE void ScoreBatchAgainstSurvivor(
    const int8_t* const* HWY_RESTRICT decoded_rows,
    const uint8_t* HWY_RESTRICT survivor_row, size_t num_chunks,
    int32_t* HWY_RESTRICT dots) {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t lanes = hn::Lanes(u8);

  VEC_GROUP_DEFINE(i32, dot);
  ForIndex<kBatchSize>(
      [&](auto k) HWY_ATTR { VEC_GROUP_ACCESSOR(dot)(k) = hn::Zero(i32); });

  for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
    const auto survivor_even = hn::LoadU(u8, survivor_row + chunk * 2 * lanes);
    const auto survivor_odd =
        hn::LoadU(u8, survivor_row + chunk * 2 * lanes + lanes);
    ForIndex<kBatchSize>([&](auto k) HWY_ATTR {
      const auto candidate_even =
          hn::LoadU(i8, decoded_rows[decltype(k)::value] + chunk * 2 * lanes);
      const auto candidate_odd = hn::LoadU(
          i8, decoded_rows[decltype(k)::value] + chunk * 2 * lanes + lanes);
      VEC_GROUP_ACCESSOR(dot)(k) = hn::SumOfMulQuadAccumulate(
          i32, survivor_even, candidate_even, VEC_GROUP_ACCESSOR(dot)(k));
      VEC_GROUP_ACCESSOR(dot)(k) = hn::SumOfMulQuadAccumulate(
          i32, survivor_odd, candidate_odd, VEC_GROUP_ACCESSOR(dot)(k));
    });
  }
  ForIndex<kBatchSize>([&](auto k) HWY_ATTR {
    dots[decltype(k)::value] =
        hn::ReduceSum(hn::ScalableTag<int32_t>(), VEC_GROUP_ACCESSOR(dot)(k));
  });
}

// Converts a bias-corrected integer dot product to a distance score.
inline float ComputeDistanceScore(bool is_l2, int32_t corrected_dot,
                                  float survivor_norm, float survivor_sq_norm,
                                  float candidate_norm,
                                  float candidate_sq_norm) {
  const float dot =
      static_cast<float>(corrected_dot) * (-survivor_norm) * candidate_norm;
  return is_l2
             ? std::max(0.0f, dot * 2.0f + survivor_sq_norm + candidate_sq_norm)
             : dot;
}

// Processes candidates in register-tiled groups of kBatchSize (default 4),
// scoring them greedily against survivors using predecoded rows.
void AcceleratedRobustPruneImpl(const TurboQuantDatasetSoA& points,
                                absl::Span<const GingerNodeId> candidates,
                                absl::Span<const float> distances_to_source,
                                GingerNodeId source_node, float alpha,
                                size_t max_survivors,
                                RobustPruneWorkspace* workspace,
                                std::vector<GingerNodeId>& out_survivors,
                                bool keep_pruned_neighbors) {
  // TODO: Support 3-bit prefix in AcceleratedRobustPrune.
  ABSL_CHECK(!points.has_3bit_prefix());
  size_t num_candidates = candidates.size();
  out_survivors.clear();
  workspace->sorted_candidate_ids.clear();
  workspace->sorted_candidate_distances.clear();
  if (keep_pruned_neighbors) {
    workspace->pruned_candidates.clear();
  }
  if (num_candidates == 0 || max_survivors == 0) return;

  const bool is_l2 =
      (points.metric() == TurboQuantDistanceMetric::kSquaredL2Distance) ||
      points.normalized_datapoints();
  const bool shift_dot_to_l2 =
      (points.metric() == TurboQuantDistanceMetric::kDotProductDistance) &&
      points.normalized_datapoints();

  // Sort candidates by ascending distance using packed (distance, id) keys,
  // filtering out source_node before sorting and skipping adjacent duplicate
  // IDs when unpacking.
  workspace->sort_keys.resize(num_candidates);
  size_t num_keys = 0;
  for (size_t i = 0; i < num_candidates; ++i) {
    if (candidates[i] == source_node) continue;
    float dist = distances_to_source[i];
    if (shift_dot_to_l2) {
      dist = std::max(0.0f, dist * 2.0f + 2.0f);
    } else if (is_l2) {
      dist = std::max(0.0f, dist);
    }
    workspace->sort_keys[num_keys++] = PackSortKey(dist, candidates[i]);
  }
  if (num_keys == 0) return;
  hwy::VQSort(workspace->sort_keys.data(), num_keys, hwy::SortAscending());

  workspace->sorted_candidate_ids.resize(num_keys);
  workspace->sorted_candidate_distances.resize(num_keys);
  num_candidates = 0;
  for (size_t i = 0; i < num_keys; ++i) {
    GingerNodeId id;
    float dist;
    UnpackSortKey(workspace->sort_keys[i], &id, &dist);
    if (num_candidates == 0 ||
        id != workspace->sorted_candidate_ids[num_candidates - 1]) {
      workspace->sorted_candidate_ids[num_candidates] = id;
      workspace->sorted_candidate_distances[num_candidates] = dist;
      ++num_candidates;
    }
  }
  workspace->sorted_candidate_ids.resize(num_candidates);
  workspace->sorted_candidate_distances.resize(num_candidates);
  const GingerNodeId* sorted_cands = workspace->sorted_candidate_ids.data();
  const float* sorted_dists = workspace->sorted_candidate_distances.data();

  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const size_t lanes = hn::Lanes(u8);
  const size_t bytes_per_point = points.num_bytes_per_datapoint();
  const size_t num_chunks = hwy::DivCeil(bytes_per_point, lanes);
  const size_t decoded_stride = DecodedQueryStride(points);
  const uint8_t* HWY_RESTRICT quantized_data = points.quantized_data();
  const size_t quantized_stride = points.stride();
  const NormStorageReader norms = points.NormReader();
  const float threshold_multiplier = is_l2 ? (1.0f / alpha) : alpha;

  // Masks replicating DecodeQueries' zeroing in the final chunk: bytes past
  // the payload remainder, and the padding nibble of an odd dimensionality.
  const size_t remainder_bytes = bytes_per_point % lanes;
  const auto last_valid = (remainder_bytes > 0)
                              ? hn::FirstN(u8, remainder_bytes)
                              : hn::FirstN(u8, lanes);
  const auto last_even_mask = hn::RebindMask(i8, last_valid);
  auto last_odd_mask = last_even_mask;
  if (points.dimensionality() % 2 != 0 && bytes_per_point > 0) {
    const auto pad_lane = hn::Eq(
        hn::Iota(u8, 0),
        hn::Set(u8, static_cast<uint8_t>((bytes_per_point - 1) % lanes)));
    last_odd_mask = hn::AndNot(hn::RebindMask(i8, pad_lane), last_odd_mask);
  }

  // Survivor state: decoded unsigned rows (L1/L2 hot) and norms.
  const size_t survivor_capacity = std::min(max_survivors, num_candidates);
  workspace->survivor_rows.resize(survivor_capacity * decoded_stride);
  workspace->survivor_norms.resize(survivor_capacity);
  if (is_l2) workspace->survivor_squared_norms.resize(survivor_capacity);
  uint8_t* HWY_RESTRICT survivor_rows = workspace->survivor_rows.data();
  float* HWY_RESTRICT survivor_norms = workspace->survivor_norms.data();
  float* HWY_RESTRICT survivor_sqnorms =
      is_l2 ? workspace->survivor_squared_norms.data() : nullptr;
  size_t num_survivors = 0;
  out_survivors.reserve(survivor_capacity);

  // Per-tile predecoded candidate rows (signed centroid bytes in the chunked
  // even/odd layout). Decoded once per tile by PredecodeCandidateRows and
  // reused by every survivor pass, Phase B, and promotion.
  workspace->batch_decoded_rows.resize(kBatchSize * decoded_stride);
  int8_t* HWY_RESTRICT decoded_rows = workspace->batch_decoded_rows.data();

  // Rolling T0 prefetch cursor bringing upcoming candidate vector blocks and
  // norm metadata into L1/L2 cache ahead of SIMD decoding.
  size_t prefetch_index = 0;
  const auto prefetch_candidates = [&](size_t target) {
    target = std::min(target, num_candidates);
    for (; prefetch_index < target; ++prefetch_index) {
      const size_t index = static_cast<size_t>(sorted_cands[prefetch_index]);
      const uint8_t* p = quantized_data + index * quantized_stride;
      for (size_t offset = 0; offset < bytes_per_point; offset += 64) {
        absl::PrefetchToLocalCache(p + offset);
      }
      absl::PrefetchToLocalCache(norms.NormAddress(index));
    }
  };

  size_t candidate_index = 0;
  while (candidate_index < num_candidates &&
         out_survivors.size() < max_survivors) {
    prefetch_candidates(candidate_index + 2 * kBatchSize);

    // Form a register tile of up to kBatchSize candidates.
    const size_t num_slots =
        std::min(kBatchSize, num_candidates - candidate_index);
    const GingerNodeId* tile_cands = sorted_cands + candidate_index;
    const float* tile_dists = sorted_dists + candidate_index;
    candidate_index += num_slots;

    const uint8_t* batch_packed_ptrs[kBatchSize];
    float batch_norms[kBatchSize];
    float batch_sqnorms[kBatchSize] = {};
    float batch_thresholds[kBatchSize];
    for (size_t slot = 0; slot < num_slots; ++slot) {
      const size_t index = static_cast<size_t>(tile_cands[slot]);
      batch_packed_ptrs[slot] = quantized_data + index * quantized_stride;
      batch_norms[slot] = norms.GetNormScalingFactor(index);
      batch_sqnorms[slot] = norms.GetSquaredNorm(index);
      batch_thresholds[slot] = tile_dists[slot] * threshold_multiplier;
    }
    // Pad unused slots to avoid reading garbage in PredecodeCandidateRows.
    for (size_t k = num_slots; k < kBatchSize; ++k) {
      batch_packed_ptrs[k] = batch_packed_ptrs[0];
    }

    // Decode the tile's rows once (fusing each slot's byte sum into the same
    // sweep). Every survivor pass below is pure load+VNNI against the
    // decoded rows.
    int32_t batch_byte_sums[kBatchSize];
    PredecodeCandidateRows(batch_packed_ptrs, num_slots, num_chunks,
                           last_even_mask, last_odd_mask, decoded_rows,
                           decoded_stride, batch_byte_sums);

    // Pruning within each register tile proceeds in two phases:
    //
    // - Phase A (Inter-tile Pruning): Scores all currently active candidates in
    //   the tile against all previously promoted survivors (survivors promoted
    //   before this tile). Candidates pruned by any pre-existing survivor are
    //   compacted out of `active[]`, reducing the SIMD width for subsequent
    //   survivor dot products.
    //
    // - Phase B (Intra-tile Resolution & Promotion): Resolves remaining live
    //   candidates in sequential candidate order against any survivors promoted
    //   *within this tile*. When a candidate survives all checks, it is
    //   immediately promoted to `survivor_rows` (via XOR 0x80 sign flip to
    //   unsigned centroids) so subsequent candidates in the current tile can be
    //   pruned against it.
    size_t active[kBatchSize];
    size_t num_active = num_slots;
    for (size_t k = 0; k < num_slots; ++k) active[k] = k;
    const size_t first_survivor_in_batch = num_survivors;
    for (size_t si = 0; si < first_survivor_in_batch && num_active > 0; ++si) {
      const int8_t* active_rows[kBatchSize];
      for (size_t j = 0; j < num_active; ++j) {
        active_rows[j] = decoded_rows + active[j] * decoded_stride;
      }
      // Pad unused slots to avoid reading garbage in ScoreBatchAgainstSurvivor.
      for (size_t j = num_active; j < kBatchSize; ++j) {
        active_rows[j] = active_rows[0];
      }
      const uint8_t* HWY_RESTRICT survivor_row =
          survivor_rows + si * decoded_stride;
      int32_t raw_dots[kBatchSize];
      switch (num_active) {
        case 1:
          ScoreBatchAgainstSurvivor<1>(active_rows, survivor_row, num_chunks,
                                       raw_dots);
          break;
        case 2:
          ScoreBatchAgainstSurvivor<2>(active_rows, survivor_row, num_chunks,
                                       raw_dots);
          break;
        case 3:
          ScoreBatchAgainstSurvivor<3>(active_rows, survivor_row, num_chunks,
                                       raw_dots);
          break;
        default:
          ScoreBatchAgainstSurvivor<4>(active_rows, survivor_row, num_chunks,
                                       raw_dots);
          break;
      }
      const float survivor_norm = survivor_norms[si];
      const float survivor_sq_norm = is_l2 ? survivor_sqnorms[si] : 0.0f;
      size_t num_kept = 0;
      for (size_t j = 0; j < num_active; ++j) {
        const size_t slot = active[j];
        const int32_t corrected_dot = raw_dots[j] - 128 * batch_byte_sums[slot];
        if (ComputeDistanceScore(is_l2, corrected_dot, survivor_norm,
                                 survivor_sq_norm, batch_norms[slot],
                                 batch_sqnorms[slot]) >=
            batch_thresholds[slot]) {
          active[num_kept++] = slot;
        }
      }
      num_active = num_kept;
    }

    // Phase B: resolve the surviving slots in candidate order against
    // survivors added within this tile, then promote. Dots reuse the
    // predecoded rows and fused byte sums.
    bool slot_survived[kBatchSize] = {};
    for (size_t j = 0; j < num_active; ++j) {
      if (out_survivors.size() >= max_survivors) break;
      const size_t slot = active[j];
      bool pruned = false;
      for (size_t si = first_survivor_in_batch; si < num_survivors && !pruned;
           ++si) {
        const int8_t* single_row[1] = {decoded_rows + slot * decoded_stride};
        int32_t raw_dot;
        ScoreBatchAgainstSurvivor<1>(single_row,
                                     survivor_rows + si * decoded_stride,
                                     num_chunks, &raw_dot);
        const int32_t corrected_dot = raw_dot - 128 * batch_byte_sums[slot];
        pruned = ComputeDistanceScore(is_l2, corrected_dot, survivor_norms[si],
                                      is_l2 ? survivor_sqnorms[si] : 0.0f,
                                      batch_norms[slot], batch_sqnorms[slot]) <
                 batch_thresholds[slot];
      }
      if (!pruned) {
        // Promote: unsigned survivor row = decoded signed row XOR 0x80
        // (kTurboQuantCentroidsUint8[i] == kTurboQuantCentroidsInt8[i] + 128).
        // Masked padding bytes become 128; they never contribute because the
        // candidate side of every dot has those bytes zeroed.
        const int8_t* HWY_RESTRICT source =
            decoded_rows + slot * decoded_stride;
        uint8_t* HWY_RESTRICT destination =
            survivor_rows + num_survivors * decoded_stride;
        const auto bias = hn::Set(i8, int8_t{-128});
        for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
          const auto even = hn::LoadU(i8, source + chunk * 2 * lanes);
          const auto odd = hn::LoadU(i8, source + chunk * 2 * lanes + lanes);
          hn::StoreU(hn::BitCast(u8, hn::Xor(even, bias)), u8,
                     destination + chunk * 2 * lanes);
          hn::StoreU(hn::BitCast(u8, hn::Xor(odd, bias)), u8,
                     destination + chunk * 2 * lanes + lanes);
        }
        survivor_norms[num_survivors] = batch_norms[slot];
        if (is_l2) survivor_sqnorms[num_survivors] = batch_sqnorms[slot];
        ++num_survivors;
        out_survivors.push_back(tile_cands[slot]);
        slot_survived[slot] = true;
      }
    }

    if (keep_pruned_neighbors && out_survivors.size() < max_survivors &&
        workspace->pruned_candidates.size() < max_survivors) {
      for (size_t slot = 0; slot < num_slots; ++slot) {
        if (!slot_survived[slot]) {
          workspace->pruned_candidates.push_back(tile_cands[slot]);
        }
      }
    }
  }

  if (keep_pruned_neighbors && out_survivors.size() < max_survivors) {
    const size_t num_to_add = std::min(max_survivors - out_survivors.size(),
                                       workspace->pruned_candidates.size());
    out_survivors.insert(out_survivors.end(),
                         workspace->pruned_candidates.begin(),
                         workspace->pruned_candidates.begin() + num_to_add);
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace ginger

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace ginger {

HWY_EXPORT(AcceleratedRobustPruneImpl);

void AcceleratedRobustPrune(const TurboQuantDatasetSoA& quantized_points,
                            absl::Span<const GingerNodeId> candidates,
                            absl::Span<const float> distances_to_source,
                            GingerNodeId source_node,
                            const RobustPruneConfig& prune_config,
                            size_t max_survivors,
                            RobustPruneWorkspace* workspace,
                            std::vector<GingerNodeId>& out_survivors) {
  HWY_DYNAMIC_DISPATCH(AcceleratedRobustPruneImpl)
  (quantized_points, candidates, distances_to_source, source_node,
   prune_config.alpha(), max_survivors, workspace, out_survivors,
   prune_config.keep_pruned_neighbors());
}

}  // namespace ginger

#endif  // HWY_ONCE
