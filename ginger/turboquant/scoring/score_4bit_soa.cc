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

// 4-bit SoA scoring kernel implementation using Highway SIMD dynamic dispatch.
//
// This file is compiled once per SIMD target via foreach_target.h.

#include "ginger/turboquant/scoring/score_4bit_soa.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/log/absl_check.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/prefetch.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/scoring/score_4bit_soa.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Processes a single lane of quantized datapoints using pre-loaded query
// vectors. Unpacks 4-bit quantized values, performs table lookup on centroids,
// and computes the dot product with query data.
//
// The odd nibbles get their own table because rows with packed scale bits
// encode the odd dimensions of their first bytes with the merged 3-bit
// codebook.
template <typename Int8, typename Uint8, typename Int32>
HWY_INLINE void AsymmetricDotProductLane(
    const hn::Vec<Uint8> mask, const hn::Vec<Uint8> even_centroids,
    const hn::Vec<Uint8> odd_centroids,
    const uint8_t* HWY_RESTRICT quantized_datapoint_lane_begin,
    const hn::Vec<Int8> query_even, const hn::Vec<Int8> query_odd,
    hn::Vec<Int32>& acc_even, hn::Vec<Int32>& acc_odd) {
  const Uint8 u8;
  const Int32 i32;

  const hn::Vec<Uint8> idxs = hn::LoadU(u8, quantized_datapoint_lane_begin);
  const hn::Vec<Uint8> even_idxs = hn::And(idxs, mask);
  const hn::Vec<Uint8> odd_idxs = hn::ShiftRight<4>(idxs);
  hn::Vec<Uint8> even_dp_vals = hn::TableLookupBytes(even_centroids, even_idxs);
  hn::Vec<Uint8> odd_dp_vals = hn::TableLookupBytes(odd_centroids, odd_idxs);

  acc_even =
      hn::SumOfMulQuadAccumulate(i32, even_dp_vals, query_even, acc_even);
  acc_odd = hn::SumOfMulQuadAccumulate(i32, odd_dp_vals, query_odd, acc_odd);
}

// Reduces an accumulator pair to a final float score.
template <typename Int32>
HWY_INLINE float ReduceScore(Int32 i32, hn::Vec<Int32> acc_even,
                             hn::Vec<Int32> acc_odd, float bias) {
  return static_cast<float>(hn::ReduceSum(i32, hn::Add(acc_even, acc_odd))) -
         bias;
}

// Iterates dimensions [start, end) in 2N-unrolled steps, loading query
// vectors and calling fn(offset, qe_vec, qo_vec) per SIMD lane.
template <typename Int8, typename Fn>
HWY_INLINE void ForEachDimLane(Int8 i8, const int8_t* HWY_RESTRICT query_even,
                               const int8_t* HWY_RESTRICT query_odd,
                               size_t start, size_t end, size_t N,
                               const Fn& fn) {
  size_t k = start;
  for (; k + 2 * N <= end; k += 2 * N) {
    fn(k, hn::Load(i8, query_even + k), hn::Load(i8, query_odd + k));
    fn(k + N, hn::Load(i8, query_even + k + N),
       hn::Load(i8, query_odd + k + N));
  }
  if (k < end) {
    fn(k, hn::Load(i8, query_even + k), hn::Load(i8, query_odd + k));
  }
}

// Reconstructs the 16-bit float16 scaling factor spread across bit 7 (the MSB)
// of each of the first 16 bytes of `row` (1 bit per byte, for kIn3BitPrefix).
// This SIMD version is faster than the scalar code in NormStorageReader.
HWY_INLINE float ReconstructScaleSimd(const uint8_t* HWY_RESTRICT row) {
  const hn::FixedTag<int8_t, 16> d16;
  const auto v = hn::LoadU(d16, reinterpret_cast<const int8_t*>(row));
  const auto mask = hn::Lt(v, hn::Zero(d16));
#if HWY_MAX_BYTES <= 64
  const uint16_t bits = static_cast<uint16_t>(hn::BitsFromMask(d16, mask));
#else
  // BitsFromMask is unavailable on SVE targets.
  uint8_t bytes[8];
  hn::StoreMaskBits(d16, mask, bytes);
  const uint16_t bits = static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
#endif
  return hwy::F32FromF16(hwy::BitCastScalar<hwy::float16_t>(bits));
}

template <bool kShouldPrefetch, typename ScoresAndIndicesAccessor,
          typename ScoreLanesFn, typename RescaleFn>
HWY_INLINE void ScoreUntiledDims(
    const uint8_t* HWY_RESTRICT quantized_data, size_t stride,
    size_t deinterleaved_end, const NormStorageReader& norms, float query_bias,
    const ScoreLanesFn& score_lanes, const RescaleFn& rescale_one,
    size_t num_datapoints, ScoresAndIndicesAccessor& scores_and_indices) {
  using Int32 = hn::ScalableTag<int32_t>;
  const Int32 i32;

  // kRowTail keeps the scale at the end of the row, so prefetch the full row.
  // Only kSeparateArray keeps the scale outside the row.
  const size_t prefetch_bytes = norms.mode() == ScalingStorageMode::kRowTail
                                    ? std::max(deinterleaved_end, stride)
                                    : deinterleaved_end;

  auto datapoint_ptr = [&](size_t i) HWY_ATTR {
    return quantized_data + stride * scores_and_indices.Index(i);
  };
  auto prefetch_datapoint = [&](size_t i) HWY_ATTR {
    if constexpr (kShouldPrefetch) {
      if (i < num_datapoints) {
        PrefetchDatapointHint<3>(datapoint_ptr(i), prefetch_bytes);
        if (norms.mode() == ScalingStorageMode::kSeparateArray) {
          PrefetchDatapointHint<3>(
              static_cast<const uint8_t*>(
                  norms.NormAddress(scores_and_indices.Index(i))),
              1);
        }
      }
    }
  };

  static constexpr size_t kPrefetchBytes = 2304;
  const size_t prefetch_datapoints =
      std::max<size_t>(8, kPrefetchBytes / prefetch_bytes);

  if constexpr (kShouldPrefetch) {
    for (size_t i = 0; i < num_datapoints && i < prefetch_datapoints; ++i) {
      prefetch_datapoint(i);
    }
  }

  for (size_t i = 0; i < num_datapoints; ++i) {
    prefetch_datapoint(i + prefetch_datapoints);
    const size_t dp_idx = scores_and_indices.Index(i);
    const uint8_t* HWY_RESTRICT datapoint_quantized =
        quantized_data + stride * dp_idx;
    hn::Vec<Int32> accumulator_even = hn::Zero(i32);
    hn::Vec<Int32> accumulator_odd = hn::Zero(i32);
    score_lanes(datapoint_quantized, 0, deinterleaved_end, accumulator_even,
                accumulator_odd);
    scores_and_indices.SetScore(
        i, rescale_one(dp_idx, datapoint_quantized,
                       ReduceScore(i32, accumulator_even, accumulator_odd,
                                   query_bias)));
  }
}

template <bool kShouldPrefetch, typename ScoresAndIndicesAccessor,
          typename ScoreLanesFn, typename RescaleFn>
HWY_INLINE void ScoreTiledDims(const uint8_t* HWY_RESTRICT quantized_data,
                               size_t stride, size_t deinterleaved_end,
                               const NormStorageReader& norms, float query_bias,
                               const ScoreLanesFn& score_lanes,
                               const RescaleFn& rescale_one,
                               size_t num_datapoints,
                               ScoresAndIndicesAccessor& scores_and_indices) {
  using Int32 = hn::ScalableTag<int32_t>;
  const Int32 i32;

  static constexpr size_t kTileBytes = 256;
  static constexpr size_t kPrefetchBytes = 2304;
  const size_t prefetch_datapoints = std::max<size_t>(
      8, hwy::RoundUpTo(kPrefetchBytes / kTileBytes, size_t{4}));

  // The scale is read after the last tile, so the last tile's prefetch also
  // covers the row tail (kRowTail) or the side-array entry (kSeparateArray).
  const size_t last_tile_prefetch_end =
      norms.mode() == ScalingStorageMode::kRowTail
          ? std::max(deinterleaved_end, stride)
          : deinterleaved_end;

  auto prefetch_tile = [&](size_t i, size_t tile_start) HWY_ATTR {
    const size_t dp_idx = scores_and_indices.Index(i);
    const bool is_last_tile = tile_start + kTileBytes >= deinterleaved_end;
    const size_t tile_end =
        is_last_tile ? last_tile_prefetch_end : tile_start + kTileBytes;
    PrefetchDatapointHint<0>(quantized_data + stride * dp_idx + tile_start,
                             tile_end - tile_start);
    if (is_last_tile && norms.mode() == ScalingStorageMode::kSeparateArray) {
      PrefetchDatapointHint<0>(
          static_cast<const uint8_t*>(norms.NormAddress(dp_idx)), 1);
    }
  };

  // Initialize scores to -query_bias; IncScore will accumulate tile results.
  for (size_t i = 0; i < num_datapoints; ++i) {
    scores_and_indices.SetScore(i, -query_bias);
  }

  // Prefetch first tile of initial datapoints.
  if constexpr (kShouldPrefetch) {
    for (size_t i = 0; i < num_datapoints && i < prefetch_datapoints; ++i) {
      prefetch_tile(i, 0);
    }
  }

  // Dim-outer loop: for each tile, score all DPs.
  for (size_t dim_offset = 0; dim_offset < deinterleaved_end;
       dim_offset += kTileBytes) {
    const bool is_last_tile = dim_offset + kTileBytes >= deinterleaved_end;
    const size_t tile_end =
        std::min(dim_offset + kTileBytes, deinterleaved_end);

    for (size_t i = 0; i < num_datapoints; ++i) {
      if constexpr (kShouldPrefetch) {
        if (i + prefetch_datapoints < num_datapoints) {
          prefetch_tile(i + prefetch_datapoints, dim_offset);
        } else if (!is_last_tile) {
          prefetch_tile((i + prefetch_datapoints) % num_datapoints,
                        dim_offset + kTileBytes);
        }
      }

      const size_t dp_idx = scores_and_indices.Index(i);
      const uint8_t* HWY_RESTRICT datapoint_quantized =
          quantized_data + stride * dp_idx;
      hn::Vec<Int32> acc_even = hn::Zero(i32);
      hn::Vec<Int32> acc_odd = hn::Zero(i32);
      score_lanes(datapoint_quantized, dim_offset, tile_end, acc_even, acc_odd);
      const float tile_score = ReduceScore(i32, acc_even, acc_odd, 0.0f);
      if (is_last_tile) {
        scores_and_indices.SetScore(
            i, rescale_one(dp_idx, datapoint_quantized,
                           scores_and_indices.GetScore(i) + tile_score));
      } else {
        scores_and_indices.IncScore(i, tile_score);
      }
    }
  }
}

template <bool kShouldPrefetch, bool kHas3BitPrefix, NormLayout kLayout,
          typename ScoresAndIndicesAccessor>
HWY_NOINLINE void TurboQuantScoringInt8BatchDimsImpl(
    const TurboQuantDatasetSoA& base_points, const TurboQuantQuery& query,
    bool is_turboslice_scoring, ScoresAndIndicesAccessor scores_and_indices) {
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int8 = hn::ScalableTag<int8_t>;
  using Int32 = hn::ScalableTag<int32_t>;
  using ginger::turboquant_codebooks::kTurboQuant3BitCentroidsUint8;
  using ginger::turboquant_codebooks::kTurboQuantCentroidsUint8;
  const Int8 i8;
  const Uint8 u8;
  const size_t N = hn::Lanes(i8);

  const uint8_t* HWY_RESTRICT quantized_data = base_points.quantized_data();
  const size_t stride = base_points.stride();
  const size_t num_bytes_per_datapoint =
      is_turboslice_scoring
          ? base_points.params().turboslice_bytes_per_datapoint()
          : base_points.num_bytes_per_datapoint();
  // The last SIMD vector may read past the coded bytes, e.g. into a kRowTail
  // norm entry. This is harmless because the query is zero-padded beyond the
  // coded bytes, so those bytes contribute 0 to the dot product.
  const size_t deinterleaved_end = hwy::RoundUpTo(num_bytes_per_datapoint, N);
  const size_t num_datapoints = scores_and_indices.size();

  const int8_t* HWY_RESTRICT query_data = query.query_data();
  const int8_t* HWY_RESTRICT query_even = query_data;
  const int8_t* HWY_RESTRICT query_odd =
      query_data + query.num_bytes_per_query() / 2;
  const float query_bias = static_cast<float>(128 * query.query_byte_sum());

  const hn::Vec<Uint8> centroids =
      hn::LoadDup128(u8, kTurboQuantCentroidsUint8.data());
  const hn::Vec<Uint8> mask = hn::Set(u8, 0xF);
  static_assert(kScaleBitBytes == 16);
  if constexpr (kHas3BitPrefix) {
    ABSL_DCHECK_GE(N, kScaleBitBytes);
  }
  const hn::Vec<Uint8> first_odd_centroids =
      kHas3BitPrefix
          ? hn::IfThenElse(
                hn::FirstN(u8, kScaleBitBytes),
                hn::LoadDup128(u8, kTurboQuant3BitCentroidsUint8.data()),
                centroids)
          : centroids;
  auto score_lanes = [&](const uint8_t* HWY_RESTRICT dp, size_t start,
                         size_t end, auto& acc_even, auto& acc_odd) HWY_ATTR {
    if constexpr (kHas3BitPrefix) {
      if (start == 0) {
        AsymmetricDotProductLane<Int8, Uint8, Int32>(
            mask, centroids, first_odd_centroids, dp, hn::Load(i8, query_even),
            hn::Load(i8, query_odd), acc_even, acc_odd);
        start = N;
      }
    }
    ForEachDimLane(i8, query_even, query_odd, start, end, N,
                   [&](size_t k, auto query_even_vec, auto query_odd_vec)
                       HWY_ATTR {
                         AsymmetricDotProductLane<Int8, Uint8, Int32>(
                             mask, centroids, centroids, dp + k, query_even_vec,
                             query_odd_vec, acc_even, acc_odd);
                       });
  };

  // Transforms raw integer-reduced dot products into distances:
  //   - Dot product: -raw * dp_scale * query_scale
  //   - Squared L2:  dp_sq_norm - 2*raw*dp_scale*query_scale + query_sq_norm
  //
  // For normalized squared L2 (`kScalingOnly`), `dp_sq_norm` is implicitly 1
  // and folded into `rescale_base_offset`.
  constexpr bool kHasStoredSqNorm =
      kLayout == NormLayout::kScalingAndSquaredNorm;
  const NormStorageReader norms = is_turboslice_scoring
                                      ? base_points.TurboSliceNormReader()
                                      : base_points.NormReader();
  const bool reconstruct_3bit_scale =
      norms.mode() == ScalingStorageMode::kIn3BitPrefix;

  const bool is_squared_l2 =
      base_points.metric() == TurboQuantDistanceMetric::kSquaredL2Distance;
  const float rescale_scale =
      (is_squared_l2 ? -2.0f : -1.0f) * query.norm_scaling_factor();
  const float rescale_base_offset =
      is_squared_l2
          ? (kHasStoredSqNorm ? query.unquantized_squared_norm()
                              : 1.0f + query.unquantized_squared_norm())
          : 0.0f;

  auto rescale_one = [&](size_t idx, const uint8_t* dp, float raw) HWY_ATTR {
    if constexpr (kHas3BitPrefix) {
      if (reconstruct_3bit_scale) {
        return raw * ReconstructScaleSimd(dp) * rescale_scale +
               rescale_base_offset;
      }
    }
    const float scale = norms.GetNormScalingFactor<kLayout>(idx);
    if constexpr (kHasStoredSqNorm) {
      return raw * scale * rescale_scale +
             (norms.GetSquaredNorm<kLayout>(idx) + rescale_base_offset);
    } else {
      return raw * scale * rescale_scale + rescale_base_offset;
    }
  };

  static constexpr int kDimTilingThreshold = 768;
  // This gates the tiled implementation to be used only on
  // zen3 and other AVX2 machines. Untiled is fastest on skylake, SPR, zen5.
  if (query.dimensionality() <= kDimTilingThreshold || N == 64) {
    ScoreUntiledDims<kShouldPrefetch>(
        quantized_data, stride, deinterleaved_end, norms, query_bias,
        score_lanes, rescale_one, num_datapoints, scores_and_indices);
  } else {
    // TODO: figure out how to make tiled fast on SPR and
    // icelake.
    ScoreTiledDims<kShouldPrefetch>(quantized_data, stride, deinterleaved_end,
                                    norms, query_bias, score_lanes, rescale_one,
                                    num_datapoints, scores_and_indices);
  }
}

template <typename ScoresAndIndicesAccessor>
HWY_INLINE void Score4BitRows(const TurboQuantDatasetSoA& base_points,
                              const TurboQuantQuery& query,
                              bool is_turboslice_scoring,
                              ScoresAndIndicesAccessor scores_and_indices) {
  if (base_points.has_3bit_prefix()) {
    ABSL_DCHECK(base_points.norm_layout() == NormLayout::kScalingOnly);
    TurboQuantScoringInt8BatchDimsImpl<
        /*kShouldPrefetch=*/true, /*kHas3BitPrefix=*/true,
        NormLayout::kScalingOnly>(base_points, query, is_turboslice_scoring,
                                  scores_and_indices);
    return;
  }
  if (base_points.norm_layout() == NormLayout::kScalingAndSquaredNorm) {
    TurboQuantScoringInt8BatchDimsImpl<
        /*kShouldPrefetch=*/true, /*kHas3BitPrefix=*/false,
        NormLayout::kScalingAndSquaredNorm>(
        base_points, query, is_turboslice_scoring, scores_and_indices);
    return;
  }
  TurboQuantScoringInt8BatchDimsImpl<
      /*kShouldPrefetch=*/true, /*kHas3BitPrefix=*/false,
      NormLayout::kScalingOnly>(base_points, query, is_turboslice_scoring,
                                scores_and_indices);
}

void TurboQuantScoringInt8BatchDims(const TurboQuantDatasetSoA& base_points,
                                    const TurboQuantQuery& query,
                                    ScoringAccessor accessor,
                                    bool is_turboslice_scoring) {
  DISPATCH_SCORING_ACCESSOR(Score4BitRows, accessor, base_points, query,
                            is_turboslice_scoring);
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {
namespace {
HWY_EXPORT(TurboQuantScoringInt8BatchDims);
}  // namespace

void ComputeDistances4BitSoA(const TurboQuantDatasetSoA& base_points,
                             const TurboQuantQuery& query,
                             ScoringAccessor accessor,
                             bool is_turboslice_scoring) {
  HWY_DYNAMIC_DISPATCH(TurboQuantScoringInt8BatchDims)(
      base_points, query, accessor, is_turboslice_scoring);
}

}  // namespace ginger
#endif
