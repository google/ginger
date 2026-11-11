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

// Extra-precision (5/6/7/8/10/12-bit) SoA scoring kernel using Highway SIMD
// dynamic dispatch.
//
// This file is compiled once per SIMD target via foreach_target.h.

#include "ginger/turboquant/scoring/score_extra_precision_soa.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "ginger/turboquant/codebooks/centroid_tables.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/prefetch.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/types/distance_metric.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/scoring/score_extra_precision_soa.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
#include "ginger/turboquant/scoring/simd_helpers-inl.h"  // NOLINT
#include "ginger/turboquant/simd/vec_group-inl.h"        // NOLINT

namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
using ::ginger::turboquant_internal::PositiveCentroidTable;
using ::ginger::turboquant_internal::SignedCentroidTable;
using turboquant_internal::LoadPackedBits;
using turboquant_internal::LookupCentroid;

// Evaluates the scalar score contribution for a single dimension `d`.
template <int kBitsPerDim>
HWY_INLINE float ScoreSingleDim(const uint8_t* HWY_RESTRICT dp_quant,
                                const uint8_t* HWY_RESTRICT dp_extra, size_t d,
                                float query_val,
                                const SignedCentroidTable<kBitsPerDim>& table) {
  // Extract 4-bit base code: even dimensions use low nibble (shift 0),
  // odd dimensions use high nibble (shift 4).
  const int nibble_shift = (d % 2) * 4;
  const uint8_t base_code = (dp_quant[d / 2] >> nibble_shift) & 0x0F;

  // Extract extra precision sub-index from contiguous bitstream.
  const size_t bit_offset = d * kBitsPerDim;
  constexpr uint8_t kExtraMask = (1u << kBitsPerDim) - 1;

  // Unconditional 2-byte read is safe because
  // TurboQuantCodec::GetExtraPrecisionBytes allocates +1 padding byte whenever
  // extra_bits_per_dim > 0, avoiding branches.
  uint16_t two_bytes;
  memcpy(&two_bytes, dp_extra + (bit_offset / 8), sizeof(two_bytes));
  const uint8_t extra_val = (two_bytes >> (bit_offset % 8)) & kExtraMask;

  return query_val * table.Get(base_code, extra_val);
}

// SIMD version of `RestorePrefixCodes`. Writing `out` with a single store
// lets the subsequent code loads forward from it.
HWY_INLINE void RestorePrefixCodesSimd(const uint8_t* HWY_RESTRICT row,
                                       const uint8_t* HWY_RESTRICT low_bits,
                                       uint8_t* HWY_RESTRICT out) {
  const hn::CappedTag<uint8_t, kScaleBitBytes> d8;
  if constexpr (hn::MaxLanes(d8) < kScaleBitBytes) {
    RestorePrefixCodes(row, low_bits, out);
  } else {
    const auto v = hn::LoadU(d8, row);
    const auto lsb = hn::LoadMaskBits(d8, low_bits);
    const auto restored =
        hn::Or(hn::And(v, hn::Set(d8, 0x0F)),
               hn::And(hn::ShiftLeft<1>(v), hn::Set(d8, 0xE0)));
    hn::StoreU(hn::Or(restored, hn::IfThenElseZero(lsb, hn::Set(d8, 0x10))), d8,
               out);
  }
}

// Computes how many candidates ahead to prefetch so that prefetches lead
// scoring by a roughly constant amount of compute work (~300-400 ns, or ~2400
// equivalent 12-bit dimensions) regardless of dimensionality or bit-width. This
// hides DRAM latency on low dimensions while bounding the in-flight prefetched
// data on high dimensions so it fits in L1D cache.
template <int kBitsPerDim>
HWY_INLINE size_t ComputePrefetchDistance(size_t dimensionality,
                                          size_t unroll_factor) {
  // Target compute lead (~300-400 ns, enough to hide DRAM latency under load),
  // expressed as equivalent 12-bit dimensions of scoring work.
  constexpr size_t kTargetLeadDims = 2400;
  // Fixed per-candidate setup cost outside the main loop, in 12-bit dimensions.
  constexpr size_t kPerCandidateOverheadDims = 64;
  // Upper cap on prefetched candidates so small dimensions do not overflow
  // prefetch queues.
  constexpr size_t kMaxPrefetchDistance = 16;
  // Per-dimension compute cost relative to 10/12-bit piecewise-cubic scoring
  // (1.0x), scaled by `kCostScale = 10` for integer arithmetic:
  // - 5/6-bit (k <= 2): 1-2 register lookups -> 0.6x
  // - 7-bit   (k == 3): 4-register lookup tree -> 1.4x
  // - 8-bit   (k == 4): 8-register lookup tree -> 2.7x
  // - 10/12-bit (k >= 6): piecewise-cubic evaluation -> 1.0x
  constexpr size_t kCostScale = 10;
  constexpr size_t kRelativeCost = kBitsPerDim <= 2   ? 6
                                   : kBitsPerDim == 3 ? 14
                                   : kBitsPerDim == 4 ? 27
                                                      : 10;

  const size_t work_per_candidate =
      kPerCandidateOverheadDims * kCostScale + dimensionality * kRelativeCost;
  const size_t raw_distance =
      hwy::DivCeil(kTargetLeadDims * kCostScale, work_per_candidate);
  return std::clamp<size_t>(hwy::RoundUpTo(raw_distance, unroll_factor),
                            unroll_factor, kMaxPrefetchDistance);
}

// Extra Precision Scoring
//
// TurboQuant uses a base 4-bit quantization (8 positive buckets + sign bit).
// Extra precision adds `kBitsPerDim` additional bits per dimension (1, 2, 3, 4,
// 6, or 8), yielding 5-bit, 6-bit, 7-bit, 8-bit, 10-bit, or 12-bit total
// quantization. Each base bucket is subdivided into 2^`kBitsPerDim`
// sub-centroids with trained float values.
//
// Memory layout per datapoint:
// - Base codes: ceil(`dim`/2) bytes of nibble-packed 4-bit codes.
// - Extra bits: ceil(`dim` * `kBitsPerDim` / 8) bytes of contiguously
// bit-packed sub-indices. All bit widths (1, 2, 3, 4, 6, 8) use the same
// contiguous packing where each dimension's `kBitsPerDim` extra bits are stored
// consecutively.
//
// Scoring pipeline (per `N`-wide SIMD iteration):
// 1. Extract `N` base 4-bit codes from packed nibbles.
// 2. Extract `N` extra sub-indices from the packed bit stream.
// 3. Combine bucket + sub-index to look up float centroids.
// 4. Apply sign via XOR (for 1/2-bit paths; all k >= 3 paths use the signed
//    centroids table with the sign baked in).
// 5. Multiply by float query coordinates and accumulate.
//
// Precondition: `kBitsPerDim` must be in {1, 2, 3, 4, 6, 8}.
//
// With a 3-bit prefix, the full 4-bit codes of the first `kScaleBitDims`
// dimensions are restored per datapoint from the row and the
// `kPrefixLowBitBytes` stored after the packed extra bits.
template <TurboQuantDistanceMetric Metric, int kBitsPerDim,
          typename ScoresAndIndicesAccessor>
HWY_NOINLINE void ExtraPrecisionScoringImpl(
    const TurboQuantDatasetSoA& base_points, const TurboQuantQuery& query,
    ScoresAndIndicesAccessor scores_and_indices) {
  static_assert(kBitsPerDim >= 1 &&
                (kBitsPerDim <= 4 || kBitsPerDim == 6 || kBitsPerDim == 8));
  ABSL_DCHECK_GT(scores_and_indices.size(), 0);
  ABSL_DCHECK_GT(base_points.num_datapoints(), 0);
  constexpr int kCap = 16;
  using Float = hn::CappedTag<float, kCap>;
  using Int32 = hn::CappedTag<int32_t, kCap>;
  using UInt32 = hn::CappedTag<uint32_t, kCap>;
  const Float df;
  const Int32 di;
  const UInt32 du;
  const size_t N = hn::Lanes(df);
  // Code loads for d < kScaleBitDims must stay within the restored prefix.
  static_assert(kScaleBitDims % kCap == 0);

  static constexpr SignedCentroidTable<kBitsPerDim> signed_centroids{};
  static constexpr PositiveCentroidTable<(kBitsPerDim <= 2 ? kBitsPerDim : 1)>
      pos_centroids{};

  // Data pointers and layout
  const uint8_t* quantized_data = base_points.quantized_data();
  const size_t stride = base_points.stride();
  const size_t dimensionality = base_points.dimensionality();
  const float* query_data = query.float_query_data().data();
  constexpr float kRescaleScale =
      Metric == TurboQuantDistanceMetric::kSquaredL2Distance ? -2.0f : -1.0f;
  const float query_sqr_norm = query.unquantized_squared_norm();
  const NormStorageReader norms = base_points.NormReader();

  const size_t extra_bytes_per_point = base_points.extra_precision_stride();
  const size_t scale_offset = base_points.extra_precision_bytes();
  const bool has_3bit_prefix = base_points.has_3bit_prefix();
  const size_t low_bits_offset =
      has_3bit_prefix ? base_points.prefix_low_bits_offset() : 0;

  const uint8_t* extra_data_base = base_points.extra_precision_data();

  auto rescale_score = [&](size_t dp_idx, const uint8_t* dp_extra,
                           float raw_sum) HWY_ATTR {
    float scale_4plusk;
    std::memcpy(&scale_4plusk, dp_extra + scale_offset, sizeof(float));
    const float scaled = raw_sum * scale_4plusk * kRescaleScale;
    if constexpr (Metric == TurboQuantDistanceMetric::kSquaredL2Distance) {
      return norms.GetSquaredNorm(dp_idx) + scaled + query_sqr_norm;
    } else {
      return scaled;
    }
  };

  const size_t num_candidates = scores_and_indices.size();
  const size_t quant_bytes = base_points.num_bytes_per_datapoint();
  const size_t simd_end = hwy::RoundDownTo(dimensionality, N);

  constexpr size_t kUnrollFactor =
      (kBitsPerDim <= 3) ? 1 : (kBitsPerDim == 4 ? 8 : 4);
  const size_t prefetch_distance =
      ComputePrefetchDistance<kBitsPerDim>(dimensionality, kUnrollFactor);

  // Dimension stride for prefetching the next cache line of 4-bit base codes
  // (`next_quant`), since each byte packs 2 dimensions.
  constexpr size_t kCodesPrefetchDims = 2 * ABSL_CACHELINE_SIZE;
  // Dimension stride for prefetching the next cache line of `kBitsPerDim`-bit
  // extra-precision data (`next_extra`), using the largest power of two fitting
  // in a cache line so `d += N` hits every multiple and advances by at most one
  // cache line.
  constexpr size_t kExtraPrefetchDims =
      std::bit_floor<size_t>((ABSL_CACHELINE_SIZE * 8) / kBitsPerDim);

  // Helper to prefetch squared norms when stored outside the quantized row.
  const bool prefetch_norm =
      Metric == TurboQuantDistanceMetric::kSquaredL2Distance &&
      base_points.norm_layout() == NormLayout::kScalingAndSquaredNorm &&
      norms.mode() != ScalingStorageMode::kIn3BitPrefix;
  auto prefetch_norm_data = [&](size_t dp_idx) HWY_ATTR {
    if (prefetch_norm) {
      absl::PrefetchToLocalCache(norms.NormAddress(dp_idx));
    }
  };

  // Helper to prefetch the trailing cache-lines of `quant` and `extra` not
  // covered by the main loop.
  auto prefetch_row_tails = [&](const uint8_t* quant,
                                const uint8_t* extra) HWY_ATTR {
    absl::PrefetchToLocalCache(quant + quant_bytes -
                               std::min<size_t>(quant_bytes, 64));
    absl::PrefetchToLocalCache(quant + quant_bytes - 1);
    absl::PrefetchToLocalCache(extra + extra_bytes_per_point -
                               std::min<size_t>(extra_bytes_per_point, 64));
    absl::PrefetchToLocalCache(extra + extra_bytes_per_point - 1);
  };

  // Prefetch the initial window of candidates before starting the main loop.
  for (size_t p = 0; p < num_candidates && p < prefetch_distance; ++p) {
    const size_t dp_idx = scores_and_indices.Index(p);
    PrefetchDatapointHint<3>(quantized_data + dp_idx * stride, quant_bytes);
    PrefetchDatapointHint<3>(extra_data_base + dp_idx * extra_bytes_per_point,
                             extra_bytes_per_point);
    prefetch_norm_data(dp_idx);
  }

  size_t i = 0;
  for (; i + kUnrollFactor <= num_candidates; i += kUnrollFactor) {
    const uint8_t* HWY_RESTRICT dp_quant[kUnrollFactor];
    const uint8_t* HWY_RESTRICT dp_extra[kUnrollFactor];
    alignas(kScaleBitBytes) uint8_t prefix_codes[kUnrollFactor][kScaleBitBytes];
    for (size_t k = 0; k < kUnrollFactor; ++k) {
      const size_t idx = scores_and_indices.Index(i + k);
      dp_quant[k] = quantized_data + idx * stride;
      dp_extra[k] = extra_data_base + idx * extra_bytes_per_point;
      if (has_3bit_prefix) {
        RestorePrefixCodesSimd(dp_quant[k], dp_extra[k] + low_bits_offset,
                               prefix_codes[k]);
      }
    }

    // Pointers to the future batch (`i + prefetch_distance`) prefetched while
    // scoring this batch, clamped to `num_candidates - 1` to avoid branches.
    const uint8_t* next_quant[kUnrollFactor];
    const uint8_t* next_extra[kUnrollFactor];
    for (size_t k = 0; k < kUnrollFactor; ++k) {
      const size_t idx = scores_and_indices.Index(
          std::min(i + prefetch_distance + k, num_candidates - 1));
      next_quant[k] = quantized_data + idx * stride;
      next_extra[k] = extra_data_base + idx * extra_bytes_per_point;
      prefetch_norm_data(idx);
    }

    VEC_GROUP_DEFINE(df, acc);
    auto get_acc = VEC_GROUP_ACCESSOR(acc);
    ForIndex<kUnrollFactor>([&](auto k) { get_acc(k) = hn::Zero(df); });

    const uint8_t* codes_src[kUnrollFactor];
    for (size_t k = 0; k < kUnrollFactor; ++k) {
      codes_src[k] = has_3bit_prefix ? prefix_codes[k] : dp_quant[k];
    }
    for (size_t d = 0; d < simd_end; d += N) {
      if (d == kScaleBitDims) {
        for (size_t k = 0; k < kUnrollFactor; ++k) codes_src[k] = dp_quant[k];
      }

      if (d % kCodesPrefetchDims == 0) {
        for (size_t k = 0; k < kUnrollFactor; ++k) {
          absl::PrefetchToLocalCache(next_quant[k] + d / 2);
        }
      }
      if (d % kExtraPrefetchDims == 0) {
        for (size_t k = 0; k < kUnrollFactor; ++k) {
          absl::PrefetchToLocalCache(next_extra[k] + (d * kBitsPerDim) / 8);
        }
      }

      const auto q = hn::LoadU(df, query_data + d);
      ForIndex<kUnrollFactor>([&](auto k) {
        const auto codes = LoadPackedBits<4>(du, codes_src[k.value], d);
        const auto extra =
            LoadPackedBits<kBitsPerDim>(du, dp_extra[k.value], d);
        const auto c = LookupCentroid<kBitsPerDim>(df, du, di, codes, extra,
                                                   pos_centroids.table,
                                                   signed_centroids.table);
        get_acc(k) = hn::MulAdd(q, c, get_acc(k));
      });
    }

    for (size_t k = 0; k < kUnrollFactor; ++k) {
      prefetch_row_tails(next_quant[k], next_extra[k]);
    }

    float sum[kUnrollFactor] = {};
    ForIndex<kUnrollFactor>(
        [&](auto k) { sum[k.value] = hn::ReduceSum(df, get_acc(k)); });

    // simd_end >= kScaleBitDims, so the tail never reads the prefix.
    for (size_t d = simd_end; d < dimensionality; ++d) {
      const float q_val = query_data[d];
      for (size_t k = 0; k < kUnrollFactor; ++k) {
        sum[k] += ScoreSingleDim(dp_quant[k], dp_extra[k], d, q_val,
                                 signed_centroids);
      }
    }

    for (size_t k = 0; k < kUnrollFactor; ++k) {
      scores_and_indices.SetScore(
          i + k,
          rescale_score(scores_and_indices.Index(i + k), dp_extra[k], sum[k]));
    }
  }

  // Score any remaining candidates when `num_candidates` is not a multiple of
  // `kUnrollFactor` (we do not include any prefetching here as they already
  // prefetched by preceding batches).
  for (; i < num_candidates; ++i) {
    const size_t dp_idx = scores_and_indices.Index(i);
    const uint8_t* HWY_RESTRICT dp_quant = quantized_data + dp_idx * stride;
    const uint8_t* HWY_RESTRICT dp_extra =
        extra_data_base + dp_idx * extra_bytes_per_point;
    alignas(kScaleBitBytes) uint8_t prefix_codes[kScaleBitBytes];
    if (has_3bit_prefix) {
      RestorePrefixCodesSimd(dp_quant, dp_extra + low_bits_offset,
                             prefix_codes);
    }

    hn::Vec<Float> sum_vec = hn::Zero(df);

    // SIMD scoring part.
    const uint8_t* codes_src = has_3bit_prefix ? prefix_codes : dp_quant;
    for (size_t d = 0; d < simd_end; d += N) {
      if (d == kScaleBitDims) codes_src = dp_quant;
      const auto codes = LoadPackedBits<4>(du, codes_src, d);
      const auto extra = LoadPackedBits<kBitsPerDim>(du, dp_extra, d);
      const auto c = LookupCentroid<kBitsPerDim>(df, du, di, codes, extra,
                                                 pos_centroids.table,
                                                 signed_centroids.table);
      const auto q = hn::LoadU(df, query_data + d);
      sum_vec = hn::MulAdd(q, c, sum_vec);
    }

    float sum = hn::ReduceSum(df, sum_vec);

    // Scalar tail for remaining dimensions (`dimensionality` % `N` != 0).
    for (size_t d = simd_end; d < dimensionality; ++d) {
      sum += ScoreSingleDim(dp_quant, dp_extra, d, query_data[d],
                            signed_centroids);
    }

    scores_and_indices.SetScore(i, rescale_score(dp_idx, dp_extra, sum));
  }
}

// Dispatches to the correct `extra_bits_per_dim` specialization.
template <TurboQuantDistanceMetric Metric, typename ScoresAndIndicesAccessor>
void ExtraPrecisionScoringDispatch(
    const TurboQuantDatasetSoA& base_points, const TurboQuantQuery& query,
    ScoresAndIndicesAccessor scores_and_indices) {
  if (scores_and_indices.size() == 0 || base_points.num_datapoints() == 0) {
    return;
  }
  ABSL_QCHECK(!query.float_query_data().empty())
      << "Extra precision scoring requires float query data. "
         "Ensure the query was created with "
         "enable_higher_precision_scoring=true, or via "
         "CreateForExtraPrecision, "
         "CreateFromQuantizedDataPointWithExtraPrecision, or "
         "ResetFromQuantizedDataPointWithExtraPrecision.";
  switch (base_points.extra_bits_per_dim()) {
    case 1:
      ExtraPrecisionScoringImpl<Metric, 1>(base_points, query,
                                           scores_and_indices);
      break;
    case 2:
      ExtraPrecisionScoringImpl<Metric, 2>(base_points, query,
                                           scores_and_indices);
      break;
    case 3:
      ExtraPrecisionScoringImpl<Metric, 3>(base_points, query,
                                           scores_and_indices);
      break;
    case 4:
      ExtraPrecisionScoringImpl<Metric, 4>(base_points, query,
                                           scores_and_indices);
      break;
    case 6:
      ExtraPrecisionScoringImpl<Metric, 6>(base_points, query,
                                           scores_and_indices);
      break;
    case 8:
      ExtraPrecisionScoringImpl<Metric, 8>(base_points, query,
                                           scores_and_indices);
      break;
    default:
      ABSL_LOG(FATAL) << "Unsupported extra_bits_per_dim: "
                      << base_points.extra_bits_per_dim()
                      << ". Supported values are 1, 2, 3, 4, 6, 8.";
  }
}

void ExtraPrecisionScoringImplWithAccessor(
    const TurboQuantDatasetSoA& base_points, const TurboQuantQuery& query,
    ScoringAccessor accessor) {
  if (base_points.metric() == TurboQuantDistanceMetric::kSquaredL2Distance) {
    DISPATCH_SCORING_ACCESSOR(ExtraPrecisionScoringDispatch<
                                  TurboQuantDistanceMetric::kSquaredL2Distance>,
                              accessor, base_points, query);
  } else {
    DISPATCH_SCORING_ACCESSOR(
        ExtraPrecisionScoringDispatch<
            TurboQuantDistanceMetric::kDotProductDistance>,
        accessor, base_points, query);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace ginger {

HWY_EXPORT(ExtraPrecisionScoringImplWithAccessor);

void ComputeDistancesExtraPrecisionSoA(const TurboQuantDatasetSoA& base_points,
                                       const TurboQuantQuery& query,
                                       ScoringAccessor accessor) {
  HWY_DYNAMIC_DISPATCH(ExtraPrecisionScoringImplWithAccessor)
  (base_points, query, accessor);
}

}  // namespace ginger

#endif  // HWY_ONCE
