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

#ifndef GINGER_TURBOQUANT_LAYOUT_NORM_STORAGE_H_
#define GINGER_TURBOQUANT_LAYOUT_NORM_STORAGE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "hwy/base.h"

namespace ginger {

// How the per-datapoint norm data is stored.
enum class NormLayout {
  // One float16 per datapoint: the norm scaling factor.
  //
  // Used whenever no squared norm has to be stored, which covers both dot
  // product (whose score has no squared-norm term) and squared L2 over
  // normalized datapoints (whose squared norm is 1 by construction).
  kScalingOnly,

  // Two floats per datapoint, interleaved: the norm scaling factor followed by
  // the squared norm. Used for squared L2 over unnormalized datapoints.
  kScalingAndSquaredNorm,
};

// The layout a dataset with this metric and normalization has to use.
//
// Only squared L2 over unnormalized datapoints needs a stored squared norm.
// Everything else gets the compact float16 layout.
constexpr NormLayout NormLayoutFor(TurboQuantDistanceMetric metric,
                                   bool normalized_datapoints) {
  return (metric == TurboQuantDistanceMetric::kSquaredL2Distance &&
          !normalized_datapoints)
             ? NormLayout::kScalingAndSquaredNorm
             : NormLayout::kScalingOnly;
}

// Returns the number of trailing bytes of a quantized row that are reserved
// for the norm data embedded in the row itself.
constexpr size_t NormEntryBytes(NormLayout layout) {
  return layout == NormLayout::kScalingAndSquaredNorm ? 2 * sizeof(float)
                                                      : sizeof(hwy::float16_t);
}

// Only used for ScalingStorageMode::kIn3BitPrefix. Number of leading row bytes
// that carry one bit of the scaling factor, and the number of dimensions they
// cover.
inline constexpr size_t kScaleBitBytes = 8 * sizeof(hwy::float16_t);
inline constexpr size_t kScaleBitDims = 2 * kScaleBitBytes;

// Where a row's scaling factor is stored and read from during scoring.
enum class ScalingStorageMode {
  // Stored in the separate NormStorage side array (`norm_data_`), so scoring a
  // candidate touches a second cache line. Preferable when the coded bytes
  // exactly fill whole cache lines, where embedding the scale would add another
  // cache line per row.
  kSeparateArray,

  // Stored at the end of each quantized row (`row + stride -
  // NormEntryBytes(layout)`), either inside spare cache-line padding or by
  // appending `NormEntryBytes(layout)` bytes per datapoint. Avoids the second
  // cache miss. It is free when the padding has room, and without padding it
  // costs the same memory as the side array.
  kRowTail,

  // Stored in the MSBs of the row's first 16 bytes. The corresponding
  // dimensions are encoded with a 3-bit codebook instead of the usual 4-bit.
  kIn3BitPrefix,
};

// Computes the per-datapoint byte stride for `coded_bytes` of quantized codes.
// When `mode == ScalingStorageMode::kRowTail`, guarantees at least
// `NormEntryBytes(layout)` trailing bytes per row so the scaling factor can
// be stored alongside the codes.
constexpr size_t RowStrideFor(size_t coded_bytes, NormLayout layout,
                              bool pad_to_cache_line_size,
                              ScalingStorageMode mode) {
  const size_t min_bytes =
      coded_bytes +
      (mode == ScalingStorageMode::kRowTail ? NormEntryBytes(layout) : 0);
  if (!pad_to_cache_line_size) {
    return min_bytes;
  }
  return hwy::RoundUpTo(min_bytes, static_cast<size_t>(ABSL_CACHELINE_SIZE));
}

// Returns whether `kIn3BitPrefix` can hold the scaling factor. The prefix only
// has room for a 16-bit scale (kScalingAndSquaredNorm would need 64 bits over
// 64 dimensions) and needs at least `kScaleBitDims` dimensions.
constexpr bool CanUse3BitPrefix(NormLayout layout, size_t dimensionality) {
  return layout == NormLayout::kScalingOnly && dimensionality >= kScaleBitDims;
}

// Recommends a `ScalingStorageMode` for a dataset with `dimensionality`,
// `layout`, and `pad_to_cache_line_size`. It favors `kRowTail` (no second
// cache miss during scoring) whenever that does not cost extra memory
// compared with `kSeparateArray`:
//   - When `pad_to_cache_line_size` is true, returns `kRowTail` if the norm
//     entry fits in the spare cache-line padding, i.e. does not increase the
//     stride.
//   - When `pad_to_cache_line_size` is false, both modes need
//     `NormEntryBytes(layout)` per datapoint. Returns `kRowTail` unless
//     `coded_bytes` is 32-byte (half-cache-line) aligned, where the extra
//     bytes would make more rows straddle cache lines.
//   - Otherwise, returns `kIn3BitPrefix` if `allow_3bit_prefix` is true and
//     `CanUse3BitPrefix(layout, dimensionality)` (avoids cache-line splits by
//     encoding the first 16 odd dimensions with a 3-bit codebook).
//   - Otherwise, returns `kSeparateArray`.
constexpr ScalingStorageMode RecommendedScalingStorageMode(
    size_t dimensionality, NormLayout layout, bool pad_to_cache_line_size,
    bool allow_3bit_prefix) {
  const size_t coded_bytes = (dimensionality + 1) / 2;
  if (pad_to_cache_line_size) {
    if (RowStrideFor(coded_bytes, layout, /*pad_to_cache_line_size=*/true,
                     ScalingStorageMode::kRowTail) ==
        RowStrideFor(coded_bytes, layout, /*pad_to_cache_line_size=*/true,
                     ScalingStorageMode::kSeparateArray)) {
      return ScalingStorageMode::kRowTail;
    }
  } else {
    constexpr size_t kHalfCacheLineBytes = ABSL_CACHELINE_SIZE / 2;
    if (coded_bytes % kHalfCacheLineBytes != 0) {
      return ScalingStorageMode::kRowTail;
    }
  }
  if (allow_3bit_prefix && CanUse3BitPrefix(layout, dimensionality)) {
    return ScalingStorageMode::kIn3BitPrefix;
  }
  return ScalingStorageMode::kSeparateArray;
}

namespace norm_storage_internal {

// Returns a mask with `count` bits set, spaced `step` bits apart starting at
// `offset`.
constexpr uint64_t SpacedBits(int count, int step, int offset) {
  uint64_t mask = 0;
  for (int i = 0; i < count; ++i) {
    mask |= uint64_t{1} << (i * step + offset);
  }
  return mask;
}

// Gathers the MSBs of the 8 bytes at `ptr` into bits [7:0].
inline uint8_t GatherByteMsbs64(const uint8_t* ptr) {
  constexpr uint64_t kMsbMask = SpacedBits(8, /*step=*/8, /*offset=*/7);
  constexpr uint64_t kGatherMult = SpacedBits(8, /*step=*/7, /*offset=*/0);
  uint64_t w;
  std::memcpy(&w, ptr, sizeof(w));
  return static_cast<uint8_t>(((w & kMsbMask) * kGatherMult) >> 56);
}

// Scatters the low 4 bits of `nibble` into the MSBs of the
// 4 bytes at `ptr`. Bit j (0..3) shifted by 7*(j + 1) lands at bit (8*j + 7).
inline void ScatterNibbleToMsbs32(uint8_t* ptr, uint32_t nibble) {
  constexpr uint32_t kMsbMask =
      static_cast<uint32_t>(SpacedBits(4, /*step=*/8, /*offset=*/7));
  constexpr uint32_t kScatterMult =
      static_cast<uint32_t>(SpacedBits(4, /*step=*/7, /*offset=*/7));
  uint32_t w;
  std::memcpy(&w, ptr, sizeof(w));
  w = (w & ~kMsbMask) | (((nibble & 0x0Fu) * kScatterMult) & kMsbMask);
  std::memcpy(ptr, &w, sizeof(w));
}

}  // namespace norm_storage_internal

// Packs the 16 bits of `scaling_factor` (float16) into the MSB of each
// of the first 16 bytes (`kScaleBitBytes`) of `row` (1 bit per byte, used by
// `ScalingStorageMode::kIn3BitPrefix`).
inline void PackScaleBits(uint8_t* row, hwy::float16_t scaling_factor) {
  const uint16_t bits = hwy::BitCastScalar<uint16_t>(scaling_factor);
  for (int i = 0; i < 4; ++i) {
    norm_storage_internal::ScatterNibbleToMsbs32(row + 4 * i, bits >> (4 * i));
  }
}

// Inverse operation of `PackScaleBits`.
// Reconstructs the 16-bit float16 scaling factor that was spread across the
// MSBs of the first 16 bytes (`kScaleBitBytes`) of `row`.
inline float ReconstructScale(const uint8_t* row) {
  const uint16_t bits =
      static_cast<uint16_t>(norm_storage_internal::GatherByteMsbs64(row)) |
      (static_cast<uint16_t>(norm_storage_internal::GatherByteMsbs64(row + 8))
       << 8);
  return hwy::F32FromF16(hwy::BitCastScalar<hwy::float16_t>(bits));
}

// 4+k datapoints with a 3-bit prefix append the bucket LSB that the prefix
// drops from each odd dimension (bit b for byte b) to their extra-precision
// data, so that reranking can restore the full 4-bit codes.
inline constexpr size_t kPrefixLowBitBytes = kScaleBitBytes / 8;

// Writes the full 4-bit codes of the first kScaleBitBytes bytes of a 3-bit
// prefix `row` to `out`, using the `low_bits` kept in the extra-precision data.
// Inverse of `ApplyThreeBitPrefix` in encode_datapoint.cc.
inline void RestorePrefixCodes(const uint8_t* row, const uint8_t* low_bits,
                               uint8_t* out) {
  for (size_t b = 0; b < kScaleBitBytes; ++b) {
    const uint8_t lsb = (low_bits[b / 8] >> (b % 8)) & 1;
    // The 3-bit code in bits 4..6 moves to bits 5..7, above the restored LSB.
    out[b] = (row[b] & 0x0F) | ((row[b] << 1) & 0xE0) | (lsb << 4);
  }
}

// Writes a single datapoint's norm entry (`NormEntryBytes(layout)` bytes) at
// `dst`. Used by both `NormStorage` and `kRowTail` row-embedded norms.
inline void WriteNormEntry(uint8_t* dst, NormLayout layout,
                           float norm_scaling_factor, float sqr_norm) {
  if (layout == NormLayout::kScalingAndSquaredNorm) {
    const float vals[2] = {norm_scaling_factor, sqr_norm};
    std::memcpy(dst, vals, sizeof(vals));
  } else {
    const hwy::float16_t packed = hwy::F16FromF32(norm_scaling_factor);
    std::memcpy(dst, &packed, sizeof(packed));
  }
}

// Side-array storage for per-datapoint norm scaling factors (and squared norms
// for `NormLayout::kScalingAndSquaredNorm`) when not embedded in the row.
class NormStorage {
 public:
  NormStorage(size_t num_datapoints, NormLayout layout)
      : layout_(layout),
        entry_bytes_(NormEntryBytes(layout)),
        data_(num_datapoints * entry_bytes_) {}

  NormLayout layout() const { return layout_; }

  void Set(size_t index, float norm_scaling_factor, float sqr_norm) {
    WriteNormEntry(data_.data() + index * entry_bytes_, layout_,
                   norm_scaling_factor, sqr_norm);
  }

  void Resize(size_t num_datapoints) {
    data_.resize(num_datapoints * entry_bytes_, 0);
  }

  const uint8_t* data() const { return data_.data(); }

 private:
  NormLayout layout_;
  size_t entry_bytes_;
  std::vector<uint8_t> data_;
};

// A resolved, per-element view over a NormStorage or row-embedded norms.
class NormStorageReader {
 public:
  // For side-array storage (NormStorage or TurboSlice):
  explicit NormStorageReader(const NormStorage& storage)
      : layout_(storage.layout()),
        mode_(ScalingStorageMode::kSeparateArray),
        norm_base_(storage.data()),
        norm_stride_(NormEntryBytes(layout_)) {}

  // For row-embedded norms:
  NormStorageReader(NormLayout layout, const uint8_t* norm_base,
                    size_t norm_stride, ScalingStorageMode mode)
      : layout_(layout),
        mode_(mode),
        norm_base_(norm_base),
        norm_stride_(norm_stride) {}

  ScalingStorageMode mode() const { return mode_; }

  template <NormLayout kLayout>
  HWY_INLINE float GetNormScalingFactor(size_t index) const {
    ABSL_DCHECK(mode_ != ScalingStorageMode::kIn3BitPrefix);
    const uint8_t* p = norm_base_ + index * norm_stride_;
    if constexpr (kLayout == NormLayout::kScalingAndSquaredNorm) {
      float nscale;
      std::memcpy(&nscale, p, sizeof(nscale));
      return nscale;
    } else {
      hwy::float16_t packed;
      std::memcpy(&packed, p, sizeof(packed));
      return hwy::F32FromF16(packed);
    }
  }

  float GetNormScalingFactor(size_t index) const {
    if (mode_ == ScalingStorageMode::kIn3BitPrefix) {
      return ReconstructScale(norm_base_ + index * norm_stride_);
    }
    return layout_ == NormLayout::kScalingAndSquaredNorm
               ? GetNormScalingFactor<NormLayout::kScalingAndSquaredNorm>(index)
               : GetNormScalingFactor<NormLayout::kScalingOnly>(index);
  }

  // The squared norm of the unquantized datapoint, or 1 for kScalingOnly,
  // which does not store one. For squared L2 that is the true value, since
  // kScalingOnly is only chosen for L2 when the datapoints are normalized.
  // For dot product the squared norm plays no part in the score, so the value
  // is never read.
  template <NormLayout kLayout>
  HWY_INLINE float GetSquaredNorm(size_t index) const {
    if constexpr (kLayout != NormLayout::kScalingAndSquaredNorm) {
      return 1.0f;
    } else {
      const uint8_t* p = norm_base_ + index * norm_stride_;
      float sqnorm;
      std::memcpy(&sqnorm, p + sizeof(float), sizeof(sqnorm));
      return sqnorm;
    }
  }

  float GetSquaredNorm(size_t index) const {
    return layout_ == NormLayout::kScalingAndSquaredNorm
               ? GetSquaredNorm<NormLayout::kScalingAndSquaredNorm>(index)
               : 1.0f;
  }

  // Address of the norm data for `index`, for prefetching.
  const void* NormAddress(size_t index) const {
    return norm_base_ + index * norm_stride_;
  }

 private:
  const NormLayout layout_;
  const ScalingStorageMode mode_;
  const uint8_t* norm_base_ = nullptr;
  size_t norm_stride_ = 0;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_LAYOUT_NORM_STORAGE_H_
