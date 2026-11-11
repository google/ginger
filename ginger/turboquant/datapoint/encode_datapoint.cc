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

#include "ginger/turboquant/datapoint/encode_datapoint.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/codebooks/codebooks_10bit.h"
#include "ginger/turboquant/codebooks/codebooks_12bit.h"
#include "ginger/turboquant/codebooks/codebooks_8bit.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/rotation/rotation.h"
#include "ginger/turboquant/types/quantization_bits.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/datapoint/encode_datapoint.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace HWY_NAMESPACE {

namespace {

namespace hn = hwy::HWY_NAMESPACE;

float FourBitEncodeImpl(const float* HWY_RESTRICT coordinates,
                        size_t dimensionality, uint8_t* HWY_RESTRICT output) {
  using F32 = hn::ScalableTag<float>;
  const F32 f32;
  const size_t N = hn::Lanes(f32);
  using ginger::turboquant_codebooks::kBoundaries;
  using ginger::turboquant_codebooks::kSquaredCentroids;

  const auto b0 = hn::Set(f32, kBoundaries[0]);
  const auto b1 = hn::Set(f32, kBoundaries[1]);
  const auto b2 = hn::Set(f32, kBoundaries[2]);
  const auto b3 = hn::Set(f32, kBoundaries[3]);
  const auto b4 = hn::Set(f32, kBoundaries[4]);
  const auto b5 = hn::Set(f32, kBoundaries[5]);
  const auto b6 = hn::Set(f32, kBoundaries[6]);
  const auto zero = hn::Zero(f32);
  const auto one = hn::Set(f32, 1.0f);
  const auto two = hn::Set(f32, 2.0f);
  const auto four = hn::Set(f32, 4.0f);
  const auto eight = hn::Set(f32, 8.0f);

  const auto c0 = hn::Set(f32, kSquaredCentroids[0]);
  const auto c1 = hn::Set(f32, kSquaredCentroids[1]);
  const auto c2 = hn::Set(f32, kSquaredCentroids[2]);
  const auto c3 = hn::Set(f32, kSquaredCentroids[3]);
  const auto c4 = hn::Set(f32, kSquaredCentroids[4]);
  const auto c5 = hn::Set(f32, kSquaredCentroids[5]);
  const auto c6 = hn::Set(f32, kSquaredCentroids[6]);
  const auto c7 = hn::Set(f32, kSquaredCentroids[7]);

  const auto compute_codes_and_norms =
      [&](const auto& coords, hn::Vec<F32>& out_code, hn::Vec<F32>& out_sq) {
        const auto abs_v = hn::Abs(coords);
        auto code = hn::IfThenElse(hn::Gt(coords, zero), zero, eight);

        const auto m_L1 = hn::Ge(abs_v, b3);

        const auto b_L2 = hn::IfThenElse(m_L1, b5, b1);
        const auto m_L2 = hn::Ge(abs_v, b_L2);

        const auto b_L3 = hn::IfThenElse(m_L1, hn::IfThenElse(m_L2, b6, b4),
                                         hn::IfThenElse(m_L2, b2, b0));
        const auto m_L3 = hn::Ge(abs_v, b_L3);

        code = hn::Add(code, hn::IfThenElse(m_L1, four, zero));
        code = hn::Add(code, hn::IfThenElse(m_L2, two, zero));
        code = hn::Add(code, hn::IfThenElse(m_L3, one, zero));

        const auto sq0 = hn::IfThenElse(m_L3, c1, c0);
        const auto sq1 = hn::IfThenElse(m_L3, c3, c2);
        const auto sq2 = hn::IfThenElse(m_L3, c5, c4);
        const auto sq3 = hn::IfThenElse(m_L3, c7, c6);

        const auto sq_low = hn::IfThenElse(m_L2, sq1, sq0);
        const auto sq_high = hn::IfThenElse(m_L2, sq3, sq2);

        out_code = code;
        out_sq = hn::IfThenElse(m_L1, sq_high, sq_low);
      };

  auto sqr_norm_acc = hn::Zero(f32);

  using U32 = hn::ScalableTag<uint32_t>;
  using I32 = hn::ScalableTag<int32_t>;
  const U32 u32;
  const I32 i32;

  using U16From32 = hn::Rebind<uint16_t, U32>;
  using U8From16 = hn::Rebind<uint8_t, U16From32>;
  const U16From32 u16from32;
  const U8From16 u8from16;

  const auto pack_nibbles = [&](auto v_i32) {
    auto shifted =
        hn::ShiftLeft<4>(hn::BitCast(u32, hn::SlideDownLanes(i32, v_i32, 1)));
    return hn::Or(hn::BitCast(u32, v_i32), shifted);
  };

  for (size_t d = 0; d < dimensionality; d += 2 * N) {
    const size_t remaining = dimensionality - d;
    const bool is_tail = remaining < 2 * N;

    const auto va = is_tail
                        ? hn::LoadN(f32, coordinates + d, HWY_MIN(remaining, N))
                        : hn::LoadU(f32, coordinates + d);
    hn::Vec<F32> codes_a = hn::Zero(f32), sq_a = hn::Zero(f32);
    compute_codes_and_norms(va, codes_a, sq_a);

    const size_t lanes_b = remaining > N ? HWY_MIN(remaining - N, N) : 0;
    const auto vb = !is_tail      ? hn::LoadU(f32, coordinates + d + N)
                    : lanes_b > 0 ? hn::LoadN(f32, coordinates + d + N, lanes_b)
                                  : hn::Zero(f32);
    hn::Vec<F32> codes_b = hn::Zero(f32), sq_b = hn::Zero(f32);
    compute_codes_and_norms(vb, codes_b, sq_b);

    if (is_tail) {
      if (remaining < N) {
        sq_a = hn::IfThenElseZero(hn::FirstN(f32, remaining), sq_a);
      }
      sq_b = hn::IfThenElseZero(hn::FirstN(f32, lanes_b), sq_b);
    }

    sqr_norm_acc = hn::Add(sqr_norm_acc, hn::Add(sq_a, sq_b));

    const auto ca = hn::ConvertTo(i32, codes_a);
    const auto cb = hn::ConvertTo(i32, codes_b);
    const auto packed = hn::ConcatEven(u32, pack_nibbles(cb), pack_nibbles(ca));
    const auto packed_u16 = hn::TruncateTo(u16from32, packed);
    const auto packed_u8 = hn::TruncateTo(u8from16, packed_u16);

    if (is_tail) {
      hn::StoreN(packed_u8, u8from16, output + d / 2, (remaining + 1) / 2);
    } else {
      hn::StoreU(packed_u8, u8from16, output + d / 2);
    }
  }

  return hn::ReduceSum(f32, sqr_norm_acc);
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {
HWY_EXPORT(FourBitEncodeImpl);

namespace {

// Debiasing factors for 4+k bit encoding.
constexpr std::array<float, 8> kDebiasing4PlusKBit = {
    1.0012649f, 1.0003301f, 1.0000854f,               // k=1, 2, 3
    1.0f,       1.0f,       1.0f,       1.0f, 1.0f};  // k=4, 5, 6, 7, 8

// Correction of `debiasing_4bit` when `num_3bit_dims` of the `dimensionality`
// coordinates use the merged 3-bit codebook
// (ScalingStorageMode::kIn3BitPrefix).
//
// Let x be the rotated unit vector, whose coordinates are approximately
// N(0, 1/d), and x^ its reconstruction from conditional-mean centroids. The
// centroid condition gives E[x_i x^_i] = E[x^_i^2] = (1 - D_i) / d, where D_i
// is the relative MSE of the codebook of coordinate i (see `k4BitDistortion`).
// Averaged over the random rotation, the scaled estimate
// <q, x^> / ||x^|| then shrinks <q, x> by
//
//   E[<x, x^>] / sqrt(E[||x^||^2]) = sqrt(1 - D),  D = (1 / d) sum_i D_i.
//
// `debiasing_4bit` undoes this for D = D_4 (1 / sqrt(1 - D_4) = 1.00479).
// With m = `num_3bit_dims` coordinates on the merged codebook,
//
//   D_mix = (m D_3 + (d - m) D_4) / d,
//
// so the shrinkage changes by sqrt((1 - D_mix) / (1 - D_4)) relative to the
// 4-bit baseline, and the returned factor is its inverse. For
// m = `kScaleBitBytes` it is 1.0032 at d = 64, 1.0016 at d = 128 and 1.0003 at
// d = 768.
float ThreeBitPrefixDebiasingCorrection(size_t dimensionality,
                                        size_t num_3bit_dims) {
  using turboquant_codebooks::k4BitDistortion;
  using turboquant_codebooks::kMerged3BitDistortion;
  const float d = static_cast<float>(dimensionality);
  const float m = static_cast<float>(num_3bit_dims);
  const float mixed_distortion =
      (m * kMerged3BitDistortion + (d - m) * k4BitDistortion) / d;
  return std::sqrt((1.0f - k4BitDistortion) / (1.0f - mixed_distortion));
}

// Debiasing factor for 4-bit encoding: 1 / sqrt(1 - D_4) (see
// `ThreeBitPrefixDebiasingCorrection`).
float debiasing_4bit(size_t dimensionality, bool has_3bit_prefix) {
  const float base = 1.0047848f;
  if (!has_3bit_prefix) return base;
  return base *
         ThreeBitPrefixDebiasingCorrection(
             dimensionality, std::min(dimensionality / 2, kScaleBitBytes));
}

// Returns the 3 lowest bits of `x`.
uint8_t MaskBucket(const uint8_t x) { return x & 7; }

struct FourPlusKBitResult {
  uint8_t base_code;
  uint8_t sub_index;
  float centroid_sq_value_4bit;
  float centroid_sq_value_4plusk;
};

FourPlusKBitResult FourPlusKBitEncoding(float value, int extra_bits_per_dim) {
  const uint8_t four_bit_encoding =
      turboquant_codebooks::FourBitEncoding(value);
  const float centroid_sq_value_4bit =
      turboquant_codebooks::kSquaredCentroids[MaskBucket(four_bit_encoding)];
  if (extra_bits_per_dim == 0) {
    return FourPlusKBitResult{
        .base_code = four_bit_encoding,
        .sub_index = 0,
        .centroid_sq_value_4bit = centroid_sq_value_4bit,
        .centroid_sq_value_4plusk = centroid_sq_value_4bit};
  }

  const float abs_value = std::abs(value);
  int base_idx = MaskBucket(four_bit_encoding);

  absl::Span<const float> boundaries;
  absl::Span<const float> centroids;
  if (extra_bits_per_dim == 1) {
    boundaries = turboquant_codebooks::kBoundaries4Plus1Bit[base_idx];
    centroids = turboquant_codebooks::kCentroids4Plus1BitPos[base_idx];
  } else if (extra_bits_per_dim == 2) {
    boundaries = turboquant_codebooks::kBoundaries4Plus2Bit[base_idx];
    centroids = turboquant_codebooks::kCentroids4Plus2BitPos[base_idx];
  } else if (extra_bits_per_dim == 3) {
    boundaries = turboquant_codebooks::kBoundaries4Plus3Bit[base_idx];
    centroids = turboquant_codebooks::kCentroids4Plus3BitPos[base_idx];
  } else if (extra_bits_per_dim == 4) {
    boundaries = turboquant_codebooks::kBoundaries4Plus4Bit[base_idx];
    centroids = turboquant_codebooks::kCentroids4Plus4BitPos[base_idx];
  } else if (extra_bits_per_dim == 6) {
    boundaries = turboquant_codebooks::kBoundaries4Plus6Bit[base_idx];
    centroids = turboquant_codebooks::kCentroids4Plus6BitFitted[base_idx];
  } else {
    boundaries = turboquant_codebooks::kBoundaries4Plus8Bit[base_idx];
    centroids = turboquant_codebooks::kCentroids4Plus8BitFitted[base_idx];
  }

  const uint8_t sub_idx =
      std::upper_bound(boundaries.begin(), boundaries.end(), abs_value) -
      boundaries.begin();
  const float best_abs_centroid = centroids[sub_idx];

  return FourPlusKBitResult{
      .base_code = four_bit_encoding,
      .sub_index = sub_idx,
      .centroid_sq_value_4bit = centroid_sq_value_4bit,
      .centroid_sq_value_4plusk = best_abs_centroid * best_abs_centroid};
}

// Re-encodes the odd dimension (high nibble) of each of the first
// `kScaleBitBytes` bytes of `codes` with the merged 3-bit codebook. Shifting
// the 4-bit code `code4 = (sign << 3) | bucket` right by 1 moves `(sign << 2) |
// (bucket >> 1)` into bits 4-6 (dropping the bucket LSB in bit 4) and leaves
// bit 7 as the don't-care bit written by `PackScaleBits`. `quantized_sqr_norm`
// must be initialized to the 4-bit quantized squared norm of `codes` and is
// updated in place to swap the 4-bit centroids for the merged ones so the
// scaling factor matches the stored codes. Returns the dropped bucket LSBs
// (bit b for byte b). `RestorePrefixCodes` in norm_storage.h is the inverse.
uint16_t ApplyThreeBitPrefix(absl::Span<uint8_t> codes,
                             float& quantized_sqr_norm) {
  ABSL_DCHECK_GE(codes.size(), kScaleBitBytes);
  uint16_t low_bits = 0;
  for (size_t b = 0; b < kScaleBitBytes; ++b) {
    const uint8_t odd = codes[b] >> 4;
    quantized_sqr_norm +=
        turboquant_codebooks::kMerged3BitSquaredCentroids[(odd & 7) >> 1] -
        turboquant_codebooks::kSquaredCentroids[odd & 7];
    low_bits |= static_cast<uint16_t>(odd & 1) << b;
    codes[b] = (codes[b] & 0x0F) | ((odd >> 1) << 4);
  }
  return low_bits;
}

absl::StatusOr<std::pair<float, float>> EncodeDatapoint4PlusKBits(
    absl::Span<const float> rotated_coordinates,
    absl::Span<uint8_t> output_coordinates, int extra_bits_per_dim,
    absl::Span<uint8_t> output_extra_precision_data) {
  const int dimension = rotated_coordinates.size();
  if (extra_bits_per_dim > 0) {
    std::fill(output_extra_precision_data.begin(),
              output_extra_precision_data.end(), 0);
  }

  float quantized_sqr_norm_4bit = 0.0;
  float quantized_sqr_norm_4plusk = 0.0;
  int current_bit = 0;
  for (int i = 0; i < dimension; ++i) {
    auto result =
        FourPlusKBitEncoding(rotated_coordinates[i], extra_bits_per_dim);

    if (i % 2 == 0) {
      output_coordinates[i / 2] = result.base_code;
    } else {
      output_coordinates[i / 2] |= (result.base_code << 4);
    }

    if (extra_bits_per_dim > 0) {
      // All `extra_bits_per_dim` values (1, 2, 3, 4, 6, 8) use contiguous
      // packing. With +1 padding byte guaranteed, we unconditionally write both
      // bytes.
      int bit_offset = current_bit % 8;
      int byte_idx = current_bit / 8;
      uint16_t val = static_cast<uint16_t>(result.sub_index) << bit_offset;
      output_extra_precision_data[byte_idx] |= static_cast<uint8_t>(val);
      output_extra_precision_data[byte_idx + 1] |=
          static_cast<uint8_t>(val >> 8);
      current_bit += extra_bits_per_dim;
    }

    quantized_sqr_norm_4bit += result.centroid_sq_value_4bit;
    quantized_sqr_norm_4plusk += result.centroid_sq_value_4plusk;
  }
  return std::make_pair(quantized_sqr_norm_4bit, quantized_sqr_norm_4plusk);
}

}  // namespace

size_t GetExtraPrecisionBytes(size_t dimensionality, int extra_bits_per_dim) {
  // Add 1 padding byte whenever extra_bits_per_dim > 0 to enable unaligned
  // 2-byte scalar extractions without branching or buffer overreads.
  const size_t padding = (extra_bits_per_dim > 0) ? 1 : 0;
  return hwy::DivCeil(dimensionality * extra_bits_per_dim, 8) + padding;
}

size_t GetExtraPrecisionBytes(size_t dimensionality, int extra_bits_per_dim,
                              bool has_3bit_prefix) {
  const size_t bytes =
      GetExtraPrecisionBytes(dimensionality, extra_bits_per_dim);
  return (has_3bit_prefix && extra_bits_per_dim > 0)
             ? bytes + kPrefixLowBitBytes
             : bytes;
}

namespace internal {

absl::StatusOr<EncodeDatapointResult> EncodeDatapointStructured(
    absl::Span<const float> coordinates, absl::Span<uint8_t> output_coordinates,
    int quantization_bits, absl::Span<uint8_t> output_extra_precision_data,
    size_t dimensionality, size_t num_bytes_per_datapoint,
    size_t turboslice_dimensionality, const HadamardAndMatrixRotation& rotation,
    bool has_3bit_prefix) {
  if (coordinates.size() != dimensionality) {
    return absl::InvalidArgumentError(
        "Datapoint dimensionality does not match dimensionality.");
  }
  if (turboslice_dimensionality > dimensionality) {
    return absl::InvalidArgumentError(
        "TurboSlice dimensionality cannot exceed dimensionality.");
  }
  if (num_bytes_per_datapoint != (dimensionality + 1) / 2) {
    return absl::InvalidArgumentError(
        "Number of bytes per datapoint does not match dimensionality.");
  }
  if (output_coordinates.size() != num_bytes_per_datapoint) {
    return absl::InvalidArgumentError(
        "Output coordinates size does not match number of bytes per "
        "datapoint.");
  }
  if (has_3bit_prefix && dimensionality < kScaleBitDims) {
    return absl::InvalidArgumentError(
        "3-bit prefix encoding needs at least kScaleBitDims dimensions.");
  }
  if (has_3bit_prefix && turboslice_dimensionality > 0 &&
      turboslice_dimensionality < kScaleBitDims) {
    return absl::InvalidArgumentError(
        "3-bit prefix encoding needs at least kScaleBitDims TurboSlice "
        "dimensions.");
  }

  const float sqr_norm =
      Eigen::Map<Eigen::VectorXf>(const_cast<float*>(coordinates.data()),
                                  coordinates.size())
          .squaredNorm();

  if (!std::isfinite(sqr_norm)) {
    return absl::InvalidArgumentError("Infinite norm");
  }
  const int extra_bits_per_dim = quantization_bits - 4;
  ABSL_RETURN_IF_ERROR(ValidateTurboQuantBits(quantization_bits));
  if (extra_bits_per_dim > 0) {
    const size_t extra_bits_size = GetExtraPrecisionBytes(
        dimensionality, extra_bits_per_dim, has_3bit_prefix);
    if (output_extra_precision_data.size() != extra_bits_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Output extra data span has wrong size for 4+", extra_bits_per_dim,
          " bit encoding. Expected: ", extra_bits_size,
          " Got: ", output_extra_precision_data.size()));
    }
  }
  if (sqr_norm == 0.0f) {
    std::fill(output_coordinates.begin(), output_coordinates.end(), 0);
    if (extra_bits_per_dim > 0) {
      std::fill(output_extra_precision_data.begin(),
                output_extra_precision_data.end(), 0);
    }
    return EncodeDatapointResult{0.0f, 0.0f, 0.0f, 0.0f};
  }
  const float norm = std::sqrt(sqr_norm);

  std::vector<float> rotated_coordinates(dimensionality);
  ABSL_RETURN_IF_ERROR(rotation.FlipRandomSignsNormalizeAndRotate(
      coordinates, norm, absl::MakeSpan(rotated_coordinates)));

  float quantized_sqr_norm_4bit;
  float quantized_sqr_norm_4plusk = 0.0f;
  if (extra_bits_per_dim == 0) {
    quantized_sqr_norm_4bit = HWY_DYNAMIC_DISPATCH(FourBitEncodeImpl)(
        rotated_coordinates.data(), dimensionality, output_coordinates.data());
  } else {
    ABSL_ASSIGN_OR_RETURN((auto [sq_4bit, sq_4plusk]),
                          EncodeDatapoint4PlusKBits(
                              rotated_coordinates, output_coordinates,
                              extra_bits_per_dim, output_extra_precision_data));
    quantized_sqr_norm_4bit = sq_4bit;
    quantized_sqr_norm_4plusk = sq_4plusk;
  }
  if (has_3bit_prefix) {
    const uint16_t low_bits =
        ApplyThreeBitPrefix(output_coordinates, quantized_sqr_norm_4bit);
    // Only the 4-bit codes lose the bucket LSBs. Keep them after the packed
    // extra bits so that 4+k scoring can restore the full codes.
    if (extra_bits_per_dim > 0) {
      static_assert(kPrefixLowBitBytes == sizeof(low_bits));
      std::memcpy(output_extra_precision_data.last(kPrefixLowBitBytes).data(),
                  &low_bits, sizeof(low_bits));
    }
  }

  const float scale4 = (norm / std::sqrt(quantized_sqr_norm_4bit)) *
                       debiasing_4bit(dimensionality, has_3bit_prefix);
  const float scale4pk = extra_bits_per_dim > 0
                             ? (norm / std::sqrt(quantized_sqr_norm_4plusk)) *
                                   kDebiasing4PlusKBit[extra_bits_per_dim - 1]
                             : scale4;

  float scale4_turboslice = 0.0f;
  if (turboslice_dimensionality > 0) {
    float quantized_sqr_norm_turboslice = 0.0f;
    const size_t turboslice_bytes = turboslice_dimensionality / 2;
    for (size_t b = 0; b < turboslice_bytes; ++b) {
      const uint8_t byte = output_coordinates[b];
      const bool use_3bit = has_3bit_prefix && b < kScaleBitBytes;
      quantized_sqr_norm_turboslice +=
          turboquant_codebooks::kSquaredCentroids[byte & 7] +
          (use_3bit
               ? turboquant_codebooks::kMerged3BitSquaredCentroids[(byte >> 4) &
                                                                   3]
               : turboquant_codebooks::kSquaredCentroids[(byte >> 4) & 7]);
    }
    if (turboslice_dimensionality % 2 != 0) {
      quantized_sqr_norm_turboslice += turboquant_codebooks::kSquaredCentroids
          [output_coordinates[turboslice_bytes] & 7];
    }
    const float debiasing_turboslice =
        debiasing_4bit(turboslice_dimensionality, has_3bit_prefix);
    scale4_turboslice = (norm / std::sqrt(quantized_sqr_norm_turboslice)) *
                        debiasing_turboslice;
  }

  return EncodeDatapointResult{
      .unquantized_squared_norm = sqr_norm,
      .norm_scaling_factor_4bit = scale4,
      .norm_scaling_factor_4plusk = scale4pk,
      .norm_scaling_factor_turboslice = scale4_turboslice};
}

}  // namespace internal

int32_t ComputeQueryByteSum(size_t dimensionality,
                            absl::Span<const uint8_t> quantized_data,
                            bool has_3bit_prefix) {
  ABSL_DCHECK_GE(quantized_data.size(),
                 hwy::DivCeil(dimensionality, size_t{2}));
  int32_t byte_sum = 0;
  const size_t num_full_bytes = dimensionality / 2;
  for (size_t i = 0; i < num_full_bytes; ++i) {
    const uint8_t byte = quantized_data[i];
    const bool use_3bit = has_3bit_prefix && i < kScaleBitBytes;
    byte_sum +=
        turboquant_codebooks::kTurboQuantCentroidsInt8[byte & 0xF] +
        (use_3bit
             ? turboquant_codebooks::kTurboQuant3BitCentroidsInt8[byte >> 4]
             : turboquant_codebooks::kTurboQuantCentroidsInt8[byte >> 4]);
  }
  if (dimensionality % 2 != 0) {
    byte_sum += turboquant_codebooks::kTurboQuantCentroidsInt8
        [quantized_data[num_full_bytes] & 0xF];
  }
  return byte_sum;
}

}  // namespace ginger
#endif  // HWY_ONCE
