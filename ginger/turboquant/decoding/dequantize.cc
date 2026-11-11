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

#include "ginger/turboquant/decoding/dequantize.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/log/absl_log.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codebooks/centroid_tables.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/norm_storage.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/decoding/dequantize.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
#include "ginger/turboquant/scoring/simd_helpers-inl.h"  // NOLINT

namespace ginger {
namespace HWY_NAMESPACE {

namespace {

namespace hn = hwy::HWY_NAMESPACE;
using ::ginger::turboquant_internal::PositiveCentroidTable;
using ::ginger::turboquant_internal::SignedCentroidTable;
using turboquant_internal::LoadPackedBits;
using turboquant_internal::LookupCentroid;

void DequantizeToRotatedSpace4BitOnlyImpl(
    int dimensionality, const uint8_t* HWY_RESTRICT quantized_data,
    float norm_scaling_factor_4bit, float* HWY_RESTRICT output) {
  using u32 = hn::ScalableTag<uint32_t>;
  const u32 du;
  const hn::ScalableTag<int8_t> di8;
  const hn::ScalableTag<int32_t> di;
  const hn::ScalableTag<float> df;
  const hn::Repartition<uint8_t, u32> du8;
  const size_t N = hn::Lanes(df);

  const auto v_scale = hn::Set(df, norm_scaling_factor_4bit);
  const auto centroids =
      hn::LoadDup128(di8, ginger::internal::kTurboQuantCentroidsInt8.data());

  size_t d = 0;
  for (; d + N <= static_cast<size_t>(dimensionality); d += N) {
    const hn::Vec<u32> codes = LoadPackedBits<4>(du, quantized_data, d);
    const auto dp_vals_i32 = hn::BitCast(
        di, hn::TableLookupBytes(centroids, hn::BitCast(du8, codes)));
    // Sign-extend from 8-bit to 32-bit: shift left 24, arithmetic shift
    // right 24.
    const auto extended = hn::ShiftRight<24>(hn::ShiftLeft<24>(dp_vals_i32));
    const auto dequantized = hn::Mul(hn::ConvertTo(df, extended), v_scale);
    hn::StoreU(dequantized, df, output + d);
  }

  // Tail.
  for (; d < static_cast<size_t>(dimensionality); ++d) {
    const uint8_t byte = quantized_data[d / 2];
    const uint8_t nibble = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
    output[d] =
        static_cast<float>(ginger::internal::kTurboQuantCentroidsInt8[nibble]) *
        norm_scaling_factor_4bit;
  }
}

template <int kBitsPerDim>
void DequantizeToRotatedSpace4PlusKBitImpl(int dimensionality,
                                           const uint8_t* HWY_RESTRICT dp_quant,
                                           const uint8_t* HWY_RESTRICT dp_extra,
                                           float scale,
                                           float* HWY_RESTRICT output) {
  static_assert(kBitsPerDim >= 1 &&
                (kBitsPerDim <= 4 || kBitsPerDim == 6 || kBitsPerDim == 8));
  using Float = hn::ScalableTag<float>;
  using Int32 = hn::ScalableTag<int32_t>;
  using UInt32 = hn::ScalableTag<uint32_t>;
  const Float df;
  const Int32 di;
  const UInt32 du;
  const size_t N = hn::Lanes(df);

  static constexpr SignedCentroidTable<kBitsPerDim> signed_centroids{};
  static constexpr PositiveCentroidTable<(kBitsPerDim <= 2 ? kBitsPerDim : 1)>
      pos_centroids{};

  const auto v_scale = hn::Set(df, scale);

  const size_t simd_end = hwy::RoundDownTo(dimensionality, N);
  size_t d = 0;
  for (; d < simd_end; d += N) {
    const auto codes = LoadPackedBits<4>(du, dp_quant, d);
    const auto extra = LoadPackedBits<kBitsPerDim>(du, dp_extra, d);
    const auto centroid_v = LookupCentroid<kBitsPerDim>(
        df, du, di, codes, extra, pos_centroids.table, signed_centroids.table);
    hn::StoreU(hn::Mul(centroid_v, v_scale), df, output + d);
  }

  // Tail.
  for (; d < static_cast<size_t>(dimensionality); ++d) {
    const uint8_t byte = dp_quant[d >> 1];
    const uint8_t base_code = (d & 1) ? (byte >> 4) : (byte & 0xF);
    const int bit = d * kBitsPerDim;
    uint16_t two_bytes = 0;
    memcpy(&two_bytes, dp_extra + (bit >> 3), 2);
    const uint8_t extra_val =
        (two_bytes >> (bit & 7)) & ((1 << kBitsPerDim) - 1);
    float centroid_val = signed_centroids.Get(base_code, extra_val);
    output[d] = centroid_val * scale;
  }
}

void DequantizeToRotatedSpace4PlusKBitDispatch(
    int dimensionality, const uint8_t* HWY_RESTRICT dp_quant,
    const uint8_t* HWY_RESTRICT dp_extra, int extra_bits_per_dim, float scale,
    float* HWY_RESTRICT output) {
  switch (extra_bits_per_dim) {
    case 1:
      DequantizeToRotatedSpace4PlusKBitImpl<1>(dimensionality, dp_quant,
                                               dp_extra, scale, output);
      break;
    case 2:
      DequantizeToRotatedSpace4PlusKBitImpl<2>(dimensionality, dp_quant,
                                               dp_extra, scale, output);
      break;
    case 3:
      DequantizeToRotatedSpace4PlusKBitImpl<3>(dimensionality, dp_quant,
                                               dp_extra, scale, output);
      break;
    case 4:
      DequantizeToRotatedSpace4PlusKBitImpl<4>(dimensionality, dp_quant,
                                               dp_extra, scale, output);
      break;
    case 6:
      DequantizeToRotatedSpace4PlusKBitImpl<6>(dimensionality, dp_quant,
                                               dp_extra, scale, output);
      break;
    case 8:
      DequantizeToRotatedSpace4PlusKBitImpl<8>(dimensionality, dp_quant,
                                               dp_extra, scale, output);
      break;
    default:
      ABSL_LOG(FATAL) << "Unsupported extra_bits_per_dim: "
                      << extra_bits_per_dim
                      << ". Supported values are 1, 2, 3, 4, 6, 8.";
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {
HWY_EXPORT(DequantizeToRotatedSpace4BitOnlyImpl);
HWY_EXPORT(DequantizeToRotatedSpace4PlusKBitDispatch);

void DequantizeToRotatedSpace4BitOnly(int dimensionality,
                                      absl::Span<const uint8_t> quantized_data,
                                      float norm_scaling_factor_4bit,
                                      absl::Span<float> output) {
  ABSL_CHECK_EQ(output.size(), dimensionality);
  HWY_DYNAMIC_DISPATCH(DequantizeToRotatedSpace4BitOnlyImpl)
  (dimensionality, quantized_data.data(), norm_scaling_factor_4bit,
   output.data());
}

void DequantizeToRotatedSpace4PlusKBit(
    int dimensionality, absl::Span<const uint8_t> quantized_data,
    absl::Span<const uint8_t> extra_precision_data, int extra_bits_per_dim,
    float norm_scaling_factor_4plusk, absl::Span<float> output) {
  ABSL_CHECK_EQ(output.size(), dimensionality);
  HWY_DYNAMIC_DISPATCH(DequantizeToRotatedSpace4PlusKBitDispatch)
  (dimensionality, quantized_data.data(), extra_precision_data.data(),
   extra_bits_per_dim, norm_scaling_factor_4plusk, output.data());
}

void DequantizeToRotatedSpace(size_t dimensionality, int extra_bits_per_dim,
                              const TurboQuantBasePointView& datapoint,
                              absl::Span<float> output) {
  if (extra_bits_per_dim == 0) {
    DequantizeToRotatedSpace4BitOnly(dimensionality, datapoint.quantized_data,
                                     datapoint.norm_scaling_factor_4bit,
                                     output);
    if (datapoint.has_3bit_prefix) {
      // The odd dimensions of the prefix hold merged 3-bit codes.
      for (size_t d = 1; d < kScaleBitDims; d += 2) {
        output[d] = turboquant_codebooks::kTurboQuant3BitCentroidsInt8
                        [datapoint.quantized_data[d / 2] >> 4] *
                    datapoint.norm_scaling_factor_4bit;
      }
    }
    return;
  }
  absl::Span<const uint8_t> codes = datapoint.quantized_data;
  absl::Span<const uint8_t> extra = datapoint.extra_precision_data;
  size_t d = 0;
  if (datapoint.has_3bit_prefix) {
    // Decode the prefix dimensions from the restored 4-bit codes. Their packed
    // extra bits end on a byte boundary.
    static_assert(kScaleBitDims % 8 == 0);
    uint8_t restored[kScaleBitBytes];
    RestorePrefixCodes(codes.data(), extra.last(kPrefixLowBitBytes).data(),
                       restored);
    DequantizeToRotatedSpace4PlusKBit(
        kScaleBitDims, restored, extra, extra_bits_per_dim,
        datapoint.norm_scaling_factor_4plusk, output.first(kScaleBitDims));
    d = kScaleBitDims;
    codes = codes.subspan(kScaleBitBytes);
    extra = extra.subspan(kScaleBitDims * extra_bits_per_dim / 8);
  }
  DequantizeToRotatedSpace4PlusKBit(
      dimensionality - d, codes, extra, extra_bits_per_dim,
      datapoint.norm_scaling_factor_4plusk, output.subspan(d));
}

}  // namespace ginger
#endif  // HWY_ONCE
