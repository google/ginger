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

#include "ginger/turboquant/query/encode_query.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/rotation/rotation.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/query/encode_query.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace HWY_NAMESPACE {

namespace {

namespace hn = hwy::HWY_NAMESPACE;

void EncodeQuantizedDataPointAsQueryImpl(
    const uint8_t* HWY_RESTRICT quantized_point,
    int8_t* HWY_RESTRICT encoded_query, size_t num_bytes_per_datapoint,
    size_t deinterleaved_end, size_t dimensionality, bool has_3bit_prefix) {
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int8 = hn::ScalableTag<int8_t>;
  const Uint8 u8;
  const Int8 i8;
  const size_t N = hn::Lanes(u8);

  static_assert(kScaleBitBytes == 16);
  const hn::Vec<Int8> centroids =
      hn::LoadDup128(i8, internal::kTurboQuantCentroidsInt8.data());
  // `hn::TableLookupBytes` operates independently within each 128-bit (16-byte)
  // block, so on `i == 0` with wider vectors (`N > 16`), bytes `0..15` look up
  // into the 3-bit table in block 0 while bytes `16..N-1` look up into
  // `centroids` in blocks `1..`.
  const hn::Vec<Int8> first_odd_centroids =
      has_3bit_prefix
          ? hn::IfThenElse(
                hn::RebindMask(
                    i8,
                    hn::Lt(hn::Iota(u8, 0),
                           hn::Set(u8, static_cast<uint8_t>(kScaleBitBytes)))),
                hn::LoadDup128(
                    i8,
                    turboquant_codebooks::kTurboQuant3BitCentroidsInt8.data()),
                centroids)
          : centroids;
  const hn::Vec<Uint8> mask = hn::Set(u8, 0xF);

  size_t i = 0;
  for (; i + N <= num_bytes_per_datapoint; i += N) {
    const hn::Vec<Uint8> packed = hn::LoadU(u8, quantized_point + i);
    const hn::Vec<Uint8> even_idxs = hn::And(packed, mask);
    const hn::Vec<Uint8> odd_idxs = hn::ShiftRight<4>(packed);
    const hn::Vec<Int8> even_vals =
        hn::BitCast(i8, hn::TableLookupBytes(centroids, even_idxs));
    const hn::Vec<Int8> odd_vals = hn::BitCast(
        i8, hn::TableLookupBytes(i == 0 ? first_odd_centroids : centroids,
                                 odd_idxs));
    hn::StoreU(even_vals, i8, encoded_query + i);
    hn::StoreU(odd_vals, i8, encoded_query + deinterleaved_end + i);
  }
  if (i < num_bytes_per_datapoint) {
    const size_t remaining = num_bytes_per_datapoint - i;
    const hn::Vec<Uint8> packed =
        hn::LoadNOr(hn::Zero(u8), u8, quantized_point + i, remaining);
    const hn::Vec<Uint8> even_idxs = hn::And(packed, mask);
    const hn::Vec<Uint8> odd_idxs = hn::ShiftRight<4>(packed);
    const hn::Vec<Int8> even_vals =
        hn::BitCast(i8, hn::TableLookupBytes(centroids, even_idxs));
    const hn::Vec<Int8> odd_vals = hn::BitCast(
        i8, hn::TableLookupBytes(i == 0 ? first_odd_centroids : centroids,
                                 odd_idxs));
    hn::StoreU(even_vals, i8, encoded_query + i);
    hn::StoreU(odd_vals, i8, encoded_query + deinterleaved_end + i);
  }
  std::memset(encoded_query + num_bytes_per_datapoint, 0,
              deinterleaved_end - num_bytes_per_datapoint);
  std::memset(encoded_query + deinterleaved_end + dimensionality / 2, 0,
              deinterleaved_end - dimensionality / 2);
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {
HWY_EXPORT(EncodeQuantizedDataPointAsQueryImpl);

namespace {

struct PrepareQueryResult {
  std::vector<float> rotated_coordinates;
  float sqr_norm;
  float norm;
  float scale_factor;
};

absl::StatusOr<PrepareQueryResult> PrepareQueryForQuantization(
    absl::Span<const float> query, absl::Span<float> rotated_float_query,
    const HadamardAndMatrixRotation& rotation) {
  const size_t dimensionality = query.size();
  const float sqr_norm = Eigen::Map<Eigen::VectorXf>(
                             const_cast<float*>(query.data()), dimensionality)
                             .squaredNorm();

  if (!std::isfinite(sqr_norm) || sqr_norm == 0.0f) {
    return absl::InvalidArgumentError(
        absl::StrCat("Query vector has invalid squared norm: ", sqr_norm));
  }
  const float norm = std::sqrt(sqr_norm);

  std::vector<float> rotated_coordinates(dimensionality);
  ABSL_RETURN_IF_ERROR(rotation.FlipRandomSignsNormalizeAndRotate(
      query, norm, absl::MakeSpan(rotated_coordinates)));

  if (!rotated_float_query.empty()) {
    const float scale = norm / std::sqrt(static_cast<float>(dimensionality));
    Eigen::Map<Eigen::VectorXf> float_query_map(rotated_float_query.data(),
                                                dimensionality);
    float_query_map = Eigen::Map<const Eigen::VectorXf>(
                          rotated_coordinates.data(), dimensionality) *
                      scale;
  }
  constexpr float kValueCap = 3.91724f;
  float max_value = 0.0f;
  for (float& x : rotated_coordinates) {
    x = std::clamp(x, -kValueCap, kValueCap);
    max_value = std::max(max_value, std::abs(x));
  }
  const float scale_factor = 127.0f / max_value;
  return PrepareQueryResult{
      .rotated_coordinates = std::move(rotated_coordinates),
      .sqr_norm = sqr_norm,
      .norm = norm,
      .scale_factor = scale_factor};
}

}  // namespace

namespace internal {

absl::StatusOr<EncodeQueryResult> EncodeQuery(
    absl::Span<const float> query, absl::Span<int8_t> encoded_query,
    absl::Span<float> rotated_float_query, size_t dimensionality,
    size_t num_bytes_per_query, size_t num_bytes_per_datapoint,
    const HadamardAndMatrixRotation& rotation,
    size_t turboslice_dimensionality) {
  if (query.size() != dimensionality) {
    return absl::InvalidArgumentError(
        "Query storage size does not match dimensionality.");
  }
  if (turboslice_dimensionality > dimensionality) {
    return absl::InvalidArgumentError(
        "TurboSlice dimensionality cannot exceed dimensionality.");
  }
  if (num_bytes_per_datapoint != (dimensionality + 1) / 2) {
    return absl::InvalidArgumentError(
        "Number of bytes per datapoint does not match dimensionality.");
  }
  const size_t deinterleaved_end = num_bytes_per_query / 2;
  if (num_bytes_per_datapoint > deinterleaved_end) {
    return absl::InvalidArgumentError(
        "Number of bytes per query is too small for dimensionality.");
  }
  if (encoded_query.size() != 2 * deinterleaved_end) {
    return absl::InvalidArgumentError(
        "Encoded query size does not match expected size.");
  }
  if (!rotated_float_query.empty() &&
      rotated_float_query.size() != dimensionality) {
    return absl::InvalidArgumentError(
        "Float query size does not match dimensionality.");
  }

  ABSL_ASSIGN_OR_RETURN(
      const auto prep,
      PrepareQueryForQuantization(query, rotated_float_query, rotation));

  int quant_norm_int8 = 0;
  int quant_norm_int8_turboslice = 0;
  std::fill(encoded_query.begin() + num_bytes_per_datapoint,
            encoded_query.begin() + deinterleaved_end, 0);
  std::fill(encoded_query.begin() + deinterleaved_end + dimensionality / 2,
            encoded_query.end(), 0);

  auto coordinate_position = [&](size_t i) {
    return (i / 2) + (i % 2) * deinterleaved_end;
  };

  for (size_t i = 0; i < dimensionality; ++i) {
    const float scaled_int8 =
        std::clamp(std::round(prep.rotated_coordinates[i] * prep.scale_factor),
                   -127.0f, 127.0f);
    const int8_t snapped = static_cast<int8_t>(scaled_int8);
    encoded_query[coordinate_position(i)] = snapped;
    quant_norm_int8 += snapped * snapped;
    if (i < turboslice_dimensionality) {
      quant_norm_int8_turboslice += snapped * snapped;
    }
  }

  const float norm_scaling_factor_turboslice =
      quant_norm_int8_turboslice > 0
          ? prep.norm /
                std::sqrt(static_cast<float>(quant_norm_int8_turboslice))
          : 0.0f;

  return EncodeQueryResult{
      .norm_scaling_factor =
          prep.norm / std::sqrt(static_cast<float>(quant_norm_int8)),
      .unquantized_squared_norm = prep.sqr_norm,
      .norm_scaling_factor_turboslice = norm_scaling_factor_turboslice};
}

absl::StatusOr<EncodeQueryResult> EncodeQueryForExtraPrecision(
    absl::Span<const float> query, absl::Span<float> rotated_float_query,
    size_t dimensionality, const HadamardAndMatrixRotation& rotation) {
  if (query.size() != dimensionality) {
    return absl::InvalidArgumentError(
        "Query storage size does not match dimensionality.");
  }
  if (rotated_float_query.size() != dimensionality) {
    return absl::InvalidArgumentError(
        "Float query size does not match dimensionality.");
  }
  const float sqr_norm =
      Eigen::Map<const Eigen::VectorXf>(query.data(), dimensionality)
          .squaredNorm();
  if (!std::isfinite(sqr_norm) || sqr_norm == 0.0f) {
    return absl::InvalidArgumentError(
        absl::StrCat("Query vector has invalid squared norm: ", sqr_norm));
  }
  ABSL_RETURN_IF_ERROR(rotation.FlipRandomSignsNormalizeAndRotate(
      query, std::sqrt(static_cast<float>(dimensionality)),
      rotated_float_query));
  return EncodeQueryResult{.norm_scaling_factor = 0.0f,
                           .unquantized_squared_norm = sqr_norm,
                           .norm_scaling_factor_turboslice = 0.0f};
}

absl::StatusOr<EncodeQueryResult> EncodeTurboSliceQuery(
    absl::Span<const float> query, absl::Span<int8_t> encoded_turboslice_query,
    size_t dimensionality, size_t turboslice_dimensionality,
    size_t num_bytes_per_turboslice_query,
    size_t turboslice_bytes_per_datapoint,
    const HadamardAndMatrixRotation& rotation) {
  if (query.size() != dimensionality) {
    return absl::InvalidArgumentError(
        "Query storage size does not match dimensionality.");
  }
  if (turboslice_dimensionality == 0) {
    return absl::InvalidArgumentError(
        "TurboSlice dimensionality not configured.");
  }
  if (turboslice_dimensionality > dimensionality) {
    return absl::InvalidArgumentError(
        "TurboSlice dimensionality cannot exceed dimensionality.");
  }
  if (turboslice_bytes_per_datapoint != (turboslice_dimensionality + 1) / 2) {
    return absl::InvalidArgumentError(
        "TurboSlice bytes per datapoint does not match TurboSlice "
        "dimensionality.");
  }
  const size_t deinterleaved_end_turboslice =
      num_bytes_per_turboslice_query / 2;
  if (turboslice_bytes_per_datapoint > deinterleaved_end_turboslice) {
    return absl::InvalidArgumentError(
        "Number of bytes per TurboSlice query is too small for TurboSlice "
        "dimensionality.");
  }
  if (encoded_turboslice_query.size() != 2 * deinterleaved_end_turboslice) {
    return absl::InvalidArgumentError(
        "Encoded TurboSlice query size does not match expected size.");
  }

  ABSL_ASSIGN_OR_RETURN(const auto prep,
                        PrepareQueryForQuantization(query, {}, rotation));

  std::fill(encoded_turboslice_query.begin() + turboslice_bytes_per_datapoint,
            encoded_turboslice_query.begin() + deinterleaved_end_turboslice, 0);
  std::fill(encoded_turboslice_query.begin() + deinterleaved_end_turboslice +
                turboslice_dimensionality / 2,
            encoded_turboslice_query.end(), 0);

  auto turboslice_coordinate_position = [&](size_t i) {
    return (i / 2) + (i % 2) * deinterleaved_end_turboslice;
  };

  int quant_norm_int8_turboslice = 0;
  for (size_t i = 0; i < turboslice_dimensionality; ++i) {
    const float scaled_int8 =
        std::clamp(std::round(prep.rotated_coordinates[i] * prep.scale_factor),
                   -127.0f, 127.0f);
    const int8_t snapped = static_cast<int8_t>(scaled_int8);
    encoded_turboslice_query[turboslice_coordinate_position(i)] = snapped;
    quant_norm_int8_turboslice += snapped * snapped;
  }

  const float norm_scaling_factor_turboslice =
      quant_norm_int8_turboslice > 0
          ? prep.norm /
                std::sqrt(static_cast<float>(quant_norm_int8_turboslice))
          : 0.0f;

  return EncodeQueryResult{
      .norm_scaling_factor = 0.0f,
      .unquantized_squared_norm = prep.sqr_norm,
      .norm_scaling_factor_turboslice = norm_scaling_factor_turboslice};
}

absl::Status EncodeQuantizedDataPointAsQuery(
    absl::Span<const uint8_t> quantized_point, absl::Span<int8_t> encoded_query,
    size_t num_bytes_per_datapoint, size_t num_bytes_per_query,
    size_t dimensionality, bool has_3bit_prefix) {
  if (quantized_point.size() != num_bytes_per_datapoint) {
    return absl::InvalidArgumentError(
        "Quantized point size does not match number of bytes per datapoint.");
  }
  const size_t deinterleaved_end = num_bytes_per_query / 2;
  if (encoded_query.size() != 2 * deinterleaved_end) {
    return absl::InvalidArgumentError(
        "Encoded query size does not match expected size.");
  }

  HWY_DYNAMIC_DISPATCH(EncodeQuantizedDataPointAsQueryImpl)(
      quantized_point.data(), encoded_query.data(), num_bytes_per_datapoint,
      deinterleaved_end, dimensionality, has_3bit_prefix);

  return absl::OkStatus();
}

}  // namespace internal
}  // namespace ginger
#endif  // HWY_ONCE
