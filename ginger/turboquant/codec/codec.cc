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

#include "ginger/turboquant/codec/codec.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/datapoint/encode_datapoint.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/turboquant/types/quantization_bits.h"
#include "hwy/highway.h"

namespace ginger {

size_t TurboQuantCodec::QueryPadding() {
  // Returns the number of lanes used for padding the query coordinates. We use
  // a constant value of 64, which is the maximum possible number of lanes for
  // int8_t across all current targets (e.g., 64 for AVX-512). This ensures the
  // query layout is consistent across targets and independent of the cache line
  // size.
  return 64;
}

size_t TurboQuantCodec::GetExtraPrecisionBytes(size_t dimensionality,
                                               int extra_bits_per_dim) {
  return ::ginger::GetExtraPrecisionBytes(dimensionality, extra_bits_per_dim);
}

size_t TurboQuantCodec::num_bytes_per_query() const {
  return 2 * hwy::RoundUpTo(num_bytes_per_datapoint_, QueryPadding());
}

size_t TurboQuantCodec::num_bytes_per_turboslice_query() const {
  return 2 * hwy::RoundUpTo(turboslice_bytes_per_datapoint_, QueryPadding());
}

// Returns a pair of:
// - number of bytes for 4-bit base quantization: ceil(`dimensionality` / 2).
// - number of bytes for additional precision data:
//   ceil(`dimensionality` * `extra_bits_per_dim` / 8) + (extra_bits_per_dim > 0
//   ? 1 : 0).
//
// Without the 1-byte padding, base bytes make up roughly 4 / (4 + k) of the
// total bytes, where k is `extra_bits_per_dim`. Integer division rounds down,
// which underestimates the base byte count by 1 for certain vector dimensions.
// Adding k / 2 as a rounding bias to the numerator neatly prevents this
// underestimation across all supported bit modes (1, 2, 3, 4, 6, and 8).
std::pair<size_t, size_t> TurboQuantCodec::InferBaseAndAdditionalPrecisionBytes(
    size_t total_bytes, int extra_bits_per_dim) {
  ABSL_QCHECK(IsValidTurboQuantExtraBits(extra_bits_per_dim));
  if (extra_bits_per_dim == 0 || total_bytes == 0) {
    return {total_bytes, 0};
  }
  const size_t total_no_pad = total_bytes - 1;
  const size_t base_bytes =
      (4 * total_no_pad + extra_bits_per_dim / 2) / (4 + extra_bits_per_dim);
  return {base_bytes, total_bytes - base_bytes};
}

TurboQuantCodec::TurboQuantCodec(size_t dimensionality, uint32_t seed,
                                 TurboQuantDistanceMetric metric,
                                 size_t turboslice_dimensionality)
    : dimensionality_(dimensionality),
      turboslice_dimensionality_(turboslice_dimensionality),
      seed_(seed),
      num_bytes_per_datapoint_((dimensionality_ + 1) / 2),
      turboslice_bytes_per_datapoint_((turboslice_dimensionality + 1) / 2),
      rotation_(dimensionality, seed),
      metric_(metric) {
  ABSL_QCHECK_LE(turboslice_dimensionality, dimensionality);
}

absl::StatusOr<TurboQuantBasePoint> EncodeDatapoint(
    const TurboQuantCodec& turboquant, absl::Span<const float> coordinates,
    int quantization_bits, bool has_3bit_prefix) {
  std::vector<uint8_t> output_coordinates(turboquant.num_bytes_per_datapoint());
  const int extra_bits_per_dim = quantization_bits - 4;
  std::vector<uint8_t> extra_precision_data;
  if (extra_bits_per_dim > 0) {
    extra_precision_data.resize(GetExtraPrecisionBytes(
        turboquant.dimensionality(), extra_bits_per_dim, has_3bit_prefix));
  }
  ABSL_ASSIGN_OR_RETURN(
      EncodeDatapointResult res,
      turboquant.EncodeDatapointStructured(
          coordinates, absl::MakeSpan(output_coordinates), quantization_bits,
          absl::MakeSpan(extra_precision_data), has_3bit_prefix));
  const int32_t byte_sum = ComputeQueryByteSum(
      turboquant.dimensionality(), absl::MakeConstSpan(output_coordinates),
      has_3bit_prefix);
  return TurboQuantBasePoint(
      std::move(output_coordinates), std::move(extra_precision_data),
      res.norm_scaling_factor_4bit, res.norm_scaling_factor_4plusk,
      res.unquantized_squared_norm, res.norm_scaling_factor_turboslice,
      byte_sum, has_3bit_prefix, extra_bits_per_dim);
}

}  // namespace ginger
