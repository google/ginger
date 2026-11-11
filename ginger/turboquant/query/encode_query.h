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

#ifndef GINGER_TURBOQUANT_QUERY_ENCODE_QUERY_H_
#define GINGER_TURBOQUANT_QUERY_ENCODE_QUERY_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/turboquant/rotation/rotation.h"

namespace ginger {

struct EncodeQueryResult {
  float norm_scaling_factor;
  float unquantized_squared_norm;
  float norm_scaling_factor_turboslice;
};

// Implementation details of the `TurboQuantCodec` query encoding methods.
namespace internal {

// Prepares the query vector for scoring. Rotates and scales the vector, and
// reorders coordinates to the format expected by the scoring loop.
// `query` must have length `dimensionality`. `encoded_query` must have
// length `num_bytes_per_query`. If `rotated_float_query` is non-empty, it is
// populated with the rotated query coordinates scaled for residual scoring.
// `turboslice_dimensionality` specifies the number of prefix dimensions to use
// for computing the TurboSlice norm scaling factor; when set to 0 (default),
// TurboSlice is disabled and no TurboSlice scaling factor is computed.
// Returns norm multipliers and the squared norm of the query.
absl::StatusOr<EncodeQueryResult> EncodeQuery(
    absl::Span<const float> query, absl::Span<int8_t> encoded_query,
    absl::Span<float> rotated_float_query, size_t dimensionality,
    size_t num_bytes_per_query, size_t num_bytes_per_datapoint,
    const HadamardAndMatrixRotation& rotation,
    size_t turboslice_dimensionality = 0);

// Prepares the query vector solely for extra-precision (4+k bit) scoring,
// skipping int8 query quantization and deinterleaving.
// `query` and `rotated_float_query` must have length `dimensionality`.
absl::StatusOr<EncodeQueryResult> EncodeQueryForExtraPrecision(
    absl::Span<const float> query, absl::Span<float> rotated_float_query,
    size_t dimensionality, const HadamardAndMatrixRotation& rotation);

// Prepares the TurboSlice query vector for scoring. Rotates and scales the
// vector using the full vector dimensionality, and quantizes the prefix of
// `turboslice_dimensionality` coordinates.
// `query` must have length `dimensionality`.
// `encoded_turboslice_query` must have length
// `num_bytes_per_turboslice_query`.
absl::StatusOr<EncodeQueryResult> EncodeTurboSliceQuery(
    absl::Span<const float> query, absl::Span<int8_t> encoded_turboslice_query,
    size_t dimensionality, size_t turboslice_dimensionality,
    size_t num_bytes_per_turboslice_query,
    size_t turboslice_bytes_per_datapoint,
    const HadamardAndMatrixRotation& rotation);

// Encodes a quantized datapoint as a query vector.
// This is useful to implement symmetric scoring (base against base point)
// using asymmetric scoring. Notably, this is as efficient as a custom
// implementation. `encoded_query` must have length `num_bytes_per_query`.
absl::Status EncodeQuantizedDataPointAsQuery(
    absl::Span<const uint8_t> quantized_point, absl::Span<int8_t> encoded_query,
    size_t num_bytes_per_datapoint, size_t num_bytes_per_query,
    size_t dimensionality, bool has_3bit_prefix);

}  // namespace internal
}  // namespace ginger

#endif  // GINGER_TURBOQUANT_QUERY_ENCODE_QUERY_H_
