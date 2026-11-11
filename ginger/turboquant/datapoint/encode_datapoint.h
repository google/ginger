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

#ifndef GINGER_TURBOQUANT_DATAPOINT_ENCODE_DATAPOINT_H_
#define GINGER_TURBOQUANT_DATAPOINT_ENCODE_DATAPOINT_H_

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/turboquant/rotation/rotation.h"

namespace ginger {

struct EncodeDatapointResult {
  float unquantized_squared_norm;
  float norm_scaling_factor_4bit;
  float norm_scaling_factor_4plusk;
  float norm_scaling_factor_turboslice;
};

// Returns the number of packed extra precision bytes per datapoint (including
// one padding byte when `extra_bits_per_dim > 0`).
size_t GetExtraPrecisionBytes(size_t dimensionality, int extra_bits_per_dim);

// As above, plus `kPrefixLowBitBytes` when `has_3bit_prefix` and
// `extra_bits_per_dim > 0`.
size_t GetExtraPrecisionBytes(size_t dimensionality, int extra_bits_per_dim,
                              bool has_3bit_prefix);

// Implementation detail of `TurboQuantCodec::EncodeDatapointStructured`.
namespace internal {

// `coordinates` must have exactly `dimensionality` entries.
// `output_coordinates` must have exactly `num_bytes_per_datapoint` entries.
// The encoded datapoint is stored in output_coordinates,
// and the squared norm of the original coordinates, the ratio of the original
// norm and the 4 bit quantized norm and the ratio of the original norm and
// the 4+k bit quantized norm is returned in the EncodeDatapointResult struct.
//
// quantization_bits parameter below is the total number of bits per dimension
// used for quantization. The extra bits k are stored in
// output_extra_precision_data. It defaults to 4 and currently setting it to
// 4, 5, 6, 7, 8, 10, or 12 is supported.
//
// Set `has_3bit_prefix` to true when encoding for a dataset that uses
// `ScalingStorageMode::kIn3BitPrefix` (i.e., `dataset.has_3bit_prefix()`).
absl::StatusOr<EncodeDatapointResult> EncodeDatapointStructured(
    absl::Span<const float> coordinates, absl::Span<uint8_t> output_coordinates,
    int quantization_bits, absl::Span<uint8_t> output_extra_precision_data,
    size_t dimensionality, size_t num_bytes_per_datapoint,
    size_t turboslice_dimensionality, const HadamardAndMatrixRotation& rotation,
    bool has_3bit_prefix);

}  // namespace internal

// Computes the query byte sum for a quantized datapoint directly by decoding
// codebook centroids without reencoding into a full query vector.
int32_t ComputeQueryByteSum(size_t dimensionality,
                            absl::Span<const uint8_t> quantized_data,
                            bool has_3bit_prefix);

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_DATAPOINT_ENCODE_DATAPOINT_H_
