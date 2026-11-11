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

#ifndef GINGER_TURBOQUANT_DECODING_DEQUANTIZE_H_
#define GINGER_TURBOQUANT_DECODING_DEQUANTIZE_H_

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "ginger/turboquant/datapoint/datapoint.h"

namespace ginger {

// Dequantizes the given quantized data to the rotated space using 4-bit
// quantization.
// - `output` must have `dimensionality` size.
void DequantizeToRotatedSpace4BitOnly(int dimensionality,
                                      absl::Span<const uint8_t> quantized_data,
                                      float norm_scaling_factor_4bit,
                                      absl::Span<float> output);

// Dequantizes the given quantized data to the rotated space using 4+k bit
// quantization.
// - `output` must have `dimensionality` size.
void DequantizeToRotatedSpace4PlusKBit(
    int dimensionality, absl::Span<const uint8_t> quantized_data,
    absl::Span<const uint8_t> extra_precision_data, int extra_bits_per_dim,
    float norm_scaling_factor_4plusk, absl::Span<float> output);

// Dequantizes `datapoint` to the rotated space, using 4+k bits when
// `extra_bits_per_dim > 0` and 4 bits otherwise. Handles 3-bit prefix
// datapoints.
// - `output` must have `dimensionality` size.
void DequantizeToRotatedSpace(size_t dimensionality, int extra_bits_per_dim,
                              const TurboQuantBasePointView& datapoint,
                              absl::Span<float> output);

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_DECODING_DEQUANTIZE_H_
