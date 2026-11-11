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

#ifndef GINGER_TURBOQUANT_TYPES_QUANTIZATION_BITS_H_
#define GINGER_TURBOQUANT_TYPES_QUANTIZATION_BITS_H_

#include "absl/status/status.h"

namespace ginger {

constexpr bool IsValidTurboQuantExtraBits(int extra_bits) {
  return (extra_bits >= 0 && extra_bits <= 4) || extra_bits == 6 ||
         extra_bits == 8;
}

constexpr bool IsValidTurboQuantBits(int quantization_bits) {
  return IsValidTurboQuantExtraBits(quantization_bits - 4);
}

inline absl::Status ValidateTurboQuantExtraBits(int extra_bits) {
  if (!IsValidTurboQuantExtraBits(extra_bits)) {
    return absl::InvalidArgumentError(
        "Extra bits per dimension must be 0, 1, 2, 3, 4, 6, or 8.");
  }
  return absl::OkStatus();
}

inline absl::Status ValidateTurboQuantBits(int quantization_bits) {
  if (!IsValidTurboQuantBits(quantization_bits)) {
    return absl::InvalidArgumentError(
        "quantization_bits must be 4, 5, 6, 7, 8, 10, or 12.");
  }
  return absl::OkStatus();
}

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_TYPES_QUANTIZATION_BITS_H_
