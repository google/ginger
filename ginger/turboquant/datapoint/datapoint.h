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

#ifndef GINGER_TURBOQUANT_DATAPOINT_DATAPOINT_H_
#define GINGER_TURBOQUANT_DATAPOINT_DATAPOINT_H_

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace ginger {

inline constexpr int32_t kQueryByteSumNotSet =
    std::numeric_limits<int32_t>::max();

// A non-owning view into a quantized TurboQuant datapoint with a custom memory
// layout. Views are cheap to copy and can be reassigned.
struct TurboQuantBasePointView {
  absl::Span<const uint8_t> quantized_data;
  absl::Span<const uint8_t> extra_precision_data;

  // The norm scaling factor (norm(unquantized) / norm(quantized)) for 4 bit.
  float norm_scaling_factor_4bit = 0.0f;

  // The norm scaling factor (norm(unquantized) / norm(quantized)) for 4+k bit.
  float norm_scaling_factor_4plusk = 0.0f;

  // The square norm of the unquantized vector.
  float unquantized_squared_norm = 0.0f;

  // The norm scaling factor for TurboSlice scoring.
  float norm_scaling_factor_turboslice = 0.0f;

  // The precomputed byte sum of the quantized data when reencoded as a query.
  int32_t query_byte_sum = kQueryByteSumNotSet;

  // Whether the first `kScaleBitBytes` odd nibbles use the merged 3-bit
  // codebook.
  bool has_3bit_prefix = false;

  // Number of extra precision bits per dimension stored in
  // `extra_precision_data` (0 when only 4-bit data is stored).
  int extra_bits_per_dim = 0;
};

static_assert(std::is_copy_assignable_v<TurboQuantBasePointView>);
static_assert(std::is_move_assignable_v<TurboQuantBasePointView>);

// Owning version of TurboQuantBasePointView. Owns the storage for
// quantized data and metadata. Use `view()` to get a non-owning view.
class TurboQuantBasePoint {
 public:
  TurboQuantBasePoint(std::vector<uint8_t> quantized_data,
                      std::vector<uint8_t> extra_precision_data,
                      float norm_scaling_factor_4bit,
                      float norm_scaling_factor_4plusk,
                      float unquantized_squared_norm,
                      float norm_scaling_factor_turboslice = 0.0f,
                      int32_t query_byte_sum = kQueryByteSumNotSet,
                      bool has_3bit_prefix = false, int extra_bits_per_dim = 0)
      : quantized_data_(std::move(quantized_data)),
        extra_precision_data_(std::move(extra_precision_data)),
        norm_scaling_factor_4bit_(norm_scaling_factor_4bit),
        norm_scaling_factor_4plusk_(norm_scaling_factor_4plusk),
        unquantized_squared_norm_(unquantized_squared_norm),
        norm_scaling_factor_turboslice_(norm_scaling_factor_turboslice),
        query_byte_sum_(query_byte_sum),
        has_3bit_prefix_(has_3bit_prefix),
        extra_bits_per_dim_(extra_bits_per_dim) {}

  explicit TurboQuantBasePoint(const TurboQuantBasePointView& view)
      : quantized_data_(view.quantized_data.begin(), view.quantized_data.end()),
        extra_precision_data_(view.extra_precision_data.begin(),
                              view.extra_precision_data.end()),
        norm_scaling_factor_4bit_(view.norm_scaling_factor_4bit),
        norm_scaling_factor_4plusk_(view.norm_scaling_factor_4plusk),
        unquantized_squared_norm_(view.unquantized_squared_norm),
        norm_scaling_factor_turboslice_(view.norm_scaling_factor_turboslice),
        query_byte_sum_(view.query_byte_sum),
        has_3bit_prefix_(view.has_3bit_prefix),
        extra_bits_per_dim_(view.extra_bits_per_dim) {}

  TurboQuantBasePointView view() const {
    return {.quantized_data = absl::MakeConstSpan(quantized_data_),
            .extra_precision_data = absl::MakeConstSpan(extra_precision_data_),
            .norm_scaling_factor_4bit = norm_scaling_factor_4bit_,
            .norm_scaling_factor_4plusk = norm_scaling_factor_4plusk_,
            .unquantized_squared_norm = unquantized_squared_norm_,
            .norm_scaling_factor_turboslice = norm_scaling_factor_turboslice_,
            .query_byte_sum = query_byte_sum_,
            .has_3bit_prefix = has_3bit_prefix_,
            .extra_bits_per_dim = extra_bits_per_dim_};
  }

 private:
  std::vector<uint8_t> quantized_data_;
  std::vector<uint8_t> extra_precision_data_;
  float norm_scaling_factor_4bit_ = 0.0f;
  float norm_scaling_factor_4plusk_ = 0.0f;
  float unquantized_squared_norm_ = 0.0f;
  float norm_scaling_factor_turboslice_ = 0.0f;
  int32_t query_byte_sum_ = kQueryByteSumNotSet;
  bool has_3bit_prefix_ = false;
  int extra_bits_per_dim_ = 0;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_DATAPOINT_DATAPOINT_H_
