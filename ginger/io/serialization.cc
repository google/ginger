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

#include "ginger/io/serialization.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/log/absl_check.h"
#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/norm_storage.h"

namespace ginger {
namespace {

void StoreLittleEndianFloat(float value, char* dst) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if constexpr (std::endian::native != std::endian::little) {
    bits = absl::byteswap(bits);
  }
  std::memcpy(dst, &bits, sizeof(bits));
}

float LoadLittleEndianFloat(const char* src) {
  uint32_t bits;
  std::memcpy(&bits, src, sizeof(bits));
  if constexpr (std::endian::native != std::endian::little) {
    bits = absl::byteswap(bits);
  }
  return std::bit_cast<float>(bits);
}

}  // namespace

size_t SerializedPointSize(const TurboQuantBasePointView& point) {
  return sizeof(float) * 3 + point.quantized_data.size() +
         point.extra_precision_data.size();
}

void SerializePoint(const TurboQuantBasePointView& point,
                    absl::Span<char> out) {
  ABSL_CHECK_EQ(out.size(), SerializedPointSize(point));
  char* ptr = out.data();
  StoreLittleEndianFloat(point.norm_scaling_factor_4bit, ptr);
  ptr += sizeof(float);
  StoreLittleEndianFloat(point.norm_scaling_factor_4plusk, ptr);
  ptr += sizeof(float);
  StoreLittleEndianFloat(point.unquantized_squared_norm, ptr);
  ptr += sizeof(float);
  if (!point.quantized_data.empty()) {
    std::memcpy(ptr, point.quantized_data.data(), point.quantized_data.size());
    // advancing ptr by number of bytes in quantized data because each entry in
    // quantized data is 1 byte.
    ptr += point.quantized_data.size();
  }
  if (!point.extra_precision_data.empty()) {
    std::memcpy(ptr, point.extra_precision_data.data(),
                point.extra_precision_data.size());
  }
}

SerializedPoint SerializePoint(const TurboQuantBasePointView& point) {
  SerializedPoint result;
  result.resize(SerializedPointSize(point));
  SerializePoint(point, absl::MakeSpan(result));
  return result;
}

TurboQuantBasePointView ParseSerializedTurboQuantPoint(
    SerializedPointView point, int bits_per_dim, bool has_3bit_prefix) {
  const char* ptr = point.data();
  // The `kPrefixLowBitBytes` of 4+k points with a 3-bit prefix are not part of
  // the inference.
  const size_t prefix_low_bit_bytes =
      has_3bit_prefix && bits_per_dim > 0 ? kPrefixLowBitBytes : 0;
  auto [num_bytes_per_datapoint, num_bytes_extra_precision_data] =
      TurboQuantCodec::InferBaseAndAdditionalPrecisionBytes(
          point.size() - 3 * sizeof(float) - prefix_low_bit_bytes,
          bits_per_dim);
  num_bytes_extra_precision_data += prefix_low_bit_bytes;
  return TurboQuantBasePointView{
      .quantized_data = absl::MakeConstSpan(
          reinterpret_cast<const uint8_t*>(ptr + 3 * sizeof(float)),
          num_bytes_per_datapoint),
      .extra_precision_data = absl::MakeConstSpan(
          reinterpret_cast<const uint8_t*>(ptr + 3 * sizeof(float) +
                                           num_bytes_per_datapoint),
          num_bytes_extra_precision_data),
      .norm_scaling_factor_4bit = LoadLittleEndianFloat(ptr),
      .norm_scaling_factor_4plusk = LoadLittleEndianFloat(ptr + sizeof(float)),
      .unquantized_squared_norm =
          LoadLittleEndianFloat(ptr + 2 * sizeof(float)),
      .has_3bit_prefix = has_3bit_prefix,
      .extra_bits_per_dim = bits_per_dim};
}

}  // namespace ginger
