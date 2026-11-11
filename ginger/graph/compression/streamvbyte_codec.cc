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

#include "ginger/graph/compression/streamvbyte_codec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/types.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "ginger/graph/compression/streamvbyte_codec.cc"

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace {

// LengthTable maps each control byte to the total byte length of the four
// encoded values it represents.
using LengthTable = std::array<uint8_t, 256>;

// ShuffleTable maps each control byte to a 128-bit shuffle mask used to
// decode the four corresponding encoded values.
using ShuffleMask = std::array<uint8_t, 16>;
using ShuffleTable = std::array<ShuffleMask, 256>;

namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void DecodeStreamVByte(const LengthTable& length_table,
                       const ShuffleTable& shuffle_table,
                       absl::Span<const uint8_t> compressed_data,
                       absl::Span<GingerNodeId> output_buffer) {
  const size_t num_values = output_buffer.size();
  const size_t num_control_bytes = (num_values + 3) / 4;

  const uint8_t* control_bytes_ptr = compressed_data.data();
  const uint8_t* data_ptr = control_bytes_ptr + num_control_bytes;
  uint32_t* output_ptr = reinterpret_cast<uint32_t*>(output_buffer.data());

  // A 128-bit register containing four 32-bit values.
  constexpr hn::CappedTag<uint32_t, 4> d32;
  // A 128-bit register containing sixteen 8-bit values.
  constexpr hn::CappedTag<uint8_t, 16> d8;

  auto prev = hn::Zero(d32);
  const auto decode_quad = [&](const uint8_t control_byte) {
    const uint8_t length = length_table[control_byte];

    // Load the encoded data into a 128-bit register and advance `data_ptr` by
    // `length` bytes to point to the next encoded block of four values.
    auto raw_data = hn::LoadU(d8, data_ptr);
    data_ptr += length;

    // Retrieve the pre-computed shuffle mask for the current `control_byte`,
    // load it into a 128-bit register, and apply it to the `raw_data`.
    // The result, `gap_data`, is a 128-bit register containing four 32-bit
    // values, which represent the gaps of the original values.
    const ShuffleMask& shuffle_mask = shuffle_table[control_byte];
    const auto mask = hn::LoadU(d8, shuffle_mask.data());
    auto gap_data = hn::BitCast(d32, hn::TableLookupBytesOr0(raw_data, mask));

    // The `gap_data` contains the differences between consecutive values. The
    // following operations decode the gaps back to the original values by
    // performing a prefix sum. The last value of the previous block (`prev`)
    // is used as the starting point for the current block's accumulation.
    const auto temp = hn::Add(hn::ShiftLeftBytes<8>(gap_data), gap_data);
    prev = hn::Add(hn::Add(hn::ShiftLeftBytes<4>(temp), temp),
                   hn::BroadcastLane<3>(prev));
  };

  const size_t num_full_quads = num_values / 4;
  for (size_t i = 0; i < num_full_quads; ++i) {
    const uint8_t control_byte = control_bytes_ptr[i];
    decode_quad(control_byte);

    hn::StoreU(prev, d32, output_ptr);
    output_ptr += 4;
  }

  const size_t num_remaining_values = num_values % 4;
  if (num_remaining_values > 0) {
    const uint8_t control_byte = control_bytes_ptr[num_full_quads];
    decode_quad(control_byte);

    hn::StoreN(prev, d32, output_ptr, num_remaining_values);
  }
}

}  // namespace HWY_NAMESPACE

}  // namespace
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace ginger {
HWY_EXPORT(DecodeStreamVByte);

namespace {

// Generates a lookup table where each control byte maps to the total byte
// length required to store the four values it encodes.
consteval LengthTable ComputeLengthTable() {
  LengthTable length_table{};

  for (size_t control_byte = 0; control_byte < 256; ++control_byte) {
    length_table[control_byte] = 0;

    for (size_t i = 0; i < 4; ++i) {
      const uint8_t header = (control_byte >> (2 * i)) & 0b11;
      const uint8_t length = header + 1;
      length_table[control_byte] += length;
    }
  }

  return length_table;
}

// Generates a lookup table where each control byte maps to a 128-bit shuffle
// mask. This mask is used to decode a block of four values encoded with that
// control byte.
consteval ShuffleTable ComputeShuffleTable() {
  ShuffleTable shuffle_table{};

  for (size_t control_byte = 0; control_byte < 256; ++control_byte) {
    uint8_t byte = 0;
    uint8_t pos = 0;

    ShuffleMask& shuffle_mask = shuffle_table[control_byte];
    for (size_t i = 0; i < 4; ++i) {
      const uint8_t header = (control_byte >> (2 * i)) & 0b11;
      const uint8_t length = header + 1;

      uint8_t j = 0;
      while (j < length) {
        shuffle_mask[pos++] = byte++;
        ++j;
      }

      while (j < 4) {
        shuffle_mask[pos++] = 0b11111111;
        ++j;
      }
    }
  }

  return shuffle_table;
}

constexpr LengthTable kLengthTable = ComputeLengthTable();
constexpr ShuffleTable kShuffleTable = ComputeShuffleTable();

class ByteBuffer {
 public:
  ByteBuffer(absl::Span<uint8_t> buffer, bool dry_run, size_t num_control_bytes)
      : buffer_(buffer),
        dry_run_(dry_run),
        num_control_bytes_(num_control_bytes),
        control_bytes_written_(0),
        data_bytes_written_(0) {}

  void WriteControlByte(uint8_t byte) {
    if (!dry_run_) {
      buffer_[control_bytes_written_] = byte;
    }
    control_bytes_written_++;
  }

  void WriteDataByte(uint8_t byte) {
    if (!dry_run_) {
      buffer_[num_control_bytes_ + data_bytes_written_] = byte;
    }
    data_bytes_written_++;
  }

  size_t control_bytes_written() const { return control_bytes_written_; }
  size_t total_bytes_written() const {
    return control_bytes_written_ + data_bytes_written_;
  }

 private:
  absl::Span<uint8_t> buffer_;
  bool dry_run_;
  size_t num_control_bytes_;
  size_t control_bytes_written_;
  size_t data_bytes_written_;
};

// Computes the control byte for the given four values.
uint8_t ComputeControlByte(std::array<GingerNodeId, 4> values) {
  // Returns the number of bytes needed to represent a value minus one.
  const auto byte_width_to_encode = [&](GingerNodeId value) {
    const size_t byte_width =
        value == 0 ? 1 : (std::bit_width<uint64_t>(value) + 7) / 8;
    return byte_width - 1;
  };

  // The control byte is made up of two bits per value to encode, representing
  // its length: 00 for 1 byte, 01 for 2 bytes, 10 for 3 bytes, and 11 for 4
  // bytes.
  return (byte_width_to_encode(values[3]) << 6) |
         (byte_width_to_encode(values[2]) << 4) |
         (byte_width_to_encode(values[1]) << 2) |
         byte_width_to_encode(values[0]);
}

}  // namespace

absl::StatusOr<size_t> StreamVByteEncoder::Encode(
    absl::Span<const GingerNodeId> values, absl::Span<uint8_t> output_buffer,
    bool dry_run) {
  const bool has_negative_values =
      std::any_of(values.begin(), values.end(),
                  [](const GingerNodeId value) { return value < 0; });
  if (has_negative_values) {
    return absl::InvalidArgumentError("Values must be non-negative.");
  }

  const size_t num_control_bytes = (values.size() + 3) / 4;
  ABSL_CHECK(dry_run || num_control_bytes <= output_buffer.size());

  ByteBuffer byte_buffer(output_buffer, dry_run, num_control_bytes);

  for (size_t i = 0; i < values.size(); i += 4) {
    std::array<GingerNodeId, 4> values_to_encode;
    values_to_encode.fill(0);

    const size_t num_values = std::min(4ul, values.size() - i);
    for (size_t j = 0; j < num_values; ++j) {
      values_to_encode[j] = values[i + j];
    }

    byte_buffer.WriteControlByte(ComputeControlByte(values_to_encode));

    for (size_t j = 0; j < num_values; ++j) {
      GingerNodeId value = values[i + j];

      // Copy the value to the output buffer, one byte at a time.
      do {
        byte_buffer.WriteDataByte(value);
        value >>= 8;
      } while (value != 0);
    }
  }

  ABSL_CHECK_EQ(byte_buffer.control_bytes_written(), num_control_bytes);
  if (!dry_run) {
    ABSL_CHECK_LE(byte_buffer.total_bytes_written(), output_buffer.size());
  }

  return byte_buffer.total_bytes_written();
}

void StreamVByteDecoder::DecodeGaps(absl::Span<const uint8_t> compressed_data,
                                    absl::Span<GingerNodeId> output_buffer) {
  return HWY_DYNAMIC_DISPATCH(DecodeStreamVByte)(
      kLengthTable, kShuffleTable, compressed_data, output_buffer);
}

}  // namespace ginger

#endif
