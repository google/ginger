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

#ifndef GINGER_GRAPH_COMPRESSION_VARINT_CODEC_H_
#define GINGER_GRAPH_COMPRESSION_VARINT_CODEC_H_

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/graph/varint.h"
#include "ginger/types.h"

namespace ginger {

// This class encodes sequences of `GingerNodeId` values using a Varint
// encoding. Unlike the classic Varint encoding, which uses the most significant
// bit of each byte as a continuation bit, this implementation encodes the
// length information within the first byte of the encoded value. This allows
// for faster decoding while retaining the same compression size.
//
// The following is an example usage with Varint:
// ```cpp
// std::vector<GingerNodeId> values_to_encode = {10, 250, 65536, 1};
//
// // Step 1: Dry run to determine the required buffer size.
// VarintEncoder dry_run_encoder({}, /*dry_run=*/true);
// for (GingerNodeId value : values_to_encode) {
//   dry_run_encoder.Encode(value);
// }
// dry_run_encoder.Flush(); // Optional for Varint, required for other encoders.
// size_t encoded_size = dry_run_encoder.CurOffset();
//
// // Step 2: Allocate the buffer and encode the values.
// std::vector<uint8_t> encoded_buffer(encoded_size +
//                                     VarintDecoder::kSlopBytes);
// VarintEncoder encoder(absl::MakeSpan(encoded_buffer));
// for (GingerNodeId value : values_to_encode) {
//   encoder.Encode(value);
// }
// encoder.Flush();
//
// // Step 3: Decode the values.
// VarintDecoder decoder(encoded_buffer);
// std::vector<GingerNodeId> decoded_values;
// for (size_t i = 0; i < values_to_encode.size(); ++i) {
//   decoded_values.push_back(decoder.Next());
// }
// // decoded_values now contains {10, 250, 65536, 1}
// ```
class VarintEncoder {
  // Returns the length in bytes of the varint representation of `value`.
  static size_t VarintLength(uint32_t value) {
    return VarintEngine::Length32(value);
  }

  // The caller is responsible for ensuring that `ptr` has sufficient space for
  // `VarintLength(value)` bytes.
  static size_t EncodeVarint(uint8_t* ptr, uint32_t value) {
    char* start = reinterpret_cast<char*>(ptr);
    char* end = VarintEngine::Encode32(start, value);
    return std::distance(start, end);
  }

 public:
  // Constructs a new encoder where the `output_buffer` is used to store the
  // encoded data. The caller must ensure it is large enough to accommodate
  // all encoded values.
  // If `dry_run` is set to true, the encoder will not write to `output_buffer`;
  // instead, it will only calculate the total size required for the encoded
  // data. Consequently, when `dry_run` is true, the contents of `output_buffer`
  // are ignored.
  explicit VarintEncoder(absl::Span<uint8_t> output_buffer,
                         bool dry_run = false)
      : output_buffer_(output_buffer), dry_run_(dry_run), cur_offset_(0) {}

  // Encodes a single `GingerNodeId` value. The value must be non-negative.
  // The caller is responsible for ensuring that the `output_buffer` provided
  // during construction has sufficient space remaining to accommodate the
  // encoded value.
  void Encode(GingerNodeId value) {
    ABSL_CHECK_GE(value, 0);

    if (dry_run_) {
      cur_offset_ += VarintLength(value);
    } else {
      ABSL_DCHECK_LE(cur_offset_ + VarintLength(value), output_buffer_.size());

      uint8_t* ptr = output_buffer_.data() + cur_offset_;
      cur_offset_ += EncodeVarint(ptr, value);
    }
  }

  // Varint encoding is always byte-aligned, so no flushing is required.
  void Flush() {}

  // Skips the next `num_bytes` bytes in the `output_buffer`. The caller is
  // responsible for ensuring that the `output_buffer` provided during
  // construction has sufficient space remaining to accommodate the skipped
  // bytes.
  void Skip(size_t num_bytes) { cur_offset_ += num_bytes; }

  // Returns true if the encoder is in dry-run mode.
  bool DryRun() const { return dry_run_; }

  // Returns the current offset into the `output_buffer` where the next encoded
  // value would be written.
  size_t CurOffset() const { return cur_offset_; }

  // Returns the remaining space in the `output_buffer` that has not yet been
  // used for encoding. If the encoder is in dry-run mode, an empty span is
  // returned.
  absl::Span<uint8_t> Remaining() const {
    return dry_run_ ? absl::Span<uint8_t>()
                    : output_buffer_.subspan(cur_offset_);
  }

 private:
  absl::Span<uint8_t> output_buffer_;

  const bool dry_run_;
  size_t cur_offset_;
};

class VarintDecoder {
  static std::pair<uint32_t, uint32_t> DecodeVarint(const uint8_t* ptr) {
    uint32_t value;
    const char* begin = reinterpret_cast<const char*>(ptr);
    const char* end = VarintEngine::Parse32(begin, &value);
    return std::make_pair(value, std::distance(begin, end));
  }

 public:
  // The decoder can read up to 7 bytes past the end of the encoded data. It is
  // recommended to add this many bytes to the end of the encoded data.
  static constexpr size_t kSlopBytes = 7;

  // Constructs a new decoder that reads from the `compressed_data` span.
  explicit VarintDecoder(absl::Span<const uint8_t> compressed_data)
      : cur_offset_(0), compressed_data_(compressed_data) {}

  // Decodes and returns the next `GingerNodeId` value from the compressed data.
  // The caller is responsible for ensuring that there is a next value to be
  // decoded.
  GingerNodeId Next() {
    ABSL_DCHECK_LT(cur_offset_, compressed_data_.size());
    const uint8_t* ptr = compressed_data_.data() + cur_offset_;
    const auto [value, length] = DecodeVarint(ptr);

    ABSL_DCHECK_LE(cur_offset_ + length, compressed_data_.size());
    cur_offset_ += length;

    return value;
  }

  // Returns the remaining space in the `compressed_data` that has not yet been
  // used for decoding.
  absl::Span<const uint8_t> Remaining() const {
    return compressed_data_.subspan(cur_offset_);
  }

 private:
  size_t cur_offset_;
  absl::Span<const uint8_t> compressed_data_;
};

}  // namespace ginger

#endif  // GINGER_GRAPH_COMPRESSION_VARINT_CODEC_H_
