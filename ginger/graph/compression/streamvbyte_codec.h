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

#ifndef GINGER_GRAPH_COMPRESSION_STREAMVBYTE_CODEC_H_
#define GINGER_GRAPH_COMPRESSION_STREAMVBYTE_CODEC_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {

// StreamVByte is a variable-byte encoding scheme that groups four 32-bit
// integers together. The encoding consists of two parts: first, a stream of
// control bytes, followed by a stream of the encoded integer data.
// Each group of four integers has a corresponding single "control byte".
// The control byte uses two bits for each of the four integers to indicate
// their encoded byte length (1 to 4 bytes). Decoding uses precomputed
// lookup tables based on the control byte to efficiently determine the total
// byte length of the encoded quad and a shuffle mask to extract the four
// 32-bit values for SIMD operations.
//
// The following is an example usage with StreamVByte:
// ```cpp
// std::vector<GingerNodeId> values = {1, 2, 3, 10, 12};
// std::vector<GingerNodeId> delta_values;
// delta_values.reserve(values.size());
//
// GingerNodeId prev_value = 0;
// for (size_t i = 0; i < values.size(); ++i) {
//   delta_values.push_back(values[i] - prev_value);
//   prev_value = values[i];
// }
//
// // Step 1: Dry run to determine the required buffer size.
// ABSL_ASSIGN_OR_RETURN(size_t encoded_size,
//                       StreamVByteEncoder::Encode(delta_values, {},
//                                                  /*dry_run=*/true));
// encoded_size += StreamVByteDecoder::kSlopBytes;
//
// // Step 2: Allocate the buffer and encode the gaps.
// std::vector<uint8_t> encoded_buffer(encoded_size);
// ABSL_RETURN_IF_ERROR(
//     StreamVByteEncoder::Encode(delta_values, absl::MakeSpan(encoded_buffer))
//         .status());
//
// // Step 3: Decode the gaps back into the original values.
// std::vector<GingerNodeId> decoded_values(delta_values.size());
// StreamVByteDecoder::DecodeGaps(encoded_buffer,
//                                absl::MakeSpan(decoded_values));
// // decoded_values == values
// ```

struct StreamVByteEncoder {
  // Encodes the given `values` into the given `output_buffer`. The values must
  // be non-negative and the `output_buffer` must have sufficient space to
  // store the encoded data. If `dry_run` is set to true, the encoder will not
  // write to `output_buffer`; instead, it will only compute the total size
  // required for the encoded data. Consequently, when `dry_run` is true, the
  // contents of `output_buffer` are ignored.
  // Returns the number of bytes written to the `output_buffer` or an error if
  // any input value is negative.
  static absl::StatusOr<size_t> Encode(absl::Span<const GingerNodeId> values,
                                       absl::Span<uint8_t> output_buffer,
                                       bool dry_run = false);
};

struct StreamVByteDecoder {
  // The decoder can read up to 15 bytes past the end of the encoded data. It is
  // recommended to add this many bytes to the end of the encoded data.
  static constexpr size_t kSlopBytes = 15;

  // Decodes the given `compressed_data` into the given `output_buffer`. The
  // `compressed_data` is interpreted as a sequence of differences of
  // non-negative 32-bit integers encoded using StreamVByte. The caller is
  // responsible for ensuring that `compressed_data` has enough values to be
  // decoded to fill the `output_buffer`.
  static void DecodeGaps(absl::Span<const uint8_t> compressed_data,
                         absl::Span<GingerNodeId> output_buffer);
};

}  // namespace ginger

#endif  // GINGER_GRAPH_COMPRESSION_STREAMVBYTE_CODEC_H_
