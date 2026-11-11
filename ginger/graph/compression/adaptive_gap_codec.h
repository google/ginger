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

#ifndef GINGER_GRAPH_COMPRESSION_ADAPTIVE_GAP_CODEC_H_
#define GINGER_GRAPH_COMPRESSION_ADAPTIVE_GAP_CODEC_H_

// Adaptive SIMD gap coding for sorted neighbor lists.
//
// Overview & Usage
// ----------------
// The caller converts sorted neighbor IDs to nonnegative gaps before encoding
// (e.g., `{1000, 1005, 1010}` -> `{1000, 5, 5}`, where the first value is the
// absolute first neighbor ID and subsequent values are differences between
// adjacent IDs). Zero gaps are supported; negative gaps and gaps whose
// cumulative sum overflows `GingerNodeId` are rejected. Decoding performs a
// prefix sum so `DecodeGaps` writes reconstructed node IDs directly:
//
//   const std::vector<GingerNodeId> gaps = {1000, 5, 5};
//   ASSERT_OK_AND_ASSIGN(const size_t size,
//                        AdaptiveGapEncoder::EncodedSize(gaps));
//   std::vector<uint8_t> encoded(size + AdaptiveGapDecoder::kSlopBytes);
//   ASSERT_OK_AND_ASSIGN(
//       const size_t written,
//       AdaptiveGapEncoder::Encode(gaps, absl::MakeSpan(encoded).first(size)));
//   std::vector<GingerNodeId> decoded(gaps.size());
//   AdaptiveGapDecoder::DecodeGaps(absl::MakeConstSpan(encoded).first(written),
//                                  absl::MakeSpan(decoded));
//   // `decoded` now holds `{1000, 1005, 1010}`.
//
// Width Books & Selection
// -----------------------
// VByte (the seven-data-bit continuation-bit Varint encoding) and
// `PrefixVarint` (`VarintEncoder` / `VarintDecoder`, which prefixes the byte
// length in the first byte) encode each integer independently at byte
// granularity. `StreamVByte` separates controls from payload, using one control
// byte per four values where each two-bit selector chooses a width from
// `{8, 16, 24, 32}` bits. This codec keeps `StreamVByte`'s 4-lane 2-bit
// selector structure, but selects one 4-width "book" per neighbor list from 16
// candidate books and optionally encodes the first entry (`gaps[0]`, the
// absolute first neighbor ID) out-of-band as a `VarintEngine` varint anchor
// before the control stream.
//
// A book is a menu of four allowed bit widths for the group-coded values, not
// one width per position. For example, with Book 3 `{4, 12, 20, 28}`, selector
// `00` -> 4 bits, `01` -> 12, `10` -> 20, and `11` -> 28, so `[5, 1000, 7,
// 30000000]` uses widths `[4, 12, 4, 28]`. Decoupling `gaps[0]` as an optional
// `VarintEngine` varint anchor prevents a large initial node ID from forcing
// the remaining gaps onto Book 0 (`{8, 16, 24, 32}`).
//
// Exactly one book applies to the group-coded portion of a list. A book is
// eligible only if its largest width fits every group-coded value. The encoder
// evaluates all eligible books both with and without the out-of-band anchor and
// chooses the representation with the minimum total byte size (including tag,
// optional anchor, control section, and byte-aligned group payloads, breaking
// ties toward in-band encoding and earlier books). The decoder reads the anchor
// flag, book index, and control mode directly from the list tag without
// searching.
//
// Wire Format
// -----------
// Each encoded list uses one of two control layouts, optionally preceded by a
// `VarintEngine` varint anchor right after `tag` (controls and payloads are
// never interleaved; degree and offset metadata are stored externally by the
// caller):
//
//   Uncompressed controls: [tag] [anchor?] [all control bytes]      [payloads]
//   Compressed controls:   [tag] [anchor?] [16-bit mask] [controls] [payloads]
//
// - `tag`: A single byte `(has_anchor << 5) | (book_index << 1) |
//   compressed_controls`, where `(tag & 0x20) != 0` selects the out-of-band
//   `VarintEngine` varint anchor for `gaps[0]`, `(tag >> 1) & 0x0f` selects the
//   book (`0..15`), and `tag & 1` selects whether the control stream is
//   sparse-compressed.
// - `anchor?`: Present when `(tag & 0x20) != 0` (used only when `n > 1`).
//   Encodes `gaps[0]` in 1-5 bytes using `VarintEngine`, leaving the remaining
//   `m = n - 1 >= 1` residual gaps (`gaps[1..n-1]`) for the group control and
//   payload streams.
// - Controls: For `m` group-coded values (`n` without anchor, `n - 1` with
//   anchor) there are `ceil(m / 4)` groups of up to 4 values. Selector `j`
//   (`0..3`) occupies bits `[2*j, 2*j+1]` of its group's control byte. A zero
//   control byte means all four values in that group use the book's minimum
//   width (`book[0]`), not that the values are zero. For `m <= 64` (`<= 16`
//   groups), the encoder replaces the control array with a little-endian 16-bit
//   presence bitmask followed only by the nonzero control bytes whenever
//   `2 + popcount(mask) < ceil(m / 4)`.
// - Group payloads: Within each 4-value group, values are bit-packed low bits
//   first. Each group begins at a byte boundary (padding any partial trailing
//   byte with zeros). The final group may contain one, two, or three values.
//   Encode only those values, round their combined payload up to a whole byte,
//   and set the unused two-bit selectors to `00`. No payload is stored for the
//   missing values (for example, three 2-bit values occupy one payload byte;
//   the fourth selector is `00`, but there is no fourth encoded gap).
//
// SIMD Decoding & Padding
// -----------------------
// For each `(book, control)` pair, compile-time lookup tables store a 16-byte
// shuffle mask, per-lane extraction shifts/masks, and the group byte length.
// Every width book guarantees that each value plus its starting bit offset
// (`bit % 8`) fits within a single 32-bit SIMD lane (`<= 32` bits) and a
// 16-byte unaligned load. A byte shuffle places each value's bytes into its
// 32-bit lane, per-lane right-shifts and masks extract the bit-packed gaps in
// parallel (skipped for Book 0's byte-aligned widths), and SIMD prefix sums
// reconstruct the neighbor IDs.
//
// Highway runtime dispatch (`HWY_DYNAMIC_DISPATCH`) selects scalar or 128-bit
// SIMD decoding (unpacking 4 groups = 16 IDs/step in 128-bit registers and
// propagating carries across groups). Input buffers must provide
// `kSlopBytes = 15` readable bytes past the logical end of the encoded stream
// to accommodate 16-byte SIMD loads (callers packing multiple lists into a
// contiguous buffer may share this 15-byte padding once at the end of the
// allocation).

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {

struct AdaptiveGapEncoder {
  // Computes the exact encoded byte count for nonnegative gaps (the first gap
  // is the first ID); returns InvalidArgumentError for negative gaps or if
  // reconstructed IDs overflow GingerNodeId. Empty input uses zero bytes.
  static absl::StatusOr<size_t> EncodedSize(
      absl::Span<const GingerNodeId> gaps);

  // Encodes nonnegative gaps (the first gap is the first ID); returns the byte
  // count, or InvalidArgumentError for negative gaps, reconstructed ID
  // overflow, or insufficient output capacity. Empty input uses zero bytes.
  static absl::StatusOr<size_t> Encode(absl::Span<const GingerNodeId> gaps,
                                       absl::Span<uint8_t> output);

  // Compatibility overload for BatchGapEncoder callers: with `dry_run`, ignores
  // `output` and returns `EncodedSize(gaps)`; otherwise encodes into `output`.
  static absl::StatusOr<size_t> Encode(absl::Span<const GingerNodeId> gaps,
                                       absl::Span<uint8_t> output,
                                       bool dry_run);
};

struct AdaptiveGapDecoder {
  static constexpr size_t kSlopBytes = 15;
  static constexpr uint8_t kCompressedControlsTagBit = 0x01;
  static constexpr int kBookTagShift = 1;
  static constexpr uint8_t kBookTagMask = 0x0f;
  static constexpr uint8_t kAnchorTagBit = 0x20;

  static constexpr bool TagHasAnchor(uint8_t tag) {
    return (tag & kAnchorTagBit) != 0;
  }
  static constexpr int TagBook(uint8_t tag) {
    return (tag >> kBookTagShift) & kBookTagMask;
  }
  static constexpr bool TagHasCompressedControls(uint8_t tag) {
    return (tag & kCompressedControlsTagBit) != 0;
  }

  // Decodes gaps and reconstructs IDs. Input must be valid encoder output
  // followed by kSlopBytes accessible bytes (outside the span is allowed).
  // output.size() must match the encoded count, and prefix sums must fit in
  // GingerNodeId. Writes exactly that many IDs; empty output is a no-op.
  static void DecodeGaps(absl::Span<const uint8_t> input,
                         absl::Span<GingerNodeId> output);
};

}  // namespace ginger
#endif  // GINGER_GRAPH_COMPRESSION_ADAPTIVE_GAP_CODEC_H_
