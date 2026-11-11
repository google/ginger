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

#include "ginger/graph/compression/adaptive_gap_codec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "ginger/graph/varint.h"
#include "ginger/types.h"

#ifndef GINGER_GRAPH_COMPRESSION_ADAPTIVE_GAP_CODEC_TABLES_ONCE_
#define GINGER_GRAPH_COMPRESSION_ADAPTIVE_GAP_CODEC_TABLES_ONCE_
namespace ginger {
namespace {

// Shared by every SIMD target and the encoder. A shuffled lane holds at most 32
// bits, including the leading bit offset (`bit % 8`), and bit-packed books
// (`book != 0`) must leave the top three bits free (`width <= 29`).
// - Book 0 covers byte widths `{8, 16, 24, 32}`.
// - Books 1-6 use multiples of 4 (`bit % 8` in `{0, 4}`) with a 28-bit ceiling.
// - Books 7-11 use even widths (`bit % 8` in `{0, 2, 4, 6}`) with a 26-bit
//   ceiling.
// - Books 12-15 use even widths with 24-bit and 22-bit ceilings, including
//   `{2, 4, 8, 24}` for densely clustered neighborhoods.
// Because `gaps[0]` (the absolute first neighbor ID) can be encoded out-of-band
// as a `VarintEngine` varint anchor when `tag & kAnchorTagBit` is set, a 29-31
// bit first ID on billion-scale graphs does not force residual gaps onto Book
// 0.
constexpr uint8_t kWidths[][4] = {
    {8, 16, 24, 32}, {12, 16, 20, 28}, {8, 12, 20, 28},  {4, 12, 20, 28},
    {8, 16, 20, 28}, {8, 12, 16, 28},  {16, 20, 24, 28}, {10, 14, 18, 26},
    {6, 14, 22, 26}, {6, 10, 18, 26},  {14, 18, 22, 26}, {6, 14, 18, 26},
    {2, 4, 8, 24},   {8, 12, 16, 24},  {12, 16, 20, 24}, {6, 10, 14, 22}};
constexpr size_t kNumBooks = sizeof(kWidths) / sizeof(kWidths[0]);
static_assert(kNumBooks <= 16);

// List tag byte layout:
// `(has_anchor << 5) | (book_index << 1) | compressed_controls`.
constexpr uint8_t kCompressedControlsTagBit =
    AdaptiveGapDecoder::kCompressedControlsTagBit;
constexpr int kBookTagShift = AdaptiveGapDecoder::kBookTagShift;
constexpr uint8_t kBookTagMask = AdaptiveGapDecoder::kBookTagMask;
constexpr uint8_t kAnchorTagBit = AdaptiveGapDecoder::kAnchorTagBit;
constexpr uint8_t kReservedTagBits = 0xc0;
// Bit-packed decode metadata stores the shift count above the extraction mask.
constexpr int kShiftMetadataOffset = 29;
// Sparse 16-bit control masks cover up to 16 four-value groups (64 values).
constexpr size_t kMaxSparseControlValues = 64;

struct alignas(32) DecodeEntry {
  uint8_t shuffle[16];
  // For bit-packed books, bits 0-28 hold the extraction mask and bits 29-31
  // hold the right-shift count. After shifting, the value is zero wherever
  // the shift metadata has set bits, so ANDing with this whole word is safe.
  // Thus extraction is (values >> (meta >> 29)) & meta. Even shifts 0/2/4/6
  // put metadata only in bits cleared by the shift (not necessarily all of
  // bits 29-31). The assertions below enforce this; book 0 ignores metadata.
  uint32_t masks_and_shifts[4];
};
static_assert(sizeof(DecodeEntry) == 32);

static_assert(
    [] {
      for (size_t book_index = 0; book_index < kNumBooks; ++book_index) {
        const auto& book = kWidths[book_index];
        for (int selector = 0; selector < 4; ++selector) {
          if (book[selector] == 0 || book[selector] > 32) {
            return false;
          }
          if (selector > 0 && book[selector - 1] >= book[selector]) {
            return false;
          }
        }
        for (int control = 0; control < 256; ++control) {
          int bit = 0;
          for (int lane = 0; lane < 4; ++lane) {
            const int width = book[(control >> (2 * lane)) & 3];
            if (width + bit % 8 > 32) {
              return false;
            }
            // Only the byte-width kernel bypasses packed shift metadata. In
            // every other book, mask bits must leave the top three bits free.
            if (book_index != 0 && width > kShiftMetadataOffset) {
              return false;
            }
            const uint32_t shift = bit % 8;
            if ((shift << kShiftMetadataOffset) & (0xffffffffu >> shift)) {
              return false;
            }
            bit += width;
          }
          if (bit > 128) {
            return false;
          }
        }
      }
      return true;
    }(),
    "Widths must be positive, increasing, fit SIMD lanes and 128-bit loads, "
    "and leave top three bits free for shift metadata in bit-packed books");

consteval std::array<std::array<DecodeEntry, 256>, kNumBooks> MakeTables() {
  std::array<std::array<DecodeEntry, 256>, kNumBooks> tables{};
  for (size_t book = 0; book < kNumBooks; ++book) {
    for (int control = 0; control < 256; ++control) {
      DecodeEntry& entry = tables[book][control];
      int bit = 0;
      for (int lane = 0; lane < 4; ++lane) {
        const int width = kWidths[book][(control >> (lane * 2)) & 3];
        const uint32_t mask = width == 32 ? 0xffffffffu : (1u << width) - 1;
        entry.masks_and_shifts[lane] =
            mask | (static_cast<uint32_t>(bit % 8) << kShiftMetadataOffset);
        for (int byte = 0; byte < 4; ++byte) {
          entry.shuffle[lane * 4 + byte] =
              byte < (width + bit % 8 + 7) / 8 ? bit / 8 + byte : 0xff;
        }
        bit += width;
      }
    }
  }
  return tables;
}

alignas(64) constexpr std::array<std::array<DecodeEntry, 256>,
                                 kNumBooks> kTables = MakeTables();

consteval std::array<std::array<uint8_t, 256>, kNumBooks> MakeLengths() {
  std::array<std::array<uint8_t, 256>, kNumBooks> lengths{};
  for (size_t book = 0; book < kNumBooks; ++book) {
    for (int control = 0; control < 256; ++control) {
      int bits = 0;
      for (int lane = 0; lane < 4; ++lane) {
        bits += kWidths[book][(control >> (2 * lane)) & 3];
      }
      lengths[book][control] = (bits + 7) / 8;
    }
  }
  return lengths;
}

constexpr std::array<std::array<uint8_t, 256>, kNumBooks> kLengths =
    MakeLengths();

constexpr int kAbsentLaneBits = 33;

// Maps `(book, bit_width(gap))` directly to the 2-bit width selector `0..3`.
consteval std::array<std::array<uint8_t, 33>, kNumBooks>
MakeEncodingSelectors() {
  std::array<std::array<uint8_t, 33>, kNumBooks> selectors{};
  for (size_t book = 0; book < kNumBooks; ++book) {
    for (int bits = 0; bits <= 32; ++bits) {
      uint8_t selector = 0;
      while (selector < 3 && bits > kWidths[book][selector]) {
        ++selector;
      }
      selectors[book][bits] = selector;
    }
  }
  return selectors;
}

constexpr std::array<std::array<uint8_t, 33>, kNumBooks> kEncodingSelectors =
    MakeEncodingSelectors();

// Index by required bit width, then book. Row `kAbsentLaneBits` (33) represents
// an absent lane in the last group and contributes no payload. Zero gaps still
// use a width. Each row holds one byte per book, so the encoder measures all
// books with one 16-byte vector per lane.
consteval std::array<std::array<uint8_t, kNumBooks>, kAbsentLaneBits + 1>
MakeEncodingWidths() {
  std::array<std::array<uint8_t, kNumBooks>, kAbsentLaneBits + 1> widths{};
  for (int bits = 0; bits <= 32; ++bits) {
    for (size_t book = 0; book < kNumBooks; ++book) {
      widths[bits][book] = kWidths[book][kEncodingSelectors[book][bits]];
    }
  }
  return widths;
}

alignas(16) constexpr std::array<std::array<uint8_t, kNumBooks>,
                                 kAbsentLaneBits + 1> kEncodingWidths =
    MakeEncodingWidths();

// Smallest width of each book. A group needs a nonzero control byte iff its
// largest bit width exceeds this.
consteval std::array<uint8_t, kNumBooks> MakeMinWidths() {
  std::array<uint8_t, kNumBooks> min_widths{};
  for (size_t book = 0; book < kNumBooks; ++book) {
    min_widths[book] = kWidths[book][0];
  }
  return min_widths;
}

alignas(16) constexpr std::array<uint8_t, kNumBooks> kMinWidths =
    MakeMinWidths();

// Bitmask of books whose maximum width (`kWidths[book][3]`) is `>= bits`.
consteval std::array<uint16_t, 33> MakeEligibleBooksForMaxBits() {
  std::array<uint16_t, 33> masks{};
  for (int bits = 0; bits <= 32; ++bits) {
    uint16_t mask = 0;
    for (size_t book = 0; book < kNumBooks; ++book) {
      if (kWidths[book][3] >= bits) {
        mask |= static_cast<uint16_t>(uint16_t{1} << book);
      }
    }
    masks[bits] = mask;
  }
  return masks;
}

constexpr std::array<uint16_t, 33> kEligibleBooksForMaxBits =
    MakeEligibleBooksForMaxBits();

// Per-book sizes of the group-coded section, measured for every book. Entries
// of books that cannot hold `max_bits` are meaningless.
struct BookSizes {
  uint32_t payload_bytes[kNumBooks];
  // Number of nonzero control bytes, only counted for lists of at most
  // `kMaxSparseControlValues` values.
  uint8_t nonzero_controls[kNumBooks];
  int max_bits;
};
}  // namespace
}  // namespace ginger
#endif  // GINGER_GRAPH_COMPRESSION_ADAPTIVE_GAP_CODEC_TABLES_ONCE_

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "ginger/graph/compression/adaptive_gap_codec.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Expands a sparse control stream starting with a little-endian 16-bit presence
// bitmask at `cursor` into `expanded_controls` (setting positions with unset
// mask bits to zero) and returns a pointer to the first group payload byte.
const uint8_t* ExpandSparseControls(const uint8_t* cursor,
                                    uint8_t (&expanded_controls)[16]) {
  const uint16_t bits = static_cast<uint16_t>(cursor[0]) |
                        (static_cast<uint16_t>(cursor[1]) << 8);
  // Use LoadExpand only where x86 has native vpexpandb (AVX3_DL+); on AVX2/SSE4
  // emulated LoadMaskBits + LoadExpand + 16-byte stack store is 14-35% slower
  // than zero-filling and scanning the few set mask bits.
#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3_DL
  constexpr hn::CappedTag<uint8_t, 16> d8;
  const auto mask = hn::LoadMaskBits(d8, cursor);
  cursor += 2;
  hn::Store(hn::LoadExpand(mask, d8, cursor), d8, expanded_controls);
  cursor += std::popcount(bits);
#else
  cursor += 2;
  std::fill_n(expanded_controls, 16, uint8_t{0});
  for (uint16_t remaining = bits; remaining != 0; remaining &= remaining - 1) {
    expanded_controls[std::countr_zero(remaining)] = *cursor++;
  }
#endif
  return cursor;
}

#if HWY_TARGET == HWY_SCALAR
const uint8_t* DecodeGroupsScalar(const uint8_t* controls,
                                  const uint8_t* cursor, int book,
                                  uint32_t initial_id,
                                  absl::Span<GingerNodeId> output) {
  uint32_t previous = initial_id;
  for (size_t i = 0; i < output.size(); i += 4) {
    const uint8_t control = *controls++;
    int bit = 0;
    for (size_t j = 0; j < 4 && i + j < output.size(); ++j) {
      const int width = kWidths[book][(control >> (2 * j)) & 3];
      uint32_t raw;
      std::memcpy(&raw, cursor + bit / 8, sizeof(uint32_t));
      const uint32_t mask = width == 32 ? 0xffffffffu : (1u << width) - 1;
      previous += (raw >> (bit % 8)) & mask;
      output[i + j] = previous;
      bit += width;
    }
    cursor += (bit + 7) / 8;
  }
  return cursor;
}
#else
// Decodes expanded group `controls` and their corresponding `data` payload
// using the selected book's `table` and `lengths`, writing `output.size()`
// reconstructed IDs starting from `initial_id`. Scans four gaps within each
// group, then propagates the previous group's last ID across groups (processing
// four independent groups per main iteration and storing only valid lanes for a
// final partial group). `kByteWidths` specializes Book 0 ({8, 16, 24, 32}).
// Advances and returns `data` past complete 4-value groups only; for a partial
// final group, the returned pointer still points to that group's payload.
template <bool kByteWidths>
const uint8_t* DecodeGroups(const uint8_t* controls, const uint8_t* data,
                            const std::array<DecodeEntry, 256>& table,
                            const std::array<uint8_t, 256>& lengths,
                            uint32_t initial_id,
                            absl::Span<GingerNodeId> output) {
  static_assert(sizeof(GingerNodeId) == sizeof(uint32_t));
  constexpr hn::CappedTag<uint32_t, 4> d32;
  constexpr hn::CappedTag<uint8_t, 16> d8;
  auto prev = hn::Set(d32, initial_id);
  auto unpack_and_scan = [&](const uint8_t* cursor, uint8_t control) {
    const DecodeEntry& entry = table[control];
    auto values =
        hn::BitCast(d32, hn::TableLookupBytesOr0(hn::LoadU(d8, cursor),
                                                 hn::Load(d8, entry.shuffle)));
    if constexpr (!kByteWidths) {
      const auto meta = hn::Load(d32, entry.masks_and_shifts);
      values = hn::And(
          hn::Shr(values, hn::ShiftRight<kShiftMetadataOffset>(meta)), meta);
    }
    values = hn::Add(values, hn::ShiftLeftBytes<4>(values));
    return hn::Add(values, hn::ShiftLeftBytes<8>(values));
  };
  uint32_t* dest = reinterpret_cast<uint32_t*>(output.data());
  size_t remaining = output.size();
  // Unpack and sum within groups independently, then resolve inter-group
  // carries.
  for (; remaining >= 16; remaining -= 16, controls += 4, dest += 16) {
    const uint8_t control0 = controls[0];
    const uint8_t control1 = controls[1];
    const uint8_t control2 = controls[2];
    const uint8_t control3 = controls[3];
    const size_t len0 = lengths[control0];
    const size_t len1 = lengths[control1];
    const size_t len2 = lengths[control2];
    const size_t len3 = lengths[control3];
    auto group0 = unpack_and_scan(data, control0);
    auto group1 = unpack_and_scan(data + len0, control1);
    auto group2 = unpack_and_scan(data + len0 + len1, control2);
    auto group3 = unpack_and_scan(data + len0 + len1 + len2, control3);
    data += len0 + len1 + len2 + len3;
    group0 = hn::Add(group0, hn::BroadcastLane<3>(prev));
    group1 = hn::Add(group1, hn::BroadcastLane<3>(group0));
    group2 = hn::Add(group2, hn::BroadcastLane<3>(group1));
    group3 = hn::Add(group3, hn::BroadcastLane<3>(group2));
    hn::StoreU(group0, d32, dest);
    hn::StoreU(group1, d32, dest + 4);
    hn::StoreU(group2, d32, dest + 8);
    hn::StoreU(group3, d32, dest + 12);
    prev = group3;
  }
  for (; remaining >= 4; remaining -= 4, dest += 4) {
    const uint8_t control = *controls++;
    prev = hn::Add(unpack_and_scan(data, control), hn::BroadcastLane<3>(prev));
    data += lengths[control];
    hn::StoreU(prev, d32, dest);
  }
  if (remaining > 0) {
    const uint8_t control = *controls;
    prev = hn::Add(unpack_and_scan(data, control), hn::BroadcastLane<3>(prev));
    hn::StoreN(prev, d32, dest, remaining);
  }
  return data;
}
#endif

void DecodeAdaptive(absl::Span<const uint8_t> input,
                    absl::Span<GingerNodeId> output) {
  if (output.empty()) {
    return;
  }
  ABSL_DCHECK(!input.empty());
  const uint8_t* cursor = input.data();
  const uint8_t* const input_end = cursor + input.size();
  const uint8_t tag = *cursor++;
  ABSL_DCHECK_EQ(tag & kReservedTagBits, 0);
  const int book = (tag >> kBookTagShift) & kBookTagMask;
  const bool compressed_controls = (tag & kCompressedControlsTagBit) != 0;
  uint32_t initial_id = 0;
  if ((tag & kAnchorTagBit) != 0) {
    ABSL_DCHECK_LT(cursor, input_end);
    cursor = reinterpret_cast<const uint8_t*>(VarintEngine::Parse32(
        reinterpret_cast<const char*>(cursor), &initial_id));
    ABSL_DCHECK_LE(cursor, input_end);
    output[0] = static_cast<GingerNodeId>(initial_id);
    output = output.subspan(1);
    ABSL_DCHECK(!output.empty());
  }
  ABSL_DCHECK(!compressed_controls || output.size() <= kMaxSparseControlValues);
  const uint8_t* controls = cursor;
  alignas(16) uint8_t expanded_controls[16];
  if (compressed_controls) {
    ABSL_DCHECK_LE(cursor + 2, input_end);
    cursor = ExpandSparseControls(cursor, expanded_controls);
    controls = expanded_controls;
  } else {
    cursor += (output.size() + 3) / 4;
  }
  ABSL_DCHECK_LE(cursor, input_end);
#if HWY_TARGET == HWY_SCALAR
  cursor = DecodeGroupsScalar(controls, cursor, book, initial_id, output);
#else
  if (book == 0) {
    cursor = DecodeGroups<true>(controls, cursor, kTables[0], kLengths[0],
                                initial_id, output);
  } else {
    cursor = DecodeGroups<false>(controls, cursor, kTables[book],
                                 kLengths[book], initial_id, output);
  }
#ifndef NDEBUG
  if (const size_t tail = output.size() % 4; tail != 0) {
    const uint8_t last_control = controls[output.size() / 4];
    int bits = 0;
    for (size_t j = 0; j < tail; ++j) {
      bits += kWidths[book][(last_control >> (2 * j)) & 3];
    }
    cursor += (bits + 7) / 8;
  }
#endif
#endif
  ABSL_DCHECK_LE(cursor, input_end);
}

#if HWY_TARGET == HWY_SCALAR
void MeasureBooks(absl::Span<const GingerNodeId> gaps, BookSizes& sizes) {
  sizes = BookSizes{};
  uint32_t all_gaps = 0;
  for (size_t i = 0; i < gaps.size(); i += 4) {
    int bits[4] = {kAbsentLaneBits, kAbsentLaneBits, kAbsentLaneBits,
                   kAbsentLaneBits};
    uint32_t group_gaps = 0;
    for (size_t j = 0; j < 4 && i + j < gaps.size(); ++j) {
      const uint32_t gap = static_cast<uint32_t>(gaps[i + j]);
      bits[j] = std::bit_width(gap);
      group_gaps |= gap;
    }
    all_gaps |= group_gaps;
    const int group_max = std::bit_width(group_gaps);
    for (size_t book = 0; book < kNumBooks; ++book) {
      const int payload_bits =
          kEncodingWidths[bits[0]][book] + kEncodingWidths[bits[1]][book] +
          kEncodingWidths[bits[2]][book] + kEncodingWidths[bits[3]][book];
      sizes.payload_bytes[book] += (payload_bits + 7) / 8;
      sizes.nonzero_controls[book] += group_max > kMinWidths[book];
    }
  }
  sizes.max_bits = std::bit_width(all_gaps);
}
#else
// Measures one group in all 16 books at once: each lane of a width row is one
// book, and the sum of four 2..32-bit widths plus 7 fits in a byte. Adds the
// group's payload bytes to `payload` and, if `kCountControls`, subtracts the
// nonzero-control mask from `controls`. Returns the OR of the group's gaps.
template <bool kCountControls, class D, class V = hn::Vec<D> >
HWY_INLINE uint32_t MeasureGroup(D d8, const uint32_t (&gaps)[4],
                                 const int (&bits)[4], V& payload,
                                 V& controls) {
  const auto row = [&](int b) {
    return hn::Load(d8, kEncodingWidths[b].data());
  };
  const V sum = hn::Add(hn::Add(row(bits[0]), row(bits[1])),
                        hn::Add(row(bits[2]), row(bits[3])));
  payload = hn::Add(payload, hn::ShiftRight<3>(hn::Add(sum, hn::Set(d8, 7))));
  const uint32_t group_gaps = gaps[0] | gaps[1] | gaps[2] | gaps[3];
  if constexpr (kCountControls) {
    const V group_max =
        hn::Set(d8, static_cast<uint8_t>(std::bit_width(group_gaps)));
    controls = hn::Sub(
        controls, hn::VecFromMask(
                      d8, hn::Gt(group_max, hn::Load(d8, kMinWidths.data()))));
  }
  return group_gaps;
}

template <bool kCountControls>
void MeasureBooksImpl(absl::Span<const GingerNodeId> gaps, BookSizes& sizes) {
  static_assert(kNumBooks == 16);
  constexpr hn::CappedTag<uint8_t, 16> d8;
  static_assert(hn::MaxLanes(d8) == 16);
  const uint32_t* data = reinterpret_cast<const uint32_t*>(gaps.data());
  const size_t num_full_groups = gaps.size() / 4;
  std::fill_n(sizes.payload_bytes, kNumBooks, 0u);
  auto controls = hn::Zero(d8);
  uint32_t all_gaps = 0;
  alignas(16) uint8_t batch_bytes[16];
  // Byte lanes hold at most 15 groups of at most 16 payload bytes each.
  constexpr size_t kGroupsPerBatch = 15;
  size_t group = 0;
  while (group < num_full_groups) {
    const size_t batch_end = std::min(num_full_groups, group + kGroupsPerBatch);
    auto payload = hn::Zero(d8);
    for (; group < batch_end; ++group, data += 4) {
      const uint32_t group_gaps[4] = {data[0], data[1], data[2], data[3]};
      const int bits[4] = {std::bit_width(data[0]), std::bit_width(data[1]),
                           std::bit_width(data[2]), std::bit_width(data[3])};
      all_gaps |=
          MeasureGroup<kCountControls>(d8, group_gaps, bits, payload, controls);
    }
    hn::Store(payload, d8, batch_bytes);
    for (size_t book = 0; book < kNumBooks; ++book) {
      sizes.payload_bytes[book] += batch_bytes[book];
    }
  }
  if (const size_t tail = gaps.size() % 4; tail != 0) {
    uint32_t group_gaps[4] = {0, 0, 0, 0};
    int bits[4] = {kAbsentLaneBits, kAbsentLaneBits, kAbsentLaneBits,
                   kAbsentLaneBits};
    for (size_t j = 0; j < tail; ++j) {
      group_gaps[j] = data[j];
      bits[j] = std::bit_width(data[j]);
    }
    auto payload = hn::Zero(d8);
    all_gaps |=
        MeasureGroup<kCountControls>(d8, group_gaps, bits, payload, controls);
    hn::Store(payload, d8, batch_bytes);
    for (size_t book = 0; book < kNumBooks; ++book) {
      sizes.payload_bytes[book] += batch_bytes[book];
    }
  }
  hn::StoreU(controls, d8, sizes.nonzero_controls);
  sizes.max_bits = std::bit_width(all_gaps);
}

void MeasureBooks(absl::Span<const GingerNodeId> gaps, BookSizes& sizes) {
  if (gaps.size() <= kMaxSparseControlValues) {
    MeasureBooksImpl</*kCountControls=*/true>(gaps, sizes);
  } else {
    MeasureBooksImpl</*kCountControls=*/false>(gaps, sizes);
  }
}
#endif
}  // namespace HWY_NAMESPACE
}  // namespace
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {
HWY_EXPORT(DecodeAdaptive);
HWY_EXPORT(MeasureBooks);
namespace {

constexpr size_t kTagBytes = 1;
constexpr size_t kSparseControlMaskBytes = 2;

struct EncodingPlan {
  int book = 0;
  bool compress_controls = false;
  bool has_anchor = false;
  size_t control_bytes = 0;
  size_t total_bytes = 0;

  uint8_t Tag() const {
    ABSL_DCHECK_LT(book, kNumBooks);
    return static_cast<uint8_t>(
        (has_anchor ? kAnchorTagBit : 0) | (book << kBookTagShift) |
        (compress_controls ? kCompressedControlsTagBit : 0));
  }
};

EncodingPlan EvaluateGroupPlan(absl::Span<const GingerNodeId> group_gaps,
                               bool has_anchor, size_t prefix_bytes) {
  ABSL_DCHECK(!group_gaps.empty());
  BookSizes sizes;
  HWY_DYNAMIC_DISPATCH(MeasureBooks)(group_gaps, sizes);
  const size_t num_controls = (group_gaps.size() + 3) / 4;
  EncodingPlan best_plan{
      .total_bytes = std::numeric_limits<size_t>::max(),
  };
  for (uint16_t mask = kEligibleBooksForMaxBits[sizes.max_bits]; mask != 0;
       mask &= mask - 1) {
    const int book = std::countr_zero(mask);
    const size_t nonzero_controls = sizes.nonzero_controls[book];
    const bool compress_controls =
        group_gaps.size() <= kMaxSparseControlValues &&
        nonzero_controls + kSparseControlMaskBytes < num_controls;
    const size_t control_bytes =
        compress_controls ? kSparseControlMaskBytes + nonzero_controls
                          : num_controls;
    const size_t total_bytes =
        prefix_bytes + control_bytes + sizes.payload_bytes[book];
    if (total_bytes < best_plan.total_bytes) {
      best_plan = EncodingPlan{
          .book = book,
          .compress_controls = compress_controls,
          .has_anchor = has_anchor,
          .control_bytes = control_bytes,
          .total_bytes = total_bytes,
      };
    }
  }
  return best_plan;
}

absl::StatusOr<EncodingPlan> SelectEncodingPlan(
    absl::Span<const GingerNodeId> gaps) {
  if (gaps.empty()) {
    return EncodingPlan{};
  }
  uint64_t reconstructed_id = 0;
  for (const GingerNodeId gap : gaps) {
    if (gap < 0) {
      return absl::InvalidArgumentError("Negative gap");
    }
    reconstructed_id += static_cast<uint32_t>(gap);
    if (reconstructed_id >
        static_cast<uint64_t>(std::numeric_limits<GingerNodeId>::max())) {
      return absl::InvalidArgumentError(
          "Reconstructed ID overflows GingerNodeId");
    }
  }
  const EncodingPlan in_band_plan =
      EvaluateGroupPlan(gaps, /*has_anchor=*/false, kTagBytes);
  if (gaps.size() == 1) {
    return in_band_plan;
  }
  const size_t anchor_bytes =
      VarintEngine::Length32(static_cast<uint32_t>(gaps[0]));
  const EncodingPlan anchor_plan = EvaluateGroupPlan(
      gaps.subspan(1), /*has_anchor=*/true, kTagBytes + anchor_bytes);
  return anchor_plan.total_bytes < in_band_plan.total_bytes ? anchor_plan
                                                            : in_band_plan;
}

}  // namespace

absl::StatusOr<size_t> AdaptiveGapEncoder::EncodedSize(
    absl::Span<const GingerNodeId> gaps) {
  ABSL_ASSIGN_OR_RETURN(const EncodingPlan plan, SelectEncodingPlan(gaps));
  return plan.total_bytes;
}

absl::StatusOr<size_t> AdaptiveGapEncoder::Encode(
    absl::Span<const GingerNodeId> gaps, absl::Span<uint8_t> output) {
  ABSL_ASSIGN_OR_RETURN(const EncodingPlan plan, SelectEncodingPlan(gaps));
  if (gaps.empty()) {
    return 0;
  }
  if (output.size() < plan.total_bytes) {
    return absl::InvalidArgumentError("Output buffer too small");
  }
  output[0] = plan.Tag();
  size_t prefix_bytes = kTagBytes;
  if (plan.has_anchor) {
    char* const start = reinterpret_cast<char*>(output.data() + prefix_bytes);
    char* const end =
        VarintEngine::Encode32(start, static_cast<uint32_t>(gaps[0]));
    prefix_bytes += static_cast<size_t>(end - start);
    gaps = gaps.subspan(1);
  }
  size_t control_offset = plan.compress_controls
                              ? prefix_bytes + kSparseControlMaskBytes
                              : prefix_bytes;
  const size_t payload_offset = prefix_bytes + plan.control_bytes;
  size_t data_offset = payload_offset;
  uint16_t control_mask = 0;
  const auto& selectors = kEncodingSelectors[plan.book];
  const auto& widths = kWidths[plan.book];
  for (size_t i = 0; i < gaps.size(); i += 4) {
    uint64_t word = 0;
    int pending_bits = 0;
    uint8_t control = 0;
    for (size_t j = 0; j < 4 && i + j < gaps.size(); ++j) {
      const uint32_t gap = static_cast<uint32_t>(gaps[i + j]);
      const int selector = selectors[std::bit_width(gap)];
      const int width = widths[selector];
      control |= selector << (2 * j);
      word |= static_cast<uint64_t>(gap) << pending_bits;
      pending_bits += width;
      // `pending_bits < 32` before each gap, so `word` never overflows.
      if (pending_bits >= 32) {
        const uint32_t low = static_cast<uint32_t>(word);
        std::memcpy(&output[data_offset], &low, sizeof(low));
        data_offset += sizeof(low);
        word >>= 32;
        pending_bits -= 32;
      }
    }
    for (; pending_bits > 0; pending_bits -= 8) {
      output[data_offset++] = static_cast<uint8_t>(word);
      word >>= 8;
    }
    if (plan.compress_controls && control != 0) {
      control_mask |= static_cast<uint16_t>(uint16_t{1} << (i / 4));
    }
    if (!plan.compress_controls || control != 0) {
      output[control_offset++] = control;
    }
  }
  if (plan.compress_controls) {
    output[prefix_bytes] = static_cast<uint8_t>(control_mask & 0xff);
    output[prefix_bytes + 1] = static_cast<uint8_t>(control_mask >> 8);
  }
  ABSL_DCHECK_EQ(control_offset, payload_offset);
  ABSL_DCHECK_EQ(data_offset, plan.total_bytes);
  return plan.total_bytes;
}

absl::StatusOr<size_t> AdaptiveGapEncoder::Encode(
    absl::Span<const GingerNodeId> gaps, absl::Span<uint8_t> output,
    bool dry_run) {
  return dry_run ? EncodedSize(gaps) : Encode(gaps, output);
}

void AdaptiveGapDecoder::DecodeGaps(absl::Span<const uint8_t> input,
                                    absl::Span<GingerNodeId> output) {
  HWY_DYNAMIC_DISPATCH(DecodeAdaptive)(input, output);
}
}  // namespace ginger
#endif
