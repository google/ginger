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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/graph/varint.h"
#include "ginger/types.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "ginger/graph/compression/adaptive_gap_codec_test.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::ginger::AdaptiveGapDecoder;
using ::ginger::AdaptiveGapEncoder;
using ::ginger::GingerNodeId;
using ::ginger::VarintEngine;
using ::testing::ElementsAreArray;

// Test-local wire-format specification of the 16 width books. Duplicating the
// specification here keeps internal tables out of the public header and catches
// accidental reinterpretation of existing wire tags.
constexpr uint8_t kSpecWidths[][4] = {
    {8, 16, 24, 32}, {12, 16, 20, 28}, {8, 12, 20, 28},  {4, 12, 20, 28},
    {8, 16, 20, 28}, {8, 12, 16, 28},  {16, 20, 24, 28}, {10, 14, 18, 26},
    {6, 14, 22, 26}, {6, 10, 18, 26},  {14, 18, 22, 26}, {6, 14, 18, 26},
    {2, 4, 8, 24},   {8, 12, 16, 24},  {12, 16, 20, 24}, {6, 10, 14, 22}};
constexpr int kNumSpecBooks = sizeof(kSpecWidths) / sizeof(kSpecWidths[0]);

// Independent bit-at-a-time group payload writer used by wire-format fixtures.
void AppendGroupBits(absl::Span<const uint32_t> values,
                     absl::Span<const int> widths,
                     std::vector<uint8_t>& bytes) {
  int total_bits = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    total_bits += widths[i];
  }
  const size_t base_byte = bytes.size();
  bytes.resize(base_byte + (total_bits + 7) / 8, 0);
  int bit_pos = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    const uint32_t value = values[i];
    const int width = widths[i];
    for (int bit = 0; bit < width; ++bit, ++bit_pos) {
      if ((value >> bit) & 1u) {
        bytes[base_byte + bit_pos / 8] |=
            static_cast<uint8_t>(1u << (bit_pos % 8));
      }
    }
  }
}

HWY_NOINLINE void AdaptiveRoundTrips() {
  std::seed_seq seed{987};
  std::mt19937 rng(seed);
  for (int iteration = 0; iteration < 600; ++iteration) {
    const int n = iteration < 140 ? iteration : rng() % 300;
    std::vector<GingerNodeId> gaps(n), expected(n), decoded(n + 2, -1);
    uint32_t sum = 0;
    for (int i = 0; i < n; ++i) {
      uint32_t value = rng() & ((1u << (iteration % 28)) - 1);
      if (iteration % 7 == 0) {
        value = i % 11 ? 1 : rng() % 10000000;
      }
      if (iteration % 13 == 0) {
        value = 0;
      }
      if (i == 0 && iteration % 17 == 0) {
        value = 0x7fffffff;
      }
      if (static_cast<uint64_t>(sum) + value > 0x7fffffff) {
        value = 0;
      }
      gaps[i] = value;
      sum += value;
      expected[i] = sum;
    }
    ASSERT_OK_AND_ASSIGN(const size_t size,
                         AdaptiveGapEncoder::EncodedSize(gaps));
    EXPECT_THAT(AdaptiveGapEncoder::Encode(gaps, {}, true), IsOkAndHolds(size));
    std::vector<uint8_t> data(size + AdaptiveGapDecoder::kSlopBytes, 0xa5);
    ASSERT_OK_AND_ASSIGN(const size_t actual, AdaptiveGapEncoder::Encode(
                                                  gaps, absl::MakeSpan(data)));
    EXPECT_EQ(size, actual);
    AdaptiveGapDecoder::DecodeGaps(absl::MakeConstSpan(data).first(size),
                                   absl::MakeSpan(decoded).subspan(1, n));
    EXPECT_EQ(decoded.front(), -1);
    EXPECT_EQ(decoded.back(), -1);
    EXPECT_THAT(absl::MakeConstSpan(decoded).subspan(1, n),
                ElementsAreArray(expected))
        << iteration;
    for (size_t j = size; j < data.size(); ++j) {
      EXPECT_EQ(data[j], 0xa5);
    }
  }
}

HWY_NOINLINE void AdaptiveAllControls() {
  for (int book = 0; book < kNumSpecBooks; ++book) {
    for (int control = 0; control < 256; ++control) {
      for (int compressed = 0; compressed < 2; ++compressed) {
        std::vector<uint8_t> bytes{static_cast<uint8_t>(2 * book + compressed)};
        if (compressed) {
          bytes.push_back(control ? 15 : 0);
          bytes.push_back(0);
        }
        if (!compressed || control) {
          for (int group = 0; group < 4; ++group) {
            bytes.push_back(control);
          }
        }
        std::vector<GingerNodeId> expected;
        uint32_t sum = 0;
        for (int group = 0; group < 4; ++group) {
          uint32_t group_values[4];
          int group_widths[4];
          for (int lane = 0; lane < 4; ++lane) {
            const int width = kSpecWidths[book][(control >> (2 * lane)) & 3];
            const uint32_t value = std::min<uint32_t>(
                (uint64_t{1} << width) - 1, 0x7fffffffu - sum);
            sum += value;
            expected.push_back(sum);
            group_values[lane] = value;
            group_widths[lane] = width;
          }
          AppendGroupBits(group_values, group_widths, bytes);
        }
        const size_t payload_size = bytes.size();
        bytes.resize(payload_size + AdaptiveGapDecoder::kSlopBytes, 0xa5);
        std::vector<GingerNodeId> out(16);
        AdaptiveGapDecoder::DecodeGaps(
            absl::MakeConstSpan(bytes).first(payload_size),
            absl::MakeSpan(out));
        EXPECT_EQ(out, expected) << "book=" << book << " control=" << control
                                 << " compressed=" << compressed;
      }
    }
  }

  // Exercise sparse control masks across low and high bytes (zero, single bits
  // 0/7/8/15, and mixed positions) with distinct nonzero controls so any
  // ordering or high-byte expansion mistake is visible.
  constexpr uint16_t kMasks[] = {0x0000, 0x0001, 0x0080, 0x0100,
                                 0x8000, 0x8181, 0xa55a, 0xffff};
  constexpr uint8_t kDistinctControls[16] = {0x01, 0x06, 0x1b, 0x27, 0x39, 0x4e,
                                             0x63, 0x78, 0x8d, 0x92, 0xa7, 0xbc,
                                             0xc1, 0xd6, 0xeb, 0xff};
  for (int book : {0, 2, 11}) {
    for (uint16_t mask : kMasks) {
      std::vector<uint8_t> bytes{static_cast<uint8_t>(2 * book + 1),
                                 static_cast<uint8_t>(mask & 0xff),
                                 static_cast<uint8_t>(mask >> 8)};
      uint8_t group_controls[16] = {};
      for (int group = 0; group < 16; ++group) {
        if ((mask >> group) & 1u) {
          group_controls[group] = kDistinctControls[group];
          bytes.push_back(group_controls[group]);
        }
      }
      std::vector<GingerNodeId> expected;
      uint32_t sum = 0;
      for (int group = 0; group < 16; ++group) {
        const uint8_t control = group_controls[group];
        uint32_t group_values[4];
        int group_widths[4];
        for (int lane = 0; lane < 4; ++lane) {
          const int selector = (control >> (2 * lane)) & 3;
          const int width = kSpecWidths[book][selector];
          const uint32_t max_val =
              width == 32 ? 0xffffffffu : (uint32_t{1} << width) - 1;
          const uint32_t value =
              (1u + group * 5 + lane * 3 + selector * 7) & max_val;
          sum += value;
          expected.push_back(sum);
          group_values[lane] = value;
          group_widths[lane] = width;
        }
        AppendGroupBits(group_values, group_widths, bytes);
      }
      const size_t payload_size = bytes.size();
      bytes.resize(payload_size + AdaptiveGapDecoder::kSlopBytes, 0xa5);
      std::vector<GingerNodeId> out(64, -1);
      AdaptiveGapDecoder::DecodeGaps(
          absl::MakeConstSpan(bytes).first(payload_size), absl::MakeSpan(out));
      EXPECT_EQ(out, expected) << "book=" << book << " mask=" << mask;
    }
  }
}

HWY_NOINLINE void AdaptiveMediumGapBoundaries() {
  // These lists contain no tiny gaps and exercise every new book. Include
  // first absolute IDs, width boundaries, partial groups, and >64 values.
  for (int n : {1, 2, 3, 4, 15, 16, 17, 63, 64, 65}) {
    for (int bits = 8; bits <= 30; ++bits) {
      std::vector<GingerNodeId> gaps(n), expected(n), out(n + 2, -1);
      uint32_t sum = 0;
      for (int i = 0; i < n; ++i) {
        uint32_t gap = (1u << bits) - 1 + (i % 3);
        if (static_cast<uint64_t>(sum) + gap > 0x7fffffff) {
          gap = 0;
        }
        gaps[i] = gap;
        expected[i] = sum += gap;
      }
      ASSERT_OK_AND_ASSIGN(const size_t size,
                           AdaptiveGapEncoder::EncodedSize(gaps));
      std::vector<uint8_t> data(size + AdaptiveGapDecoder::kSlopBytes, 0xa5);
      EXPECT_THAT(AdaptiveGapEncoder::Encode(gaps, absl::MakeSpan(data)),
                  IsOkAndHolds(size));
      AdaptiveGapDecoder::DecodeGaps(absl::MakeConstSpan(data).first(size),
                                     absl::MakeSpan(out).subspan(1, n));
      EXPECT_THAT(absl::MakeConstSpan(out).subspan(1, n),
                  ElementsAreArray(expected));
      EXPECT_EQ(out.front(), -1);
      EXPECT_EQ(out.back(), -1);
      for (size_t j = size; j < data.size(); ++j) {
        EXPECT_EQ(data[j], 0xa5);
      }
    }
  }
  for (const auto [gap, book] :
       {std::pair{1023, 7}, {4095, 1}, {16383, 10}, {65535, 6}}) {
    std::vector<GingerNodeId> gaps(64, gap);
    std::vector<uint8_t> data(512);
    ASSERT_OK_AND_ASSIGN(const size_t size, AdaptiveGapEncoder::Encode(
                                                gaps, absl::MakeSpan(data)));
    EXPECT_GT(size, 0);
    EXPECT_FALSE(AdaptiveGapDecoder::TagHasAnchor(data[0]));
    EXPECT_EQ(AdaptiveGapDecoder::TagBook(data[0]), book);
  }
}

HWY_NOINLINE void AdaptiveWideFirstId() {
  // A wide absolute first ID (including 29-31 bit IDs at billion scale) must be
  // decoupled via the VarintEngine varint anchor rather than forcing all
  // following small gaps onto a wide-floor book. Cover partial groups and the
  // compressed-control cutoff (including 65 total values = 1 anchor + 64
  // residual gaps).
  for (int n : {17, 64, 65, 66}) {
    for (int bits : {24, 26, 28, 30}) {
      std::vector<GingerNodeId> gaps(n, bits == 24 ? 1 : 256);
      gaps[0] = 1u << bits;
      ASSERT_OK_AND_ASSIGN(const size_t size,
                           AdaptiveGapEncoder::EncodedSize(gaps));
      std::vector<uint8_t> data(size + AdaptiveGapDecoder::kSlopBytes, 0xa5);
      ASSERT_OK_AND_ASSIGN(
          const size_t actual,
          AdaptiveGapEncoder::Encode(gaps, absl::MakeSpan(data)));
      EXPECT_EQ(actual, size);
      EXPECT_TRUE(AdaptiveGapDecoder::TagHasAnchor(data[0]));
      EXPECT_EQ(AdaptiveGapDecoder::TagBook(data[0]), bits == 24 ? 12 : 7);
      EXPECT_EQ(AdaptiveGapDecoder::TagHasCompressedControls(data[0]), n <= 65);
      std::vector<GingerNodeId> decoded(n);
      AdaptiveGapDecoder::DecodeGaps(absl::MakeConstSpan(data).first(size),
                                     absl::MakeSpan(decoded));
      GingerNodeId expected = 0;
      for (int i = 0; i < n; ++i) {
        expected += gaps[i];
        EXPECT_EQ(decoded[i], expected);
      }
    }
  }
}

HWY_NOINLINE void AdaptiveWireExamples() {
  std::vector<GingerNodeId> gaps{0, 1, 2, 3};
  std::vector<uint8_t> data(64);
  ASSERT_OK_AND_ASSIGN(size_t size,
                       AdaptiveGapEncoder::Encode(gaps, absl::MakeSpan(data)));
  EXPECT_EQ(size, 3);
  EXPECT_EQ(std::vector<uint8_t>(data.begin(), data.begin() + 3),
            (std::vector<uint8_t>{24, 0, 0xe4}));
  gaps.assign(16, 1);
  ASSERT_OK_AND_ASSIGN(size_t size_16,
                       AdaptiveGapEncoder::Encode(gaps, absl::MakeSpan(data)));
  EXPECT_EQ(size_16, 7);
  EXPECT_EQ(data[0], 25);
  EXPECT_EQ(data[1], 0);
  EXPECT_EQ(data[2], 0);
  for (int i = 3; i < 7; ++i) {
    EXPECT_EQ(data[i], 0x55);
  }
  std::vector<GingerNodeId> decoded(16);
  AdaptiveGapDecoder::DecodeGaps(absl::MakeConstSpan(data).first(size_16),
                                 absl::MakeSpan(decoded));
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(decoded[i], i + 1);
  }

  // 30-bit first ID decoupled as a 5-byte VarintEngine varint anchor followed
  // by four 2-bit residual gaps `{0, 1, 2, 3}` on Book 12
  // (`tag = 0x20 | (12 << 1)`).
  gaps = {1 << 29, 0, 1, 2, 3};
  ASSERT_OK_AND_ASSIGN(const size_t size_anchored,
                       AdaptiveGapEncoder::Encode(gaps, absl::MakeSpan(data)));
  std::vector<uint8_t> expected_anchored(8);
  expected_anchored[0] = 0x38;
  char* const anchor_end = VarintEngine::Encode32(
      reinterpret_cast<char*>(expected_anchored.data() + 1),
      static_cast<uint32_t>(gaps[0]));
  ASSERT_EQ(anchor_end - reinterpret_cast<char*>(expected_anchored.data()), 6);
  expected_anchored[6] = 0x00;
  expected_anchored[7] = 0xe4;
  EXPECT_EQ(size_anchored, expected_anchored.size());
  EXPECT_EQ(std::vector<uint8_t>(data.begin(), data.begin() + size_anchored),
            expected_anchored);
  decoded.assign(5, -1);
  AdaptiveGapDecoder::DecodeGaps(absl::MakeConstSpan(data).first(size_anchored),
                                 absl::MakeSpan(decoded));
  EXPECT_THAT(decoded, ElementsAreArray({(1 << 29), (1 << 29), (1 << 29) + 1,
                                         (1 << 29) + 3, (1 << 29) + 6}));
  // Empty streams have no tag and need no readable input or writable output.
  EXPECT_THAT(AdaptiveGapEncoder::EncodedSize({}), IsOkAndHolds(0));
  EXPECT_THAT(AdaptiveGapEncoder::Encode({}, {}), IsOkAndHolds(0));
  EXPECT_THAT(AdaptiveGapEncoder::Encode({}, {}, true), IsOkAndHolds(0));
  AdaptiveGapDecoder::DecodeGaps({}, {});

  // Reject undersized spans before any write, including a missing tag byte.
  for (int n : {1, 3, 16, 64, 65}) {
    gaps.assign(n, 1);
    ASSERT_OK_AND_ASSIGN(const size_t required,
                         AdaptiveGapEncoder::EncodedSize(gaps));
    data.assign(required + AdaptiveGapDecoder::kSlopBytes, 0xa5);
    for (size_t capacity : {size_t{0}, required - 1}) {
      EXPECT_FALSE(
          AdaptiveGapEncoder::Encode(gaps, absl::MakeSpan(data).first(capacity))
              .ok());
      EXPECT_EQ(data, std::vector<uint8_t>(data.size(), 0xa5));
    }
    EXPECT_THAT(
        AdaptiveGapEncoder::Encode(gaps, absl::MakeSpan(data).first(required)),
        IsOkAndHolds(required));
  }
  gaps = {1, -1};
  EXPECT_FALSE(AdaptiveGapEncoder::EncodedSize(gaps).ok());
  EXPECT_FALSE(AdaptiveGapEncoder::Encode(gaps, {}, true).ok());

  // Reconstructed ID overflow checks at GingerNodeId::max() boundaries.
  constexpr GingerNodeId kMaxId = std::numeric_limits<GingerNodeId>::max();
  for (const std::vector<GingerNodeId>& valid_max :
       {std::vector<GingerNodeId>{kMaxId}, std::vector<GingerNodeId>{kMaxId, 0},
        std::vector<GingerNodeId>{kMaxId - 1, 1}}) {
    ASSERT_OK_AND_ASSIGN(const size_t max_size,
                         AdaptiveGapEncoder::EncodedSize(valid_max));
    std::vector<uint8_t> max_buf(max_size + AdaptiveGapDecoder::kSlopBytes, 0);
    ASSERT_THAT(AdaptiveGapEncoder::Encode(valid_max, absl::MakeSpan(max_buf)),
                IsOkAndHolds(max_size));
    std::vector<GingerNodeId> max_decoded(valid_max.size(), 0);
    AdaptiveGapDecoder::DecodeGaps(absl::MakeConstSpan(max_buf).first(max_size),
                                   absl::MakeSpan(max_decoded));
    EXPECT_EQ(max_decoded.back(), kMaxId);
  }
  for (const std::vector<GingerNodeId>& overflow_gaps :
       {std::vector<GingerNodeId>{kMaxId, 1},
        std::vector<GingerNodeId>{1000000000, 1000000000, 147483647, 0, 1}}) {
    data.assign(64, 0xa5);
    EXPECT_FALSE(AdaptiveGapEncoder::EncodedSize(overflow_gaps).ok());
    EXPECT_FALSE(
        AdaptiveGapEncoder::Encode(overflow_gaps, absl::MakeSpan(data)).ok());
    EXPECT_EQ(data, std::vector<uint8_t>(data.size(), 0xa5));
  }

  // Regression test for long lists (>128 values, group index >= 32) with
  // nonzero control in group 32 when sparse controls are disabled.
  std::vector<GingerNodeId> long_gaps(129, 1);
  long_gaps[128] = 1 << 28;
  ASSERT_OK_AND_ASSIGN(const size_t long_size,
                       AdaptiveGapEncoder::EncodedSize(long_gaps));
  std::vector<uint8_t> long_bytes(long_size + AdaptiveGapDecoder::kSlopBytes,
                                  0xa5);
  ASSERT_THAT(AdaptiveGapEncoder::Encode(long_gaps, absl::MakeSpan(long_bytes)),
              IsOkAndHolds(long_size));
  EXPECT_EQ(AdaptiveGapDecoder::TagBook(long_bytes[0]),
            0);  // Book 0, uncompressed controls.
  EXPECT_FALSE(AdaptiveGapDecoder::TagHasCompressedControls(long_bytes[0]));
  std::vector<GingerNodeId> long_decoded(long_gaps.size(), -1);
  AdaptiveGapDecoder::DecodeGaps(
      absl::MakeConstSpan(long_bytes).first(long_size),
      absl::MakeSpan(long_decoded));
  EXPECT_EQ(long_decoded.front(), 1);
  EXPECT_EQ(long_decoded[127], 128);
  EXPECT_EQ(long_decoded[128], 268435584);
}
}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(AdaptiveGapCodecTest);
HWY_EXPORT_AND_TEST_P(AdaptiveGapCodecTest, AdaptiveRoundTrips);
HWY_EXPORT_AND_TEST_P(AdaptiveGapCodecTest, AdaptiveWireExamples);
HWY_EXPORT_AND_TEST_P(AdaptiveGapCodecTest, AdaptiveAllControls);
HWY_EXPORT_AND_TEST_P(AdaptiveGapCodecTest, AdaptiveMediumGapBoundaries);
HWY_EXPORT_AND_TEST_P(AdaptiveGapCodecTest, AdaptiveWideFirstId);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();
#endif
