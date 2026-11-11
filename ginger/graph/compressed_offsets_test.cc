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

#include "ginger/graph/compressed_offsets.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "ginger/internal/gunit_support.h"

namespace ginger {
namespace {

TEST(CompressedOffsets, WidthBoundaryAndRoundTrips) {
  const int64_t limit = std::numeric_limits<uint32_t>::max();
  for (auto expected : {std::vector<int64_t>{0}, std::vector<int64_t>{0, limit},
                        std::vector<int64_t>{0, limit + 1}}) {
    CompressedOffsets offsets(expected);
    const bool fits32 = expected.back() <= limit;
    EXPECT_EQ(offsets.mode(), fits32 ? CompressedOffsets::Mode::kFlat32
                                     : CompressedOffsets::Mode::kFlat64);
    EXPECT_EQ(offsets.NumBytes(),
              expected.size() * (fits32 ? sizeof(uint32_t) : sizeof(int64_t)));
    EXPECT_EQ(offsets.ToVector(), expected);
  }
}

TEST(CompressedOffsets, RelativeBlockBoundariesAndFallback) {
  std::vector<int64_t> values(258);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = (int64_t{1} << 34) + i * 500;
  }
  CompressedOffsets relative(values);
  ASSERT_EQ(relative.mode(), CompressedOffsets::Mode::kBlock16);
  const size_t expected_blocks = (values.size() - 1) / 28 + 1;
  EXPECT_EQ(relative.NumBytes(), expected_blocks * 64);
  EXPECT_EQ(relative.ToVector(), values);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(relative[i], values[i]);
    if (i + 1 < values.size()) {
      EXPECT_EQ(relative.GetRange(i), std::make_pair(values[i], values[i + 1]));
    }
  }

  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = i * 3000;
  }
  CompressedOffsets fallback(values);
  ASSERT_EQ(fallback.mode(), CompressedOffsets::Mode::kFlat32);
  EXPECT_EQ(fallback.ToVector(), values);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(fallback[i], values[i]);
    if (i + 1 < values.size()) {
      EXPECT_EQ(fallback.GetRange(i), std::make_pair(values[i], values[i + 1]));
    }
  }
}

TEST(CompressedOffsets, GetRangeBlockBoundaryAndTerminalOffsets) {
  // Size 29: block 0 has 28 entries (0..27), block 1 has 1 entry (28).
  // Index 27 spans across the block boundary to terminal offset 28.
  std::vector<int64_t> boundary_terminal(29);
  const int64_t base = int64_t{1} << 34;
  for (size_t i = 0; i < boundary_terminal.size(); ++i) {
    boundary_terminal[i] = base + i * 10;
  }
  CompressedOffsets offsets_29(boundary_terminal);
  ASSERT_EQ(offsets_29.mode(), CompressedOffsets::Mode::kBlock16);
  EXPECT_EQ(offsets_29.GetRange(26),
            std::make_pair(boundary_terminal[26], boundary_terminal[27]));
  EXPECT_EQ(offsets_29.GetRange(27),
            std::make_pair(boundary_terminal[27], boundary_terminal[28]));

  // Size 30: partial final block with 2 entries (28, 29).
  // Index 28 accesses within the partial final block to terminal offset 29.
  std::vector<int64_t> partial_final(30);
  for (size_t i = 0; i < partial_final.size(); ++i) {
    partial_final[i] = base + i * 10;
  }
  CompressedOffsets offsets_30(partial_final);
  ASSERT_EQ(offsets_30.mode(), CompressedOffsets::Mode::kBlock16);
  EXPECT_EQ(offsets_30.GetRange(27),
            std::make_pair(partial_final[27], partial_final[28]));
  EXPECT_EQ(offsets_30.GetRange(28),
            std::make_pair(partial_final[28], partial_final[29]));
}

TEST(CompressedOffsets, GetRangeAllWidthsAndRepeatedOffsets) {
  // Build a sequence spanning multiple blocks with repeated offsets (empty
  // neighborhoods) both within blocks (multiples of 7) and across block
  // boundaries (multiples of kBlockSize = 28, e.g. index 27 -> 28).
  std::vector<int64_t> base_values(260, 0);
  int64_t cur = 0;
  for (size_t i = 1; i < base_values.size(); ++i) {
    // Repeat offsets at every multiple of 7, which includes every block
    // boundary since kBlockSize = 28 = 4 * 7.
    if (i % 7 != 0) {
      cur += 25;
    }
    base_values[i] = cur;
  }

  // 1. Mode::kBlock16
  CompressedOffsets block16(base_values);
  ASSERT_EQ(block16.mode(), CompressedOffsets::Mode::kBlock16);
  for (size_t i = 0; i + 1 < base_values.size(); ++i) {
    EXPECT_EQ(block16.GetRange(i),
              std::make_pair(base_values[i], base_values[i + 1]));
  }

  // 2. Mode::kFlat32 (uint32 fallback when steps exceed 65535)
  std::vector<int64_t> wide_step_values = base_values;
  for (size_t i = 0; i < wide_step_values.size(); ++i) {
    wide_step_values[i] *= 4000;
  }
  CompressedOffsets uint32_offsets(wide_step_values);
  ASSERT_EQ(uint32_offsets.mode(), CompressedOffsets::Mode::kFlat32);
  for (size_t i = 0; i + 1 < wide_step_values.size(); ++i) {
    EXPECT_EQ(uint32_offsets.GetRange(i),
              std::make_pair(wide_step_values[i], wide_step_values[i + 1]));
  }

  // 3. Mode::kFlat64 (int64 fallback when values exceed uint32_t::max() and
  // steps exceed 65535)
  std::vector<int64_t> int64_values = wide_step_values;
  const int64_t high_base = int64_t{1} << 35;
  for (size_t i = 0; i < int64_values.size(); ++i) {
    int64_values[i] += high_base;
  }
  CompressedOffsets int64_offsets(int64_values);
  ASSERT_EQ(int64_offsets.mode(), CompressedOffsets::Mode::kFlat64);
  for (size_t i = 0; i + 1 < int64_values.size(); ++i) {
    EXPECT_EQ(int64_offsets.GetRange(i),
              std::make_pair(int64_values[i], int64_values[i + 1]));
  }
}

TEST(CompressedOffsets, ExactRelativeLimit) {
  // Size 113 spans 5 blocks (4 * 28 + 1) where block16 bytes (320) is strictly
  // less than flat uint32 bytes (452).
  std::vector<int64_t> values(113, 0);
  for (size_t i = 28; i < values.size(); ++i) {
    values[i] = 65535;
  }
  EXPECT_EQ(CompressedOffsets(values).mode(),
            CompressedOffsets::Mode::kBlock16);
  for (size_t i = 28; i < values.size(); ++i) {
    values[i] = 65536;
  }
  EXPECT_EQ(CompressedOffsets(values).mode(), CompressedOffsets::Mode::kFlat32);
  EXPECT_EQ(CompressedOffsets(std::vector<int64_t>{0}).mode(),
            CompressedOffsets::Mode::kFlat32);
}

TEST(CompressedOffsets, PrefetchRangeLastValidIndex) {
  // 1. Mode::kBlock16
  std::vector<int64_t> block16_values(260, 0);
  int64_t cur = 0;
  for (size_t i = 1; i < block16_values.size(); ++i) {
    if (i % 7 != 0) {
      cur += 25;
    }
    block16_values[i] = cur;
  }
  CompressedOffsets block16(block16_values);
  ASSERT_EQ(block16.mode(), CompressedOffsets::Mode::kBlock16);
  ASSERT_GE(block16.size(), 2);
  block16.PrefetchRange(0);
  block16.PrefetchRange(block16.size() - 2);

  // 2. Mode::kFlat32 (uint32 fallback when steps exceed 65535)
  std::vector<int64_t> flat32_values = block16_values;
  for (size_t i = 0; i < flat32_values.size(); ++i) {
    flat32_values[i] *= 4000;
  }
  CompressedOffsets flat32(flat32_values);
  ASSERT_EQ(flat32.mode(), CompressedOffsets::Mode::kFlat32);
  ASSERT_GE(flat32.size(), 2);
  flat32.PrefetchRange(0);
  flat32.PrefetchRange(flat32.size() - 2);

  // 3. Mode::kFlat64 (int64 fallback when values exceed uint32_t::max() and
  // steps exceed 65535)
  std::vector<int64_t> flat64_values = flat32_values;
  const int64_t high_base = int64_t{1} << 35;
  for (size_t i = 0; i < flat64_values.size(); ++i) {
    flat64_values[i] += high_base;
  }
  CompressedOffsets flat64(flat64_values);
  ASSERT_EQ(flat64.mode(), CompressedOffsets::Mode::kFlat64);
  ASSERT_GE(flat64.size(), 2);
  flat64.PrefetchRange(0);
  flat64.PrefetchRange(flat64.size() - 2);
}

}  // namespace
}  // namespace ginger
