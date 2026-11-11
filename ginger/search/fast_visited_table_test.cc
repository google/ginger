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

#include "ginger/search/fast_visited_table.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "ginger/internal/gunit_support.h"

namespace ginger::internal {
namespace {

TEST(FastVisitedTableTest, InsertAndDuplicate) {
  FastVisitedTable table(64);
  EXPECT_TRUE(table.Insert(42));
  EXPECT_FALSE(table.Insert(42));
}

TEST(FastVisitedTableTest, DistinctIds) {
  FastVisitedTable table(64);
  EXPECT_TRUE(table.Insert(1));
  EXPECT_TRUE(table.Insert(2));
  EXPECT_TRUE(table.Insert(3));
  EXPECT_FALSE(table.Insert(2));
}

TEST(FastVisitedTableTest, ResetClearsEntries) {
  FastVisitedTable table(64);
  table.Insert(10);
  table.Reset();
  EXPECT_TRUE(table.Insert(10));
}

TEST(FastVisitedTableTest, ResizePreservesEntries) {
  // Use a tiny table to force a resize.
  FastVisitedTable table(4);
  EXPECT_TRUE(table.Insert(1));
  EXPECT_TRUE(table.Insert(2));
  EXPECT_TRUE(table.Insert(3));  // Triggers resize (>50% full).
  EXPECT_FALSE(table.Insert(1));
  EXPECT_FALSE(table.Insert(2));
  EXPECT_FALSE(table.Insert(3));
  EXPECT_TRUE(table.Insert(4));
}

TEST(FastVisitedTableTest, ResetGrowsTableToMeetMinCapacity) {
  FastVisitedTable table(64);
  size_t old_size = table.capacity();
  table.Reset(200);
  EXPECT_GT(table.capacity(), old_size);
  EXPECT_GE(table.capacity(), 200);
  EXPECT_EQ(table.capacity(), 256);
}

TEST(FastVisitedTableTest, GenerationsIsolateEntries) {
  FastVisitedTable table(64);
  table.Insert(5);
  table.Insert(6);
  table.Reset();
  // New generation: old entries are invisible.
  EXPECT_TRUE(table.Insert(5));
  EXPECT_TRUE(table.Insert(6));
  EXPECT_FALSE(table.Insert(5));
  table.Reset();
  // Third generation: entries from both prior generations are invisible.
  EXPECT_TRUE(table.Insert(5));
}

TEST(FastVisitedTableTest, GenerationOverflowResetsSlots) {
  // Use capacity >= default capacity so Reset(0) doesn't call Init() due to
  // size increase.
  FastVisitedTable table(FastVisitedTable::kDefaultCapacity);
  table.Insert(7);
  table.Insert(42);
  table.SetCurrentGenerationForTesting(std::numeric_limits<uint32_t>::max());
  table.Reset();
  EXPECT_EQ(table.current_generation(), 1);
  for (const auto& slot : table.table()) {
    EXPECT_EQ(slot.id, -1);
    EXPECT_EQ(slot.generation, 0);
  }
  EXPECT_TRUE(table.Insert(7));
  EXPECT_TRUE(table.Insert(42));
}

}  // namespace
}  // namespace ginger::internal
