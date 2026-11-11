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

#include "ginger/vamana/fixed_size_priority_queue.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>

#include "ginger/internal/gunit_support.h"

namespace ginger {
namespace {

// Fixture parameterized by the element type.
template <typename ElementType>
class FixedSizePriorityQueueTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(FixedSizePriorityQueueTest);

TYPED_TEST_P(FixedSizePriorityQueueTest, BasicTest) {
  size_t n = 1024;
  FixedSizePriorityQueue</*PriorityType=*/int32_t, /*IndexType=*/TypeParam>
      queue(n);
  ASSERT_TRUE(queue.Empty());
  ASSERT_EQ(queue.Top(), 0);
}

TYPED_TEST_P(FixedSizePriorityQueueTest, RemoveNonTopElement) {
  size_t n = 4;
  FixedSizePriorityQueue</*PriorityType=*/int32_t, /*IndexType=*/TypeParam>
      queue(n);
  queue.InsertOrUpdate(1, 1);
  queue.InsertOrUpdate(2, 2);
  queue.InsertOrUpdate(3, 3);
  auto elt = queue.Top();
  ASSERT_EQ(elt, 3);
  ASSERT_EQ(queue.Priority(elt), 3);
  queue.Remove(2);
  queue.Remove(elt);
  elt = queue.Top();
  ASSERT_EQ(elt, 1);
  ASSERT_EQ(queue.Priority(elt), 1);
  queue.Remove(elt);
  ASSERT_TRUE(queue.Empty());
}

TYPED_TEST_P(FixedSizePriorityQueueTest, TestGetPriority) {
  size_t n = 4;
  FixedSizePriorityQueue</*PriorityType=*/int32_t, /*IndexType=*/TypeParam>
      queue(n);
  queue.InsertOrUpdate(1, 300);
  queue.InsertOrUpdate(2, 20);
  queue.InsertOrUpdate(3, 1024);
  ASSERT_EQ(queue.Priority(1), 300);
  ASSERT_EQ(queue.Priority(2), 20);
  ASSERT_EQ(queue.Priority(3), 1024);
}

TYPED_TEST_P(FixedSizePriorityQueueTest, TestLT) {
  size_t n = 4;
  FixedSizePriorityQueue</*PriorityType=*/int32_t, /*IndexType=*/TypeParam>
      queue(n);
  queue.InsertOrUpdate(1, 1);
  queue.InsertOrUpdate(2, 2);
  queue.InsertOrUpdate(3, 3);
  auto elt = queue.Top();
  ASSERT_EQ(elt, 3);
  ASSERT_EQ(queue.Priority(elt), 3);
  queue.Remove(elt);
  elt = queue.Top();
  ASSERT_EQ(elt, 2);
  ASSERT_EQ(queue.Priority(elt), 2);
  queue.Remove(elt);
  elt = queue.Top();
  ASSERT_EQ(elt, 1);
  ASSERT_EQ(queue.Priority(elt), 1);
  queue.Remove(elt);
  ASSERT_TRUE(queue.Empty());
}

TYPED_TEST_P(FixedSizePriorityQueueTest, TestInterleavedInsertRemove) {
  size_t n = 4;
  FixedSizePriorityQueue</*PriorityType=*/int32_t, /*IndexType=*/TypeParam>
      queue(n);
  queue.InsertOrUpdate(1, 300);
  queue.InsertOrUpdate(2, 20);
  queue.InsertOrUpdate(3, 1024);
  queue.Remove(3);
  ASSERT_EQ(queue.Top(), 1);
  ASSERT_EQ(queue.Priority(1), 300);
  ASSERT_EQ(queue.Priority(2), 20);
  // 3 is not in the queue; priority should be kInvalidPriority.
  ASSERT_EQ(queue.Priority(3), std::numeric_limits<int32_t>::max());
  queue.InsertOrUpdate(3, 500);
  ASSERT_EQ(queue.Top(), 3);
  ASSERT_EQ(queue.Priority(1), 300);
  ASSERT_EQ(queue.Priority(2), 20);
  ASSERT_EQ(queue.Priority(3), 500);

  auto elt = queue.Top();
  ASSERT_EQ(elt, 3);
  ASSERT_EQ(queue.Priority(elt), 500);
  queue.Remove(elt);
  elt = queue.Top();
  ASSERT_EQ(elt, 1);
  ASSERT_EQ(queue.Priority(elt), 300);
  queue.Remove(elt);
  elt = queue.Top();
  ASSERT_EQ(elt, 2);
  ASSERT_EQ(queue.Priority(elt), 20);
  queue.Remove(elt);
  ASSERT_TRUE(queue.Empty());
}

TEST(FixedSizePriorityQueueTest, Tiebreaking) {
  FixedSizePriorityQueue<int32_t> queue(4);
  queue.InsertOrUpdate(0, 1);
  queue.InsertOrUpdate(1, 1);
  queue.InsertOrUpdate(2, 1);
  queue.InsertOrUpdate(3, 1);
  EXPECT_EQ(queue.Top(), 0);
  queue.Remove(0);
  EXPECT_EQ(queue.Top(), 1);
  queue.Remove(1);
  EXPECT_EQ(queue.Top(), 2);
  queue.Remove(2);
  EXPECT_EQ(queue.Top(), 3);
  queue.Remove(3);
  EXPECT_TRUE(queue.Empty());
}

TEST(FixedSizePriorityQueueTest, TiebreakingAfterRemoval) {
  FixedSizePriorityQueue<int32_t> queue(4);
  queue.InsertOrUpdate(0, 5);
  queue.InsertOrUpdate(1, 7);
  queue.InsertOrUpdate(2, 5);
  queue.InsertOrUpdate(3, 7);
  EXPECT_EQ(queue.Top(), 1);
  queue.Remove(1);
  EXPECT_EQ(queue.Top(), 3);
  queue.InsertOrUpdate(3, 4);
  EXPECT_EQ(queue.Top(), 0);
}

TEST(FixedSizePriorityQueueTest, CustomCompare) {
  FixedSizePriorityQueue<int32_t, int32_t, std::less<int32_t>> queue(4);
  queue.InsertOrUpdate(0, 3);
  queue.InsertOrUpdate(1, 1);
  queue.InsertOrUpdate(3, 2);
  EXPECT_EQ(queue.Top(), 1);
  queue.Remove(1);
  EXPECT_EQ(queue.Top(), 3);
  queue.Remove(3);
  EXPECT_EQ(queue.Top(), 0);
  queue.Remove(0);
  EXPECT_TRUE(queue.Empty());
}

TEST(FixedSizePriorityQueueTest, CustomCompareTiebreaking) {
  FixedSizePriorityQueue<int32_t, int32_t, std::less<int32_t>> queue(4);
  queue.InsertOrUpdate(0, 5);
  queue.InsertOrUpdate(1, 5);
  queue.InsertOrUpdate(2, 2);
  queue.InsertOrUpdate(3, 5);
  EXPECT_EQ(queue.Top(), 2);
  queue.Remove(2);
  EXPECT_EQ(queue.Top(), 0);
  queue.Remove(0);
  EXPECT_EQ(queue.Top(), 1);
  queue.Remove(1);
  EXPECT_EQ(queue.Top(), 3);
  queue.Remove(3);
  EXPECT_TRUE(queue.Empty());
}

TEST(FixedSizePriorityQueueTest, CanUsePairAsPriority) {
  FixedSizePriorityQueue<std::pair<int32_t, int32_t>, int32_t,
                         std::less<std::pair<int32_t, int32_t>>>
      queue(4);
  queue.InsertOrUpdate(0, {5, 1});
  queue.InsertOrUpdate(1, {5, 2});
  queue.InsertOrUpdate(2, {2, 3});
  queue.InsertOrUpdate(3, {5, 4});
  EXPECT_EQ(queue.Top(), 2);
  queue.Remove(2);
  EXPECT_EQ(queue.Top(), 0);
}

REGISTER_TYPED_TEST_SUITE_P(FixedSizePriorityQueueTest, BasicTest,
                            RemoveNonTopElement, TestGetPriority, TestLT,
                            TestInterleavedInsertRemove);

using IndexTypes = ::testing::Types<int32_t, int64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(FixedSizePriorityQueueTest,
                               FixedSizePriorityQueueTest, IndexTypes);

}  // namespace
}  // namespace ginger
