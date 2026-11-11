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

// Unit tests for vec_group-inl.h: ForIndex, VEC_GROUP_DEFINE,
// VEC_GROUP_ACCESSOR.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/simd/vec_group_test.cc"
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

using ::testing::ElementsAre;
using ::testing::IsEmpty;

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/simd/vec_group-inl.h"

namespace ginger {
namespace HWY_NAMESPACE {
namespace {}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

// Verifies ForIndex<N> invokes the functor exactly N times
// 0, 1, ..., N-1 in order.
HWY_NOINLINE void TestForIndexOrder() {
  // N = 0: should not call the functor at all.
  {
    std::vector<size_t> collected;
    ginger::HWY_NAMESPACE::ForIndex<0>(
        [&](auto idx) { collected.push_back(idx.value); });
    EXPECT_THAT(collected, IsEmpty());
  }

  // N = 1
  {
    std::vector<size_t> collected;
    ginger::HWY_NAMESPACE::ForIndex<1>(
        [&](auto idx) { collected.push_back(idx.value); });
    EXPECT_THAT(collected, ElementsAre(0));
  }

  // N = 4
  {
    std::vector<size_t> collected;
    ginger::HWY_NAMESPACE::ForIndex<4>(
        [&](auto idx) { collected.push_back(idx.value); });
    EXPECT_THAT(collected, ElementsAre(0, 1, 2, 3));
  }

  // N = 8
  {
    std::vector<size_t> collected;
    ginger::HWY_NAMESPACE::ForIndex<8>(
        [&](auto idx) { collected.push_back(idx.value); });
    EXPECT_THAT(collected, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
  }
}

// Verifies that ForIndex index arguments are constexpr (usable in
// if-constexpr and static_assert contexts).
HWY_NOINLINE void TestForIndexConstexpr() {
  int sum = 0;
  ginger::HWY_NAMESPACE::ForIndex<4>([&](auto idx) {
    // idx.value must be usable in a constexpr context.
    if constexpr (decltype(idx)::value < 2) {
      sum += 1;
    } else {
      sum += 10;
    }
  });
  // Indices 0, 1 contribute 1 each; indices 2, 3 contribute 10 each.
  EXPECT_EQ(sum, 22);
}

// Verifies VEC_GROUP_DEFINE + VEC_GROUP_ACCESSOR: write via accessor, then
// read back the correct values.
HWY_NOINLINE void TestVecGroupAccessor() {
  const hn::ScalableTag<int32_t> i32;
  VEC_GROUP_DEFINE(i32, v);
  auto acc = VEC_GROUP_ACCESSOR(v);

  // Write distinct values via the accessor.
  ginger::HWY_NAMESPACE::ForIndex<8>([&](auto q) {
    acc(q) = hn::Set(i32, static_cast<int32_t>(q.value + 1));
  });

  // Read back and verify each element.
  ginger::HWY_NAMESPACE::ForIndex<8>([&](auto q) {
    const int32_t expected = static_cast<int32_t>(q.value + 1);
    const int32_t got = hn::GetLane(acc(q));
    EXPECT_EQ(got, expected) << "mismatch at index " << q.value;
  });
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(VecGroupTest);
HWY_EXPORT_AND_TEST_P(VecGroupTest, TestForIndexOrder);
HWY_EXPORT_AND_TEST_P(VecGroupTest, TestForIndexConstexpr);
HWY_EXPORT_AND_TEST_P(VecGroupTest, TestVecGroupAccessor);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif
