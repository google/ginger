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

#ifndef GINGER_INTERNAL_GUNIT_SUPPORT_H_
#define GINGER_INTERNAL_GUNIT_SUPPORT_H_

#include <utility>

#include "gtest/gtest.h"

#define GINGER_TEST_CONCAT_INNER(x, y) x##y
#define GINGER_TEST_CONCAT(x, y) GINGER_TEST_CONCAT_INNER(x, y)

#define GINGER_TEST_IMPL_EAT(...)
#define GINGER_TEST_IMPL_REM(...) __VA_ARGS__
#define GINGER_TEST_1_IF_EMPTY_ELSE_10(...) 1##__VA_OPT__(0)
#define GINGER_TEST_IMPL_UNPAREN_1(x) GINGER_TEST_IMPL_REM x
#define GINGER_TEST_IMPL_UNPAREN_10(x) GINGER_TEST_IMPL_REM(x)
#define GINGER_TEST_UNPAREN(...)                                         \
  GINGER_TEST_IMPL_REM(GINGER_TEST_CONCAT(                               \
      GINGER_TEST_IMPL_UNPAREN_,                                         \
      GINGER_TEST_1_IF_EMPTY_ELSE_10(GINGER_TEST_IMPL_EAT __VA_ARGS__))( \
      GINGER_TEST_IMPL_REM(__VA_ARGS__)))

#ifndef EXPECT_OK
#define EXPECT_OK(val) EXPECT_TRUE((val).ok())
#endif

#ifndef ASSERT_OK
#define ASSERT_OK(val) ASSERT_TRUE((val).ok())
#endif

#ifndef ASSERT_OK_AND_ASSIGN
#define ASSERT_OK_AND_ASSIGN(lhs, rexpr)                       \
  auto GINGER_TEST_CONCAT(_status_or_, __LINE__) = (rexpr);    \
  ASSERT_TRUE(GINGER_TEST_CONCAT(_status_or_, __LINE__).ok()); \
  GINGER_TEST_UNPAREN(lhs) =                                   \
      std::move(GINGER_TEST_CONCAT(_status_or_, __LINE__)).value()
#endif

#ifndef EXPECT_OK_AND_ASSIGN
#define EXPECT_OK_AND_ASSIGN(lhs, rexpr)                       \
  auto GINGER_TEST_CONCAT(_status_or_, __LINE__) = (rexpr);    \
  EXPECT_TRUE(GINGER_TEST_CONCAT(_status_or_, __LINE__).ok()); \
  GINGER_TEST_UNPAREN(lhs) =                                   \
      std::move(GINGER_TEST_CONCAT(_status_or_, __LINE__)).value()
#endif

#endif  // GINGER_INTERNAL_GUNIT_SUPPORT_H_
