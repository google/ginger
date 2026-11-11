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

#include "ginger/internal/portable_random.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"

namespace ginger::internal::portable_random {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;

// These sequences must never change: encodings regenerate their rotations and
// projections from a seed, so a change makes existing encodings unreadable.

TEST(PortableRandomTest, UniformIntIsStable) {
  std::mt19937 gen(42);
  std::vector<uint64_t> values;
  for (int i = 0; i < 8; ++i) values.push_back(UniformInt(gen, 1000));
  EXPECT_THAT(values, ElementsAre(102, 435, 860, 270, 106, 71, 700, 20));
}

TEST(PortableRandomTest, UniformInt64IsStable) {
  std::mt19937_64 gen(42);
  std::vector<uint64_t> values;
  for (int i = 0; i < 8; ++i) values.push_back(UniformInt(gen, 1000));
  EXPECT_THAT(values, ElementsAre(726, 680, 778, 846, 341, 892, 32, 256));
}

TEST(PortableRandomTest, BernoulliIsStable) {
  std::mt19937_64 gen(42);
  std::vector<bool> values;
  for (int i = 0; i < 8; ++i) values.push_back(Bernoulli(gen, 0.5));
  EXPECT_THAT(values,
              ElementsAre(false, false, false, true, false, true, false, true));
}

TEST(PortableRandomTest, NormalIsStable) {
  std::mt19937 gen(42);
  NormalDistribution normal;
  std::vector<float> values;
  for (int i = 0; i < 4; ++i) values.push_back(normal(gen));
  // Allow for last-bit differences between math libraries.
  constexpr float kTolerance = 1e-6f;
  EXPECT_THAT(values, ElementsAre(FloatNear(-0.516964138f, kTolerance),
                                  FloatNear(1.22192132f, kTolerance),
                                  FloatNear(0.721332669f, kTolerance),
                                  FloatNear(0.86963594f, kTolerance)));
}

TEST(PortableRandomTest, ShuffleIsStable) {
  std::mt19937 gen(42);
  std::vector<int> values(10);
  std::iota(values.begin(), values.end(), 0);
  Shuffle(absl::MakeSpan(values), gen);
  EXPECT_THAT(values, ElementsAre(6, 4, 0, 9, 2, 3, 1, 5, 7, 8));
}

TEST(PortableRandomTest, UniformIntStaysInRange) {
  std::mt19937 gen(1);
  for (uint64_t bound : {1u, 2u, 3u, 1000u, 0xFFFFFFFFu}) {
    for (int i = 0; i < 1000; ++i) EXPECT_LT(UniformInt(gen, bound), bound);
  }
}

}  // namespace
}  // namespace ginger::internal::portable_random
