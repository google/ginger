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

#include "ginger/turboquant/rotation/walsh_hadamard.h"

#include <cstddef>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/random/random.h"
#include "absl/types/span.h"

namespace ginger {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

void ReferenceHadamardTransform(float* data, size_t dim) {
  for (size_t len = 1; len < dim; len <<= 1) {
    for (size_t i = 0; i < dim; i += 2 * len) {
      for (size_t j = 0; j < len; ++j) {
        float u = data[i + j];
        float v = data[i + j + len];
        data[i + j] = u + v;
        data[i + j + len] = u - v;
      }
    }
  }
}

TEST(WalshHadamardTransformTest, InvalidInput) {
  EXPECT_FALSE(WalshHadamardTransform(absl::Span<float>(nullptr, 128)).ok());
  std::vector<float> data(3, 0.0f);
  EXPECT_FALSE(WalshHadamardTransform(absl::MakeSpan(data)).ok());
  EXPECT_FALSE(WalshHadamardTransform(absl::MakeSpan(data).first(0)).ok());
}

TEST(WalshHadamardTransformTest, Dim1IsNoOp) {
  std::vector<float> data = {42.0f};
  EXPECT_OK(WalshHadamardTransform(absl::MakeSpan(data)));
  EXPECT_EQ(data[0], 42.0f);
}

TEST(WalshHadamardTransformTest, MatchesReferenceFWHT) {
  absl::BitGen bitgen;
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (size_t power = 1; power <= 13; ++power) {
    const size_t dim = 1 << power;
    std::vector<float> actual_data(dim);
    for (size_t i = 0; i < dim; ++i) {
      actual_data[i] = dist(bitgen);
    }
    std::vector<float> expected_data = actual_data;

    ASSERT_OK(WalshHadamardTransform(absl::MakeSpan(actual_data)));
    ReferenceHadamardTransform(expected_data.data(), dim);

    EXPECT_THAT(actual_data, Pointwise(FloatNear(1e-4f), expected_data))
        << "Mismatch at dim = " << dim;
  }
}

}  // namespace
}  // namespace ginger
