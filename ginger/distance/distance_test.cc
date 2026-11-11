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

#include "ginger/distance/distance.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::FloatEq;
using ::testing::IsEmpty;

TEST(DotProductDistanceTest, ComputesDistances) {
  SpanDataset dataset(3, 4);
  dataset.Set<float>(0, {1.0, 0.0, 0.0, 0.0});
  dataset.Set<float>(1, {0.0, 1.0, 0.0, 0.0});
  dataset.Set<float>(2, {0.6, 0.8, 0.0, 0.0});

  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  DotProductDistance(0, absl::MakeSpan(neighbors), dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(0.0), 1),
                                     FieldsAre(FloatEq(-0.6), 2)));
}

TEST(DotProductDistanceTest, ComputesDistancesFromSpan) {
  SpanDataset dataset(3, 4);
  dataset.Set<float>(0, {1.0, 0.0, 0.0, 0.0});
  dataset.Set<float>(1, {0.0, 1.0, 0.0, 0.0});
  dataset.Set<float>(2, {0.6, 0.8, 0.0, 0.0});

  std::vector<float> query = {1.0, 0.0, 0.0, 0.0};
  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  DotProductDistance(absl::MakeConstSpan(query), absl::MakeSpan(neighbors),
                     dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(0.0), 1),
                                     FieldsAre(FloatEq(-0.6), 2)));
}

TEST(DotProductDistanceTest, DotProductGreaterThanOne) {
  SpanDataset dataset(3, 4);
  dataset.Set<float>(0, {2.0, 0.0, 0.0, 0.0});
  dataset.Set<float>(1, {1.0, 0.0, 0.0, 0.0});
  dataset.Set<float>(2, {2.0, 0.0, 0.0, 0.0});

  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  DotProductDistance(0, absl::MakeSpan(neighbors), dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(-2.0), 1),
                                     FieldsAre(FloatEq(-4.0), 2)));
}

TEST(DotProductDistanceTest, EmptyNeighbors) {
  SpanDataset dataset(1, 4);
  dataset.Set<float>(0, {1.0, 0.0, 0.0, 0.0});
  std::vector<DistanceAndId> neighbors = {};
  DotProductDistance(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_THAT(neighbors, IsEmpty());
}

TEST(SquaredL2DistanceTest, ComputesDistances) {
  SpanDataset dataset(3, 4);
  dataset.Set<float>(0, {1.0, 0.0, 0.0, 0.0});
  dataset.Set<float>(1, {0.0, 1.0, 0.0, 0.0});
  dataset.Set<float>(2, {2.0, 0.0, 0.0, 0.0});

  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  SquaredL2Distance(0, absl::MakeSpan(neighbors), dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(2.0), 1),
                                     FieldsAre(FloatEq(1.0), 2)));
}

TEST(SquaredL2DistanceTest, ComputesDistancesFromSpan) {
  SpanDataset dataset(3, 4);
  dataset.Set<float>(0, {1.0, 0.0, 0.0, 0.0});
  dataset.Set<float>(1, {0.0, 1.0, 0.0, 0.0});
  dataset.Set<float>(2, {2.0, 0.0, 0.0, 0.0});

  std::vector<float> query = {1.0, 0.0, 0.0, 0.0};
  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  SquaredL2Distance(absl::MakeConstSpan(query), absl::MakeSpan(neighbors),
                    dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(2.0), 1),
                                     FieldsAre(FloatEq(1.0), 2)));
}

TEST(DistanceTest, HighDimensionalAndRemainderVectors) {
  constexpr size_t kDim = 35;
  SpanDataset dataset(3, kDim);

  std::vector<float> v0(kDim, 0.0f);
  std::vector<float> v1(kDim, 0.0f);
  std::vector<float> v2(kDim, 0.0f);

  for (size_t i = 0; i < kDim; ++i) {
    v0[i] = 1.0f;
    v1[i] = (i % 2 == 0) ? 1.0f : 0.0f;  // 18 ones, 17 zeros
    v2[i] = 2.0f;
  }

  dataset.Set<float>(0, v0);
  dataset.Set<float>(1, v1);
  dataset.Set<float>(2, v2);

  std::vector<DistanceAndId> dp_neighbors = {{.id = 1}, {.id = 2}};
  DotProductDistance(0, absl::MakeSpan(dp_neighbors), dataset);
  // v0 . v1 = 18.0. Distance = -18.0
  // v0 . v2 = 70.0. Distance = -70.0
  EXPECT_THAT(dp_neighbors, ElementsAre(FieldsAre(FloatEq(-18.0), 1),
                                        FieldsAre(FloatEq(-70.0), 2)));

  std::vector<DistanceAndId> l2_neighbors = {{.id = 1}, {.id = 2}};
  SquaredL2Distance(0, absl::MakeSpan(l2_neighbors), dataset);
  // ||v0 - v1||^2 = 17 * (1 - 0)^2 = 17.0
  // ||v0 - v2||^2 = 35 * (1 - 2)^2 = 35.0
  EXPECT_THAT(l2_neighbors, ElementsAre(FieldsAre(FloatEq(17.0), 1),
                                        FieldsAre(FloatEq(35.0), 2)));
}

TEST(DistanceTest, VariedDimensionsCoverAllSimdUnrollingBranches) {
  // Dimensions chosen to exercise 4N, 2N, N, and scalar remainder branches
  // across N=4 (SSE/NEON), N=8 (AVX2), and N=16 (AVX-512).
  // Specifically:
  // - 13 = 2*4 + 1*4 + 1 (covers N=4 for 2N + N + 1)
  // - 25 = 2*8 + 1*8 + 1 (covers N=8 for 2N + N + 1)
  // - 55 = 2*16 + 1*16 + 7 (covers N=16 for 2N + N + 7)
  for (const int32_t dim : {1, 4, 7, 12, 13, 16, 24, 25, 35, 48, 55, 64, 100}) {
    SpanDataset dataset(2, dim);
    std::vector<float> v0(dim);
    std::vector<float> v1(dim);

    float expected_dot = 0.0f;
    float expected_l2 = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      v0[i] = static_cast<float>((i % 5) - 2) * 0.25f;
      v1[i] = static_cast<float>((i % 7) - 3) * 0.5f;
      expected_dot += v0[i] * v1[i];
      const float diff = v0[i] - v1[i];
      expected_l2 += diff * diff;
    }
    dataset.Set<float>(0, v0);
    dataset.Set<float>(1, v1);

    std::vector<DistanceAndId> dp_neighbors = {{.id = 1}};
    DotProductDistance(0, absl::MakeSpan(dp_neighbors), dataset);
    const float expected_dp_dist = -expected_dot;
    EXPECT_NEAR(dp_neighbors[0].distance, expected_dp_dist, 1e-4f)
        << "dim=" << dim;

    std::vector<DistanceAndId> l2_neighbors = {{.id = 1}};
    SquaredL2Distance(0, absl::MakeSpan(l2_neighbors), dataset);
    EXPECT_NEAR(l2_neighbors[0].distance, expected_l2, 1e-4f) << "dim=" << dim;
  }
}

TEST(SquaredL2DistanceTest, EmptyNeighbors) {
  SpanDataset dataset(1, 4);
  dataset.Set<float>(0, {1.0, 0.0, 0.0, 0.0});
  std::vector<DistanceAndId> neighbors = {};
  SquaredL2Distance(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_THAT(neighbors, IsEmpty());
}

TEST(DistanceDeathTest, QueryDimensionalityMismatch) {
  SpanDataset dataset(2, 4);
  dataset.Set<float>(0, {1.0, 0.0, 0.0, 0.0});
  dataset.Set<float>(1, {0.0, 1.0, 0.0, 0.0});
  const std::vector<float> query = {1.0, 0.0, 0.0};
  std::vector<DistanceAndId> neighbors = {{.id = 1}};
  EXPECT_DEBUG_DEATH(DotProductDistance(absl::MakeConstSpan(query),
                                        absl::MakeSpan(neighbors), dataset),
                     "");
  EXPECT_DEBUG_DEATH(SquaredL2Distance(absl::MakeConstSpan(query),
                                       absl::MakeSpan(neighbors), dataset),
                     "");
}

}  // namespace
}  // namespace ginger
