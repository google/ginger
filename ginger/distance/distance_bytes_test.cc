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

#include "ginger/distance/distance_bytes.h"

#include <cstddef>
#include <cstdint>
#include <limits>
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

// =============================================================================
// int8_t Dataset Tests
// =============================================================================

TEST(DistanceBytesTest, Int8DotProductDistance) {
  SpanDataset dataset(3, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {10, 0, 0, 0});
  dataset.Set<int8_t>(1, {0, 10, 0, 0});
  dataset.Set<int8_t>(2, {6, 8, 0, 0});

  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  DotProductDistanceInt8(0, absl::MakeSpan(neighbors), dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(0.0f), 1),
                                     FieldsAre(FloatEq(-60.0f), 2)));
}

TEST(DistanceBytesTest, Int8DotProductDistanceByteQuery) {
  SpanDataset dataset(3, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {10, 0, 0, 0});
  dataset.Set<int8_t>(1, {0, 10, 0, 0});
  dataset.Set<int8_t>(2, {6, 8, 0, 0});

  std::vector<int8_t> query = {2, 3, 0, 0};
  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  DotProductDistance(absl::MakeConstSpan(query), absl::MakeSpan(neighbors),
                     dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(-30.0f), 1),
                                     FieldsAre(FloatEq(-36.0f), 2)));
}

TEST(DistanceBytesTest, Int8SquaredL2Distance) {
  SpanDataset dataset(3, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {10, 0, 0, 0});
  dataset.Set<int8_t>(1, {0, 10, 0, 0});
  dataset.Set<int8_t>(2, {6, 8, 0, 0});

  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  SquaredL2DistanceInt8(0, absl::MakeSpan(neighbors), dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(200.0f), 1),
                                     FieldsAre(FloatEq(80.0f), 2)));
}

TEST(DistanceBytesTest, Int8SquaredL2DistanceByteQuery) {
  SpanDataset dataset(3, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {10, 0, 0, 0});
  dataset.Set<int8_t>(1, {0, 10, 0, 0});
  dataset.Set<int8_t>(2, {6, 8, 0, 0});

  std::vector<int8_t> query = {6, 0, 0, 0};
  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  SquaredL2Distance(absl::MakeConstSpan(query), absl::MakeSpan(neighbors),
                    dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(136.0f), 1),
                                     FieldsAre(FloatEq(64.0f), 2)));
}

TEST(DistanceBytesTest, DistanceToSelfInt8) {
  SpanDataset dataset(1, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {3, 4, -5, 6});

  std::vector<DistanceAndId> neighbors = {{.id = 0}};

  SquaredL2DistanceInt8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_FLOAT_EQ(neighbors[0].distance, 0.0f);

  // Norm squared: 3^2 + 4^2 + (-5)^2 + 6^2 = 9 + 16 + 25 + 36 = 86
  DotProductDistanceInt8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_FLOAT_EQ(neighbors[0].distance, -86.0f);
}

TEST(DistanceBytesTest, ExtremeValuesInt8) {
  constexpr size_t kDim = 4;
  SpanDataset dataset(2, kDim, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {-128, 127, -128, 127});
  dataset.Set<int8_t>(1, {127, -128, -128, 127});

  std::vector<DistanceAndId> neighbors = {{.id = 1}};

  // Dot product: (-128*127) + (127*-128) + (-128*-128) + (127*127)
  // = -16256 - 16256 + 16384 + 16129 = 1.0 (distance is -1.0)
  DotProductDistanceInt8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_FLOAT_EQ(neighbors[0].distance, -1.0f);

  // L2: (-128 - 127)^2 + (127 - -128)^2 + (-128 - -128)^2 + (127 - 127)^2
  // = (-255)^2 + 255^2 + 0 + 0 = 65025 + 65025 = 130050.0
  SquaredL2DistanceInt8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_FLOAT_EQ(neighbors[0].distance, 130050.0f);
}

TEST(DistanceBytesTest, Int8EmptyNeighbors) {
  SpanDataset dataset(1, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {1, 0, 0, 0});

  std::vector<DistanceAndId> neighbors = {};
  DotProductDistanceInt8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_THAT(neighbors, IsEmpty());

  SquaredL2DistanceInt8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_THAT(neighbors, IsEmpty());
}

TEST(DistanceBytesDeathTest, Int8QueryDimensionalityMismatch) {
  SpanDataset dataset(2, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {1, 0, 0, 0});
  dataset.Set<int8_t>(1, {0, 1, 0, 0});
  const std::vector<int8_t> query = {1, 0, 0};
  std::vector<DistanceAndId> neighbors = {{.id = 1}};
  EXPECT_DEBUG_DEATH(DotProductDistance(absl::MakeConstSpan(query),
                                        absl::MakeSpan(neighbors), dataset),
                     "");
  EXPECT_DEBUG_DEATH(SquaredL2Distance(absl::MakeConstSpan(query),
                                       absl::MakeSpan(neighbors), dataset),
                     "");
}

// =============================================================================
// uint8_t Dataset Tests
// =============================================================================

TEST(DistanceBytesTest, Uint8DotProductDistance) {
  SpanDataset dataset(3, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {10, 0, 0, 0});
  dataset.Set<uint8_t>(1, {0, 10, 0, 0});
  dataset.Set<uint8_t>(2, {6, 8, 0, 0});

  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  DotProductDistanceUint8(0, absl::MakeSpan(neighbors), dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(0.0f), 1),
                                     FieldsAre(FloatEq(-60.0f), 2)));
}

TEST(DistanceBytesTest, Uint8DotProductDistanceByteQuery) {
  SpanDataset dataset(3, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {10, 0, 0, 0});
  dataset.Set<uint8_t>(1, {0, 10, 0, 0});
  dataset.Set<uint8_t>(2, {6, 8, 0, 0});

  std::vector<uint8_t> query = {2, 3, 0, 0};
  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  DotProductDistance(absl::MakeConstSpan(query), absl::MakeSpan(neighbors),
                     dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(-30.0f), 1),
                                     FieldsAre(FloatEq(-36.0f), 2)));
}

TEST(DistanceBytesTest, Uint8SquaredL2Distance) {
  SpanDataset dataset(3, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {10, 0, 0, 0});
  dataset.Set<uint8_t>(1, {0, 10, 0, 0});
  dataset.Set<uint8_t>(2, {6, 8, 0, 0});

  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  SquaredL2DistanceUint8(0, absl::MakeSpan(neighbors), dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(200.0f), 1),
                                     FieldsAre(FloatEq(80.0f), 2)));
}

TEST(DistanceBytesTest, Uint8SquaredL2DistanceByteQuery) {
  SpanDataset dataset(3, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {10, 0, 0, 0});
  dataset.Set<uint8_t>(1, {0, 10, 0, 0});
  dataset.Set<uint8_t>(2, {6, 8, 0, 0});

  std::vector<uint8_t> query = {6, 0, 0, 0};
  std::vector<DistanceAndId> neighbors = {{.id = 1}, {.id = 2}};
  SquaredL2Distance(absl::MakeConstSpan(query), absl::MakeSpan(neighbors),
                    dataset);

  EXPECT_THAT(neighbors, ElementsAre(FieldsAre(FloatEq(136.0f), 1),
                                     FieldsAre(FloatEq(64.0f), 2)));
}

TEST(DistanceBytesTest, DistanceToSelfUint8) {
  SpanDataset dataset(1, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {3, 4, 5, 6});

  std::vector<DistanceAndId> neighbors = {{.id = 0}};

  SquaredL2DistanceUint8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_FLOAT_EQ(neighbors[0].distance, 0.0f);

  // Norm squared: 3^2 + 4^2 + 5^2 + 6^2 = 9 + 16 + 25 + 36 = 86
  DotProductDistanceUint8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_FLOAT_EQ(neighbors[0].distance, -86.0f);
}

TEST(DistanceBytesTest, ExtremeValuesUint8) {
  constexpr size_t kDim = 4;
  SpanDataset dataset(2, kDim, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {0, 255, 0, 255});
  dataset.Set<uint8_t>(1, {255, 0, 0, 255});

  std::vector<DistanceAndId> neighbors = {{.id = 1}};

  // Dot product: 0*255 + 255*0 + 0*0 + 255*255 = 65025.0
  DotProductDistanceUint8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_FLOAT_EQ(neighbors[0].distance, -65025.0f);

  // L2: (0-255)^2 + (255-0)^2 + 0 + 0 = 65025 + 65025 = 130050.0
  SquaredL2DistanceUint8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_FLOAT_EQ(neighbors[0].distance, 130050.0f);
}

TEST(DistanceBytesTest, Uint8EmptyNeighbors) {
  SpanDataset dataset(1, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {1, 0, 0, 0});

  std::vector<DistanceAndId> neighbors = {};
  DotProductDistanceUint8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_THAT(neighbors, IsEmpty());

  SquaredL2DistanceUint8(0, absl::MakeSpan(neighbors), dataset);
  EXPECT_THAT(neighbors, IsEmpty());
}

// =============================================================================
// Batching and Varied Dimensions Tests
// =============================================================================

// Tests neighbor lists exceeding 4 elements (e.g. 9 and 17 neighbors) on an
// int8_t dataset to exercise both the 4-neighbor SIMD batch loop, candidate
// vector prefetching, and the remaining single-point tail loop.
TEST(DistanceBytesTest, MultiNeighborBatchingExceedsFourNeighborsInt8) {
  constexpr size_t kNumPoints = 18;
  constexpr size_t kDim = 4;

  SpanDataset dataset(kNumPoints, kDim, CoordinateType::kInt8);
  for (size_t i = 0; i < kNumPoints; ++i) {
    dataset.Set<int8_t>(i, {static_cast<int8_t>(i), 0, 0, 0});
  }

  for (size_t count : {9, 17}) {
    std::vector<DistanceAndId> dp_neighbors;
    std::vector<DistanceAndId> l2_neighbors;
    dp_neighbors.reserve(count);
    l2_neighbors.reserve(count);
    for (size_t i = 1; i <= count; ++i) {
      dp_neighbors.push_back({.id = static_cast<GingerNodeId>(i)});
      l2_neighbors.push_back({.id = static_cast<GingerNodeId>(i)});
    }

    std::vector<int8_t> query = {2, 0, 0, 0};
    DotProductDistance(absl::MakeConstSpan(query), absl::MakeSpan(dp_neighbors),
                       dataset);
    SquaredL2Distance(absl::MakeConstSpan(query), absl::MakeSpan(l2_neighbors),
                      dataset);

    for (size_t i = 0; i < count; ++i) {
      const float expected_dp = -2.0f * static_cast<float>(i + 1);
      EXPECT_FLOAT_EQ(dp_neighbors[i].distance, expected_dp)
          << "count=" << count << ", neighbor=" << i;

      const float diff = 2.0f - static_cast<float>(i + 1);
      const float expected_l2 = diff * diff;
      EXPECT_FLOAT_EQ(l2_neighbors[i].distance, expected_l2)
          << "count=" << count << ", neighbor=" << i;
    }
  }
}

// Tests neighbor lists exceeding 4 elements (e.g. 9 and 17 neighbors) on a
// uint8_t dataset to exercise both the 4-neighbor SIMD batch loop, candidate
// vector prefetching, and the remaining single-point tail loop.
TEST(DistanceBytesTest, MultiNeighborBatchingExceedsFourNeighborsUint8) {
  constexpr size_t kNumPoints = 18;
  constexpr size_t kDim = 4;

  SpanDataset dataset(kNumPoints, kDim, CoordinateType::kUint8);
  for (size_t i = 0; i < kNumPoints; ++i) {
    dataset.Set<uint8_t>(i, {static_cast<uint8_t>(i), 0, 0, 0});
  }

  for (size_t count : {9, 17}) {
    std::vector<DistanceAndId> dp_neighbors;
    std::vector<DistanceAndId> l2_neighbors;
    dp_neighbors.reserve(count);
    l2_neighbors.reserve(count);
    for (size_t i = 1; i <= count; ++i) {
      dp_neighbors.push_back({.id = static_cast<GingerNodeId>(i)});
      l2_neighbors.push_back({.id = static_cast<GingerNodeId>(i)});
    }

    std::vector<uint8_t> query = {2, 0, 0, 0};
    DotProductDistance(absl::MakeConstSpan(query), absl::MakeSpan(dp_neighbors),
                       dataset);
    SquaredL2Distance(absl::MakeConstSpan(query), absl::MakeSpan(l2_neighbors),
                      dataset);

    for (size_t i = 0; i < count; ++i) {
      const float expected_dp = -2.0f * static_cast<float>(i + 1);
      EXPECT_FLOAT_EQ(dp_neighbors[i].distance, expected_dp)
          << "count=" << count << ", neighbor=" << i;

      const float diff = 2.0f - static_cast<float>(i + 1);
      const float expected_l2 = diff * diff;
      EXPECT_FLOAT_EQ(l2_neighbors[i].distance, expected_l2)
          << "count=" << count << ", neighbor=" << i;
    }
  }
}

// Tests a wide range of vector dimensions on int8_t datasets to exercise all
// SIMD unrolling branches (4N, 2N, N, and scalar remainder) for SSE/NEON (N=4),
// AVX2 (N=8), and AVX-512 (N=16), validating int8_t queries.
// Uses 5 neighbors per dimension to exercise both the 4-point batch loop and
// remaining single-point tail loop across every dimension.
TEST(DistanceBytesTest, VariedDimensionsCoverAllSimdUnrollingBranchesInt8) {
  for (const int32_t dim : {1, 4, 7, 12, 13, 16, 24, 25, 35, 48, 55, 64, 100}) {
    constexpr size_t kNumNeighbors = 5;
    SpanDataset dataset(kNumNeighbors + 1, dim, CoordinateType::kInt8);

    std::vector<int8_t> int8_query(dim);
    for (size_t i = 0; i < dim; ++i) {
      int8_query[i] = static_cast<int8_t>((i % 5) - 2);
    }
    dataset.Set<int8_t>(0, int8_query);

    std::vector<float> expected_dots(kNumNeighbors, 0.0f);
    std::vector<float> expected_l2s(kNumNeighbors, 0.0f);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      std::vector<int8_t> dataset_point(dim);
      for (size_t i = 0; i < dim; ++i) {
        dataset_point[i] = static_cast<int8_t>(((i + n) % 7) - 3);
        expected_dots[n] += static_cast<float>(int8_query[i]) *
                            static_cast<float>(dataset_point[i]);
        const float diff = static_cast<float>(int8_query[i]) -
                           static_cast<float>(dataset_point[i]);
        expected_l2s[n] += diff * diff;
      }
      dataset.Set<int8_t>(n + 1, dataset_point);
    }

    // Int8 query - Dot Product
    std::vector<DistanceAndId> dp_neighbors(kNumNeighbors);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      dp_neighbors[n] = {.id = static_cast<GingerNodeId>(n + 1)};
    }
    DotProductDistance(absl::MakeConstSpan(int8_query),
                       absl::MakeSpan(dp_neighbors), dataset);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      EXPECT_NEAR(dp_neighbors[n].distance, -expected_dots[n], 1e-3f)
          << "dim=" << dim << ", neighbor=" << n;
    }

    // Int8 query - Squared L2
    std::vector<DistanceAndId> l2_neighbors(kNumNeighbors);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      l2_neighbors[n] = {.id = static_cast<GingerNodeId>(n + 1)};
    }
    SquaredL2Distance(absl::MakeConstSpan(int8_query),
                      absl::MakeSpan(l2_neighbors), dataset);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      EXPECT_NEAR(l2_neighbors[n].distance, expected_l2s[n], 1e-3f)
          << "dim=" << dim << ", neighbor=" << n;
    }
  }
}

// Tests a wide range of vector dimensions on uint8_t datasets to exercise all
// SIMD unrolling branches (4N, 2N, N, and scalar remainder) for SSE/NEON (N=4),
// AVX2 (N=8), and AVX-512 (N=16), validating uint8_t queries.
// Uses 5 neighbors per dimension to exercise both the 4-point batch loop and
// remaining single-point tail loop across every dimension.
TEST(DistanceBytesTest, VariedDimensionsCoverAllSimdUnrollingBranchesUint8) {
  for (const int32_t dim : {1, 4, 7, 12, 13, 16, 24, 25, 35, 48, 55, 64, 100}) {
    constexpr size_t kNumNeighbors = 5;
    SpanDataset dataset(kNumNeighbors + 1, dim, CoordinateType::kUint8);

    std::vector<uint8_t> uint8_query(dim);
    for (size_t i = 0; i < dim; ++i) {
      uint8_query[i] = static_cast<uint8_t>((i * 13) % 255);
    }
    dataset.Set<uint8_t>(0, uint8_query);

    std::vector<float> expected_dots(kNumNeighbors, 0.0f);
    std::vector<float> expected_l2s(kNumNeighbors, 0.0f);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      std::vector<uint8_t> dataset_point(dim);
      for (size_t i = 0; i < dim; ++i) {
        dataset_point[i] = static_cast<uint8_t>((i * 17 + 5 + n) % 255);
        expected_dots[n] += static_cast<float>(uint8_query[i]) *
                            static_cast<float>(dataset_point[i]);
        const float diff = static_cast<float>(uint8_query[i]) -
                           static_cast<float>(dataset_point[i]);
        expected_l2s[n] += diff * diff;
      }
      dataset.Set<uint8_t>(n + 1, dataset_point);
    }

    // Uint8 query - Dot Product
    std::vector<DistanceAndId> dp_neighbors(kNumNeighbors);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      dp_neighbors[n] = {.id = static_cast<GingerNodeId>(n + 1)};
    }
    DotProductDistance(absl::MakeConstSpan(uint8_query),
                       absl::MakeSpan(dp_neighbors), dataset);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      EXPECT_NEAR(dp_neighbors[n].distance, -expected_dots[n], 1e-3f)
          << "dim=" << dim << ", neighbor=" << n;
    }

    // Uint8 query - Squared L2
    std::vector<DistanceAndId> l2_neighbors(kNumNeighbors);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      l2_neighbors[n] = {.id = static_cast<GingerNodeId>(n + 1)};
    }
    SquaredL2Distance(absl::MakeConstSpan(uint8_query),
                      absl::MakeSpan(l2_neighbors), dataset);
    for (size_t n = 0; n < kNumNeighbors; ++n) {
      EXPECT_NEAR(l2_neighbors[n].distance, expected_l2s[n], 1e-3f)
          << "dim=" << dim << ", neighbor=" << n;
    }
  }
}

// Tests neighbor list sizes that are exact multiples of 4 (e.g. 4, 8, 12, and
// 16 neighbors) with zero remainder tail to verify clean batch loop boundaries
// and prefetching across batch boundaries.
TEST(DistanceBytesTest, ExactMultiplesOfFourNeighborsBatchingInt8) {
  constexpr size_t kNumNeighbors = 16;
  constexpr size_t kDim = 4;

  SpanDataset dataset(kNumNeighbors + 1, kDim, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {1, 2, 3, 4});
  for (size_t i = 1; i <= kNumNeighbors; ++i) {
    dataset.Set<int8_t>(i, {static_cast<int8_t>(i), 0, 0, 0});
  }

  for (size_t count : {4, 8, 12, 16}) {
    std::vector<DistanceAndId> dp_neighbors(count);
    std::vector<DistanceAndId> l2_neighbors(count);
    for (size_t i = 0; i < count; ++i) {
      dp_neighbors[i] = {.id = static_cast<GingerNodeId>(i + 1)};
      l2_neighbors[i] = {.id = static_cast<GingerNodeId>(i + 1)};
    }
    DotProductDistanceInt8(0, absl::MakeSpan(dp_neighbors), dataset);
    SquaredL2DistanceInt8(0, absl::MakeSpan(l2_neighbors), dataset);
    for (size_t i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ(dp_neighbors[i].distance,
                      -1.0f * static_cast<float>(i + 1));
      const float diff = 1.0f - static_cast<float>(i + 1);
      const float expected_l2 = diff * diff + 4.0f + 9.0f + 16.0f;
      EXPECT_FLOAT_EQ(l2_neighbors[i].distance, expected_l2);
    }
  }
}

TEST(DistanceBytesTest, ExactMultiplesOfFourNeighborsBatchingUint8) {
  constexpr size_t kNumNeighbors = 16;
  constexpr size_t kDim = 4;

  SpanDataset dataset(kNumNeighbors + 1, kDim, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {1, 2, 3, 4});
  for (size_t i = 1; i <= kNumNeighbors; ++i) {
    dataset.Set<uint8_t>(i, {static_cast<uint8_t>(i), 0, 0, 0});
  }

  for (size_t count : {4, 8, 12, 16}) {
    std::vector<DistanceAndId> dp_neighbors(count);
    std::vector<DistanceAndId> l2_neighbors(count);
    for (size_t i = 0; i < count; ++i) {
      dp_neighbors[i] = {.id = static_cast<GingerNodeId>(i + 1)};
      l2_neighbors[i] = {.id = static_cast<GingerNodeId>(i + 1)};
    }
    DotProductDistanceUint8(0, absl::MakeSpan(dp_neighbors), dataset);
    SquaredL2DistanceUint8(0, absl::MakeSpan(l2_neighbors), dataset);
    for (size_t i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ(dp_neighbors[i].distance,
                      -1.0f * static_cast<float>(i + 1));
      const float diff = 1.0f - static_cast<float>(i + 1);
      const float expected_l2 = diff * diff + 4.0f + 9.0f + 16.0f;
      EXPECT_FLOAT_EQ(l2_neighbors[i].distance, expected_l2);
    }
  }
}

// Tests batches long enough to run the steady-state prefetch path of the batch
// kernels (more than `kBatchPrefetchDistance + 4` neighbors), with neighbor ids
// in non-sequential order and coordinates spanning the full range of the type,
// against a scalar reference.
template <typename T>
class DistanceBytesTypedTest : public ::testing::Test {};

using ByteTypes = ::testing::Types<int8_t, uint8_t>;
TYPED_TEST_SUITE(DistanceBytesTypedTest, ByteTypes);

TYPED_TEST(DistanceBytesTypedTest, LongBatchMatchesScalarReference) {
  using T = TypeParam;
  constexpr int kNumNeighbors = 50;
  // Maps any non-negative int onto the full range of `T` without overflow.
  const auto to_byte = [](int x) {
    return static_cast<T>(x % 256 + std::numeric_limits<T>::min());
  };

  for (const int dim : {128, 300}) {
    SpanDataset dataset(kNumNeighbors, dim, CoordinateTypeFor<T>());

    std::vector<T> query(dim);
    for (int i = 0; i < dim; ++i) {
      query[i] = to_byte(i * 7);
    }

    std::vector<float> expected_dots(kNumNeighbors);
    std::vector<float> expected_l2s(kNumNeighbors);
    for (int n = 0; n < kNumNeighbors; ++n) {
      std::vector<T> dataset_point(dim);
      int64_t expected_dot = 0;
      int64_t expected_l2 = 0;
      for (int i = 0; i < dim; ++i) {
        dataset_point[i] = to_byte(i * 11 + n * 19);
        const int64_t diff = int64_t{query[i]} - int64_t{dataset_point[i]};
        expected_dot += int64_t{query[i]} * int64_t{dataset_point[i]};
        expected_l2 += diff * diff;
      }
      dataset.Set<T>(n, dataset_point);
      expected_dots[n] = static_cast<float>(expected_dot);
      expected_l2s[n] = static_cast<float>(expected_l2);
    }

    // Visits the neighbors in reverse order, so that the ids differ from their
    // positions in the batch.
    std::vector<DistanceAndId> dp_neighbors(kNumNeighbors);
    std::vector<DistanceAndId> l2_neighbors(kNumNeighbors);
    for (int n = 0; n < kNumNeighbors; ++n) {
      const auto id = static_cast<GingerNodeId>(kNumNeighbors - 1 - n);
      dp_neighbors[n] = {.id = id};
      l2_neighbors[n] = {.id = id};
    }
    DotProductDistance(absl::MakeConstSpan(query), absl::MakeSpan(dp_neighbors),
                       dataset);
    SquaredL2Distance(absl::MakeConstSpan(query), absl::MakeSpan(l2_neighbors),
                      dataset);
    for (int n = 0; n < kNumNeighbors; ++n) {
      const GingerNodeId id = dp_neighbors[n].id;
      EXPECT_FLOAT_EQ(dp_neighbors[n].distance, -expected_dots[id])
          << "dim=" << dim << ", id=" << id;
      EXPECT_FLOAT_EQ(l2_neighbors[n].distance, expected_l2s[id])
          << "dim=" << dim << ", id=" << id;
    }
  }
}

TEST(DistanceBytesDeathTest, Uint8QueryDimensionalityMismatch) {
  SpanDataset dataset(2, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {1, 0, 0, 0});
  dataset.Set<uint8_t>(1, {0, 1, 0, 0});
  const std::vector<uint8_t> query = {1, 0, 0};
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
