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

#include "ginger/search/search_functors.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::FloatEq;
using ::testing::FloatNear;

// Verify concept conformance for all scoring functors.
static_assert(SearchDistanceScorer<FloatSquaredL2>);
static_assert(BuildDistanceScorer<FloatSquaredL2>);
static_assert(SearchDistanceScorer<FloatDotProduct>);
static_assert(BuildDistanceScorer<FloatDotProduct>);
static_assert(SearchDistanceScorer<Int8SquaredL2>);
static_assert(BuildDistanceScorer<Int8SquaredL2>);
static_assert(SearchDistanceScorer<Int8DotProduct>);
static_assert(BuildDistanceScorer<Int8DotProduct>);
static_assert(SearchDistanceScorer<Uint8SquaredL2>);
static_assert(BuildDistanceScorer<Uint8SquaredL2>);
static_assert(SearchDistanceScorer<Uint8DotProduct>);
static_assert(BuildDistanceScorer<Uint8DotProduct>);
static_assert(SearchDistanceScorer<TurboQuantSearchScorer>);
static_assert(SearchDistanceScorer<TurboSliceSearchScorer>);
static_assert(BuildDistanceScorer<TurboQuantBuildScorer>);

SpanDataset CreateTestFloatDataset() {
  SpanDataset dataset(3, 4);
  dataset.Set<float>(0, {1.0f, 0.0f, 0.0f, 0.0f});
  dataset.Set<float>(1, {0.0f, 1.0f, 0.0f, 0.0f});
  dataset.Set<float>(2, {0.5f, 2.0f, 0.0f, 0.0f});
  return dataset;
}

absl::StatusOr<TurboQuantDatasetSoA> CreateTestTurboQuantDataset(
    const TurboQuantCodec& quantizer, const SpanDataset& float_dataset) {
  std::vector<absl::Span<const float>> points;
  points.reserve(float_dataset.size());
  for (size_t i = 0; i < float_dataset.size(); ++i) {
    points.push_back(float_dataset.get<float>(i));
  }
  return TurboQuantDatasetSoA::Create(quantizer, points);
}

TEST(PostProcessTurboQuantDistancesTest, ClampsNegativeDistancesToZero) {
  std::vector<DistanceAndId> candidates = {
      {.distance = -0.5f, .id = 0},
      {.distance = 0.0f, .id = 1},
      {.distance = 1.5f, .id = 2},
      {.distance = -0.001f, .id = 3},
  };
  PostProcessTurboQuantDistances(absl::MakeSpan(candidates));
  EXPECT_THAT(
      candidates,
      ElementsAre(FieldsAre(FloatEq(0.0f), 0), FieldsAre(FloatEq(0.0f), 1),
                  FieldsAre(FloatEq(1.5f), 2), FieldsAre(FloatEq(0.0f), 3)));
}

TEST(PostProcessTurboQuantDistancesTest, PreservesPositiveDistances) {
  std::vector<DistanceAndId> candidates = {
      {.distance = 0.1f, .id = 10},
      {.distance = 2.5f, .id = 20},
  };
  PostProcessTurboQuantDistances(absl::MakeSpan(candidates));
  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(0.1f), 10),
                                      FieldsAre(FloatEq(2.5f), 20)));
}

TEST(PostProcessTurboQuantDistancesTest, HandlesEmptySpan) {
  std::vector<DistanceAndId> candidates;
  PostProcessTurboQuantDistances(absl::MakeSpan(candidates));
  EXPECT_TRUE(candidates.empty());
}

TEST(FloatSquaredL2Test, SearchScorerComputesDistances) {
  SpanDataset dataset = CreateTestFloatDataset();
  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  FloatSquaredL2 scorer{.dataset = &dataset,
                        .query = absl::MakeConstSpan(query)};

  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}, {.id = 2}};
  scorer(absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(0.0f), 0),
                                      FieldsAre(FloatEq(2.0f), 1),
                                      FieldsAre(FloatEq(4.25f), 2)));
}

TEST(FloatSquaredL2Test, BuildScorerComputesDistances) {
  SpanDataset dataset = CreateTestFloatDataset();
  FloatSquaredL2 scorer{.dataset = &dataset};

  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}, {.id = 2}};
  scorer(/*query_id=*/0, absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(0.0f), 0),
                                      FieldsAre(FloatEq(2.0f), 1),
                                      FieldsAre(FloatEq(4.25f), 2)));
}

TEST(FloatDotProductTest, SearchScorerComputesDistances) {
  SpanDataset dataset = CreateTestFloatDataset();
  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  FloatDotProduct scorer{.dataset = &dataset,
                         .query = absl::MakeConstSpan(query)};

  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}, {.id = 2}};
  scorer(absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(-1.0f), 0),
                                      FieldsAre(FloatEq(0.0f), 1),
                                      FieldsAre(FloatEq(-0.5f), 2)));
}

TEST(FloatDotProductTest, BuildScorerComputesDistances) {
  SpanDataset dataset = CreateTestFloatDataset();
  FloatDotProduct scorer{.dataset = &dataset};

  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}, {.id = 2}};
  scorer(/*query_id=*/0, absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(-1.0f), 0),
                                      FieldsAre(FloatEq(0.0f), 1),
                                      FieldsAre(FloatEq(-0.5f), 2)));
}

TEST(Int8SquaredL2Test, SearchScorerComputesDistances) {
  SpanDataset dataset(3, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {10, 0, 0, 0});
  dataset.Set<int8_t>(1, {0, 10, 0, 0});
  dataset.Set<int8_t>(2, {6, 8, 0, 0});

  std::vector<int8_t> query = {6, 0, 0, 0};
  Int8SquaredL2 scorer{.dataset = &dataset,
                       .query = absl::MakeConstSpan(query)};

  std::vector<DistanceAndId> candidates = {{.id = 1}, {.id = 2}};
  scorer(absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(136.0f), 1),
                                      FieldsAre(FloatEq(64.0f), 2)));
}

TEST(Int8SquaredL2Test, BuildScorerComputesDistances) {
  SpanDataset dataset(3, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {10, 0, 0, 0});
  dataset.Set<int8_t>(1, {0, 10, 0, 0});
  dataset.Set<int8_t>(2, {6, 8, 0, 0});

  Int8SquaredL2 scorer{.dataset = &dataset};

  std::vector<DistanceAndId> candidates = {{.id = 1}, {.id = 2}};
  scorer(/*query_id=*/0, absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(200.0f), 1),
                                      FieldsAre(FloatEq(80.0f), 2)));
}

TEST(Int8DotProductTest, SearchScorerComputesDistances) {
  SpanDataset dataset(3, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {10, 0, 0, 0});
  dataset.Set<int8_t>(1, {0, 10, 0, 0});
  dataset.Set<int8_t>(2, {6, 8, 0, 0});

  std::vector<int8_t> query = {2, 3, 0, 0};
  Int8DotProduct scorer{.dataset = &dataset,
                        .query = absl::MakeConstSpan(query)};

  std::vector<DistanceAndId> candidates = {{.id = 1}, {.id = 2}};
  scorer(absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(-30.0f), 1),
                                      FieldsAre(FloatEq(-36.0f), 2)));
}

TEST(Int8DotProductTest, BuildScorerComputesDistances) {
  SpanDataset dataset(3, 4, CoordinateType::kInt8);
  dataset.Set<int8_t>(0, {10, 0, 0, 0});
  dataset.Set<int8_t>(1, {0, 10, 0, 0});
  dataset.Set<int8_t>(2, {6, 8, 0, 0});

  Int8DotProduct scorer{.dataset = &dataset};

  std::vector<DistanceAndId> candidates = {{.id = 1}, {.id = 2}};
  scorer(/*query_id=*/0, absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(0.0f), 1),
                                      FieldsAre(FloatEq(-60.0f), 2)));
}

TEST(Uint8SquaredL2Test, SearchScorerComputesDistances) {
  SpanDataset dataset(3, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {10, 0, 0, 0});
  dataset.Set<uint8_t>(1, {0, 10, 0, 0});
  dataset.Set<uint8_t>(2, {6, 8, 0, 0});

  std::vector<uint8_t> query = {6, 0, 0, 0};
  Uint8SquaredL2 scorer{.dataset = &dataset,
                        .query = absl::MakeConstSpan(query)};

  std::vector<DistanceAndId> candidates = {{.id = 1}, {.id = 2}};
  scorer(absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(136.0f), 1),
                                      FieldsAre(FloatEq(64.0f), 2)));
}

TEST(Uint8SquaredL2Test, BuildScorerComputesDistances) {
  SpanDataset dataset(3, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {10, 0, 0, 0});
  dataset.Set<uint8_t>(1, {0, 10, 0, 0});
  dataset.Set<uint8_t>(2, {6, 8, 0, 0});

  Uint8SquaredL2 scorer{.dataset = &dataset};

  std::vector<DistanceAndId> candidates = {{.id = 1}, {.id = 2}};
  scorer(/*query_id=*/0, absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(200.0f), 1),
                                      FieldsAre(FloatEq(80.0f), 2)));
}

TEST(Uint8DotProductTest, SearchScorerComputesDistances) {
  SpanDataset dataset(3, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {10, 0, 0, 0});
  dataset.Set<uint8_t>(1, {0, 10, 0, 0});
  dataset.Set<uint8_t>(2, {6, 8, 0, 0});

  std::vector<uint8_t> query = {2, 3, 0, 0};
  Uint8DotProduct scorer{.dataset = &dataset,
                         .query = absl::MakeConstSpan(query)};

  std::vector<DistanceAndId> candidates = {{.id = 1}, {.id = 2}};
  scorer(absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(-30.0f), 1),
                                      FieldsAre(FloatEq(-36.0f), 2)));
}

TEST(Uint8DotProductTest, BuildScorerComputesDistances) {
  SpanDataset dataset(3, 4, CoordinateType::kUint8);
  dataset.Set<uint8_t>(0, {10, 0, 0, 0});
  dataset.Set<uint8_t>(1, {0, 10, 0, 0});
  dataset.Set<uint8_t>(2, {6, 8, 0, 0});

  Uint8DotProduct scorer{.dataset = &dataset};

  std::vector<DistanceAndId> candidates = {{.id = 1}, {.id = 2}};
  scorer(/*query_id=*/0, absl::MakeSpan(candidates));

  EXPECT_THAT(candidates, ElementsAre(FieldsAre(FloatEq(0.0f), 1),
                                      FieldsAre(FloatEq(-60.0f), 2)));
}

TEST(TurboQuantBuildScorerTest, ComputesSquaredL2DistancesAndClamps) {
  SpanDataset float_dataset = CreateTestFloatDataset();
  TurboQuantCodec quantizer(4, 123,
                            TurboQuantDistanceMetric::kSquaredL2Distance);
  ASSERT_OK_AND_ASSIGN(auto dataset,
                       CreateTestTurboQuantDataset(quantizer, float_dataset));

  TurboQuantBuildScorer scorer{.dataset = &dataset, .quantizer = &quantizer};
  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}, {.id = 2}};
  EXPECT_OK(scorer(/*query_id=*/0, absl::MakeSpan(candidates)));

  for (const auto& cand : candidates) {
    EXPECT_GE(cand.distance, 0.0f);
  }
  EXPECT_THAT(candidates[0], FieldsAre(FloatNear(0.0f, 0.15f), 0));
  EXPECT_THAT(candidates[1], FieldsAre(FloatNear(2.0f, 0.25f), 1));
  EXPECT_THAT(candidates[2], FieldsAre(FloatNear(4.25f, 0.5f), 2));
}

TEST(TurboQuantBuildScorerTest, ComputesDotProductDistances) {
  SpanDataset float_dataset = CreateTestFloatDataset();
  TurboQuantCodec quantizer(4, 123,
                            TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(auto dataset,
                       CreateTestTurboQuantDataset(quantizer, float_dataset));

  TurboQuantBuildScorer scorer{.dataset = &dataset, .quantizer = &quantizer};
  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}, {.id = 2}};
  EXPECT_OK(scorer(/*query_id=*/0, absl::MakeSpan(candidates)));

  EXPECT_THAT(candidates[0], FieldsAre(FloatNear(-1.0f, 0.15f), 0));
  EXPECT_THAT(candidates[1], FieldsAre(FloatNear(0.0f, 0.15f), 1));
  EXPECT_THAT(candidates[2], FieldsAre(FloatNear(-0.5f, 0.15f), 2));
}

TEST(ComputeTurboQuantBuildDistancesTest, ComputesDistancesDirectly) {
  SpanDataset float_dataset = CreateTestFloatDataset();
  TurboQuantCodec quantizer(4, 123,
                            TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(auto dataset,
                       CreateTestTurboQuantDataset(quantizer, float_dataset));

  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}, {.id = 2}};
  EXPECT_OK(ComputeTurboQuantBuildDistances(quantizer, dataset, /*query_id=*/0,
                                            absl::MakeSpan(candidates)));

  EXPECT_THAT(candidates[0], FieldsAre(FloatNear(-1.0f, 0.15f), 0));
  EXPECT_THAT(candidates[1], FieldsAre(FloatNear(0.0f, 0.15f), 1));
  EXPECT_THAT(candidates[2], FieldsAre(FloatNear(-0.5f, 0.15f), 2));
}

TEST(TurboQuantSearchScorerTest, ComputesDistancesWithFixedQuery) {
  SpanDataset float_dataset = CreateTestFloatDataset();
  TurboQuantCodec quantizer(4, 123,
                            TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(auto dataset,
                       CreateTestTurboQuantDataset(quantizer, float_dataset));

  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_OK_AND_ASSIGN(
      auto tq_query,
      TurboQuantQuery::Create(quantizer, absl::MakeConstSpan(query)));

  TurboQuantSearchScorer scorer{.dataset = &dataset, .fixed_query = &tq_query};
  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}, {.id = 2}};
  scorer(absl::MakeSpan(candidates));

  EXPECT_THAT(candidates[0], FieldsAre(FloatNear(-1.0f, 0.15f), 0));
  EXPECT_THAT(candidates[1], FieldsAre(FloatNear(0.0f, 0.15f), 1));
  EXPECT_THAT(candidates[2], FieldsAre(FloatNear(-0.5f, 0.15f), 2));
}

TEST(TurboSliceSearchScorerTest, ComputesDistancesWithTurboSliceQuery) {
  SpanDataset float_dataset = CreateTestFloatDataset();
  TurboQuantCodec quantizer(4, 123,
                            TurboQuantDistanceMetric::kDotProductDistance,
                            /*turboslice_dimensionality=*/2);
  ASSERT_OK_AND_ASSIGN(auto dataset,
                       CreateTestTurboQuantDataset(quantizer, float_dataset));

  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_OK_AND_ASSIGN(
      auto tq_query,
      TurboQuantQuery::Create(quantizer, absl::MakeConstSpan(query)));

  TurboSliceSearchScorer scorer{.dataset = &dataset,
                                .turboslice_query = &tq_query};
  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}, {.id = 2}};
  scorer(absl::MakeSpan(candidates));

  EXPECT_THAT(candidates[0], FieldsAre(FloatNear(-1.0f, 0.5f), 0));
  EXPECT_THAT(candidates[1], FieldsAre(FloatNear(0.0f, 0.1f), 1));
  EXPECT_THAT(candidates[2], FieldsAre(FloatNear(-0.5f, 0.5f), 2));
  EXPECT_LT(candidates[0].distance, candidates[2].distance);
  EXPECT_LT(candidates[2].distance, candidates[1].distance);
}

}  // namespace
}  // namespace ginger
