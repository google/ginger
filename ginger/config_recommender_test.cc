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

#include "ginger/config_recommender.h"

#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/searcher.h"
#include "ginger/searcher.pb.h"
#include "ginger/searcher_builder.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_reference_stub.h"

namespace ginger {
namespace {

using ::absl_testing::StatusIs;
using ::testing::SizeIs;

// Points in a random `rank`-dimensional subspace plus small noise, so that PCA
// truncates. With `normalize`, points are scaled to unit norm.
std::shared_ptr<SpanDataset> CreateLowRankDataset(GingerNodeId num_points,
                                                  int dimensionality, int rank,
                                                  bool normalize) {
  std::mt19937 rng(42);
  std::normal_distribution<float> normal;
  std::vector<std::vector<float>> basis(rank,
                                        std::vector<float>(dimensionality));
  for (std::vector<float>& row : basis) {
    for (float& x : row) x = normal(rng);
  }

  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
  std::vector<float> point(dimensionality);
  for (GingerNodeId i = 0; i < num_points; ++i) {
    for (float& x : point) x = 0.01f * normal(rng);
    for (int r = 0; r < rank; ++r) {
      const float coefficient = normal(rng);
      for (int d = 0; d < dimensionality; ++d) {
        point[d] += coefficient * basis[r][d];
      }
    }
    if (normalize) {
      float squared_norm = 0.0f;
      for (const float x : point) squared_norm += x * x;
      const float inverse_norm = 1.0f / std::sqrt(squared_norm);
      for (float& x : point) x *= inverse_norm;
    }
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }
  return dataset;
}

// Checks that the recommended config builds and searches. Parameterized by
// dimensionality (with and without PCA) and metric. Unit-norm dot-product data
// is eligible for the 3-bit prefix.
class RecommendGingerConfigTest
    : public ::testing::TestWithParam<std::tuple<int, DistanceMetric>> {
 private:
  ginger::internal::ParallelSchedulerReference scheduler_{4};
};

TEST_P(RecommendGingerConfigTest, BuildsAndSearches) {
  const auto [dimensionality, metric] = GetParam();
  constexpr GingerNodeId kNumPoints = 2000;
  // For unit-norm points, the point itself is its nearest dot-product neighbor.
  std::shared_ptr<SpanDataset> dataset =
      CreateLowRankDataset(kNumPoints, dimensionality, /*rank=*/32,
                           /*normalize=*/metric == DOT_PRODUCT);

  ASSERT_OK_AND_ASSIGN(const GingerSearcherConfig config,
                       RecommendGingerConfig(dimensionality, metric,
                                             /*robust_prune_alpha=*/1.2f));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  const SearchOptions options =
      searcher->DefaultSearchOptions(/*num_neighbors=*/10, /*beam_size=*/100);
  for (const GingerNodeId query_id : {0, 1234}) {
    const absl::Span<const float> query =
        std::as_const(*dataset).get<float>(query_id);
    ASSERT_OK_AND_ASSIGN(const std::vector<DistanceAndId> neighbors,
                         searcher->FindNeighbors(query, options));
    ASSERT_THAT(neighbors, SizeIs(options.num_neighbors()));
    EXPECT_EQ(neighbors[0].id, query_id);
  }
}

INSTANTIATE_TEST_SUITE_P(WithAndWithoutPca, RecommendGingerConfigTest,
                         ::testing::Combine(::testing::Values(128, 768),
                                            ::testing::Values(SQUARED_L2,
                                                              DOT_PRODUCT)));

TEST(RecommendGingerConfigValidationTest, RejectsInvalidArguments) {
  EXPECT_THAT(RecommendGingerConfig(/*dimensionality=*/0, SQUARED_L2,
                                    /*robust_prune_alpha=*/1.2f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(RecommendGingerConfig(/*dimensionality=*/128, UNDEFINED_DISTANCE,
                                    /*robust_prune_alpha=*/1.2f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(RecommendGingerConfig(/*dimensionality=*/128, SQUARED_L2,
                                    /*robust_prune_alpha=*/0.9f),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace ginger
