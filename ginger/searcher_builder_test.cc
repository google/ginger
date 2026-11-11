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

#include "ginger/searcher_builder.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/build_progress.h"
#include "ginger/candidate_filtering/jl.h"
#include "ginger/dataset.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/projection/config.pb.h"
#include "ginger/projection/pca_float.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/searcher.h"
#include "ginger/searcher.pb.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"
#include "ginger/vamana/config.pb.h"
#include "ginger/vamana/index_builder.h"
#include "ginger/internal/scheduler_reference_stub.h"

namespace ginger {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAreArray;
using ::absl_testing::StatusIs;

class GingerSearcherBuilderTest : public ::testing::Test {
 protected:
  using TurboQuantQuantizer = TurboQuantCodec;
  using TurboQuantDataset = TurboQuantDatasetSoA;

  std::shared_ptr<SpanDataset> CreateDataset(
      const std::vector<std::vector<float>>& points) {
    if (points.empty()) {
      return std::make_shared<SpanDataset>(/*num_points=*/0,
                                           /*dimensionality=*/0);
    }

    const GingerNodeId num_points = points.size();
    const int32_t dimensionality = points[0].size();

    auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
    for (GingerNodeId i = 0; i < num_points; ++i) {
      dataset->Set<float>(i, absl::MakeConstSpan(points[i]));
    }

    return dataset;
  }

  std::shared_ptr<SpanDataset> CreateInt8Dataset(
      const std::vector<std::vector<int8_t>>& points) {
    if (points.empty()) {
      return std::make_shared<SpanDataset>(/*num_points=*/0,
                                           /*dimensionality=*/0,
                                           CoordinateType::kInt8);
    }

    const GingerNodeId num_points = points.size();
    const int32_t dimensionality = points[0].size();

    auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality,
                                                 CoordinateType::kInt8);
    for (GingerNodeId i = 0; i < num_points; ++i) {
      dataset->Set<int8_t>(i, absl::MakeConstSpan(points[i]));
    }

    return dataset;
  }

  std::shared_ptr<SpanDataset> CreateUint8Dataset(
      const std::vector<std::vector<uint8_t>>& points) {
    if (points.empty()) {
      return std::make_shared<SpanDataset>(/*num_points=*/0,
                                           /*dimensionality=*/0,
                                           CoordinateType::kUint8);
    }

    const GingerNodeId num_points = points.size();
    const int32_t dimensionality = points[0].size();

    auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality,
                                                 CoordinateType::kUint8);
    for (GingerNodeId i = 0; i < num_points; ++i) {
      dataset->Set<uint8_t>(i, absl::MakeConstSpan(points[i]));
    }

    return dataset;
  }

  absl::StatusOr<std::pair<std::shared_ptr<TurboQuantQuantizer>,
                           std::shared_ptr<TurboQuantDataset>>>
  CreateQuantizedDataset(const std::vector<std::vector<float>>& data,
                         DistanceMetric metric, bool normalized_datapoints) {
    const GingerNodeId num_points = data.size();
    const int32_t dimensionality = data.empty() ? 4 : data[0].size();

    auto quant_metric = metric == SQUARED_L2
                            ? TurboQuantDistanceMetric::kSquaredL2Distance
                            : TurboQuantDistanceMetric::kDotProductDistance;

    auto quantizer = std::make_shared<TurboQuantQuantizer>(
        dimensionality, /*seed=*/123, quant_metric);

    ABSL_ASSIGN_OR_RETURN(
        auto dataset, TurboQuantDataset::Create(
                          *quantizer, num_points,
                          TurboQuantDatasetOptions{
                              .pad_to_cache_line_size = false,
                              .quantization_bits = 4,
                              .store_query_byte_sums = false,
                              .normalized_datapoints = normalized_datapoints}));

    for (GingerNodeId i = 0; i < num_points; ++i) {
      ABSL_RETURN_IF_ERROR(dataset.UpdatePoint(*quantizer, data[i], i));
    }

    return std::make_pair(
        quantizer, std::make_shared<TurboQuantDataset>(std::move(dataset)));
  }

 private:
  ginger::internal::ParallelSchedulerReference scheduler_{4};
};

SearchOptions MakeSearchOptions(
    SearchSpace search_space = SEARCH_SPACE_TURBO_QUANT,
    RerankingStrategy reranking = RERANKING_STRATEGY_NONE, int beam_size = 2,
    int num_neighbors = 1, bool enable_jl_filter = false) {
  SearchOptions options;
  options.set_search_space(search_space);
  options.set_reranking_strategy(reranking);
  options.set_beam_size(beam_size);
  options.set_num_neighbors(num_neighbors);
  if (enable_jl_filter) {
    options.mutable_candidate_filter()->mutable_jl();
  }
  return options;
}

// =========================================================================
// 1. Tests using `GingerSearcherConfig` and Float/Byte Dataset
// =========================================================================

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithFloatDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 2);
  EXPECT_EQ(searcher->coordinate_type(), CoordinateType::kFloat);
}

TEST_F(GingerSearcherBuilderTest, BuildVamanaWithInt8Dataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(10);

  auto dataset =
      CreateInt8Dataset({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 3);
  EXPECT_EQ(searcher->dimensionality(), 4);
  EXPECT_EQ(searcher->coordinate_type(), CoordinateType::kInt8);
}

TEST_F(GingerSearcherBuilderTest, BuildVamanaWithUint8Dataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(10);

  auto dataset =
      CreateUint8Dataset({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 3);
  EXPECT_EQ(searcher->dimensionality(), 4);
  EXPECT_EQ(searcher->coordinate_type(), CoordinateType::kUint8);
}

TEST_F(GingerSearcherBuilderTest, BuildVamanaWithInt8DatasetAndTurboQuant) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  auto dataset = CreateInt8Dataset({{1, 2, 3, 4}, {5, 6, 7, 8}});
  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 4);
  EXPECT_EQ(searcher->coordinate_type(), CoordinateType::kInt8);

  std::vector<int8_t> query = {1, 2, 3, 4};
  EXPECT_THAT(
      searcher->FindNeighbors(absl::MakeSpan(query), MakeSearchOptions()),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, BuildVamanaWithUint8DatasetAndTurboQuant) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  auto dataset = CreateUint8Dataset({{1, 2, 3, 4}, {5, 6, 7, 8}});
  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 4);
  EXPECT_EQ(searcher->coordinate_type(), CoordinateType::kUint8);

  std::vector<uint8_t> query = {1, 2, 3, 4};
  EXPECT_THAT(
      searcher->FindNeighbors(absl::MakeSpan(query), MakeSearchOptions()),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, BuildPipnnWithInt8DatasetAndTurboQuant) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_pipnn_config()->set_robust_prune_alpha(1.0f);
  config.mutable_pipnn_config()->set_max_degree(4);
  config.mutable_pipnn_config()->set_max_leaf_size(10);
  config.mutable_pipnn_config()->set_min_leaf_size(2);
  auto* leader_selection =
      config.mutable_pipnn_config()->mutable_leader_selection();
  leader_selection->set_top_level_leaders(4);
  leader_selection->set_max_num_leaders(4);
  leader_selection->set_leader_sample_rate(0.5f);
  leader_selection->add_fanout_sequence(2);
  config.mutable_turbo_quant()->set_seed(123);

  std::vector<std::vector<int8_t>> points;
  for (int i = 0; i < 20; ++i) {
    points.push_back({static_cast<int8_t>(i), static_cast<int8_t>(i + 1),
                      static_cast<int8_t>(i + 2), static_cast<int8_t>(i + 3)});
  }
  auto dataset = CreateInt8Dataset(points);
  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 20);
  EXPECT_EQ(searcher->dimensionality(), 4);
  EXPECT_EQ(searcher->coordinate_type(), CoordinateType::kInt8);

  std::vector<int8_t> query = {0, 1, 2, 3};
  EXPECT_THAT(
      searcher->FindNeighbors(absl::MakeSpan(query), MakeSearchOptions()),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, VamanaBuildStopsWhenCancelled) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .SetCancellationCheck([] { return true; })
                  .Build(),
              StatusIs(absl::StatusCode::kCancelled));

  std::atomic<int> checks = 0;
  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .SetCancellationCheck([&checks] { return checks.fetch_add(1) >= 1; })
          .Build(),
      StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(checks.load(), 2);
}

TEST_F(GingerSearcherBuilderTest, PipnnBuildStopsWhenCancelled) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_pipnn_config()->set_robust_prune_alpha(1.0f);
  config.mutable_pipnn_config()->set_max_degree(4);
  config.mutable_pipnn_config()->set_max_leaf_size(10);
  config.mutable_pipnn_config()->set_min_leaf_size(2);
  auto* leader_selection =
      config.mutable_pipnn_config()->mutable_leader_selection();
  leader_selection->set_top_level_leaders(4);
  leader_selection->set_max_num_leaders(4);
  leader_selection->set_leader_sample_rate(0.5f);
  leader_selection->add_fanout_sequence(2);
  config.mutable_turbo_quant()->set_seed(123);

  std::vector<std::vector<int8_t>> points;
  for (int i = 0; i < 20; ++i) {
    points.push_back({static_cast<int8_t>(i), static_cast<int8_t>(i + 1),
                      static_cast<int8_t>(i + 2), static_cast<int8_t>(i + 3)});
  }
  auto dataset = CreateInt8Dataset(points);
  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .SetCancellationCheck([] { return true; })
                  .Build(),
              StatusIs(absl::StatusCode::kCancelled));

  std::atomic<int> checks = 0;
  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .SetCancellationCheck([&checks] { return checks.fetch_add(1) >= 1; })
          .Build(),
      StatusIs(absl::StatusCode::kCancelled));
  EXPECT_GE(checks.load(), 2);
}

TEST_F(GingerSearcherBuilderTest, BuildPipnnWithUint8DatasetWithoutTurboQuant) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_pipnn_config()->set_robust_prune_alpha(1.0f);
  config.mutable_pipnn_config()->set_max_degree(4);
  config.mutable_pipnn_config()->set_max_leaf_size(10);
  config.mutable_pipnn_config()->set_min_leaf_size(2);
  auto* leader_selection =
      config.mutable_pipnn_config()->mutable_leader_selection();
  leader_selection->set_top_level_leaders(4);
  leader_selection->set_max_num_leaders(4);
  leader_selection->set_leader_sample_rate(0.5f);
  leader_selection->add_fanout_sequence(2);

  std::vector<std::vector<uint8_t>> points;
  for (int i = 0; i < 20; ++i) {
    points.push_back({static_cast<uint8_t>(i), static_cast<uint8_t>(i + 1),
                      static_cast<uint8_t>(i + 2),
                      static_cast<uint8_t>(i + 3)});
  }
  auto dataset = CreateUint8Dataset(points);
  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());
  ASSERT_NE(searcher, nullptr);
  EXPECT_FALSE(searcher->config().has_turbo_quant());
  EXPECT_EQ(searcher->size(), 20);
  EXPECT_EQ(searcher->dimensionality(), 4);
  EXPECT_EQ(searcher->coordinate_type(), CoordinateType::kUint8);

  std::vector<uint8_t> query = {0, 1, 2, 3};
  EXPECT_THAT(searcher->FindNeighbors(absl::MakeSpan(query),
                                      MakeSearchOptions(SEARCH_SPACE_FLOAT)),
              IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithDotProductNormalized) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  EXPECT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 2);
  EXPECT_EQ(searcher->coordinate_type(), CoordinateType::kFloat);
}

TEST_F(GingerSearcherBuilderTest, ValidateAlphaBoundsForMetrics) {
  auto dataset = CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}});

  // Squared L2 with alpha < 1.0 must fail
  {
    GingerSearcherConfig config;
    config.set_distance_metric(SQUARED_L2);
    config.mutable_vamana_config()->set_max_degree(4);
    auto* pass = config.mutable_vamana_config()->add_passes();
    pass->set_max_beam_width(2);
    pass->mutable_prune_config()->set_alpha(0.5f);
    EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("robust_prune alpha must be >= 1.0")));
  }

  // Squared L2 with normalized datapoints and alpha < 1.0 must fail
  {
    GingerSearcherConfig config;
    config.set_distance_metric(SQUARED_L2);
    config.set_normalized_datapoints(true);
    config.mutable_vamana_config()->set_max_degree(4);
    auto* pass = config.mutable_vamana_config()->add_passes();
    pass->set_max_beam_width(2);
    pass->mutable_prune_config()->set_alpha(0.5f);
    EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("robust_prune alpha must be >= 1.0")));
  }

  // Dot Product (MIPS) with alpha < 1.0 must fail
  {
    GingerSearcherConfig config;
    config.set_distance_metric(DOT_PRODUCT);
    config.mutable_vamana_config()->set_max_degree(4);
    auto* pass = config.mutable_vamana_config()->add_passes();
    pass->set_max_beam_width(2);
    pass->mutable_prune_config()->set_alpha(0.5f);
    EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("robust_prune alpha must be >= 1.0")));
  }

  // PiPNN with robust_prune_alpha < 1.0 must fail
  {
    GingerSearcherConfig config;
    config.set_distance_metric(DOT_PRODUCT);
    config.mutable_pipnn_config()->set_robust_prune_alpha(0.5f);
    EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("robust_prune alpha must be >= 1.0")));
  }

  // PiPNN with prune_config.alpha < 1.0 must fail
  {
    GingerSearcherConfig config;
    config.set_distance_metric(DOT_PRODUCT);
    config.mutable_pipnn_config()->mutable_prune_config()->set_alpha(0.5f);
    EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("robust_prune alpha must be >= 1.0")));
  }

  // PiPNN with both robust_prune_alpha and prune_config must fail
  {
    GingerSearcherConfig config;
    config.set_distance_metric(DOT_PRODUCT);
    config.mutable_pipnn_config()->set_robust_prune_alpha(1.2f);
    config.mutable_pipnn_config()->mutable_prune_config()->set_alpha(1.2f);
    EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Cannot specify both robust_prune_alpha and "
                                   "prune_config")));
  }
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureMissingDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  GingerSearcherBuilder builder(config);

  EXPECT_THAT(builder.Build(), StatusIs(absl::StatusCode::kInvalidArgument,
                                        "No dataset provided."));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureMissingVamanaConfig) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  // We do not set the vamana_config.

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  EXPECT_THAT(
      GingerSearcherBuilder(config).AddDataset(dataset).Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Must provide either Vamana or PiPNN config to build an index."));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureEmptyFloatDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  // We use an empty dataset to test the empty dataset check.
  auto dataset = CreateDataset({});

  EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument, "Empty dataset."));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureBuildCannotBeCalledMultipleTimes) {
  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  VamanaConfig vamana_config;
  vamana_config.set_max_degree(4);
  vamana_config.add_passes()->set_max_beam_width(2);

  GingerSearcherBuilder builder;
  builder.AddDataset(dataset)
      .SetDistanceMetric(SQUARED_L2)
      .SetVamanaConfig(vamana_config);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       builder.Build());
  EXPECT_NE(searcher, nullptr);

  EXPECT_THAT(
      builder.Build(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               "GingerSearcherBuilder is single-use and cannot be reused."));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureByteCoordinatesWithPca) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_pca()->set_dimension(2);

  auto dataset = CreateInt8Dataset({{1, 2, 3, 4}, {5, 6, 7, 8}});
  EXPECT_THAT(
      GingerSearcherBuilder(config).AddDataset(dataset).Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("PCA projection is not supported for byte coordinates")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailurePcaConfigWithoutTurboQuant) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_pca()->set_dimension(2);

  auto dataset =
      CreateDataset({{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}});
  EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "PCA projection is only supported with TurboQuant."));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureProjectionWithoutTurboQuant) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset =
      CreateDataset({{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}});
  Eigen::MatrixXf mat(2, 4);
  mat.setZero();
  ASSERT_OK_AND_ASSIGN(auto proj, FloatPcaProjection::Create(mat));

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddProjection(std::move(proj))
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "PCA projection is only supported with TurboQuant."));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureByteCoordinatesWithNormalizedDatapoints) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.set_normalized_datapoints(true);

  auto dataset = CreateInt8Dataset({{1, 2, 3, 4}, {5, 6, 7, 8}});
  EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("normalized_datapoints is only supported with "
                                 "float coordinates")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureInt8CoordinatesWithJl) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_jl()->set_jl_dimensionality(64);

  auto dataset = CreateInt8Dataset({{1, 2, 3, 4}, {5, 6, 7, 8}});
  EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("JL candidate filtering requires float "
                                 "coordinates")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureUint8CoordinatesWithJl) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_jl()->set_jl_dimensionality(64);

  auto dataset = CreateUint8Dataset({{1, 2, 3, 4}, {5, 6, 7, 8}});
  EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("JL candidate filtering requires float "
                                 "coordinates")));
}

TEST_F(GingerSearcherBuilderTest,
       NormalizedDatapointsAutoDetectedWhenUnsetForUnitNormDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());
  EXPECT_TRUE(searcher->config().normalized_datapoints());
}

TEST_F(GingerSearcherBuilderTest,
       NormalizedDatapointsNotAutoDetectedWhenExplicitlyFalse) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.set_normalized_datapoints(false);

  auto dataset = CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());
  EXPECT_FALSE(searcher->config().normalized_datapoints());
}

// =============================================================================
// 2. Tests setting options via Builder Setters
// =============================================================================

TEST_F(GingerSearcherBuilderTest, FluentInterface) {
  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  VamanaConfig vamana_config;
  vamana_config.set_max_degree(4);
  vamana_config.add_passes()->set_max_beam_width(2);

  bool callback_called = false;
  auto progress_callback = [&](const IndexBuildProgress&) {
    callback_called = true;
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder()
                           .AddDataset(dataset)
                           .SetDistanceMetric(SQUARED_L2)
                           .SetVamanaConfig(vamana_config)
                           .SetProgressCallback(progress_callback)
                           .Build());

  EXPECT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 2);
  EXPECT_TRUE(callback_called);
}

TEST_F(GingerSearcherBuilderTest, BuildWithTurboQuantConfigSetter) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  TurboQuantConfig tq_config;
  tq_config.set_seed(123);
  tq_config.set_extra_bits(0);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .SetTurboQuantConfig(tq_config)
                           .Build());

  EXPECT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 2);
}

TEST_F(GingerSearcherBuilderTest, BuildWithPcaConfigSetter) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  auto dataset = CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}});

  PcaProjectionConfig pca_config;
  pca_config.set_dimension(1);

  GingerSearcherBuilder::BuildStats stats;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .SetPcaProjectionConfig(pca_config)
                           .Build(&stats));

  EXPECT_NE(searcher, nullptr);
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithTurboQuantPca) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  auto dataset = CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}});

  PcaProjectionConfig pca_config;
  pca_config.set_dimension(1);
  pca_config.set_matrix_representation(MATRIX_REPRESENTATION_TURBOQUANT);

  GingerSearcherBuilder::BuildStats stats;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .SetPcaProjectionConfig(pca_config)
                           .Build(&stats));

  EXPECT_NE(searcher, nullptr);
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithPcaTruncation) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  // 3D dataset where variance is mostly in 1D
  auto dataset = CreateDataset({{1.0f, 0.0f, 0.0f},
                                {1.1f, 0.01f, 0.0f},
                                {0.9f, -0.01f, 0.0f},
                                {1.05f, 0.005f, 0.0f}});

  PcaProjectionConfig pca_config;
  auto* truncation = pca_config.mutable_truncation();
  truncation->set_significance_threshold(0.99f);
  truncation->set_truncation_threshold(0.5f);  // Require max 50% of 3D -> 1D

  GingerSearcherBuilder::BuildStats stats;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .SetPcaProjectionConfig(pca_config)
                           .Build(&stats));
  ASSERT_NE(searcher, nullptr);

  std::vector<float> query = {1.0f, 0.0f, 0.0f};
  EXPECT_THAT(
      searcher->FindNeighbors(absl::MakeSpan(query),
                              MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                                                RERANKING_STRATEGY_FLOAT)),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithPcaTruncationFallback) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  // 2D dataset where variance is spread
  auto dataset =
      CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}});

  PcaProjectionConfig pca_config;
  auto* truncation = pca_config.mutable_truncation();
  truncation->set_significance_threshold(0.9f);
  // Impossible to reduce 2D to less than 1D (0.5)
  truncation->set_truncation_threshold(0.1f);

  GingerSearcherBuilder::BuildStats stats;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .SetPcaProjectionConfig(pca_config)
                           .Build(&stats));
  ASSERT_NE(searcher, nullptr);

  std::vector<float> query = {1.0f, 0.0f};
  EXPECT_THAT(
      searcher->FindNeighbors(absl::MakeSpan(query),
                              MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                                                RERANKING_STRATEGY_FLOAT)),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, BuildFailureInvalidPcaDimension) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  auto dataset = CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}});

  PcaProjectionConfig pca_config;
  pca_config.set_dimension(3);  // Invalid: > dataset dim (2)

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .SetPcaProjectionConfig(pca_config)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(GingerSearcherBuilderTest,
       BuildFailureTurboSliceLargerThanPcaDimension) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_turbo_quant()->set_turboslice_dimensionality(3);
  config.mutable_pca()->set_dimension(2);

  auto dataset = CreateDataset({{1.0f, 0.0f, 0.0f, 0.0f},
                                {0.0f, 1.0f, 0.0f, 0.0f},
                                {0.0f, 0.0f, 1.0f, 0.0f},
                                {0.0f, 0.0f, 0.0f, 1.0f}});

  EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("turboslice_dimensionality")));
}

TEST_F(GingerSearcherBuilderTest, PcaExtraBitsRerankUsesFullDimensionality) {
  constexpr int kNumPoints = 32;
  constexpr int kDim = 16;
  std::vector<std::vector<float>> points(kNumPoints, std::vector<float>(kDim));
  for (int i = 0; i < kNumPoints; ++i) {
    for (int d = 0; d < kDim; ++d) {
      points[i][d] = static_cast<float>((i * 7 + d * 13) % 17) - 8.0f;
    }
  }

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(8);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(8);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_turbo_quant()->set_extra_bits(4);

  // The graph is built without PCA, so both searchers share the same index and
  // their full-dimensional extra-bits reranking must produce identical results.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> full_searcher,
      GingerSearcherBuilder(config).AddDataset(CreateDataset(points)).Build());

  config.mutable_pca()->set_dimension(kDim / 2);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> pca_searcher,
      GingerSearcherBuilder(config).AddDataset(CreateDataset(points)).Build());

  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> reordered_pca_searcher,
      GingerSearcherBuilder(config).AddDataset(CreateDataset(points)).Build());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> in_place_reordered_pca_searcher,
      GingerSearcherBuilder(config)
          .AddDataset(CreateDataset(points))
          .SetReorderInPlace(true)
          .Build());

  const SearchOptions options = MakeSearchOptions(
      SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
      /*beam_size=*/kNumPoints, /*num_neighbors=*/kNumPoints);
  const std::vector<float>& query = points[3];
  ASSERT_OK_AND_ASSIGN(
      std::vector<DistanceAndId> expected,
      full_searcher->FindNeighbors(absl::MakeConstSpan(query), options));
  for (const GingerSearcher* searcher :
       {pca_searcher.get(), reordered_pca_searcher.get(),
        in_place_reordered_pca_searcher.get()}) {
    ASSERT_OK_AND_ASSIGN(
        std::vector<DistanceAndId> actual,
        searcher->FindNeighbors(absl::MakeConstSpan(query), options));

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(actual[i].id, expected[i].id);
      EXPECT_FLOAT_EQ(actual[i].distance, expected[i].distance);
    }
  }

  std::vector<int8_t> int8_query(kDim, 1);
  EXPECT_THAT(
      pca_searcher->FindNeighbors(absl::MakeConstSpan(int8_query), options),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Extra bits reranking with a rerank dataset requires "
                         "float query vectors.")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureUndefinedDistanceMetric) {
  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  EXPECT_THAT(GingerSearcherBuilder().AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Distance metric is not set."));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureInvalidTurboQuantSeed) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  // We use an invalid TurboQuant seed to test the validation failure.
  TurboQuantConfig tq_config;
  tq_config.set_seed(-1);

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .SetTurboQuantConfig(tq_config)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "TurboQuant seed must be non-negative."));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureInvalidTurboQuantExtraBits) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  // We use an invalid TurboQuant extra bits to test the validation failure.
  TurboQuantConfig tq_config;
  tq_config.set_extra_bits(5);

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .SetTurboQuantConfig(tq_config)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Extra bits per dimension must be 0, 1, 2, 3, 4, 6, or "
                       "8."));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureInvalidTurboSliceDimensionalityNegative) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  TurboQuantConfig tq_config;
  tq_config.set_turboslice_dimensionality(-1);

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .SetTurboQuantConfig(tq_config)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "TurboQuant turboslice_dimensionality must be non-negative."));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureInvalidTurboSliceDimensionalityExceedsDim) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  TurboQuantConfig tq_config;
  tq_config.set_turboslice_dimensionality(4);  // Dataset dimension is 2.

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .SetTurboQuantConfig(tq_config)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "TurboSlice dimensionality cannot be greater than vector "
               "dimensionality."));
}

// =============================================================================
// 3. Tests using Pre-built Components
// =============================================================================

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithPreQuantizedDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, dataset]),
      CreateQuantizedDataset({{1.0f, 2.0f}, {3.0f, 4.0f}}, SQUARED_L2,
                             /*normalized_datapoints=*/false));

  auto searcher_or = GingerSearcherBuilder(config)
                         .AddQuantizedDataset(dataset, quantizer)
                         .Build();
  ASSERT_OK(searcher_or);

  auto searcher = *std::move(searcher_or);
  EXPECT_NE(searcher, nullptr);

  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 2);
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithPrebuiltIndex) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  // Create a placeholder VamanaIndex (FlatGraph) of size 2.
  auto index = std::make_unique<const VamanaIndex>(/*num_nodes=*/2,
                                                   /*max_degree=*/4);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(index))
                           .Build());

  EXPECT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 2);
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithPrebuiltCompressedIndex) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  // Create a placeholder CompressedCSRGraph of size 2.
  std::vector<int64_t> offsets = {0, 0, 0};
  std::vector<uint8_t> compressed_targets = {};
  int64_t num_arcs = 0;

  auto compressed_index = std::make_unique<const CompressedCSRGraph<>>(
      std::move(offsets), std::move(compressed_targets), num_arcs);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(compressed_index))
                           .Build());

  EXPECT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 2);
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureQuantizerTurboSliceDimMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_turbo_quant()->set_turboslice_dimensionality(2);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});
  // Create quantized dataset with turboslice_dimensionality 0 (default) to
  // mismatch with the 2 in the config.
  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, quantized_dataset]),
      CreateQuantizedDataset({{1.0f, 2.0f}, {3.0f, 4.0f}}, SQUARED_L2,
                             /*normalized_datapoints=*/false));

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(quantized_dataset, quantizer)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Quantizer TurboSlice dimensionality mismatch")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureQuantizerMetricMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});
  // Create quantized dataset with DOT_PRODUCT metric to mismatch with the
  // SQUARED_L2 metric in the config.
  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, quantized_dataset]),
      CreateQuantizedDataset({{1.0f, 2.0f}, {3.0f, 4.0f}}, DOT_PRODUCT,
                             /*normalized_datapoints=*/false));

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(quantized_dataset, quantizer)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Quantizer metric does not match the metric specified in the "
               "config."));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureIndexSizeMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});  // Size 2

  // We create an index of size 3 to mismatch with the dataset size.
  auto index = std::make_unique<const VamanaIndex>(/*num_nodes=*/3,
                                                   /*max_degree=*/4);

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddIndex(std::move(index))
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index size mismatch")));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureCompressedIndexSizeMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);

  // We use a dataset of size 2 to mismatch with the index size.
  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  // Create compressed index of size 3 (4 offsets).
  std::vector<int64_t> offsets = {0, 0, 0, 0};
  std::vector<uint8_t> compressed_targets = {};
  int64_t num_arcs = 0;

  auto compressed_index = std::make_unique<const CompressedCSRGraph<>>(
      std::move(offsets), std::move(compressed_targets), num_arcs);

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddIndex(std::move(compressed_index))
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index size mismatch")));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureQuantizedDatasetMissingQuantizer) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, quantized_dataset]),
      CreateQuantizedDataset({{1.0f, 2.0f}, {3.0f, 4.0f}}, SQUARED_L2,
                             /*normalized_datapoints=*/false));

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddQuantizedDataset(quantized_dataset, nullptr)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Quantizer must be provided if a quantized dataset is provided."));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureQuantizerMissingDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto quantizer = std::make_shared<TurboQuantQuantizer>(
      /*dimensionality=*/2, /*seed=*/42,
      TurboQuantDistanceMetric::kSquaredL2Distance);

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(nullptr, quantizer)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Quantized dataset must be provided if a quantizer is provided."));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureQuantizerDatasetDimMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  ASSERT_OK_AND_ASSIGN((auto [unused_quantizer, quantized_dataset]),
                       CreateQuantizedDataset(
                           {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}},
                           SQUARED_L2, /*normalized_datapoints=*/false));
  auto wrong_dim_quantizer = std::make_shared<TurboQuantQuantizer>(
      /*dimensionality=*/2, /*seed=*/42,
      TurboQuantDistanceMetric::kSquaredL2Distance);

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddQuantizedDataset(quantized_dataset, wrong_dim_quantizer)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("quantizer has 2 dimensions, quantized dataset has 4 "
                         "dimensions")));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureQuantizerDatasetTurboSliceDimMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  ASSERT_OK_AND_ASSIGN((auto [unused_quantizer, quantized_dataset]),
                       CreateQuantizedDataset(
                           {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}},
                           SQUARED_L2, /*normalized_datapoints=*/false));
  auto ts_quantizer = std::make_shared<TurboQuantQuantizer>(
      /*dimensionality=*/4, /*seed=*/42,
      TurboQuantDistanceMetric::kSquaredL2Distance,
      /*turboslice_dimensionality=*/2);

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddQuantizedDataset(quantized_dataset, ts_quantizer)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Quantizer TurboSlice dimensionality mismatch")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureFloatQuantizedSizeMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  // We create a float dataset of size 2 and a quantized dataset of size 3.
  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});
  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, quantized_dataset]),
      CreateQuantizedDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                             SQUARED_L2, /*normalized_datapoints=*/false));

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddQuantizedDataset(quantized_dataset, quantizer)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dataset size mismatch")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureFloatQuantizedDimMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);

  // We create a float dataset of dimension 2 and a quantized dataset of
  // dimension 3.
  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});
  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, quantized_dataset]),
      CreateQuantizedDataset({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                             SQUARED_L2, /*normalized_datapoints=*/false));

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddQuantizedDataset(quantized_dataset, quantizer)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimensionality mismatch")));
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithJlConfig) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_jl()->set_jl_dimensionality(128);
  config.mutable_jl()->set_seed(42);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  std::vector<float> query = {1.0f, 2.0f};
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/2, /*num_neighbors=*/1,
                            /*enable_jl_filter=*/true)),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithPreBuiltJlDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});
  ASSERT_OK_AND_ASSIGN(auto jl_dataset,
                       JlDataset::Create(64, 42, SQUARED_L2, *dataset));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddJlDataset(std::move(jl_dataset))
                           .Build());

  std::vector<float> query = {1.0f, 2.0f};
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/2, /*num_neighbors=*/1,
                            /*enable_jl_filter=*/true)),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureInvalidJlDim) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_jl()->set_jl_dimensionality(100);  // not a multiple of 64

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("multiple of 64")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureJlWithoutFloatDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_jl()->set_jl_dimensionality(64);

  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, dataset]),
      CreateQuantizedDataset({{1.0f, 2.0f}, {3.0f, 4.0f}}, SQUARED_L2,
                             /*normalized_datapoints=*/false));

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddQuantizedDataset(dataset, quantizer)
          .Build(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Building JL dataset requires a float dataset")));
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithGraphCompressionAndJl) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_jl()->set_jl_dimensionality(128);
  config.mutable_jl()->set_seed(42);
  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  std::vector<float> query = {1.0f, 2.0f};
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/2, /*num_neighbors=*/1,
                            /*enable_jl_filter=*/true)),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest,
       BuildSuccessWithGraphCompressionRecursivePartitioning) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  std::vector<float> query = {1.0f, 2.0f};
  EXPECT_THAT(searcher->FindNeighbors(
                  absl::MakeSpan(query),
                  MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                                    /*beam_size=*/2, /*num_neighbors=*/1)),
              IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest,
       BuildSuccessWithGraphCompressionRecursiveBisection) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_BISECTION);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  std::vector<float> query = {1.0f, 2.0f};
  EXPECT_THAT(searcher->FindNeighbors(
                  absl::MakeSpan(query),
                  MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                                    /*beam_size=*/2, /*num_neighbors=*/1)),
              IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest,
       BuildFailureWithUnsupportedReorderingAlgorithm) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(static_cast<ReorderingAlgorithm>(999));

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});

  EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported reordering algorithm: 999")));
}

TEST_F(GingerSearcherBuilderTest,
       BuildFailureWithUnspecifiedReorderingAlgorithm) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(REORDERING_ALGORITHM_UNSPECIFIED);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});

  EXPECT_THAT(GingerSearcherBuilder(config).AddDataset(dataset).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Reordering algorithm must be specified when "
                                 "reordering is enabled.")));
}

TEST_F(GingerSearcherBuilderTest,
       BuildAppliesReorderingWhenGraphCompressionDisabled) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.set_use_graph_compression(false);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto flat1 = std::make_unique<FlatGraph>(4, 2);
  auto flat2 = std::make_unique<FlatGraph>(4, 2);
  for (int i = 0; i < 4; ++i) {
    flat1->SetNeighbors(i, {(i + 1) % 4});
    flat2->SetNeighbors(i, {(i + 1) % 4});
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> unreordered,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat1))
                           .Build());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> reordered,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat2))
                           .AddReordering({3, 2, 1, 0})
                           .Build());

  EXPECT_TRUE(std::holds_alternative<std::unique_ptr<const FlatGraph>>(
      reordered->index()));

  SearchOptions options;
  options.set_num_neighbors(4);
  options.set_beam_size(4);
  options.set_search_space(SEARCH_SPACE_FLOAT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);

  std::vector<float> query = {6.5f, 7.5f};
  ASSERT_OK_AND_ASSIGN(auto expected_neighbors,
                       unreordered->FindNeighbors(query, options));
  ASSERT_OK_AND_ASSIGN(auto actual_neighbors,
                       reordered->FindNeighbors(query, options));
  EXPECT_EQ(actual_neighbors, expected_neighbors);
}

TEST_F(GingerSearcherBuilderTest,
       BuildSuccessGraphCompressionWithoutReordering) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.set_use_graph_compression(true);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto flat = std::make_unique<FlatGraph>(4, 2);
  for (int i = 0; i < 4; ++i) {
    flat->SetNeighbors(i, {(i + 1) % 4});
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat))
                           .Build());

  EXPECT_TRUE(
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          searcher->index()));
}

TEST_F(GingerSearcherBuilderTest,
       BuildSuccessWithPrebuiltIndexReorderingTurboQuantAndJl) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_jl()->set_jl_dimensionality(128);
  config.mutable_jl()->set_seed(42);
  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto flat = std::make_unique<FlatGraph>(4, 2);
  for (int i = 0; i < 4; ++i) {
    flat->SetNeighbors(i, {(i + 1) % 4});
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat))
                           .Build());

  EXPECT_TRUE(
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          searcher->index()));

  std::vector<float> query = {1.0f, 2.0f};
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/2, /*num_neighbors=*/1,
                            /*enable_jl_filter=*/true)),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest,
       BuildSuccessWithReorderingAlgorithmWhenGraphCompressionDisabled) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.set_use_graph_compression(false);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto flat1 = std::make_unique<FlatGraph>(4, 2);
  auto flat2 = std::make_unique<FlatGraph>(4, 2);
  for (int i = 0; i < 4; ++i) {
    flat1->SetNeighbors(i, {(i + 1) % 4});
    flat2->SetNeighbors(i, {(i + 1) % 4});
  }

  GingerSearcherConfig uncompressed_config;
  uncompressed_config.set_distance_metric(SQUARED_L2);
  uncompressed_config.set_use_graph_compression(false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> unreordered,
                       GingerSearcherBuilder(uncompressed_config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat1))
                           .Build());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> reordered,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat2))
                           .Build());

  EXPECT_TRUE(std::holds_alternative<std::unique_ptr<const FlatGraph>>(
      reordered->index()));

  SearchOptions options;
  options.set_num_neighbors(4);
  options.set_beam_size(4);
  options.set_search_space(SEARCH_SPACE_FLOAT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);

  std::vector<float> query = {6.5f, 7.5f};
  ASSERT_OK_AND_ASSIGN(auto expected_neighbors,
                       unreordered->FindNeighbors(query, options));
  ASSERT_OK_AND_ASSIGN(auto actual_neighbors,
                       reordered->FindNeighbors(query, options));
  EXPECT_EQ(actual_neighbors, expected_neighbors);
}

TEST_F(GingerSearcherBuilderTest,
       BuildSuccessWithExplicitReorderingAlgorithmNone) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.set_reordering_algorithm(REORDERING_ALGORITHM_NONE);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto flat1 = std::make_unique<FlatGraph>(4, 2);
  auto flat2 = std::make_unique<FlatGraph>(4, 2);
  for (int i = 0; i < 4; ++i) {
    flat1->SetNeighbors(i, {(i + 1) % 4});
    flat2->SetNeighbors(i, {(i + 1) % 4});
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher_uncompressed,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat1))
                           .Build());

  // REORDERING_ALGORITHM_NONE means no reordering was applied, so FlatGraph is
  // preserved.
  EXPECT_TRUE(std::holds_alternative<std::unique_ptr<const FlatGraph>>(
      searcher_uncompressed->index()));

  config.set_use_graph_compression(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher_compressed,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat2))
                           .Build());

  EXPECT_TRUE(
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          searcher_compressed->index()));
}

TEST_F(GingerSearcherBuilderTest,
       BuildAppliesReorderingOutOfPlaceWithTurboQuant) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.set_use_graph_compression(false);
  config.mutable_turbo_quant()->set_seed(123);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto flat1 = std::make_unique<FlatGraph>(4, 2);
  auto flat2 = std::make_unique<FlatGraph>(4, 2);
  for (int i = 0; i < 4; ++i) {
    flat1->SetNeighbors(i, {(i + 1) % 4});
    flat2->SetNeighbors(i, {(i + 1) % 4});
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> in_place,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat1))
                           .AddReordering({3, 2, 1, 0})
                           .SetReorderInPlace(true)
                           .Build());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> out_of_place,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat2))
                           .AddReordering({3, 2, 1, 0})
                           .SetReorderInPlace(false)
                           .Build());

  SearchOptions options;
  options.set_num_neighbors(4);
  options.set_beam_size(4);
  options.set_search_space(SEARCH_SPACE_TURBO_QUANT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);

  std::vector<float> query = {6.5f, 7.5f};
  ASSERT_OK_AND_ASSIGN(auto expected_neighbors,
                       in_place->FindNeighbors(query, options));
  ASSERT_OK_AND_ASSIGN(auto actual_neighbors,
                       out_of_place->FindNeighbors(query, options));
  EXPECT_EQ(actual_neighbors, expected_neighbors);
}

TEST_F(GingerSearcherBuilderTest, BuildAppliesReorderingToJlDataset) {
  constexpr int kPoints = 64;
  constexpr int kDimensions = 32;
  std::vector<std::vector<float>> points(kPoints,
                                         std::vector<float>(kDimensions));
  for (int i = 0; i < kPoints; ++i) {
    for (int j = 0; j < kDimensions; ++j) {
      points[i][j] =
          static_cast<float>((i * kDimensions + j) * 37 % 101) / 101.0f - 0.5f;
    }
  }
  auto dataset = CreateDataset(points);
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.set_use_graph_compression(false);
  config.mutable_jl()->set_jl_dimensionality(64);
  config.mutable_jl()->set_seed(7);
  auto make_graph = [&] {
    auto graph = std::make_unique<FlatGraph>(kPoints, 2);
    for (int i = 0; i < kPoints; ++i) {
      graph->SetNeighbors(i, {(i + 1) % kPoints, (i + 7) % kPoints});
    }
    return graph;
  };
  // Asymmetric, so that the permutation differs from its inverse.
  std::vector<GingerNodeId> reordering;
  for (int i = 0; i < kPoints; ++i) reordering.push_back((i + 5) % kPoints);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> plain,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(make_graph())
                           .Build());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> reordered,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(make_graph())
                           .AddReordering(reordering)
                           .Build());

  // Results use original IDs, so the JL candidate filter must see the same
  // points in both searchers.
  SearchOptions options;
  options.set_num_neighbors(5);
  options.set_beam_size(20);
  options.set_search_space(SEARCH_SPACE_FLOAT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);
  options.mutable_candidate_filter()->mutable_jl();
  for (int q = 0; q < 4; ++q) {
    std::vector<float> query(kDimensions);
    for (int j = 0; j < kDimensions; ++j) {
      query[j] = static_cast<float>((q * 13 + j * 7) % 17) - 8.0f;
    }
    ASSERT_OK_AND_ASSIGN(auto expected, plain->FindNeighbors(query, options));
    ASSERT_OK_AND_ASSIGN(auto actual, reordered->FindNeighbors(query, options));
    EXPECT_EQ(actual, expected) << "query " << q;
  }
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureEntryPointOutOfBounds) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto flat = std::make_unique<FlatGraph>(4, 2);

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddIndex(std::move(flat))
                  .SetEntryPoint(4)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Entry point is out of bounds")));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureInverseReorderingNotPermutation) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto flat = std::make_unique<FlatGraph>(4, 2);

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddIndex(std::move(flat))
          .AddInverseReordering({0, 1, 1, 3})
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Inverse reordering contains duplicate indices")));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureInverseReorderingWithEnabledReordering) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddIndex(std::make_unique<FlatGraph>(4, 2))
          .AddReordering({3, 2, 1, 0})
          .AddInverseReordering({3, 2, 1, 0})
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Cannot specify inverse reordering when reordering is enabled")));

  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);
  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddIndex(std::make_unique<FlatGraph>(4, 2))
          .AddInverseReordering({3, 2, 1, 0})
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Cannot specify inverse reordering when reordering is enabled")));
}

TEST_F(GingerSearcherBuilderTest, SetEntryPointRemappedWhenReordered) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.set_use_graph_compression(true);
  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto flat = std::make_unique<FlatGraph>(4, 2);
  for (int i = 0; i < 4; ++i) {
    flat->SetNeighbors(i, {(i + 1) % 4});
  }

  // With reordering {3, 2, 1, 0}, original node 1 is mapped to stored node 2,
  // and the inverse reordering is {3, 2, 1, 0}.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> reordered_searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(flat))
                           .AddReordering({3, 2, 1, 0})
                           .SetEntryPoint(1)
                           .Build());

  auto compressed_copy = std::make_unique<const CompressedCSRGraph<>>(
      *std::get<std::unique_ptr<const CompressedCSRGraph<>>>(
          reordered_searcher->index()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> restored_searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddIndex(std::move(compressed_copy))
                           .SetEntryPoint(2)
                           .AddInverseReordering({3, 2, 1, 0})
                           .Build());

  EXPECT_EQ(*reordered_searcher, *restored_searcher);
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureJlSourceDimMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto other_dataset = CreateDataset({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  ASSERT_OK_AND_ASSIGN(auto jl_dataset,
                       JlDataset::Create(64, 42, SQUARED_L2, *other_dataset));

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddJlDataset(std::move(jl_dataset))
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimensionality mismatch")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureJlMetricMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});
  ASSERT_OK_AND_ASSIGN(auto jl_dataset,
                       JlDataset::Create(64, 42, DOT_PRODUCT, *dataset));

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddJlDataset(std::move(jl_dataset))
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("JL dataset metric does not match")));
}

TEST_F(GingerSearcherBuilderTest, ValidationFailureProjectionInputDimMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}});
  ASSERT_OK_AND_ASSIGN((auto [quantizer, q_dataset]),
                       CreateQuantizedDataset({{1.0f}, {0.0f}}, DOT_PRODUCT,
                                              /*normalized_datapoints=*/false));

  Eigen::MatrixXf mat(1, 3);
  mat.setRandom();
  ASSERT_OK_AND_ASSIGN(auto proj, FloatPcaProjection::Create(mat));

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddQuantizedDataset(q_dataset, quantizer)
                  .AddProjection(std::move(proj))
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("projection input has 3 dimensions")));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureProjectionOutputDimMismatch) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateDataset({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}});
  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, q_dataset]),
      CreateQuantizedDataset({{1.0f, 0.0f}, {0.0f, 1.0f}}, DOT_PRODUCT,
                             /*normalized_datapoints=*/false));

  Eigen::MatrixXf mat(1, 3);
  mat.setRandom();
  ASSERT_OK_AND_ASSIGN(auto proj, FloatPcaProjection::Create(mat));

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .AddQuantizedDataset(q_dataset, quantizer)
                  .AddProjection(std::move(proj))
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected quantized dataset dimensionality to "
                                 "be 1, but got 2")));
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureProjectionOutputDimMismatchWithoutFloatDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, q_dataset]),
      CreateQuantizedDataset({{1.0f, 0.0f}, {0.0f, 1.0f}}, DOT_PRODUCT,
                             /*normalized_datapoints=*/false));

  Eigen::MatrixXf mat(1, 3);
  mat.setRandom();
  ASSERT_OK_AND_ASSIGN(auto proj, FloatPcaProjection::Create(mat));

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddQuantizedDataset(q_dataset, quantizer)
                  .AddProjection(std::move(proj))
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected quantized dataset dimensionality to "
                                 "be 1, but got 2")));
}

TEST_F(GingerSearcherBuilderTest, BuildWithQuantizedDatasetProjectionAndJl) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_jl()->set_jl_dimensionality(64);
  config.mutable_jl()->set_seed(42);

  ASSERT_OK_AND_ASSIGN((auto [quantizer, q_dataset]),
                       CreateQuantizedDataset({{1.0f}, {2.0f}}, SQUARED_L2,
                                              /*normalized_datapoints=*/false));

  Eigen::MatrixXf mat(1, 3);
  mat.setRandom();
  ASSERT_OK_AND_ASSIGN(auto proj, FloatPcaProjection::Create(mat));

  auto float_for_jl = CreateDataset({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  ASSERT_OK_AND_ASSIGN(auto jl_dataset,
                       JlDataset::Create(64, 42, SQUARED_L2, *float_for_jl));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddQuantizedDataset(q_dataset, quantizer)
                           .AddProjection(std::move(proj))
                           .AddJlDataset(std::move(jl_dataset))
                           .Build());
  EXPECT_NE(searcher, nullptr);
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureJlDimensionMismatchWithProjectionWithoutFloatDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  ASSERT_OK_AND_ASSIGN((auto [quantizer, q_dataset]),
                       CreateQuantizedDataset({{1.0f}, {2.0f}}, SQUARED_L2,
                                              /*normalized_datapoints=*/false));

  Eigen::MatrixXf mat(1, 3);
  mat.setRandom();
  ASSERT_OK_AND_ASSIGN(auto proj, FloatPcaProjection::Create(mat));

  auto float_for_jl = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});
  ASSERT_OK_AND_ASSIGN(auto jl_dataset,
                       JlDataset::Create(64, 42, SQUARED_L2, *float_for_jl));

  EXPECT_THAT(
      GingerSearcherBuilder(config)
          .AddQuantizedDataset(q_dataset, quantizer)
          .AddProjection(std::move(proj))
          .AddJlDataset(std::move(jl_dataset))
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Dimensionality mismatch: JL source dimensionality")));
}

TEST_F(GingerSearcherBuilderTest,
       BuildSuccessByteCoordinatesWithPreQuantizedDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateInt8Dataset({{1, 2}, {3, 4}});
  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, quantized_dataset]),
      CreateQuantizedDataset({{1.0f, 2.0f}, {3.0f, 4.0f}}, SQUARED_L2,
                             /*normalized_datapoints=*/false));

  ASSERT_OK_AND_ASSIGN(auto searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddQuantizedDataset(quantized_dataset, quantizer)
                           .Build());
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 2);
  EXPECT_EQ(searcher->coordinate_type(), CoordinateType::kInt8);

  std::vector<int8_t> query = {1, 2};
  EXPECT_THAT(
      searcher->FindNeighbors(absl::MakeSpan(query), MakeSearchOptions()),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, BuildSuccessWithDropDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .SetDropDataset(true)
                           .Build());

  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 2);
  EXPECT_EQ(searcher->dimensionality(), 2);

  std::vector<float> query = {1.0f, 2.0f};
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE)),
      IsOkAndHolds(SizeIs(1)));

  // Searching in float space should fail since the float dataset was dropped.
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Float dataset not available")));

  // Float reranking should fail since the float dataset was dropped.
  EXPECT_THAT(
      searcher->FindNeighbors(absl::MakeSpan(query),
                              MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                                                RERANKING_STRATEGY_FLOAT)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Reranking requires float dataset")));
}

// Verifies that `drop_dataset` in the config drops the float dataset.
TEST_F(GingerSearcherBuilderTest, BuildSuccessWithDropDatasetInConfig) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);
  config.set_drop_dataset(true);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  std::vector<float> query = {1.0f, 2.0f};
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Float dataset not available")));
}

TEST_F(GingerSearcherBuilderTest,
       BuildSuccessWithDropDatasetAndGraphCompression) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);
  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);

  auto dataset =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .SetDropDataset(true)
                           .Build());

  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->size(), 4);
  EXPECT_EQ(searcher->dimensionality(), 2);

  std::vector<float> query = {1.0f, 2.0f};
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE)),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest,
       BuildSuccessWithDropDatasetPrebuiltIndexReorderingTurboQuantAndJl) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_jl()->set_jl_dimensionality(128);
  config.mutable_jl()->set_seed(42);
  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);

  auto dataset1 =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto dataset2 =
      CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});

  auto flat1 = std::make_unique<FlatGraph>(4, 2);
  auto flat2 = std::make_unique<FlatGraph>(4, 2);
  for (int i = 0; i < 4; ++i) {
    flat1->SetNeighbors(i, {(i + 1) % 4});
    flat2->SetNeighbors(i, {(i + 1) % 4});
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> without_drop,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset1)
                           .AddIndex(std::move(flat1))
                           .SetDropDataset(false)
                           .Build());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> with_drop,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset2)
                           .AddIndex(std::move(flat2))
                           .SetDropDataset(true)
                           .Build());

  EXPECT_EQ(without_drop->size(), with_drop->size());
  EXPECT_EQ(without_drop->dimensionality(), with_drop->dimensionality());

  std::vector<float> query = {1.0f, 2.0f};
  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/2, /*num_neighbors=*/2,
                        /*enable_jl_filter=*/true);
  ASSERT_OK_AND_ASSIGN(auto res_without_drop,
                       without_drop->FindNeighbors(query, options));
  ASSERT_OK_AND_ASSIGN(auto res_with_drop,
                       with_drop->FindNeighbors(query, options));
  EXPECT_EQ(res_without_drop, res_with_drop);
}

TEST_F(GingerSearcherBuilderTest,
       ValidationFailureDropDatasetWithoutQuantizedDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);

  auto dataset = CreateDataset({{1.0f, 2.0f}, {3.0f, 4.0f}});

  EXPECT_THAT(GingerSearcherBuilder(config)
                  .AddDataset(dataset)
                  .SetDropDataset(true)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot drop full-precision dataset without a "
                                 "quantized dataset provided or configured")));
}

TEST_F(GingerSearcherBuilderTest,
       BuildEnablesNormalizedDotProductWhenDatasetIsUnitNorm) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);

  auto dataset =
      CreateDataset({{1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}});

  GingerSearcherBuilder builder(config);
  builder.AddDataset(dataset);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       builder.Build());

  ASSERT_NE(searcher, nullptr);

  std::vector<float> query = {1.0f, 0.0f};
  ASSERT_OK_AND_ASSIGN(
      std::vector<DistanceAndId> results,
      searcher->FindNeighbors(
          absl::MakeSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                            /*beam_size=*/4, /*num_neighbors=*/2)));

  ASSERT_THAT(results, SizeIs(2));
  EXPECT_EQ(results[0].id, 0);
  // distance = -dot(q, x) = -1.0
  EXPECT_NEAR(results[0].distance, -1.0f, 1e-4f);
}

TEST_F(GingerSearcherBuilderTest,
       VamanaUsesNonpositiveDistancesOnNonUnitNormDotProductDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(3);
  auto* pass = config.mutable_vamana_config()->add_passes();
  pass->set_max_beam_width(5);

  auto dataset = CreateDataset(
      {{2.0f, 0.0f}, {2.0f, 0.0f}, {2.0f, 0.0f}, {2.0f, 0.0f}, {2.0f, 0.0f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  const auto& graph =
      *std::get<std::unique_ptr<const FlatGraph>>(searcher->index());
  // Points have norm 2.0 (not unit norm), so BuildVamanaIndex sets
  // nonpositive_distances = true without shifting distances. If
  // nonpositive_distances were false (or shifted by +1.0, giving -3.0 < 0),
  // RobustPrune would discard all negative-distance candidates and return 0
  // neighbors instead of max_degree = 3.
  EXPECT_THAT(graph.Neighbors(0), SizeIs(3));
}

TEST_F(GingerSearcherBuilderTest,
       VamanaEnablesNormalizedDotProductOnUnitNormFloatDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(3);
  auto* pass = config.mutable_vamana_config()->add_passes();
  pass->set_max_beam_width(6);

  auto dataset = CreateDataset({{1.0f, 0.0f},
                                {1.0f, 0.0f},
                                {1.0f, 0.0f},
                                {1.0f, 0.0f},
                                {1.0f, 0.0f},
                                {1.0f, 0.0f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  const auto& graph =
      *std::get<std::unique_ptr<const FlatGraph>>(searcher->index());
  // Unit-norm float dataset shifts build-time distances by +1.0 to 0.0, so
  // zero-distance deduplication caps node 0's degree at max_dupe_candidates = 2
  // instead of max_degree = 3.
  EXPECT_THAT(graph.Neighbors(0), SizeIs(2));
}

TEST_F(GingerSearcherBuilderTest,
       VamanaClearsNonpositiveDistancesForNormalizedDotProduct) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(3);
  auto* pass = config.mutable_vamana_config()->add_passes();
  pass->set_max_beam_width(6);
  // Pre-configured with nonpositive_distances = true (as done for dot product).
  pass->mutable_prune_config()->set_nonpositive_distances(true);

  auto dataset = CreateDataset({{1.0f, 0.0f},
                                {0.0f, 1.0f},
                                {-1.0f, 0.0f},
                                {0.0f, -1.0f},
                                {0.7071f, 0.7071f},
                                {-0.7071f, 0.7071f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  const auto& graph =
      *std::get<std::unique_ptr<const FlatGraph>>(searcher->index());
  // Because distances are shifted to [0, 2], nonpositive_distances must be
  // cleared to false so RobustPrune does not drop positive-distance neighbors.
  EXPECT_FALSE(graph.Neighbors(0).empty());
}

TEST_F(GingerSearcherBuilderTest,
       PipnnUsesL2PruningOnUnitNormDotProductDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  auto* pipnn = config.mutable_pipnn_config();
  pipnn->set_max_degree(4);
  pipnn->set_max_leaf_size(10);
  pipnn->set_min_leaf_size(2);
  pipnn->set_pruning_strategy(
      PipnnConfig::PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY);
  pipnn->set_robust_prune_alpha(2.0f);
  pipnn->mutable_leader_selection()->add_fanout_sequence(2);
  config.mutable_turbo_quant()->set_seed(123);

  auto unit_dataset = CreateDataset({{1.0f, 0.0f, 0.0f, 0.0f},
                                     {0.8f, 0.6f, 0.0f, 0.0f},
                                     {0.6f, 0.8f, 0.0f, 0.0f},
                                     {0.6f, 0.8f, 0.0f, 0.0f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> unit_searcher,
      GingerSearcherBuilder(config).AddDataset(unit_dataset).Build());
  const auto& unit_graph =
      *std::get<std::unique_ptr<const FlatGraph>>(unit_searcher->index());
  // On unit-norm vectors, PiPNN prunes in Squared L2 space where node 1
  // ({0.8, 0.6}) prunes nodes 2 and 3 ({0.6, 0.8}) from node 0, leaving 1
  // neighbor.
  EXPECT_THAT(unit_graph.Neighbors(0), SizeIs(1));

  auto non_unit_dataset = CreateDataset({{2.0f, 0.0f, 0.0f, 0.0f},
                                         {1.6f, 1.2f, 0.0f, 0.0f},
                                         {1.2f, 1.6f, 0.0f, 0.0f},
                                         {1.2f, 1.6f, 0.0f, 0.0f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> non_unit_searcher,
      GingerSearcherBuilder(config).AddDataset(non_unit_dataset).Build());
  const auto& non_unit_graph =
      *std::get<std::unique_ptr<const FlatGraph>>(non_unit_searcher->index());
  // On non-unit-norm vectors, PiPNN uses unshifted MIPS pruning where node 1
  // does not prune node 2 at alpha = 2.0, leaving 2 neighbors.
  EXPECT_THAT(non_unit_graph.Neighbors(0), SizeIs(2));
}

TEST_F(GingerSearcherBuilderTest,
       VamanaAutoEnablesNormalizedDotProductOnNormalizedQuantizedDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(3);
  auto* pass = config.mutable_vamana_config()->add_passes();
  pass->set_max_beam_width(6);

  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, dataset]),
      CreateQuantizedDataset({{1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f}},
                             DOT_PRODUCT, /*normalized_datapoints=*/true));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddQuantizedDataset(dataset, quantizer)
                           .Build());

  const auto& graph =
      *std::get<std::unique_ptr<const FlatGraph>>(searcher->index());
  // With normalized_datapoints = true, distance_offset is set to 1.0f, so
  // zero-distance deduplication caps node 0's degree at max_dupe_candidates
  // = 2.
  EXPECT_THAT(graph.Neighbors(0), SizeIs(2));
}

TEST_F(
    GingerSearcherBuilderTest,
    VamanaDoesNotEnableNormalizedDotProductWhenQuantizedDatasetNotNormalized) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(3);
  auto* pass = config.mutable_vamana_config()->add_passes();
  pass->set_max_beam_width(6);

  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, dataset]),
      CreateQuantizedDataset({{1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f}},
                             DOT_PRODUCT, /*normalized_datapoints=*/false));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddQuantizedDataset(dataset, quantizer)
                           .Build());

  const auto& graph =
      *std::get<std::unique_ptr<const FlatGraph>>(searcher->index());
  // Without normalized_datapoints, distance_offset is not set (remains 0.0f),
  // so zero-distance deduplication does not trigger and degree reaches
  // max_degree = 3.
  EXPECT_THAT(graph.Neighbors(0), SizeIs(3));
}

TEST_F(GingerSearcherBuilderTest,
       VamanaEnablesNormalizedDotProductFromConfigFlagOnQuantizedDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.set_normalized_datapoints(true);
  config.mutable_vamana_config()->set_max_degree(3);
  auto* pass = config.mutable_vamana_config()->add_passes();
  pass->set_max_beam_width(6);

  ASSERT_OK_AND_ASSIGN(
      (auto [quantizer, dataset]),
      CreateQuantizedDataset({{1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f},
                              {1.0f, 0.0f}},
                             DOT_PRODUCT, /*normalized_datapoints=*/false));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddQuantizedDataset(dataset, quantizer)
                           .Build());

  const auto& graph =
      *std::get<std::unique_ptr<const FlatGraph>>(searcher->index());
  // With config.normalized_datapoints() = true, distance_offset is set to 1.0f,
  // so zero-distance deduplication caps node 0's degree at max_dupe_candidates
  // = 2.
  EXPECT_THAT(graph.Neighbors(0), SizeIs(2));
}

TEST_F(GingerSearcherBuilderTest,
       SquaredL2AutoDetectsNormalizedDatapointsForTurboQuant) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_vamana_config()->set_max_degree(2);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(4);
  config.mutable_turbo_quant()->set_seed(42);

  // Vectors with norm 0.9991 (squared norm ~0.9982) are within the
  // [0.999, 1.001] auto-detection tolerance. When TurboQuantDataset is built
  // with normalized_datapoints = true, it assumes ||x||^2 = 1.0 instead of
  // storing 0.9982, shifting the raw squared L2 distance up by (1.0 - 0.9982)
  // compared to an unnormalized dataset.
  auto dataset = CreateDataset({{0.9991f, 0.0f, 0.0f, 0.0f},
                                {0.0f, 0.9991f, 0.0f, 0.0f},
                                {0.0f, 0.0f, 0.9991f, 0.0f},
                                {0.0f, 0.0f, 0.0f, 0.9991f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher_auto,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  auto unnormalized_dataset = CreateDataset({{0.9991f, 0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.9991f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.9991f, 0.0f},
                                             {0.0f, 0.0f, 0.0f, 2.0f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher_unnormalized,
      GingerSearcherBuilder(config).AddDataset(unnormalized_dataset).Build());

  config.set_normalized_datapoints(true);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher_explicit,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/4, /*num_neighbors=*/1);
  // Use a query with non-zero distance to node 0 so distance estimations do
  // not clamp to zero under PostProcessTurboQuantDistances.
  const std::vector<float> query = {0.5f, 0.0f, 0.0f, 0.0f};
  ASSERT_OK_AND_ASSIGN(auto results_auto,
                       searcher_auto->FindNeighbors(query, options));
  ASSERT_OK_AND_ASSIGN(auto results_explicit,
                       searcher_explicit->FindNeighbors(query, options));
  ASSERT_OK_AND_ASSIGN(auto results_unnormalized,
                       searcher_unnormalized->FindNeighbors(query, options));
  ASSERT_THAT(results_auto, SizeIs(1));
  ASSERT_THAT(results_explicit, SizeIs(1));
  ASSERT_THAT(results_unnormalized, SizeIs(1));
  EXPECT_EQ(results_auto[0].distance, results_explicit[0].distance);
  EXPECT_GT(results_auto[0].distance, results_unnormalized[0].distance);
}

TEST_F(GingerSearcherBuilderTest,
       PcaProjectionDisablesNormalizedDatapointsInQuantizedDataset) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.set_normalized_datapoints(true);
  config.mutable_vamana_config()->set_max_degree(3);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(4);
  config.mutable_turbo_quant()->set_seed(42);
  config.mutable_pca()->set_dimension(2);

  // All points have exact unit norm (0.6^2 + 0.8^2 = 1.0), with a constant
  // offset of 0.8 along dimension 2. After PCA centering and projection to 2D,
  // each projected point has squared norm ||Px||^2 = 0.36 < 1.0. If
  // TurboQuantDataset assumed ||Px||^2 = 1.0, self-distance would be ~0.64
  // instead of ~0.0.
  auto dataset = CreateDataset({{0.6f, 0.0f, 0.8f, 0.0f},
                                {-0.6f, 0.0f, 0.8f, 0.0f},
                                {0.0f, 0.6f, 0.8f, 0.0f},
                                {0.0f, -0.6f, 0.8f, 0.0f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/4, /*num_neighbors=*/1);
  const std::vector<float> query = {0.6f, 0.0f, 0.8f, 0.0f};
  ASSERT_OK_AND_ASSIGN(auto results, searcher->FindNeighbors(query, options));
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_EQ(results[0].id, 0);
  EXPECT_LT(results[0].distance, 0.05f);
}

TEST_F(GingerSearcherBuilderTest,
       VamanaEnablesNormalizedDotProductEvenWhenPcaIsEnabled) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(3);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(6);
  config.mutable_turbo_quant()->set_seed(42);
  config.mutable_pca()->set_dimension(2);

  // Even when PCA is configured (and projection_ is built prior to index
  // construction), index construction occurs on the unprojected unit-norm
  // vectors, so normalized_dot_product should still shift distances by +1.0 and
  // cap duplicate degree at max_dupe_candidates = 2.
  auto dataset = CreateDataset({{1.0f, 0.0f, 0.0f, 0.0f},
                                {1.0f, 0.0f, 0.0f, 0.0f},
                                {1.0f, 0.0f, 0.0f, 0.0f},
                                {1.0f, 0.0f, 0.0f, 0.0f},
                                {1.0f, 0.0f, 0.0f, 0.0f},
                                {1.0f, 0.0f, 0.0f, 0.0f}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  const auto& graph =
      *std::get<std::unique_ptr<const FlatGraph>>(searcher->index());
  EXPECT_THAT(graph.Neighbors(0), SizeIs(2));
}

TEST_F(GingerSearcherBuilderTest,
       BuildVamanaWithRecommendedModeAndAllowThreeBitPrefixAt128Dims) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_turbo_quant()->set_allow_three_bit_prefix(true);

  std::vector<std::vector<float>> points(8, std::vector<float>(128, 0.1f));
  for (int i = 0; i < 8; ++i) {
    points[i][i] = 1.0f;
  }
  auto dataset = CreateDataset(points);
  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());
  ASSERT_NE(searcher, nullptr);

  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(points[0]),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE)),
      IsOkAndHolds(SizeIs(1)));
}

TEST_F(GingerSearcherBuilderTest, PipnnBuildsWithThreeBitPrefixDataset) {
  auto make_pipnn_config = [](DistanceMetric metric) {
    GingerSearcherConfig config;
    config.set_distance_metric(metric);
    config.mutable_pipnn_config()->set_robust_prune_alpha(1.0f);
    config.mutable_pipnn_config()->set_max_degree(4);
    config.mutable_pipnn_config()->set_max_leaf_size(10);
    config.mutable_pipnn_config()->set_min_leaf_size(2);
    auto* leader_selection =
        config.mutable_pipnn_config()->mutable_leader_selection();
    leader_selection->set_top_level_leaders(4);
    leader_selection->set_max_num_leaders(4);
    leader_selection->set_leader_sample_rate(0.5f);
    leader_selection->add_fanout_sequence(2);
    config.mutable_turbo_quant()->set_seed(123);
    config.mutable_turbo_quant()->set_scaling_storage_mode(
        TurboQuantConfig::SCALING_IN_THREE_BIT_PREFIX);
    return config;
  };

  std::vector<std::vector<float>> points(20, std::vector<float>(128, 0.1f));
  for (int i = 0; i < 20; ++i) {
    points[i][i] = 1.0f;
  }
  auto dataset = CreateDataset(points);

  // PiPNN builds the index on a dataset without the prefix, which is then
  // re-quantized with the prefix.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(make_pipnn_config(DOT_PRODUCT))
                           .AddDataset(dataset)
                           .Build());
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeSpan(points[0]),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE)),
      IsOkAndHolds(SizeIs(1)));

  // Unnormalized SQUARED_L2 is ineligible for the 3-bit prefix.
  EXPECT_THAT(GingerSearcherBuilder(make_pipnn_config(SQUARED_L2))
                  .AddDataset(dataset)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("SCALING_IN_THREE_BIT_PREFIX requires")));
}

GingerSearcherConfig MakeExplicitThreeBitPrefixConfig(DistanceMetric metric) {
  GingerSearcherConfig config;
  config.set_distance_metric(metric);
  config.mutable_vamana_config()->set_max_degree(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_turbo_quant()->set_scaling_storage_mode(
      TurboQuantConfig::SCALING_IN_THREE_BIT_PREFIX);
  return config;
}

std::vector<std::vector<float>> MakeRandomPoints(int num_points, int dim,
                                                 int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist;
  std::vector<std::vector<float>> points(num_points, std::vector<float>(dim));
  for (auto& point : points) {
    for (float& x : point) x = dist(rng);
  }
  return points;
}

TEST_F(GingerSearcherBuilderTest, PipnnIgnoresPrebuiltProjectionForIndexBuild) {
  GingerSearcherConfig config;
  config.set_distance_metric(SQUARED_L2);
  config.mutable_turbo_quant()->set_seed(123);
  auto* pipnn = config.mutable_pipnn_config();
  pipnn->set_max_degree(4);
  pipnn->set_max_leaf_size(10);
  pipnn->set_min_leaf_size(2);
  pipnn->set_robust_prune_alpha(1.0f);
  pipnn->mutable_leader_selection()->add_fanout_sequence(2);

  constexpr int kNumPoints = 64;
  constexpr int kDim = 8;
  auto dataset = CreateDataset(MakeRandomPoints(kNumPoints, kDim, /*seed=*/7));
  // A projection that keeps only the first two coordinates, which changes the
  // neighborhoods if PiPNN builds the graph on the projected points.
  Eigen::MatrixXf mat = Eigen::MatrixXf::Zero(kDim, kDim);
  mat(0, 0) = 1.0f;
  mat(1, 1) = 1.0f;
  ASSERT_OK_AND_ASSIGN(auto proj, FloatPcaProjection::Create(mat));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> expected_searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddProjection(std::move(proj))
                           .Build());

  // Index building operates in the full-dimensional space, so the pre-built
  // projection must not affect the PiPNN graph.
  const auto& expected_graph =
      *std::get<std::unique_ptr<const FlatGraph>>(expected_searcher->index());
  const auto& graph =
      *std::get<std::unique_ptr<const FlatGraph>>(searcher->index());
  for (GingerNodeId i = 0; i < kNumPoints; ++i) {
    EXPECT_THAT(graph.Neighbors(i),
                UnorderedElementsAreArray(expected_graph.Neighbors(i)))
        << "node " << i;
  }
}

TEST_F(GingerSearcherBuilderTest,
       ExplicitThreeBitPrefixRejectedBeforeIndexConstruction) {
  // Unnormalized squared L2 needs a stored squared norm.
  GingerSearcherConfig l2 = MakeExplicitThreeBitPrefixConfig(SQUARED_L2);
  // Normalized squared L2 is eligible, but the PCA re-quantization after index
  // construction is not.
  GingerSearcherConfig pca = l2;
  pca.set_normalized_datapoints(true);
  pca.mutable_pca()->set_dimension(64);
  // Too few TurboSlice dimensions.
  GingerSearcherConfig turboslice =
      MakeExplicitThreeBitPrefixConfig(DOT_PRODUCT);
  turboslice.mutable_turbo_quant()->set_turboslice_dimensionality(16);
  // The last case has too few dimensions.
  const std::pair<GingerSearcherConfig, int> cases[] = {
      {l2, 128},
      {pca, 128},
      {turboslice, 128},
      {MakeExplicitThreeBitPrefixConfig(DOT_PRODUCT), 16}};
  for (const auto& [config, dim] : cases) {
    EXPECT_THAT(
        GingerSearcherBuilder(config)
            .AddDataset(CreateDataset(MakeRandomPoints(8, dim, /*seed=*/1)))
            .Build(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("SCALING_IN_THREE_BIT_PREFIX requires")))
        << config.ShortDebugString();
  }
}

TEST_F(GingerSearcherBuilderTest, ExplicitThreeBitPrefixAcceptedForDotProduct) {
  auto dataset = CreateDataset(MakeRandomPoints(8, 32, /*seed=*/1));
  EXPECT_OK(GingerSearcherBuilder(MakeExplicitThreeBitPrefixConfig(DOT_PRODUCT))
                .AddDataset(dataset)
                .Build());
}

TEST_F(GingerSearcherBuilderTest,
       ResolveScalingStorageModeAllowsThreeBitPrefixWithExtraBits) {
  TurboQuantConfig config;
  config.set_extra_bits(2);
  EXPECT_EQ(internal::ResolveScalingStorageMode(config, /*dimensionality=*/128,
                                                NormLayout::kScalingOnly),
            ScalingStorageMode::kSeparateArray);
  config.set_allow_three_bit_prefix(true);
  EXPECT_EQ(internal::ResolveScalingStorageMode(config, /*dimensionality=*/128,
                                                NormLayout::kScalingOnly),
            ScalingStorageMode::kIn3BitPrefix);
}

}  // namespace
}  // namespace ginger
