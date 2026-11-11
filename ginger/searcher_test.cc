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

#include "ginger/searcher.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/build_progress.h"
#include "ginger/candidate_filtering/jl.h"
#include "ginger/dataset.h"
#include "ginger/internal/portable_random.h"
#include "ginger/projection/config.pb.h"
#include "ginger/searcher.pb.h"
#include "ginger/searcher_builder.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"
#include "ginger/vamana/config.pb.h"
#include "ginger/internal/scheduler_reference_stub.h"

namespace ginger {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::FloatNear;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

SearchOptions MakeSearchOptions(SearchSpace search, RerankingStrategy reranking,
                                int beam_size = 3, int num_neighbors = 2) {
  SearchOptions options;
  options.set_search_space(search);
  options.set_reranking_strategy(reranking);
  options.set_beam_size(beam_size);
  options.set_num_neighbors(num_neighbors);
  return options;
}

std::shared_ptr<SpanDataset> CreateOneHotDataset(int num_points,
                                                 int dimensionality) {
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality, 0.0f);
    point[i % dimensionality] = 1.0f;
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }
  return dataset;
}

std::shared_ptr<SpanDataset> CreateConstantDataset(int num_points,
                                                   int dimensionality,
                                                   float val) {
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality, val);
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }
  return dataset;
}

std::shared_ptr<SpanDataset> CreateNormalizedRandomDataset(int num_points,
                                                           int dimensionality,
                                                           int seed = 42) {
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality);
    float norm = 0.0f;
    for (int d = 0; d < dimensionality; ++d) {
      point[d] = dist(gen);
      norm += point[d] * point[d];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
      for (int d = 0; d < dimensionality; ++d) {
        point[d] /= norm;
      }
    }
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }
  return dataset;
}

GingerSearcherConfig CreateSearcherConfig(DistanceMetric metric = DOT_PRODUCT,
                                          int max_degree = 8, int seed = 123,
                                          int max_beam_width = 10) {
  GingerSearcherConfig config;
  config.set_distance_metric(metric);
  config.mutable_vamana_config()->set_max_degree(max_degree);
  config.mutable_vamana_config()->set_seed(seed);
  auto* pass = config.mutable_vamana_config()->add_passes();
  pass->set_max_beam_width(max_beam_width);
  return config;
}

struct QuantizedDatasetFixture {
  std::shared_ptr<TurboQuantCodec> quantizer;
  std::shared_ptr<TurboQuantDatasetSoA> quantized_dataset;
};

// Creates a quantized dataset fixture. If quantization_bits > 0, uses that
// value explicitly; otherwise uses the library default (4 bits).
absl::StatusOr<QuantizedDatasetFixture> CreateQuantizedDataset(
    int num_points, int dimensionality, DistanceMetric metric = DOT_PRODUCT,
    int seed = 123, int quantization_bits = 0, int turboslice_dim = 0,
    bool normalized_datapoints = false) {
  auto quant_metric = metric == SQUARED_L2
                          ? TurboQuantDistanceMetric::kSquaredL2Distance
                          : TurboQuantDistanceMetric::kDotProductDistance;
  auto quantizer = std::make_shared<TurboQuantCodec>(
      dimensionality, seed, quant_metric, turboslice_dim);
  absl::StatusOr<TurboQuantDatasetSoA> quantized_dataset_or;
  if (quantization_bits > 0) {
    quantized_dataset_or = TurboQuantDatasetSoA::Create(
        *quantizer, num_points,
        TurboQuantDatasetOptions{
            .pad_to_cache_line_size = false,
            .quantization_bits = quantization_bits,
            .store_query_byte_sums = false,
            .normalized_datapoints = normalized_datapoints,
            .scaling_storage_mode = ScalingStorageMode::kSeparateArray});
  } else {
    quantized_dataset_or = TurboQuantDatasetSoA::Create(
        *quantizer, num_points,
        TurboQuantDatasetOptions{
            .pad_to_cache_line_size = false,
            .quantization_bits = 4,
            .store_query_byte_sums = false,
            .normalized_datapoints = normalized_datapoints,
            .scaling_storage_mode = ScalingStorageMode::kSeparateArray});
  }
  if (!quantized_dataset_or.ok()) {
    return quantized_dataset_or.status();
  }
  return QuantizedDatasetFixture{
      .quantizer = quantizer,
      .quantized_dataset = std::make_shared<TurboQuantDatasetSoA>(
          std::move(*quantized_dataset_or))};
}

// Creates a quantized dataset fixture and populates it with points from the
// given float dataset. If quantization_bits > 0, uses that value explicitly.
absl::StatusOr<QuantizedDatasetFixture> CreateAndInitializeQuantizedDataset(
    const SpanDataset& dataset, DistanceMetric metric = DOT_PRODUCT,
    int seed = 123, int quantization_bits = 0, int turboslice_dim = 0,
    bool normalized_datapoints = false) {
  int num_points = dataset.size();
  int dimensionality = dataset.dimensionality();
  auto fixture_or = CreateQuantizedDataset(
      num_points, dimensionality, metric, seed, quantization_bits,
      turboslice_dim, normalized_datapoints);
  if (!fixture_or.ok()) {
    return fixture_or.status();
  }
  auto fixture = std::move(*fixture_or);
  for (int i = 0; i < num_points; ++i) {
    auto status = fixture.quantized_dataset->UpdatePoint(
        *fixture.quantizer, dataset.get<float>(i), i);
    if (!status.ok()) {
      return status;
    }
  }
  return fixture;
}

// Verifies that a GingerSearcher can be built with a basic float dataset
// and that the resulting index has the expected number of nodes.
TEST(GingerSearcherTest, BasicBuild) {
  auto dataset = CreateConstantDataset(10, 4, 0.1f);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 2);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));
  EXPECT_EQ(searcher->size(), 10);
  EXPECT_EQ(searcher->dimensionality(), 4);
}

// Verifies that searching with float coordinates in a basic dataset returns
// the expected nearest neighbor with correct distance.
TEST(GingerSearcherTest, BasicSearch) {
  auto dataset = CreateOneHotDataset(10, 4);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 5);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));
  std::vector<float> query(4, 0.0f);
  query[0] = 1.0f;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/5,
                            /*num_neighbors=*/3)));
  EXPECT_THAT(results, ElementsAre(FieldsAre(FloatNear(-1.0f, 1e-6), 0),
                                   FieldsAre(FloatNear(-1.0f, 1e-6), 4),
                                   FieldsAre(FloatNear(-1.0f, 1e-6), 8)));
}

// Verifies that a GingerSearcher can be built using TurboQuant for
// quantization and that the resulting index is correctly initialized.
TEST(GingerSearcherTest, TurboQuantBuild) {
  auto dataset = CreateConstantDataset(10, 64, 0.1f);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 2);

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());
  EXPECT_EQ(searcher->size(), 10);
  EXPECT_EQ(searcher->dimensionality(), 64);
}

// Verifies that searching with TurboQuant works correctly, including
// reranking with float coordinates and forced float search strategy.
TEST(GingerSearcherTest, TurboQuantSearch) {
  auto dataset = CreateOneHotDataset(100, 64);
  auto config = CreateSearcherConfig();

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;
  // Search with TQ without reranking.
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/10,
                            /*num_neighbors=*/2)));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results[0], FieldsAre(_, 0));

  // Search with reranking.
  GingerSearcherConfig rerank_config = config;

  ASSERT_OK_AND_ASSIGN(
      auto searcher_rerank,
      GingerSearcherBuilder(rerank_config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  ASSERT_OK_AND_ASSIGN(
      auto results_rerank,
      searcher_rerank->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                            /*beam_size=*/10,
                            /*num_neighbors=*/2)));
  ASSERT_THAT(results_rerank, Not(IsEmpty()));
  EXPECT_THAT(results_rerank[0], FieldsAre(FloatNear(-1.0f, 1e-6), 0));

  // Search with forced float search.
  ASSERT_OK_AND_ASSIGN(
      auto results_float,
      searcher_rerank->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/10,
                            /*num_neighbors=*/2)));
  ASSERT_THAT(results_float, Not(IsEmpty()));
  EXPECT_THAT(results_float[0], FieldsAre(FloatNear(-1.0f, 1e-6), 0));
}

// Verifies that searching with TurboQuant and extra bits reranking works
// with DOT_PRODUCT metric.
TEST(GingerSearcherTest, TurboQuantSearchExtraBitsRerankingDotProduct) {
  auto dataset = CreateOneHotDataset(100, 64);
  auto config = CreateSearcherConfig();

  // Create with 2 extra bits (quantization_bits = 6).
  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, DOT_PRODUCT, 123, 6));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  // DOT_PRODUCT rescoring applies 1.0f + cand.distance, clamped to >= 0.
  // For the nearest neighbor (identical vector), distance should be ~0.
  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                            RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
                            /*beam_size=*/10, /*num_neighbors=*/2),
          &stats));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results[0], FieldsAre(FloatNear(-1.0f, 1e-3), 0));
  // Distances must be non-positive (DOT_PRODUCT: distance = -dot).
  EXPECT_THAT(results, Each(FieldsAre(Le(0.0f), _)));
  // Verify stats track reranking.
  EXPECT_GT(stats.num_reranked, 0);

  // Verify searching with a quantized query (TurboQuantBasePointView) and
  // extra-bits reranking also succeeds and returns the expected neighbor.
  BeamSearchResult::Statistics quant_query_stats;
  ASSERT_OK_AND_ASSIGN(
      auto quant_query_results,
      searcher->FindNeighbors(
          fixture.quantized_dataset->DataPointView(0),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                            RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
                            /*beam_size=*/10, /*num_neighbors=*/2),
          &quant_query_stats));
  ASSERT_THAT(quant_query_results, Not(IsEmpty()));
  EXPECT_THAT(quant_query_results[0], FieldsAre(FloatNear(-1.0f, 1e-3), 0));
  EXPECT_GT(quant_query_stats.num_reranked, 0);
}

// Verifies that searching with TurboQuant and extra bits reranking works
// with SQUARED_L2 metric.
TEST(GingerSearcherTest, TurboQuantSearchExtraBitsRerankingSquaredL2) {
  const int dimensionality = 64;
  const int num_points = 100;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality);
    for (int d = 0; d < dimensionality; ++d) point[d] = dis(gen);
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  auto config = CreateSearcherConfig(SQUARED_L2);

  // Create with 2 extra bits (quantization_bits = 6).
  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, SQUARED_L2, 123, 6));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  // Query is point 0 itself.
  std::vector<float> query(dataset->get<float>(0).begin(),
                           dataset->get<float>(0).end());

  // SQUARED_L2 rescoring applies max(0, cand.distance) without the +1 offset.
  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                            RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
                            /*beam_size=*/10, /*num_neighbors=*/2),
          &stats));
  ASSERT_THAT(results, Not(IsEmpty()));
  // Because the query IS point 0, distance should be ~0.
  EXPECT_THAT(results[0], FieldsAre(FloatNear(0.0f, 0.1f), 0));
  // All distances must be non-negative.
  EXPECT_THAT(results, Each(FieldsAre(Ge(0.0f), _)));
  EXPECT_GT(stats.num_reranked, 0);
}

std::shared_ptr<SpanDataset> CreateGaussianDataset(int num_points,
                                                   int dimensionality,
                                                   int seed) {
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 4.0f);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality);
    for (float& x : point) x = dist(gen);
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }
  return dataset;
}

// Verifies that float search with extra-bits reranking works without a
// separate rerank dataset, and that it yields the same clamped distances as
// TurboQuant search with extra-bits reranking.
TEST(GingerSearcherTest, FloatSearchExtraBitsRerankingSquaredL2) {
  constexpr int kNumPoints = 32;
  constexpr int kDim = 16;
  auto dataset = CreateGaussianDataset(kNumPoints, kDim, /*seed=*/42);
  GingerSearcherConfig config = CreateSearcherConfig(SQUARED_L2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_turbo_quant()->set_extra_bits(2);
  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());

  const SearchOptions float_options = MakeSearchOptions(
      SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
      /*beam_size=*/kNumPoints, /*num_neighbors=*/kNumPoints);
  const SearchOptions tq_options = MakeSearchOptions(
      SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
      /*beam_size=*/kNumPoints, /*num_neighbors=*/kNumPoints);
  for (GingerNodeId q = 0; q < kNumPoints; ++q) {
    const absl::Span<const float> query = std::as_const(*dataset).get<float>(q);
    BeamSearchResult::Statistics stats;
    ASSERT_OK_AND_ASSIGN(auto results,
                         searcher->FindNeighbors(query, float_options, &stats));
    EXPECT_GT(stats.num_reranked, 0);
    EXPECT_THAT(results, Each(FieldsAre(Ge(0.0f), _)));
    ASSERT_OK_AND_ASSIGN(auto expected,
                         searcher->FindNeighbors(query, tq_options));
    ASSERT_EQ(results.size(), expected.size());
    for (size_t i = 0; i < results.size(); ++i) {
      EXPECT_EQ(results[i].id, expected[i].id);
      EXPECT_FLOAT_EQ(results[i].distance, expected[i].distance);
    }
  }
}

// Verifies that float search with extra-bits reranking from the separate
// full-dimensional rerank dataset (used with PCA) clamps the squared L2
// distances like TurboQuant search does.
TEST(GingerSearcherTest, FloatSearchExtraBitsRerankingWithPcaSquaredL2) {
  ginger::internal::ParallelSchedulerReference scheduler{4};
  constexpr int kNumPoints = 32;
  constexpr int kDim = 16;
  auto dataset = CreateGaussianDataset(kNumPoints, kDim, /*seed=*/42);
  GingerSearcherConfig config = CreateSearcherConfig(SQUARED_L2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_turbo_quant()->set_extra_bits(2);

  // The graph is built without PCA, so both searchers share the same index and
  // their full-dimensional extra-bits reranking must produce identical results.
  ASSERT_OK_AND_ASSIGN(
      auto full_searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());
  config.mutable_pca()->set_dimension(kDim / 2);
  ASSERT_OK_AND_ASSIGN(
      auto pca_searcher,
      GingerSearcherBuilder(config).AddDataset(dataset).Build());

  const SearchOptions float_options = MakeSearchOptions(
      SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
      /*beam_size=*/kNumPoints, /*num_neighbors=*/kNumPoints);
  // Confirms that the PCA searcher reranks with a separate rerank dataset.
  std::vector<int8_t> int8_query(kDim, 1);
  EXPECT_THAT(
      pca_searcher->FindNeighbors(absl::MakeConstSpan(int8_query),
                                  float_options),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Extra bits reranking with a rerank dataset requires "
                         "float query vectors.")));

  const SearchOptions tq_options = MakeSearchOptions(
      SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
      /*beam_size=*/kNumPoints, /*num_neighbors=*/kNumPoints);
  for (GingerNodeId q = 0; q < kNumPoints; ++q) {
    const absl::Span<const float> query = std::as_const(*dataset).get<float>(q);
    BeamSearchResult::Statistics stats;
    ASSERT_OK_AND_ASSIGN(auto results, pca_searcher->FindNeighbors(
                                           query, float_options, &stats));
    EXPECT_GT(stats.num_reranked, 0);
    EXPECT_THAT(results, Each(FieldsAre(Ge(0.0f), _)));
    ASSERT_OK_AND_ASSIGN(auto expected,
                         full_searcher->FindNeighbors(query, tq_options));
    ASSERT_EQ(results.size(), expected.size());
    for (size_t i = 0; i < results.size(); ++i) {
      EXPECT_EQ(results[i].id, expected[i].id);
      // The rerank query is prepared differently than the search query, which
      // causes rounding differences when the distance is close to 0.
      EXPECT_THAT(results[i].distance, FloatNear(expected[i].distance, 1e-3f));
    }
  }
}

// Verifies that searching with TurboQuant and float reranking works
// with SQUARED_L2 metric.
TEST(GingerSearcherTest, TurboQuantSearchFloatRerankingSquaredL2) {
  const int dimensionality = 64;
  const int num_points = 100;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality);
    for (int d = 0; d < dimensionality; ++d) point[d] = dis(gen);
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  auto config = CreateSearcherConfig(SQUARED_L2);

  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, SQUARED_L2, 123));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  // Query is point 0 itself.
  std::vector<float> query(dataset->get<float>(0).begin(),
                           dataset->get<float>(0).end());

  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                            /*beam_size=*/10,
                            /*num_neighbors=*/2),
          &stats));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results[0], FieldsAre(FloatNear(0.0f, 1e-5f), 0));
  EXPECT_THAT(results, Each(FieldsAre(Ge(0.0f), _)));
  EXPECT_GT(stats.num_reranked, 0);
}

// Verifies that `DefaultSearchOptions` picks the search space and reranking
// strategy from the available datasets.
TEST(GingerSearcherTest, DefaultSearchOptions) {
  auto dataset = CreateOneHotDataset(100, 64);
  auto config = CreateSearcherConfig();

  ASSERT_OK_AND_ASSIGN(auto float_searcher,
                       GingerSearcher::Create(config, dataset));
  SearchOptions options = float_searcher->DefaultSearchOptions(
      /*num_neighbors=*/3, /*beam_size=*/20);
  EXPECT_EQ(options.num_neighbors(), 3);
  EXPECT_EQ(options.beam_size(), 20);
  EXPECT_EQ(options.search_space(), SEARCH_SPACE_FLOAT);
  EXPECT_EQ(options.reranking_strategy(), RERANKING_STRATEGY_NONE);
  EXPECT_FALSE(options.has_rerank_factor());

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));
  ASSERT_OK_AND_ASSIGN(
      auto tq_searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());
  options = tq_searcher->DefaultSearchOptions(3, 20);
  EXPECT_EQ(options.search_space(), SEARCH_SPACE_TURBO_QUANT);
  EXPECT_EQ(options.reranking_strategy(), RERANKING_STRATEGY_FLOAT);
  EXPECT_EQ(options.rerank_factor(), 6.0f);

  ASSERT_OK_AND_ASSIGN(
      auto extra_bits_fixture,
      CreateAndInitializeQuantizedDataset(*dataset, DOT_PRODUCT, 123,
                                          /*quantization_bits=*/6));
  ASSERT_OK_AND_ASSIGN(
      auto extra_bits_searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(extra_bits_fixture.quantized_dataset,
                               extra_bits_fixture.quantizer)
          .Build());
  options = extra_bits_searcher->DefaultSearchOptions(3, 20);
  EXPECT_EQ(options.search_space(), SEARCH_SPACE_TURBO_QUANT);
  EXPECT_EQ(options.reranking_strategy(),
            RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS);
  EXPECT_EQ(options.rerank_factor(), 6.0f);
  EXPECT_THAT(extra_bits_searcher->FindNeighbors(
                  std::as_const(*dataset).get<float>(0), options),
              IsOkAndHolds(SizeIs(3)));
}

// Verifies that extra bits reranking fails gracefully when the quantized
// dataset has no extra bits (quantization_bits = 4).
TEST(GingerSearcherTest,
     TurboQuantSearchExtraBitsRerankingFailsWithoutExtraBits) {
  auto dataset = CreateConstantDataset(10, 64, 0.1f);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 5);

  // Default quantization_bits = 4, so bits_per_dim() == 0.
  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(64, 0.1f);
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                            RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
                            /*beam_size=*/5, /*num_neighbors=*/2)),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          "Extra bits reranking requested but quantized dataset has no extra "
          "bits."));
}

// Verifies that extra bits reranking produces more accurate distances than
// plain TurboQuant, confirming the reranking actually happens.
TEST(GingerSearcherTest,
     TurboQuantExtraBitsRerankingIsMoreAccurateThanPlainTQ) {
  const int dimensionality = 32;
  const int num_points = 20;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);

  std::mt19937 gen(123);
  internal::portable_random::NormalDistribution dist;
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality);
    float norm_sq = 0.0f;
    for (int d = 0; d < dimensionality; ++d) {
      point[d] = dist(gen);
      norm_sq += point[d] * point[d];
    }
    float norm = std::sqrt(norm_sq);
    for (int d = 0; d < dimensionality; ++d) point[d] /= norm;
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  auto config = CreateSearcherConfig(DOT_PRODUCT, 16, 123, 20);

  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, DOT_PRODUCT, 123, 7));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  // Queries are slightly perturbed copies of every data point. The searcher
  // reports the negated dot product, so each result is compared with its exact
  // negated dot product. Single neighbors can go either way, so the errors are
  // summed over all queries.
  constexpr int kNumNeighbors = 5;
  float tq_error = 0.0f;
  float extra_error = 0.0f;
  for (int q = 0; q < num_points; ++q) {
    std::vector<float> query_vec(dataset->get<float>(q).begin(),
                                 dataset->get<float>(q).end());
    query_vec[q % dimensionality] += 0.05f;
    float q_norm = 0.0f;
    for (float f : query_vec) q_norm += f * f;
    q_norm = std::sqrt(q_norm);
    for (float& f : query_vec) f /= q_norm;
    absl::Span<const float> query = absl::MakeConstSpan(query_vec);

    ASSERT_OK_AND_ASSIGN(
        auto results_tq,
        searcher->FindNeighbors(
            query,
            MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                              /*beam_size=*/20,
                              /*num_neighbors=*/kNumNeighbors)));
    ASSERT_THAT(results_tq, SizeIs(kNumNeighbors));
    ASSERT_OK_AND_ASSIGN(
        auto results_extra,
        searcher->FindNeighbors(
            query, MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                                     RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS,
                                     /*beam_size=*/20,
                                     /*num_neighbors=*/kNumNeighbors)));
    ASSERT_THAT(results_extra, SizeIs(kNumNeighbors));

    auto exact_error = [&](const DistanceAndId& result) {
      float dot = 0.0f;
      auto point = dataset->get<float>(result.id);
      for (int d = 0; d < dimensionality; ++d) dot += point[d] * query_vec[d];
      return std::abs(result.distance + dot);
    };
    for (int i = 0; i < kNumNeighbors; ++i) {
      tq_error += exact_error(results_tq[i]);
      extra_error += exact_error(results_extra[i]);
    }
  }
  // Extra bits cut the error to about an eighth.
  EXPECT_LT(extra_error, 0.25f * tq_error);
}

// Verifies that `FindNeighbors` correctly projects the query when PCA is
// enabled and returns valid results in TurboQuant search space.
TEST(GingerSearcherTest, FindNeighborsWithPcaAndTurboQuant) {
  ginger::internal::ParallelSchedulerReference scheduler{4};

  std::shared_ptr<SpanDataset> dataset = std::make_shared<SpanDataset>(4, 4);
  std::vector<float> p0 = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {0.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> p2 = {0.5f, 0.5f, 0.0f, 0.0f};
  std::vector<float> p3 = {0.0f, 0.0f, 0.0f, 0.0f};
  dataset->Set<float>(0, absl::MakeConstSpan(p0));
  dataset->Set<float>(1, absl::MakeConstSpan(p1));
  dataset->Set<float>(2, absl::MakeConstSpan(p2));
  dataset->Set<float>(3, absl::MakeConstSpan(p3));

  GingerSearcherConfig config = CreateSearcherConfig(DOT_PRODUCT, 2, 123, 2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_pca()->set_dimension(2);  // Project to 2D

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcher::Create(config, dataset));

  // Query is 4D
  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  SearchOptions options;
  options.set_search_space(SEARCH_SPACE_TURBO_QUANT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);
  options.set_num_neighbors(1);
  options.set_beam_size(2);

  ASSERT_OK_AND_ASSIGN(std::vector<DistanceAndId> res,
                       searcher->FindNeighbors(absl::MakeSpan(query), options));

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].id, 0);  // Point 0 is exactly the query
}

// Verifies that search works when using TurboQuant quantized PCA projection.
TEST(GingerSearcherTest, SearchWithTurboQuantPca) {
  ginger::internal::ParallelSchedulerReference scheduler{4};

  std::shared_ptr<SpanDataset> dataset = std::make_shared<SpanDataset>(4, 4);
  std::vector<float> p0 = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {0.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> p2 = {0.5f, 0.5f, 0.0f, 0.0f};
  std::vector<float> p3 = {0.0f, 0.0f, 0.0f, 0.0f};
  dataset->Set<float>(0, absl::MakeConstSpan(p0));
  dataset->Set<float>(1, absl::MakeConstSpan(p1));
  dataset->Set<float>(2, absl::MakeConstSpan(p2));
  dataset->Set<float>(3, absl::MakeConstSpan(p3));

  GingerSearcherConfig config = CreateSearcherConfig(DOT_PRODUCT, 2, 123, 2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_pca()->set_dimension(2);
  config.mutable_pca()->set_matrix_representation(
      MATRIX_REPRESENTATION_TURBOQUANT);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcher::Create(config, dataset));

  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  SearchOptions options;
  options.set_search_space(SEARCH_SPACE_TURBO_QUANT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);
  options.set_num_neighbors(1);
  options.set_beam_size(2);

  ASSERT_OK_AND_ASSIGN(std::vector<DistanceAndId> res,
                       searcher->FindNeighbors(absl::MakeSpan(query), options));

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].id, 0);
}

// Verifies that pre-quantized queries fail when PCA is enabled.
TEST(GingerSearcherTest, PreQuantizedQueryFailsWithPca) {
  ginger::internal::ParallelSchedulerReference scheduler{4};

  std::shared_ptr<SpanDataset> dataset = std::make_shared<SpanDataset>(4, 4);
  std::vector<float> p0 = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {0.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> p2 = {0.5f, 0.5f, 0.0f, 0.0f};
  std::vector<float> p3 = {0.0f, 0.0f, 0.0f, 0.0f};
  dataset->Set<float>(0, absl::MakeConstSpan(p0));
  dataset->Set<float>(1, absl::MakeConstSpan(p1));
  dataset->Set<float>(2, absl::MakeConstSpan(p2));
  dataset->Set<float>(3, absl::MakeConstSpan(p3));

  GingerSearcherConfig config = CreateSearcherConfig(DOT_PRODUCT, 2, 123, 2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_pca()->set_dimension(2);  // Project to 2D

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcher::Create(config, dataset));

  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  TurboQuantCodec quantizer(4, 123,
                            TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(
      TurboQuantQuery tq_query,
      TurboQuantQuery::Create(quantizer, absl::MakeConstSpan(query)));

  SearchOptions options;
  options.set_search_space(SEARCH_SPACE_TURBO_QUANT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);
  options.set_num_neighbors(1);
  options.set_beam_size(2);

  EXPECT_THAT(
      searcher->FindNeighbors(tq_query, options),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "PCA projection is not supported for pre-quantized queries."));
}

// Verifies that `FindNeighbors` returns an error if no beam size is specified
// in `SearchOptions`.
TEST(GingerSearcherTest, FindNeighborsFailsWithoutBeamSize) {
  std::shared_ptr<SpanDataset> dataset = CreateConstantDataset(10, 4, 0.0f);
  GingerSearcherConfig config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 5);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcher::Create(config, dataset));

  std::vector<float> query(4, 0.0f);
  SearchOptions options;
  options.set_search_space(SEARCH_SPACE_FLOAT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);
  options.set_num_neighbors(2);
  // beam_size is unset (0).
  EXPECT_THAT(searcher->FindNeighbors(absl::MakeConstSpan(query), options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "beam_size must be positive in SearchOptions."));
}

// Verifies that `FindNeighbors` returns an error if no `num_neighbors` is
// specified in `SearchOptions`.
TEST(GingerSearcherTest, FindNeighborsFailsWithoutNumNeighbors) {
  std::shared_ptr<SpanDataset> dataset = CreateConstantDataset(10, 4, 0.0f);
  GingerSearcherConfig config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 5);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcher::Create(config, dataset));

  std::vector<float> query(4, 0.0f);
  SearchOptions options;
  options.set_search_space(SEARCH_SPACE_FLOAT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);
  options.set_beam_size(5);
  // num_neighbors is unset (0).
  EXPECT_THAT(searcher->FindNeighbors(absl::MakeConstSpan(query), options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "num_neighbors must be positive in SearchOptions."));
}

// Verifies that `FindNeighbors` returns an error if a non-positive or
// non-finite `rerank_factor` is specified in `SearchOptions`.
TEST(GingerSearcherTest, FindNeighborsFailsWithInvalidRerankFactor) {
  std::shared_ptr<SpanDataset> dataset = CreateConstantDataset(10, 4, 0.0f);
  GingerSearcherConfig config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 5);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcher::Create(config, dataset));

  std::vector<float> query(4, 0.0f);
  SearchOptions options;
  options.set_search_space(SEARCH_SPACE_FLOAT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);
  options.set_beam_size(5);
  options.set_num_neighbors(2);
  for (float rerank_factor :
       {-1.0f, 0.0f, std::nanf(""), std::numeric_limits<float>::infinity()}) {
    options.set_rerank_factor(rerank_factor);
    EXPECT_THAT(
        searcher->FindNeighbors(absl::MakeConstSpan(query), options),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 "rerank_factor must be positive and finite when set."));
  }
}

// Verifies that `FindNeighbors` rejects queries whose dimensionality does not
// match the searcher.
TEST(GingerSearcherTest, FindNeighborsFailsWithWrongQueryDimensionality) {
  std::shared_ptr<SpanDataset> dataset = CreateConstantDataset(10, 4, 0.0f);
  GingerSearcherConfig config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 5);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcher::Create(config, dataset));

  SearchOptions options;
  options.set_search_space(SEARCH_SPACE_FLOAT);
  options.set_reranking_strategy(RERANKING_STRATEGY_NONE);
  options.set_beam_size(5);
  options.set_num_neighbors(2);
  for (size_t size : {0, 3, 5}) {
    std::vector<float> query(size, 0.0f);
    EXPECT_THAT(searcher->FindNeighbors(absl::MakeConstSpan(query), options),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << "size=" << size;
  }
}

// Verifies that `FindNeighbors` rejects a precomputed TurboQuant query whose
// dimensionality does not match the searcher's codec.
TEST(GingerSearcherTest,
     FindNeighborsFailsWithWrongTurboQuantQueryDimensionality) {
  auto dataset = CreateOneHotDataset(100, 64);
  auto config = CreateSearcherConfig();
  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, DOT_PRODUCT, 123));
  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  TurboQuantCodec other_codec(32, 123,
                              TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(
      TurboQuantQuery tq_query,
      TurboQuantQuery::Create(other_codec, std::vector<float>(32, 1.0f)));
  EXPECT_THAT(searcher->FindNeighbors(
                  tq_query, MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                                              RERANKING_STRATEGY_NONE)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GingerSearcherTest,
     FindNeighborsFailsWhenTurboQuantQueryMissingFloatDataForExtraBits) {
  auto dataset = CreateOneHotDataset(100, 64);
  auto config = CreateSearcherConfig();
  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, DOT_PRODUCT,
                                         /*seed=*/123,
                                         /*quantization_bits=*/6));
  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  ASSERT_OK_AND_ASSIGN(
      TurboQuantQuery no_float_query,
      TurboQuantQuery::Create(*fixture.quantizer, std::vector<float>(64, 1.0f),
                              /*enable_higher_precision_scoring=*/false));
  EXPECT_THAT(searcher->FindNeighbors(
                  no_float_query,
                  MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                                    RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("missing float query data")));
}

// Verifies that Create returns an error if the metric used to build the
// TurboQuant quantizer does not match the metric in the configuration.
TEST(GingerSearcherTest, CreateFailsWithMetricMismatch) {
  int dimensionality = 4;
  auto dataset = CreateOneHotDataset(10, dimensionality);

  auto config = CreateSearcherConfig(SQUARED_L2, 4, 123, 2);

  auto quantizer = std::make_shared<TurboQuantCodec>(
      dimensionality, 123, TurboQuantDistanceMetric::kDotProductDistance);

  auto searcher_or = GingerSearcherBuilder(config)
                         .AddDataset(dataset)
                         .AddQuantizedDataset(nullptr, quantizer)
                         .Build();
  EXPECT_THAT(
      searcher_or.status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Quantizer metric does not match the metric specified in the "
               "config."));
}

// Verifies that GingerSearcher can be built and queried with an empty dataset.
TEST(GingerSearcherTest, EmptyDataset) {
  auto dataset = std::make_shared<SpanDataset>(0, 4);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 2);

  auto searcher_or = GingerSearcher::Create(config, dataset);
  EXPECT_THAT(searcher_or.status(),
              StatusIs(absl::StatusCode::kInvalidArgument, "Empty dataset."));
}

// Verifies searching with SQUARED_L2 distance metric.
TEST(GingerSearcherTest, SquaredL2Search) {
  int dimensionality = 2;
  int num_points = 3;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
  // Point 0: (0, 0)
  // Point 1: (1, 0)
  // Point 2: (0, 2)
  dataset->Set<float>(0, {0.0f, 0.0f});
  dataset->Set<float>(1, {1.0f, 0.0f});
  dataset->Set<float>(2, {0.0f, 2.0f});

  auto config = CreateSearcherConfig(SQUARED_L2, 4, 123, 5);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));

  // Query: (0.1, 0.1)
  // Dist to 0: 0.01 + 0.01 = 0.02
  // Dist to 1: 0.81 + 0.01 = 0.82
  // Dist to 2: 0.01 + 3.61 = 3.62
  std::vector<float> query = {0.1f, 0.1f};
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/5,
                            /*num_neighbors=*/3)));
  EXPECT_THAT(results, ElementsAre(FieldsAre(FloatNear(0.02f, 1e-6), 0),
                                   FieldsAre(FloatNear(0.82f, 1e-6), 1),
                                   FieldsAre(FloatNear(3.62f, 1e-6), 2)));
}

// Verifies that Create fails if there is a dimensionality mismatch between
// the dataset and the quantized dataset.
TEST(GingerSearcherTest, DimensionalityMismatch) {
  int dim1 = 4;
  int dim2 = 8;
  int num_points = 10;
  auto dataset = std::make_shared<SpanDataset>(num_points, dim1);

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);

  ASSERT_OK_AND_ASSIGN(auto fixture, CreateQuantizedDataset(num_points, dim2));

  auto searcher_or =
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build();
  EXPECT_THAT(searcher_or.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("Dimensionality mismatch")));
}

// Verifies that Create fails if there is a size mismatch between
// the dataset and the quantized dataset.
TEST(GingerSearcherTest, SizeMismatch) {
  int dimensionality = 4;
  auto dataset = std::make_shared<SpanDataset>(10, dimensionality);

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateQuantizedDataset(20, dimensionality));

  auto searcher_or =
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build();
  EXPECT_THAT(searcher_or.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("size mismatch")));
}

// Verifies that TurboQuant search fails if no quantized dataset was provided.
TEST(GingerSearcherTest, TurboQuantSearchFailsWithoutDataset) {
  int dimensionality = 4;
  auto dataset = CreateConstantDataset(10, dimensionality, 0.1f);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 2);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));

  std::vector<float> query(dimensionality, 0.1f);

  // 1. Try strategy TurboQuant
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/5,
                            /*num_neighbors=*/2)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               "TurboQuant coordinates not available."));

  // 2. Try strategy TurboQuant with float reranking
  EXPECT_THAT(
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                            /*beam_size=*/5,
                            /*num_neighbors=*/2)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               "TurboQuant coordinates not available."));

  // 3. Try TQ query overload
  TurboQuantCodec quantizer_local(
      dimensionality, 123, TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(
      auto tq_query,
      TurboQuantQuery::Create(quantizer_local, absl::MakeConstSpan(query)));
  EXPECT_THAT(searcher->FindNeighbors(
                  tq_query, MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                                              RERANKING_STRATEGY_NONE,
                                              /*beam_size=*/5,
                                              /*num_neighbors=*/2)),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "TurboQuant coordinates not available."));
}

// Verifies that Create fails if no dataset is provided.
TEST(GingerSearcherTest, CreateFailsWithoutDatasets) {
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(4);
  auto searcher_or = GingerSearcher::Create(config, nullptr);
  EXPECT_THAT(searcher_or.status(), StatusIs(absl::StatusCode::kInvalidArgument,
                                             "No dataset provided."));
}

// Verifies that float search fails if no float dataset was provided.
TEST(GingerSearcherTest, FloatSearchFailsWithoutDataset) {
  int dimensionality = 4;
  auto dataset = CreateConstantDataset(10, dimensionality, 0.1f);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 2);

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(dimensionality, 0.1f);
  EXPECT_THAT(searcher->FindNeighbors(
                  absl::MakeConstSpan(query),
                  MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                                    /*beam_size=*/5, /*num_neighbors=*/2)),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "Float dataset not available."));
}

// Verifies that searching with explicit search options (float vs TurboQuant)
// works as expected and respects the requested strategy.
TEST(GingerSearcherTest, SearchOptionsExplicit) {
  auto dataset = CreateOneHotDataset(20, 64);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 16, 123, 20);

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(64, 0.0f);
  query[7] = 1.0f;

  // Test float search options
  {
    ASSERT_OK_AND_ASSIGN(
        auto results,
        searcher->FindNeighbors(
            absl::MakeConstSpan(query),
            MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                              /*beam_size=*/20,
                              /*num_neighbors=*/1)));
    EXPECT_THAT(results, ElementsAre(FieldsAre(FloatNear(-1.0f, 1e-6), 7)));
  }

  // Test TurboQuant search options
  {
    ASSERT_OK_AND_ASSIGN(
        auto results,
        searcher->FindNeighbors(
            absl::MakeConstSpan(query),
            MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                              /*beam_size=*/20,
                              /*num_neighbors=*/1)));
    EXPECT_THAT(results, ElementsAre(FieldsAre(_, 7)));
  }

  // Test default SearchOptions (should fail because UNSPECIFIED is invalid)
  {
    SearchOptions options;
    options.set_beam_size(20);
    options.set_num_neighbors(1);
    EXPECT_THAT(searcher->FindNeighbors(absl::MakeConstSpan(query), options),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

// Verifies that float search is more accurate than TurboQuant (without
// reranking) and that default SearchOptions correctly selects the strategy
// based on available data.
TEST(GingerSearcherTest, SearchOptionsAccuracy) {
  const int dimensionality = 32;
  const int num_points = 20;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);

  std::mt19937 gen(123);
  std::normal_distribution<float> dist(0.0, 1.0);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality);
    float norm_sq = 0.0f;
    for (int d = 0; d < dimensionality; ++d) {
      point[d] = dist(gen);
      norm_sq += point[d] * point[d];
    }
    float norm = std::sqrt(norm_sq);
    for (int d = 0; d < dimensionality; ++d) point[d] /= norm;
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  auto config = CreateSearcherConfig(DOT_PRODUCT, 16, 123, 20);

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));

  // 1. Searcher with both datasets.
  ASSERT_OK_AND_ASSIGN(
      auto searcher_both,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query_vec(dataset->get<float>(10).begin(),
                               dataset->get<float>(10).end());
  // Jitter the query slightly so it's not identical, making TQ error more
  // likely to exceed 1e-6.
  query_vec[0] += 0.05f;
  float q_norm = 0.0f;
  for (float f : query_vec) q_norm += f * f;
  q_norm = std::sqrt(q_norm);
  for (float& f : query_vec) f /= q_norm;
  absl::Span<const float> query = absl::MakeConstSpan(query_vec);

  float dot = 0.0f;
  auto p10 = dataset->get<float>(10);
  for (int d = 0; d < dimensionality; ++d) dot += p10[d] * query_vec[d];
  float expected_distance = -dot;

  // Float: should have high accuracy (1e-6).
  ASSERT_OK_AND_ASSIGN(
      auto results_float,
      searcher_both->FindNeighbors(
          query, MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                                   /*beam_size=*/20, /*num_neighbors=*/1)));
  EXPECT_THAT(results_float[0].distance, FloatNear(expected_distance, 1e-6));

  // TurboQuant: should have lower accuracy (fails 1e-6).
  ASSERT_OK_AND_ASSIGN(
      auto results_tq,
      searcher_both->FindNeighbors(
          query,
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/20, /*num_neighbors=*/1)));
  EXPECT_THAT(results_tq[0].distance, Not(FloatNear(expected_distance, 1e-6)));

  // TQ with float reranking: should be accurate.
  ASSERT_OK_AND_ASSIGN(
      auto results_rerank,
      searcher_both->FindNeighbors(
          query,
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                            /*beam_size=*/20, /*num_neighbors=*/1)));
  EXPECT_THAT(results_rerank[0].distance, FloatNear(expected_distance, 1e-6));

  // 2. Searcher with ONLY quantized dataset.
  auto quantizer_only = std::make_shared<TurboQuantCodec>(
      dimensionality, 123, TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(
      auto searcher_only,
      GingerSearcherBuilder(config)
          .AddQuantizedDataset(fixture.quantized_dataset, quantizer_only)
          .Build());

  // TQ without reranking: succeeds where float would fail.
  ASSERT_OK_AND_ASSIGN(
      auto results_tq_only,
      searcher_only->FindNeighbors(
          query,
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/20, /*num_neighbors=*/1)));
  EXPECT_THAT(results_tq_only[0].distance,
              Not(FloatNear(expected_distance, 1e-6)));
}

// Verifies FindNeighbors overloads for pre-quantized queries and base point
// views.
TEST(GingerSearcherTest, FindNeighborsOverloads) {
  int dimensionality = 64;
  int num_points = 50;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
  for (int i = 0; i < num_points; ++i) {
    // Denser dataset for more reliable TQ results
    std::vector<float> point(dimensionality, 0.1f);
    point[i % dimensionality] += 1.0f;
    // Simple normalization
    float norm = 0.0f;
    for (float f : point) norm += f * f;
    norm = std::sqrt(norm);
    for (float& f : point) f /= norm;
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  auto config = CreateSearcherConfig(DOT_PRODUCT, 16, 123, 20);

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  // 1. Test TurboQuantQuery overload
  {
    TurboQuantCodec quantizer_local(
        dimensionality, 123, TurboQuantDistanceMetric::kDotProductDistance);
    ASSERT_OK_AND_ASSIGN(
        auto tq_query,
        TurboQuantQuery::Create(quantizer_local, dataset->get<float>(5)));

    ASSERT_OK_AND_ASSIGN(
        auto results,
        searcher->FindNeighbors(
            tq_query,
            MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                              /*beam_size=*/20, /*num_neighbors=*/5)));
    EXPECT_THAT(results, AllOf(SizeIs(Ge(1)), Each(FieldsAre(Le(0.0f), _))));
    EXPECT_EQ(results[0].id, 5);
  }

  // 2. Test TurboQuantBasePointView overload
  {
    auto base_point_view = fixture.quantized_dataset->DataPointView(10);
    BeamSearchResult::Statistics stats;
    ASSERT_OK_AND_ASSIGN(
        auto results,
        searcher->FindNeighbors(
            base_point_view,
            MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                              /*beam_size=*/20,
                              /*num_neighbors=*/5),
            &stats));
    EXPECT_THAT(results, AllOf(SizeIs(Ge(1)), Each(FieldsAre(Le(0.0f), _))));
    EXPECT_EQ(results[0].id, 10);
    EXPECT_GT(stats.num_distance_computations, 0);

    // Test without quantizer
    ASSERT_OK_AND_ASSIGN(auto searcher_no_quantizer,
                         GingerSearcher::Create(config, dataset));
    EXPECT_THAT(
        searcher_no_quantizer->FindNeighbors(
            base_point_view,
            MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                              /*beam_size=*/20, /*num_neighbors=*/5)),
        absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  }
}

// Verifies that requesting float reranking with a pre-quantized query returns
// an error instead of crashing.
TEST(GingerSearcherTest, FloatRerankingWithPreQuantizedQueryReturnsError) {
  int dimensionality = 64;
  int num_points = 50;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality, 0.1f);
    point[i % dimensionality] += 1.0f;
    Eigen::Map<Eigen::VectorXf>(point.data(), point.size()).normalize();
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  auto config = CreateSearcherConfig(DOT_PRODUCT, 16, 123, 20);

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  // Create a pre-quantized query.
  TurboQuantCodec quantizer_local(
      dimensionality, 123, TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(
      auto tq_query,
      TurboQuantQuery::Create(quantizer_local, dataset->get<float>(0)));

  // Float reranking should fail because there's no float query vector.
  EXPECT_THAT(searcher->FindNeighbors(
                  tq_query, MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                                              RERANKING_STRATEGY_FLOAT,
                                              /*beam_size=*/20,
                                              /*num_neighbors=*/5)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Float search space should fail because there's no float query vector.
  EXPECT_THAT(searcher->FindNeighbors(
                  tq_query,
                  MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                                    /*beam_size=*/20,
                                    /*num_neighbors=*/5)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// A more complex test that uses random points and verifies that a query
// very close to a specific point returns that point as the top result.
TEST(GingerSearcherTest, RecallAccuracyRandomPoints) {
  const int dimensionality = 32;
  const int num_points = 1000;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality);
    for (int d = 0; d < dimensionality; ++d) {
      point[d] = dis(gen);
    }
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  // Plant a point at index 500
  std::vector<float> planted_point(dimensionality, 0.5f);
  dataset->Set<float>(500, absl::MakeConstSpan(planted_point));

  auto config = CreateSearcherConfig(SQUARED_L2, 32, 123, 50);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));

  // Query is very close to the planted point
  std::vector<float> query = planted_point;
  query[0] += 0.001f;

  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/100,
                            /*num_neighbors=*/1)));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_EQ(results[0].id, 500);
}

// Verifies that GingerSearcher handles cosine similarity correctly when
// vectors are normalized and DOT_PRODUCT is used.
TEST(GingerSearcherTest, CosineSimilarityWithNormalization) {
  const int dimensionality = 128;
  const int num_points = 100;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);

  auto normalize = [](std::vector<float>& v) {
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 0) {
      for (float& x : v) x /= norm;
    }
  };

  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality, 0.0f);
    point[i % dimensionality] = 1.0f;
    point[(i + 1) % dimensionality] = 0.5f;
    normalize(point);
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  auto config = CreateSearcherConfig(DOT_PRODUCT, 16, 123, 20);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));

  // Query is identical to point 10
  std::vector<float> query(dimensionality, 0.0f);
  query[10 % dimensionality] = 1.0f;
  query[11 % dimensionality] = 0.5f;
  normalize(query);

  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/20,
                            /*num_neighbors=*/5)));
  ASSERT_THAT(results, Not(IsEmpty()));

  EXPECT_EQ(results[0].id, 10);
  EXPECT_NEAR(results[0].distance, -1.0f, 1e-5);
}

// Verifies that a GingerSearcher built with multiple passes works as expected.
TEST(GingerSearcherTest, MultiplePassesBuildAndSearch) {
  auto dataset = CreateOneHotDataset(8, 8);

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(8);
  config.mutable_vamana_config()->set_seed(123);
  // Add multiple passes with increasing beam widths
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(2);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(4);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(8);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));

  std::vector<float> query(8, 0.0f);
  query[3] = 1.0f;

  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/10,
                            /*num_neighbors=*/5)));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_EQ(results[0].id, 3);
  EXPECT_NEAR(results[0].distance, -1.0f, 1e-6);
}

// A test that uses Gaussian random points and verifies that we can find
// neighbors with reasonable accuracy.
TEST(GingerSearcherTest, RecallAccuracyGaussianData) {
  const int dimensionality = 128;
  const int num_points = 1000;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);

  std::mt19937 gen(123);
  std::normal_distribution<float> dist(0.0, 1.0);

  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality);
    float norm_sq = 0.0f;
    for (int d = 0; d < dimensionality; ++d) {
      point[d] = dist(gen);
      norm_sq += point[d] * point[d];
    }
    // Normalize to unit length
    float norm = std::sqrt(norm_sq);
    for (int d = 0; d < dimensionality; ++d) {
      point[d] /= norm;
    }
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  auto config = CreateSearcherConfig(DOT_PRODUCT, 32, 456, 50);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));

  // Query is one of the points in the dataset
  int query_idx = 123;
  absl::Span<const float> query = dataset->get<float>(query_idx);

  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          query, MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                                   /*beam_size=*/100, /*num_neighbors=*/10)));
  ASSERT_THAT(results, Not(IsEmpty()));
  // The closest point should be the point itself.
  EXPECT_EQ(results[0].id, query_idx);
  EXPECT_NEAR(results[0].distance, -1.0f, 1e-5);
}

// Verifies that asking for more neighbors than the beam size or dataset size
// is handled gracefully.
TEST(GingerSearcherTest, FindNeighborsWithLargeK) {
  auto dataset = CreateOneHotDataset(50, 16);
  auto config = CreateSearcherConfig(SQUARED_L2, 8, 123, 10);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));

  std::vector<float> query(16, 0.0f);
  query[0] = 1.1f;

  // Ask for 100 neighbors when dataset only has 50.
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/20,
                            /*num_neighbors=*/100)));
  // Should return at most num_points neighbors.
  EXPECT_LE(results.size(), 50);
  EXPECT_THAT(results, Not(IsEmpty()));
}

// Verifies search with a very small beam size. Recall might be low but it
// should still function.
TEST(GingerSearcherTest, FindNeighborsWithSmallBeamSize) {
  const int dimensionality = 8;
  const int num_points = 20;
  auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);

  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dimensionality, static_cast<float>(i));
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }

  auto config = CreateSearcherConfig(SQUARED_L2, 4, 123, 2);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));

  std::vector<float> query(dimensionality, 5.0f);

  // Search with beam_size = 1.
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/1,
                            /*num_neighbors=*/1)));
  EXPECT_THAT(results, SizeIs(1));
}

// Verifies that search statistics are correctly reported.
TEST(GingerSearcherTest, SearchStatistics) {
  auto dataset = CreateOneHotDataset(100, 4);
  auto config = CreateSearcherConfig();

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateAndInitializeQuantizedDataset(*dataset));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(4, 0.0f);
  query[0] = 1.0f;

  // 1. Basic float search stats
  {
    BeamSearchResult::Statistics stats;
    ASSERT_OK(searcher
                  ->FindNeighbors(absl::MakeConstSpan(query),
                                  MakeSearchOptions(SEARCH_SPACE_FLOAT,
                                                    RERANKING_STRATEGY_NONE,
                                                    /*beam_size=*/10,
                                                    /*num_neighbors=*/5),
                                  &stats)
                  .status());
    EXPECT_GT(stats.num_distance_computations, 0);
    EXPECT_GT(stats.num_nodes_visited, 0);
  }

  // 2. TurboQuant search with reranking stats
  {
    BeamSearchResult::Statistics stats;
    ASSERT_OK_AND_ASSIGN(
        auto results,
        searcher->FindNeighbors(absl::MakeConstSpan(query),
                                MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT,
                                                  RERANKING_STRATEGY_FLOAT,
                                                  /*beam_size=*/10,
                                                  /*num_neighbors=*/5),
                                &stats));
    EXPECT_GT(stats.num_distance_computations, 0);
    EXPECT_GT(stats.num_nodes_visited, 0);
    EXPECT_GT(stats.num_reranked, 0);
  }

  // 3. Optional stats (should not crash if null)
  {
    EXPECT_THAT(
        searcher->FindNeighbors(
            absl::MakeConstSpan(query),
            MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                              /*beam_size=*/10,
                              /*num_neighbors=*/5)),
        IsOk());
  }
}

// Verifies that ties in distance are broken by node id in ascending order.
TEST(GingerSearcherTest, TieBreakingById) {
  auto dataset = CreateConstantDataset(5, 4, 1.0f);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 10);

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(config, dataset));

  std::vector<float> query(4, 1.0f);

  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/10,
                            /*num_neighbors=*/5)));
  ASSERT_THAT(results, SizeIs(5));

  // All distances should be -4.0, and IDs should be 0, 1, 2, 3, 4.
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(results[i].distance, -4.0f, 1e-6);
    EXPECT_EQ(results[i].id, i);
  }
}

TEST(GingerSearcherTest, CreateFailsWithQuantizedDatasetButNoQuantizer) {
  int dimensionality = 64;
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 2);

  ASSERT_OK_AND_ASSIGN(auto fixture,
                       CreateQuantizedDataset(10, dimensionality));

  auto searcher_or =
      GingerSearcherBuilder(config)
          .AddQuantizedDataset(fixture.quantized_dataset, nullptr)
          .Build();
  EXPECT_THAT(
      searcher_or.status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Quantizer must be provided if a quantized dataset is provided."));
}

TEST(GingerSearcherTest, CompressedGraphSearch) {
  auto dataset = CreateOneHotDataset(10, 4);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 5);
  config.set_use_graph_compression(true);

  std::vector<GingerNodeId> reordering(dataset->size());
  std::iota(reordering.begin(), reordering.end(), 0);

  ASSERT_OK_AND_ASSIGN(auto compressed_searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .AddReordering(std::move(reordering))
                           .Build());

  std::vector<float> query(4, 0.0f);
  query[0] = 1.0f;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      compressed_searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                            /*beam_size=*/5,
                            /*num_neighbors=*/3)));
  EXPECT_THAT(results, ElementsAre(FieldsAre(FloatNear(-1.0f, 1e-6), 0),
                                   FieldsAre(FloatNear(-1.0f, 1e-6), 4),
                                   FieldsAre(FloatNear(-1.0f, 1e-6), 8)));
}

// Verifies that Create returns an error if Vamana passes are not configured.
TEST(GingerSearcherTest, CreateFailsWithEmptyVamanaPasses) {
  auto dataset = CreateConstantDataset(10, 4, 0.1f);
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config();  // Create empty config
  auto searcher_or = GingerSearcher::Create(config, dataset);
  EXPECT_THAT(searcher_or.status(), StatusIs(absl::StatusCode::kInvalidArgument,
                                             "passes must not be empty."));
}

// Verifies that a progress callback can be passed to GingerSearcherBuilder and
// is invoked.
TEST(GingerSearcherTest, BuilderInvokesProgressCallback) {
  auto dataset = CreateConstantDataset(10, 4, 0.1f);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 2);

  std::vector<IndexBuildProgress> progress_updates;
  auto progress_callback = [&](const IndexBuildProgress& progress) {
    progress_updates.push_back(progress);
  };

  ASSERT_OK_AND_ASSIGN(auto searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(dataset)
                           .SetProgressCallback(progress_callback)
                           .Build());
  EXPECT_GT(progress_updates.size(), 0);
  EXPECT_EQ(progress_updates.back().total_completed_points, 10);
}

// Verifies that a progress callback can be passed to GingerSearcher::Create and
// is invoked.
TEST(GingerSearcherTest, CreateInvokesProgressCallback) {
  auto dataset = CreateConstantDataset(10, 4, 0.1f);
  auto config = CreateSearcherConfig(DOT_PRODUCT, 4, 123, 2);

  std::vector<IndexBuildProgress> progress_updates;
  auto progress_callback = [&](const IndexBuildProgress& progress) {
    progress_updates.push_back(progress);
  };

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcher::Create(
                                          config, dataset, progress_callback));
  EXPECT_GT(progress_updates.size(), 0);
  EXPECT_EQ(progress_updates.back().total_completed_points, 10);
}

TEST(GingerSearcherTest, FindNeighborsDoesNotLimitReranking) {
  auto dataset = CreateOneHotDataset(100, 64);
  auto config = CreateSearcherConfig();

  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, DOT_PRODUCT, 123));
  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  // Search with kTurboQuantWithFloatReranking and beam_size=10.
  // As we do not set rerank_factor, we rerank all candidates returned by beam
  // search, so we expect num_reranked to be equal to the beam size.
  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(
          absl::MakeConstSpan(query),
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                            /*beam_size=*/10,
                            /*num_neighbors=*/2),
          &stats));

  EXPECT_THAT(results, SizeIs(2));
  EXPECT_EQ(stats.num_reranked, 10);
}

TEST(GingerSearcherTest, FindNeighborsRespectsRerankFactor) {
  auto dataset = CreateOneHotDataset(100, 64);
  auto config = CreateSearcherConfig();

  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, DOT_PRODUCT, 123));
  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT);
  options.set_beam_size(10);
  options.set_num_neighbors(2);
  options.set_rerank_factor(2.0f);

  // Search with kTurboQuantWithFloatReranking and beam_size=10.
  // As we set rerank_factor=2.0f and num_neighbors=2, we should rerank only 4
  // candidates.
  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(absl::MakeConstSpan(query), options, &stats));

  EXPECT_THAT(results, SizeIs(2));
  EXPECT_EQ(stats.num_reranked, 4);
}

// A rerank factor below 1 still reranks at least `num_neighbors` candidates.
TEST(GingerSearcherTest, FindNeighborsRerankFactorBelowOne) {
  auto dataset = CreateOneHotDataset(100, 64);
  auto config = CreateSearcherConfig();

  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, DOT_PRODUCT, 123));
  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT);
  options.set_beam_size(10);
  options.set_num_neighbors(4);
  options.set_rerank_factor(0.5f);

  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher->FindNeighbors(absl::MakeConstSpan(query), options, &stats));

  EXPECT_THAT(results, SizeIs(4));
  EXPECT_EQ(stats.num_reranked, 4);
}

TEST(GingerSearcherTest, SearchOptionsBeamSizeAndRerankFactorTest) {
  std::shared_ptr<SpanDataset> dataset =
      CreateOneHotDataset(/*num_points=*/100, /*dimensionality=*/64);
  GingerSearcherConfig config = CreateSearcherConfig();
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_vamana_config()->set_max_degree(16);
  config.mutable_vamana_config()->add_passes()->set_max_beam_width(100);

  ASSERT_OK_AND_ASSIGN(
      QuantizedDatasetFixture fixture,
      CreateAndInitializeQuantizedDataset(*dataset, DOT_PRODUCT, 123));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT);
  options.set_beam_size(10);
  options.set_num_neighbors(2);
  options.set_rerank_factor(2.0f);

  BeamSearchResult::Statistics stats;
  EXPECT_THAT(
      searcher->FindNeighbors(absl::MakeConstSpan(query), options, &stats),
      IsOkAndHolds(SizeIs(2)));
  EXPECT_EQ(stats.num_reranked, 4);
}

class TurboSliceCandidateFilterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    InitFixture(CreateOneHotDataset(100, 64), /*turboslice_dim=*/16);
  }

  void InitFixture(std::shared_ptr<SpanDataset> dataset, int turboslice_dim) {
    dataset_ = std::move(dataset);
    config_ = CreateSearcherConfig();
    ASSERT_OK_AND_ASSIGN(auto fixture,
                         CreateAndInitializeQuantizedDataset(
                             *dataset_, DOT_PRODUCT, 123, 0, turboslice_dim));
    quantizer_ = fixture.quantizer;
    quantized_dataset_ = fixture.quantized_dataset;
    ASSERT_OK_AND_ASSIGN(
        searcher_, GingerSearcherBuilder(config_)
                       .AddDataset(dataset_)
                       .AddQuantizedDataset(quantized_dataset_, quantizer_)
                       .Build());
  }

  void InitFixture(int num_points, int dimensionality, int turboslice_dim) {
    InitFixture(CreateOneHotDataset(num_points, dimensionality),
                turboslice_dim);
  }

  std::shared_ptr<SpanDataset> dataset_;
  GingerSearcherConfig config_;
  std::shared_ptr<TurboQuantCodec> quantizer_;
  std::shared_ptr<TurboQuantDatasetSoA> quantized_dataset_;
  std::unique_ptr<GingerSearcher> searcher_;
};

TEST_F(TurboSliceCandidateFilterTest, SearchSucceeds) {
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_turbo_slice();

  ASSERT_OK_AND_ASSIGN(auto results, searcher_->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results[0], FieldsAre(FloatNear(-1.0f, 0.1f), 0));
}

TEST_F(TurboSliceCandidateFilterTest, SearchWithPreQuantizedQuery) {
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  ASSERT_OK_AND_ASSIGN(auto tq_query,
                       TurboQuantQuery::Create(*quantizer_, query));

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_turbo_slice();

  ASSERT_OK_AND_ASSIGN(auto results,
                       searcher_->FindNeighbors(tq_query, options));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results[0], FieldsAre(FloatNear(-1.0f, 0.1f), 0));
}

TEST_F(TurboSliceCandidateFilterTest, FailsWhenTurboSliceDisabled) {
  InitFixture(/*num_points=*/100, /*dimensionality=*/64, /*turboslice_dim=*/0);
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_turbo_slice();

  EXPECT_THAT(searcher_->FindNeighbors(absl::MakeConstSpan(query), options),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(TurboSliceCandidateFilterTest, CandidateFilterOnFloatSearchFails) {
  ASSERT_OK_AND_ASSIGN(auto float_searcher,
                       GingerSearcher::Create(config_, dataset_));

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_turbo_slice();

  EXPECT_THAT(
      float_searcher->FindNeighbors(absl::MakeConstSpan(query), options),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TurboSliceCandidateFilterTest,
       CandidateFilterWithoutTurboSliceSucceeds) {
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  // Has candidate_filter, but turbo_slice is NOT set.
  options.mutable_candidate_filter();

  ASSERT_OK_AND_ASSIGN(auto results, searcher_->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results[0], FieldsAre(FloatNear(-1.0f, 0.1f), 0));
}

TEST_F(TurboSliceCandidateFilterTest, CustomBeamMeanWidth) {
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_turbo_slice();
  options.mutable_candidate_filter()->set_beam_mean_width(10);

  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher_->FindNeighbors(absl::MakeConstSpan(query), options, &stats));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results[0], FieldsAre(FloatNear(-1.0f, 0.1f), 0));
  EXPECT_GT(stats.num_distance_computations, 0);

  options.mutable_candidate_filter()->set_beam_mean_width(1);
  EXPECT_THAT(
      searcher_->FindNeighbors(absl::MakeConstSpan(query), options, &stats),
      absl_testing::IsOk());
}

TEST_F(TurboSliceCandidateFilterTest,
       TurboQuantScoringOverridesTurboSliceScores) {
  InitFixture(CreateNormalizedRandomDataset(50, 64), /*turboslice_dim=*/16);
  auto query_dataset = CreateNormalizedRandomDataset(1, 64, /*seed=*/99);
  auto query = query_dataset->get<float>(0);

  SearchOptions options_ts =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/50, /*num_neighbors=*/5);
  options_ts.mutable_candidate_filter()->mutable_turbo_slice();

  SearchOptions options_no_filter =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/50, /*num_neighbors=*/5);

  ASSERT_OK_AND_ASSIGN(auto results_ts,
                       searcher_->FindNeighbors(query, options_ts));
  ASSERT_OK_AND_ASSIGN(auto results_no_filter,
                       searcher_->FindNeighbors(query, options_no_filter));

  ASSERT_THAT(results_ts, Not(IsEmpty()));
  ASSERT_THAT(results_no_filter, Not(IsEmpty()));

  EXPECT_EQ(results_ts[0].id, results_no_filter[0].id);
  EXPECT_FLOAT_EQ(results_ts[0].distance, results_no_filter[0].distance);
}

class JlCandidateFilterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    InitFixture(CreateOneHotDataset(100, 64), /*jl_dim=*/128);
  }

  void InitFixture(std::shared_ptr<SpanDataset> dataset, int jl_dim,
                   DistanceMetric metric = DOT_PRODUCT,
                   bool use_graph_compression = false) {
    dataset_ = std::move(dataset);
    config_ = CreateSearcherConfig(metric);
    if (use_graph_compression) {
      config_.set_use_graph_compression(true);
      config_.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);
    }
    ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                           *dataset_, metric, 123, 0, 0));
    quantizer_ = fixture.quantizer;
    quantized_dataset_ = fixture.quantized_dataset;
    ASSERT_OK_AND_ASSIGN(jl_dataset_,
                         JlDataset::Create(jl_dim, 42, metric, *dataset_));
    GingerSearcherBuilder builder(config_);
    builder.AddDataset(dataset_);
    builder.AddQuantizedDataset(quantized_dataset_, quantizer_);
    builder.AddJlDataset(jl_dataset_);
    ASSERT_OK_AND_ASSIGN(searcher_, builder.Build());
  }

  std::shared_ptr<SpanDataset> dataset_;
  GingerSearcherConfig config_;
  std::shared_ptr<TurboQuantCodec> quantizer_;
  std::shared_ptr<TurboQuantDatasetSoA> quantized_dataset_;
  std::shared_ptr<JlDataset> jl_dataset_;
  std::unique_ptr<GingerSearcher> searcher_;
};

TEST_F(JlCandidateFilterTest, SearchSucceedsWithTurboQuant) {
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();

  ASSERT_OK_AND_ASSIGN(auto results, searcher_->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results,
              UnorderedElementsAre(FieldsAre(FloatNear(-1.0f, 0.1f), 0),
                                   FieldsAre(FloatNear(-1.0f, 0.1f), 64)));
}

TEST_F(JlCandidateFilterTest, SearchSucceedsWithSquaredL2) {
  InitFixture(CreateOneHotDataset(100, 64), /*jl_dim=*/128, SQUARED_L2);
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();

  ASSERT_OK_AND_ASSIGN(auto results, searcher_->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results,
              UnorderedElementsAre(FieldsAre(FloatNear(0.0f, 0.1f), 0),
                                   FieldsAre(FloatNear(0.0f, 0.1f), 64)));
}

TEST_F(JlCandidateFilterTest, SearchSucceedsWithGraphCompression) {
  InitFixture(CreateOneHotDataset(100, 64), /*jl_dim=*/128, DOT_PRODUCT,
              /*use_graph_compression=*/true);
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();

  ASSERT_OK_AND_ASSIGN(auto results, searcher_->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results,
              UnorderedElementsAre(FieldsAre(FloatNear(-1.0f, 0.1f), 0),
                                   FieldsAre(FloatNear(-1.0f, 0.1f), 64)));
}

TEST_F(JlCandidateFilterTest, SearchSucceedsWithFloatSearchSpace) {
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();

  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher_->FindNeighbors(absl::MakeConstSpan(query), options, &stats));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results,
              UnorderedElementsAre(FieldsAre(FloatNear(-1.0f, 0.1f), 0),
                                   FieldsAre(FloatNear(-1.0f, 0.1f), 64)));
  EXPECT_GT(stats.num_distance_computations, 0);
}

TEST_F(JlCandidateFilterTest, SearchSucceedsWithFloatSearchSpaceSquaredL2) {
  InitFixture(CreateOneHotDataset(100, 64), /*jl_dim=*/128, SQUARED_L2);
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();

  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher_->FindNeighbors(absl::MakeConstSpan(query), options, &stats));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results,
              UnorderedElementsAre(FieldsAre(FloatNear(0.0f, 0.1f), 0),
                                   FieldsAre(FloatNear(0.0f, 0.1f), 64)));
  EXPECT_GT(stats.num_distance_computations, 0);
}

TEST_F(JlCandidateFilterTest,
       SearchSucceedsWithFloatSearchSpaceWithoutTurboQuant) {
  GingerSearcherBuilder builder(config_);
  builder.AddDataset(dataset_);
  builder.AddJlDataset(jl_dataset_);
  ASSERT_OK_AND_ASSIGN(auto searcher_no_tq, builder.Build());

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();

  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(auto results,
                       searcher_no_tq->FindNeighbors(absl::MakeConstSpan(query),
                                                     options, &stats));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results,
              UnorderedElementsAre(FieldsAre(FloatNear(-1.0f, 0.1f), 0),
                                   FieldsAre(FloatNear(-1.0f, 0.1f), 64)));
  EXPECT_GT(stats.num_distance_computations, 0);
}

TEST_F(JlCandidateFilterTest, FailsWhenJlDatasetNotConfigured) {
  ASSERT_OK_AND_ASSIGN(auto no_jl_searcher,
                       GingerSearcherBuilder(config_)
                           .AddDataset(dataset_)
                           .AddQuantizedDataset(quantized_dataset_, quantizer_)
                           .Build());

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();

  EXPECT_THAT(
      no_jl_searcher->FindNeighbors(absl::MakeConstSpan(query), options),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(JlCandidateFilterTest, FailsWithNegativeBeamMeanWidth) {
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();
  options.mutable_candidate_filter()->set_beam_mean_width(-1);

  EXPECT_THAT(searcher_->FindNeighbors(absl::MakeConstSpan(query), options),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(JlCandidateFilterTest, FailsWithPreQuantizedQuery) {
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  ASSERT_OK_AND_ASSIGN(auto tq_query,
                       TurboQuantQuery::Create(*quantizer_, query));

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();

  EXPECT_THAT(searcher_->FindNeighbors(tq_query, options),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(JlCandidateFilterTest, CustomBeamMeanWidth) {
  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/10, /*num_neighbors=*/2);
  options.mutable_candidate_filter()->mutable_jl();
  options.mutable_candidate_filter()->set_beam_mean_width(10);

  BeamSearchResult::Statistics stats;
  ASSERT_OK_AND_ASSIGN(
      auto results,
      searcher_->FindNeighbors(absl::MakeConstSpan(query), options, &stats));
  ASSERT_THAT(results, Not(IsEmpty()));
  EXPECT_THAT(results[0], FieldsAre(FloatNear(-1.0f, 0.1f), 0));
  EXPECT_GT(stats.num_distance_computations, 0);

  options.mutable_candidate_filter()->set_beam_mean_width(1);
  EXPECT_THAT(
      searcher_->FindNeighbors(absl::MakeConstSpan(query), options, &stats),
      absl_testing::IsOk());
}

TEST_F(JlCandidateFilterTest, ScoringOverridesJlScores) {
  auto dataset = CreateNormalizedRandomDataset(50, 64);
  ASSERT_OK_AND_ASSIGN(auto jl_dataset,
                       JlDataset::Create(128, 42, DOT_PRODUCT, *dataset));
  std::shared_ptr<const JlDataset> jl_ptr = std::move(jl_dataset);
  ASSERT_OK_AND_ASSIGN(auto fixture, CreateAndInitializeQuantizedDataset(
                                         *dataset, DOT_PRODUCT, 123, 0, 0));
  GingerSearcherBuilder builder(config_);
  builder.AddDataset(dataset);
  builder.AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer);
  builder.AddJlDataset(jl_ptr);
  ASSERT_OK_AND_ASSIGN(auto searcher, builder.Build());

  auto query_dataset = CreateNormalizedRandomDataset(1, 64, /*seed=*/99);
  auto query = query_dataset->get<float>(0);

  SearchOptions options_jl =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/50, /*num_neighbors=*/5);
  options_jl.mutable_candidate_filter()->mutable_jl();

  SearchOptions options_no_filter =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE,
                        /*beam_size=*/50, /*num_neighbors=*/5);

  ASSERT_OK_AND_ASSIGN(auto results_jl,
                       searcher->FindNeighbors(query, options_jl));
  ASSERT_OK_AND_ASSIGN(auto results_no_filter,
                       searcher->FindNeighbors(query, options_no_filter));

  ASSERT_THAT(results_jl, Not(IsEmpty()));
  ASSERT_THAT(results_no_filter, Not(IsEmpty()));

  EXPECT_EQ(results_jl[0].id, results_no_filter[0].id);
  EXPECT_FLOAT_EQ(results_jl[0].distance, results_no_filter[0].distance);
}

TEST(GingerSearcherTest, SquaredL2NormalizedDatapointsSearch) {
  std::shared_ptr<SpanDataset> dataset =
      CreateOneHotDataset(/*num_points=*/20, /*dimensionality=*/64);
  GingerSearcherConfig config = CreateSearcherConfig(SQUARED_L2);
  config.set_normalized_datapoints(true);
  config.mutable_turbo_quant();

  ASSERT_OK_AND_ASSIGN(
      QuantizedDatasetFixture fixture,
      CreateAndInitializeQuantizedDataset(*dataset, SQUARED_L2, 123,
                                          /*quantization_bits=*/0,
                                          /*turboslice_dim=*/0,
                                          /*normalized_datapoints=*/true));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GingerSearcher> searcher,
      GingerSearcherBuilder(config)
          .AddDataset(dataset)
          .AddQuantizedDataset(fixture.quantized_dataset, fixture.quantizer)
          .Build());

  std::vector<float> query(64, 0.0f);
  query[0] = 1.0f;

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_FLOAT);
  options.set_beam_size(10);
  options.set_num_neighbors(5);

  ASSERT_OK_AND_ASSIGN(
      auto neighbors,
      searcher->FindNeighbors(absl::MakeConstSpan(query), options));
  EXPECT_THAT(neighbors, SizeIs(5));
  for (const auto& neighbor : neighbors) {
    EXPECT_GE(neighbor.distance, 0.0f);
  }
}

TEST(GingerSearcherTest, Int8SearchSquaredL2WithInt8Query) {
  constexpr size_t kDimensionality = 4;
  constexpr GingerNodeId kNumPoints = 5;

  auto dataset = std::make_shared<SpanDataset>(kNumPoints, kDimensionality,
                                               CoordinateType::kInt8);
  dataset->Set<int8_t>(0, {10, 0, 0, 0});
  dataset->Set<int8_t>(1, {0, 10, 0, 0});
  dataset->Set<int8_t>(2, {0, 0, 10, 0});
  dataset->Set<int8_t>(3, {0, 0, 0, 10});
  dataset->Set<int8_t>(4, {6, 8, 0, 0});

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcher::Create(CreateSearcherConfig(SQUARED_L2), dataset));

  std::vector<int8_t> query = {10, 0, 0, 0};
  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE);
  options.set_beam_size(5);
  options.set_num_neighbors(2);

  ASSERT_OK_AND_ASSIGN(auto results, searcher->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, SizeIs(2));
  EXPECT_EQ(results[0].id, 0);
  EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
}

TEST(GingerSearcherTest, Int8SearchDotProductWithInt8Query) {
  constexpr size_t kDimensionality = 4;
  constexpr GingerNodeId kNumPoints = 5;

  auto dataset = std::make_shared<SpanDataset>(kNumPoints, kDimensionality,
                                               CoordinateType::kInt8);
  dataset->Set<int8_t>(0, {10, 0, 0, 0});
  dataset->Set<int8_t>(1, {0, 10, 0, 0});
  dataset->Set<int8_t>(2, {0, 0, 10, 0});
  dataset->Set<int8_t>(3, {0, 0, 0, 10});
  dataset->Set<int8_t>(4, {6, 8, 0, 0});

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcher::Create(CreateSearcherConfig(DOT_PRODUCT), dataset));

  std::vector<int8_t> query = {1, 0, 0, 0};
  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE);
  options.set_beam_size(5);
  options.set_num_neighbors(2);

  ASSERT_OK_AND_ASSIGN(auto results, searcher->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, SizeIs(2));
  EXPECT_EQ(results[0].id, 0);
  EXPECT_FLOAT_EQ(results[0].distance, -10.0f);
}

TEST(GingerSearcherTest, ByteSearchRerankingFails) {
  constexpr size_t kDimensionality = 4;
  constexpr GingerNodeId kNumPoints = 5;

  auto dataset = std::make_shared<SpanDataset>(kNumPoints, kDimensionality,
                                               CoordinateType::kInt8);
  dataset->Set<int8_t>(0, {10, 0, 0, 0});
  dataset->Set<int8_t>(1, {0, 10, 0, 0});
  dataset->Set<int8_t>(2, {0, 0, 10, 0});
  dataset->Set<int8_t>(3, {0, 0, 0, 10});
  dataset->Set<int8_t>(4, {6, 8, 0, 0});

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcher::Create(CreateSearcherConfig(SQUARED_L2), dataset));

  std::vector<int8_t> query = {6, 8, 0, 0};
  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_FLOAT);
  options.set_beam_size(5);
  options.set_num_neighbors(2);

  EXPECT_THAT(searcher->FindNeighbors(absl::MakeConstSpan(query), options),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Reranking requires float dataset.")));
}

TEST(GingerSearcherTest, Uint8SearchSquaredL2WithUint8Query) {
  constexpr size_t kDimensionality = 4;
  constexpr GingerNodeId kNumPoints = 5;

  auto dataset = std::make_shared<SpanDataset>(kNumPoints, kDimensionality,
                                               CoordinateType::kUint8);
  dataset->Set<uint8_t>(0, {10, 0, 0, 0});
  dataset->Set<uint8_t>(1, {0, 10, 0, 0});
  dataset->Set<uint8_t>(2, {0, 0, 10, 0});
  dataset->Set<uint8_t>(3, {0, 0, 0, 10});
  dataset->Set<uint8_t>(4, {6, 8, 0, 0});

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcher::Create(CreateSearcherConfig(SQUARED_L2), dataset));

  std::vector<uint8_t> query = {10, 0, 0, 0};
  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE);
  options.set_beam_size(5);
  options.set_num_neighbors(2);

  ASSERT_OK_AND_ASSIGN(auto results, searcher->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, SizeIs(2));
  EXPECT_EQ(results[0].id, 0);
  EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
}

TEST(GingerSearcherTest, Uint8SearchDotProductWithUint8Query) {
  constexpr size_t kDimensionality = 4;
  constexpr GingerNodeId kNumPoints = 5;

  auto dataset = std::make_shared<SpanDataset>(kNumPoints, kDimensionality,
                                               CoordinateType::kUint8);
  dataset->Set<uint8_t>(0, {10, 0, 0, 0});
  dataset->Set<uint8_t>(1, {0, 10, 0, 0});
  dataset->Set<uint8_t>(2, {0, 0, 10, 0});
  dataset->Set<uint8_t>(3, {0, 0, 0, 10});
  dataset->Set<uint8_t>(4, {6, 8, 0, 0});

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcher::Create(CreateSearcherConfig(DOT_PRODUCT), dataset));

  std::vector<uint8_t> query = {1, 0, 0, 0};
  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE);
  options.set_beam_size(5);
  options.set_num_neighbors(2);

  ASSERT_OK_AND_ASSIGN(auto results, searcher->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, SizeIs(2));
  EXPECT_EQ(results[0].id, 0);
  EXPECT_FLOAT_EQ(results[0].distance, -10.0f);
}

TEST(GingerSearcherTest, ByteSearchErrorValidation) {
  constexpr size_t kDimensionality = 4;
  constexpr GingerNodeId kNumPoints = 5;

  auto int8_dataset = std::make_shared<SpanDataset>(kNumPoints, kDimensionality,
                                                    CoordinateType::kInt8);
  for (int i = 0; i < kNumPoints; ++i) {
    int8_dataset->Set<int8_t>(i, {1, 2, 3, 4});
  }

  ASSERT_OK_AND_ASSIGN(
      auto int8_searcher,
      GingerSearcher::Create(CreateSearcherConfig(SQUARED_L2), int8_dataset));

  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_FLOAT, RERANKING_STRATEGY_NONE);

  // Float query on int8 dataset
  std::vector<float> float_query = {1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_THAT(
      int8_searcher->FindNeighbors(absl::MakeConstSpan(float_query), options),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("float query is only supported for float datasets.")));

  // Uint8 query on int8 dataset
  std::vector<uint8_t> uint8_query = {1, 2, 3, 4};
  EXPECT_THAT(
      int8_searcher->FindNeighbors(absl::MakeConstSpan(uint8_query), options),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // TurboQuant search space requested on byte dataset without quantizer
  SearchOptions tq_options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE);
  std::vector<int8_t> int8_query = {1, 2, 3, 4};
  EXPECT_THAT(
      int8_searcher->FindNeighbors(absl::MakeConstSpan(int8_query), tq_options),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("TurboQuant coordinates not available.")));

  auto uint8_dataset = std::make_shared<SpanDataset>(
      kNumPoints, kDimensionality, CoordinateType::kUint8);
  for (int i = 0; i < kNumPoints; ++i) {
    uint8_dataset->Set<uint8_t>(i, {1, 2, 3, 4});
  }

  ASSERT_OK_AND_ASSIGN(
      auto uint8_searcher,
      GingerSearcher::Create(CreateSearcherConfig(SQUARED_L2), uint8_dataset));

  // Float query on uint8 dataset
  EXPECT_THAT(
      uint8_searcher->FindNeighbors(absl::MakeConstSpan(float_query), options),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("float query is only supported for float datasets.")));
}

TEST(GingerSearcherTest, Int8SearchTurboQuantWithInt8Query) {
  constexpr size_t kDimensionality = 4;
  constexpr GingerNodeId kNumPoints = 5;

  auto dataset = std::make_shared<SpanDataset>(kNumPoints, kDimensionality,
                                               CoordinateType::kInt8);
  dataset->Set<int8_t>(0, {10, 0, 0, 0});
  dataset->Set<int8_t>(1, {0, 10, 0, 0});
  dataset->Set<int8_t>(2, {0, 0, 10, 0});
  dataset->Set<int8_t>(3, {0, 0, 0, 10});
  dataset->Set<int8_t>(4, {6, 8, 0, 0});

  GingerSearcherConfig config = CreateSearcherConfig(SQUARED_L2);
  config.mutable_turbo_quant()->set_seed(123);
  config.mutable_turbo_quant()->set_extra_bits(2);

  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());

  std::vector<int8_t> query = {10, 0, 0, 0};
  SearchOptions options = MakeSearchOptions(
      SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS);
  options.set_beam_size(5);
  options.set_num_neighbors(2);

  ASSERT_OK_AND_ASSIGN(auto results, searcher->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, SizeIs(2));
  EXPECT_EQ(results[0].id, 0);
}

TEST(GingerSearcherTest, Uint8SearchTurboQuantWithUint8Query) {
  constexpr size_t kDimensionality = 4;
  constexpr GingerNodeId kNumPoints = 5;

  auto dataset = std::make_shared<SpanDataset>(kNumPoints, kDimensionality,
                                               CoordinateType::kUint8);
  dataset->Set<uint8_t>(0, {10, 0, 0, 0});
  dataset->Set<uint8_t>(1, {0, 10, 0, 0});
  dataset->Set<uint8_t>(2, {0, 0, 10, 0});
  dataset->Set<uint8_t>(3, {0, 0, 0, 10});
  dataset->Set<uint8_t>(4, {6, 8, 0, 0});

  GingerSearcherConfig config = CreateSearcherConfig(DOT_PRODUCT);
  config.mutable_turbo_quant()->set_seed(123);

  ASSERT_OK_AND_ASSIGN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());

  std::vector<uint8_t> query = {10, 0, 0, 0};
  SearchOptions options =
      MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, RERANKING_STRATEGY_NONE);
  options.set_beam_size(5);
  options.set_num_neighbors(2);

  ASSERT_OK_AND_ASSIGN(auto results, searcher->FindNeighbors(
                                         absl::MakeConstSpan(query), options));
  ASSERT_THAT(results, SizeIs(2));
  EXPECT_EQ(results[0].id, 0);
}

TEST(GingerSearcherTest, ThreeBitPrefixWithExtraBitsMatchesSeparateArray) {
  constexpr int kDim = 128;
  constexpr int kNumPoints = 500;
  constexpr int kNumQueries = 20;
  constexpr int kNumNeighbors = 10;
  auto make_points = [](int n, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist;
    std::vector<std::vector<float>> points(n, std::vector<float>(kDim));
    for (auto& point : points) {
      for (float& x : point) x = dist(rng);
    }
    return points;
  };
  const auto points = make_points(kNumPoints, /*seed=*/1);
  const auto queries = make_points(kNumQueries, /*seed=*/2);
  auto dataset = std::make_shared<SpanDataset>(kNumPoints, kDim);
  for (int i = 0; i < kNumPoints; ++i) {
    dataset->Set<float>(i, absl::MakeConstSpan(points[i]));
  }
  auto build = [&](bool allow_three_bit_prefix) {
    GingerSearcherConfig config =
        CreateSearcherConfig(DOT_PRODUCT, /*max_degree=*/16, /*seed=*/123,
                             /*max_beam_width=*/32);
    config.mutable_turbo_quant()->set_seed(123);
    config.mutable_turbo_quant()->set_extra_bits(2);
    config.mutable_turbo_quant()->set_allow_three_bit_prefix(
        allow_three_bit_prefix);
    // Reorders the quantized dataset (with extra bits) in place.
    config.set_use_graph_compression(true);
    config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);
    return GingerSearcherBuilder(config).AddDataset(dataset).Build();
  };
  ASSERT_OK_AND_ASSIGN(auto prefix_searcher,
                       build(/*allow_three_bit_prefix=*/true));
  ASSERT_OK_AND_ASSIGN(auto separate_searcher,
                       build(/*allow_three_bit_prefix=*/false));

  // 4+k reranking is bit-exact, while the 4-bit fast pass differs because it
  // uses the 3-bit prefix.
  int num_hits = 0;
  bool fast_pass_differs = false;
  for (const auto& query : queries) {
    for (RerankingStrategy reranking :
         {RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS, RERANKING_STRATEGY_NONE}) {
      const SearchOptions options =
          MakeSearchOptions(SEARCH_SPACE_TURBO_QUANT, reranking,
                            /*beam_size=*/64, kNumNeighbors);
      ASSERT_OK_AND_ASSIGN(auto prefix_results,
                           prefix_searcher->FindNeighbors(query, options));
      ASSERT_OK_AND_ASSIGN(auto separate_results,
                           separate_searcher->FindNeighbors(query, options));
      for (const DistanceAndId& p : prefix_results) {
        for (const DistanceAndId& s : separate_results) {
          if (p.id != s.id) continue;
          if (reranking == RERANKING_STRATEGY_NONE) {
            fast_pass_differs |= p.distance != s.distance;
          } else {
            EXPECT_EQ(p.distance, s.distance) << "id " << p.id;
          }
        }
      }
      if (reranking == RERANKING_STRATEGY_NONE) continue;

      // Brute-force top-k for recall.
      std::vector<std::pair<float, GingerNodeId>> scored;
      for (GingerNodeId i = 0; i < points.size(); ++i) {
        scored.emplace_back(-std::inner_product(query.begin(), query.end(),
                                                points[i].begin(), 0.0f),
                            i);
      }
      absl::c_partial_sort(scored, scored.begin() + kNumNeighbors);
      for (const DistanceAndId& p : prefix_results) {
        num_hits += absl::c_any_of(
            absl::MakeConstSpan(scored).first(kNumNeighbors),
            [&](const auto& truth) { return truth.second == p.id; });
      }
    }
  }
  EXPECT_GE(num_hits, 0.8 * kNumNeighbors * queries.size());
  EXPECT_TRUE(fast_pass_differs);
}

}  // namespace
}  // namespace ginger
