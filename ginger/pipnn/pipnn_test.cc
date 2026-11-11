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

#include "ginger/pipnn/pipnn.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/search/search_functors.h"
#include "ginger/searcher.h"
#include "ginger/searcher.pb.h"
#include "ginger/searcher_builder.h"
#include "ginger/testing/parse_text_proto.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_test_stub.h"

namespace ginger {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::ginger::internal::ParallelSchedulerReferenceTest;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

std::vector<float> MakeRandomData(size_t num_points, size_t dim,
                                  std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<float> data(num_points * dim);
  for (auto& v : data) v = dist(rng);
  return data;
}

struct QuantizedData {
  TurboQuantCodec quantizer;
  TurboQuantDatasetSoA points;
};

QuantizedData MakeQuantizedPoints(
    size_t dim, size_t num_points, const std::vector<float>& data,
    TurboQuantDistanceMetric metric =
        TurboQuantDistanceMetric::kDotProductDistance) {
  std::vector<absl::Span<const float>> datapoints;
  for (size_t i = 0; i < num_points; ++i) {
    datapoints.push_back(absl::MakeConstSpan(data.data() + i * dim, dim));
  }
  TurboQuantCodec tq(dim, 42, metric);
  auto bp_or = TurboQuantDatasetSoA::Create(
      tq, datapoints,
      TurboQuantDatasetOptions{
          .pad_to_cache_line_size = false,
          .quantization_bits = 4,
          .store_query_byte_sums = false,
          .normalized_datapoints = false,
          .scaling_storage_mode = ScalingStorageMode::kSeparateArray});
  ABSL_CHECK_OK(bp_or);
  return {std::move(tq), *std::move(bp_or)};
}

// Returns a valid PipnnConfig suitable for small test datasets.
PipnnConfig MakeValidConfig() {
  return PARSE_TEXT_PROTO(R"pb(
    max_leaf_size: 50
    num_passes: 1
    max_degree: 16
    robust_prune_alpha: 1.0
    min_leaf_size: 10
    hash_prune { reservoir_size: 32 num_hash_bits: 8 }
    leader_selection {
      leader_sample_rate: 0.1
      max_num_leaders: 32
      top_level_leaders: 16
      fanout_sequence: [ 10, 3, 1 ]
    }
  )pb");
}

// Validates structural invariants of a PiPNN graph.
void VerifyGraphStructure(const FlatGraph& graph, int32_t num_nodes,
                          int32_t max_degree) {
  EXPECT_EQ(graph.NumNodes(), num_nodes);
  EXPECT_EQ(graph.MaxDegree(), max_degree);
  for (int32_t u = 0; u < num_nodes; ++u) {
    int32_t deg = graph.Degree(u);
    EXPECT_GE(deg, 0) << "node=" << u;
    EXPECT_LE(deg, max_degree) << "node=" << u;
    auto neighbors = graph.Neighbors(u);
    EXPECT_EQ(static_cast<int32_t>(neighbors.size()), deg) << "node=" << u;
    for (int32_t j = 0; j < deg; ++j) {
      EXPECT_GE(neighbors[j], 0) << "node=" << u << " j=" << j;
      EXPECT_LT(neighbors[j], num_nodes) << "node=" << u << " j=" << j;
      EXPECT_NE(neighbors[j], u)
          << "node=" << u << " j=" << j << " self-loop detected";
    }
  }
}

class PipnnTestBase
    : public ginger::internal::ParallelSchedulerReferenceTest {
 protected:
  QuantizedData CreateQuantizedData(
      size_t dim, size_t num_points,
      TurboQuantDistanceMetric metric =
          TurboQuantDistanceMetric::kDotProductDistance) {
    std::mt19937 rng(42);
    auto data = MakeRandomData(num_points, dim, rng);
    return MakeQuantizedPoints(dim, num_points, data, metric);
  }
};

class ValidatePipnnConfigTest : public PipnnTestBase {
 protected:
  void SetUp() override {
    // Create a small dataset that works with valid configs.
    const size_t dim = 32;
    const size_t num_points = 100;
    qd_ = std::make_unique<QuantizedData>(CreateQuantizedData(dim, num_points));
  }

  absl::Status Build(const PipnnConfig& config) {
    return BuildPipnnIndex(qd_->points, qd_->quantizer, config).status();
  }

  std::unique_ptr<QuantizedData> qd_;
};

TEST_F(ValidatePipnnConfigTest, AcceptsValidConfig) {
  EXPECT_THAT(Build(MakeValidConfig()), IsOk());
}

TEST_F(ValidatePipnnConfigTest, RejectsZeroPasses) {
  PipnnConfig config = MakeValidConfig();
  config.set_num_passes(0);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("num_passes")));
}

TEST_F(ValidatePipnnConfigTest, RejectsSmallLeafSize) {
  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(3);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("max_leaf_size")));
}

TEST_F(ValidatePipnnConfigTest, RejectsInvalidLeaderSampleRateZero) {
  PipnnConfig config = MakeValidConfig();
  config.mutable_leader_selection()->set_leader_sample_rate(0.0f);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("leader_sample_rate")));
}

TEST_F(ValidatePipnnConfigTest, RejectsInvalidLeaderSampleRateAboveOne) {
  PipnnConfig config = MakeValidConfig();
  config.mutable_leader_selection()->set_leader_sample_rate(1.5f);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("leader_sample_rate")));
}

TEST_F(ValidatePipnnConfigTest, RejectsZeroReservoirSize) {
  PipnnConfig config = MakeValidConfig();
  config.mutable_hash_prune()->set_reservoir_size(0);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("reservoir_size")));
}

TEST_F(ValidatePipnnConfigTest, RejectsInvalidNumHashBits) {
  PipnnConfig config = MakeValidConfig();
  config.mutable_hash_prune()->set_num_hash_bits(0);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("num_hash_bits")));
}

TEST_F(ValidatePipnnConfigTest, RejectsNumHashBitsTooLarge) {
  PipnnConfig config = MakeValidConfig();
  config.mutable_hash_prune()->set_num_hash_bits(17);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("num_hash_bits")));
}

TEST_F(ValidatePipnnConfigTest, RejectsTooFewMaxNumLeaders) {
  PipnnConfig config = MakeValidConfig();
  config.mutable_leader_selection()->set_max_num_leaders(2);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("max_num_leaders")));
}

TEST_F(ValidatePipnnConfigTest, RejectsTooFewTopLevelLeaders) {
  PipnnConfig config = MakeValidConfig();
  config.mutable_leader_selection()->set_top_level_leaders(0);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("top_level_leaders")));
}

TEST_F(ValidatePipnnConfigTest, RejectsZeroMaxDegree) {
  PipnnConfig config = MakeValidConfig();
  config.set_max_degree(0);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("max_degree")));
}

TEST_F(ValidatePipnnConfigTest, RejectsLowPruneThreshold) {
  PipnnConfig config = MakeValidConfig();
  config.set_robust_prune_alpha(0.5f);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("robust_prune")));
}

TEST_F(ValidatePipnnConfigTest, RejectsLowPruneConfigAlpha) {
  PipnnConfig config = MakeValidConfig();
  config.clear_robust_prune_alpha();
  config.mutable_prune_config()->set_alpha(0.5f);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("robust_prune")));
}

TEST_F(ValidatePipnnConfigTest, RejectsBothRobustPruneAlphaAndPruneConfig) {
  PipnnConfig config = MakeValidConfig();
  config.set_robust_prune_alpha(1.2f);
  config.mutable_prune_config()->set_alpha(1.2f);
  EXPECT_THAT(Build(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot specify both robust_prune_alpha and "
                                 "prune_config")));
}

TEST_F(ValidatePipnnConfigTest, AcceptsPruneConfigWithoutAlpha) {
  PipnnConfig config = MakeValidConfig();
  config.clear_robust_prune_alpha();
  config.mutable_prune_config();
  EXPECT_THAT(Build(config), IsOk());
}

TEST_F(ValidatePipnnConfigTest, AcceptsUnsetRobustPruneAlpha) {
  PipnnConfig config = MakeValidConfig();
  config.clear_robust_prune_alpha();
  EXPECT_THAT(Build(config), IsOk());
}

TEST_F(ValidatePipnnConfigTest, RejectsZeroMinLeafSize) {
  PipnnConfig config = MakeValidConfig();
  config.set_min_leaf_size(0);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("min_leaf_size")));
}

TEST_F(ValidatePipnnConfigTest, RejectsMinLeafSizeTooLarge) {
  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(40);
  config.set_min_leaf_size(30);
  EXPECT_THAT(Build(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("min_leaf_size must be less than "
                                 "or equal to max_leaf_size / 2")));
}

TEST_F(ValidatePipnnConfigTest, RejectsEmptyFanoutSequence) {
  PipnnConfig config = MakeValidConfig();
  config.mutable_leader_selection()->clear_fanout_sequence();
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("fanout_sequence")));
}

TEST_F(ValidatePipnnConfigTest, RejectsZeroFanout) {
  PipnnConfig config = MakeValidConfig();
  config.mutable_leader_selection()->clear_fanout_sequence();
  config.mutable_leader_selection()->add_fanout_sequence(5);
  config.mutable_leader_selection()->add_fanout_sequence(0);
  EXPECT_THAT(Build(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("fanout")));
}

TEST_F(ValidatePipnnConfigTest, RejectsInvalidRobustPruneChunkFraction) {
  PipnnConfig config = MakeValidConfig();
  config.set_robust_prune_chunk_fraction(0.0f);
  EXPECT_THAT(Build(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("robust_prune_chunk_fraction")));
  config.set_robust_prune_chunk_fraction(1.5f);
  EXPECT_THAT(Build(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("robust_prune_chunk_fraction")));
}

class BuildPipnnIndexTest : public PipnnTestBase {};

TEST_F(BuildPipnnIndexTest, SmallDatasetReturnsValidGraph) {
  const size_t dim = 128;
  const size_t num_points = 200;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(100);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());
}

TEST_F(BuildPipnnIndexTest, AllNodesHaveNeighbors) {
  const size_t dim = 64;
  const size_t num_points = 200;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(100);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  for (int32_t u = 0; u < static_cast<int32_t>(num_points); ++u) {
    EXPECT_GT(graph.Degree(u), 0) << "node=" << u << " is isolated";
  }
}

class PipnnDeterminismTest : public PipnnTestBase,
                             public ::testing::WithParamInterface<PipnnConfig> {
};

TEST_P(PipnnDeterminismTest, DeterminismAndPhasedBuildEquivalence) {
  const size_t dim = 64;
  const size_t num_points = 150;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config = GetParam();

  // 1. Verify normal build determinism (phased_build = false)
  config.set_phased_build(false);
  ASSERT_OK_AND_ASSIGN(auto graph1,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  ASSERT_OK_AND_ASSIGN(auto graph2,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph1, num_points, config.max_degree());
  VerifyGraphStructure(graph2, num_points, config.max_degree());
  EXPECT_EQ(graph1, graph2);

  // 2. Verify phased build determinism (phased_build = true)
  config.set_phased_build(true);
  ASSERT_OK_AND_ASSIGN(auto graph_phased1,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  ASSERT_OK_AND_ASSIGN(auto graph_phased2,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph_phased1, num_points, config.max_degree());
  VerifyGraphStructure(graph_phased2, num_points, config.max_degree());
  EXPECT_EQ(graph_phased1, graph_phased2);

  // 3. Verify equivalence between normal build and phased build
  EXPECT_EQ(graph1, graph_phased1);
}

std::vector<PipnnConfig> MakeDeterminismTestConfigs() {
  std::vector<PipnnConfig> configs;

  // 1. Base valid config
  PipnnConfig base = MakeValidConfig();
  base.set_max_leaf_size(80);
  configs.push_back(base);

  // 2. Multiple passes
  PipnnConfig multi_pass = base;
  multi_pass.set_num_passes(3);
  configs.push_back(multi_pass);

  // 3. Custom fanout sequence
  PipnnConfig custom_fanout = base;
  custom_fanout.mutable_leader_selection()->clear_fanout_sequence();
  custom_fanout.mutable_leader_selection()->add_fanout_sequence(5);
  custom_fanout.mutable_leader_selection()->add_fanout_sequence(2);
  custom_fanout.mutable_leader_selection()->add_fanout_sequence(1);
  configs.push_back(custom_fanout);

  // 4. Small leaf size
  PipnnConfig small_leaf = base;
  small_leaf.set_max_leaf_size(10);
  small_leaf.set_min_leaf_size(5);
  configs.push_back(small_leaf);

  // 5. Reverse RNG backfill with keep_pruned_neighbors
  PipnnConfig rev_backfill = base;
  rev_backfill.clear_robust_prune_alpha();
  rev_backfill.mutable_prune_config()->set_alpha(1.0f);
  rev_backfill.mutable_prune_config()->set_keep_pruned_neighbors(true);
  rev_backfill.mutable_prune_config()->set_reverse_rng_backfill(true);
  configs.push_back(rev_backfill);

  return configs;
}

INSTANTIATE_TEST_SUITE_P(VariousConfigs, PipnnDeterminismTest,
                         ::testing::ValuesIn(MakeDeterminismTestConfigs()));

TEST_F(BuildPipnnIndexTest, DifferentSeedProducesDifferentGraph) {
  const size_t dim = 64;
  const size_t num_points = 150;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config1 = MakeValidConfig();
  config1.set_max_leaf_size(80);
  config1.set_seed(12345);

  PipnnConfig config2 = config1;
  config2.set_seed(54321);

  ASSERT_OK_AND_ASSIGN(auto graph1,
                       BuildPipnnIndex(qd.points, qd.quantizer, config1));
  ASSERT_OK_AND_ASSIGN(auto graph2,
                       BuildPipnnIndex(qd.points, qd.quantizer, config2));

  VerifyGraphStructure(graph1, num_points, config1.max_degree());
  VerifyGraphStructure(graph2, num_points, config2.max_degree());

  // Different seed produces different graph.
  EXPECT_NE(graph1, graph2);
}

TEST_F(BuildPipnnIndexTest, SmallLeafSizeProducesValidGraph) {
  const size_t dim = 64;
  const size_t num_points = 100;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(4);
  config.set_min_leaf_size(2);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());
}

TEST_F(BuildPipnnIndexTest, MultiplePassesProducesDifferentValidGraph) {
  const size_t dim = 64;
  const size_t num_points = 200;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config1 = MakeValidConfig();
  config1.set_max_leaf_size(100);
  config1.set_num_passes(1);

  PipnnConfig config3 = config1;
  config3.set_num_passes(3);

  ASSERT_OK_AND_ASSIGN(auto graph1,
                       BuildPipnnIndex(qd.points, qd.quantizer, config1));
  ASSERT_OK_AND_ASSIGN(auto graph3,
                       BuildPipnnIndex(qd.points, qd.quantizer, config3));

  VerifyGraphStructure(graph1, num_points, config1.max_degree());
  VerifyGraphStructure(graph3, num_points, config3.max_degree());

  // Multiple passes should partition the data differently due to pass-dependent
  // leader selection seeds, leading to a different graph.
  EXPECT_NE(graph1, graph3);
}

TEST_F(BuildPipnnIndexTest, PhasedBuildProducesValidGraph) {
  const size_t dim = 64;
  const size_t num_points = 200;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(100);
  config.set_phased_build(true);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());
  for (int32_t u = 0; u < static_cast<int32_t>(num_points); ++u) {
    EXPECT_GT(graph.Degree(u), 0) << "node=" << u << " is isolated";
  }
}

TEST_F(BuildPipnnIndexTest, CustomFanoutSequence) {
  const size_t dim = 64;
  const size_t num_points = 200;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(100);
  config.mutable_leader_selection()->clear_fanout_sequence();
  config.mutable_leader_selection()->add_fanout_sequence(5);
  config.mutable_leader_selection()->add_fanout_sequence(2);
  config.mutable_leader_selection()->add_fanout_sequence(1);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());
}

TEST_F(BuildPipnnIndexTest, L2DistanceMetric) {
  const size_t dim = 64;
  const size_t num_points = 200;
  auto qd = CreateQuantizedData(dim, num_points,
                                TurboQuantDistanceMetric::kSquaredL2Distance);

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(100);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());
}

TEST_F(BuildPipnnIndexTest, MinimalDataset) {
  const size_t dim = 32;
  const size_t num_points = 5;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(10);
  config.mutable_leader_selection()->set_top_level_leaders(3);
  config.mutable_leader_selection()->set_max_num_leaders(3);
  config.set_min_leaf_size(2);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());
}

TEST_F(BuildPipnnIndexTest, AllNodesReachableInFewHops) {
  const size_t dim = 64;
  const int32_t num_points = 500;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(250);
  config.set_num_passes(2);
  config.mutable_hash_prune()->set_reservoir_size(64);
  config.set_max_degree(32);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());

  // BFS from node 0. All nodes should be reachable in few hops.
  std::vector<int32_t> hop_distance(num_points, -1);
  std::vector<int32_t> frontier = {0};
  hop_distance[0] = 0;
  int32_t max_hops = 0;

  while (!frontier.empty()) {
    std::vector<int32_t> next_frontier;
    for (int32_t u : frontier) {
      for (int32_t v : graph.Neighbors(u)) {
        if (hop_distance[v] == -1) {
          hop_distance[v] = hop_distance[u] + 1;
          max_hops = std::max(max_hops, hop_distance[v]);
          next_frontier.push_back(v);
        }
      }
    }
    frontier = std::move(next_frontier);
  }

  int32_t num_reachable = 0;
  for (int32_t d : hop_distance) {
    if (d >= 0) ++num_reachable;
  }

  EXPECT_EQ(num_reachable, num_points)
      << "Not all nodes are reachable from node 0. Reachable: " << num_reachable
      << " / " << num_points;

  // A well-connected ANN graph with degree 32 on 500 points should have very
  // small diameter. With degree 32, two hops cover up to ~1024 nodes, so the
  // graph should be traversable in very few hops.
  EXPECT_LE(max_hops, 6) << "Graph diameter is too large: " << max_hops
                         << " hops";
}

TEST_F(BuildPipnnIndexTest, BeamSearchRecallSmokeTest) {
  const size_t dim = 64;
  const int32_t num_points = 500;
  std::mt19937 rng(42);
  auto data = MakeRandomData(num_points, dim, rng);

  // Wrap the data in SpanDataset for the searcher.
  auto dataset = std::make_shared<SpanDataset>(num_points, dim);
  for (int32_t i = 0; i < num_points; ++i) {
    dataset->Set(i, absl::MakeConstSpan(data.data() + i * dim, dim));
  }

  // Create quantized base points.
  auto qd = MakeQuantizedPoints(dim, num_points, data);
  auto quantized_dataset =
      std::make_shared<TurboQuantDatasetSoA>(std::move(qd.points));
  auto quantizer = std::make_shared<TurboQuantCodec>(std::move(qd.quantizer));

  // Build the PiPNN graph.
  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(250);
  config.set_num_passes(2);
  config.mutable_hash_prune()->set_reservoir_size(64);
  config.set_max_degree(32);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(*quantized_dataset, *quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());

  // Wrap graph in GingerSearcher.
  GingerSearcherConfig searcher_config;
  searcher_config.set_distance_metric(DOT_PRODUCT);

  auto index = std::make_unique<const FlatGraph>(std::move(graph));
  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(searcher_config)
          .AddDataset(dataset)
          .AddQuantizedDataset(quantized_dataset, quantizer)
          .AddIndex(std::move(index))
          .Build());

  // Perform searches and evaluate recall.
  int32_t num_queries = 50;
  int32_t hits = 0;
  for (int32_t q = 0; q < num_queries; ++q) {
    // Generate a random query vector.
    auto query = MakeRandomData(1, dim, rng);

    // Find brute-force 1-NN (nearest neighbor) using float dot product.
    float best_dot = -std::numeric_limits<float>::infinity();
    int32_t best_j = -1;
    for (int32_t j = 0; j < num_points; ++j) {
      float dot = 0.0f;
      for (size_t d = 0; d < dim; ++d) {
        dot += query[d] * data[j * dim + d];
      }
      if (dot > best_dot) {
        best_dot = dot;
        best_j = j;
      }
    }

    SearchOptions search_options = PARSE_TEXT_PROTO(R"pb(
      search_space: SEARCH_SPACE_TURBO_QUANT
      reranking_strategy: RERANKING_STRATEGY_NONE
      beam_size: 64
      num_neighbors: 10
    )pb");
    ASSERT_OK_AND_ASSIGN(std::vector<DistanceAndId> results,
                         searcher->FindNeighbors(query, search_options));

    for (const auto& res : results) {
      if (res.id == best_j) {
        ++hits;
        break;
      }
    }
  }

  double recall = static_cast<double>(hits) / num_queries;
  EXPECT_GE(recall, 0.7) << "Beam search recall is too low: " << recall;
}

TEST_F(BuildPipnnIndexTest, RobustPruneOnTheFlyStructure) {
  const size_t dim = 64;
  const int32_t num_points = 300;
  auto qd = CreateQuantizedData(dim, num_points);

  for (bool phased_build : {false, true}) {
    PipnnConfig config = MakeValidConfig();
    config.set_max_leaf_size(100);
    config.set_num_passes(2);
    config.set_max_degree(24);
    config.set_phased_build(phased_build);
    config.set_pruning_strategy(
        PipnnConfig::PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY);

    ASSERT_OK_AND_ASSIGN(auto graph,
                         BuildPipnnIndex(qd.points, qd.quantizer, config));
    VerifyGraphStructure(graph, num_points, config.max_degree());
  }
}

TEST_F(BuildPipnnIndexTest, RobustPruneOnTheFlyBeamSearchRecall) {
  const size_t dim = 64;
  const int32_t num_points = 500;
  std::mt19937 rng(42);
  auto data = MakeRandomData(num_points, dim, rng);

  auto dataset = std::make_shared<SpanDataset>(num_points, dim);
  for (int32_t i = 0; i < num_points; ++i) {
    dataset->Set(i, absl::MakeConstSpan(data.data() + i * dim, dim));
  }

  auto qd = MakeQuantizedPoints(dim, num_points, data);
  auto quantized_dataset =
      std::make_shared<TurboQuantDatasetSoA>(std::move(qd.points));
  auto quantizer = std::make_shared<TurboQuantCodec>(std::move(qd.quantizer));

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(250);
  config.set_num_passes(2);
  config.set_max_degree(32);
  config.set_pruning_strategy(
      PipnnConfig::PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(*quantized_dataset, *quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());

  GingerSearcherConfig searcher_config;
  searcher_config.set_distance_metric(DOT_PRODUCT);

  auto index = std::make_unique<const FlatGraph>(std::move(graph));
  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(searcher_config)
          .AddDataset(dataset)
          .AddQuantizedDataset(quantized_dataset, quantizer)
          .AddIndex(std::move(index))
          .Build());

  int32_t num_queries = 50;
  int32_t hits = 0;
  for (int32_t q = 0; q < num_queries; ++q) {
    auto query = MakeRandomData(1, dim, rng);

    float best_dot = -std::numeric_limits<float>::infinity();
    int32_t best_j = -1;
    for (int32_t j = 0; j < num_points; ++j) {
      float dot = 0.0f;
      for (size_t d = 0; d < dim; ++d) {
        dot += query[d] * data[j * dim + d];
      }
      if (dot > best_dot) {
        best_dot = dot;
        best_j = j;
      }
    }

    SearchOptions search_options = PARSE_TEXT_PROTO(R"pb(
      search_space: SEARCH_SPACE_TURBO_QUANT
      reranking_strategy: RERANKING_STRATEGY_NONE
      beam_size: 64
      num_neighbors: 10
    )pb");
    ASSERT_OK_AND_ASSIGN(std::vector<DistanceAndId> results,
                         searcher->FindNeighbors(query, search_options));

    for (const auto& res : results) {
      if (res.id == best_j) {
        ++hits;
        break;
      }
    }
  }

  double recall = static_cast<double>(hits) / num_queries;
  EXPECT_GE(recall, 0.7) << "Beam search recall is too low: " << recall;
}

TEST_F(BuildPipnnIndexTest, DeepRecursiveClustering) {
  size_t dim = 8;
  size_t num_points = 500;
  std::mt19937 rng(12345);
  std::vector<float> data = MakeRandomData(num_points, dim, rng);
  QuantizedData qd = MakeQuantizedPoints(dim, num_points, data);

  PipnnConfig config = PARSE_TEXT_PROTO(R"pb(
    max_leaf_size: 15
    num_passes: 1
    max_degree: 8
    robust_prune_alpha: 1.0
    min_leaf_size: 5
    hash_prune { reservoir_size: 32 num_hash_bits: 8 }
    leader_selection {
      leader_sample_rate: 0.3
      max_num_leaders: 16
      top_level_leaders: 8
      fanout_sequence: [ 4, 3, 2, 2, 1 ]
    }
  )pb");

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());
}

TEST_F(BuildPipnnIndexTest, RobustPruneOnTheFlyOverflowPruning) {
  size_t dim = 16;
  size_t num_points = 400;
  std::mt19937 rng(42);
  std::vector<float> data = MakeRandomData(num_points, dim, rng);
  QuantizedData qd = MakeQuantizedPoints(dim, num_points, data);

  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(100);
  config.set_num_passes(4);
  config.set_max_degree(8);
  config.set_pruning_strategy(
      PipnnConfig::PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY);

  ASSERT_OK_AND_ASSIGN(auto graph,
                       BuildPipnnIndex(qd.points, qd.quantizer, config));
  VerifyGraphStructure(graph, num_points, config.max_degree());

  int total_degree = 0;
  for (int i = 0; i < num_points; ++i) {
    total_degree += graph.Degree(i);
  }
  // With on-the-fly pruning to 75% of max_degree (8 * 0.75 = 6), nodes prune
  // to at most 6 survivors on overflow, yielding an average degree ~5.7.
  EXPECT_GE(total_degree, num_points * 5);
}

TEST_F(BuildPipnnIndexTest, BidirectionalConnectivity) {
  const size_t dim = 64;
  const int32_t num_points = 8;
  auto qd = CreateQuantizedData(dim, num_points,
                                TurboQuantDistanceMetric::kSquaredL2Distance);

  for (auto strategy :
       {PipnnConfig::PRUNING_STRATEGY_HASH_PRUNE,
        PipnnConfig::PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY}) {
    PipnnConfig config = MakeValidConfig();
    config.set_max_leaf_size(20);
    config.set_min_leaf_size(4);
    config.set_num_passes(1);
    config.set_max_degree(32);
    config.set_robust_prune_alpha(1000.0f);
    config.mutable_hash_prune()->set_reservoir_size(32);
    config.mutable_hash_prune()->set_num_hash_bits(16);
    config.set_pruning_strategy(strategy);

    ASSERT_OK_AND_ASSIGN(auto graph,
                         BuildPipnnIndex(qd.points, qd.quantizer, config));
    VerifyGraphStructure(graph, num_points, config.max_degree());

    // With a single leaf and high alpha (no pruning), every node gets its
    // 2 forward edges (top-2) plus any reverse edges from nodes that selected
    // it as their best or second-best neighbor, yielding a symmetric graph.
    int32_t max_node_degree = 0;
    int32_t total_degree = 0;
    for (int32_t u = 0; u < num_points; ++u) {
      int32_t deg = graph.Degree(u);
      EXPECT_GE(deg, 2) << "node=" << u << " strategy=" << strategy;
      max_node_degree = std::max(max_node_degree, deg);
      total_degree += deg;
      for (GingerNodeId v : graph.Neighbors(u)) {
        EXPECT_THAT(graph.Neighbors(v), ::testing::Contains(u))
            << "Missing reverse edge (" << v << " -> " << u << ") for edge ("
            << u << " -> " << v << ") with strategy=" << strategy;
      }
    }

    EXPECT_GT(max_node_degree, 2)
        << "Graph lacks reverse edges; max degree should exceed 2 with "
           "bidirectional insertion.";
    EXPECT_GT(total_degree, num_points * 2)
        << "Total degree should exceed num_points * 2 due to reverse edges.";
  }
}

TEST_F(BuildPipnnIndexTest, IsCancelledImmediateCancel) {
  const size_t dim = 16;
  const int32_t num_points = 100;
  auto qd = CreateQuantizedData(dim, num_points);
  PipnnConfig config = MakeValidConfig();

  EXPECT_THAT(
      BuildPipnnIndex(qd.points, qd.quantizer, config, []() { return true; }),
      StatusIs(absl::StatusCode::kCancelled));
}

TEST_F(BuildPipnnIndexTest, IsCancelledMidClusteringCancel) {
  const size_t dim = 16;
  const int32_t num_points = 200;
  auto qd = CreateQuantizedData(dim, num_points);
  PipnnConfig config = MakeValidConfig();
  config.set_max_leaf_size(15);
  config.set_min_leaf_size(5);
  config.set_num_passes(2);

  // The first check is at the start of the build and the second one is the
  // first inside partitioning.
  int call_count = 0;
  auto is_cancelled = [&call_count]() { return ++call_count == 2; };

  EXPECT_THAT(BuildPipnnIndex(qd.points, qd.quantizer, config, is_cancelled),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(call_count, 2);
}

TEST_F(BuildPipnnIndexTest, KeepPrunedNeighborsIncreasesHashPruneDegree) {
  const size_t dim = 64;
  const int32_t num_points = 200;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config_default = MakeValidConfig();
  config_default.clear_robust_prune_alpha();
  config_default.set_max_leaf_size(100);
  config_default.set_max_degree(24);
  config_default.mutable_prune_config()->set_alpha(1.0f);
  config_default.mutable_prune_config()->set_keep_pruned_neighbors(false);

  PipnnConfig config_keep = config_default;
  config_keep.mutable_prune_config()->set_keep_pruned_neighbors(true);

  ASSERT_OK_AND_ASSIGN(
      auto graph_default,
      BuildPipnnIndex(qd.points, qd.quantizer, config_default));
  ASSERT_OK_AND_ASSIGN(auto graph_keep,
                       BuildPipnnIndex(qd.points, qd.quantizer, config_keep));

  VerifyGraphStructure(graph_default, num_points, config_default.max_degree());
  VerifyGraphStructure(graph_keep, num_points, config_keep.max_degree());

  int32_t total_default = 0;
  int32_t total_keep = 0;
  for (int32_t u = 0; u < num_points; ++u) {
    EXPECT_GE(graph_keep.Degree(u), graph_default.Degree(u)) << "node=" << u;
    total_default += graph_default.Degree(u);
    total_keep += graph_keep.Degree(u);
  }
  EXPECT_GT(total_keep, total_default);
}

TEST_F(BuildPipnnIndexTest,
       KeepPrunedNeighborsRespectedInFinalPruneWithRobustPruneOnTheFly) {
  const size_t dim = 64;
  // Use a single leaf (num_points <= max_leaf_size) so that on-the-fly
  // overflow pruning is deterministic across runs.
  const int32_t num_points = 80;
  auto qd = CreateQuantizedData(dim, num_points);

  PipnnConfig config_false = MakeValidConfig();
  config_false.clear_robust_prune_alpha();
  config_false.set_max_leaf_size(100);
  config_false.set_max_degree(8);
  config_false.set_pruning_strategy(
      PipnnConfig::PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY);
  config_false.mutable_prune_config()->set_alpha(1.0f);
  config_false.mutable_prune_config()->set_keep_pruned_neighbors(false);

  PipnnConfig config_true = config_false;
  config_true.mutable_prune_config()->set_keep_pruned_neighbors(true);

  ASSERT_OK_AND_ASSIGN(auto graph_false,
                       BuildPipnnIndex(qd.points, qd.quantizer, config_false));
  ASSERT_OK_AND_ASSIGN(auto graph_true,
                       BuildPipnnIndex(qd.points, qd.quantizer, config_true));

  VerifyGraphStructure(graph_false, num_points, config_false.max_degree());
  VerifyGraphStructure(graph_true, num_points, config_true.max_degree());

  // Because in-flight overflow prunes ignore keep_pruned_neighbors, the
  // candidate sets entering FinalizeAndPruneGraph are identical, so
  // graph_false.Neighbors(u) is a prefix of graph_true.Neighbors(u), and the
  // final prune increases total degree.
  int32_t total_false = 0;
  int32_t total_true = 0;
  for (int32_t u = 0; u < num_points; ++u) {
    const auto nbrs_false = graph_false.Neighbors(u);
    const auto nbrs_true = graph_true.Neighbors(u);
    ASSERT_LE(nbrs_false.size(), nbrs_true.size()) << "node=" << u;
    for (size_t i = 0; i < nbrs_false.size(); ++i) {
      EXPECT_EQ(nbrs_true[i], nbrs_false[i]) << "node=" << u << " i=" << i;
    }
    total_false += graph_false.Degree(u);
    total_true += graph_true.Degree(u);
  }
  EXPECT_GT(total_true, total_false);
}

double ComputeReciprocity(const FlatGraph& graph) {
  int64_t total_edges = 0;
  int64_t mutual_edges = 0;
  for (int32_t u = 0; u < graph.NumNodes(); ++u) {
    for (GingerNodeId v : graph.Neighbors(u)) {
      ++total_edges;
      const auto nbrs_v = graph.Neighbors(v);
      if (absl::c_find(nbrs_v, u) != nbrs_v.end()) {
        ++mutual_edges;
      }
    }
  }
  return total_edges > 0 ? static_cast<double>(mutual_edges) / total_edges
                         : 0.0;
}

TEST_F(BuildPipnnIndexTest,
       ReverseRngBackfillSplitsBudgetAndIncreasesReciprocity) {
  const size_t dim = 64;
  const int32_t num_points = 200;
  auto qd = CreateQuantizedData(dim, num_points);

  // Stage-1 reference graph with max_degree = 8 (half of 16).
  PipnnConfig config_s1 = MakeValidConfig();
  config_s1.clear_robust_prune_alpha();
  config_s1.set_max_leaf_size(100);
  config_s1.set_max_degree(8);
  config_s1.mutable_prune_config()->set_alpha(1.0f);
  config_s1.mutable_prune_config()->set_keep_pruned_neighbors(false);
  config_s1.mutable_prune_config()->set_reverse_rng_backfill(false);

  // Full Stage-1 reference graph with max_degree = 16 and
  // keep_pruned_neighbors = false (all unoccluded RNG survivors up to 16).
  PipnnConfig config_s1_full = config_s1;
  config_s1_full.set_max_degree(16);

  // Standard max_degree = 16 graph with keep_pruned_neighbors = true.
  PipnnConfig config_std = config_s1;
  config_std.set_max_degree(16);
  config_std.mutable_prune_config()->set_keep_pruned_neighbors(true);

  // Reverse-RNG backfill with max_degree = 16 and keep_pruned_neighbors =
  // false.
  PipnnConfig config_rev = config_s1;
  config_rev.set_max_degree(16);
  config_rev.mutable_prune_config()->set_reverse_rng_backfill(true);

  // Reverse-RNG backfill with max_degree = 16 and keep_pruned_neighbors = true.
  PipnnConfig config_rev_keep = config_rev;
  config_rev_keep.mutable_prune_config()->set_keep_pruned_neighbors(true);

  // Merging the reverse edges in source-node batches of 20 nodes
  // (chunk_fraction = 0.1) must produce a bit-for-bit identical graph.
  PipnnConfig config_rev_chunked = config_rev_keep;
  config_rev_chunked.set_robust_prune_chunk_fraction(0.1f);

  ASSERT_OK_AND_ASSIGN(auto graph_s1,
                       BuildPipnnIndex(qd.points, qd.quantizer, config_s1));
  ASSERT_OK_AND_ASSIGN(
      auto graph_s1_full,
      BuildPipnnIndex(qd.points, qd.quantizer, config_s1_full));
  ASSERT_OK_AND_ASSIGN(auto graph_std,
                       BuildPipnnIndex(qd.points, qd.quantizer, config_std));
  ASSERT_OK_AND_ASSIGN(auto graph_rev,
                       BuildPipnnIndex(qd.points, qd.quantizer, config_rev));
  ASSERT_OK_AND_ASSIGN(
      auto graph_rev_keep,
      BuildPipnnIndex(qd.points, qd.quantizer, config_rev_keep));
  ASSERT_OK_AND_ASSIGN(
      auto graph_rev_chunked,
      BuildPipnnIndex(qd.points, qd.quantizer, config_rev_chunked));
  EXPECT_EQ(graph_rev_chunked, graph_rev_keep);

  VerifyGraphStructure(graph_s1, num_points, 8);
  VerifyGraphStructure(graph_s1_full, num_points, 16);
  VerifyGraphStructure(graph_std, num_points, 16);
  VerifyGraphStructure(graph_rev, num_points, 16);
  VerifyGraphStructure(graph_rev_keep, num_points, 16);

  int32_t total_s1 = 0;
  int32_t total_rev = 0;
  int32_t total_rev_keep = 0;
  for (int32_t u = 0; u < num_points; ++u) {
    const auto nbrs_s1 = graph_s1.Neighbors(u);
    const auto nbrs_s1_full = graph_s1_full.Neighbors(u);
    const auto nbrs_rev = graph_rev.Neighbors(u);
    const auto nbrs_rev_keep = graph_rev_keep.Neighbors(u);

    // 1. No duplicate neighbors in any node's adjacency list.
    for (size_t i = 0; i < nbrs_rev_keep.size(); ++i) {
      for (size_t j = i + 1; j < nbrs_rev_keep.size(); ++j) {
        EXPECT_NE(nbrs_rev_keep[i], nbrs_rev_keep[j]) << "node=" << u;
      }
    }

    // 2. Stage-1 primary forward survivors (max_degree / 2 = 8) form an exact
    // prefix of graph_rev.Neighbors(u), and every remaining edge in graph_rev
    // is either an incoming reverse edge from some node v's Stage-1 primary
    // survivors or one of u's own remaining unoccluded RNG survivors.
    ASSERT_LE(nbrs_s1.size(), nbrs_rev.size()) << "node=" << u;
    for (size_t i = 0; i < nbrs_s1.size(); ++i) {
      EXPECT_EQ(nbrs_rev[i], nbrs_s1[i]) << "node=" << u << " i=" << i;
    }
    for (size_t i = nbrs_s1.size(); i < nbrs_rev.size(); ++i) {
      const GingerNodeId v = nbrs_rev[i];
      const bool is_reverse_from_s1 =
          absl::c_linear_search(graph_s1.Neighbors(v), u);
      const bool is_secondary_rng_survivor =
          absl::c_linear_search(nbrs_s1_full, v);
      EXPECT_TRUE(is_reverse_from_s1 || is_secondary_rng_survivor)
          << "Backfilled neighbor v=" << v << " of u=" << u
          << " is neither a Stage-1 reverse edge nor a secondary RNG survivor";
    }

    // 3. Enabling keep_pruned_neighbors preserves graph_rev.Neighbors(u) as a
    // prefix and tops up remaining capacity with pruned candidates.
    ASSERT_LE(nbrs_rev.size(), nbrs_rev_keep.size()) << "node=" << u;
    for (size_t i = 0; i < nbrs_rev.size(); ++i) {
      EXPECT_EQ(nbrs_rev_keep[i], nbrs_rev[i]) << "node=" << u << " i=" << i;
    }

    total_s1 += graph_s1.Degree(u);
    total_rev += graph_rev.Degree(u);
    total_rev_keep += graph_rev_keep.Degree(u);
  }

  EXPECT_GT(total_rev, total_s1);
  EXPECT_GT(total_rev_keep, total_rev);
  EXPECT_GT(ComputeReciprocity(graph_rev_keep), ComputeReciprocity(graph_std));
}

TEST_F(BuildPipnnIndexTest,
       ReverseRngBackfillKeepsClosestReverseEdgesWhenOverCapacity) {
  const size_t dim = 64;
  const int32_t num_points = 200;
  const int32_t max_degree = 6;
  auto qd = CreateQuantizedData(dim, num_points);

  // Stage-1 reference graph holding each node's primary survivors, i.e., its
  // closest (max_degree + 1) / 2 = 3 RobustPrune survivors.
  PipnnConfig config_primary = MakeValidConfig();
  config_primary.clear_robust_prune_alpha();
  config_primary.set_max_leaf_size(100);
  config_primary.set_num_passes(2);
  config_primary.set_max_degree((max_degree + 1) / 2);
  config_primary.mutable_prune_config()->set_alpha(1.0f);

  // Stage-1 reference graph holding all RobustPrune survivors up to
  // max_degree.
  PipnnConfig config_survivors = config_primary;
  config_survivors.set_max_degree(max_degree);

  PipnnConfig config_rev = config_survivors;
  config_rev.mutable_prune_config()->set_reverse_rng_backfill(true);

  // Merging the reverse edges of a single source node per batch
  // (num_points * chunk_fraction < 1) must produce an identical graph.
  PipnnConfig config_rev_single_source = config_rev;
  config_rev_single_source.set_robust_prune_chunk_fraction(0.001f);

  ASSERT_OK_AND_ASSIGN(
      auto graph_primary,
      BuildPipnnIndex(qd.points, qd.quantizer, config_primary));
  ASSERT_OK_AND_ASSIGN(
      auto graph_survivors,
      BuildPipnnIndex(qd.points, qd.quantizer, config_survivors));
  ASSERT_OK_AND_ASSIGN(auto graph_rev,
                       BuildPipnnIndex(qd.points, qd.quantizer, config_rev));
  ASSERT_OK_AND_ASSIGN(
      auto graph_rev_single_source,
      BuildPipnnIndex(qd.points, qd.quantizer, config_rev_single_source));
  VerifyGraphStructure(graph_rev, num_points, max_degree);
  EXPECT_EQ(graph_rev_single_source, graph_rev);

  // incoming[u] holds the nodes v with u among their primary survivors, in
  // increasing id order.
  std::vector<std::vector<GingerNodeId>> incoming(num_points);
  for (GingerNodeId v = 0; v < num_points; ++v) {
    for (GingerNodeId u : graph_primary.Neighbors(v)) {
      incoming[u].push_back(v);
    }
  }

  const TurboQuantBuildScorer scorer{.dataset = &qd.points,
                                     .quantizer = &qd.quantizer};
  int num_over_capacity = 0;
  int num_topped_up = 0;
  for (GingerNodeId u = 0; u < num_points; ++u) {
    const absl::Span<const GingerNodeId> primary = graph_primary.Neighbors(u);
    std::vector<DistanceAndId> reverse;
    for (GingerNodeId v : incoming[u]) {
      if (!absl::c_linear_search(primary, v)) reverse.push_back({0.0f, v});
    }
    const size_t reverse_capacity = max_degree - primary.size();
    if (reverse.size() > reverse_capacity) {
      // Only the closest reverse edges by (distance, id) are kept, in that
      // order.
      ++num_over_capacity;
      ASSERT_OK(scorer(u, absl::MakeSpan(reverse)));
      absl::c_sort(reverse);
      reverse.resize(reverse_capacity);
    }

    std::vector<GingerNodeId> expected(primary.begin(), primary.end());
    for (const DistanceAndId& r : reverse) expected.push_back(r.id);
    // Any remaining capacity is topped up with u's remaining RobustPrune
    // survivors.
    const size_t num_primary_and_reverse = expected.size();
    for (GingerNodeId w :
         graph_survivors.Neighbors(u).subspan(primary.size())) {
      if (expected.size() >= max_degree) break;
      if (!absl::c_linear_search(expected, w)) expected.push_back(w);
    }
    if (expected.size() > num_primary_and_reverse) ++num_topped_up;

    EXPECT_THAT(graph_rev.Neighbors(u), ElementsAreArray(expected))
        << "node=" << u;
  }
  // Both the over-capacity and the top-up paths must be exercised.
  EXPECT_GT(num_over_capacity, 0);
  EXPECT_GT(num_topped_up, 0);
}

TEST_F(BuildPipnnIndexTest,
       ReverseRngBackfillSupportedWithRobustPruneOnTheFly) {
  const size_t dim = 64;
  const int32_t num_points = 80;
  const int32_t max_degree = 8;
  const size_t primary_cap = (max_degree + 1) / 2;
  auto qd = CreateQuantizedData(dim, num_points);

  // Use max_leaf_size > num_points so each pass forms a single leaf and the
  // pre-finalization graph is identical across all configs below.
  PipnnConfig config_std = MakeValidConfig();
  config_std.clear_robust_prune_alpha();
  config_std.set_max_leaf_size(100);
  config_std.set_max_degree(max_degree);
  config_std.set_pruning_strategy(
      PipnnConfig::PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY);
  config_std.mutable_prune_config()->set_alpha(1.0f);
  config_std.mutable_prune_config()->set_keep_pruned_neighbors(false);

  PipnnConfig config_rev = config_std;
  config_rev.mutable_prune_config()->set_reverse_rng_backfill(true);

  PipnnConfig config_rev_keep = config_rev;
  config_rev_keep.mutable_prune_config()->set_keep_pruned_neighbors(true);

  PipnnConfig config_rev_chunked = config_rev_keep;
  config_rev_chunked.set_robust_prune_chunk_fraction(0.05f);

  ASSERT_OK_AND_ASSIGN(auto graph_std,
                       BuildPipnnIndex(qd.points, qd.quantizer, config_std));
  ASSERT_OK_AND_ASSIGN(auto graph_rev,
                       BuildPipnnIndex(qd.points, qd.quantizer, config_rev));
  ASSERT_OK_AND_ASSIGN(
      auto graph_rev_keep,
      BuildPipnnIndex(qd.points, qd.quantizer, config_rev_keep));
  ASSERT_OK_AND_ASSIGN(
      auto graph_rev_chunked,
      BuildPipnnIndex(qd.points, qd.quantizer, config_rev_chunked));

  VerifyGraphStructure(graph_std, num_points, max_degree);
  VerifyGraphStructure(graph_rev, num_points, max_degree);
  VerifyGraphStructure(graph_rev_keep, num_points, max_degree);
  EXPECT_EQ(graph_rev_chunked, graph_rev_keep);

  int32_t total_rev = 0;
  int32_t total_rev_keep = 0;
  for (GingerNodeId u = 0; u < num_points; ++u) {
    const auto nbrs_std = graph_std.Neighbors(u);
    const auto nbrs_rev = graph_rev.Neighbors(u);
    const auto nbrs_rev_keep = graph_rev_keep.Neighbors(u);

    // Primary survivors (up to primary_cap) must match graph_std's prefix, and
    // remaining edges must be Stage-1 reverse edges or secondary RNG survivors.
    const auto primary =
        nbrs_std.subspan(0, std::min(nbrs_std.size(), primary_cap));
    ASSERT_LE(primary.size(), nbrs_rev.size()) << "node=" << u;
    EXPECT_THAT(nbrs_rev.subspan(0, primary.size()), ElementsAreArray(primary))
        << "node=" << u;
    for (GingerNodeId v : nbrs_rev.subspan(primary.size())) {
      const auto v_std = graph_std.Neighbors(v);
      const auto v_primary =
          v_std.subspan(0, std::min(v_std.size(), primary_cap));
      EXPECT_TRUE(absl::c_linear_search(v_primary, u) ||
                  absl::c_linear_search(nbrs_std, v))
          << "node=" << u << " v=" << v;
    }

    // Enabling keep_pruned_neighbors preserves graph_rev as a prefix and tops
    // up remaining slots with pruned candidates.
    ASSERT_LE(nbrs_rev.size(), nbrs_rev_keep.size()) << "node=" << u;
    EXPECT_THAT(nbrs_rev_keep.subspan(0, nbrs_rev.size()),
                ElementsAreArray(nbrs_rev))
        << "node=" << u;

    total_rev += graph_rev.Degree(u);
    total_rev_keep += graph_rev_keep.Degree(u);
  }
  EXPECT_GT(total_rev_keep, total_rev);
  EXPECT_GT(ComputeReciprocity(graph_rev), ComputeReciprocity(graph_std));
}

}  // namespace
}  // namespace ginger
