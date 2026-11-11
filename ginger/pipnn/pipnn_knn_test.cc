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

#include "ginger/pipnn/pipnn_knn.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/distance/distance.h"
#include "ginger/pipnn/config.pb.h"
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
struct SyntheticData {
  std::vector<float> flat_data;
  SpanDataset float_dataset;
  TurboQuantCodec quantizer;
  TurboQuantDatasetSoA points;
};

SyntheticData MakeSyntheticDataset(
    size_t num_points, size_t dim,
    TurboQuantDistanceMetric metric =
        TurboQuantDistanceMetric::kDotProductDistance,
    uint32_t seed = 42, int quantization_bits = 4) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<float> flat_data(num_points * dim);
  for (float& v : flat_data) {
    v = dist(rng);
  }

  SpanDataset float_dataset(static_cast<GingerNodeId>(num_points), dim);
  std::vector<absl::Span<const float>> spans;
  spans.reserve(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    absl::Span<const float> span =
        absl::MakeConstSpan(flat_data.data() + i * dim, dim);
    float_dataset.Set<float>(static_cast<GingerNodeId>(i), span);
    spans.push_back(span);
  }

  TurboQuantCodec quantizer(dim, seed, metric);
  absl::StatusOr<TurboQuantDatasetSoA> points =
      num_points == 0
          ? TurboQuantDatasetSoA::Create(
                quantizer, /*num_datapoints=*/0,
                TurboQuantDatasetOptions{
                    .pad_to_cache_line_size = false,
                    .quantization_bits = quantization_bits,
                    .store_query_byte_sums = false,
                    .normalized_datapoints = false,
                    .scaling_storage_mode = ScalingStorageMode::kSeparateArray})
          : TurboQuantDatasetSoA::Create(
                quantizer, spans,
                TurboQuantDatasetOptions{
                    .pad_to_cache_line_size = false,
                    .quantization_bits = quantization_bits,
                    .store_query_byte_sums = false,
                    .normalized_datapoints = false,
                    .scaling_storage_mode =
                        ScalingStorageMode::kSeparateArray});
  ABSL_CHECK_OK(points);

  return SyntheticData{
      .flat_data = std::move(flat_data),
      .float_dataset = std::move(float_dataset),
      .quantizer = std::move(quantizer),
      .points = *std::move(points),
  };
}

PipnnConfig MakePipnnConfig(int max_leaf_size = 64, int num_passes = 2) {
  PipnnConfig config;
  config.set_min_leaf_size(8);
  config.set_max_leaf_size(max_leaf_size);
  config.set_num_passes(num_passes);
  config.set_max_degree(10);
  PipnnConfig::LeaderSelectionConfig* leader_selection =
      config.mutable_leader_selection();
  leader_selection->set_leader_sample_rate(0.1f);
  leader_selection->set_max_num_leaders(16);
  leader_selection->set_top_level_leaders(8);
  leader_selection->add_fanout_sequence(2);
  leader_selection->add_fanout_sequence(2);
  return config;
}

class PipnnKnnTest : public ParallelSchedulerReferenceTest {};

TEST_F(PipnnKnnTest, InvalidArguments) {
  SyntheticData data = MakeSyntheticDataset(32, 32);
  PipnnConfig config = MakePipnnConfig();
  auto noop_cb = [](GingerNodeId, std::vector<DistanceAndId>) {};

  EXPECT_THAT(BuildPipnnKnnGraph(data.points, data.quantizer, config, /*k=*/0,
                                 /*overretrieve_factor=*/1, noop_cb),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(BuildPipnnKnnGraph(data.points, data.quantizer, config, /*k=*/5,
                                 /*overretrieve_factor=*/0, noop_cb),
              StatusIs(absl::StatusCode::kInvalidArgument));

  PipnnConfig bad_config = config;
  bad_config.set_num_passes(0);
  EXPECT_THAT(
      BuildPipnnKnnGraph(data.points, data.quantizer, bad_config, /*k=*/5,
                         /*overretrieve_factor=*/1, noop_cb),
      StatusIs(absl::StatusCode::kInvalidArgument));

  PipnnConfig empty_fanout_config = config;
  empty_fanout_config.mutable_leader_selection()->clear_fanout_sequence();
  EXPECT_THAT(
      BuildPipnnKnnGraph(data.points, data.quantizer, empty_fanout_config,
                         /*k=*/5, /*overretrieve_factor=*/1, noop_cb),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // The partitioning config is validated even for an empty dataset.
  SyntheticData empty_data = MakeSyntheticDataset(0, 32);
  EXPECT_THAT(BuildPipnnKnnGraph(empty_data.points, empty_data.quantizer,
                                 empty_fanout_config, /*k=*/5,
                                 /*overretrieve_factor=*/1, noop_cb),
              StatusIs(absl::StatusCode::kInvalidArgument));

  SpanDataset wrong_size_dataset(10, 32);
  EXPECT_THAT(BuildPipnnKnnGraph(data.points, data.quantizer, config, /*k=*/5,
                                 /*overretrieve_factor=*/1, noop_cb,
                                 &wrong_size_dataset),
              StatusIs(absl::StatusCode::kInvalidArgument));

  SpanDataset non_float_dataset(32, 32, CoordinateType::kInt8);
  EXPECT_THAT(BuildPipnnKnnGraph(data.points, data.quantizer, config, /*k=*/5,
                                 /*overretrieve_factor=*/1, noop_cb,
                                 &non_float_dataset),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(PipnnKnnTest, EmptyDataset) {
  SyntheticData data = MakeSyntheticDataset(0, 32);
  PipnnConfig config = MakePipnnConfig();
  int callback_invocations = 0;
  EXPECT_THAT(BuildPipnnKnnGraph(data.points, data.quantizer, config, /*k=*/5,
                                 /*overretrieve_factor=*/1,
                                 [&](GingerNodeId, std::vector<DistanceAndId>) {
                                   ++callback_invocations;
                                 }),
              IsOk());
  EXPECT_EQ(callback_invocations, 0);
}

TEST_F(PipnnKnnTest, EndToEndInvariantsNoSelfLoopsNoDuplicates) {
  constexpr size_t kNumPoints = 200;
  constexpr size_t kDim = 32;
  constexpr int kK = 10;
  constexpr int kOverretrieve = 2;

  SyntheticData data = MakeSyntheticDataset(kNumPoints, kDim);
  PipnnConfig config = MakePipnnConfig(/*max_leaf_size=*/64, /*num_passes=*/3);

  std::vector<std::vector<DistanceAndId>> graph(kNumPoints);
  EXPECT_THAT(BuildPipnnKnnGraph(
                  data.points, data.quantizer, config, kK, kOverretrieve,
                  [&](GingerNodeId u, std::vector<DistanceAndId> neighbors) {
                    graph[u] = std::move(neighbors);
                  },
                  &data.float_dataset),
              IsOk());

  for (size_t u = 0; u < kNumPoints; ++u) {
    const std::vector<DistanceAndId>& neighbors = graph[u];
    EXPECT_LE(neighbors.size(), static_cast<size_t>(kK));
    EXPECT_GT(neighbors.size(), 0u);
    EXPECT_TRUE(absl::c_is_sorted(neighbors))
        << "Neighbors for node " << u << " are not sorted ascending by score";

    absl::flat_hash_set<GingerNodeId> seen;
    for (const DistanceAndId& edge : neighbors) {
      EXPECT_NE(edge.id, static_cast<GingerNodeId>(u))
          << "Self-loop detected for node " << u;
      EXPECT_GE(edge.id, 0);
      EXPECT_LT(edge.id, static_cast<GingerNodeId>(kNumPoints));
      EXPECT_TRUE(seen.insert(edge.id).second)
          << "Duplicate neighbor " << edge.id << " detected for node " << u;
    }
  }
}

TEST_F(PipnnKnnTest, CompactionTriggeredWhenCandidatesExceedTwoM) {
  // Choose small k and overretrieve_factor so M = 4, capacity = 2 * M = 8,
  // while leaf size is 50 and num_passes = 5, so each node collects ~250
  // candidates across passes, repeatedly triggering buffer compaction.
  constexpr size_t kNumPoints = 120;
  constexpr size_t kDim = 32;
  constexpr int kK = 4;
  constexpr int kOverretrieve = 1;

  SyntheticData data = MakeSyntheticDataset(kNumPoints, kDim);
  PipnnConfig config = MakePipnnConfig(/*max_leaf_size=*/50, /*num_passes=*/5);

  std::vector<std::vector<DistanceAndId>> graph(kNumPoints);
  EXPECT_THAT(BuildPipnnKnnGraph(
                  data.points, data.quantizer, config, kK, kOverretrieve,
                  [&](GingerNodeId u, std::vector<DistanceAndId> neighbors) {
                    graph[u] = std::move(neighbors);
                  }),
              IsOk());

  for (size_t u = 0; u < kNumPoints; ++u) {
    EXPECT_EQ(graph[u].size(), static_cast<size_t>(kK));
    EXPECT_TRUE(absl::c_is_sorted(graph[u]));
    absl::flat_hash_set<GingerNodeId> seen;
    for (const DistanceAndId& edge : graph[u]) {
      EXPECT_NE(edge.id, static_cast<GingerNodeId>(u));
      EXPECT_TRUE(seen.insert(edge.id).second);
    }
  }
}

TEST_F(PipnnKnnTest, DeterministicTieBreakingUnderCompaction) {
  // All points are identical, so every pair has identical quantized distance.
  // With M = 4 (capacity = 8), max_leaf_size = 64, and num_passes = 5,
  // every node repeatedly compacts tied candidates. Because compaction sorts
  // full 64-bit (ordered_score << 32 | id) keys, ties must deterministically
  // break in favor of smaller neighbor IDs.
  constexpr size_t kNumPoints = 40;
  constexpr size_t kDim = 32;
  constexpr int kK = 4;
  constexpr int kOverretrieve = 1;

  std::vector<float> flat_data(kNumPoints * kDim, 0.5f);
  std::vector<absl::Span<const float>> spans;
  spans.reserve(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    spans.push_back(absl::MakeConstSpan(flat_data.data() + i * kDim, kDim));
  }
  TurboQuantCodec quantizer(kDim, /*seed=*/42,
                            TurboQuantDistanceMetric::kSquaredL2Distance);
  ASSERT_OK_AND_ASSIGN(
      TurboQuantDatasetSoA points,
      TurboQuantDatasetSoA::Create(
          quantizer, spans,
          TurboQuantDatasetOptions{
              .pad_to_cache_line_size = false,
              .quantization_bits = 4,
              .store_query_byte_sums = false,
              .normalized_datapoints = false,
              .scaling_storage_mode = ScalingStorageMode::kSeparateArray}));

  PipnnConfig config = MakePipnnConfig(/*max_leaf_size=*/64, /*num_passes=*/5);

  std::vector<std::vector<DistanceAndId>> graph(kNumPoints);
  ASSERT_THAT(BuildPipnnKnnGraph(
                  points, quantizer, config, kK, kOverretrieve,
                  [&](GingerNodeId u, std::vector<DistanceAndId> neighbors) {
                    graph[u] = std::move(neighbors);
                  }),
              IsOk());

  for (size_t u = 0; u < kNumPoints; ++u) {
    ASSERT_EQ(graph[u].size(), static_cast<size_t>(kK));
    std::vector<GingerNodeId> expected_ids;
    for (GingerNodeId v = 0; expected_ids.size() < static_cast<size_t>(kK);
         ++v) {
      if (v != static_cast<GingerNodeId>(u)) {
        expected_ids.push_back(v);
      }
    }
    for (int j = 0; j < kK; ++j) {
      EXPECT_EQ(graph[u][j].id, expected_ids[j])
          << "Tie-breaking failed for node " << u << " at rank " << j;
    }
  }
}

TEST_F(PipnnKnnTest, ExactFloatRerankingImprovesRecall) {
  constexpr size_t kNumPoints = 150;
  constexpr size_t kDim = 64;
  constexpr int kK = 10;

  for (TurboQuantDistanceMetric metric :
       {TurboQuantDistanceMetric::kDotProductDistance,
        TurboQuantDistanceMetric::kSquaredL2Distance}) {
    SyntheticData data = MakeSyntheticDataset(kNumPoints, kDim, metric);
    // Single leaf containing all points to isolate exact float reranking vs
    // quantized top-k accuracy.
    PipnnConfig config =
        MakePipnnConfig(/*max_leaf_size=*/200, /*num_passes=*/1);

    // Compute ground truth exact float top-k for each point.
    std::vector<absl::flat_hash_set<GingerNodeId>> ground_truth(kNumPoints);
    for (size_t u = 0; u < kNumPoints; ++u) {
      std::vector<DistanceAndId> all_candidates;
      all_candidates.reserve(kNumPoints - 1);
      for (size_t v = 0; v < kNumPoints; ++v) {
        if (u == v) continue;
        all_candidates.push_back(
            DistanceAndId{.id = static_cast<GingerNodeId>(v)});
      }
      if (metric == TurboQuantDistanceMetric::kSquaredL2Distance) {
        SquaredL2Distance(static_cast<GingerNodeId>(u),
                          absl::MakeSpan(all_candidates), data.float_dataset);
      } else {
        DotProductDistance(static_cast<GingerNodeId>(u),
                           absl::MakeSpan(all_candidates), data.float_dataset);
      }
      absl::c_partial_sort(all_candidates, all_candidates.begin() + kK);
      for (int i = 0; i < kK; ++i) {
        ground_truth[u].insert(all_candidates[i].id);
      }
    }

    std::vector<std::vector<DistanceAndId>> reranked_graph(kNumPoints);
    EXPECT_THAT(
        BuildPipnnKnnGraph(
            data.points, data.quantizer, config, kK, /*overretrieve_factor=*/4,
            [&](GingerNodeId u, std::vector<DistanceAndId> neighbors) {
              reranked_graph[u] = std::move(neighbors);
            },
            &data.float_dataset),
        IsOk());

    size_t hits = 0;
    for (size_t u = 0; u < kNumPoints; ++u) {
      EXPECT_EQ(reranked_graph[u].size(), static_cast<size_t>(kK));
      for (const DistanceAndId& edge : reranked_graph[u]) {
        if (ground_truth[u].contains(edge.id)) {
          ++hits;
        }
      }
    }
    const double recall =
        static_cast<double>(hits) / static_cast<double>(kNumPoints * kK);
    EXPECT_GT(recall, 0.90)
        << "Recall too low for metric " << static_cast<int>(metric);
  }
}

TEST_F(PipnnKnnTest, ExtraPrecision7BitRerankingImprovesRecallWithoutFloat) {
  constexpr size_t kNumPoints = 150;
  constexpr size_t kDim = 64;
  constexpr int kK = 10;

  for (TurboQuantDistanceMetric metric :
       {TurboQuantDistanceMetric::kDotProductDistance,
        TurboQuantDistanceMetric::kSquaredL2Distance}) {
    SyntheticData data_7bit = MakeSyntheticDataset(
        kNumPoints, kDim, metric, /*seed=*/42, /*quantization_bits=*/7);
    PipnnConfig config =
        MakePipnnConfig(/*max_leaf_size=*/200, /*num_passes=*/1);

    std::vector<absl::flat_hash_set<GingerNodeId>> ground_truth(kNumPoints);
    for (size_t u = 0; u < kNumPoints; ++u) {
      std::vector<DistanceAndId> all_candidates;
      all_candidates.reserve(kNumPoints - 1);
      for (size_t v = 0; v < kNumPoints; ++v) {
        if (u == v) continue;
        all_candidates.push_back(
            DistanceAndId{.id = static_cast<GingerNodeId>(v)});
      }
      if (metric == TurboQuantDistanceMetric::kSquaredL2Distance) {
        SquaredL2Distance(static_cast<GingerNodeId>(u),
                          absl::MakeSpan(all_candidates),
                          data_7bit.float_dataset);
      } else {
        DotProductDistance(static_cast<GingerNodeId>(u),
                           absl::MakeSpan(all_candidates),
                           data_7bit.float_dataset);
      }
      absl::c_partial_sort(all_candidates, all_candidates.begin() + kK);
      for (int i = 0; i < kK; ++i) {
        ground_truth[u].insert(all_candidates[i].id);
      }
    }

    // 4-bit only (overretrieve_factor = 1, float_dataset = nullptr).
    std::vector<std::vector<DistanceAndId>> graph_4bit(kNumPoints);
    ASSERT_THAT(BuildPipnnKnnGraph(
                    data_7bit.points, data_7bit.quantizer, config, kK,
                    /*overretrieve_factor=*/1,
                    [&](GingerNodeId u, std::vector<DistanceAndId> neighbors) {
                      graph_4bit[u] = std::move(neighbors);
                    },
                    /*float_dataset=*/nullptr),
                IsOk());

    // 7-bit extra-precision reranking (overretrieve_factor = 4,
    // float_dataset = nullptr).
    std::vector<std::vector<DistanceAndId>> graph_7bit(kNumPoints);
    ASSERT_THAT(BuildPipnnKnnGraph(
                    data_7bit.points, data_7bit.quantizer, config, kK,
                    /*overretrieve_factor=*/4,
                    [&](GingerNodeId u, std::vector<DistanceAndId> neighbors) {
                      graph_7bit[u] = std::move(neighbors);
                    },
                    /*float_dataset=*/nullptr),
                IsOk());

    size_t hits_4bit = 0;
    size_t hits_7bit = 0;
    for (size_t u = 0; u < kNumPoints; ++u) {
      EXPECT_EQ(graph_7bit[u].size(), static_cast<size_t>(kK));
      EXPECT_TRUE(absl::c_is_sorted(graph_7bit[u]));
      for (const DistanceAndId& edge : graph_4bit[u]) {
        if (ground_truth[u].contains(edge.id)) ++hits_4bit;
      }
      for (const DistanceAndId& edge : graph_7bit[u]) {
        if (ground_truth[u].contains(edge.id)) ++hits_7bit;
      }
    }
    // Observed at seed 42: 4-bit recall ~0.90, 7-bit rerank recall ~0.99.
    const double recall_4bit =
        static_cast<double>(hits_4bit) / static_cast<double>(kNumPoints * kK);
    const double recall_7bit =
        static_cast<double>(hits_7bit) / static_cast<double>(kNumPoints * kK);
    EXPECT_GT(recall_7bit, 0.95)
        << "Recall too low for metric " << static_cast<int>(metric);
    EXPECT_GT(recall_7bit, recall_4bit + 0.05)
        << "7-bit extra-precision reranking should improve recall over 4-bit "
           "by more than 5 percentage points for metric "
        << static_cast<int>(metric);
  }
}

TEST_F(PipnnKnnTest, CancellationImmediateAndMidClustering) {
  SyntheticData data = MakeSyntheticDataset(200, 16);
  PipnnConfig config = MakePipnnConfig(/*max_leaf_size=*/15, /*num_passes=*/2);
  config.set_min_leaf_size(5);
  auto noop_cb = [](GingerNodeId, std::vector<DistanceAndId>) {};

  // Immediate cancel.
  EXPECT_THAT(
      BuildPipnnKnnGraph(data.points, data.quantizer, config, /*k=*/8,
                         /*overretrieve_factor=*/1, noop_cb,
                         /*float_dataset=*/nullptr, []() { return true; }),
      StatusIs(absl::StatusCode::kCancelled));

  // Mid-clustering cancel. The first check is at the start of the build and
  // the second one is the first inside partitioning.
  int call_count = 0;
  auto is_cancelled = [&call_count]() { return ++call_count == 2; };

  EXPECT_THAT(BuildPipnnKnnGraph(data.points, data.quantizer, config, /*k=*/8,
                                 /*overretrieve_factor=*/1, noop_cb,
                                 /*float_dataset=*/nullptr, is_cancelled),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(call_count, 2);

  // Null callback succeeds.
  std::function<bool()> null_callback = nullptr;
  EXPECT_THAT(BuildPipnnKnnGraph(data.points, data.quantizer, config, /*k=*/8,
                                 /*overretrieve_factor=*/1, noop_cb,
                                 /*float_dataset=*/nullptr, null_callback),
              IsOk());
}

}  // namespace
}  // namespace ginger
