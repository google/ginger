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

#include "ginger/pipnn/pipnn_impl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/testing/parse_text_proto.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_test_stub.h"
#include "parlay/sequence.h"

namespace ginger::pipnn_internal {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::ginger::internal::ParallelSchedulerReferenceTest;
using ::testing::HasSubstr;

struct QuantizedData {
  TurboQuantCodec quantizer;
  TurboQuantDatasetSoA points;
};

QuantizedData MakeQuantizedData(size_t dim, size_t num_points) {
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<float> data(num_points * dim);
  for (float& v : data) {
    v = dist(rng);
  }
  std::vector<absl::Span<const float>> datapoints;
  datapoints.reserve(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    datapoints.push_back(absl::MakeConstSpan(data.data() + i * dim, dim));
  }
  TurboQuantCodec tq(dim, 42, TurboQuantDistanceMetric::kDotProductDistance);
  auto points_or = TurboQuantDatasetSoA::Create(
      tq, datapoints,
      TurboQuantDatasetOptions{
          .pad_to_cache_line_size = false,
          .quantization_bits = 4,
          .store_query_byte_sums = false,
          .normalized_datapoints = false,
          .scaling_storage_mode = ScalingStorageMode::kSeparateArray});
  ABSL_CHECK_OK(points_or);
  return {std::move(tq), *std::move(points_or)};
}

PipnnConfig MakeValidPartitionConfig() {
  return PARSE_TEXT_PROTO(R"pb(
    max_leaf_size: 32
    min_leaf_size: 8
    num_passes: 2
    seed: 123
    leader_selection {
      leader_sample_rate: 0.2
      max_num_leaders: 16
      top_level_leaders: 8
      fanout_sequence: [ 2, 1 ]
    }
  )pb");
}

class PipnnImplTest : public ParallelSchedulerReferenceTest {};

TEST_F(PipnnImplTest, ValidatesConfigParameters) {
  EXPECT_THAT(ValidatePipnnPartitioningConfig(MakeValidPartitionConfig()),
              IsOk());

  {
    PipnnConfig cfg = MakeValidPartitionConfig();
    cfg.set_num_passes(0);
    EXPECT_THAT(
        ValidatePipnnPartitioningConfig(cfg),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("num_passes")));
  }
  {
    PipnnConfig cfg = MakeValidPartitionConfig();
    cfg.set_max_leaf_size(3);
    EXPECT_THAT(ValidatePipnnPartitioningConfig(cfg),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("max_leaf_size")));
  }
  {
    PipnnConfig cfg = MakeValidPartitionConfig();
    cfg.set_min_leaf_size(0);
    EXPECT_THAT(ValidatePipnnPartitioningConfig(cfg),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("min_leaf_size")));
  }
  {
    PipnnConfig cfg = MakeValidPartitionConfig();
    cfg.set_max_leaf_size(20);
    cfg.set_min_leaf_size(11);
    EXPECT_THAT(ValidatePipnnPartitioningConfig(cfg),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("min_leaf_size")));
  }
  {
    PipnnConfig cfg = MakeValidPartitionConfig();
    cfg.mutable_leader_selection()->set_leader_sample_rate(0.0f);
    EXPECT_THAT(ValidatePipnnPartitioningConfig(cfg),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("leader_sample_rate")));
  }
  {
    PipnnConfig cfg = MakeValidPartitionConfig();
    cfg.mutable_leader_selection()->set_max_num_leaders(2);
    EXPECT_THAT(ValidatePipnnPartitioningConfig(cfg),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("max_num_leaders")));
  }
  {
    PipnnConfig cfg = MakeValidPartitionConfig();
    cfg.mutable_leader_selection()->set_top_level_leaders(0);
    EXPECT_THAT(ValidatePipnnPartitioningConfig(cfg),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("top_level_leaders")));
  }
  {
    PipnnConfig cfg = MakeValidPartitionConfig();
    cfg.mutable_leader_selection()->clear_fanout_sequence();
    EXPECT_THAT(ValidatePipnnPartitioningConfig(cfg),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("fanout_sequence")));
  }
  {
    PipnnConfig cfg = MakeValidPartitionConfig();
    cfg.mutable_leader_selection()->set_fanout_sequence(0, 0);
    EXPECT_THAT(
        ValidatePipnnPartitioningConfig(cfg),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("fanout")));
  }
}

TEST_F(PipnnImplTest, SingleLeafWhenDatasetFitsInMaxLeafSize) {
  const size_t num_points = 20;
  QuantizedData qd = MakeQuantizedData(/*dim=*/32, num_points);
  PipnnConfig cfg = MakeValidPartitionConfig();
  cfg.set_max_leaf_size(32);
  cfg.set_print_stats(true);

  std::vector<std::vector<GingerNodeId>> leaves;
  ASSERT_THAT(PartitionPointsIntoLeaves(
                  qd.points, qd.quantizer, cfg, /*pass=*/0,
                  [&](absl::Span<const GingerNodeId> leaf) {
                    leaves.emplace_back(leaf.begin(), leaf.end());
                  },
                  /*on_chunk_done=*/[]() {},
                  /*is_cancelled=*/[]() { return false; }),
              IsOk());

  ASSERT_EQ(leaves.size(), 1);
  EXPECT_EQ(leaves[0].size(), num_points);

  ASSERT_OK_AND_ASSIGN(
      parlay::sequence<parlay::sequence<GingerNodeId>> collected_leaves,
      PartitionPointsIntoLeaves(qd.points, qd.quantizer, cfg, /*pass=*/0));
  ASSERT_EQ(collected_leaves.size(), 1);
  EXPECT_EQ(collected_leaves[0].size(), num_points);
}

TEST_F(PipnnImplTest, ClampsTopLevelLeadersToClusterSize) {
  // More top-level leaders than points, but too many points for a single leaf.
  const size_t num_points = 40;
  QuantizedData qd = MakeQuantizedData(/*dim=*/32, num_points);
  PipnnConfig cfg = MakeValidPartitionConfig();
  cfg.mutable_leader_selection()->set_top_level_leaders(64);

  ASSERT_OK_AND_ASSIGN(
      parlay::sequence<parlay::sequence<GingerNodeId>> leaves,
      PartitionPointsIntoLeaves(qd.points, qd.quantizer, cfg, /*pass=*/0));
  std::vector<bool> covered(num_points, false);
  for (const auto& leaf : leaves) {
    EXPECT_LE(leaf.size(), static_cast<size_t>(cfg.max_leaf_size()));
    for (const GingerNodeId id : leaf) covered[id] = true;
  }
  EXPECT_THAT(covered, ::testing::Each(true));
}

TEST_F(PipnnImplTest, PartitionsIntoBoundedLeavesAndCoversAllPoints) {
  const size_t num_points = 120;
  QuantizedData qd = MakeQuantizedData(/*dim=*/32, num_points);
  PipnnConfig cfg = MakeValidPartitionConfig();
  cfg.set_max_leaf_size(24);
  cfg.set_min_leaf_size(6);
  cfg.set_print_stats(true);

  int max_leaves_per_point = 1;
  for (int f : cfg.leader_selection().fanout_sequence()) {
    max_leaves_per_point *= f;
  }

  absl::Mutex mu;
  std::vector<std::vector<GingerNodeId>> callback_leaves;
  std::vector<int> leaf_counts(num_points, 0);
  ASSERT_THAT(PartitionPointsIntoLeaves(
                  qd.points, qd.quantizer, cfg, /*pass=*/0,
                  [&](absl::Span<const GingerNodeId> leaf) {
                    EXPECT_GT(leaf.size(), 0);
                    EXPECT_LE(leaf.size(),
                              static_cast<size_t>(cfg.max_leaf_size()));
                    absl::MutexLock lock(mu);
                    callback_leaves.emplace_back(leaf.begin(), leaf.end());
                    for (GingerNodeId id : leaf) {
                      ++leaf_counts[id];
                    }
                  },
                  /*on_chunk_done=*/[]() {},
                  /*is_cancelled=*/[]() { return false; }),
              IsOk());

  ASSERT_OK_AND_ASSIGN(
      parlay::sequence<parlay::sequence<GingerNodeId>> collected_leaves,
      PartitionPointsIntoLeaves(qd.points, qd.quantizer, cfg, /*pass=*/0));

  EXPECT_GT(callback_leaves.size(), 1);
  EXPECT_EQ(collected_leaves.size(), callback_leaves.size());
  size_t num_points_at_max_leaves = 0;
  for (size_t u = 0; u < num_points; ++u) {
    EXPECT_GE(leaf_counts[u], 1);
    EXPECT_LE(leaf_counts[u], max_leaves_per_point);
    if (leaf_counts[u] == max_leaves_per_point) {
      ++num_points_at_max_leaves;
    }
  }
  EXPECT_GE(num_points_at_max_leaves, num_points / 2);
}

TEST_F(PipnnImplTest, LargeFanoutCoversAllPoints) {
  // Fanouts above 10 take the ManyToManyTurboQuantTopK fallback path.
  constexpr int kFanout = 12;
  const size_t num_points = 120;
  QuantizedData qd = MakeQuantizedData(/*dim=*/32, num_points);
  PipnnConfig cfg = MakeValidPartitionConfig();
  cfg.set_max_leaf_size(24);
  cfg.set_min_leaf_size(6);
  cfg.mutable_leader_selection()->set_top_level_leaders(kFanout);
  cfg.mutable_leader_selection()->clear_fanout_sequence();
  cfg.mutable_leader_selection()->add_fanout_sequence(kFanout);

  ASSERT_OK_AND_ASSIGN(
      parlay::sequence<parlay::sequence<GingerNodeId>> leaves,
      PartitionPointsIntoLeaves(qd.points, qd.quantizer, cfg, /*pass=*/0));

  std::vector<int> leaf_counts(num_points, 0);
  for (const auto& leaf : leaves) {
    EXPECT_LE(leaf.size(), static_cast<size_t>(cfg.max_leaf_size()));
    for (GingerNodeId id : leaf) {
      ++leaf_counts[id];
    }
  }
  for (size_t u = 0; u < num_points; ++u) {
    EXPECT_GE(leaf_counts[u], 1);
    EXPECT_LE(leaf_counts[u], kFanout);
  }
}

TEST_F(PipnnImplTest, SplitsStuckClusterRandomly) {
  const size_t dim = 32;
  const size_t num_points = 20;
  std::vector<float> identical_data(num_points * dim, 0.1f);
  std::vector<absl::Span<const float>> datapoints(
      num_points, absl::MakeConstSpan(identical_data.data(), dim));
  TurboQuantCodec tq(dim, 42, TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(
      auto points,
      TurboQuantDatasetSoA::Create(
          tq, datapoints,
          TurboQuantDatasetOptions{
              .pad_to_cache_line_size = false,
              .quantization_bits = 4,
              .store_query_byte_sums = false,
              .normalized_datapoints = false,
              .scaling_storage_mode = ScalingStorageMode::kSeparateArray}));

  PipnnConfig cfg = MakeValidPartitionConfig();
  cfg.set_max_leaf_size(8);
  cfg.set_min_leaf_size(4);
  cfg.set_print_stats(true);
  cfg.mutable_leader_selection()->set_top_level_leaders(3);
  cfg.mutable_leader_selection()->set_max_num_leaders(3);
  cfg.mutable_leader_selection()->clear_fanout_sequence();
  cfg.mutable_leader_selection()->add_fanout_sequence(1);

  absl::Mutex mu;
  std::vector<int> leaf_counts(num_points, 0);
  ASSERT_THAT(PartitionPointsIntoLeaves(
                  points, tq, cfg, /*pass=*/0,
                  [&](absl::Span<const GingerNodeId> leaf) {
                    EXPECT_LE(leaf.size(),
                              static_cast<size_t>(cfg.max_leaf_size()));
                    absl::MutexLock lock(mu);
                    for (GingerNodeId id : leaf) {
                      ++leaf_counts[id];
                    }
                  },
                  /*on_chunk_done=*/[]() {},
                  /*is_cancelled=*/[]() { return false; }),
              IsOk());
  for (size_t u = 0; u < num_points; ++u) {
    EXPECT_EQ(leaf_counts[u], 1);
  }

  ASSERT_OK_AND_ASSIGN(
      parlay::sequence<parlay::sequence<GingerNodeId>> collected_leaves,
      PartitionPointsIntoLeaves(points, tq, cfg, /*pass=*/0));
  std::vector<int> collected_counts(num_points, 0);
  for (const auto& leaf : collected_leaves) {
    EXPECT_LE(leaf.size(), static_cast<size_t>(cfg.max_leaf_size()));
    for (GingerNodeId id : leaf) {
      ++collected_counts[id];
    }
  }
  for (size_t u = 0; u < num_points; ++u) {
    EXPECT_EQ(collected_counts[u], 1);
  }
}

TEST_F(PipnnImplTest, DeterministicWithinPassAndVariesAcrossPasses) {
  QuantizedData qd = MakeQuantizedData(/*dim=*/32, /*num_points=*/120);
  PipnnConfig cfg = MakeValidPartitionConfig();

  auto collect_leaves = [&](int pass) {
    auto collected =
        PartitionPointsIntoLeaves(qd.points, qd.quantizer, cfg, pass);
    ABSL_CHECK_OK(collected);
    std::vector<std::vector<GingerNodeId>> leaves;
    leaves.reserve(collected->size());
    for (const parlay::sequence<GingerNodeId>& leaf : *collected) {
      leaves.emplace_back(leaf.begin(), leaf.end());
    }
    std::sort(leaves.begin(), leaves.end());
    return leaves;
  };

  auto pass0_a = collect_leaves(0);
  auto pass0_b = collect_leaves(0);
  auto pass1 = collect_leaves(1);

  EXPECT_EQ(pass0_a, pass0_b);
  EXPECT_NE(pass0_a, pass1);
}

TEST_F(PipnnImplTest, RespectsCancellation) {
  QuantizedData qd = MakeQuantizedData(/*dim=*/32, /*num_points=*/120);
  PipnnConfig cfg = MakeValidPartitionConfig();
  auto noop_leaf = [](absl::Span<const GingerNodeId>) {};
  auto noop_chunk = []() {};

  EXPECT_THAT(PartitionPointsIntoLeaves(qd.points, qd.quantizer, cfg,
                                        /*pass=*/0, noop_leaf, noop_chunk,
                                        /*is_cancelled=*/[]() { return true; }),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(
      PartitionPointsIntoLeaves(qd.points, qd.quantizer, cfg, /*pass=*/0,
                                /*is_cancelled=*/[]() { return true; }),
      StatusIs(absl::StatusCode::kCancelled));
}

TEST(ChunkBudgetTest, FractionOfTotalOrUnlimited) {
  EXPECT_EQ(ChunkBudget(1000, 0.25f), 250);
  EXPECT_EQ(ChunkBudget(1000, 1.0f), std::numeric_limits<size_t>::max());
  EXPECT_EQ(ChunkBudget(1000, 1.5f), std::numeric_limits<size_t>::max());
}

TEST_F(PipnnImplTest, ChunksRespectBudgetAndMatchPhasedLeaves) {
  const size_t num_points = 2000;
  QuantizedData qd = MakeQuantizedData(/*dim=*/32, num_points);
  PipnnConfig cfg = MakeValidPartitionConfig();
  cfg.mutable_leader_selection()->set_fanout_sequence(1, 2);
  const size_t max_leaf_points = num_points * 2 * 2;

  ASSERT_OK_AND_ASSIGN(
      parlay::sequence<parlay::sequence<GingerNodeId>> collected,
      PartitionPointsIntoLeaves(qd.points, qd.quantizer, cfg, /*pass=*/0));
  std::vector<std::vector<GingerNodeId>> phased_leaves;
  for (const parlay::sequence<GingerNodeId>& leaf : collected) {
    phased_leaves.emplace_back(leaf.begin(), leaf.end());
  }
  std::sort(phased_leaves.begin(), phased_leaves.end());

  for (float fraction : {0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 1.0f}) {
    SCOPED_TRACE(fraction);
    cfg.set_robust_prune_chunk_fraction(fraction);
    const size_t budget = ChunkBudget(max_leaf_points, fraction);
    std::atomic<int> in_flight = 0;
    std::atomic<size_t> chunk_points = 0;
    int num_chunks = 0;
    absl::Mutex mu;
    std::vector<std::vector<GingerNodeId>> leaves;
    ASSERT_THAT(PartitionPointsIntoLeaves(
                    qd.points, qd.quantizer, cfg, /*pass=*/0,
                    [&](absl::Span<const GingerNodeId> leaf) {
                      ++in_flight;
                      chunk_points += leaf.size();
                      {
                        absl::MutexLock lock(mu);
                        leaves.emplace_back(leaf.begin(), leaf.end());
                      }
                      --in_flight;
                    },
                    [&]() {
                      EXPECT_EQ(in_flight, 0);
                      EXPECT_LE(chunk_points,
                                std::max<size_t>(budget, cfg.max_leaf_size()));
                      chunk_points = 0;
                      ++num_chunks;
                    },
                    /*is_cancelled=*/[]() { return false; }),
                IsOk());

    if (fraction < 1.0f) {
      EXPECT_GT(num_chunks, 1);
    } else {
      EXPECT_EQ(num_chunks, 1);
    }
    std::sort(leaves.begin(), leaves.end());
    EXPECT_EQ(leaves, phased_leaves);
  }
}

}  // namespace
}  // namespace ginger::pipnn_internal
