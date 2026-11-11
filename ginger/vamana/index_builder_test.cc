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

#include "ginger/vamana/index_builder.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "ginger/build_progress.h"
#include "ginger/graph/graph_stats.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/types.h"
#include "ginger/vamana/config.pb.h"

namespace ginger::internal {
namespace {

using ::testing::Pair;

TEST(NumReachableNodesTest, SingleNodeGraph) {
  VamanaIndex graph(1, 0);
  EXPECT_THAT(internal::NumReachableNodes(0, graph), Pair(1, 0));
}

TEST(NumReachableNodesTest, DisconnectedGraph) {
  VamanaIndex graph(3, 1);
  graph.SetNeighbors(0, {1});
  EXPECT_THAT(internal::NumReachableNodes(0, graph), Pair(2, 1));
  EXPECT_THAT(internal::NumReachableNodes(1, graph), Pair(1, 0));
  EXPECT_THAT(internal::NumReachableNodes(2, graph), Pair(1, 0));
}

TEST(NumReachableNodesTest, FullyConnectedGraph) {
  VamanaIndex graph(3, 2);
  graph.SetNeighbors(0, {1, 2});
  graph.SetNeighbors(1, {0, 2});
  graph.SetNeighbors(2, {0, 1});
  EXPECT_THAT(internal::NumReachableNodes(0, graph), Pair(3, 1));
  EXPECT_THAT(internal::NumReachableNodes(1, graph), Pair(3, 1));
  EXPECT_THAT(internal::NumReachableNodes(2, graph), Pair(3, 1));
}

TEST(NumReachableNodesTest, LineGraph) {
  VamanaIndex graph(4, 2);
  graph.SetNeighbors(0, {1});
  graph.SetNeighbors(1, {0, 2});
  graph.SetNeighbors(2, {1, 3});
  graph.SetNeighbors(3, {2});
  EXPECT_THAT(internal::NumReachableNodes(0, graph), Pair(4, 3));
  EXPECT_THAT(internal::NumReachableNodes(1, graph), Pair(4, 2));
  EXPECT_THAT(internal::NumReachableNodes(3, graph), Pair(4, 3));
}

TEST(NumReachableNodesTest, GraphWithCycle) {
  VamanaIndex graph(3, 2);
  graph.SetNeighbors(0, {1});
  graph.SetNeighbors(1, {2});
  graph.SetNeighbors(2, {0});
  EXPECT_THAT(internal::NumReachableNodes(0, graph), Pair(3, 2));
  EXPECT_THAT(internal::NumReachableNodes(1, graph), Pair(3, 2));
  EXPECT_THAT(internal::NumReachableNodes(2, graph), Pair(3, 2));
}

TEST(NumReachableNodesTest, StarGraph) {
  VamanaIndex graph(5, 4);
  graph.SetNeighbors(0, {1, 2, 3, 4});
  graph.SetNeighbors(1, {0});
  graph.SetNeighbors(2, {0});
  graph.SetNeighbors(3, {0});
  graph.SetNeighbors(4, {0});
  EXPECT_THAT(internal::NumReachableNodes(0, graph), Pair(5, 1));
  EXPECT_THAT(internal::NumReachableNodes(1, graph), Pair(5, 2));
}

}  // namespace
}  // namespace ginger::internal

namespace ginger {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(BuildVamanaIndexTest, ProgressCallbackWorks) {
  VamanaConfig config;
  config.set_max_degree(4);
  config.set_seed(12345);
  auto* pass1 = config.add_passes();
  pass1->set_max_beam_width(8);
  pass1->mutable_prune_config()->set_alpha(1.2f);
  auto* pass2 = config.add_passes();
  pass2->set_max_beam_width(8);
  pass2->mutable_prune_config()->set_alpha(1.2f);

  constexpr GingerNodeId kNumPoints = 200;

  auto compute_distances = [](GingerNodeId query_id,
                              absl::Span<DistanceAndId> candidates) {
    for (auto& candidate : candidates) {
      candidate.distance = std::abs(static_cast<float>(query_id) -
                                    static_cast<float>(candidate.id));
    }
  };

  std::vector<IndexBuildProgress> progress_history;
  auto progress_callback = [&](const IndexBuildProgress& progress) {
    progress_history.push_back(progress);
  };

  auto index_or = BuildVamanaIndex(kNumPoints, config, compute_distances,
                                   progress_callback);
  ASSERT_TRUE(index_or.ok());

  // Since kNumPoints = 200, batches will be generated.
  // Let's verify we received multiple progress reports.
  ASSERT_GT(progress_history.size(), 1);

  GingerNodeId last_completed = 0;
  absl::Duration last_elapsed = absl::ZeroDuration();

  for (size_t i = 0; i < progress_history.size(); ++i) {
    const auto& progress = progress_history[i];
    EXPECT_EQ(progress.num_points, kNumPoints);
    EXPECT_EQ(progress.total_passes, 2);
    EXPECT_GE(progress.current_pass, 0);
    EXPECT_LT(progress.current_pass, 2);
    EXPECT_GT(progress.total_completed_points, last_completed);
    EXPECT_GE(progress.elapsed_time, last_elapsed);
    EXPECT_GT(progress.last_batch_size, 0);
    EXPECT_GE(progress.last_batch_duration, absl::ZeroDuration());

    last_completed = progress.total_completed_points;
    last_elapsed = progress.elapsed_time;
  }

  // The final report should show all work done.
  // total work = kNumPoints * total_passes = 200 * 2 = 400.
  EXPECT_EQ(last_completed, 400);
}

TEST(BuildVamanaIndexTest, RejectsReverseRngBackfill) {
  VamanaConfig config;
  config.set_max_degree(4);
  auto* pass = config.add_passes();
  pass->set_max_beam_width(8);
  pass->mutable_prune_config()->set_alpha(1.2f);
  pass->mutable_prune_config()->set_reverse_rng_backfill(true);

  auto compute_distances = [](GingerNodeId /*query_id*/,
                              absl::Span<DistanceAndId> candidates) {
    for (auto& candidate : candidates) candidate.distance = 1.0f;
  };

  EXPECT_THAT(BuildVamanaIndex(/*num_points=*/16, config, compute_distances),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("reverse_rng_backfill")));
}

TEST(BuildVamanaIndexTest, RejectsAlphaBelowOne) {
  VamanaConfig config;
  config.set_max_degree(4);
  auto* pass = config.add_passes();
  pass->set_max_beam_width(8);
  pass->mutable_prune_config()->set_alpha(0.9f);

  auto compute_distances = [](GingerNodeId /*query_id*/,
                              absl::Span<DistanceAndId> candidates) {
    for (auto& candidate : candidates) candidate.distance = 1.0f;
  };

  EXPECT_THAT(BuildVamanaIndex(/*num_points=*/16, config, compute_distances),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("prune_config.alpha")));
}

TEST(BuildVamanaIndexTest, RejectsNonPositiveNumPoints) {
  VamanaConfig config;
  config.set_max_degree(4);
  auto* pass = config.add_passes();
  pass->set_max_beam_width(8);
  pass->mutable_prune_config()->set_alpha(1.2f);

  auto compute_distances = [](GingerNodeId /*query_id*/,
                              absl::Span<DistanceAndId> /*candidates*/) {};

  EXPECT_THAT(BuildVamanaIndex(/*num_points=*/0, config, compute_distances),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("num_points must be positive")));
  EXPECT_THAT(BuildVamanaIndex(/*num_points=*/-1, config, compute_distances),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("num_points must be positive")));
}
}  // namespace
}  // namespace ginger
