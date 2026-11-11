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

#include "ginger/eval/recall.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/types.h"
#include "parlay/parallel.h"

namespace ginger {
namespace {

void ComputeDistances(GingerNodeId source_node_id,
                      absl::Span<DistanceAndId> distances) {
  std::vector<float> pos = {0, 1, 3};
  for (auto& dist_and_id : distances) {
    dist_and_id.distance = std::abs(pos[source_node_id] - pos[dist_and_id.id]);
  }
}

TEST(GraphRecallEstimatorTest, PerfectRecall) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 3;

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances));

  std::vector<DistanceAndId> neighbors0 = {{1.0, 1}};
  std::vector<DistanceAndId> neighbors1 = {{1.0, 0}};
  std::vector<DistanceAndId> neighbors2 = {{2.0, 1}};
  ASSERT_OK(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)));
  ASSERT_OK(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)));
  ASSERT_OK(estimator->ReportNeighborsFound(2, absl::MakeSpan(neighbors2)));

  EXPECT_DOUBLE_EQ(estimator->RecallEstimate(), 1.0);
}

TEST(GraphRecallEstimatorTest, ZeroRecall) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 3;

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances));

  // Neighbors with distance > k-th neighbor distance.
  // For node 0, 1-NN is 1 with dist 1.0.
  // For node 1, 1-NN is 0 with dist 1.0.
  // For node 2, 1-NN is 1 with dist 2.0.
  std::vector<DistanceAndId> neighbors0 = {{3.0, 2}};
  std::vector<DistanceAndId> neighbors1 = {{2.0, 2}};
  std::vector<DistanceAndId> neighbors2 = {{3.0, 0}};
  ASSERT_OK(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)));
  ASSERT_OK(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)));
  ASSERT_OK(estimator->ReportNeighborsFound(2, absl::MakeSpan(neighbors2)));

  EXPECT_DOUBLE_EQ(estimator->RecallEstimate(), 0.0);
}

TEST(GraphRecallEstimatorTest, PartialRecall) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 3;

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances));

  // For node 0, 1-NN is 1 with dist 1.0. If we report 2 with dist 3.0, recall
  // for 0 is 0.
  std::vector<DistanceAndId> neighbors0 = {{3.0, 2}};
  // For node 1, 1-NN is 0 with dist 1.0. If we report 0 with dist 1.0, recall
  // for 1 is 1.
  std::vector<DistanceAndId> neighbors1 = {{1.0, 0}};
  // For node 2, 1-NN is 1 with dist 2.0. If we report 1 with dist 2.0, recall
  // for 2 is 1.
  std::vector<DistanceAndId> neighbors2 = {{2.0, 1}};
  ASSERT_OK(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)));
  ASSERT_OK(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)));
  ASSERT_OK(estimator->ReportNeighborsFound(2, absl::MakeSpan(neighbors2)));

  EXPECT_DOUBLE_EQ(estimator->RecallEstimate(), 2.0 / 3.0);
}

TEST(GraphRecallEstimatorTest, FewerNeighborsReported) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 2;
  GingerNodeId num_recall_samples = 3;

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances));

  // For node 0, 2-NNs are 1 (dist 1.0) and 2 (dist 3.0). Reporting 1 neighbor.
  std::vector<DistanceAndId> neighbors0 = {{1.0, 1}};
  // For node 1, 2-NNs are 0 (dist 1.0) and 2 (dist 2.0). Reporting 1 neighbor.
  std::vector<DistanceAndId> neighbors1 = {{1.0, 0}};
  // For node 2, 2-NNs are 1 (dist 2.0) and 0 (dist 3.0). Reporting 1 neighbor.
  std::vector<DistanceAndId> neighbors2 = {{2.0, 1}};
  ASSERT_OK(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)));
  ASSERT_OK(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)));
  ASSERT_OK(estimator->ReportNeighborsFound(2, absl::MakeSpan(neighbors2)));

  EXPECT_DOUBLE_EQ(estimator->RecallEstimate(), 0.5);
}

TEST(GraphRecallEstimatorTest, NoNodes) {
  GingerNodeId num_nodes = 0;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 0;

  EXPECT_THAT(GraphRecallEstimator::Create(
                  num_nodes, num_neighbors, num_recall_samples,
                  [](GingerNodeId, absl::Span<DistanceAndId>) {}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GraphRecallEstimatorTest, NoRecallSamples) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 0;

  EXPECT_THAT(
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GraphRecallEstimatorTest, NoNeighbors) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 0;
  GingerNodeId num_recall_samples = 1;

  EXPECT_THAT(
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GraphRecallEstimatorTest, NumNeighborsTooLarge) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 2;
  GingerNodeId num_recall_samples = 1;
  ASSERT_OK(GraphRecallEstimator::Create(num_nodes, num_neighbors,
                                         num_recall_samples, ComputeDistances));

  num_neighbors = 3;
  EXPECT_THAT(
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  num_neighbors = 4;
  EXPECT_THAT(
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GraphRecallEstimatorTest, NumRecallSamplesTooLarge) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 3;
  ASSERT_OK(GraphRecallEstimator::Create(num_nodes, num_neighbors,
                                         num_recall_samples, ComputeDistances));

  num_recall_samples = 4;
  EXPECT_THAT(
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GraphRecallEstimatorTest, SampledRecall) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 1;

  double total_recall = 0;
  int num_runs = 1000;
  for (int i = 0; i < num_runs; ++i) {
    ASSERT_OK_AND_ASSIGN(
        auto estimator,
        GraphRecallEstimator::Create(num_nodes, num_neighbors,
                                     num_recall_samples, ComputeDistances));
    // Node 0: Wrong neighbor
    std::vector<DistanceAndId> neighbors0 = {{3.0, 2}};
    // Node 1: Correct neighbor
    std::vector<DistanceAndId> neighbors1 = {{1.0, 0}};
    // Node 2: Wrong neighbor
    std::vector<DistanceAndId> neighbors2 = {{3.0, 0}};
    ASSERT_OK(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)));
    ASSERT_OK(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)));
    ASSERT_OK(estimator->ReportNeighborsFound(2, absl::MakeSpan(neighbors2)));
    total_recall += estimator->RecallEstimate();
  }
  // Since num_recall_samples = 1, exactly one node is sampled each time.
  // Node 0 has recall 0.
  // Node 1 has recall 1.
  // Node 2 has recall 0.
  // The probability of picking node 0 is 1/3, node 1 is 1/3, and node 2 is 1/3.
  // The expected recall is (0 * 1/3) + (1 * 1/3) + (0 * 1/3) = 1/3.
  // The following checks should hold with probability roughly 1 - 10^-6.
  EXPECT_GE(total_recall / num_runs, 0.269);
  EXPECT_LE(total_recall / num_runs, 0.4);
}

TEST(GraphRecallEstimatorTest, ParallelPartialRecall) {
  GingerNodeId num_nodes = 100;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 50;

  auto compute_distances = [&](GingerNodeId source_node_id,
                               absl::Span<DistanceAndId> distances) {
    for (auto& dist_and_id : distances) {
      int diff = std::abs(static_cast<int>(source_node_id) -
                          static_cast<int>(dist_and_id.id));
      dist_and_id.distance = std::min(diff, static_cast<int>(num_nodes) - diff);
    }
  };

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   compute_distances));

  parlay::parallel_for(0, num_nodes, [&](GingerNodeId i) {
    std::vector<DistanceAndId> neighbors;
    if (i % 2 == 0) {
      // Correct neighbor for even nodes: (i + 1) % num_nodes
      GingerNodeId neighbor_id = (i + 1) % num_nodes;
      neighbors.push_back({1.0, neighbor_id});
    } else {
      // Incorrect neighbor for odd nodes: (i + 2) % num_nodes
      GingerNodeId neighbor_id = (i + 2) % num_nodes;
      neighbors.push_back({2.0, neighbor_id});
    }
    ASSERT_OK(estimator->ReportNeighborsFound(i, absl::MakeSpan(neighbors)));
  });

  // Since even nodes (50% of total) have perfect recall and odd nodes (50% of
  // total) have zero recall, and we sample 50% of the nodes, we expect the
  // recall to be close to 0.5. The following check should hold with
  // probability roughly 1 - 10^-6.
  EXPECT_NEAR(estimator->RecallEstimate(), 0.5, 0.23);
}

TEST(GraphRecallEstimatorTest, DistinctDistances) {
  GingerNodeId num_nodes = 5;
  GingerNodeId num_neighbors = 3;
  GingerNodeId num_recall_samples = 5;

  std::vector<float> pos = {0, 10, 100, 1000, 10000};
  auto compute_distances = [&](GingerNodeId source_node_id,
                               absl::Span<DistanceAndId> distances) {
    for (auto& dist_and_id : distances) {
      dist_and_id.distance =
          std::abs(pos[source_node_id] - pos[dist_and_id.id]);
    }
  };

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   compute_distances));

  // True Top-3 Neighbors:
  // 0: {1 (10), 2 (100), 3 (1000)}
  // 1: {0 (10), 2 (90), 3 (990)}
  // 2: {1 (90), 0 (100), 3 (900)}
  // 3: {2 (900), 1 (990), 0 (1000)}
  // 4: {3 (9000), 2 (9900), 1 (9990)}

  // Reported Neighbors:
  // Node 0: {1, 2, 4} -> 2/3 recall
  std::vector<DistanceAndId> neighbors0 = {{10.0, 1}, {100.0, 2}, {10000.0, 4}};
  // Node 1: {0, 2, 3} -> 3/3 recall
  std::vector<DistanceAndId> neighbors1 = {{10.0, 0}, {90.0, 2}, {990.0, 3}};
  // Node 2: {1, 0, 4} -> 2/3 recall
  std::vector<DistanceAndId> neighbors2 = {{90.0, 1}, {100.0, 0}, {9900.0, 4}};
  // Node 3: {2, 1, 4} -> 2/3 recall
  std::vector<DistanceAndId> neighbors3 = {{900.0, 2}, {990.0, 1}, {9000.0, 4}};
  // Node 4: {3, 2, 0} -> 2/3 recall (order doesn't matter for the set)
  std::vector<DistanceAndId> neighbors4 = {
      {9000.0, 3}, {9900.0, 2}, {10000.0, 0}};

  ASSERT_OK(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)));
  ASSERT_OK(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)));
  ASSERT_OK(estimator->ReportNeighborsFound(2, absl::MakeSpan(neighbors2)));
  ASSERT_OK(estimator->ReportNeighborsFound(3, absl::MakeSpan(neighbors3)));
  ASSERT_OK(estimator->ReportNeighborsFound(4, absl::MakeSpan(neighbors4)));

  // Total correct neighbors reported: 2 + 3 + 2 + 2 + 2 = 11
  // Total true neighbors: 3 * 5 = 15
  EXPECT_DOUBLE_EQ(estimator->RecallEstimate(), 11.0 / 15.0);
}

TEST(GraphRecallEstimatorTest, InvalidNodeId) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 3;

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances));

  std::vector<DistanceAndId> neighbors0 = {{1.0, 3}};  // Invalid node id 3
  EXPECT_THAT(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  std::vector<DistanceAndId> neighbors1 = {{1.0, -1}};  // Invalid node id -1
  EXPECT_THAT(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GraphRecallEstimatorTest, DuplicateNodeId) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 2;
  GingerNodeId num_recall_samples = 3;

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances));

  std::vector<DistanceAndId> neighbors0 = {{1.0, 1},
                                           {2.0, 1}};  // Duplicate id 1
  EXPECT_THAT(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GraphRecallEstimatorTest, DistancesAreRecomputed) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 3;

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   ComputeDistances));

  // Distances provided here are wrong, but ReportNeighborsFound should use
  // ComputeDistances to get the correct ones.
  // True distances: d(0,1)=1, d(1,0)=1, d(2,1)=2
  std::vector<DistanceAndId> neighbors0 = {{99.0, 1}};  // Correct neighbor id
  std::vector<DistanceAndId> neighbors1 = {{99.0, 0}};  // Correct neighbor id
  std::vector<DistanceAndId> neighbors2 = {{99.0, 1}};  // Correct neighbor id

  ASSERT_OK(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)));
  ASSERT_OK(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)));
  ASSERT_OK(estimator->ReportNeighborsFound(2, absl::MakeSpan(neighbors2)));

  // Since the IDs are correct, the recall should be perfect.
  EXPECT_DOUBLE_EQ(estimator->RecallEstimate(), 1.0);

  // Check that the distances were recomputed.
  EXPECT_EQ(neighbors0[0].distance, 1.0);
  EXPECT_EQ(neighbors1[0].distance, 1.0);
  EXPECT_EQ(neighbors2[0].distance, 2.0);
}

TEST(GraphRecallEstimatorTest, RecallWithDistanceTies) {
  GingerNodeId num_nodes = 3;
  GingerNodeId num_neighbors = 1;
  GingerNodeId num_recall_samples = 3;

  auto compute_distances_with_ties = [&](GingerNodeId source_node_id,
                                         absl::Span<DistanceAndId> distances) {
    std::vector<float> pos = {0, 1, 1};
    for (auto& dist_and_id : distances) {
      dist_and_id.distance =
          std::abs(pos[source_node_id] - pos[dist_and_id.id]);
    }
  };

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   compute_distances_with_ties));

  // For node 0, 1-NN is 1 or 2, both with dist 1.0.
  // If we report neighbors 1 and 2, recall should be 1/1 = 1.
  std::vector<DistanceAndId> neighbors0 = {{1.0, 1}, {1.0, 2}};
  // For node 1, 1-NN is 2 with dist 0.0.
  std::vector<DistanceAndId> neighbors1 = {{0.0, 2}};
  // For node 2, 1-NN is 1 with dist 0.0.
  std::vector<DistanceAndId> neighbors2 = {{0.0, 1}};
  ASSERT_OK(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)));
  ASSERT_OK(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)));
  ASSERT_OK(estimator->ReportNeighborsFound(2, absl::MakeSpan(neighbors2)));

  EXPECT_DOUBLE_EQ(estimator->RecallEstimate(), 1.0);
}

TEST(GraphRecallEstimatorTest, RecallWithDistanceTiesAndMissedCloserNeighbor) {
  GingerNodeId num_nodes = 5;
  GingerNodeId num_neighbors = 2;
  GingerNodeId num_recall_samples = 5;

  auto compute_distances_with_ties = [&](GingerNodeId source_node_id,
                                         absl::Span<DistanceAndId> distances) {
    std::vector<float> pos = {0, 1, 2, 2, 2};
    for (auto& dist_and_id : distances) {
      dist_and_id.distance =
          std::abs(pos[source_node_id] - pos[dist_and_id.id]);
    }
  };

  ASSERT_OK_AND_ASSIGN(
      auto estimator,
      GraphRecallEstimator::Create(num_nodes, num_neighbors, num_recall_samples,
                                   compute_distances_with_ties));

  // For node 0, pos=0. Others: 1 at 1, 2,3,4 at 2.
  // Dists for 0: d(0,1)=1, d(0,2)=2, d(0,3)=2, d(0,4)=2.
  // For k=2, 2-NN dist is 2. Node 1 is closer (dist 1).
  // So for node 0: k=2, d=2, c_true=1.
  // If we report {2,3}, h_c=0, h_t=2.
  // Recall for 0 = (0 + min(2, 2 - max(0,1))) / 2 = 1/2.
  std::vector<DistanceAndId> neighbors0 = {{2.0, 2}, {2.0, 3}};
  // For node 1, pos=1. d(1,0)=1, d(1,2)=1, d(1,3)=1, d(1,4)=1.
  // k=2, d=1, c_true=0. Report {0,2} -> h_c=0, h_t=2.
  // Recall for 1 = (0 + min(2, 2-max(0,0)))/2 = 1.
  std::vector<DistanceAndId> neighbors1 = {{1.0, 0}, {1.0, 2}};
  // For node 2, pos=2. d(2,3)=0, d(2,4)=0, d(2,1)=1, d(2,0)=2.
  // k=2, d=0, c_true=0. Report {3,1} -> h_c=0, h_t=1
  // Recall for 2 = (0 + min(1, 2-max(0,0)))/2 = 1/2.
  std::vector<DistanceAndId> neighbors2 = {{0.0, 3}, {1.0, 1}};
  // For node 3, pos=2. d(3,2)=0, d(3,4)=0, d(3,1)=1, d(3,0)=2.
  // k=2, d=0, c_true=0. Report {1,0} -> h_c=0, h_t=0
  // Recall for 3 = (0 + min(0, 2-max(0,0)))/2 = 0.
  std::vector<DistanceAndId> neighbors3 = {{1.0, 1}, {2.0, 0}};
  // For node 4, pos=2. d(4,2)=0, d(4,3)=0, d(4,1)=1, d(4,0)=2.
  // k=2, d=0, c_true=0. Report {1,0} -> h_c=0, h_t=0
  // Recall for 4 = (0 + min(0, 2-max(0,0)))/2 = 0.
  std::vector<DistanceAndId> neighbors4 = {{1.0, 1}, {2.0, 0}};

  ASSERT_OK(estimator->ReportNeighborsFound(0, absl::MakeSpan(neighbors0)));
  ASSERT_OK(estimator->ReportNeighborsFound(1, absl::MakeSpan(neighbors1)));
  ASSERT_OK(estimator->ReportNeighborsFound(2, absl::MakeSpan(neighbors2)));
  ASSERT_OK(estimator->ReportNeighborsFound(3, absl::MakeSpan(neighbors3)));
  ASSERT_OK(estimator->ReportNeighborsFound(4, absl::MakeSpan(neighbors4)));

  // Recall = (0.5 + 1 + 0.5 + 0 + 0) / 5 = 2 / 5 = 0.4.
  EXPECT_DOUBLE_EQ(estimator->RecallEstimate(), 0.4);
}

}  // namespace
}  // namespace ginger
