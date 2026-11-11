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

#include "ginger/reordering/recursive_graph_partitioning_impl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/algorithm/container.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"

namespace ginger::internal {
namespace {

using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Ge;
using ::testing::IsEmpty;
using ::testing::Lt;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::TestWithParam;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

ginger::CSRGraph CreateCSRGraph(
    const std::vector<std::vector<GingerNodeId>>& adjacency_list) {
  const size_t num_nodes = adjacency_list.size();
  size_t num_arcs = 0;
  for (const auto& neighbors : adjacency_list) {
    num_arcs += neighbors.size();
  }

  std::vector<int64_t> offsets = {0};
  std::vector<GingerNodeId> targets;
  targets.reserve(num_arcs);
  for (const auto& neighbors : adjacency_list) {
    for (const auto neighbor : neighbors) {
      EXPECT_LT(neighbor, num_nodes);
      targets.push_back(neighbor);
    }
    offsets.push_back(targets.size());
  }
  return ginger::CSRGraph(std::move(offsets), std::move(targets));
}

TEST(CSRGraphTest, SingleNodeGraph) {
  rgp::CSRGraph graph(1, 0);
  std::fill(graph.Offsets().begin(), graph.Offsets().end(), 0);

  EXPECT_EQ(graph.NumNodes(), 1);
  EXPECT_EQ(graph.NumArcs(), 0);
  EXPECT_EQ(graph.Degree(0), 0);
  EXPECT_THAT(graph.Neighbors(0), IsEmpty());
}

TEST(CSRGraphTest, GraphWithNodesAndEdges) {
  std::vector<rgp::NodeId> offsets = {0, 1, 3, 4};
  std::vector<rgp::EdgeId> targets = {1, 0, 2, 1};
  std::vector<GingerNodeId> mapping = {10, 11, 12};

  rgp::CSRGraph graph(3, 4);
  absl::c_copy(offsets, graph.Offsets().begin());
  absl::c_copy(targets, graph.Targets().begin());
  absl::c_copy(mapping, graph.Mapping().begin());

  ASSERT_EQ(graph.NumNodes(), 3);
  ASSERT_EQ(graph.NumArcs(), 4);

  EXPECT_EQ(graph.Degree(0), 1);
  EXPECT_EQ(graph.Degree(1), 2);
  EXPECT_EQ(graph.Degree(2), 1);

  EXPECT_THAT(graph.Neighbors(0), ElementsAre(1));
  EXPECT_THAT(graph.Neighbors(1), ElementsAre(0, 2));
  EXPECT_THAT(graph.Neighbors(2), ElementsAre(1));

  EXPECT_EQ(graph.OriginalId(0), 10);
  EXPECT_EQ(graph.OriginalId(1), 11);
  EXPECT_EQ(graph.OriginalId(2), 12);
}

TEST(CopyGraphTest, GraphWithNodesButNoEdges) {
  ginger::CSRGraph ginger_graph({0, 0, 0, 0}, {});
  rgp::CSRGraph graph = CopyGraph(ginger_graph);

  EXPECT_EQ(graph.NumNodes(), 3);
  EXPECT_EQ(graph.NumArcs(), 0);

  EXPECT_EQ(graph.Degree(0), 0);
  EXPECT_EQ(graph.Degree(1), 0);
  EXPECT_EQ(graph.Degree(2), 0);

  EXPECT_THAT(graph.Neighbors(0), IsEmpty());
  EXPECT_THAT(graph.Neighbors(1), IsEmpty());
  EXPECT_THAT(graph.Neighbors(2), IsEmpty());

  EXPECT_EQ(graph.OriginalId(0), 0);
  EXPECT_EQ(graph.OriginalId(1), 1);
  EXPECT_EQ(graph.OriginalId(2), 2);
}

TEST(CopyGraphTest, GraphWithOneDirectedEdge) {
  ginger::CSRGraph ginger_graph({0, 1, 1}, {1});
  rgp::CSRGraph graph = CopyGraph(ginger_graph);

  EXPECT_EQ(graph.NumNodes(), 2);
  EXPECT_EQ(graph.NumArcs(), 2);

  EXPECT_EQ(graph.Degree(0), 1);
  EXPECT_EQ(graph.Degree(1), 1);

  EXPECT_THAT(graph.Neighbors(0), UnorderedElementsAre(1));
  EXPECT_THAT(graph.Neighbors(1), UnorderedElementsAre(0));

  EXPECT_EQ(graph.OriginalId(0), 0);
  EXPECT_EQ(graph.OriginalId(1), 1);
}

TEST(CopyGraphTest, GraphWithUndirectedEdge) {
  ginger::CSRGraph ginger_graph = CreateCSRGraph({{1}, {0}});
  rgp::CSRGraph graph = CopyGraph(ginger_graph);

  EXPECT_EQ(graph.NumNodes(), 2);
  EXPECT_EQ(graph.NumArcs(), 4);

  EXPECT_EQ(graph.Degree(0), 2);
  EXPECT_EQ(graph.Degree(1), 2);

  EXPECT_THAT(graph.Neighbors(0), UnorderedElementsAre(1, 1));
  EXPECT_THAT(graph.Neighbors(1), UnorderedElementsAre(0, 0));

  EXPECT_EQ(graph.OriginalId(0), 0);
  EXPECT_EQ(graph.OriginalId(1), 1);
}

TEST(CopyGraphTest, DirectedCycle) {
  ginger::CSRGraph ginger_graph({0, 1, 2, 3}, {1, 2, 0});
  rgp::CSRGraph graph = CopyGraph(ginger_graph);

  EXPECT_EQ(graph.NumNodes(), 3);
  EXPECT_EQ(graph.NumArcs(), 6);

  EXPECT_EQ(graph.Degree(0), 2);
  EXPECT_EQ(graph.Degree(1), 2);
  EXPECT_EQ(graph.Degree(2), 2);

  EXPECT_THAT(graph.Neighbors(0), UnorderedElementsAre(1, 2));
  EXPECT_THAT(graph.Neighbors(1), UnorderedElementsAre(0, 2));
  EXPECT_THAT(graph.Neighbors(2), UnorderedElementsAre(0, 1));

  EXPECT_EQ(graph.OriginalId(0), 0);
  EXPECT_EQ(graph.OriginalId(1), 1);
  EXPECT_EQ(graph.OriginalId(2), 2);
}

TEST(CopyGraphTest, PreservesOutgoingOrderAndSortsReverseNeighbors) {
  constexpr size_t kNumNodes = 1000;
  std::vector<std::vector<GingerNodeId>> adjacency_list(kNumNodes);
  // Node 0 has outgoing edges in unsorted order: 3, 1.
  adjacency_list[0] = {3, 1};
  // Every node u in [1, kNumNodes) has a directed edge to node 0.
  for (size_t u = 1; u < kNumNodes; ++u) {
    adjacency_list[u] = {0};
  }

  ginger::CSRGraph ginger_graph = CreateCSRGraph(adjacency_list);
  rgp::CSRGraph graph = CopyGraph(ginger_graph);

  ASSERT_EQ(graph.NumNodes(), kNumNodes);
  ASSERT_EQ(graph.Degree(0), 2 + (kNumNodes - 1));

  absl::Span<const rgp::NodeId> neighbors0 = graph.Neighbors(0);
  EXPECT_THAT(neighbors0.subspan(0, 2), ElementsAre(3, 1));

  std::vector<rgp::NodeId> expected_reverse(kNumNodes - 1);
  std::iota(expected_reverse.begin(), expected_reverse.end(), 1);
  EXPECT_THAT(neighbors0.subspan(2), ElementsAreArray(expected_reverse));

  EXPECT_THAT(graph.Neighbors(1), ElementsAre(0, 0));
  EXPECT_THAT(graph.Neighbors(2), ElementsAre(0));
  EXPECT_THAT(graph.Neighbors(3), ElementsAre(0, 0));
}

TEST(PartitionTest, EmptyGraph) {
  RecursiveGraphPartitioningReorderingConfig config;
  rgp::CSRGraph graph(0, 0);

  std::vector<rgp::PartitionId> partition =
      Partition(config, graph, /*num_subgraphs=*/2, /*parallelism=*/1);
  EXPECT_THAT(partition, IsEmpty());
}

TEST(PartitionTest, GraphWithNodesButNoEdges) {
  RecursiveGraphPartitioningReorderingConfig config;
  rgp::CSRGraph graph(3, 0);
  std::fill(graph.Offsets().begin(), graph.Offsets().end(), 0);

  std::vector<rgp::PartitionId> partition =
      Partition(config, graph, /*num_subgraphs=*/2, /*parallelism=*/1);
  EXPECT_THAT(partition, SizeIs(3));
  EXPECT_THAT(partition, Each(AllOf(Ge(0), Lt(2))));
}

TEST(PartitionTest, MultiplePartitions) {
  RecursiveGraphPartitioningReorderingConfig config;

  std::vector<rgp::EdgeId> offsets = {0, 1, 3, 4};
  std::vector<rgp::NodeId> targets = {1, 0, 2, 1};
  std::vector<GingerNodeId> mapping = {10, 11, 12};

  rgp::CSRGraph graph(3, 4);
  absl::c_copy(offsets, graph.Offsets().begin());
  absl::c_copy(targets, graph.Targets().begin());
  absl::c_copy(mapping, graph.Mapping().begin());

  std::vector<rgp::PartitionId> partition =
      Partition(config, graph, /*num_subgraphs=*/2, /*parallelism=*/1);
  EXPECT_THAT(partition, SizeIs(3));
  EXPECT_THAT(partition, Each(AllOf(Ge(0), Lt(2))));
}

struct ExtractSubgraphsTestCase {
  bool parallel = false;
};
using ExtractSubgraphsTest = TestWithParam<ExtractSubgraphsTestCase>;

TEST_P(ExtractSubgraphsTest, EmptyGraphNonZeroSubgraphs) {
  const ExtractSubgraphsTestCase& test_case = GetParam();
  RecursiveGraphPartitioningReorderingConfig config;

  rgp::CSRGraph graph(0, 0);
  std::vector<rgp::PartitionId> partition = {};
  std::vector<rgp::CSRGraph> subgraphs =
      test_case.parallel
          ? ExtractSubgraphsParallel(config, graph, partition,
                                     /*num_subgraphs=*/1, /*parallelism=*/4)
          : ExtractSubgraphs(config, graph, partition, /*num_subgraphs=*/1);
  EXPECT_THAT(subgraphs, SizeIs(1));

  EXPECT_EQ(subgraphs[0].NumNodes(), 0);
  EXPECT_EQ(subgraphs[0].NumArcs(), 0);
}

TEST_P(ExtractSubgraphsTest, GraphWithNodesButNoEdges) {
  const ExtractSubgraphsTestCase& test_case = GetParam();
  RecursiveGraphPartitioningReorderingConfig config;

  std::vector<rgp::EdgeId> offsets = {0, 0, 0};
  std::vector<rgp::NodeId> targets = {};
  std::vector<GingerNodeId> mapping = {10, 11};

  rgp::CSRGraph graph(2, 0);
  absl::c_copy(offsets, graph.Offsets().begin());
  absl::c_copy(targets, graph.Targets().begin());
  absl::c_copy(mapping, graph.Mapping().begin());

  std::vector<rgp::PartitionId> partition = {0, 1};
  std::vector<rgp::CSRGraph> subgraphs =
      test_case.parallel
          ? ExtractSubgraphsParallel(config, graph, partition,
                                     /*num_subgraphs=*/2, /*parallelism=*/4)
          : ExtractSubgraphs(config, graph, partition, /*num_subgraphs=*/2);
  ASSERT_THAT(subgraphs, SizeIs(2));

  EXPECT_EQ(subgraphs[0].NumNodes(), 1);
  EXPECT_EQ(subgraphs[0].NumArcs(), 0);
  EXPECT_EQ(subgraphs[0].OriginalId(0), 10);

  EXPECT_EQ(subgraphs[1].NumNodes(), 1);
  EXPECT_EQ(subgraphs[1].NumArcs(), 0);
  EXPECT_EQ(subgraphs[1].OriginalId(0), 11);
}

TEST_P(ExtractSubgraphsTest, SinglePartition) {
  const ExtractSubgraphsTestCase& test_case = GetParam();
  RecursiveGraphPartitioningReorderingConfig config;

  std::vector<rgp::EdgeId> offsets = {0, 1, 3, 4};
  std::vector<rgp::NodeId> targets = {1, 0, 2, 1};
  std::vector<GingerNodeId> mapping = {10, 11, 12};

  rgp::CSRGraph graph(3, 4);
  absl::c_copy(offsets, graph.Offsets().begin());
  absl::c_copy(targets, graph.Targets().begin());
  absl::c_copy(mapping, graph.Mapping().begin());

  std::vector<rgp::PartitionId> partition = {0, 0, 0};
  std::vector<rgp::CSRGraph> subgraphs =
      test_case.parallel
          ? ExtractSubgraphsParallel(config, graph, partition,
                                     /*num_subgraphs=*/1, /*parallelism=*/4)
          : ExtractSubgraphs(config, graph, partition, /*num_subgraphs=*/1);
  EXPECT_THAT(subgraphs, SizeIs(1));

  const auto& subgraph = subgraphs[0];
  EXPECT_EQ(subgraph.NumNodes(), 3);
  EXPECT_EQ(subgraph.NumArcs(), 4);

  EXPECT_EQ(subgraph.Degree(0), 1);
  EXPECT_EQ(subgraph.Degree(1), 2);
  EXPECT_EQ(subgraph.Degree(2), 1);

  EXPECT_THAT(subgraph.Neighbors(0), ElementsAre(1));
  EXPECT_THAT(subgraph.Neighbors(1), ElementsAre(0, 2));
  EXPECT_THAT(subgraph.Neighbors(2), ElementsAre(1));

  EXPECT_EQ(subgraph.OriginalId(0), 10);
  EXPECT_EQ(subgraph.OriginalId(1), 11);
  EXPECT_EQ(subgraph.OriginalId(2), 12);
}

TEST_P(ExtractSubgraphsTest, MultiplePartitions) {
  const ExtractSubgraphsTestCase& test_case = GetParam();
  RecursiveGraphPartitioningReorderingConfig config;

  std::vector<rgp::EdgeId> offsets = {0, 1, 3, 4};
  std::vector<rgp::NodeId> targets = {1, 0, 2, 1};
  std::vector<GingerNodeId> mapping = {10, 11, 12};

  rgp::CSRGraph graph(3, 4);
  absl::c_copy(offsets, graph.Offsets().begin());
  absl::c_copy(targets, graph.Targets().begin());
  absl::c_copy(mapping, graph.Mapping().begin());

  std::vector<rgp::PartitionId> partition = {0, 1, 1};
  std::vector<rgp::CSRGraph> subgraphs =
      test_case.parallel
          ? ExtractSubgraphsParallel(config, graph, partition,
                                     /*num_subgraphs=*/2, /*parallelism=*/4)
          : ExtractSubgraphs(config, graph, partition, /*num_subgraphs=*/2);
  EXPECT_THAT(subgraphs, SizeIs(2));

  // Subgraph 0 contains node 0
  const auto& subgraph0 = subgraphs[0];
  EXPECT_EQ(subgraph0.NumNodes(), 1);
  EXPECT_EQ(subgraph0.NumArcs(), 0);

  EXPECT_EQ(subgraph0.Degree(0), 0);
  EXPECT_EQ(subgraph0.OriginalId(0), 10);

  // Subgraph 1 contains nodes 1 and 2
  const auto& subgraph1 = subgraphs[1];
  EXPECT_EQ(subgraph1.NumNodes(), 2);
  EXPECT_EQ(subgraph1.NumArcs(), 2);

  EXPECT_EQ(subgraph1.OriginalId(0), 11);
  EXPECT_EQ(subgraph1.OriginalId(1), 12);

  EXPECT_EQ(subgraph1.Degree(0), 1);
  EXPECT_EQ(subgraph1.Degree(1), 1);

  EXPECT_THAT(subgraph1.Neighbors(0), ElementsAre(1));
  EXPECT_THAT(subgraph1.Neighbors(1), ElementsAre(0));
}

TEST_P(ExtractSubgraphsTest, EveryNodeInItsOwnPartition) {
  const ExtractSubgraphsTestCase& test_case = GetParam();
  RecursiveGraphPartitioningReorderingConfig config;

  std::vector<rgp::EdgeId> offsets = {0, 1, 3, 4};
  std::vector<rgp::NodeId> targets = {1, 0, 2, 1};
  std::vector<GingerNodeId> mapping = {10, 11, 12};

  rgp::CSRGraph graph(3, 4);
  absl::c_copy(offsets, graph.Offsets().begin());
  absl::c_copy(targets, graph.Targets().begin());
  absl::c_copy(mapping, graph.Mapping().begin());

  std::vector<rgp::PartitionId> partition = {0, 1, 2};
  std::vector<rgp::CSRGraph> subgraphs =
      test_case.parallel
          ? ExtractSubgraphsParallel(config, graph, partition,
                                     /*num_subgraphs=*/3, /*parallelism=*/4)
          : ExtractSubgraphs(config, graph, partition, /*num_subgraphs=*/3);
  EXPECT_THAT(subgraphs, SizeIs(3));

  const auto& subgraph0 = subgraphs[0];
  EXPECT_EQ(subgraph0.NumNodes(), 1);
  EXPECT_EQ(subgraph0.NumArcs(), 0);
  EXPECT_EQ(subgraph0.OriginalId(0), 10);

  const auto& subgraph1 = subgraphs[1];
  EXPECT_EQ(subgraph1.NumNodes(), 1);
  EXPECT_EQ(subgraph1.NumArcs(), 0);
  EXPECT_EQ(subgraph1.OriginalId(0), 11);

  const auto& subgraph2 = subgraphs[2];
  EXPECT_EQ(subgraph2.NumNodes(), 1);
  EXPECT_EQ(subgraph2.NumArcs(), 0);
  EXPECT_EQ(subgraph2.OriginalId(0), 12);
}

INSTANTIATE_TEST_SUITE_P(ExtractSubgraphsTestSuite, ExtractSubgraphsTest,
                         ::testing::ValuesIn<ExtractSubgraphsTestCase>({
                             {/*parallel=*/false},
                             {/*parallel=*/true},
                         }),
                         [](const auto& info) {
                           return info.param.parallel ? "Parallel"
                                                      : "Sequential";
                         });

TEST(ComputeReorderingTest, DisconnectedGraph) {
  constexpr rgp::NodeId kNumNodesPerClique = 10;
  std::vector<std::vector<GingerNodeId>> adjacency_list(2 * kNumNodesPerClique);

  std::vector<GingerNodeId> permutation(2 * kNumNodesPerClique);
  std::iota(permutation.begin(), permutation.end(), 0);

  absl::BitGen bitgen;
  std::shuffle(permutation.begin(), permutation.end(), bitgen);

  // Create the first clique
  for (rgp::NodeId i = 0; i < kNumNodesPerClique; ++i) {
    const rgp::NodeId u = permutation[i];
    for (rgp::NodeId j = 0; j < kNumNodesPerClique; ++j) {
      const rgp::NodeId v = permutation[j];
      if (u == v) continue;
      adjacency_list[u].push_back(v);
    }
  }

  // Create the second clique
  for (rgp::NodeId i = kNumNodesPerClique; i < 2 * kNumNodesPerClique; ++i) {
    const rgp::NodeId u = permutation[i];
    for (rgp::NodeId j = kNumNodesPerClique; j < 2 * kNumNodesPerClique; ++j) {
      const rgp::NodeId v = permutation[j];
      if (u == v) continue;
      adjacency_list[u].push_back(v);
    }
  }

  RecursiveGraphPartitioningReorderingConfig config;
  config.set_base_case_threshold(1);
  config.set_top_level_branching_factor(2);

  auto graph = CreateCSRGraph(adjacency_list);
  auto reordering = ComputeReordering(graph, config);

  std::vector<rgp::NodeId> first_half(kNumNodesPerClique);
  for (rgp::NodeId i = 0; i < kNumNodesPerClique; ++i) {
    first_half[i] = reordering[permutation[i]];
  }

  std::vector<rgp::NodeId> second_half(kNumNodesPerClique);
  for (rgp::NodeId i = kNumNodesPerClique; i < 2 * kNumNodesPerClique; ++i) {
    second_half[i - kNumNodesPerClique] = reordering[permutation[i]];
  }

  std::vector<rgp::NodeId> expected_first_half(kNumNodesPerClique);
  std::iota(expected_first_half.begin(), expected_first_half.end(), 0);

  std::vector<rgp::NodeId> expected_second_half(kNumNodesPerClique);
  std::iota(expected_second_half.begin(), expected_second_half.end(),
            kNumNodesPerClique);

  // The two cliques are disconnected, so the reordering should group all nodes
  // of each clique together. Since the order of the cliques is not fixed,
  // either `first_half` contains the reordered indices of the first clique
  // and `second_half` contains the reordered indices of the second clique,
  // or vice versa.
  EXPECT_THAT(std::make_pair(first_half, second_half),
              AnyOf(Pair(UnorderedElementsAreArray(expected_first_half),
                         UnorderedElementsAreArray(expected_second_half)),
                    Pair(UnorderedElementsAreArray(expected_second_half),
                         UnorderedElementsAreArray(expected_first_half))));
}

}  // namespace
}  // namespace ginger::internal
