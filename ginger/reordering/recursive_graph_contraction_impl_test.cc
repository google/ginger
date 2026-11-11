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

#include "ginger/reordering/recursive_graph_contraction_impl.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/types.h"
#include "parlay/sequence.h"

namespace ginger::internal {
namespace {

using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

// Helper to create CSRGraph from adjacency list
CSRGraph CreateCSRGraph(
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
  return CSRGraph(std::move(offsets), std::move(targets));
}

// Tests for HandleBaseCase
TEST(HandleBaseCaseTest, EmptyNodes) {
  parlay::sequence<GingerNodeId> nodes;
  std::vector<GingerNodeId> sizes_vec = {};
  absl::Span<GingerNodeId> sizes = absl::MakeSpan(sizes_vec);

  rgc::HandleBaseCase(nodes, sizes);

  EXPECT_THAT(sizes_vec, IsEmpty());
}

TEST(HandleBaseCaseTest, SingleNode) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0});
  std::vector<GingerNodeId> sizes_vec = {10};
  absl::Span<GingerNodeId> sizes = absl::MakeSpan(sizes_vec);

  rgc::HandleBaseCase(nodes, sizes);

  EXPECT_THAT(sizes_vec, ElementsAre(0));
}

TEST(HandleBaseCaseTest, MultipleNodes) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  std::vector<GingerNodeId> sizes_vec = {1, 2, 3};
  absl::Span<GingerNodeId> sizes = absl::MakeSpan(sizes_vec);

  rgc::HandleBaseCase(nodes, sizes);

  EXPECT_THAT(sizes_vec, ElementsAre(0, 1, 3));
}

TEST(HandleBaseCaseTest, UnorderedNodes) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{1, 0});
  std::vector<GingerNodeId> sizes_vec = {10, 5, 100};
  absl::Span<GingerNodeId> sizes = absl::MakeSpan(sizes_vec);

  rgc::HandleBaseCase(nodes, sizes);

  EXPECT_THAT(sizes_vec, ElementsAre(5, 0, 100));
}

// Tests for ComputeMatching
TEST(ComputeMatchingTest, EmptyGraph) {
  parlay::sequence<GingerNodeId> nodes;
  parlay::sequence<rgc::Edge> edges;
  std::vector<GingerNodeId> sizes;

  parlay::sequence<GingerNodeId> parent_buffer;
  parlay::sequence<rgc::AtomicTaggedPriority> priorities_buffer;

  parlay::sequence<rgc::MatchingEdge> matching = rgc::ComputeMatching(
      nodes, edges, priorities_buffer, parent_buffer, absl::MakeSpan(sizes));

  EXPECT_THAT(matching, IsEmpty());
  EXPECT_THAT(parent_buffer, IsEmpty());
  EXPECT_THAT(sizes, IsEmpty());
}

TEST(ComputeMatchingTest, NoEdges) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  parlay::sequence<rgc::Edge> edges;
  std::vector<GingerNodeId> sizes = {1, 1, 1};

  auto parent_buffer = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  parlay::sequence<rgc::AtomicTaggedPriority> priorities_buffer(3);

  parlay::sequence<rgc::MatchingEdge> matching = rgc::ComputeMatching(
      nodes, edges, priorities_buffer, parent_buffer, absl::MakeSpan(sizes));

  EXPECT_THAT(matching, IsEmpty());
  EXPECT_THAT(parent_buffer, ElementsAre(0, 1, 2));
  EXPECT_THAT(sizes, ElementsAre(1, 1, 1));
}

TEST(ComputeMatchingTest, SingleEdge) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1});
  auto edges = parlay::to_sequence(std::vector<rgc::Edge>{{{0, 1}, 1.0f}});
  std::vector<GingerNodeId> sizes = {1, 1};

  auto parent_buffer = parlay::to_sequence(std::vector<GingerNodeId>{0, 1});
  parlay::sequence<rgc::AtomicTaggedPriority> priorities_buffer(2);

  parlay::sequence<rgc::MatchingEdge> matching = rgc::ComputeMatching(
      nodes, edges, priorities_buffer, parent_buffer, absl::MakeSpan(sizes));

  ASSERT_THAT(matching, SizeIs(1));
  EXPECT_EQ(matching[0].u, 0);
  EXPECT_EQ(matching[0].v, 1);
  EXPECT_EQ(matching[0].u_size, 1);

  EXPECT_THAT(parent_buffer, ElementsAre(0, 0));
  EXPECT_THAT(sizes, ElementsAre(2, 1));
}

TEST(ComputeMatchingTest, PathGraph) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  auto edges = parlay::to_sequence(
      std::vector<rgc::Edge>{{{0, 1}, 2.0f}, {{1, 2}, 3.0f}});
  std::vector<GingerNodeId> sizes = {1, 2, 1};

  auto parent_buffer = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  parlay::sequence<rgc::AtomicTaggedPriority> priorities_buffer(3);

  parlay::sequence<rgc::MatchingEdge> matching = rgc::ComputeMatching(
      nodes, edges, priorities_buffer, parent_buffer, absl::MakeSpan(sizes));

  ASSERT_THAT(matching, SizeIs(1));
  EXPECT_EQ(matching[0].u, 1);
  EXPECT_EQ(matching[0].v, 2);
  EXPECT_EQ(matching[0].u_size, 2);

  // Node 0 is unmatched, 1 and 2 are matched.
  EXPECT_THAT(parent_buffer, ElementsAre(0, 1, 1));
  EXPECT_THAT(sizes, ElementsAre(1, 3, 1));
}

TEST(ComputeMatchingTest, CycleGraph) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2, 3});
  auto edges = parlay::to_sequence(std::vector<rgc::Edge>{
      {{0, 1}, 1.0f}, {{1, 2}, 1.0f}, {{2, 3}, 1.0f}, {{0, 3}, 1.0f}});
  std::vector<GingerNodeId> sizes = {1, 1, 1, 1};

  auto parent_buffer =
      parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2, 3});
  parlay::sequence<rgc::AtomicTaggedPriority> priorities_buffer(4);

  parlay::sequence<rgc::MatchingEdge> matching = rgc::ComputeMatching(
      nodes, edges, priorities_buffer, parent_buffer, absl::MakeSpan(sizes));

  ASSERT_THAT(matching, SizeIs(1));
  EXPECT_EQ(matching[0].u, 0);
  EXPECT_EQ(matching[0].v, 3);
  EXPECT_EQ(matching[0].u_size, 1);

  // Nodes 0 and 3 are matched, 1 and 2 are unmatched.
  EXPECT_THAT(parent_buffer, ElementsAre(0, 1, 2, 0));
  EXPECT_THAT(sizes, ElementsAre(2, 1, 1, 1));
}

TEST(ComputeMatchingTest, PrioritiesArePositive) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1});
  auto edges = parlay::to_sequence(std::vector<rgc::Edge>{{{0, 1}, 1.0f}});
  std::vector<GingerNodeId> sizes = {1, 1};

  auto parent_buffer = parlay::to_sequence(std::vector<GingerNodeId>{0, 1});
  parlay::sequence<rgc::AtomicTaggedPriority> priorities_buffer(2);

  rgc::ComputeMatching(nodes, edges, priorities_buffer, parent_buffer,
                       absl::MakeSpan(sizes));

  for (const auto& p : priorities_buffer) {
    if (p.load().i != rgc::kInvalidEdgeIndex) {
      EXPECT_GT(p.load().weight, 0);
    }
  }
}

TEST(ComputeMatchingTest, LargeClusterSizesDoNotOverflow) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  auto edges = parlay::to_sequence(
      std::vector<rgc::Edge>{{{0, 1}, 100.0f}, {{1, 2}, 1.0f}});
  // 50000 * 50000 = 2,500,000,000, which overflows signed 32-bit int32_t.
  std::vector<GingerNodeId> sizes = {50000, 50000, 1};

  auto parent_buffer = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  parlay::sequence<rgc::AtomicTaggedPriority> priorities_buffer(3);

  parlay::sequence<rgc::MatchingEdge> matching = rgc::ComputeMatching(
      nodes, edges, priorities_buffer, parent_buffer, absl::MakeSpan(sizes));

  EXPECT_GT(priorities_buffer[0].load().weight, 0.0f);
  ASSERT_THAT(matching, SizeIs(1));
  EXPECT_EQ(matching[0].u, 1);
  EXPECT_EQ(matching[0].v, 2);
}

// Tests for UpdateGraph
TEST(UpdateGraphTest, EmptyGraph) {
  parlay::sequence<GingerNodeId> nodes;
  parlay::sequence<rgc::Edge> edges;
  parlay::sequence<GingerNodeId> parent_buffer;

  rgc::UpdateGraph(nodes, edges, parent_buffer);

  EXPECT_THAT(nodes, IsEmpty());
  EXPECT_THAT(edges, IsEmpty());
}

TEST(UpdateGraphTest, NoEdges) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  parlay::sequence<rgc::Edge> edges;
  auto parent_buffer = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 1});

  rgc::UpdateGraph(nodes, edges, parent_buffer);

  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1));
  EXPECT_THAT(edges, IsEmpty());
}

TEST(UpdateGraphTest, NoContraction) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  auto edges = parlay::to_sequence(
      std::vector<rgc::Edge>{{{0, 1}, 1.0f}, {{1, 2}, 2.0f}});
  auto parent_buffer = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});

  rgc::UpdateGraph(nodes, edges, parent_buffer);

  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1, 2));
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 1), 1.0f),
                                          Pair(Pair(1, 2), 2.0f)));
}

TEST(UpdateGraphTest, ContractNode) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2});
  auto edges = parlay::to_sequence(
      std::vector<rgc::Edge>{{{0, 1}, 1.0f}, {{1, 2}, 2.0f}});
  auto parent_buffer = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 1});

  rgc::UpdateGraph(nodes, edges, parent_buffer);

  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1));
  // Edge {1, 2} becomes {1, 1} and is removed as a self-loop.
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 1), 1.0f)));
}

TEST(UpdateGraphTest, ParallelEdgesCreatedAndMerged) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2, 3});
  auto edges = parlay::to_sequence(
      std::vector<rgc::Edge>{{{0, 2}, 1.0f}, {{1, 3}, 2.0f}});
  // Contract 2->0 and 3->1
  auto parent_buffer =
      parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 0, 1});

  rgc::UpdateGraph(nodes, edges, parent_buffer);

  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1));
  // Both edges become self-loops and are removed.
  EXPECT_THAT(edges, IsEmpty());
}

TEST(UpdateGraphTest, ParallelEdgesCreatedAndMerged2) {
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2, 3});
  auto edges = parlay::to_sequence(
      std::vector<rgc::Edge>{{{0, 1}, 1.0f}, {{2, 3}, 2.0f}});
  // Contract 1->0 and 3->2
  auto parent_buffer =
      parlay::to_sequence(std::vector<GingerNodeId>{0, 0, 2, 2});

  rgc::UpdateGraph(nodes, edges, parent_buffer);

  EXPECT_THAT(nodes, UnorderedElementsAre(0, 2));
  // Edge {0,1} becomes {0,0} -> self-loop
  // Edge {2,3} becomes {2,2} -> self-loop
  EXPECT_THAT(edges, IsEmpty());
}

TEST(UpdateGraphTest, StarGraphContraction) {
  // 0-1, 0-2, 0-3
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2, 3});
  auto edges = parlay::to_sequence(
      std::vector<rgc::Edge>{{{0, 1}, 1.0f}, {{0, 2}, 1.0f}, {{0, 3}, 1.0f}});
  // Contract 1->0
  auto parent_buffer =
      parlay::to_sequence(std::vector<GingerNodeId>{0, 0, 2, 3});

  rgc::UpdateGraph(nodes, edges, parent_buffer);

  EXPECT_THAT(nodes, UnorderedElementsAre(0, 2, 3));
  // Edge {0,1} becomes {0,0} -> self loop.
  // Edge {0,2} -> {0,2}
  // Edge {0,3} -> {0,3}
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 2), 1.0f),
                                          Pair(Pair(0, 3), 1.0f)));
}

TEST(UpdateGraphTest, PathGraphContraction) {
  // 0-1-2-3
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2, 3});
  auto edges = parlay::to_sequence(
      std::vector<rgc::Edge>{{{0, 1}, 1.0f}, {{1, 2}, 1.0f}, {{2, 3}, 1.0f}});
  // Contract 1->0 and 3->2
  auto parent_buffer =
      parlay::to_sequence(std::vector<GingerNodeId>{0, 0, 2, 2});
  rgc::UpdateGraph(nodes, edges, parent_buffer);
  EXPECT_THAT(nodes, UnorderedElementsAre(0, 2));
  // Edge {0,1} becomes {0,0} -> self loop.
  // Edge {1,2} becomes {0,2}
  // Edge {2,3} becomes {2,2} -> self loop.
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 2), 1.0f)));
}

TEST(UpdateGraphTest, MergeEdges) {
  // 0-2 w=1, 1-3 w=2.
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2, 3});
  auto edges = parlay::to_sequence(
      std::vector<rgc::Edge>{{{0, 2}, 1.0f}, {{1, 3}, 2.0f}});
  // Contract 2->0, 1->0, 3->0 => all nodes map to 0
  auto parent_buffer =
      parlay::to_sequence(std::vector<GingerNodeId>{0, 0, 0, 0});
  rgc::UpdateGraph(nodes, edges, parent_buffer);
  EXPECT_THAT(nodes, UnorderedElementsAre(0));
  // All edges become self-loops.
  EXPECT_THAT(edges, IsEmpty());
}

TEST(UpdateGraphTest, MergeEdges2) {
  // 0-1 w=1, 0-2 w=2, 1-3 w=3, 2-3 w=4
  auto nodes = parlay::to_sequence(std::vector<GingerNodeId>{0, 1, 2, 3});
  auto edges = parlay::to_sequence(std::vector<rgc::Edge>{
      {{0, 1}, 1.0f}, {{0, 2}, 2.0f}, {{1, 3}, 3.0f}, {{2, 3}, 4.0f}});
  // Contract 1->0 and 3->2, so parents = {0,0,2,2}
  auto parent_buffer =
      parlay::to_sequence(std::vector<GingerNodeId>{0, 0, 2, 2});

  rgc::UpdateGraph(nodes, edges, parent_buffer);

  EXPECT_THAT(nodes, UnorderedElementsAre(0, 2));
  // Edge {0,1}w1 -> {0,0}w1 -> self-loop
  // Edge {0,2}w2 -> {0,2}w2
  // Edge {1,3}w3 -> {0,2}w3
  // Edge {2,3}w4 -> {2,2}w4 -> self-loop
  // Edges {0,2}w2 and {0,2}w3 merge to {0,2}w5.
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 2), 5.0f)));
}

// Tests for WriteOffsets
TEST(WriteOffsetsTest, EmptyMatching) {
  parlay::sequence<rgc::MatchingEdge> matching;
  std::vector<GingerNodeId> sizes_vec = {};
  absl::Span<GingerNodeId> sizes = absl::MakeSpan(sizes_vec);

  rgc::WriteOffsets(matching, sizes);

  EXPECT_THAT(sizes_vec, IsEmpty());
}

TEST(WriteOffsetsTest, EmptyMatchingNonEmptySizes) {
  parlay::sequence<rgc::MatchingEdge> matching;
  std::vector<GingerNodeId> sizes_vec = {10, 20};
  absl::Span<GingerNodeId> sizes = absl::MakeSpan(sizes_vec);

  rgc::WriteOffsets(matching, sizes);

  EXPECT_THAT(sizes_vec, ElementsAre(10, 20));
}

TEST(WriteOffsetsTest, SingleEdge) {
  auto matching =
      parlay::to_sequence(std::vector<rgc::MatchingEdge>{{0, 1, 10}});
  std::vector<GingerNodeId> sizes_vec = {100, 0};
  absl::Span<GingerNodeId> sizes = absl::MakeSpan(sizes_vec);

  rgc::WriteOffsets(matching, sizes);

  EXPECT_THAT(sizes_vec, ElementsAre(100, 110));
}

TEST(WriteOffsetsTest, MultipleEdges) {
  auto matching = parlay::to_sequence(
      std::vector<rgc::MatchingEdge>{{0, 1, 10}, {2, 3, 8}});
  std::vector<GingerNodeId> sizes_vec = {100, 0, 0, 200};
  absl::Span<GingerNodeId> sizes = absl::MakeSpan(sizes_vec);

  rgc::WriteOffsets(matching, sizes);

  EXPECT_THAT(sizes_vec, ElementsAre(100, 110, 0, 8));
}

TEST(WriteOffsetsTest, NonContiguousNodeIds) {
  auto matching =
      parlay::to_sequence(std::vector<rgc::MatchingEdge>{{0, 2, 10}});
  std::vector<GingerNodeId> sizes_vec = {100, 99, 0};
  absl::Span<GingerNodeId> sizes = absl::MakeSpan(sizes_vec);

  rgc::WriteOffsets(matching, sizes);

  EXPECT_THAT(sizes_vec, ElementsAre(100, 99, 110));
}

// End-to-end tests for ComputeRecursiveGraphContractionReordering
TEST(ComputeRecursiveGraphContractionReorderingTest, EmptyGraph) {
  CSRGraph graph = CreateCSRGraph({});
  auto reordering = ComputeRecursiveGraphContractionReordering(graph);
  EXPECT_THAT(reordering, IsEmpty());
}

TEST(ComputeRecursiveGraphContractionReorderingTest, SingleNode) {
  CSRGraph graph = CreateCSRGraph({{}});
  auto reordering = ComputeRecursiveGraphContractionReordering(graph);
  EXPECT_THAT(reordering, ElementsAre(0));
}

TEST(ComputeRecursiveGraphContractionReorderingTest, TwoNodesNoEdge) {
  CSRGraph graph = CreateCSRGraph({{}, {}});
  auto reordering = ComputeRecursiveGraphContractionReordering(graph);
  EXPECT_THAT(reordering, UnorderedElementsAre(0, 1));
}

TEST(ComputeRecursiveGraphContractionReorderingTest, DisconnectedGraph) {
  constexpr GingerNodeId kNumNodesPerClique = 10;
  const GingerNodeId num_nodes = 2 * kNumNodesPerClique;

  // We are constructing a graph made up of two disconnected cliques. The nodes
  // with IDs 0, 2, 4, 6, ... are in the first clique, and nodes with IDs 1, 3,
  // 5, ... are in the second clique. After reordering, we expect one clique
  // to be placed at indices 0 to `kNumNodesPerClique - 1` and the other at
  // indices `kNumNodesPerClique` to `num_nodes - 1`.

  std::vector<GingerNodeId> first_clique;
  first_clique.reserve(kNumNodesPerClique);
  for (GingerNodeId i = 0; i < num_nodes; i += 2) {
    first_clique.push_back(i);
  }

  std::vector<GingerNodeId> second_clique;
  second_clique.reserve(kNumNodesPerClique);
  for (GingerNodeId i = 1; i < num_nodes; i += 2) {
    second_clique.push_back(i);
  }

  std::vector<std::vector<GingerNodeId>> adjacency_list(num_nodes);
  for (const GingerNodeId u : first_clique) {
    for (const GingerNodeId v : first_clique) {
      if (u == v) continue;
      adjacency_list[u].push_back(v);
    }
  }
  for (const GingerNodeId u : second_clique) {
    for (const GingerNodeId v : second_clique) {
      if (u == v) continue;
      adjacency_list[u].push_back(v);
    }
  }

  auto graph = CreateCSRGraph(adjacency_list);
  auto reordering = ComputeRecursiveGraphContractionReordering(graph);

  std::vector<GingerNodeId> first_clique_reordered;
  for (const GingerNodeId u : first_clique) {
    first_clique_reordered.push_back(reordering[u]);
  }
  std::vector<GingerNodeId> second_clique_reordered;
  for (const GingerNodeId u : second_clique) {
    second_clique_reordered.push_back(reordering[u]);
  }

  std::vector<GingerNodeId> expected_half_1(kNumNodesPerClique);
  std::iota(expected_half_1.begin(), expected_half_1.end(), 0);

  std::vector<GingerNodeId> expected_half_2(kNumNodesPerClique);
  std::iota(expected_half_2.begin(), expected_half_2.end(), kNumNodesPerClique);

  EXPECT_THAT(std::make_pair(first_clique_reordered, second_clique_reordered),
              AnyOf(Pair(UnorderedElementsAreArray(expected_half_1),
                         UnorderedElementsAreArray(expected_half_2)),
                    Pair(UnorderedElementsAreArray(expected_half_2),
                         UnorderedElementsAreArray(expected_half_1))));
}

// Tests for CopyGraph
TEST(CopyGraphTest, EmptyGraph) {
  CSRGraph graph = CreateCSRGraph({});
  auto [nodes, edges] = rgc::CopyGraph(graph);
  EXPECT_THAT(nodes, IsEmpty());
  EXPECT_THAT(edges, IsEmpty());
}

TEST(CopyGraphTest, GraphWithNodesButNoEdges) {
  CSRGraph graph = CreateCSRGraph({{}, {}, {}});
  auto [nodes, edges] = rgc::CopyGraph(graph);
  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1, 2));
  EXPECT_THAT(edges, IsEmpty());
}

TEST(CopyGraphTest, GraphWithOneEdge) {
  CSRGraph graph = CreateCSRGraph({{1}, {}});
  auto [nodes, edges] = rgc::CopyGraph(graph);
  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1));
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 1), 1.0f)));
}

TEST(CopyGraphTest, GraphWithOneBidirectionalEdge) {
  CSRGraph graph = CreateCSRGraph({{1}, {0}});
  auto [nodes, edges] = rgc::CopyGraph(graph);
  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1));
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 1), 2.0f)));
}

TEST(CopyGraphTest, GraphWithParallelEdges) {
  CSRGraph graph = CreateCSRGraph({{1, 1}, {0}});
  auto [nodes, edges] = rgc::CopyGraph(graph);
  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1));
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 1), 3.0f)));
}

TEST(CopyGraphTest, TriangleGraph) {
  CSRGraph graph = CreateCSRGraph({{1, 2}, {0, 2}, {0, 1}});
  auto [nodes, edges] = rgc::CopyGraph(graph);
  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1, 2));
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 1), 2.0f),
                                          Pair(Pair(0, 2), 2.0f),
                                          Pair(Pair(1, 2), 2.0f)));
}

TEST(CopyGraphTest, FiltersSelfLoops) {
  CSRGraph graph = CreateCSRGraph({{0, 1}, {1}});
  auto [nodes, edges] = rgc::CopyGraph(graph);
  EXPECT_THAT(nodes, UnorderedElementsAre(0, 1));
  EXPECT_THAT(edges, UnorderedElementsAre(Pair(Pair(0, 1), 1.0f)));
}

}  // namespace
}  // namespace ginger::internal
