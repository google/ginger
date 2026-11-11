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

#include "ginger/reordering/recursive_graph_bisection_impl.h"

#include <array>
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

namespace ginger::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

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

TEST(QueryDataGraphTest, EmptyGraph) {
  std::vector<int64_t> offsets = {};
  std::vector<GingerNodeId> targets = {};
  std::vector<GingerNodeId> mapping = {};
  QueryDataGraph graph(/*num_query_nodes=*/0, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  EXPECT_EQ(graph.NumQueryNodes(), 0);
  EXPECT_EQ(graph.NumDataNodes(), 0);
  EXPECT_EQ(graph.NumEdges(), 0);

  EXPECT_THAT(graph.Offsets(), ElementsAreArray(offsets));
  EXPECT_THAT(graph.Targets(), ElementsAreArray(targets));
  EXPECT_THAT(graph.Mapping(), ElementsAreArray(mapping));
}

TEST(QueryDataGraphTest, SingleDataNode) {
  std::vector<int64_t> offsets = {0};
  std::vector<GingerNodeId> targets = {0, 1};
  std::vector<GingerNodeId> mapping = {10};
  QueryDataGraph graph(/*num_query_nodes=*/2, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  EXPECT_EQ(graph.NumQueryNodes(), 2);
  EXPECT_EQ(graph.NumDataNodes(), 1);
  EXPECT_EQ(graph.NumEdges(), 2);
  EXPECT_EQ(graph.Degree(0), 2);
  EXPECT_THAT(graph.AdjacentQueryNodes(0), ElementsAre(0, 1));
  EXPECT_EQ(graph.OriginalId(0), 10);

  EXPECT_THAT(graph.Offsets(), ElementsAreArray(offsets));
  EXPECT_THAT(graph.Targets(), ElementsAreArray(targets));
  EXPECT_THAT(graph.Mapping(), ElementsAreArray(mapping));
}

TEST(QueryDataGraphTest, GraphWithDataNodesButNoQueryNodes) {
  std::vector<int64_t> offsets = {0, 0, 0};
  std::vector<GingerNodeId> targets = {};
  std::vector<GingerNodeId> mapping = {10, 0, 50};
  QueryDataGraph graph(/*num_query_nodes=*/0, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  EXPECT_EQ(graph.NumQueryNodes(), 0);
  EXPECT_EQ(graph.NumDataNodes(), 3);
  EXPECT_EQ(graph.NumEdges(), 0);

  EXPECT_EQ(graph.Degree(0), 0);
  EXPECT_EQ(graph.Degree(1), 0);
  EXPECT_EQ(graph.Degree(2), 0);

  EXPECT_THAT(graph.AdjacentQueryNodes(0), IsEmpty());
  EXPECT_THAT(graph.AdjacentQueryNodes(1), IsEmpty());
  EXPECT_THAT(graph.AdjacentQueryNodes(2), IsEmpty());

  EXPECT_EQ(graph.OriginalId(0), 10);
  EXPECT_EQ(graph.OriginalId(1), 0);
  EXPECT_EQ(graph.OriginalId(2), 50);

  EXPECT_THAT(graph.Offsets(), ElementsAreArray(offsets));
  EXPECT_THAT(graph.Targets(), ElementsAreArray(targets));
  EXPECT_THAT(graph.Mapping(), ElementsAreArray(mapping));
}

TEST(QueryDataGraphTest, GraphWithDataNodesAndQueryNodes) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  EXPECT_EQ(graph.NumQueryNodes(), 5);
  EXPECT_EQ(graph.NumDataNodes(), 3);
  EXPECT_EQ(graph.NumEdges(), 6);

  EXPECT_EQ(graph.Degree(0), 2);
  EXPECT_EQ(graph.Degree(1), 1);
  EXPECT_EQ(graph.Degree(2), 3);

  EXPECT_THAT(graph.AdjacentQueryNodes(0), ElementsAre(0, 1));
  EXPECT_THAT(graph.AdjacentQueryNodes(1), ElementsAre(2));
  EXPECT_THAT(graph.AdjacentQueryNodes(2), ElementsAre(0, 3, 4));

  EXPECT_EQ(graph.OriginalId(0), 100);
  EXPECT_EQ(graph.OriginalId(1), 0);
  EXPECT_EQ(graph.OriginalId(2), 12);

  EXPECT_THAT(graph.Offsets(), ElementsAreArray(offsets));
  EXPECT_THAT(graph.Targets(), ElementsAreArray(targets));
  EXPECT_THAT(graph.Mapping(), ElementsAreArray(mapping));
}

TEST(BisectionTest, InitArbitrarilyEmpty) {
  std::vector<GingerNodeId> left_data_nodes = {};
  std::vector<GingerNodeId> right_data_nodes = {};
  std::vector<GingerNodeId> mapping = {};
  Bisection bisection(absl::MakeSpan(left_data_nodes),
                      absl::MakeSpan(right_data_nodes),
                      absl::MakeSpan(mapping));

  bisection.InitArbitrarily();
  EXPECT_EQ(bisection.Size(), 0);
  EXPECT_EQ(bisection.LeftSize(), 0);
  EXPECT_EQ(bisection.RightSize(), 0);
  EXPECT_THAT(bisection.LeftDataNodes(), IsEmpty());
  EXPECT_THAT(bisection.RightDataNodes(), IsEmpty());
}

TEST(BisectionTest, InitArbitrarily) {
  std::vector<GingerNodeId> left_data_nodes(3);
  std::vector<GingerNodeId> right_data_nodes(2);
  std::vector<GingerNodeId> mapping(5);
  Bisection bisection(absl::MakeSpan(left_data_nodes),
                      absl::MakeSpan(right_data_nodes),
                      absl::MakeSpan(mapping));

  bisection.InitArbitrarily();
  EXPECT_EQ(bisection.Size(), 5);
  EXPECT_EQ(bisection.LeftSize(), 3);
  EXPECT_EQ(bisection.RightSize(), 2);
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(0, 1, 2));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(3, 4));
}

TEST(BisectionTest, SwapNodes) {
  std::vector<GingerNodeId> left_data_nodes(3);
  std::vector<GingerNodeId> right_data_nodes(2);
  std::vector<GingerNodeId> mapping(5);
  Bisection bisection(absl::MakeSpan(left_data_nodes),
                      absl::MakeSpan(right_data_nodes),
                      absl::MakeSpan(mapping));

  bisection.InitArbitrarily();
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(0, 1, 2));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(3, 4));

  // Swap node 1 (left) with node 3 (right)
  bisection.Swap(/*left_data_node=*/1, /*right_data_node=*/3);
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(0, 3, 2));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(1, 4));

  // Swap node 0 (left) with node 4 (right)
  bisection.Swap(/*left_data_node=*/0, /*right_data_node=*/4);
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(4, 3, 2));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(1, 0));

  // Swap node 2 (left) with node 1 (right)
  bisection.Swap(/*left_data_node=*/2, /*right_data_node=*/1);
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(4, 3, 1));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(2, 0));

  // Swap node 4 (left) with node 0 (right)
  bisection.Swap(/*left_data_node=*/4, /*right_data_node=*/0);
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(0, 3, 1));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(2, 4));
}

TEST(BisectionTest, SplitEmpty) {
  std::vector<GingerNodeId> left_data_nodes = {};
  std::vector<GingerNodeId> right_data_nodes = {};
  std::vector<GingerNodeId> mapping = {};
  Bisection bisection(absl::MakeSpan(left_data_nodes),
                      absl::MakeSpan(right_data_nodes),
                      absl::MakeSpan(mapping));

  bisection.InitArbitrarily();
  auto [left_subproblem, right_subproblem] = bisection.Split();

  EXPECT_EQ(left_subproblem.Size(), 0);
  EXPECT_EQ(left_subproblem.LeftSize(), 0);
  EXPECT_EQ(left_subproblem.RightSize(), 0);
  EXPECT_THAT(left_subproblem.LeftDataNodes(), IsEmpty());
  EXPECT_THAT(left_subproblem.RightDataNodes(), IsEmpty());

  EXPECT_EQ(right_subproblem.Size(), 0);
  EXPECT_EQ(right_subproblem.LeftSize(), 0);
  EXPECT_EQ(right_subproblem.RightSize(), 0);
  EXPECT_THAT(right_subproblem.LeftDataNodes(), IsEmpty());
  EXPECT_THAT(right_subproblem.RightDataNodes(), IsEmpty());
}

TEST(BisectionTest, Split) {
  std::vector<GingerNodeId> left_data_nodes(3);
  std::vector<GingerNodeId> right_data_nodes(2);
  std::vector<GingerNodeId> mapping(5);
  Bisection bisection(absl::MakeSpan(left_data_nodes),
                      absl::MakeSpan(right_data_nodes),
                      absl::MakeSpan(mapping));

  bisection.InitArbitrarily();
  auto [left_subproblem, right_subproblem] = bisection.Split();

  EXPECT_EQ(left_subproblem.Size(), 3);
  EXPECT_EQ(left_subproblem.LeftSize(), 1);
  EXPECT_EQ(left_subproblem.RightSize(), 2);

  EXPECT_EQ(right_subproblem.Size(), 2);
  EXPECT_EQ(right_subproblem.LeftSize(), 1);
  EXPECT_EQ(right_subproblem.RightSize(), 1);
}

TEST(MovesTest, Moves) {
  std::vector<Move> left_to_right_moves = {{.data_node = 0, .gain = 0.1f},
                                           {.data_node = 1, .gain = 0.5f}};
  std::vector<Move> right_to_left_moves = {{.data_node = 2, .gain = 0.2f}};
  Moves moves(absl::MakeSpan(left_to_right_moves),
              absl::MakeSpan(right_to_left_moves));

  EXPECT_EQ(moves.NumLeftMoves(), 2);
  EXPECT_EQ(moves.NumRightMoves(), 1);
  EXPECT_THAT(moves.LeftToRightMoves(),
              ElementsAre(FieldsAre(0, 0.1f), FieldsAre(1, 0.5f)));
  EXPECT_THAT(moves.RightToLeftMoves(), ElementsAre(FieldsAre(2, 0.2f)));
}

TEST(MovesTest, SortByGain) {
  std::vector<Move> left_to_right_moves = {
      {.data_node = 2, .gain = 0.4f},
      {.data_node = 0, .gain = 0.1f},
      {.data_node = 1, .gain = 0.5f},
  };
  std::vector<Move> right_to_left_moves = {{.data_node = 3, .gain = 0.2f},
                                           {.data_node = 4, .gain = 0.8f},
                                           {.data_node = 5, .gain = 0.6f}};
  Moves moves(absl::MakeSpan(left_to_right_moves),
              absl::MakeSpan(right_to_left_moves));

  moves.SortByGain();
  EXPECT_THAT(
      moves.LeftToRightMoves(),
      ElementsAre(FieldsAre(1, 0.5f), FieldsAre(2, 0.4f), FieldsAre(0, 0.1f)));
  EXPECT_THAT(
      moves.RightToLeftMoves(),
      ElementsAre(FieldsAre(4, 0.8f), FieldsAre(5, 0.6f), FieldsAre(3, 0.2f)));
}

TEST(MovesTest, SplitMoves) {
  std::vector<Move> left_to_right_moves = {{.data_node = 0, .gain = 0.1f},
                                           {.data_node = 1, .gain = 0.5f},
                                           {.data_node = 2, .gain = 0.4f}};
  std::vector<Move> right_to_left_moves = {{.data_node = 3, .gain = 0.2f},
                                           {.data_node = 4, .gain = 0.8f},
                                           {.data_node = 5, .gain = 0.6f},
                                           {.data_node = 6, .gain = 0.3f}};
  Moves moves(absl::MakeSpan(left_to_right_moves),
              absl::MakeSpan(right_to_left_moves));

  auto [left_subproblem, right_subproblem] = moves.Split();

  EXPECT_EQ(left_subproblem.NumLeftMoves(), 1);
  EXPECT_EQ(left_subproblem.NumRightMoves(), 2);

  EXPECT_EQ(right_subproblem.NumLeftMoves(), 2);
  EXPECT_EQ(right_subproblem.NumRightMoves(), 2);
}

TEST(ComputeDegreesTest, NoDataNodes) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> data_nodes = {};
  std::vector<GingerNodeId> degrees(5);
  ComputeDegrees(graph, data_nodes, absl::MakeSpan(degrees));

  EXPECT_THAT(degrees, ElementsAre(0, 0, 0, 0, 0));
}

TEST(ComputeDegreesTest, AllDataNodes) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> data_nodes = {0, 1, 2};
  std::vector<GingerNodeId> degrees(5);
  ComputeDegrees(graph, data_nodes, absl::MakeSpan(degrees));

  EXPECT_THAT(degrees, ElementsAre(2, 1, 1, 1, 1));
}

TEST(ComputeDegreesTest, SomeDataNodes) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> data_nodes = {0, 2};
  std::vector<GingerNodeId> degrees(5);
  ComputeDegrees(graph, data_nodes, absl::MakeSpan(degrees));

  EXPECT_THAT(degrees, ElementsAre(2, 1, 0, 1, 1));
}

TEST(ComputeDegreesTest, SingleDataNode) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};

  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> data_nodes = {1};
  std::vector<GingerNodeId> degrees(5);
  ComputeDegrees(graph, data_nodes, absl::MakeSpan(degrees));

  EXPECT_THAT(degrees, ElementsAre(0, 0, 1, 0, 0));
}

TEST(UpdateDegreesTest, MoveDataNode0) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> from_degrees = {2, 1, 1, 1, 1};
  std::vector<GingerNodeId> to_degrees = {0, 0, 0, 0, 0};
  UpdateDegrees(graph, /*data_node=*/0, absl::MakeSpan(from_degrees),
                absl::MakeSpan(to_degrees));

  EXPECT_THAT(from_degrees, ElementsAre(1, 0, 1, 1, 1));
  EXPECT_THAT(to_degrees, ElementsAre(1, 1, 0, 0, 0));
}

TEST(UpdateDegreesTest, MoveDataNode1) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> from_degrees = {2, 1, 1, 1, 1};
  std::vector<GingerNodeId> to_degrees = {0, 0, 0, 0, 0};
  UpdateDegrees(graph, /*data_node=*/1, absl::MakeSpan(from_degrees),
                absl::MakeSpan(to_degrees));

  EXPECT_THAT(from_degrees, ElementsAre(2, 1, 0, 1, 1));
  EXPECT_THAT(to_degrees, ElementsAre(0, 0, 1, 0, 0));
}

TEST(UpdateDegreesTest, MoveDataNode2) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> from_degrees = {2, 1, 1, 1, 1};
  std::vector<GingerNodeId> to_degrees = {0, 0, 0, 0, 0};
  UpdateDegrees(graph, /*data_node=*/2, absl::MakeSpan(from_degrees),
                absl::MakeSpan(to_degrees));

  EXPECT_THAT(from_degrees, ElementsAre(1, 1, 1, 0, 0));
  EXPECT_THAT(to_degrees, ElementsAre(1, 0, 0, 1, 1));
}

TEST(ComputeMovesTest, LeftToRightMoves) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<QueryCost> query_costs = {
      {.cost = 1.0f, .left_to_right_cost = 0.5f, .right_to_left_cost = 1.5f},
      {.cost = 2.0f, .left_to_right_cost = 1.0f, .right_to_left_cost = 3.0f},
      {.cost = 0.5f, .left_to_right_cost = 0.0f, .right_to_left_cost = 1.0f},
      {.cost = 3.0f, .left_to_right_cost = 2.0f, .right_to_left_cost = 4.0f},
      {.cost = 1.5f, .left_to_right_cost = 1.0f, .right_to_left_cost = 2.0f}};

  std::vector<GingerNodeId> data_nodes = {0, 1, 2};
  std::vector<Move> moves(3);
  ComputeMoves(graph, data_nodes, query_costs, absl::MakeSpan(moves),
               /*left_to_right=*/true);

  EXPECT_THAT(moves, ElementsAre(FieldsAre(0, 1.5f), FieldsAre(1, 0.5f),
                                 FieldsAre(2, 2.0f)));
}

TEST(ComputeMovesTest, RightToLeftMoves) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<QueryCost> query_costs = {
      {.cost = 1.0f, .left_to_right_cost = 0.5f, .right_to_left_cost = 1.5f},
      {.cost = 2.0f, .left_to_right_cost = 1.0f, .right_to_left_cost = 3.0f},
      {.cost = 0.5f, .left_to_right_cost = 0.0f, .right_to_left_cost = 1.0f},
      {.cost = 3.0f, .left_to_right_cost = 2.0f, .right_to_left_cost = 4.0f},
      {.cost = 1.5f, .left_to_right_cost = 1.0f, .right_to_left_cost = 2.0f}};

  std::vector<GingerNodeId> data_nodes = {0, 1, 2};
  std::vector<Move> moves(3);
  ComputeMoves(graph, data_nodes, query_costs, absl::MakeSpan(moves),
               /*left_to_right=*/false);

  EXPECT_THAT(moves, ElementsAre(FieldsAre(0, -1.5f), FieldsAre(1, -0.5f),
                                 FieldsAre(2, -2.0f)));
}

TEST(PerformSwapsAndUpdateDegreesTest, EmptyMoves) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> left_data_nodes_vec(2);
  std::vector<GingerNodeId> right_data_nodes_vec(1);
  std::vector<GingerNodeId> bisection_mapping_vec(3);
  Bisection bisection(absl::MakeSpan(left_data_nodes_vec),
                      absl::MakeSpan(right_data_nodes_vec),
                      absl::MakeSpan(bisection_mapping_vec));
  bisection.InitArbitrarily();

  std::array<std::vector<GingerNodeId>, 2> degrees = {
      std::vector<GingerNodeId>{1, 1, 1, 0, 0},
      std::vector<GingerNodeId>{1, 0, 0, 1, 1},
  };

  std::vector<Move> left_to_right_moves_vec = {};
  std::vector<Move> right_to_left_moves_vec = {};
  Moves moves(absl::MakeSpan(left_to_right_moves_vec),
              absl::MakeSpan(right_to_left_moves_vec));

  EXPECT_EQ(PerformSwapsAndUpdateDegrees(graph, bisection, moves, degrees), 0);
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(0, 1));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(2));
  EXPECT_THAT(degrees[0], ElementsAre(1, 1, 1, 0, 0));
  EXPECT_THAT(degrees[1], ElementsAre(1, 0, 0, 1, 1));
}

TEST(PerformSwapsAndUpdateDegreesTest, NoSwaps) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> left_data_nodes_vec(2);
  std::vector<GingerNodeId> right_data_nodes_vec(1);
  std::vector<GingerNodeId> bisection_mapping_vec(3);
  Bisection bisection(absl::MakeSpan(left_data_nodes_vec),
                      absl::MakeSpan(right_data_nodes_vec),
                      absl::MakeSpan(bisection_mapping_vec));
  bisection.InitArbitrarily();

  std::array<std::vector<GingerNodeId>, 2> degrees = {
      std::vector<GingerNodeId>{1, 1, 1, 0, 0},
      std::vector<GingerNodeId>{1, 0, 0, 1, 1},
  };

  std::vector<Move> left_to_right_moves_vec = {
      {.data_node = 0, .gain = 0.1f},
      {.data_node = 1, .gain = 0.05f},
  };
  std::vector<Move> right_to_left_moves_vec = {
      {.data_node = 2, .gain = -0.2f},
  };
  Moves moves(absl::MakeSpan(left_to_right_moves_vec),
              absl::MakeSpan(right_to_left_moves_vec));

  EXPECT_EQ(PerformSwapsAndUpdateDegrees(graph, bisection, moves, degrees), 0);
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(0, 1));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(2));
  EXPECT_THAT(degrees[0], ElementsAre(1, 1, 1, 0, 0));
  EXPECT_THAT(degrees[1], ElementsAre(1, 0, 0, 1, 1));
}

TEST(PerformSwapsAndUpdateDegreesTest, OneSwap) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> left_data_nodes(2);
  std::vector<GingerNodeId> right_data_nodes(1);
  std::vector<GingerNodeId> bisection_mapping(3);
  Bisection bisection(absl::MakeSpan(left_data_nodes),
                      absl::MakeSpan(right_data_nodes),
                      absl::MakeSpan(bisection_mapping));
  bisection.InitArbitrarily();

  std::array<std::vector<GingerNodeId>, 2> degrees = {
      std::vector<GingerNodeId>{1, 1, 1, 0, 0},
      std::vector<GingerNodeId>{1, 0, 0, 1, 1},
  };

  std::vector<Move> left_to_right_moves = {
      {.data_node = 0, .gain = 0.3f},
      {.data_node = 1, .gain = 0.05f},
  };
  std::vector<Move> right_to_left_moves = {
      {.data_node = 2, .gain = -0.2f},
  };
  Moves moves(absl::MakeSpan(left_to_right_moves),
              absl::MakeSpan(right_to_left_moves));

  EXPECT_EQ(PerformSwapsAndUpdateDegrees(graph, bisection, moves, degrees), 1);
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(2, 1));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(0));
  EXPECT_THAT(degrees[0], ElementsAre(1, 0, 1, 1, 1));
  EXPECT_THAT(degrees[1], ElementsAre(1, 1, 0, 0, 0));
}

TEST(PerformSwapsAndUpdateDegreesTest, MultipleSwaps) {
  std::vector<int64_t> offsets = {0, 2, 3, 6};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4, 1, 2, 3};
  std::vector<GingerNodeId> mapping = {100, 0, 12, 1};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  std::vector<GingerNodeId> left_data_nodes_vec(2);
  std::vector<GingerNodeId> right_data_nodes_vec(2);
  std::vector<GingerNodeId> bisection_mapping_vec(4);
  Bisection bisection(absl::MakeSpan(left_data_nodes_vec),
                      absl::MakeSpan(right_data_nodes_vec),
                      absl::MakeSpan(bisection_mapping_vec));
  bisection.InitArbitrarily();  // L={0,1}, R={2,3}

  // Degrees for L={0,1}, R={2,3}:
  // L: 0->{0,1}, 1->{2}. QNs: 0,1,2.
  // R: 2->{0,3,4}, 3->{1,2,3}. QNs: 0,1,2,3,4
  // deg[0]: for q0: 0 in L, 2 in R -> 1L, 1R
  // for q1: 0 in L, 3 in R -> 1L, 1R
  // for q2: 1 in L, 3 in R -> 1L, 1R
  // for q3: 2 in R, 3 in R -> 0L, 2R
  // for q4: 2 in R -> 0L, 1R
  std::array<std::vector<GingerNodeId>, 2> degrees = {
      // left degrees for QN 0,1,2,3,4
      std::vector<GingerNodeId>{1, 1, 1, 0, 0},
      // right degrees for QN 0,1,2,3,4
      std::vector<GingerNodeId>{1, 1, 1, 2, 1},
  };

  std::vector<Move> left_to_right_moves_vec = {
      {.data_node = 0, .gain = 0.8f},
      {.data_node = 1, .gain = 0.7f},
  };
  std::vector<Move> right_to_left_moves_vec = {
      {.data_node = 2, .gain = 0.6f},
      {.data_node = 3, .gain = 0.5f},
  };
  Moves moves(absl::MakeSpan(left_to_right_moves_vec),
              absl::MakeSpan(right_to_left_moves_vec));
  const GingerNodeId num_swaps_performed =
      PerformSwapsAndUpdateDegrees(graph, bisection, moves, degrees);

  EXPECT_EQ(num_swaps_performed, 2);
  EXPECT_THAT(bisection.LeftDataNodes(), ElementsAre(2, 3));
  EXPECT_THAT(bisection.RightDataNodes(), ElementsAre(0, 1));

  // After swapping 0 with 2, and 1 with 3:
  // L={2,3}, R={0,1}
  // New degrees:
  // deg[0] should be degrees of R in old partition
  // deg[1] should be degrees of L in old partition
  EXPECT_THAT(degrees[0], ElementsAre(1, 1, 1, 2, 1));
  EXPECT_THAT(degrees[1], ElementsAre(1, 1, 1, 0, 0));
}

TEST(ComputeCompactQueryNodeIdsTest, AllDataNodes) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  // DN 0 -> {0,1}, DN 1 -> {2}, DN 2 -> {0,3,4}
  // All QNs {0,1,2,3,4} are covered.
  std::vector<GingerNodeId> data_nodes = {0, 1, 2};
  std::vector<GingerNodeId> query_node_ids(5);
  const GingerNodeId num_query_nodes = ComputeCompactQueryNodeIds(
      graph, data_nodes, absl::MakeSpan(query_node_ids));

  EXPECT_EQ(num_query_nodes, 5);
  EXPECT_THAT(query_node_ids, ElementsAre(0, 1, 2, 3, 4));
}

TEST(ComputeCompactQueryNodeIdsTest, SomeDataNodes) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> mapping = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(mapping));

  // DN 0 -> {0,1}, DN 2 -> {0,3,4}.
  // QNs covered: {0,1,3,4}.
  std::vector<GingerNodeId> data_nodes = {0, 2};
  std::vector<GingerNodeId> query_node_ids(5);
  const GingerNodeId num_query_nodes = ComputeCompactQueryNodeIds(
      graph, data_nodes, absl::MakeSpan(query_node_ids));

  EXPECT_EQ(num_query_nodes, 4);
  EXPECT_THAT(query_node_ids, ElementsAre(0, 1, 2, 2, 3));
}

TEST(ComputeSubgraphDataTest, SingleDataNode) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> map = {100, 4711, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(map));

  std::vector<GingerNodeId> data_nodes_in_subgraph = {1};
  std::vector<GingerNodeId> query_node_ids_map(5);
  ComputeCompactQueryNodeIds(graph, data_nodes_in_subgraph,
                             absl::MakeSpan(query_node_ids_map));

  std::vector<int64_t> subgraph_offsets(1);
  std::vector<GingerNodeId> subgraph_targets(1);
  std::vector<GingerNodeId> subgraph_mapping(1);
  ComputeSubgraphData(graph, data_nodes_in_subgraph, query_node_ids_map,
                      absl::MakeSpan(subgraph_offsets),
                      absl::MakeSpan(subgraph_targets),
                      absl::MakeSpan(subgraph_mapping));

  EXPECT_THAT(subgraph_offsets, ElementsAre(0));
  EXPECT_THAT(subgraph_targets, ElementsAre(0));
  EXPECT_THAT(subgraph_mapping, ElementsAre(4711));
}

TEST(ComputeSubgraphDataTest, SomeDataNodes) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> map = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(map));

  std::vector<GingerNodeId> data_nodes_in_subgraph = {0, 2};
  std::vector<GingerNodeId> query_node_ids_map(5);
  ComputeCompactQueryNodeIds(graph, data_nodes_in_subgraph,
                             absl::MakeSpan(query_node_ids_map));

  std::vector<int64_t> subgraph_offsets(2);
  std::vector<GingerNodeId> subgraph_targets(5);
  std::vector<GingerNodeId> subgraph_mapping(2);

  ComputeSubgraphData(graph, data_nodes_in_subgraph, query_node_ids_map,
                      absl::MakeSpan(subgraph_offsets),
                      absl::MakeSpan(subgraph_targets),
                      absl::MakeSpan(subgraph_mapping));

  EXPECT_THAT(subgraph_offsets, ElementsAre(0, 2));
  EXPECT_THAT(subgraph_targets, ElementsAre(0, 1, 0, 2, 3));
  EXPECT_THAT(subgraph_mapping, ElementsAre(100, 12));
}

TEST(ComputeSubgraphDataTest, AllDataNodes) {
  std::vector<int64_t> offsets = {0, 2, 3};
  std::vector<GingerNodeId> targets = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> map = {100, 0, 12};
  QueryDataGraph graph(/*num_query_nodes=*/5, absl::MakeSpan(offsets),
                       absl::MakeSpan(targets), absl::MakeSpan(map));

  std::vector<GingerNodeId> data_nodes_in_subgraph = {0, 1, 2};
  std::vector<GingerNodeId> query_node_ids_map(5);
  ComputeCompactQueryNodeIds(graph, data_nodes_in_subgraph,
                             absl::MakeSpan(query_node_ids_map));

  std::vector<int64_t> subgraph_offsets(3);
  std::vector<GingerNodeId> subgraph_targets(6);
  std::vector<GingerNodeId> subgraph_mapping(3);

  ComputeSubgraphData(graph, data_nodes_in_subgraph, query_node_ids_map,
                      absl::MakeSpan(subgraph_offsets),
                      absl::MakeSpan(subgraph_targets),
                      absl::MakeSpan(subgraph_mapping));

  EXPECT_THAT(subgraph_offsets, ElementsAre(0, 2, 3));
  EXPECT_THAT(subgraph_targets, ElementsAre(0, 1, 2, 0, 3, 4));
  EXPECT_THAT(subgraph_mapping, ElementsAre(100, 0, 12));
}

TEST(ExtractSubgraphsTest, OneSidedBisection) {
  std::vector<int64_t> offsets_vec = {0, 2, 3};
  std::vector<GingerNodeId> targets_vec = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> map_vec = {100, 0, 12};
  QueryDataGraph graph(5, absl::MakeSpan(offsets_vec),
                       absl::MakeSpan(targets_vec), absl::MakeSpan(map_vec));

  std::vector<int64_t> buffer_offsets(3);
  std::vector<GingerNodeId> buffer_targets(6);
  std::vector<GingerNodeId> buffer_mapping(3);
  QueryDataGraphBuffers graph_buffers = {
      .offsets = absl::MakeSpan(buffer_offsets),
      .targets = absl::MakeSpan(buffer_targets),
      .mapping = absl::MakeSpan(buffer_mapping)};

  std::vector<GingerNodeId> left_nodes(3);
  std::vector<GingerNodeId> right_nodes(0);
  std::vector<GingerNodeId> bisection_mapping(3);
  Bisection bisection(absl::MakeSpan(left_nodes), absl::MakeSpan(right_nodes),
                      absl::MakeSpan(bisection_mapping));
  bisection.InitArbitrarily();  // L={0,1,2}, R={}

  auto result = ExtractSubgraphs(graph, graph_buffers, bisection);

  // Check left subgraph
  EXPECT_EQ(result.left_subgraph.NumQueryNodes(), 5);
  EXPECT_EQ(result.left_subgraph.NumDataNodes(), 3);
  EXPECT_EQ(result.left_subgraph.NumEdges(), 6);
  EXPECT_THAT(result.left_subgraph.Offsets(), ElementsAre(0, 2, 3));
  EXPECT_THAT(result.left_subgraph.Targets(), ElementsAre(0, 1, 2, 0, 3, 4));
  EXPECT_THAT(result.left_subgraph.Mapping(), ElementsAre(100, 0, 12));

  // Check right subgraph
  EXPECT_EQ(result.right_subgraph.NumQueryNodes(), 0);
  EXPECT_EQ(result.right_subgraph.NumDataNodes(), 0);
  EXPECT_EQ(result.right_subgraph.NumEdges(), 0);
  EXPECT_THAT(result.right_subgraph.Offsets(), IsEmpty());
  EXPECT_THAT(result.right_subgraph.Targets(), IsEmpty());
  EXPECT_THAT(result.right_subgraph.Mapping(), IsEmpty());
}

TEST(ExtractSubgraphsTest, SplitThreeNodes) {
  std::vector<int64_t> offsets_vec = {0, 2, 3};
  std::vector<GingerNodeId> targets_vec = {0, 1, 2, 0, 3, 4};
  std::vector<GingerNodeId> map_vec = {100, 0, 12};
  QueryDataGraph graph(5, absl::MakeSpan(offsets_vec),
                       absl::MakeSpan(targets_vec), absl::MakeSpan(map_vec));

  std::vector<int64_t> buffer_offsets(3);
  std::vector<GingerNodeId> buffer_targets(6);
  std::vector<GingerNodeId> buffer_mapping(3);
  QueryDataGraphBuffers graph_buffers = {
      .offsets = absl::MakeSpan(buffer_offsets),
      .targets = absl::MakeSpan(buffer_targets),
      .mapping = absl::MakeSpan(buffer_mapping)};

  std::vector<GingerNodeId> left_nodes(1);
  std::vector<GingerNodeId> right_nodes(2);
  std::vector<GingerNodeId> bisection_mapping(3);
  Bisection bisection(absl::MakeSpan(left_nodes), absl::MakeSpan(right_nodes),
                      absl::MakeSpan(bisection_mapping));
  bisection.InitArbitrarily();  // L={0}, R={1,2}

  auto result = ExtractSubgraphs(graph, graph_buffers, bisection);

  // Check left subgraph
  EXPECT_EQ(result.left_subgraph.NumQueryNodes(), 2);
  EXPECT_EQ(result.left_subgraph.NumDataNodes(), 1);
  EXPECT_EQ(result.left_subgraph.NumEdges(), 2);
  EXPECT_THAT(result.left_subgraph.Offsets(), ElementsAre(0));
  EXPECT_THAT(result.left_subgraph.Targets(), ElementsAre(0, 1));
  EXPECT_THAT(result.left_subgraph.Mapping(), ElementsAre(100));

  // Check right subgraph
  EXPECT_EQ(result.right_subgraph.NumQueryNodes(), 4);
  EXPECT_EQ(result.right_subgraph.NumDataNodes(), 2);
  EXPECT_EQ(result.right_subgraph.NumEdges(), 4);
  EXPECT_THAT(result.right_subgraph.Offsets(), ElementsAre(0, 1));
  EXPECT_THAT(result.right_subgraph.Targets(), ElementsAre(1, 0, 2, 3));
  EXPECT_THAT(result.right_subgraph.Mapping(), ElementsAre(0, 12));
}

TEST(CopyGraphTest, EmptyGraph) {
  CSRGraph graph = CreateCSRGraph({});

  std::vector<int64_t> offsets;
  std::vector<GingerNodeId> targets;
  CopyGraph(graph, absl::MakeSpan(offsets), absl::MakeSpan(targets));

  EXPECT_THAT(offsets, IsEmpty());
  EXPECT_THAT(targets, IsEmpty());
}

TEST(CopyGraphTest, GraphWithNoEdges) {
  CSRGraph graph = CreateCSRGraph({{}, {}, {}});

  std::vector<int64_t> offsets(3);
  std::vector<GingerNodeId> targets;
  CopyGraph(graph, absl::MakeSpan(offsets), absl::MakeSpan(targets));

  EXPECT_THAT(offsets, ElementsAre(0, 0, 0));
  EXPECT_THAT(targets, IsEmpty());
}

TEST(CopyGraphTest, GraphWithEdges) {
  CSRGraph graph = CreateCSRGraph({{1, 2}, {0, 2}, {1}});

  std::vector<int64_t> offsets(3);
  std::vector<GingerNodeId> targets(5);
  CopyGraph(graph, absl::MakeSpan(offsets), absl::MakeSpan(targets));

  EXPECT_THAT(offsets, ElementsAre(0, 2, 4));
  EXPECT_THAT(targets, ElementsAre(1, 2, 0, 2, 1));
}

TEST(ComputeReorderingTest, EdgeAndIsolatedNode) {
  // Create a graph with an edge between nodes 0 and 2, and an isolated node 1.
  CSRGraph graph = CreateCSRGraph({{2}, {}, {0}});
  auto reordering = ComputeReordering(graph);

  // We expect the isolated node 2 to be moved to the end.
  EXPECT_THAT(reordering, UnorderedElementsAre(0, 1, 2));
  EXPECT_EQ(reordering[2], 2);
}

TEST(ComputeReorderingTest, DisconnectedGraph) {
  // Create a graph with two disconnected components.
  CSRGraph graph = CreateCSRGraph({{2}, {3}, {0}, {1}});
  auto reordering = ComputeReordering(graph);
  ASSERT_EQ(reordering.size(), 4);

  // We expect the bisection to separate the two components.
  std::vector<GingerNodeId> first_half = {reordering[0], reordering[1]};
  std::vector<GingerNodeId> second_half = {reordering[2], reordering[3]};
  if (reordering[0] == 0 || reordering[0] == 1) {
    EXPECT_THAT(first_half, UnorderedElementsAre(0, 1));
    EXPECT_THAT(second_half, UnorderedElementsAre(2, 3));
  } else {
    EXPECT_THAT(first_half, UnorderedElementsAre(2, 3));
    EXPECT_THAT(second_half, UnorderedElementsAre(0, 1));
  }
}

TEST(ComputeReorderingTest, StarGraph) {
  // Create a star graph with kNumNodes. Node kNumNodes - 1 is the center,
  // connected to nodes 0 to kNumNodes - 2.
  constexpr GingerNodeId kNumNodes = 101;
  constexpr GingerNodeId kCenterNode = kNumNodes - 1;
  std::vector<std::vector<GingerNodeId>> adjacency_list(kNumNodes);

  for (GingerNodeId u = 0; u < kNumNodes; ++u) {
    if (u == kCenterNode) continue;
    adjacency_list[u].push_back(kCenterNode);
    adjacency_list[kCenterNode].push_back(u);
  }

  CSRGraph graph = CreateCSRGraph(adjacency_list);
  auto reordering = ComputeReordering(graph);

  std::vector<GingerNodeId> expected_nodes(kNumNodes);
  std::iota(expected_nodes.begin(), expected_nodes.end(), 0);
  EXPECT_THAT(reordering, UnorderedElementsAreArray(expected_nodes));

  // The reordering is expected to place the center of the start at the front.
  EXPECT_EQ(reordering[kCenterNode], 0);
}

}  // namespace
}  // namespace ginger::internal
