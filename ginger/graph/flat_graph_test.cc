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

#include "ginger/graph/flat_graph.h"

#include <utility>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"

namespace ginger {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(FlatGraphTest, EmptyGraph) {
  FlatGraph graph(0, 0);
  EXPECT_EQ(graph.NumNodes(), 0);
}

TEST(FlatGraphTest, GraphWithNodesButNoEdges) {
  FlatGraph graph(3, 5);
  EXPECT_EQ(graph.NumNodes(), 3);
  EXPECT_EQ(graph.Degree(0), 0);
  EXPECT_EQ(graph.Degree(1), 0);
  EXPECT_EQ(graph.Degree(2), 0);
  EXPECT_THAT(graph.Neighbors(0), IsEmpty());
  EXPECT_THAT(graph.Neighbors(1), IsEmpty());
  EXPECT_THAT(graph.Neighbors(2), IsEmpty());
}

TEST(FlatGraphTest, SetNeighbors) {
  FlatGraph graph(4, 3);
  graph.SetNeighbors(0, {1, 2});
  graph.SetNeighbors(1, {0});
  graph.SetNeighbors(2, {0, 3});
  graph.SetNeighbors(3, {2});

  EXPECT_EQ(graph.NumNodes(), 4);
  EXPECT_EQ(graph.Degree(0), 2);
  EXPECT_EQ(graph.Degree(1), 1);
  EXPECT_EQ(graph.Degree(2), 2);
  EXPECT_EQ(graph.Degree(3), 1);

  EXPECT_THAT(graph.Neighbors(0), ElementsAre(1, 2));
  EXPECT_THAT(graph.Neighbors(1), ElementsAre(0));
  EXPECT_THAT(graph.Neighbors(2), ElementsAre(0, 3));
  EXPECT_THAT(graph.Neighbors(3), ElementsAre(2));
}

TEST(FlatGraphTest, SetNeighborsMaxDegree) {
  FlatGraph graph(3, 2);
  graph.SetNeighbors(0, {1, 0});
  graph.SetNeighbors(1, {2});
  graph.SetNeighbors(2, {1});

  EXPECT_EQ(graph.Degree(0), 2);
  EXPECT_EQ(graph.Degree(1), 1);
  EXPECT_EQ(graph.Degree(2), 1);
  EXPECT_THAT(graph.Neighbors(0), ElementsAre(1, 0));
  EXPECT_THAT(graph.Neighbors(1), ElementsAre(2));
  EXPECT_THAT(graph.Neighbors(2), ElementsAre(1));
}

TEST(FlatGraphTest, OverwriteNeighbors) {
  FlatGraph graph(2, 2);
  graph.SetNeighbors(0, {1});
  EXPECT_THAT(graph.Neighbors(0), ElementsAre(1));
  EXPECT_EQ(graph.Degree(0), 1);

  graph.SetNeighbors(0, {1, 0});
  EXPECT_THAT(graph.Neighbors(0), ElementsAre(1, 0));
  EXPECT_EQ(graph.Degree(0), 2);

  graph.SetNeighbors(0, {});
  EXPECT_THAT(graph.Neighbors(0), IsEmpty());
  EXPECT_EQ(graph.Degree(0), 0);
}

TEST(FlatGraphTest, AppendNeighbors) {
  FlatGraph graph(2, 3);
  graph.AppendNeighbors(0, {1});
  EXPECT_THAT(graph.Neighbors(0), ElementsAre(1));
  EXPECT_EQ(graph.Degree(0), 1);

  graph.AppendNeighbors(0, {0, 1});
  EXPECT_THAT(graph.Neighbors(0), ElementsAre(1, 0, 1));
  EXPECT_EQ(graph.Degree(0), 3);
}

TEST(FlatGraphDeathTest, SetNeighborsExceedsMaxDegree) {
  FlatGraph graph(1, 1);
  EXPECT_DEATH(graph.SetNeighbors(0, {0, 1}),
               "Degree 2 for node 0 exceeds max_degree_ 1");
}

TEST(FlatGraphDeathTest, AppendNeighborsExceedsMaxDegree) {
  FlatGraph graph(1, 2);
  graph.SetNeighbors(0, {0});
  EXPECT_DEATH(graph.AppendNeighbors(0, {0, 1}),
               "New degree 3 for node 0 exceeds max_degree_ 2");
}

TEST(FlatGraphTest, CopyConstructorAndAssignment) {
  FlatGraph original(2, 3);
  original.SetNeighbors(0, {1});
  original.SetNeighbors(1, {0});

  FlatGraph copy(original);
  EXPECT_EQ(copy.NumNodes(), 2);
  EXPECT_EQ(copy.MaxDegree(), 3);
  EXPECT_THAT(copy.Neighbors(0), ElementsAre(1));
  EXPECT_THAT(copy.Neighbors(1), ElementsAre(0));

  // Verify deep copy: mutating copy does not modify original.
  copy.SetNeighbors(0, {1, 0});
  EXPECT_THAT(copy.Neighbors(0), ElementsAre(1, 0));
  EXPECT_THAT(original.Neighbors(0), ElementsAre(1));

  // Copy assignment.
  FlatGraph assigned(1, 1);
  assigned = original;
  EXPECT_EQ(assigned.NumNodes(), 2);
  EXPECT_EQ(assigned.MaxDegree(), 3);
  EXPECT_THAT(assigned.Neighbors(0), ElementsAre(1));
  EXPECT_THAT(assigned.Neighbors(1), ElementsAre(0));
}

TEST(FlatGraphTest, MoveConstructorAndAssignment) {
  FlatGraph original(2, 3);
  original.SetNeighbors(0, {1});

  FlatGraph moved(std::move(original));
  EXPECT_EQ(moved.NumNodes(), 2);
  EXPECT_EQ(moved.MaxDegree(), 3);
  EXPECT_THAT(moved.Neighbors(0), ElementsAre(1));

  FlatGraph moved_assign(1, 1);
  moved_assign = std::move(moved);
  EXPECT_EQ(moved_assign.NumNodes(), 2);
  EXPECT_EQ(moved_assign.MaxDegree(), 3);
  EXPECT_THAT(moved_assign.Neighbors(0), ElementsAre(1));
}

}  // namespace
}  // namespace ginger
