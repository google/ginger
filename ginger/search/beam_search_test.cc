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

#include "ginger/search/beam_search.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/base/optimization.h"
#include "absl/types/span.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/compressed_graph_builder.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/search/visited_set.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

// Mock graph class for testing.
class MockGraph {
 public:
  explicit MockGraph(GingerNodeId num_nodes) : adjacency_list_(num_nodes) {}

  void AddEdge(GingerNodeId u, GingerNodeId v) {
    adjacency_list_[u].push_back(v);
    adjacency_list_[v].push_back(u);
  }

  absl::Span<const GingerNodeId> Neighbors(GingerNodeId node_id) const {
    return absl::MakeSpan(adjacency_list_[node_id]);
  }

  GingerNodeId NumNodes() const { return adjacency_list_.size(); }

 private:
  std::vector<std::vector<GingerNodeId>> adjacency_list_;
};

TEST(UpdateBeamViaSetUnionTest, EmptyBeamEmptyCandidates) {
  internal::SearchToken s;
  s.Reset(5);
  EXPECT_THAT(s.Beam(), IsEmpty());
  s.AddToBeam({});
  EXPECT_THAT(s.Beam(), IsEmpty());
}

TEST(UpdateBeamViaSetUnionTest, EmptyBeam) {
  internal::SearchToken s;
  s.Reset(5);
  std::vector<DistanceAndId> candidates({{0.3f, 3}, {0.2f, 2}, {0.1f, 1}});
  s.AddToBeam(absl::MakeSpan(candidates));
  auto beam = s.Beam();
  EXPECT_EQ(beam.size(), 3);
  EXPECT_FLOAT_EQ(beam[0].distance, 0.1f);
  EXPECT_EQ(beam[0].ExtractNodeId(), 1);
  EXPECT_FLOAT_EQ(beam[1].distance, 0.2f);
  EXPECT_EQ(beam[1].ExtractNodeId(), 2);
  EXPECT_FLOAT_EQ(beam[2].distance, 0.3f);
  EXPECT_EQ(beam[2].ExtractNodeId(), 3);
}

TEST(UpdateBeamViaSetUnionTest, EmptyCandidates) {
  internal::SearchToken s;
  s.Reset(5);
  std::vector<DistanceAndId> initial_beam({{0.1f, 1}, {0.2f, 2}, {0.3f, 3}});
  s.AddToBeam(absl::MakeSpan(initial_beam));
  auto beam = s.Beam();
  EXPECT_EQ(beam.size(), 3);
  EXPECT_FLOAT_EQ(beam[0].distance, 0.1f);
  EXPECT_EQ(beam[0].ExtractNodeId(), 1);
  EXPECT_FLOAT_EQ(beam[1].distance, 0.2f);
  EXPECT_EQ(beam[1].ExtractNodeId(), 2);
  EXPECT_FLOAT_EQ(beam[2].distance, 0.3f);
  EXPECT_EQ(beam[2].ExtractNodeId(), 3);
}

TEST(UpdateBeamViaSetUnionTest, Disjoint) {
  internal::SearchToken s;
  s.Reset(5);
  std::vector<DistanceAndId> initial_beam({{0.1f, 1}, {0.2f, 2}});
  s.AddToBeam(absl::MakeSpan(initial_beam));
  std::vector<DistanceAndId> candidates({{0.3f, 3}, {0.4f, 4}});
  s.AddToBeam(absl::MakeSpan(candidates));
  auto beam = s.Beam();
  EXPECT_EQ(beam.size(), 4);
  EXPECT_FLOAT_EQ(beam[0].distance, 0.1f);
  EXPECT_EQ(beam[0].ExtractNodeId(), 1);
  EXPECT_FLOAT_EQ(beam[1].distance, 0.2f);
  EXPECT_EQ(beam[1].ExtractNodeId(), 2);
  EXPECT_FLOAT_EQ(beam[2].distance, 0.3f);
  EXPECT_EQ(beam[2].ExtractNodeId(), 3);
  EXPECT_FLOAT_EQ(beam[3].distance, 0.4f);
  EXPECT_EQ(beam[3].ExtractNodeId(), 4);
}

TEST(UpdateBeamViaSetUnionTest, BeamSizeLimit) {
  internal::SearchToken s;
  s.Reset(3);
  std::vector<DistanceAndId> initial_beam({{0.1f, 1}, {0.5f, 5}});
  s.AddToBeam(absl::MakeSpan(initial_beam));
  std::vector<DistanceAndId> candidates({{0.2f, 2}, {0.3f, 3}, {0.4f, 4}});
  s.AddToBeam(absl::MakeSpan(candidates));
  auto beam = s.Beam();
  EXPECT_EQ(beam.size(), 3);
  EXPECT_FLOAT_EQ(beam[0].distance, 0.1f);
  EXPECT_EQ(beam[0].ExtractNodeId(), 1);
  EXPECT_FLOAT_EQ(beam[1].distance, 0.2f);
  EXPECT_EQ(beam[1].ExtractNodeId(), 2);
  EXPECT_FLOAT_EQ(beam[2].distance, 0.3f);
  EXPECT_EQ(beam[2].ExtractNodeId(), 3);
}

template <typename Graph, typename DistanceComparator>
BeamSearchResult RunNavigationBeamSearchImpl(
    const Graph& graph, DistanceComparator&& compute_distances,
    absl::Span<const GingerNodeId> entry_points, int max_beam_size,
    internal::VisitedStrategy strategy, bool emit_explored = false) {
  internal::SearchToken s;
  switch (strategy) {
    case internal::VisitedStrategy::kBitvector: {
      internal::BitvectorVisitedTracker tracker;
      return internal::NavigationBeamSearchImpl(
          graph, std::forward<DistanceComparator>(compute_distances),
          entry_points, max_beam_size, s, tracker, emit_explored);
    }
    case internal::VisitedStrategy::kFastTable: {
      internal::FastTableVisitedTracker tracker;
      return internal::NavigationBeamSearchImpl(
          graph, std::forward<DistanceComparator>(compute_distances),
          entry_points, max_beam_size, s, tracker, emit_explored);
    }
    case internal::VisitedStrategy::kFlatHashSet: {
      internal::FlatHashSetVisitedTracker tracker;
      return internal::NavigationBeamSearchImpl(
          graph, std::forward<DistanceComparator>(compute_distances),
          entry_points, max_beam_size, s, tracker, emit_explored);
    }
  }
  ABSL_UNREACHABLE();
}

class NavigationBeamSearchTest
    : public ::testing::TestWithParam<internal::VisitedStrategy> {};

TEST(NavigationBeamSearchDispatcherTest, DefaultDispatcher) {
  MockGraph graph(5);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(1, 3);
  graph.AddEdge(2, 4);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results =
      NavigationBeamSearch(graph, nav_cmp, absl::MakeSpan(entry_points), 3);

  EXPECT_THAT(results.final_beam,
              UnorderedElementsAre(FieldsAre(0.0f, 0), FieldsAre(1.0f, 1),
                                   FieldsAre(2.0f, 2)));
}

TEST_P(NavigationBeamSearchTest, BasicTest) {
  MockGraph graph(5);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(1, 3);
  graph.AddEdge(2, 4);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam());

  EXPECT_THAT(results.final_beam,
              UnorderedElementsAre(FieldsAre(0.0f, 0), FieldsAre(1.0f, 1),
                                   FieldsAre(2.0f, 2)));
}

TEST_P(NavigationBeamSearchTest, BeamSizeLimit) {
  MockGraph graph(5);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(0, 3);
  graph.AddEdge(0, 4);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 2, GetParam());

  EXPECT_THAT(results.final_beam,
              UnorderedElementsAre(FieldsAre(0.0f, 0), FieldsAre(1.0f, 1)));
}

TEST_P(NavigationBeamSearchTest, MultipleEntryPoints) {
  MockGraph graph(5);
  graph.AddEdge(0, 1);
  graph.AddEdge(2, 3);
  graph.AddEdge(3, 4);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      // Distance is node_id for this test
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0, 2};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 4, GetParam());

  EXPECT_THAT(results.final_beam,
              UnorderedElementsAre(FieldsAre(0.0f, 0), FieldsAre(1.0f, 1),
                                   FieldsAre(2.0f, 2), FieldsAre(3.0f, 3)));
}

TEST_P(NavigationBeamSearchTest, PruneOddIdsFromDistanceComparisons) {
  MockGraph graph(5);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(0, 3);
  graph.AddEdge(0, 4);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      if (neighbor.id % 2 != 0) {
        neighbor.distance = std::numeric_limits<float>::infinity();
      } else {
        neighbor.distance = static_cast<float>(neighbor.id);
      }
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam());

  EXPECT_THAT(results.final_beam,
              UnorderedElementsAre(FieldsAre(0.0f, 0), FieldsAre(2.0f, 2),
                                   FieldsAre(4.0f, 4)));
}

TEST_P(NavigationBeamSearchTest, FilterCandidatesByRemoval) {
  MockGraph graph(5);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(0, 3);
  graph.AddEdge(0, 4);

  // Filter out odd IDs by shrinking the span, instead of setting distance
  // to infinity.
  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    auto it =
        std::remove_if(neighbors.begin(), neighbors.end(),
                       [](const DistanceAndId& d) { return d.id % 2 != 0; });
    neighbors = neighbors.first(std::distance(neighbors.begin(), it));
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam());

  // Only even IDs should be in the beam.
  EXPECT_THAT(results.final_beam,
              UnorderedElementsAre(FieldsAre(0.0f, 0), FieldsAre(2.0f, 2),
                                   FieldsAre(4.0f, 4)));

  // Distance computations should only count the kept candidates.
  // Entry point 0: 1 candidate (id=0), kept.
  // Node 0 has neighbors {1,2,3,4}: after filtering odd IDs, 2 kept (2,4).
  // Nodes 2 and 4 have neighbor 0, already seen, so 0 new candidates.
  EXPECT_EQ(results.stats.num_distance_computations, 3);
}

TEST_P(NavigationBeamSearchTest, FilterEntryPointByRemoval) {
  MockGraph graph(3);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);

  // Filter out the entry point. Node 1 is scored before node 2 but is farther
  // away, so it must not be explored first.
  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    auto it = std::remove_if(neighbors.begin(), neighbors.end(),
                             [](const DistanceAndId& d) { return d.id == 0; });
    neighbors = neighbors.first(std::distance(neighbors.begin(), it));
    for (auto& neighbor : neighbors) {
      neighbor.distance = neighbor.id == 1 ? 2.0f : 1.0f;
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam(),
      /*emit_explored=*/true);

  EXPECT_THAT(results.final_beam,
              ElementsAre(FieldsAre(1.0f, 2), FieldsAre(2.0f, 1)));
  EXPECT_THAT(results.explored,
              ElementsAre(FieldsAre(1.0f, 2), FieldsAre(2.0f, 1)));
  EXPECT_EQ(results.stats.num_candidates_considered, 3);
  EXPECT_EQ(results.stats.num_distance_computations, 2);
  EXPECT_EQ(results.stats.num_nodes_visited, 2);
}

TEST_P(NavigationBeamSearchTest, FilterEntryPointByInfiniteDistance) {
  MockGraph graph(3);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);

  // Filter out the entry point by setting its distance to infinity. Node 1 is
  // scored before node 2 but is farther away, so it must not be explored first.
  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      if (neighbor.id == 0) {
        neighbor.distance = std::numeric_limits<float>::infinity();
      } else {
        neighbor.distance = neighbor.id == 1 ? 2.0f : 1.0f;
      }
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam(),
      /*emit_explored=*/true);

  EXPECT_THAT(results.final_beam,
              ElementsAre(FieldsAre(1.0f, 2), FieldsAre(2.0f, 1)));
  EXPECT_THAT(results.explored,
              ElementsAre(FieldsAre(1.0f, 2), FieldsAre(2.0f, 1)));
  EXPECT_EQ(results.stats.num_candidates_considered, 3);
  EXPECT_EQ(results.stats.num_distance_computations, 3);
  EXPECT_EQ(results.stats.num_nodes_visited, 2);
}

TEST_P(NavigationBeamSearchTest, FilterAllCandidatesByRemoval) {
  MockGraph graph(2);
  graph.AddEdge(0, 1);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    return neighbors.first(0);
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam(),
      /*emit_explored=*/true);

  EXPECT_THAT(results.final_beam, IsEmpty());
  EXPECT_THAT(results.explored, IsEmpty());
  EXPECT_EQ(results.stats.num_candidates_considered, 2);
  EXPECT_EQ(results.stats.num_distance_computations, 0);
  EXPECT_EQ(results.stats.num_nodes_visited, 0);
}

TEST_P(NavigationBeamSearchTest, FilterEntryPointExpandsSurvivingNeighbors) {
  MockGraph graph(4);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(1, 3);

  // Filter out the entry point. Node 3 is only reachable through node 1.
  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    auto it = std::remove_if(neighbors.begin(), neighbors.end(),
                             [](const DistanceAndId& d) { return d.id == 0; });
    neighbors = neighbors.first(std::distance(neighbors.begin(), it));
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 4, GetParam(),
      /*emit_explored=*/true);

  EXPECT_THAT(
      results.final_beam,
      ElementsAre(FieldsAre(1.0f, 1), FieldsAre(2.0f, 2), FieldsAre(3.0f, 3)));
  EXPECT_THAT(
      results.explored,
      ElementsAre(FieldsAre(1.0f, 1), FieldsAre(2.0f, 2), FieldsAre(3.0f, 3)));
  EXPECT_EQ(results.stats.num_distance_computations, 3);
  EXPECT_EQ(results.stats.num_nodes_visited, 3);
}

TEST_P(NavigationBeamSearchTest, DisconnectedComponent) {
  MockGraph graph(5);
  graph.AddEdge(0, 1);
  graph.AddEdge(2, 3);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 4, GetParam());

  EXPECT_THAT(results.final_beam,
              UnorderedElementsAre(FieldsAre(0.0f, 0), FieldsAre(1.0f, 1)));
}

TEST_P(NavigationBeamSearchTest, StatisticsAccuracy) {
  MockGraph graph(5);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(1, 3);
  graph.AddEdge(2, 4);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam());

  EXPECT_EQ(results.stats.num_distance_computations, 5);
  EXPECT_EQ(results.stats.num_nodes_visited, 3);
}

TEST_P(NavigationBeamSearchTest, StatisticsAccuracy_CompleteGraph) {
  MockGraph graph(4);
  // Complete graph on 4 nodes
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(0, 3);
  graph.AddEdge(1, 2);
  graph.AddEdge(1, 3);
  graph.AddEdge(2, 3);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 4, GetParam());

  // Entry point 0 -> 1 computation.
  // Node 0 has neighbors 1, 2, 3 -> 3 computations.
  // All other nodes have all their neighbors already seen -> 0 computations.
  EXPECT_EQ(results.stats.num_distance_computations, 4);
  // All 4 nodes will be added to the beam and each will be visited (since beam
  // size is 4).
  EXPECT_EQ(results.stats.num_nodes_visited, 4);
}

TEST_P(NavigationBeamSearchTest, StatisticsAccuracy_EmptyGraph) {
  MockGraph graph(3);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam());

  // Entry point 0 -> 1 computation.
  // Node 0 has no neighbors -> 0 computations.
  EXPECT_EQ(results.stats.num_distance_computations, 1);
  // Only node 0 is visited.
  EXPECT_EQ(results.stats.num_nodes_visited, 1);
}

TEST_P(NavigationBeamSearchTest, EmitExplored) {
  MockGraph graph(5);
  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(1, 3);
  graph.AddEdge(2, 4);

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  std::vector<GingerNodeId> entry_points = {0};
  BeamSearchResult results_not_emitted = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam(),
      /*emit_explored=*/false);
  EXPECT_THAT(results_not_emitted.explored, IsEmpty());

  BeamSearchResult results = RunNavigationBeamSearchImpl(
      graph, nav_cmp, absl::MakeSpan(entry_points), 3, GetParam(),
      /*emit_explored=*/true);
  EXPECT_THAT(
      results.explored,
      ElementsAre(FieldsAre(0.0f, 0), FieldsAre(1.0f, 1), FieldsAre(2.0f, 2)));
}

INSTANTIATE_TEST_SUITE_P(
    AllVisitedStrategies, NavigationBeamSearchTest,
    ::testing::Values(internal::VisitedStrategy::kBitvector,
                      internal::VisitedStrategy::kFastTable,
                      internal::VisitedStrategy::kFlatHashSet),
    [](const ::testing::TestParamInfo<internal::VisitedStrategy>& info) {
      switch (info.param) {
        case internal::VisitedStrategy::kBitvector:
          return "Bitvector";
        case internal::VisitedStrategy::kFastTable:
          return "FastTable";
        case internal::VisitedStrategy::kFlatHashSet:
          return "FlatHashSet";
      }
      return "Unknown";
    });

CSRGraph CreateCSRGraph(std::vector<std::vector<GingerNodeId>> adjacency_list) {
  size_t num_arcs = 0;
  for (auto& neighbors : adjacency_list) {
    std::sort(neighbors.begin(), neighbors.end());
    num_arcs += neighbors.size();
  }

  std::vector<int64_t> offsets = {0};
  std::vector<GingerNodeId> targets;
  targets.reserve(num_arcs);
  for (const auto& neighbors : adjacency_list) {
    for (const auto neighbor : neighbors) {
      targets.push_back(neighbor);
    }
    offsets.push_back(targets.size());
  }
  return CSRGraph(std::move(offsets), std::move(targets));
}

template <typename Graph>
class NavigationBeamSearchPrefetchableGraphTest : public ::testing::Test {
 protected:
  static Graph CreateTestGraph(const CSRGraph& csr) {
    if constexpr (std::is_same_v<Graph, CSRGraph>) {
      return csr;
    } else if constexpr (std::is_same_v<Graph, CompressedCSRGraph<>>) {
      return CompressGraph<CSRGraph, CompressedGraphConfig>(csr);
    } else if constexpr (std::is_same_v<Graph, AdaptiveCompressedGraph>) {
      return CompressGraph<CSRGraph, AdaptiveCompressedGraphConfig>(csr);
    }
  }

  template <typename DistanceComparator>
  void RunAndAssertIdentical(const CSRGraph& csr_graph,
                             DistanceComparator&& nav_cmp,
                             absl::Span<const GingerNodeId> entry_points,
                             int max_beam_size, bool emit_explored = false) {
    BeamSearchResult baseline = NavigationBeamSearch(
        csr_graph, nav_cmp, entry_points, max_beam_size, emit_explored);

    Graph graph = CreateTestGraph(csr_graph);
    BeamSearchResult actual = NavigationBeamSearch(
        graph, nav_cmp, entry_points, max_beam_size, emit_explored);

    EXPECT_EQ(actual.final_beam, baseline.final_beam);
    if (emit_explored) {
      EXPECT_EQ(actual.explored, baseline.explored);
    }
    EXPECT_EQ(actual.stats.num_candidates_considered,
              baseline.stats.num_candidates_considered);
    EXPECT_EQ(actual.stats.num_distance_computations,
              baseline.stats.num_distance_computations);
    EXPECT_EQ(actual.stats.num_nodes_visited, baseline.stats.num_nodes_visited);
  }
};

using PrefetchableGraphTypes =
    ::testing::Types<CSRGraph, CompressedCSRGraph<>, AdaptiveCompressedGraph>;
TYPED_TEST_SUITE(NavigationBeamSearchPrefetchableGraphTest,
                 PrefetchableGraphTypes);

TYPED_TEST(NavigationBeamSearchPrefetchableGraphTest, BasicSingleEntryPoint) {
  CSRGraph graph = CreateCSRGraph({{1, 2}, {0, 3}, {0, 4}, {1}, {2}});

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  const std::vector<GingerNodeId> entry_points = {0};
  this->RunAndAssertIdentical(graph, nav_cmp, absl::MakeSpan(entry_points), 3,
                              /*emit_explored=*/true);
  this->RunAndAssertIdentical(graph, nav_cmp, absl::MakeSpan(entry_points), 2,
                              /*emit_explored=*/false);
  this->RunAndAssertIdentical(graph, nav_cmp, absl::MakeSpan(entry_points), 5,
                              /*emit_explored=*/true);
}

TYPED_TEST(NavigationBeamSearchPrefetchableGraphTest, MultipleEntryPoints) {
  CSRGraph graph = CreateCSRGraph({{1}, {0}, {3}, {2, 4}, {3}});

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  const std::vector<GingerNodeId> entry_points = {0, 2};
  this->RunAndAssertIdentical(graph, nav_cmp, absl::MakeSpan(entry_points), 4,
                              /*emit_explored=*/true);
  this->RunAndAssertIdentical(graph, nav_cmp, absl::MakeSpan(entry_points), 2,
                              /*emit_explored=*/false);
}

TYPED_TEST(NavigationBeamSearchPrefetchableGraphTest,
           MultiHopRingGraphWithMultipleQueries) {
  constexpr int kNumNodes = 32;
  std::vector<std::vector<GingerNodeId>> adj(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i) {
    adj[i].push_back((i + 1) % kNumNodes);
    adj[i].push_back((i + kNumNodes - 1) % kNumNodes);
    adj[i].push_back((i + 4) % kNumNodes);
    adj[i].push_back((i + kNumNodes - 4) % kNumNodes);
  }
  CSRGraph graph = CreateCSRGraph(adj);

  for (GingerNodeId target : {0, 7, 16, 29}) {
    auto target_cmp = [target](absl::Span<DistanceAndId> neighbors) {
      for (auto& neighbor : neighbors) {
        neighbor.distance = static_cast<float>(std::abs(neighbor.id - target));
      }
      return neighbors;
    };

    const std::vector<GingerNodeId> single_entry = {0};
    this->RunAndAssertIdentical(graph, target_cmp, absl::MakeSpan(single_entry),
                                4,
                                /*emit_explored=*/true);
    this->RunAndAssertIdentical(graph, target_cmp, absl::MakeSpan(single_entry),
                                16,
                                /*emit_explored=*/true);

    const std::vector<GingerNodeId> multi_entry = {0, 16};
    this->RunAndAssertIdentical(graph, target_cmp, absl::MakeSpan(multi_entry),
                                8, /*emit_explored=*/false);
    this->RunAndAssertIdentical(graph, target_cmp, absl::MakeSpan(multi_entry),
                                16, /*emit_explored=*/true);
  }
}

TYPED_TEST(NavigationBeamSearchPrefetchableGraphTest,
           DisconnectedAndIsolatedNodes) {
  CSRGraph graph = CreateCSRGraph({{1}, {0}, {}, {4}, {3}});

  auto nav_cmp = [](absl::Span<DistanceAndId> neighbors) {
    for (auto& neighbor : neighbors) {
      neighbor.distance = static_cast<float>(neighbor.id);
    }
    return neighbors;
  };

  const std::vector<GingerNodeId> entry_points = {0};
  this->RunAndAssertIdentical(graph, nav_cmp, absl::MakeSpan(entry_points), 3,
                              /*emit_explored=*/true);

  const std::vector<GingerNodeId> isolated_entry = {2};
  this->RunAndAssertIdentical(graph, nav_cmp, absl::MakeSpan(isolated_entry), 3,
                              /*emit_explored=*/true);
}

}  // namespace

namespace internal {

TEST(MergeBeamAndCandidatesTest, EmptyBeamEmptyCandidates) {
  std::vector<BeamEntry> beam;
  std::vector<DistanceAndId> candidates;
  size_t next_unexplored = 0;
  MergeBeamAndCandidates(beam, absl::MakeSpan(candidates), 5, next_unexplored);
  EXPECT_THAT(beam, IsEmpty());
}

TEST(MergeBeamAndCandidatesTest, EmptyBeam) {
  std::vector<BeamEntry> beam;
  std::vector<DistanceAndId> candidates = {{0.1f, 1}, {0.2f, 2}, {0.3f, 3}};
  size_t next_unexplored = 0;
  MergeBeamAndCandidates(beam, absl::MakeSpan(candidates), 5, next_unexplored);
  EXPECT_EQ(beam.size(), 3);
  EXPECT_FLOAT_EQ(beam[0].distance, 0.1f);
  EXPECT_EQ(beam[0].ExtractNodeId(), 1);
  EXPECT_FLOAT_EQ(beam[1].distance, 0.2f);
  EXPECT_EQ(beam[1].ExtractNodeId(), 2);
  EXPECT_FLOAT_EQ(beam[2].distance, 0.3f);
  EXPECT_EQ(beam[2].ExtractNodeId(), 3);
}

TEST(MergeBeamAndCandidatesTest, EmptyCandidates) {
  std::vector<BeamEntry> beam = {{0.1f, 1}, {0.2f, 2}};
  std::vector<DistanceAndId> candidates;
  size_t next_unexplored = 0;
  MergeBeamAndCandidates(beam, absl::MakeSpan(candidates), 5, next_unexplored);
  EXPECT_EQ(beam.size(), 2);
  EXPECT_FLOAT_EQ(beam[0].distance, 0.1f);
  EXPECT_EQ(beam[0].ExtractNodeId(), 1);
  EXPECT_FLOAT_EQ(beam[1].distance, 0.2f);
  EXPECT_EQ(beam[1].ExtractNodeId(), 2);
}

TEST(MergeBeamAndCandidatesTest, Disjoint) {
  std::vector<BeamEntry> beam = {{0.1f, 1}, {0.2f, 2}};
  std::vector<DistanceAndId> candidates = {{0.3f, 3}, {0.4f, 4}};
  size_t next_unexplored = 0;
  MergeBeamAndCandidates(beam, absl::MakeSpan(candidates), 5, next_unexplored);
  EXPECT_EQ(beam.size(), 4);
  EXPECT_FLOAT_EQ(beam[0].distance, 0.1f);
  EXPECT_EQ(beam[0].ExtractNodeId(), 1);
  EXPECT_FLOAT_EQ(beam[1].distance, 0.2f);
  EXPECT_EQ(beam[1].ExtractNodeId(), 2);
  EXPECT_FLOAT_EQ(beam[2].distance, 0.3f);
  EXPECT_EQ(beam[2].ExtractNodeId(), 3);
  EXPECT_FLOAT_EQ(beam[3].distance, 0.4f);
  EXPECT_EQ(beam[3].ExtractNodeId(), 4);
}

TEST(MergeBeamAndCandidatesTest, BeamSizeLimit) {
  std::vector<BeamEntry> beam = {{0.1f, 1}, {0.5f, 5}};
  std::vector<DistanceAndId> candidates = {{0.2f, 2}, {0.3f, 3}, {0.4f, 4}};
  size_t next_unexplored = 0;
  MergeBeamAndCandidates(beam, absl::MakeSpan(candidates), 3, next_unexplored);
  EXPECT_EQ(beam.size(), 3);
  EXPECT_FLOAT_EQ(beam[0].distance, 0.1f);
  EXPECT_EQ(beam[0].ExtractNodeId(), 1);
  EXPECT_FLOAT_EQ(beam[1].distance, 0.2f);
  EXPECT_EQ(beam[1].ExtractNodeId(), 2);
  EXPECT_FLOAT_EQ(beam[2].distance, 0.3f);
  EXPECT_EQ(beam[2].ExtractNodeId(), 3);
}

TEST(MergeBeamAndCandidatesTest, Interleaved) {
  std::vector<BeamEntry> beam = {{0.1f, 1}, {0.3f, 3}, {0.5f, 5}};
  std::vector<DistanceAndId> candidates = {{0.2f, 2}, {0.4f, 4}};
  size_t next_unexplored = 0;
  MergeBeamAndCandidates(beam, absl::MakeSpan(candidates), 5, next_unexplored);
  EXPECT_EQ(beam.size(), 5);
  EXPECT_FLOAT_EQ(beam[0].distance, 0.1f);
  EXPECT_EQ(beam[0].ExtractNodeId(), 1);
  EXPECT_FLOAT_EQ(beam[1].distance, 0.2f);
  EXPECT_EQ(beam[1].ExtractNodeId(), 2);
  EXPECT_FLOAT_EQ(beam[2].distance, 0.3f);
  EXPECT_EQ(beam[2].ExtractNodeId(), 3);
  EXPECT_FLOAT_EQ(beam[3].distance, 0.4f);
  EXPECT_EQ(beam[3].ExtractNodeId(), 4);
  EXPECT_FLOAT_EQ(beam[4].distance, 0.5f);
  EXPECT_EQ(beam[4].ExtractNodeId(), 5);
}

TEST(MergeBeamAndCandidatesTest,
     NextUnexploredUpdatesWhenCandidateInsertedBefore) {
  std::vector<BeamEntry> beam = {{0.1f, 1}, {0.3f, 3}};
  std::vector<DistanceAndId> candidates = {{0.2f, 2}};
  size_t next_unexplored = 2;
  MergeBeamAndCandidates(beam, absl::MakeSpan(candidates), 5, next_unexplored);
  EXPECT_EQ(next_unexplored, 1);
}

TEST(MergeBeamAndCandidatesTest,
     NextUnexploredUnchangedWhenCandidatesInsertedAfter) {
  std::vector<BeamEntry> beam = {{0.1f, 1}, {0.2f, 2}};
  std::vector<DistanceAndId> candidates = {{0.3f, 3}, {0.4f, 4}};
  size_t next_unexplored = 1;
  MergeBeamAndCandidates(beam, absl::MakeSpan(candidates), 5, next_unexplored);
  EXPECT_EQ(next_unexplored, 1);
}

TEST(SearchTokenTest, PeekDoesNotMarkExplored) {
  SearchToken s;
  s.Reset(5);
  std::vector<DistanceAndId> candidates = {{0.3f, 3}, {0.5f, 5}};
  s.AddToBeam(absl::MakeSpan(candidates));

  const DistanceAndId peek1 = s.PeekNodeToExplore();
  EXPECT_FLOAT_EQ(peek1.distance, 0.3f);
  EXPECT_EQ(peek1.id, 3);
  EXPECT_FALSE(s.Beam()[0].WasExplored());

  const DistanceAndId peek2 = s.PeekNodeToExplore();
  EXPECT_EQ(peek2, peek1);
  EXPECT_FALSE(s.Beam()[0].WasExplored());
}

TEST(SearchTokenTest, PeekFollowedByExtractReturnsSameNode) {
  SearchToken s;
  s.Reset(5);
  std::vector<DistanceAndId> candidates = {{0.2f, 2}, {0.4f, 4}};
  s.AddToBeam(absl::MakeSpan(candidates));

  const DistanceAndId peeked = s.PeekNodeToExplore();
  EXPECT_FLOAT_EQ(peeked.distance, 0.2f);
  EXPECT_EQ(peeked.id, 2);

  const DistanceAndId extracted = s.ExtractNodeToExplore();
  EXPECT_EQ(extracted, peeked);
  EXPECT_TRUE(s.Beam()[0].WasExplored());

  const DistanceAndId next_peek = s.PeekNodeToExplore();
  EXPECT_FLOAT_EQ(next_peek.distance, 0.4f);
  EXPECT_EQ(next_peek.id, 4);
}

TEST(SearchTokenTest, PeekReturnsNewHeadAfterBetterCandidateAdded) {
  SearchToken s;
  s.Reset(5);
  std::vector<DistanceAndId> candidates = {{0.3f, 3}, {0.5f, 5}};
  s.AddToBeam(absl::MakeSpan(candidates));

  EXPECT_EQ(s.PeekNodeToExplore(), (DistanceAndId{0.3f, 3}));

  // Insert a better candidate (lower distance).
  std::vector<DistanceAndId> better_candidate = {{0.1f, 1}};
  s.AddToBeam(absl::MakeSpan(better_candidate));

  // Peek returns the new head.
  EXPECT_EQ(s.PeekNodeToExplore(), (DistanceAndId{0.1f, 1}));
  EXPECT_EQ(s.ExtractNodeToExplore(), (DistanceAndId{0.1f, 1}));

  // After extracting the new head, peek returns the previous candidate.
  EXPECT_EQ(s.PeekNodeToExplore(), (DistanceAndId{0.3f, 3}));
}

TEST(SearchTokenTest, PeekOnEmptyOrFullyExploredBeam) {
  SearchToken s;
  s.Reset(5);
  EXPECT_EQ(s.PeekNodeToExplore().id, -1);
  EXPECT_EQ(s.ExtractNodeToExplore().id, -1);

  std::vector<DistanceAndId> candidates = {{0.1f, 1}};
  s.AddToBeam(absl::MakeSpan(candidates));
  EXPECT_EQ(s.ExtractNodeToExplore().id, 1);
  EXPECT_EQ(s.PeekNodeToExplore().id, -1);
  EXPECT_EQ(s.ExtractNodeToExplore().id, -1);
}

}  // namespace internal
}  // namespace ginger
