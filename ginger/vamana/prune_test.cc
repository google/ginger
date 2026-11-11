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

#include "ginger/vamana/prune.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

TEST(RobustPruneTest, SelfLoop) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {0.0f, 0},
      {1.0f, 1},
  };
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {};
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1));
}

TEST(RobustPruneTest, SelfLoopOnly) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {0.0f, 0},
  };
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {};
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, IsEmpty());
}

TEST(RobustPruneTest, DistZeroNonSelfLoop) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {0.0f, 1}, {1.0f, 2}, {2.0f, 5}, {0.0f, 3}, {0.0f, 4}};
  size_t max_degree = 3;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(std::abs(query - node.id));
    }
  };
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, UnorderedElementsAre(AnyOf(1, 3, 4), AnyOf(1, 3, 4), 2));
}

TEST(RobustPruneTest, PointWithZeroDistanceToEverything) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {{0.0f, 1},   {100.0f, 2},
                                           {200.0f, 3}, {300.0f, 4},
                                           {400.0f, 5}, {500.0f, 6}};
  size_t max_degree = 6;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      if (query == 1 || node.id == 1) {
        node.distance = 0.0f;
      } else {
        node.distance = 100 * static_cast<float>(std::abs(query - node.id));
      }
    }
  };

  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);
  prune_config.set_desired_degree(3);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  // Neighbor 1 must be selected. Since it is distance 0 to all other
  // candidates, it prunes all other candidates in RobustPrune.
  // In AlphaTuning, all other candidates have limit_dist = 0, and get
  // priority=+inf, so they are not added.
  EXPECT_THAT(result, UnorderedElementsAre(1));
}

TEST(RobustPruneTest, ZeroDistanceDedupingPositiveVsNegative) {
  // Test that candidates with distance 0 are deduped for positive distance
  // metrics, but NOT for non-positive distance metrics.
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {0.0f, 1}, {0.0f, 2}, {0.0f, 3}, {0.0f, 4}, {0.0f, 5}};
  size_t max_degree = 5;

  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = 0.0f;
    }
  };

  // Case 1: Positive distances (deduping should happen, keeping at most
  // max_dupe_candidates = 2)
  {
    RobustPruneConfig prune_config;
    prune_config.set_alpha(1.0);
    prune_config.set_nonpositive_distances(false);

    std::vector<DistanceAndId> candidates_copy = candidates;
    std::vector<GingerNodeId> result =
        RobustPrune(absl::MakeSpan(candidates_copy), node_id, max_degree,
                    compute_distances, prune_config);
    EXPECT_EQ(result.size(), 2);
  }

  // Case 2: Non-positive distances (deduping should not happen initially,
  // all 5 points should survive)
  {
    RobustPruneConfig prune_config;
    prune_config.set_alpha(1.0);
    prune_config.set_nonpositive_distances(true);

    std::vector<DistanceAndId> candidates_copy = candidates;
    std::vector<GingerNodeId> result =
        RobustPrune(absl::MakeSpan(candidates_copy), node_id, max_degree,
                    compute_distances, prune_config);
    EXPECT_EQ(result.size(), 5);
    EXPECT_THAT(result, UnorderedElementsAre(1, 2, 3, 4, 5));
  }
}

TEST(RobustPruneTest, EmptyCandidates) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates;
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {};
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, IsEmpty());
}

TEST(RobustPruneTest, MaxDegree) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {1.0f, 1},
      {2.0f, 2},
      {3.0f, 3},
      {4.0f, 4},
  };

  size_t max_degree = 2;

  // Distance function where candidates are far from each other so no pruning
  // happens.
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = 1000.0f;
    }
  };
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);

  EXPECT_THAT(result, ElementsAre(1, 2));
}

TEST(RobustPruneTest, TestVaryingAlpha) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {1.0f, 1},
      {2.0f, 2},
      {3.0f, 3},
  };
  size_t max_degree = 5;

  // Simple distance function: 1D line logic where coordinate = id.
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(std::abs(query - node.id));
    }
  };

  // Boundary checks for alpha:
  // A candidate c is pruned by a selected node s if alpha * dist(c, s) <
  // dist(c, query).
  RobustPruneConfig prune_config;

  // Boundary 1: alpha = 1.5
  // Affects if 3 is pruned by 1 (when 2 is pruned by 1).
  // alpha * 2 < 3 => alpha < 1.5
  prune_config.set_alpha(1.5f - 0.0001f);
  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1));

  prune_config.set_alpha(1.5f);
  result = RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                       compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1, 3));

  // Boundary 2: alpha = 2.0
  // Affects if 2 is pruned by 1.
  // alpha * 1 < 2 => alpha < 2.0
  prune_config.set_alpha(2.0f - 0.0001f);
  result = RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                       compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1, 3));

  prune_config.set_alpha(2.0f);
  result = RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                       compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1, 2));

  // Boundary 3: alpha = 3.0
  // Affects if 3 is pruned by 2 (when 2 is NOT pruned by 1).
  // alpha * 1 < 3 => alpha < 3.0
  prune_config.set_alpha(3.0f - 0.0001f);
  result = RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                       compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1, 2));

  prune_config.set_alpha(3.0f);
  result = RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                       compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1, 2, 3));
}

TEST(RobustPruneTest, UnsortedCandidates) {
  GingerNodeId node_id = 0;
  // Candidates are unsorted by distance.
  std::vector<DistanceAndId> candidates = {
      {3.0f, 3},
      {1.0f, 1},
      {2.0f, 2},
  };
  size_t max_degree = 5;

  // Simple distance function: 1D line logic where coordinate = id.
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(std::abs(query - node.id));
    }
  };

  // Logic is the same as TestVaryingAlpha with alpha=1.0.
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);
  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1));
}

TEST(RobustPruneTest, DuplicateCandidates) {
  GingerNodeId node_id = 0;
  // Candidates contain duplicates of node 1.
  std::vector<DistanceAndId> candidates = {
      {1.0f, 1},
      {1.0f, 1},
      {2.0f, 2},
  };
  size_t max_degree = 5;

  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      if (node.id == query) {
        node.distance = 0.0f;
      } else {
        node.distance = static_cast<float>(std::abs(query - node.id));
      }
    }
  };
  RobustPruneConfig prune_config;
  prune_config.set_alpha(3.0);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);

  // 1 should appear only once. 2 should also appear.
  EXPECT_THAT(result, ElementsAre(1, 2));
}

TEST(RobustPruneTest, PruningHappens) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {1.0f, 1},
      {1.1f, 2},
  };
  size_t max_degree = 5;

  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      if ((query == 1 && node.id == 2) || (query == 2 && node.id == 1)) {
        node.distance = 0.8f;
      } else {
        node.distance = static_cast<float>(std::abs(query - node.id));
      }
    }
  };
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.2);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);

  // Node 1 is selected first.
  // For node 2: threshold = 1.1 / 1.2 = 0.9166...
  // d(1, 2) = 0.8. Since 0.8 < 0.9166, node 2 is pruned.
  EXPECT_THAT(result, ElementsAre(1));
}

TEST(RobustPruneTest, NegativeDistancesInvertAlpha) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {-1.0f, 1},
      {-2.0f, 2},
  };
  size_t max_degree = 5;

  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = -1.5f;
    }
  };

  RobustPruneConfig prune_config;
  prune_config.set_nonpositive_distances(true);

  // Node 2 is closer to query node 0 than node 1, so node 2 is selected first
  // as a survivor. For node 1, we check if it is pruned by node 2. This pruning
  // occurs if d(1, 2) < d(0, 1) / alpha since we have negative distances.

  // With alpha = 2.0, the alpha is inverted to 0.5. The threshold is then
  // d(0, 1) / alpha = -1.0 / 0.5 = -2.0.
  // Since d(1, 2) = -1.5 is not < -2.0, node 1 is not pruned by node 2.
  prune_config.set_alpha(2.0);
  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(2, 1));

  // With alpha = 1.0, the alpha is inverted to 1.0. The threshold is then
  // d(0, 1) / alpha = -1.0 / 1.0 = -1.0.
  // Since d(1, 2) = -1.5 is < -1.0, node 1 is pruned by node 2.
  prune_config.set_alpha(1.0);
  result = RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                       compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(2));
}

TEST(RobustPruneTest, TieBreaking) {
  GingerNodeId node_id = 0;
  // Candidates 1 and 2 have same distance to node 0.
  // We pass them in reverse id order to test that the implementation tie-breaks
  // lexicographically.
  std::vector<DistanceAndId> candidates = {
      {1.0f, 2},
      {1.0f, 1},
  };
  size_t max_degree = 1;

  // Distance function where all nodes are distance 1 from each other.
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = (query == node.id) ? 0.0f : 1.0f;
    }
  };
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.2);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);

  // With max_degree=1, and candidates 1 and 2 being equidistant from 0,
  // node 1 should be selected due to smaller id.
  EXPECT_THAT(result, ElementsAre(1));
}

TEST(RobustPruneTest, BasicAlphaTuning) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {0.0f, 0}, {1.0f, 1}, {2.0f, 2}, {3.0f, 3}, {4.0f, 4},
  };
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(std::abs(query - node.id));
    }
  };
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);
  prune_config.set_desired_degree(3);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  // For alpha=1, only 1 is selected.
  // With desired_degree=3, AlphaTuning selects 4 (min_alpha = 1.33), then 2
  // (min_alpha = 2).
  EXPECT_THAT(result, ElementsAre(1, 4, 2));

  prune_config.set_desired_degree(1);
  result = RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                       compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1));
}

TEST(RobustPruneTest, AlphaTuningNoPruningForSmallCandidateSet) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {0.0f, 0},             // self-loop
      {1.0f, 1}, {1.0f, 1},  // duplicate
      {2.0f, 2}, {2.0f, 2},  // duplicate
      {0.0f, 0},             // self-loop
  };
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(std::abs(query - node.id));
    }
  };
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);
  prune_config.set_desired_degree(3);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1, 2));
}

TEST(RobustPruneTest, AlphaTuningMultipleRounds) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {1.0f, 1}, {2.0f, 2}, {3.0f, 3}, {4.0f, 4}, {5.0f, 5},
  };
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(std::abs(query - node.id));
    }
  };
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);
  prune_config.set_desired_degree(3);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, UnorderedElementsAre(1, 5, 3));
}

TEST(RobustPruneTest, AlphaTuningLargeDesiredDegree) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {1.0f, 1},
      {2.0f, 2},
      {3.0f, 3},
  };
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(std::abs(query - node.id));
    }
  };
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);
  prune_config.set_desired_degree(4);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1, 3, 2));
}

TEST(RobustPruneTest, AlphaTuningNegativeDistances) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {-1.0f, 1}, {-2.0f, 2}, {-3.0f, 3}, {-4.0f, 4}};
  size_t desired_degree = 3;
  size_t max_degree = 5;

  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    auto get_dist = [](int a, int b) -> float {
      if (a > b) std::swap(a, b);

      if (a == 0) {
        if (b == 1) return -1.0f;
        if (b == 2) return -2.0f;
        if (b == 3) return -3.0f;
        if (b == 4) return -4.0f;
      } else if (b == 4) {
        return -5.0f;
      } else if (a == 2 && b == 3) {
        return -12.0f;
      } else if (a == 1 && b == 3) {
        return -1.5f;
      } else if (a == 1 && b == 2) {
        return -2.0f;
      }

      return -100.0f;
    };

    for (auto& node : nodes) {
      node.distance = get_dist(query, node.id);
    }
  };

  RobustPruneConfig prune_config;
  prune_config.set_nonpositive_distances(true);
  prune_config.set_alpha(1.0);
  prune_config.set_desired_degree(desired_degree);

  // In the first pass:
  // - Candidates are sorted to: 4, 3, 2, 1.
  // - 4 is added to survivors.
  // - 3, 2, and 1 are pruned because their distances to 4 are -5.0
  // Surviving neighbors is initially {4}.
  //
  // In AlphaTuning:
  // 1. Critical alpha values against candidate 4:
  //   - Candidate 3: -5.0 / -3.0 = 1.6667 (smallest)
  //   - Candidate 2: -5.0 / -2.0 = 2.5
  //   - Candidate 1: -5.0 / -1.0 = 5.0
  //   - 3 is selected first and added to survivors (survivors: {4, 3}).
  // 2. PQ is updated with distances from new survivor 3:
  //   - Candidate 2: -12.0 / -2.0 = 6.0 -> priority is 6.0
  //   - Candidate 1: -1.5 / -1.0 = 1.5 -> priority remains 5.0 from survivor 4
  //   - 1 is selected next and added to survivors (survivors: {4, 3, 1}).
  // 3. Desired degree is reached, so AlphaTuning stops.
  //
  // Resulting survivors check out as: 4, 3, 1.
  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(4, 3, 1));
}

TEST(AlphaTuningTest, AlphaTuningAddsCandidate) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {{1.0f, 1}};
  size_t desired_degree = 2;
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(
          std::abs(static_cast<int>(query) - static_cast<int>(node.id)));
    }
  };
  std::vector<DistanceAndId> surviving_neighbors = {{0.0f, 2}};
  std::vector<bool> survived_pruning = {false};
  RobustPruneConfig prune_config;
  prune_config.set_desired_degree(desired_degree);
  internal::AlphaTuning(candidates, node_id, prune_config, max_degree,
                        compute_distances, surviving_neighbors,
                        survived_pruning);
  EXPECT_THAT(surviving_neighbors, ElementsAre(Field(&DistanceAndId::id, 2),
                                               Field(&DistanceAndId::id, 1)));
}

TEST(AlphaTuningTest, TieBreaking) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {{2.0f, 2}, {3.0f, 3}};
  size_t desired_degree = 2;
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      if (query == 2 && node.id == 1) {
        node.distance = 1.0f;
      } else if (query == 1 && node.id == 2) {
        node.distance = 1.0f;
      } else if (query == 3 && node.id == 1) {
        node.distance = 1.5f;
      } else if (query == 1 && node.id == 3) {
        node.distance = 1.5f;
      } else {
        node.distance = 100.0f;  // some large distance
      }
    }
  };
  std::vector<DistanceAndId> surviving_neighbors = {{0.0f, 1}};
  std::vector<bool> survived_pruning = {false, false};
  RobustPruneConfig prune_config;
  prune_config.set_desired_degree(desired_degree);
  internal::AlphaTuning(candidates, node_id, prune_config, max_degree,
                        compute_distances, surviving_neighbors,
                        survived_pruning);
  // With query = 0 and survivor = 1:
  // Priority(2) = 2.0 / 1.0 = 2.0
  // Priority(3) = 3.0 / 1.5 = 2.0
  // So candidates 2 and 3 both have priority 2.0. Since the candidates list is
  // sorted by id, tie-breaking selects candidate 2.
  EXPECT_THAT(surviving_neighbors, ElementsAre(Field(&DistanceAndId::id, 1),
                                               Field(&DistanceAndId::id, 2)));
}

TEST(AlphaTuningTest, PointsWithZeroDistanceNotAdded) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {{1.0f, 1}, {2.0f, 2}, {3.0f, 3}};
  size_t desired_degree = 3;
  size_t max_degree = 5;
  // All other nodes are duplicates (0 distance) from node 4.
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = 0;
    }
  };
  std::vector<DistanceAndId> surviving_neighbors = {{0.0f, 4}};
  std::vector<bool> survived_pruning = {false, false, false};
  RobustPruneConfig prune_config;
  prune_config.set_desired_degree(desired_degree);
  internal::AlphaTuning(candidates, node_id, prune_config, max_degree,
                        compute_distances, surviving_neighbors,
                        survived_pruning);
  EXPECT_THAT(surviving_neighbors, ElementsAre(Field(&DistanceAndId::id, 4)));
}

TEST(AlphaTuningTest, HitsDesiredDegree) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {{2.0f, 2}, {3.0f, 3}, {4.0f, 4}};
  size_t desired_degree = 3;
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(
          std::abs(static_cast<int>(query) - static_cast<int>(node.id)));
    }
  };
  // Start with one neighbor {1}.
  std::vector<DistanceAndId> surviving_neighbors = {{1.0f, 1}};
  std::vector<bool> survived_pruning = {false, false, false};
  RobustPruneConfig prune_config;
  prune_config.set_desired_degree(desired_degree);
  internal::AlphaTuning(candidates, node_id, prune_config, max_degree,
                        compute_distances, surviving_neighbors,
                        survived_pruning);
  // AlphaTuning should add 2 more neighbors to reach the desired_degree.
  // Based on the distances, nodes 4 and 2 are added, but 3 is not added.
  EXPECT_THAT(surviving_neighbors,
              UnorderedElementsAre(Field(&DistanceAndId::id, 1),
                                   Field(&DistanceAndId::id, 2),
                                   Field(&DistanceAndId::id, 4)));
}

TEST(AlphaTuningTest, RunsOutOfCandidates) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {{2.0f, 2}, {3.0f, 3}};
  // Desired degree is larger than the number of available candidates.
  size_t desired_degree = 4;
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(
          std::abs(static_cast<int>(query) - static_cast<int>(node.id)));
    }
  };
  std::vector<DistanceAndId> surviving_neighbors = {{1.0f, 1}};
  std::vector<bool> survived_pruning = {false, false};
  RobustPruneConfig prune_config;
  prune_config.set_desired_degree(desired_degree);
  internal::AlphaTuning(candidates, node_id, prune_config, max_degree,
                        compute_distances, surviving_neighbors,
                        survived_pruning);
  // AlphaTuning should add all available candidates.
  EXPECT_THAT(surviving_neighbors,
              UnorderedElementsAre(Field(&DistanceAndId::id, 1),
                                   Field(&DistanceAndId::id, 2),
                                   Field(&DistanceAndId::id, 3)));
}

TEST(AlphaTuningTest, SomeCandidatesHaveZeroDistance) {
  // Candidate 3 has zero distance to candidate 2.
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {{2.0f, 2}, {3.0f, 3}};
  size_t desired_degree = 3;
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      if ((query == 1 && node.id == 2) || (query == 2 && node.id == 1)) {
        node.distance = 2.0f;
      } else if ((query == 1 && node.id == 3) || (query == 3 && node.id == 1)) {
        node.distance = 3.0f;
      } else if ((query == 2 && node.id == 3) || (query == 3 && node.id == 2)) {
        node.distance = 0.0f;
      } else {
        node.distance = 100.0f;  // some large distance
      }
    }
  };
  std::vector<DistanceAndId> surviving_neighbors = {{1.0f, 1}};
  std::vector<bool> survived_pruning = {false, false};
  RobustPruneConfig prune_config;
  prune_config.set_desired_degree(desired_degree);
  internal::AlphaTuning(candidates, node_id, prune_config, max_degree,
                        compute_distances, surviving_neighbors,
                        survived_pruning);
  // Candidate 2 is added first. Candidate 3 is not added because its distance
  // to candidate 2 is zero, resulting in an infinite priority.
  EXPECT_THAT(surviving_neighbors,
              UnorderedElementsAre(Field(&DistanceAndId::id, 1),
                                   Field(&DistanceAndId::id, 2)));
}

TEST(AlphaTuningTest, ZeroDistanceNonPositive) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {{-2.0f, 2}, {-3.0f, 3}};
  size_t desired_degree = 3;
  size_t max_degree = 5;

  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      if ((query == 1 && node.id == 2) || (query == 2 && node.id == 1)) {
        node.distance = -2.0f;
      } else if ((query == 1 && node.id == 3) || (query == 3 && node.id == 1)) {
        node.distance = -3.0f;
      } else if ((query == 2 && node.id == 3) || (query == 3 && node.id == 2)) {
        node.distance = 0.0f;
      } else {
        node.distance = 0.0f;
      }
    }
  };

  RobustPruneConfig prune_config;
  prune_config.set_desired_degree(desired_degree);
  prune_config.set_nonpositive_distances(true);

  std::vector<DistanceAndId> surviving_neighbors = {{-1.0f, 1}};
  std::vector<bool> survived_pruning = {false, false};
  internal::AlphaTuning(candidates, node_id, prune_config, max_degree,
                        compute_distances, surviving_neighbors,
                        survived_pruning);

  // Surviving neighbors is initially {1}.
  //
  // In AlphaTuning:
  // 1. Critical alpha values against candidate 1:
  //   - Candidate 2: -2.0 / -2.0 = 1.0
  //   - Candidate 3: -3.0 / -3.0 = 1.0
  //   - 2 is selected first and added to survivors (survivors: {1, 2}).
  // 2. PQ is updated with distances from new survivor 2:
  //   - Candidate 3: distance to 2 is 0.0f (orthogonal) -> it is not updated!
  //     So its priority remains 1.0 (from initialization)
  //   - 3 is selected next and added to survivors (survivors: {1, 2, 3}).
  // 3. Desired degree is reached, so AlphaTuning stops.
  //
  // Resulting survivors check out as: 1, 2, 3.
  EXPECT_THAT(surviving_neighbors,
              UnorderedElementsAre(Field(&DistanceAndId::id, 1),
                                   Field(&DistanceAndId::id, 2),
                                   Field(&DistanceAndId::id, 3)));
}

TEST(RobustPruneTest, EmptyAfterSelfLoopRemovalWithAlphaTuning) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {1.0f, 0},
  };
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {};
  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0);
  prune_config.set_desired_degree(3);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, IsEmpty());
}

TEST(RobustPruneTest, KeepPrunedNeighborsFillsUpToMaxDegreeInDistanceOrder) {
  GingerNodeId node_id = 0;
  // 1D line: node 1 prunes nodes 2, 3, 4, 5 when alpha = 1.0, except let's
  // place node -1 at distance 1.5 in the opposite direction so {1, 6} survive
  // and {2, 3, 4, 5} are pruned.
  std::vector<DistanceAndId> candidates = {
      {4.0f, 4}, {1.0f, 1}, {3.0f, 3}, {1.5f, 6}, {2.0f, 2}, {5.0f, 5},
  };
  size_t max_degree = 4;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    auto pos = [](GingerNodeId id) -> float {
      return id == 6 ? -1.5f : static_cast<float>(id);
    };
    for (auto& node : nodes) {
      node.distance = std::abs(pos(query) - pos(node.id));
    }
  };

  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0f);
  prune_config.set_keep_pruned_neighbors(false);

  std::vector<DistanceAndId> candidates_copy = candidates;
  std::vector<GingerNodeId> without_keep =
      RobustPrune(absl::MakeSpan(candidates_copy), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(without_keep, ElementsAre(1, 6));

  prune_config.set_keep_pruned_neighbors(true);
  candidates_copy = candidates;
  std::vector<GingerNodeId> with_keep =
      RobustPrune(absl::MakeSpan(candidates_copy), node_id, max_degree,
                  compute_distances, prune_config);
  // Survivors {1, 6} followed by pruned candidates {2, 3} in increasing
  // distance order until max_degree = 4 is reached.
  EXPECT_THAT(with_keep, ElementsAre(1, 6, 2, 3));
}

TEST(RobustPruneTest, KeepPrunedNeighborsFewerCandidatesThanMaxDegree) {
  GingerNodeId node_id = 0;
  std::vector<DistanceAndId> candidates = {
      {0.0f, 0},
      {3.0f, 3},
      {1.0f, 1},
      {2.0f, 2},
  };
  size_t max_degree = 5;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    for (auto& node : nodes) {
      node.distance = static_cast<float>(std::abs(query - node.id));
    }
  };

  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0f);
  prune_config.set_keep_pruned_neighbors(true);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1, 2, 3));
}

TEST(RobustPruneTest, KeepPrunedNeighborsWithAlphaTuning) {
  GingerNodeId node_id = 0;
  // 1D line at positions +1, +2, +4, +5:
  // Initial pass (alpha = 1.0) keeps {1} and prunes {2, 3, 4}.
  // AlphaTuning (desired_degree = 3) first promotes 4 (priority 5/4 = 1.25),
  // then promotes 2 (priority 2/1 = 2.0 vs 3's updated priority 4/1 = 4.0).
  // Backfill (keep_pruned_neighbors = true, max_degree = 4) skips 2 (already
  // promoted by AlphaTuning) and appends 3.
  std::vector<DistanceAndId> candidates = {
      {1.0f, 1},
      {2.0f, 2},
      {4.0f, 3},
      {5.0f, 4},
  };
  size_t max_degree = 4;
  auto compute_distances = [](GingerNodeId query,
                              absl::Span<DistanceAndId> nodes) {
    auto pos = [](GingerNodeId id) -> float {
      if (id == 3) return 4.0f;
      if (id == 4) return 5.0f;
      return static_cast<float>(id);
    };
    for (auto& node : nodes) {
      node.distance = std::abs(pos(query) - pos(node.id));
    }
  };

  RobustPruneConfig prune_config;
  prune_config.set_alpha(1.0f);
  prune_config.set_desired_degree(3);
  prune_config.set_keep_pruned_neighbors(true);

  std::vector<GingerNodeId> result =
      RobustPrune(absl::MakeSpan(candidates), node_id, max_degree,
                  compute_distances, prune_config);
  EXPECT_THAT(result, ElementsAre(1, 4, 2, 3));
}

}  // namespace

}  // namespace ginger
