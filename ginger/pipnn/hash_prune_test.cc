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

#include "ginger/pipnn/hash_prune.h"

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"
#include "ginger/pipnn/simhash.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::FloatNear;
using ::testing::Ge;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::Lt;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

TEST(HashPruneTest, MergeAndGetNeighbors) {
  std::vector<float> p0 = {1.0f, 0.0f};
  std::vector<float> p1 = {0.9f, 0.1f};
  std::vector<float> p2 = {0.0f, 1.0f};
  std::vector<float> p3 = {0.1f, 0.9f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2, p3};

  const size_t kReservoirSize = 2;
  const size_t kNumProjections = 4;
  SimHashProjections simhash(data, 2, kNumProjections);

  HashPrune tables(kReservoirSize, std::move(simhash));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(0, neighbors);
  EXPECT_THAT(neighbors, IsEmpty());

  std::vector<DistanceAndId> edges = {{.distance = 0.1f, .id = 1},
                                      {.distance = 1.5f, .id = 2},
                                      {.distance = 1.4f, .id = 3}};

  tables.MergeNeighbors(0, absl::MakeSpan(edges));

  tables.GetNeighbors(0, neighbors);
  EXPECT_THAT(neighbors, SizeIs(Le(kReservoirSize)));
  for (const auto& neighbor : neighbors) {
    EXPECT_THAT(neighbor.id, AllOf(Ge(0), Lt(4)));
    EXPECT_THAT(neighbor.distance, Ge(0.0f));
  }

  // Test bulk export
  std::vector<std::vector<DistanceAndId>> all_neighbors =
      tables.ExportAllNeighbors();
  EXPECT_THAT(all_neighbors, SizeIs(4));
  EXPECT_THAT(all_neighbors[0], SizeIs(neighbors.size()));
  EXPECT_THAT(all_neighbors[1], IsEmpty());
  EXPECT_THAT(all_neighbors[2], IsEmpty());
  EXPECT_THAT(all_neighbors[3], IsEmpty());
}

TEST(HashPruneTest, DeduplicationOnHashCollision) {
  // Construct points such that p1 and p2 have identical coordinates and thus
  // guaranteed identical SimHashes relative to p0.
  std::vector<float> p0 = {0.0f, 0.0f};
  std::vector<float> p1 = {1.0f, 1.0f};
  std::vector<float> p2 = {1.0f, 1.0f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2};

  SimHashProjections simhash(data, 2, 8);
  HashPrune tables(5, std::move(simhash));

  // Merge p2 first (farther distance 0.5f)
  std::vector<DistanceAndId> edge1 = {{.distance = 0.5f, .id = 2}};
  tables.MergeNeighbors(0, absl::MakeSpan(edge1));

  // Merge p1 second (closer distance 0.1f)
  std::vector<DistanceAndId> edge2 = {{.distance = 0.1f, .id = 1}};
  tables.MergeNeighbors(0, absl::MakeSpan(edge2));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(0, neighbors);

  // Since p1 and p2 collide on hash, HashPrune should keep only the nearer p1.
  EXPECT_THAT(neighbors, ElementsAre(AllOf(Field(&DistanceAndId::id, 1),
                                           Field(&DistanceAndId::distance,
                                                 FloatNear(0.1f, 1e-2f)))));
}

TEST(HashPruneTest, ReservoirFullReplacesWorst) {
  // Orthogonal points to ensure distinct SimHashes
  std::vector<float> p0 = {0.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {1.0f, 0.0f, 0.0f};
  std::vector<float> p2 = {0.0f, 1.0f, 0.0f};
  std::vector<float> p3 = {0.0f, 0.0f, 1.0f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2, p3};

  SimHashProjections simhash(data, 3, 16);
  // Restrict reservoir size to 2
  HashPrune tables(2, std::move(simhash));

  std::vector<DistanceAndId> initial_edges = {{.distance = 1.0f, .id = 1},
                                              {.distance = 2.0f, .id = 2}};
  tables.MergeNeighbors(0, absl::MakeSpan(initial_edges));

  // Now insert a better candidate (distance 0.5f)
  std::vector<DistanceAndId> better_edge = {{.distance = 0.5f, .id = 3}};
  tables.MergeNeighbors(0, absl::MakeSpan(better_edge));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(0, neighbors);

  // The worst candidate (id 2, dist 2.0f) should have been evicted/replaced.
  EXPECT_THAT(neighbors, UnorderedElementsAre(Field(&DistanceAndId::id, 1),
                                              Field(&DistanceAndId::id, 3)));
}

TEST(HashPruneTest, IgnoresWorseWhenFull) {
  std::vector<float> p0 = {0.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {1.0f, 0.0f, 0.0f};
  std::vector<float> p2 = {0.0f, 1.0f, 0.0f};
  std::vector<float> p3 = {0.0f, 0.0f, 1.0f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2, p3};

  SimHashProjections simhash(data, 3, 16);
  HashPrune tables(2, std::move(simhash));

  std::vector<DistanceAndId> initial_edges = {{.distance = 0.2f, .id = 1},
                                              {.distance = 0.4f, .id = 2}};
  tables.MergeNeighbors(0, absl::MakeSpan(initial_edges));

  // Insert a worse candidate (distance 5.0f)
  std::vector<DistanceAndId> worse_edge = {{.distance = 5.0f, .id = 3}};
  tables.MergeNeighbors(0, absl::MakeSpan(worse_edge));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(0, neighbors);

  EXPECT_THAT(neighbors, UnorderedElementsAre(Field(&DistanceAndId::id, 1),
                                              Field(&DistanceAndId::id, 2)));
}

TEST(HashPruneTest, MultipleSourcesIndependent) {
  std::vector<float> p0 = {0.0f, 0.0f};
  std::vector<float> p1 = {1.0f, 0.0f};
  std::vector<float> p2 = {0.0f, 1.0f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2};

  SimHashProjections simhash(data, 2, 8);
  HashPrune tables(5, std::move(simhash));

  std::vector<DistanceAndId> edges0 = {{.distance = 0.5f, .id = 1}};
  std::vector<DistanceAndId> edges1 = {{.distance = 0.3f, .id = 2}};

  tables.MergeNeighbors(0, absl::MakeSpan(edges0));
  tables.MergeNeighbors(1, absl::MakeSpan(edges1));

  auto all_neighbors = tables.ExportAllNeighbors();
  EXPECT_THAT(
      all_neighbors,
      ElementsAre(ElementsAre(Field(&DistanceAndId::id, 1)),
                  ElementsAre(Field(&DistanceAndId::id, 2)), IsEmpty()));
}

TEST(HashPruneTest, SimdScanWithLargeReservoir) {
  // To trigger the SIMD loop in FindHashSimdImpl, deg must be >= N (e.g. 32 for
  // AVX-512). We generate 500 random points in 16D space to ensure at least 64
  // distinct SimHashes.
  const int kNumPoints = 500;
  const int kDim = 16;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<std::vector<float>> point_vecs(kNumPoints,
                                             std::vector<float>(kDim));
  std::vector<absl::Span<const float>> data;
  for (int i = 0; i < kNumPoints; ++i) {
    for (int j = 0; j < kDim; ++j) {
      point_vecs[i][j] = dist(rng);
    }
    data.push_back(point_vecs[i]);
  }

  SimHashProjections simhash(data, kDim, 16);
  HashPrune tables(64, std::move(simhash));

  std::vector<DistanceAndId> edges;
  for (int i = 1; i < kNumPoints; ++i) {
    edges.push_back(
        {.distance = 0.01f * i, .id = static_cast<GingerNodeId>(i)});
  }
  tables.MergeNeighbors(0, absl::MakeSpan(edges));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(0, neighbors);
  EXPECT_THAT(neighbors, SizeIs(64));
}

TEST(HashPruneTest, HashCollisionWorseCandidateIgnored) {
  std::vector<float> p0 = {0.0f, 0.0f};
  std::vector<float> p1 = {1.0f, 1.0f};
  std::vector<float> p2 = {1.0f, 1.0f};  // Collides with p1
  std::vector<absl::Span<const float>> data = {p0, p1, p2};

  SimHashProjections simhash(data, 2, 8);
  HashPrune tables(5, std::move(simhash));

  // Merge p1 first (closer distance 0.1f)
  std::vector<DistanceAndId> edge1 = {{.distance = 0.1f, .id = 1}};
  tables.MergeNeighbors(0, absl::MakeSpan(edge1));

  // Merge p2 second (farther distance 0.5f). Since it collides on hash with p1
  // but has a worse distance, it should be ignored.
  std::vector<DistanceAndId> edge2 = {{.distance = 0.5f, .id = 2}};
  tables.MergeNeighbors(0, absl::MakeSpan(edge2));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(0, neighbors);

  EXPECT_THAT(neighbors, ElementsAre(AllOf(Field(&DistanceAndId::id, 1),
                                           Field(&DistanceAndId::distance,
                                                 FloatNear(0.1f, 1e-2f)))));
}

TEST(HashPruneTest, HashCollisionReplacesWorstCandidate) {
  std::vector<float> p0 = {0.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {1.0f, 0.0f, 0.0f};
  std::vector<float> p2 = {0.0f, 1.0f, 0.0f};
  std::vector<float> p3 = {0.0f, 1.0f, 0.0f};  // Collides with p2
  std::vector<float> p4 = {0.0f, 0.0f, 1.0f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2, p3, p4};

  SimHashProjections simhash(data, 3, 16);
  HashPrune tables(2, std::move(simhash));

  // Insert p1 (dist 0.5f) and p2 (dist 1.0f). p2 becomes max_val (worst).
  std::vector<DistanceAndId> initial_edges = {{.distance = 0.5f, .id = 1},
                                              {.distance = 1.0f, .id = 2}};
  tables.MergeNeighbors(0, absl::MakeSpan(initial_edges));

  // Insert p3 (dist 0.2f), which collides on hash with p2.
  // This triggers `pos == max_idx`, which calls `recompute_worst()`.
  // The new worst distance in the table becomes p1 (dist 0.5f).
  std::vector<DistanceAndId> collision_edge = {{.distance = 0.2f, .id = 3}};
  tables.MergeNeighbors(0, absl::MakeSpan(collision_edge));

  // Now insert p4 (dist 0.4f). Since the table is full (size 2) and p4's
  // distance (0.4f) is better than the new worst distance (p1, dist 0.5f), p4
  // should replace p1.
  std::vector<DistanceAndId> final_edge = {{.distance = 0.4f, .id = 4}};
  tables.MergeNeighbors(0, absl::MakeSpan(final_edge));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(0, neighbors);

  EXPECT_THAT(neighbors, UnorderedElementsAre(Field(&DistanceAndId::id, 3),
                                              Field(&DistanceAndId::id, 4)));
}

TEST(HashPruneTest, RecomputeWorstSelectsNonZeroIndex) {
  std::vector<float> p0 = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> p2 = {0.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> p3 = {0.0f, 0.0f, 1.0f, 0.0f};
  std::vector<float> p4 = {0.0f, 0.0f, 0.0f, 1.0f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2, p3, p4};

  SimHashProjections simhash(data, 4, 16);
  HashPrune tables(3, std::move(simhash));

  // Insert p1 (dist 1.0f), p2 (dist 2.0f), p3 (dist 3.0f). p3 becomes max_idx
  // (index 2).
  std::vector<DistanceAndId> initial_edges = {{.distance = 1.0f, .id = 1},
                                              {.distance = 2.0f, .id = 2},
                                              {.distance = 3.0f, .id = 3}};
  tables.MergeNeighbors(0, absl::MakeSpan(initial_edges));

  // Insert p4 (dist 0.5f), which replaces p3 (index 2).
  // recompute_worst() is called. mv starts at d[0] (1.0f).
  // When j = 1, v = d[1] (2.0f). v > mv (2.0f > 1.0f) is true, setting mi = 1,
  // mv = 2.0f.
  std::vector<DistanceAndId> better_edge = {{.distance = 0.5f, .id = 4}};
  tables.MergeNeighbors(0, absl::MakeSpan(better_edge));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(0, neighbors);

  EXPECT_THAT(neighbors, UnorderedElementsAre(Field(&DistanceAndId::id, 1),
                                              Field(&DistanceAndId::id, 2),
                                              Field(&DistanceAndId::id, 4)));
}

TEST(HashPruneTest, SkipsWorseEdgeWithoutEvaluatingHash) {
  std::vector<float> p0 = {0.0f, 0.0f};
  std::vector<float> p1 = {1.0f, 0.0f};
  std::vector<absl::Span<const float>> data = {p0, p1};
  SimHashProjections simhash(data, 2, 8);

  HashPrune tables(1, std::move(simhash));
  std::vector<DistanceAndId> edge1 = {{.distance = 0.1f, .id = 1}};
  tables.MergeNeighbors(0, absl::MakeSpan(edge1));

  // Now the reservoir is full (size 1, max_val = 0.1f).
  // Pass an edge with dist = 10.0f and an invalid id (999999).
  std::vector<DistanceAndId> invalid_edge = {{.distance = 10.0f, .id = 999999}};
  tables.MergeNeighbors(0, absl::MakeSpan(invalid_edge));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(0, neighbors);
  EXPECT_THAT(neighbors, SizeIs(1));
}

TEST(HashPruneTest, FillsLastPointAndUsesAllTableSlots) {
  const int kNumPoints = 50;
  const int kDim = 16;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<std::vector<float>> point_vecs(kNumPoints,
                                             std::vector<float>(kDim));
  std::vector<absl::Span<const float>> data;
  for (int i = 0; i < kNumPoints; ++i) {
    for (int j = 0; j < kDim; ++j) {
      point_vecs[i][j] = dist(rng);
    }
    data.push_back(point_vecs[i]);
  }

  SimHashProjections simhash(data, kDim, 16);
  const size_t kReservoirSize = 32;
  HashPrune tables(kReservoirSize, std::move(simhash));

  const GingerNodeId last_point = kNumPoints - 1;
  std::vector<DistanceAndId> edges;
  for (int i = 0; i < kNumPoints - 1; ++i) {
    edges.push_back(
        {.distance = 0.01f * (i + 1), .id = static_cast<GingerNodeId>(i)});
  }
  tables.MergeNeighbors(last_point, absl::MakeSpan(edges));

  std::vector<DistanceAndId> neighbors;
  tables.GetNeighbors(last_point, neighbors);
  EXPECT_THAT(neighbors, SizeIs(kReservoirSize));
}

TEST(HashPruneTest, DeterministicTieBreaking) {
  std::vector<float> p0 = {0.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {1.0f, 0.0f, 0.0f};
  std::vector<float> p2 = {0.0f, 1.0f, 0.0f};
  std::vector<float> p3 = {0.0f, 0.0f, 1.0f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2, p3};

  SimHashProjections simhash1(data, 3, 16);
  SimHashProjections simhash2(data, 3, 16);

  // We set reservoir_size to 2.
  HashPrune tables1(2, std::move(simhash1));
  HashPrune tables2(2, std::move(simhash2));

  // Edges all with the exact same distance (1.5f).
  DistanceAndId e1 = {.distance = 1.5f, .id = 1};
  DistanceAndId e2 = {.distance = 1.5f, .id = 2};
  DistanceAndId e3 = {.distance = 1.5f, .id = 3};

  // Insertion order A: 1, 2, 3
  std::vector<DistanceAndId> edges_a = {e1, e2, e3};
  tables1.MergeNeighbors(0, absl::MakeSpan(edges_a));

  // Insertion order B: 3, 2, 1
  std::vector<DistanceAndId> edges_b = {e3, e2, e1};
  tables2.MergeNeighbors(0, absl::MakeSpan(edges_b));

  std::vector<DistanceAndId> neighbors1;
  tables1.GetNeighbors(0, neighbors1);

  std::vector<DistanceAndId> neighbors2;
  tables2.GetNeighbors(0, neighbors2);

  // Verify that both tables ended up with the exact same set of neighbors.
  EXPECT_THAT(neighbors1, UnorderedElementsAreArray(neighbors2));
}

}  // namespace
}  // namespace ginger
