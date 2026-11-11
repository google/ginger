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

#ifndef GINGER_REORDERING_RECURSIVE_GRAPH_CONTRACTION_IMPL_H_
#define GINGER_REORDERING_RECURSIVE_GRAPH_CONTRACTION_IMPL_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"
#include "parlay/sequence.h"

namespace ginger {
namespace internal {
namespace rgc {

// An edge consists of two endpoints and an edge weight. We use `std::pair`
// instead of a struct because the parlay interface expects pairs for reduce
// operations.
using Edge = std::pair<std::pair<GingerNodeId, GingerNodeId>, float>;

// A TaggedEdge is an edge stored together with its associated index, referring
// to its position inside an array.
using TaggedEdge = std::pair<Edge, int64_t>;

// A MatchingEdge consists of two endpoints `u` and `v`, the size of the
// endpoint `u` before contraction.
struct MatchingEdge {
  GingerNodeId u;
  GingerNodeId v;
  GingerNodeId u_size;
};

// A TaggedPriority is a priority score together with its associated index,
// referring to its position inside an array. We use a double instead of a float
// because this avoids padding bytes between the weight and the index, which
// would be uninitialized when constructing this value and can cause memory
// sanitizer errors due to atomic operations.
struct TaggedPriority {
  double weight;
  int64_t i;
};
using AtomicTaggedPriority = std::atomic<TaggedPriority>;

// The minimum priority score used to initialize the priorities buffer.
inline constexpr float kUnitEdgeWeight = 1.0f;
inline constexpr float kMinEdgeWeight = std::numeric_limits<float>::lowest();
inline constexpr int64_t kInvalidEdgeIndex = -1;
inline constexpr TaggedPriority kMinTaggedPriority = {kMinEdgeWeight,
                                                      kInvalidEdgeIndex};

// Returns true if the priority score of `a` is greater than `b`, breaking ties
// by comparing the edge index.
constexpr bool IsPriorityGreater(TaggedPriority a, TaggedPriority b) {
  return a.weight > b.weight || (a.weight == b.weight && a.i > b.i);
}

// Copies the graph from an adjacency list based format to a list of nodes ID's
// present in the graph and an edge list consisting of all edges in the graph.
template <typename Graph>
std::pair<parlay::sequence<GingerNodeId>, parlay::sequence<Edge>> CopyGraph(
    const Graph& graph);

// Computes a matching on the current set of `nodes`. For each edge (u, v), a
// priority is calculated based on `w(u,v) / (|u| * |v|)`. An edge is included
// in the matching if it has the highest priority for both endpoints `u` and
// `v`. The sizes of the matched nodes are updated, and `parent_buffer` is
// updated to reflect the contraction. Returns a sequence of `MatchingEdge`
// structs containing the matched pairs and their sizes before contraction.
parlay::sequence<MatchingEdge> ComputeMatching(
    const parlay::sequence<GingerNodeId>& nodes,
    const parlay::sequence<Edge>& edges,
    parlay::sequence<AtomicTaggedPriority>& priorities_buffer,
    parlay::sequence<GingerNodeId>& parent_buffer,
    absl::Span<GingerNodeId> sizes);

// Updates the graph representation after a contraction step. This involves
// filtering out nodes that have been contracted, updating the endpoints of
// edges to refer to their new supernode parents, and combining any resulting
// parallel edges by summing their weights.
void UpdateGraph(parlay::sequence<GingerNodeId>& nodes,
                 parlay::sequence<Edge>& edges,
                 const parlay::sequence<GingerNodeId>& parent_buffer);

// Base case for the recursive graph contraction. When no edges remain, all
// current `nodes` are isolated (super-)nodes. We write each node's offset into
// the `sizes` buffer.
void HandleBaseCase(const parlay::sequence<GingerNodeId>& nodes,
                    absl::Span<GingerNodeId> sizes);

// Updates the `sizes` array to store the starting offset for each supernode
// created by a matching edge in the final reordering.
void WriteOffsets(const parlay::sequence<MatchingEdge>& matching,
                  absl::Span<GingerNodeId> sizes);

}  // namespace rgc

// Computes a node reordering by recursively contracting the graph.
template <typename Graph>
std::vector<GingerNodeId> ComputeRecursiveGraphContractionReordering(
    const Graph& graph, RecursiveGraphContractionReorderingConfig config = {});

}  // namespace internal
}  // namespace ginger

#endif  // GINGER_REORDERING_RECURSIVE_GRAPH_CONTRACTION_IMPL_H_
