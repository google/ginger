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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_reference_stub.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/sequence.h"
#include "parlay/utilities.h"

namespace ginger {
namespace internal {
namespace rgc {

template <typename Graph>
std::pair<parlay::sequence<GingerNodeId>, parlay::sequence<Edge>> CopyGraph(
    const Graph& graph) {
  // Initialize `nodes` to contain all original nodes in the graph.
  parlay::sequence<GingerNodeId> nodes = parlay::tabulate(
      graph.NumNodes(), [](const GingerNodeId u) { return u; });

  // We write all non-self edges of the graph to an edge list consisting of both
  // endpoints together with a unit edge weight.
  parlay::sequence<Edge> edges =
      parlay::flatten(parlay::map(nodes, [&](const GingerNodeId u) {
        return parlay::map_maybe(
            graph.Neighbors(u),
            [&](const GingerNodeId v) -> std::optional<Edge> {
              if (u == v) return std::nullopt;
              const auto endpoints =
                  std::make_pair(std::min(u, v), std::max(u, v));
              return Edge(endpoints, kUnitEdgeWeight);
            });
      }));

  // Combine redundant edges by summing their weights and shuffle the edges.
  edges = parlay::reduce_by_key(edges, parlay::plus<float>());
  edges = parlay::random_shuffle(edges);

  return std::make_pair(std::move(nodes), std::move(edges));
}

template std::pair<parlay::sequence<GingerNodeId>, parlay::sequence<Edge>>
CopyGraph<CSRGraph>(const CSRGraph& graph);
template std::pair<parlay::sequence<GingerNodeId>, parlay::sequence<Edge>>
CopyGraph<FlatGraph>(const FlatGraph& graph);

parlay::sequence<MatchingEdge> ComputeMatching(
    const parlay::sequence<GingerNodeId>& nodes,
    const parlay::sequence<Edge>& edges,
    parlay::sequence<AtomicTaggedPriority>& priorities_buffer,
    parlay::sequence<GingerNodeId>& parent_buffer,
    absl::Span<GingerNodeId> sizes) {
  // For each edge (u, v), calculate a priority based on `w(u,v) / (|u| * |v|)`
  // and write it to the `priorities_buffer` for both endpoints. These
  // priorities are tagged with the edge index to break ties. Before computing
  // the priorities, we initialize the `priorities_buffer` for all active nodes
  // to a minimum value.
  parlay::for_each(nodes, [&](const GingerNodeId u) {
    priorities_buffer[u].store(kMinTaggedPriority);
  });
  parlay::parallel_for(0, edges.size(), [&](const int64_t i) {
    const Edge edge = edges[i];
    const auto [u, v] = edge.first;
    const float w = edge.second;

    const TaggedPriority priority{
        w / (static_cast<float>(sizes[u]) * static_cast<float>(sizes[v])), i};
    parlay::write_min(&priorities_buffer[v], priority, IsPriorityGreater);
    parlay::write_min(&priorities_buffer[u], priority, IsPriorityGreater);
  });

  // Get the matched edges that win on both sides and for those:
  // 1) update size of u to include size of v
  // 2) have parent_buffer[v] point to u
  // 3) return (u, v, old size of u) for use when returning from recursion
  return parlay::map_maybe(
      nodes, [&](const GingerNodeId u) -> std::optional<MatchingEdge> {
        // If the priority of u is invalid, then we have no edges incident to u
        // and we can skip it.
        const TaggedPriority u_priority = priorities_buffer[u].load();
        if (u_priority.i == kInvalidEdgeIndex) return std::nullopt;

        // Get the other endpoint of the edge incident to u with the highest
        // priority. As we only want each edge in the matching once, we only
        // add it for the node with the lower index.
        const auto& [x, y] = edges[u_priority.i].first;
        const GingerNodeId v = (u != x) ? x : y;
        if (u >= v) return std::nullopt;

        // If the edge incident with the highest priority to u is not the same
        // as the one incident to v, then we have no matching edge.
        const TaggedPriority v_priority = priorities_buffer[v].load();
        if (u_priority.i != v_priority.i) return std::nullopt;

        const GingerNodeId u_size = sizes[u];
        const GingerNodeId v_size = sizes[v];
        sizes[u] += v_size;
        parent_buffer[v] = u;
        return MatchingEdge(u, v, u_size);
      });
}

void UpdateGraph(parlay::sequence<GingerNodeId>& nodes,
                 parlay::sequence<Edge>& edges,
                 const parlay::sequence<GingerNodeId>& parent_buffer) {
  // Update `nodes` to only include remaining vertices after contraction
  nodes = parlay::filter(
      nodes, [&](const GingerNodeId v) { return parent_buffer[v] == v; });

  // Update edge endpoints orientation and remove any self edges
  edges = parlay::map_maybe(edges, [&](const Edge edge) -> std::optional<Edge> {
    const auto& [u, v] = edge.first;

    GingerNodeId pu = parent_buffer[u];
    GingerNodeId pv = parent_buffer[v];
    if (pu == pv) return std::nullopt;  // remove self edges
    if (pu > pv) std::swap(pu, pv);     // keep oriented low to high

    return Edge(std::make_pair(pu, pv), /*weight=*/edge.second);
  });

  // Combine redundant edges by summing their weights.
  edges = parlay::reduce_by_key(edges, parlay::plus<float>());
}

void HandleBaseCase(const parlay::sequence<GingerNodeId>& nodes,
                    absl::Span<GingerNodeId> sizes) {
  // Handles the base case of the recursion. This is reached when all remaining
  // (super-)nodes are isolated, meaning no more edges can be matched.
  // We compute an ordering for these super-nodes by performing a prefix sum
  // over their current sizes. The size of a super-node represents the total
  // number of original nodes contracted into it. By assigning each super-node
  // the offset produced by the prefix sum, we establish the starting position
  // for placing the original nodes within each super-node's range in the final
  // reordering.
  const auto [offsets, unused_sum] = parlay::scan(parlay::delayed_map(
      nodes, [&](const GingerNodeId u) { return sizes[u]; }));
  parlay::parallel_for(0, nodes.size(),
                       [&](const size_t i) { sizes[nodes[i]] = offsets[i]; });
}

void WriteOffsets(const parlay::sequence<MatchingEdge>& matching,
                  absl::Span<GingerNodeId> sizes) {
  parlay::for_each(matching, [&](const MatchingEdge edge) {
    const auto& [representative, contracted, representative_size] = edge;
    sizes[contracted] = sizes[representative] + representative_size;
  });
}

}  // namespace rgc

template <typename Graph>
std::vector<GingerNodeId> ComputeRecursiveGraphContractionReordering(
    const Graph& graph, RecursiveGraphContractionReorderingConfig config) {
  ginger::internal::ParallelSchedulerReference scheduler;
  auto [nodes, edges] = rgc::CopyGraph(graph);

  // Initialize the priorities buffer to contain one entry for each node. This
  // is used to store the priority score of the best incident edge.
  parlay::sequence<rgc::AtomicTaggedPriority> priorities_buffer(nodes.size());

  // Initialize the parent buffer to contain one entry for each node. This is
  // used to update the endpoints of matched edges to refer to their new
  // supernode parents. We initialize it to the original nodes
  parlay::sequence<GingerNodeId> parent_buffer = nodes;

  // On the way down the recursion, the `reordering` vector stores the sizes of
  // the supernodes. Since they are initially all just regular nodes, we
  // initialize the weight to one. On the way up, we update the offsets in the
  // reordering to store the new IDs for the supernodes at the current level.
  std::vector<GingerNodeId> reordering(graph.NumNodes(), 1);

  // Main loop: In each iteration, we compute a matching of edges and update the
  // graph representation after contraction until no more edges can be matched.
  // We store the matchings so that we can write the reordering in a second pass
  // after we have encountered the base case.
  std::vector<parlay::sequence<rgc::MatchingEdge>> matchings;
  while (!edges.empty()) {
    matchings.push_back(rgc::ComputeMatching(nodes, edges, priorities_buffer,
                                             parent_buffer,
                                             absl::MakeSpan(reordering)));
    rgc::UpdateGraph(nodes, edges, parent_buffer);

    // We abort the recursion if either of the following conditions is met:
    // 1) The number of nodes is smaller than the base case threshold.
    // 2) The recursion depth exceeds the maximum recursion depth.
    if (nodes.size() <= config.base_case_threshold() ||
        matchings.size() >= config.max_recursion_depth()) {
      break;
    }
  }

  // Base case: When no edges remain, all current `nodes` are isolated
  // (super-)nodes. We write each node's offset into the `sizes` buffer.
  rgc::HandleBaseCase(nodes, absl::MakeSpan(reordering));

  // We now compute the final reordering by unwinding the recursion. We iterate
  // through the `matchings` from the last level of recursion back to the first.
  // In each step, we use the stored matching information to compute the offsets
  // for the contracted nodes based on the sizes of their representatives,
  // effectively projecting the reordering up through each level of contraction.
  while (!matchings.empty()) {
    rgc::WriteOffsets(matchings.back(), absl::MakeSpan(reordering));
    matchings.pop_back();
  }

  return reordering;
}

template std::vector<GingerNodeId>
ComputeRecursiveGraphContractionReordering<CSRGraph>(
    const CSRGraph& graph, RecursiveGraphContractionReorderingConfig config);
template std::vector<GingerNodeId>
ComputeRecursiveGraphContractionReordering<FlatGraph>(
    const FlatGraph& graph, RecursiveGraphContractionReorderingConfig config);

}  // namespace internal
}  // namespace ginger
