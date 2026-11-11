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

#include "ginger/graph/graph_stats.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

namespace ginger {
namespace internal {

std::pair<GingerNodeId, GingerNodeId> NumReachableNodes(GingerNodeId start,
                                                        const FlatGraph& G) {
  if (G.NumNodes() == 0 || start >= G.NumNodes()) {
    return std::make_pair(0, 0);
  }
  auto visited = parlay::tabulate<std::atomic<bool>>(
      G.NumNodes(), [&](int64_t i) { return (i == start) ? true : false; });

  parlay::sequence<GingerNodeId> frontier(1, start);
  GingerNodeId num_hops = 0;
  GingerNodeId num_nodes_visited = 0;
  std::string hop_hist;
  while (!frontier.empty()) {
    num_nodes_visited += frontier.size();
    absl::StrAppend(&hop_hist, "h", num_hops, ":", frontier.size(), " ");
    // Claim newly visited neighbors before flattening to avoid allocating
    // O(total_edges) temporary buffers on large graphs.
    auto out_seqs = parlay::map(frontier, [&](GingerNodeId u) {
      parlay::sequence<GingerNodeId> next;
      for (GingerNodeId v : G.Neighbors(u)) {
        bool expected = false;
        if (!visited[v].load(std::memory_order_relaxed) &&
            visited[v].compare_exchange_strong(expected, true,
                                               std::memory_order_relaxed)) {
          next.push_back(v);
        }
      }
      return next;
    });
    frontier = parlay::flatten(std::move(out_seqs));
    ++num_hops;
  }
  ABSL_LOG(INFO) << "BFS hop distribution from node " << start << ": "
                 << hop_hist;
  return std::make_pair(num_nodes_visited,
                        num_hops > 0 ? num_hops - 1 : num_hops);
}

}  // namespace internal

GraphStats ComputeGraphStats(const FlatGraph& graph, GingerNodeId start_node) {
  GraphStats stats;
  stats.num_nodes = graph.NumNodes();
  if (stats.num_nodes == 0) {
    return stats;
  }

  struct DegreeSummary {
    size_t total_edges = 0;
    size_t max_degree = 0;
    size_t zero_degree_nodes = 0;
  };
  auto degree_monoid = parlay::make_monoid(
      [](const DegreeSummary& a, const DegreeSummary& b) {
        DegreeSummary res;
        res.total_edges = a.total_edges + b.total_edges;
        res.max_degree = std::max(a.max_degree, b.max_degree);
        res.zero_degree_nodes = a.zero_degree_nodes + b.zero_degree_nodes;
        return res;
      },
      DegreeSummary{});

  DegreeSummary deg_summary = parlay::reduce(
      parlay::delayed_tabulate(graph.NumNodes(),
                               [&](size_t i) -> DegreeSummary {
                                 const size_t d =
                                     graph.Degree(static_cast<GingerNodeId>(i));
                                 DegreeSummary s;
                                 s.total_edges = d;
                                 s.max_degree = d;
                                 if (d == 0) s.zero_degree_nodes = 1;
                                 return s;
                               }),
      degree_monoid);

  stats.total_edges = deg_summary.total_edges;
  stats.max_degree = static_cast<GingerNodeId>(deg_summary.max_degree);
  stats.zero_degree_nodes = deg_summary.zero_degree_nodes;
  stats.avg_degree = static_cast<double>(stats.total_edges) / stats.num_nodes;

  auto in_degree = parlay::sequence<uint32_t>(graph.NumNodes(), 0);
  parlay::parallel_for(0, graph.NumNodes(), [&](size_t u) {
    for (GingerNodeId v : graph.Neighbors(static_cast<GingerNodeId>(u))) {
      std::atomic_ref<uint32_t>(in_degree[v])
          .fetch_add(1, std::memory_order_relaxed);
    }
  });
  stats.zero_in_degree_nodes =
      parlay::count_if(in_degree, [](uint32_t d) { return d == 0; });

  const auto [reachable_nodes, num_hops] =
      internal::NumReachableNodes(start_node, graph);
  stats.reachable_nodes_from_start = reachable_nodes;
  stats.max_hops_from_start = num_hops;

  return stats;
}

void PrintGraphStats(const FlatGraph& graph, absl::string_view graph_name,
                     GingerNodeId start_node) {
  const GraphStats stats = ComputeGraphStats(graph, start_node);
  const std::string prefix =
      graph_name.empty() ? "Final graph stats"
                         : absl::StrCat(graph_name, " final graph stats");
  ABSL_LOG(INFO) << prefix << ": num_nodes = " << stats.num_nodes
                 << ", total_edges = " << stats.total_edges
                 << ", avg_degree = " << stats.avg_degree
                 << ", max_degree = " << stats.max_degree
                 << ", 0-degree nodes = " << stats.zero_degree_nodes
                 << ", 0-in-degree nodes = " << stats.zero_in_degree_nodes;
  ABSL_LOG(INFO) << "Number of nodes reachable from node " << start_node << ": "
                 << stats.reachable_nodes_from_start;
  ABSL_LOG(INFO) << "Maximum hop distance from node " << start_node
                 << " to any reachable node: " << stats.max_hops_from_start;
}

}  // namespace ginger
