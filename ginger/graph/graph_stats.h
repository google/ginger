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

#ifndef GINGER_GRAPH_GRAPH_STATS_H_
#define GINGER_GRAPH_GRAPH_STATS_H_

#include <cstddef>
#include <utility>

#include "absl/strings/string_view.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/types.h"

namespace ginger {

namespace internal {

// Returns the number of reachable nodes from the start vertex and the maximum
// hop distance from the source to any node.
std::pair<GingerNodeId, GingerNodeId> NumReachableNodes(GingerNodeId start,
                                                        const FlatGraph& G);

}  // namespace internal

struct GraphStats {
  GingerNodeId num_nodes = 0;
  size_t total_edges = 0;
  double avg_degree = 0.0;
  GingerNodeId max_degree = 0;
  size_t zero_degree_nodes = 0;
  size_t zero_in_degree_nodes = 0;
  GingerNodeId reachable_nodes_from_start = 0;
  GingerNodeId max_hops_from_start = 0;
};

// Computes graph connectivity and degree statistics from start_node.
GraphStats ComputeGraphStats(const FlatGraph& graph,
                             GingerNodeId start_node = 0);

// Computes and logs formatted graph statistics via ABSL_LOG(INFO).
void PrintGraphStats(const FlatGraph& graph, absl::string_view graph_name = "",
                     GingerNodeId start_node = 0);

}  // namespace ginger

#endif  // GINGER_GRAPH_GRAPH_STATS_H_
