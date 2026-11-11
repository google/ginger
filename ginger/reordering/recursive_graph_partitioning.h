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

#ifndef GINGER_REORDERING_RECURSIVE_GRAPH_PARTITIONING_H_
#define GINGER_REORDERING_RECURSIVE_GRAPH_PARTITIONING_H_

#include <vector>

#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"

namespace ginger {

// This reordering algorithm recursively partitions the graph using a graph
// partitioner, aiming to minimize the number of edges between the parts. The
// final node ordering is obtained by concatenating the orderings from the
// resulting partitions.

// Computes a reordering of the input graph. Settings for the algorithm can be
// optionally provided with a `config` parameter, which is by default set with
// the default arguments. It returns a vector `reordering` of length n where
// `reordering[u]` equals to the new node ID of node u.
template <typename Graph>
std::vector<GingerNodeId> ComputeRecursiveGraphPartitioningReordering(
    const Graph& graph, RecursiveGraphPartitioningReorderingConfig config = {});

}  // namespace ginger

#endif  // GINGER_REORDERING_RECURSIVE_GRAPH_PARTITIONING_H_
