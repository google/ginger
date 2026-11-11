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

#ifndef GINGER_REORDERING_RECURSIVE_GRAPH_CONTRACTION_H_
#define GINGER_REORDERING_RECURSIVE_GRAPH_CONTRACTION_H_

#include <vector>

#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"

namespace ginger {

// This algorithm computes a node ordering for a graph based on recursive graph
// contraction. The basic idea is to perform repeated rounds of parallel
// matching. In each round, a set of pairs of clusters are contracted into new,
// larger clusters. Each potential edge `(u, v)` between clusters is assigned a
// priority score calculated as `w(u, v) / (s(u) * s(v))`. An edge `(u, v)` is
// selected for matching and contraction if it has the highest priority among
// all edges incident to both endpoint clusters `u` and `v`. This algorithm is
// first described in https://dl.acm.org/doi/pdf/10.5555/644108.644219.

// Computes a reordering of the input graph. Settings for the algorithm can be
// optionally provided with a `config` parameter, which is by default set with
// the default arguments. It returns a vector `reordering` of length n where
// `reordering[u]` equals to the new node ID of node u.
template <typename Graph>
std::vector<GingerNodeId> ComputeRecursiveGraphContractionReordering(
    const Graph& graph, RecursiveGraphContractionReorderingConfig config = {});

}  // namespace ginger

#endif  // GINGER_REORDERING_RECURSIVE_GRAPH_CONTRACTION_H_
