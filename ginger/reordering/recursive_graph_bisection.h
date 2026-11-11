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

#ifndef GINGER_REORDERING_RECURSIVE_GRAPH_BISECTION_H_
#define GINGER_REORDERING_RECURSIVE_GRAPH_BISECTION_H_

#include <vector>

#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"

namespace ginger {

// This algorithm computes a node ordering for a graph, which is optimized to
// reduce the size of the graph when compressed using gap-encoding. It
// reimplements the recursive graph bisection algorithm described in
// https://arxiv.org/abs/1602.08820.
//
// The algorithm works by recursively splitting the nodes of the graph into two
// bisections of equal size. The final node ordering is obtained by
// concatenating the orderings from the resulting bisections. At each recursive
// step, the graph is split by first computing an initial bisection. This
// initial bisection is then improved through a local search algorithm. The
// local search consists of multiple rounds. In each round, the algorithm
// computes the gain for moving each node to the other side of the bisection.
// These moves are sorted in decreasing order of gain, and pairwise swaps are
// performed based on this order, continuing as long as the sum of the gains is
// positive. The gains are derived from an objective function that corresponds
// to the size of the graph when compressed using delta-encoding. This function
// is given by `Sum_u Sum_{i=1}^{k-1} log(v_{i+1} - v_i)`, where k is the degree
// of u, and v_0 < v_1 < ... < v_k are the adjacent nodes of u sorted by
// increasing node ID.

// Computes a reordering of the input graph. Settings for the algorithm can be
// optionally provided with a `config` parameter, which is by default set with
// the default arguments. It returns a vector `reordering` of length n where
// `reordering[u]` equals to the new node ID of node u.
template <typename Graph>
std::vector<GingerNodeId> ComputeRecursiveGraphBisectionReordering(
    const Graph& graph, RecursiveGraphBisectionReorderingConfig config = {});

}  // namespace ginger

#endif  // GINGER_REORDERING_RECURSIVE_GRAPH_BISECTION_H_
