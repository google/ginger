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

#ifndef GINGER_VAMANA_PRUNE_H_
#define GINGER_VAMANA_PRUNE_H_

#include <cstddef>
#include <functional>
#include <vector>

#include "absl/types/span.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/types.h"

namespace ginger {

// This function implements an extended version of the RobustPrune
// algorithm from the DiskANN paper (see the summary of this algorithm in
// https://arxiv.org/abs/2305.04359). The key inputs to the algorithm are a set
// of candidate edges, `candidates` and the parameters in `prune_config` and
// `max_degree`. The goal of the algorithm is to sub-select a diverse (as
// defined later) set of at most `max_degree` candidate edges that connect
// `node_id` to its near neighbors while ensuring good long-range connectivity
// for `node_id`.
//
// In particular, the algorithm 1) sorts the input points by distance from
// the source and 2) inductively builds up a set of points that survive.
// Given the survivors, the next point p' survives if for all p* in the
// survivor set, d(p*, p') > d(node_id, p) / alpha. If a point p' does not
// satisfy this criteria due to some survivor p*, the edge (p, p') can be viewed
// as redundant, as a search can traverse from (p, p*) and then navigate to p'
// via p*.
//
// One issue with blindly applying the RobustPrune procedure described above is
// that if alpha is set incorrectly, the set of edges that are output can be too
// small, leading to poor navigability of the overall graph. The usual fix for
// this is to tune alpha so that the average degree of the graph is at least
// some value.
//
// To perform this tuning without manual intervention, this function provides an
// "AlphaTuning" option, which automatically tunes alpha while trying to find a
// subset of edges of size at least `desired_degree`. Alpha tuning is enabled by
// setting the `desired_degree` parameter in `prune_config`. If alpha
// tuning is enabled, it happens as a second phase after regular pruning, and is
// only applied if the number of survivors from regular pruning is less than
// `desired_degree`. Alpha tuning greedily add candidates that were pruned,
// until `desired_degree` is reached or no more candidates can be added. In
// particular, the greedy approach we use selects candidates based on minimum
// alpha that would be required for them to not be pruned by any current
// survivor.
//
// Note that if the size of the candidate set after we filter out duplicates
// and self-loops is smaller than `desired_degree`, we do not perform
// pruning, and simply return this candidate set. Similarly, alpha tuning can
// return fewer than `desired_degree` neighbors in the case where some points
// have distance 0 to each other.
//
// Alternatively, if `prune_config.keep_pruned_neighbors` is true, pruned
// candidates are appended in increasing distance order after pruning until
// `max_degree` is reached.
//
// In the case of non-negative distances, the pruning algorithm also handles
// duplicate points. If there are too many candidates with distance 0
// (representing duplicate points), the algorithm directly prunes them to keep
// only a small, random subset. This ensures the cluster of duplicate points
// remains well connected, while preventing duplicate points from dominating all
// neighbors of the node.
//
// Input requirements:
// - `prune_config.alpha` is >= 1
// - The caller must ensure that the distances are filled in `candidates`. We
// make this assumption because the caller may often have this information
// already.
// - The distances are always non-positive (i.e., <= 0) if
// `prune_config.nonpositive_distances` is true. Otherwise, the distances have
// to be non-negative. Candidates violating this assumption are discarded and
// not considered for pruning.
// - It is acceptable to pass duplicate node ids in `candidates`, however
// they must have exactly equal distances. When multiple nodes have the same
// distance, tie-breaking is done lexicographically by ID.
//
// Outputs:
// - A set of candidates without duplicates or self-loops. Note that the
// output is not necessarily sorted by distance from `node_id`.
std::vector<GingerNodeId> RobustPrune(
    absl::Span<DistanceAndId> candidates, GingerNodeId node_id,
    size_t max_degree,
    std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
        compute_distances,
    const RobustPruneConfig& prune_config);

namespace internal {

void AlphaTuning(absl::Span<const DistanceAndId> candidates,
                 GingerNodeId node_id, const RobustPruneConfig& prune_config,
                 size_t max_degree,
                 std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
                     compute_distances,
                 std::vector<DistanceAndId>& surviving_neighbors,
                 std::vector<bool>& survived_pruning);

}  // namespace internal

}  // namespace ginger

#endif  // GINGER_VAMANA_PRUNE_H_
