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

#ifndef GINGER_VAMANA_INDEX_BUILDER_H_
#define GINGER_VAMANA_INDEX_BUILDER_H_

#include <cstddef>
#include <functional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "ginger/build_progress.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/types.h"
#include "ginger/vamana/config.pb.h"

namespace ginger {

using VamanaIndex = FlatGraph;

constexpr GingerNodeId kVamanaEntryPoint = GingerNodeId(0);

namespace internal {

// Inserts a batch of nodes into the VamanaIndex graph. For each node in the
// batch, it performs a beam search to find potential neighbors, prunes these
// candidates using RobustPrune, and then updates the graph with the pruned
// neighbors. The `compute_distances` function is used to calculate distances
// between nodes.
//
// To ensure a deterministic update when performing insertions in parallel, the
// process is split into three steps:
// 1) For each node in the `batch`, perform a beam search and prune the results
//    in parallel. All searches operate on a snapshot of the graph before any
//    updates from this batch are applied. The results are stored but not yet
//    committed to the graph.
// 2) After all beam searches are complete, the forward edges for all nodes in
//    the batch are updated in the graph.
// 3) The reverse edges are grouped by the target node, merged with existing
//    neighbors, and used to update the graph.
void InsertBatch(absl::Span<const GingerNodeId> batch, FlatGraph& graph,
                 const VamanaConfig& config, size_t pass_index,
                 std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
                     compute_distances);

}  // namespace internal

// Builds an approximate nearest neighbor index using the Vamana build
// procedure.
//
// This function constructs a `VamanaIndex` (currently a `FlatGraph`) by
// inserting `num_points` nodes in batches. The insertion process involves:
// 1.  Randomly permuting the node IDs (except the first one).
// 2.  Generating batches of points to insert with exponentially increasing
//     sizes, up to a fixed maximum batch size (currently 1M points).
// 3.  For each batch, performing a beam search to find potential neighbors
//     for each node in the batch, pruning these candidates using
//     `RobustPrune'.  It then reverses the edges that survive the prune to try
//      and add new out-edges for the receiving endpoint (these will be nodes in
//      earlier batches). If such a node recieves more than max_degree
//      neighbors, we calculate its new neighborhood using RobustPrune.
//
// The index construction is deterministic given a fixed `config.seed()`.
//
// Args:
//   `num_points`: The total number of points that we build on in the index.
//   `config`: The `VamanaConfig` proto specifying the build parameters.
//   `compute_distances`: A function that takes a `GingerNodeId` (the query
//     point) and a span of `DistanceAndId`s (the candidates) and, for each
//     candidate, sets its `distance` field in place to the distance between
//     the query point and the candidate's `id`.
//
// Returns:
//   An `absl::StatusOr<FlatGraph>` containing the built index on success,
//   or an error status if the provided `config` is invalid.
//
// TODO: add deterministic tests for BuildVamanaIndex.
absl::StatusOr<FlatGraph> BuildVamanaIndex(
    GingerNodeId num_points, VamanaConfig config,
    std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
        compute_distances,
    std::function<void(const IndexBuildProgress&)> progress_callback = nullptr);

// Same as above, but polls `is_cancelled` between batches and returns a
// `CancelledError` once it returns true.
absl::StatusOr<FlatGraph> BuildVamanaIndex(
    GingerNodeId num_points, VamanaConfig config,
    std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
        compute_distances,
    std::function<void(const IndexBuildProgress&)> progress_callback,
    const std::function<bool()>& is_cancelled);
}  // namespace ginger

#endif  // GINGER_VAMANA_INDEX_BUILDER_H_
