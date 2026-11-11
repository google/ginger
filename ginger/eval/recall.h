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

#ifndef GINGER_EVAL_RECALL_H_
#define GINGER_EVAL_RECALL_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {

constexpr float kRecallDistanceEpsilon = 1e-5;

// Stats about k-th neighbor distance used for recall computation.
struct KthNeighborStats {
  // Distance to k-th neighbor.
  float distance_to_kth_neighbor = 0.0;
  // Number of neighbors with distance < distance_to_kth_neighbor -
  // kRecallDistanceEpsilon.
  int num_closer_neighbors = 0;
};

// Estimates the recall of an approximate k-NN graph.
// The estimator samples a subset of nodes, and for each sampled node, computes
// the true k nearest neighbors. The reported estimated recall is the exact
// recall of the sampled nodes.
// The exact process of computing recall of a given node is as follows:
// 1. Compute the distance `d` to the k-th nearest neighbor of the given node
//    (ignoring a self-loop), and the true number of neighbors (`c_true`) with
//    distance `< d - kRecallDistanceEpsilon`.
// 2. Given a set of neighbors found, recompute the distances to these
// neighbors.
// 3. Let `h_c` be the number of found neighbors with distance `< d -
//    kRecallDistanceEpsilon`, and `h_t` be the number of found neighbors with
//    distance in `[d - kRecallDistanceEpsilon, d + kRecallDistanceEpsilon]`.
//    The recall of the given node is
//    `(h_c + min(h_t, k - c_true)) / k`.
// 4. The recall of the sampled nodes is the average of the recall of each
//    sampled node.
class GraphRecallEstimator {
 public:
  // Creates a recall estimator for a a k-NN graph with `num_nodes` nodes and k
  // = `num_neighbors`. `num_recall_samples` specifies the number of nodes used
  // to estimate the recall. It must be smaller than `num_nodes`.
  // `compute_distances` is used to compute the distances between a source node
  // and several query nodes. It must be thread-safe.
  static absl::StatusOr<std::unique_ptr<GraphRecallEstimator>> Create(
      GingerNodeId num_nodes, GingerNodeId num_neighbors,
      GingerNodeId num_recall_samples,
      std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
          compute_distances);

  // Reports the neighbors found for a given node.
  // This function should be called exactly once for each node in the graph.
  // `neighbors` must contain distinct node ids between 0 and `num_nodes_ - 1`,
  // otherwise an error is returned.
  // The distances in `neighbors` are recomputed by the provided distance
  // computation function, and thus can be arbitrary.
  absl::Status ReportNeighborsFound(GingerNodeId node_id,
                                    absl::Span<DistanceAndId> neighbors);

  // Returns the recall estimate. The estimate is computed as the average
  // recall over the sampled nodes, where the recall for each node is
  // calculated as described in the class comment.
  double RecallEstimate() {
    if (num_nodes_ == 0) return 0.0;
    absl::MutexLock lock(mutex_);
    if (node_stats_.empty()) return 0.0;
    return total_recall_ / node_stats_.size();
  }

 private:
  GraphRecallEstimator(
      GingerNodeId num_nodes, GingerNodeId num_neighbors,
      std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
          compute_distances,
      absl::flat_hash_map<GingerNodeId, KthNeighborStats>&& node_stats)
      : num_nodes_(num_nodes),
        num_neighbors_(num_neighbors),
        compute_distances_(std::move(compute_distances)),
        node_stats_(std::move(node_stats)) {}

  bool InRecallSet(GingerNodeId node_id) const {
    return node_stats_.contains(node_id);
  }

  const GingerNodeId num_nodes_;

  const GingerNodeId num_neighbors_;

  const std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
      compute_distances_;

  // Kth neighbor stats for each node in the recall set.
  absl::flat_hash_map<GingerNodeId, KthNeighborStats> node_stats_;

  absl::Mutex mutex_;
  double total_recall_ = 0.0;  // Guarded by mutex_.
};

}  // namespace ginger

#endif  // GINGER_EVAL_RECALL_H_
