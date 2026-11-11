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

#include "ginger/eval/recall.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "ginger/types.h"
#include "parlay/parallel.h"

namespace ginger {

absl::StatusOr<std::unique_ptr<GraphRecallEstimator>>
GraphRecallEstimator::Create(
    GingerNodeId num_nodes, GingerNodeId num_neighbors,
    GingerNodeId num_recall_samples,
    std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
        compute_distances) {
  if (num_nodes <= 0) {
    return absl::InvalidArgumentError("num_nodes must be positive");
  }
  if (num_neighbors <= 0) {
    return absl::InvalidArgumentError("num_neighbors must be positive");
  }
  if (num_neighbors >= num_nodes) {
    return absl::InvalidArgumentError(
        "num_neighbors must be less than num_nodes");
  }
  if (num_recall_samples <= 0 || num_recall_samples > num_nodes) {
    return absl::InvalidArgumentError(
        "num_recall_samples must be between 1 and num_nodes");
  }

  absl::flat_hash_map<GingerNodeId, KthNeighborStats> node_stats;
  absl::BitGen gen;
  while (node_stats.size() < num_recall_samples) {
    GingerNodeId sample_id = absl::Uniform<GingerNodeId>(gen, 0, num_nodes);
    node_stats.try_emplace(sample_id);
  }

  std::vector<GingerNodeId> recall_sample_ids;
  recall_sample_ids.reserve(num_recall_samples);
  for (const auto& [id, _] : node_stats) {
    recall_sample_ids.push_back(id);
  }

  // We compute k-th neighbor distances in tiles to save memory.
  // For each of num_recall_samples nodes, we maintain num_neighbors smallest
  // distances found so far.
  std::vector<std::vector<DistanceAndId>> top_k_neighbors(num_recall_samples);
  for (int i = 0; i < num_recall_samples; ++i) {
    top_k_neighbors[i].reserve(num_neighbors);
  }

  const GingerNodeId block_size =
      std::max<int>(num_recall_samples, num_neighbors);
  for (GingerNodeId block_start = 0; block_start < num_nodes;
       block_start += block_size) {
    GingerNodeId block_end =
        std::min<GingerNodeId>(block_start + block_size, num_nodes);
    parlay::parallel_for(0, num_recall_samples, [&](int i) {
      const GingerNodeId node_id = recall_sample_ids[i];
      std::vector<DistanceAndId>& current_top_k = top_k_neighbors[i];
      const size_t initial_size = current_top_k.size();
      // Reserve space for the new block.
      current_top_k.reserve(initial_size + (block_end - block_start));

      for (GingerNodeId j = block_start; j < block_end; ++j) {
        if (j != node_id) {
          current_top_k.push_back({0.0, j});
        }
      }
      if (current_top_k.size() == initial_size) return;

      // Compute distances for the newly added elements.
      compute_distances(node_id,
                        absl::MakeSpan(current_top_k).subspan(initial_size));

      // If the vector size exceeds num_neighbors, use nth_element to keep only
      // the num_neighbors smallest elements.
      if (current_top_k.size() > num_neighbors) {
        std::nth_element(current_top_k.begin(),
                         current_top_k.begin() + num_neighbors - 1,
                         current_top_k.end());
        current_top_k.resize(num_neighbors);
      }
    });
  }

  for (int i = 0; i < num_recall_samples; ++i) {
    ABSL_CHECK_EQ(top_k_neighbors[i].size(), num_neighbors);
    float kth_dist =
        std::max_element(top_k_neighbors[i].begin(), top_k_neighbors[i].end())
            ->distance;
    int num_closer = 0;
    for (const auto& neighbor : top_k_neighbors[i]) {
      if (neighbor.distance < kth_dist - kRecallDistanceEpsilon) {
        ++num_closer;
      }
    }
    node_stats[recall_sample_ids[i]] = {kth_dist, num_closer};
  }

  // Can't use make_unique because the constructor is private.
  return std::unique_ptr<GraphRecallEstimator>(new GraphRecallEstimator(
      num_nodes, num_neighbors, compute_distances, std::move(node_stats)));
}

absl::Status GraphRecallEstimator::ReportNeighborsFound(
    GingerNodeId node_id, absl::Span<DistanceAndId> neighbors) {
  if (!InRecallSet(node_id)) {
    return absl::OkStatus();
  }

  for (const auto& [_, neighbor_id] : neighbors) {
    if (neighbor_id < 0 || neighbor_id >= num_nodes_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid neighbor id ", neighbor_id, " found for node ",
                       node_id, " with num_nodes_ = ", num_nodes_));
    }
  }

  compute_distances_(node_id, neighbors);

  absl::flat_hash_set<GingerNodeId> seen_ids;
  const KthNeighborStats& stats = node_stats_.at(node_id);
  float kth_distance = stats.distance_to_kth_neighbor;
  int hits_closer = 0;
  int hits_tie = 0;
  for (const auto& neighbor : neighbors) {
    if (!seen_ids.insert(neighbor.id).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Duplicate neighbor id ", neighbor.id, " found for node ", node_id));
    }
    if (neighbor.distance < kth_distance - kRecallDistanceEpsilon) {
      ++hits_closer;
    } else if (neighbor.distance <= kth_distance + kRecallDistanceEpsilon) {
      ++hits_tie;
    }
  }

  // The reported neighbors cannot be closer than the ground truth nearest
  // neighbors.
  ABSL_QCHECK_LE(hits_closer, stats.num_closer_neighbors);

  double recall_credits =
      hits_closer +
      std::min(hits_tie, num_neighbors_ - stats.num_closer_neighbors);
  if (recall_credits > num_neighbors_) recall_credits = num_neighbors_;

  absl::MutexLock lock(mutex_);
  total_recall_ += recall_credits / num_neighbors_;
  return absl::OkStatus();
}

}  // namespace ginger
