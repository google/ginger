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

#ifndef GINGER_PIPNN_HASH_PRUNE_H_
#define GINGER_PIPNN_HASH_PRUNE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/pipnn/simhash.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// https://arxiv.org/abs/2602.21247
// HashPrune is an online, history-independent pruning algorithm that maintains
// a sparse navigation graph for Approximate Nearest Neighbor Search (ANNS)
// index construction. It uses locality-sensitive hashing (LSH) to divide space
// into angular cones around each vertex, retaining only the closest candidate
// neighbor in each direction. The resulting adjacency lists are deterministic
// and independent of the order in which candidate edges are inserted.
class HashPrune {
 public:
  HashPrune(size_t reservoir_size, SimHashProjections simhash_projections);

  // Thread-safe.
  void MergeNeighbors(GingerNodeId source,
                      absl::Span<const DistanceAndId> edges);

  // Populates the caller-provided vector with the neighbors of point_id. Not
  // thread-safe.
  void GetNeighbors(GingerNodeId point_id,
                    std::vector<DistanceAndId>& neighbors) const;

  // Bulk export: Returns the entire pruned graph as an adjacency list.
  std::vector<std::vector<DistanceAndId>> ExportAllNeighbors() const;

  struct Stats {
    double avg_degree = 0.0;
    size_t max_degree = 0;
    size_t total_edges = 0;
  };
  Stats GetStats() const;
  void LogStats() const;

  // Read-only accessors for node i's degree, distances, and point IDs.
  // Not thread-safe against concurrent MergeNeighbors on node i.
  // Note: Distances(i) and PointIds(i) return spans of length `stride_` (padded
  // for SIMD alignment). Only the first `GetDegree(i)` elements are valid.
  // Prefer ValidDistances(i) / ValidPointIds(i) which return spans of length
  // `GetDegree(i)`.
  size_t GetDegree(GingerNodeId i) const {
    return static_cast<size_t>(degrees_[i]);
  }
  absl::Span<const Eigen::bfloat16> Distances(GingerNodeId i) const {
    return absl::MakeSpan(distances_.get() + i * stride_, stride_);
  }
  absl::Span<const GingerNodeId> PointIds(GingerNodeId i) const {
    return absl::MakeSpan(point_ids_.get() + i * stride_, stride_);
  }
  absl::Span<const Eigen::bfloat16> ValidDistances(GingerNodeId i) const {
    return absl::MakeSpan(distances_.get() + i * stride_, GetDegree(i));
  }
  absl::Span<const GingerNodeId> ValidPointIds(GingerNodeId i) const {
    return absl::MakeSpan(point_ids_.get() + i * stride_, GetDegree(i));
  }

 private:
  absl::Span<uint16_t> Hashes(GingerNodeId i) {
    return absl::MakeSpan(hashes_.get() + i * stride_, stride_);
  }
  absl::Span<const uint16_t> Hashes(GingerNodeId i) const {
    return absl::MakeSpan(hashes_.get() + i * stride_, stride_);
  }

  absl::Span<Eigen::bfloat16> Distances(GingerNodeId i) {
    return absl::MakeSpan(distances_.get() + i * stride_, stride_);
  }

  absl::Span<GingerNodeId> PointIds(GingerNodeId i) {
    return absl::MakeSpan(point_ids_.get() + i * stride_, stride_);
  }

  static size_t FindHash(const uint16_t* hashes, size_t deg, uint16_t target);

  uint16_t Hash(GingerNodeId point_id, GingerNodeId source) const {
    return simhash_projections_.RelativeHash(source, point_id);
  }

  const size_t reservoir_size_;
  const size_t num_points_;
  const size_t stride_;  // Padded to 32 elements for SIMD alignment.

  // Table entries.
  hwy::AlignedFreeUniquePtr<uint16_t[]> hashes_;
  hwy::AlignedFreeUniquePtr<Eigen::bfloat16[]> distances_;
  hwy::AlignedFreeUniquePtr<GingerNodeId[]> point_ids_;

  // Per-point metadata (small, accessed once per MergeNeighbors call).
  std::vector<GingerNodeId> degrees_;
  std::vector<Eigen::bfloat16> worst_distances_;
  std::vector<uint16_t> worst_indices_;
  std::vector<absl::Mutex> table_mutexes_;

  SimHashProjections simhash_projections_;
};

}  // namespace ginger

#endif  // GINGER_PIPNN_HASH_PRUNE_H_
