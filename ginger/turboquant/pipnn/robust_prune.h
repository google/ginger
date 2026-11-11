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

#ifndef GINGER_TURBOQUANT_PIPNN_ROBUST_PRUNE_H_
#define GINGER_TURBOQUANT_PIPNN_ROBUST_PRUNE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// Reusable scratch buffers for robust pruning.
//
// Create once per worker thread and reuse across calls. All buffers grow
// as needed and retain their capacity between calls to avoid reallocation.
struct RobustPruneWorkspace {
  // Decoded 8-bit rows for the current batch of candidates (signed int8).
  hwy::AlignedVector<int8_t> batch_decoded_rows;

  // Decoded 8-bit rows for promoted survivors (unsigned uint8, +128 bias).
  // Kept hot in L1/L2 so distance computations against new candidates avoid
  // re-reading the original 4-bit data from DRAM.
  hwy::AlignedVector<uint8_t> survivor_rows;

  // Norm scaling factors for each survivor.
  hwy::AlignedVector<float> survivor_norms;

  // Unquantized squared norms for each survivor (SquaredL2 only).
  hwy::AlignedVector<float> survivor_squared_norms;

  // Packed 64-bit sort keys (distance | node_id) for VQSort.
  std::vector<uint64_t> sort_keys;

  // Sorted, deduplicated candidate node IDs and distances (excluding
  // `source_node`), cleared on entry to `AcceleratedRobustPrune` and resized to
  // the number of unique valid candidates upon return.
  std::vector<GingerNodeId> sorted_candidate_ids;
  std::vector<float> sorted_candidate_distances;

  // Pruned candidate node IDs in increasing distance order (used when
  // keep_pruned_neighbors is true).
  std::vector<GingerNodeId> pruned_candidates;
};

// Performs robust pruning on a TurboQuant dataset.
//
// Clears `out_survivors` on entry and populates it with at most `max_survivors`
// surviving neighbor indices for `source_node`.
//
// Sorts candidates by ascending distance, then applies greedy robust pruning.
// Uses register-tiled SIMD to compute distances directly from 4-bit TurboQuant
// data without materializing full distance matrices.
//
// Preconditions:
//   - candidates and distances_to_source must have identical sizes.
//   - workspace must be non-null.
//
// Behavior & Invariants:
//   - out_survivors is cleared on entry and contains at most max_survivors
//     elements upon return.
//   - Candidates are sorted internally via VQSort; callers need not pre-sort.
//   - source_node is the graph vertex whose outgoing neighborhood is being
//     pruned. If source_node appears in candidates (e.g. from an ANN search or
//     graph traversal), it is explicitly skipped to prevent self-loops.
//   - Duplicate candidate IDs are supported; subsequent duplicates are pruned
//     against the earlier instance.
//   - Supported metrics: SquaredL2Distance (distances >= 0) and
//     DotProductDistance (TurboQuant distances <= 0, where smaller values
//     represent closer neighbors).
//   - A candidate c is pruned by survivor s if dist(c, s) < threshold,
//     where threshold = dist(c, source_node) / alpha for L2, and
//     dist(c, source_node) * alpha for dot product.
//     For unit-norm vectors, we convert dot-product distance to L2 and use the
//     L2 threshold.
//   - If prune_config.keep_pruned_neighbors() is true, fills out_survivors
//     with pruned candidates in increasing distance order until max_survivors
//     is reached.
//
// Parameters:
//   quantized_points    - the TurboQuant dataset (4-bit).
//   candidates          - candidate neighbor indices (need not be sorted).
//   distances_to_source - distances_to_source[i] = distance(candidates[i],
//                         source_node).
//   source_node         - the graph vertex whose neighborhood is being pruned
//                         (excluded from survivors to prevent self-loops).
//   prune_config        - robust pruning configuration.
//   max_survivors       - capacity limit on returned neighbors.
//   workspace           - reusable scratch buffers (non-null, thread-local
//                         recommended).
//   out_survivors       - [out] cleared on entry and populated with at most
//                         max_survivors surviving neighbor indices.
void AcceleratedRobustPrune(const TurboQuantDatasetSoA& quantized_points,
                            absl::Span<const GingerNodeId> candidates,
                            absl::Span<const float> distances_to_source,
                            GingerNodeId source_node,
                            const RobustPruneConfig& prune_config,
                            size_t max_survivors,
                            RobustPruneWorkspace* workspace,
                            std::vector<GingerNodeId>& out_survivors);

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_PIPNN_ROBUST_PRUNE_H_
