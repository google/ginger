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

#ifndef GINGER_PIPNN_PIPNN_KNN_H_
#define GINGER_PIPNN_PIPNN_KNN_H_

#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "ginger/dataset.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"

namespace ginger {

// Callback invoked once per node `u` in `[0, points.num_datapoints())` with up
// to `k` nearest neighbors sorted ascending by distance (with no self-loops or
// duplicates). Invoked concurrently from worker threads; must be thread-safe.
using AddNeighborsCallback =
    std::function<void(GingerNodeId, std::vector<DistanceAndId>)>;

// Builds a directed approximate k-NN graph using PiPNN recursive leaf
// partitioning and all-to-all comparisons within each leaf (via
// `ManyToManyTurboQuantTopK`).
//
// Each node retains up to `max_candidates = overretrieve_factor * k` candidate
// neighbors across `pipnn_config.num_passes()` partitioning passes when
// reranking is enabled (`k` when no reranking is used):
// - If `float_dataset != nullptr`, the retained candidates are reranked using
//   exact float distances (`SquaredL2Distance` or `DotProductDistance`) before
//   selecting the top `k`.
// - Otherwise, if `points.extra_bits_per_dim() > 0`, the retained 4-bit
//   candidates are reranked using TurboQuant 4+k-bit (e.g., 7-bit)
//   extra-precision distances.
// - Otherwise, the top `k` 4-bit quantized candidates and distances are emitted
//   directly.
//
// Only the partitioning fields of `pipnn_config` (`max_leaf_size`,
// `min_leaf_size`, `num_passes`, `leader_selection`, and `seed`) and
// `print_stats` are used; all other fields (`max_degree`, `robust_prune_alpha`,
// `hash_prune`, `pruning_strategy`, and `phased_build`) are ignored. Invalid
// partitioning fields are rejected, as in `BuildPipnnIndex`.
absl::Status BuildPipnnKnnGraph(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    const PipnnConfig& pipnn_config, int k, int overretrieve_factor,
    AddNeighborsCallback add_neighbors,
    const SpanDataset* float_dataset = nullptr,
    const std::function<bool()>& is_cancelled = []() { return false; });

}  // namespace ginger

#endif  // GINGER_PIPNN_PIPNN_KNN_H_
