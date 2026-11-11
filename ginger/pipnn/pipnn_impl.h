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

#ifndef GINGER_PIPNN_PIPNN_IMPL_H_
#define GINGER_PIPNN_PIPNN_IMPL_H_

#include <cstddef>
#include <functional>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"
#include "parlay/sequence.h"

namespace ginger::pipnn_internal {

// Callback invoked for each terminal leaf cluster produced during recursive
// partitioning. `leaf` contains the node IDs in the leaf. Must be thread-safe.
using LeafCallback =
    absl::FunctionRef<void(absl::Span<const GingerNodeId> leaf)>;

// Returns the chunk budget in leaf points: `fraction` of `total_leaf_points`,
// or unlimited if `fraction >= 1`.
size_t ChunkBudget(size_t total_leaf_points, float fraction);

// Validates the hierarchical partitioning parameters in `config`
// (`num_passes`, `max_leaf_size`, `min_leaf_size`, and `leader_selection`).
// TODO: Factor partitioning parameters into a submessage of
// PipnnConfig.
absl::Status ValidatePipnnPartitioningConfig(const PipnnConfig& config);

// Recursively partitions all points in `points` into leaf clusters of size at
// most `config.max_leaf_size()` using the PiPNN hierarchical leader-clustering
// phase (see https://arxiv.org/abs/2602.21247), invoking `on_leaf` in parallel
// for each chunk of terminal leaf clusters and `on_chunk_done` sequentially
// after each chunk completes.
//
// At each recursion level `depth`, if a cluster exceeds `max_leaf_size`:
//   1. A subset of leader points is sampled from the cluster
//      (`top_level_leaders` at `depth == 0`, or
//      `clamp(size * leader_sample_rate, 3, max_num_leaders)` at `depth > 0`).
//   2. Each point in the cluster is assigned to its `fanout_sequence[depth]`
//      (or 1 if `depth >= fanout_sequence.size()`) nearest leaders.
//   3. Sub-clusters smaller than `config.min_leaf_size()` are merged together
//      without exceeding `config.max_leaf_size()`.
//   4. Remaining oversized sub-clusters recurse to `depth + 1`. To bound
//      recursion depth, clusters that exceed a maximum depth are randomly
//      partitioned into leaves of size `config.max_leaf_size()`.
//
// `pass` (0-indexed pass number) is combined with `config.seed()` to seed PRNGs
// during leader sampling and random cluster splitting so that each pass
// produces a distinct partitioning (and is used in stats logging when
// `config.print_stats()` is enabled). The set of leaf clusters produced is
// deterministic given `config.seed()` and `pass`, though the order in which
// `on_leaf` is invoked within a chunk is non-deterministic due to thread
// scheduling.
absl::Status PartitionPointsIntoLeaves(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    const PipnnConfig& config, int pass, LeafCallback on_leaf,
    absl::FunctionRef<void()> on_chunk_done,
    const std::function<bool()>& is_cancelled);

// Overload that collects all terminal leaf clusters up the recursion tree and
// returns them as a flat sequence instead of invoking a callback on the fly.
// This is used for "phased builds", where we separate the recursive clustering
// from the subsequent leaf-processing phase.
absl::StatusOr<parlay::sequence<parlay::sequence<GingerNodeId>>>
PartitionPointsIntoLeaves(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    const PipnnConfig& config, int pass,
    const std::function<bool()>& is_cancelled = []() { return false; });

}  // namespace ginger::pipnn_internal

#endif  // GINGER_PIPNN_PIPNN_IMPL_H_
