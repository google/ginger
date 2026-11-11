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

#include "ginger/pipnn/pipnn_knn.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/distance/distance.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/pipnn/pipnn_impl.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/many_to_many_topk.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_reference_stub.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/sort/vqsort.h"
#include "parlay/parallel.h"

namespace ginger {
namespace {

using ::ginger::pipnn_internal::PartitionPointsIntoLeaves;
using ::ginger::pipnn_internal::ValidatePipnnPartitioningConfig;

absl::Status CheckCancelled(const std::function<bool()>& is_cancelled) {
  if (is_cancelled != nullptr && is_cancelled()) {
    return absl::CancelledError();
  }
  return absl::OkStatus();
}

// Co-locates per-node synchronization and reservoir metadata to touch a single
// cache line per node update and reduce cross-thread false sharing.
struct alignas(32) NodeReservoirState {
  absl::Mutex mutex;
  std::atomic<uint64_t> threshold_packed{std::numeric_limits<uint64_t>::max()};
  uint32_t size = 0;
};

// Packs `distance` and `id` into a 64-bit candidate key such that comparing
// keys as unsigned integers compares `(distance, id)` lexicographically: the
// upper 32 bits hold an order-preserving encoding of `distance` and the lower
// 32 bits hold `id`.
uint64_t MakeCandidateKey(float distance, GingerNodeId id) {
  uint32_t bits = absl::bit_cast<uint32_t>(distance);
  // Flip all bits of negative floats (whose magnitude order is reversed) and
  // only the sign bit of non-negative floats (to order them above negatives).
  bits = (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
  return (uint64_t{bits} << 32) | static_cast<uint32_t>(id);
}

// Returns the `distance` of a key created by `MakeCandidateKey`.
float CandidateKeyDistance(uint64_t key) {
  uint32_t bits = static_cast<uint32_t>(key >> 32);
  bits = (bits & 0x80000000U) ? (bits & ~0x80000000U) : ~bits;
  return absl::bit_cast<float>(bits);
}

// Returns the `id` of a key created by `MakeCandidateKey`.
GingerNodeId CandidateKeyId(uint64_t key) {
  return static_cast<GingerNodeId>(static_cast<uint32_t>(key));
}

// Sorts packed 64-bit `(ordered_score << 32) | neighbor_id` candidates in
// ascending order and removes duplicate neighbors. Because the 4-bit
// TurboQuant score for a pair `(u, v)` is bit-for-bit deterministic across
// all leaves and passes, duplicate occurrences of neighbor `v` in `u`'s
// buffer have the exact same 64-bit value, so sorting places duplicates
// adjacent to each other while leaving the unique prefix sorted by score.
size_t DeduplicateAndSortCandidates(uint64_t* ptr, size_t count) {
  if (count <= 1) return count;
  hwy::VQSort(ptr, count, hwy::SortAscending());
  return std::unique(ptr, ptr + count) - ptr;
}

void CompactNodeCandidates(uint64_t* ptr, uint32_t& size,
                           std::atomic<uint64_t>& threshold_atomic,
                           size_t max_candidates) {
  const size_t unique_count = DeduplicateAndSortCandidates(ptr, size);
  if (unique_count >= max_candidates) {
    size = static_cast<uint32_t>(max_candidates);
    threshold_atomic.store(ptr[max_candidates - 1], std::memory_order_relaxed);
  } else {
    size = static_cast<uint32_t>(unique_count);
    threshold_atomic.store(std::numeric_limits<uint64_t>::max(),
                           std::memory_order_relaxed);
  }
}

// Compares all pairs of points in `leaf` using 4-bit TurboQuant distances and
// merges the nearest neighbors of each point within the leaf into its candidate
// buffer. The buffer of node `u` holds `node_states[u].size` keys (see
// `MakeCandidateKey`) starting at `candidates + u * capacity_per_node`, and is
// compacted to its `max_candidates` smallest unique keys whenever it fills up.
// Safe to call concurrently: the buffer of node `u` is only accessed while
// holding `node_states[u].mutex`.
void ProcessLeaf(const TurboQuantDatasetSoA& points,
                 absl::Span<const GingerNodeId> leaf, size_t max_candidates,
                 size_t capacity_per_node, uint64_t* candidates,
                 absl::Span<NodeReservoirState> node_states) {
  const size_t leaf_size = leaf.size();
  if (leaf_size <= 1) return;

  const int leaf_k =
      static_cast<int>(std::min<size_t>(leaf_size, max_candidates + 1));
  thread_local ManyToManyTopKState state;
  thread_local std::vector<uint64_t> leaf_thresholds;
  thread_local std::vector<float> initial_thresholds;
  thread_local std::vector<uint64_t> query_candidates;
  if (leaf_thresholds.size() < leaf_size) {
    leaf_thresholds.resize(leaf_size);
    initial_thresholds.resize(leaf_size);
  }
  if (query_candidates.size() < static_cast<size_t>(leaf_k)) {
    query_candidates.resize(leaf_k);
  }
  for (size_t q_idx = 0; q_idx < leaf_size; ++q_idx) {
    const uint64_t cur_thresh = node_states[leaf[q_idx]].threshold_packed.load(
        std::memory_order_relaxed);
    leaf_thresholds[q_idx] = cur_thresh;
    // `ManyToManyTurboQuantTopK` only collects scores strictly below the
    // initial threshold. Using
    // `nextafter(CandidateKeyDistance(cur_thresh), +inf)` keeps all scores
    // `s <= CandidateKeyDistance(cur_thresh)` so equal-distance ties with
    // smaller global `node_id`s are still considered.
    initial_thresholds[q_idx] =
        cur_thresh == std::numeric_limits<uint64_t>::max()
            ? std::numeric_limits<float>::max()
            : std::nextafter(CandidateKeyDistance(cur_thresh),
                             std::numeric_limits<float>::infinity());
  }

  const absl::Span<const TopKResult> results = ManyToManyTurboQuantTopK(
      points, points, IndexSubset::Gathered(leaf), IndexSubset::Gathered(leaf),
      leaf_k, state, absl::MakeConstSpan(initial_thresholds).first(leaf_size));

  for (size_t q_idx = 0; q_idx < leaf_size; ++q_idx) {
    const GingerNodeId u = leaf[q_idx];
    const uint64_t cur_thresh = leaf_thresholds[q_idx];

    size_t valid_count = 0;
    for (const TopKResult& result : results.subspan(q_idx * leaf_k, leaf_k)) {
      // `result.db_idx` is a leaf-local index. Results are sorted, and padded
      // at the end with an out-of-range `db_idx` if fewer than `leaf_k` scores
      // are below the initial threshold.
      if (result.db_idx >= leaf_size) break;
      if (result.db_idx == q_idx) continue;
      const GingerNodeId v = leaf[result.db_idx];
      if (v == u) continue;
      const uint64_t key = MakeCandidateKey(result.distance, v);
      if (key >= cur_thresh) continue;
      query_candidates[valid_count++] = key;
    }
    if (valid_count == 0) continue;

    NodeReservoirState& u_state = node_states[u];
    uint64_t* node_buf =
        candidates + static_cast<size_t>(u) * capacity_per_node;
    absl::MutexLock lock(u_state.mutex);
    uint64_t thresh = u_state.threshold_packed.load(std::memory_order_relaxed);
    for (size_t r = 0; r < valid_count; ++r) {
      if (query_candidates[r] >= thresh) continue;
      node_buf[u_state.size++] = query_candidates[r];
      if (u_state.size == capacity_per_node) {
        CompactNodeCandidates(node_buf, u_state.size, u_state.threshold_packed,
                              max_candidates);
        thresh = u_state.threshold_packed.load(std::memory_order_relaxed);
      }
    }
  }
}

// Computes TurboQuant 4+k-bit extra-precision distances from dataset point `u`
// to `neighbors` in place, reusing a thread_local query buffer to avoid
// per-call memory allocation.
absl::Status ComputeExtraPrecisionDistances(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    GingerNodeId u, absl::Span<DistanceAndId> neighbors) {
  thread_local std::optional<TurboQuantQuery> query;
  // `num_bytes_per_query()` is padded, so it can match across dimensionalities.
  if (!query.has_value() ||
      query->num_bytes_per_query() != quantizer.num_bytes_per_query() ||
      query->dimensionality() != quantizer.dimensionality()) {
    ABSL_ASSIGN_OR_RETURN(
        query, TurboQuantQuery::CreateFromQuantizedDataPointWithExtraPrecision(
                   quantizer, points.DataPointView(u)));
  } else {
    ABSL_RETURN_IF_ERROR(query->ResetFromQuantizedDataPointWithExtraPrecision(
        quantizer, points.DataPointView(u)));
  }
  points.UnsafeComputeDistancesExtraPrecision(*query,
                                              CreateScoringAccessor(neighbors));
  // Clamp small negative values that may arise due to quantization error under
  // Squared L2 distance.
  if (quantizer.metric() == TurboQuantDistanceMetric::kSquaredL2Distance) {
    for (DistanceAndId& neighbor : neighbors) {
      neighbor.distance = std::max(0.0f, neighbor.distance);
    }
  }
  return absl::OkStatus();
}

// Deduplicates `candidate_keys` (the candidate buffer of node `u`) in place and
// returns up to `max_candidates` unique neighbors sorted ascending by 4-bit
// TurboQuant distance. If reranking is enabled, the neighbors are instead
// reranked and sorted by exact float distance (if `float_dataset != nullptr`)
// or else by TurboQuant extra-precision distance (if
// `points.extra_bits_per_dim() > 0`), and only the best `k` are returned.
std::vector<DistanceAndId> DeduplicateAndRerank(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    const SpanDataset* float_dataset, int k, size_t max_candidates,
    GingerNodeId u, absl::Span<uint64_t> candidate_keys) {
  const bool use_float_rerank = (float_dataset != nullptr);
  const bool use_extra_precision_rerank =
      (!use_float_rerank && points.extra_bits_per_dim() > 0);
  const bool should_rerank = use_float_rerank || use_extra_precision_rerank;
  const size_t unique_count = DeduplicateAndSortCandidates(
      candidate_keys.data(), candidate_keys.size());
  const size_t m = std::min<size_t>(unique_count, max_candidates);

  std::vector<DistanceAndId> neighbors(m);
  for (size_t j = 0; j < m; ++j) {
    const uint64_t item = candidate_keys[j];
    const GingerNodeId neighbor_id = CandidateKeyId(item);
    const float dist = should_rerank ? 0.0f : CandidateKeyDistance(item);
    neighbors[j] = DistanceAndId{
        .distance = dist,
        .id = neighbor_id,
    };
  }

  if (use_float_rerank && m > 0) {
    if (quantizer.metric() == TurboQuantDistanceMetric::kSquaredL2Distance) {
      SquaredL2Distance(u, absl::MakeSpan(neighbors), *float_dataset);
    } else {
      DotProductDistance(u, absl::MakeSpan(neighbors), *float_dataset);
    }
  } else if (use_extra_precision_rerank && m > 0) {
    ABSL_CHECK_OK(ComputeExtraPrecisionDistances(points, quantizer, u,
                                                 absl::MakeSpan(neighbors)));
  }

  if (should_rerank) {
    if (neighbors.size() > static_cast<size_t>(k)) {
      absl::c_nth_element(neighbors, neighbors.begin() + k);
      neighbors.resize(k);
    }
    absl::c_sort(neighbors);
  }
  return neighbors;
}

absl::Status ValidatePipnnKnnArgs(const TurboQuantDatasetSoA& points,
                                  const PipnnConfig& pipnn_config, int k,
                                  int overretrieve_factor,
                                  const SpanDataset* float_dataset) {
  if (k <= 0) {
    return absl::InvalidArgumentError("k must be positive.");
  }
  if (overretrieve_factor < 1) {
    return absl::InvalidArgumentError(
        "overretrieve_factor must be at least 1.");
  }
  ABSL_RETURN_IF_ERROR(ValidatePipnnPartitioningConfig(pipnn_config));
  if (float_dataset != nullptr) {
    if (static_cast<size_t>(float_dataset->size()) != points.num_datapoints()) {
      return absl::InvalidArgumentError(
          "float_dataset size must match points.num_datapoints().");
    }
    if (float_dataset->coordinate_type() != CoordinateType::kFloat) {
      return absl::InvalidArgumentError(
          "float_dataset must have coordinate_type == kFloat.");
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status BuildPipnnKnnGraph(const TurboQuantDatasetSoA& points,
                                const TurboQuantCodec& quantizer,
                                const PipnnConfig& pipnn_config, int k,
                                int overretrieve_factor,
                                AddNeighborsCallback add_neighbors,
                                const SpanDataset* float_dataset,
                                const std::function<bool()>& is_cancelled) {
  ginger::internal::ParallelSchedulerReference scheduler;
  ABSL_RETURN_IF_ERROR(ValidatePipnnKnnArgs(
      points, pipnn_config, k, overretrieve_factor, float_dataset));
  const std::function<bool()> cancel_fn =
      is_cancelled != nullptr ? is_cancelled : []() { return false; };
  ABSL_RETURN_IF_ERROR(CheckCancelled(cancel_fn));
  const size_t num_nodes = points.num_datapoints();
  if (num_nodes == 0) {
    return absl::OkStatus();
  }

  const absl::Time total_start = absl::Now();
  if (pipnn_config.print_stats()) {
    ABSL_LOG(INFO) << "Building PiPNN k-NN graph for " << num_nodes
                   << " points with k=" << k
                   << ", overretrieve_factor=" << overretrieve_factor
                   << ", num_passes=" << pipnn_config.num_passes()
                   << ", max_leaf_size=" << pipnn_config.max_leaf_size() << ".";
  }

  const bool use_float_rerank = (float_dataset != nullptr);
  const bool use_extra_precision_rerank =
      (!use_float_rerank && points.extra_bits_per_dim() > 0);
  const bool should_rerank = use_float_rerank || use_extra_precision_rerank;

  const size_t max_candidates =
      should_rerank ? static_cast<size_t>(overretrieve_factor) * k
                    : static_cast<size_t>(k);
  const size_t capacity_per_node = 2 * max_candidates;

  hwy::AlignedFreeUniquePtr<uint64_t[]> candidates =
      hwy::AllocateAligned<uint64_t>(num_nodes * capacity_per_node);
  std::vector<NodeReservoirState> node_states(num_nodes);

  for (int pass = 0; pass < pipnn_config.num_passes(); ++pass) {
    const absl::Time pass_start = absl::Now();
    ABSL_RETURN_IF_ERROR(PartitionPointsIntoLeaves(
        points, quantizer, pipnn_config, pass,
        [&](absl::Span<const GingerNodeId> leaf) {
          ProcessLeaf(points, leaf, max_candidates, capacity_per_node,
                      candidates.get(), absl::MakeSpan(node_states));
        },
        /*on_chunk_done=*/[]() {}, cancel_fn));
    if (pipnn_config.print_stats()) {
      ABSL_LOG(INFO) << "PiPNN k-NN pass " << pass << " finished in "
                     << absl::Now() - pass_start;
    }
  }

  ABSL_RETURN_IF_ERROR(CheckCancelled(cancel_fn));

  const absl::Time rerank_start = absl::Now();

  parlay::parallel_for(0, num_nodes, [&](size_t u) {
    const absl::Span<uint64_t> candidate_keys = absl::MakeSpan(
        candidates.get() + u * capacity_per_node, node_states[u].size);
    std::vector<DistanceAndId> neighbors = DeduplicateAndRerank(
        points, quantizer, float_dataset, k, max_candidates,
        static_cast<GingerNodeId>(u), candidate_keys);
    add_neighbors(static_cast<GingerNodeId>(u), std::move(neighbors));
  });

  if (pipnn_config.print_stats()) {
    ABSL_LOG(INFO) << "PiPNN k-NN deduplication and reranking finished in "
                   << absl::Now() - rerank_start;
    ABSL_LOG(INFO) << "PiPNN k-NN graph build finished in "
                   << absl::Now() - total_start;
  }

  return absl::OkStatus();
}

}  // namespace ginger
