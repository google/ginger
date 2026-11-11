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

#include "ginger/pipnn/pipnn_impl.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "ginger/internal/parallel_for_with_status.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/many_to_many_topk.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "ginger/turboquant/pipnn/assign_leaders.h"
#include "ginger/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

namespace ginger::pipnn_internal {
namespace {
absl::Status CheckCancelled(const std::function<bool()>& is_cancelled) {
  if (is_cancelled()) {
    return absl::CancelledError();
  }
  return absl::OkStatus();
}

constexpr int kMaxDepth = 8;
constexpr int kConcerningDepth = 6;
constexpr double kShrinkage = 0.8;

// Thread-safe statistics collector for PiPNN leaf clusters across worker
// threads, tracking leaf counts, size distributions, and depth.
struct LeafStats {
  std::atomic<int64_t> num_leaves{0};
  std::atomic<int64_t> total_leaf_points{0};
  std::atomic<int64_t> min_leaf_size{std::numeric_limits<int64_t>::max()};
  std::atomic<int64_t> max_leaf_size{0};
  std::atomic<int> max_leaf_depth{0};
  std::atomic<int64_t> total_leaf_depth{0};

  void RecordLeaf(size_t leaf_size, int depth) {
    num_leaves.fetch_add(1, std::memory_order_relaxed);
    total_leaf_points.fetch_add(leaf_size, std::memory_order_relaxed);
    const int64_t sz = static_cast<int64_t>(leaf_size);
    int64_t cur_min = min_leaf_size.load(std::memory_order_relaxed);
    while (sz < cur_min && !min_leaf_size.compare_exchange_weak(
                               cur_min, sz, std::memory_order_relaxed)) {
    }
    int64_t cur_max = max_leaf_size.load(std::memory_order_relaxed);
    while (sz > cur_max && !max_leaf_size.compare_exchange_weak(
                               cur_max, sz, std::memory_order_relaxed)) {
    }
    if (depth > 0) {
      RecordLeafDepth(depth);
    }
  }

  void RecordLeafDepth(int depth) {
    total_leaf_depth.fetch_add(depth, std::memory_order_relaxed);
    int cur_max = max_leaf_depth.load(std::memory_order_relaxed);
    while (depth > cur_max && !max_leaf_depth.compare_exchange_weak(
                                  cur_max, depth, std::memory_order_relaxed)) {
    }
  }

  void LogStats(int pass) const {
    const int64_t leaves = num_leaves.load(std::memory_order_relaxed);
    const int64_t total_pts = total_leaf_points.load(std::memory_order_relaxed);
    const int64_t min_sz =
        (leaves == 0) ? 0 : min_leaf_size.load(std::memory_order_relaxed);
    const int64_t max_sz = max_leaf_size.load(std::memory_order_relaxed);
    const double avg_sz =
        leaves == 0 ? 0.0 : static_cast<double>(total_pts) / leaves;
    const int max_depth = max_leaf_depth.load(std::memory_order_relaxed);
    const double avg_depth =
        leaves == 0 ? 0.0
                    : static_cast<double>(
                          total_leaf_depth.load(std::memory_order_relaxed)) /
                          leaves;

    ABSL_LOG(INFO) << "PiPNN pass " << pass + 1 << " leaf statistics: "
                   << "num_leaves = " << leaves
                   << ", min_leaf_size = " << min_sz
                   << ", avg_leaf_size = " << avg_sz
                   << ", max_leaf_size = " << max_sz
                   << ", total_leaf_points = " << total_pts
                   << ", leaf_depth (avg/max) = " << avg_depth << "/"
                   << max_depth;
  }
};

struct LeaderAssignment {
  parlay::sequence<int16_t> assignment;
  int num_leaders = 0;
};

// Merges undersized clusters (< `min_leaf_size`) into larger clusters without
// exceeding `max_leaf_size`.
//
// Invariant: Assumes that the node IDs within each input cluster are sorted in
// ascending order, and preserves this sorted node ID order in merged clusters.
//
// Behavior:
// 1. Clusters are sorted descending by cluster size.
// 2. In each iteration, the smallest cluster (`clusters.back()`) is popped and
//    tested for merging into the second-smallest cluster (the new
//    `clusters.back()`). Merging the smallest clusters with each other builds
//    them up to `min_leaf_size` without prematurely bloating larger clusters.
// 3. Since clusters are sorted descending by size, if the smallest remaining
//    cluster cannot accommodate `small_cluster` without exceeding
//    `max_leaf_size`, no other remaining cluster can fit it either (assuming
//    disjoint clusters). In this case, merging stops.
// 4. Node IDs are merged via `absl::c_set_union` to deduplicate replicated
//    points (which occur when `fanout > 1`), maintaining the sorted node ID
//    invariant.
// 5. Merged clusters are bubbled up to maintain descending order of cluster
//    sizes.
void MergeSmallClusters(
    parlay::sequence<parlay::sequence<GingerNodeId>>& clusters,
    int min_leaf_size, int max_leaf_size) {
  // Use stable_sort to maintain the deterministic leader_idx order when
  // cluster sizes are equal.
  absl::c_stable_sort(clusters, [](const parlay::sequence<GingerNodeId>& a,
                                   const parlay::sequence<GingerNodeId>& b) {
    return a.size() > b.size();
  });
  while (clusters.size() > 1 && clusters.back().size() < min_leaf_size) {
    parlay::sequence<GingerNodeId> small_cluster = std::move(clusters.back());
    clusters.pop_back();

    // Since clusters is sorted descending by size, clusters.back() is the
    // smallest remaining cluster. If small_cluster doesn't fit here, no other
    // cluster can fit it.
    if (clusters.back().size() + small_cluster.size() <= max_leaf_size) {
      int merge_target_idx = clusters.size() - 1;
      ABSL_DCHECK(absl::c_is_sorted(clusters[merge_target_idx]));
      ABSL_DCHECK(absl::c_is_sorted(small_cluster));
      parlay::sequence<GingerNodeId> merged;
      merged.reserve(clusters[merge_target_idx].size() + small_cluster.size());
      // Use absl::c_set_union because when fanout > 1, points can be assigned
      // to multiple clusters and must be deduplicated when merging.
      absl::c_set_union(clusters[merge_target_idx], small_cluster,
                        std::back_inserter(merged));
      clusters[merge_target_idx] = std::move(merged);
      int current_idx = merge_target_idx;
      while (current_idx > 0 &&
             clusters[current_idx].size() > clusters[current_idx - 1].size()) {
        std::swap(clusters[current_idx], clusters[current_idx - 1]);
        current_idx--;
      }
    } else {
      clusters.push_back(std::move(small_cluster));
      break;
    }
  }
}

LeaderAssignment NearestLeaders(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    const parlay::sequence<GingerNodeId>& cluster, int depth, int pass,
    const PipnnConfig::LeaderSelectionConfig& leader_config,
    absl::Span<const int> fanout_sequence, int seed, bool print_stats) {
  const int num_leaders =
      (depth == 0)
          ? std::min<int>(leader_config.top_level_leaders(), cluster.size())
          : std::clamp<int>(cluster.size() * leader_config.leader_sample_rate(),
                            3, leader_config.max_num_leaders());
  ABSL_QCHECK_LE(num_leaders, cluster.size());
  ABSL_DCHECK_LE(num_leaders, std::numeric_limits<int16_t>::max());
  const int fanout =
      std::min(num_leaders, depth >= static_cast<int>(fanout_sequence.size())
                                ? 1
                                : fanout_sequence[depth]);
  std::vector<GingerNodeId> leaders(num_leaders);
  std::mt19937 prng(cluster.size() + depth + pass + seed);
  std::sample(cluster.begin(), cluster.end(), leaders.begin(), num_leaders,
              prng);
  PreTransposedTurboQuantDataset leader_dataset =
      PreTransposedTurboQuantDataset::Build(points, absl::MakeSpan(leaders),
                                            quantizer);

  parlay::sequence<int16_t> assignment(cluster.size() * fanout);
  const int block_size = (depth == 0) ? 1024 : 2048;
  const absl::Time start_assignment = absl::Now();
  parlay::blocked_for(
      0, cluster.size(), block_size,
      [&](size_t /*block_id*/, size_t start, size_t end) {
        IndexSubset query_subset = IndexSubset::Gathered(
            absl::MakeSpan(cluster).subspan(start, end - start));
        auto assign_results = [&](const auto& results) {
          for (size_t j = start; j < end; ++j) {
            for (int k = 0; k < fanout; ++k) {
              auto entry = results[j - start].Get(k);
              assignment[j * fanout + k] = static_cast<int16_t>(entry.index);
            }
          }
        };
        if (fanout == 1) {
          assign_results(
              AssignToLeadersTopK1(points, query_subset, leader_dataset));
        } else if (fanout <= 3) {
          assign_results(
              AssignToLeadersTopK3(points, query_subset, leader_dataset));
        } else if (fanout <= 10) {
          assign_results(
              AssignToLeadersTopK10(points, query_subset, leader_dataset));
        } else {
          // Slow fallback case.
          std::vector<TopKResult> results = ManyToManyTurboQuantTopK(
              points, leader_dataset, query_subset, fanout);
          for (size_t j = start; j < end; ++j) {
            for (int k = 0; k < fanout; ++k) {
              assignment[j * fanout + k] = static_cast<int16_t>(
                  results[(j - start) * fanout + k].db_idx);
            }
          }
        }
        // For 5% of points, deterministically replace the last assigned leader
        // with a distinct random leader to promote cross-cluster connectivity.
        if (leader_config.random_leader_replacement() && fanout > 1 &&
            num_leaders > 1) {
          constexpr float kRandomLeaderSamplingRate = 0.05f;
          std::mt19937 block_prng(cluster.size() + start + depth + pass + seed);
          std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
          std::uniform_int_distribution<int16_t> dist(0, num_leaders - 1);
          const int rand_slot = fanout - 1;
          for (size_t j = start; j < end; ++j) {
            if (prob_dist(block_prng) >= kRandomLeaderSamplingRate) {
              continue;
            }
            const absl::Span<const int16_t> existing(&assignment[j * fanout],
                                                     rand_slot);
            for (int retry = 0; retry < 10; ++retry) {
              const int16_t rand_leader = dist(block_prng);
              if (!absl::c_linear_search(existing, rand_leader)) {
                assignment[j * fanout + rand_slot] = rand_leader;
                break;
              }
            }
          }
        }
      });
  if (depth == 0 && print_stats) {
    ABSL_LOG(INFO) << "PiPNN pass " << pass + 1
                   << " top-level leader assignment (depth 0) took "
                   << absl::Now() - start_assignment << " (fanout = " << fanout
                   << ", num leaders = " << num_leaders
                   << ", num points = " << cluster.size() << ")";
  }
  return LeaderAssignment{
      .assignment = std::move(assignment),
      .num_leaders = num_leaders,
  };
}

// Groups points in `cluster` by their assigned leader indices from `assignment`
// and sorts each resulting sub-cluster by node ID.
parlay::sequence<parlay::sequence<GingerNodeId>> GroupByLeaders(
    const parlay::sequence<GingerNodeId>& cluster,
    const parlay::sequence<int16_t>& assignment, int fanout, int num_leaders) {
  const auto pairs = parlay::delayed_tabulate(
      assignment.size(), [&](size_t i) -> std::pair<int16_t, GingerNodeId> {
        return {assignment[i], cluster[i / fanout]};
      });
  parlay::sequence<parlay::sequence<GingerNodeId>> clusters =
      parlay::group_by_index(pairs, num_leaders);
  // parlay::group_by_index uses parallel bucket hashing which scrambles
  // element order within clusters. Sort node lists to ensure subsequent
  // sampling and splitting are deterministic and satisfy the sorted
  // node ID invariant required by MergeSmallClusters (for efficient merging).
  parlay::parallel_for(0, clusters.size(),
                       [&](size_t c) { absl::c_sort(clusters[c]); });
  return clusters;
}

// Randomly partitions an oversized cluster into terminal leaf clusters of size
// at most `max_leaf_size`.
//
// Node IDs are shuffled pseudo-randomly to achieve balanced partitioning.
// Note: Clusters produced by SplitClusterRandomly are terminal leaves that are
// immediately processed by ProcessLeaf (or collected into `leaves`); they never
// enter MergeSmallClusters or subsequent recursive partitioning, so their node
// IDs do not need to be sorted.
parlay::sequence<parlay::sequence<GingerNodeId>> SplitClusterRandomly(
    parlay::sequence<GingerNodeId>& cluster, int depth, int pass, int seed,
    int max_leaf_size) {
  std::mt19937 prng(cluster.size() + depth + pass + seed);
  std::shuffle(cluster.begin(), cluster.end(), prng);
  const size_t num_splits =
      (cluster.size() + max_leaf_size - 1) / max_leaf_size;
  parlay::sequence<parlay::sequence<GingerNodeId>> splits;
  splits.reserve(num_splits);
  for (size_t i = 0; i < cluster.size(); i += max_leaf_size) {
    size_t count = std::min<size_t>(max_leaf_size, cluster.size() - i);
    splits.emplace_back(cluster.begin() + i, cluster.begin() + i + count);
  }
  cluster.clear();
  return splits;
}

// Bundles the immutable parameters threaded through recursive clustering.
struct ClusteringContext {
  int Fanout(int depth) const {
    return depth >= static_cast<int>(fanout_sequence.size())
               ? 1
               : fanout_sequence[depth];
  }

  // Upper bound on how often each point of a non-leaf cluster at `depth` is
  // replicated across the leaves below it.
  size_t MaxReplication(int depth) const {
    size_t factor = 1;
    for (int d = depth; d < static_cast<int>(fanout_sequence.size()); ++d) {
      factor *= fanout_sequence[d];
    }
    return factor;
  }

  bool IsLeaf(absl::Span<const GingerNodeId> cluster, int depth) const {
    if (cluster.size() <= config.max_leaf_size()) {
      if (leaf_stats != nullptr) {
        leaf_stats->RecordLeaf(cluster.size(), depth);
      }
      return true;
    }
    return false;
  }

  const TurboQuantDatasetSoA& points;
  const TurboQuantCodec& quantizer;
  const PipnnConfig& config;
  absl::Span<const int> fanout_sequence;
  LeafStats* leaf_stats;  // Null when stats are disabled.
  int pass;
  const std::function<bool()>& is_cancelled;
};

// Returns true when a cluster is stuck (too deep or not shrinking fast enough)
// and should be randomly split into leaf-sized pieces instead of recursing.
bool ShouldSplit(size_t cluster_size, size_t parent_size, int depth) {
  return depth > kMaxDepth ||
         (depth > kConcerningDepth && cluster_size > kShrinkage * parent_size);
}

absl::StatusOr<parlay::sequence<parlay::sequence<GingerNodeId>>>
PartitionCluster(const ClusteringContext& ctx,
                 parlay::sequence<GingerNodeId>& cluster, int depth,
                 size_t parent_size) {
  ABSL_RETURN_IF_ERROR(CheckCancelled(ctx.is_cancelled));
  if (depth > 0 && ShouldSplit(cluster.size(), parent_size, depth - 1)) {
    return SplitClusterRandomly(cluster, depth - 1, ctx.pass, ctx.config.seed(),
                                ctx.config.max_leaf_size());
  }
  const int fanout = ctx.Fanout(depth);
  LeaderAssignment leaders =
      NearestLeaders(ctx.points, ctx.quantizer, cluster, depth, ctx.pass,
                     ctx.config.leader_selection(), ctx.fanout_sequence,
                     ctx.config.seed(), ctx.config.print_stats());
  parlay::sequence<parlay::sequence<GingerNodeId>> clusters =
      GroupByLeaders(cluster, leaders.assignment, fanout, leaders.num_leaders);
  // Eagerly free memory in the parent frame's cluster and assignment buffers
  // so that peak memory usage does not scale with the recursion depth.
  cluster.clear();
  leaders.assignment.clear();
  MergeSmallClusters(clusters, ctx.config.min_leaf_size(),
                     ctx.config.max_leaf_size());
  return clusters;
}

// Runs `fn(c)` for `c` in `[0, num_clusters)`, executing in parallel at shallow
// depths (`depth < 3`) and sequentially at `depth >= 3` to bound stack growth
// from nested parlay work-stealing joins.
template <typename Fn>
absl::Status ParallelForSubclusters(size_t num_clusters, int depth, Fn fn) {
  return parallel_for_with_status(0, num_clusters, std::move(fn),
                                  depth >= 3 ? num_clusters : 1);
}

// Partitions a non-leaf `cluster` and returns all leaves below it.
absl::StatusOr<parlay::sequence<parlay::sequence<GingerNodeId>>> CollectLeaves(
    const ClusteringContext& ctx, parlay::sequence<GingerNodeId>& cluster,
    int depth, size_t parent_size) {
  const size_t num_points = cluster.size();
  ABSL_ASSIGN_OR_RETURN(
      parlay::sequence<parlay::sequence<GingerNodeId>> clusters,
      PartitionCluster(ctx, cluster, depth, parent_size));
  parlay::sequence<parlay::sequence<GingerNodeId>> leaves;
  parlay::sequence<parlay::sequence<GingerNodeId>> inner;
  for (parlay::sequence<GingerNodeId>& sub : clusters) {
    if (ctx.IsLeaf(sub, depth + 1)) {
      leaves.push_back(std::move(sub));
    } else {
      inner.push_back(std::move(sub));
    }
  }
  if (inner.empty()) {
    return leaves;
  }
  parlay::sequence<parlay::sequence<parlay::sequence<GingerNodeId>>>
      per_cluster_leaves(inner.size());
  ABSL_RETURN_IF_ERROR(ParallelForSubclusters(
      inner.size(), depth, [&](size_t c) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(
            per_cluster_leaves[c],
            CollectLeaves(ctx, inner[c], depth + 1, num_points));
        return absl::OkStatus();
      }));
  per_cluster_leaves.push_back(std::move(leaves));
  return parlay::flatten(std::move(per_cluster_leaves));
}

// Recursively partitions `cluster` and invokes `on_leaf` (possibly in
// parallel) on each leaf as soon as it is formed.
absl::Status StreamLeaves(const ClusteringContext& ctx,
                          parlay::sequence<GingerNodeId>& cluster, int depth,
                          size_t parent_size, LeafCallback on_leaf) {
  if (ctx.IsLeaf(cluster, depth)) {
    on_leaf(absl::MakeConstSpan(cluster));
    cluster.clear();
    return absl::OkStatus();
  }
  const size_t num_points = cluster.size();
  ABSL_ASSIGN_OR_RETURN(
      parlay::sequence<parlay::sequence<GingerNodeId>> clusters,
      PartitionCluster(ctx, cluster, depth, parent_size));
  return ParallelForSubclusters(
      clusters.size(), depth, [&](size_t c) -> absl::Status {
        return StreamLeaves(ctx, clusters[c], depth + 1, num_points, on_leaf);
      });
}

// Sequentially walks the partitioning tree and groups clusters into chunks of
// at most `chunk_budget` estimated leaf points. Clusters exceeding the budget
// are partitioned here and their subclusters visited recursively. The budget is
// a soft limit: a leaf exceeding it forms a chunk on its own.
// Each chunk is processed in parallel via `StreamLeaves` (partition and leaf
// processing), followed by `on_chunk_done` once no leaf is in flight.
class ChunkedPartitioner {
 public:
  ChunkedPartitioner(const ClusteringContext& ctx, LeafCallback on_leaf,
                     absl::FunctionRef<void()> on_chunk_done)
      : ctx_(ctx), on_leaf_(on_leaf), on_chunk_done_(on_chunk_done) {}

  absl::Status Run(parlay::sequence<GingerNodeId>& point_ids) {
    const size_t root_replication = ctx_.MaxReplication(/*depth=*/0);
    chunk_budget_ = ChunkBudget(point_ids.size() * root_replication,
                                ctx_.config.robust_prune_chunk_fraction());
    const size_t estimate =
        EstimatedLeafPoints(point_ids.size(), root_replication);
    ABSL_RETURN_IF_ERROR(
        Visit(point_ids, /*depth=*/0, /*parent_size=*/0, estimate));
    return RunPending();
  }

 private:
  struct PendingCluster {
    parlay::sequence<GingerNodeId> ids;
    int depth;
    size_t parent_size;
  };

  size_t EstimatedLeafPoints(size_t size, size_t replication) const {
    return size <= ctx_.config.max_leaf_size() ? size : size * replication;
  }

  // `estimate` is the estimated number of leaf points below `cluster`.
  absl::Status Visit(parlay::sequence<GingerNodeId>& cluster, int depth,
                     size_t parent_size, size_t estimate) {
    if (estimate > chunk_budget_ &&
        cluster.size() > ctx_.config.max_leaf_size()) {
      const size_t num_points = cluster.size();
      ABSL_ASSIGN_OR_RETURN(
          parlay::sequence<parlay::sequence<GingerNodeId>> clusters,
          PartitionCluster(ctx_, cluster, depth, parent_size));
      const size_t child_replication = ctx_.MaxReplication(depth + 1);
      for (parlay::sequence<GingerNodeId>& child : clusters) {
        ABSL_RETURN_IF_ERROR(
            Visit(child, depth + 1, num_points,
                  EstimatedLeafPoints(child.size(), child_replication)));
      }
      return absl::OkStatus();
    }
    if (!pending_.empty() && pending_points_ + estimate > chunk_budget_) {
      ABSL_RETURN_IF_ERROR(RunPending());
    }
    pending_.push_back({std::move(cluster), depth, parent_size});
    pending_points_ += estimate;
    return absl::OkStatus();
  }

  absl::Status RunPending() {
    if (pending_.empty()) return absl::OkStatus();
    ABSL_RETURN_IF_ERROR(parallel_for_with_status(
        0, pending_.size(),
        [&](size_t i) -> absl::Status {
          PendingCluster& p = pending_[i];
          return StreamLeaves(ctx_, p.ids, p.depth, p.parent_size, on_leaf_);
        },
        /*granularity=*/1));
    on_chunk_done_();
    pending_.clear();
    pending_points_ = 0;
    return absl::OkStatus();
  }

  const ClusteringContext& ctx_;
  size_t chunk_budget_ = 0;
  LeafCallback on_leaf_;
  absl::FunctionRef<void()> on_chunk_done_;
  std::vector<PendingCluster> pending_;
  size_t pending_points_ = 0;
};

template <typename ClusterFn>
absl::Status RunPartitioningPass(const TurboQuantDatasetSoA& points,
                                 const TurboQuantCodec& quantizer,
                                 const PipnnConfig& config, int pass,
                                 const std::function<bool()>& is_cancelled,
                                 ClusterFn cluster_fn) {
  ABSL_DCHECK(is_cancelled != nullptr);
  ABSL_RETURN_IF_ERROR(CheckCancelled(is_cancelled));
  std::vector<int> fanout_sequence(
      config.leader_selection().fanout_sequence().begin(),
      config.leader_selection().fanout_sequence().end());
  parlay::sequence<GingerNodeId> point_ids =
      parlay::to_sequence(parlay::iota<GingerNodeId>(points.num_datapoints()));
  std::optional<LeafStats> leaf_stats;
  if (config.print_stats()) {
    leaf_stats.emplace();
  }
  ClusteringContext ctx{
      .points = points,
      .quantizer = quantizer,
      .config = config,
      .fanout_sequence = absl::MakeSpan(fanout_sequence),
      .leaf_stats = leaf_stats.has_value() ? &*leaf_stats : nullptr,
      .pass = pass,
      .is_cancelled = is_cancelled,
  };
  ABSL_RETURN_IF_ERROR(cluster_fn(ctx, point_ids));
  if (leaf_stats.has_value()) {
    leaf_stats->LogStats(pass);
  }
  return absl::OkStatus();
}

}  // namespace

size_t ChunkBudget(size_t total_leaf_points, float fraction) {
  return fraction >= 1.0f ? std::numeric_limits<size_t>::max()
                          : static_cast<size_t>(total_leaf_points * fraction);
}

absl::Status ValidatePipnnPartitioningConfig(const PipnnConfig& config) {
  if (config.num_passes() < 1) {
    return absl::InvalidArgumentError("num_passes must be at least 1.");
  }
  if (config.max_leaf_size() < 4) {
    return absl::InvalidArgumentError("max_leaf_size must be at least 4.");
  }
  const PipnnConfig::LeaderSelectionConfig& leader_config =
      config.leader_selection();
  if (leader_config.leader_sample_rate() <= 0.0f ||
      leader_config.leader_sample_rate() > 1.0f) {
    return absl::InvalidArgumentError("leader_sample_rate must be in (0, 1].");
  }
  if (leader_config.max_num_leaders() < 3 ||
      leader_config.max_num_leaders() > std::numeric_limits<int16_t>::max()) {
    return absl::InvalidArgumentError("max_num_leaders must be in [3, 32767].");
  }
  if (leader_config.top_level_leaders() < 1 ||
      leader_config.top_level_leaders() > std::numeric_limits<int16_t>::max()) {
    return absl::InvalidArgumentError(
        "top_level_leaders must be in [1, 32767].");
  }
  if (config.min_leaf_size() <= 0) {
    return absl::InvalidArgumentError("min_leaf_size must be positive.");
  }
  if (config.min_leaf_size() > config.max_leaf_size() / 2) {
    return absl::InvalidArgumentError(
        "min_leaf_size must be less than or equal to max_leaf_size / 2.");
  }
  if (leader_config.fanout_sequence().empty()) {
    return absl::InvalidArgumentError(
        "fanout_sequence must contain at least one element.");
  }
  for (int fanout : leader_config.fanout_sequence()) {
    if (fanout < 1) {
      return absl::InvalidArgumentError("fanout values must be at least 1.");
    }
  }
  return absl::OkStatus();
}

absl::Status PartitionPointsIntoLeaves(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    const PipnnConfig& config, int pass, LeafCallback on_leaf,
    absl::FunctionRef<void()> on_chunk_done,
    const std::function<bool()>& is_cancelled) {
  return RunPartitioningPass(
      points, quantizer, config, pass, is_cancelled,
      [&](const ClusteringContext& ctx,
          parlay::sequence<GingerNodeId>& point_ids) {
        return ChunkedPartitioner(ctx, on_leaf, on_chunk_done).Run(point_ids);
      });
}

absl::StatusOr<parlay::sequence<parlay::sequence<GingerNodeId>>>
PartitionPointsIntoLeaves(const TurboQuantDatasetSoA& points,
                          const TurboQuantCodec& quantizer,
                          const PipnnConfig& config, int pass,
                          const std::function<bool()>& is_cancelled) {
  parlay::sequence<parlay::sequence<GingerNodeId>> leaves;
  ABSL_RETURN_IF_ERROR(RunPartitioningPass(
      points, quantizer, config, pass, is_cancelled,
      [&](const ClusteringContext& ctx,
          parlay::sequence<GingerNodeId>& point_ids) -> absl::Status {
        if (ctx.IsLeaf(point_ids, /*depth=*/0)) {
          // only called at root --> thread-safe.
          leaves.push_back(std::move(point_ids));
          return absl::OkStatus();
        }
        ABSL_ASSIGN_OR_RETURN(leaves, CollectLeaves(ctx, point_ids, /*depth=*/0,
                                                    /*parent_size=*/0));
        return absl::OkStatus();
      }));
  return leaves;
}

}  // namespace ginger::pipnn_internal
