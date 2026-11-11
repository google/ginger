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

#include "ginger/pipnn/pipnn.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/internal/parallel_for_with_status.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/pipnn/hash_prune.h"
#include "ginger/pipnn/pipnn_impl.h"
#include "ginger/pipnn/simhash.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/search/search_functors.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/pipnn/process_leaf.h"
#include "ginger/turboquant/pipnn/robust_prune.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_reference_stub.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

namespace ginger {
namespace {

using ::ginger::pipnn_internal::ChunkBudget;
using ::ginger::pipnn_internal::PartitionPointsIntoLeaves;
using ::ginger::pipnn_internal::ValidatePipnnPartitioningConfig;
absl::Status CheckCancelled(const std::function<bool()>& is_cancelled) {
  if (is_cancelled()) {
    return absl::CancelledError();
  }
  return absl::OkStatus();
}

constexpr float kOnTheFlyPruneDegreeFraction = 0.75f;

void PruneAndSetNeighbors(GingerNodeId u, absl::Span<DistanceAndId> candidates,
                          FlatGraph& graph, const TurboQuantDatasetSoA& points,
                          const TurboQuantBuildScorer& scorer,
                          const RobustPruneConfig& prune_config,
                          size_t target_degree) {
  ABSL_CHECK_OK(scorer(u, candidates));

  thread_local RobustPruneWorkspace workspace;
  thread_local std::vector<GingerNodeId> cand_ids;
  thread_local std::vector<float> cand_dists;
  thread_local std::vector<GingerNodeId> survivors;

  cand_ids.resize(candidates.size());
  cand_dists.resize(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    cand_ids[i] = candidates[i].id;
    cand_dists[i] = candidates[i].distance;
  }

  AcceleratedRobustPrune(points, cand_ids, cand_dists, u, prune_config,
                         target_degree, &workspace, survivors);
  graph.SetNeighbors(u, absl::MakeSpan(survivors));
}

struct RobustPruneOnTheFly {
  RobustPruneOnTheFly(int32_t num_nodes, int32_t max_degree,
                      const TurboQuantDatasetSoA& points,
                      const TurboQuantCodec& quantizer,
                      RobustPruneConfig prune_config)
      : graph(num_nodes, max_degree),
        node_mutexes(num_nodes),
        points(&points),
        scorer(&points, &quantizer),
        in_flight_prune_config(std::move(prune_config)),
        on_the_fly_prune_degree(std::max<size_t>(
            1,
            static_cast<size_t>(max_degree * kOnTheFlyPruneDegreeFraction))) {
    // Disable keep_pruned_neighbors during in-flight prunes to free extra
    // capacity when fewer than on_the_fly_prune_degree candidates survive and
    // avoid re-scoring backfilled edges on subsequent overflows.
    in_flight_prune_config.set_keep_pruned_neighbors(false);
  }

  FlatGraph graph;
  std::vector<absl::Mutex> node_mutexes;
  const TurboQuantDatasetSoA* points;
  TurboQuantBuildScorer scorer;
  RobustPruneConfig in_flight_prune_config;
  size_t on_the_fly_prune_degree;
};

// We expect that the size of `new_targets` is small, 4 on average.
void AddEdgesAndPruneIfNeeded(GingerNodeId u,
                              absl::Span<const GingerNodeId> new_targets,
                              RobustPruneOnTheFly& ctx) {
  absl::MutexLock lock(ctx.node_mutexes[u]);
  absl::Span<const GingerNodeId> current_neighbors = ctx.graph.Neighbors(u);

  // Filter duplicates.
  absl::InlinedVector<GingerNodeId, 8> valid_new_targets;
  valid_new_targets.reserve(new_targets.size());
  for (GingerNodeId v : new_targets) {
    if (v == u || absl::c_linear_search(current_neighbors, v) ||
        absl::c_linear_search(valid_new_targets, v)) {
      continue;
    }
    valid_new_targets.push_back(v);
  }
  if (valid_new_targets.empty()) return;

  if (current_neighbors.size() + valid_new_targets.size() <=
      static_cast<size_t>(ctx.graph.MaxDegree())) {
    ctx.graph.AppendNeighbors(u, valid_new_targets);
  } else {
    absl::InlinedVector<DistanceAndId, 96> candidates;
    candidates.reserve(current_neighbors.size() + valid_new_targets.size());
    for (GingerNodeId v : current_neighbors) {
      candidates.push_back({0.0f, v});
    }
    for (GingerNodeId v : valid_new_targets) {
      candidates.push_back({0.0f, v});
    }
    PruneAndSetNeighbors(u, absl::MakeSpan(candidates), ctx.graph, *ctx.points,
                         ctx.scorer, ctx.in_flight_prune_config,
                         ctx.on_the_fly_prune_degree);
  }
}

// Computes bidirected kNN edges for a leaf cluster and merges each
// edge (along with its distance) directly into the per-node angular-cone
// reservoir.
void MergeLeafIntoHashPrune(const TurboQuantDatasetSoA& points,
                            const TurboQuantCodec& quantizer,
                            absl::Span<const GingerNodeId> cluster,
                            ProcessLeafWorkspace& workspace,
                            HashPrune& hash_prune) {
  if (cluster.size() <= 3) return;
  ProcessLeafTop2(quantizer, points, cluster, workspace);

  const size_t n = cluster.size();
  for (size_t i = 0; i < n; ++i) {
    const Top2Result& r = workspace.top2_results[i];
    const std::array<DistanceAndId, 2> fwd_edges = {{
        {r.best_score, cluster[r.best_idx]},
        {r.second_best_score, cluster[r.second_best_idx]},
    }};
    hash_prune.MergeNeighbors(cluster[i], absl::MakeSpan(fwd_edges));

    const DistanceAndId rev_edge1 = {r.best_score, cluster[i]};
    hash_prune.MergeNeighbors(cluster[r.best_idx],
                              absl::MakeSpan(&rev_edge1, 1));

    const DistanceAndId rev_edge2 = {r.second_best_score, cluster[i]};
    hash_prune.MergeNeighbors(cluster[r.second_best_idx],
                              absl::MakeSpan(&rev_edge2, 1));
  }
}

// Bidirected kNN using ProcessLeafTop2. Edges are grouped by destination node
// first via counting sort, then passed in batch to the consumer lambda.
template <typename Consumer>
void ForEachLeafEdgeGrouped(const TurboQuantDatasetSoA& points,
                            const TurboQuantCodec& quantizer,
                            absl::Span<const GingerNodeId> cluster,
                            ProcessLeafWorkspace& workspace, Consumer consume) {
  if (cluster.size() <= 3) return;
  ProcessLeafTop2(quantizer, points, cluster, workspace);

  const size_t n = cluster.size();
  auto& offsets = workspace.leaf_edge_offsets;
  auto& targets = workspace.leaf_edge_targets;

  // Count total edges incident to each node i (2 forward + reverse edges).
  offsets.assign(n + 1, 0);
  for (size_t i = 0; i < n; ++i) {
    const Top2Result& r = workspace.top2_results[i];
    offsets[i] += 2;
    offsets[r.best_idx] += 1;
    offsets[r.second_best_idx] += 1;
  }

  // Exclusive prefix sum to compute start offsets for each node's edge range.
  size_t total_edges = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t deg = offsets[i];
    offsets[i] = total_edges;
    total_edges += deg;
  }
  offsets[n] = total_edges;

  if (targets.size() < total_edges) {
    targets.resize(total_edges);
  }

  // Scatter forward and reverse edges; afterwards offsets[i] is the end of
  // node i's range.
  for (size_t i = 0; i < n; ++i) {
    const Top2Result& r = workspace.top2_results[i];
    targets[offsets[i]++] = cluster[r.best_idx];
    targets[offsets[i]++] = cluster[r.second_best_idx];
    targets[offsets[r.best_idx]++] = cluster[i];
    targets[offsets[r.second_best_idx]++] = cluster[i];
  }

  size_t start = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t end = offsets[i];
    absl::Span<const GingerNodeId> edge_targets(&targets[start], end - start);
    consume(cluster[i], edge_targets);
    start = end;
  }
}

void ProcessLeaf(const TurboQuantDatasetSoA& points,
                 const TurboQuantCodec& quantizer,
                 absl::Span<const GingerNodeId> cluster,
                 ProcessLeafWorkspace& workspace, HashPrune* hash_prune,
                 RobustPruneOnTheFly* on_the_fly_ctx) {
  if (hash_prune != nullptr) {
    MergeLeafIntoHashPrune(points, quantizer, cluster, workspace, *hash_prune);
  } else {
    ForEachLeafEdgeGrouped(
        points, quantizer, cluster, workspace,
        [&](GingerNodeId u, absl::Span<const GingerNodeId> edges) {
          AddEdgesAndPruneIfNeeded(u, edges, *on_the_fly_ctx);
        });
  }
}

// TODO: Remove fallback once config.robust_prune_alpha is deleted.
// We still have configs serialized that use this field.
absl::StatusOr<RobustPruneConfig> GetRobustPruneConfig(
    const PipnnConfig& config) {
  ABSL_ASSIGN_OR_RETURN(const float alpha, GetRobustPruneAlpha(config));
  RobustPruneConfig prune_config = config.prune_config();
  prune_config.set_alpha(alpha);
  return prune_config;
}

// Maximum number of source nodes whose reverse edges are merged per batch in
// `PruneWithReverseEdgeBackfill`.
constexpr size_t kMaxReverseBackfillBatchSize = 1'000'000;

// Per-worker scratch buffers for `PruneWithReverseEdgeBackfill`, aligned to a
// cache line to avoid false sharing between workers.
struct alignas(64) PruneScratch {
  RobustPruneWorkspace workspace;
  std::vector<GingerNodeId> cand_ids;
  std::vector<float> cand_dists;
  std::vector<DistanceAndId> candidates;
  std::vector<GingerNodeId> survivors;
  std::vector<GingerNodeId> new_reverse;
  std::vector<GingerNodeId> final_nbrs;
};

// The reverse edge `(target, source)` of a primary survivor edge
// `source -> target`.
using ReverseEdge = std::pair<GingerNodeId, GingerNodeId>;

// Merges `edges`, which share the same target `v` and are sorted by increasing
// source id, into the adjacency list of `v`. The list is laid out as
// `[primary survivors, reverse edges, tail]`, where the first two sections hold
// `primary_count` and `reverse_count` entries and the tail holds the remaining
// RobustPrune survivors and pruned candidates of `v` that still fit. The
// primary survivors are never modified, since `PruneWithReverseEdgeBackfill`
// reads them back when `v` is processed as a source in a later batch.
void MergeReverseEdges(absl::Span<const ReverseEdge> edges,
                       int32_t primary_count, int32_t& reverse_count,
                       const TurboQuantBuildScorer& scorer, FlatGraph& graph,
                       PruneScratch& s) {
  const GingerNodeId v = edges.front().first;
  const size_t max_degree = graph.MaxDegree();
  const absl::Span<const GingerNodeId> nbrs = graph.Neighbors(v);
  const absl::Span<const GingerNodeId> primary = nbrs.first(primary_count);
  const absl::Span<const GingerNodeId> reverse =
      nbrs.subspan(primary_count, reverse_count);
  const absl::Span<const GingerNodeId> tail =
      nbrs.subspan(primary_count + reverse_count);

  // Skip reverse edges that duplicate a primary survivor edge of `v`.
  s.new_reverse.clear();
  for (const auto& [target, source] : edges) {
    if (!absl::c_linear_search(primary, source)) {
      s.new_reverse.push_back(source);
    }
  }
  if (s.new_reverse.empty()) return;

  const size_t reverse_capacity = max_degree - primary.size();
  s.final_nbrs.assign(primary.begin(), primary.end());
  if (reverse.size() + s.new_reverse.size() <= reverse_capacity) {
    // Append the new reverse edges, then keep as much of the tail as still
    // fits, skipping entries that just became reverse edges. Truncating the
    // tail is lossless: whenever `j` tail entries later become reverse edges,
    // the reverse edges grow by at least `j`, so truncated entries never fit
    // again.
    s.final_nbrs.insert(s.final_nbrs.end(), reverse.begin(), reverse.end());
    s.final_nbrs.insert(s.final_nbrs.end(), s.new_reverse.begin(),
                        s.new_reverse.end());
    reverse_count += static_cast<int32_t>(s.new_reverse.size());
    for (const GingerNodeId candidate : tail) {
      if (s.final_nbrs.size() >= max_degree) break;
      if (!absl::c_linear_search(s.new_reverse, candidate)) {
        s.final_nbrs.push_back(candidate);
      }
    }
  } else {
    // Keep only the closest `reverse_capacity` reverse edges by
    // `(distance, id)`, which leaves no room for the tail.
    s.candidates.clear();
    for (const GingerNodeId candidate : reverse) {
      s.candidates.push_back({0.0f, candidate});
    }
    for (const GingerNodeId candidate : s.new_reverse) {
      s.candidates.push_back({0.0f, candidate});
    }
    ABSL_CHECK_OK(scorer(v, absl::MakeSpan(s.candidates)));
    absl::c_partial_sort(s.candidates, s.candidates.begin() + reverse_capacity);
    for (size_t i = 0; i < reverse_capacity; ++i) {
      s.final_nbrs.push_back(s.candidates[i].id);
    }
    reverse_count = static_cast<int32_t>(reverse_capacity);
  }
  graph.SetNeighbors(v, absl::MakeSpan(s.final_nbrs));
}

// Fills the adjacency list of each node `u` in `graph` up to `max_degree`
// entries in the following order:
//   1. The closest `(max_degree + 1) / 2` RobustPrune survivors of `u`
//      ("primary survivors").
//   2. Reverse edges to the nodes `v` that have `u` among their primary
//      survivors. If they exceed the remaining capacity, only the closest ones
//      by `(distance, id)` are kept.
//   3. The remaining RobustPrune survivors of `u` in increasing distance order.
//   4. If `keep_pruned_neighbors` is set, the pruned candidates of `u` in
//      increasing distance order.
//
// `get_candidates(u, scratch)` must fill `scratch.cand_ids` and
// `scratch.cand_dists` with the candidates of `u` and their distances to `u`.
// `release_candidates()` is called once all candidates have been consumed, so
// that their memory can be freed before the reverse edges are merged.
template <typename GetCandidatesFn, typename ReleaseCandidatesFn>
void PruneWithReverseEdgeBackfill(FlatGraph& graph,
                                  const TurboQuantDatasetSoA& points,
                                  const TurboQuantBuildScorer& scorer,
                                  const RobustPruneConfig& prune_config,
                                  float chunk_fraction,
                                  GetCandidatesFn get_candidates,
                                  ReleaseCandidatesFn release_candidates) {
  const size_t num_nodes = graph.NumNodes();
  const size_t max_degree = graph.MaxDegree();

  const size_t forward_degree = (max_degree + 1) / 2;
  RobustPruneConfig forward_prune_config = prune_config;
  forward_prune_config.set_keep_pruned_neighbors(false);

  // Stage 1: Lay out each node's adjacency list as
  // `[primary survivors, remaining survivors, pruned candidates]`, where the
  // pruned candidates are only included if `keep_pruned_neighbors` is set.
  std::vector<int32_t> primary_counts(num_nodes);
  std::vector<PruneScratch> scratch(parlay::num_workers());
  parlay::parallel_for(0, num_nodes, [&](size_t u) {
    PruneScratch& s = scratch[parlay::worker_id()];

    get_candidates(static_cast<GingerNodeId>(u), s);
    AcceleratedRobustPrune(points, s.cand_ids, s.cand_dists,
                           static_cast<GingerNodeId>(u), forward_prune_config,
                           max_degree, &s.workspace, s.survivors);

    const size_t out_degree = s.survivors.size();
    primary_counts[u] =
        static_cast<int32_t>(std::min(out_degree, forward_degree));
    if (prune_config.keep_pruned_neighbors()) {
      // The survivors are a subsequence of the sorted candidates, so the other
      // candidates are the pruned ones in increasing distance order.
      size_t surv_idx = 0;
      for (const GingerNodeId candidate : s.workspace.sorted_candidate_ids) {
        if (s.survivors.size() >= max_degree) break;
        if (surv_idx < out_degree && candidate == s.survivors[surv_idx]) {
          ++surv_idx;
        } else {
          s.survivors.push_back(candidate);
        }
      }
    }
    graph.SetNeighbors(static_cast<GingerNodeId>(u),
                       absl::MakeSpan(s.survivors));
  });

  release_candidates();

  // Stage 2: Merge the reverse edges of all primary survivor edges into the
  // adjacency lists of their targets, processing the source nodes in batches
  // to bound the memory used by the reverse edges.
  std::vector<int32_t> reverse_counts(num_nodes, 0);
  const size_t batch_size = std::max<size_t>(
      1, std::min<size_t>(ChunkBudget(num_nodes, chunk_fraction),
                          kMaxReverseBackfillBatchSize));
  for (size_t batch_start = 0; batch_start < num_nodes;
       batch_start += batch_size) {
    const size_t batch_end = std::min(num_nodes, batch_start + batch_size);
    // Note: A source in this batch may already have been updated as a target
    // in an earlier batch. Reading its primary survivors from `graph` is only
    // correct because `MergeReverseEdges` keeps them as a prefix of the
    // adjacency list.
    parlay::sequence<ReverseEdge> reverse_edges = parlay::flatten(
        parlay::delayed_tabulate(batch_end - batch_start, [&](size_t i) {
          const GingerNodeId u = static_cast<GingerNodeId>(batch_start + i);
          return parlay::delayed_map(
              graph.Neighbors(u).first(primary_counts[u]),
              [u](GingerNodeId v) { return ReverseEdge(v, u); });
        }));

    // Group the reverse edges by target, with sources in increasing id order.
    parlay::sort_inplace(reverse_edges);
    const parlay::sequence<size_t> group_starts = parlay::pack_index(
        parlay::delayed_seq<bool>(reverse_edges.size(), [&](size_t i) {
          return i == 0 || reverse_edges[i].first != reverse_edges[i - 1].first;
        }));
    parlay::parallel_for(0, group_starts.size(), [&](size_t g) {
      const size_t start = group_starts[g];
      const size_t end = g + 1 < group_starts.size() ? group_starts[g + 1]
                                                     : reverse_edges.size();
      const GingerNodeId v = reverse_edges[start].first;
      MergeReverseEdges(
          absl::MakeConstSpan(reverse_edges).subspan(start, end - start),
          primary_counts[v], reverse_counts[v], scorer, graph,
          scratch[parlay::worker_id()]);
    });
  }
}

// Extracts candidate edge lists from the HashPrune reservoir (which partitions
// space into angular cones) and applies robust pruning (which sorts
// and prunes internally) to produce the final FlatGraph. Takes ownership of
// `hash_prune` so that the reservoir can be freed once it is no longer needed.
FlatGraph ExtractEdgeListsAndConvertToFlatGraph(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    const PipnnConfig& config, const RobustPruneConfig& prune_config,
    std::optional<HashPrune> hash_prune) {
  const size_t max_degree = config.max_degree();
  const absl::Time start_extract = absl::Now();
  FlatGraph graph(points.num_datapoints(), config.max_degree());

  if (prune_config.reverse_rng_backfill()) {
    const TurboQuantBuildScorer scorer{.dataset = &points,
                                       .quantizer = &quantizer};
    PruneWithReverseEdgeBackfill(
        graph, points, scorer, prune_config,
        config.robust_prune_chunk_fraction(),
        [&](GingerNodeId u, PruneScratch& s) {
          const auto ids = hash_prune->ValidPointIds(u);
          const auto dists = hash_prune->ValidDistances(u);
          s.cand_ids.assign(ids.begin(), ids.end());
          s.cand_dists.resize(dists.size());
          for (size_t i = 0; i < dists.size(); ++i) {
            s.cand_dists[i] = static_cast<float>(dists[i]);
          }
        },
        [&] { hash_prune.reset(); });
  } else {
    parlay::parallel_for(0, points.num_datapoints(), [&](size_t u) {
      // Thread-local scratch buffers reused across iterations on the worker
      // thread.
      thread_local RobustPruneWorkspace workspace;
      thread_local std::vector<GingerNodeId> survivors;
      thread_local std::vector<float> cand_dist;

      const auto candidate_distances = hash_prune->ValidDistances(u);
      const auto candidate_ids = hash_prune->ValidPointIds(u);
      cand_dist.resize(candidate_distances.size());
      for (size_t i = 0; i < candidate_distances.size(); ++i) {
        cand_dist[i] = static_cast<float>(candidate_distances[i]);
      }

      AcceleratedRobustPrune(points, candidate_ids,
                             absl::MakeConstSpan(cand_dist), u, prune_config,
                             max_degree, &workspace, survivors);
      graph.SetNeighbors(u, absl::MakeSpan(survivors));
    });
  }
  const absl::Duration extract_time = absl::Now() - start_extract;
  if (config.print_stats()) {
    ABSL_LOG(INFO) << "PiPNN graph extraction and robust pruning took "
                   << extract_time;
  }
  return graph;
}

void FinalizeAndPruneGraph(FlatGraph& graph, const TurboQuantDatasetSoA& points,
                           const TurboQuantBuildScorer& scorer,
                           const RobustPruneConfig& prune_config,
                           float chunk_fraction) {
  if (prune_config.reverse_rng_backfill()) {
    PruneWithReverseEdgeBackfill(
        graph, points, scorer, prune_config, chunk_fraction,
        [&](GingerNodeId u, PruneScratch& s) {
          const auto nbrs = graph.Neighbors(u);
          s.candidates.resize(nbrs.size());
          for (size_t i = 0; i < nbrs.size(); ++i) {
            s.candidates[i] = {0.0f, nbrs[i]};
          }
          ABSL_CHECK_OK(scorer(u, absl::MakeSpan(s.candidates)));
          s.cand_ids.resize(s.candidates.size());
          s.cand_dists.resize(s.candidates.size());
          for (size_t i = 0; i < s.candidates.size(); ++i) {
            s.cand_ids[i] = s.candidates[i].id;
            s.cand_dists[i] = s.candidates[i].distance;
          }
        },
        [] {});
    return;
  }
  parlay::blocked_for(0, graph.NumNodes(), /*block_size=*/1024,
                      [&](size_t /*block_id*/, size_t start, size_t end) {
                        std::vector<DistanceAndId> candidates;
                        for (GingerNodeId u = start; u < end; ++u) {
                          candidates.clear();
                          for (GingerNodeId v : graph.Neighbors(u)) {
                            candidates.push_back({0.0f, v});
                          }
                          PruneAndSetNeighbors(u, absl::MakeSpan(candidates),
                                               graph, points, scorer,
                                               prune_config, graph.MaxDegree());
                        }
                      });
}

absl::Status ValidatePipnnConfig(const PipnnConfig& config) {
  ABSL_RETURN_IF_ERROR(ValidatePipnnPartitioningConfig(config));

  const auto& hash_prune_config = config.hash_prune();
  if (hash_prune_config.reservoir_size() <= 0) {
    return absl::InvalidArgumentError("reservoir_size must be positive.");
  }
  if (hash_prune_config.num_hash_bits() <= 0 ||
      hash_prune_config.num_hash_bits() > 16) {
    return absl::InvalidArgumentError("num_hash_bits must be in [1, 16].");
  }
  if (config.max_degree() <= 0) {
    return absl::InvalidArgumentError("max_degree must be positive.");
  }
  ABSL_ASSIGN_OR_RETURN(const float alpha, GetRobustPruneAlpha(config));
  if (alpha < 1.0f) {
    return absl::InvalidArgumentError(
        "robust_prune alpha must be at least 1.0.");
  }
  if (config.robust_prune_chunk_fraction() <= 0.0f ||
      config.robust_prune_chunk_fraction() > 1.0f) {
    return absl::InvalidArgumentError(
        "robust_prune_chunk_fraction must be in (0, 1].");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<float> GetRobustPruneAlpha(const PipnnConfig& config) {
  if (config.has_robust_prune_alpha() && config.prune_config().has_alpha()) {
    return absl::InvalidArgumentError(
        "Cannot specify both robust_prune_alpha and prune_config.alpha.");
  }
  return config.prune_config().has_alpha() ? config.prune_config().alpha()
                                           : config.robust_prune_alpha();
}

absl::StatusOr<FlatGraph> BuildPipnnIndex(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    const PipnnConfig& config, const std::function<bool()>& is_cancelled) {
  ginger::internal::ParallelSchedulerReference scheduler;
  ABSL_DCHECK(is_cancelled != nullptr);
  ABSL_RETURN_IF_ERROR(ValidatePipnnConfig(config));
  ABSL_RETURN_IF_ERROR(CheckCancelled(is_cancelled));

  ABSL_ASSIGN_OR_RETURN(const RobustPruneConfig prune_config,
                        GetRobustPruneConfig(config));

  auto log_step = [&](absl::string_view label, absl::Time start) {
    if (config.print_stats()) {
      ABSL_LOG(INFO) << label << " took " << absl::Now() - start;
    }
  };

  const absl::Time total_start = absl::Now();
  if (config.print_stats()) {
    ABSL_LOG(INFO) << "Building PiPNN index for " << points.num_datapoints()
                   << " points with " << config.num_passes()
                   << " passes (max_leaf_size=" << config.max_leaf_size()
                   << ", min_leaf_size=" << config.min_leaf_size()
                   << ", max_degree=" << config.max_degree()
                   << ", robust_prune_alpha=" << prune_config.alpha()
                   << ", phased_build="
                   << (config.phased_build() ? "true" : "false")
                   << ", strategy="
                   << (config.pruning_strategy() ==
                               PipnnConfig::PRUNING_STRATEGY_HASH_PRUNE
                           ? "HashPrune"
                           : "RobustPruneOnTheFly")
                   << ").";
  }

  std::vector<ProcessLeafWorkspace> process_leaf_workspaces(
      parlay::num_workers());

  std::optional<HashPrune> hash_prune;
  std::optional<RobustPruneOnTheFly> on_the_fly_ctx;

  if (config.pruning_strategy() ==
      PipnnConfig::PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY) {
    on_the_fly_ctx.emplace(points.num_datapoints(), config.max_degree(), points,
                           quantizer, prune_config);
  } else {
    const absl::Time simhash_start = absl::Now();
    auto simhash_projections =
        SimHashProjections(points, config.hash_prune().num_hash_bits());
    log_step("SimHash projections computed", simhash_start);
    hash_prune.emplace(config.hash_prune().reservoir_size(),
                       std::move(simhash_projections));
  }

  for (int pass = 0; pass < config.num_passes(); ++pass) {
    const absl::Time pass_start = absl::Now();
    const absl::Time clustering_start = absl::Now();
    if (config.phased_build()) {
      ABSL_ASSIGN_OR_RETURN(
          parlay::sequence<parlay::sequence<GingerNodeId>> leaves,
          PartitionPointsIntoLeaves(points, quantizer, config, pass,
                                    is_cancelled));
      log_step(absl::StrCat("PiPNN pass ", pass + 1, " clustering"),
               clustering_start);
      size_t total_leaf_points = 0;
      for (const auto& leaf : leaves) total_leaf_points += leaf.size();
      const size_t chunk_budget =
          ChunkBudget(total_leaf_points, config.robust_prune_chunk_fraction());
      for (size_t start = 0, end = 0; start < leaves.size(); start = end) {
        size_t pts = 0;
        do {
          pts += leaves[end++].size();
        } while (end < leaves.size() &&
                 pts + leaves[end].size() <= chunk_budget);
        ABSL_RETURN_IF_ERROR(
            parallel_for_with_status(start, end, [&](size_t i) -> absl::Status {
              ABSL_RETURN_IF_ERROR(CheckCancelled(is_cancelled));
              ProcessLeaf(points, quantizer, leaves[i],
                          process_leaf_workspaces[parlay::worker_id()],
                          hash_prune ? &*hash_prune : nullptr,
                          on_the_fly_ctx ? &*on_the_fly_ctx : nullptr);
              return absl::OkStatus();
            }));
      }
    } else {
      ABSL_RETURN_IF_ERROR(PartitionPointsIntoLeaves(
          points, quantizer, config, pass,
          [&](absl::Span<const GingerNodeId> leaf) {
            ProcessLeaf(points, quantizer, leaf,
                        process_leaf_workspaces[parlay::worker_id()],
                        hash_prune ? &*hash_prune : nullptr,
                        on_the_fly_ctx ? &*on_the_fly_ctx : nullptr);
          },
          /*on_chunk_done=*/[]() {}, is_cancelled));
      log_step(absl::StrCat("PiPNN pass ", pass + 1, " clustering"),
               clustering_start);
    }
    if (config.print_stats()) {
      ABSL_LOG(INFO) << "PiPNN pass " << pass + 1 << " finished in "
                     << absl::Now() - pass_start;
    }
  }

  const absl::Time start_final = absl::Now();

  if (hash_prune.has_value()) {
    if (config.print_stats()) {
      hash_prune->LogStats();
    }
    FlatGraph graph = ExtractEdgeListsAndConvertToFlatGraph(
        points, quantizer, config, prune_config, std::move(hash_prune));
    log_step("PiPNN final robust pruning", start_final);
    log_step("PiPNN index build finished", total_start);
    return graph;
  } else {
    ABSL_CHECK(on_the_fly_ctx.has_value());
    FinalizeAndPruneGraph(on_the_fly_ctx->graph, *on_the_fly_ctx->points,
                          on_the_fly_ctx->scorer, prune_config,
                          config.robust_prune_chunk_fraction());
    log_step("PiPNN final robust pruning", start_final);
    log_step("PiPNN index build finished", total_start);
    return std::move(on_the_fly_ctx->graph);
  }
}

}  // namespace ginger
