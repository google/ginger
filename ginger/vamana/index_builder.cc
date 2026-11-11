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

#include "ginger/vamana/index_builder.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "ginger/build_progress.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/internal/portable_random.h"
#include "ginger/search/beam_search.h"
#include "ginger/types.h"
#include "ginger/vamana/config.pb.h"
#include "ginger/vamana/prune.h"
#include "ginger/internal/scheduler_reference_stub.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

namespace ginger {

namespace {

std::string FormatWithSuffix(double n) {
  if (n < 1000) {
    return absl::StrFormat("%.1f", n);
  } else if (n < 1e6) {
    return absl::StrFormat("%.1fk", n / 1e3);
  } else if (n < 1e9) {
    return absl::StrFormat("%.1fM", n / 1e6);
  } else if (n < 1e12) {
    return absl::StrFormat("%.1fB", n / 1e9);
  } else {
    return absl::StrFormat("%.1fT", n / 1e12);
  }
}

absl::Status ResolveAndValidateConfig(VamanaConfig& config) {
  if (config.passes().empty()) {
    return absl::InvalidArgumentError("passes must not be empty.");
  }
  if (config.max_degree() <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "max_degree must be positive, but is %d.", config.max_degree()));
  }

  for (int i = 0; i < config.passes_size(); ++i) {
    VamanaConfig::Pass* pass = config.mutable_passes(i);
    if (i > 0) {
      // Use the previous pass's values as a default.
      VamanaConfig::Pass temp_pass = config.passes(i - 1);
      temp_pass.MergeFrom(*pass);
      *pass = temp_pass;
    }
    if (pass->max_beam_width() <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "max_beam_width must be positive in pass %d, but is %d.", i,
          pass->max_beam_width()));
    }
    if (!(pass->prune_config().alpha() >= 1.0f)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "prune_config.alpha must be at least 1.0 in pass %d, but is %f.%s", i,
          pass->prune_config().alpha(),
          pass->prune_config().nonpositive_distances()
              ? " RobustPrune already inverts alpha for nonpositive distances "
                "such as negative dot products, so use alpha >= 1 for them "
                "too."
              : ""));
    }
    if (pass->prune_config().has_desired_degree()) {
      if (pass->prune_config().desired_degree() <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "desired_degree must be positive in pass %d, but is %d.", i,
            pass->prune_config().desired_degree()));
      }
      if (pass->prune_config().desired_degree() > config.max_degree()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "desired_degree must be less than or equal to max_degree in pass "
            "%d, but is %d and max_degree is %d.",
            i, pass->prune_config().desired_degree(), config.max_degree()));
      }
    }
    if (pass->prune_config().reverse_rng_backfill()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "reverse_rng_backfill is not supported in Vamana (pass %d).", i));
    }
  }
  return absl::OkStatus();
}

void LogOrPrintProgress(const IndexBuildProgress& progress,
                        bool log_progress_to_stdout) {
  const double total_work =
      static_cast<double>(progress.num_points) * progress.total_passes;
  if (total_work > 0) {
    double percent_complete =
        100.0 * progress.total_completed_points / total_work;
    absl::Duration remaining = absl::ZeroDuration();
    double avg_items_per_second = 0.0;
    double current_items_per_second = 0.0;
    if (progress.last_batch_duration > absl::ZeroDuration()) {
      current_items_per_second =
          static_cast<double>(progress.last_batch_size) /
          absl::ToDoubleSeconds(progress.last_batch_duration);
    }
    if (progress.total_completed_points > 0) {
      if (progress.elapsed_time > absl::ZeroDuration()) {
        avg_items_per_second = progress.total_completed_points /
                               absl::ToDoubleSeconds(progress.elapsed_time);
        remaining =
            absl::Seconds((total_work - progress.total_completed_points) /
                          avg_items_per_second);
      }
    }
    std::string progress_str = absl::StrFormat(
        "Vamana index build: %5.1f%% | elapsed: %9s | ETA: %9s | "
        "avg throughput: %6s items/s | current throughput: %6s items/s | "
        "pass: %d/%d",
        percent_complete,
        absl::FormatDuration(
            absl::Trunc(progress.elapsed_time, absl::Seconds(1))),
        absl::FormatDuration(absl::Trunc(remaining, absl::Seconds(1))),
        FormatWithSuffix(avg_items_per_second),
        FormatWithSuffix(current_items_per_second), progress.current_pass + 1,
        progress.total_passes);
    if (log_progress_to_stdout) {
      std::cout << "\r" << progress_str << std::flush;
      if (progress.total_completed_points == total_work) {
        std::cout << std::endl;
      }
    } else {
      ABSL_LOG(INFO) << progress_str;
    }
  }
}

// Generates batches of `GingerNodeId`s from `points_to_insert` with exponential
// growth. The first batch has size `clamp(0.02 * existing_nodes, 1,
// max_batch_size)`. Subsequent batches double in size, up to `max_batch_size`.
// Returns a vector of `absl::Span`s, each representing a batch.
std::vector<absl::Span<const GingerNodeId>> GenerateBatches(
    absl::Span<const GingerNodeId> points_to_insert, size_t existing_nodes) {
  std::vector<absl::Span<const GingerNodeId>> batches;
  if (points_to_insert.empty()) {
    return batches;
  }

  const size_t total_nodes = existing_nodes + points_to_insert.size();
  size_t max_batch_size = std::clamp<size_t>(0.02 * total_nodes, 1, 1'000'000);

  size_t target_batch_size =
      std::clamp<size_t>(0.02 * existing_nodes, 1, max_batch_size);
  size_t points_processed = 0;

  while (points_processed < points_to_insert.size()) {
    size_t first = points_processed;
    size_t remaining_points = points_to_insert.size() - points_processed;
    // The actual size of the current batch is limited by the current
    // exponential size, the max_batch_size, and the number of remaining points.
    size_t actual_batch_size =
        std::min({target_batch_size, max_batch_size, remaining_points});

    batches.push_back(points_to_insert.subspan(first, actual_batch_size));
    points_processed += actual_batch_size;

    // Double the target_batch_size for the *next* iteration.
    if (target_batch_size < max_batch_size) {
      target_batch_size *= 2;
    }
  }
  return batches;
}

// Sorts the input `span` in-place. Then, groups consecutive elements that are
// considered equal by `eq_fn`. The `process_group` function is then applied in
// parallel to each group, passed as an `absl::Span`.
template <class T, class EqFn, class ProcessGroup>
void GroupBy(absl::Span<T> span, EqFn&& eq_fn, ProcessGroup&& process_group) {
  parlay::sort_inplace(span);
  auto flags = parlay::delayed_seq<bool>(span.size(), [&](size_t i) {
    return i == 0 || !eq_fn(span[i], span[i - 1]);
  });
  parlay::sequence<size_t> offsets = parlay::pack_index(flags);

  parlay::parallel_for(
      0, offsets.size(),
      [&](size_t i) {
        size_t start = offsets[i];
        size_t end = (i == offsets.size() - 1) ? span.size() : offsets[i + 1];
        process_group(span.subspan(start, end - start));
      },
      1);
}
}  // namespace

namespace internal {

template <typename Graph>
void InsertBatch(absl::Span<const GingerNodeId> batch, Graph& graph,
                 const VamanaConfig& config, size_t pass_index,
                 std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
                     compute_distances) {
  parlay::sequence<std::vector<GingerNodeId>> forward_edges(batch.size());

  const VamanaConfig::Pass& pass = config.passes(pass_index);

  parlay::parallel_for(0, batch.size(), [&](size_t i) {
    // The point to insert.
    GingerNodeId node_id = batch[i];
    // Set the entry point and run the beam search.
    auto distance_from_source = [&](absl::Span<DistanceAndId> neighbors) {
      compute_distances(node_id, neighbors);
      return neighbors;
    };

    auto search_result =
        NavigationBeamSearch(graph, distance_from_source, {kVamanaEntryPoint},
                             config.passes(pass_index).max_beam_width(),
                             /*emit_explored=*/true);

    auto& result = search_result.explored;
    const size_t num_search_results = result.size();
    for (GingerNodeId neighbor : graph.Neighbors(node_id)) {
      result.push_back({0.0f, neighbor});
    }
    compute_distances(node_id,
                      absl::MakeSpan(result).subspan(num_search_results));

    // Prune the result using RobustPrune.
    std::vector<GingerNodeId> neighbors;
    // TODO: note that we could directly fill into
    // result from RobustPrune, and avoid a copy later in this code
    // block.
    if (result.size() > config.max_degree()) {
      neighbors =
          RobustPrune(absl::MakeSpan(result), node_id, config.max_degree(),
                      compute_distances, pass.prune_config());
    } else {
      neighbors.reserve(result.size());
      for (const auto& neighbor : result) {
        if (neighbor.id == node_id) continue;
        neighbors.push_back(neighbor.id);
      }
    }
    forward_edges[i] = std::move(neighbors);
  });

  // Insert the forward edges for the current batch. Note that we do not
  // insert them directly when computing them in the previous parallel loop to
  // ensure a deterministic result. By first computing all forward edges and
  // then inserting them, all parallel computations within this batch operate
  // on the same snapshot of the graph.
  parlay::parallel_for(0, batch.size(), [&](size_t i) {
    graph.SetNeighbors(batch[i], std::move(forward_edges[i]));
  });

  parlay::sequence<std::pair<GingerNodeId, GingerNodeId>> reverse_edges =
      parlay::flatten(parlay::delayed_tabulate(batch.size(), [&](size_t i) {
        const GingerNodeId source_id = batch[i];
        return parlay::delayed_map(
            graph.Neighbors(source_id), [source_id](GingerNodeId neighbor_id) {
              // Reverse edge direction.
              return std::make_pair(neighbor_id, source_id);
            });
      }));

  // Sort the reversed edges, and find the starts of each node's incoming
  // edges. Note that a semisort would suffice, but a fast implementation is
  // not yet available in parlaylib.

  GroupBy(
      absl::MakeSpan(reverse_edges),
      [](const std::pair<GingerNodeId, GingerNodeId>& a,
         const std::pair<GingerNodeId, GingerNodeId>& b) {
        return a.first == b.first;
      },
      [&](absl::Span<std::pair<GingerNodeId, GingerNodeId>> edges) {
        ABSL_CHECK(!edges.empty());
        const GingerNodeId node_id_to_update = edges[0].first;

        // We know that the incoming neighbors don't have duplicates, but they
        // may overlap with our current neighbors (e.g., in passes after the
        // first), so we merge and deduplicate next.
        absl::Span<const GingerNodeId> current_neighbors =
            graph.Neighbors(node_id_to_update);
        absl::flat_hash_set<GingerNodeId> current_neighbors_set(
            current_neighbors.begin(), current_neighbors.end());

        std::vector<DistanceAndId> candidates;
        candidates.reserve(current_neighbors.size() + edges.size());

        // Add current neighbors as DistanceAndId.
        for (GingerNodeId neighbor_id : current_neighbors) {
          candidates.push_back({0.0f, neighbor_id});
        }

        // Add new incoming neighbors, deduplicating.
        for (const auto& [_, incoming_neighbor_id] : edges) {
          if (!current_neighbors_set.contains(incoming_neighbor_id)) {
            candidates.push_back({0.0f, incoming_neighbor_id});
          }
        }

        if (candidates.size() > config.max_degree()) {
          // Compute distances for all candidates.
          compute_distances(node_id_to_update, absl::MakeSpan(candidates));
          std::vector<GingerNodeId> pruned_neighbors = RobustPrune(
              absl::MakeSpan(candidates), node_id_to_update,
              config.max_degree(), compute_distances, pass.prune_config());
          graph.SetNeighbors(node_id_to_update, std::move(pruned_neighbors));
        } else {
          // Extract GingerNodeId from DistanceAndId.
          std::vector<GingerNodeId> final_neighbors;
          final_neighbors.reserve(candidates.size());
          for (const auto& dist_id : candidates) {
            final_neighbors.push_back(dist_id.id);
          }
          graph.SetNeighbors(node_id_to_update, std::move(final_neighbors));
        }
      });
}

void InsertBatch(absl::Span<const GingerNodeId> batch, FlatGraph& graph,
                 const VamanaConfig& config, size_t pass_index,
                 std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
                     compute_distances) {
  InsertBatch<FlatGraph>(batch, graph, config, pass_index,
                         std::move(compute_distances));
}

}  // namespace internal

absl::StatusOr<FlatGraph> BuildVamanaIndex(
    GingerNodeId num_points, VamanaConfig config,
    std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
        compute_distances,
    std::function<void(const IndexBuildProgress&)> progress_callback) {
  return BuildVamanaIndex(num_points, std::move(config),
                          std::move(compute_distances),
                          std::move(progress_callback), [] { return false; });
}

absl::StatusOr<FlatGraph> BuildVamanaIndex(
    GingerNodeId num_points, VamanaConfig config,
    std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
        compute_distances,
    std::function<void(const IndexBuildProgress&)> progress_callback,
    const std::function<bool()>& is_cancelled) {
  ginger::internal::ParallelSchedulerReference scheduler;
  if (num_points <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("num_points must be positive, but is %d.", num_points));
  }
  ABSL_RETURN_IF_ERROR(ResolveAndValidateConfig(config));
  // Allocate a graph with the given max degree.
  FlatGraph graph(num_points, config.max_degree());

  // Generate the batches.
  std::vector<GingerNodeId> points_to_insert(num_points);
  std::iota(points_to_insert.begin(), points_to_insert.end(), 0);
  std::mt19937 rng(config.seed());
  // Note: this ensures that we have a random permutation on all points other
  // than the start point, which is always the first node.
  internal::portable_random::Shuffle(
      absl::MakeSpan(points_to_insert).subspan(1), rng);
  auto batches = GenerateBatches(points_to_insert, /*existing_nodes=*/0);

  ABSL_LOG(INFO) << "Max degree: " << config.max_degree();
  ABSL_LOG(INFO) << "Number of points: " << num_points;
  ABSL_LOG(INFO) << "Number of batches: " << batches.size();

  // Insert each batch into the graph.
  GingerNodeId nodes_processed = 0;
  auto overall_start_time = absl::Now();
  for (size_t pass_index = 0; pass_index < config.passes_size(); ++pass_index) {
    for (const auto& batch : batches) {
      if (is_cancelled()) return absl::CancelledError();

      absl::Time batch_start = absl::Now();
      internal::InsertBatch(batch, graph, config, pass_index,
                            compute_distances);
      absl::Time batch_end = absl::Now();

      nodes_processed += batch.size();
      absl::Duration elapsed = absl::Now() - overall_start_time;

      IndexBuildProgress progress;
      progress.num_points = num_points;
      progress.total_passes = config.passes_size();
      progress.current_pass = pass_index;
      progress.total_completed_points = nodes_processed;
      progress.elapsed_time = elapsed;
      progress.last_batch_duration = batch_end - batch_start;
      progress.last_batch_size = batch.size();

      LogOrPrintProgress(progress, config.log_progress_to_stdout());

      if (progress_callback != nullptr) {
        progress_callback(progress);
      }
    }
  }

  return graph;
}

}  // namespace ginger
