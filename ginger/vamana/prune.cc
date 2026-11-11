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

#include "ginger/vamana/prune.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/internal/portable_random.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/types.h"
#include "ginger/vamana/fixed_size_priority_queue.h"

namespace ginger {

namespace internal {

namespace {

// This computes the next value of alpha that will result in the candidate
// *not* being pruned. `point_distance` is the distance from the candidate to
// the node p being considered for pruning and `limit_distance` is the
// distance from the closest survivor p_prime to the candidate. We need the
// alpha to be larger than the ratio d(p, candidate) / d(p_prime, candidate)
// for the candidate not to be pruned.
float ComputeAlphaTuningPriority(float point_distance, float limit_distance,
                                 const RobustPruneConfig& prune_config) {
  ABSL_QCHECK_NE(limit_distance, 0.0f);
  const float priority = point_distance / limit_distance;

  // For non-positive distances, the alpha required to keep the point becomes
  // d(p_prime, candidate) / d(p, candidate). Inverting the priority ensures
  // it evaluates to this alpha.
  if (prune_config.nonpositive_distances()) {
    return 1.0f / priority;
  } else {
    return priority;
  }
}

}  // namespace

// Given an initial set of `surviving_neighbors` (neighbors that survived the
// initial pass of RobustPrune, with a fixed alpha), the following function adds
// the remaining candidates to `surviving_neighbors` one-by-one until
// `surviving_neighbors.size()` is at least `desired_degree`, or all candidates
// have been added, up to a maximum of `max_degree` total neighbors.
//
// This is done by maintaining a priority queue of candidates that are not yet
// in `surviving_neighbors`. For each such candidate, we calculate the minimum
// value of alpha such that all points in `surviving_neighbors` would fail to
// prune this candidate with this value of alpha (this is the point's
// "priority"). In each iteration we select the candidate with smallest
// priority, add it to `surviving_neighbors`, and update the priorities of other
// candidates in the priority queue.
//
// At the end of this function, if |`surviving_neighbors`| < desired_degree,
// this can only be for two reasons.  Either 1) |`candidates`| < desired_degree,
// or 2) all remaining elements in `candidates` that were not added to
// `surviving_neighbors` have distance 0 to some point in `surviving_neighbors`.
void AlphaTuning(absl::Span<const DistanceAndId> candidates,
                 GingerNodeId node_id, const RobustPruneConfig& prune_config,
                 size_t max_degree,
                 std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
                     compute_distances,
                 std::vector<DistanceAndId>& surviving_neighbors,
                 std::vector<bool>& survived_pruning) {
  const size_t desired_degree = prune_config.desired_degree();
  ABSL_QCHECK_GT(surviving_neighbors.size(), 0);
  ABSL_QCHECK_GE(max_degree, desired_degree);
  if (surviving_neighbors.size() >= desired_degree) return;

  // A min priority queue where priorities correspond to the smallest alpha
  // value that would prune the candidate out. The values associated with each
  // priority are the indices of nodes in `candidates`.
  FixedSizePriorityQueue<float, int, std::less<float>> pq(candidates.size());

  // Compute the initial priority for every candidate.
  // TODO: try optimizing the number of comparisons we do here by only
  // comparing each candidate with the suffix of surviving_points that it did
  // not compare with when it was first pruned.
  for (int i = 0; i < candidates.size(); ++i) {
    if (survived_pruning[i]) continue;
    compute_distances(candidates[i].id, absl::MakeSpan(surviving_neighbors));
    // limit_distance is the closest distance
    float limit_dist =
        std::min_element(surviving_neighbors.begin(), surviving_neighbors.end())
            ->distance;

    // For non-negative distances, distance 0 implies duplicates, so we prune
    // them. For non-positive distances, a candidate with distance 0 to a all
    // survivors would have been picked in the initial pass of RobustPrune. So
    // we can ignore this case here.
    if (limit_dist != 0) {
      pq.InsertOrUpdate(
          i, ComputeAlphaTuningPriority(candidates[i].distance, limit_dist,
                                        prune_config));
    }
  }

  while (surviving_neighbors.size() < desired_degree && !pq.Empty()) {
    int candidate_idx = pq.Top();
    pq.Remove(candidate_idx);
    survived_pruning[candidate_idx] = true;
    const auto& new_survivor = candidates[candidate_idx];
    surviving_neighbors.push_back({0.0f, new_survivor.id});

    // For candidate point newly added to survivors, compute its distances to
    // all candidates still in PQ.
    std::vector<DistanceAndId> candidates_in_pq_dist;
    std::vector<int> indices_in_pq;
    for (int j = 0; j < candidates.size(); ++j) {
      if (pq.Priority(j) != pq.kInvalidPriority) {
        candidates_in_pq_dist.push_back({0.0f, candidates[j].id});
        indices_in_pq.push_back(j);
      }
    }
    if (candidates_in_pq_dist.empty()) continue;
    compute_distances(new_survivor.id, absl::MakeSpan(candidates_in_pq_dist));

    // With new survivor, update priorities of elements still in PQ.
    for (int k = 0; k < indices_in_pq.size(); ++k) {
      int idx_in_pq = indices_in_pq[k];
      float dist_to_new_survivor = candidates_in_pq_dist[k].distance;

      if (dist_to_new_survivor != 0) {
        const float priority = ComputeAlphaTuningPriority(
            candidates[idx_in_pq].distance, dist_to_new_survivor, prune_config);

        if (priority > pq.Priority(idx_in_pq)) {
          pq.InsertOrUpdate(idx_in_pq, priority);
        }
      } else if (!prune_config.nonpositive_distances()) {
        // For non-negative distances, distance 0 means we found a duplicate, so
        // we unambiguously drop it. For non-positive distances, distance 0
        // implies orthogonality. This means that the new survivor cannot prune
        // it under any alpha. The min alpha required to keep the candidate is
        // therefore not changed.
        pq.Remove(idx_in_pq);
      }
    }
  }
}

}  // namespace internal

std::vector<GingerNodeId> RobustPrune(
    absl::Span<DistanceAndId> candidates, GingerNodeId node_id,
    size_t max_degree,
    std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>
        compute_distances,
    const RobustPruneConfig& prune_config) {
  // Remove candidates whose distances do not match the expected sign.
  const bool nonpositive = prune_config.nonpositive_distances();
  auto wrong_sign_it = std::remove_if(
      candidates.begin(), candidates.end(),
      [nonpositive](const DistanceAndId& c) {
        return nonpositive ? (c.distance > 0.0f) : (c.distance < 0.0f);
      });
  candidates.remove_suffix(candidates.end() - wrong_sign_it);

  // Remove duplicates. Note: this assumes that equal ids have equal distances.
  absl::c_sort(candidates);
  auto duplicates_it =
      std::unique(candidates.begin(), candidates.end(),
                  [](const DistanceAndId& a, const DistanceAndId& b) {
                    return a.id == b.id;
                  });
  candidates.remove_suffix(candidates.end() - duplicates_it);

  // Remove self-loops.
  auto self_loop_it =
      std::remove_if(candidates.begin(), candidates.end(),
                     [&](const DistanceAndId& c) { return c.id == node_id; });
  candidates.remove_suffix(candidates.end() - self_loop_it);

  if (candidates.empty()) return {};

  // The alpha parameter controls the pruning threshold. Larger values of alpha
  // result in more edges being kept. If we have negative distances, we need to
  // invert alpha to maintain the same interpretation.
  float alpha = prune_config.alpha();
  ABSL_QCHECK_GE(alpha, 1.0);
  if (prune_config.nonpositive_distances()) {
    alpha = 1.0 / alpha;
  }
  const bool perform_alpha_tuning = prune_config.has_desired_degree();
  const bool keep_pruned = prune_config.keep_pruned_neighbors();

  // `surviving_neighbors` stores pairs of distance and id.
  // The distance field will be used by `compute_distances` to fill
  // `dist(candidate.id, pruning_node.id)` where `pruning_node` is one of the
  // ids that will be emitted in the final result.
  std::vector<DistanceAndId> surviving_neighbors;
  surviving_neighbors.reserve(max_degree);
  std::vector<bool> survived_pruning;
  if (perform_alpha_tuning || keep_pruned) {
    survived_pruning.resize(candidates.size(), false);
  }

  // If we're using non-negative distances and we have too many candidates with
  // distance 0, prune them directly. Keep a random subset of duplicates to
  // ensure the cluster of duplicate points is well connected.
  // TODO: Implement a similar mechanism for negative distances in
  // the case where we know that the vectors are normalized.
  const int max_dupe_candidates = std::clamp<int>(max_degree * 0.25, 2, 20);
  constexpr float kEpsilon = 1e-6f;
  int i = 0;
  if (!prune_config.nonpositive_distances()) {
    while (i < candidates.size() && candidates[i].distance < kEpsilon) {
      ++i;
    }
  }
  if (i > max_dupe_candidates) {
    std::mt19937 rng(node_id);
    internal::portable_random::Shuffle(absl::MakeSpan(candidates).first(i),
                                       rng);
    for (int j = 0; j < max_dupe_candidates; ++j) {
      surviving_neighbors.push_back({0.0f, candidates[j].id});
      if (perform_alpha_tuning || keep_pruned) {
        survived_pruning[j] = true;
      }
    }
  } else {
    for (int j = 0; j < i; ++j) {
      surviving_neighbors.push_back({0.0f, candidates[j].id});
      if (perform_alpha_tuning || keep_pruned) {
        survived_pruning[j] = true;
      }
    }
  }

  // Fill `surviving_neighbors` with all candidates that survive being pruned
  // with the given `alpha`.
  for (; i < candidates.size() && surviving_neighbors.size() < max_degree;
       ++i) {
    const auto& candidate = candidates[i];
    bool pruned = false;

    // Check if this point is pruned by any of the current surviving
    // neighbors.
    if (!surviving_neighbors.empty()) {
      compute_distances(candidate.id, absl::MakeSpan(surviving_neighbors));
      const float threshold = candidate.distance / alpha;
      for (const auto& pruning_node : surviving_neighbors) {
        if (pruning_node.distance < threshold) {
          pruned = true;

          break;
        }
      }
    }

    if (!pruned) {
      surviving_neighbors.push_back({0.0f, candidate.id});
      if (perform_alpha_tuning || keep_pruned) {
        survived_pruning[i] = true;
      }
    }
  }

  if (perform_alpha_tuning) {
    internal::AlphaTuning(candidates, node_id, prune_config, max_degree,
                          compute_distances, surviving_neighbors,
                          survived_pruning);
  }

  if (keep_pruned) {
    for (size_t j = 0;
         j < candidates.size() && surviving_neighbors.size() < max_degree;
         ++j) {
      if (!survived_pruning[j]) {
        surviving_neighbors.push_back({0.0f, candidates[j].id});
      }
    }
  }

  std::vector<GingerNodeId> result(surviving_neighbors.size());
  for (size_t i = 0; i < surviving_neighbors.size(); ++i) {
    result[i] = surviving_neighbors[i].id;
  }
  return result;
}

}  // namespace ginger
