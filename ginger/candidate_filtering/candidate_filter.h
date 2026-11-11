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

#ifndef GINGER_CANDIDATE_FILTERING_CANDIDATE_FILTER_H_
#define GINGER_CANDIDATE_FILTERING_CANDIDATE_FILTER_H_

#include <algorithm>
#include <cstddef>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {

struct NoFilter {
  absl::Span<DistanceAndId> FilterCandidates(
      absl::Span<DistanceAndId> candidates,
      absl::Span<const BeamEntry> /*beam*/ = {}) const {
    return candidates;
  }
};

// BeamMeanThresholdPolicy computes a threshold for filtering candidates, that
// is the running mean of the tail `width` elements of the beam, using their
// approximate distances (e.g., JL or TurboSlice). We apply the filtering only
// once the beam has reached `min_beam_size`.
// As an optimization, we only recompute the tail distances if the beam has
// changed since the last update, otherwise we reuse the previous mean of the
// tail distances to update the running mean.
class BeamMeanThresholdPolicy {
 public:
  // Args:
  //   min_beam_size: The minimum beam size required before filtering is enabled
  //     (typically set to the target operational beam size).
  //   width: The number of elements at the tail of the beam to average when
  //     computing the threshold (typically 5 in practice).
  explicit BeamMeanThresholdPolicy(size_t min_beam_size, size_t width)
      : min_beam_size_(min_beam_size), width_(width) {}

  // Updates the running threshold based on the current beam before candidate
  // filtering takes place. If the beam size has reached `min_beam_size` and the
  // beam has changed since the last update, it recomputes the distances for the
  // tail elements in the beam to update the mean threshold.
  // Returns the number of extra distance computations performed during the
  // update (0 if the beam hasn't changed or is smaller than `min_beam_size`).
  template <typename ComputeFilterDistancesFunc>
  int UpdateBeforeFiltering(absl::Span<const BeamEntry> beam,
                            ComputeFilterDistancesFunc& compute_distances) {
    if (beam.empty() || beam.size() < min_beam_size_) return 0;

    int extra_computations = 0;
    const GingerNodeId current_back_id = beam.back().ExtractNodeId();

    // Recompute tail distances if this is the first time or if the beam
    // changed. Checking both beam size and the back element's ID is sufficient
    // to detect any change to the beam: if the beam is not full, any insertion
    // increases its size; if the beam is full, any valid insertion evicts the
    // worst element, changing beam.back().
    if (threshold_count_ == 0 || last_beam_back_id_ != current_back_id ||
        last_beam_size_ != beam.size()) {
      const size_t actual_width = std::min(width_, beam.size());
      if (actual_width <= 0) return 0;

      tail_nodes_.clear();
      for (const auto& entry : beam.last(actual_width)) {
        tail_nodes_.push_back({0.0f, entry.ExtractNodeId()});
      }

      absl::Span<DistanceAndId> tail_span = absl::MakeSpan(tail_nodes_);
      compute_distances(tail_span);
      extra_computations = tail_span.size();
      if (tail_span.empty()) return 0;

      float tail_sum = 0.0f;
      for (const auto& entry : tail_span) {
        tail_sum += entry.distance;
      }
      last_tail_mean_ = tail_sum / tail_span.size();
      last_beam_back_id_ = current_back_id;
      last_beam_size_ = beam.size();
    }

    threshold_sum_ += last_tail_mean_;
    ++threshold_count_;
    threshold_ = threshold_sum_ / threshold_count_;
    return extra_computations;
  }

  bool ShouldFilter() const { return threshold_count_ > 0; }

  float threshold() const { return threshold_; }

 private:
  size_t min_beam_size_;
  size_t width_;
  // Number of times the threshold has been updated.
  int threshold_count_ = 0;
  // The node ID at the back of the beam during the last update.
  GingerNodeId last_beam_back_id_ = -1;
  // The size of the beam during the last update.
  size_t last_beam_size_ = 0;
  // The cached mean distance of the tail nodes from the last computation.
  float last_tail_mean_ = 0.0f;
  // Running sum of tail means to compute the average threshold over time.
  float threshold_sum_ = 0.0f;
  // The current running average threshold.
  float threshold_ = 0.0f;
  // Buffer to hold tail nodes for distance computation.
  absl::InlinedVector<DistanceAndId, 8> tail_nodes_;
};

// CandidateFilter provides approximate-distance-based candidate filtering for
// beam search. It computes approximate distances, determines which candidates
// survive according to a given `ThresholdPolicy`, and updates the policy based
// on the survivors.
//
// This filter is stateful and specific to a single query. A new CandidateFilter
// instance must be constructed for each new beam search call.
template <typename ComputeDistancesFunc, typename ThresholdPolicy>
class CandidateFilter {
 public:
  CandidateFilter(ComputeDistancesFunc compute_distances,
                  ThresholdPolicy policy)
      : compute_distances_(std::move(compute_distances)),
        policy_(std::move(policy)) {}

  // Convenience operator() to allow CandidateFilter to be used directly as a
  // DistanceComparator in NavigationBeamSearch.
  absl::Span<DistanceAndId> operator()(absl::Span<DistanceAndId> candidates,
                                       absl::Span<const BeamEntry> beam = {}) {
    return FilterCandidates(candidates, beam);
  }

  // Filters candidates using approximate distances. Candidates with approximate
  // distance >= the current threshold are removed via in-place compaction.
  // Returns a subspan of `candidates` containing only the survivors.
  absl::Span<DistanceAndId> FilterCandidates(
      absl::Span<DistanceAndId> candidates,
      absl::Span<const BeamEntry> beam = {}) {
    if (candidates.empty()) return candidates;

    num_filter_computations_ +=
        policy_.UpdateBeforeFiltering(beam, compute_distances_);
    if (!policy_.ShouldFilter()) {
      return candidates;
    }

    num_filter_computations_ += candidates.size();
    compute_distances_(candidates);

    float threshold = policy_.threshold();
    auto it = std::remove_if(candidates.begin(), candidates.end(),
                             [threshold](const DistanceAndId& cand) {
                               return cand.distance >= threshold;
                             });
    return candidates.first(it - candidates.begin());
  }

  const ThresholdPolicy& policy() const { return policy_; }

  size_t num_filter_computations() const { return num_filter_computations_; }

 private:
  ComputeDistancesFunc compute_distances_;
  ThresholdPolicy policy_;
  size_t num_filter_computations_ = 0;
};

// Explicit deduction guide.
template <typename ComputeDistancesFunc, typename ThresholdPolicy>
CandidateFilter(ComputeDistancesFunc, ThresholdPolicy)
    -> CandidateFilter<ComputeDistancesFunc, ThresholdPolicy>;

}  // namespace ginger

#endif  // GINGER_CANDIDATE_FILTERING_CANDIDATE_FILTER_H_
