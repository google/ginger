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

#ifndef GINGER_SEARCH_BEAM_SEARCH_H_
#define GINGER_SEARCH_BEAM_SEARCH_H_

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/search/visited_set.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"

namespace ginger {
namespace internal {

template <typename Graph>
concept HasOffsetRangePrefetch = requires(const Graph& graph, GingerNodeId u) {
  graph.PrefetchOffsetRange(u);
};

template <typename Graph>
concept HasNeighborPrefetch = requires(const Graph& graph, GingerNodeId u) {
  graph.PrefetchNeighbors(u);
};

template <typename Graph>
void PrefetchOffsetRange(const Graph& graph, GingerNodeId u) {
  if constexpr (HasOffsetRangePrefetch<Graph>) {
    graph.PrefetchOffsetRange(u);
  }
}

template <typename Graph>
void PrefetchNeighbors(const Graph& graph, GingerNodeId u) {
  if constexpr (HasNeighborPrefetch<Graph>) {
    graph.PrefetchNeighbors(u);
  }
}

// Merges the beam and the candidates into a new beam. The new beam will contain
// at most `max_beam_size` elements with the lowest distances.
// Requires that `candidates` and `beam` are disjoint, and both are sorted in
// increasing order by distance. All node ids in `candidates` must be >= 0.
// Updates `next_unexplored` if a candidates was placed at an index smaller than
// the current `next_unexplored`.
void MergeBeamAndCandidates(std::vector<BeamEntry>& beam,
                            absl::Span<const DistanceAndId> candidates,
                            size_t max_beam_size, size_t& next_unexplored);

// The state of a beam search.
// The contents of the beam is a collection of (node, distance) entries. The
// beam maintains (at most) top `max_beam_size` entries with the lowest
// distances. Some of the nodes in the beam are marked as explored.
class SearchToken {
 public:
  SearchToken() = default;

  // Resets the search token.
  // Reusing the same SearchToken object across multiple searches helps to avoid
  // memory allocations and initialization time.
  void Reset(int max_beam_size) {
    max_beam_size_ = max_beam_size;
    next_unexplored_ = 0;
    beam_.clear();
    beam_.reserve(max_beam_size);
  }

  // Returns the lowest-distance unexplored node in the beam and its distance
  // without marking it as explored, or {_, -1} if there are no such nodes.
  DistanceAndId PeekNodeToExplore() {
    for (size_t i = next_unexplored_; i < beam_.size(); ++i) {
      if (!beam_[i].WasExplored()) {
        next_unexplored_ = i;
        return {beam_[i].distance, beam_[i].ExtractNodeId()};
      }
    }
    next_unexplored_ = beam_.size();
    return {0, -1};
  }

  // Returns the lowest-distance unexplored node in the beam and its distance,
  // or {_, -1} if there are no such nodes.
  DistanceAndId ExtractNodeToExplore() {
    const DistanceAndId next = PeekNodeToExplore();
    if (next.id != -1) {
      beam_[next_unexplored_++].MarkExplored();
    }
    return next;
  }

  // Adds candidates to the beam. Expects candidates to already be marked seen.
  void AddToBeam(absl::Span<DistanceAndId> candidates) {
    AddToBeam(candidates, [](auto&&) {});
  }

  // Adds candidates to the beam, invoking `before_merge(candidates)` on the
  // sorted surviving candidates prior to merging into the beam.
  template <typename BeforeMergeFn>
  void AddToBeam(absl::Span<DistanceAndId> candidates,
                 BeforeMergeFn&& before_merge) {
    const float threshold = beam_.size() == static_cast<size_t>(max_beam_size_)
                                ? beam_.back().distance
                                : std::numeric_limits<float>::infinity();
    // Remove any neighbors with distance larger than the end of the current
    // beam.
    auto it = std::remove_if(candidates.begin(), candidates.end(),
                             [&](const DistanceAndId& neighbor) {
                               return neighbor.distance >= threshold;
                             });
    candidates = candidates.first(std::distance(candidates.begin(), it));
    if (candidates.empty()) {
      return;
    }

    if (candidates.size() > static_cast<size_t>(max_beam_size_)) {
      std::nth_element(candidates.begin(), candidates.begin() + max_beam_size_,
                       candidates.end());
      candidates = candidates.first(max_beam_size_);
    }
    std::sort(candidates.begin(), candidates.end());

    before_merge(candidates);
    internal::MergeBeamAndCandidates(beam_, candidates, max_beam_size_,
                                     next_unexplored_);
  }

  absl::Span<const BeamEntry> Beam() const {
    return absl::MakeConstSpan(beam_);
  }

  std::vector<DistanceAndId> Result() const {
    std::vector<DistanceAndId> result;
    result.reserve(beam_.size());
    // Place the converged beam into result.
    for (const BeamEntry& entry : beam_) {
      result.push_back(
          {entry.distance, static_cast<int>(entry.ExtractNodeId())});
    }
    return result;
  }

 private:
  // The maximum size of the beam; if we see more neighbors than this size, we
  // will prune the beam to this size.
  int max_beam_size_ = 0;

  // The index of the first candidate in the beam that might be unexplored.
  size_t next_unexplored_ = 0;

  // A vector representing the beam.
  std::vector<BeamEntry> beam_;
};

template <typename VisitedTracker, typename Graph, typename DistanceComparator>
BeamSearchResult NavigationBeamSearchImpl(
    const Graph& graph, DistanceComparator&& compute_distances,
    absl::Span<const GingerNodeId> entry_points, int max_beam_size,
    SearchToken& s, VisitedTracker& tracker, bool emit_explored = false) {
  BeamSearchResult result;
  result.stats.num_candidates_considered = 0;
  result.stats.num_distance_computations = 0;
  result.stats.num_nodes_visited = 0;

  s.Reset(max_beam_size);
  tracker.Reset(max_beam_size, graph.NumNodes());

  // Evaluates distances for `cands` using `compute_distances` and updates
  // search stats. Returns a span of scored candidates, which may be a subspan
  // of `cands` if the scorer filtered candidates out.
  auto score_candidates = [&](absl::Span<DistanceAndId> cands) {
    absl::Span<DistanceAndId> scored;
    if constexpr (std::is_invocable_v<DistanceComparator,
                                      absl::Span<DistanceAndId>,
                                      absl::Span<const BeamEntry>>) {
      scored = compute_distances(cands, s.Beam());
    } else {
      scored = compute_distances(cands);
    }
    result.stats.num_candidates_considered += cands.size();
    result.stats.num_distance_computations += scored.size();
    return scored;
  };

  // For graphs that use an offset-based representation (e.g., CSR), warms up
  // the offset entries for the top surviving candidates so their offset lookups
  // are already in cache if they are explored in upcoming iterations.
  auto prefetch_top_candidates = [&](absl::Span<const DistanceAndId> sorted) {
    if constexpr (HasOffsetRangePrefetch<Graph>) {
      constexpr size_t kNumCandidateOffsetsToPrefetch = 4;
      const size_t num_to_prefetch =
          std::min<size_t>(sorted.size(), kNumCandidateOffsetsToPrefetch);
      for (size_t i = 0; i < num_to_prefetch; ++i) {
        PrefetchOffsetRange(graph, sorted[i].id);
      }
    }
  };

  thread_local hwy::AlignedVector<DistanceAndId> neighbor_candidates;
  neighbor_candidates.clear();
  neighbor_candidates.reserve(entry_points.size());
  tracker.ForEachNotSeenAndMarkSeen(entry_points, [&](GingerNodeId id) {
    neighbor_candidates.push_back({0.0f, id});
  });

  // When there is a single entry point, it is guaranteed to be explored first.
  // Pre-populating its neighbors into the initial candidate pool combines the
  // entry point and its neighbors into a single batch distance evaluation.
  if (entry_points.size() == 1) {
    const GingerNodeId entry_point = entry_points[0];
    tracker.ForEachNotSeenAndMarkSeen(
        graph.Neighbors(entry_point), [&](GingerNodeId neighbor) {
          neighbor_candidates.push_back({0.0f, neighbor});
        });

    absl::Span<DistanceAndId> scored =
        score_candidates(absl::MakeSpan(neighbor_candidates));

    // As we already pre-expanded the entry point's neighbors above, mark it as
    // explored so the main search loop does not re-explore it. The comparator
    // may have filtered out the entry point, either by removing it or by
    // setting its distance to infinity. In that case there is nothing to mark,
    // and its surviving neighbors are already scored.
    if (!scored.empty() && scored[0].id == entry_point &&
        scored[0].distance < std::numeric_limits<float>::infinity()) {
      s.AddToBeam(scored.first(1));
      scored = scored.subspan(1);

      DistanceAndId entry = s.ExtractNodeToExplore();
      ABSL_DCHECK_EQ(entry.id, entry_point);

      ++result.stats.num_nodes_visited;
      if (emit_explored) {
        result.explored.push_back(entry);
      }
    }

    s.AddToBeam(scored, prefetch_top_candidates);
  } else {
    s.AddToBeam(score_candidates(absl::MakeSpan(neighbor_candidates)),
                prefetch_top_candidates);
  }
  neighbor_candidates.clear();

  // As an optimization, we buffer up to this many candidates later in the
  // search before updating the beam and performance distance computations to
  // better amortize their cost.
  constexpr int kMinCandidates = 64;
  const int desired_num_candidates =
      std::min(kMinCandidates, max_beam_size / 8);
  DistanceAndId to_explore;
  while ((to_explore = s.ExtractNodeToExplore()).id != -1) {
    ++result.stats.num_nodes_visited;
    if (emit_explored) {
      result.explored.push_back(to_explore);
    }

    // Hide the memory latency of fetching the adjacency list for the next
    // unexplored beam entry:
    // - If the graph uses an offset-based representation (e.g., CSR),
    //   1. Prefetch its offset entry while decoding `to_explore`'s neighbors
    //      (assumes `Neighbors()` eagerly decodes into a buffer).
    //   2. Resolve the offset and prefetch its adjacency list while filtering
    //      `to_explore`'s neighbors against the visited set.
    // - If the graph uses a flat representation (e.g., FlatGraph), step 1 is a
    //   no-op and step 2 directly prefetches `next_to_explore`'s neighbor block
    //   while filtering `to_explore`'s neighbors against the visited set.
    const DistanceAndId next_to_explore = s.PeekNodeToExplore();
    const bool has_next = next_to_explore.id != -1;
    if (has_next) PrefetchOffsetRange(graph, next_to_explore.id);

    const auto neighbors = graph.Neighbors(to_explore.id);
    if (has_next) PrefetchNeighbors(graph, next_to_explore.id);

    tracker.ForEachNotSeenAndMarkSeen(neighbors, [&](GingerNodeId neighbor) {
      neighbor_candidates.push_back({0.0f, neighbor});
    });

    // skip if we're not at the beginning of the search (visited >= beam_size/2)
    // and don't have enough candidates; or if there's no new candidates.
    if (neighbor_candidates.empty() ||
        (result.stats.num_nodes_visited >= max_beam_size / 2 &&
         neighbor_candidates.size() < desired_num_candidates && has_next)) {
      continue;
    }

    s.AddToBeam(score_candidates(absl::MakeSpan(neighbor_candidates)),
                prefetch_top_candidates);
    neighbor_candidates.clear();
  }

  result.final_beam = s.Result();
  return result;
}

}  // namespace internal

// Performs a beam search on the provided `graph`, using `compute_distances` for
// distance comparisons, starting from the specified `entry_points`. The search
// will maintain at most `max_beam_size` entries with the lowest distances.
//
// Note: Internally, this function uses a thread_local buffer to avoid
// re-allocating vectors and bitsets on every call. This makes navigation
// slightly faster when called repeatedly with the same graph size and
// max_beam_size.
//
// Template Parameters:
// -   `Graph`: An object providing access to graph neighbors. It must implement
//     `Neighbors(GingerNodeId node_id)` which returns an iterable range of
//     neighbor node IDs, and `NumNodes()` which returns the total number of
//     nodes in the graph.
//     Optionally, it may implement `PrefetchOffsetRange(GingerNodeId)` and
//     `PrefetchNeighbors(GingerNodeId)`. If present, beam search calls them for
//     upcoming unexplored nodes to hide memory latency.
// -   `DistanceComparator`: A callable object (e.g., a functor or lambda) that
//     takes `absl::Span<DistanceAndId>` (and optionally
//     `absl::Span<const BeamEntry>`) by value and returns
//     `absl::Span<DistanceAndId>`. It is responsible for computing and updating
//     the `distance` field of each entry in the span by evaluating a distance
//     function, and returning the span.
//     Optionally, the lambda may filter out candidates by returning a
//     smaller span. This should be a subspan of the input (e.g.,
//     `candidates.first(new_size)` after a remove_if). If the comparator
//     accepts a second parameter of type `absl::Span<const BeamEntry>`, the
//     current search beam is passed to allow beam-stateful filtering decisions.
template <typename Graph, typename DistanceComparator>
BeamSearchResult NavigationBeamSearch(
    const Graph& graph, DistanceComparator&& compute_distances,
    absl::Span<const GingerNodeId> entry_points, int max_beam_size,
    bool emit_explored = false) {
  thread_local internal::SearchToken s;
  switch (internal::SelectVisitedStrategy(graph.NumNodes(), max_beam_size)) {
    case internal::VisitedStrategy::kBitvector: {
      thread_local internal::BitvectorVisitedTracker tracker;
      return internal::NavigationBeamSearchImpl(
          graph, std::forward<DistanceComparator>(compute_distances),
          entry_points, max_beam_size, s, tracker, emit_explored);
    }
    case internal::VisitedStrategy::kFastTable: {
      thread_local internal::FastTableVisitedTracker tracker;
      return internal::NavigationBeamSearchImpl(
          graph, std::forward<DistanceComparator>(compute_distances),
          entry_points, max_beam_size, s, tracker, emit_explored);
    }
    case internal::VisitedStrategy::kFlatHashSet: {
      thread_local internal::FlatHashSetVisitedTracker tracker;
      return internal::NavigationBeamSearchImpl(
          graph, std::forward<DistanceComparator>(compute_distances),
          entry_points, max_beam_size, s, tracker, emit_explored);
    }
  }
  ABSL_UNREACHABLE();
}

}  // namespace ginger

#endif  // GINGER_SEARCH_BEAM_SEARCH_H_
