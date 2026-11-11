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

#include "ginger/pipnn/hash_prune.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/pipnn/simhash.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"
#include "parlay/parallel.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ginger/pipnn/hash_prune.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace HWY_NAMESPACE {

size_t FindHashSimdImpl(const uint16_t* HWY_RESTRICT hashes, size_t deg,
                        uint16_t target) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<uint16_t> du16;
  const size_t N = hn::Lanes(du16);
  const auto t = hn::Set(du16, target);
  size_t j = 0;
  for (; j + N <= deg; j += N) {
    const auto v = hn::Load(du16, hashes + j);
    const auto eq = hn::Eq(v, t);
    if (!hn::AllFalse(du16, eq)) {
      return j + hn::FindFirstTrue(du16, eq);
    }
  }
  // Scalar tail.
  for (; j < deg; ++j) {
    if (hashes[j] == target) return j;
  }
  return deg;
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {

HWY_EXPORT(FindHashSimdImpl);

size_t HashPrune::FindHash(const uint16_t* hashes, size_t deg,
                           uint16_t target) {
  return HWY_DYNAMIC_DISPATCH(FindHashSimdImpl)(hashes, deg, target);
}

HashPrune::HashPrune(size_t reservoir_size,
                     SimHashProjections simhash_projections)
    : reservoir_size_(reservoir_size),
      num_points_(simhash_projections.num_points()),
      // Pad stride to 32 uint16_t's for SIMD and cache line alignment.
      stride_(hwy::RoundUpTo(reservoir_size, 32)),
      hashes_(hwy::AllocateAligned<uint16_t>(num_points_ * stride_)),
      distances_(hwy::AllocateAligned<Eigen::bfloat16>(num_points_ * stride_)),
      point_ids_(hwy::AllocateAligned<GingerNodeId>(num_points_ * stride_)),
      degrees_(num_points_, 0),
      worst_distances_(num_points_, Eigen::bfloat16(0)),
      worst_indices_(num_points_, 0),
      table_mutexes_(num_points_),
      simhash_projections_(std::move(simhash_projections)) {
  const size_t total_elements = num_points_ * stride_;
  parlay::blocked_for(
      0, total_elements, /*block_size=*/65536,
      [&](size_t /*block_id*/, size_t start, size_t end) {
        const size_t len = end - start;
        std::memset(hashes_.get() + start, 0, len * sizeof(uint16_t));
        std::memset(distances_.get() + start, 0, len * sizeof(Eigen::bfloat16));
        std::memset(point_ids_.get() + start, 0, len * sizeof(GingerNodeId));
      });
}

// Streams in new edges into `source`'s reservoir. A hash bucket has either 0 or
// 1 elemment. Retains the closest element per hash bucket. If the reservoir is
// full, discards the furthest element in the whole reservoir and replaces it
// with the new edge if closer.
// To save memory, this is not a standard hash table with predetermined slots
// for a hash value, but an array that we linearly scan.
void HashPrune::MergeNeighbors(const GingerNodeId source,
                               absl::Span<const DistanceAndId> edges) {
  absl::MutexLock lock(table_mutexes_[source]);

  absl::Span<uint16_t> const hashes = Hashes(source);
  absl::Span<Eigen::bfloat16> const distances = Distances(source);
  absl::Span<GingerNodeId> const neighbor_ids = PointIds(source);

  size_t deg = static_cast<size_t>(degrees_[source]);

  // Cached worst (kept in sync with per-point metadata).
  size_t max_idx = static_cast<size_t>(worst_indices_[source]);
  float max_val = static_cast<float>(worst_distances_[source]);

  auto set_worst = [&](size_t idx, float v) {
    max_idx = idx;
    max_val = v;
  };

  auto recompute_worst = [&]() {
    ABSL_DCHECK_GT(deg, 0);
    size_t worst_idx = 0;
    float worst_dist = static_cast<float>(distances[0]);
    GingerNodeId worst_id = neighbor_ids[0];

    for (size_t i = 1; i < deg; ++i) {
      const float d = static_cast<float>(distances[i]);
      if (d > worst_dist) {
        worst_idx = i;
        worst_dist = d;
        worst_id = neighbor_ids[i];
      } else if (d == worst_dist) {
        // Short-circuit: only read neighbor_ids[i] on distance tie.
        const GingerNodeId id = neighbor_ids[i];
        if (id > worst_id) {
          worst_idx = i;
          worst_id = id;
        }
      }
    }
    set_worst(worst_idx, worst_dist);
  };

  for (const DistanceAndId& e : edges) {
    // Round incoming distance to bfloat16 precision before comparisons to
    // ensure deterministic eviction decisions regardless of edge arrival order.
    const float dist = static_cast<float>(Eigen::bfloat16(e.distance));

    // Skip: this edge can't improve the table. We use explicit short-circuiting
    // comparisons instead of std::tie to avoid reading neighbor_ids[max_idx]
    // from memory unless distances tie.
    if (deg == reservoir_size_ &&
        (dist > max_val ||
         (dist == max_val && e.id >= neighbor_ids[max_idx]))) {
      continue;
    }

    const uint16_t cand_hash = Hash(e.id, source);

    const size_t pos = FindHash(hashes.data(), deg, cand_hash);

    // Hash collision: keep the nearer point. Break distance ties
    // deterministically using node ID. Short-circuit to avoid reading
    // neighbor_ids[pos] unless distances tie.
    if (pos < deg) {
      const float old_dist = static_cast<float>(distances[pos]);
      if (old_dist > dist || (old_dist == dist && e.id < neighbor_ids[pos])) {
        neighbor_ids[pos] = e.id;
        distances[pos] = Eigen::bfloat16(dist);
        // NOMUTANTS -- Only recompute worst if we replaced the
        // current worst.
        if (pos == max_idx) recompute_worst();  // NOLINT
      }
      continue;
    }

    if (deg < reservoir_size_) {
      // Append at end.
      neighbor_ids[deg] = e.id;
      distances[deg] = Eigen::bfloat16(dist);
      hashes[deg] = cand_hash;
      if (deg == 0 || dist > max_val ||
          (dist == max_val && e.id > neighbor_ids[max_idx])) {
        set_worst(deg, dist);
      }
      ++deg;
    } else {
      // Full: replace worst if better (already checked).
      neighbor_ids[max_idx] = e.id;
      distances[max_idx] = Eigen::bfloat16(dist);
      hashes[max_idx] = cand_hash;
      recompute_worst();
    }
  }

  degrees_[source] = static_cast<GingerNodeId>(deg);
  worst_indices_[source] = static_cast<uint16_t>(max_idx);
  worst_distances_[source] = Eigen::bfloat16(max_val);
}

void HashPrune::GetNeighbors(GingerNodeId point_id,
                             std::vector<DistanceAndId>& neighbors) const {
  const size_t deg = static_cast<size_t>(degrees_[point_id]);
  absl::Span<const Eigen::bfloat16> d = Distances(point_id);
  absl::Span<const GingerNodeId> p = PointIds(point_id);
  neighbors.clear();
  neighbors.reserve(deg);
  for (size_t i = 0; i < deg; ++i) {
    neighbors.push_back({static_cast<float>(d[i]), p[i]});
  }
}

std::vector<std::vector<DistanceAndId>> HashPrune::ExportAllNeighbors() const {
  std::vector<std::vector<DistanceAndId>> all_neighbors(num_points_);
  for (size_t i = 0; i < num_points_; ++i) {
    GetNeighbors(static_cast<GingerNodeId>(i), all_neighbors[i]);
  }
  return all_neighbors;
}

HashPrune::Stats HashPrune::GetStats() const {
  Stats stats;
  for (GingerNodeId deg : degrees_) {
    stats.total_edges += deg;
    stats.max_degree = std::max<size_t>(stats.max_degree, deg);
  }
  if (!degrees_.empty()) {
    stats.avg_degree = static_cast<double>(stats.total_edges) / degrees_.size();
  }
  return stats;
}

void HashPrune::LogStats() const {
  const Stats hp_stats = GetStats();
  ABSL_LOG(INFO) << "HashPrune reservoir stats after leaf-building: "
                 << "avg degree = " << hp_stats.avg_degree << " / "
                 << reservoir_size_ << ", max degree = " << hp_stats.max_degree
                 << ", total candidate edges = " << hp_stats.total_edges;
}

}  // namespace ginger
#endif  // HWY_ONCE
