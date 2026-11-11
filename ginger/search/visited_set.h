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

#ifndef GINGER_SEARCH_VISITED_SET_H_
#define GINGER_SEARCH_VISITED_SET_H_

#include <cstddef>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "ginger/search/fast_visited_table.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"

namespace ginger {
namespace internal {

// Strategy used to track visited nodes during beam search.
// - kBitvector: Uses a dense bitvector. Best for smaller graphs (<2.5M nodes).
// - kFastTable: Uses a custom linear probing table with generation counter.
// Best for small/medium beam sizes (<800).
// - kFlatHashSet: Uses absl::flat_hash_set. Best for large beam sizes (>=800).
enum class VisitedStrategy { kBitvector, kFastTable, kFlatHashSet };

inline constexpr int kBitvectorStrategyMaxNodes = 2'500'000;
inline constexpr int kFastTableMaxBeamSize = 800;

constexpr VisitedStrategy SelectVisitedStrategy(int num_nodes,
                                                int max_beam_size) {
  if (num_nodes >= 0 && num_nodes < kBitvectorStrategyMaxNodes) {
    return VisitedStrategy::kBitvector;
  }
  if (max_beam_size < kFastTableMaxBeamSize) {
    return VisitedStrategy::kFastTable;
  }
  return VisitedStrategy::kFlatHashSet;
}

// Tracks visited nodes during beam search using a dense bitvector.
// Best for smaller graphs (<2.5M nodes).
class BitvectorVisitedTracker {
 public:
  void Reset(int /*max_beam_size*/, int num_nodes) {
    if (was_seen_bits_.size() != num_nodes) {
      was_seen_bits_.assign(num_nodes, false);
      seen_nodes_.clear();
    } else {
      for (GingerNodeId node_id : seen_nodes_) {
        was_seen_bits_[node_id] = false;
      }
      seen_nodes_.clear();
    }
  }

  template <typename Range, typename Func>
  void ForEachNotSeenAndMarkSeen(const Range& range, Func&& func) {
    for (GingerNodeId node_id : range) {
      ABSL_DCHECK_LT(node_id, was_seen_bits_.size());
      if (!was_seen_bits_[node_id]) {
        was_seen_bits_[node_id] = true;
        seen_nodes_.push_back(node_id);
        func(node_id);
      }
    }
  }

 private:
  std::vector<bool> was_seen_bits_;
  hwy::AlignedVector<GingerNodeId> seen_nodes_;
};

// Tracks visited nodes using a custom linear probing table with a generation
// counter. Best for small/medium beam sizes (<800).
class FastTableVisitedTracker {
 public:
  static constexpr size_t ComputeVisitedTableCapacity(int max_beam_size) {
    if (max_beam_size >= 800) {
      return 1048576;  // 2^20
    }
    if (max_beam_size >= 400) {
      return 262144;  // 2^18
    }
    if (max_beam_size >= 200) {
      return 65536;  // 2^16
    }
    return FastVisitedTable::kDefaultCapacity;
  }

  void Reset(int max_beam_size, int /*num_nodes*/) {
    fast_was_seen_set_.Reset(ComputeVisitedTableCapacity(max_beam_size));
  }

  template <typename Range, typename Func>
  void ForEachNotSeenAndMarkSeen(const Range& range, Func&& func) {
    for (GingerNodeId node_id : range) {
      if (fast_was_seen_set_.Insert(node_id)) {
        func(node_id);
      }
    }
  }

 private:
  FastVisitedTable fast_was_seen_set_;
};

// Tracks visited nodes using absl::flat_hash_set. Best for large beam
// sizes (>=800).
class FlatHashSetVisitedTracker {
 public:
  void Reset(int /*max_beam_size*/, int /*num_nodes*/) {
    flat_was_seen_set_.clear();
  }

  template <typename Range, typename Func>
  void ForEachNotSeenAndMarkSeen(const Range& range, Func&& func) {
    for (GingerNodeId node_id : range) {
      if (flat_was_seen_set_.insert(node_id).second) {
        func(node_id);
      }
    }
  }

 private:
  absl::flat_hash_set<GingerNodeId> flat_was_seen_set_;
};

}  // namespace internal
}  // namespace ginger

#endif  // GINGER_SEARCH_VISITED_SET_H_
