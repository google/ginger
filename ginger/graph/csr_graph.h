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

#ifndef GINGER_GRAPH_CSR_GRAPH_H_
#define GINGER_GRAPH_CSR_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {

// A static directed graph in compressed sparse row (CSR) format.
class CSRGraph {
 public:
  // Builds a graph with `offsets.size() - 1` nodes. The neighbors of node `u`
  // are `targets[offsets[u]]` to `targets[offsets[u + 1] - 1]`. `offsets` must
  // start at 0, be non-decreasing, and end at `targets.size()`. Every entry of
  // `targets` must be a valid node ID.
  CSRGraph(std::vector<int64_t> offsets, std::vector<GingerNodeId> targets)
      : targets_(std::move(targets)), offsets_(std::move(offsets)) {
    ABSL_CHECK(!offsets_.empty());
    ABSL_CHECK_EQ(offsets_.front(), 0);
    ABSL_CHECK_EQ(offsets_.back(), static_cast<int64_t>(targets_.size()));
  }

  GingerNodeId NumNodes() const { return offsets_.size() - 1; }
  int64_t NumArcs() const { return targets_.size(); }

  int64_t Degree(GingerNodeId node_id) const {
    return offsets_[node_id + 1] - offsets_[node_id];
  }

  absl::Span<const GingerNodeId> Neighbors(GingerNodeId node_id) const {
    return absl::MakeSpan(targets_).subspan(offsets_[node_id], Degree(node_id));
  }

  absl::Span<GingerNodeId> Neighbors(GingerNodeId node_id) {
    return absl::MakeSpan(targets_).subspan(offsets_[node_id], Degree(node_id));
  }

  // Prefetches the offset table entry for `node_id`.
  void PrefetchOffsetRange(GingerNodeId node_id) const {
    ABSL_DCHECK_GE(node_id, 0);
    ABSL_DCHECK_LT(node_id, NumNodes());
    absl::PrefetchToLocalCache(&offsets_[node_id]);
  }

  // Resolves the offset range for `node_id` and prefetches the neighbor IDs.
  void PrefetchNeighbors(GingerNodeId node_id) const {
    ABSL_DCHECK_GE(node_id, 0);
    ABSL_DCHECK_LT(node_id, NumNodes());

    const int64_t offset = offsets_[node_id];
    const int64_t end_offset = offsets_[node_id + 1];
    if (offset >= end_offset) return;

    const uintptr_t first_line =
        reinterpret_cast<uintptr_t>(targets_.data() + offset) &
        ~(static_cast<uintptr_t>(ABSL_CACHELINE_SIZE) - 1);
    const uintptr_t last_byte =
        reinterpret_cast<uintptr_t>(targets_.data() + end_offset - 1);

    for (uintptr_t line = first_line; line <= last_byte;
         line += ABSL_CACHELINE_SIZE) {
      absl::PrefetchToLocalCache(reinterpret_cast<const void*>(line));
    }
  }

  size_t NumBytes() const {
    return offsets_.size() * sizeof(int64_t) +
           targets_.size() * sizeof(GingerNodeId);
  }

  bool operator==(const CSRGraph& other) const {
    return offsets_ == other.offsets_ && targets_ == other.targets_;
  }
  bool operator!=(const CSRGraph& other) const { return !(*this == other); }

 private:
  std::vector<GingerNodeId> targets_;
  std::vector<int64_t> offsets_;
};

}  // namespace ginger

#endif  // GINGER_GRAPH_CSR_GRAPH_H_
