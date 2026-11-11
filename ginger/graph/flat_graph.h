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

#ifndef GINGER_GRAPH_FLAT_GRAPH_H_
#define GINGER_GRAPH_FLAT_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"
#include "parlay/parallel.h"

namespace ginger {

// A graph data structure with a flat memory layout, similar to ParlayANN's
// graph. For each node, we store its degree followed by its neighbors in a
// contiguous memory block of size max_degree + 1.
class FlatGraph {
 public:
  FlatGraph(GingerNodeId num_nodes, GingerNodeId max_degree)
      : num_nodes_(num_nodes), max_degree_(max_degree), data_(nullptr) {
    const size_t total_elements = TotalElements();
    if (total_elements > 0) {
      data_ = hwy::AllocateAligned<GingerNodeId>(total_elements);
      InitData();
    }
  }

  FlatGraph(const FlatGraph& other)
      : num_nodes_(other.num_nodes_),
        max_degree_(other.max_degree_),
        data_(nullptr) {
    const size_t total_elements = TotalElements();
    if (total_elements > 0) {
      data_ = hwy::AllocateAligned<GingerNodeId>(total_elements);
      CopyDataFrom(other);
    }
  }

  FlatGraph& operator=(const FlatGraph& other) {
    if (this != &other) {
      num_nodes_ = other.num_nodes_;
      max_degree_ = other.max_degree_;
      const size_t total_elements = TotalElements();
      if (total_elements > 0) {
        data_ = hwy::AllocateAligned<GingerNodeId>(total_elements);
        CopyDataFrom(other);
      } else {
        data_.reset();
      }
    }
    return *this;
  }

  FlatGraph(FlatGraph&&) noexcept = default;
  FlatGraph& operator=(FlatGraph&&) noexcept = default;

  // Returns the degree of node_id.
  GingerNodeId Degree(GingerNodeId node_id) const {
    return data_[GetStartOfNodeData(node_id)];
  }

  GingerNodeId NumNodes() const { return num_nodes_; }
  GingerNodeId MaxDegree() const { return max_degree_; }

  // Returns a span of the internal data buffer.
  absl::Span<const GingerNodeId> GetData() const {
    return absl::MakeSpan(data_.get(), TotalElements());
  }
  absl::Span<GingerNodeId> MutableData() {
    return absl::MakeSpan(data_.get(), TotalElements());
  }

  // Returns the memory footprint of the graph in bytes.
  size_t NumBytes() const { return TotalElements() * sizeof(GingerNodeId); }

  bool operator==(const FlatGraph& other) const {
    if (num_nodes_ != other.num_nodes_ || max_degree_ != other.max_degree_) {
      return false;
    }
    for (GingerNodeId u = 0; u < num_nodes_; ++u) {
      if (Neighbors(u) != other.Neighbors(u)) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const FlatGraph& other) const { return !(*this == other); }

  absl::Span<const GingerNodeId> Neighbors(GingerNodeId node_id) const {
    return GetData().subspan(GetStartOfNodeData(node_id) + 1, Degree(node_id));
  }

  absl::Span<GingerNodeId> Neighbors(GingerNodeId node_id) {
    return MutableData().subspan(GetStartOfNodeData(node_id) + 1,
                                 Degree(node_id));
  }

  // Prefetches the degree and neighbor IDs for `node_id`.
  void PrefetchNeighbors(GingerNodeId node_id) const {
    ABSL_DCHECK_GE(node_id, 0);
    ABSL_DCHECK_LT(node_id, num_nodes_);

    const GingerNodeId* node_data = data_.get() + GetStartOfNodeData(node_id);
    const uintptr_t first_line =
        reinterpret_cast<uintptr_t>(node_data) &
        ~(static_cast<uintptr_t>(ABSL_CACHELINE_SIZE) - 1);
    const uintptr_t last_byte =
        reinterpret_cast<uintptr_t>(node_data + max_degree_);

    for (uintptr_t line = first_line; line <= last_byte;
         line += ABSL_CACHELINE_SIZE) {
      absl::PrefetchToLocalCache(reinterpret_cast<const void*>(line));
    }
  }

  // Sets the neighbors of `node_id` to the elements in `neighbors`.
  // The degree of `node_id` is set to `neighbors.size()`.
  // Requires `neighbors.size()` to be less than or equal to `max_degree_`.
  void SetNeighbors(GingerNodeId node_id,
                    absl::Span<const GingerNodeId> neighbors) {
    int64_t degree = neighbors.size();
    ABSL_CHECK_LE(degree, max_degree_)
        << "Degree " << degree << " for node " << node_id
        << " exceeds max_degree_ " << max_degree_;
    size_t offset = GetStartOfNodeData(node_id);
    data_[offset] = degree;
    for (size_t i = 0; i < degree; ++i) {
      data_[offset + 1 + i] = neighbors[i];
    }
  }

  // Appends `neighbors` to the existing neighbor list of `node_id`.
  // Requires the resulting degree to be less than or equal to `max_degree_`.
  void AppendNeighbors(GingerNodeId node_id,
                       absl::Span<const GingerNodeId> neighbors) {
    ABSL_DCHECK_LT(node_id, num_nodes_);
    size_t offset = GetStartOfNodeData(node_id);
    GingerNodeId current_degree = data_[offset];
    GingerNodeId new_degree =
        current_degree + static_cast<GingerNodeId>(neighbors.size());
    ABSL_CHECK_LE(new_degree, max_degree_)
        << "New degree " << new_degree << " for node " << node_id
        << " exceeds max_degree_ " << max_degree_;
    absl::c_copy(neighbors, data_.get() + offset + 1 + current_degree);
    data_[offset] = new_degree;
  }

 private:
  // Returns the total number of elements in the internal buffer.
  size_t TotalElements() const {
    return static_cast<size_t>(num_nodes_) *
           (static_cast<size_t>(max_degree_) + 1);
  }

  // Returns the starting index of the block for a given vertex.
  size_t GetStartOfNodeData(GingerNodeId node_id) const {
    return static_cast<size_t>(node_id) * (max_degree_ + 1);
  }

  // Zero-initializes the internal data buffer in parallel.
  void InitData() {
    parlay::blocked_for(
        0, TotalElements(), /*block_size=*/65536,
        [&](size_t /*block_id*/, const size_t start, const size_t end) {
          std::memset(data_.get() + start, 0,
                      (end - start) * sizeof(GingerNodeId));
        });
  }

  // Copies the internal data buffer from another FlatGraph in parallel.
  void CopyDataFrom(const FlatGraph& other) {
    ABSL_DCHECK_EQ(TotalElements(), other.TotalElements());
    parlay::blocked_for(
        0, TotalElements(), /*block_size=*/65536,
        [&](size_t /*block_id*/, const size_t start, const size_t end) {
          std::memcpy(data_.get() + start, other.data_.get() + start,
                      (end - start) * sizeof(GingerNodeId));
        });
  }

  GingerNodeId num_nodes_;
  GingerNodeId max_degree_;
  hwy::AlignedFreeUniquePtr<GingerNodeId[]> data_;
};

}  // namespace ginger

#endif  // GINGER_GRAPH_FLAT_GRAPH_H_
