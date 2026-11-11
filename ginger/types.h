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

#ifndef GINGER_TYPES_H_
#define GINGER_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/str_format.h"

namespace ginger {

using GingerNodeId = int32_t;

// The underlying data type representing vector coordinate values in datasets
// and queries (32-bit floats, signed 8-bit ints, or unsigned 8-bit ints).
enum class CoordinateType {
  kFloat,
  kInt8,
  kUint8,
};

// Stores a distance and a node ID. We use negative numbers to indicate
// uninitialized values.
struct DistanceAndId {
  float distance = -1.0;
  GingerNodeId id = -1;

  bool operator<(const DistanceAndId& other) const {
    return std::tie(distance, id) < std::tie(other.distance, other.id);
  }

  bool operator==(const DistanceAndId& other) const {
    return distance == other.distance && id == other.id;
  }

  // This makes unit test errors much more readable.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DistanceAndId& d) {
    absl::Format(&sink, "{distance: %f, id: %d}", d.distance, d.id);
  }
};

// BeamEntry stores a triple of a (distance, node_id, explored) in the beam.
// We use a separate struct instead of a triple to enable the following
// optimization: we steal a bit from the node_id's to mark nodes as already
// explored or not (i.e., we have fetched and scored all of its neighbors in the
// graph).
//
// Note that this optimization means we can index at most 2^31 - 1 nodes on a
// single machine when GingerNodeId is int32_t.
struct BeamEntry {
 public:
  float distance;

  // Default constructor.
  BeamEntry() : distance(0.0f), node_id_with_explored_bit(0) {}

  // The node_id is initially marked as not explored.
  BeamEntry(float distance, GingerNodeId node_id)
      : distance(distance), node_id_with_explored_bit(node_id) {}

  // Copy constructor from DistanceAndId. The explored bit is cleared.
  explicit BeamEntry(const DistanceAndId& other)
      : distance(other.distance), node_id_with_explored_bit(other.id) {
    ABSL_CHECK_GE(other.id, 0);
  }

  // Assignment operator from DistanceAndId. The explored bit is cleared.
  BeamEntry& operator=(const DistanceAndId& other) {
    ABSL_CHECK_GE(other.id, 0);
    distance = other.distance;
    node_id_with_explored_bit = other.id;
    return *this;
  }

  bool operator<(const BeamEntry& other) const {
    return distance < other.distance;
  }

  bool WasExplored() const {
    // use top bit as marker
    return node_id_with_explored_bit & kExploredBitMask;
  }

  void MarkExplored() {
    // use top bit as marker
    node_id_with_explored_bit |= kExploredBitMask;
  }

  // Returns the node_id ignoring the explored bit.
  GingerNodeId ExtractNodeId() const {
    return node_id_with_explored_bit & (~kExploredBitMask);
  }

 private:
  GingerNodeId node_id_with_explored_bit;
  static constexpr GingerNodeId kExploredBitMask =
      static_cast<GingerNodeId>(1) << ((sizeof(GingerNodeId) * 8) - 1);
};

// Stores the result of a beam search, including the final beam and search
// statistics.
struct BeamSearchResult {
  std::vector<DistanceAndId> final_beam;

  // Nodes explored during the search. Only populated when emit_explored is
  // true.
  std::vector<DistanceAndId> explored;

  struct Statistics {
    // Number of graph nodes visited and expanded during beam search.
    size_t num_nodes_visited = 0;

    // Number of candidate nodes examined for potential beam insertion.
    size_t num_candidates_considered = 0;

    // Number of distance computations performed by the primary navigation
    // distance scorer during beam search.
    size_t num_distance_computations = 0;

    // Number of approximate distance evaluations performed by candidate
    // filtering policies (e.g., TurboSlice or JL filtering).
    size_t num_filter_computations = 0;

    // Number of candidate nodes rescored during post-search reranking.
    size_t num_reranked = 0;
  };

  Statistics stats;
};

}  // namespace ginger

#endif  // GINGER_TYPES_H_
