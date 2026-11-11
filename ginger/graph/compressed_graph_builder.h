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

#ifndef GINGER_GRAPH_COMPRESSED_GRAPH_BUILDER_H_
#define GINGER_GRAPH_COMPRESSED_GRAPH_BUILDER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/slice.h"

namespace ginger {
namespace internal {

template <typename Config>
size_t CountIntervals(absl::Span<const GingerNodeId> neighbors) {
  constexpr size_t kIntervalThreshold = Config::kIntervalThreshold;
  if (neighbors.size() < kIntervalThreshold) {
    return 0;
  }

  size_t num_intervals = 0;

  size_t current_length = 1;
  for (size_t i = 1; i < neighbors.size(); ++i) {
    if (neighbors[i] == neighbors[i - 1] + 1) {
      ++current_length;
    } else {
      if (current_length >= kIntervalThreshold) {
        ++num_intervals;
      }

      current_length = 1;
    }
  }

  if (current_length >= kIntervalThreshold) {
    ++num_intervals;
  }

  return num_intervals;
}

// Writes the interval encoding of a sorted list of node IDs to the provided
// encoder. As a side effect, this function moves all elements not part of an
// encoded interval to the front of `sorted_neighbors` and returns the number
// of such elements.
template <typename Config, typename Encoder>
GingerNodeId EncodeIntervals(Encoder& encoder,
                             absl::Span<GingerNodeId> sorted_neighbors) {
  constexpr size_t kIntervalThreshold = Config::kIntervalThreshold;
  static_assert(kIntervalThreshold > 1);

  ABSL_DCHECK_GE(sorted_neighbors.size(), kIntervalThreshold);
  ABSL_DCHECK(std::is_sorted(sorted_neighbors.begin(), sorted_neighbors.end()));
  ABSL_DCHECK(std::all_of(sorted_neighbors.begin(), sorted_neighbors.end(),
                          [](const GingerNodeId value) { return value >= 0; }));

  // While traversing the neighborhood to identify intervals, we move all gaps
  // (elements not part of an encoded interval) to the front of the
  // `sorted_neighbors` span. This allows the subsequent `EncodeGaps` routine
  // to simply process the prefix of `sorted_neighbors` without needing to
  // re-identify the intervals. `cur_gap_offset` tracks the current index
  // where the next gap should be written. It is incremented for each neighbor
  // and decremented whenever an interval is encoded.
  size_t cur_gap_offset = 1;

  size_t cur_interval_length = 1;

  GingerNodeId prev_right_extreme = 0;
  GingerNodeId cur_left_extreme = sorted_neighbors[0];

  const auto encode_interval = [&] {
    const GingerNodeId delta_left_extreme =
        cur_left_extreme - prev_right_extreme;
    const GingerNodeId delta_length = cur_interval_length - kIntervalThreshold;

    encoder.Encode(delta_left_extreme);
    encoder.Encode(delta_length);

    prev_right_extreme = cur_left_extreme + (cur_interval_length - 1);
    cur_gap_offset -= cur_interval_length;
  };

  GingerNodeId prev_neighbor = sorted_neighbors[0];
  for (size_t i = 1; i < sorted_neighbors.size(); ++i) {
    const GingerNodeId neighbor = sorted_neighbors[i];

    if (neighbor == prev_neighbor + 1) {
      ++cur_interval_length;
    } else {
      if (cur_interval_length >= kIntervalThreshold) {
        encode_interval();
      }

      cur_left_extreme = sorted_neighbors[i];
      cur_interval_length = 1;
    }

    sorted_neighbors[cur_gap_offset++] = neighbor;
    prev_neighbor = neighbor;
  }

  if (cur_interval_length >= kIntervalThreshold) {
    encode_interval();
  }

  return cur_gap_offset;
}

// Writes the gap encoding of a sorted list of node IDs to the provided encoder.
template <typename Config, typename Encoder>
void EncodeGaps(Encoder& encoder, absl::Span<GingerNodeId> sorted_neighbors) {
  ABSL_DCHECK(std::is_sorted(sorted_neighbors.begin(), sorted_neighbors.end()));
  ABSL_DCHECK(std::all_of(sorted_neighbors.begin(), sorted_neighbors.end(),
                          [](const GingerNodeId value) { return value >= 0; }));

  // Transform the sorted list of neighbors into a list of gaps.
  GingerNodeId prev_neighbor = 0;
  for (size_t i = 0; i < sorted_neighbors.size(); ++i) {
    const GingerNodeId neighbor = sorted_neighbors[i];
    sorted_neighbors[i] = neighbor - prev_neighbor;
    prev_neighbor = neighbor;
  }

  constexpr bool kUseBatchGapDecoding = Config::kUseBatchGapDecoding;
  using BatchGapEncoder = Config::BatchGapEncoder;

  if constexpr (kUseBatchGapDecoding) {
    encoder.Flush();
    const size_t num_bytes_written =
        BatchGapEncoder::Encode(sorted_neighbors, encoder.Remaining(),
                                encoder.DryRun())
            .value();
    encoder.Skip(num_bytes_written);
  } else {
    for (const GingerNodeId gap : sorted_neighbors) {
      encoder.Encode(gap);
    }
  }
}

// Encodes the neighbors of the given node using the provided `encoder`. The
// encoding scheme depends on the `Config` template parameter. Note that
// `sorted_neighbors` must contains the neighbors of the given node and has to
// be sorted in ascending order.
template <typename Config, typename Graph, typename Encoder>
void EncodeNeighbors(const Graph& graph, const GingerNodeId node_id,
                     Encoder& encoder,
                     absl::Span<GingerNodeId> sorted_neighbors) {
  ABSL_DCHECK(std::is_sorted(sorted_neighbors.begin(), sorted_neighbors.end()));
  ABSL_DCHECK(std::all_of(sorted_neighbors.begin(), sorted_neighbors.end(),
                          [](const GingerNodeId value) { return value >= 0; }));

  if constexpr (Config::kUseIntervalEncoding) {
    const size_t num_intervals = CountIntervals<Config>(sorted_neighbors);
    const GingerNodeId degree = graph.Degree(node_id);

    // Ensure that the left shift does not cause loss of the MSB.
    ABSL_CHECK_LE(degree, std::numeric_limits<GingerNodeId>::max() >> 1);
    const GingerNodeId tagged_degree =
        (degree << 1) | (num_intervals > 0 ? 1 : 0);
    encoder.Encode(tagged_degree);

    GingerNodeId num_remaining_nodes = degree;
    if (num_intervals > 0) {
      encoder.Encode(num_intervals);
      num_remaining_nodes = EncodeIntervals<Config>(encoder, sorted_neighbors);
    }

    // If intervals are present, `EncodeIntervals` stably moves the remaining
    // gaps to the front of `sorted_neighbors`. We thus know that the
    // remaining nodes are at the beginning of the `sorted_neighbors` array.
    absl::Span<GingerNodeId> remaining_nodes =
        absl::MakeSpan(sorted_neighbors).first(num_remaining_nodes);

    EncodeGaps<Config>(encoder, remaining_nodes);
  } else {
    encoder.Encode(graph.Degree(node_id));
    EncodeGaps<Config>(encoder, sorted_neighbors);
  }
}

inline std::vector<GingerNodeId>& BuilderScratchBuffer() {
  thread_local std::vector<GingerNodeId> buffer;
  return buffer;
}

template <typename Config, typename Graph>
std::vector<int64_t> ComputeCompressedOffsets(const Graph& graph) {
  // We compute the offsets in two steps:
  // 1) In parallel, for each node `u`, we compute the size of its encoded
  //    neighborhood. This size is written to `offsets[u]`.
  // 2) A parallel prefix sum (scan) is performed on the `offsets` array. After
  //    the scan, `offsets[u]` will contain the starting offset for node `u`'s
  //    compressed neighborhood in the `compressed_targets` array.
  std::vector<int64_t> offsets(graph.NumNodes() + 1, 0);

  parlay::parallel_for(0, graph.NumNodes(), [&](const GingerNodeId node_id) {
    std::vector<GingerNodeId>& sorted_neighbors_buffer = BuilderScratchBuffer();
    // We use a dry-run encoder to compute the size of the compressed
    // representation for each node's neighborhood without writing any data.
    using VariableLengthEncoder = Config::VariableLengthEncoder;
    VariableLengthEncoder encoder(/*output_buffer=*/{}, /*dry_run=*/true);

    const absl::Span<const GingerNodeId> neighbors = graph.Neighbors(node_id);
    sorted_neighbors_buffer.assign(neighbors.begin(), neighbors.end());
    std::sort(sorted_neighbors_buffer.begin(), sorted_neighbors_buffer.end());

    EncodeNeighbors<Config>(graph, node_id, encoder,
                            absl::MakeSpan(sorted_neighbors_buffer));
    encoder.Flush();

    offsets[node_id] = encoder.CurOffset();
  });

  parlay::scan_inplace(parlay::make_slice(offsets));

  return offsets;
}

template <typename Config, typename Graph>
std::vector<uint8_t> ComputeCompressedTargets(
    const Graph& graph, absl::Span<const int64_t> offsets) {
  using VariableLengthEncoder = Config::VariableLengthEncoder;

  const size_t compressed_targets_size =
      offsets.back() + CompressedCSRGraph<Config>::kSlopBytes;
  std::vector<uint8_t> compressed_targets(compressed_targets_size);

  parlay::parallel_for(0, graph.NumNodes(), [&](const GingerNodeId node_id) {
    std::vector<GingerNodeId>& sorted_neighbors_buffer = BuilderScratchBuffer();
    const size_t neighborhood_size = offsets[node_id + 1] - offsets[node_id];
    const absl::Span<uint8_t> compressed_neighborhood = absl::MakeSpan(
        compressed_targets.data() + offsets[node_id], neighborhood_size);
    VariableLengthEncoder encoder(compressed_neighborhood);

    const absl::Span<const GingerNodeId> neighbors = graph.Neighbors(node_id);
    sorted_neighbors_buffer.assign(neighbors.begin(), neighbors.end());
    std::sort(sorted_neighbors_buffer.begin(), sorted_neighbors_buffer.end());

    EncodeNeighbors<Config>(graph, node_id, encoder,
                            absl::MakeSpan(sorted_neighbors_buffer));
    encoder.Flush();
  });

  return compressed_targets;
}

}  // namespace internal

// Constructs a compressed graph from a graph stored in a format, which provides
// `NumNodes()`, `Degree(node)`, and `Neighbors(node)` methods. The template
// parameter `Config` contains the settings to use for graph compression.
template <typename Graph, typename Config = CompressedGraphConfig>
CompressedCSRGraph<Config> CompressGraph(const Graph& graph) {
  // We construct the graph in two passes. The first pass is used to determine
  // the sizes of the compressed target array and compute offsets into it.
  std::vector<int64_t> offsets =
      internal::ComputeCompressedOffsets<Config>(graph);

  // The second pass computes the compressed neighborhoods and writes the
  // compressed data into the targets array using the pre-computed offsets.
  std::vector<uint8_t> compressed_targets =
      internal::ComputeCompressedTargets<Config>(graph, offsets);

  const int64_t num_arcs = parlay::reduce(parlay::delayed_tabulate(
      graph.NumNodes(),
      [&](GingerNodeId node) -> int64_t { return graph.Degree(node); }));

  return CompressedCSRGraph<Config>(std::move(offsets),
                                    std::move(compressed_targets), num_arcs);
}

}  // namespace ginger

#endif  // GINGER_GRAPH_COMPRESSED_GRAPH_BUILDER_H_
