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

#ifndef GINGER_GRAPH_COMPRESSED_GRAPH_H_
#define GINGER_GRAPH_COMPRESSED_GRAPH_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/graph/compressed_offsets.h"
#include "ginger/graph/compression/adaptive_gap_codec.h"
#include "ginger/graph/compression/varint_codec.h"
#include "ginger/types.h"

namespace ginger {

// This header contains a compressed graph data structure optimized for space
// efficiency. The `compressed_graph_builder.h` header contains an interface to
// construct compressed graphs from uncompressed graphs. To construct a
// compressed graph, you can call the interface as follows:
//
// ```cpp
// const FlatGraph graph = ...;
// CompressedCSRGraph<> compressed_graph = CompressGraph(graph);
// ```
//
// The format of the compressed graph data structure is based on the Compressed
// Sparse Row (CSR) representation, consisting of an offsets array and a
// compressed targets array. The offsets array stores the starting position for
// each node's neighbor list in the targets array, pointing to the first byte of
// encoded data.
//
// All data within a neighborhood is encoded using a variable-length codec. For
// improved compression, gap encoding is used for neighbor IDs: neighbor lists
// are sorted by ID, and the differences (gaps) between consecutive IDs are
// stored instead of absolute values. Since gaps are generally smaller than the
// original IDs, the variable-length codec can represent them using less space.
//
// Additionally, interval encoding can be optionally used for further
// compression. It identifies consecutive runs of neighbor IDs in the sorted
// neighbor list and stores them as intervals, represented by a starting ID
// (left extreme) and a length. Neighbors within these intervals are not
// gap-encoded.
//
// The layout for each compressed neighborhood starts with the node's degree. If
// interval encoding is enabled, the degree's least significant bit (LSB) acts
// as a flag to indicate the presence of interval data, and the actual degree is
// found by right-shifting. If intervals are present, the interval count is
// stored next, followed by a sequence of (left extreme, length) pairs for each
// interval. Finally, any remaining neighbors not covered by intervals are
// stored using gap encoding. Note that all interval data for a neighborhood is
// written before the gap data, which means that the two are not interleaved.
//
// The following is an example layout with the configuration set to:
// (kUseIntervalEncoding = true, kIntervalThreshold = 3):
//
// Suppose a node has neighbors {1, 2, 3, 4, 8, 9, 10, 15, 19}
// - Degree: 9. Since intervals are present, the least significant bit is set,
//              so the encoded degree is (9 << 1) | 1 = 19
// - Intervals:
//     -   {1, 2, 3, 4}: Left Extreme = 1, Length = 4
//     -   {8, 9, 10}: Left Extreme = 8, Length = 3
// - Gaps: {15, 19}
//
// The encoded neighborhood data would be:
// 1. (Tagged) Degree: 19
// 2. Number of Intervals: 2
// 3. Interval 1:
//     -   Delta Left Extreme: 1 (since initial right extreme is 0)
//     -   Delta Length: 4 - kIntervalThreshold = 4 - 3 = 1
//     Encoded: {1, 1}
// 4.  Interval 2:
//     -   Delta Left Extreme: 8 - 4 = 4 (previous right extreme was 4)
//     -   Delta Length: 3 - kIntervalThreshold = 3 - 3 = 0
//     Encoded: {4, 0}
// 5.  Gaps: There are two gaps remaining
//     -   Gap: 15 - 0 = 15 (initial neighbor is 0)
//     -   Gap: 19 - 15 = 4
//     Encoded: {15, 4}
//
// The full encoded sequence for this neighborhood is {19, 2, 1, 1, 4, 0, 15, 4}

namespace internal {

// Returns a thread-local buffer for storing the neighbors of a node. This
// avoids the overhead of constructing a temporary vector every time
// `Neighbors` is called.
std::vector<GingerNodeId>& LocalBuffer();

}  // namespace internal

// This struct contains the default settings for graph compression used by
// `CompressedCSRGraph`.
struct CompressedGraphConfig {
  // Specifies which variable length codec to use.
  using VariableLengthEncoder = VarintEncoder;
  using VariableLengthDecoder = VarintDecoder;

  // Specifies whether to use batch gap decoding and the codecs to use.
  static constexpr bool kUseBatchGapDecoding = true;
  using BatchGapEncoder = AdaptiveGapEncoder;
  using BatchGapDecoder = AdaptiveGapDecoder;

  // Specifies whether to use interval encoding for compressed neighborhoods.
  static constexpr bool kUseIntervalEncoding = false;

  // Specifies the minimum length of a contiguous run of neighbors to be encoded
  // as an interval. Note this is only used if `kUseIntervalEncoding` is true.
  static constexpr size_t kIntervalThreshold = 3;
};

template <typename Config>
class CompressedCSRGraph;

// Optional adaptive SIMD format for sorted neighbor gap coding. To construct an
// `AdaptiveCompressedGraph` via `compressed_graph_builder.h`:
//
//   AdaptiveCompressedGraph compressed =
//       CompressGraph<FlatGraph, AdaptiveCompressedGraphConfig>(flat);
struct AdaptiveCompressedGraphConfig : public CompressedGraphConfig {
  static constexpr bool kUseBatchGapDecoding = true;
  using BatchGapEncoder = AdaptiveGapEncoder;
  using BatchGapDecoder = AdaptiveGapDecoder;
};

using AdaptiveCompressedGraph =
    CompressedCSRGraph<AdaptiveCompressedGraphConfig>;

// A compressed graph data structure based on the CSR format. We store in a
// contiguous block of memory for each node its degree followed by its outgoing
// edges. The neighbor IDs are compressed using variable-length encoding, gap
// encoding, and optionally interval encoding. The template parameter `Config`
// specifies the concrete settings.
template <typename Config = CompressedGraphConfig>
class CompressedCSRGraph {
  using VariableLengthDecoder = Config::VariableLengthDecoder;

  static constexpr bool kUseBatchGapDecoding = Config::kUseBatchGapDecoding;
  using BatchGapDecoder = Config::BatchGapDecoder;

  static constexpr bool kUseIntervalEncoding = Config::kUseIntervalEncoding;
  static constexpr size_t kIntervalThreshold = Config::kIntervalThreshold;

 public:
  static constexpr size_t kSlopBytes =
      std::max(VariableLengthDecoder::kSlopBytes,
               kUseBatchGapDecoding ? BatchGapDecoder::kSlopBytes : size_t{0});

  // Constructs a `CompressedCSRGraph` from pre-computed offsets and compressed
  // targets.
  // The `offsets` vector contains pointers into the `compressed_targets` array
  // and must have a size of `NumNodes() + 1`. In particular, `offsets[u]`
  // stores the index of the first byte in `compressed_targets` where the
  // neighborhood of node `u` is encoded. The last entry should point to the
  // byte immediately following the last encoded neighborhood, i.e., its value
  // may be less than `compressed_targets.size()` if padding is already present.
  // The constructor canonicalizes decoder padding to `end + kSlopBytes`. The
  // `num_arcs` parameter is the total number of directed edges encoded within
  // `compressed_targets`.
  CompressedCSRGraph(std::vector<int64_t> offsets,
                     std::vector<uint8_t> compressed_targets, int64_t num_arcs)
      : compressed_targets_(
            PadCompressedTargets(offsets, std::move(compressed_targets))),
        offsets_(std::move(offsets)),
        num_arcs_(num_arcs) {}

  int64_t Degree(GingerNodeId node_id) const {
    VariableLengthDecoder decoder = GetDecoder(node_id);
    const auto [degree, unused_has_intervals] = DecodeDegree(decoder);
    return degree;
  }

  GingerNodeId NumNodes() const { return offsets_.size() - 1; }
  int64_t NumArcs() const { return num_arcs_; }

  // Returns a span of the neighbors of the given node. Note that the span is
  // valid until the next call to `Neighbors` from the same thread.
  absl::Span<GingerNodeId> Neighbors(GingerNodeId node_id) const {
    std::vector<GingerNodeId>& neighbors_buffer = internal::LocalBuffer();
    FillNeighbors(node_id, neighbors_buffer);
    return absl::MakeSpan(neighbors_buffer);
  }

  // Writes the neighbors of a given node into the given buffer. The buffer is
  // resized to the degree before writing the neighbors.
  void FillNeighbors(GingerNodeId node_id,
                     std::vector<GingerNodeId>& neighbors_buffer) const {
    VariableLengthDecoder decoder = GetDecoder(node_id);

    const auto [degree, has_intervals] = DecodeDegree(decoder);
    neighbors_buffer.resize(degree);

    absl::Span<GingerNodeId> output_buffer = absl::MakeSpan(neighbors_buffer);
    if constexpr (kUseIntervalEncoding) {
      if (has_intervals) {
        const GingerNodeId total_interval_length =
            DecodeIntervals(decoder, output_buffer);

        // `DecodeIntervals` has written `total_interval_length` values to the
        // start of `output_buffer`. We advance the buffer to skip these.
        output_buffer = output_buffer.subspan(total_interval_length);
      }
    }

    DecodeGaps(decoder, output_buffer);
  }

  // Prefetches the offset table entry for `node_id`.
  void PrefetchOffsetRange(GingerNodeId node_id) const {
    ABSL_DCHECK_GE(node_id, 0);
    ABSL_DCHECK_LT(node_id, NumNodes());
    offsets_.PrefetchRange(node_id);
  }

  // Resolves the offset range for `node_id` and prefetches the compressed
  // neighbor IDs.
  void PrefetchNeighbors(GingerNodeId node_id) const {
    ABSL_DCHECK_GE(node_id, 0);
    ABSL_DCHECK_LT(node_id, NumNodes());

    const auto [offset, end_offset] = offsets_.GetRange(node_id);
    if (offset >= end_offset) return;

    const uintptr_t first_line =
        reinterpret_cast<uintptr_t>(compressed_targets_.data() + offset) &
        ~(static_cast<uintptr_t>(ABSL_CACHELINE_SIZE) - 1);
    const uintptr_t last_byte = reinterpret_cast<uintptr_t>(
        compressed_targets_.data() + end_offset - 1);

    for (uintptr_t line = first_line; line <= last_byte;
         line += ABSL_CACHELINE_SIZE) {
      absl::PrefetchToLocalCache(reinterpret_cast<const void*>(line));
    }
  }

  size_t NumBytes() const {
    return offsets_.NumBytes() + compressed_targets_.size() * sizeof(uint8_t);
  }

  float BitsPerEdge() const {
    return NumArcs() == 0 ? 0.0f : 8.0f * NumBytes() / NumArcs();
  }

  const CompressedOffsets& Offsets() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return offsets_;
  }

  absl::Span<const uint8_t> CompressedTargets() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return compressed_targets_;
  }

  bool operator==(const CompressedCSRGraph& other) const {
    return num_arcs_ == other.num_arcs_ && offsets_ == other.offsets_ &&
           compressed_targets_ == other.compressed_targets_;
  }
  bool operator!=(const CompressedCSRGraph& other) const {
    return !(*this == other);
  }

 private:
  static std::vector<uint8_t> PadCompressedTargets(
      absl::Span<const int64_t> offsets,
      std::vector<uint8_t> compressed_targets) {
    ABSL_CHECK(!offsets.empty());
    const int64_t end = offsets.back();
    ABSL_CHECK_GE(end, 0);
    const size_t end_offset = static_cast<size_t>(end);
    ABSL_CHECK_LE(end_offset, compressed_targets.size());
    compressed_targets.resize(end_offset);
    compressed_targets.resize(end_offset + kSlopBytes, 0);
    return compressed_targets;
  }

  std::pair<GingerNodeId, bool> DecodeDegree(
      VariableLengthDecoder& decoder) const {
    // If interval encoding is enabled, we store in the least significant bit of
    // the degree value whether there are intervals in the neighborhood. To get
    // the degree, we simply drop the least significant bit.
    if constexpr (kUseIntervalEncoding) {
      const GingerNodeId tagged_degree = decoder.Next();
      const GingerNodeId degree = tagged_degree >> 1;
      const bool has_intervals = tagged_degree & 1;
      return {degree, has_intervals};
    } else {
      const GingerNodeId degree = decoder.Next();
      return {degree, /*has_intervals=*/false};
    }
  }

  GingerNodeId DecodeIntervals(VariableLengthDecoder& decoder,
                               absl::Span<GingerNodeId> output_buffer) const {
    const GingerNodeId num_intervals = decoder.Next();

    // To know how many gaps to decode after the intervals, we keep track of the
    // total length of all intervals by adding up the individual lengths.
    GingerNodeId total_interval_length = 0;

    GingerNodeId prev_right_extreme = 0;
    for (GingerNodeId i = 0; i < num_intervals; ++i) {
      const GingerNodeId delta_left_extreme = decoder.Next();
      const GingerNodeId delta_length = decoder.Next();

      const GingerNodeId left_extreme = delta_left_extreme + prev_right_extreme;
      const GingerNodeId length = delta_length + kIntervalThreshold;

      prev_right_extreme = left_extreme + (length - 1);

      for (GingerNodeId id = left_extreme; id < left_extreme + length; ++id) {
        output_buffer[total_interval_length++] = id;
      }
    }

    return total_interval_length;
  }

  void DecodeGaps(VariableLengthDecoder& decoder,
                  absl::Span<GingerNodeId> output_buffer) const {
    if constexpr (kUseBatchGapDecoding) {
      BatchGapDecoder::DecodeGaps(decoder.Remaining(), output_buffer);
    } else {
      GingerNodeId previous_neighbor = 0;
      for (GingerNodeId i = 0; i < output_buffer.size(); ++i) {
        const GingerNodeId gap = decoder.Next();
        const GingerNodeId neighbor = previous_neighbor + gap;

        output_buffer[i] = neighbor;
        previous_neighbor = neighbor;
      }
    }
  }

  VariableLengthDecoder GetDecoder(GingerNodeId node_id) const {
    const auto [offset, end_offset] = offsets_.GetRange(node_id);
    const uint8_t* neighborhood = compressed_targets_.data() + offset;
    return VariableLengthDecoder(
        absl::MakeConstSpan(neighborhood, end_offset - offset));
  }

  const std::vector<uint8_t> compressed_targets_;
  const CompressedOffsets offsets_;
  const int64_t num_arcs_;
};

}  // namespace ginger

#endif  // GINGER_GRAPH_COMPRESSED_GRAPH_H_
