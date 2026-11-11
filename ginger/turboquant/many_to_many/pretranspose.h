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

#ifndef GINGER_TURBOQUANT_MANY_TO_MANY_PRETRANSPOSE_H_
#define GINGER_TURBOQUANT_MANY_TO_MANY_PRETRANSPOSE_H_

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// A pre-decoded, block-transposed representation of a TurboQuant-encoded
// database, ready for direct SIMD scoring without per-batch decoding.
//
// Memory layout
// -------------
// Points are grouped into *panels* of `points_per_panel` points each
// (points_per_panel = SIMD-width / 4, i.e. the number of 4-bit point lanes
// that fit in one SIMD
// register). Each panel stores decoded, interleaved tile data contiguously in
// `panel_data`, sized at `panel_bytes` bytes per panel.
//
// Norm arrays (`norm_scaling_factors`, `sqr_norms`) are indexed by linear point
// index, padded to full panel boundaries so that SIMD loads never read
// out-of-bounds. Padding entries are set to +infinity (norm_scaling_factors) or
// +max (sqr norms) so they sort to the bottom and never appear in top-k
// results.
//
// When to use
// -----------
// Pretransposition pays off when the same database is scored against multiple
// query batches (e.g., batched graph search where each node's neighbors form
// a separate query set but the database shard is reused). For a single
// query-vs-database call, the standard path is equally fast because it fuses
// decode+score into a single L3-tiled pass.
struct PreTransposedTurboQuantDataset {
  // Builds a pretransposed dataset from the given base points.
  //
  // `indices` selects and reorders a subset of points from `base_points`.
  // The resulting dataset contains `indices.size()` points in the order
  // specified by `indices`. The `quantizer` is accepted for API symmetry
  // with the scoring functions but is not used during pretransposition (the
  // codebook is embedded in the nibble-packed data).
  static PreTransposedTurboQuantDataset Build(
      const TurboQuantDatasetSoA& base_points, absl::Span<const int> indices,
      const TurboQuantCodec& quantizer);

  // Convenience overload: transposes all points in sequential order.
  static PreTransposedTurboQuantDataset Build(
      const TurboQuantDatasetSoA& base_points,
      const TurboQuantCodec& quantizer);

  // Number of database points stored in this dataset.
  size_t num_points = 0;
  // Vector dimensionality.
  size_t dimensionality = 0;
  // Nibble-packed bytes per point in the source encoding.
  size_t bytes_per_point = 0;
  // Number of panels (ceil(num_points / points_per_panel)).
  size_t num_panels = 0;
  // Byte size of one decoded panel.
  size_t panel_bytes = 0;
  // Total number of SIMD tiles across all dimension chunks per panel
  // (num_dim_chunks * tiles_per_chunk).
  size_t total_tiles = 0;
  // Points per panel (SIMD-width / 4).
  size_t points_per_panel = 0;

  // Contiguous decoded panel buffer.
  // Layout: num_panels consecutive panels, each `panel_bytes` bytes.
  hwy::AlignedVector<uint8_t> panel_data;
  // Per-point norm scaling factors, indexed by linear point index.
  // Padded to full panels with +infinity.
  hwy::AlignedVector<float> norm_scaling_factors;
  // Per-point squared L2 norms of the unquantized vectors.
  // Only populated when the source metric is kSquaredL2Distance.
  // Padded to full panels with float max.
  hwy::AlignedVector<float> sqr_norms;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_PRETRANSPOSE_H_
