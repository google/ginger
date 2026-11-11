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

#ifndef GINGER_TURBOQUANT_PIPNN_PROCESS_LEAF_H_
#define GINGER_TURBOQUANT_PIPNN_PROCESS_LEAF_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// Per-query result of fused top-2 neighbor selection.
// Scores are approximate distances.
struct Top2Result {
  float best_score = std::numeric_limits<float>::max();
  float second_best_score = std::numeric_limits<float>::max();
  // Local offsets [0, indices.size()) into the `indices` span passed to
  // ProcessLeafTop2. To map back to global indices, use indices[best_idx].
  int32_t best_idx = -1;
  int32_t second_best_idx = -1;

  // Each index must be presented to Update() at most once; the lower-triangle
  // optimization guarantees this by computing each (query, db) pair exactly
  // once.
  void Update(float score, int32_t idx) {
    ABSL_DCHECK_NE(idx, best_idx);
    if (score < best_score) {
      second_best_score = best_score;
      second_best_idx = best_idx;
      best_score = score;
      best_idx = idx;
    } else if (score < second_best_score) {
      second_best_score = score;
      second_best_idx = idx;
    }
    // TODO: Figure out a way to turn this into a CHECK without
    // performance impact.
    ABSL_DCHECK_NE(best_idx, second_best_idx);
  }
};

// Reusable workspace for ProcessLeaf to avoid repeated heap allocations.
// Create once, pass to ProcessLeaf on each call. Buffers are resized as needed
// but not reallocated if capacity is sufficient.
struct ProcessLeafWorkspace {
  hwy::AlignedVector<uint8_t> panel_data;
  // Reuse buffer for decoding one batch of kMq queries at a time.
  hwy::AlignedVector<int8_t> decoded_buf;
  // Padded to num_panels * points_per_panel with 0 (norms) or +inf (sqr_norms)
  // for SIMD panel access.
  hwy::AlignedVector<float> norms;
  // Squared norms of unquantized vectors (only populated for L2 metric).
  hwy::AlignedVector<float> sqr_norms;
  hwy::AlignedVector<Top2Result> top2_results;

  // Struct-of-Arrays db-side min trackers (hoisted to avoid re-allocation per
  // call).
  hwy::AlignedVector<float> db_min1_s;
  hwy::AlignedVector<float> db_min2_s;
  hwy::AlignedVector<int32_t> db_min1_idx_s;
  hwy::AlignedVector<int32_t> db_min2_idx_s;

  // Reusable buffers for CSR-like flat storage of edges per leaf in PiPNN
  // build.
  std::vector<int32_t> leaf_edge_offsets;
  std::vector<int32_t> leaf_edge_targets;
};

// Fused top-2 neighbor selection using lower triangle (symmetric update).
// Returns top-2 results per point in workspace->top2_results.
// The `indices` indicate which datapoints of `quantized_points` to use.
// The indices returned in `workspace.top2_results` (best_idx/second_best_idx)
// are local offsets [0, indices.size()) into the `indices` span, NOT global
// indices.
// IMPORTANT: duplicate indices are not allowed. If a duplicated point is
// someone's top 1, it will also show up as their top 2.
//
// The output is written to `workspace.top2_results`.
// There is no need to clean up the workspace before calling
// ProcessLeaf or initialize it in a certain way.
// For best performance, reuse the same workspace across multiple calls.
//
// Output score semantics:
//   Dot product: score ≈ −dot(q, db) (lower = more similar)
//   L2: score ≈ ||q − db||² (lower = more similar)
// The `quantizer` object stores whether dot product or squared L2 is used.
//
// `quantizer` must be the same object used to encode `quantized_points`.
void ProcessLeafTop2(const TurboQuantCodec& quantizer,
                     const TurboQuantDatasetSoA& quantized_points,
                     absl::Span<const int> indices,
                     ProcessLeafWorkspace& workspace);

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_PIPNN_PROCESS_LEAF_H_
