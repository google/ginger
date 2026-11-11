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

#ifndef GINGER_TURBOQUANT_CHAMFER_CHAMFER_H_
#define GINGER_TURBOQUANT_CHAMFER_CHAMFER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/chamfer/point_cloud_query.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// A dataset of multi-vector point clouds encoded as TurboQuant vectors for
// high-throughput Chamfer distance computation.
//
// Each document is a variable-length point cloud of token embeddings sharing
// the same dimensionality. Embeddings are pre-transposed during dataset
// initialization into SIMD-friendly panels with 4 bits per dimension.
class TurboQuantChamferDataset {
 public:
  TurboQuantChamferDataset() = default;

  TurboQuantChamferDataset(const ginger::TurboQuantDatasetSoA& tokens,
                           std::vector<size_t> doc_offsets);

  TurboQuantChamferDataset(TurboQuantChamferDataset&&) noexcept = default;
  TurboQuantChamferDataset& operator=(TurboQuantChamferDataset&&) noexcept =
      default;

  GingerNodeId size() const { return size_; }

  size_t dimensionality() const { return dimensionality_; }

  // Number of tokens in document `id`.
  size_t num_tokens(GingerNodeId id) const {
    ABSL_DCHECK_GE(id, 0);
    ABSL_DCHECK_LT(id, size_);
    return doc_offsets_[id + 1] - doc_offsets_[id];
  }

  absl::Span<const size_t> doc_offsets() const { return doc_offsets_; }

  // Accessors for pre-transposed 4-bit panel storage and precomputed norms.
  // Each panel contains `points_per_panel` points packed as 4-bit nibbles.
  const uint8_t* transposed_panels() const { return transposed_panels_.data(); }
  const float* panel_norms() const { return panel_norms_.data(); }
  absl::Span<const size_t> doc_panel_offsets() const {
    return doc_panel_offsets_;
  }
  size_t panel_bytes() const { return panel_bytes_; }
  size_t points_per_panel() const { return points_per_panel_; }

  // Computes the Chamfer distance from `query` to each neighbor's document,
  // writing results into neighbors[i].distance.
  void ChamferDistance(const TurboQuantPointCloudQuery& query,
                       absl::Span<DistanceAndId> neighbors) const;

 private:
  size_t dimensionality_ = 0;
  GingerNodeId size_ = 0;
  std::vector<size_t> doc_offsets_;
  size_t panel_bytes_ = 0;
  size_t points_per_panel_ = 0;
  std::vector<size_t> doc_panel_offsets_;
  hwy::AlignedVector<uint8_t> transposed_panels_;
  hwy::AlignedVector<float> panel_norms_;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_CHAMFER_CHAMFER_H_
