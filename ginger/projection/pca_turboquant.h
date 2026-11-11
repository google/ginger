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

#ifndef GINGER_PROJECTION_PCA_TURBOQUANT_H_
#define GINGER_PROJECTION_PCA_TURBOQUANT_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/projection/config.pb.h"
#include "ginger/projection/pca.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// PCA projection implementation where the rows of the projection matrix are
// quantized to 4 bits per weight using TurboQuant. This reduces memory traffic
// during projection by ~8x compared to single-precision float matrices.
class TurboQuantPcaProjection : public PcaProjection {
 public:
  // Bring the 1-argument overload Project(query) into scope from PcaProjection
  // so that overriding Project(query, output) does not hide it.
  using PcaProjection::Project;

  static absl::StatusOr<std::unique_ptr<TurboQuantPcaProjection>> Create(
      const Eigen::MatrixXf& projection_matrix);

  static absl::StatusOr<std::unique_ptr<TurboQuantPcaProjection>> Create(
      int32_t input_dim, int32_t output_dim,
      hwy::AlignedVector<uint8_t> packed_weights,
      hwy::AlignedVector<float> row_inv_scales);

  absl::Status Project(absl::Span<const float> query,
                       absl::Span<float> output) const override;

  absl::Span<const uint8_t> packed_weights() const {
    return absl::MakeConstSpan(packed_weights_);
  }
  absl::Span<const float> row_inv_scales() const {
    return absl::MakeConstSpan(row_inv_scales_);
  }

 private:
  explicit TurboQuantPcaProjection(int32_t input_dim, int32_t output_dim,
                                   hwy::AlignedVector<uint8_t> packed_weights,
                                   hwy::AlignedVector<float> row_inv_scales);

  hwy::AlignedVector<uint8_t> packed_weights_;
  hwy::AlignedVector<float> row_inv_scales_;
};

}  // namespace ginger

#endif  // GINGER_PROJECTION_PCA_TURBOQUANT_H_
