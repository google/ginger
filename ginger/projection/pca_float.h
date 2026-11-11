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

#ifndef GINGER_PROJECTION_PCA_FLOAT_H_
#define GINGER_PROJECTION_PCA_FLOAT_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/projection/config.pb.h"
#include "ginger/projection/pca.h"

namespace ginger {

// PCA projection implementation utilizing floats for the projection matrix.
class FloatPcaProjection : public PcaProjection {
 public:
  // Bring the 1-argument overload Project(query) into scope from PcaProjection
  // so that overriding Project(query, output) does not hide it.
  using PcaProjection::Project;

  static absl::StatusOr<std::unique_ptr<FloatPcaProjection>> Create(
      Eigen::MatrixXf projection_matrix);

  absl::Status Project(absl::Span<const float> query,
                       absl::Span<float> output) const override;

  const Eigen::MatrixXf& projection_matrix() const {
    return projection_matrix_;
  }

 private:
  explicit FloatPcaProjection(int32_t input_dim,
                              Eigen::MatrixXf projection_matrix);

  Eigen::MatrixXf projection_matrix_;
};

}  // namespace ginger

#endif  // GINGER_PROJECTION_PCA_FLOAT_H_
