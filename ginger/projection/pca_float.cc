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

#include "ginger/projection/pca_float.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/projection/config.pb.h"
#include "ginger/projection/pca.h"

namespace ginger {

absl::StatusOr<std::unique_ptr<FloatPcaProjection>> FloatPcaProjection::Create(
    Eigen::MatrixXf projection_matrix) {
  const int32_t output_dim = projection_matrix.rows();
  const int32_t input_dim = projection_matrix.cols();
  if (output_dim == 0 || input_dim == 0) {
    return absl::InvalidArgumentError(
        "Projection matrix dimensions must be strictly positive.");
  }

  if (output_dim > input_dim) {
    return absl::InvalidArgumentError(
        "Output dimension must be smaller than or equal to input dimension.");
  }

  return std::unique_ptr<FloatPcaProjection>(
      new FloatPcaProjection(input_dim, std::move(projection_matrix)));
}

FloatPcaProjection::FloatPcaProjection(int32_t input_dim,
                                       Eigen::MatrixXf projection_matrix)
    : PcaProjection(input_dim, projection_matrix.rows(),
                    MATRIX_REPRESENTATION_FLOAT),
      projection_matrix_(std::move(projection_matrix)) {}

absl::Status FloatPcaProjection::Project(absl::Span<const float> query,
                                         absl::Span<float> output) const {
  if (query.size() != static_cast<size_t>(input_dim())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Query dimensionality (%d) does not match PCA input "
                        "dimensionality (%d).",
                        query.size(), input_dim()));
  }
  const int32_t output_dim = projection_matrix_.rows();
  if (output.size() != output_dim) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Output dimensionality (%d) does not match PCA output "
                        "dimensionality (%d).",
                        output.size(), output_dim));
  }

  Eigen::Map<const Eigen::VectorXf> query_vec(query.data(), query.size());
  Eigen::Map<Eigen::VectorXf> result_vec(output.data(), output.size());
  result_vec.noalias() = projection_matrix_ * query_vec;

  return absl::OkStatus();
}

}  // namespace ginger
