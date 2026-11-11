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

#ifndef GINGER_PROJECTION_PCA_H_
#define GINGER_PROJECTION_PCA_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/projection/config.pb.h"

namespace ginger {

// This projection fits a linear transformation matrix to a high-dimensional
// dataset to project points into a lower-dimensional subspace while maximizing
// the variance of the projected points.
class PcaProjection {
 public:
  virtual ~PcaProjection() = default;

  // Creates a new PCA projection instance by fitting a projection matrix to
  // the given dataset.
  //
  // Requirements:
  // - If `config.dimension()` is selected:
  //   - `config.dimension()` must be positive and less than or equal to the
  //     dimensionality of the points in `dataset`.
  // - If `config.truncation()` is selected:
  //   - `config.truncation().significance_threshold()` must be in (0, 1].
  //   - `config.truncation().truncation_threshold()` must be in (0, 1].
  //   - `config.truncation().alignment()` must be non-negative. If set to 0,
  //     no alignment is performed.
  //   - `config.truncation().alignment_tail()` must be in [0, alignment).
  // - `config.matrix_representation()` must be specified and supported.
  // - `dataset` must contain at least 2 points and have float coordinates.
  //
  // Returns:
  // - A unique pointer to the initialized PcaProjection instance if successful.
  // - An absl::InvalidArgumentError if any of the requirements are violated.
  // - An absl::InternalError if the eigendecomposition solver fails.
  // - An absl::FailedPreconditionError if truncation is selected but the target
  //   dimensionality is not sufficiently reduced.
  static absl::StatusOr<std::unique_ptr<PcaProjection>> Create(
      const PcaProjectionConfig& config, const SpanDataset& dataset);

  // Projects a single vector into the lower-dimensional PCA subspace,
  // allocating and returning a new vector for the result.
  //
  // Preconditions:
  // - `query` must have the same dimensionality as the original input dataset
  //    used to fit this PCA projection.
  //
  // Returns:
  // - A vector of size `target_dim` containing the projected coordinates.
  // - An absl::InvalidArgumentError if `query` has an incorrect dimensionality.
  virtual absl::StatusOr<std::vector<float>> Project(
      absl::Span<const float> query) const;

  // Projects a single vector into the lower-dimensional PCA subspace,
  // writing the result into the provided `output` buffer.
  //
  // Preconditions:
  // - `query` must have size equal to the input dimensionality of this PCA
  //   projection.
  // - `output` must have size equal to `output_dim()`.
  //
  // Returns:
  // - absl::OkStatus() if successful.
  // - An absl::InvalidArgumentError if `query` or `output` have incorrect
  //   dimensionalities.
  virtual absl::Status Project(absl::Span<const float> query,
                               absl::Span<float> output) const = 0;

  // Returns the dimensionality of the input space of this projection.
  int32_t input_dim() const { return input_dim_; }

  // Returns the dimensionality of the output space of this projection.
  int32_t output_dim() const { return output_dim_; }

  // Returns the matrix representation of this projection.
  MatrixRepresentation matrix_representation() const {
    return matrix_representation_;
  }

 protected:
  PcaProjection(int32_t input_dim, int32_t output_dim,
                MatrixRepresentation matrix_representation)
      : input_dim_(input_dim),
        output_dim_(output_dim),
        matrix_representation_(matrix_representation) {}

 private:
  int32_t input_dim_ = 0;
  int32_t output_dim_ = 0;
  MatrixRepresentation matrix_representation_ =
      MATRIX_REPRESENTATION_UNSPECIFIED;
};

}  // namespace ginger

#endif  // GINGER_PROJECTION_PCA_H_
