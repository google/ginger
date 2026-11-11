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

#ifndef GINGER_PROJECTION_PCA_TRAINING_H_
#define GINGER_PROJECTION_PCA_TRAINING_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"
#include "ginger/dataset.h"
#include "ginger/projection/config.pb.h"
#include "ginger/types.h"

namespace ginger {

// Trains PCA on the given dataset, returning the projection matrix.
absl::StatusOr<Eigen::MatrixXf> TrainPca(const PcaProjectionConfig& config,
                                         const SpanDataset& dataset);

namespace internal {

// Validates that the configuration parameters are valid for the input dataset.
absl::Status ValidateConfig(const PcaProjectionConfig& config,
                            int32_t input_dim);

// Resolves the number of points to use for PCA training based on the sampling
// configuration. Returns `dataset_size` if sampling is not configured or if the
// requested sample size is at least `dataset_size`.
GingerNodeId GetSampleSize(const PcaProjectionConfig& config,
                           GingerNodeId dataset_size, int32_t input_dim);

// Samples `sample_size` distinct points uniformly at random from `dataset`
// using `seed` and returns them as a new `SpanDataset` for PCA training.
SpanDataset CreateTrainingDataset(const SpanDataset& dataset,
                                  GingerNodeId sample_size, int32_t seed);

// Computes the component-wise arithmetic mean across all points in the dataset.
Eigen::VectorXd ComputeMeanVector(const SpanDataset& dataset);

// Constructs the centered covariance matrix from the dataset.
Eigen::MatrixXd ComputeCovarianceMatrix(const SpanDataset& dataset);

struct EigenDecompositionResult {
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXf eigenvectors;
};

// Performs an eigendecomposition of the covariance matrix and extracts the top
// `target_dim` principal directions and corresponding eigenvalues.
absl::StatusOr<EigenDecompositionResult> ComputeEigenvectors(
    const Eigen::MatrixXd& covariance_matrix, int32_t target_dim);

// Truncates the eigenvectors and eigenvalues as specified by the configuration
// to keep only the most significant components. Returns an
// `absl::FailedPreconditionError` if the target dimensionality is not
// sufficiently reduced relative to the original dimensionality.
absl::Status PerformTruncation(const PcaProjectionConfig::Truncation& config,
                               EigenDecompositionResult& result);

}  // namespace internal
}  // namespace ginger

#endif  // GINGER_PROJECTION_PCA_TRAINING_H_
