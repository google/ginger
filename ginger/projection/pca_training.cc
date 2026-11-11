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

#include "ginger/projection/pca_training.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "ginger/dataset.h"
#include "ginger/projection/config.pb.h"
#include "ginger/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace ginger {
namespace internal {

absl::Status ValidateConfig(const PcaProjectionConfig& config,
                            const int32_t input_dim) {
  if (input_dim <= 0) {
    return absl::InvalidArgumentError(
        "The dataset must have a non-zero dimension.");
  }
  switch (config.mode_case()) {
    case PcaProjectionConfig::kDimension: {
      const int32_t target_dim = config.dimension();
      if (target_dim <= 0) {
        return absl::InvalidArgumentError(
            "The target dimension must be positive.");
      }
      if (target_dim > input_dim) {
        return absl::InvalidArgumentError(
            "The target dimension must be smaller than or equal to the "
            "dimensionality of the dataset.");
      }
      break;
    }
    case PcaProjectionConfig::kTruncation: {
      const auto& truncation = config.truncation();
      if (truncation.significance_threshold() <= 0.0f ||
          truncation.significance_threshold() > 1.0f) {
        return absl::InvalidArgumentError(
            "The significance threshold must be in (0, 1].");
      }
      if (truncation.truncation_threshold() <= 0.0f ||
          truncation.truncation_threshold() > 1.0f) {
        return absl::InvalidArgumentError(
            "The truncation threshold must be in (0, 1].");
      }
      if (truncation.alignment() < 0) {
        return absl::InvalidArgumentError(
            "The alignment must be non-negative.");
      }
      if (truncation.alignment_tail() < 0 ||
          (truncation.alignment_tail() > 0 &&
           truncation.alignment_tail() >= truncation.alignment())) {
        return absl::InvalidArgumentError(
            "The alignment tail must be in [0, alignment).");
      }
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "The PcaProjectionConfig must specify either dimension or truncation "
          "options.");
  }

  switch (config.matrix_representation()) {
    case MATRIX_REPRESENTATION_FLOAT:
    case MATRIX_REPRESENTATION_TURBOQUANT:
      break;
    case MATRIX_REPRESENTATION_UNSPECIFIED:
      return absl::InvalidArgumentError(
          "Matrix representation must be specified in PcaProjectionConfig.");
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported matrix representation: ",
                       config.matrix_representation()));
  }

  if (config.has_sampling()) {
    const auto& sampling = config.sampling();
    switch (sampling.strategy_case()) {
      case PcaProjectionConfig::Sampling::kSampleSize:
        if (sampling.sample_size() < 2) {
          return absl::InvalidArgumentError(
              "The sample size must be at least 2.");
        }
        break;
      case PcaProjectionConfig::Sampling::kSamplesPerDimension:
        if (sampling.samples_per_dimension() <= 0) {
          return absl::InvalidArgumentError(
              "The number of samples per dimension must be positive.");
        }
        if (sampling.samples_per_dimension() * input_dim < 2) {
          return absl::InvalidArgumentError(
              "The effective sample size (samples_per_dimension * input_dim) "
              "must be at least 2.");
        }
        break;
      case PcaProjectionConfig::Sampling::STRATEGY_NOT_SET:
        return absl::InvalidArgumentError(
            "The PcaProjectionConfig::Sampling must specify either sample_size "
            "or samples_per_dimension.");
    }
  }

  return absl::OkStatus();
}

GingerNodeId GetSampleSize(const PcaProjectionConfig& config,
                           GingerNodeId dataset_size, int32_t input_dim) {
  if (!config.has_sampling()) {
    return dataset_size;
  }

  const auto& sampling = config.sampling();
  switch (sampling.strategy_case()) {
    case PcaProjectionConfig::Sampling::kSampleSize:
      return std::min(dataset_size, sampling.sample_size());
    case PcaProjectionConfig::Sampling::kSamplesPerDimension:
      return std::min(dataset_size,
                      sampling.samples_per_dimension() * input_dim);
    case PcaProjectionConfig::Sampling::STRATEGY_NOT_SET:
      return dataset_size;
  }
}

SpanDataset CreateTrainingDataset(const SpanDataset& dataset,
                                  GingerNodeId sample_size, int32_t seed) {
  const GingerNodeId num_points = dataset.size();
  ABSL_CHECK_GT(sample_size, 0);
  ABSL_CHECK_LE(sample_size, num_points);

  std::mt19937 rng(seed);
  std::vector<GingerNodeId> indices(sample_size);
  const auto all_indices = parlay::iota(num_points);
  std::sample(all_indices.begin(), all_indices.end(), indices.begin(),
              sample_size, rng);

  SpanDataset training_dataset(sample_size, dataset.dimensionality(),
                               dataset.coordinate_type());
  parlay::parallel_for(0, sample_size, [&](size_t i) {
    training_dataset.SetBytes(i, dataset.get_bytes(indices[i]));
  });

  return training_dataset;
}

// A generic parallel map-reduce style function where inner batches compute
// locally without synchronization on a provided per-outer-batch state, and
// the resulting state for each outer batch is reduced into a global state
// inside a single critical section.
//
// Arguments:
//   - size: total number of elements to process.
//   - batch_size: size of each local batch.
//   - create_state: creates the initial local state for a batch.
//   - reduce_inner_batch: processes a batch locally.
//   - update_global: merges local state into the global state.
template <typename State, typename CreateStateFn, typename ReduceInnerBatchFn,
          typename UpdateGlobalFn>
void ParallelBatchReduce(size_t size, size_t batch_size,
                         CreateStateFn create_state,
                         ReduceInnerBatchFn reduce_inner_batch,
                         UpdateGlobalFn update_global) {
  const auto ceil_div = [](size_t a, size_t b) { return (a + b - 1) / b; };
  const size_t num_batches = ceil_div(size, batch_size);
  if (num_batches == 0) {
    return;
  }

  const size_t num_batches_per_outer_batch =
      ceil_div(num_batches, parlay::num_workers());

  absl::Mutex mutex;
  parlay::blocked_for(
      0, num_batches, num_batches_per_outer_batch,
      [&](size_t outer_id, size_t inner_start, size_t inner_end) {
        State local_state = create_state();

        for (size_t inner_id = inner_start; inner_id < inner_end; ++inner_id) {
          const size_t start_point = inner_id * batch_size;
          const size_t current_batch_size =
              std::min(size - start_point, batch_size);
          reduce_inner_batch(local_state, start_point, current_batch_size);
        }

        absl::MutexLock lock(mutex);
        update_global(local_state);
      });
}

Eigen::VectorXd ComputeMeanVector(const SpanDataset& dataset) {
  const size_t num_points = dataset.size();
  const size_t dim = dataset.dimensionality();
  ABSL_CHECK_GT(num_points, 0);
  ABSL_CHECK_GT(dim, 0);

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
  ParallelBatchReduce<Eigen::VectorXd>(
      num_points, /*batch_size=*/256,
      /*create_state=*/[&]() { return Eigen::VectorXd::Zero(dim); },
      /*reduce_inner_batch=*/
      [&](Eigen::VectorXd& local_sum, size_t begin, size_t batch_size) {
        for (size_t i = 0; i < batch_size; ++i) {
          Eigen::Map<const Eigen::VectorXf> point_vec(
              dataset.get<float>(begin + i).data(), dim);
          local_sum += point_vec.cast<double>();
        }
      },
      /*update_global=*/
      [&](const Eigen::VectorXd& local_sum) { mean += local_sum; });

  mean /= num_points;
  return mean;
}

Eigen::MatrixXd ComputeCovarianceMatrix(const SpanDataset& dataset) {
  const size_t num_points = dataset.size();
  ABSL_CHECK_GT(num_points, 1);
  const size_t dim = dataset.dimensionality();
  ABSL_CHECK_GT(dim, 0);

  const Eigen::VectorXf mean_vec = ComputeMeanVector(dataset).cast<float>();

  struct LocalState {
    Eigen::MatrixXf covariance_matrix;
    Eigen::MatrixXf centered_matrix;
  };
  Eigen::MatrixXd covariance_matrix = Eigen::MatrixXd::Zero(dim, dim);

  constexpr size_t kBatchSize = 256;
  ParallelBatchReduce<LocalState>(
      num_points, kBatchSize,
      /*create_state=*/
      [&]() {
        return LocalState{
            /*covariance=*/Eigen::MatrixXf::Zero(dim, dim),
            /*centered_matrix=*/Eigen::MatrixXf(dim, kBatchSize),
        };
      },
      /*reduce_inner_batch=*/
      [&](LocalState& state, size_t begin, size_t batch_size) {
        auto active_cols = state.centered_matrix.leftCols(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
          Eigen::Map<const Eigen::VectorXf> point_vec(
              dataset.get<float>(begin + i).data(), dim);
          active_cols.col(i) = point_vec - mean_vec;
        }

        // Accumulate the product of centered points into the lower triangle.
        // Since the covariance matrix is symmetric, we can compute only the
        // lower triangle and reconstruct the upper triangle later.
        state.covariance_matrix.selfadjointView<Eigen::Lower>().rankUpdate(
            active_cols);
      },
      /*update_global=*/
      [&](const LocalState& state) {
        covariance_matrix.triangularView<Eigen::Lower>() +=
            state.covariance_matrix.cast<double>();
      });

  // Reconstruct the symmetric upper triangle from the lower triangle, which we
  // omitted in the parallel reduction.
  covariance_matrix.triangularView<Eigen::Upper>() =
      covariance_matrix.transpose();

  // As we check for `num_points > 1` above, this division is safe.
  covariance_matrix /= (num_points - 1);

  return covariance_matrix;
}

absl::StatusOr<EigenDecompositionResult> ComputeEigenvectors(
    const Eigen::MatrixXd& covariance_matrix, int32_t target_dim) {
  if (covariance_matrix.rows() != covariance_matrix.cols()) {
    return absl::InvalidArgumentError("Covariance matrix must be square.");
  }
  if (target_dim <= 0) {
    return absl::InvalidArgumentError("The target dimension must be positive.");
  }
  const int32_t input_dim = covariance_matrix.rows();
  if (target_dim > input_dim) {
    return absl::InvalidArgumentError(
        "The target dimension must be smaller than or equal to the "
        "dimensionality of the covariance matrix.");
  }

  // TODO: Consider using a parallel solver as this step can become
  // a bottleneck on some datasets.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(
      covariance_matrix.cast<float>());
  if (solver.info() != Eigen::Success) {
    return absl::InternalError(
        "Eigen solver failed to compute eigenvalues/eigenvectors.");
  }

  EigenDecompositionResult result;
  result.eigenvalues.resize(target_dim);
  result.eigenvectors.resize(target_dim, input_dim);

  for (int32_t i = 0; i < target_dim; ++i) {
    // Eigen stores the eigenvalues/eigenvectors in increasing order. We extract
    // them starting from the end to place the most important components first.
    const int32_t col_idx = input_dim - 1 - i;

    result.eigenvalues(i) = solver.eigenvalues()(col_idx);
    result.eigenvectors.row(i) = solver.eigenvectors().col(col_idx);
  }

  return result;
}

absl::Status PerformTruncation(const PcaProjectionConfig::Truncation& config,
                               EigenDecompositionResult& result) {
  ABSL_CHECK_GT(config.significance_threshold(), 0.0f);
  ABSL_CHECK_LE(config.significance_threshold(), 1.0f);
  ABSL_CHECK_GT(config.truncation_threshold(), 0.0f);
  ABSL_CHECK_LE(config.truncation_threshold(), 1.0f);
  ABSL_CHECK_GE(config.alignment(), 0);
  ABSL_CHECK_GT(result.eigenvalues.size(), 0);
  ABSL_CHECK_EQ(result.eigenvectors.rows(), result.eigenvalues.size());

  const int32_t num_eigenvalues = result.eigenvalues.size();
  const double threshold =
      config.significance_threshold() * result.eigenvalues.sum();

  // Find the minimum number of components needed to preserve at least
  // `config.significance_threshold()` fraction of the total variance.
  double sum = 0.0;
  int32_t target_dim = num_eigenvalues;
  for (int32_t i = 0; i < num_eigenvalues; ++i) {
    sum += result.eigenvalues(i);

    if (sum >= threshold) {
      target_dim = i + 1;
      break;
    }
  }

  // Round up the target dimension to the next value of the form
  // `k * alignment - alignment_tail`, if specified.
  if (config.alignment() > 0) {
    const int32_t align_dims = config.alignment();
    const int32_t tail_dims = config.alignment_tail();
    target_dim =
        ((target_dim + tail_dims + align_dims - 1) / align_dims) * align_dims -
        tail_dims;
    target_dim = std::min(target_dim, num_eigenvalues);
  }

  // Only apply truncation if the target dimensionality is sufficiently reduced
  // relative to the original dimensionality.
  if (target_dim >= num_eigenvalues * config.truncation_threshold()) {
    return absl::FailedPreconditionError(
        "Target dimensionality is not sufficiently reduced.");
  }

  result.eigenvalues.conservativeResize(target_dim);
  result.eigenvectors.conservativeResize(target_dim,
                                         result.eigenvectors.cols());

  return absl::OkStatus();
}

}  // namespace internal

absl::StatusOr<Eigen::MatrixXf> TrainPca(const PcaProjectionConfig& config,
                                         const SpanDataset& dataset) {
  if (dataset.coordinate_type() != CoordinateType::kFloat) {
    return absl::InvalidArgumentError(
        "PCA training is only supported for float datasets.");
  }
  if (dataset.size() <= 1) {
    return absl::InvalidArgumentError(
        "The dataset must contain at least 2 points for PCA.");
  }
  const int32_t input_dim = dataset.dimensionality();
  if (input_dim == 0) {
    return absl::InvalidArgumentError(
        "The dataset must have a non-zero dimension.");
  }
  ABSL_RETURN_IF_ERROR(internal::ValidateConfig(config, input_dim));

  // We compute the dimension of the PCA projection based on two settings:
  // either the user supplies a fixed dimension to project into or they set
  // the amount of variance to preserve from which we derive how many
  // dimensions to truncate. In the latter case, we compute all eigenvectors
  // and their eigenvalues are used to derive the variance.
  const int32_t target_dim =
      config.has_dimension() ? config.dimension() : input_dim;

  // To avoid the cost of computing the covariance matrix over the entire
  // dataset, we optionally train PCA on a smaller subset of points. The
  // training dataset is constructed by uniformly sampling points without
  // replacement based on either a fixed sample size or a multiple of the input
  // dimension.
  const GingerNodeId sample_size =
      internal::GetSampleSize(config, dataset.size(), input_dim);
  std::optional<SpanDataset> sampled_dataset;
  if (sample_size < dataset.size()) {
    sampled_dataset = internal::CreateTrainingDataset(dataset, sample_size,
                                                      config.sampling().seed());
  }
  const SpanDataset& training_dataset =
      sampled_dataset.has_value() ? *sampled_dataset : dataset;

  // We compute the PCA projection by computing an eigendecomposition on the
  // covariance matrix which is faster than doing singular value decomposition
  // as we usually deal with datasets where the dimension is much smaller
  // than the number of points.
  Eigen::MatrixXd covariance_matrix =
      internal::ComputeCovarianceMatrix(training_dataset);
  ABSL_ASSIGN_OR_RETURN(
      auto eigendecomposition,
      internal::ComputeEigenvectors(covariance_matrix, target_dim));

  if (config.has_truncation()) {
    ABSL_RETURN_IF_ERROR(
        internal::PerformTruncation(config.truncation(), eigendecomposition));
  }

  ABSL_LOG(INFO) << "PCA projection matrix size: "
                 << eigendecomposition.eigenvectors.rows() << " x "
                 << eigendecomposition.eigenvectors.cols();

  return eigendecomposition.eigenvectors;
}

}  // namespace ginger
