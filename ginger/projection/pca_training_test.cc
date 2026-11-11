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

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/dataset.h"
#include "ginger/projection/config.pb.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_reference_stub.h"

namespace ginger::internal {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

class PcaTrainingTest : public ::testing::Test {
 protected:
  std::unique_ptr<SpanDataset> CreateDataset(
      const std::vector<std::vector<float>>& points) {
    if (points.empty()) {
      return std::make_unique<SpanDataset>(/*num_points=*/0,
                                           /*dimensionality=*/0);
    }

    const GingerNodeId num_points = points.size();
    const int32_t dim = points[0].size();

    auto dataset =
        std::make_unique<SpanDataset>(num_points, /*dimensionality=*/dim);
    for (GingerNodeId i = 0; i < num_points; ++i) {
      dataset->Set<float>(i, absl::MakeConstSpan(points[i]));
    }

    return dataset;
  }

 private:
  ginger::internal::ParallelSchedulerReference scheduler_{
      /*num_workers=*/4};
};

TEST_F(PcaTrainingTest, GetSampleSize) {
  PcaProjectionConfig config;
  config.set_dimension(2);
  EXPECT_EQ(GetSampleSize(config, /*dataset_size=*/100, /*input_dim=*/4), 100);

  config.mutable_sampling()->set_sample_size(25);
  EXPECT_EQ(GetSampleSize(config, /*dataset_size=*/100, /*input_dim=*/4), 25);
  EXPECT_EQ(GetSampleSize(config, /*dataset_size=*/10, /*input_dim=*/4), 10);

  config.mutable_sampling()->set_samples_per_dimension(10);
  EXPECT_EQ(GetSampleSize(config, /*dataset_size=*/100, /*input_dim=*/4), 40);
  EXPECT_EQ(GetSampleSize(config, /*dataset_size=*/30, /*input_dim=*/4), 30);
}

TEST_F(PcaTrainingTest, CreateTrainingDataset_SmallAndLargeFractions) {
  std::vector<std::vector<float>> points;
  points.reserve(100);
  for (int i = 0; i < 100; ++i) {
    points.push_back({static_cast<float>(i), static_cast<float>(2 * i)});
  }
  auto dataset = CreateDataset(points);

  // Small sample fraction (uses Floyd's algorithm branch: sample_size <= N/4).
  SpanDataset small_sample =
      CreateTrainingDataset(*dataset, /*sample_size=*/10, /*seed=*/42);
  ASSERT_EQ(small_sample.size(), 10);
  ASSERT_EQ(small_sample.dimensionality(), 2);
  for (GingerNodeId i = 0; i < small_sample.size(); ++i) {
    const auto pt = small_sample.get<float>(i);
    EXPECT_FLOAT_EQ(pt[1], 2.0f * pt[0]);
    if (i > 0) {
      // Sampled points should be in strictly increasing index order.
      EXPECT_GT(pt[0], small_sample.get<float>(i - 1)[0]);
    }
  }

  // Large sample fraction (uses std::sample branch: sample_size > N/4).
  SpanDataset large_sample =
      CreateTrainingDataset(*dataset, /*sample_size=*/60, /*seed=*/42);
  ASSERT_EQ(large_sample.size(), 60);
  ASSERT_EQ(large_sample.dimensionality(), 2);
  for (GingerNodeId i = 0; i < large_sample.size(); ++i) {
    const auto pt = large_sample.get<float>(i);
    EXPECT_FLOAT_EQ(pt[1], 2.0f * pt[0]);
    if (i > 0) {
      EXPECT_GT(pt[0], large_sample.get<float>(i - 1)[0]);
    }
  }
}

TEST_F(PcaTrainingTest, ComputeMeanVector_Standard) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f},
      {3.0f, 4.0f},
      {5.0f, 6.0f},
  });

  Eigen::VectorXd mean = ComputeMeanVector(*dataset);
  ASSERT_EQ(mean.size(), 2);

  EXPECT_DOUBLE_EQ(mean(0), 3.0);
  EXPECT_DOUBLE_EQ(mean(1), 4.0);
}

TEST_F(PcaTrainingTest, ComputeMeanVector_SinglePoint) {
  auto dataset = CreateDataset({
      {4.2f, -1.5f},
  });

  Eigen::VectorXd mean = ComputeMeanVector(*dataset);
  ASSERT_EQ(mean.size(), 2);

  EXPECT_NEAR(mean(0), 4.2, 1e-6);
  EXPECT_NEAR(mean(1), -1.5, 1e-6);
}

TEST_F(PcaTrainingTest, ComputeMeanVector_LargeDataset) {
  std::vector<std::vector<float>> points(500, std::vector<float>{1.5f, -2.5f});
  auto dataset = CreateDataset(points);

  Eigen::VectorXd mean = ComputeMeanVector(*dataset);
  ASSERT_EQ(mean.size(), 2);

  EXPECT_DOUBLE_EQ(mean(0), 1.5);
  EXPECT_DOUBLE_EQ(mean(1), -2.5);
}

TEST_F(PcaTrainingTest, ComputeMeanVector_MixedSigns) {
  auto dataset = CreateDataset({
      {-10.0f, 50.0f},
      {0.0f, 0.0f},
      {10.0f, -50.0f},
  });

  Eigen::VectorXd mean = ComputeMeanVector(*dataset);
  ASSERT_EQ(mean.size(), 2);

  EXPECT_DOUBLE_EQ(mean(0), 0.0);
  EXPECT_DOUBLE_EQ(mean(1), 0.0);
}

TEST_F(PcaTrainingTest, ComputeMeanVector_EmptyDataset_Dies) {
  auto dataset = CreateDataset({});
  EXPECT_DEATH(ComputeMeanVector(*dataset), "");
}

TEST_F(PcaTrainingTest, ComputeMeanVector_ZeroDimensionalDataset_Dies) {
  auto dataset = CreateDataset({{}});
  EXPECT_DEATH(ComputeMeanVector(*dataset), "");
}

TEST_F(PcaTrainingTest, ComputeCovarianceMatrix_StandardValues) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f},
      {4.0f, 6.0f},
  });

  Eigen::MatrixXd cov = ComputeCovarianceMatrix(*dataset);
  ASSERT_EQ(cov.rows(), 2);
  ASSERT_EQ(cov.cols(), 2);

  EXPECT_NEAR(cov(0, 0), 4.5, 1e-9);
  EXPECT_NEAR(cov(0, 1), 6.0, 1e-9);
  EXPECT_NEAR(cov(1, 0), 6.0, 1e-9);
  EXPECT_NEAR(cov(1, 1), 8.0, 1e-9);
}

TEST_F(PcaTrainingTest, ComputeCovarianceMatrix_LargeDataset) {
  std::vector<std::vector<float>> points;
  for (int i = 0; i < 250; ++i) {
    points.push_back({1.0f, 1.0f});
  }
  for (int i = 0; i < 250; ++i) {
    points.push_back({2.0f, 2.0f});
  }
  auto dataset = CreateDataset(points);

  Eigen::MatrixXd cov = ComputeCovarianceMatrix(*dataset);
  ASSERT_EQ(cov.rows(), 2);
  ASSERT_EQ(cov.cols(), 2);

  const double expected_cov = 125.0 / 499.0;
  EXPECT_NEAR(cov(0, 0), expected_cov, 1e-9);
  EXPECT_NEAR(cov(0, 1), expected_cov, 1e-9);
  EXPECT_NEAR(cov(1, 0), expected_cov, 1e-9);
  EXPECT_NEAR(cov(1, 1), expected_cov, 1e-9);
}

TEST_F(PcaTrainingTest, ComputeCovarianceMatrix_SinglePoint_Dies) {
  auto dataset = CreateDataset({{1.0f, 2.0f}});
  EXPECT_DEATH(ComputeCovarianceMatrix(*dataset), "");
}

TEST_F(PcaTrainingTest, ComputeCovarianceMatrix_EmptyDataset_Dies) {
  auto dataset = CreateDataset({});
  EXPECT_DEATH(ComputeCovarianceMatrix(*dataset), "");
}

TEST_F(PcaTrainingTest, ComputeEigenvectors_Success_Partially) {
  Eigen::MatrixXd cov(2, 2);
  cov << 2.0, 0.0, 0.0, 1.0;

  ASSERT_OK_AND_ASSIGN(auto result, ComputeEigenvectors(cov, /*target_dim=*/1));

  ASSERT_EQ(result.eigenvalues.size(), 1);
  EXPECT_NEAR(result.eigenvalues(0), 2.0, 1e-9);

  ASSERT_EQ(result.eigenvectors.rows(), 1);
  ASSERT_EQ(result.eigenvectors.cols(), 2);

  EXPECT_NEAR(std::abs(result.eigenvectors(0, 0)), 1.0, 1e-9);
  EXPECT_NEAR(result.eigenvectors(0, 1), 0.0, 1e-9);
}

TEST_F(PcaTrainingTest, ComputeEigenvectors_Success_AllDimensions) {
  Eigen::MatrixXd cov(2, 2);
  cov << 2.0, 0.0, 0.0, 1.0;

  ASSERT_OK_AND_ASSIGN(auto result, ComputeEigenvectors(cov, /*target_dim=*/2));
  ASSERT_EQ(result.eigenvalues.size(), 2);

  EXPECT_NEAR(result.eigenvalues(0), 2.0, 1e-9);
  EXPECT_NEAR(result.eigenvalues(1), 1.0, 1e-9);

  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);

  EXPECT_NEAR(std::abs(result.eigenvectors(0, 0)), 1.0, 1e-9);
  EXPECT_NEAR(result.eigenvectors(0, 1), 0.0, 1e-9);

  EXPECT_NEAR(result.eigenvectors(1, 0), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(result.eigenvectors(1, 1)), 1.0, 1e-9);
}

TEST_F(PcaTrainingTest, ComputeEigenvectors_NonSquareMatrix_Fails) {
  Eigen::MatrixXd cov(2, 3);
  cov << 2.0, 0.0, 0.0, 0.0, 1.0, 0.0;

  EXPECT_THAT(ComputeEigenvectors(cov, 1).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Covariance matrix must be square."));
}

TEST_F(PcaTrainingTest, ComputeEigenvectors_TargetDimensionNegative_Fails) {
  Eigen::MatrixXd cov(2, 2);
  cov << 2.0, 0.0, 0.0, 1.0;

  EXPECT_THAT(ComputeEigenvectors(cov, -1).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The target dimension must be positive."));
}

TEST_F(PcaTrainingTest, ComputeEigenvectors_TargetDimensionTooLarge_Fails) {
  Eigen::MatrixXd cov(2, 2);
  cov << 2.0, 0.0, 0.0, 1.0;

  EXPECT_THAT(
      ComputeEigenvectors(cov, 3).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "The target dimension must be smaller than or equal to the "
               "dimensionality of the covariance matrix."));
}

TEST_F(PcaTrainingTest, PerformTruncation_Success) {
  EigenDecompositionResult result;
  result.eigenvalues.resize(2);
  result.eigenvalues << 3.0, 1.0;
  result.eigenvectors = Eigen::MatrixXf::Zero(2, 2);

  PcaProjectionConfig::Truncation config;
  config.set_significance_threshold(0.5f);
  config.set_truncation_threshold(1.0f);

  EXPECT_OK(PerformTruncation(config, result));

  EXPECT_EQ(result.eigenvalues.size(), 1);
  EXPECT_EQ(result.eigenvectors.rows(), 1);
}

TEST_F(PcaTrainingTest, PerformTruncation_WithAlignment) {
  EigenDecompositionResult result;
  result.eigenvalues = Eigen::VectorXd::Ones(40);
  result.eigenvectors = Eigen::MatrixXf::Zero(40, 5);

  PcaProjectionConfig::Truncation config;
  config.set_significance_threshold(0.01f);
  config.set_truncation_threshold(1.0f);
  config.set_alignment(32);

  EXPECT_OK(PerformTruncation(config, result));

  EXPECT_EQ(result.eigenvalues.size(), 32);
  EXPECT_EQ(result.eigenvectors.rows(), 32);
}

TEST_F(PcaTrainingTest, PerformTruncation_WithAlignmentTail) {
  EigenDecompositionResult result;
  result.eigenvalues = Eigen::VectorXd::Ones(400);
  result.eigenvectors = Eigen::MatrixXf::Zero(400, 5);

  // 0.3 keeps 120 dimensions, rounded up to 2 * 128 - 16 = 240 (since
  // 1 * 128 - 16 = 112 < 120).
  PcaProjectionConfig::Truncation config;
  config.set_significance_threshold(0.3f);
  config.set_truncation_threshold(1.0f);
  config.set_alignment(128);
  config.set_alignment_tail(16);

  EXPECT_OK(PerformTruncation(config, result));

  EXPECT_EQ(result.eigenvalues.size(), 240);
  EXPECT_EQ(result.eigenvectors.rows(), 240);
}

TEST_F(PcaTrainingTest, PerformTruncation_TruncationThresholdRejected) {
  EigenDecompositionResult result;
  result.eigenvalues.resize(2);
  result.eigenvalues << 3.0, 1.0;
  result.eigenvectors = Eigen::MatrixXf::Zero(2, 2);

  PcaProjectionConfig::Truncation config;
  config.set_significance_threshold(0.5f);
  config.set_truncation_threshold(0.4f);

  EXPECT_THAT(PerformTruncation(config, result),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "Target dimensionality is not sufficiently reduced."));

  EXPECT_EQ(result.eigenvalues.size(), 2);
  EXPECT_EQ(result.eigenvectors.rows(), 2);
}

TEST_F(PcaTrainingTest, PerformTruncation_NoTruncationRequired) {
  EigenDecompositionResult result;
  result.eigenvalues.resize(2);
  result.eigenvalues << 3.0, 1.0;
  result.eigenvectors = Eigen::MatrixXf::Zero(2, 2);

  PcaProjectionConfig::Truncation config;
  config.set_significance_threshold(0.8f);
  config.set_truncation_threshold(1.0f);

  EXPECT_THAT(PerformTruncation(config, result),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "Target dimensionality is not sufficiently reduced."));

  EXPECT_EQ(result.eigenvalues.size(), 2);
  EXPECT_EQ(result.eigenvectors.rows(), 2);
}

TEST_F(PcaTrainingTest, PerformTruncation_TooLargeThreshold_Dies) {
  EigenDecompositionResult result;
  result.eigenvalues.resize(2);
  result.eigenvalues << 3.0, 1.0;
  result.eigenvectors = Eigen::MatrixXf::Zero(2, 2);

  PcaProjectionConfig::Truncation config;
  config.set_significance_threshold(1.5f);
  config.set_truncation_threshold(1.0f);

  EXPECT_DEATH((void)PerformTruncation(config, result), "");
}

TEST_F(PcaTrainingTest, PerformTruncation_MismatchEigenDimensions_Dies) {
  EigenDecompositionResult result;
  result.eigenvalues.resize(2);
  result.eigenvalues << 3.0, 1.0;
  result.eigenvectors = Eigen::MatrixXf::Zero(1, 2);

  PcaProjectionConfig::Truncation config;
  config.set_significance_threshold(0.5f);
  config.set_truncation_threshold(1.0f);

  EXPECT_DEATH((void)PerformTruncation(config, result), "");
}

TEST_F(PcaTrainingTest, PerformTruncation_ZeroEigenvectors_Dies) {
  EigenDecompositionResult result;
  result.eigenvalues.resize(0);
  result.eigenvectors = Eigen::MatrixXf::Zero(0, 2);

  PcaProjectionConfig::Truncation config;
  config.set_significance_threshold(0.5f);
  config.set_truncation_threshold(1.0f);

  EXPECT_DEATH((void)PerformTruncation(config, result), "");
}

TEST_F(PcaTrainingTest, ValidateConfig_ValidDimension) {
  PcaProjectionConfig config;
  config.set_dimension(3);
  EXPECT_THAT(ValidateConfig(config, /*input_dim=*/5), IsOk());
}

TEST_F(PcaTrainingTest, ValidateConfig_ZeroInputDim_Fails) {
  PcaProjectionConfig config;
  config.set_dimension(1);
  EXPECT_THAT(ValidateConfig(config, /*input_dim=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The dataset must have a non-zero dimension."));
}

TEST_F(PcaTrainingTest, ValidateConfig_UnspecifiedMode_Fails) {
  PcaProjectionConfig config;
  EXPECT_THAT(
      ValidateConfig(config, /*input_dim=*/5),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "The PcaProjectionConfig must specify either dimension or truncation "
          "options."));
}

TEST_F(PcaTrainingTest, ValidateConfig_InvalidDimension_Fails) {
  PcaProjectionConfig config_zero;
  config_zero.set_dimension(0);
  EXPECT_THAT(ValidateConfig(config_zero, /*input_dim=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The target dimension must be positive."));

  PcaProjectionConfig config_large;
  config_large.set_dimension(10);
  EXPECT_THAT(ValidateConfig(config_large, /*input_dim=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The target dimension must be smaller than or equal to "
                       "the dimensionality of the dataset."));
}

TEST_F(PcaTrainingTest, ValidateConfig_InvalidTruncation_Fails) {
  PcaProjectionConfig config;
  auto* truncation = config.mutable_truncation();
  truncation->set_significance_threshold(0.0f);
  truncation->set_truncation_threshold(0.5f);
  EXPECT_THAT(ValidateConfig(config, /*input_dim=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The significance threshold must be in (0, 1]."));

  truncation->set_significance_threshold(0.8f);
  truncation->set_truncation_threshold(1.5f);
  EXPECT_THAT(ValidateConfig(config, /*input_dim=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The truncation threshold must be in (0, 1]."));

  truncation->set_truncation_threshold(0.5f);
  truncation->set_alignment(-1);
  EXPECT_THAT(ValidateConfig(config, /*input_dim=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The alignment must be non-negative."));
}

TEST_F(PcaTrainingTest, ValidateConfig_UnspecifiedMatrixRepresentation_Fails) {
  PcaProjectionConfig config;
  config.set_dimension(3);
  config.set_matrix_representation(MATRIX_REPRESENTATION_UNSPECIFIED);
  EXPECT_THAT(
      ValidateConfig(config, /*input_dim=*/5),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Matrix representation must be specified in PcaProjectionConfig."));
}

TEST_F(PcaTrainingTest, ValidateConfig_ValidSampling) {
  PcaProjectionConfig config;
  config.set_dimension(3);
  config.mutable_sampling()->set_sample_size(100);
  EXPECT_THAT(ValidateConfig(config, /*input_dim=*/5), IsOk());

  config.mutable_sampling()->set_samples_per_dimension(10);
  EXPECT_THAT(ValidateConfig(config, /*input_dim=*/5), IsOk());
}

TEST_F(PcaTrainingTest, ValidateConfig_InvalidSampling_Fails) {
  PcaProjectionConfig config;
  config.set_dimension(3);
  config.mutable_sampling();
  EXPECT_THAT(
      ValidateConfig(config, /*input_dim=*/5),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "The PcaProjectionConfig::Sampling must specify either sample_size "
          "or samples_per_dimension."));

  config.mutable_sampling()->set_sample_size(1);
  EXPECT_THAT(ValidateConfig(config, /*input_dim=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The sample size must be at least 2."));

  config.mutable_sampling()->set_samples_per_dimension(0);
  EXPECT_THAT(
      ValidateConfig(config, /*input_dim=*/5),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "The number of samples per dimension must be positive."));

  config.set_dimension(1);
  config.mutable_sampling()->set_samples_per_dimension(1);
  EXPECT_THAT(
      ValidateConfig(config, /*input_dim=*/1),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "The effective sample size (samples_per_dimension * input_dim) "
               "must be at least 2."));
}

TEST_F(PcaTrainingTest, TrainPca_Success) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(2);

  ASSERT_OK_AND_ASSIGN(auto projection_matrix, TrainPca(config, *dataset));
  EXPECT_EQ(projection_matrix.rows(), 2);
  EXPECT_EQ(projection_matrix.cols(), 3);
}

TEST_F(PcaTrainingTest, TrainPca_SingleVectorDataset_Fails) {
  auto dataset = CreateDataset({{1.0f, 2.0f, 3.0f}});

  PcaProjectionConfig config;
  config.set_dimension(2);

  EXPECT_THAT(TrainPca(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The dataset must contain at least 2 points for PCA."));
}

TEST_F(PcaTrainingTest, TrainPca_Truncation_Success) {
  auto dataset = CreateDataset({
      {10.0f, 0.1f, 0.0f},
      {20.0f, 0.2f, 0.0f},
      {30.0f, 0.3f, 0.0f},
      {40.0f, 0.4f, 0.0f},
  });

  PcaProjectionConfig config;
  auto* truncation = config.mutable_truncation();
  truncation->set_significance_threshold(0.5f);
  truncation->set_truncation_threshold(1.0f);

  ASSERT_OK_AND_ASSIGN(auto projection_matrix, TrainPca(config, *dataset));
  EXPECT_LT(projection_matrix.rows(), 3);
  EXPECT_EQ(projection_matrix.cols(), 3);
}

TEST_F(PcaTrainingTest, TrainPca_WithSampling_Success) {
  std::vector<std::vector<float>> points;
  points.reserve(50);
  for (int i = 0; i < 50; ++i) {
    points.push_back({static_cast<float>(i), 0.1f * i, 0.0f});
  }
  auto dataset = CreateDataset(points);

  PcaProjectionConfig config;
  config.set_dimension(2);
  config.mutable_sampling()->set_sample_size(10);
  config.mutable_sampling()->set_seed(123);

  ASSERT_OK_AND_ASSIGN(auto projection_matrix, TrainPca(config, *dataset));
  EXPECT_EQ(projection_matrix.rows(), 2);
  EXPECT_EQ(projection_matrix.cols(), 3);

  config.mutable_sampling()->set_samples_per_dimension(5);
  ASSERT_OK_AND_ASSIGN(auto projection_matrix_per_dim,
                       TrainPca(config, *dataset));
  EXPECT_EQ(projection_matrix_per_dim.rows(), 2);
  EXPECT_EQ(projection_matrix_per_dim.cols(), 3);
}

}  // namespace
}  // namespace ginger::internal
