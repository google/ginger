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

#include "ginger/projection/pca.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
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
#include "ginger/projection/pca_float.h"
#include "ginger/projection/pca_turboquant.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_reference_stub.h"

namespace ginger {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class PcaTest : public ::testing::Test {
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

// =============================================================================
// PcaProjection Tests
// =============================================================================

TEST_F(PcaTest, Project_Success_Projection) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(2);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 2);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);

  std::vector<float> query = {2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(auto projected, pca->Project(query));
  EXPECT_EQ(projected.size(), 2);
}

TEST_F(PcaTest, Project_Success_ExactFit) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(3);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 3);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);

  std::vector<float> query = {2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(auto projected, pca->Project(query));
  EXPECT_EQ(projected.size(), 3);
}

TEST_F(PcaTest, Project_Success_ZeroVarianceDimension) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 0.0f},
      {2.0f, 1.0f, 0.0f},
      {3.0f, 3.0f, 0.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(2);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 2);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);

  std::vector<float> query = {1.0f, 2.0f, 0.0f};
  ASSERT_OK_AND_ASSIGN(auto projected, pca->Project(query));
  EXPECT_EQ(projected.size(), 2);
}

TEST_F(PcaTest, Project_DominantAxis) {
  const float s2 = std::sqrt(2.0f);
  // Construct a dataset with clearly separated principal components:
  // V1 (largest): [1/sqrt(2), 1/sqrt(2), 0]
  // V2 (medium): [-1/sqrt(2), 1/sqrt(2), 0]
  // V3 (smallest): [0, 0, 1]
  auto dataset = CreateDataset({
      {10.0f / s2, 10.0f / s2, 0.0f},
      {-10.0f / s2, -10.0f / s2, 0.0f},
      {-2.0f / s2, 2.0f / s2, 0.0f},
      {2.0f / s2, -2.0f / s2, 0.0f},
      {0.0f, 0.0f, 0.1f},
      {0.0f, 0.0f, -0.1f},
  });

  PcaProjectionConfig config;
  config.set_dimension(1);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 1);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);

  // Projecting the FIRST principal component vector V1:
  // Since it is the dominant direction of variance, its projection should be
  // aligned with the dominant axis (length is 1.0).
  std::vector<float> v1 = {1.0f / s2, 1.0f / s2, 0.0f};
  ASSERT_OK_AND_ASSIGN(auto proj_v1, pca->Project(v1));
  ASSERT_EQ(proj_v1.size(), 1);

  // The projection should be perfectly aligned with the principal axis
  // (length is 1.0, accounting for sign ambiguity).
  EXPECT_NEAR(std::abs(proj_v1[0]), 1.0f, 1e-4f);
}

TEST_F(PcaTest, Project_Success_Truncation) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.4f);
  config.mutable_truncation()->set_truncation_threshold(1.0f);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 1);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);

  std::vector<float> query = {2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(auto projected, pca->Project(query));
  EXPECT_EQ(projected.size(), 1);
}

TEST_F(PcaTest, Create_TruncationSignificanceOne_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
      {0.0f, 0.0f, 0.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(1.0f);
  config.mutable_truncation()->set_truncation_threshold(1.0f);

  EXPECT_THAT(PcaProjection::Create(config, *dataset),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "Target dimensionality is not sufficiently reduced."));
}

TEST_F(PcaTest, Create_TruncationThresholdRejected_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.6f);
  config.mutable_truncation()->set_truncation_threshold(0.5f);

  EXPECT_THAT(PcaProjection::Create(config, *dataset),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "Target dimensionality is not sufficiently reduced."));
}

TEST_F(PcaTest, Project_Success_TruncationThresholdAllowed) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.4f);
  config.mutable_truncation()->set_truncation_threshold(0.5f);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 1);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);
}

TEST_F(PcaTest, Project_Success_TruncationAlignmentRoundUp) {
  std::vector<std::vector<float>> points;
  for (int i = 0; i < 20; ++i) {
    std::vector<float> pt(20, 0.0f);
    pt[i] = 1.0f;
    points.push_back(pt);
  }
  auto dataset = CreateDataset(points);

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.22f);
  config.mutable_truncation()->set_truncation_threshold(1.0f);
  config.mutable_truncation()->set_alignment(8);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 8);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);
}

TEST_F(PcaTest, Create_TruncationAlignmentCap_Fails) {
  std::vector<std::vector<float>> points;
  for (int i = 0; i < 10; ++i) {
    std::vector<float> pt(10, 0.0f);
    pt[i] = 1.0f;
    points.push_back(pt);
  }
  auto dataset = CreateDataset(points);

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.9f);
  config.mutable_truncation()->set_truncation_threshold(1.0f);
  config.mutable_truncation()->set_alignment(8);

  EXPECT_THAT(PcaProjection::Create(config, *dataset),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "Target dimensionality is not sufficiently reduced."));
}

TEST_F(PcaTest, Create_SingleVectorDataset_Fails) {
  auto dataset = CreateDataset({{1.0f, 2.0f, 3.0f}});

  PcaProjectionConfig config;
  config.set_dimension(2);

  EXPECT_THAT(PcaProjection::Create(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The dataset must contain at least 2 points for PCA."));
}

TEST_F(PcaTest, Create_TargetDimensionNull_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(0);

  EXPECT_THAT(PcaProjection::Create(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The target dimension must be positive."));
}

TEST_F(PcaTest, Create_TargetDimensionNegative_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(-1);

  EXPECT_THAT(PcaProjection::Create(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The target dimension must be positive."));
}

TEST_F(PcaTest, Create_TargetDimensionTooLarge_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(4);

  EXPECT_THAT(
      PcaProjection::Create(config, *dataset).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "The target dimension must be smaller than or equal to the "
               "dimensionality of the dataset."));
}

TEST_F(PcaTest, Create_SignificanceThresholdOutOfBounds_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(1.5f);
  config.mutable_truncation()->set_truncation_threshold(0.5f);

  EXPECT_THAT(PcaProjection::Create(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The significance threshold must be in (0, 1]."));
}

TEST_F(PcaTest, Create_TruncationThresholdOutOfBounds_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.9f);
  config.mutable_truncation()->set_truncation_threshold(-0.1f);

  EXPECT_THAT(PcaProjection::Create(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The truncation threshold must be in (0, 1]."));
}

TEST_F(PcaTest, Create_SignificanceThresholdZero_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.0f);
  config.mutable_truncation()->set_truncation_threshold(0.5f);

  EXPECT_THAT(PcaProjection::Create(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The significance threshold must be in (0, 1]."));
}

TEST_F(PcaTest, Create_TruncationThresholdZero_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.9f);
  config.mutable_truncation()->set_truncation_threshold(0.0f);

  EXPECT_THAT(PcaProjection::Create(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The truncation threshold must be in (0, 1]."));
}

TEST_F(PcaTest, Create_TruncationThresholdTooLarge_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.9f);
  config.mutable_truncation()->set_truncation_threshold(1.1f);

  EXPECT_THAT(PcaProjection::Create(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The truncation threshold must be in (0, 1]."));
}

TEST_F(PcaTest, Create_AlignmentNegative_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.mutable_truncation()->set_significance_threshold(0.9f);
  config.mutable_truncation()->set_truncation_threshold(0.5f);
  config.mutable_truncation()->set_alignment(-1);

  EXPECT_THAT(PcaProjection::Create(config, *dataset).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "The alignment must be non-negative."));
}

TEST_F(PcaTest, Create_UnspecifiedConfigMode_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;

  EXPECT_THAT(
      PcaProjection::Create(config, *dataset).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "The PcaProjectionConfig must specify either dimension or "
               "truncation options."));
}

TEST_F(PcaTest, Create_UnspecifiedMatrixRepresentation_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(2);
  config.set_matrix_representation(MATRIX_REPRESENTATION_UNSPECIFIED);

  EXPECT_THAT(
      PcaProjection::Create(config, *dataset).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Matrix representation must be specified in PcaProjectionConfig."));
}

TEST_F(PcaTest, Project_DimensionMismatch_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(2);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 2);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);

  std::vector<float> query = {2.0f, 3.0f};
  EXPECT_THAT(pca->Project(query).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Query dimensionality (2) does not match PCA input "
                       "dimensionality (3)."));
}

TEST_F(PcaTest, Project_PreallocatedBuffer_Success) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(2);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 2);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);

  std::vector<float> query = {2.0f, 3.0f, 4.0f};
  std::vector<float> output(2);
  EXPECT_THAT(pca->Project(query, absl::MakeSpan(output)), IsOk());

  ASSERT_OK_AND_ASSIGN(auto expected, pca->Project(query));
  EXPECT_EQ(output, expected);
}

TEST_F(PcaTest, Project_PreallocatedBuffer_OutputDimensionMismatch_Fails) {
  auto dataset = CreateDataset({
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
  });

  PcaProjectionConfig config;
  config.set_dimension(2);

  ASSERT_OK_AND_ASSIGN(auto pca, PcaProjection::Create(config, *dataset));
  EXPECT_EQ(pca->output_dim(), 2);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);

  std::vector<float> query = {2.0f, 3.0f, 4.0f};
  std::vector<float> output(3);
  EXPECT_THAT(pca->Project(query, absl::MakeSpan(output)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Output dimensionality (3) does not match PCA output "
                       "dimensionality (2)."));
}
// =============================================================================
// FloatPcaProjection Tests
// =============================================================================

TEST(FloatPcaTest, Create_Success) {
  Eigen::MatrixXf matrix(2, 3);
  matrix.setZero();
  ASSERT_OK_AND_ASSIGN(auto pca, FloatPcaProjection::Create(matrix));
  EXPECT_EQ(pca->output_dim(), 2);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_FLOAT);
}

TEST(FloatPcaTest, Create_ZeroDimension_Fails) {
  Eigen::MatrixXf matrix(0, 3);
  EXPECT_THAT(
      FloatPcaProjection::Create(matrix),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Projection matrix dimensions must be strictly positive."));

  Eigen::MatrixXf matrix2(3, 0);
  EXPECT_THAT(
      FloatPcaProjection::Create(matrix2),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Projection matrix dimensions must be strictly positive."));
}

TEST(FloatPcaTest, Create_OutputDimensionGreaterThanInputDimension_Fails) {
  Eigen::MatrixXf matrix(4, 3);
  matrix.setZero();
  EXPECT_THAT(FloatPcaProjection::Create(matrix),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Output dimension must be smaller than or equal to "
                       "input dimension."));
}

// =============================================================================
// TurboQuantPcaProjection Tests
// =============================================================================

class TurboQuantPcaTest : public ::testing::Test {
 protected:
  void SetUp() override { rng_.seed(42); }

  Eigen::MatrixXf GenerateRandomMatrix(int rows, int cols) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Eigen::MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        mat(i, j) = dist(rng_);
      }
    }
    return mat;
  }

  std::vector<float> GenerateRandomVector(int size) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(size);
    for (int i = 0; i < size; ++i) {
      vec[i] = dist(rng_);
    }
    return vec;
  }

  std::mt19937 rng_;

 private:
  ginger::internal::ParallelSchedulerReference scheduler_{
      /*num_workers=*/4};
};

TEST_F(TurboQuantPcaTest, Create_Success) {
  Eigen::MatrixXf matrix(2, 3);
  matrix.setZero();
  ASSERT_OK_AND_ASSIGN(auto pca, TurboQuantPcaProjection::Create(matrix));
  EXPECT_EQ(pca->output_dim(), 2);
  EXPECT_EQ(pca->matrix_representation(), MATRIX_REPRESENTATION_TURBOQUANT);
}

TEST_F(TurboQuantPcaTest, Create_ZeroDimension_Fails) {
  Eigen::MatrixXf matrix(0, 3);
  EXPECT_THAT(
      TurboQuantPcaProjection::Create(matrix),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Projection matrix dimensions must be strictly positive."));

  Eigen::MatrixXf matrix2(3, 0);
  EXPECT_THAT(
      TurboQuantPcaProjection::Create(matrix2),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Projection matrix dimensions must be strictly positive."));
}

TEST_F(TurboQuantPcaTest, Project_DimensionMismatch_Fails) {
  Eigen::MatrixXf matrix = GenerateRandomMatrix(4, 16);
  ASSERT_OK_AND_ASSIGN(auto pca, TurboQuantPcaProjection::Create(matrix));

  std::vector<float> short_query(8, 0.0f);
  EXPECT_THAT(pca->Project(short_query).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Query dimensionality (8) does not match PCA input "
                       "dimensionality (16)."));
}

TEST_F(TurboQuantPcaTest, Project_OutputDimensionMismatch_Fails) {
  Eigen::MatrixXf matrix = GenerateRandomMatrix(4, 16);
  ASSERT_OK_AND_ASSIGN(auto pca, TurboQuantPcaProjection::Create(matrix));

  std::vector<float> query(16, 0.0f);
  std::vector<float> wrong_output(6, 0.0f);
  EXPECT_THAT(pca->Project(query, absl::MakeSpan(wrong_output)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Output dimensionality (6) does not match PCA output "
                       "dimensionality (4)."));
}

TEST_F(TurboQuantPcaTest, TurboQuantProjection_Precision_NonMultiples_15x765) {
  const int rows = 15;
  const int cols = 765;
  Eigen::MatrixXf W = GenerateRandomMatrix(rows, cols);
  std::vector<float> query = GenerateRandomVector(cols);

  ASSERT_OK_AND_ASSIGN(auto proj, TurboQuantPcaProjection::Create(W));
  ASSERT_OK_AND_ASSIGN(std::vector<float> result, proj->Project(query));
  EXPECT_EQ(result.size(), rows);

  Eigen::Map<const Eigen::VectorXf> query_map(query.data(), query.size());
  Eigen::VectorXf expected = W * query_map;

  for (int i = 0; i < rows; ++i) {
    EXPECT_NEAR(result[i], expected(i), 3.5f);
  }
}

TEST_F(TurboQuantPcaTest, SafeScaling_ZeroVectors) {
  const int rows = 10;
  const int cols = 30;
  Eigen::MatrixXf W = Eigen::MatrixXf::Zero(rows, cols);
  std::vector<float> query(cols, 0.0f);

  ASSERT_OK_AND_ASSIGN(auto tq_proj, TurboQuantPcaProjection::Create(W));
  ASSERT_OK_AND_ASSIGN(auto tq_res, tq_proj->Project(query));
  for (int i = 0; i < rows; ++i) {
    EXPECT_FALSE(std::isnan(tq_res[i]));
    EXPECT_FLOAT_EQ(tq_res[i], 0.0f);
  }
}

TEST_F(TurboQuantPcaTest, SafeScaling_ExtremeValues) {
  const int rows = 8;
  const int cols = 16;
  Eigen::MatrixXf W = GenerateRandomMatrix(rows, cols) * 1e25f;
  std::vector<float> query = GenerateRandomVector(cols);
  for (auto& val : query) val *= 1e-25f;

  ASSERT_OK_AND_ASSIGN(auto tq_proj, TurboQuantPcaProjection::Create(W));
  ASSERT_OK_AND_ASSIGN(auto tq_res, tq_proj->Project(query));
  for (int i = 0; i < rows; ++i) {
    EXPECT_FALSE(std::isnan(tq_res[i]));
    EXPECT_FALSE(std::isinf(tq_res[i]));
  }
}

TEST_F(TurboQuantPcaTest, CreateFromPackedWeights_Success) {
  hwy::AlignedVector<uint8_t> weights(1024);
  hwy::AlignedVector<float> scales(16, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto pca, TurboQuantPcaProjection::Create(128, 16, std::move(weights),
                                                std::move(scales)));
  EXPECT_EQ(pca->input_dim(), 128);
  EXPECT_EQ(pca->output_dim(), 16);
}

TEST_F(TurboQuantPcaTest, CreateFromPackedWeights_InvalidDimensions_Fails) {
  hwy::AlignedVector<uint8_t> weights(1024);
  hwy::AlignedVector<float> scales(16, 1.0f);

  EXPECT_THAT(
      TurboQuantPcaProjection::Create(0, 16, hwy::AlignedVector<uint8_t>(),
                                      hwy::AlignedVector<float>()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Projection dimensions must be strictly positive.")));

  EXPECT_THAT(
      TurboQuantPcaProjection::Create(128, 0, hwy::AlignedVector<uint8_t>(),
                                      hwy::AlignedVector<float>()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Projection dimensions must be strictly positive.")));

  EXPECT_THAT(
      TurboQuantPcaProjection::Create(16, 32, hwy::AlignedVector<uint8_t>(),
                                      hwy::AlignedVector<float>()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Output dimension must be smaller than or equal to "
                         "input dimension.")));
}

TEST_F(TurboQuantPcaTest, CreateFromPackedWeights_WeightsSizeMismatch_Fails) {
  hwy::AlignedVector<uint8_t> bad_weights(100);
  hwy::AlignedVector<float> scales(16, 1.0f);

  EXPECT_THAT(TurboQuantPcaProjection::Create(128, 16, std::move(bad_weights),
                                              std::move(scales)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Packed weights size mismatch")));
}

TEST_F(TurboQuantPcaTest, CreateFromPackedWeights_ScalesSizeMismatch_Fails) {
  hwy::AlignedVector<uint8_t> weights(1024);
  hwy::AlignedVector<float> bad_scales(10, 1.0f);

  EXPECT_THAT(TurboQuantPcaProjection::Create(128, 16, std::move(weights),
                                              std::move(bad_scales)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Row inverse scales size mismatch")));
}

}  // namespace
}  // namespace ginger
