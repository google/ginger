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

#include "ginger/pipnn/structured_random_matrix.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "Eigen/Core"

namespace ginger {
namespace {

using ::testing::FloatNear;

class SimpleStats {
 public:
  template <typename Iter>
  SimpleStats(Iter begin, Iter end) : sorted_(begin, end) {
    std::sort(sorted_.begin(), sorted_.end());
    double sum = 0.0;
    for (float v : sorted_) {
      sum += v;
    }
    mean_ = sorted_.empty() ? 0.0f : static_cast<float>(sum / sorted_.size());
    double sq_sum = 0.0;
    for (float v : sorted_) {
      sq_sum += (v - mean_) * (v - mean_);
    }
    stddev_ =
        sorted_.size() <= 1
            ? 0.0f
            : static_cast<float>(std::sqrt(sq_sum / (sorted_.size() - 1)));
  }

  explicit SimpleStats(const std::vector<float>& values)
      : SimpleStats(values.begin(), values.end()) {}

  float mean() const { return mean_; }
  float stddev() const { return stddev_; }
  float median() const { return Quantile(0.50); }
  float pc10() const { return Quantile(0.10); }
  float pc25() const { return Quantile(0.25); }
  float pc75() const { return Quantile(0.75); }
  float pc90() const { return Quantile(0.90); }

 private:
  float Quantile(double q) const {
    if (sorted_.empty()) return 0.0f;
    if (sorted_.size() == 1) return sorted_[0];
    double pos = q * (sorted_.size() - 1);
    size_t idx = static_cast<size_t>(pos);
    double frac = pos - idx;
    if (idx + 1 < sorted_.size()) {
      return static_cast<float>(sorted_[idx] * (1.0 - frac) +
                                sorted_[idx + 1] * frac);
    }
    return sorted_[idx];
  }

  std::vector<float> sorted_;
  float mean_;
  float stddev_;
};

class StructuredRandomOrthogonalMatrixTest : public ::testing::Test {
 protected:
  // Verify that the components of the output vector have roughly the right
  // quantiles.
  void RunRotateStatisticsTest(const int size) {
    StructuredRandomOrthogonalMatrix rotator(size);
    std::vector<float> vec(size, 0.0f);
    vec[0] = 1;
    rotator.Rotate(&vec);
    SimpleStats stats(vec);
    const float kExpectedStdDev = 1.0 / std::sqrt(size);
    EXPECT_THAT(stats.mean() / kExpectedStdDev,
                FloatNear(0.0, 3.0 / std::sqrt(size)));
    EXPECT_THAT(stats.stddev() / kExpectedStdDev, FloatNear(1, 0.1));
    const float ok_quantile_error = 3.0 / std::sqrt(size);
    EXPECT_THAT(stats.median() / kExpectedStdDev,
                FloatNear(0.0, ok_quantile_error));
    EXPECT_THAT(stats.pc10() / kExpectedStdDev,
                FloatNear(-1.28, ok_quantile_error));
    EXPECT_THAT(stats.pc25() / kExpectedStdDev,
                FloatNear(-0.68, ok_quantile_error));
    EXPECT_THAT(stats.pc90() / kExpectedStdDev,
                FloatNear(1.28, ok_quantile_error));
    EXPECT_THAT(stats.pc75() / kExpectedStdDev,
                FloatNear(0.68, ok_quantile_error));
  }
};

TEST_F(StructuredRandomOrthogonalMatrixTest, RoundUpSize) {
  EXPECT_EQ(StructuredRandomOrthogonalMatrix::RoundUpSize(0), 2);
  EXPECT_EQ(StructuredRandomOrthogonalMatrix::RoundUpSize(1), 2);
  EXPECT_EQ(StructuredRandomOrthogonalMatrix::RoundUpSize(2), 2);
  EXPECT_EQ(StructuredRandomOrthogonalMatrix::RoundUpSize(3), 4);
  EXPECT_EQ(StructuredRandomOrthogonalMatrix::RoundUpSize(4), 4);
  EXPECT_EQ(StructuredRandomOrthogonalMatrix::RoundUpSize(5), 8);
  EXPECT_EQ(StructuredRandomOrthogonalMatrix::RoundUpSize(128), 128);
  EXPECT_EQ(StructuredRandomOrthogonalMatrix::RoundUpSize(129), 256);
}

TEST_F(StructuredRandomOrthogonalMatrixTest, VectorRotateTest) {
  StructuredRandomOrthogonalMatrix rotator(4, /*seed=*/12345);
  std::vector<float> vec = {1.0f, -2.0f, 3.0f, 0.5f};
  Eigen::Map<Eigen::VectorXf> wrapped_vec(vec.data(), vec.size());
  const float old_norm = wrapped_vec.norm();
  rotator.Rotate(&vec);
  EXPECT_NEAR(wrapped_vec.norm(), old_norm, 1e-5f);
}

TEST_F(StructuredRandomOrthogonalMatrixTest, EigenRotateTest) {
  StructuredRandomOrthogonalMatrix rotator(8, /*seed=*/42);
  Eigen::VectorXf vec(8);
  vec << 1.0f, -2.0f, 3.0f, 0.5f, -1.5f, 2.5f, -0.5f, 1.2f;
  const float old_norm = vec.norm();
  rotator.Rotate(&vec);
  EXPECT_NEAR(vec.norm(), old_norm, 1e-5f);
}

TEST_F(StructuredRandomOrthogonalMatrixTest, RotateTestTiny) {
  StructuredRandomOrthogonalMatrix rotator(2, /*seed=*/54324);
  std::vector<float> vec = {1.0f, 0.5f};
  Eigen::Map<Eigen::VectorXf> wrapped_vec(vec.data(), vec.size());
  const float old_norm = wrapped_vec.norm();
  rotator.Rotate(&vec);
  EXPECT_NEAR(wrapped_vec.norm(), old_norm, 1e-5f);
}

TEST_F(StructuredRandomOrthogonalMatrixTest, RotateTestLarge) {
  StructuredRandomOrthogonalMatrix rotator(128, /*seed=*/999);
  std::vector<float> vec(128, 0.0f);
  vec[0] = 1.0f;
  rotator.Rotate(&vec);
  Eigen::Map<Eigen::VectorXf> wrapped_vec(vec.data(), vec.size());
  EXPECT_NEAR(wrapped_vec.norm(), 1.0f, 1e-5f);
}

TEST_F(StructuredRandomOrthogonalMatrixTest, Consistent) {
  StructuredRandomOrthogonalMatrix rotator1(4, /*seed=*/101);
  StructuredRandomOrthogonalMatrix rotator2(4, /*seed=*/101);
  Eigen::VectorXf vec1(4), vec2(4);
  vec1 << 1.0f, -2.0f, 3.0f, 0.5f;
  vec2 = vec1;
  rotator1.Rotate(&vec1);
  rotator2.Rotate(&vec2);
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(vec1[i], vec2[i]);
  }
}

TEST_F(StructuredRandomOrthogonalMatrixTest, UsesSeed) {
  StructuredRandomOrthogonalMatrix rotator1(4, /*seed=*/42);
  StructuredRandomOrthogonalMatrix rotator2(4, /*seed=*/223);
  Eigen::VectorXf vec1(4), vec2(4);
  vec1 << 1.0f, -2.0f, 3.0f, 0.5f;
  vec2 = vec1;
  rotator1.Rotate(&vec1);
  rotator2.Rotate(&vec2);
  EXPECT_GT((vec1 - vec2).norm(), 0.1f);
}

TEST_F(StructuredRandomOrthogonalMatrixTest, RotateTestStatistics64) {
  RunRotateStatisticsTest(64);
}

TEST_F(StructuredRandomOrthogonalMatrixTest, RotateTestStatistics128) {
  RunRotateStatisticsTest(128);
}

TEST_F(StructuredRandomOrthogonalMatrixTest, RotateTestStatistics256) {
  RunRotateStatisticsTest(256);
}

TEST_F(StructuredRandomOrthogonalMatrixTest, RotateTestStatistics1024) {
  RunRotateStatisticsTest(1024);
}

class StructuredRandomProjectionsTest : public ::testing::Test {
 protected:
  // Verify that the components of the output vector have roughly mean 0 and
  // roughly normal distribution.
  void RunProjectorStatisticsTest(const int size_out, const int size_in,
                                  StructuredRandomProjections::Mode mode) {
    StructuredRandomProjections projector(size_out, size_in, mode, 432432);
    Eigen::VectorXf vec_in = Eigen::VectorXf::Zero(size_in);
    vec_in(0) = 1;
    Eigen::VectorXf vec_out = projector.Project(vec_in);
    ASSERT_EQ(vec_out.size(), size_out);
    SimpleStats stats(vec_out.begin(), vec_out.end());
    const float sigma = stats.stddev();
    EXPECT_THAT(stats.mean() / sigma,
                FloatNear(0.0, 3.0 / std::sqrt(size_out)));
    const float ok_quantile_error = 6.0 / std::sqrt(size_out);
    EXPECT_THAT(stats.median() / sigma, FloatNear(0.0, ok_quantile_error));
    EXPECT_THAT(stats.pc10() / sigma, FloatNear(-1.28, ok_quantile_error));
    EXPECT_THAT(stats.pc90() / sigma, FloatNear(1.28, ok_quantile_error));
    EXPECT_THAT(stats.pc25() / sigma, FloatNear(-0.68, ok_quantile_error));
    EXPECT_THAT(stats.pc75() / sigma, FloatNear(0.68, ok_quantile_error));
  }

  // Verify that SimHash works as expected, i.e. the probability that the sign
  // of the projections of two vectors differ is proportional to the angle
  // between them.
  void CheckSimHash(StructuredRandomProjections::Mode mode) {
    const int size_in = 128;
    const int size_out = 1024;
    Eigen::VectorXf vec_in1 = Eigen::VectorXf::Zero(size_in);
    Eigen::VectorXf vec_in2 = Eigen::VectorXf::Zero(size_in);
    vec_in1(0) = 1.0f;
    vec_in2(0) = 1.0f;
    vec_in2(1) = -1.0f;
    StructuredRandomProjections projector(size_out, size_in, mode, 2159801);
    Eigen::VectorXf vec_out1 = projector.Project(vec_in1);
    Eigen::VectorXf vec_out2 = projector.Project(vec_in2);
    ASSERT_EQ(vec_out1.size(), size_out);
    ASSERT_EQ(vec_out2.size(), size_out);
    int num_match = 0;
    for (int i = 0; i < size_out; ++i) {
      if (vec_out1[i] * vec_out2[i] > 0) num_match++;
    }
    const double p = 1.0 - std::acos(std::sqrt(0.5)) / 3.14159265358979323846;
    const double ok_error = 3.0 * std::sqrt(size_out * p * (1.0 - p));
    EXPECT_GE(num_match, size_out * p - ok_error);
    EXPECT_LE(num_match, size_out * p + ok_error);
  }
};

TEST_F(StructuredRandomProjectionsTest, Stats_10k_4_explicit) {
  RunProjectorStatisticsTest(
      10000, 4, StructuredRandomProjections::Mode::EXPLICIT_MATRIX);
}

TEST_F(StructuredRandomProjectionsTest, Stats_10k_4_hd) {
  RunProjectorStatisticsTest(10000, 4,
                             StructuredRandomProjections::Mode::HDHDHD);
}

TEST_F(StructuredRandomProjectionsTest, Stats_10k_16_hd) {
  RunProjectorStatisticsTest(10000, 16,
                             StructuredRandomProjections::Mode::HDHDHD);
}

TEST_F(StructuredRandomProjectionsTest, Stats_10k_64_hd) {
  RunProjectorStatisticsTest(10000, 64,
                             StructuredRandomProjections::Mode::HDHDHD);
}

TEST_F(StructuredRandomProjectionsTest, Stats_10k_128_hd) {
  RunProjectorStatisticsTest(10000, 128,
                             StructuredRandomProjections::Mode::HDHDHD);
}

TEST_F(StructuredRandomProjectionsTest, Stats_10k_256_hd) {
  RunProjectorStatisticsTest(10000, 256,
                             StructuredRandomProjections::Mode::HDHDHD);
}

TEST_F(StructuredRandomProjectionsTest, Stats_10k_1024_hd) {
  RunProjectorStatisticsTest(10000, 1024,
                             StructuredRandomProjections::Mode::HDHDHD);
}

TEST_F(StructuredRandomProjectionsTest, Stats_8k_8k_hd) {
  RunProjectorStatisticsTest(8192, 8192,
                             StructuredRandomProjections::Mode::HDHDHD);
}

TEST_F(StructuredRandomProjectionsTest, ExplicitWorksForSimHash) {
  CheckSimHash(StructuredRandomProjections::Mode::EXPLICIT_MATRIX);
}

TEST_F(StructuredRandomProjectionsTest, HDHDHDWorksForSimHash) {
  CheckSimHash(StructuredRandomProjections::Mode::HDHDHD);
}

TEST_F(StructuredRandomProjectionsTest, ExplictConsistent) {
  const int size_in = 128;
  const int size_out = 1024;
  StructuredRandomProjections projector1(
      size_out, size_in, StructuredRandomProjections::Mode::EXPLICIT_MATRIX,
      2159801);
  StructuredRandomProjections projector2(
      size_out, size_in, StructuredRandomProjections::Mode::EXPLICIT_MATRIX,
      2159801);
  Eigen::VectorXf vec_in = Eigen::VectorXf::Zero(size_in);
  vec_in(0) = 1.0f;
  Eigen::VectorXf out1 = projector1.Project(vec_in);
  Eigen::VectorXf out2 = projector2.Project(vec_in);
  for (int i = 0; i < size_out; ++i) {
    EXPECT_NEAR(out1[i], out2[i], 1e-6f);
  }
}

TEST_F(StructuredRandomProjectionsTest, HDHDHDConsistent) {
  const int size_in = 128;
  const int size_out = 1024;
  StructuredRandomProjections projector1(
      size_out, size_in, StructuredRandomProjections::Mode::HDHDHD, 2159801);
  StructuredRandomProjections projector2(
      size_out, size_in, StructuredRandomProjections::Mode::HDHDHD, 2159801);
  Eigen::VectorXf vec_in = Eigen::VectorXf::Zero(size_in);
  vec_in(0) = 1.0f;
  Eigen::VectorXf out1 = projector1.Project(vec_in);
  Eigen::VectorXf out2 = projector2.Project(vec_in);
  for (int i = 0; i < size_out; ++i) {
    EXPECT_NEAR(out1[i], out2[i], 1e-6f);
  }
}

TEST_F(StructuredRandomProjectionsTest, NonFloatType) {
  const int size_in = 2;
  const int size_out = 10;
  StructuredRandomProjections projector(
      size_out, size_in, StructuredRandomProjections::Mode::EXPLICIT_MATRIX,
      2159801);
  Eigen::VectorXf float_vec_in = Eigen::VectorXf::Zero(size_in);
  float_vec_in(0) = 1.0f;
  Eigen::VectorXi int_vec_in = Eigen::VectorXi::Zero(size_in);
  int_vec_in(0) = 1;
  Eigen::VectorXf out_float = projector.Project(float_vec_in);
  Eigen::VectorXf out_int = projector.Project(int_vec_in);
  for (int i = 0; i < size_out; ++i) {
    EXPECT_NEAR(out_float[i], out_int[i], 1e-6f);
  }
}

}  // namespace
}  // namespace ginger
