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
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "ginger/turboquant/rotation/walsh_hadamard.h"

namespace ginger {

StructuredRandomOrthogonalMatrix::StructuredRandomOrthogonalMatrix(
    const int dimension, uint32_t seed)
    : dimension_(dimension) {
  std::mt19937 gen(seed);
  Init(gen);
}

void StructuredRandomOrthogonalMatrix::Init(std::mt19937& gen) {
  ABSL_CHECK_EQ(dimension_, RoundUpSize(dimension_));
  const double sqrt_dim = std::sqrt(dimension_);
  std::uniform_int_distribution<int> dist(0, 1);
  rand_diagonals_.clear();
  rand_diagonals_.reserve(kNumSteps);
  for (int i = 0; i < kNumSteps; ++i) {
    rand_diagonals_.emplace_back(dimension_);
    for (int j = 0; j < dimension_; ++j) {
      // Diagonal entries are +/- 1/sqrt_dim to correct for the fast Hadamard
      // transform's lack of normalization.
      const double plus_minus_one = dist(gen) ? 1.0 : -1.0;
      rand_diagonals_[i][j] = static_cast<float>(plus_minus_one / sqrt_dim);
    }
  }
}

void StructuredRandomOrthogonalMatrix::Rotate(float* data,
                                              const int size) const {
  ABSL_CHECK_EQ(dimension_, size);
  for (int i = 0; i < kNumSteps; ++i) {
    // D step: multiply by diagonal matrix.
    Eigen::Map<Eigen::RowVectorXf> wrapped_vec(data, size);
    wrapped_vec *= rand_diagonals_[i].asDiagonal();
    // H step: Fast Walsh-Hadamard Transform.
    ABSL_CHECK_OK(WalshHadamardTransform(absl::MakeSpan(data, size)));
  }
}

StructuredRandomProjections::StructuredRandomProjections(int dimension_to,
                                                         int dimension_from,
                                                         Mode mode,
                                                         uint32_t seed)
    : dimension_to_(dimension_to), dimension_from_(dimension_from) {
  Init(mode, seed);
}

void StructuredRandomProjections::Init(Mode mode, uint32_t seed) {
  ABSL_CHECK_GT(dimension_to_, 0);
  ABSL_CHECK_GT(dimension_from_, 0);
  // Minimum dimension of rotator that acts sufficiently normal-like.
  const int kMinRotatorSize = 1 << 7;
  rotator_size_ = StructuredRandomOrthogonalMatrix::RoundUpSize(
      std::max(kMinRotatorSize, dimension_from_));
  const int num_rotators = (dimension_to_ + rotator_size_ - 1) / rotator_size_;
  const bool use_explicit_matrix = [&]() {
    switch (mode) {
      case Mode::EXPLICIT_MATRIX:
        return true;
      case Mode::HDHDHD:
        return false;
      case Mode::AUTO: {
        const int64_t kMagicRuntimeConstant = 6;
        const int64_t log2_rotator =
            std::bit_width(static_cast<unsigned int>(rotator_size_)) - 1;
        const int64_t rotators_runtime =
            kMagicRuntimeConstant * num_rotators * rotator_size_ * log2_rotator;
        const int64_t naive_runtime =
            static_cast<int64_t>(dimension_to_) * dimension_from_;
        return naive_runtime <= rotators_runtime;
      }
    }
  }();

  std::mt19937 gen(seed);
  if (use_explicit_matrix) {
    Eigen::MatrixXf& proj =
        projector_.emplace<Eigen::MatrixXf>(dimension_to_, dimension_from_);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimension_to_));
    for (int i = 0; i < dimension_to_; ++i) {
      for (int j = 0; j < dimension_from_; ++j) {
        proj(i, j) = norm_dist(gen) * scale;
      }
    }
  } else {
    std::vector<StructuredRandomOrthogonalMatrix>& rotators =
        projector_.emplace<std::vector<StructuredRandomOrthogonalMatrix>>();
    rotators.reserve(num_rotators);
    for (int i = 0; i < num_rotators; ++i) {
      rotators.emplace_back(rotator_size_, gen());
    }
  }
}

}  // namespace ginger
