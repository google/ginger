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

// Classes for structured-random rotation or projection using alternating
// Hadamard transforms and diagonal +1/-1 random matrices (i.e. HDHDHD).

#ifndef GINGER_PIPNN_STRUCTURED_RANDOM_MATRIX_H_
#define GINGER_PIPNN_STRUCTURED_RANDOM_MATRIX_H_

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <variant>
#include <vector>

#include "absl/log/absl_check.h"
#include "Eigen/Core"
#include "Eigen/Dense"

namespace ginger {

// This class implements an HDHDHD pseudo-random rotation where "H" is a Fast
// Hadamard Transform and each D is a diagonal matrix with random +1/-1 entries.
class StructuredRandomOrthogonalMatrix {
 public:
  // Rounds a non-negative dimensionality up to the next supported transform
  // size (a power of 2, at least 2).
  static int RoundUpSize(int dim) {
    ABSL_CHECK_GE(dim, 0);
    if (dim <= 2) return 2;
    return static_cast<int>(std::bit_ceil(static_cast<unsigned int>(dim)));
  }

  // Use given seed for creating the random diagonal matrices.
  // Requires dimension to be a power of 2 at least 2, i.e.
  // Precondition: dimension == RoundUpSize(dimension).
  explicit StructuredRandomOrthogonalMatrix(int dimension,
                                            uint32_t seed = kDefaultSeed);

  // This applies HDHDHD to a vector. The input vector is modified in-place. The
  // L2-norm of the returned vector equals the L2-norm of the input vector.
  // Precondition: size == dimension_.
  void Rotate(float* data, int size) const;
  void Rotate(std::vector<float>* vec) const {
    Rotate(vec->data(), static_cast<int>(vec->size()));
  }
  void Rotate(Eigen::VectorXf* vec) const {
    Rotate(vec->data(), static_cast<int>(vec->innerSize()));
  }

 private:
  // Number of HDs. Two papers indicate that 3 is correct for this:
  //   - The TripleSpin paper: https://arxiv.org/abs/1605.09046 .
  //   - The cross polytope hash paper: https://arxiv.org/pdf/1509.02897.pdf .
  static constexpr int kNumSteps = 3;
  static constexpr uint32_t kDefaultSeed = 54324;

  void Init(std::mt19937& gen);

  // The diagonals of the three random +1/-1 "D" matrices.
  std::vector<Eigen::VectorXf> rand_diagonals_;
  // The size of the implicit "H" and "D" matrices.
  int dimension_;
};

// This class implicitly multiplies vectors by a dimension_to by
// dimension_from structured random "projection" matrix M. Matrix M acts
// similarly to a matrix that has components that are independent normally-
// distributed random variables with mean 0 and standard deviation sigma. The
// value of sigma is an implementation detail that may be different for
// different StructuredRandomProjections objects. Applications that care about
// the norm of the returned vector should normalize it.
// A StructuredRandomProjections is a good replacement for a matrix of random
// normals in e.g. SimHash, cross polytope LSH, or the JL-lemma.
//
// The current implementation uses an actual matrix of independent normally-
// distributed random variables if dimension_to is sufficiently small (<50ish).
// Otherwise it uses one or more StructuredRandomOrthogonalMatrix(s). Each
// StructuredRandomOrthogonalMatrix produces
// k=StructuredRandomOrthogonalMatrix::RoundUpSize(dimension_from) components so
// it uses ceiling(dimension_to/k) StructuredRandomOrthogonalMatrix(s).
// See the TripleSpin paper (https://arxiv.org/abs/1605.09046) for a proof that
// this works for e.g. cross polytope sketching.
//
// All StructuredRandomProjections objects that are passed the same constructor
// parameters will behave identically up to floating-point rounding. This
// property is true even for instances on different machines (but using the
// same binary). Implementors: see comment at the top of the .cc file for more.
class StructuredRandomProjections {
 public:
  enum class Mode {
    EXPLICIT_MATRIX,  // Matrix of independent normals
    HDHDHD,           // StructuredRandomOrthogonalMatrixs
    AUTO  // Choose one of the above automatically based on dimensions.
  };

  // Requires dimension_from > 0 and dimension_to > 0.
  StructuredRandomProjections(int dimension_to, int dimension_from, Mode mode,
                              uint32_t seed);
  StructuredRandomProjections(int dimension_to, int dimension_from, Mode mode)
      : StructuredRandomProjections(dimension_to, dimension_from, mode,
                                    kDefaultSeed) {}
  StructuredRandomProjections(int dimension_to, int dimension_from)
      : StructuredRandomProjections(dimension_to, dimension_from, Mode::AUTO,
                                    kDefaultSeed) {}

  // Computes M * vec.
  template <typename Derived>
  Eigen::VectorXf Project(const Eigen::DenseBase<Derived>& vec) const {
    ABSL_CHECK_EQ(vec.innerSize(), dimension_from_);
    if (!StructuredMode()) {
      return std::get<Eigen::MatrixXf>(projector_) * vec.template cast<float>();
    }
    Eigen::VectorXf result_vec(dimension_to_);
    Eigen::VectorXf work_vec(rotator_size_);
    int start_index = 0;
    const float scale =
        std::sqrt(static_cast<float>(rotator_size_) / dimension_to_);
    for (const StructuredRandomOrthogonalMatrix& rotator :
         std::get<std::vector<StructuredRandomOrthogonalMatrix>>(projector_)) {
      work_vec.setZero();
      work_vec.head(dimension_from_) = vec.template cast<float>();
      rotator.Rotate(&work_vec);
      for (int i = 0; i < rotator_size_ && i + start_index < dimension_to_;
           ++i) {
        result_vec[i + start_index] = work_vec[i] * scale;
      }
      start_index += rotator_size_;
    }
    return result_vec;
  }

 private:
  static constexpr uint32_t kDefaultSeed = 136317631;

  void Init(Mode mode, uint32_t seed);
  bool StructuredMode() const {
    return std::holds_alternative<
        std::vector<StructuredRandomOrthogonalMatrix>>(projector_);
  }

  int dimension_to_;
  int dimension_from_;
  int rotator_size_;
  // Either a single projection matrix for EXPLICIT_MATRIX mode, or a vector of
  // rotators for HDHDHD mode.
  std::variant<Eigen::MatrixXf, std::vector<StructuredRandomOrthogonalMatrix>>
      projector_;
};

}  // namespace ginger

#endif  // GINGER_PIPNN_STRUCTURED_RANDOM_MATRIX_H_
