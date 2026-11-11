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

#include "ginger/turboquant/rotation/rotation.h"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "Eigen/Core"

namespace ginger {

using ::absl_testing::StatusIs;
using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(HadamardAndMatrixRotationTest, DeterministicWithSeed) {
  const int dim = 3 * 8;
  const int seed = 123;

  HadamardAndMatrixRotation rotator1(dim, seed);
  std::vector<float> input1(dim);
  std::iota(input1.begin(), input1.end(), 5.0f);
  std::vector<float> output1(dim);
  EXPECT_OK(rotator1.Rotate(absl::MakeSpan(input1), absl::MakeSpan(output1)));

  HadamardAndMatrixRotation rotator2(dim, seed);
  std::vector<float> input2(dim);
  std::iota(input2.begin(), input2.end(), 5.0f);
  std::vector<float> output2(dim);
  EXPECT_OK(rotator2.Rotate(absl::MakeSpan(input2), absl::MakeSpan(output2)));

  EXPECT_EQ(output1, output2);
}

// Pins the rotation for a fixed seed, so that changes in the random number
// generation, its order, or Eigen's QR decomposition are caught.
TEST(HadamardAndMatrixRotationTest, GoldenOutput) {
  const int dim = 3 * 8;
  HadamardAndMatrixRotation rotator(dim, /*seed=*/123);
  std::vector<float> input(dim);
  std::iota(input.begin(), input.end(), 1.0f);
  std::vector<float> output(dim);
  ASSERT_OK(rotator.Rotate(absl::MakeSpan(input), absl::MakeSpan(output)));
  const std::vector<float> expected = {
      -65.6460648f, 4.78501558f, 9.57003117f,  0.0f,         19.1400623f,
      0.0f,         0.0f,        0.0f,         208.344574f,  -2.45905566f,
      -4.91811132f, 0.0f,        -9.83622265f, 0.0f,         0.0f,
      0.0f,         258.57132f,  -10.7264471f, -21.4528942f, 0.0f,
      -42.9057884f, 0.0f,        0.0f,         0.0f};
  EXPECT_THAT(output, Pointwise(FloatNear(1e-4f), expected));
}

TEST(HadamardAndMatrixRotationTest, DifferentSeedDifferentRotation) {
  const int dim = 3 * 4;
  std::vector<float> input(dim);
  std::iota(input.begin(), input.end(), 1.0f);

  HadamardAndMatrixRotation rotator1(dim, 101);
  std::vector<float> output1(dim);
  EXPECT_OK(rotator1.Rotate(absl::MakeSpan(input), absl::MakeSpan(output1)));

  HadamardAndMatrixRotation rotator2(dim, 202);
  std::vector<float> output2(dim);
  EXPECT_OK(rotator2.Rotate(absl::MakeSpan(input), absl::MakeSpan(output2)));

  EXPECT_NE(output1, output2);
}

// This test exercises the subspace_dimensionality_ = 1 case, where we can't
// call into the Hadamard transform.
TEST(HadamardAndMatrixRotationTest, Dimension13) {
  const int dim = 13;
  std::vector<float> input(dim);
  std::iota(input.begin(), input.end(), 1.0f);

  HadamardAndMatrixRotation rotator(dim, 101);
  std::vector<float> output(dim);
  EXPECT_OK(rotator.Rotate(absl::MakeSpan(input), absl::MakeSpan(output)));
  EXPECT_EQ(output.size(), dim);
}

TEST(HadamardAndMatrixRotationTest, NormBlowsUpByFactorSqrtDim) {
  const int dim = 3 * 16;
  const int seed = 123;

  std::vector<float> input(dim);
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0, 1.0);
  for (int i = 0; i < dim; ++i) {
    input[i] = dist(gen);
  }

  float initial_norm = 0.0f;
  for (float value : input) {
    initial_norm += value * value;
  }
  initial_norm = std::sqrt(initial_norm);

  std::vector<float> output(dim);
  HadamardAndMatrixRotation rotator(dim, seed);
  EXPECT_OK(rotator.Rotate(absl::MakeSpan(input), absl::MakeSpan(output)));

  float rotated_norm = 0.0f;
  for (float value : output) {
    rotated_norm += value * value;
  }
  rotated_norm = std::sqrt(rotated_norm);

  EXPECT_NEAR(rotated_norm / initial_norm, std::sqrt(static_cast<float>(dim)),
              1e-5);
}

TEST(RotationMatrixTest, EigenVsHandrolled) {
  const int subspace_dim = 8;
  const int num_subspaces = 5;
  const int dim = subspace_dim * num_subspaces;

  HadamardAndMatrixRotation rotator(dim, 555);
  Eigen::MatrixXf rotation_matrix = rotator.GetRotationMatrix();

  std::mt19937 gen(556);
  std::normal_distribution<float> dist(0.0, 1.0);
  std::vector<float> input(dim);
  for (int i = 0; i < dim; ++i) {
    input[i] = dist(gen);
  }
  std::vector<float> output_coordinates_handrolled(dim, 0.0f);

  for (size_t i = 0; i < subspace_dim; i++) {
    for (size_t j = 0; j < num_subspaces; j++) {
      for (size_t k = 0; k < num_subspaces; k++) {
        output_coordinates_handrolled[i + j * subspace_dim] +=
            rotation_matrix(k, j) * input[i + k * subspace_dim];
      }
    }
  }

  std::vector<float> output_coordinates_eigen(dim);

  Eigen::Map<Eigen::MatrixXf> input_map(input.data(), subspace_dim,
                                        num_subspaces);
  Eigen::Map<Eigen::MatrixXf> output_coordinates_map(
      output_coordinates_eigen.data(), subspace_dim, num_subspaces);
  output_coordinates_map = input_map * rotation_matrix;

  EXPECT_THAT(output_coordinates_handrolled,
              Pointwise(FloatNear(1e-5), output_coordinates_eigen));
}

TEST(HadamardAndMatrixRotationTest, InPlaceRotation) {
  for (int dim : {16, 64, 48, 768}) {
    HadamardAndMatrixRotation rotator(dim, 123);
    std::vector<float> input(dim);
    std::iota(input.begin(), input.end(), 1.0f);
    std::vector<float> input_inplace = input;

    std::vector<float> output_out_of_place(dim);
    EXPECT_OK(rotator.Rotate(absl::MakeSpan(input),
                             absl::MakeSpan(output_out_of_place)));
    EXPECT_OK(rotator.Rotate(absl::MakeSpan(input_inplace),
                             absl::MakeSpan(input_inplace)));

    EXPECT_THAT(input_inplace, Pointwise(FloatNear(1e-5), output_out_of_place));
  }
}

TEST(HadamardAndMatrixRotationTest, InPlaceOverloadAndR3SIMD) {
  const int dim = 3 * 8;  // num_subspaces = 3
  const int seed = 42;

  std::vector<float> input(dim);
  std::iota(input.begin(), input.end(), 1.0f);

  HadamardAndMatrixRotation rotator(dim, seed);

  std::vector<float> out_of_place_output(dim);
  std::vector<float> in1 = input;
  EXPECT_OK(
      rotator.Rotate(absl::MakeSpan(in1), absl::MakeSpan(out_of_place_output)));

  std::vector<float> in_place_output = input;
  EXPECT_OK(rotator.Rotate(absl::MakeSpan(in_place_output)));

  EXPECT_THAT(in_place_output, Pointwise(FloatNear(1e-5), out_of_place_output));
}

namespace {

// Covers all code paths of `Rotate` and `InverseRotate`: num_subspaces = 1
// (powers of two), num_subspaces = 3 (SIMD mixing), other odd num_subspaces
// (Eigen mixing), and subspace_dimensionality = 1 (no Hadamard transform).
constexpr int kInverseTestDimensionalities[] = {
    1, 2, 3, 6, 8, 13, 16, 24, 40, 127, 128, 768, 1024, 3072};

std::vector<float> RandomGaussianVector(int dim, int seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0, 1.0);
  std::vector<float> values(dim);
  for (float& value : values) {
    value = dist(gen);
  }
  return values;
}

}  // namespace

TEST(HadamardAndMatrixRotationTest, InverseRotateUndoesRotate) {
  for (int dim : kInverseTestDimensionalities) {
    SCOPED_TRACE(testing::Message() << "dim = " << dim);
    HadamardAndMatrixRotation rotator(dim, 123);
    const std::vector<float> input = RandomGaussianVector(dim, 456);

    std::vector<float> scratch = input;
    std::vector<float> rotated(dim);
    ASSERT_OK(rotator.Rotate(absl::MakeSpan(scratch), absl::MakeSpan(rotated)));
    std::vector<float> recovered(dim);
    ASSERT_OK(rotator.InverseRotate(absl::MakeSpan(rotated),
                                    absl::MakeSpan(recovered)));

    EXPECT_THAT(recovered, Pointwise(FloatNear(1e-4), input));
  }
}

TEST(HadamardAndMatrixRotationTest, RotateUndoesInverseRotate) {
  for (int dim : kInverseTestDimensionalities) {
    SCOPED_TRACE(testing::Message() << "dim = " << dim);
    HadamardAndMatrixRotation rotator(dim, 123);
    const std::vector<float> input = RandomGaussianVector(dim, 789);

    std::vector<float> scratch = input;
    std::vector<float> inverse_rotated(dim);
    ASSERT_OK(rotator.InverseRotate(absl::MakeSpan(scratch),
                                    absl::MakeSpan(inverse_rotated)));
    std::vector<float> recovered(dim);
    ASSERT_OK(rotator.Rotate(absl::MakeSpan(inverse_rotated),
                             absl::MakeSpan(recovered)));

    EXPECT_THAT(recovered, Pointwise(FloatNear(1e-4), input));
  }
}

TEST(HadamardAndMatrixRotationTest, InPlaceInverseRotation) {
  for (int dim : kInverseTestDimensionalities) {
    SCOPED_TRACE(testing::Message() << "dim = " << dim);
    HadamardAndMatrixRotation rotator(dim, 123);
    std::vector<float> input = RandomGaussianVector(dim, 42);
    std::vector<float> in_place_output = input;

    std::vector<float> out_of_place_output(dim);
    ASSERT_OK(rotator.InverseRotate(absl::MakeSpan(input),
                                    absl::MakeSpan(out_of_place_output)));
    ASSERT_OK(rotator.InverseRotate(absl::MakeSpan(in_place_output)));

    EXPECT_THAT(in_place_output,
                Pointwise(FloatNear(1e-6), out_of_place_output));
  }
}

TEST(HadamardAndMatrixRotationTest, InverseRotateFailsWithInvalidSize) {
  const int dim = 3 * 8;
  HadamardAndMatrixRotation rotator(dim, 123);
  std::vector<float> valid(dim);
  std::vector<float> invalid(dim - 1);

  EXPECT_THAT(
      rotator.InverseRotate(absl::MakeSpan(invalid), absl::MakeSpan(valid)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      rotator.InverseRotate(absl::MakeSpan(valid), absl::MakeSpan(invalid)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

namespace {

// Returns `rotator.Rotate(input)` without modifying `input`.
std::vector<float> RotateCopy(const HadamardAndMatrixRotation& rotator,
                              std::vector<float> input) {
  std::vector<float> output(input.size());
  ABSL_CHECK_OK(rotator.Rotate(absl::MakeSpan(input), absl::MakeSpan(output)));
  return output;
}

std::vector<float> BasisVector(int dim, int i) {
  std::vector<float> e(dim, 0.0f);
  e[i] = 1.0f;
  return e;
}

}  // namespace

TEST(HadamardAndMatrixRotationTest,
     RandomPermutationIsDeterministicAndApplied) {
  for (int dim : kInverseTestDimensionalities) {
    SCOPED_TRACE(testing::Message() << "dim = " << dim);
    HadamardAndMatrixRotation rotator(dim, 123, /*permute_coordinates=*/true);
    const std::vector<float> input = RandomGaussianVector(dim, 456);

    // Deterministic for the same seed.
    HadamardAndMatrixRotation same_seed(dim, 123, /*permute_coordinates=*/true);
    EXPECT_EQ(RotateCopy(same_seed, input), RotateCopy(rotator, input));

    // The permutation changes the rotation unless the dimensionality is a
    // power of 2 (see `RandomPermutationIsNoOpForPowersOfTwo`). Small
    // dimensionalities are skipped since their permutation may be the identity.
    if ((dim & (dim - 1)) != 0 && dim >= 24) {
      HadamardAndMatrixRotation unpermuted(dim, 123,
                                           /*permute_coordinates=*/false);
      EXPECT_NE(RotateCopy(unpermuted, input), RotateCopy(rotator, input));
    }
  }
}

// The permuted rotation equals the unpermuted rotation applied to permuted
// coordinates, i.e. its columns (the images of the basis vectors) are a
// permutation of the columns of the unpermuted rotation.
TEST(HadamardAndMatrixRotationTest,
     RandomPermutationEqualsContiguousOnPermutedInput) {
  for (int dim : {3, 6, 13, 24, 40, 96}) {
    SCOPED_TRACE(testing::Message() << "dim = " << dim);
    HadamardAndMatrixRotation contiguous(dim, 123,
                                         /*permute_coordinates=*/false);
    HadamardAndMatrixRotation permuted(dim, 123, /*permute_coordinates=*/true);

    std::vector<std::vector<float>> contiguous_columns(dim);
    for (int j = 0; j < dim; ++j) {
      contiguous_columns[j] = RotateCopy(contiguous, BasisVector(dim, j));
    }

    std::vector<bool> matched(dim, false);
    for (int i = 0; i < dim; ++i) {
      SCOPED_TRACE(testing::Message() << "i = " << i);
      const std::vector<float> column =
          RotateCopy(permuted, BasisVector(dim, i));
      // The columns are orthogonal with squared norm `dim`, so the matching
      // column is the one with the largest inner product.
      int best_j = 0;
      float best_dot = -1.0f;
      for (int j = 0; j < dim; ++j) {
        const float dot = std::inner_product(
            column.begin(), column.end(), contiguous_columns[j].begin(), 0.0f);
        if (dot > best_dot) {
          best_dot = dot;
          best_j = j;
        }
      }
      EXPECT_FALSE(matched[best_j]) << "column " << best_j << " matched twice";
      matched[best_j] = true;
      EXPECT_THAT(column,
                  Pointwise(FloatNear(1e-4), contiguous_columns[best_j]));
    }
  }
}

TEST(HadamardAndMatrixRotationTest, RandomPermutationInPlaceRotation) {
  for (int dim : kInverseTestDimensionalities) {
    SCOPED_TRACE(testing::Message() << "dim = " << dim);
    HadamardAndMatrixRotation permuted(dim, 123, /*permute_coordinates=*/true);
    const std::vector<float> input = RandomGaussianVector(dim, 456);
    const std::vector<float> out_of_place = RotateCopy(permuted, input);

    std::vector<float> in_place = input;
    ASSERT_OK(permuted.Rotate(absl::MakeSpan(in_place)));
    EXPECT_THAT(in_place, Pointwise(FloatNear(1e-5), out_of_place));
  }
}

TEST(HadamardAndMatrixRotationTest, RandomPermutationInverseRotation) {
  for (int dim : kInverseTestDimensionalities) {
    SCOPED_TRACE(testing::Message() << "dim = " << dim);
    HadamardAndMatrixRotation rotator(dim, 123, /*permute_coordinates=*/true);
    const std::vector<float> input = RandomGaussianVector(dim, 789);

    // InverseRotate(Rotate(x)) == x.
    std::vector<float> scratch = input;
    std::vector<float> rotated(dim);
    ASSERT_OK(rotator.Rotate(absl::MakeSpan(scratch), absl::MakeSpan(rotated)));
    std::vector<float> rotated_copy = rotated;
    std::vector<float> recovered(dim);
    ASSERT_OK(rotator.InverseRotate(absl::MakeSpan(rotated),
                                    absl::MakeSpan(recovered)));
    EXPECT_THAT(recovered, Pointwise(FloatNear(1e-4), input));

    // In-place InverseRotate matches out-of-place.
    ASSERT_OK(rotator.InverseRotate(absl::MakeSpan(rotated_copy)));
    EXPECT_THAT(rotated_copy, Pointwise(FloatNear(1e-6), recovered));

    // Rotate(InverseRotate(x)) == x.
    scratch = input;
    std::vector<float> inverse_rotated(dim);
    ASSERT_OK(rotator.InverseRotate(absl::MakeSpan(scratch),
                                    absl::MakeSpan(inverse_rotated)));
    ASSERT_OK(rotator.Rotate(absl::MakeSpan(inverse_rotated)));
    EXPECT_THAT(inverse_rotated, Pointwise(FloatNear(1e-4), input));
  }
}

TEST(HadamardAndMatrixRotationTest,
     FlipRandomSignsNormalizeAndRotateMatchesRotate) {
  for (bool permute_coordinates : {false, true}) {
    for (int dim : kInverseTestDimensionalities) {
      SCOPED_TRACE(testing::Message()
                   << "dim = " << dim
                   << ", permute_coordinates = " << permute_coordinates);
      HadamardAndMatrixRotation rotator(dim, 123, permute_coordinates);
      const std::vector<float> input = RandomGaussianVector(dim, 321);
      const float norm = 2.5f;

      std::vector<float> expected(dim);
      for (int i = 0; i < dim; ++i) {
        expected[i] = input[i] * rotator.signs()[i] * (1.0f / norm);
      }
      ASSERT_OK(rotator.Rotate(absl::MakeSpan(expected)));

      std::vector<float> fused(dim);
      ASSERT_OK(rotator.FlipRandomSignsNormalizeAndRotate(
          input, norm, absl::MakeSpan(fused)));
      EXPECT_THAT(fused, Pointwise(FloatNear(1e-5), expected));

      std::vector<float> in_place = input;
      ASSERT_OK(rotator.FlipRandomSignsNormalizeAndRotate(
          in_place, norm, absl::MakeSpan(in_place)));
      EXPECT_THAT(in_place, Pointwise(FloatNear(1e-5), expected));
    }
  }
}

TEST(HadamardAndMatrixRotationTest, RandomPermutationIsNoOpForPowersOfTwo) {
  for (int dim : {1, 2, 8, 128, 1024}) {
    SCOPED_TRACE(testing::Message() << "dim = " << dim);
    HadamardAndMatrixRotation contiguous(dim, 123,
                                         /*permute_coordinates=*/false);
    HadamardAndMatrixRotation permuted(dim, 123, /*permute_coordinates=*/true);
    std::vector<float> input1 = RandomGaussianVector(dim, 1);
    std::vector<float> input2 = input1;
    std::vector<float> output1(dim);
    std::vector<float> output2(dim);
    ASSERT_OK(
        contiguous.Rotate(absl::MakeSpan(input1), absl::MakeSpan(output1)));
    ASSERT_OK(permuted.Rotate(absl::MakeSpan(input2), absl::MakeSpan(output2)));
    EXPECT_EQ(output1, output2);
  }
}

// For a vector whose energy is concentrated in the first third of the
// coordinates (as for Matryoshka embeddings), the random permutation balances
// the energy across the output subspaces in expectation. A single permutation
// fluctuates (standard deviation ~0.03 per subspace), so we average the energy
// fractions over many seeds.
TEST(HadamardAndMatrixRotationTest,
     RandomPermutationBalancesPrefixHeavyEnergy) {
  const int dim = 768;
  const int s = 256;
  const int kNumSeeds = 50;
  std::vector<float> input(dim, 0.0f);
  std::mt19937 gen(7);
  std::bernoulli_distribution coin(0.5);
  for (int i = 0; i < s; ++i) input[i] = coin(gen) ? 1.0f : -1.0f;

  double mean_fraction[3] = {0.0, 0.0, 0.0};
  for (int seed = 0; seed < kNumSeeds; ++seed) {
    HadamardAndMatrixRotation rotator(dim, seed, /*permute_coordinates=*/true);
    const std::vector<float> output = RotateCopy(rotator, input);
    double total_energy = 0.0;
    for (float x : output) total_energy += x * x;
    for (int b = 0; b < 3; ++b) {
      double energy = 0.0;
      for (int j = 0; j < s; ++j) {
        energy += output[b * s + j] * output[b * s + j];
      }
      mean_fraction[b] += energy / total_energy / kNumSeeds;
    }
  }
  for (int b = 0; b < 3; ++b) {
    EXPECT_NEAR(mean_fraction[b], 1.0 / 3.0, 0.03) << "subspace " << b;
  }
}

}  // namespace ginger
