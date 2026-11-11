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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/QR"
#include "ginger/internal/portable_random.h"
#include "ginger/turboquant/rotation/walsh_hadamard.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/rotation/rotation.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

// Computes output[i] = coordinates[i] * signs[i] / norm. `coordinates` and
// `output` may alias, since each element is read before it is written.
void FlipSignsAndScaleImpl(const float* coordinates,
                           const float* HWY_RESTRICT signs, float norm,
                           size_t dimensionality, float* output) {
  const float inv_norm = 1.0f / norm;
  using F32 = hn::ScalableTag<float>;
  const F32 f32;
  const size_t N = hn::Lanes(f32);
  const auto inv_norm_v = hn::Set(f32, inv_norm);
  size_t i = 0;
  for (; i + N <= dimensionality; i += N) {
    const auto c = hn::LoadU(f32, coordinates + i);
    const auto s = hn::LoadU(f32, signs + i);
    hn::StoreU(hn::Mul(hn::Mul(c, s), inv_norm_v), f32, output + i);
  }
  for (; i < dimensionality; ++i) {
    output[i] = coordinates[i] * signs[i] * inv_norm;
  }
}

void SubspaceMixR3Impl(const float* in, const float* HWY_RESTRICT r_mat,
                       size_t sub_dim, float* out) {
  using F32 = hn::ScalableTag<float>;
  const F32 f32;
  const size_t N = hn::Lanes(f32);
  const auto r00 = hn::Set(f32, r_mat[0]);
  const auto r10 = hn::Set(f32, r_mat[1]);
  const auto r20 = hn::Set(f32, r_mat[2]);
  const auto r01 = hn::Set(f32, r_mat[3]);
  const auto r11 = hn::Set(f32, r_mat[4]);
  const auto r21 = hn::Set(f32, r_mat[5]);
  const auto r02 = hn::Set(f32, r_mat[6]);
  const auto r12 = hn::Set(f32, r_mat[7]);
  const auto r22 = hn::Set(f32, r_mat[8]);

  const float* x0_ptr = in;
  const float* x1_ptr = in + sub_dim;
  const float* x2_ptr = in + 2 * sub_dim;
  float* y0_ptr = out;
  float* y1_ptr = out + sub_dim;
  float* y2_ptr = out + 2 * sub_dim;

  size_t j = 0;
  for (; j + N <= sub_dim; j += N) {
    const auto x0 = hn::LoadU(f32, x0_ptr + j);
    const auto x1 = hn::LoadU(f32, x1_ptr + j);
    const auto x2 = hn::LoadU(f32, x2_ptr + j);

    const auto y0 = hn::MulAdd(x2, r20, hn::MulAdd(x1, r10, hn::Mul(x0, r00)));
    const auto y1 = hn::MulAdd(x2, r21, hn::MulAdd(x1, r11, hn::Mul(x0, r01)));
    const auto y2 = hn::MulAdd(x2, r22, hn::MulAdd(x1, r12, hn::Mul(x0, r02)));

    hn::StoreU(y0, f32, y0_ptr + j);
    hn::StoreU(y1, f32, y1_ptr + j);
    hn::StoreU(y2, f32, y2_ptr + j);
  }
  for (; j < sub_dim; ++j) {
    const float x0 = x0_ptr[j];
    const float x1 = x1_ptr[j];
    const float x2 = x2_ptr[j];
    y0_ptr[j] = r_mat[0] * x0 + r_mat[1] * x1 + r_mat[2] * x2;
    y1_ptr[j] = r_mat[3] * x0 + r_mat[4] * x1 + r_mat[5] * x2;
    y2_ptr[j] = r_mat[6] * x0 + r_mat[7] * x1 + r_mat[8] * x2;
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {
HWY_EXPORT(FlipSignsAndScaleImpl);
HWY_EXPORT(SubspaceMixR3Impl);

namespace {

// Applies the unnormalized Walsh-Hadamard transform in-place to each contiguous
// block of `subspace_dimensionality` coordinates. No-op if
// `subspace_dimensionality` is 1.
void HadamardTransformSubspaces(/*INOUT*/ absl::Span<float> coordinates,
                                const int subspace_dimensionality) {
  if (subspace_dimensionality <= 1) return;
  for (size_t i = 0; i < coordinates.size(); i += subspace_dimensionality) {
    WalshHadamardTransform(
        /*INOUT*/ coordinates.subspan(i, subspace_dimensionality))
        .IgnoreError();
  }
}

// Applies the permutation: output[k] = input[permutation[k]]. `input` and
// `output` must not alias.
void Gather(absl::Span<const float> input,
            absl::Span<const int32_t> permutation, absl::Span<float> output) {
  for (size_t k = 0; k < permutation.size(); ++k) {
    output[k] = input[permutation[k]];
  }
}

// Undoes the permutation applied by `Gather`: the k-th input coordinate is
// moved back to its original position `permutation[k]`. `input` and `output`
// must not alias.
void Scatter(absl::Span<const float> input,
             absl::Span<const int32_t> permutation, absl::Span<float> output) {
  for (size_t k = 0; k < permutation.size(); ++k) {
    output[permutation[k]] = input[k];
  }
}

}  // namespace

Eigen::MatrixXf GenerateRandomOrthonormalMatrix(int d, uint32_t seed) {
  std::mt19937 gen(seed);
  internal::portable_random::NormalDistribution dist;

  Eigen::MatrixXf random_matrix(d, d);
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      random_matrix(i, j) = dist(gen);
    }
  }

  Eigen::HouseholderQR<Eigen::MatrixXf> qr(random_matrix);
  Eigen::MatrixXf q_matrix = qr.householderQ();
  return (std::sqrt(d) * q_matrix);
}

HadamardAndMatrixRotation::HadamardAndMatrixRotation(
    const int dimensionality, const uint32_t seed,
    const bool permute_coordinates)
    : dimensionality_(dimensionality), signs_(dimensionality) {
  ABSL_CHECK_GT(dimensionality, 0);
  num_subspaces_ = dimensionality_;
  while (num_subspaces_ > 1 && num_subspaces_ % 2 == 0) {
    num_subspaces_ /= 2;
  }
  subspace_dimensionality_ = dimensionality_ / num_subspaces_;
  rotation_matrix_ = GenerateRandomOrthonormalMatrix(num_subspaces_, seed);

  std::mt19937 gen(seed);
  for (int i = 0; i < dimensionality; ++i) {
    const int f =
        static_cast<int>(internal::portable_random::UniformInt(gen, 2));
    signs_[i] = (2.0f * f - 1);
  }

  // A permutation is only needed if there are multiple subspaces to mix. It is
  // seeded with `seed + 1` so that it differs from the random streams used for
  // the signs and the mixing matrix, which are seeded with `seed`.
  if (permute_coordinates && num_subspaces_ > 1) {
    permutation_.resize(dimensionality_);
    std::iota(permutation_.begin(), permutation_.end(), 0);
    std::mt19937 permutation_gen(static_cast<uint32_t>(seed) + 1);
    internal::portable_random::Shuffle(absl::MakeSpan(permutation_),
                                       permutation_gen);
  }
}

absl::Status HadamardAndMatrixRotation::Rotate(
    /*INOUT*/ absl::Span<float> input_coordinates,
    /*OUT*/ absl::Span<float> output_coordinates) const {
  if (input_coordinates.size() != dimensionality_ ||
      output_coordinates.size() != dimensionality_) {
    return absl::InvalidArgumentError(
        "'input_coordinates' and 'output_coordinates' must have the same "
        "dimensionality as specified in the constructor.");
  }

  // Apply the permutation P, i.e. position k of the permuted coordinates holds
  // input_coordinates[permutation_[k]]. The gather requires distinct buffers,
  // so in-place calls gather into a scratch buffer. Either way, the permuted
  // coordinates become the input of the Hadamard transform and the mixing.
  std::vector<float> scratch;
  if (permute()) {
    absl::Span<float> permuted_coordinates = output_coordinates;
    if (input_coordinates.data() == output_coordinates.data()) {
      scratch.resize(dimensionality_);
      permuted_coordinates = absl::MakeSpan(scratch);
    }
    Gather(input_coordinates, permutation_, permuted_coordinates);
    input_coordinates = permuted_coordinates;
  }

  HadamardTransformSubspaces(input_coordinates, subspace_dimensionality_);

  const bool is_aliased =
      (input_coordinates.data() == output_coordinates.data());

  if (num_subspaces_ == 1) {
    if (!is_aliased) {
      std::copy_n(input_coordinates.data(), dimensionality_,
                  output_coordinates.data());
    }
  } else if (num_subspaces_ == 3) {
    HWY_DYNAMIC_DISPATCH(SubspaceMixR3Impl)(
        input_coordinates.data(), rotation_matrix_.data(),
        subspace_dimensionality_, output_coordinates.data());
  } else {
    const Eigen::Map<const Eigen::MatrixXf> coordinates_map(
        input_coordinates.data(), subspace_dimensionality_, num_subspaces_);
    Eigen::Map<Eigen::MatrixXf> output_coordinates_map(
        output_coordinates.data(), subspace_dimensionality_, num_subspaces_);
    if (is_aliased) {
      output_coordinates_map = coordinates_map * rotation_matrix_;
    } else {
      output_coordinates_map.noalias() = coordinates_map * rotation_matrix_;
    }
  }
  return absl::OkStatus();
}

absl::Status HadamardAndMatrixRotation::InverseRotate(
    /*INOUT*/ absl::Span<float> input_coordinates,
    /*OUT*/ absl::Span<float> output_coordinates) const {
  if (input_coordinates.size() != dimensionality_ ||
      output_coordinates.size() != dimensionality_) {
    return absl::InvalidArgumentError(
        "'input_coordinates' and 'output_coordinates' must have the same "
        "dimensionality as specified in the constructor.");
  }

  // View the coordinates as a column-major s x r matrix X, where
  // s = subspace_dimensionality_ and r = num_subspaces_. `Rotate` computes
  // H * X * R, where H is the unnormalized Walsh-Hadamard matrix (H = H^T and
  // H * H = s * I) and R is the mixing matrix (R * R^T = r * I). The inverse is
  // thus H * Y * R^T / (s * r). Since H multiplies from the left and R^T from
  // the right, we apply them in the same order as `Rotate`.
  HadamardTransformSubspaces(input_coordinates, subspace_dimensionality_);

  const float inv_dimensionality = 1.0f / dimensionality_;
  if (num_subspaces_ == 1) {
    // `Rotate` skips the mixing in this case, so we only need to rescale.
    for (int i = 0; i < dimensionality_; ++i) {
      output_coordinates[i] = input_coordinates[i] * inv_dimensionality;
    }
    return absl::OkStatus();
  }

  // With a permutation, the mixed coordinates are scattered into
  // `output_coordinates` at the end to undo P. The mixing supports aliasing,
  // so we mix `input_coordinates` in-place (it is an INOUT parameter anyway)
  // and scatter from it. In-place calls mix into a scratch buffer instead,
  // since the scatter requires distinct buffers.
  std::vector<float> scratch;
  absl::Span<float> mixed_coordinates = output_coordinates;
  if (permute()) {
    mixed_coordinates = input_coordinates;
    if (input_coordinates.data() == output_coordinates.data()) {
      scratch.resize(dimensionality_);
      mixed_coordinates = absl::MakeSpan(scratch);
    }
  }

  const bool is_aliased =
      (input_coordinates.data() == mixed_coordinates.data());
  if (num_subspaces_ == 3) {
    // Column-major R^T / d, which is the layout `SubspaceMixR3Impl` expects.
    const Eigen::Matrix3f inverse_mixing_matrix =
        rotation_matrix_.transpose() * inv_dimensionality;
    HWY_DYNAMIC_DISPATCH(SubspaceMixR3Impl)(
        input_coordinates.data(), inverse_mixing_matrix.data(),
        subspace_dimensionality_, mixed_coordinates.data());
  } else {
    const Eigen::Map<const Eigen::MatrixXf> coordinates_map(
        input_coordinates.data(), subspace_dimensionality_, num_subspaces_);
    Eigen::Map<Eigen::MatrixXf> mixed_coordinates_map(
        mixed_coordinates.data(), subspace_dimensionality_, num_subspaces_);
    // The scalar is kept on the left so that Eigen folds it into the product.
    if (is_aliased) {
      mixed_coordinates_map =
          inv_dimensionality * (coordinates_map * rotation_matrix_.transpose());
    } else {
      mixed_coordinates_map.noalias() =
          inv_dimensionality * (coordinates_map * rotation_matrix_.transpose());
    }
  }

  if (permute()) {
    Scatter(mixed_coordinates, permutation_, output_coordinates);
  }
  return absl::OkStatus();
}

absl::Status HadamardAndMatrixRotation::FlipRandomSignsNormalizeAndRotate(
    absl::Span<const float> coordinates, float norm,
    absl::Span<float> rotated_coordinates) const {
  if (norm == 0.0f) {
    return absl::InvalidArgumentError("Vector has a norm of zero.");
  }
  if (coordinates.size() != dimensionality_ ||
      rotated_coordinates.size() != dimensionality_) {
    return absl::InvalidArgumentError(
        "'coordinates' and 'rotated_coordinates' must have the same "
        "dimensionality as specified in the constructor.");
  }
  HWY_DYNAMIC_DISPATCH(FlipSignsAndScaleImpl)(coordinates.data(), signs_.data(),
                                              norm, dimensionality_,
                                              rotated_coordinates.data());
  return Rotate(rotated_coordinates, rotated_coordinates);
}

}  // namespace ginger
#endif  // HWY_ONCE
