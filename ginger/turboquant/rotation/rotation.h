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

#ifndef GINGER_TURBOQUANT_ROTATION_ROTATION_H_
#define GINGER_TURBOQUANT_ROTATION_ROTATION_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "Eigen/Core"

namespace ginger {

// A fast random rotation for vectors of arbitrary dimensionality d, used by
// TurboQuant. The Walsh-Hadamard transform runs in O(d log d) but requires d to
// be a power of 2, and padding to the next power of 2 would increase the memory
// footprint. Instead, we write d = r * s with r odd and s = 2^k, and `Rotate`
//   1. optionally permutes the coordinates (see `permute_coordinates`),
//   2. applies the unnormalized Hadamard transform H to each of the r
//      contiguous subspaces of size s, and
//   3. mixes the subspaces with a random r x r orthogonal matrix scaled by
//      sqrt(r), called R below.
// Viewing the (permuted) vector as a column-major s x r matrix X, the result is
// H * X * R, which scales norms by sqrt(d). Note that this strategy is
// particularly effective for Cohere, Gecko, OpenAI and Gemini embeddings, which
// all have r=3 (3x3 mixing).
//
// All randomness (R, the permutation, and the random signs used by
// `FlipRandomSignsNormalizeAndRotate`) is derived from the seed, so the
// rotation can be reconstructed from the constructor arguments instead of
// being stored.
class HadamardAndMatrixRotation {
 public:
  // `permute_coordinates`: Without a permutation, subspace b consists of the
  // contiguous coordinates [b * s, (b + 1) * s). If the energy of the input is
  // unevenly distributed across these blocks, e.g. for Matryoshka embeddings
  // whose prefixes carry more energy, R cannot equalize it, so the rotated
  // coordinates have unequal variances, which increases the quantization error.
  // With `permute_coordinates`, the coordinates are first shuffled with a
  // random permutation (seeded with `seed + 1`), so each subspace receives a
  // random subset of the coordinates and the subspace energies are balanced in
  // expectation. Has no effect if d is a power of 2 (r = 1). Since the
  // permutation changes the rotation, data must be encoded and decoded with the
  // same setting. Defaults to false for backward compatibility.
  HadamardAndMatrixRotation(int dimensionality, uint32_t seed,
                            bool permute_coordinates = false);

  // Rotates `input_coordinates` and stores the result in `output_coordinates`.
  // Both spans must have the dimensionality passed to the constructor and may
  // refer to the same buffer. `input_coordinates` may be overwritten, since it
  // is used as scratch space. With `permute_coordinates`, in-place calls
  // allocate a temporary buffer.
  absl::Status Rotate(/*INOUT*/ absl::Span<float> input_coordinates,
                      /*OUT*/ absl::Span<float> output_coordinates) const;

  // Convenience overload that rotates `coordinates` in-place.
  absl::Status Rotate(/*INOUT*/ absl::Span<float> coordinates) const {
    return Rotate(coordinates, coordinates);
  }

  // Inverse of `Rotate`, i.e. `InverseRotate(Rotate(x)) == x` up to
  // floating-point error. Since `Rotate` scales norms by sqrt(d), this is the
  // transpose of `Rotate` scaled by 1/d. Same buffer contract as `Rotate`.
  absl::Status InverseRotate(
      /*INOUT*/ absl::Span<float> input_coordinates,
      /*OUT*/ absl::Span<float> output_coordinates) const;

  // Convenience overload that inverse-rotates `coordinates` in-place.
  absl::Status InverseRotate(/*INOUT*/ absl::Span<float> coordinates) const {
    return InverseRotate(coordinates, coordinates);
  }

  // Computes `Rotate(D * coordinates / norm)`, where D is the diagonal matrix
  // of random signs (see `signs()`), indexed by the original (unpermuted)
  // coordinates. `norm` must be the non-zero L2 norm of `coordinates`; it is
  // passed in because callers already compute it. The spans may alias.
  absl::Status FlipRandomSignsNormalizeAndRotate(
      absl::Span<const float> coordinates, float norm,
      absl::Span<float> rotated_coordinates) const;

  // The r x r mixing matrix R. Exposed for testing.
  const Eigen::MatrixXf& GetRotationMatrix() const { return rotation_matrix_; }

  // The random signs (+1 or -1) used by `FlipRandomSignsNormalizeAndRotate`.
  absl::Span<const float> signs() const { return signs_; }
  void SetSignsForTesting(const std::vector<float>& signs) { signs_ = signs; }

 private:
  // True iff a permutation is applied, i.e. if `permute_coordinates` was set
  // and the dimensionality is not a power of 2.
  bool permute() const { return !permutation_.empty(); }

  // d = r * s in the notation above.
  int dimensionality_;               // d
  int num_subspaces_;                // r (odd)
  int subspace_dimensionality_;      // s (a power of 2)
  Eigen::MatrixXf rotation_matrix_;  // R
  std::vector<float> signs_;
  // The permutation P: position k of the permuted vector holds coordinate
  // permutation_[k] of the input. `Rotate` gathers with it and `InverseRotate`
  // scatters with it, so no inverse permutation is stored. Empty if no
  // permutation is applied (see `permute()`).
  std::vector<int32_t> permutation_;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_ROTATION_ROTATION_H_
