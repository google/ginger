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

#ifndef GINGER_TURBOQUANT_CODEBOOKS_PIECEWISE_CUBIC_H_
#define GINGER_TURBOQUANT_CODEBOOKS_PIECEWISE_CUBIC_H_

// Piecewise-cubic inner codebooks for 4+6 and 4+8 bit encoding.
//
// The Lloyd-Max inner centroids are smooth in the sub-index within each of the
// outer buckets 0-6, so one cubic per bucket reproduces them almost exactly.
// The unbounded top bucket is split into `kTailSegments<k>` equal sub-index
// ranges with one cubic each. With 15 (k = 6) or 23 (k = 8) segments the
// coefficients fit in one or two 16-lane registers, so SIMD scoring evaluates
// centroids with in-register table lookups and FMAs instead of a gather.
//
// The fitted values are the library codebook: encoding and scoring both use
// them. Coefficients are least-squares fits to the Lloyd-Max tables, computed
// at compile time.

#include <array>

#include "ginger/turboquant/codebooks/codebooks_10bit.h"
#include "ginger/turboquant/codebooks/codebooks_12bit.h"

namespace ginger::turboquant_codebooks {

template <int k>
inline constexpr int kTailSegments = k == 6 ? 8 : 16;

// Sub-index bits that select the position within a top-bucket segment.
template <int k>
inline constexpr int kTailSegmentBits = k == 6 ? 3 : 4;

template <int k>
inline constexpr int kNumCubicSegments = 7 + kTailSegments<k>;

// coef[j][segment] multiplies t^j, where t is the sub-index relative to the
// first sub-index of the segment. Padded to 32 segments (two 16-lane vectors).
struct PiecewiseCubicCoefficients {
  alignas(64) float coef[4][32] = {};
};

template <int k>
constexpr float LloydCentroidPos(int bucket, int sub_index) {
  static_assert(k == 6 || k == 8);
  if constexpr (k == 6) {
    return kCentroids4Plus6BitPos[bucket][sub_index];
  } else {
    return kCentroids4Plus8BitPos[bucket][sub_index];
  }
}

// Fits one cubic per segment to the Lloyd-Max centroids by least squares.
template <int k>
constexpr PiecewiseCubicCoefficients FitPiecewiseCubic() {
  PiecewiseCubicCoefficients result;
  constexpr int kSubIndices = 1 << k;
  for (int segment = 0; segment < kNumCubicSegments<k>; ++segment) {
    // Segments 0-6 cover buckets 0-6; the rest split bucket 7 equally.
    const bool tail = segment >= 7;
    const int bucket = tail ? 7 : segment;
    const int segment_len = tail ? kSubIndices / kTailSegments<k> : kSubIndices;
    const int first_sub_index = tail ? (segment - 7) * segment_len : 0;

    // Augmented normal equations [V^T V | V^T y] for y ~= sum_j coef_u[j] u^j,
    // with u = t / segment_len in [0, 1) instead of t for conditioning.
    double normal[4][5] = {};
    for (int t = 0; t < segment_len; ++t) {
      const double u = static_cast<double>(t) / segment_len;
      const double y = LloydCentroidPos<k>(bucket, first_sub_index + t);
      const double powers[4] = {1.0, u, u * u, u * u * u};
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
          normal[row][col] += powers[row] * powers[col];
        }
        normal[row][4] += powers[row] * y;
      }
    }
    // Gaussian elimination to upper-triangular form. V^T V is positive
    // definite, so no pivoting is needed.
    for (int pivot = 0; pivot < 4; ++pivot) {
      for (int row = pivot + 1; row < 4; ++row) {
        const double factor = normal[row][pivot] / normal[pivot][pivot];
        for (int col = pivot; col < 5; ++col) {
          normal[row][col] -= factor * normal[pivot][col];
        }
      }
    }
    // Back substitution.
    double coef_u[4] = {};
    for (int row = 3; row >= 0; --row) {
      double sum = normal[row][4];
      for (int col = row + 1; col < 4; ++col) {
        sum -= normal[row][col] * coef_u[col];
      }
      coef_u[row] = sum / normal[row][row];
    }
    // Convert from powers of u to powers of t: divide by segment_len^j.
    double len_pow = 1.0;
    for (int j = 0; j < 4; ++j) {
      result.coef[j][segment] = static_cast<float>(coef_u[j] / len_pow);
      len_pow *= segment_len;
    }
  }
  return result;
}

template <int k>
inline constexpr PiecewiseCubicCoefficients kPiecewiseCubic =
    FitPiecewiseCubic<k>();

// Evaluates the fitted positive centroid. Each Horner step rounds like a float
// FMA (the product of two floats is exact in double), matching SIMD scoring.
template <int k>
constexpr float EvalPiecewiseCubic(int bucket, int sub_index) {
  // In the top bucket, the high sub-index bits select the segment and the low
  // bits are the offset t within it.
  constexpr int kOffsetBits = k - kTailSegmentBits<k>;
  const bool tail = bucket == 7;
  const int segment = tail ? 7 + (sub_index >> kOffsetBits) : bucket;
  const int t = tail ? sub_index & ((1 << kOffsetBits) - 1) : sub_index;
  const auto& coef = kPiecewiseCubic<k>.coef;
  // Horner: ((coef[3] t + coef[2]) t + coef[1]) t + coef[0].
  float value = coef[3][segment];
  for (int j = 2; j >= 0; --j) {
    value = static_cast<float>(static_cast<double>(value) * t +
                               static_cast<double>(coef[j][segment]));
  }
  return value;
}

template <int k>
constexpr std::array<std::array<float, (1 << k)>, 8> MakePiecewiseCubicTable() {
  std::array<std::array<float, (1 << k)>, 8> table{};
  for (int bucket = 0; bucket < 8; ++bucket) {
    for (int sub = 0; sub < (1 << k); ++sub) {
      table[bucket][sub] = EvalPiecewiseCubic<k>(bucket, sub);
    }
  }
  return table;
}

// Positive centroids [bucket][sub_index] of the 4+6 and 4+8 bit codebooks.
inline constexpr auto kCentroids4Plus6BitFitted = MakePiecewiseCubicTable<6>();
inline constexpr auto kCentroids4Plus8BitFitted = MakePiecewiseCubicTable<8>();

// Encoding with midpoint boundaries requires strictly increasing centroids.
template <int k>
constexpr bool IsStrictlyIncreasing(
    const std::array<std::array<float, (1 << k)>, 8>& table) {
  for (int bucket = 0; bucket < 8; ++bucket) {
    for (int sub = 1; sub < (1 << k); ++sub) {
      if (!(table[bucket][sub] > table[bucket][sub - 1])) return false;
    }
  }
  return true;
}
static_assert(IsStrictlyIncreasing<6>(kCentroids4Plus6BitFitted));
static_assert(IsStrictlyIncreasing<8>(kCentroids4Plus8BitFitted));

}  // namespace ginger::turboquant_codebooks

#endif  // GINGER_TURBOQUANT_CODEBOOKS_PIECEWISE_CUBIC_H_
