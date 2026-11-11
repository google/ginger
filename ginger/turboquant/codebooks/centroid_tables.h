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

#ifndef GINGER_TURBOQUANT_CODEBOOKS_CENTROID_TABLES_H_
#define GINGER_TURBOQUANT_CODEBOOKS_CENTROID_TABLES_H_

// Constexpr centroid and scoring tables for TurboQuant with extra precision
// bits, extracted for testability.
// Note: These centroid tables assume a 4-bit base quantizer (16 signed codes
// or 8 magnitude buckets). Therefore, the codec supports total quantization
// bit depths >= 4 bits per dimension (4-bit base code + kBitsPerDim extra
// bits).

#include "ginger/turboquant/codebooks/codebooks.h"

namespace ginger {
namespace turboquant_internal {

// Flat centroid table with sign baked in (used by the 3-bit/GatherIndex path).
// Layout: table[base_code * 2^kBitsPerDim + sub_index] = signed centroid.
// Use Get(base_code, sub_index) to access individual centroids without
// depending on the raw linear array layout.
template <int kBitsPerDim>
struct SignedCentroidTable {
  static constexpr int kNumSubCentroids = 1 << kBitsPerDim;
  static constexpr int kTableSize = 16 * kNumSubCentroids;
  alignas(64) float table[kTableSize] = {};

  constexpr SignedCentroidTable() {
    for (int base_code = 0; base_code < 16; ++base_code) {
      for (int sub = 0; sub < kNumSubCentroids; ++sub) {
        table[base_code * kNumSubCentroids + sub] =
            turboquant_codebooks::LookupCentroid<kBitsPerDim>(base_code, sub);
      }
    }
  }

  constexpr float Get(int base_code, int sub) const {
    return table[base_code * kNumSubCentroids + sub];
  }
  const float* data() const { return table; }
};

// Positive-only centroid table (unsigned magnitudes). Used by the
// extra-precision (4+1)-bit and (4+2)-bit paths where sign is applied
// separately via XOR, enabling fast SIMD table lookups (TableLookupLanes)
// instead of GatherIndex. Layout: table[bucket * 2^kBitsPerDim + sub_index] =
// |centroid|. Use Get(bucket, sub_index) to access individual centroids without
// depending on the raw linear array layout.
template <int kBitsPerDim>
struct PositiveCentroidTable {
  static constexpr int kNumSubCentroids = 1 << kBitsPerDim;
  static constexpr int kSize = 8 * kNumSubCentroids;
  alignas(64) float table[kSize] = {};

  constexpr PositiveCentroidTable() {
    for (int bucket = 0; bucket < 8; ++bucket) {
      for (int sub = 0; sub < kNumSubCentroids; ++sub) {
        table[bucket * kNumSubCentroids + sub] =
            turboquant_codebooks::LookupCentroidPos<kBitsPerDim>(bucket, sub);
      }
    }
  }

  constexpr float Get(int bucket, int sub) const {
    return table[bucket * kNumSubCentroids + sub];
  }
  const float* data() const { return table; }
};

}  // namespace turboquant_internal
}  // namespace ginger

#endif  // GINGER_TURBOQUANT_CODEBOOKS_CENTROID_TABLES_H_
