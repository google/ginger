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

#ifndef GINGER_TURBOQUANT_CODEBOOKS_CODEBOOKS_H_
#define GINGER_TURBOQUANT_CODEBOOKS_CODEBOOKS_H_

// Trained codebooks for TurboQuant 4+k bit encoding.
//
// The centroids are symmetric around zero, so we only store positive values.
// The sign bit is handled separately during encoding/scoring, which is why we
// use 8 outer buckets instead of 16. The outer codebook entries are 16-entry
// Lloyd-max. The inner codebook entries are 2^k-entry Lloyd-max within each
// outer bucket. For k <= 4 they were trained on samples assigned to the bucket.
// For k = 6 and 8 the Lloyd-max tables are exact for the unit Gaussian
// restricted to the bucket (closed-form truncated-Gaussian moments), because
// sample-based training does not converge in the unbounded top bucket. The
// library codebook for k = 6 and 8 is a piecewise-cubic fit of those tables
// (see piecewise_cubic.h).

#include <array>
#include <cmath>
#include <cstdint>

#include "ginger/turboquant/codebooks/codebooks_10bit.h"
#include "ginger/turboquant/codebooks/codebooks_12bit.h"
#include "ginger/turboquant/codebooks/codebooks_8bit.h"
#include "ginger/turboquant/codebooks/piecewise_cubic.h"

namespace ginger::turboquant_codebooks {

// Codebook for 4+1 bit encoding (2 sub-centroids per bucket).
inline constexpr float kCentroids4Plus1BitPos[8][2] = {
    {0.06429295f, 0.19320825f}, {0.32435477f, 0.45861828f},
    {0.59504265f, 0.73458207f}, {0.87637275f, 1.0211549f},
    {1.1752363f, 1.3411912f},   {1.5226129f, 1.725043f},
    {1.955263f, 2.2278156f},    {2.5911248f, 3.1743689f}};

// Codebook for 4+2 bit encoding (4 sub-centroids per bucket).
inline constexpr float kCentroids4Plus2BitPos[8][4] = {
    {0.03218001f, 0.0965295f, 0.16102298f, 0.22570142f},
    {0.29126957f, 0.35793915f, 0.4250963f, 0.4928815f},
    {0.56110424f, 0.6298732f, 0.6996884f, 0.770649f},
    {0.8416932f, 0.9125828f, 0.9849607f, 1.0592102f},
    {1.1363492f, 1.2165966f, 1.2997268f, 1.3859141f},
    {1.4768435f, 1.573173f, 1.6746373f, 1.7821963f},
    {1.8978078f, 2.0233622f, 2.1605513f, 2.3128238f},
    {2.5041966f, 2.7596662f, 3.094675f, 3.6143968f}};

// Codebook for 4+3 bit encoding (8 sub-centroids per bucket).
inline constexpr float kCentroids4Plus3BitPos[8][8] = {
    {0.01616118f, 0.04832271f, 0.08046247f, 0.11269487f, 0.14494665f,
     0.1772535f, 0.20956624f, 0.24194811f},
    {0.27466837f, 0.30787098f, 0.34123892f, 0.374825f, 0.40835336f, 0.44219938f,
     0.47601205f, 0.51006347f},
    {0.544149f, 0.5781752f, 0.6125676f, 0.6472863f, 0.68223864f, 0.7172138f,
     0.7527806f, 0.7886175f},
    {0.8243816f, 0.8594819f, 0.89489985f, 0.9304192f, 0.966468f, 1.0034072f,
     1.0405338f, 1.0782677f},
    {1.1168792f, 1.1564443f, 1.1964256f, 1.2372665f, 1.2789235f, 1.3210907f,
     1.3642502f, 1.4082732f},
    {1.4538847f, 1.5009207f, 1.5491327f, 1.5985374f, 1.6493175f, 1.7015269f,
     1.7553362f, 1.8109263f},
    {1.8687574f, 1.9291992f, 1.992032f, 2.0576959f, 2.1264176f, 2.1986542f,
     2.2749264f, 2.3559053f},
    {2.4544358f, 2.5763671f, 2.712404f, 2.8676207f, 3.0502949f, 3.274234f,
     3.57299f, 4.0428066f}};

// Returns the positive centroid value for a given bucket and sub-index.
template <int kBitsPerDim>
constexpr float LookupCentroidPos(int bucket, uint8_t sub_index) {
  static_assert(kBitsPerDim == 1 || kBitsPerDim == 2 || kBitsPerDim == 3 ||
                    kBitsPerDim == 4 || kBitsPerDim == 6 || kBitsPerDim == 8,
                "Unsupported kBitsPerDim");
  if constexpr (kBitsPerDim == 1) {
    return kCentroids4Plus1BitPos[bucket][sub_index];
  } else if constexpr (kBitsPerDim == 2) {
    return kCentroids4Plus2BitPos[bucket][sub_index];
  } else if constexpr (kBitsPerDim == 3) {
    return kCentroids4Plus3BitPos[bucket][sub_index];
  } else if constexpr (kBitsPerDim == 4) {
    return kCentroids4Plus4BitPos[bucket][sub_index];
  } else if constexpr (kBitsPerDim == 6) {
    return kCentroids4Plus6BitFitted[bucket][sub_index];
  } else if constexpr (kBitsPerDim == 8) {
    return kCentroids4Plus8BitFitted[bucket][sub_index];
  }
}

// Returns the centroid value with sign applied.
// base_code is the 4-bit code: lower 3 bits = bucket, bit 3 = sign.
template <int kBitsPerDim>
constexpr float LookupCentroid(uint8_t base_code, uint8_t sub_index) {
  float val = LookupCentroidPos<kBitsPerDim>(base_code & 7, sub_index);
  return (base_code & 8) ? -val : val;
}

// Decision boundaries between the 8 base 4-bit buckets. These are the
// midpoints between the (positive) float centroids before quantizing to int8.
inline constexpr std::array<float, 7> kBoundaries = {
    0.2581972f, 0.5271527f, 0.806866f, 1.097338f,
    1.430843f,  1.839655f,  2.399083f};

// Squared int8 centroids (`kTurboQuantCentroidsInt8`) of the 8 base 4-bit
// buckets, used for quantized squared norm computation.
inline constexpr std::array<float, 8> kSquaredCentroids = {
    36.0f, 324.0f, 961.0f, 1936.0f, 3364.0f, 5625.0f, 9216.0f, 16129.0f};

// Returns the 4-bit bucket index (0–7) for a non-negative value.
inline uint8_t FindBucket(float x) {
  uint8_t idx = kBoundaries.size();
  while (idx > 0 && x < kBoundaries[idx - 1]) --idx;
  return idx;
}

// Returns the full 4-bit encoding: lower 3 bits = bucket, bit 3 = sign.
inline uint8_t FourBitEncoding(float x) {
  return FindBucket(std::abs(x)) | (x > 0 ? 0 : 8);
}

// Number of sub-centroids per bucket: 2^k
template <int kBitsPerDim>
inline constexpr int kNumSubCentroids = 1 << kBitsPerDim;

// Number of decision boundaries per bucket: 2^k - 1
template <int kBitsPerDim>
inline constexpr int kNumBoundaries = kNumSubCentroids<kBitsPerDim> - 1;

// Computes decision boundaries as midpoints of adjacent positive centroids
// for all 8 buckets, evaluated at compile time.
// boundary[bucket][i] = (centroid[bucket][i] + centroid[bucket][i+1]) / 2
template <int kBitsPerDim>
constexpr std::array<std::array<float, kNumBoundaries<kBitsPerDim>>, 8>
ComputeAllBoundaries() {
  std::array<std::array<float, kNumBoundaries<kBitsPerDim>>, 8> result{};
  for (int bucket = 0; bucket < 8; ++bucket) {
    for (int i = 0; i < kNumBoundaries<kBitsPerDim>; ++i) {
      result[bucket][i] = (LookupCentroidPos<kBitsPerDim>(bucket, i) +
                           LookupCentroidPos<kBitsPerDim>(bucket, i + 1)) /
                          2.0f;
    }
  }
  return result;
}

// Precomputed boundary arrays, generated at compile time from centroids.
// All arrays have shape [8][kNumBoundaries<kBitsPerDim>].
inline constexpr auto kBoundaries4Plus1Bit = ComputeAllBoundaries<1>();
inline constexpr auto kBoundaries4Plus2Bit = ComputeAllBoundaries<2>();
inline constexpr auto kBoundaries4Plus3Bit = ComputeAllBoundaries<3>();
inline constexpr auto kBoundaries4Plus4Bit = ComputeAllBoundaries<4>();
inline constexpr auto kBoundaries4Plus6Bit = ComputeAllBoundaries<6>();
inline constexpr auto kBoundaries4Plus8Bit = ComputeAllBoundaries<8>();

// Scaled TurboQuant codebook quantized to int8 for the base 4-bit scoring path.
inline constexpr std::array<int8_t, 16> kTurboQuantCentroidsInt8 = {
    6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};

// Unsigned (bias-shifted by +128) version of the int8 codebook, used by the
// database panel decoder for uint8 x int8 dot-product accumulation with
// vpdpbusd instructions.
inline constexpr std::array<uint8_t, 16> kTurboQuantCentroidsUint8 = {
    134, 146, 159, 172, 186, 203, 224, 255, 122, 110, 97, 84, 70, 53, 32, 1};

// Merged 3-bit codebook for ScalingStorageMode::kIn3BitPrefix.
//
// Dropping every second decision boundary of the 4-bit codebook merges pairs
// of adjacent buckets {2j, 2j+1} into 4 magnitude cells j = bucket >> 1.
// Shifting a 4-bit code `code4 = (sign << 3) | bucket` right by 1 bit produces
// `code3 = (sign << 2) | (bucket >> 1)` in bits 0..2 and leaves bit 3 clear
// for the scale bit. Entries c and c | 8 are identical so bit 3 is a
// don't-care during table lookup.

inline constexpr std::array<int8_t, 16> kTurboQuant3BitCentroidsInt8 = {
    12, 37, 65, 104, -12, -37, -65, -104, 12, 37, 65, 104, -12, -37, -65, -104};

inline constexpr std::array<uint8_t, 16> kTurboQuant3BitCentroidsUint8 = {
    140, 165, 193, 232, 116, 91, 63, 24, 140, 165, 193, 232, 116, 91, 63, 24};

constexpr std::array<float, 4> ComputeMerged3BitSquaredCentroids() {
  std::array<float, 4> result{};
  for (int i = 0; i < 4; ++i) {
    const float centroid = kTurboQuant3BitCentroidsInt8[i];
    result[i] = centroid * centroid;
  }
  return result;
}

inline constexpr std::array<float, 4> kMerged3BitSquaredCentroids =
    ComputeMerged3BitSquaredCentroids();

// Relative mean squared error D = E[(x - c(x))^2] of the 4-bit and the merged
// 3-bit codebook for a standard normal coordinate x, with c(x) the conditional
// mean of the cell that contains x. Computed offline by integrating N(0, 1)
// over the cells delimited by `kBoundaries` (every boundary for 4 bits, every
// second one for the merged 3 bits). 1 / sqrt(1 - k4BitDistortion) is the
// debiasing factor of 4-bit encoding.
inline constexpr float k4BitDistortion = 0.0095064f;
inline constexpr float kMerged3BitDistortion = 0.0348364f;

}  // namespace ginger::turboquant_codebooks

// Keep the int8 codebook accessible from its historical location.
namespace ginger::internal {
using turboquant_codebooks::kTurboQuantCentroidsInt8;
}  // namespace ginger::internal

#endif  // GINGER_TURBOQUANT_CODEBOOKS_CODEBOOKS_H_
