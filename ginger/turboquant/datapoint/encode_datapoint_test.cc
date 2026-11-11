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

#include "ginger/turboquant/datapoint/encode_datapoint.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/rotation/rotation.h"
#include "hwy/base.h"

namespace ginger {
namespace {

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::absl_testing::StatusIs;

// Dot product of the first `num_dims` coordinates of the rotated query with
// the int8 centroids of `codes`, as computed by the fast pass (but with an
// unquantized query).
double FastPassDot(absl::Span<const float> rotated_query,
                   absl::Span<const uint8_t> codes, size_t num_dims,
                   bool has_3bit_prefix) {
  double dot = 0.0;
  for (size_t i = 0; i < num_dims; ++i) {
    const size_t b = i / 2;
    const bool odd = i % 2 != 0;
    const uint8_t code = odd ? codes[b] >> 4 : codes[b] & 0xF;
    const int8_t centroid =
        (odd && has_3bit_prefix && b < kScaleBitBytes)
            ? turboquant_codebooks::kTurboQuant3BitCentroidsInt8[code]
            : turboquant_codebooks::kTurboQuantCentroidsInt8[code];
    dot += rotated_query[i] * centroid;
  }
  return dot;
}

// The 3-bit prefix loses more precision on `kScaleBitBytes` coordinates, which
// shrinks the scaled dot product estimate. The prefix debiasing correction must
// restore the expected estimate of plain 4-bit codes, for both the
// full-dimensional and the TurboSlice scaling factor. 4+k datapoints keep the
// full 4-bit codes for reranking, so their 4+k scaling factor must not change.
TEST(EncodeDatapointTest, ThreeBitPrefixKeepsFourBitDotProductBias) {
  constexpr int kNumPoints = 4000;
  for (const auto [dim, quantization_bits] :
       {std::pair<size_t, int>{64, 4}, {128, 4}, {256, 4}, {128, 6}}) {
    SCOPED_TRACE(absl::StrCat("dim=", dim, " bits=", quantization_bits));
    const int extra_bits = quantization_bits - 4;
    const size_t turboslice_dim = dim / 2;
    const HadamardAndMatrixRotation rotation(dim, /*seed=*/7);
    std::mt19937 rng(dim);
    std::normal_distribution<float> normal;
    std::vector<float> x(dim);
    std::vector<float> x_rot(dim);
    std::vector<uint8_t> codes(dim / 2);
    // Sums of estimate / ||x||^2 for the self dot product <x, x>, indexed by
    // has_3bit_prefix.
    double full_ratio[2] = {0.0, 0.0};
    double turboslice_ratio[2] = {0.0, 0.0};
    for (int n = 0; n < kNumPoints; ++n) {
      for (float& v : x) v = normal(rng);
      float sqr_norm = 0.0f;
      for (float v : x) sqr_norm += v * v;
      const float norm = std::sqrt(sqr_norm);
      ASSERT_OK(rotation.FlipRandomSignsNormalizeAndRotate(
          x, norm, absl::MakeSpan(x_rot)));
      float scale_4plusk[2] = {0.0f, 0.0f};
      for (const bool prefix : {false, true}) {
        std::vector<uint8_t> extra(
            extra_bits > 0 ? GetExtraPrecisionBytes(dim, extra_bits, prefix)
                           : 0);
        const auto res = internal::EncodeDatapointStructured(
            x, absl::MakeSpan(codes), quantization_bits, absl::MakeSpan(extra),
            dim, dim / 2, turboslice_dim, rotation, prefix);
        ASSERT_OK(res);
        scale_4plusk[prefix] = res->norm_scaling_factor_4plusk;
        full_ratio[prefix] += res->norm_scaling_factor_4bit *
                              FastPassDot(x_rot, codes, dim, prefix) / norm;
        turboslice_ratio[prefix] +=
            res->norm_scaling_factor_turboslice *
            FastPassDot(x_rot, codes, turboslice_dim, prefix) / norm;
      }
      if (extra_bits > 0) {
        ASSERT_EQ(scale_4plusk[1], scale_4plusk[0]);
      }
    }
    // Without the correction, the prefix ratios are 1 - 0.8e-3 (d = 256) to
    // 1 - 3.1e-3 (d = 64) for the full dimensionality, and 1 - 1.6e-3 to
    // 1 - 6.0e-3 for TurboSlice. With 32 TurboSlice dimensions, half of which
    // are 3-bit, the first-order correction leaves a residual of about 4e-4.
    EXPECT_NEAR(full_ratio[1] / full_ratio[0], 1.0, 3e-4);
    EXPECT_NEAR(turboslice_ratio[1] / turboslice_ratio[0], 1.0, 1e-3);
  }
}

TEST(EncodeDatapointTest, PackAndReconstructScaleBits) {
  constexpr size_t kRowBytes = kScaleBitBytes + 8;
  std::vector<uint8_t> initial_row(kRowBytes);
  for (size_t i = 0; i < kRowBytes; ++i) {
    // Include both 0 and 1 in bit 7 before packing to verify PackScaleBits
    // overwrites existing MSBs while preserving bits 0..6.
    initial_row[i] = static_cast<uint8_t>((37 * i + 13) & 0xFF);
  }
  for (const uint16_t raw_bits :
       {uint16_t{0x0000}, uint16_t{0x7BFF}, uint16_t{0x3C00}, uint16_t{0xBC00},
        uint16_t{0x5555}, uint16_t{0xAAAA}, uint16_t{0x0001},
        uint16_t{0x3555}}) {
    const hwy::float16_t scale_f16 =
        hwy::BitCastScalar<hwy::float16_t>(raw_bits);
    std::vector<uint8_t> row = initial_row;
    PackScaleBits(row.data(), scale_f16);
    EXPECT_EQ(ReconstructScale(row.data()), hwy::F32FromF16(scale_f16));
    for (size_t i = 0; i < kScaleBitBytes; ++i) {
      EXPECT_EQ(row[i] & 0x7F, initial_row[i] & 0x7F) << "byte " << i;
      EXPECT_EQ((row[i] >> 7) & 1, (raw_bits >> i) & 1) << "byte " << i;
    }
    for (size_t i = kScaleBitBytes; i < kRowBytes; ++i) {
      EXPECT_EQ(row[i], initial_row[i]) << "byte " << i;
    }
  }
}

// The 3-bit prefix only rewrites the odd nibbles of the first `kScaleBitBytes`
// bytes. 4+k datapoints append the dropped bucket LSBs to their packed extra
// bits, from which RestorePrefixCodes recovers the plain 4-bit codes.
TEST(EncodeDatapointTest, ThreeBitPrefixByteLayout) {
  const size_t dim = 128;
  const HadamardAndMatrixRotation rotation(dim, /*seed=*/7);
  std::mt19937 rng(1);
  std::normal_distribution<float> normal;
  std::vector<float> x(dim);
  for (float& v : x) v = normal(rng);
  for (const int extra_bits : {0, 2, 6}) {
    SCOPED_TRACE(absl::StrCat("extra_bits=", extra_bits));
    std::vector<uint8_t> codes[2];
    std::vector<uint8_t> extra[2];
    float scale_4plusk[2];
    for (const bool prefix : {false, true}) {
      codes[prefix].resize(dim / 2);
      extra[prefix].resize(
          extra_bits > 0 ? GetExtraPrecisionBytes(dim, extra_bits, prefix) : 0);
      ASSERT_OK_AND_ASSIGN(
          auto res, internal::EncodeDatapointStructured(
                        x, absl::MakeSpan(codes[prefix]), 4 + extra_bits,
                        absl::MakeSpan(extra[prefix]), dim, dim / 2,
                        /*turboslice_dimensionality=*/0, rotation, prefix));
      scale_4plusk[prefix] = res.norm_scaling_factor_4plusk;
    }
    const std::vector<uint8_t>& plain = codes[false];
    const std::vector<uint8_t>& prefixed = codes[true];
    for (size_t b = 0; b < plain.size(); ++b) {
      if (b < kScaleBitBytes) {
        EXPECT_EQ(prefixed[b] & 0xF, plain[b] & 0xF) << "byte " << b;
        // The odd nibble holds `code4 >> 1`, leaving bit 7 clear.
        EXPECT_EQ(prefixed[b] >> 4, plain[b] >> 5) << "byte " << b;
      } else {
        EXPECT_EQ(prefixed[b], plain[b]) << "byte " << b;
      }
    }
    if (extra_bits == 0) continue;
    const size_t packed_bytes = GetExtraPrecisionBytes(dim, extra_bits);
    ASSERT_EQ(extra[true].size(), packed_bytes + kPrefixLowBitBytes);
    EXPECT_TRUE(std::equal(extra[false].begin(), extra[false].end(),
                           extra[true].begin()));
    uint8_t restored[kScaleBitBytes];
    RestorePrefixCodes(prefixed.data(), extra[true].data() + packed_bytes,
                       restored);
    EXPECT_THAT(restored, ElementsAreArray(plain.data(), kScaleBitBytes));
    EXPECT_EQ(scale_4plusk[true], scale_4plusk[false]);
  }
}

TEST(EncodeDatapointTest, ThreeBitPrefixRejectsInvalidArguments) {
  const size_t dim = 128;
  const int extra_bits = 2;
  const HadamardAndMatrixRotation rotation(dim, /*seed=*/7);
  const std::vector<float> x(dim, 1.0f);
  std::vector<uint8_t> codes(dim / 2);
  std::vector<uint8_t> extra(
      GetExtraPrecisionBytes(dim, extra_bits, /*has_3bit_prefix=*/true));
  // Extra span without the kPrefixLowBitBytes.
  EXPECT_THAT(
      internal::EncodeDatapointStructured(
          x, absl::MakeSpan(codes), 4 + extra_bits,
          absl::MakeSpan(extra).first(GetExtraPrecisionBytes(dim, extra_bits)),
          dim, dim / 2, /*turboslice_dimensionality=*/0, rotation,
          /*has_3bit_prefix=*/true),
      StatusIs(absl::StatusCode::kInvalidArgument));
  // TurboSlice prefix too short to hold the scale bits.
  EXPECT_THAT(internal::EncodeDatapointStructured(
                  x, absl::MakeSpan(codes), 4 + extra_bits,
                  absl::MakeSpan(extra), dim, dim / 2,
                  /*turboslice_dimensionality=*/kScaleBitDims / 2, rotation,
                  /*has_3bit_prefix=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EncodeDatapointTest, RejectsTurboSliceDimensionalityAboveDimensionality) {
  const size_t dim = 128;
  const HadamardAndMatrixRotation rotation(dim, /*seed=*/7);
  const std::vector<float> x(dim, 1.0f);
  std::vector<uint8_t> codes(dim / 2);
  EXPECT_THAT(internal::EncodeDatapointStructured(
                  x, absl::MakeSpan(codes), /*quantization_bits=*/4,
                  /*output_extra_precision_data=*/{}, dim, dim / 2,
                  /*turboslice_dimensionality=*/dim + 2, rotation,
                  /*has_3bit_prefix=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("TurboSlice dimensionality cannot exceed")));
}

TEST(EncodeDatapointTest, RejectsBytesPerDatapointNotMatchingDimensionality) {
  const size_t dim = 128;
  const HadamardAndMatrixRotation rotation(dim, /*seed=*/7);
  const std::vector<float> x(dim, 1.0f);
  // Matching but too-small output span and byte count.
  std::vector<uint8_t> codes(dim / 4);
  EXPECT_THAT(internal::EncodeDatapointStructured(
                  x, absl::MakeSpan(codes), /*quantization_bits=*/4,
                  /*output_extra_precision_data=*/{}, dim, codes.size(),
                  /*turboslice_dimensionality=*/0, rotation,
                  /*has_3bit_prefix=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("bytes per datapoint does not match")));
}

}  // namespace
}  // namespace ginger
