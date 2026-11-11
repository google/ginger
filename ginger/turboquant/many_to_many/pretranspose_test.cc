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

#include "ginger/turboquant/many_to_many/pretranspose.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "ginger/internal/gunit_support.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/types/distance_metric.h"

namespace ginger {
namespace {

std::vector<std::vector<float>> GenerateRandomVectors(size_t count, size_t dim,
                                                      absl::BitGen& bitgen) {
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<std::vector<float>> vecs(count, std::vector<float>(dim));
  for (auto& v : vecs) {
    for (float& x : v) x = dist(bitgen);
  }
  return vecs;
}

TurboQuantDatasetSoA EncodePoints(const TurboQuantCodec& tq,
                                  absl::Span<const std::vector<float>> vecs) {
  std::vector<absl::Span<const float>> spans;
  spans.reserve(vecs.size());
  for (const auto& v : vecs) spans.push_back(absl::MakeConstSpan(v));
  auto bp = TurboQuantDatasetSoA::Create(tq, spans);
  ABSL_CHECK_OK(bp);
  return *std::move(bp);
}

// Helper: ceiling division.
size_t DivCeil(size_t a, size_t b) { return (a + b - 1) / b; }

uint8_t ToNibble(uint8_t byte, bool is_high) {
  return is_high ? (byte >> 4) : (byte & 0x0F);
}

uint8_t GetExpectedValue(const TurboQuantDatasetSoA& bp, size_t idx, size_t D) {
  const size_t packed_byte_idx = D / 2;
  const bool is_high = (D % 2 == 1);
  const uint8_t byte = bp.quantized_data()[idx * bp.stride() + packed_byte_idx];
  const uint8_t nibble = ToNibble(byte, is_high);
  return turboquant_codebooks::kTurboQuantCentroidsUint8[nibble];
}

size_t GetOffset(const PreTransposedTurboQuantDataset& pt, size_t P, size_t D) {
  const size_t points_per_panel = pt.points_per_panel;
  const size_t N = points_per_panel * 4;
  const size_t tiles_per_chunk = N / 2;
  const size_t packed_byte_idx = D / 2;

  const size_t panel_idx = P / points_per_panel;
  const size_t i_local = P % points_per_panel;
  const size_t chunk = packed_byte_idx / N;
  const size_t parity = D % 2;
  const size_t offset_in_chunk = packed_byte_idx % N;
  const size_t group = offset_in_chunk / 4;
  const size_t byte_idx = offset_in_chunk % 4;

  return panel_idx * pt.panel_bytes +
         (chunk * tiles_per_chunk + parity * points_per_panel + group) * N +
         i_local * 4 + byte_idx;
}

class PreTransposeMetadataTest
    : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

TEST_P(PreTransposeMetadataTest, MetadataIsConsistent) {
  const auto [dim, num_points] = GetParam();

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  // points_per_panel = N/4 where N = SIMD lanes for uint8_t. Derive N.
  const size_t points_per_panel = pt.points_per_panel;
  EXPECT_GT(points_per_panel, 0u);
  const size_t N = points_per_panel * 4;

  EXPECT_EQ(pt.num_points, num_points);
  EXPECT_EQ(pt.dimensionality, dim);
  EXPECT_EQ(pt.bytes_per_point, bp.num_bytes_per_datapoint());

  const size_t expected_num_panels = DivCeil(num_points, points_per_panel);
  EXPECT_EQ(pt.num_panels, expected_num_panels);

  const size_t num_dim_chunks = DivCeil(pt.bytes_per_point, N);
  const size_t tiles_per_chunk = N / 2;
  const size_t expected_total_tiles = num_dim_chunks * tiles_per_chunk;
  EXPECT_EQ(pt.total_tiles, expected_total_tiles);

  const size_t expected_panel_bytes = expected_total_tiles * N;
  EXPECT_EQ(pt.panel_bytes, expected_panel_bytes);

  // panel_data_ should be exactly num_panels * panel_bytes.
  EXPECT_NE(pt.panel_data.data(), nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    DimAndCount, PreTransposeMetadataTest,
    ::testing::Combine(::testing::Values(4, 16, 64, 128, 256, 3, 7, 17, 129),
                       ::testing::Values(1, 2, 3, 7, 8, 15, 16, 17, 32, 50)));

class PreTransposeNormsTest
    : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

TEST_P(PreTransposeNormsTest, NormsMatchBasePoints) {
  const auto [dim, num_points] = GetParam();

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  // Each norm should match the base point's norm_scaling_factor.
  for (size_t i = 0; i < num_points; ++i) {
    EXPECT_EQ(pt.norm_scaling_factors[i], bp.norm_scaling_factor(i))
        << "norm mismatch at index " << i;
  }
  // Padding norms (beyond num_points, up to num_panels * kpoints) should be
  // infinity to ensure padded points never win in distance comparisons.
  const size_t padded_count = pt.num_panels * pt.points_per_panel;
  for (size_t i = num_points; i < padded_count; ++i) {
    EXPECT_EQ(pt.norm_scaling_factors[i],
              std::numeric_limits<float>::infinity())
        << "padding norm at index " << i << " should be infinity";
  }
}

INSTANTIATE_TEST_SUITE_P(DimAndCount, PreTransposeNormsTest,
                         ::testing::Combine(::testing::Values(4, 64, 128, 129),
                                            ::testing::Values(1, 7, 16, 17,
                                                              50)));

TEST(PreTransposeL2Test, SqrNormsPopulatedForL2) {
  const size_t dim = 128;
  const size_t num_points = 20;

  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kSquaredL2Distance);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  ASSERT_TRUE(!pt.sqr_norms.empty());

  for (size_t i = 0; i < num_points; ++i) {
    EXPECT_EQ(pt.sqr_norms[i], bp.unquantized_squared_norm(i))
        << "sqr_norm mismatch at index " << i;
  }

  // Padding squared norms should be max float.
  const size_t padded_count = pt.num_panels * pt.points_per_panel;
  for (size_t i = num_points; i < padded_count; ++i) {
    EXPECT_EQ(pt.sqr_norms[i], std::numeric_limits<float>::max())
        << "padding sqr_norm at index " << i << " should be max";
  }
}

TEST(PreTransposeL2Test, SqrNormsEmptyForDotProduct) {
  const size_t dim = 64;
  const size_t num_points = 10;

  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kDotProductDistance);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  EXPECT_TRUE(pt.sqr_norms.empty());
}

TEST(PreTransposeBuildIndicesTest, SubsetMetadata) {
  const size_t dim = 128;
  const size_t num_points = 50;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  std::vector<int> indices = {3, 10, 25, 49};
  auto pt = PreTransposedTurboQuantDataset::Build(bp, indices, tq);

  EXPECT_EQ(pt.num_points, indices.size());
  EXPECT_EQ(pt.dimensionality, dim);
  EXPECT_EQ(pt.bytes_per_point, bp.num_bytes_per_datapoint());
}

TEST(PreTransposeBuildIndicesTest, SubsetNorms) {
  const size_t dim = 64;
  const size_t num_points = 30;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  std::vector<int> indices = {0, 5, 10, 29};
  auto pt = PreTransposedTurboQuantDataset::Build(bp, indices, tq);

  ASSERT_EQ(pt.num_points, indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(pt.norm_scaling_factors[i], bp.norm_scaling_factor(indices[i]))
        << "norm mismatch at subset position " << i << " (original index "
        << indices[i] << ")";
  }
}

TEST(PreTransposeBuildIndicesTest, SubsetL2SqrNorms) {
  const size_t dim = 128;
  const size_t num_points = 40;

  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kSquaredL2Distance);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  std::vector<int> indices = {1, 7, 20, 39};
  auto pt = PreTransposedTurboQuantDataset::Build(bp, indices, tq);

  ASSERT_TRUE(!pt.sqr_norms.empty());
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(pt.sqr_norms[i], bp.unquantized_squared_norm(indices[i]))
        << "sqr_norm mismatch at subset position " << i;
  }
}

TEST(PreTransposeBuildTest, BuildAllMatchesBuildWithIota) {
  const size_t dim = 128;
  const size_t num_points = 20;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt_all = PreTransposedTurboQuantDataset::Build(bp, tq);

  std::vector<int> all_indices(num_points);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  auto pt_explicit = PreTransposedTurboQuantDataset::Build(bp, all_indices, tq);

  EXPECT_EQ(pt_all.num_points, pt_explicit.num_points);
  EXPECT_EQ(pt_all.dimensionality, pt_explicit.dimensionality);
  EXPECT_EQ(pt_all.bytes_per_point, pt_explicit.bytes_per_point);
  EXPECT_EQ(pt_all.num_panels, pt_explicit.num_panels);
  EXPECT_EQ(pt_all.panel_bytes, pt_explicit.panel_bytes);
  EXPECT_EQ(pt_all.total_tiles, pt_explicit.total_tiles);
  EXPECT_EQ(pt_all.points_per_panel, pt_explicit.points_per_panel);

  // Norms should be identical.
  for (size_t i = 0; i < num_points; ++i) {
    EXPECT_EQ(pt_all.norm_scaling_factors[i],
              pt_explicit.norm_scaling_factors[i])
        << "norm mismatch at " << i;
  }

  EXPECT_EQ(pt_all.panel_data, pt_explicit.panel_data);
}

TEST(PreTransposeEdgeTest, SinglePoint) {
  const size_t dim = 128;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(1, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  EXPECT_EQ(pt.num_points, 1u);
  EXPECT_EQ(pt.num_panels, 1u);
  EXPECT_EQ(pt.norm_scaling_factors[0], bp.norm_scaling_factor(0));
}

TEST(PreTransposeEdgeTest, EmptyDataset) {
  const size_t dim = 128;

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto bps = TurboQuantDatasetSoA::Create(tq, 0);
  ABSL_CHECK_OK(bps);
  auto bp = *std::move(bps);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  EXPECT_EQ(pt.num_points, 0u);
  EXPECT_EQ(pt.num_panels, 0u);
  EXPECT_TRUE(pt.panel_data.empty());
}

TEST(PreTransposeBuildIndicesTest, ReversedIndicesCorrectNorms) {
  const size_t dim = 64;
  const size_t num_points = 20;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  std::vector<int> reversed(num_points);
  std::iota(reversed.begin(), reversed.end(), 0);
  std::reverse(reversed.begin(), reversed.end());

  auto pt = PreTransposedTurboQuantDataset::Build(bp, reversed, tq);

  ASSERT_EQ(pt.num_points, num_points);
  for (size_t i = 0; i < num_points; ++i) {
    EXPECT_EQ(pt.norm_scaling_factors[i], bp.norm_scaling_factor(reversed[i]))
        << "reversed norm mismatch at position " << i;
  }
}

TEST(PreTransposeBuildIndicesTest, DuplicateIndicesCorrect) {
  const size_t dim = 64;
  const size_t num_points = 10;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  std::vector<int> duplicates(num_points, 0);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, duplicates, tq);

  ASSERT_EQ(pt.num_points, num_points);
  for (size_t i = 0; i < num_points; ++i) {
    EXPECT_EQ(pt.norm_scaling_factors[i], bp.norm_scaling_factor(0));
  }

  for (size_t D : {0, 1, 31, 32, 63}) {
    const uint8_t expected_val = GetExpectedValue(bp, 0, D);
    for (size_t P = 0; P < num_points; ++P) {
      const size_t offset = GetOffset(pt, P, D);
      EXPECT_EQ(pt.panel_data[offset], expected_val)
          << "Mismatch at point " << P << " (source 0) dim " << D;
    }
  }
}

TEST(PreTransposeBuildIndicesTest, EmptyIndicesReturnsEmptyDataset) {
  const size_t dim = 128;
  const size_t num_points = 10;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  std::vector<int> empty_indices;
  auto pt = PreTransposedTurboQuantDataset::Build(bp, empty_indices, tq);

  EXPECT_EQ(pt.num_points, 0u);
  EXPECT_EQ(pt.num_panels, 0u);
  EXPECT_TRUE(pt.panel_data.empty());
  EXPECT_TRUE(pt.norm_scaling_factors.empty());
}

TEST(PreTransposeBuildIndicesTest, IndicesLargerThanBaseCapacity) {
  const size_t dim = 64;
  const size_t num_base_points = 5;
  const size_t num_requested_points = 100;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_base_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  std::vector<int> indices(num_requested_points);
  for (size_t i = 0; i < num_requested_points; ++i) {
    indices[i] = i % num_base_points;
  }

  auto pt = PreTransposedTurboQuantDataset::Build(bp, indices, tq);

  EXPECT_EQ(pt.num_points, num_requested_points);
  EXPECT_EQ(pt.num_panels, DivCeil(num_requested_points, pt.points_per_panel));

  for (size_t P : {0, 4, 5, 99}) {
    const size_t base_p = P % num_base_points;
    for (size_t D : {0, 31, 63}) {
      const uint8_t expected_val = GetExpectedValue(bp, base_p, D);
      const size_t offset = GetOffset(pt, P, D);
      EXPECT_EQ(pt.panel_data[offset], expected_val)
          << "Mismatch at point " << P << " (base " << base_p << ") dim " << D;
    }
  }
}

TEST(PreTransposeLayoutTest, VerifyByteLayout) {
  const size_t dim = 128;
  const size_t num_points = 10;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  for (size_t P : {0, 1, 7, 8, 9}) {
    for (size_t D : {0, 1, 31, 32, 33, 63, 64, 65, 127}) {
      const uint8_t expected_val = GetExpectedValue(bp, P, D);
      const size_t offset = GetOffset(pt, P, D);
      EXPECT_EQ(pt.panel_data[offset], expected_val)
          << "Mismatch at point " << P << " dim " << D;
    }
  }
}

TEST(PreTransposeLayoutTest, PaddedLeftoversZeroCleared) {
  const size_t dim = 64;
  const size_t num_points = 3;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  const size_t points_per_panel = pt.points_per_panel;
  ASSERT_EQ(pt.num_panels, 1u);

  for (size_t P = num_points; P < points_per_panel; ++P) {
    for (size_t D = 0; D < dim; ++D) {
      const size_t offset = GetOffset(pt, P, D);
      EXPECT_EQ(pt.panel_data[offset], 0u)
          << "Padding not zeroed at local point " << P << " dim " << D;
    }
  }
}

class PreTransposeOddDimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(PreTransposeOddDimTest, BuildSucceeds) {
  const size_t dim = GetParam();
  const size_t num_points = 15;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  EXPECT_EQ(pt.num_points, num_points);
  EXPECT_EQ(pt.dimensionality, dim);
  EXPECT_GT(pt.panel_bytes, 0u);
  EXPECT_GT(pt.total_tiles, 0u);
}

INSTANTIATE_TEST_SUITE_P(OddDims, PreTransposeOddDimTest,
                         ::testing::Values(1, 3, 5, 7, 9, 15, 17, 33, 65, 129,
                                           257, 513));

class PreTransposePanelCountTest : public ::testing::TestWithParam<size_t> {};

TEST_P(PreTransposePanelCountTest, PanelCountCorrect) {
  const size_t num_points = GetParam();
  const size_t dim = 128;

  TurboQuantCodec tq(dim, /*seed=*/42);
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(num_points, dim, bitgen);
  auto bp = EncodePoints(tq, vecs);

  auto pt = PreTransposedTurboQuantDataset::Build(bp, tq);

  const size_t points_per_panel = pt.points_per_panel;
  EXPECT_EQ(pt.num_panels, DivCeil(num_points, points_per_panel));
}

INSTANTIATE_TEST_SUITE_P(PanelCounts, PreTransposePanelCountTest,
                         ::testing::Values(1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32,
                                           33, 47, 48, 49, 64, 100));

}  // namespace
}  // namespace ginger
