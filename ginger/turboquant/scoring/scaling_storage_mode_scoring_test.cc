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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/scoring/scoring_scalar_reference.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "hwy/base.h"

namespace ginger {
namespace {

using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::absl_testing::StatusIs;

constexpr size_t kSeed = 555;
constexpr size_t kNumDatapoints = 300;

std::vector<std::vector<float>> RandomVectors(size_t count, size_t dim,
                                              absl::BitGen& bitgen,
                                              bool normalize) {
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<std::vector<float>> vecs(count, std::vector<float>(dim));
  for (auto& v : vecs) {
    float sq_norm = 0.0f;
    for (float& x : v) {
      x = dist(bitgen);
      sq_norm += x * x;
    }
    if (normalize) {
      const float inv_norm = 1.0f / std::sqrt(sq_norm);
      for (float& x : v) x *= inv_norm;
    }
  }
  return vecs;
}

TurboQuantDatasetSoA Encode(const TurboQuantCodec& tq,
                            absl::Span<const std::vector<float>> vecs,
                            bool normalized_datapoints, ScalingStorageMode mode,
                            bool pad_to_cache_line_size,
                            int quantization_bits) {
  const std::vector<absl::Span<const float>> spans(vecs.begin(), vecs.end());
  auto bp = TurboQuantDatasetSoA::Create(
      tq, spans,
      TurboQuantDatasetOptions{.pad_to_cache_line_size = pad_to_cache_line_size,
                               .quantization_bits = quantization_bits,
                               .store_query_byte_sums = false,
                               .normalized_datapoints = normalized_datapoints,
                               .scaling_storage_mode = mode});
  ABSL_CHECK_OK(bp);
  return *std::move(bp);
}

using ScoreFn = void (TurboQuantDatasetSoA::*)(const TurboQuantQuery&,
                                               ScoringAccessor) const;

std::vector<float> ScoreSimd(const TurboQuantDatasetSoA& db,
                             const TurboQuantQuery& query, ScoreFn score_fn,
                             absl::Span<const int32_t> indices) {
  std::vector<float> scores(indices.size());
  (db.*score_fn)(
      query, ScoresAndIndicesPairOfSpansView(indices, absl::MakeSpan(scores)));
  return scores;
}

std::vector<int32_t> AllIndices(size_t n) {
  std::vector<int32_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}

// Parameterized over (dim, metric, normalized_datapoints, padding).
class ScalingStorageModeScoringTest
    : public testing::TestWithParam<
          std::tuple<size_t, TurboQuantDistanceMetric, bool, bool>> {};

// 32: below one vector. 100: norm fits in the padding. 128: 32B-aligned, norm
// spills into another cache line when padded. 1536: tiled kernel on AVX2.
INSTANTIATE_TEST_SUITE_P(
    DimsMetricsAndPadding, ScalingStorageModeScoringTest,
    testing::Combine(
        testing::Values(size_t{32}, size_t{100}, size_t{128}, size_t{1536}),
        testing::Values(TurboQuantDistanceMetric::kDotProductDistance,
                        TurboQuantDistanceMetric::kSquaredL2Distance),
        testing::Bool(), testing::Bool()));

TEST_P(ScalingStorageModeScoringTest, RowTailMatchesSeparateArray) {
  const auto [dim, metric, normalized_datapoints, pad] = GetParam();
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, metric);
  auto vecs = RandomVectors(kNumDatapoints, dim, bitgen, normalized_datapoints);
  auto db_row_tail = Encode(tq, vecs, normalized_datapoints,
                            ScalingStorageMode::kRowTail, pad, 4);
  auto db_separate = Encode(tq, vecs, normalized_datapoints,
                            ScalingStorageMode::kSeparateArray, pad, 4);

  const size_t row_bytes = db_row_tail.num_bytes_per_datapoint() +
                           NormEntryBytes(db_row_tail.norm_layout());
  EXPECT_EQ(db_row_tail.scaling_storage_mode(), ScalingStorageMode::kRowTail);
  EXPECT_FALSE(db_row_tail.has_3bit_prefix());
  EXPECT_EQ(db_row_tail.stride(),
            pad ? hwy::RoundUpTo(row_bytes, ABSL_CACHELINE_SIZE) : row_bytes);
  EXPECT_EQ(db_row_tail.norm_data().data(), nullptr);
  // An explicit kSeparateArray request is honored even if padding has room.
  EXPECT_EQ(db_separate.scaling_storage_mode(),
            ScalingStorageMode::kSeparateArray);
  EXPECT_NE(db_separate.norm_data().data(), nullptr);

  auto query_vecs = RandomVectors(1, dim, bitgen, normalized_datapoints);
  ASSERT_OK_AND_ASSIGN(auto query, TurboQuantQuery::Create(tq, query_vecs[0]));

  const auto indices = AllIndices(db_row_tail.num_datapoints());
  const auto actual =
      ScoreSimd(db_row_tail, query,
                &TurboQuantDatasetSoA::UnsafeComputeDistances, indices);
  const auto expected =
      ScoreSimd(db_separate, query,
                &TurboQuantDatasetSoA::UnsafeComputeDistances, indices);

  EXPECT_THAT(actual, Pointwise(FloatEq(), expected));
}

// kRowTail and kIn3BitPrefix (whose dropped bucket LSBs are kept in the
// extra-precision bytes) give 4+k scores identical to kSeparateArray.
TEST(ScalingStorageModeExtraPrecisionTest, MatchesSeparateArray) {
  for (const int bits : {5, 6, 7, 8, 10, 12}) {
    for (const size_t dim : {size_t{32}, size_t{100}, size_t{128}}) {
      for (const auto [metric, normalized_datapoints] :
           {std::make_pair(TurboQuantDistanceMetric::kDotProductDistance,
                           false),
            std::make_pair(TurboQuantDistanceMetric::kSquaredL2Distance, false),
            std::make_pair(TurboQuantDistanceMetric::kSquaredL2Distance,
                           true)}) {
        for (const bool pad : {false, true}) {
          SCOPED_TRACE(testing::Message()
                       << "bits=" << bits << " dim=" << dim << " metric="
                       << static_cast<int>(metric) << " pad=" << pad);
          absl::BitGen bitgen;
          TurboQuantCodec tq(dim, kSeed, metric);
          const auto vecs =
              RandomVectors(kNumDatapoints, dim, bitgen, normalized_datapoints);
          const auto query_vecs =
              RandomVectors(1, dim, bitgen, normalized_datapoints);
          ASSERT_OK_AND_ASSIGN(
              auto query,
              TurboQuantQuery::Create(
                  tq, query_vecs[0], /*enable_higher_precision_scoring=*/true));
          const auto indices = AllIndices(kNumDatapoints);
          constexpr ScoreFn kScoreFn =
              &TurboQuantDatasetSoA::UnsafeComputeDistancesExtraPrecision;
          const auto expected =
              ScoreSimd(Encode(tq, vecs, normalized_datapoints,
                               ScalingStorageMode::kSeparateArray, pad, bits),
                        query, kScoreFn, indices);
          for (const ScalingStorageMode mode :
               {ScalingStorageMode::kRowTail,
                ScalingStorageMode::kIn3BitPrefix}) {
            if (mode == ScalingStorageMode::kIn3BitPrefix &&
                NormLayoutFor(metric, normalized_datapoints) !=
                    NormLayout::kScalingOnly) {
              continue;
            }
            const auto db =
                Encode(tq, vecs, normalized_datapoints, mode, pad, bits);
            EXPECT_EQ(db.has_3bit_prefix(),
                      mode == ScalingStorageMode::kIn3BitPrefix);
            const auto actual = ScoreSimd(db, query, kScoreFn, indices);
            EXPECT_THAT(actual, Pointwise(FloatEq(), expected))
                << "mode=" << static_cast<int>(mode);
            std::vector<float> reference(indices.size());
            TurboQuantScoringExtraPrecisionScalarReference(
                db, query, absl::MakeSpan(reference), indices);
            EXPECT_THAT(actual, Pointwise(FloatNear(1e-4), reference))
                << "mode=" << static_cast<int>(mode);
          }
        }
      }
    }
  }
}

TEST_P(ScalingStorageModeScoringTest, In3BitPrefixMatchesScalarReference) {
  const auto [dim, metric, normalized_datapoints, pad] = GetParam();
  if (NormLayoutFor(metric, normalized_datapoints) !=
      NormLayout::kScalingOnly) {
    GTEST_SKIP() << "kIn3BitPrefix only applies to kScalingOnly.";
  }
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, metric);
  auto vecs = RandomVectors(kNumDatapoints, dim, bitgen, normalized_datapoints);
  auto db = Encode(tq, vecs, normalized_datapoints,
                   ScalingStorageMode::kIn3BitPrefix, pad, 4);
  ASSERT_TRUE(db.has_3bit_prefix());
  EXPECT_EQ(db.norm_data().data(), nullptr);

  auto query_vecs = RandomVectors(1, dim, bitgen, normalized_datapoints);
  ASSERT_OK_AND_ASSIGN(auto query, TurboQuantQuery::Create(tq, query_vecs[0]));

  const auto indices = AllIndices(db.num_datapoints());
  const auto actual = ScoreSimd(
      db, query, &TurboQuantDatasetSoA::UnsafeComputeDistances, indices);

  std::vector<float> expected(indices.size());
  TurboQuantScoringInt8ScalarReference(db, query, absl::MakeSpan(expected),
                                       indices);
  EXPECT_THAT(actual, Pointwise(FloatNear(1e-4), expected));
}

// Checks kIn3BitPrefix against kSeparateArray scores, independently of the
// scalar reference, which shares the 3-bit decoding logic.
TEST_P(ScalingStorageModeScoringTest, In3BitPrefixCloseToSeparateArray) {
  const auto [dim, metric, normalized_datapoints, pad] = GetParam();
  if (NormLayoutFor(metric, normalized_datapoints) !=
      NormLayout::kScalingOnly) {
    GTEST_SKIP() << "kIn3BitPrefix only applies to kScalingOnly.";
  }
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, metric);
  auto vecs = RandomVectors(kNumDatapoints, dim, bitgen, normalized_datapoints);
  auto db_prefix = Encode(tq, vecs, normalized_datapoints,
                          ScalingStorageMode::kIn3BitPrefix, pad, 4);
  auto db_separate = Encode(tq, vecs, normalized_datapoints,
                            ScalingStorageMode::kSeparateArray, pad, 4);

  auto query_vecs = RandomVectors(1, dim, bitgen, normalized_datapoints);
  ASSERT_OK_AND_ASSIGN(auto query, TurboQuantQuery::Create(tq, query_vecs[0]));
  const auto indices = AllIndices(kNumDatapoints);
  const auto actual = ScoreSimd(
      db_prefix, query, &TurboQuantDatasetSoA::UnsafeComputeDistances, indices);
  const auto expected =
      ScoreSimd(db_separate, query,
                &TurboQuantDatasetSoA::UnsafeComputeDistances, indices);

  double sum = 0.0, sum_sq = 0.0, abs_diff = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    sum += expected[i];
    sum_sq += expected[i] * expected[i];
    abs_diff += std::abs(actual[i] - expected[i]);
  }
  const double n = expected.size();
  const double stddev = std::sqrt(sum_sq / n - (sum / n) * (sum / n));
  // Only the kScaleBitDims / 2 odd prefix dims lose precision, so the relative
  // error shrinks with dim. The bound has ~2x margin over observed errors.
  EXPECT_LT(abs_diff / n, 1.2 / std::sqrt(dim) * stddev)
      << "dim=" << dim << " ratio=" << (abs_diff / n) / stddev;
}

TEST(ScalingStorageMode3BitPrefixTest, ExtraPrecisionRoundTrip) {
  const size_t dim = 128;
  const int bits = 6;
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, TurboQuantDistanceMetric::kDotProductDistance);
  const auto vecs = RandomVectors(8, dim, bitgen, /*normalize=*/false);
  const std::vector<absl::Span<const float>> spans(vecs.begin(), vecs.end());
  ASSERT_OK_AND_ASSIGN(
      auto db,
      TurboQuantDatasetSoA::Create(
          tq, spans,
          TurboQuantDatasetOptions{
              .pad_to_cache_line_size = false,
              .quantization_bits = bits,
              .store_query_byte_sums = false,
              .normalized_datapoints = false,
              .scaling_storage_mode = ScalingStorageMode::kIn3BitPrefix}));

  // Rebuild via encoded datapoints (UpdatePoint and AppendPoint) and via
  // DataPointView copies.
  auto db_encoded = TurboQuantDatasetSoA::UninitializedLike(db);
  auto db_copied = TurboQuantDatasetSoA::UninitializedLike(db);
  ASSERT_OK_AND_ASSIGN(
      auto db_appended,
      TurboQuantDatasetSoA::Create(
          tq, /*num_datapoints=*/0,
          TurboQuantDatasetOptions{
              .pad_to_cache_line_size = false,
              .quantization_bits = bits,
              .store_query_byte_sums = false,
              .normalized_datapoints = false,
              .scaling_storage_mode = ScalingStorageMode::kIn3BitPrefix}));
  for (size_t i = 0; i < vecs.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(auto dp, EncodeDatapoint(tq, vecs[i], bits,
                                                  /*has_3bit_prefix=*/true));
    ASSERT_OK(db_encoded.UpdatePoint(dp.view(), i));
    ASSERT_OK(db_appended.AppendPoint(dp.view()));
    ASSERT_OK(db_copied.UpdatePoint(db.DataPointView(i), i));
  }

  const auto query_vecs = RandomVectors(1, dim, bitgen, /*normalize=*/false);
  ASSERT_OK_AND_ASSIGN(
      auto query,
      TurboQuantQuery::Create(tq, query_vecs[0],
                              /*enable_higher_precision_scoring=*/true));
  const auto indices = AllIndices(vecs.size());
  constexpr ScoreFn kScoreFn =
      &TurboQuantDatasetSoA::UnsafeComputeDistancesExtraPrecision;
  const auto expected = ScoreSimd(db, query, kScoreFn, indices);
  EXPECT_THAT(ScoreSimd(db_encoded, query, kScoreFn, indices),
              Pointwise(FloatEq(), expected));
  EXPECT_THAT(ScoreSimd(db_appended, query, kScoreFn, indices),
              Pointwise(FloatEq(), expected));
  EXPECT_THAT(ScoreSimd(db_copied, query, kScoreFn, indices),
              Pointwise(FloatEq(), expected));
}

TEST(ScalingStorageModeEligibilityTest, CreateRejectsIneligibleDatasets) {
  struct Case {
    size_t dim;
    TurboQuantDistanceMetric metric;
    size_t turboslice_dimensionality;
  };
  constexpr Case kCases[] = {
      {kScaleBitDims - 2, TurboQuantDistanceMetric::kDotProductDistance, 0},
      // kScalingAndSquaredNorm layout.
      {128, TurboQuantDistanceMetric::kSquaredL2Distance, 0},
      // TurboSlice prefix too short to hold the scale bits.
      {128, TurboQuantDistanceMetric::kDotProductDistance, 16},
  };
  for (const Case& c : kCases) {
    for (const int quantization_bits : {4, 6}) {
      TurboQuantCodec tq(c.dim, kSeed, c.metric, c.turboslice_dimensionality);
      EXPECT_THAT(
          TurboQuantDatasetSoA::Create(
              tq, /*num_datapoints=*/16,
              TurboQuantDatasetOptions{
                  .pad_to_cache_line_size = false,
                  .quantization_bits = quantization_bits,
                  .store_query_byte_sums = false,
                  .normalized_datapoints = false,
                  .scaling_storage_mode = ScalingStorageMode::kIn3BitPrefix}),
          StatusIs(absl::StatusCode::kInvalidArgument))
          << "dim=" << c.dim << " bits=" << quantization_bits;
    }
  }
}

TEST(ScalingStorageModeTurboSliceTest, BothPassesMatchScalarReference) {
  const size_t dim = 128;
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, TurboQuantDistanceMetric::kDotProductDistance,
                     /*turboslice_dimensionality=*/64);
  auto db = Encode(
      tq, RandomVectors(kNumDatapoints, dim, bitgen, /*normalize=*/false),
      /*normalized_datapoints=*/false, ScalingStorageMode::kIn3BitPrefix,
      /*pad_to_cache_line_size=*/false, 4);
  ASSERT_TRUE(db.has_3bit_prefix());
  ASSERT_TRUE(db.IsTurboSliceEnabled());
  EXPECT_EQ(db.scaling_storage_mode_4bit(), ScalingStorageMode::kSeparateArray);
  EXPECT_EQ(db.scaling_storage_mode_turboslice(),
            ScalingStorageMode::kIn3BitPrefix);
  EXPECT_NE(db.norm_data().data(), nullptr);

  // Also exercise GrowStorageForAppend when TurboSlice is in 3-bit prefix and
  // full 4-bit is in the separate norm array.
  for (const auto& v : RandomVectors(5, dim, bitgen, /*normalize=*/false)) {
    ASSERT_OK(db.AppendPoint(tq, v));
  }
  EXPECT_EQ(db.num_datapoints(), kNumDatapoints + 5);

  auto query_vecs = RandomVectors(1, dim, bitgen, /*normalize=*/false);
  const auto indices = AllIndices(db.num_datapoints());

  ASSERT_OK_AND_ASSIGN(
      auto turboslice_query,
      TurboQuantQuery::CreateTurboSliceQuery(tq, query_vecs[0]));
  const auto turboslice_actual = ScoreSimd(
      db, turboslice_query,
      &TurboQuantDatasetSoA::UnsafeComputeDistancesTurboSlice, indices);
  std::vector<float> turboslice_expected(indices.size());
  TurboQuantScoringInt8ScalarReference(
      db, turboslice_query, absl::MakeSpan(turboslice_expected), indices,
      /*is_turboslice_scoring=*/true);
  EXPECT_THAT(turboslice_actual,
              Pointwise(FloatNear(1e-4), turboslice_expected));

  ASSERT_OK_AND_ASSIGN(auto query, TurboQuantQuery::Create(tq, query_vecs[0]));
  const auto actual = ScoreSimd(
      db, query, &TurboQuantDatasetSoA::UnsafeComputeDistances, indices);
  std::vector<float> expected(indices.size());
  TurboQuantScoringInt8ScalarReference(db, query, absl::MakeSpan(expected),
                                       indices);
  EXPECT_THAT(actual, Pointwise(FloatNear(1e-4), expected));
}

// With TurboSlice, the 3-bit prefix holds the TurboSlice scale, while the 4-bit
// and 4+k scales live in their separate arrays.
TEST(ScalingStorageModeTurboSliceTest, ExtraPrecisionWith3BitPrefix) {
  const size_t dim = 128;
  const int bits = 6;
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, TurboQuantDistanceMetric::kDotProductDistance,
                     /*turboslice_dimensionality=*/64);
  const auto vecs =
      RandomVectors(kNumDatapoints, dim, bitgen, /*normalize=*/false);
  const auto db_prefix = Encode(tq, vecs, /*normalized_datapoints=*/false,
                                ScalingStorageMode::kIn3BitPrefix,
                                /*pad_to_cache_line_size=*/false, bits);
  const auto db_separate = Encode(tq, vecs, /*normalized_datapoints=*/false,
                                  ScalingStorageMode::kSeparateArray,
                                  /*pad_to_cache_line_size=*/false, bits);
  ASSERT_TRUE(db_prefix.has_3bit_prefix());
  ASSERT_TRUE(db_prefix.IsTurboSliceEnabled());

  const auto query_vecs = RandomVectors(1, dim, bitgen, /*normalize=*/false);
  ASSERT_OK_AND_ASSIGN(
      auto query,
      TurboQuantQuery::Create(tq, query_vecs[0],
                              /*enable_higher_precision_scoring=*/true));
  const auto indices = AllIndices(kNumDatapoints);
  constexpr ScoreFn kScoreFn =
      &TurboQuantDatasetSoA::UnsafeComputeDistancesExtraPrecision;
  const auto actual = ScoreSimd(db_prefix, query, kScoreFn, indices);
  std::vector<float> expected(indices.size());
  TurboQuantScoringExtraPrecisionScalarReference(
      db_prefix, query, absl::MakeSpan(expected), indices);
  EXPECT_THAT(actual, Pointwise(FloatNear(1e-4), expected));
  EXPECT_THAT(actual, Pointwise(FloatEq(), ScoreSimd(db_separate, query,
                                                     kScoreFn, indices)));
}

TEST(ScalingStorageMode3BitPrefixTest, DequantizeToRotatedSpace4Bit) {
  const size_t dim = 128;
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, TurboQuantDistanceMetric::kDotProductDistance);
  const auto vecs = RandomVectors(16, dim, bitgen, /*normalize=*/false);
  const auto db_prefix = Encode(tq, vecs, /*normalized_datapoints=*/false,
                                ScalingStorageMode::kIn3BitPrefix,
                                /*pad_to_cache_line_size=*/false, 4);
  const auto db_separate = Encode(tq, vecs, /*normalized_datapoints=*/false,
                                  ScalingStorageMode::kSeparateArray,
                                  /*pad_to_cache_line_size=*/false, 4);
  std::vector<float> prefix(dim);
  std::vector<float> separate(dim);
  for (size_t i = 0; i < vecs.size(); ++i) {
    db_prefix.DequantizeToRotatedSpace(i, absl::MakeSpan(prefix));
    db_separate.DequantizeToRotatedSpace(i, absl::MakeSpan(separate));
    // Odd prefix dims hold the merged 3-bit centroid of the 4-bit code, which
    // is the high nibble of the separate row's byte.
    const auto separate_dp = db_separate.DataPointView(i);
    const float prefix_scale =
        db_prefix.DataPointView(i).norm_scaling_factor_4bit;
    const float separate_scale = separate_dp.norm_scaling_factor_4bit;
    for (size_t d = 0; d < dim; ++d) {
      const float expected =
          d < kScaleBitDims && d % 2 == 1
              ? turboquant_codebooks::kTurboQuant3BitCentroidsInt8
                    [separate_dp.quantized_data[d / 2] >> 5]
              : separate[d] / separate_scale;
      EXPECT_NEAR(prefix[d] / prefix_scale, expected, 1e-3) << "d=" << d;
    }
  }
}

// The dropped bucket LSBs are restored, so 4+k dequantization is exact.
TEST(ScalingStorageMode3BitPrefixTest, DequantizeToRotatedSpaceExtraPrecision) {
  const size_t dim = 128;
  for (const int bits : {6, 10}) {
    SCOPED_TRACE(testing::Message() << "bits=" << bits);
    absl::BitGen bitgen;
    TurboQuantCodec tq(dim, kSeed,
                       TurboQuantDistanceMetric::kDotProductDistance);
    const auto vecs = RandomVectors(16, dim, bitgen, /*normalize=*/false);
    const auto db_prefix = Encode(tq, vecs, /*normalized_datapoints=*/false,
                                  ScalingStorageMode::kIn3BitPrefix,
                                  /*pad_to_cache_line_size=*/false, bits);
    const auto db_separate = Encode(tq, vecs, /*normalized_datapoints=*/false,
                                    ScalingStorageMode::kSeparateArray,
                                    /*pad_to_cache_line_size=*/false, bits);
    std::vector<float> prefix(dim);
    std::vector<float> separate(dim);
    for (size_t i = 0; i < vecs.size(); ++i) {
      db_prefix.DequantizeToRotatedSpace(i, absl::MakeSpan(prefix));
      db_separate.DequantizeToRotatedSpace(i, absl::MakeSpan(separate));
      EXPECT_THAT(prefix, Pointwise(FloatEq(), separate));
    }
  }
}

TEST(ScalingStorageMode3BitPrefixTest,
     QueryFromDataPointWithExtraPrecisionMatchesSeparateArray) {
  const size_t dim = 128;
  const int bits = 6;
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, TurboQuantDistanceMetric::kDotProductDistance);
  const auto vecs = RandomVectors(16, dim, bitgen, /*normalize=*/false);
  const auto db_prefix = Encode(tq, vecs, /*normalized_datapoints=*/false,
                                ScalingStorageMode::kIn3BitPrefix,
                                /*pad_to_cache_line_size=*/false, bits);
  const auto db_separate = Encode(tq, vecs, /*normalized_datapoints=*/false,
                                  ScalingStorageMode::kSeparateArray,
                                  /*pad_to_cache_line_size=*/false, bits);
  ASSERT_OK_AND_ASSIGN(
      auto query_prefix,
      TurboQuantQuery::CreateFromQuantizedDataPointWithExtraPrecision(
          tq, db_prefix.DataPointView(0)));
  ASSERT_OK_AND_ASSIGN(
      auto query_separate,
      TurboQuantQuery::CreateFromQuantizedDataPointWithExtraPrecision(
          tq, db_separate.DataPointView(0)));
  const auto indices = AllIndices(vecs.size());
  constexpr ScoreFn kScoreFn =
      &TurboQuantDatasetSoA::UnsafeComputeDistancesExtraPrecision;
  EXPECT_THAT(ScoreSimd(db_prefix, query_prefix, kScoreFn, indices),
              Pointwise(FloatEq(), ScoreSimd(db_separate, query_separate,
                                             kScoreFn, indices)));
}

TEST(ScalingStorageModeUpdatePointTest,
     RejectsMismatched3BitPrefixAndScoresDataPointAsQuery) {
  const size_t dim = 128;
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, TurboQuantDistanceMetric::kDotProductDistance);
  auto vecs = RandomVectors(4, dim, bitgen, /*normalize=*/false);

  ASSERT_OK_AND_ASSIGN(
      auto db_3bit,
      TurboQuantDatasetSoA::Create(
          tq, vecs.size(),
          TurboQuantDatasetOptions{
              .pad_to_cache_line_size = false,
              .quantization_bits = 4,
              .store_query_byte_sums = false,
              .normalized_datapoints = false,
              .scaling_storage_mode = ScalingStorageMode::kIn3BitPrefix}));
  ASSERT_OK_AND_ASSIGN(
      auto db_4bit,
      TurboQuantDatasetSoA::Create(
          tq, vecs.size(),
          TurboQuantDatasetOptions{
              .pad_to_cache_line_size = false,
              .quantization_bits = 4,
              .store_query_byte_sums = true,
              .normalized_datapoints = false,
              .scaling_storage_mode = ScalingStorageMode::kSeparateArray}));

  ASSERT_OK_AND_ASSIGN(auto dp_4bit,
                       EncodeDatapoint(tq, vecs[0], /*quantization_bits=*/4,
                                       /*has_3bit_prefix=*/false));
  ASSERT_OK_AND_ASSIGN(auto dp_3bit,
                       EncodeDatapoint(tq, vecs[0], /*quantization_bits=*/4,
                                       /*has_3bit_prefix=*/true));

  EXPECT_FALSE(db_3bit.UpdatePoint(dp_4bit.view(), 0).ok());
  EXPECT_FALSE(db_4bit.UpdatePoint(dp_3bit.view(), 0).ok());
  EXPECT_OK(db_3bit.UpdatePoint(dp_3bit.view(), 0));
  EXPECT_OK(db_4bit.UpdatePoint(dp_4bit.view(), 0));

  for (size_t i = 1; i < vecs.size(); ++i) {
    ASSERT_OK(db_3bit.UpdatePoint(tq, vecs[i], i));
  }
  ASSERT_OK_AND_ASSIGN(auto q_before_sums,
                       TurboQuantQuery::CreateFromQuantizedDataPoint(
                           tq, db_3bit.DataPointView(0)));
  db_3bit.EnableQueryByteSums();
  ASSERT_OK_AND_ASSIGN(auto q_after_sums,
                       TurboQuantQuery::CreateFromQuantizedDataPoint(
                           tq, db_3bit.DataPointView(0)));
  EXPECT_EQ(q_before_sums.query_byte_sum(), q_after_sums.query_byte_sum());
  EXPECT_EQ(q_before_sums.query_byte_sum(), dp_3bit.view().query_byte_sum);

  const auto indices = AllIndices(db_3bit.num_datapoints());
  const auto actual =
      ScoreSimd(db_3bit, q_after_sums,
                &TurboQuantDatasetSoA::UnsafeComputeDistances, indices);
  std::vector<float> expected(indices.size());
  TurboQuantScoringInt8ScalarReference(db_3bit, q_after_sums,
                                       absl::MakeSpan(expected), indices);
  EXPECT_THAT(actual, Pointwise(FloatNear(1e-5), expected));
}

struct RecommendedModeCase {
  size_t dim;
  NormLayout layout;
  bool pad;
  ScalingStorageMode expected;
  // With `allow_3bit_prefix`.
  ScalingStorageMode expected_3bit;
};

class RecommendedScalingStorageModeTest
    : public testing::TestWithParam<RecommendedModeCase> {};

constexpr NormLayout kScalingOnly = NormLayout::kScalingOnly;
constexpr NormLayout kWithSquaredNorm = NormLayout::kScalingAndSquaredNorm;
constexpr ScalingStorageMode kRowTail = ScalingStorageMode::kRowTail;
constexpr ScalingStorageMode kSeparate = ScalingStorageMode::kSeparateArray;
constexpr ScalingStorageMode k3Bit = ScalingStorageMode::kIn3BitPrefix;

INSTANTIATE_TEST_SUITE_P(
    BoundaryCases, RecommendedScalingStorageModeTest,
    testing::Values(
        // Padded: kRowTail iff `coded_bytes + NormEntryBytes(layout)` does
        // not cross a 64B boundary. kScalingOnly takes 2 bytes.
        RecommendedModeCase{1, kScalingOnly, true, kRowTail, kRowTail},
        RecommendedModeCase{124, kScalingOnly, true, kRowTail, kRowTail},
        RecommendedModeCase{125, kScalingOnly, true, kSeparate, k3Bit},
        RecommendedModeCase{128, kScalingOnly, true, kSeparate, k3Bit},
        RecommendedModeCase{129, kScalingOnly, true, kRowTail, kRowTail},
        RecommendedModeCase{252, kScalingOnly, true, kRowTail, kRowTail},
        RecommendedModeCase{253, kScalingOnly, true, kSeparate, k3Bit},
        RecommendedModeCase{1536, kScalingOnly, true, kSeparate, k3Bit},
        // kScalingAndSquaredNorm takes 8 bytes (16 dims).
        RecommendedModeCase{112, kWithSquaredNorm, true, kRowTail, kRowTail},
        RecommendedModeCase{113, kWithSquaredNorm, true, kSeparate, kSeparate},
        RecommendedModeCase{128, kWithSquaredNorm, true, kSeparate, kSeparate},
        // Unpadded: kRowTail unless `coded_bytes` is 32B-aligned.
        RecommendedModeCase{1, kScalingOnly, false, kRowTail, kRowTail},
        RecommendedModeCase{63, kScalingOnly, false, kSeparate, k3Bit},
        RecommendedModeCase{64, kScalingOnly, false, kSeparate, k3Bit},
        RecommendedModeCase{126, kScalingOnly, false, kRowTail, kRowTail},
        RecommendedModeCase{127, kScalingOnly, false, kSeparate, k3Bit},
        RecommendedModeCase{128, kScalingOnly, false, kSeparate, k3Bit},
        RecommendedModeCase{129, kScalingOnly, false, kRowTail, kRowTail},
        RecommendedModeCase{254, kScalingOnly, false, kRowTail, kRowTail},
        RecommendedModeCase{1536, kScalingOnly, false, kSeparate, k3Bit},
        RecommendedModeCase{126, kWithSquaredNorm, false, kRowTail, kRowTail},
        RecommendedModeCase{127, kWithSquaredNorm, false, kSeparate,
                            kSeparate}));

TEST_P(RecommendedScalingStorageModeTest, MatchesExpected) {
  const RecommendedModeCase& c = GetParam();
  EXPECT_EQ(RecommendedScalingStorageMode(c.dim, c.layout, c.pad,
                                          /*allow_3bit_prefix=*/false),
            c.expected)
      << "dim=" << c.dim << " pad=" << c.pad;
  EXPECT_EQ(RecommendedScalingStorageMode(c.dim, c.layout, c.pad,
                                          /*allow_3bit_prefix=*/true),
            c.expected_3bit)
      << "dim=" << c.dim << " pad=" << c.pad;
}

TEST(ScalingStorageModeRowTailTest,
     TurboSliceWithRowTailKeeps4BitInRowTailAndTurboSliceInSideArray) {
  const size_t dim = 128;
  const size_t turboslice_dim = 32;
  absl::BitGen bitgen;
  TurboQuantCodec tq(dim, kSeed, TurboQuantDistanceMetric::kDotProductDistance,
                     turboslice_dim);
  auto vecs = RandomVectors(64, dim, bitgen, /*normalize=*/false);
  auto db_row_tail =
      Encode(tq, vecs, /*normalized_datapoints=*/false,
             ScalingStorageMode::kRowTail, /*pad_to_cache_line_size=*/false, 4);
  auto db_separate = Encode(tq, vecs, /*normalized_datapoints=*/false,
                            ScalingStorageMode::kSeparateArray,
                            /*pad_to_cache_line_size=*/false, 4);

  auto appended_vecs = RandomVectors(4, dim, bitgen, /*normalize=*/false);
  for (const auto& v : appended_vecs) {
    ASSERT_OK(db_row_tail.AppendPoint(tq, v));
    ASSERT_OK(db_separate.AppendPoint(tq, v));
  }

  EXPECT_EQ(db_row_tail.norm_data().data(), nullptr);

  auto query_vecs = RandomVectors(1, dim, bitgen, /*normalize=*/false);
  ASSERT_OK_AND_ASSIGN(auto query, TurboQuantQuery::Create(tq, query_vecs[0]));
  ASSERT_OK_AND_ASSIGN(
      auto ts_query, TurboQuantQuery::CreateTurboSliceQuery(tq, query_vecs[0]));

  const auto indices = AllIndices(db_row_tail.num_datapoints());
  EXPECT_THAT(ScoreSimd(db_row_tail, query,
                        &TurboQuantDatasetSoA::UnsafeComputeDistances, indices),
              Pointwise(FloatEq(),
                        ScoreSimd(db_separate, query,
                                  &TurboQuantDatasetSoA::UnsafeComputeDistances,
                                  indices)));
  EXPECT_THAT(
      ScoreSimd(db_row_tail, ts_query,
                &TurboQuantDatasetSoA::UnsafeComputeDistancesTurboSlice,
                indices),
      Pointwise(
          FloatEq(),
          ScoreSimd(db_separate, ts_query,
                    &TurboQuantDatasetSoA::UnsafeComputeDistancesTurboSlice,
                    indices)));
}

TEST(TurboSliceQueryTest, DerivedFromFullQueryMatchesFloatEncoding) {
  const size_t dim = 128;
  absl::BitGen bitgen;
  const auto query_vecs = RandomVectors(1, dim, bitgen, /*normalize=*/false);
  for (size_t ts_dim : {63, 64, 65, 127, 128}) {
    TurboQuantCodec tq(dim, kSeed,
                       TurboQuantDistanceMetric::kDotProductDistance, ts_dim);
    ASSERT_OK_AND_ASSIGN(auto full_query,
                         TurboQuantQuery::Create(tq, query_vecs[0]));
    ASSERT_OK_AND_ASSIGN(auto expected, TurboQuantQuery::CreateTurboSliceQuery(
                                            tq, query_vecs[0]));
    ASSERT_OK_AND_ASSIGN(
        auto derived, TurboQuantQuery::CreateTurboSliceQuery(tq, full_query));
    ASSERT_EQ(derived.num_bytes_per_query(), expected.num_bytes_per_query());
    EXPECT_THAT(absl::MakeConstSpan(derived.query_data(),
                                    derived.num_bytes_per_query()),
                ::testing::ElementsAreArray(absl::MakeConstSpan(
                    expected.query_data(), expected.num_bytes_per_query())))
        << "ts_dim=" << ts_dim;
    EXPECT_EQ(derived.query_byte_sum(), expected.query_byte_sum())
        << "ts_dim=" << ts_dim;
  }
}

}  // namespace
}  // namespace ginger
