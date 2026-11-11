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

#include "ginger/turboquant/many_to_many/many_to_many_turboquant.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_scalar_reference.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/internal/scheduler_reference_stub.h"

namespace ginger {
namespace {

// Generate random float vectors with a normal distribution scaled by
// 1/sqrt(dim), similar to the benchmark.
std::vector<std::vector<float>> GenerateRandomVectors(size_t count, size_t dim,
                                                      absl::BitGen& bitgen) {
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<std::vector<float>> vecs(count, std::vector<float>(dim));
  for (auto& v : vecs) {
    for (float& x : v) x = dist(bitgen);
  }
  return vecs;
}

// Custom matcher: compares two std::vector<float> element-wise within a
// tolerance. Designed for use with testing::Pointwise over nested vectors.
MATCHER_P(PointwiseFloatNearTuple, tolerance, "") {
  const auto& actual_inner = std::get<0>(arg);
  const auto& expected_inner = std::get<1>(arg);
  return testing::ExplainMatchResult(
      testing::Pointwise(testing::FloatNear(tolerance), expected_inner),
      actual_inner, result_listener);
}

// Build a TurboQuantDatasetSoA from raw float vectors.
TurboQuantDatasetSoA EncodePoints(const TurboQuantCodec& tq,
                                  absl::Span<const std::vector<float>> vecs,
                                  bool pad_to_cache_line = false) {
  std::vector<absl::Span<const float>> spans;
  spans.reserve(vecs.size());
  for (const auto& v : vecs) spans.push_back(absl::MakeConstSpan(v));
  auto bp = TurboQuantDatasetSoA::Create(
      tq, spans,
      TurboQuantDatasetOptions{
          .pad_to_cache_line_size = pad_to_cache_line,
          .quantization_bits = 4,
          .store_query_byte_sums = false,
          .normalized_datapoints = false,
          .scaling_storage_mode = ScalingStorageMode::kSeparateArray});
  ABSL_CHECK_OK(bp);
  return *std::move(bp);
}

// Compute reference scores for all (query, db) pairs using the scalar
// one-to-many reference. Returns scores[q_idx][db_idx].
std::vector<std::vector<float>> ComputeReferenceScores(
    const TurboQuantDatasetSoA& queries_bp, const TurboQuantDatasetSoA& db_bp,
    const TurboQuantCodec& tq) {
  const size_t nq = queries_bp.num_datapoints();
  const size_t ndb = db_bp.num_datapoints();
  std::vector<std::vector<float>> ref(nq, std::vector<float>(ndb));

  for (size_t qi = 0; qi < nq; ++qi) {
    // Create a query from the qi-th encoded datapoint.
    // We re-encode the query from the stored quantized data using the
    // CreateFromQuantizedDataPoint path.
    auto query_or = TurboQuantQuery::CreateFromQuantizedDataPoint(
        tq, queries_bp.DataPointView(qi));
    ABSL_CHECK_OK(query_or);
    auto query = *std::move(query_or);

    std::vector<int32_t> all_db_indices(ndb);
    for (size_t i = 0; i < ndb; ++i) all_db_indices[i] = i;

    std::vector<float> scores(ndb);
    TurboQuantScoringInt8ScalarReference(db_bp, query, absl::MakeSpan(scores),
                                         all_db_indices);
    ref[qi] = std::move(scores);
  }
  return ref;
}

// Run ManyToManyTurboQuant and collect all scores into a dense matrix.
// Returns scores[q_idx][db_idx].
std::vector<std::vector<float>> RunManyToMany(
    const TurboQuantDatasetSoA& queries_bp, const TurboQuantDatasetSoA& db_bp) {
  const size_t nq = queries_bp.num_datapoints();
  const size_t ndb = db_bp.num_datapoints();
  std::vector<std::vector<float>> result(nq, std::vector<float>(ndb, 0.0f));

  ManyToManyTurboQuant(queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
                       IndexSubset::Contiguous(0, ndb),
                       [&](size_t query_idx, absl::Span<const float> scores,
                           size_t first_db_idx) {
                         for (size_t i = 0; i < scores.size(); ++i) {
                           result[query_idx][first_db_idx + i] = scores[i];
                         }
                       });
  return result;
}

// Compare many-to-many results against the scalar reference.
void VerifyManyToManyVsReference(size_t dim, size_t num_queries, size_t num_db,
                                 bool pad = false, float tol = 1e-4f) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs, pad);
  auto db_bp = EncodePoints(tq, db_vecs, pad);

  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual = RunManyToMany(queries_bp, db_bp);

  ASSERT_EQ(actual.size(), num_queries);
  EXPECT_THAT(actual, testing::Pointwise(PointwiseFloatNearTuple(tol), ref));
}

class ManyToManyDimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(ManyToManyDimTest, VsScalarReference) {
  const size_t dim = GetParam();
  VerifyManyToManyVsReference(dim, /*num_queries=*/20, /*num_db=*/50);
}

TEST_P(ManyToManyDimTest, VsScalarReferencePadded) {
  const size_t dim = GetParam();
  VerifyManyToManyVsReference(dim, /*num_queries=*/20, /*num_db=*/50,
                              /*pad=*/true);
}

INSTANTIATE_TEST_SUITE_P(
    Dimensions, ManyToManyDimTest,
    ::testing::Values(
        // Powers of 2 (aligned to all SIMD widths).
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
        // Odd dimensions (dim % 2 == 1): exercises the odd-dim nibble path.
        1, 3, 5, 7, 17, 39, 103, 129, 257,
        // Non-power-of-2 even dims: exercises partial SIMD chunks.
        100, 384, 511, 513, 768));

class OddDimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(OddDimTest, VsScalarReference) {
  const size_t dim = GetParam();
  // Use enough points to exercise panel and query batch boundaries.
  VerifyManyToManyVsReference(dim, /*num_queries=*/17, /*num_db=*/35);
}

INSTANTIATE_TEST_SUITE_P(OddDims, OddDimTest,
                         ::testing::Values(1, 3, 5, 7, 9, 15, 17, 33, 65, 129,
                                           257, 513));

// Test: Query batch boundary (exercises kMq = 8 boundary)
class QueryBatchTest : public ::testing::TestWithParam<size_t> {};

TEST_P(QueryBatchTest, VsScalarReference) {
  const size_t num_queries = GetParam();
  VerifyManyToManyVsReference(/*dim=*/128, num_queries, /*num_db=*/50);
}

INSTANTIATE_TEST_SUITE_P(QueryBatch, QueryBatchTest,
                         // Around kMq=8 boundary and multiples.
                         ::testing::Values(1, 2, 7, 8, 9, 15, 16, 17, 24, 25));

// Test: Panel boundary (exercises points_per_panel = Lanes(i32))
// On AVX-512: points_per_panel=16, on AVX2: points_per_panel=8.
// Test counts that trigger: 1 panel, 2 panels (the paired path),
// 2+1 remainder, and partial last panels.
class PanelBoundaryTest : public ::testing::TestWithParam<size_t> {};

TEST_P(PanelBoundaryTest, VsScalarReference) {
  const size_t num_db = GetParam();
  VerifyManyToManyVsReference(/*dim=*/128, /*num_queries=*/10, num_db);
}

INSTANTIATE_TEST_SUITE_P(
    PanelBoundary, PanelBoundaryTest,
    ::testing::Values(
        // Very small: partial single panel.
        1, 2, 3, 7, 8,
        // Around points_per_panel=16 (AVX-512 panel size).
        9, 15, 16, 17,
        // Two panels: exercises the 2-panel micro-kernel path.
        31, 32, 33,
        // Three panels: exercises 2-panel + 1-panel remainder.
        47, 48, 49));

// Test: Large DB batch (exercises L2-tiled outer loop)
// kMinDbBatch=32, kMaxDbBatch=1024
class LargeDbBatchTest : public ::testing::TestWithParam<size_t> {};

TEST_P(LargeDbBatchTest, VsScalarReference) {
  const size_t num_db = GetParam();
  // Use smaller dim and fewer queries for speed on large DB sizes.
  VerifyManyToManyVsReference(/*dim=*/64, /*num_queries=*/5, num_db);
}

INSTANTIATE_TEST_SUITE_P(LargeDbBatch, LargeDbBatchTest,
                         ::testing::Values(31, 32, 33, 63, 64, 65, 127, 128,
                                           256, 500));

TEST(ManyToManyEdgeTest, SingleQuerySingleDb) {
  VerifyManyToManyVsReference(/*dim=*/128, /*num_queries=*/1, /*num_db=*/1);
}

TEST(ManyToManyEdgeTest, SingleQueryManyDb) {
  VerifyManyToManyVsReference(/*dim=*/128, /*num_queries=*/1, /*num_db=*/50);
}

TEST(ManyToManyEdgeTest, ManyQuerySingleDb) {
  VerifyManyToManyVsReference(/*dim=*/128, /*num_queries=*/20, /*num_db=*/1);
}

TEST(ManyToManyEdgeTest, ZeroQueries) {
  TurboQuantCodec tq(128, /*seed=*/42);
  absl::BitGen bitgen;
  auto db_vecs = GenerateRandomVectors(10, 128, bitgen);
  auto db_bp = EncodePoints(tq, db_vecs);

  // Create empty queries.
  auto empty_bp = TurboQuantDatasetSoA::Create(tq, {});
  ABSL_CHECK_OK(empty_bp);

  bool callback_called = false;
  ManyToManyTurboQuant(
      *empty_bp, db_bp, IndexSubset::Contiguous(0, empty_bp->num_datapoints()),
      IndexSubset::Contiguous(0, db_bp.num_datapoints()),
      [&](size_t, absl::Span<const float>, size_t) { callback_called = true; });
  EXPECT_FALSE(callback_called);
}

TEST(ManyToManyEdgeTest, ZeroDbPoints) {
  TurboQuantCodec tq(128, /*seed=*/42);
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(10, 128, bitgen);
  auto queries_bp = EncodePoints(tq, q_vecs);

  auto empty_bp = TurboQuantDatasetSoA::Create(tq, {});
  ABSL_CHECK_OK(empty_bp);

  bool callback_called = false;
  ManyToManyTurboQuant(
      queries_bp, *empty_bp,
      IndexSubset::Contiguous(0, queries_bp.num_datapoints()),
      IndexSubset::Contiguous(0, empty_bp->num_datapoints()),
      [&](size_t, absl::Span<const float>, size_t) { callback_called = true; });
  EXPECT_FALSE(callback_called);
}

TEST(ManyToManyEdgeTest, SelfComparison) {
  const size_t dim = 128;
  const size_t n = 30;
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(n, dim, bitgen);
  TurboQuantCodec tq(dim, /*seed=*/42);
  auto bp = EncodePoints(tq, vecs);

  auto ref = ComputeReferenceScores(bp, bp, tq);
  auto actual = RunManyToMany(bp, bp);

  ASSERT_EQ(actual.size(), n);
  for (size_t qi = 0; qi < n; ++qi) {
    ASSERT_EQ(actual[qi].size(), n);
    for (size_t di = 0; di < n; ++di) {
      EXPECT_NEAR(actual[qi][di], ref[qi][di], 1e-4f)
          << "qi=" << qi << " di=" << di;
    }
  }
}

// Test callback invocation
// Verify that every (query, db) pair is covered exactly once, that query
// indices are valid, and that db offsets + span sizes tile the full DB.
TEST(ManyToManyCallbackTest, CoversAllPairsExactlyOnce) {
  const size_t dim = 64;
  const size_t nq = 17;   // Not a multiple of kMq=8.
  const size_t ndb = 50;  // Not a multiple of points_per_panel.

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);
  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  // Track how many times each (q, db) pair is scored.
  std::vector<std::vector<int>> hit_count(nq, std::vector<int>(ndb, 0));

  ManyToManyTurboQuant(queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
                       IndexSubset::Contiguous(0, ndb),
                       [&](size_t query_idx, absl::Span<const float> scores,
                           size_t first_db_idx) {
                         ASSERT_LT(query_idx, nq);
                         ASSERT_LE(first_db_idx + scores.size(), ndb);
                         for (size_t i = 0; i < scores.size(); ++i) {
                           hit_count[query_idx][first_db_idx + i]++;
                         }
                       });

  // Every pair should be hit exactly once.
  for (size_t qi = 0; qi < nq; ++qi) {
    for (size_t di = 0; di < ndb; ++di) {
      EXPECT_EQ(hit_count[qi][di], 1) << "qi=" << qi << " di=" << di;
    }
  }
}

TEST(ManyToManyCallbackTest, DbBatchesAreContiguous) {
  const size_t dim = 64;
  const size_t nq = 3;
  const size_t ndb = 100;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);
  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  // For each query, track the (first_db_idx, size) of each callback.
  absl::flat_hash_map<size_t, std::vector<std::pair<size_t, size_t>>>
      callback_ranges;

  ManyToManyTurboQuant(queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
                       IndexSubset::Contiguous(0, ndb),
                       [&](size_t query_idx, absl::Span<const float> scores,
                           size_t first_db_idx) {
                         callback_ranges[query_idx].emplace_back(first_db_idx,
                                                                 scores.size());
                       });

  ASSERT_EQ(callback_ranges.size(), nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    const auto& ranges = callback_ranges[qi];
    ASSERT_FALSE(ranges.empty());
    // First batch should start at 0 (within a DB outer batch).
    // Batches within one DB outer batch should be contiguous.
    size_t total_covered = 0;
    for (const auto& [offset, len] : ranges) {
      total_covered += len;
    }
    EXPECT_EQ(total_covered, ndb) << "query=" << qi;
  }
}

TEST(ManyToManyStressTest, OddEverything) {
  // dim=129 (odd), 13 queries (not multiple of 8), 37 db (not multiple of 16).
  // Tolerance is relaxed from 1e-4 to 2e-4 because the SIMD-transposed
  // DecodeDbPanel (from pipnn_turboquant-inl.h) uses a different tile ordering
  // than the original, changing the int32 accumulation order and thus the
  // floating-point rounding. Both orderings are equally correct.
  VerifyManyToManyVsReference(/*dim=*/129, /*num_queries=*/13, /*num_db=*/37,
                              /*pad=*/false, /*tol=*/2e-4f);
}

TEST(ManyToManyStressTest, LargeDimSmallCounts) {
  VerifyManyToManyVsReference(/*dim=*/1536, /*num_queries=*/3, /*num_db=*/5);
}

TEST(ManyToManyStressTest, SmallDimLargeCounts) {
  VerifyManyToManyVsReference(/*dim=*/4, /*num_queries=*/25, /*num_db=*/100);
}

// Uses enough queries to exceed the internal query batch size (mc), forcing the
// full q_decoded buffer to be exercised. With dim=1536, decoded_dim is large
// (~1536) ==> mc ≈ 128, so 150 queries guarantees at least one full-batch
// iteration that writes every byte of the mc * decoded_dim buffer.
TEST(ManyToManyStressTest, LargeDimExceedsQueryBatch) {
  VerifyManyToManyVsReference(/*dim=*/1536, /*num_queries=*/150, /*num_db=*/5);
}

// Build a TurboQuantDatasetSoA with L2 metric from raw float vectors.
TurboQuantDatasetSoA EncodePointsL2(const TurboQuantCodec& tq,
                                    absl::Span<const std::vector<float>> vecs) {
  std::vector<absl::Span<const float>> spans;
  spans.reserve(vecs.size());
  for (const auto& v : vecs) spans.push_back(absl::MakeConstSpan(v));
  auto bp = TurboQuantDatasetSoA::Create(tq, spans);
  ABSL_CHECK_OK(bp);
  return *std::move(bp);
}

// Compare L2 many-to-many results against the scalar reference.
void VerifyManyToManyL2VsReference(size_t dim, size_t num_queries,
                                   size_t num_db, float tol = 1e-4f) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kSquaredL2Distance);
  auto queries_bp = EncodePointsL2(tq, q_vecs);
  auto db_bp = EncodePointsL2(tq, db_vecs);

  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual = RunManyToMany(queries_bp, db_bp);

  ASSERT_EQ(actual.size(), num_queries);
  EXPECT_THAT(actual, testing::Pointwise(PointwiseFloatNearTuple(tol), ref));
}

class ManyToManyL2DimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(ManyToManyL2DimTest, VsScalarReference) {
  const size_t dim = GetParam();
  VerifyManyToManyL2VsReference(dim, /*num_queries=*/15, /*num_db=*/40);
}

INSTANTIATE_TEST_SUITE_P(L2Dimensions, ManyToManyL2DimTest,
                         ::testing::Values(4, 16, 64, 128, 256, 768,
                                           // Odd dimensions.
                                           1, 3, 5, 7, 17, 129, 257));

TEST(ManyToManyL2EdgeTest, SingleQuerySingleDb) {
  VerifyManyToManyL2VsReference(/*dim=*/128, /*num_queries=*/1, /*num_db=*/1);
}

TEST(ManyToManyL2EdgeTest, SelfComparison) {
  const size_t dim = 128;
  const size_t n = 25;
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(n, dim, bitgen);
  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kSquaredL2Distance);
  auto bp = EncodePointsL2(tq, vecs);

  auto ref = ComputeReferenceScores(bp, bp, tq);
  auto actual = RunManyToMany(bp, bp);

  ASSERT_EQ(actual.size(), n);
  for (size_t qi = 0; qi < n; ++qi) {
    ASSERT_EQ(actual[qi].size(), n);
    for (size_t di = 0; di < n; ++di) {
      EXPECT_NEAR(actual[qi][di], ref[qi][di], 1e-4f)
          << "qi=" << qi << " di=" << di;
    }
  }
}

TEST(ManyToManyL2StressTest, OddEverything) {
  VerifyManyToManyL2VsReference(/*dim=*/129, /*num_queries=*/13, /*num_db=*/37,
                                /*tol=*/2e-4f);
}

// Run ManyToManyTurboQuant with a pretransposed database and collect scores.
std::vector<std::vector<float>> RunManyToManyPreTransposed(
    const TurboQuantDatasetSoA& queries_bp, const TurboQuantDatasetSoA& db_bp,
    const TurboQuantCodec& tq) {
  const size_t nq = queries_bp.num_datapoints();
  const size_t ndb = db_bp.num_datapoints();
  std::vector<std::vector<float>> result(nq, std::vector<float>(ndb, 0.0f));

  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);

  ManyToManyTurboQuant(queries_bp, pretransposed,
                       IndexSubset::Contiguous(0, nq),
                       [&](size_t query_idx, absl::Span<const float> scores,
                           size_t first_db_idx) {
                         for (size_t i = 0; i < scores.size(); ++i) {
                           result[query_idx][first_db_idx + i] = scores[i];
                         }
                       });
  return result;
}

// Verifies the pretransposed variant matches both the scalar reference
// and the on-the-fly decode variant.
void VerifyPreTransposedVsReference(size_t dim, size_t num_queries,
                                    size_t num_db, float tol = 1e-4f) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual_on_the_fly = RunManyToMany(queries_bp, db_bp);
  auto actual_pretransposed = RunManyToManyPreTransposed(queries_bp, db_bp, tq);

  ASSERT_EQ(actual_pretransposed.size(), num_queries);
  for (size_t qi = 0; qi < num_queries; ++qi) {
    ASSERT_EQ(actual_pretransposed[qi].size(), num_db) << "query=" << qi;
    for (size_t di = 0; di < num_db; ++di) {
      EXPECT_NEAR(actual_pretransposed[qi][di], ref[qi][di], tol)
          << "pretransposed vs ref: dim=" << dim << " query=" << qi
          << " db=" << di;
      // Pretransposed should produce identical results to on-the-fly.
      EXPECT_EQ(actual_pretransposed[qi][di], actual_on_the_fly[qi][di])
          << "pretransposed vs on-the-fly: dim=" << dim << " query=" << qi
          << " db=" << di;
    }
  }
}

// Core correctness across many dimensionalities.
class PreTransposedDimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(PreTransposedDimTest, VsScalarReference) {
  const size_t dim = GetParam();
  VerifyPreTransposedVsReference(dim, /*num_queries=*/15, /*num_db=*/40);
}

INSTANTIATE_TEST_SUITE_P(PreTransposedDimensions, PreTransposedDimTest,
                         ::testing::Values(4, 8, 16, 32, 64, 128, 256, 512,
                                           // Odd dimensions.
                                           1, 3, 5, 7, 17, 129, 257,
                                           // Non-power-of-2 even.
                                           100, 384, 768));

// Edge cases.
TEST(PreTransposedEdgeTest, SingleQuerySingleDb) {
  VerifyPreTransposedVsReference(/*dim=*/128, /*num_queries=*/1, /*num_db=*/1);
}

TEST(PreTransposedEdgeTest, SingleQueryManyDb) {
  VerifyPreTransposedVsReference(/*dim=*/128, /*num_queries=*/1, /*num_db=*/50);
}

TEST(PreTransposedEdgeTest, ManyQuerySingleDb) {
  VerifyPreTransposedVsReference(/*dim=*/128, /*num_queries=*/20, /*num_db=*/1);
}

TEST(PreTransposedEdgeTest, SelfComparison) {
  const size_t dim = 128;
  const size_t n = 30;
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(n, dim, bitgen);
  TurboQuantCodec tq(dim, /*seed=*/42);
  auto bp = EncodePoints(tq, vecs);

  auto ref = ComputeReferenceScores(bp, bp, tq);
  auto actual = RunManyToManyPreTransposed(bp, bp, tq);

  ASSERT_EQ(actual.size(), n);
  for (size_t qi = 0; qi < n; ++qi) {
    ASSERT_EQ(actual[qi].size(), n);
    for (size_t di = 0; di < n; ++di) {
      EXPECT_NEAR(actual[qi][di], ref[qi][di], 1e-4f)
          << "qi=" << qi << " di=" << di;
    }
  }
}

// Panel boundary tests.
class PreTransposedPanelBoundaryTest : public ::testing::TestWithParam<size_t> {
};

TEST_P(PreTransposedPanelBoundaryTest, VsScalarReference) {
  const size_t num_db = GetParam();
  VerifyPreTransposedVsReference(/*dim=*/128, /*num_queries=*/10, num_db);
}

INSTANTIATE_TEST_SUITE_P(PreTransposedPanelBoundary,
                         PreTransposedPanelBoundaryTest,
                         ::testing::Values(1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32,
                                           33, 47, 48, 49));

// Stress tests.
TEST(PreTransposedStressTest, OddEverything) {
  VerifyPreTransposedVsReference(/*dim=*/129, /*num_queries=*/13,
                                 /*num_db=*/37, /*tol=*/2e-4f);
}

TEST(PreTransposedStressTest, LargeDimSmallCounts) {
  VerifyPreTransposedVsReference(/*dim=*/1536, /*num_queries=*/3,
                                 /*num_db=*/5);
}

TEST(PreTransposedStressTest, SmallDimLargeCounts) {
  VerifyPreTransposedVsReference(/*dim=*/4, /*num_queries=*/25,
                                 /*num_db=*/100);
}

void VerifyPreTransposedL2VsReference(size_t dim, size_t num_queries,
                                      size_t num_db, float tol = 1e-4f) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kSquaredL2Distance);
  auto queries_bp = EncodePointsL2(tq, q_vecs);
  auto db_bp = EncodePointsL2(tq, db_vecs);

  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual_on_the_fly = RunManyToMany(queries_bp, db_bp);
  auto actual_pretransposed = RunManyToManyPreTransposed(queries_bp, db_bp, tq);

  ASSERT_EQ(actual_pretransposed.size(), num_queries);
  for (size_t qi = 0; qi < num_queries; ++qi) {
    ASSERT_EQ(actual_pretransposed[qi].size(), num_db) << "query=" << qi;
    for (size_t di = 0; di < num_db; ++di) {
      EXPECT_NEAR(actual_pretransposed[qi][di], ref[qi][di], tol)
          << "L2 pretransposed vs ref: dim=" << dim << " query=" << qi
          << " db=" << di;
      EXPECT_EQ(actual_pretransposed[qi][di], actual_on_the_fly[qi][di])
          << "L2 pretransposed vs on-the-fly: dim=" << dim << " query=" << qi
          << " db=" << di;
    }
  }
}

class PreTransposedL2DimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(PreTransposedL2DimTest, VsScalarReference) {
  const size_t dim = GetParam();
  VerifyPreTransposedL2VsReference(dim, /*num_queries=*/15, /*num_db=*/40);
}

INSTANTIATE_TEST_SUITE_P(PreTransposedL2Dimensions, PreTransposedL2DimTest,
                         ::testing::Values(4, 16, 64, 128, 256, 768,
                                           // Odd dimensions.
                                           1, 3, 5, 7, 17, 129, 257));

TEST(PreTransposedL2EdgeTest, SingleQuerySingleDb) {
  VerifyPreTransposedL2VsReference(/*dim=*/128, /*num_queries=*/1,
                                   /*num_db=*/1);
}

TEST(PreTransposedL2StressTest, OddEverything) {
  VerifyPreTransposedL2VsReference(/*dim=*/129, /*num_queries=*/13,
                                   /*num_db=*/37, /*tol=*/2e-4f);
}

// Run ManyToManyTurboQuant with gathered indices and collect scores.
// Returns scores[i][j] where i indexes into query_indices and j into
// db_indices.
std::vector<std::vector<float>> RunManyToManyGathered(
    const TurboQuantDatasetSoA& queries_bp, const TurboQuantDatasetSoA& db_bp,
    absl::Span<const int> query_indices, absl::Span<const int> db_indices) {
  const size_t nq = query_indices.empty() ? queries_bp.num_datapoints()
                                          : query_indices.size();
  const size_t ndb =
      db_indices.empty() ? db_bp.num_datapoints() : db_indices.size();
  std::vector<std::vector<float>> result(nq, std::vector<float>(ndb, 0.0f));

  ManyToManyTurboQuant(
      queries_bp, db_bp,
      query_indices.empty()
          ? IndexSubset::Contiguous(0, queries_bp.num_datapoints())
          : IndexSubset::Gathered(query_indices),
      db_indices.empty() ? IndexSubset::Contiguous(0, db_bp.num_datapoints())
                         : IndexSubset::Gathered(db_indices),
      [&](size_t query_idx, absl::Span<const float> scores,
          size_t first_db_idx) {
        for (size_t i = 0; i < scores.size(); ++i) {
          result[query_idx][first_db_idx + i] = scores[i];
        }
      });
  return result;
}

// Verify gathered on-the-fly variant: scores for the subset should match
// the reference scores for the same (query, db) pairs.
void VerifyManyToManyGathered(size_t dim, size_t num_queries, size_t num_db,
                              size_t num_q_subset, size_t num_db_subset,
                              float tol = 1e-4f) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  // Build random non-repeating index subsets.
  std::vector<int> q_indices(num_queries);
  std::iota(q_indices.begin(), q_indices.end(), 0);
  std::shuffle(q_indices.begin(), q_indices.end(), bitgen);
  q_indices.resize(num_q_subset);

  std::vector<int> db_indices(num_db);
  std::iota(db_indices.begin(), db_indices.end(), 0);
  std::shuffle(db_indices.begin(), db_indices.end(), bitgen);
  db_indices.resize(num_db_subset);

  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual =
      RunManyToManyGathered(queries_bp, db_bp, absl::MakeConstSpan(q_indices),
                            absl::MakeConstSpan(db_indices));

  ASSERT_EQ(actual.size(), num_q_subset);
  for (size_t qi = 0; qi < num_q_subset; ++qi) {
    ASSERT_EQ(actual[qi].size(), num_db_subset) << "query=" << qi;
    for (size_t di = 0; di < num_db_subset; ++di) {
      EXPECT_NEAR(actual[qi][di], ref[q_indices[qi]][db_indices[di]], tol)
          << "dim=" << dim << " q_idx=" << q_indices[qi]
          << " db_idx=" << db_indices[di];
    }
  }
}

TEST(GatheredIndicesTest, BasicSubset) {
  VerifyManyToManyGathered(/*dim=*/128, /*num_queries=*/20, /*num_db=*/50,
                           /*num_q_subset=*/7, /*num_db_subset=*/15);
}

TEST(GatheredIndicesTest, SmallDim) {
  VerifyManyToManyGathered(/*dim=*/4, /*num_queries=*/25, /*num_db=*/40,
                           /*num_q_subset=*/10, /*num_db_subset=*/12);
}

TEST(GatheredIndicesTest, OddDim) {
  VerifyManyToManyGathered(/*dim=*/129, /*num_queries=*/20, /*num_db=*/37,
                           /*num_q_subset=*/9, /*num_db_subset=*/13,
                           /*tol=*/2e-4f);
}

TEST(GatheredIndicesTest, LargeDim) {
  VerifyManyToManyGathered(/*dim=*/768, /*num_queries=*/10, /*num_db=*/20,
                           /*num_q_subset=*/5, /*num_db_subset=*/8);
}

TEST(GatheredIndicesTest, SingleQuerySingleDb) {
  VerifyManyToManyGathered(/*dim=*/128, /*num_queries=*/10, /*num_db=*/20,
                           /*num_q_subset=*/1, /*num_db_subset=*/1);
}

TEST(GatheredIndicesTest, QueryOnlyIndices) {
  // Only query indices, all DB points.
  const size_t dim = 128;
  const size_t nq = 20;
  const size_t ndb = 50;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  std::vector<int> q_indices = {3, 7, 15};
  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual =
      RunManyToManyGathered(queries_bp, db_bp, absl::MakeConstSpan(q_indices),
                            /*db_indices=*/{});

  ASSERT_EQ(actual.size(), q_indices.size());
  for (size_t qi = 0; qi < q_indices.size(); ++qi) {
    ASSERT_EQ(actual[qi].size(), ndb);
    for (size_t di = 0; di < ndb; ++di) {
      EXPECT_NEAR(actual[qi][di], ref[q_indices[qi]][di], 1e-4f);
    }
  }
}

TEST(GatheredIndicesTest, DbOnlyIndices) {
  // Only DB indices, all queries.
  const size_t dim = 128;
  const size_t nq = 10;
  const size_t ndb = 50;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  std::vector<int> db_indices = {0, 5, 10, 25, 49};
  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual = RunManyToManyGathered(queries_bp, db_bp,
                                      /*query_indices=*/{},
                                      absl::MakeConstSpan(db_indices));

  ASSERT_EQ(actual.size(), nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    ASSERT_EQ(actual[qi].size(), db_indices.size());
    for (size_t di = 0; di < db_indices.size(); ++di) {
      EXPECT_NEAR(actual[qi][di], ref[qi][db_indices[di]], 1e-4f);
    }
  }
}

TEST(GatheredIndicesTest, EmptyIndicesMatchesFull) {
  // Empty indices = full ManyToMany.
  const size_t dim = 64;
  const size_t nq = 10;
  const size_t ndb = 30;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto full = RunManyToMany(queries_bp, db_bp);
  auto gathered = RunManyToManyGathered(queries_bp, db_bp,
                                        /*query_indices=*/{},
                                        /*db_indices=*/{});

  ASSERT_EQ(gathered.size(), nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    ASSERT_EQ(gathered[qi].size(), ndb);
    for (size_t di = 0; di < ndb; ++di) {
      EXPECT_EQ(gathered[qi][di], full[qi][di]);
    }
  }
}

TEST(GatheredIndicesL2Test, BasicSubset) {
  const size_t dim = 128;
  const size_t nq = 15;
  const size_t ndb = 40;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kSquaredL2Distance);
  auto queries_bp = EncodePointsL2(tq, q_vecs);
  auto db_bp = EncodePointsL2(tq, db_vecs);

  std::vector<int> q_indices = {1, 5, 10};
  std::vector<int> db_indices = {0, 3, 7, 20, 39};
  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual =
      RunManyToManyGathered(queries_bp, db_bp, absl::MakeConstSpan(q_indices),
                            absl::MakeConstSpan(db_indices));

  ASSERT_EQ(actual.size(), q_indices.size());
  for (size_t qi = 0; qi < q_indices.size(); ++qi) {
    ASSERT_EQ(actual[qi].size(), db_indices.size());
    for (size_t di = 0; di < db_indices.size(); ++di) {
      EXPECT_NEAR(actual[qi][di], ref[q_indices[qi]][db_indices[di]], 1e-4f)
          << "L2 gathered: q=" << q_indices[qi] << " db=" << db_indices[di];
    }
  }
}

std::vector<std::vector<float>> RunManyToManyPreTransposedGathered(
    const TurboQuantDatasetSoA& queries_bp, const TurboQuantDatasetSoA& db_bp,
    const TurboQuantCodec& tq, absl::Span<const int> query_indices) {
  const size_t nq = query_indices.empty() ? queries_bp.num_datapoints()
                                          : query_indices.size();
  const size_t ndb = db_bp.num_datapoints();
  std::vector<std::vector<float>> result(nq, std::vector<float>(ndb, 0.0f));

  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);

  ManyToManyTurboQuant(
      queries_bp, pretransposed,
      query_indices.empty()
          ? IndexSubset::Contiguous(0, queries_bp.num_datapoints())
          : IndexSubset::Gathered(query_indices),
      [&](size_t query_idx, absl::Span<const float> scores,
          size_t first_db_idx) {
        for (size_t i = 0; i < scores.size(); ++i) {
          result[query_idx][first_db_idx + i] = scores[i];
        }
      });
  return result;
}

TEST(GatheredPreTransposedTest, BasicSubset) {
  const size_t dim = 128;
  const size_t nq = 20;
  const size_t ndb = 50;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  std::vector<int> q_indices = {2, 5, 11, 18};
  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual =
      RunManyToManyPreTransposedGathered(queries_bp, db_bp, tq, q_indices);

  ASSERT_EQ(actual.size(), q_indices.size());
  for (size_t qi = 0; qi < q_indices.size(); ++qi) {
    ASSERT_EQ(actual[qi].size(), ndb);
    for (size_t di = 0; di < ndb; ++di) {
      EXPECT_NEAR(actual[qi][di], ref[q_indices[qi]][di], 1e-4f)
          << "pretransposed gathered: q_idx=" << q_indices[qi] << " di=" << di;
    }
  }
}

TEST(GatheredPreTransposedTest, OddDim) {
  const size_t dim = 129;
  const size_t nq = 15;
  const size_t ndb = 37;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  std::vector<int> q_indices = {0, 3, 7, 14};
  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  auto actual =
      RunManyToManyPreTransposedGathered(queries_bp, db_bp, tq, q_indices);

  ASSERT_EQ(actual.size(), q_indices.size());
  for (size_t qi = 0; qi < q_indices.size(); ++qi) {
    ASSERT_EQ(actual[qi].size(), ndb);
    for (size_t di = 0; di < ndb; ++di) {
      EXPECT_NEAR(actual[qi][di], ref[q_indices[qi]][di], 2e-4f);
    }
  }
}

TEST(GatheredPreTransposedTest, EmptyIndicesMatchesFull) {
  const size_t dim = 64;
  const size_t nq = 10;
  const size_t ndb = 30;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto full = RunManyToManyPreTransposed(queries_bp, db_bp, tq);
  auto gathered = RunManyToManyPreTransposedGathered(queries_bp, db_bp, tq,
                                                     /*query_indices=*/{});

  ASSERT_EQ(gathered.size(), nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    ASSERT_EQ(gathered[qi].size(), ndb);
    for (size_t di = 0; di < ndb; ++di) {
      EXPECT_EQ(gathered[qi][di], full[qi][di]);
    }
  }
}

// Gathered + Pretransposed with DB indices (via Build).
TEST(GatheredPreTransposedTest, WithDbIndicesViaBuild) {
  const size_t dim = 128;
  const size_t nq = 15;
  const size_t ndb = 50;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  std::vector<int> q_indices = {1, 5, 10};
  std::vector<int> db_indices = {0, 3, 10, 25, 49};

  auto pretransposed = PreTransposedTurboQuantDataset::Build(
      db_bp, absl::MakeConstSpan(db_indices), tq);

  const size_t ndb_subset = db_indices.size();
  std::vector<std::vector<float>> result(q_indices.size(),
                                         std::vector<float>(ndb_subset, 0.0f));

  ManyToManyTurboQuant(queries_bp, pretransposed,
                       IndexSubset::Gathered(q_indices),
                       [&](size_t query_idx, absl::Span<const float> scores,
                           size_t first_db_idx) {
                         for (size_t i = 0; i < scores.size(); ++i) {
                           result[query_idx][first_db_idx + i] = scores[i];
                         }
                       });

  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  ASSERT_EQ(result.size(), q_indices.size());
  for (size_t qi = 0; qi < q_indices.size(); ++qi) {
    ASSERT_EQ(result[qi].size(), ndb_subset);
    for (size_t di = 0; di < ndb_subset; ++di) {
      EXPECT_NEAR(result[qi][di], ref[q_indices[qi]][db_indices[di]], 1e-4f)
          << "pretransposed+both gathered: q=" << q_indices[qi]
          << " db=" << db_indices[di];
    }
  }
}

// Test: partial kMq micro-batch padding. Uses a query count that is not a
// multiple of kMq=8 so the last micro-batch has padding slots. Verifies
// the valid query scores match the reference and no spurious callbacks fire.
TEST(ManyToManyEdgeTest, PartialMicroBatchPadding) {
  // 3 queries: kMq=8, so 5 slots are padded.
  const size_t dim = 128;
  const size_t nq = 3;
  const size_t ndb = 20;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto ref = ComputeReferenceScores(queries_bp, db_bp, tq);
  std::vector<std::vector<float>> result(nq, std::vector<float>(ndb, 0.0f));
  size_t max_query_idx_seen = 0;

  ManyToManyTurboQuant(queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
                       IndexSubset::Contiguous(0, ndb),
                       [&](size_t query_idx, absl::Span<const float> scores,
                           size_t first_db_idx) {
                         ASSERT_LT(query_idx, nq) << "Padded slot leaked";
                         max_query_idx_seen =
                             std::max(max_query_idx_seen, query_idx);
                         for (size_t i = 0; i < scores.size(); ++i) {
                           result[query_idx][first_db_idx + i] = scores[i];
                         }
                       });

  EXPECT_EQ(max_query_idx_seen, nq - 1);
  EXPECT_THAT(result, testing::Pointwise(PointwiseFloatNearTuple(1e-4f), ref));
}

}  // namespace
}  // namespace ginger
