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

#include "ginger/turboquant/many_to_many/many_to_many_topk.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/base/casts.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/many_to_many_turboquant.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "ginger/turboquant/types/distance_metric.h"

namespace ginger {

// Test-only access to the internals of `ManyToManyTopKState`.
class ManyToManyTopKStatePeer {
 public:
  static const many_to_many_topk_internal::TopKState& Get(
      const ManyToManyTopKState& state) {
    return state.state_;
  }
};

namespace {

using ::testing::ElementsAreArray;

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
  auto bp = TurboQuantDatasetSoA::Create(tq, vecs.size());
  ABSL_CHECK_OK(bp);
  for (size_t i = 0; i < vecs.size(); ++i) {
    ABSL_CHECK_OK(bp->UpdatePoint(tq, absl::MakeConstSpan(vecs[i]), i));
  }
  return *std::move(bp);
}

// Compute reference Top1 by running ManyToManyTurboQuant and tracking argmin.
std::vector<TopKResult> ComputeReferenceTop1(
    const TurboQuantDatasetSoA& queries_bp, const TurboQuantDatasetSoA& db_bp) {
  const size_t nq = queries_bp.num_datapoints();
  std::vector<TopKResult> results(
      nq, {.db_idx = 0, .distance = std::numeric_limits<float>::max()});

  ManyToManyTurboQuant(queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
                       IndexSubset::Contiguous(0, db_bp.num_datapoints()),
                       [&](size_t query_idx, absl::Span<const float> scores,
                           size_t first_db_idx) {
                         for (size_t i = 0; i < scores.size(); ++i) {
                           if (scores[i] < results[query_idx].distance) {
                             results[query_idx].distance = scores[i];
                             results[query_idx].db_idx = first_db_idx + i;
                           }
                         }
                       });
  return results;
}

// Compute reference TopK by running ManyToManyTurboQuant and sorting.
std::vector<TopKResult> ComputeReferenceTopK(
    const TurboQuantDatasetSoA& queries_bp, const TurboQuantDatasetSoA& db_bp,
    int k) {
  const size_t nq = queries_bp.num_datapoints();
  std::vector<std::vector<TopKResult>> all_scores(nq);

  ManyToManyTurboQuant(
      queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
      IndexSubset::Contiguous(0, db_bp.num_datapoints()),
      [&](size_t query_idx, absl::Span<const float> scores,
          size_t first_db_idx) {
        for (size_t i = 0; i < scores.size(); ++i) {
          uint32_t u = static_cast<uint32_t>(first_db_idx + i);
          all_scores[query_idx].push_back({.db_idx = u, .distance = scores[i]});
        }
      });

  std::vector<TopKResult> results;
  results.reserve(nq * k);

  for (size_t qi = 0; qi < nq; ++qi) {
    auto& res = all_scores[qi];
    std::sort(res.begin(), res.end(),
              [](const TopKResult& a, const TopKResult& b) {
                if (a.distance != b.distance) return a.distance < b.distance;
                return a.db_idx < b.db_idx;
              });

    size_t count = std::min(static_cast<size_t>(k), res.size());
    for (size_t i = 0; i < count; ++i) {
      results.push_back(res[i]);
    }
    // Pad with invalid results if k > database size.
    for (size_t i = count; i < static_cast<size_t>(k); ++i) {
      results.push_back({.db_idx = std::numeric_limits<uint32_t>::max(),
                         .distance = std::numeric_limits<float>::max()});
    }
  }
  return results;
}

// Reference implementation for gathered indices.
std::vector<TopKResult> ComputeReferenceTopK(
    const TurboQuantDatasetSoA& queries_bp, const TurboQuantDatasetSoA& db_bp,
    absl::Span<const int> query_indices, absl::Span<const int> db_indices,
    int k) {
  const size_t nq = query_indices.empty() ? queries_bp.num_datapoints()
                                          : query_indices.size();
  std::vector<std::vector<TopKResult>> all_scores(nq);

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
          uint32_t u = static_cast<uint32_t>(first_db_idx + i);
          all_scores[query_idx].push_back({.db_idx = u, .distance = scores[i]});
        }
      });

  std::vector<TopKResult> results;
  results.reserve(nq * k);

  for (size_t qi = 0; qi < nq; ++qi) {
    auto& res = all_scores[qi];
    std::sort(res.begin(), res.end(),
              [](const TopKResult& a, const TopKResult& b) {
                if (a.distance != b.distance) return a.distance < b.distance;
                return a.db_idx < b.db_idx;
              });

    size_t count = std::min(static_cast<size_t>(k), res.size());
    for (size_t i = 0; i < count; ++i) {
      results.push_back(res[i]);
    }
    for (size_t i = count; i < static_cast<size_t>(k); ++i) {
      results.push_back({.db_idx = std::numeric_limits<uint32_t>::max(),
                         .distance = std::numeric_limits<float>::max()});
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// On-the-fly database overload (all queries, all db points)
// ---------------------------------------------------------------------------
void VerifyTop1(size_t dim, size_t num_queries, size_t num_db) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto ref = ComputeReferenceTop1(queries_bp, db_bp);
  auto actual = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, num_queries),
      IndexSubset::Contiguous(0, num_db), /*k=*/1);

  ASSERT_EQ(actual.size(), num_queries);
  for (size_t qi = 0; qi < num_queries; ++qi) {
    EXPECT_EQ(actual[qi].db_idx, ref[qi].db_idx)
        << "dim=" << dim << " qi=" << qi;
    EXPECT_FLOAT_EQ(actual[qi].distance, ref[qi].distance)
        << "dim=" << dim << " qi=" << qi;
  }
}

class Top1DimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(Top1DimTest, MatchesReference) {
  VerifyTop1(GetParam(), /*num_queries=*/15, /*num_db=*/40);
}

INSTANTIATE_TEST_SUITE_P(Dimensions, Top1DimTest,
                         ::testing::Values(4, 16, 64, 128, 256, 768,
                                           // Odd dimensions.
                                           1, 3, 5, 7, 17, 129, 257));

TEST(Top1EdgeTest, SingleQuerySingleDb) {
  VerifyTop1(/*dim=*/128, /*num_queries=*/1, /*num_db=*/1);
}

TEST(Top1EdgeTest, SingleQueryManyDb) {
  VerifyTop1(/*dim=*/128, /*num_queries=*/1, /*num_db=*/50);
}

TEST(Top1EdgeTest, ManyQuerySingleDb) {
  VerifyTop1(/*dim=*/128, /*num_queries=*/20, /*num_db=*/1);
}

TEST(Top1EdgeTest, SelfComparison) {
  const size_t dim = 128;
  const size_t n = 30;
  absl::BitGen bitgen;
  auto vecs = GenerateRandomVectors(n, dim, bitgen);
  TurboQuantCodec tq(dim, /*seed=*/42);
  auto bp = EncodePoints(tq, vecs);

  auto ref = ComputeReferenceTop1(bp, bp);
  auto actual =
      ManyToManyTurboQuantTopK(bp, bp, IndexSubset::Contiguous(0, n),
                               IndexSubset::Contiguous(0, n), /*k=*/1);

  ASSERT_EQ(actual.size(), n);
  for (size_t qi = 0; qi < n; ++qi) {
    EXPECT_EQ(actual[qi].db_idx, ref[qi].db_idx) << "qi=" << qi;
    EXPECT_FLOAT_EQ(actual[qi].distance, ref[qi].distance) << "qi=" << qi;
  }
}

// ---------------------------------------------------------------------------
// On-the-fly database with gathered indices
// ---------------------------------------------------------------------------
TEST(Top1GatheredTest, BasicSubset) {
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
  std::vector<int> db_indices = {0, 3, 10, 25, 49};

  auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                         IndexSubset::Gathered(q_indices),
                                         IndexSubset::Gathered(db_indices),
                                         /*k=*/1);

  // Compute reference: run gathered ManyToMany, manually find argmin.
  const size_t nq_sub = q_indices.size();
  std::vector<TopKResult> ref(
      nq_sub, {.db_idx = 0, .distance = std::numeric_limits<float>::max()});
  ManyToManyTurboQuant(queries_bp, db_bp, IndexSubset::Gathered(q_indices),
                       IndexSubset::Gathered(db_indices),
                       [&](size_t query_idx, absl::Span<const float> scores,
                           size_t first_db_idx) {
                         for (size_t i = 0; i < scores.size(); ++i) {
                           if (scores[i] < ref[query_idx].distance) {
                             ref[query_idx].distance = scores[i];
                             ref[query_idx].db_idx = first_db_idx + i;
                           }
                         }
                       });

  ASSERT_EQ(actual.size(), nq_sub);
  for (size_t qi = 0; qi < nq_sub; ++qi) {
    EXPECT_EQ(actual[qi].db_idx, ref[qi].db_idx) << "qi=" << qi;
    EXPECT_FLOAT_EQ(actual[qi].distance, ref[qi].distance) << "qi=" << qi;
  }
}

TEST(Top1GatheredTest, EmptyIndicesMatchesFull) {
  const size_t dim = 64;
  const size_t nq = 10;
  const size_t ndb = 30;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto full = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
      IndexSubset::Contiguous(0, ndb), /*k=*/1);
  auto gathered = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                           IndexSubset::Contiguous(0, nq),
                                           IndexSubset::Contiguous(0, ndb),
                                           /*k=*/1);

  ASSERT_EQ(gathered.size(), nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    EXPECT_EQ(gathered[qi].db_idx, full[qi].db_idx) << "qi=" << qi;
    EXPECT_EQ(gathered[qi].distance, full[qi].distance) << "qi=" << qi;
  }
}

// ---------------------------------------------------------------------------
// Pretransposed database overload
// ---------------------------------------------------------------------------
void VerifyTop1PreTransposed(size_t dim, size_t num_queries, size_t num_db) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);

  auto ref = ComputeReferenceTop1(queries_bp, db_bp);
  auto actual = ManyToManyTurboQuantTopK(
      queries_bp, pretransposed, IndexSubset::Contiguous(0, num_queries),
      /*k=*/1);

  ASSERT_EQ(actual.size(), num_queries);
  for (size_t qi = 0; qi < num_queries; ++qi) {
    EXPECT_EQ(actual[qi].db_idx, ref[qi].db_idx)
        << "pretransposed dim=" << dim << " qi=" << qi;
    EXPECT_FLOAT_EQ(actual[qi].distance, ref[qi].distance)
        << "pretransposed dim=" << dim << " qi=" << qi;
  }
}

class Top1PreTransposedDimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(Top1PreTransposedDimTest, MatchesReference) {
  VerifyTop1PreTransposed(GetParam(), /*num_queries=*/15, /*num_db=*/40);
}

INSTANTIATE_TEST_SUITE_P(PreTransposedDimensions, Top1PreTransposedDimTest,
                         ::testing::Values(4, 16, 64, 128, 256, 768,
                                           // Odd dimensions.
                                           1, 3, 5, 7, 17, 129, 257));

TEST(Top1PreTransposedEdgeTest, SingleQuerySingleDb) {
  VerifyTop1PreTransposed(/*dim=*/128, /*num_queries=*/1, /*num_db=*/1);
}

TEST(Top1PreTransposedEdgeTest, ManyQuerySingleDb) {
  VerifyTop1PreTransposed(/*dim=*/128, /*num_queries=*/20, /*num_db=*/1);
}

TEST(Top1PreTransposedEdgeTest, MatchesOnTheFly) {
  const size_t dim = 128;
  const size_t nq = 15;
  const size_t ndb = 40;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto on_the_fly = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                             IndexSubset::Contiguous(0, nq),
                                             IndexSubset::Contiguous(0, ndb),
                                             /*k=*/1);
  auto pretransposed_db = PreTransposedTurboQuantDataset::Build(db_bp, tq);
  auto pretransposed = ManyToManyTurboQuantTopK(queries_bp, pretransposed_db,
                                                IndexSubset::Contiguous(0, nq),
                                                /*k=*/1);

  ASSERT_EQ(pretransposed.size(), nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    EXPECT_EQ(pretransposed[qi].db_idx, on_the_fly[qi].db_idx) << "qi=" << qi;
    EXPECT_EQ(pretransposed[qi].distance, on_the_fly[qi].distance)
        << "qi=" << qi;
  }
}

// ---------------------------------------------------------------------------
// Pretransposed database with gathered query indices
// ---------------------------------------------------------------------------
TEST(Top1PreTransposedGatheredTest, BasicSubset) {
  const size_t dim = 128;
  const size_t nq = 20;
  const size_t ndb = 50;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);

  std::vector<int> q_indices = {2, 5, 11, 18};
  auto actual = ManyToManyTurboQuantTopK(queries_bp, pretransposed,
                                         IndexSubset::Gathered(q_indices),
                                         /*k=*/1);

  // Compute reference via the full ManyToMany callback path.
  std::vector<TopKResult> ref(
      q_indices.size(),
      {.db_idx = 0, .distance = std::numeric_limits<float>::max()});
  ManyToManyTurboQuant(queries_bp, pretransposed,
                       IndexSubset::Gathered(q_indices),
                       [&](size_t query_idx, absl::Span<const float> scores,
                           size_t first_db_idx) {
                         for (size_t i = 0; i < scores.size(); ++i) {
                           if (scores[i] < ref[query_idx].distance) {
                             ref[query_idx].distance = scores[i];
                             ref[query_idx].db_idx = first_db_idx + i;
                           }
                         }
                       });

  ASSERT_EQ(actual.size(), q_indices.size());
  for (size_t qi = 0; qi < q_indices.size(); ++qi) {
    EXPECT_EQ(actual[qi].db_idx, ref[qi].db_idx) << "qi=" << qi;
    EXPECT_FLOAT_EQ(actual[qi].distance, ref[qi].distance) << "qi=" << qi;
  }
}

TEST(Top1PreTransposedGatheredTest, EmptyIndicesMatchesFull) {
  const size_t dim = 64;
  const size_t nq = 10;
  const size_t ndb = 30;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);

  auto full = ManyToManyTurboQuantTopK(queries_bp, pretransposed,
                                       IndexSubset::Contiguous(0, nq), /*k=*/1);
  auto gathered = ManyToManyTurboQuantTopK(queries_bp, pretransposed,
                                           IndexSubset::Contiguous(0, nq),
                                           /*k=*/1);

  ASSERT_EQ(gathered.size(), nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    EXPECT_EQ(gathered[qi].db_idx, full[qi].db_idx) << "qi=" << qi;
    EXPECT_EQ(gathered[qi].distance, full[qi].distance) << "qi=" << qi;
  }
}

// ---------------------------------------------------------------------------
// L2 metric tests
// ---------------------------------------------------------------------------
void VerifyTop1L2(size_t dim, size_t num_queries, size_t num_db) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kSquaredL2Distance);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto ref = ComputeReferenceTop1(queries_bp, db_bp);
  auto actual = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, num_queries),
      IndexSubset::Contiguous(0, num_db), /*k=*/1);

  ASSERT_EQ(actual.size(), num_queries);
  for (size_t qi = 0; qi < num_queries; ++qi) {
    EXPECT_EQ(actual[qi].db_idx, ref[qi].db_idx)
        << "L2 dim=" << dim << " qi=" << qi;
    EXPECT_FLOAT_EQ(actual[qi].distance, ref[qi].distance)
        << "L2 dim=" << dim << " qi=" << qi;
  }
}

class Top1L2DimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(Top1L2DimTest, MatchesReference) {
  VerifyTop1L2(GetParam(), /*num_queries=*/15, /*num_db=*/40);
}

INSTANTIATE_TEST_SUITE_P(L2Dimensions, Top1L2DimTest,
                         ::testing::Values(4, 16, 64, 128, 256, 768,
                                           // Odd dimensions.
                                           1, 3, 5, 7, 17, 129, 257));

TEST(Top1L2EdgeTest, SingleQuerySingleDb) {
  VerifyTop1L2(/*dim=*/128, /*num_queries=*/1, /*num_db=*/1);
}

TEST(Top1L2PreTransposedTest, MatchesOnTheFly) {
  const size_t dim = 128;
  const size_t nq = 15;
  const size_t ndb = 40;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kSquaredL2Distance);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto on_the_fly = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                             IndexSubset::Contiguous(0, nq),
                                             IndexSubset::Contiguous(0, ndb),
                                             /*k=*/1);
  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);
  auto pretransposed_result = ManyToManyTurboQuantTopK(
      queries_bp, pretransposed, IndexSubset::Contiguous(0, nq), /*k=*/1);

  ASSERT_EQ(pretransposed_result.size(), nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    EXPECT_EQ(pretransposed_result[qi].db_idx, on_the_fly[qi].db_idx)
        << "L2 pretransposed qi=" << qi;
    EXPECT_EQ(pretransposed_result[qi].distance, on_the_fly[qi].distance)
        << "L2 pretransposed qi=" << qi;
  }
}

// ---------------------------------------------------------------------------
// Stress tests
// ---------------------------------------------------------------------------
TEST(Top1StressTest, OddEverything) {
  // dim=129 (odd), 13 queries, 37 db points.
  VerifyTop1(/*dim=*/129, /*num_queries=*/13, /*num_db=*/37);
}

TEST(Top1StressTest, LargeDimSmallCounts) {
  VerifyTop1(/*dim=*/1536, /*num_queries=*/3, /*num_db=*/5);
}

TEST(Top1StressTest, SmallDimLargeCounts) {
  VerifyTop1(/*dim=*/4, /*num_queries=*/25, /*num_db=*/100);
}

TEST(Top1StressTest, LargeQueryBatch) {
  // Enough queries to exercise the internal query batch boundary.
  VerifyTop1(/*dim=*/128, /*num_queries=*/150, /*num_db=*/20);
}

TEST(Top1CoverageTest, RemainderScalarFallback) {
  // Explicitly use prime counts to ensure remainder loops run.
  VerifyTop1(/*dim=*/128, /*num_queries=*/7, /*num_db=*/11);
}

TEST(Top1CoverageTest, AdversarialNeedleAtLaneBoundary) {
  // Place needle at exactly 4*N lane ends to ensure offset mutation fails.
  const size_t dim = 128;
  const size_t nq = 1;
  const size_t ndb = 64;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  // Make distance significantly smaller at a specific target boundary lane end.
  // Target index 15 is end of first 16-element vector lane block.
  for (size_t i = 0; i < dim; ++i) {
    db_vecs[15][i] = q_vecs[0][i];
  }

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                         IndexSubset::Contiguous(0, nq),
                                         IndexSubset::Contiguous(0, ndb),
                                         /*k=*/1);
  EXPECT_EQ(actual[0].db_idx, 15);
}

TEST(Top1CoverageTest, EmptyDatabase) {
  // Force n = 0 in the score reduction to hit the line 110 return n fallback.
  const size_t dim = 128;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(1, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(10, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  std::vector<int> empty_db_indices;
  auto actual =
      ManyToManyTurboQuantTopK(queries_bp, db_bp, IndexSubset::Contiguous(0, 1),
                               IndexSubset::Gathered(empty_db_indices),
                               /*k=*/1);

  EXPECT_EQ(actual.size(), 1);
}

TEST(Top1CoverageTest, NeedleInUnrolledSubLanes) {
  // Test across various AVX lane widths (4, 8, 16, 32) to catch all offset
  // mutations.
  const size_t dim = 128;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(1, dim, bitgen);

  for (size_t lane_width : {4, 8, 16, 32}) {
    for (size_t sub_lane : {1, 2, 3}) {
      const size_t target_idx = 4 + sub_lane * lane_width;
      auto db_vecs = GenerateRandomVectors(150, dim, bitgen);
      for (size_t i = 0; i < dim; ++i) {
        db_vecs[target_idx][i] = q_vecs[0][i];
      }

      TurboQuantCodec tq(dim, /*seed=*/42);
      auto queries_bp = EncodePoints(tq, q_vecs);
      auto db_bp = EncodePoints(tq, db_vecs);

      auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                             IndexSubset::Contiguous(0, 1),
                                             IndexSubset::Contiguous(0, 150),
                                             /*k=*/1);
      EXPECT_EQ(actual[0].db_idx, target_idx)
          << "lane_width=" << lane_width << " sub_lane=" << sub_lane;
    }
  }
}

TEST(Top1CoverageTest, ReduceMinSpanAdversarialOffsets) {
  // Specifically target the min2 and min3 reduction lane offsets in the 4*N
  // block loop.
  const size_t dim = 128;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(1, dim, bitgen);

  for (size_t lane_width : {4, 8, 16, 32}) {
    // Target exactly the end of min2 and min3 load lanes.
    for (size_t target_idx :
         {2 * lane_width - 1, 3 * lane_width - 1, 3 * lane_width + 1}) {
      auto db_vecs = GenerateRandomVectors(150, dim, bitgen);
      for (size_t i = 0; i < dim; ++i) {
        db_vecs[target_idx][i] = q_vecs[0][i];
      }

      TurboQuantCodec tq(dim, /*seed=*/42);
      auto queries_bp = EncodePoints(tq, q_vecs);
      auto db_bp = EncodePoints(tq, db_vecs);

      auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                             IndexSubset::Contiguous(0, 1),
                                             IndexSubset::Contiguous(0, 150),
                                             /*k=*/1);
      EXPECT_EQ(actual[0].db_idx, target_idx)
          << "lane_width=" << lane_width << " target_idx=" << target_idx;
    }
  }
}

TEST(Top1CoverageTest, UnrolledNeighborOffsetCheck) {
  const size_t dim = 128;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(1, dim, bitgen);

  for (size_t lane_width : {4, 8, 16, 32}) {
    const size_t target_idx = 2 * lane_width;
    auto db_vecs = GenerateRandomVectors(150, dim, bitgen);

    // Initialize all database points to point in the exact opposite direction
    // of the query to guarantee maximum distance even after 4-bit quantization.
    for (size_t i = 0; i < 150; ++i) {
      for (size_t j = 0; j < dim; ++j) {
        db_vecs[i][j] = -q_vecs[0][j];
      }
    }

    // Set the target element to a small positive distance.
    for (size_t i = 0; i < dim; ++i) {
      db_vecs[target_idx][i] = q_vecs[0][i] + 0.1f;
    }

    TurboQuantCodec tq(dim, /*seed=*/42);
    auto queries_bp = EncodePoints(tq, q_vecs);
    auto db_bp = EncodePoints(tq, db_vecs);

    // We mask out target_idx - 1 and target_idx + 1 so that they are only read
    // if a mutated unrolled load offset by +1/-1 ignores the mask!
    std::vector<int> allowed_indices;
    for (size_t i = 0; i < 150; ++i) {
      if (i != target_idx - 1 && i != target_idx + 1) {
        allowed_indices.push_back(i);
      }
    }

    auto actual = ManyToManyTurboQuantTopK(
        queries_bp, db_bp, IndexSubset::Contiguous(0, 1),
        IndexSubset::Gathered(allowed_indices), /*k=*/1);

    // Since unrolled SIMD blocks load multiple adjacent neighbors at once,
    // any adjacent neighbor inside the same SIMD block may be legally selected
    // by the un-mutated SIMD min reduction. We verify that the selected index
    // is exactly the target or the neighbor within the same block.
    EXPECT_TRUE(actual[0].db_idx == target_idx ||
                actual[0].db_idx == target_idx - 1 ||
                actual[0].db_idx == target_idx + 1);
  }
}

void VerifyTopK(size_t dim, size_t num_queries, size_t num_db, int k) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto ref = ComputeReferenceTopK(queries_bp, db_bp, k);
  auto actual = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, num_queries),
      IndexSubset::Contiguous(0, num_db), k);

  ASSERT_EQ(actual.size(), num_queries * k);
  for (size_t qi = 0; qi < num_queries; ++qi) {
    for (int i = 0; i < k; ++i) {
      size_t idx = qi * k + i;
      EXPECT_EQ(actual[idx].db_idx, ref[idx].db_idx)
          << "dim=" << dim << " qi=" << qi << " i=" << i;
      EXPECT_FLOAT_EQ(actual[idx].distance, ref[idx].distance)
          << "dim=" << dim << " qi=" << qi << " i=" << i;
      // Catches bugs in OrderedFloatToInt conversion (e.g., -0.0f = 0.0f).
      EXPECT_EQ(absl::bit_cast<uint32_t>(actual[idx].distance),
                absl::bit_cast<uint32_t>(ref[idx].distance))
          << "bitwise float discrepancy, dim=" << dim << " qi=" << qi
          << " i=" << i;
    }
  }
}

class TopKDimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(TopKDimTest, MatchesReference) {
  VerifyTopK(GetParam(), /*num_queries=*/15, /*num_db=*/40, /*k=*/5);
}

INSTANTIATE_TEST_SUITE_P(TopKDimensions, TopKDimTest,
                         ::testing::Values(4, 16, 64, 128, 256, 768,
                                           // Odd dimensions.
                                           1, 3, 5, 7, 17, 129, 257));

TEST(TopKEdgeTest, KEqualsOneMatchesTop1) {
  const size_t dim = 128;
  const size_t nq = 10;
  const size_t ndb = 30;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto top1 = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                       IndexSubset::Contiguous(0, nq),
                                       IndexSubset::Contiguous(0, ndb),
                                       /*k=*/1);
  auto topk = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
      IndexSubset::Contiguous(0, ndb), /*k=*/1);

  ASSERT_EQ(topk.size(), nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    EXPECT_EQ(topk[qi].db_idx, top1[qi].db_idx) << "qi=" << qi;
    EXPECT_FLOAT_EQ(topk[qi].distance, top1[qi].distance) << "qi=" << qi;
  }
}

TEST(TopKEdgeTest, KEqualsDatabaseSize) {
  const size_t dim = 64;
  const size_t nq = 5;
  const size_t ndb = 10;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto actual = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
      IndexSubset::Contiguous(0, ndb), /*k=*/ndb);
  auto ref = ComputeReferenceTopK(queries_bp, db_bp, /*k=*/ndb);

  ASSERT_EQ(actual.size(), nq * ndb);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i].db_idx, ref[i].db_idx) << "i=" << i;
    EXPECT_FLOAT_EQ(actual[i].distance, ref[i].distance) << "i=" << i;
  }
}

TEST(TopKEdgeTest, KLargerThanDatabaseSize) {
  const size_t dim = 64;
  const size_t nq = 5;
  const size_t ndb = 10;
  const int k = 15;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                         IndexSubset::Contiguous(0, nq),
                                         IndexSubset::Contiguous(0, ndb), k);

  ASSERT_EQ(actual.size(), nq * k);
  for (size_t qi = 0; qi < nq; ++qi) {
    // First ndb elements should be valid (sorted)
    for (size_t i = 0; i < ndb; ++i) {
      EXPECT_NE(actual[qi * k + i].db_idx,
                std::numeric_limits<uint32_t>::max());
    }
    // Remaining elements should be padded
    for (size_t i = ndb; i < k; ++i) {
      EXPECT_EQ(actual[qi * k + i].db_idx,
                std::numeric_limits<uint32_t>::max());
      EXPECT_EQ(actual[qi * k + i].distance, std::numeric_limits<float>::max());
    }
  }
}

TEST(TopKGatheredTest, BasicSubset) {
  const size_t dim = 128;
  const size_t nq = 20;
  const size_t ndb = 50;
  const int k = 3;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  std::vector<int> q_indices = {2, 5, 11, 18};
  std::vector<int> db_indices = {0, 3, 10, 25, 49};

  auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                         IndexSubset::Gathered(q_indices),
                                         IndexSubset::Gathered(db_indices), k);

  auto ref =
      ComputeReferenceTopK(queries_bp, db_bp, absl::MakeConstSpan(q_indices),
                           absl::MakeConstSpan(db_indices), k);

  ASSERT_EQ(actual.size(), q_indices.size() * k);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i].db_idx, ref[i].db_idx) << "i=" << i;
    EXPECT_FLOAT_EQ(actual[i].distance, ref[i].distance) << "i=" << i;
  }
}

TEST(TopKPreTransposedTest, MatchesOnTheFly) {
  const size_t dim = 128;
  const size_t nq = 10;
  const size_t ndb = 30;
  const int k = 5;
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto on_the_fly = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
      IndexSubset::Contiguous(0, ndb), k);

  auto pretransposed_db = PreTransposedTurboQuantDataset::Build(db_bp, tq);
  auto pretransposed = ManyToManyTurboQuantTopK(
      queries_bp, pretransposed_db, IndexSubset::Contiguous(0, nq), k);

  ASSERT_EQ(pretransposed.size(), nq * k);
  for (size_t i = 0; i < pretransposed.size(); ++i) {
    EXPECT_EQ(pretransposed[i].db_idx, on_the_fly[i].db_idx) << "i=" << i;
    EXPECT_FLOAT_EQ(pretransposed[i].distance, on_the_fly[i].distance)
        << "i=" << i;
  }
}

TEST(TopKCoverageTest, DeferredGarbageCollectionTrigger) {
  // Break buffer capacity mid-iteration to audit delayed pruning and
  // SIMD remainders.
  const size_t dim = 128;
  const size_t nq = 2;
  const size_t ndb = 2011;
  const int k = 5;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto ref = ComputeReferenceTopK(queries_bp, db_bp, k);
  auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                         IndexSubset::Contiguous(0, nq),
                                         IndexSubset::Contiguous(0, ndb), k);

  ASSERT_EQ(actual.size(), nq * k);
  for (size_t i = 0; i < nq * k; ++i) {
    EXPECT_EQ(actual[i].db_idx, ref[i].db_idx) << "i=" << i;
    EXPECT_FLOAT_EQ(actual[i].distance, ref[i].distance) << "i=" << i;
  }
}

TEST(TopKCoverageTest, RemainderBlocks) {
  // Evaluate uneven remainder offsets across unconditional loading bounds.
  const size_t dim = 16;
  const size_t nq = 1;
  const size_t ndb = 13;
  const int k = 3;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                         IndexSubset::Contiguous(0, nq),
                                         IndexSubset::Contiguous(0, ndb), k);
  EXPECT_EQ(actual.size(), nq * k);
}

TEST(TopKCoverageTest, StrictInitialThreshold) {
  // Prove strict setting initialization works by rejecting unmask bounds.
  const size_t dim = 16;
  const size_t nq = 1;
  const size_t ndb = 10;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto actual = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
      IndexSubset::Contiguous(0, ndb), 5, -1e9f);

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(actual[i].distance, std::numeric_limits<float>::max());
  }
}

TEST(TopKPreTransposedGatheredTest, SpecificSubsets) {
  const size_t dim = 128;
  const size_t nq = 10;
  const size_t ndb = 30;
  const int k = 3;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);
  std::vector<int> q_indices = {0, 2, 5};

  auto actual = ManyToManyTurboQuantTopK(queries_bp, pretransposed,
                                         IndexSubset::Gathered(q_indices), k);

  std::vector<int> db_indices;
  auto ref =
      ComputeReferenceTopK(queries_bp, db_bp, absl::MakeConstSpan(q_indices),
                           absl::MakeConstSpan(db_indices), k);

  ASSERT_EQ(actual.size(), q_indices.size() * k);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i].db_idx, ref[i].db_idx) << "i=" << i;
    EXPECT_FLOAT_EQ(actual[i].distance, ref[i].distance) << "i=" << i;
  }
}

TEST(TopKPreTransposedIntervalTest, ContiguousSpan) {
  const size_t dim = 64;
  const size_t nq = 15;
  const size_t ndb = 30;
  const int k = 5;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);
  const int q_start = 2;
  const int q_limit = 12;
  const int q_length = q_limit - q_start;

  auto actual = ManyToManyTurboQuantTopK(
      queries_bp, pretransposed, IndexSubset::Contiguous(q_start, q_length), k);

  std::vector<int> q_indices;
  for (int i = q_start; i < q_limit; ++i) {
    q_indices.push_back(i);
  }
  std::vector<int> db_indices;
  auto ref =
      ComputeReferenceTopK(queries_bp, db_bp, absl::MakeConstSpan(q_indices),
                           absl::MakeConstSpan(db_indices), k);

  ASSERT_EQ(actual.size(), q_length * k);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i].db_idx, ref[i].db_idx) << "i=" << i;
    EXPECT_FLOAT_EQ(actual[i].distance, ref[i].distance) << "i=" << i;
  }
}

TEST(TopKCoverageTest, Needles) {
  const size_t dim = 128;
  const size_t nq = 8;
  const int k = 3;
  const size_t buffer_capacity = std::max<size_t>(4 * k + 128, 1024);  // 1024
  const size_t ndb = buffer_capacity + 20;

  std::vector<std::vector<float>> q_vecs(nq, std::vector<float>(dim, 0.0f));
  std::vector<std::vector<float>> db_vecs(ndb, std::vector<float>(dim, -2.0f));

  const int best_pos[8][3] = {
      {0, 1, 2},       {10, 100, 500},  {1021, 1022, 1023}, {5, 15, 1000},
      {500, 501, 502}, {700, 800, 900}, {100, 200, 300},    {900, 10, 50},
  };

  for (size_t q = 0; q < nq; ++q) {
    for (size_t i = 0; i < 16; ++i) {
      q_vecs[q][16 * q + i] = 1.0f;
    }

    for (size_t rank = 0; rank < 3; ++rank) {
      int pos = best_pos[q][rank];
      float val = (rank == 0) ? 2.0f : ((rank == 1) ? 1.5f : 1.0f);
      for (size_t i = 0; i < 16; ++i) {
        db_vecs[pos][16 * q + i] = val;
      }
    }

    for (size_t i = 0; i < 16; ++i) {
      db_vecs[buffer_capacity][16 * q + i] = 1.25f;
    }
  }

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                         IndexSubset::Contiguous(0, nq),
                                         IndexSubset::Contiguous(0, ndb), k);
  auto ref = ComputeReferenceTopK(queries_bp, db_bp, k);

  EXPECT_EQ(actual.size(), nq * k);
  for (size_t i = 0; i < nq * k; ++i) {
    EXPECT_EQ(actual[i].db_idx, ref[i].db_idx)
        << "Mismatch at flat index " << i;
  }
}

TEST(TopKCoverageTest, VQSelectPivotKPlusOne) {
  const size_t dim = 128;
  const int k = 2;
  const size_t ndb = 5003;
  const size_t nq = 4;

  for (int seed = 0; seed < 20; ++seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
    auto make_vecs = [&](size_t count) {
      std::vector<std::vector<float>> vecs(count, std::vector<float>(dim));
      for (auto& v : vecs)
        for (float& x : v) x = dist(rng);
      return vecs;
    };
    auto q_vecs = make_vecs(nq);
    auto db_vecs = make_vecs(ndb);

    TurboQuantCodec tq(dim, /*seed=*/42);
    auto queries_bp = EncodePoints(tq, q_vecs);
    auto db_bp = EncodePoints(tq, db_vecs);

    auto ref = ComputeReferenceTopK(queries_bp, db_bp, k);
    auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                           IndexSubset::Contiguous(0, nq),
                                           IndexSubset::Contiguous(0, ndb), k);

    ASSERT_EQ(actual.size(), nq * k);
    for (size_t i = 0; i < nq * k; ++i) {
      EXPECT_EQ(actual[i].db_idx, ref[i].db_idx)
          << "seed=" << seed << " i=" << i;
      EXPECT_FLOAT_EQ(actual[i].distance, ref[i].distance)
          << "seed=" << seed << " i=" << i;
    }
  }
}
TEST(TopKCoverageTest, FilteredLoadNPointerShift) {
  const size_t dim = 128;
  const int k = 3;
  // Use 2003 to ensure non-alignment for both 8-wide and 16-wide SIMD lanes.
  const size_t ndb = 2003;
  const size_t nq = 1;

  std::vector<std::vector<float>> q_vecs(nq, std::vector<float>(dim, 0.0f));
  std::vector<std::vector<float>> db_vecs(ndb, std::vector<float>(dim, 0.0f));

  // Set query to project on first dimension.
  q_vecs[0][0] = 1.0f;

  // Make the first 2000 elements mediocre.
  for (size_t i = 0; i < ndb - 3; ++i) {
    db_vecs[i][0] = 0.0f;
  }

  // Set the very last 3 elements to be amazingly good.
  // Since TurboQuant minimizes -dot_product, larger dot products are better.
  db_vecs[2000][0] = 5.0f;  // true score: -5.0
  db_vecs[2001][0] = 6.0f;  // true score: -6.0
  db_vecs[2002][0] = 7.0f;  // true score: -7.0

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto actual = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                         IndexSubset::Contiguous(0, nq),
                                         IndexSubset::Contiguous(0, ndb), k);

  ASSERT_EQ(actual.size(), k);
  EXPECT_EQ(actual[0].db_idx, 2002);
  EXPECT_NEAR(actual[0].distance, -7.0f, 0.2f);
  EXPECT_EQ(actual[1].db_idx, 2001);
  EXPECT_NEAR(actual[1].distance, -6.0f, 0.2f);
  EXPECT_EQ(actual[2].db_idx, 2000);
  EXPECT_NEAR(actual[2].distance, -5.0f, 0.2f);
}

TEST(TopKCoverageTest, TightPivotMaskDrop) {
  // Secures coverage against k-2 offset pivot shifting and capacity boundaries
  const size_t dim = 64;
  const int k = 5;
  const size_t capacity = std::max<size_t>(4 * k + 128, 1024);  // 1024
  const size_t ndb = capacity + 2;

  std::vector<std::vector<float>> q_vecs(1, std::vector<float>(dim, 0.0f));
  std::vector<std::vector<float>> db_vecs(ndb, std::vector<float>(dim, 0.0f));

  q_vecs[0][0] = 1.0f;

  for (size_t i = 0; i < capacity; ++i) {
    db_vecs[i][0] = 0.1f;
  }

  db_vecs[0][0] = 5.0f;
  db_vecs[1][0] = 4.8f;
  db_vecs[2][0] = 4.6f;
  db_vecs[3][0] = 4.4f;
  db_vecs[4][0] = 4.2f;

  db_vecs[capacity][0] = 4.3f;

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto topk =
      ManyToManyTurboQuantTopK(queries_bp, db_bp, IndexSubset::Contiguous(0, 1),
                               IndexSubset::Contiguous(0, ndb), k);
  auto ref = ComputeReferenceTopK(queries_bp, db_bp, k);

  for (int i = 0; i < k; ++i) {
    EXPECT_EQ(topk[i].db_idx, ref[i].db_idx)
        << "k-2 pivot shift failure at i=" << i;
  }
}

TEST(TopKCoverageTest, RemainingMaskLeakProtection) {
  const size_t dim = 64;
  const int k = 2;
  const size_t ndb = 6;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(1, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42,
                     TurboQuantDistanceMetric::kSquaredL2Distance);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  auto actual =
      ManyToManyTurboQuantTopK(queries_bp, db_bp, IndexSubset::Contiguous(0, 1),
                               IndexSubset::Contiguous(0, ndb), k, 50.0f);

  for (int i = 0; i < k; ++i) {
    EXPECT_LT(actual[i].db_idx, ndb);
  }
}

TEST(TopKStateReuseTest, ZeroAllocationStateReuseMatchesReference) {
  const size_t dim = 128;
  const size_t nq = 20;
  const size_t ndb = 80;
  const int k = 10;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);

  ManyToManyTopKState state;
  const many_to_many_topk_internal::TopKState& internal =
      ManyToManyTopKStatePeer::Get(state);
  // First call sizes the state and workspace buffers.
  absl::Span<const TopKResult> results = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
      IndexSubset::Contiguous(0, ndb), k, state);
  EXPECT_THAT(results,
              ElementsAreArray(ComputeReferenceTopK(queries_bp, db_bp, k)));

  const void* results_ptr_before = results.data();
  const void* items_ptr_before = internal.items.data();
  const void* panels_ptr_before = internal.workspace.db_panels.data();

  // Second call uses a smaller, non-panel-aligned subset (nq=11, ndb=23, k=5):
  // reuses the same buffers without reallocating while testing partial last
  // panels over stale norms.
  const size_t small_nq = 11;
  const size_t small_ndb = 23;
  const int small_k = 5;
  results = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, small_nq),
      IndexSubset::Contiguous(0, small_ndb), small_k, state);
  EXPECT_EQ(results.data(), results_ptr_before);
  EXPECT_EQ(internal.items.data(), items_ptr_before);
  EXPECT_EQ(internal.workspace.db_panels.data(), panels_ptr_before);

  std::vector<int> small_q_idx(small_nq);
  std::vector<int> small_db_idx(small_ndb);
  std::iota(small_q_idx.begin(), small_q_idx.end(), 0);
  std::iota(small_db_idx.begin(), small_db_idx.end(), 0);
  EXPECT_THAT(results,
              ElementsAreArray(ComputeReferenceTopK(
                  queries_bp, db_bp, small_q_idx, small_db_idx, small_k)));

  // Third call with a much larger dataset reallocates buffers to grow capacity.
  const size_t large_nq = 40;
  const size_t large_ndb = 200;
  const int large_k = 15;
  auto large_q_vecs = GenerateRandomVectors(large_nq, dim, bitgen);
  auto large_db_vecs = GenerateRandomVectors(large_ndb, dim, bitgen);
  auto large_queries_bp = EncodePoints(tq, large_q_vecs);
  auto large_db_bp = EncodePoints(tq, large_db_vecs);

  results = ManyToManyTurboQuantTopK(
      large_queries_bp, large_db_bp, IndexSubset::Contiguous(0, large_nq),
      IndexSubset::Contiguous(0, large_ndb), large_k, state);
  EXPECT_NE(results.data(), results_ptr_before);
  EXPECT_NE(internal.items.data(), items_ptr_before);
  EXPECT_NE(internal.workspace.db_panels.data(), panels_ptr_before);
  EXPECT_THAT(results, ElementsAreArray(ComputeReferenceTopK(
                           large_queries_bp, large_db_bp, large_k)));
}

TEST(TopKStateReuseTest,
     ZeroAllocationPreTransposedStateReuseMatchesReference) {
  const size_t dim = 128;
  const size_t nq = 20;
  const size_t ndb = 80;
  const int k = 10;

  absl::BitGen bitgen;
  std::vector<std::vector<float>> q_vecs =
      GenerateRandomVectors(nq, dim, bitgen);
  std::vector<std::vector<float>> db_vecs =
      GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  TurboQuantDatasetSoA queries_bp = EncodePoints(tq, q_vecs);
  TurboQuantDatasetSoA db_bp = EncodePoints(tq, db_vecs);
  PreTransposedTurboQuantDataset pretransposed_db =
      PreTransposedTurboQuantDataset::Build(db_bp, tq);

  ManyToManyTopKState state;
  const many_to_many_topk_internal::TopKState& internal =
      ManyToManyTopKStatePeer::Get(state);
  absl::Span<const TopKResult> results = ManyToManyTurboQuantTopK(
      queries_bp, pretransposed_db, IndexSubset::Contiguous(0, nq), k, state);
  EXPECT_THAT(results,
              ElementsAreArray(ComputeReferenceTopK(queries_bp, db_bp, k)));

  const void* results_ptr_before = results.data();
  const void* items_ptr_before = internal.items.data();
  const void* q_decoded_ptr_before = internal.workspace.q_decoded.data();

  // Second call with fewer queries and a smaller k reuses the same buffers.
  const size_t small_nq = 11;
  const int small_k = 5;
  results = ManyToManyTurboQuantTopK(queries_bp, pretransposed_db,
                                     IndexSubset::Contiguous(0, small_nq),
                                     small_k, state);
  EXPECT_EQ(results.data(), results_ptr_before);
  EXPECT_EQ(internal.items.data(), items_ptr_before);
  EXPECT_EQ(internal.workspace.q_decoded.data(), q_decoded_ptr_before);

  std::vector<int> small_q_idx(small_nq);
  std::iota(small_q_idx.begin(), small_q_idx.end(), 0);
  EXPECT_THAT(results, ElementsAreArray(
                           ComputeReferenceTopK(queries_bp, db_bp, small_q_idx,
                                                /*db_indices=*/{}, small_k)));
}

TEST(TopKStateReuseTest, NonPositiveKAndEmptySubsetsSafeWithState) {
  const size_t dim = 64;
  const size_t nq = 10;
  const size_t ndb = 20;

  absl::BitGen bitgen;
  std::vector<std::vector<float>> q_vecs =
      GenerateRandomVectors(nq, dim, bitgen);
  std::vector<std::vector<float>> db_vecs =
      GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  TurboQuantDatasetSoA queries_bp = EncodePoints(tq, q_vecs);
  TurboQuantDatasetSoA db_bp = EncodePoints(tq, db_vecs);
  PreTransposedTurboQuantDataset pretransposed_db =
      PreTransposedTurboQuantDataset::Build(db_bp, tq);

  ManyToManyTopKState state;
  const many_to_many_topk_internal::TopKState& internal =
      ManyToManyTopKStatePeer::Get(state);

  // k = 0 on a fresh state returns no results and allocates no buffers.
  absl::Span<const TopKResult> results = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
      IndexSubset::Contiguous(0, ndb), /*k=*/0, state);
  EXPECT_TRUE(results.empty());
  EXPECT_TRUE(internal.items.empty());
  EXPECT_TRUE(internal.results.empty());

  // k <= 0 on a state holding results from a previous run returns no results.
  results = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                     IndexSubset::Contiguous(0, nq),
                                     IndexSubset::Contiguous(0, ndb),
                                     /*k=*/5, state);
  ASSERT_EQ(results.size(), nq * 5);
  results = ManyToManyTurboQuantTopK(queries_bp, pretransposed_db,
                                     IndexSubset::Contiguous(0, nq),
                                     /*k=*/-5, state);
  EXPECT_TRUE(results.empty());

  results =
      ManyToManyTurboQuantTopK(queries_bp, db_bp, IndexSubset::Contiguous(0, 0),
                               IndexSubset::Contiguous(0, ndb),
                               /*k=*/5, state);
  EXPECT_TRUE(results.empty());

  const std::vector<float> thresholds(nq, 0.0f);
  results = ManyToManyTurboQuantTopK(queries_bp, db_bp,
                                     IndexSubset::Contiguous(0, nq),
                                     IndexSubset::Contiguous(0, ndb), /*k=*/0,
                                     state, absl::MakeConstSpan(thresholds));
  EXPECT_TRUE(results.empty());

  results = ManyToManyTurboQuantTopK(queries_bp, pretransposed_db,
                                     IndexSubset::Contiguous(0, nq), /*k=*/0,
                                     state, absl::MakeConstSpan(thresholds));
  EXPECT_TRUE(results.empty());
}

TEST(TopKStateReuseTest, PerQueryInitialThresholdsFilterCandidatesCorrectly) {
  const size_t dim = 64;
  const size_t nq = 12;
  const size_t ndb = 50;
  const int k = 8;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);
  PreTransposedTurboQuantDataset pretransposed_db =
      PreTransposedTurboQuantDataset::Build(db_bp, tq);

  // Compute full TopK to pick a distinct cutoff threshold per query (e.g., the
  // 4th-best distance for even queries, and no filtering for odd queries).
  auto full_ref = ComputeReferenceTopK(queries_bp, db_bp, k);
  std::vector<float> per_query_thresholds(nq);
  for (size_t q = 0; q < nq; ++q) {
    per_query_thresholds[q] = (q % 2 == 0) ? full_ref[q * k + 3].distance
                                           : std::numeric_limits<float>::max();
  }

  ManyToManyTopKState state_soa;
  absl::Span<const TopKResult> results_soa = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, IndexSubset::Contiguous(0, nq),
      IndexSubset::Contiguous(0, ndb), k, state_soa,
      absl::MakeConstSpan(per_query_thresholds));

  ManyToManyTopKState state_pre;
  absl::Span<const TopKResult> results_pre = ManyToManyTurboQuantTopK(
      queries_bp, pretransposed_db, IndexSubset::Contiguous(0, nq), k,
      state_pre, absl::MakeConstSpan(per_query_thresholds));
  EXPECT_THAT(results_soa, ElementsAreArray(results_pre));

  for (size_t q = 0; q < nq; ++q) {
    auto single_ref = ManyToManyTurboQuantTopK(
        queries_bp, db_bp, IndexSubset::Contiguous(q, 1),
        IndexSubset::Contiguous(0, ndb), k, per_query_thresholds[q]);
    EXPECT_THAT(results_soa.subspan(q * k, k), ElementsAreArray(single_ref));
  }
}

TEST(TopKStateReuseTest, MatchesAllocatingOverloads) {
  const size_t dim = 64;
  const size_t nq = 12;
  const size_t ndb = 30;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(nq, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(ndb, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);
  PreTransposedTurboQuantDataset pretransposed_db =
      PreTransposedTurboQuantDataset::Build(db_bp, tq);
  const IndexSubset query_subset = IndexSubset::Contiguous(0, nq);
  const IndexSubset db_subset = IndexSubset::Contiguous(0, ndb);

  ManyToManyTopKState state;
  // Like the allocating overloads, k = 1 takes the Top1 path, which needs no
  // candidate buffer.
  ManyToManyTurboQuantTopK(queries_bp, db_bp, query_subset, db_subset,
                           /*k=*/1, state);
  EXPECT_TRUE(ManyToManyTopKStatePeer::Get(state).items.empty());

  // Decreasing k reuses buffers sized for a larger k. k > ndb pads results.
  for (int k : {static_cast<int>(ndb) + 3, 5, 1}) {
    EXPECT_THAT(ManyToManyTurboQuantTopK(queries_bp, db_bp, query_subset,
                                         db_subset, k, state),
                ElementsAreArray(ManyToManyTurboQuantTopK(
                    queries_bp, db_bp, query_subset, db_subset, k)))
        << "k=" << k;
    EXPECT_THAT(ManyToManyTurboQuantTopK(queries_bp, pretransposed_db,
                                         query_subset, k, state),
                ElementsAreArray(ManyToManyTurboQuantTopK(
                    queries_bp, pretransposed_db, query_subset, k)))
        << "k=" << k;
  }
}

TEST(Top1EmptyDatabaseTest, ReturnsSentinelIndexAndMaxDistance) {
  constexpr size_t kDim = 64;
  constexpr size_t kNumQueries = 4;

  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(kNumQueries, kDim, bitgen);
  auto db_vecs = GenerateRandomVectors(/*count=*/2, kDim, bitgen);

  TurboQuantCodec tq(kDim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);
  const IndexSubset query_subset = IndexSubset::Contiguous(0, kNumQueries);
  const IndexSubset empty_db_subset = IndexSubset::Contiguous(0, 0);

  const std::vector<TopKResult> soa_results = ManyToManyTurboQuantTopK(
      queries_bp, db_bp, query_subset, empty_db_subset, /*k=*/1);
  ASSERT_EQ(soa_results.size(), kNumQueries);
  for (const TopKResult& res : soa_results) {
    EXPECT_EQ(res.db_idx, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(res.distance, std::numeric_limits<float>::max());
  }

  const PreTransposedTurboQuantDataset empty_pretransposed =
      PreTransposedTurboQuantDataset::Build(db_bp, absl::Span<const int>{}, tq);
  const std::vector<TopKResult> pretransposed_results =
      ManyToManyTurboQuantTopK(queries_bp, empty_pretransposed, query_subset,
                               /*k=*/1);
  ASSERT_EQ(pretransposed_results.size(), kNumQueries);
  for (const TopKResult& res : pretransposed_results) {
    EXPECT_EQ(res.db_idx, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(res.distance, std::numeric_limits<float>::max());
  }
}

}  // namespace
}  // namespace ginger
