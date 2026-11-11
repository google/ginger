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

#include "ginger/turboquant/pipnn/assign_leaders.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ginger/internal/gunit_support.h"
#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/many_to_many_turboquant.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "ginger/turboquant/pipnn/topk_result.h"
#include "ginger/turboquant/types/distance_metric.h"

namespace ginger {
namespace {

// Test utilities

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

// Compute reference scores [num_queries][num_db] by materializing the full
// score matrix via ManyToManyTurboQuant callback.
template <size_t K>
std::vector<std::vector<float>> ComputeReferenceTopK(
    const TurboQuantDatasetSoA& queries_bp,
    const PreTransposedTurboQuantDataset& db, size_t num_queries,
    absl::Span<const int> query_indices = {}) {
  std::vector<std::vector<float>> per_query(num_queries,
                                            std::vector<float>(db.num_points));

  if (query_indices.empty()) {
    ManyToManyTurboQuant(
        queries_bp, db, IndexSubset::Contiguous(0, num_queries),
        [&](size_t qi, absl::Span<const float> scores, size_t first_db) {
          for (size_t i = 0; i < scores.size(); ++i) {
            per_query[qi][first_db + i] = scores[i];
          }
        });
  } else {
    ManyToManyTurboQuant(
        queries_bp, db, IndexSubset::Gathered(query_indices),
        [&](size_t qi, absl::Span<const float> scores, size_t first_db) {
          for (size_t i = 0; i < scores.size(); ++i) {
            per_query[qi][first_db + i] = scores[i];
          }
        });
  }
  return per_query;
}

// Verify that the actual top-K results match the reference scores.
// Tie-breaking across equal scores differs between in-register lane reduction
// (K=1, 3) and scalar sorted insertion (K=10), so we verify sorted ascending
// scores, distinct valid indices, exact top-K score profile, and that every
// returned index achieves its reported score in the reference matrix.
template <size_t K>
void VerifyFixedTopKResults(const std::vector<FixedTopKResult<K>>& actual,
                            const std::vector<std::vector<float>>& ref,
                            absl::string_view label = "") {
  ASSERT_EQ(actual.size(), ref.size()) << label;
  for (size_t qi = 0; qi < actual.size(); ++qi) {
    const std::vector<float>& true_scores = ref[qi];
    const size_t expected_k = std::min(K, true_scores.size());

    std::vector<std::pair<float, int32_t>> a_entries;
    for (size_t k = 0; k < K; ++k) {
      auto actual_entry = actual[qi].Get(k);
      if (actual_entry.index >= 0) {
        a_entries.push_back({actual_entry.score, actual_entry.index});
      }
    }
    auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
    EXPECT_TRUE(std::is_sorted(a_entries.begin(), a_entries.end(), cmp))
        << label << " qi=" << qi << " actual results not sorted ascending";

    ASSERT_EQ(a_entries.size(), expected_k) << label << " qi=" << qi;

    std::vector<float> sorted_ref = true_scores;
    absl::c_partial_sort(sorted_ref, sorted_ref.begin() + expected_k);

    for (size_t k = 0; k < expected_k; ++k) {
      const int32_t idx = a_entries[k].second;
      ASSERT_LT(idx, static_cast<int32_t>(true_scores.size()))
          << label << " qi=" << qi << " k=" << k;
      for (size_t prev = 0; prev < k; ++prev) {
        EXPECT_NE(idx, a_entries[prev].second)
            << label << " qi=" << qi << " k=" << k;
      }
      EXPECT_FLOAT_EQ(a_entries[k].first, sorted_ref[k])
          << label << " qi=" << qi << " k=" << k;
      EXPECT_FLOAT_EQ(true_scores[idx], a_entries[k].first)
          << label << " qi=" << qi << " k=" << k << " idx=" << idx;
    }
  }
}

// Deduplicated test helper for verifying Top-K leader assignment.
template <size_t K, typename AssignFunc>
void VerifyTopK(size_t dim, size_t num_queries, size_t num_db,
                AssignFunc assign_fn, const std::string& label) {
  absl::BitGen bitgen;
  auto q_vecs = GenerateRandomVectors(num_queries, dim, bitgen);
  auto db_vecs = GenerateRandomVectors(num_db, dim, bitgen);

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);
  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);

  std::vector<FixedTopKResult<K>> actual = assign_fn(queries_bp, pretransposed);
  auto ref = ComputeReferenceTopK<K>(queries_bp, pretransposed, num_queries);
  VerifyFixedTopKResults<K>(actual, ref, label + " dim=" + std::to_string(dim));
}

// Top-1 fused tests

void VerifyTopK1Gathered(size_t dim, size_t num_queries, size_t num_db) {
  std::vector<int> q_indices(num_queries);
  std::iota(q_indices.begin(), q_indices.end(), 0);
  VerifyTopK<1>(
      dim, num_queries, num_db,
      [&](const TurboQuantDatasetSoA& queries_bp,
          const PreTransposedTurboQuantDataset& pretransposed) {
        return AssignToLeadersTopK1(
            queries_bp, IndexSubset::Gathered(q_indices), pretransposed);
      },
      "TopK1Gathered");
}

class TopK1DimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(TopK1DimTest, MatchesReference) {
  VerifyTopK1Gathered(GetParam(), /*num_queries=*/15, /*num_db=*/40);
}

INSTANTIATE_TEST_SUITE_P(Dimensions, TopK1DimTest,
                         ::testing::Values(4, 16, 64, 128, 256, 768, 1, 3, 5, 7,
                                           17, 129, 257));

TEST(TopK1EdgeTest, SingleQueryManyDb) {
  VerifyTopK1Gathered(/*dim=*/128, /*num_queries=*/1, /*num_db=*/50);
}

TEST(TopK1EdgeTest, ManyQueries) {
  VerifyTopK1Gathered(/*dim=*/128, /*num_queries=*/25, /*num_db=*/40);
}

// Top-3 gathered tests

void VerifyTopK3Gathered(size_t dim, size_t num_queries, size_t num_db) {
  std::vector<int> q_indices(num_queries);
  std::iota(q_indices.begin(), q_indices.end(), 0);
  VerifyTopK<3>(
      dim, num_queries, num_db,
      [&](const TurboQuantDatasetSoA& queries_bp,
          const PreTransposedTurboQuantDataset& pretransposed) {
        return AssignToLeadersTopK3(
            queries_bp, IndexSubset::Gathered(q_indices), pretransposed);
      },
      "TopK3Gathered");
}

class TopK3DimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(TopK3DimTest, MatchesReference) {
  VerifyTopK3Gathered(GetParam(), /*num_queries=*/15, /*num_db=*/40);
}

INSTANTIATE_TEST_SUITE_P(Dimensions, TopK3DimTest,
                         ::testing::Values(4, 16, 64, 128, 256, 768, 1, 3, 5, 7,
                                           17, 129, 257));

TEST(TopK3EdgeTest, SingleQueryManyDb) {
  VerifyTopK3Gathered(/*dim=*/128, /*num_queries=*/1, /*num_db=*/50);
}

TEST(TopK3EdgeTest, FewDbPoints) {
  // K=3 with only 3 db points: all must appear.
  VerifyTopK3Gathered(/*dim=*/128, /*num_queries=*/5, /*num_db=*/3);
}

TEST(TopK3EdgeTest, ManyQueries) {
  // Enough queries to exercise the kQueryUnroll=6 batching boundary.
  VerifyTopK3Gathered(/*dim=*/128, /*num_queries=*/25, /*num_db=*/40);
}

TEST(TopK3GatheredSubsetTest, BasicSubset) {
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
  std::vector<FixedTopKResult<3>> actual = AssignToLeadersTopK3(
      queries_bp, IndexSubset::Gathered(q_indices), pretransposed);

  auto ref =
      ComputeReferenceTopK<3>(queries_bp, pretransposed, q_indices.size(),
                              absl::MakeConstSpan(q_indices));

  VerifyFixedTopKResults<3>(actual, ref, "TopK3GatheredSubset");
}

// Top-10 consecutive tests

void VerifyTopK10Consecutive(size_t dim, size_t num_queries, size_t num_db) {
  VerifyTopK<10>(
      dim, num_queries, num_db,
      [&](const TurboQuantDatasetSoA& queries_bp,
          const PreTransposedTurboQuantDataset& pretransposed) {
        return AssignToLeadersTopK10(
            queries_bp, IndexSubset::Contiguous(0, num_queries), pretransposed);
      },
      "TopK10Consec");
}

class TopK10ConsecutiveDimTest : public ::testing::TestWithParam<size_t> {};

TEST_P(TopK10ConsecutiveDimTest, MatchesReference) {
  VerifyTopK10Consecutive(GetParam(), /*num_queries=*/15, /*num_db=*/40);
}

INSTANTIATE_TEST_SUITE_P(Dimensions, TopK10ConsecutiveDimTest,
                         ::testing::Values(4, 16, 64, 128, 256, 768, 1, 3, 5, 7,
                                           17, 129, 257));

TEST(TopK10ConsecutiveEdgeTest, SingleQuery) {
  VerifyTopK10Consecutive(/*dim=*/128, /*num_queries=*/1, /*num_db=*/50);
}

// L2 metric tests

TEST(L2MetricTest, TopK3Gathered) {
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
  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);

  std::vector<int> q_indices(nq);
  std::iota(q_indices.begin(), q_indices.end(), 0);

  std::vector<FixedTopKResult<3>> actual = AssignToLeadersTopK3(
      queries_bp, IndexSubset::Gathered(q_indices), pretransposed);

  auto ref = ComputeReferenceTopK<3>(queries_bp, pretransposed, nq);
  VerifyFixedTopKResults<3>(actual, ref, "L2 TopK3");
}

TEST(L2MetricTest, TopK10Consecutive) {
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
  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);

  std::vector<FixedTopKResult<10>> actual = AssignToLeadersTopK10(
      queries_bp, IndexSubset::Contiguous(0, nq), pretransposed);

  auto ref = ComputeReferenceTopK<10>(queries_bp, pretransposed, nq);
  VerifyFixedTopKResults<10>(actual, ref, "L2 TopK10");
}

TEST(ScoreTiesTest, DuplicateLeadersInteriorAndBoundary) {
  const size_t dim = 128;
  const size_t nq = 7;
  const size_t ndb = 40;
  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<std::vector<float>> q_vecs(nq, std::vector<float>(dim));
  std::vector<std::vector<float>> db_vecs(ndb, std::vector<float>(dim));
  for (auto& v : q_vecs) {
    for (float& x : v) x = dist(rng);
  }
  for (auto& v : db_vecs) {
    for (float& x : v) x = dist(rng);
  }

  // Place 5 identical copies across different SIMD lanes/panels (straddling
  // K=1 and K=3 boundaries, and strictly inside K=10), plus 8 copies of a
  // second vector so K=10 also has a boundary tie.
  for (size_t idx :
       {size_t{0}, size_t{5}, size_t{16}, size_t{22}, size_t{35}}) {
    db_vecs[idx] = q_vecs[0];
  }
  for (size_t idx : {size_t{1}, size_t{7}, size_t{11}, size_t{17}, size_t{23},
                     size_t{29}, size_t{31}, size_t{38}}) {
    db_vecs[idx] = db_vecs[2];
  }

  TurboQuantCodec tq(dim, /*seed=*/42);
  auto queries_bp = EncodePoints(tq, q_vecs);
  auto db_bp = EncodePoints(tq, db_vecs);
  auto pretransposed = PreTransposedTurboQuantDataset::Build(db_bp, tq);

  const IndexSubset subset = IndexSubset::Contiguous(0, nq);
  const auto ref = ComputeReferenceTopK<10>(queries_bp, pretransposed, nq);
  VerifyFixedTopKResults<1>(
      AssignToLeadersTopK1(queries_bp, subset, pretransposed), ref,
      "Ties TopK1");
  VerifyFixedTopKResults<3>(
      AssignToLeadersTopK3(queries_bp, subset, pretransposed), ref,
      "Ties TopK3");
  VerifyFixedTopKResults<10>(
      AssignToLeadersTopK10(queries_bp, subset, pretransposed), ref,
      "Ties TopK10");
}

// Stress tests

TEST(StressTest, OddEverything) {
  VerifyTopK1Gathered(/*dim=*/129, /*num_queries=*/13, /*num_db=*/37);
  VerifyTopK3Gathered(/*dim=*/129, /*num_queries=*/13, /*num_db=*/37);
  VerifyTopK10Consecutive(/*dim=*/129, /*num_queries=*/13, /*num_db=*/37);
}

TEST(StressTest, LargeDimSmallCounts) {
  VerifyTopK1Gathered(/*dim=*/1536, /*num_queries=*/3, /*num_db=*/5);
  VerifyTopK3Gathered(/*dim=*/1536, /*num_queries=*/3, /*num_db=*/5);
  VerifyTopK10Consecutive(/*dim=*/1536, /*num_queries=*/3, /*num_db=*/5);
}

TEST(StressTest, SmallDimLargeCounts) {
  VerifyTopK1Gathered(/*dim=*/4, /*num_queries=*/25, /*num_db=*/100);
  VerifyTopK3Gathered(/*dim=*/4, /*num_queries=*/25, /*num_db=*/100);
  VerifyTopK10Consecutive(/*dim=*/4, /*num_queries=*/25, /*num_db=*/100);
}

TEST(StressTest, LargeQueryBatch) {
  VerifyTopK10Consecutive(/*dim=*/128, /*num_queries=*/150, /*num_db=*/20);
}

}  // namespace
}  // namespace ginger
