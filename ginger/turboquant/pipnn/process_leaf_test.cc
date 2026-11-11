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

#include "ginger/turboquant/pipnn/process_leaf.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ginger/internal/gunit_support.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/types/distance_metric.h"

namespace ginger {
namespace {

std::vector<float> MakeRandomData(size_t num_points, size_t dim,
                                  std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<float> data(num_points * dim);
  for (auto& v : data) v = dist(rng);
  return data;
}

std::pair<TurboQuantCodec, TurboQuantDatasetSoA> MakeQuantizedPoints(
    size_t dim, size_t num_points, const std::vector<float>& data,
    TurboQuantDistanceMetric metric =
        TurboQuantDistanceMetric::kDotProductDistance) {
  std::vector<absl::Span<const float>> datapoints;
  for (size_t i = 0; i < num_points; ++i) {
    datapoints.push_back(absl::MakeConstSpan(data.data() + i * dim, dim));
  }
  TurboQuantCodec tq(dim, 42, metric);
  auto bp_or = TurboQuantDatasetSoA::Create(
      tq, datapoints,
      TurboQuantDatasetOptions{
          .pad_to_cache_line_size = false,
          .quantization_bits = 4,
          .store_query_byte_sums = false,
          .normalized_datapoints = false,
          .scaling_storage_mode = ScalingStorageMode::kSeparateArray});
  ABSL_CHECK_OK(bp_or);
  return {std::move(tq), *std::move(bp_or)};
}

class ProcessLeafTop2Test
    : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

// Shuffling the indices of the leaf must produce the same nearest neighbors
// (mapped back to global IDs).
TEST_P(ProcessLeafTop2Test, ShuffleInvariance) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  // Original order.
  ProcessLeafWorkspace workspace1;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace1);

  // Shuffled order.
  std::vector<int> indices2 = indices;
  std::shuffle(indices2.begin(), indices2.end(), rng);

  ProcessLeafWorkspace workspace2;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices2), workspace2);

  // Build mapping: for each global index, find its local position in run 2.
  std::vector<int> global_to_local2(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    global_to_local2[indices2[i]] = i;
  }

  // Compare: for each point (by global index), the top-1 neighbor should be
  // the same global point in both runs.
  for (size_t i = 0; i < num_points; ++i) {
    const int global_i = indices[i];
    const auto& r1 = workspace1.top2_results[i];
    const int local2_i = global_to_local2[global_i];
    const auto& r2 = workspace2.top2_results[local2_i];

    if (r1.best_idx < 0 || r1.best_idx >= static_cast<int32_t>(num_points))
      continue;
    if (r2.best_idx < 0 || r2.best_idx >= static_cast<int32_t>(num_points))
      continue;

    const int global_nn1 = indices[r1.best_idx];
    const int global_nn2 = indices2[r2.best_idx];

    EXPECT_EQ(global_nn1, global_nn2)
        << "point=" << global_i << " dim=" << dim << " n=" << num_points
        << " score1_run1=" << r1.best_score << " score1_run2=" << r2.best_score;
  }
}

// Top-2 scores are ordered (score1 <= score2).
TEST_P(ProcessLeafTop2Test, ScoresOrdered) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_LE(r.best_score, r.second_best_score)
        << "point=" << i << " dim=" << dim << " n=" << num_points;
    EXPECT_GE(r.best_idx, 0) << "point=" << i;
    EXPECT_LT(r.best_idx, static_cast<int32_t>(num_points))
        << "point=" << i << " dim=" << dim;
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i))
        << "point=" << i << " nearest neighbor should not be self";
  }
}

// All scores are finite.
TEST_P(ProcessLeafTop2Test, ScoresFinite) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(77);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_TRUE(std::isfinite(r.best_score))
        << "point=" << i << " score1=" << r.best_score;
    EXPECT_TRUE(std::isfinite(r.second_best_score))
        << "point=" << i << " score2=" << r.second_best_score;
  }
}

// Self-exclusion — no point selects itself as a neighbor.
TEST_P(ProcessLeafTop2Test, SelfExclusion) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i))
        << "point=" << i << " selected itself as top-1";
    if (r.second_best_idx >= 0) {
      EXPECT_NE(r.second_best_idx, static_cast<int32_t>(i))
          << "point=" << i << " selected itself as top-2";
    }
  }
}

// Workspace reuse — calling twice yields same results.
TEST_P(ProcessLeafTop2Test, WorkspaceReuse) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;

  // First call.
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);
  auto results1 = workspace.top2_results;

  // Second call (reuse workspace).
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  ASSERT_EQ(results1.size(), workspace.top2_results.size());
  for (size_t i = 0; i < num_points; ++i) {
    EXPECT_EQ(results1[i].best_idx, workspace.top2_results[i].best_idx)
        << "point=" << i;
    EXPECT_EQ(results1[i].second_best_idx,
              workspace.top2_results[i].second_best_idx)
        << "point=" << i;
    EXPECT_FLOAT_EQ(results1[i].best_score,
                    workspace.top2_results[i].best_score)
        << "point=" << i;
    EXPECT_FLOAT_EQ(results1[i].second_best_score,
                    workspace.top2_results[i].second_best_score)
        << "point=" << i;
  }
}

// Workspace reuse with different-sized inputs.
TEST_P(ProcessLeafTop2Test, WorkspaceResizeShrink) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;

  // First call with all points.
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  // Second call with fewer points (reuse workspace).
  const size_t half = std::max(size_t{2}, num_points / 2);
  std::vector<int> small_indices(indices.begin(), indices.begin() + half);
  ProcessLeafTop2(tq, bp, absl::MakeSpan(small_indices), workspace);

  // Results should be valid for the smaller size.
  ASSERT_EQ(workspace.top2_results.size(), half);
  for (size_t i = 0; i < half; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_GE(r.best_idx, 0) << "point=" << i;
    EXPECT_LT(r.best_idx, static_cast<int32_t>(half)) << "point=" << i;
    EXPECT_LE(r.best_score, r.second_best_score) << "point=" << i;
  }
}

// Distinct top-2: idx1 != idx2 for all points.
TEST_P(ProcessLeafTop2Test, DistinctNeighbors) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    if (r.second_best_idx >= 0) {
      EXPECT_NE(r.best_idx, r.second_best_idx)
          << "point=" << i << " top-1 and top-2 are the same";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Dims, ProcessLeafTop2Test,
    ::testing::Values(std::make_tuple(32, 16),
                      std::make_tuple(64, 32),    // small
                      std::make_tuple(96, 16),    // non-power-of-2 dim
                      std::make_tuple(128, 32),   // typical
                      std::make_tuple(256, 32),   // medium
                      std::make_tuple(768, 16)),  // large
    [](const ::testing::TestParamInfo<ProcessLeafTop2Test::ParamType>& info) {
      return "dim" + std::to_string(std::get<0>(info.param)) + "_n" +
             std::to_string(std::get<1>(info.param));
    });

class ProcessLeafTop2L2Test
    : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

TEST_P(ProcessLeafTop2L2Test, ScoresOrderedAndNonNegative) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(
      dim, num_points, data, TurboQuantDistanceMetric::kSquaredL2Distance);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_LE(r.best_score, r.second_best_score)
        << "point=" << i << " dim=" << dim << " n=" << num_points;
    EXPECT_GE(r.best_idx, 0) << "point=" << i;
    EXPECT_LT(r.best_idx, static_cast<int32_t>(num_points)) << "point=" << i;
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i))
        << "point=" << i << " nearest neighbor should not be self";
    // L2 distances should be approximately non-negative.
    // Allow small negative due to quantization error.
    EXPECT_GE(r.best_score, -1.0f) << "point=" << i
                                   << " L2 score1 should be approximately "
                                      "non-negative";
  }
}

TEST_P(ProcessLeafTop2L2Test, SelfExclusion) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(
      dim, num_points, data, TurboQuantDistanceMetric::kSquaredL2Distance);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i)) << "point=" << i;
    if (r.second_best_idx >= 0) {
      EXPECT_NE(r.second_best_idx, static_cast<int32_t>(i)) << "point=" << i;
    }
  }
}

TEST_P(ProcessLeafTop2L2Test, ScoresFinite) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(
      dim, num_points, data, TurboQuantDistanceMetric::kSquaredL2Distance);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_TRUE(std::isfinite(r.best_score))
        << "point=" << i << " score1=" << r.best_score;
    EXPECT_TRUE(std::isfinite(r.second_best_score))
        << "point=" << i << " score2=" << r.second_best_score;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Dims, ProcessLeafTop2L2Test,
    ::testing::Values(std::make_tuple(32, 16), std::make_tuple(64, 32),
                      std::make_tuple(128, 32), std::make_tuple(256, 32),
                      std::make_tuple(768, 16)),
    [](const ::testing::TestParamInfo<ProcessLeafTop2L2Test::ParamType>& info) {
      return "dim" + std::to_string(std::get<0>(info.param)) + "_n" +
             std::to_string(std::get<1>(info.param));
    });

// Test with num_points that are NOT multiples of SIMD width
// to detect OOB reads/writes or incorrect padding.

class ProcessLeafTop2PaddingTest
    : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

TEST_P(ProcessLeafTop2PaddingTest, NoSigsegvAndCorrectResults) {
  const auto [dim, num_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  ASSERT_EQ(workspace.top2_results.size(), num_points);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    // All indices must be valid (within [0, num_points)).
    EXPECT_GE(r.best_idx, 0) << "point=" << i;
    EXPECT_LT(r.best_idx, static_cast<int32_t>(num_points)) << "point=" << i;
    // All scores must be finite (no +inf from padding).
    EXPECT_TRUE(std::isfinite(r.best_score))
        << "point=" << i << " score1=" << r.best_score;
    EXPECT_TRUE(std::isfinite(r.second_best_score))
        << "point=" << i << " score2=" << r.second_best_score;
    // No self-neighbors.
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i)) << "point=" << i;
    if (r.second_best_idx >= 0) {
      EXPECT_NE(r.second_best_idx, static_cast<int32_t>(i)) << "point=" << i;
    }
    // Ordering.
    EXPECT_LE(r.best_score, r.second_best_score) << "point=" << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    SIMDEdgeCases, ProcessLeafTop2PaddingTest,
    ::testing::Values(
        // Non-multiples of 16 (AVX-512 int32 lane count)
        std::make_tuple(128, 17), std::make_tuple(128, 33),
        std::make_tuple(128, 65), std::make_tuple(128, 100),
        // Non-multiples of 4 (NEON int32 lane count)
        std::make_tuple(128, 5), std::make_tuple(128, 7),
        std::make_tuple(128, 9), std::make_tuple(128, 13),
        // Near SIMD boundaries
        std::make_tuple(128, 15), std::make_tuple(128, 16),
        std::make_tuple(128, 31), std::make_tuple(128, 32)),
    [](const ::testing::TestParamInfo<ProcessLeafTop2PaddingTest::ParamType>&
           info) {
      return "dim" + std::to_string(std::get<0>(info.param)) + "_n" +
             std::to_string(std::get<1>(info.param));
    });

class ProcessLeafTop2SubsetTest
    : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

TEST_P(ProcessLeafTop2SubsetTest, NonContiguousIndices) {
  const auto [dim, total_points] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(total_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, total_points, data);

  // Use only even-indexed points.
  std::vector<int> indices;
  for (size_t i = 0; i < total_points; i += 2) {
    indices.push_back(i);
  }
  const size_t n = indices.size();

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  ASSERT_EQ(workspace.top2_results.size(), n);

  for (size_t i = 0; i < n; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_GE(r.best_idx, 0) << "point=" << i;
    EXPECT_LT(r.best_idx, static_cast<int32_t>(n)) << "point=" << i;
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i))
        << "point=" << i << " nearest neighbor should not be self";
    EXPECT_TRUE(std::isfinite(r.best_score)) << "point=" << i;
    EXPECT_LE(r.best_score, r.second_best_score) << "point=" << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Dims, ProcessLeafTop2SubsetTest,
    ::testing::Values(std::make_tuple(64, 32), std::make_tuple(128, 64),
                      std::make_tuple(256, 32)),
    [](const ::testing::TestParamInfo<ProcessLeafTop2SubsetTest::ParamType>&
           info) {
      return "dim" + std::to_string(std::get<0>(info.param)) + "_total" +
             std::to_string(std::get<1>(info.param));
    });

TEST(Top2ResultUpdateTest, BasicInsertions) {
  Top2Result r;
  r.Update(5.0f, 0);
  EXPECT_FLOAT_EQ(r.best_score, 5.0f);
  EXPECT_EQ(r.best_idx, 0);

  r.Update(3.0f, 1);
  EXPECT_FLOAT_EQ(r.best_score, 3.0f);
  EXPECT_EQ(r.best_idx, 1);
  EXPECT_FLOAT_EQ(r.second_best_score, 5.0f);
  EXPECT_EQ(r.second_best_idx, 0);

  // Worse than both — no change.
  r.Update(10.0f, 2);
  EXPECT_FLOAT_EQ(r.best_score, 3.0f);
  EXPECT_EQ(r.best_idx, 1);
  EXPECT_FLOAT_EQ(r.second_best_score, 5.0f);
  EXPECT_EQ(r.second_best_idx, 0);
}

TEST(Top2ResultUpdateTest, SecondBestOnly) {
  Top2Result r;
  r.Update(1.0f, 0);
  r.Update(3.0f, 1);
  // Insert between first and second.
  r.Update(2.0f, 2);
  EXPECT_FLOAT_EQ(r.best_score, 1.0f);
  EXPECT_EQ(r.best_idx, 0);
  EXPECT_FLOAT_EQ(r.second_best_score, 2.0f);
  EXPECT_EQ(r.second_best_idx, 2);
}

TEST(Top2ResultUpdateTest, ReverseOrderInsertion) {
  Top2Result r;
  r.Update(10.0f, 0);
  r.Update(8.0f, 1);
  r.Update(6.0f, 2);
  r.Update(4.0f, 3);
  r.Update(2.0f, 4);
  EXPECT_FLOAT_EQ(r.best_score, 2.0f);
  EXPECT_EQ(r.best_idx, 4);
  EXPECT_FLOAT_EQ(r.second_best_score, 4.0f);
  EXPECT_EQ(r.second_best_idx, 3);
}

// Ground-truth brute-force comparison
// Computes all-pairs distances via UnsafeComputeDistances and checks that
// ProcessLeafTop2's results are consistent with this independent reference.
//
// NOTE: ProcessLeafTop2 uses symmetric computation (processes the lower
// triangle and updates both query-side and db-side trackers), whereas
// UnsafeComputeDistances uses an asymmetric query-vs-base code path.
// Quantization causes these two paths to produce slightly different
// distances for the same (i,j) pair, so exact index/score equality is
// not expected. Instead we verify:
//  1. Score ranking: ProcessLeaf's top-1 score is within a tolerance
//     of the brute-force top-1 score.
//  2. Basic sanity: self-exclusion, distinct neighbors, finite scores.

class ProcessLeafBruteForceTest
    : public ::testing::TestWithParam<
          std::tuple<size_t, size_t, TurboQuantDistanceMetric>> {};

TEST_P(ProcessLeafBruteForceTest, MatchesBruteForce) {
  const auto [dim, num_points, metric] = GetParam();
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data, metric);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  // --- ProcessLeaf result ---
  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  // --- Brute-force reference via UnsafeComputeDistances ---
  for (size_t i = 0; i < num_points; ++i) {
    auto query_or =
        TurboQuantQuery::CreateFromQuantizedDataPoint(tq, bp.DataPointView(i));
    ABSL_CHECK_OK(query_or);
    auto query = *std::move(query_or);

    std::vector<int32_t> all_indices(num_points);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::vector<float> scores(num_points);
    bp.UnsafeComputeDistances(query,
                              CreateScoringAccessor(absl::MakeSpan(all_indices),
                                                    absl::MakeSpan(scores)));

    // Build brute-force top-2 (excluding self).
    Top2Result bf;
    for (size_t j = 0; j < num_points; ++j) {
      if (j == i) continue;
      bf.Update(scores[j], static_cast<int32_t>(j));
    }

    const auto& r = workspace.top2_results[i];

    // Sanity: self-exclusion.
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i))
        << "point=" << i << " selected itself as idx1";
    if (r.second_best_idx >= 0) {
      EXPECT_NE(r.second_best_idx, static_cast<int32_t>(i))
          << "point=" << i << " selected itself as idx2";
    }

    // Sanity: distinct neighbors.
    if (r.second_best_idx >= 0) {
      EXPECT_NE(r.best_idx, r.second_best_idx)
          << "point=" << i << " idx1==idx2";
    }

    // Sanity: scores finite and ordered.
    EXPECT_TRUE(std::isfinite(r.best_score)) << "point=" << i;
    EXPECT_TRUE(std::isfinite(r.second_best_score)) << "point=" << i;
    EXPECT_LE(r.best_score, r.second_best_score) << "point=" << i;

    const float tol = 1e-4f;
    EXPECT_NEAR(r.best_score, bf.best_score, tol)
        << "point=" << i << " dim=" << dim << " idx1=" << r.best_idx
        << " bf.idx1=" << bf.best_idx
        << " ProcessLeaf vs brute-force top-1 score mismatch";
    EXPECT_EQ(r.best_idx, bf.best_idx)
        << "point=" << i << " dim=" << dim << " top-1 index mismatch";

    if (bf.second_best_idx >= 0) {
      EXPECT_NEAR(r.second_best_score, bf.second_best_score, tol)
          << "point=" << i << " dim=" << dim << " idx2=" << r.second_best_idx
          << " bf.idx2=" << bf.second_best_idx
          << " ProcessLeaf vs brute-force top-2 score mismatch";
      EXPECT_EQ(r.second_best_idx, bf.second_best_idx)
          << "point=" << i << " dim=" << dim << " top-2 index mismatch";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    BruteForce, ProcessLeafBruteForceTest,
    ::testing::Values(
        std::make_tuple(64, 2, TurboQuantDistanceMetric::kDotProductDistance),
        std::make_tuple(64, 3, TurboQuantDistanceMetric::kDotProductDistance),
        std::make_tuple(64, 16, TurboQuantDistanceMetric::kDotProductDistance),
        std::make_tuple(128, 32, TurboQuantDistanceMetric::kDotProductDistance),
        std::make_tuple(64, 16, TurboQuantDistanceMetric::kSquaredL2Distance),
        std::make_tuple(128, 32, TurboQuantDistanceMetric::kSquaredL2Distance)),
    [](const ::testing::TestParamInfo<ProcessLeafBruteForceTest::ParamType>&
           info) {
      return "dim" + std::to_string(std::get<0>(info.param)) + "_n" +
             std::to_string(std::get<1>(info.param)) +
             (std::get<2>(info.param) ==
                      TurboQuantDistanceMetric::kSquaredL2Distance
                  ? "_l2"
                  : "_dot");
    });

// Minimum input sizes (n=2, n=3)
class ProcessLeafMinSizeTest : public ::testing::TestWithParam<size_t> {};

TEST_P(ProcessLeafMinSizeTest, MinimalPoints) {
  const size_t num_points = GetParam();
  const size_t dim = 64;
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  ASSERT_EQ(workspace.top2_results.size(), num_points);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    // top-1 must exist and not be self.
    EXPECT_GE(r.best_idx, 0) << "point=" << i;
    EXPECT_LT(r.best_idx, static_cast<int32_t>(num_points)) << "point=" << i;
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i)) << "point=" << i;
    EXPECT_TRUE(std::isfinite(r.best_score)) << "point=" << i;

    if (num_points == 2) {
      // With n=2 there is only one other point, so top-2 should be
      // unset (score2 = +inf or idx2 = -1).
      EXPECT_EQ(r.second_best_idx, -1)
          << "point=" << i << " n=2 should not have a second neighbor";
    } else {
      // With n=3, top-2 should also be valid.
      EXPECT_GE(r.second_best_idx, 0) << "point=" << i;
      EXPECT_NE(r.second_best_idx, static_cast<int32_t>(i)) << "point=" << i;
      EXPECT_NE(r.best_idx, r.second_best_idx) << "point=" << i;
      EXPECT_LE(r.best_score, r.second_best_score) << "point=" << i;
    }
  }
}

// n=2: each point's only neighbor is the other.
// n=3: smallest case with full top-2.
INSTANTIATE_TEST_SUITE_P(MinSizes, ProcessLeafMinSizeTest,
                         ::testing::Values(2, 3));

// kMq batch boundary sizes (n=5, 6, 7)
// kMq=6 in ProcessLeafTop2Impl, so these exercise:
//   n=5: single partial batch
//   n=6: exactly one full batch
//   n=7: one full batch + one partial

class ProcessLeafKMqBoundaryTest : public ::testing::TestWithParam<size_t> {};

TEST_P(ProcessLeafKMqBoundaryTest, BatchBoundary) {
  const size_t num_points = GetParam();
  const size_t dim = 128;
  std::mt19937 rng(42);

  auto data = MakeRandomData(num_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  ASSERT_EQ(workspace.top2_results.size(), num_points);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_GE(r.best_idx, 0) << "point=" << i;
    EXPECT_LT(r.best_idx, static_cast<int32_t>(num_points)) << "point=" << i;
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i)) << "point=" << i;
    EXPECT_TRUE(std::isfinite(r.best_score)) << "point=" << i;
    EXPECT_LE(r.best_score, r.second_best_score) << "point=" << i;
    if (r.second_best_idx >= 0) {
      EXPECT_NE(r.second_best_idx, static_cast<int32_t>(i)) << "point=" << i;
      EXPECT_NE(r.best_idx, r.second_best_idx) << "point=" << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(KMqBounds, ProcessLeafKMqBoundaryTest,
                         ::testing::Values(5, 6, 7, 11, 12, 13));

// Identical / near-duplicate points
TEST(ProcessLeafDuplicatesTest, AllIdenticalPoints) {
  const size_t dim = 128;
  const size_t num_points = 16;

  // All points are the same vector.
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<float> single_vec(dim);
  for (auto& v : single_vec) v = dist(rng);

  std::vector<float> data;
  data.reserve(num_points * dim);
  for (size_t i = 0; i < num_points; ++i) {
    data.insert(data.end(), single_vec.begin(), single_vec.end());
  }

  auto [tq, bp] = MakeQuantizedPoints(dim, num_points, data);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  ASSERT_EQ(workspace.top2_results.size(), num_points);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    // Must not select self.
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i)) << "point=" << i;
    if (r.second_best_idx >= 0) {
      EXPECT_NE(r.second_best_idx, static_cast<int32_t>(i)) << "point=" << i;
    }
    // Distinct neighbors.
    if (r.second_best_idx >= 0) {
      EXPECT_NE(r.best_idx, r.second_best_idx) << "point=" << i;
    }
    // Scores should be finite.
    EXPECT_TRUE(std::isfinite(r.best_score)) << "point=" << i;
    EXPECT_TRUE(std::isfinite(r.second_best_score)) << "point=" << i;
    // Ordering preserved.
    EXPECT_LE(r.best_score, r.second_best_score) << "point=" << i;
  }
}

TEST(ProcessLeafDuplicatesTest, AllIdenticalPointsL2) {
  const size_t dim = 128;
  const size_t num_points = 16;

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<float> single_vec(dim);
  for (auto& v : single_vec) v = dist(rng);

  std::vector<float> data;
  data.reserve(num_points * dim);
  for (size_t i = 0; i < num_points; ++i) {
    data.insert(data.end(), single_vec.begin(), single_vec.end());
  }

  auto [tq, bp] = MakeQuantizedPoints(
      dim, num_points, data, TurboQuantDistanceMetric::kSquaredL2Distance);

  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  for (size_t i = 0; i < num_points; ++i) {
    const auto& r = workspace.top2_results[i];
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i)) << "point=" << i;
    // L2 distances between identical points will NOT be exactly zero
    // because 4-bit quantization introduces rounding error. The
    // quantized L2 distance of identical float vectors is dominated by
    // quantization noise, which can be substantial (several units).
    // We just verify the score is finite and non-negative.
    EXPECT_TRUE(std::isfinite(r.best_score))
        << "point=" << i << " non-finite L2 score";
    EXPECT_GE(r.best_score, -0.1f)
        << "point=" << i << " L2 distance should be non-negative";
  }
}

TEST(ProcessLeafSmallNTest, NonContiguousIndicesLocalOffsets) {
  const size_t dim = 64;
  const size_t total_points = 10;
  std::mt19937 rng(42);

  auto data = MakeRandomData(total_points, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, total_points, data);

  // Use a small subset of non-contiguous indices to force ProcessLeafSmallN.
  std::vector<int> indices = {2, 5, 8};
  const size_t n = indices.size();

  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(indices), workspace);

  ASSERT_EQ(workspace.top2_results.size(), n);

  for (size_t i = 0; i < n; ++i) {
    const auto& r = workspace.top2_results[i];
    // The returned indices must be local offsets [0, n), NOT global indices.
    EXPECT_GE(r.best_idx, 0) << "point=" << i;
    EXPECT_LT(r.best_idx, static_cast<int32_t>(n)) << "point=" << i;
    EXPECT_NE(r.best_idx, static_cast<int32_t>(i))
        << "point=" << i << " selected itself";

    if (r.second_best_idx >= 0) {
      EXPECT_LT(r.second_best_idx, static_cast<int32_t>(n)) << "point=" << i;
      EXPECT_NE(r.second_best_idx, static_cast<int32_t>(i)) << "point=" << i;
    }
  }
}

TEST(ProcessLeafSmallNTest, EmptyIndicesReturnsEmptyResults) {
  const size_t dim = 64;
  std::mt19937 rng(42);
  auto data = MakeRandomData(/*num_points=*/4, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, /*num_points=*/4, data);

  std::vector<int> empty_indices;
  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(empty_indices), workspace);
  EXPECT_TRUE(workspace.top2_results.empty());
}

TEST(ProcessLeafSmallNTest, SinglePointReturnsUnsetTop2) {
  const size_t dim = 64;
  std::mt19937 rng(42);
  auto data = MakeRandomData(/*num_points=*/4, dim, rng);
  auto [tq, bp] = MakeQuantizedPoints(dim, /*num_points=*/4, data);

  std::vector<int> single_index = {2};
  ProcessLeafWorkspace workspace;
  ProcessLeafTop2(tq, bp, absl::MakeSpan(single_index), workspace);
  ASSERT_EQ(workspace.top2_results.size(), 1u);
  EXPECT_EQ(workspace.top2_results[0].best_idx, -1);
  EXPECT_EQ(workspace.top2_results[0].second_best_idx, -1);
}

}  // namespace
}  // namespace ginger
