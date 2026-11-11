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

#include "ginger/turboquant/pipnn/robust_prune.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"
#include "ginger/vamana/prune.h"

namespace ginger {
namespace {

using ::ginger::DistanceAndId;
using ::ginger::RobustPruneConfig;

std::vector<float> MakeRandomData(size_t num_points, size_t dim,
                                  std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dim));
  std::vector<float> data(num_points * dim);
  for (auto& v : data) v = dist(rng);
  return data;
}

std::pair<TurboQuantCodec, TurboQuantDatasetSoA> MakeQuantizedPoints(
    size_t num_points, size_t dim, TurboQuantDistanceMetric metric,
    const std::vector<float>& float_data, bool normalized_datapoints = false) {
  TurboQuantCodec tq(dim, 42, metric);
  auto bp = TurboQuantDatasetSoA::Create(
      tq, num_points,
      TurboQuantDatasetOptions{
          .pad_to_cache_line_size = false,
          .quantization_bits = 4,
          .store_query_byte_sums = false,
          .normalized_datapoints = normalized_datapoints,
          .scaling_storage_mode = ScalingStorageMode::kSeparateArray});
  ABSL_CHECK_OK(bp);
  for (size_t i = 0; i < num_points; ++i) {
    ABSL_CHECK_OK(bp->UpdatePoint(
        tq, absl::MakeConstSpan(float_data.data() + i * dim, dim), i));
  }
  return {std::move(tq), *std::move(bp)};
}

// Reference pruning via the canonical RobustPrune implementation using the
// same bf16 TurboQuant distances as RobustPrune.
std::vector<GingerNodeId> RobustPruneReference(
    const TurboQuantDatasetSoA& bp, const TurboQuantCodec& tq,
    absl::Span<const GingerNodeId> sorted_candidates,
    absl::Span<const float> dist_to_p, GingerNodeId p_idx, float alpha,
    size_t max_survivors, bool keep_pruned_neighbors = false) {
  const bool is_dot_product =
      (bp.metric() == TurboQuantDistanceMetric::kDotProductDistance);
  const bool shift_dot = is_dot_product && bp.normalized_datapoints();

  // Build DistanceAndId candidates with pre-filled bf16 distances to p.
  std::vector<DistanceAndId> candidates(sorted_candidates.size());
  for (size_t i = 0; i < sorted_candidates.size(); ++i) {
    float d = dist_to_p[i];
    if (shift_dot) {
      d = std::max(0.0f, 1.0f + d);
    } else if (!is_dot_product) {
      d = std::max(0.0f, d);
    }
    candidates[i] = {.distance = d, .id = sorted_candidates[i]};
  }

  // compute_distances: given query node, compute TurboQuant bf16 distances to
  // each node in the span, writing into DistanceAndId::distance.
  auto compute_distances = [&tq, &bp, shift_dot, is_dot_product](
                               GingerNodeId query_id,
                               absl::Span<DistanceAndId> nodes) {
    auto query_or = TurboQuantQuery::CreateFromQuantizedDataPoint(
        tq, bp.DataPointView(query_id));
    ABSL_CHECK_OK(query_or);
    bp.UnsafeComputeDistances(*query_or, CreateScoringAccessor(nodes));
    if (shift_dot) {
      for (auto& node : nodes) {
        node.distance = std::max(0.0f, 1.0f + node.distance);
      }
    } else if (!is_dot_product) {
      for (auto& node : nodes) {
        node.distance = std::max(0.0f, node.distance);
      }
    }
  };

  RobustPruneConfig prune_config;
  prune_config.set_alpha(alpha);
  prune_config.set_nonpositive_distances(is_dot_product && !shift_dot);
  prune_config.set_keep_pruned_neighbors(keep_pruned_neighbors);

  return ::ginger::RobustPrune(absl::MakeSpan(candidates), p_idx, max_survivors,
                               compute_distances, prune_config);
}

// Returns the number of elements in sorted_a that also appear in sorted_b.
// Both inputs must be sorted.
size_t CountSetIntersection(const std::vector<GingerNodeId>& sorted_a,
                            const std::vector<GingerNodeId>& sorted_b) {
  std::vector<GingerNodeId> intersection;
  absl::c_set_intersection(sorted_a, sorted_b,
                           std::back_inserter(intersection));
  return intersection.size();
}

RobustPruneConfig MakePruneConfig(float alpha, bool keep_pruned_neighbors) {
  RobustPruneConfig config;
  config.set_alpha(alpha);
  config.set_keep_pruned_neighbors(keep_pruned_neighbors);
  return config;
}

RobustPruneConfig MakePruneConfig(float alpha) {
  return MakePruneConfig(alpha, /*keep_pruned_neighbors=*/false);
}

class AcceleratedRobustPruneTest
    : public ::testing::TestWithParam<
          std::pair<size_t, TurboQuantDistanceMetric>> {};

TEST_P(AcceleratedRobustPruneTest, MatchesRobustPrune) {
  const auto [dim, metric] = GetParam();
  constexpr size_t kNumPoints = 120;
  std::mt19937 rng(42);
  const auto float_data = MakeRandomData(kNumPoints, dim, rng);
  const auto [tq, bp] =
      MakeQuantizedPoints(kNumPoints, dim, metric, float_data);

  // Choose point 0 as p_idx.
  const GingerNodeId p_idx = 0;
  ASSERT_OK_AND_ASSIGN(auto query,
                       TurboQuantQuery::CreateFromQuantizedDataPoint(
                           tq, bp.DataPointView(p_idx)));

  std::vector<GingerNodeId> all_indices(kNumPoints);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::vector<float> scores(kNumPoints);
  bp.UnsafeComputeDistances(query,
                            CreateScoringAccessor(absl::MakeSpan(all_indices),
                                                  absl::MakeSpan(scores)));

  // Candidate list excluding p_idx.
  std::vector<GingerNodeId> candidates;
  for (GingerNodeId i = 0; i < static_cast<GingerNodeId>(kNumPoints); ++i) {
    if (i != p_idx) candidates.push_back(i);
  }

  // Sort candidates by ascending distance to p_idx.
  std::sort(
      candidates.begin(), candidates.end(),
      [&](GingerNodeId a, GingerNodeId b) { return scores[a] < scores[b]; });

  std::vector<float> dist_to_p(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    dist_to_p[i] = scores[candidates[i]];
  }

  RobustPruneWorkspace workspace;
  std::vector<GingerNodeId> pruning_survivors;
  constexpr float kAlpha = 1.1f;
  constexpr size_t kMaxSurvivors = 20;

  AcceleratedRobustPrune(
      bp, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      MakePruneConfig(kAlpha), kMaxSurvivors, &workspace, pruning_survivors);

  std::vector<GingerNodeId> surv_robust = RobustPruneReference(
      bp, tq, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      kAlpha, kMaxSurvivors);

  // RobustPrune output is not necessarily sorted, so sort both for comparison.
  std::sort(pruning_survivors.begin(), pruning_survivors.end());
  std::sort(surv_robust.begin(), surv_robust.end());

  EXPECT_FALSE(pruning_survivors.empty());
  EXPECT_LE(pruning_survivors.size(), kMaxSurvivors);
  // The SIMD integer dot product pipeline in RobustPrune may produce
  // slightly different float results than the standard float pipeline used by
  // RobustPrune at numerical boundaries, causing at most a few marginal
  // pruning differences.
  const size_t overlap = CountSetIntersection(pruning_survivors, surv_robust);
  const size_t max_size =
      std::max(pruning_survivors.size(), surv_robust.size());
  EXPECT_GE(overlap, max_size - 2)
      << "Too many differences between RobustPrune and reference RobustPrune";
}

TEST_P(AcceleratedRobustPruneTest, MatchesRobustPruneWithNormalizedDatapoints) {
  const auto [dim, metric] = GetParam();
  constexpr size_t kNumPoints = 120;
  std::mt19937 rng(42);
  auto float_data = MakeRandomData(kNumPoints, dim, rng);
  // Normalize each vector to unit norm.
  for (size_t i = 0; i < kNumPoints; ++i) {
    float norm = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      norm += float_data[i * dim + d] * float_data[i * dim + d];
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
      for (size_t d = 0; d < dim; ++d) {
        float_data[i * dim + d] /= norm;
      }
    }
  }
  const auto [tq, bp] = MakeQuantizedPoints(kNumPoints, dim, metric, float_data,
                                            /*normalized_datapoints=*/true);

  const GingerNodeId p_idx = 0;
  ASSERT_OK_AND_ASSIGN(auto query,
                       TurboQuantQuery::CreateFromQuantizedDataPoint(
                           tq, bp.DataPointView(p_idx)));

  std::vector<GingerNodeId> all_indices(kNumPoints);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::vector<float> scores(kNumPoints);
  bp.UnsafeComputeDistances(query,
                            CreateScoringAccessor(absl::MakeSpan(all_indices),
                                                  absl::MakeSpan(scores)));

  std::vector<GingerNodeId> candidates;
  for (GingerNodeId i = 0; i < static_cast<GingerNodeId>(kNumPoints); ++i) {
    if (i != p_idx) candidates.push_back(i);
  }

  std::sort(
      candidates.begin(), candidates.end(),
      [&](GingerNodeId a, GingerNodeId b) { return scores[a] < scores[b]; });

  std::vector<float> dist_to_p(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    dist_to_p[i] = scores[candidates[i]];
  }

  RobustPruneWorkspace workspace;
  std::vector<GingerNodeId> pruning_survivors;
  constexpr float kAlpha = 1.1f;
  constexpr size_t kMaxSurvivors = 20;

  AcceleratedRobustPrune(
      bp, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      MakePruneConfig(kAlpha), kMaxSurvivors, &workspace, pruning_survivors);

  std::vector<GingerNodeId> surv_robust = RobustPruneReference(
      bp, tq, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      kAlpha, kMaxSurvivors);

  std::sort(pruning_survivors.begin(), pruning_survivors.end());
  std::sort(surv_robust.begin(), surv_robust.end());

  EXPECT_FALSE(pruning_survivors.empty());
  EXPECT_LE(pruning_survivors.size(), kMaxSurvivors);
  const size_t overlap = CountSetIntersection(pruning_survivors, surv_robust);
  const size_t max_size =
      std::max(pruning_survivors.size(), surv_robust.size());
  EXPECT_GE(overlap, max_size - 2)
      << "Too many differences between RobustPrune and reference RobustPrune";
}

TEST_P(AcceleratedRobustPruneTest, LargerAlphaRelaxesPruning) {
  const auto [dim, metric] = GetParam();
  constexpr size_t kNumPoints = 120;
  std::mt19937 rng(42);
  const auto float_data = MakeRandomData(kNumPoints, dim, rng);
  const auto [tq, bp] =
      MakeQuantizedPoints(kNumPoints, dim, metric, float_data);

  const GingerNodeId p_idx = 0;
  ASSERT_OK_AND_ASSIGN(auto query,
                       TurboQuantQuery::CreateFromQuantizedDataPoint(
                           tq, bp.DataPointView(p_idx)));

  std::vector<GingerNodeId> all_indices(kNumPoints);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::vector<float> scores(kNumPoints);
  bp.UnsafeComputeDistances(query,
                            CreateScoringAccessor(absl::MakeSpan(all_indices),
                                                  absl::MakeSpan(scores)));

  // Candidate list excluding p_idx.
  std::vector<GingerNodeId> candidates;
  for (GingerNodeId i = 0; i < static_cast<GingerNodeId>(kNumPoints); ++i) {
    if (i != p_idx) candidates.push_back(i);
  }

  // Sort candidates by ascending distance to p_idx.
  std::sort(
      candidates.begin(), candidates.end(),
      [&](GingerNodeId a, GingerNodeId b) { return scores[a] < scores[b]; });

  std::vector<float> dist_to_p(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    dist_to_p[i] = scores[candidates[i]];
  }

  RobustPruneWorkspace workspace;
  constexpr size_t kMaxSurvivors = 60;

  std::vector<GingerNodeId> surv_tight;
  AcceleratedRobustPrune(
      bp, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      MakePruneConfig(1.0f), kMaxSurvivors, &workspace, surv_tight);

  std::vector<GingerNodeId> surv_relaxed;
  AcceleratedRobustPrune(
      bp, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      MakePruneConfig(1.3f), kMaxSurvivors, &workspace, surv_relaxed);

  EXPECT_GE(surv_relaxed.size(), surv_tight.size());
}

TEST_P(AcceleratedRobustPruneTest, KeepPrunedNeighborsFillsUpToMaxSurvivors) {
  const auto [dim, metric] = GetParam();
  constexpr size_t kNumPoints = 120;
  std::mt19937 rng(42);
  const auto float_data = MakeRandomData(kNumPoints, dim, rng);
  const auto [tq, bp] =
      MakeQuantizedPoints(kNumPoints, dim, metric, float_data);

  const GingerNodeId p_idx = 0;
  ASSERT_OK_AND_ASSIGN(auto query,
                       TurboQuantQuery::CreateFromQuantizedDataPoint(
                           tq, bp.DataPointView(p_idx)));

  std::vector<GingerNodeId> all_indices(kNumPoints);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::vector<float> scores(kNumPoints);
  bp.UnsafeComputeDistances(query,
                            CreateScoringAccessor(absl::MakeSpan(all_indices),
                                                  absl::MakeSpan(scores)));

  std::vector<GingerNodeId> candidates;
  std::vector<float> dist_to_p;
  for (GingerNodeId i = 0; i < static_cast<GingerNodeId>(kNumPoints); ++i) {
    if (i != p_idx) {
      candidates.push_back(i);
      dist_to_p.push_back(scores[i]);
    }
  }

  RobustPruneWorkspace workspace;
  constexpr size_t kMaxSurvivors = 60;
  constexpr float kAlpha = 1.0f;

  std::vector<GingerNodeId> surv_without_keep;
  AcceleratedRobustPrune(
      bp, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      MakePruneConfig(kAlpha, /*keep_pruned_neighbors=*/false), kMaxSurvivors,
      &workspace, surv_without_keep);

  std::vector<GingerNodeId> surv_with_keep;
  AcceleratedRobustPrune(
      bp, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      MakePruneConfig(kAlpha, /*keep_pruned_neighbors=*/true), kMaxSurvivors,
      &workspace, surv_with_keep);

  EXPECT_EQ(surv_with_keep.size(), kMaxSurvivors);
  ASSERT_LE(surv_without_keep.size(), surv_with_keep.size());
  // The prefix of surv_with_keep must be identical to surv_without_keep.
  for (size_t i = 0; i < surv_without_keep.size(); ++i) {
    EXPECT_EQ(surv_with_keep[i], surv_without_keep[i]);
  }
  // Backfilled pruned candidates must be sorted by increasing distance to
  // p_idx.
  for (size_t i = surv_without_keep.size() + 1; i < surv_with_keep.size();
       ++i) {
    EXPECT_LE(scores[surv_with_keep[i - 1]], scores[surv_with_keep[i]]);
  }
  // All returned IDs must be unique and not equal to p_idx.
  std::vector<GingerNodeId> sorted_keep = surv_with_keep;
  std::sort(sorted_keep.begin(), sorted_keep.end());
  EXPECT_EQ(std::unique(sorted_keep.begin(), sorted_keep.end()),
            sorted_keep.end());
  EXPECT_FALSE(absl::c_binary_search(sorted_keep, p_idx));
}

TEST(RobustPruneBoundaryTest, EmptyCandidates) {
  constexpr size_t kDim = 128;
  constexpr size_t kNumPoints = 10;
  std::mt19937 rng(42);
  const auto float_data = MakeRandomData(kNumPoints, kDim, rng);
  const auto [tq, bp] = MakeQuantizedPoints(
      kNumPoints, kDim, TurboQuantDistanceMetric::kDotProductDistance,
      float_data);

  RobustPruneWorkspace workspace;
  std::vector<GingerNodeId> survivors;
  std::vector<GingerNodeId> empty_cands;
  std::vector<float> empty_dists;
  AcceleratedRobustPrune(bp, absl::MakeSpan(empty_cands),
                         absl::MakeSpan(empty_dists), /*source_node=*/0,
                         MakePruneConfig(1.2f),
                         /*max_survivors=*/10, &workspace, survivors);
  EXPECT_TRUE(survivors.empty());
}

TEST(RobustPruneBoundaryTest, MaxSurvivorsZero) {
  constexpr size_t kDim = 128;
  constexpr size_t kNumPoints = 20;
  std::mt19937 rng(42);
  const auto float_data = MakeRandomData(kNumPoints, kDim, rng);
  const auto [tq, bp] = MakeQuantizedPoints(
      kNumPoints, kDim, TurboQuantDistanceMetric::kDotProductDistance,
      float_data);

  const GingerNodeId p_idx = 0;
  ASSERT_OK_AND_ASSIGN(auto query,
                       TurboQuantQuery::CreateFromQuantizedDataPoint(
                           tq, bp.DataPointView(p_idx)));

  std::vector<GingerNodeId> all_indices(kNumPoints);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::vector<float> scores(kNumPoints);
  bp.UnsafeComputeDistances(query,
                            CreateScoringAccessor(absl::MakeSpan(all_indices),
                                                  absl::MakeSpan(scores)));

  std::vector<GingerNodeId> candidates;
  for (GingerNodeId i = 0; i < static_cast<GingerNodeId>(kNumPoints); ++i) {
    if (i != p_idx) candidates.push_back(i);
  }
  std::sort(
      candidates.begin(), candidates.end(),
      [&](GingerNodeId a, GingerNodeId b) { return scores[a] < scores[b]; });
  std::vector<float> dist_to_p(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    dist_to_p[i] = scores[candidates[i]];
  }

  RobustPruneWorkspace workspace;
  std::vector<GingerNodeId> survivors;
  AcceleratedRobustPrune(bp, absl::MakeSpan(candidates),
                         absl::MakeSpan(dist_to_p), p_idx,
                         MakePruneConfig(1.2f),
                         /*max_survivors=*/0, &workspace, survivors);
  EXPECT_TRUE(survivors.empty());
}

TEST(RobustPruneBoundaryTest, MaxSurvivorsOne) {
  constexpr size_t kDim = 128;
  constexpr size_t kNumPoints = 20;
  std::mt19937 rng(42);
  const auto float_data = MakeRandomData(kNumPoints, kDim, rng);
  const auto [tq, bp] = MakeQuantizedPoints(
      kNumPoints, kDim, TurboQuantDistanceMetric::kDotProductDistance,
      float_data);

  const GingerNodeId p_idx = 0;
  ASSERT_OK_AND_ASSIGN(auto query,
                       TurboQuantQuery::CreateFromQuantizedDataPoint(
                           tq, bp.DataPointView(p_idx)));

  std::vector<GingerNodeId> all_indices(kNumPoints);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::vector<float> scores(kNumPoints);
  bp.UnsafeComputeDistances(query,
                            CreateScoringAccessor(absl::MakeSpan(all_indices),
                                                  absl::MakeSpan(scores)));

  std::vector<GingerNodeId> candidates;
  for (GingerNodeId i = 0; i < static_cast<GingerNodeId>(kNumPoints); ++i) {
    if (i != p_idx) candidates.push_back(i);
  }
  std::sort(
      candidates.begin(), candidates.end(),
      [&](GingerNodeId a, GingerNodeId b) { return scores[a] < scores[b]; });
  std::vector<float> dist_to_p(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    dist_to_p[i] = scores[candidates[i]];
  }

  RobustPruneWorkspace workspace;
  std::vector<GingerNodeId> survivors;
  AcceleratedRobustPrune(bp, absl::MakeSpan(candidates),
                         absl::MakeSpan(dist_to_p), p_idx,
                         MakePruneConfig(1.2f),
                         /*max_survivors=*/1, &workspace, survivors);
  EXPECT_EQ(survivors.size(), 1);
}

TEST(RobustPruneBoundaryTest, CandidatesContainOnlyPIdx) {
  constexpr size_t kDim = 128;
  constexpr size_t kNumPoints = 10;
  std::mt19937 rng(42);
  const auto float_data = MakeRandomData(kNumPoints, kDim, rng);
  const auto [tq, bp] = MakeQuantizedPoints(
      kNumPoints, kDim, TurboQuantDistanceMetric::kDotProductDistance,
      float_data);

  RobustPruneWorkspace workspace;
  std::vector<GingerNodeId> survivors;
  std::vector<GingerNodeId> cands = {0};
  std::vector<float> dists = {0.0f};
  AcceleratedRobustPrune(bp, absl::MakeSpan(cands), absl::MakeSpan(dists),
                         /*source_node=*/0, MakePruneConfig(1.2f),
                         /*max_survivors=*/10, &workspace, survivors);
  EXPECT_TRUE(survivors.empty());
}

TEST(RobustPruneBoundaryTest, FewerCandidatesThanMaxSurvivors) {
  constexpr size_t kDim = 128;
  constexpr size_t kNumPoints = 20;
  std::mt19937 rng(42);
  const auto float_data = MakeRandomData(kNumPoints, kDim, rng);
  const auto [tq, bp] = MakeQuantizedPoints(
      kNumPoints, kDim, TurboQuantDistanceMetric::kDotProductDistance,
      float_data);

  const GingerNodeId p_idx = 0;
  ASSERT_OK_AND_ASSIGN(auto query,
                       TurboQuantQuery::CreateFromQuantizedDataPoint(
                           tq, bp.DataPointView(p_idx)));

  // Use only 5 candidates but set max_survivors to 100.
  std::vector<GingerNodeId> all_indices(kNumPoints);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::vector<float> scores(kNumPoints);
  bp.UnsafeComputeDistances(query,
                            CreateScoringAccessor(absl::MakeSpan(all_indices),
                                                  absl::MakeSpan(scores)));

  std::vector<GingerNodeId> candidates;
  for (GingerNodeId i = 1; i <= 5; ++i) {
    candidates.push_back(i);
  }
  std::sort(
      candidates.begin(), candidates.end(),
      [&](GingerNodeId a, GingerNodeId b) { return scores[a] < scores[b]; });
  std::vector<float> dist_to_p(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    dist_to_p[i] = scores[candidates[i]];
  }

  RobustPruneWorkspace workspace;
  std::vector<GingerNodeId> survivors;
  constexpr size_t kLargeMaxSurvivors = 100;
  AcceleratedRobustPrune(
      bp, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      MakePruneConfig(1.2f), kLargeMaxSurvivors, &workspace, survivors);
  // All candidates could potentially survive, but never more than
  // candidates.size().
  EXPECT_LE(survivors.size(), candidates.size());
  EXPECT_FALSE(survivors.empty());

  // Verify consistency with reference RobustPrune.
  std::vector<GingerNodeId> surv_robust = RobustPruneReference(
      bp, tq, absl::MakeSpan(candidates), absl::MakeSpan(dist_to_p), p_idx,
      1.2f, kLargeMaxSurvivors);
  std::sort(survivors.begin(), survivors.end());
  std::sort(surv_robust.begin(), surv_robust.end());
  const size_t overlap = CountSetIntersection(survivors, surv_robust);
  EXPECT_GE(overlap, std::max(survivors.size(), surv_robust.size()) - 2);
}

TEST(RobustPruneBoundaryTest, WorkspaceReuse) {
  std::mt19937 rng(42);
  RobustPruneWorkspace workspace;

  // Run AcceleratedRobustPrune with different dimensions and metrics using the
  // same workspace to verify no state leaks between calls.
  for (const auto& [dim, metric] : {
           std::pair<size_t, TurboQuantDistanceMetric>{
               128, TurboQuantDistanceMetric::kDotProductDistance},
           std::pair<size_t, TurboQuantDistanceMetric>{
               127, TurboQuantDistanceMetric::kSquaredL2Distance},
           std::pair<size_t, TurboQuantDistanceMetric>{
               3, TurboQuantDistanceMetric::kDotProductDistance},
       }) {
    constexpr size_t kNumPoints = 30;
    const auto float_data = MakeRandomData(kNumPoints, dim, rng);
    const auto [tq, bp] =
        MakeQuantizedPoints(kNumPoints, dim, metric, float_data);

    const GingerNodeId p_idx = 0;
    ASSERT_OK_AND_ASSIGN(auto query,
                         TurboQuantQuery::CreateFromQuantizedDataPoint(
                             tq, bp.DataPointView(p_idx)));

    std::vector<GingerNodeId> all_indices(kNumPoints);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::vector<float> scores(kNumPoints);
    bp.UnsafeComputeDistances(query,
                              CreateScoringAccessor(absl::MakeSpan(all_indices),
                                                    absl::MakeSpan(scores)));

    std::vector<GingerNodeId> candidates;
    for (GingerNodeId i = 0; i < static_cast<GingerNodeId>(kNumPoints); ++i) {
      if (i != p_idx) candidates.push_back(i);
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [&](GingerNodeId a, GingerNodeId b) { return scores[a] < scores[b]; });
    std::vector<float> dist_to_p(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      dist_to_p[i] = scores[candidates[i]];
    }

    std::vector<GingerNodeId> pruning_survivors;
    AcceleratedRobustPrune(bp, absl::MakeSpan(candidates),
                           absl::MakeSpan(dist_to_p), p_idx,
                           MakePruneConfig(1.1f),
                           /*max_survivors=*/10, &workspace, pruning_survivors);

    std::vector<GingerNodeId> surv_robust =
        RobustPruneReference(bp, tq, absl::MakeSpan(candidates),
                             absl::MakeSpan(dist_to_p), p_idx, 1.1f, 10);
    std::sort(pruning_survivors.begin(), pruning_survivors.end());
    std::sort(surv_robust.begin(), surv_robust.end());
    const size_t overlap = CountSetIntersection(pruning_survivors, surv_robust);
    EXPECT_GE(overlap,
              std::max(pruning_survivors.size(), surv_robust.size()) - 2)
        << "Mismatch for dim=" << dim << " metric=" << static_cast<int>(metric);
  }
}

INSTANTIATE_TEST_SUITE_P(
    AcceleratedRobustPruneTests, AcceleratedRobustPruneTest,
    ::testing::Values(
        std::make_pair(size_t{3},
                       TurboQuantDistanceMetric::kDotProductDistance),
        std::make_pair(size_t{127},
                       TurboQuantDistanceMetric::kDotProductDistance),
        std::make_pair(size_t{128},
                       TurboQuantDistanceMetric::kDotProductDistance),
        std::make_pair(size_t{256},
                       TurboQuantDistanceMetric::kDotProductDistance),
        std::make_pair(size_t{3}, TurboQuantDistanceMetric::kSquaredL2Distance),
        std::make_pair(size_t{127},
                       TurboQuantDistanceMetric::kSquaredL2Distance),
        std::make_pair(size_t{128},
                       TurboQuantDistanceMetric::kSquaredL2Distance),
        std::make_pair(size_t{256},
                       TurboQuantDistanceMetric::kSquaredL2Distance)));

}  // namespace
}  // namespace ginger
