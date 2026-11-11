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

#include "ginger/candidate_filtering/jl.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/candidate_filtering/candidate_filter.h"
#include "ginger/dataset.h"
#include "ginger/searcher.pb.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

std::unique_ptr<SpanDataset> CreateRandomDataset(size_t num_points, size_t dim,
                                                 uint64_t seed = 42) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  auto dataset = std::make_unique<SpanDataset>(num_points, dim);
  for (size_t i = 0; i < num_points; ++i) {
    auto pt = dataset->get<float>(i);
    for (size_t d = 0; d < dim; ++d) {
      pt[d] = dist(rng);
    }
  }
  return dataset;
}

std::unique_ptr<SpanDataset> CreateOneHotDataset(size_t num_points,
                                                 size_t dim) {
  auto dataset = std::make_unique<SpanDataset>(num_points, dim);
  for (size_t i = 0; i < num_points; ++i) {
    auto pt = dataset->get<float>(i);
    for (size_t d = 0; d < dim; ++d) {
      pt[d] = (d == (i % dim)) ? 1.0f : 0.0f;
    }
  }
  return dataset;
}

TEST(JlDatasetTest, CreateValidationFails) {
  auto dataset = CreateRandomDataset(10, 64);
  const size_t num_points = 10;

  // Invalid source_dim for preallocated dataset
  EXPECT_THAT(JlDataset::Create(0, 512, 123, DOT_PRODUCT, num_points),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(JlDataset::Create(-1, 512, 123, DOT_PRODUCT, num_points),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Invalid dataset point size
  SpanDataset zero_dim_dataset(10, 0);
  EXPECT_THAT(JlDataset::Create(512, 123, DOT_PRODUCT, zero_dim_dataset),
              StatusIs(absl::StatusCode::kInvalidArgument));
  SpanDataset unaligned_dataset(10, 3, CoordinateType::kInt8);
  EXPECT_THAT(JlDataset::Create(512, 123, DOT_PRODUCT, unaligned_dataset),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Invalid jl_dim
  EXPECT_THAT(JlDataset::Create(0, 123, DOT_PRODUCT, *dataset),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(JlDataset::Create(64, 0, 123, DOT_PRODUCT, num_points),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      JlDataset::Create(100, 123, DOT_PRODUCT, *dataset),
      StatusIs(absl::StatusCode::kInvalidArgument));  // Not multiple of 64
  EXPECT_THAT(JlDataset::Create(64, 100, 123, DOT_PRODUCT, num_points),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Invalid seed
  EXPECT_THAT(JlDataset::Create(512, -1, DOT_PRODUCT, *dataset),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(JlDataset::Create(64, 512, -1, DOT_PRODUCT, num_points),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Undefined metric
  EXPECT_THAT(JlDataset::Create(512, 123, UNDEFINED_DISTANCE, *dataset),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(JlDataset::Create(64, 512, 123, UNDEFINED_DISTANCE, num_points),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched data size in Create overload with pre-populated data
  hwy::AlignedVector<uint64_t> wrong_sized_data(1);
  EXPECT_THAT(JlDataset::Create(64, 128, 123, DOT_PRODUCT, num_points,
                                std::move(wrong_sized_data)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JlDatasetTest, CreateFromPrepopulatedData) {
  constexpr int32_t kSourceDim = 64;
  constexpr int32_t kJlDim = 128;
  constexpr int32_t kSeed = 42;
  constexpr size_t kNumPoints = 5;
  constexpr size_t kWordsPerPoint = kJlDim / 64;

  hwy::AlignedVector<uint64_t> data(kNumPoints * kWordsPerPoint);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i + 100;
  }
  std::vector<uint64_t> expected(data.begin(), data.end());

  ASSERT_OK_AND_ASSIGN(auto dataset,
                       JlDataset::Create(kSourceDim, kJlDim, kSeed, DOT_PRODUCT,
                                         kNumPoints, std::move(data)));

  EXPECT_EQ(dataset->source_dim(), kSourceDim);
  EXPECT_EQ(dataset->jl_dim(), kJlDim);
  EXPECT_EQ(dataset->seed(), kSeed);
  EXPECT_EQ(dataset->metric(), DOT_PRODUCT);
  EXPECT_EQ(dataset->size(), kNumPoints);
  EXPECT_THAT(dataset->data(), ElementsAreArray(expected));
}

TEST(JlDatasetTest, QuantizeIdenticalPointsHaveZeroDistance) {
  auto dataset = CreateRandomDataset(10, 128, /*seed=*/1);
  ASSERT_OK_AND_ASSIGN(
      auto jl_dataset,
      JlDataset::Create(1024, /*seed=*/42, SQUARED_L2, *dataset));

  for (size_t i = 0; i < dataset->size(); ++i) {
    auto query = dataset->get<float>(i);
    JlQuery jl_query = jl_dataset->QuantizeQuery(query);

    std::vector<DistanceAndId> candidates = {
        {.id = static_cast<GingerNodeId>(i)}};
    jl_dataset->UnsafeComputeDistances(jl_query, absl::MakeSpan(candidates));
    EXPECT_FLOAT_EQ(candidates[0].distance, 0.0f);
  }
}

TEST(JlDatasetTest, QuantizeOppositePointsHaveMaxDistanceDotProduct) {
  SpanDataset dataset(2, 64);
  for (size_t d = 0; d < 64; ++d) {
    dataset.get<float>(0)[d] = 1.0f;
    dataset.get<float>(1)[d] = -1.0f;
  }
  ASSERT_OK_AND_ASSIGN(
      auto jl_dataset,
      JlDataset::Create(512, /*seed=*/123, DOT_PRODUCT, dataset));

  JlQuery jl_query = jl_dataset->QuantizeQuery(dataset.get<float>(0));
  std::vector<DistanceAndId> candidates = {{.id = 0}, {.id = 1}};
  jl_dataset->UnsafeComputeDistances(jl_query, absl::MakeSpan(candidates));

  EXPECT_FLOAT_EQ(candidates[0].distance, 0.0f);
  EXPECT_FLOAT_EQ(candidates[1].distance, 512.0f);
}

TEST(JlDatasetTest, DistanceCorrelationEuclidean) {
  const size_t dim = 128;
  const size_t num_points = 5;
  std::mt19937_64 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> query(dim);
  std::vector<float> direction(dim);
  for (size_t d = 0; d < dim; ++d) {
    query[d] = dist(rng);
    direction[d] = dist(rng);
  }

  SpanDataset dataset(num_points, dim);
  for (size_t i = 0; i < num_points; ++i) {
    auto pt = dataset.get<float>(i);
    const float scale = static_cast<float>(i + 1) * 0.4f;
    for (size_t d = 0; d < dim; ++d) {
      pt[d] = query[d] + scale * direction[d];
    }
  }

  ASSERT_OK_AND_ASSIGN(auto jl_dataset, JlDataset::Create(1024, /*seed=*/777,
                                                          SQUARED_L2, dataset));

  JlQuery jl_query = jl_dataset->QuantizeQuery(query);
  std::vector<DistanceAndId> candidates;
  for (size_t i = 0; i < num_points; ++i) {
    candidates.push_back({.id = static_cast<GingerNodeId>(i)});
  }

  jl_dataset->UnsafeComputeDistances(jl_query, absl::MakeSpan(candidates));

  // Verify that distances increase as points are perturbed further away.
  for (size_t i = 1; i < num_points; ++i) {
    EXPECT_LT(candidates[i - 1].distance, candidates[i].distance);
  }
}

TEST(JlDatasetTest, UpdatePointAndGetPoint) {
  auto dataset = CreateRandomDataset(5, 64, /*seed=*/99);
  ASSERT_OK_AND_ASSIGN(
      auto jl_dataset,
      JlDataset::Create(256, /*seed=*/1234, DOT_PRODUCT, *dataset));

  std::vector<float> new_point(64, 2.5f);
  jl_dataset->UpdatePoint(2, new_point);

  JlQuery jl_query = jl_dataset->QuantizeQuery(new_point);
  std::vector<DistanceAndId> candidates = {{.id = 2}};
  jl_dataset->UnsafeComputeDistances(jl_query, absl::MakeSpan(candidates));
  EXPECT_FLOAT_EQ(candidates[0].distance, 0.0f);

  absl::Span<const uint64_t> pt_slice = jl_dataset->GetPoint(2);
  EXPECT_EQ(pt_slice.size(), 4);  // 256 / 64 = 4 words
  EXPECT_THAT(pt_slice, ElementsAre(jl_query.data()[0], jl_query.data()[1],
                                    jl_query.data()[2], jl_query.data()[3]));
}

TEST(JlDatasetTest, Reorder) {
  auto dataset = CreateOneHotDataset(4, 64);
  ASSERT_OK_AND_ASSIGN(
      auto jl_dataset,
      JlDataset::Create(256, /*seed=*/55, DOT_PRODUCT, *dataset));

  std::vector<GingerNodeId> permutation = {3, 2, 1, 0};
  auto reordered = jl_dataset->Reorder(permutation);

  for (size_t i = 0; i < 4; ++i) {
    auto orig_pt = jl_dataset->GetPoint(permutation[i]);
    auto reord_pt = reordered->GetPoint(i);
    EXPECT_EQ(orig_pt, reord_pt);
  }

  jl_dataset->ReorderInPlace(permutation);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(jl_dataset->GetPoint(i), reordered->GetPoint(i));
  }
}

TEST(JlDatasetTest, CandidateFilterIntegration) {
  const size_t num_points = 20;
  const size_t dim = 64;
  auto dataset = CreateRandomDataset(num_points, dim, /*seed=*/10);

  ASSERT_OK_AND_ASSIGN(
      auto jl_dataset,
      JlDataset::Create(512, /*seed=*/42, SQUARED_L2, *dataset));

  auto query = dataset->get<float>(0);
  JlQuery jl_query = jl_dataset->QuantizeQuery(query);

  JlSearchScorer scorer{jl_dataset.get(), &jl_query};
  CandidateFilter candidate_filter(
      scorer, BeamMeanThresholdPolicy(/*min_beam_size=*/5, /*width=*/3));

  // Initialize beam with closest point and some other points.
  std::vector<BeamEntry> beam = {
      {0.0f, 0}, {10.0f, 1}, {20.0f, 2}, {30.0f, 3}, {40.0f, 4}};

  // Candidates: point 0 (identical), and all other points
  std::vector<DistanceAndId> candidates;
  for (size_t i = 0; i < num_points; ++i) {
    candidates.push_back({.id = static_cast<GingerNodeId>(i)});
  }

  absl::Span<DistanceAndId> survivors =
      candidate_filter(absl::MakeSpan(candidates), absl::MakeConstSpan(beam));

  // Point 0 must survive because its distance is 0.
  bool found_zero = false;
  for (const auto& surv : survivors) {
    if (surv.id == 0) {
      found_zero = true;
      EXPECT_FLOAT_EQ(surv.distance, 0.0f);
    }
  }
  EXPECT_TRUE(found_zero);
}

TEST(JlDatasetTest, CreateEmptyStorageAndPopulate) {
  const size_t dim = 64;
  const size_t num_points = 5;
  ASSERT_OK_AND_ASSIGN(
      auto jl_dataset,
      JlDataset::Create(dim, 256, /*seed=*/42, DOT_PRODUCT, num_points));

  EXPECT_EQ(jl_dataset->size(), num_points);
  EXPECT_EQ(jl_dataset->words_per_point(), 4);

  std::vector<float> point(dim, 1.0f);
  jl_dataset->UpdatePoint(0, point);

  JlQuery jl_query = jl_dataset->QuantizeQuery(point);
  std::vector<DistanceAndId> candidates = {{.id = 0}};
  jl_dataset->UnsafeComputeDistances(jl_query, absl::MakeSpan(candidates));
  EXPECT_FLOAT_EQ(candidates[0].distance, 0.0f);
}

TEST(JlDatasetTest, VaryingCandidateBatchSizes) {
  const size_t num_points = 20;
  const size_t dim = 128;
  auto dataset = CreateRandomDataset(num_points, dim, /*seed=*/123);

  ASSERT_OK_AND_ASSIGN(
      auto jl_dataset,
      JlDataset::Create(512, /*seed=*/42, SQUARED_L2, *dataset));

  auto query = dataset->get<float>(0);
  JlQuery jl_query = jl_dataset->QuantizeQuery(query);

  for (size_t batch_size = 0; batch_size <= num_points; ++batch_size) {
    std::vector<DistanceAndId> candidates(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      candidates[i].id = static_cast<GingerNodeId>(i);
    }
    jl_dataset->UnsafeComputeDistances(jl_query, absl::MakeSpan(candidates));

    if (batch_size > 0) {
      EXPECT_FLOAT_EQ(candidates[0].distance, 0.0f);
    }
  }
}

TEST(JlDatasetTest, EmptyDataset) {
  const size_t dim = 64;
  ASSERT_OK_AND_ASSIGN(
      auto jl_dataset,
      JlDataset::Create(dim, 256, /*seed=*/42, DOT_PRODUCT, /*num_points=*/0));

  EXPECT_EQ(jl_dataset->size(), 0);
  EXPECT_EQ(jl_dataset->words_per_point(), 4);

  std::vector<GingerNodeId> empty_perm;
  jl_dataset->ReorderInPlace(empty_perm);
  auto reordered = jl_dataset->Reorder(empty_perm);
  EXPECT_EQ(reordered->size(), 0);
}

TEST(JlDatasetTest, ComputeDistancesNonMultipleDimensionsAndBatches) {
  for (const int32_t jl_dim : {64, 128, 192, 320, 384, 448}) {
    const size_t dim = 64;
    const size_t num_points = 15;
    auto dataset = CreateRandomDataset(num_points, dim, /*seed=*/321);

    ASSERT_OK_AND_ASSIGN(
        auto jl_dataset,
        JlDataset::Create(jl_dim, /*seed=*/42, SQUARED_L2, *dataset));

    auto query = dataset->get<float>(0);
    JlQuery jl_query = jl_dataset->QuantizeQuery(query);

    for (size_t batch_size : {1, 2, 3, 4, 5, 7, 8, 11, 15}) {
      std::vector<DistanceAndId> candidates(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        candidates[i].id = static_cast<GingerNodeId>(i);
      }
      jl_dataset->UnsafeComputeDistances(jl_query, absl::MakeSpan(candidates));
      EXPECT_FLOAT_EQ(candidates[0].distance, 0.0f);
      if (batch_size > 1) {
        EXPECT_GT(candidates[1].distance, 0.0f);
      }
    }
  }
}

TEST(JlDatasetTest, SquaredL2TranslationInvariance) {
  const size_t dim = 64;
  auto dataset = CreateRandomDataset(1, dim, /*seed=*/123);
  ASSERT_OK_AND_ASSIGN(
      auto jl_dataset,
      JlDataset::Create(256, /*seed=*/42, SQUARED_L2, *dataset));

  auto orig_query = dataset->get<float>(0);
  std::vector<float> shifted_query(dim);
  for (size_t d = 0; d < dim; ++d) {
    shifted_query[d] = orig_query[d] + 50.0f;
  }

  JlQuery q_orig = jl_dataset->QuantizeQuery(orig_query);
  JlQuery q_shifted = jl_dataset->QuantizeQuery(shifted_query);

  std::vector<uint64_t> orig_vec(q_orig.data().begin(), q_orig.data().end());
  std::vector<uint64_t> shifted_vec(q_shifted.data().begin(),
                                    q_shifted.data().end());
  EXPECT_EQ(shifted_vec, orig_vec);
}

TEST(JlDatasetTest, QuantizePointDeterministicOutput) {
  const size_t dim = 64;
  // Exact integer coordinates keep the point independent of the standard
  // library and the math library.
  auto dataset = std::make_unique<SpanDataset>(1, dim);
  std::mt19937 rng(1);
  auto pt = dataset->get<float>(0);
  for (size_t d = 0; d < dim; ++d) {
    pt[d] = static_cast<float>(static_cast<int32_t>(rng() & 0xFFFF) - 32768);
  }

  ASSERT_OK_AND_ASSIGN(
      auto l2_dataset,
      JlDataset::Create(256, /*seed=*/42, SQUARED_L2, *dataset));
  ASSERT_OK_AND_ASSIGN(
      auto dot_dataset,
      JlDataset::Create(256, /*seed=*/42, DOT_PRODUCT, *dataset));

  JlQuery l2_query = l2_dataset->QuantizeQuery(pt);
  JlQuery dot_query = dot_dataset->QuantizeQuery(pt);

  EXPECT_EQ(l2_query.num_words(), 4);
  EXPECT_EQ(dot_query.num_words(), 4);

  EXPECT_EQ(
      std::vector<uint64_t>(l2_query.data().begin(), l2_query.data().end()),
      std::vector<uint64_t>({15618935420853057523ULL, 1594968201155040316ULL,
                             7131374709824027337ULL, 4860090755940486459ULL}));
  EXPECT_EQ(
      std::vector<uint64_t>(dot_query.data().begin(), dot_query.data().end()),
      std::vector<uint64_t>({15238663268288680061ULL, 2684362716900467699ULL,
                             17267941001708281584ULL, 3741790203313902433ULL}));
}

}  // namespace
}  // namespace ginger
