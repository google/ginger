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

// Unit tests for gemm_loop-inl.h.

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
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/query/query.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/many_to_many/gemm_loop_test.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/decode_db-inl.h"  // NOLINT
#include "ginger/turboquant/many_to_many/decode_queries-inl.h"  // NOLINT
#include "ginger/turboquant/many_to_many/gemm_loop-inl.h"  // NOLINT

namespace ginger {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pair;
using ::testing::Pointwise;

std::vector<std::vector<float>> GenerateRandomVectors(
    size_t count, size_t dimensionality, std::mt19937& random_generator) {
  std::normal_distribution<float> distribution(
      0.0f, 1.0f / std::sqrt(dimensionality));
  std::vector<std::vector<float>> vectors(count,
                                          std::vector<float>(dimensionality));
  for (std::vector<float>& vector : vectors) {
    for (float& value : vector) value = distribution(random_generator);
  }
  return vectors;
}

TurboQuantDatasetSoA EncodePoints(
    const TurboQuantCodec& turboquant,
    absl::Span<const std::vector<float>> vectors) {
  absl::StatusOr<TurboQuantDatasetSoA> base_points =
      TurboQuantDatasetSoA::Create(turboquant, vectors.size());
  ABSL_CHECK_OK(base_points);
  for (size_t i = 0; i < vectors.size(); ++i) {
    ABSL_CHECK_OK(base_points->UpdatePoint(turboquant, vectors[i], i));
  }
  return *std::move(base_points);
}

std::vector<std::vector<float>> ComputeReferenceScores(
    const TurboQuantDatasetSoA& queries_base_points,
    const TurboQuantDatasetSoA& database_base_points,
    const TurboQuantCodec& turboquant) {
  const size_t num_queries = queries_base_points.num_datapoints();
  const size_t num_db_points = database_base_points.num_datapoints();
  std::vector<std::vector<float>> reference_scores(
      num_queries, std::vector<float>(num_db_points));

  std::vector<int32_t> all_database_indices(num_db_points);
  std::iota(all_database_indices.begin(), all_database_indices.end(), 0);

  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    TurboQuantQuery query = *TurboQuantQuery::CreateFromQuantizedDataPoint(
        turboquant, queries_base_points.DataPointView(query_idx));
    database_base_points.UnsafeComputeDistances(
        query,
        CreateScoringAccessor(absl::MakeSpan(all_database_indices),
                              absl::MakeSpan(reference_scores[query_idx])));
  }
  return reference_scores;
}

struct GemmTestData {
  size_t num_queries;
  size_t num_db_points;
  size_t num_panels;
  size_t points_per_panel;
  size_t tiles_per_vector;
  size_t panel_bytes;
  size_t decoded_dimension;

  hwy::AlignedVector<int8_t> query_decoded_data;
  std::vector<int32_t> query_byte_sums;
  std::vector<float> query_norms;
  std::vector<float> query_squared_norms;

  hwy::AlignedVector<uint8_t> database_panels;
  hwy::AlignedVector<float> database_norm_scaling_factors;
  hwy::AlignedVector<float> database_squared_norms;

  bool is_l2;
  std::vector<std::vector<float>> reference_scores;
  DecodedQueryView decoded_view;
};

GemmTestData SetupGemmData(size_t dimensionality, size_t num_queries,
                           size_t num_db_points, bool is_l2) {
  std::mt19937 random_generator(42);
  std::vector<std::vector<float>> query_vectors =
      GenerateRandomVectors(num_queries, dimensionality, random_generator);
  std::vector<std::vector<float>> database_vectors =
      GenerateRandomVectors(num_db_points, dimensionality, random_generator);

  TurboQuantDistanceMetric metric =
      is_l2 ? TurboQuantDistanceMetric::kSquaredL2Distance
            : TurboQuantDistanceMetric::kDotProductDistance;
  TurboQuantCodec turboquant(dimensionality, /*seed=*/42, metric);
  TurboQuantDatasetSoA queries_base_points =
      EncodePoints(turboquant, query_vectors);
  TurboQuantDatasetSoA database_base_points =
      EncodePoints(turboquant, database_vectors);

  GemmTestData data;
  data.num_queries = num_queries;
  data.num_db_points = num_db_points;
  data.is_l2 = is_l2;
  data.reference_scores = ComputeReferenceScores(
      queries_base_points, database_base_points, turboquant);

  data.decoded_dimension = DecodedQueryStride(queries_base_points);
  data.query_decoded_data.resize(num_queries * data.decoded_dimension);
  data.query_byte_sums.resize(num_queries);
  data.query_norms.resize(num_queries);
  data.query_squared_norms.resize(num_queries);
  data.decoded_view = DecodeQueriesAndMetadata(
      queries_base_points, IndexSubset::Contiguous(0, num_queries),
      absl::MakeSpan(data.query_decoded_data),
      absl::MakeSpan(data.query_byte_sums), absl::MakeSpan(data.query_norms),
      absl::MakeSpan(data.query_squared_norms));
  data.points_per_panel = DbPanelNumPoints();
  data.panel_bytes = DbPanelBytes(database_base_points);
  data.tiles_per_vector =
      data.panel_bytes / hn::Lanes(hn::ScalableTag<int8_t>());
  data.num_panels = hwy::DivCeil(num_db_points, data.points_per_panel);
  data.database_panels.resize(data.num_panels * data.panel_bytes);
  for (size_t panel_idx = 0; panel_idx < data.num_panels; ++panel_idx) {
    const size_t start = panel_idx * data.points_per_panel;
    const size_t size = std::min(data.points_per_panel, num_db_points - start);
    absl::Span<uint8_t> panel_span(
        data.database_panels.data() + panel_idx * data.panel_bytes,
        data.panel_bytes);
    DecodeDbPanel(database_base_points, IndexSubset::Contiguous(start, size),
                  panel_span);
  }
  const size_t padded_database_size = data.num_panels * data.points_per_panel;
  data.database_norm_scaling_factors.assign(padded_database_size, 0.0f);
  data.database_squared_norms.assign(padded_database_size, 0.0f);
  for (size_t database_idx = 0; database_idx < num_db_points; ++database_idx) {
    data.database_norm_scaling_factors[database_idx] =
        database_base_points.norm_scaling_factor(database_idx);
    if (is_l2) {
      data.database_squared_norms[database_idx] =
          database_base_points.unquantized_squared_norm(database_idx);
    }
  }
  return data;
}

template <size_t kQueriesPerBatch>
std::vector<std::vector<float>> RunGemmLoop(const GemmTestData& data) {
  hn::ScalableTag<float> f32;
  const size_t NF = hn::Lanes(f32);

  std::vector<float> buffer(data.num_queries * data.num_panels * NF, -9999.0f);

  DecodedQueryView decoded_view = data.decoded_view;

  RunQueryLoop<kQueriesPerBatch>(
      decoded_view, data.num_queries, data.num_panels, data.tiles_per_vector,
      data.database_norm_scaling_factors.data(),
      data.is_l2 ? data.database_squared_norms.data() : nullptr,
      /*lower_triangle=*/false,
      [&](size_t panel_idx) {
        return data.database_panels.data() + panel_idx * data.panel_bytes;
      },
      [&](size_t /*query_start_idx*/, size_t /*num_active_queries*/,
          const QueryBatch<kQueriesPerBatch>& /*query_batch*/) {},
      [&](auto scores_accessor, size_t query_start_idx,
          size_t num_active_queries, size_t panel_base_idx) HWY_ATTR {
        ForIndex<kQueriesPerBatch>([&](auto query_idx) HWY_ATTR {
          constexpr size_t q = decltype(query_idx)::value;
          if (q < num_active_queries) {
            hn::StoreU(scores_accessor(query_idx), f32,
                       buffer.data() +
                           (query_start_idx + q) * data.num_panels * NF +
                           panel_base_idx);
          }
        });
      },
      [&](size_t /*query_start_idx*/,
          const QueryBatch<kQueriesPerBatch>& /*query_batch*/) {});

  std::vector<std::vector<float>> result(
      data.num_queries, std::vector<float>(data.num_db_points));
  for (size_t query_idx = 0; query_idx < data.num_queries; ++query_idx) {
    for (size_t database_idx = 0; database_idx < data.num_db_points;
         ++database_idx) {
      const size_t panel_idx = database_idx / NF;
      const size_t lane = database_idx % NF;
      result[query_idx][database_idx] =
          buffer[query_idx * data.num_panels * NF + panel_idx * NF + lane];
    }
  }
  return result;
}

MATCHER_P(PointwiseFloatNearTuple, tolerance, "") {
  const std::vector<float>& actual_inner = std::get<0>(arg);
  const std::vector<float>& expected_inner = std::get<1>(arg);
  return testing::ExplainMatchResult(
      Pointwise(FloatNear(tolerance), expected_inner), actual_inner,
      result_listener);
}

void VerifyVsReference(size_t dimensionality, size_t num_queries,
                       size_t num_db_points, bool is_l2 = false,
                       float tolerance = 1e-4f) {
  GemmTestData data =
      SetupGemmData(dimensionality, num_queries, num_db_points, is_l2);
  std::vector<std::vector<float>> actual_scores = RunGemmLoop<4>(data);

  ASSERT_EQ(actual_scores.size(), num_queries);
  EXPECT_THAT(actual_scores, Pointwise(PointwiseFloatNearTuple(tolerance),
                                       data.reference_scores));
}

HWY_NOINLINE void TestPrepareQueryBatchFull() {
  constexpr size_t kQueriesPerBatch = 4;
  constexpr size_t decoded_dimension = 32;
  constexpr size_t num_queries = kQueriesPerBatch;

  std::vector<int8_t> decoded_data(num_queries * decoded_dimension, 1);
  std::vector<int32_t> byte_sums(num_queries);
  std::vector<float> norms(num_queries);
  std::vector<float> squared_norms(num_queries);
  std::vector<const int8_t*> expected_pointers(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    byte_sums[i] = static_cast<int32_t>(i * 3 + 5);
    norms[i] = 0.1f * static_cast<float>(i + 1);
    squared_norms[i] = 0.5f * static_cast<float>(i + 1);
    expected_pointers[i] = decoded_data.data() + i * decoded_dimension;
  }

  DecodedQueryView decoded_view{
      .decoded_data = decoded_data.data(),
      .byte_sums = byte_sums.data(),
      .norm_scaling_factors = norms.data(),
      .squared_norms = squared_norms.data(),
      .decoded_dimension = decoded_dimension,
      .is_l2 = true,
  };

  QueryBatch<kQueriesPerBatch> query_batch;
  internal::PrepareQueryBatch(decoded_view, /*query_offset=*/0, num_queries,
                              query_batch);

  EXPECT_EQ(query_batch.num_active_queries, num_queries);
  EXPECT_THAT(absl::MakeConstSpan(query_batch.query_ptrs),
              ElementsAreArray(expected_pointers));
  EXPECT_THAT(absl::MakeConstSpan(query_batch.norm_scaling_factors),
              Pointwise(FloatEq(), norms));
  EXPECT_THAT(absl::MakeConstSpan(query_batch.squared_norms),
              Pointwise(FloatEq(), squared_norms));
  EXPECT_THAT(absl::MakeConstSpan(query_batch.byte_sums),
              ElementsAreArray(byte_sums));
}

HWY_NOINLINE void TestPrepareQueryBatchPartial() {
  constexpr size_t kQueriesPerBatch = 4;
  constexpr size_t decoded_dimension = 16;
  constexpr size_t num_active_queries = 1;

  std::vector<int8_t> decoded_data(decoded_dimension, 2);
  int32_t byte_sums[1] = {42};
  float norms[1] = {0.7f};
  float squared_norms[1] = {1.2f};

  DecodedQueryView decoded_view{
      .decoded_data = decoded_data.data(),
      .byte_sums = byte_sums,
      .norm_scaling_factors = norms,
      .squared_norms = squared_norms,
      .decoded_dimension = decoded_dimension,
      .is_l2 = true,
  };

  QueryBatch<kQueriesPerBatch> query_batch;
  internal::PrepareQueryBatch(decoded_view, /*query_offset=*/0,
                              num_active_queries, query_batch);

  EXPECT_EQ(query_batch.num_active_queries, num_active_queries);
  EXPECT_THAT(absl::MakeConstSpan(query_batch.query_ptrs),
              Each(decoded_data.data()));
  EXPECT_THAT(absl::MakeConstSpan(query_batch.norm_scaling_factors),
              ElementsAre(0.7f, 0.0f, 0.0f, 0.0f));
  EXPECT_THAT(absl::MakeConstSpan(query_batch.squared_norms),
              ElementsAre(1.2f, 0.0f, 0.0f, 0.0f));
  EXPECT_THAT(absl::MakeConstSpan(query_batch.byte_sums),
              ElementsAre(42, 0, 0, 0));
}

HWY_NOINLINE void TestComputePanelScoresImpl(bool is_l2) {
  constexpr size_t kQueriesPerBatch = 4;
  hn::ScalableTag<int32_t> i32;
  hn::ScalableTag<float> f32;
  const size_t NF = hn::Lanes(f32);

  QueryBatch<kQueriesPerBatch> query_batch;
  query_batch.num_active_queries = kQueriesPerBatch;
  for (size_t query_idx = 0; query_idx < kQueriesPerBatch; ++query_idx) {
    query_batch.norm_scaling_factors[query_idx] =
        0.5f * static_cast<float>(query_idx + 1);
    query_batch.squared_norms[query_idx] =
        is_l2 ? 1.2f * static_cast<float>(query_idx + 1) : 0.0f;
    query_batch.byte_sums[query_idx] = static_cast<int32_t>(query_idx + 1);
  }

  VEC_GROUP_DEFINE(i32, a0);
  VEC_GROUP_DEFINE(i32, bias_correction);

  ForIndex<kQueriesPerBatch>([&](auto query_idx) HWY_ATTR {
    constexpr size_t q = decltype(query_idx)::value;
    VEC_GROUP_ACCESSOR(a0)
    (query_idx) = hn::Set(i32, static_cast<int32_t>(1000 + q * 100));
    VEC_GROUP_ACCESSOR(bias_correction)
    (query_idx) = hn::Set(i32, 128 * query_batch.byte_sums[q]);
  });

  std::vector<float> negated_database_norms(NF);
  std::vector<float> database_squared_norms(NF);
  for (size_t i = 0; i < NF; ++i) {
    negated_database_norms[i] = -0.1f * static_cast<float>(i + 1);
    database_squared_norms[i] = is_l2 ? 0.3f * static_cast<float>(i + 1) : 0.0f;
  }
  hn::Vec<hn::ScalableTag<float>> negated_db_norm_scaling =
      hn::LoadU(f32, negated_database_norms.data());
  hn::Vec<hn::ScalableTag<float>> db_sqr_norms =
      hn::LoadU(f32, database_squared_norms.data());

  std::vector<std::vector<float>> actual_scores(kQueriesPerBatch,
                                                std::vector<float>(NF));

  VEC_GROUP_DEFINE(f32, scores);
  internal::ComputePanelScores<kQueriesPerBatch>(
      VEC_GROUP_ACCESSOR(a0), VEC_GROUP_ACCESSOR(bias_correction), query_batch,
      negated_db_norm_scaling, db_sqr_norms, is_l2, VEC_GROUP_ACCESSOR(scores));

  ForIndex<kQueriesPerBatch>([&](auto query_idx) HWY_ATTR {
    constexpr size_t q = decltype(query_idx)::value;
    hn::StoreU(VEC_GROUP_ACCESSOR(scores)(query_idx), f32,
               actual_scores[q].data());
  });

  for (size_t query_idx = 0; query_idx < kQueriesPerBatch; ++query_idx) {
    int32_t accumulator_val = static_cast<int32_t>(1000 + query_idx * 100);
    int32_t correction_val = 128 * query_batch.byte_sums[query_idx];
    int32_t corrected_accumulator = accumulator_val - correction_val;
    std::vector<float> expected_values(NF);
    for (size_t i = 0; i < NF; ++i) {
      float raw_dot_product =
          static_cast<float>(corrected_accumulator) * negated_database_norms[i];
      expected_values[i] =
          is_l2 ? raw_dot_product * 2.0f *
                          query_batch.norm_scaling_factors[query_idx] +
                      database_squared_norms[i] +
                      query_batch.squared_norms[query_idx]
                : raw_dot_product * query_batch.norm_scaling_factors[query_idx];
    }
    EXPECT_THAT(actual_scores[query_idx],
                Pointwise(FloatNear(3e-4f), expected_values));
  }
}

HWY_NOINLINE void TestComputePanelScoresVariations() {
  TestComputePanelScoresImpl(false);
  TestComputePanelScoresImpl(true);
}

template <size_t kQueriesPerBatch>
HWY_NOINLINE void TestRunPanelLoopImpl(size_t num_panels, size_t panel_start) {
  hn::ScalableTag<int8_t> i8;
  const size_t N = hn::Lanes(i8);
  const size_t tiles_per_vector = 2;
  const size_t panel_bytes = tiles_per_vector * N;

  std::vector<int8_t> dummy_query(tiles_per_vector * 4, 1);
  const int8_t* query_ptrs[kQueriesPerBatch];
  for (size_t query_idx = 0; query_idx < kQueriesPerBatch; ++query_idx) {
    query_ptrs[query_idx] = dummy_query.data();
  }

  std::vector<hwy::AlignedVector<uint8_t>> panels(num_panels);
  for (size_t panel_idx = 0; panel_idx < num_panels; ++panel_idx) {
    panels[panel_idx].resize(panel_bytes, static_cast<uint8_t>(panel_idx + 1));
  }

  std::vector<size_t> visited_panels;
  internal::RunPanelLoop<kQueriesPerBatch>(
      query_ptrs, num_panels, tiles_per_vector, panel_start,
      [&](size_t panel_idx) { return panels[panel_idx].data(); },
      [&](auto /*acc*/, size_t panel_idx)
          HWY_ATTR { visited_panels.push_back(panel_idx); });

  std::vector<size_t> expected_panels(num_panels - panel_start);
  std::iota(expected_panels.begin(), expected_panels.end(), panel_start);
  EXPECT_THAT(visited_panels, ElementsAreArray(expected_panels));
}

HWY_NOINLINE void TestRunPanelLoopVariations() {
  TestRunPanelLoopImpl<4>(1, 0);
  TestRunPanelLoopImpl<4>(2, 0);
  TestRunPanelLoopImpl<4>(4, 0);
  TestRunPanelLoopImpl<4>(7, 0);
  TestRunPanelLoopImpl<4>(5, 2);
}

HWY_NOINLINE void TestRunQueryLoopVsReference() {
  VerifyVsReference(128, 3, 5, false);
  VerifyVsReference(128, 4, 8, false);
  VerifyVsReference(128, 9, 20, false);
  VerifyVsReference(128, 2, 5, true);
  VerifyVsReference(128, 7, 20, true);
}

HWY_NOINLINE void TestRunQueryLoopLowerTriangle() {
  constexpr size_t kQueriesPerBatch = 2;
  hn::ScalableTag<float> f32;
  const size_t NF = hn::Lanes(f32);
  const size_t num_db_points = std::max<size_t>(NF * 3, kQueriesPerBatch * 4);
  const size_t dimensionality = 128;

  GemmTestData data =
      SetupGemmData(dimensionality, num_db_points, num_db_points,
                    /*is_l2=*/false);
  DecodedQueryView decoded_view = data.decoded_view;

  std::vector<std::pair<size_t, size_t>> batch_min_panel;
  size_t current_query_start_idx = 0;

  RunQueryLoop<kQueriesPerBatch>(
      decoded_view, data.num_queries, data.num_panels, data.tiles_per_vector,
      data.database_norm_scaling_factors.data(),
      /*db_sqr_norms=*/nullptr,
      /*lower_triangle=*/true,
      [&](size_t panel_idx) {
        return data.database_panels.data() + panel_idx * data.panel_bytes;
      },
      [&](size_t query_start_idx, size_t num_active_queries,
          const QueryBatch<kQueriesPerBatch>& /*query_batch*/) {
        EXPECT_EQ(
            num_active_queries,
            std::min(kQueriesPerBatch, data.num_queries - query_start_idx));
        current_query_start_idx = query_start_idx;
        batch_min_panel.emplace_back(query_start_idx, data.num_panels);
      },
      [&](auto /*scores_accessor*/, size_t query_start_idx,
          size_t num_active_queries, size_t panel_base_idx) HWY_ATTR {
        EXPECT_EQ(
            num_active_queries,
            std::min(kQueriesPerBatch, data.num_queries - query_start_idx));
        size_t panel_idx = panel_base_idx / NF;
        std::pair<size_t, size_t>& back = batch_min_panel.back();
        back.second = std::min(back.second, panel_idx);
      },
      [&](size_t query_start_idx,
          const QueryBatch<kQueriesPerBatch>& query_batch) {
        EXPECT_EQ(query_start_idx, current_query_start_idx);
        EXPECT_EQ(
            query_batch.num_active_queries,
            std::min(kQueriesPerBatch, data.num_queries - query_start_idx));
      });

  std::vector<std::pair<size_t, size_t>> expected_batches;
  for (size_t query_start_idx = 0; query_start_idx < data.num_queries;
       query_start_idx += kQueriesPerBatch) {
    expected_batches.emplace_back(query_start_idx, query_start_idx / NF);
  }
  EXPECT_THAT(batch_min_panel, ElementsAreArray(expected_batches));

  ASSERT_GT(batch_min_panel.size(), 1u);
  EXPECT_GT(batch_min_panel.back().second, 0u);
}

HWY_NOINLINE void TestRunQueryLoopCallbacks() {
  constexpr size_t kQueriesPerBatch = 2;
  const size_t dimensionality = 128;
  const size_t num_queries = 5;
  const size_t num_db_points = 10;

  GemmTestData data = SetupGemmData(dimensionality, num_queries, num_db_points,
                                    /*is_l2=*/false);
  std::vector<std::pair<size_t, size_t>> initialize_calls;
  std::vector<size_t> reduce_calls;
  size_t current_query_start_idx = 0;

  RunQueryLoop<kQueriesPerBatch>(
      num_queries, /*is_l2=*/false, data.num_panels, data.tiles_per_vector,
      data.database_norm_scaling_factors.data(),
      /*db_sqr_norms=*/nullptr,
      /*lower_triangle=*/false,
      [&](size_t panel_idx) {
        return data.database_panels.data() + panel_idx * data.panel_bytes;
      },
      [&](size_t query_start_idx, size_t num_active_queries,
          QueryBatch<kQueriesPerBatch>& query_batch) {
        current_query_start_idx = query_start_idx;
        initialize_calls.emplace_back(query_start_idx, num_active_queries);
        for (size_t query_idx = 0; query_idx < kQueriesPerBatch; ++query_idx) {
          size_t source_idx =
              std::min(query_start_idx + query_idx, num_queries - 1);
          query_batch.query_ptrs[query_idx] =
              data.query_decoded_data.data() +
              source_idx * data.decoded_dimension;
          query_batch.norm_scaling_factors[query_idx] =
              (query_idx < num_active_queries)
                  ? data.query_norms[query_start_idx + query_idx]
                  : 0.0f;
          query_batch.squared_norms[query_idx] = 0.0f;
          query_batch.byte_sums[query_idx] =
              (query_idx < num_active_queries)
                  ? data.query_byte_sums[query_start_idx + query_idx]
                  : 0;
        }
        query_batch.num_active_queries = num_active_queries;
      },
      [&](auto /*scores_accessor*/, size_t query_start_idx,
          size_t num_active_queries, size_t panel_base_idx) HWY_ATTR {
        EXPECT_EQ(num_active_queries,
                  std::min(kQueriesPerBatch, num_queries - query_start_idx));
        EXPECT_EQ(panel_base_idx % hn::Lanes(hn::ScalableTag<float>()), 0u);
      },
      [&](size_t query_start_idx,
          const QueryBatch<kQueriesPerBatch>& query_batch) {
        EXPECT_EQ(query_start_idx, current_query_start_idx);
        EXPECT_EQ(query_batch.num_active_queries,
                  std::min(kQueriesPerBatch, num_queries - query_start_idx));
        reduce_calls.push_back(query_start_idx);
      });

  EXPECT_THAT(initialize_calls,
              ElementsAre(Pair(0u, 2u), Pair(2u, 2u), Pair(4u, 1u)));
  EXPECT_THAT(reduce_calls, ElementsAre(0u, 2u, 4u));
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {
namespace {
HWY_BEFORE_TEST(GemmLoopTest);
HWY_EXPORT_AND_TEST_P(GemmLoopTest, TestPrepareQueryBatchFull);
HWY_EXPORT_AND_TEST_P(GemmLoopTest, TestPrepareQueryBatchPartial);
HWY_EXPORT_AND_TEST_P(GemmLoopTest, TestComputePanelScoresVariations);
HWY_EXPORT_AND_TEST_P(GemmLoopTest, TestRunPanelLoopVariations);
HWY_EXPORT_AND_TEST_P(GemmLoopTest, TestRunQueryLoopVsReference);
HWY_EXPORT_AND_TEST_P(GemmLoopTest, TestRunQueryLoopLowerTriangle);
HWY_EXPORT_AND_TEST_P(GemmLoopTest, TestRunQueryLoopCallbacks);
HWY_AFTER_TEST();
}  // namespace
}  // namespace ginger
HWY_TEST_MAIN();
#endif
