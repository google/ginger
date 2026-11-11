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

#include "ginger/turboquant/chamfer/chamfer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ginger/turboquant/chamfer/point_cloud_query.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::ginger::IndexSubset;
using ::ginger::TurboQuantCodec;
using ::ginger::TurboQuantDatasetSoA;
using ::ginger::TurboQuantQuery;

std::vector<float> GenerateEmbeddings(size_t count, size_t dim,
                                      float scale = 1.0f, int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> data(count * dim);
  for (float& val : data) val = dist(gen);
  return data;
}

TurboQuantDatasetSoA Quantize(const std::vector<float>& data, size_t dim,
                              const TurboQuantCodec& tq) {
  if (data.empty()) {
    auto empty_ds = TurboQuantDatasetSoA::Create(tq, 0);
    ABSL_CHECK_OK(empty_ds.status());
    return *std::move(empty_ds);
  }
  std::vector<absl::Span<const float>> spans;
  for (size_t i = 0; i < data.size() / dim; ++i) {
    spans.push_back(absl::MakeConstSpan(data).subspan(i * dim, dim));
  }
  auto ds = TurboQuantDatasetSoA::Create(tq, spans);
  ABSL_CHECK_OK(ds.status());
  return *std::move(ds);
}

std::vector<float> ComputeReferenceMatrix(
    const TurboQuantDatasetSoA& queries_tq, const TurboQuantDatasetSoA& docs_tq,
    const TurboQuantCodec& tq, absl::Span<const int> qi = {},
    absl::Span<const int> di = {}) {
  size_t nq = qi.empty() ? queries_tq.num_datapoints() : qi.size();
  size_t nd = di.empty() ? docs_tq.num_datapoints() : di.size();
  std::vector<int32_t> eff_di(nd);
  if (di.empty()) {
    std::iota(eff_di.begin(), eff_di.end(), 0);
  } else {
    for (size_t i = 0; i < nd; ++i) eff_di[i] = di[i];
  }

  std::vector<float> ref(nq * nd);
  for (size_t q = 0; q < nq; ++q) {
    auto query = TurboQuantQuery::CreateFromQuantizedDataPoint(
        tq, queries_tq.DataPointView(qi.empty() ? q : qi[q]));
    ABSL_CHECK_OK(query.status());
    ABSL_CHECK_OK(
        docs_tq.ComputeDistances(*query, absl::MakeSpan(eff_di),
                                 absl::MakeSpan(ref).subspan(q * nd, nd)));
  }
  return ref;
}

class ChamferSimilarityTest : public ::testing::Test {
 protected:
  void VerifyChamfer(int nq, int nd, int dim, int seed,
                     absl::Span<const int> qi = {}, float scale = 1.0f,
                     float tol = 1e-3f) {
    TurboQuantCodec tq(dim, seed);
    auto q_tq = Quantize(GenerateEmbeddings(nq, dim, scale, 1), dim, tq);
    auto d_tq = Quantize(GenerateEmbeddings(nd, dim, scale, 2), dim, tq);

    ASSERT_OK_AND_ASSIGN(auto query_pc,
                         TurboQuantPointCloudQuery::CreateFromQuantizedDataset(
                             q_tq, qi.empty() ? IndexSubset::Contiguous(0, nq)
                                              : IndexSubset::Gathered(qi)));

    std::vector<DistanceAndId> neighbors(1);
    neighbors[0].id = 0;
    TurboQuantChamferDataset dataset(d_tq, {0, static_cast<size_t>(nd)});
    dataset.ChamferDistance(query_pc, absl::MakeSpan(neighbors));

    std::vector<float> ref = ComputeReferenceMatrix(q_tq, d_tq, tq, qi);
    size_t aq = qi.empty() ? nq : qi.size();
    float expected_sum = 0.0f;
    for (size_t q = 0; q < aq; ++q) {
      float min_dist = ref[q * nd];
      for (size_t d = 1; d < nd; ++d) {
        min_dist = std::min(min_dist, ref[q * nd + d]);
      }
      expected_sum += min_dist;
    }
    float expected = aq == 0 ? 0.0f : expected_sum / static_cast<float>(aq);
    EXPECT_NEAR(neighbors[0].distance, expected, tol);
  }
};

TEST_F(ChamferSimilarityTest, ConformanceStandardConfigurations) {
  const std::vector<std::pair<int, int>> batch_configs = {
      {1, 1}, {4, 16}, {8, 32}, {16, 64}, {32, 128}};
  for (int dim : {128, 256, 512}) {
    for (const auto& [num_queries, num_docs] : batch_configs) {
      SCOPED_TRACE(
          absl::StrCat("Dim: ", dim, ", Shape: ", num_queries, "x", num_docs));
      VerifyChamfer(num_queries, num_docs, dim, 2001 + dim + num_queries);
    }
  }
}

TEST_F(ChamferSimilarityTest, NonPowerOfTwoEmbeddings) {
  for (int dim : {32, 48, 96, 160}) {
    SCOPED_TRACE(absl::StrCat("Dim: ", dim));
    VerifyChamfer(9, 27, dim, 2002);
  }
}

TEST_F(ChamferSimilarityTest, IndexedSubsetsSparseSelection) {
  VerifyChamfer(32, 128, 256, 2003, {1, 3, 7, 15, 31});
}

TEST_F(ChamferSimilarityTest, IndexedSubsetsInvertedOrdering) {
  std::vector<int> q_reverse(8);
  for (int i = 0; i < 8; ++i) q_reverse[i] = 7 - i;
  VerifyChamfer(8, 32, 128, 2004, q_reverse);
}

TEST_F(ChamferSimilarityTest, ExtremeCoordinateMagnitudes) {
  VerifyChamfer(8, 32, 128, 2005, {}, 100.0f, 1e-1f);
}

TEST_F(ChamferSimilarityTest, AsymmetricAndNonStandardShapes) {
  const std::vector<std::pair<int, int>> shapes = {
      {1, 1000}, {1000, 1}, {17, 65}, {15, 63}, {31, 127}, {7, 23}};
  for (const auto& [num_queries, num_docs] : shapes) {
    SCOPED_TRACE(absl::StrCat("Shape: ", num_queries, "x", num_docs));
    VerifyChamfer(num_queries, num_docs, 128, 2006 + num_queries + num_docs);
  }
}

TEST_F(ChamferSimilarityTest, AsymmetricBatchManyQueries) {
  VerifyChamfer(6, 4, 256, 3001, {}, 1.0f, 1e-3f);
}

TEST_F(ChamferSimilarityTest, AsymmetricBatchManyDocs) {
  VerifyChamfer(4, 10, 256, 3002, {}, 1.0f, 1e-3f);
}

TEST_F(ChamferSimilarityTest, HighDimensionalityVectors) {
  VerifyChamfer(6, 10, 512, 3003, {}, 1.0f, 1e-3f);
}

TEST_F(ChamferSimilarityTest, IndexedLargeDataset) {
  std::vector<int> qi = {0, 1, 2, 3, 4, 5, 6, 7};
  VerifyChamfer(16, 64, 128, 3004, qi, 1.0f, 1e-3f);
}

TEST_F(ChamferSimilarityTest, IndexedSubsetsLargeSelection) {
  std::vector<int> query_subset;
  for (int i = 0; i < 30; ++i) query_subset.push_back(i * 2);
  VerifyChamfer(64, 256, 128, 2007, query_subset, 1.0f, 1e-3f);
}

TEST_F(ChamferSimilarityTest, IntervalSubsets) {
  std::vector<int> query_indices;
  for (int i = 0; i < 5; ++i) query_indices.push_back(i);
  for (int i = 10; i < 20; ++i) query_indices.push_back(i);
  VerifyChamfer(64, 256, 128, 2008, query_indices, 1.0f, 1e-3f);
}

TEST_F(ChamferSimilarityTest, EmptyInputs) {
  const int dim = 128;
  TurboQuantCodec tq(dim, /*seed=*/4001);
  auto q_tq = Quantize(GenerateEmbeddings(4, dim, 1.0f, 1), dim, tq);
  auto d_tq = Quantize(GenerateEmbeddings(16, dim, 1.0f, 2), dim, tq);
  auto empty_data = Quantize({}, dim, tq);

  ASSERT_OK_AND_ASSIGN(
      auto valid_query,
      TurboQuantPointCloudQuery::CreateFromQuantizedDataset(
          q_tq, IndexSubset::Contiguous(0, q_tq.num_datapoints())));
  ASSERT_OK_AND_ASSIGN(auto empty_query,
                       TurboQuantPointCloudQuery::CreateFromQuantizedDataset(
                           q_tq, IndexSubset::Contiguous(0, 0)));

  std::vector<DistanceAndId> neighbors(1);
  neighbors[0].id = 0;

  TurboQuantChamferDataset valid_dataset(d_tq, {0, 16});
  valid_dataset.ChamferDistance(empty_query, absl::MakeSpan(neighbors));
  EXPECT_EQ(neighbors[0].distance, 0.0f);

  valid_dataset.ChamferDistance(valid_query, {});

  TurboQuantChamferDataset empty_doc_dataset(empty_data, {0, 0});
  empty_doc_dataset.ChamferDistance(valid_query, absl::MakeSpan(neighbors));
  EXPECT_EQ(neighbors[0].distance, 0.0f);
}

TEST_F(ChamferSimilarityTest, OneToManyBatchConformance) {
  const int dim = 128;
  const int nq = 8;
  TurboQuantCodec tq(dim, 2026);
  auto q_tq = Quantize(GenerateEmbeddings(nq, dim, 1.0f, 10), dim, tq);
  ASSERT_OK_AND_ASSIGN(auto query_pc,
                       TurboQuantPointCloudQuery::CreateFromQuantizedDataset(
                           q_tq, IndexSubset::Contiguous(0, nq)));

  // Generate 5 documents with varying token lengths (4, 15, 32, 7, 23).
  const std::vector<int> doc_lengths = {4, 15, 32, 7, 23};
  const int total_tokens =
      std::accumulate(doc_lengths.begin(), doc_lengths.end(), 0);
  auto d_tq =
      Quantize(GenerateEmbeddings(total_tokens, dim, 1.0f, 20), dim, tq);

  std::vector<size_t> doc_offsets;
  doc_offsets.push_back(0);
  for (int len : doc_lengths) {
    doc_offsets.push_back(doc_offsets.back() + len);
  }

  std::vector<DistanceAndId> neighbors = {{0.0f, 4}, {0.0f, 2}, {0.0f, 0}};

  TurboQuantChamferDataset tq_dataset(d_tq, doc_offsets);
  tq_dataset.ChamferDistance(query_pc, absl::MakeSpan(neighbors));

  const float inv_nq = 1.0f / static_cast<float>(nq);
  for (size_t i = 0; i < neighbors.size(); ++i) {
    const GingerNodeId doc_id = neighbors[i].id;
    const size_t start = doc_offsets[doc_id];
    const size_t len = doc_offsets[doc_id + 1] - start;
    EXPECT_EQ(tq_dataset.num_tokens(doc_id), len);

    std::vector<int> di(len);
    std::iota(di.begin(), di.end(), start);
    std::vector<float> ref = ComputeReferenceMatrix(q_tq, d_tq, tq, {}, di);
    float expected_sum = 0.0f;
    for (size_t q = 0; q < nq; ++q) {
      float min_dist = ref[q * len];
      for (size_t d = 1; d < len; ++d) {
        min_dist = std::min(min_dist, ref[q * len + d]);
      }
      expected_sum += min_dist;
    }
    EXPECT_NEAR(neighbors[i].distance, expected_sum * inv_nq, 1e-4f);
  }
}

TEST_F(ChamferSimilarityTest, ManyNeighborsBatchingAcrossChunks) {
  const int dim = 128;
  const int nq = 4;
  const size_t num_docs = 600;  // Spans multiple 256-sized batches.
  TurboQuantCodec tq(dim, 3030);
  auto q_tq = Quantize(GenerateEmbeddings(nq, dim, 1.0f, 11), dim, tq);
  ASSERT_OK_AND_ASSIGN(auto query_pc,
                       TurboQuantPointCloudQuery::CreateFromQuantizedDataset(
                           q_tq, IndexSubset::Contiguous(0, nq)));

  // Each doc has 5 tokens.
  const size_t tokens_per_doc = 5;
  auto d_tq = Quantize(
      GenerateEmbeddings(num_docs * tokens_per_doc, dim, 1.0f, 22), dim, tq);

  std::vector<size_t> doc_offsets;
  doc_offsets.reserve(num_docs + 1);
  for (size_t i = 0; i <= num_docs; ++i) {
    doc_offsets.push_back(i * tokens_per_doc);
  }

  TurboQuantChamferDataset tq_dataset(d_tq, doc_offsets);

  // All 600 docs in one large neighbors call.
  std::vector<DistanceAndId> batched_neighbors(num_docs);
  for (size_t i = 0; i < num_docs; ++i) {
    batched_neighbors[i] = {0.0f, static_cast<GingerNodeId>(i)};
  }
  tq_dataset.ChamferDistance(query_pc, absl::MakeSpan(batched_neighbors));

  // Compare against individual single-neighbor calls.
  for (size_t i = 0; i < num_docs; i += 37) {
    std::vector<DistanceAndId> single_neighbor = {
        {0.0f, static_cast<GingerNodeId>(i)}};
    tq_dataset.ChamferDistance(query_pc, absl::MakeSpan(single_neighbor));
    EXPECT_FLOAT_EQ(batched_neighbors[i].distance, single_neighbor[0].distance);
  }
}

}  // namespace
}  // namespace ginger
