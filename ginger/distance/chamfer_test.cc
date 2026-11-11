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

#include "ginger/distance/chamfer.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::testing::IsEmpty;

float ReferenceChamfer(absl::Span<const float> query,
                       absl::Span<const float> doc, size_t dim) {
  if (query.empty() || doc.empty() || dim == 0) return 0.0f;
  const size_t nq = query.size() / dim, nd = doc.size() / dim;
  float sum_max = 0.0f;
  for (size_t i = 0; i < nq; ++i) {
    float max_dot = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < nd; ++j) {
      float dot = 0.0f;
      for (size_t k = 0; k < dim; ++k) {
        dot += query[i * dim + k] * doc[j * dim + k];
      }
      max_dot = std::max(max_dot, dot);
    }
    sum_max += max_dot;
  }
  return -sum_max / static_cast<float>(nq);
}

std::vector<float> GenFloats(size_t count, int mod = 7, float scale = 0.2f) {
  std::vector<float> v(count);
  for (size_t i = 0; i < count; ++i) {
    v[i] = ((static_cast<int>(i) % mod) - mod / 2) * scale;
  }
  return v;
}

std::vector<std::vector<float>> MakeDocs(size_t dim,
                                         std::vector<size_t> token_counts) {
  std::vector<std::vector<float>> docs;
  docs.reserve(token_counts.size());
  for (size_t i = 0; i < token_counts.size(); ++i) {
    docs.push_back(GenFloats(token_counts[i] * dim, 7 + (i % 5)));
  }
  return docs;
}

ChamferDataset MakeDataset(size_t dim,
                           absl::Span<const std::vector<float>> docs) {
  std::vector<size_t> offsets = {0};
  for (const auto& d : docs) offsets.push_back(offsets.back() + d.size() / dim);
  ChamferDataset dataset(dim, offsets);
  for (GingerNodeId i = 0; i < docs.size(); ++i) dataset.SetDoc(i, docs[i]);
  return dataset;
}

std::vector<DistanceAndId> MakeNeighbors(size_t count) {
  std::vector<DistanceAndId> nbrs(count);
  for (GingerNodeId i = 0; i < count; ++i) nbrs[i].id = i;
  return nbrs;
}

std::vector<DistanceAndId> MakeNeighbors(
    std::initializer_list<GingerNodeId> ids) {
  std::vector<DistanceAndId> nbrs;
  nbrs.reserve(ids.size());
  for (GingerNodeId id : ids) nbrs.push_back({.id = id});
  return nbrs;
}

std::vector<DistanceAndId> MakeNeighbors(absl::Span<const GingerNodeId> ids) {
  std::vector<DistanceAndId> nbrs;
  nbrs.reserve(ids.size());
  for (GingerNodeId id : ids) nbrs.push_back({.id = id});
  return nbrs;
}

void CheckChamfer(const ChamferDataset& dataset, absl::Span<const float> query,
                  absl::Span<const std::vector<float>> docs,
                  absl::Span<DistanceAndId> neighbors) {
  dataset.ChamferDistance(query, neighbors);
  for (const auto& nbr : neighbors) {
    EXPECT_NEAR(nbr.distance,
                ReferenceChamfer(query, docs[nbr.id], dataset.dimensionality()),
                1e-4f)
        << "doc_id=" << nbr.id;
  }
}

void CheckChamfer(const ChamferDataset& dataset, absl::Span<const float> query,
                  absl::Span<const std::vector<float>> docs) {
  auto neighbors = MakeNeighbors(docs.size());
  CheckChamfer(dataset, query, docs, absl::MakeSpan(neighbors));
}

void CheckChamfer(size_t dim, absl::Span<const float> query,
                  absl::Span<const std::vector<float>> docs) {
  CheckChamfer(MakeDataset(dim, docs), query, docs);
}

TEST(ChamferDistanceTest, BasicAndNeighbors) {
  const std::vector<std::vector<float>> docs = {
      {1, 0, 0, 0, 0, 1, 0, 0}, {0, 1, 0, 0}, {0.6f, 0.8f, 0, 0, 1, 0, 0, 0}};
  const std::vector<float> query = {1, 0, 0, 0, 0, 1, 0, 0};
  auto dataset = MakeDataset(4, docs);
  CheckChamfer(dataset, query, docs);
  auto neighbors = MakeNeighbors({2, 0});
  CheckChamfer(dataset, query, docs, absl::MakeSpan(neighbors));
}

TEST(ChamferDistanceTest, EmptyInputs) {
  const std::vector<std::vector<float>> docs = {{1, 0, 0, 0}, {}};
  auto dataset = MakeDataset(4, docs);
  std::vector<DistanceAndId> neighbors;
  dataset.ChamferDistance({1, 0, 0, 0}, absl::MakeSpan(neighbors));
  EXPECT_THAT(neighbors, IsEmpty());
  CheckChamfer(dataset, {}, docs);
  CheckChamfer(dataset, std::vector<float>{1, 0, 0, 0}, docs);
}

TEST(ChamferDistanceTest, ConstructFromSpanDataset) {
  const std::vector<std::vector<float>> docs = {
      {1, 0, 0, 0}, {0, 1, 0, 0}, {0.6f, 0.8f, 0, 0}};
  SpanDataset tokens(3, 4);
  for (GingerNodeId i = 0; i < 3; ++i) tokens.Set<float>(i, docs[i]);
  ChamferDataset dataset(std::move(tokens), {0, 1, 2, 3});
  CheckChamfer(dataset, std::vector<float>{1, 0, 0, 0}, docs);
}

TEST(ChamferDistanceTest, RejectNonFloatSpanDataset) {
  SpanDataset tokens(3, 4, CoordinateType::kInt8);
  EXPECT_DEATH(ChamferDataset(std::move(tokens), {0, 1, 2, 3}), "");
}

TEST(ChamferDistanceTest, VariedDimensionsAndTokenCounts) {
  struct Shape {
    size_t nq, nd;
  };
  constexpr Shape kShapes[] = {{1, 1},  {1, 2},  {2, 4},  {3, 8},
                               {4, 15}, {7, 16}, {8, 17}, {4, 32}};
  for (int dim : {1, 4, 7, 12, 13, 16, 24, 25, 35, 48, 55, 64, 100, 256}) {
    for (auto [nq, nd] : kShapes) {
      auto docs = MakeDocs(dim, {nd});
      CheckChamfer(dim, GenFloats(nq * dim, 5), docs);
    }
  }
}

TEST(ChamferDistanceTest, MixedSizesAndCrossDocBatching) {
  const std::vector<size_t> doc_sizes = {1,  3, 16, 2,  17, 5,
                                         32, 4, 1,  48, 15, 8};
  for (int dim : {4, 8, 16, 55, 64, 128}) {
    auto docs = MakeDocs(dim, doc_sizes);
    auto dataset = MakeDataset(dim, docs);
    for (size_t nq : {1, 2, 3, 4, 8, 16}) {
      CheckChamfer(dataset, GenFloats(nq * dim, 3), docs);
    }
  }
}

TEST(ChamferDistanceTest, UpdateDocWithSetDoc) {
  const size_t dim = 8;
  const std::vector<size_t> sizes = {5, 16, 20, 11, 25};
  auto docs = MakeDocs(dim, sizes);
  auto dataset = MakeDataset(dim, docs);
  for (GingerNodeId id = 0; id < docs.size(); ++id) {
    docs[id] = GenFloats(sizes[id] * dim, 11, 0.15f * (id + 1));
    dataset.SetDoc(id, docs[id]);
  }
  CheckChamfer(dataset, GenFloats(4 * dim, 5), docs);
}

TEST(ChamferDistanceTest, RowMajorLastTokenBestMatch) {
  for (int dim : {4, 8, 16, 32, 64}) {
    for (size_t nd : {1, 3, 5, 7, 9, 11, 15, 25}) {
      std::vector<float> doc(nd * dim, -1.0f);
      // Make the final token the undisputed best match with high positive
      // values.
      for (size_t k = 0; k < dim; ++k) {
        doc[(nd - 1) * dim + k] = 10.0f;
      }
      const std::vector<std::vector<float>> docs = {doc};
      // Score against 4 queries to trigger ScoreTileRowMajor4.
      const size_t nq = 4;
      const std::vector<float> query(nq * dim, 1.0f);
      CheckChamfer(dim, query, docs);
    }
  }
}

TEST(ChamferDistanceTest, SetDocDoesNotOverrunNextDoc) {
  for (int dim : {4, 8, 16, 32}) {
    for (size_t size_0 : {8, 9, 15, 16, 24, 25}) {
      const size_t size_1 = 4;
      std::vector<float> doc_0_buf((size_0 + 2) * dim, -100.0f);
      std::vector<float> doc_0(doc_0_buf.begin(),
                               doc_0_buf.begin() + size_0 * dim);
      std::vector<float> doc_1(size_1 * dim, -100.0f);
      // Make only the first token of doc_1 have high positive values.
      for (size_t k = 0; k < dim; ++k) {
        doc_1[k] = 100.0f;
      }
      const std::vector<std::vector<float>> docs = {doc_0, doc_1};
      std::vector<size_t> offsets = {0, size_0, size_0 + size_1};
      ChamferDataset dataset(dim, offsets);

      // Populate doc_1 first, then doc_0 using a span backed by doc_0_buf so
      // any out-of-bounds remainder read/write in SetDoc(0, ...) predictably
      // copies negative values over doc_1's first token.
      dataset.SetDoc(1, doc_1);
      dataset.SetDoc(0, absl::MakeConstSpan(doc_0_buf).first(size_0 * dim));

      const std::vector<float> query(dim, 1.0f);
      CheckChamfer(dataset, query, docs);
    }
  }
}

TEST(ChamferDistanceTest, SevenWayConcurrentOccupancy) {
  // Configures document token counts such that exactly 3 documents remain
  // pending in pending_16 (e.g., sizes 16, 17, 18) and 3 documents remain
  // pending in pending_8 (e.g., sizes 8, 9, 10) simultaneously. Starting the
  // 7th document requires AllocSlot to pop free_stack[0] (slot index 6), which
  // fails if slot indices are mutated out of bounds.
  const std::vector<std::vector<size_t>> test_cases = {
      {16, 16, 16, 8, 8, 8, 8},
      {8, 8, 8, 16, 16, 16, 16},
      {17, 18, 19, 9, 10, 11, 24},
  };
  for (int dim : {4, 8, 16, 32}) {
    for (const auto& sizes : test_cases) {
      auto docs = MakeDocs(dim, sizes);
      auto dataset = MakeDataset(dim, docs);
      for (size_t nq : {1, 2, 4}) {
        CheckChamfer(dataset, GenFloats(nq * dim, 5), docs);
      }
    }
  }
}

TEST(ChamferDistanceTest, MaxQueriesSupported) {
  constexpr size_t kMaxQueries = 16;
  const std::vector<size_t> doc_sizes = {1, 7, 8, 15, 16, 23, 24, 31, 32, 35};
  for (int dim : {4, 8, 16, 32, 64}) {
    auto docs = MakeDocs(dim, doc_sizes);
    auto dataset = MakeDataset(dim, docs);
    CheckChamfer(dataset, GenFloats(kMaxQueries * dim, 11), docs);
  }
}

TEST(ChamferDistanceTest, LargeQueryCountSupported) {
  const std::vector<size_t> doc_sizes = {1, 7, 8, 15, 16, 23, 24, 31, 32, 35};
  for (int dim : {4, 8, 16, 32}) {
    auto docs = MakeDocs(dim, doc_sizes);
    auto dataset = MakeDataset(dim, docs);
    for (size_t nq : {17, 24, 32, 35, 64}) {
      CheckChamfer(dataset, GenFloats(nq * dim, 11), docs);
    }
  }
}

TEST(ChamferDistanceTest, DocumentTokenCountSweep) {
  // Verifies all transition boundaries:
  // - row-major: nd in [1, 7]
  // - micro-block: nd in [8, 15]
  // - full block: nd in [16, 23]
  // - full + micro-block: nd in [24, 31]
  // - full + full block: nd >= 32
  std::vector<size_t> all_sizes;
  all_sizes.reserve(35);
  for (size_t nd = 1; nd <= 35; ++nd) {
    all_sizes.push_back(nd);
  }

  for (int dim : {4, 8, 13, 16, 32}) {
    // 1. All sizes in one dataset to exercise cross-document batching and queue
    // transitions.
    auto docs = MakeDocs(dim, all_sizes);
    auto dataset = MakeDataset(dim, docs);
    for (size_t nq : {1, 4, 8, 16}) {
      CheckChamfer(dataset, GenFloats(nq * dim, 7), docs);
    }

    // 2. Each nd individually to verify isolated document scoring and flushing.
    for (size_t nd = 1; nd <= 35; ++nd) {
      auto single_doc = MakeDocs(dim, {nd});
      CheckChamfer(dim, GenFloats(4 * dim, 7), single_doc);
    }
  }
}

TEST(ChamferDistanceTest, NegativeDotProducts) {
  // Query all +1, doc all -1.
  // Dot products are all -dim, max dot product is -dim, and similarity is -dim.
  // Negation yields distance = +dim.
  // Verifies that max-dot tracking initializes to -infinity rather than 0.0f
  // (which would incorrectly produce a distance of 0.0f), and verifies sign
  // handling across all block structures.
  for (int dim : {4, 8, 16, 32}) {
    const std::vector<size_t> sizes = {1, 5, 8, 12, 16, 20, 24, 30, 32, 35};
    std::vector<std::vector<float>> docs;
    for (size_t nd : sizes) {
      docs.push_back(std::vector<float>(nd * dim, -1.0f));
    }
    auto dataset = MakeDataset(dim, docs);

    for (size_t nq : {1, 2, 4, 8, 16}) {
      const std::vector<float> query(nq * dim, 1.0f);
      auto neighbors = MakeNeighbors(docs.size());
      dataset.ChamferDistance(query, absl::MakeSpan(neighbors));
      for (const auto& nbr : neighbors) {
        EXPECT_NEAR(nbr.distance, static_cast<float>(dim), 1e-4f)
            << "dim=" << dim << " nq=" << nq << " doc_id=" << nbr.id;
      }
    }
  }

  // Verify that max selection chooses the least negative dot product.
  for (int dim : {4, 8, 16, 32}) {
    // 3 doc tokens with values -3, -1, -2. With query all +1, dot products
    // are -3*dim, -1*dim, -2*dim. Maximum is -1*dim, yielding distance +dim.
    std::vector<float> doc(3 * dim);
    for (size_t k = 0; k < dim; ++k) {
      doc[0 * dim + k] = -3.0f;
      doc[1 * dim + k] = -1.0f;
      doc[2 * dim + k] = -2.0f;
    }
    const std::vector<std::vector<float>> docs = {doc};
    const std::vector<float> query(2 * dim, 1.0f);
    auto dataset = MakeDataset(dim, docs);
    auto neighbors = MakeNeighbors(1);
    dataset.ChamferDistance(query, absl::MakeSpan(neighbors));
    EXPECT_NEAR(neighbors[0].distance, static_cast<float>(dim), 1e-4f);
  }
}

TEST(ChamferDistanceTest, DuplicateNeighborIds) {
  for (int dim : {4, 8, 16, 32}) {
    // Documents covering various block transitions:
    // small (<8), micro-block (8-15), full block (16-23),
    // full+micro (24-31), full+full (>=32).
    const std::vector<size_t> sizes = {3, 10, 16, 25, 33};
    auto docs = MakeDocs(dim, sizes);
    auto dataset = MakeDataset(dim, docs);

    // 1. Interleaved duplicate IDs.
    const std::vector<GingerNodeId> interleaved_ids = {0, 1, 0, 2, 1, 3,
                                                       4, 2, 0, 4, 3, 1};
    for (size_t nq : {1, 4, 8, 16}) {
      auto nbrs = MakeNeighbors(interleaved_ids);
      CheckChamfer(dataset, GenFloats(nq * dim, 3), docs, absl::MakeSpan(nbrs));
    }

    // 2. Consecutive duplicate IDs exceeding kMaxInFlight (7) to test slot
    // recycling and concurrent tracking for identical document IDs.
    std::vector<GingerNodeId> consecutive_ids;
    for (GingerNodeId id = 0; id < static_cast<GingerNodeId>(docs.size());
         ++id) {
      for (int rep = 0; rep < 10; ++rep) {
        consecutive_ids.push_back(id);
      }
    }
    for (size_t nq : {1, 4, 8, 16}) {
      auto nbrs = MakeNeighbors(consecutive_ids);
      CheckChamfer(dataset, GenFloats(nq * dim, 5), docs, absl::MakeSpan(nbrs));
      for (size_t i = 1; i < nbrs.size(); ++i) {
        if (nbrs[i].id == nbrs[i - 1].id) {
          EXPECT_FLOAT_EQ(nbrs[i].distance, nbrs[i - 1].distance)
              << "Mismatch between index " << i << " and " << i - 1
              << " for doc id " << nbrs[i].id;
        }
      }
    }
  }
}

}  // namespace
}  // namespace ginger
