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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/distance/chamfer.h"
#include "ginger/turboquant/chamfer/chamfer.h"
#include "ginger/turboquant/chamfer/point_cloud_query.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"

namespace ginger {
namespace {

template <typename DocAccessor>
void ChamferDistanceEigen(absl::Span<const float> query,
                          absl::Span<DistanceAndId> neighbors,
                          DocAccessor&& get_doc, size_t dim) {
  using MatrixRowMajor =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  if (query.empty() || dim == 0) return;
  const size_t nq = query.size() / dim;
  Eigen::Map<const MatrixRowMajor> Q(query.data(), nq, dim);

  for (auto& [distance, neighbor_id] : neighbors) {
    absl::Span<const float> candidate = get_doc(neighbor_id);
    if (candidate.empty()) {
      distance = 0.0f;
      continue;
    }
    const size_t nd = candidate.size() / dim;
    Eigen::Map<const MatrixRowMajor> D(candidate.data(), nd, dim);
    Eigen::MatrixXf all_similarities = Q * D.transpose();
    distance = -all_similarities.rowwise().maxCoeff().mean();
  }
}

struct BenchmarkData {
  static constexpr size_t kNumQueries = 20;
  static constexpr size_t kTargetPoolTokens = 2100000;
  static constexpr size_t kNumMixedDocs = 100002;

  ChamferDataset dataset;
  std::vector<std::vector<float>> queries;
  std::vector<std::vector<DistanceAndId>> neighbors_storage;
  std::vector<float> sample_doc;
  size_t total_doc_tokens_in_neighbors = 0;

  // Passing nd == 0 creates mixed document token counts in range [4, 32].
  BenchmarkData(size_t dim, size_t nq, size_t nd, size_t num_neighbors)
      : dataset(dim, MakeOffsets(nd)),
        queries(kNumQueries, std::vector<float>(nq * dim, 0.0f)),
        neighbors_storage(kNumQueries,
                          std::vector<DistanceAndId>(num_neighbors)),
        sample_doc((nd == 0 ? 32 : nd) * dim) {
    const size_t num_pool = dataset.size();
    absl::BitGen rng;
    for (auto& query : queries) {
      for (size_t i = 0; i < query.size(); ++i) {
        query[i] = absl::Uniform<float>(rng, -1.0f, 1.0f);
      }
    }
    for (size_t i = 0; i < sample_doc.size(); ++i) {
      sample_doc[i] = (static_cast<int>(i % 251) - 125) * 0.008f;
    }
    for (size_t id = 0; id < num_pool; ++id) {
      dataset.SetDoc(id, GetDoc(id));
    }
    for (size_t q = 0; q < kNumQueries; ++q) {
      for (size_t i = 0; i < num_neighbors; ++i) {
        GingerNodeId id =
            static_cast<GingerNodeId>(absl::Uniform<size_t>(rng, 0, num_pool));
        neighbors_storage[q][i].id = id;
        total_doc_tokens_in_neighbors += dataset.num_tokens(id);
      }
    }
  }

  absl::Span<const float> GetDoc(GingerNodeId id) const {
    return absl::MakeConstSpan(sample_doc)
        .first(dataset.num_tokens(id) * dataset.dimensionality());
  }

  void SetCounters(benchmark::State& state, size_t nq, size_t dim) const {
    const size_t total_tokens = kNumQueries * neighbors_storage[0].size() * nq +
                                total_doc_tokens_in_neighbors;
    const size_t bytes = total_tokens * dim * sizeof(float);
    const size_t flops = 2 * nq * total_doc_tokens_in_neighbors * dim;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            total_tokens);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes);
    state.SetLabel(absl::StrCat(flops, " FLOPS"));
  }

 private:
  static std::vector<size_t> MakeOffsets(size_t nd) {
    if (nd == 0) {
      // Document token counts evenly covering all integers in range [4, 32].
      std::vector<size_t> offsets(kNumMixedDocs + 1, 0);
      for (size_t i = 0; i < kNumMixedDocs; ++i) {
        offsets[i + 1] = offsets[i] + (4 + (i % 29));
      }
      return offsets;
    }
    const size_t num_pool = std::max<size_t>(500, kTargetPoolTokens / nd);
    std::vector<size_t> offsets(num_pool + 1);
    for (size_t i = 0; i <= num_pool; ++i) {
      offsets[i] = i * nd;
    }
    return offsets;
  }
};

struct TurboQuantBenchmarkData : public BenchmarkData {
  TurboQuantCodec tq;
  std::vector<TurboQuantPointCloudQuery> queries_pc;
  TurboQuantChamferDataset tq_dataset;

  TurboQuantBenchmarkData(size_t dim, size_t nq, size_t nd,
                          size_t num_neighbors)
      : BenchmarkData(dim, nq, nd, num_neighbors), tq(dim, 42) {
    for (const auto& query : queries) {
      auto query_pc = TurboQuantPointCloudQuery::Create(tq, query);
      ABSL_CHECK_OK(query_pc);
      queries_pc.push_back(std::move(*query_pc));
    }

    std::vector<absl::Span<const float>> d_spans;
    d_spans.reserve(dataset.doc_offsets().back());
    for (size_t id = 0; id < dataset.size(); ++id) {
      absl::Span<const float> doc = GetDoc(id);
      const size_t n_tokens = doc.size() / dim;
      for (size_t t = 0; t < n_tokens; ++t) {
        d_spans.push_back(absl::MakeConstSpan(doc).subspan(t * dim, dim));
      }
    }
    auto tq_flattened = TurboQuantDatasetSoA::Create(tq, d_spans);
    ABSL_CHECK_OK(tq_flattened);
    tq_dataset = TurboQuantChamferDataset(
        *tq_flattened, std::vector<size_t>(dataset.doc_offsets().begin(),
                                           dataset.doc_offsets().end()));
  }
};

void BM_ChamferEigen(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t nq = state.range(1);
  const size_t nd = state.range(2);
  const size_t num_neighbors = 50;
  BenchmarkData data(dim, nq, nd, num_neighbors);

  for (auto _ : state) {
    for (size_t q = 0; q < BenchmarkData::kNumQueries; ++q) {
      ChamferDistanceEigen(
          data.queries[q], absl::MakeSpan(data.neighbors_storage[q]),
          [&](GingerNodeId id) { return data.GetDoc(id); }, dim);
      benchmark::DoNotOptimize(data.neighbors_storage[q]);
    }
  }
  data.SetCounters(state, nq, dim);
}

void BM_ChamferHighway(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t nq = state.range(1);
  const size_t nd = state.range(2);
  const size_t num_neighbors = 50;
  BenchmarkData data(dim, nq, nd, num_neighbors);

  for (auto _ : state) {
    for (size_t q = 0; q < BenchmarkData::kNumQueries; ++q) {
      data.dataset.ChamferDistance(data.queries[q],
                                   absl::MakeSpan(data.neighbors_storage[q]));
      benchmark::DoNotOptimize(data.neighbors_storage[q]);
    }
  }
  data.SetCounters(state, nq, dim);
}

void BM_ChamferTurboQuant(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t nq = state.range(1);
  const size_t nd = state.range(2);
  const size_t num_neighbors = 50;
  TurboQuantBenchmarkData data(dim, nq, nd, num_neighbors);

  for (auto _ : state) {
    for (size_t q = 0; q < BenchmarkData::kNumQueries; ++q) {
      data.tq_dataset.ChamferDistance(
          data.queries_pc[q], absl::MakeSpan(data.neighbors_storage[q]));
      benchmark::DoNotOptimize(data.neighbors_storage[q]);
    }
  }
  data.SetCounters(state, nq, dim);
}

static void BenchmarkArguments(benchmark::Benchmark* b) {
  for (int dim : {64, 128, 256}) {
    for (int nq : {4, 8}) {
      for (int nd : {16, 32, 48}) {
        b->Args({dim, nq, nd});
      }
    }
    for (int nd : {64, 128}) {
      b->Args({dim, 16, nd});
      b->Args({dim, 32, nd});
    }
  }
}

// Benchmark arguments focused on small doc token counts: 1-8 tokens per doc,
// various query lengths and dimensions.
static void SmallDocBenchmarkArguments(benchmark::Benchmark* b) {
  for (int dim : {64, 128}) {
    for (int nq : {4, 8}) {
      for (int nd : {1, 2, 3, 4, 8}) {
        b->Args({dim, nq, nd});
      }
    }
  }
}

static void MixedBenchmarkArguments(benchmark::Benchmark* b) {
  for (int dim : {64, 128, 256}) {
    for (int nq : {4, 8, 16, 32}) {
      b->Args({dim, nq, 0});
    }
  }
}

BENCHMARK(BM_ChamferEigen)->Apply(BenchmarkArguments);
BENCHMARK(BM_ChamferEigen)->Apply(SmallDocBenchmarkArguments);
BENCHMARK(BM_ChamferEigen)->Apply(MixedBenchmarkArguments);
BENCHMARK(BM_ChamferHighway)->Apply(BenchmarkArguments);
BENCHMARK(BM_ChamferHighway)->Apply(SmallDocBenchmarkArguments);
BENCHMARK(BM_ChamferHighway)->Apply(MixedBenchmarkArguments);
BENCHMARK(BM_ChamferTurboQuant)->Apply(BenchmarkArguments);
BENCHMARK(BM_ChamferTurboQuant)->Apply(SmallDocBenchmarkArguments);
BENCHMARK(BM_ChamferTurboQuant)->Apply(MixedBenchmarkArguments);

}  // namespace
}  // namespace ginger

BENCHMARK_MAIN();
