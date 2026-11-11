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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/dataset.h"
#include "ginger/distance/distance.h"
#include "ginger/types.h"

namespace ginger {
namespace {

void DotProductDistanceOld(absl::Span<const float> query,
                           absl::Span<DistanceAndId> neighbors,
                           const SpanDataset& dataset) {
  Eigen::Map<const Eigen::VectorXf> query_vec(query.data(), query.size());
  for (auto& [distance, neighbor_id] : neighbors) {
    absl::Span<const float> candidate = dataset.get<float>(neighbor_id);
    Eigen::Map<const Eigen::VectorXf> candidate_vec(candidate.data(),
                                                    candidate.size());
    distance = -query_vec.dot(candidate_vec);
  }
}

void SquaredL2DistanceOld(absl::Span<const float> query,
                          absl::Span<DistanceAndId> neighbors,
                          const SpanDataset& dataset) {
  Eigen::Map<const Eigen::VectorXf> query_vec(query.data(), query.size());
  for (auto& [distance, neighbor_id] : neighbors) {
    absl::Span<const float> candidate = dataset.get<float>(neighbor_id);
    Eigen::Map<const Eigen::VectorXf> candidate_vec(candidate.data(),
                                                    candidate.size());
    distance = (query_vec - candidate_vec).squaredNorm();
  }
}

struct BenchmarkData {
  SpanDataset dataset;
  std::vector<float> query;
  std::vector<DistanceAndId> neighbors;

  BenchmarkData(size_t dim, size_t num_neighbors)
      : dataset(num_neighbors + 1, dim),
        query(dim, 0.0f),
        neighbors(num_neighbors) {
    absl::BitGen bg;
    for (size_t i = 0; i < dim; ++i) {
      query[i] = absl::Uniform<float>(bg, -1.0f, 1.0f);
    }
    for (size_t i = 0; i < num_neighbors; ++i) {
      std::vector<float> point(dim);
      for (size_t j = 0; j < dim; ++j) {
        point[j] = absl::Uniform<float>(bg, -1.0f, 1.0f);
      }
      dataset.Set<float>(i, point);
      neighbors[i].id = static_cast<GingerNodeId>(i);
    }
  }
};

void BM_DotProductOld(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t num_neighbors = 50;
  BenchmarkData data(dim, num_neighbors);

  for (auto _ : state) {
    DotProductDistanceOld(data.query, absl::MakeSpan(data.neighbors),
                          data.dataset);
    benchmark::DoNotOptimize(data.neighbors);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors * dim * sizeof(float));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors);
}

void BM_DotProductHighway(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t num_neighbors = 50;
  BenchmarkData data(dim, num_neighbors);

  for (auto _ : state) {
    DotProductDistance(data.query, absl::MakeSpan(data.neighbors),
                       data.dataset);
    benchmark::DoNotOptimize(data.neighbors);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors * dim * sizeof(float));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors);
}

void BM_SquaredL2Old(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t num_neighbors = 50;
  BenchmarkData data(dim, num_neighbors);

  for (auto _ : state) {
    SquaredL2DistanceOld(data.query, absl::MakeSpan(data.neighbors),
                         data.dataset);
    benchmark::DoNotOptimize(data.neighbors);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors * dim * sizeof(float));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors);
}

void BM_SquaredL2Highway(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t num_neighbors = 50;
  BenchmarkData data(dim, num_neighbors);

  for (auto _ : state) {
    SquaredL2Distance(data.query, absl::MakeSpan(data.neighbors), data.dataset);
    benchmark::DoNotOptimize(data.neighbors);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors * dim * sizeof(float));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors);
}

BENCHMARK(BM_DotProductOld)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_DotProductHighway)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_SquaredL2Old)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_SquaredL2Highway)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

}  // namespace
}  // namespace ginger

BENCHMARK_MAIN();
