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
#include <random>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/types/span.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/benchmark_util.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/many_to_many_topk.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "ginger/turboquant/pipnn/assign_leaders.h"
#include "hwy/targets.h"

namespace ginger {
namespace {

void BenchmarkArgsK1(benchmark::Benchmark* b) {
  for (int dim : {128, 256, 768}) {
    for (int num_leaders : {10, 30, 64, 256, 1024, 4096}) {
      b->Args({dim, num_leaders});
    }
  }
}

void BenchmarkArgsK3K10(benchmark::Benchmark* b) {
  for (int dim : {128, 256, 768}) {
    for (int num_leaders : {64, 256, 1024, 4096}) {
      b->Args({dim, num_leaders});
    }
  }
}

template <typename BenchmarkFunc>
void RunBenchmarkImpl(benchmark::State& state, BenchmarkFunc run_kernel) {
  hwy::GetChosenTarget().Update(hwy::SupportedTargets());
  const size_t dim = state.range(0);
  const size_t num_leaders = state.range(1);
  const size_t num_queries = 1024;  // Fixed query batch size for benchmark.
  const size_t num_queries_pool = 10000;  // Larger pool for random access.

  SharedData* shared = GetSharedData(dim, num_leaders, num_queries_pool);
  auto leaders =
      PreTransposedTurboQuantDataset::Build(*shared->db_points, *shared->tq);

  std::vector<int> q_indices(num_queries);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, num_queries_pool - 1);
  for (int& idx : q_indices) {
    idx = dist(rng);
  }

  run_kernel(state, *shared->query_points, leaders,
             absl::MakeConstSpan(q_indices));

  state.counters["queries/sec"] = benchmark::Counter(
      num_queries, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["GMACs/sec"] = benchmark::Counter(
      static_cast<double>(num_queries) * num_leaders * dim / 1e9,
      benchmark::Counter::kIsIterationInvariantRate);
}

void BM_AssignToLeadersTopK1(benchmark::State& state) {
  RunBenchmarkImpl(state, [](benchmark::State& state,
                             const TurboQuantDatasetSoA& query_bp,
                             const PreTransposedTurboQuantDataset& leaders,
                             absl::Span<const int> q_indices) {
    for (auto _ : state) {
      auto topk_results = AssignToLeadersTopK1(
          query_bp, IndexSubset::Gathered(q_indices), leaders);
      benchmark::DoNotOptimize(topk_results);
    }
  });
}

void BM_AssignToLeadersTopK3(benchmark::State& state) {
  RunBenchmarkImpl(state, [](benchmark::State& state,
                             const TurboQuantDatasetSoA& query_bp,
                             const PreTransposedTurboQuantDataset& leaders,
                             absl::Span<const int> q_indices) {
    for (auto _ : state) {
      auto topk_results = AssignToLeadersTopK3(
          query_bp, IndexSubset::Gathered(q_indices), leaders);
      benchmark::DoNotOptimize(topk_results);
    }
  });
}

void BM_AssignToLeadersConsecutiveTopK10(benchmark::State& state) {
  RunBenchmarkImpl(state, [](benchmark::State& state,
                             const TurboQuantDatasetSoA& query_bp,
                             const PreTransposedTurboQuantDataset& leaders,
                             absl::Span<const int> q_indices) {
    for (auto _ : state) {
      auto topk_results = AssignToLeadersTopK10(
          query_bp, IndexSubset::Contiguous(0, q_indices.size()), leaders);
      benchmark::DoNotOptimize(topk_results);
    }
  });
}

void BM_ManyToManyTopK1(benchmark::State& state) {
  RunBenchmarkImpl(state, [](benchmark::State& state,
                             const TurboQuantDatasetSoA& query_bp,
                             const PreTransposedTurboQuantDataset& leaders,
                             absl::Span<const int> q_indices) {
    for (auto _ : state) {
      auto results = ManyToManyTurboQuantTopK(
          query_bp, leaders, IndexSubset::Gathered(q_indices), 1);
      benchmark::DoNotOptimize(results);
    }
  });
}

void BM_ManyToManyTopK3(benchmark::State& state) {
  RunBenchmarkImpl(state, [](benchmark::State& state,
                             const TurboQuantDatasetSoA& query_bp,
                             const PreTransposedTurboQuantDataset& leaders,
                             absl::Span<const int> q_indices) {
    for (auto _ : state) {
      auto results = ManyToManyTurboQuantTopK(
          query_bp, leaders, IndexSubset::Gathered(q_indices), 3);
      benchmark::DoNotOptimize(results);
    }
  });
}

void BM_ManyToManyConsecutiveTopK10(benchmark::State& state) {
  RunBenchmarkImpl(state, [](benchmark::State& state,
                             const TurboQuantDatasetSoA& query_bp,
                             const PreTransposedTurboQuantDataset& leaders,
                             absl::Span<const int> q_indices) {
    for (auto _ : state) {
      auto results = ManyToManyTurboQuantTopK(
          query_bp, leaders, IndexSubset::Contiguous(0, q_indices.size()), 10);
      benchmark::DoNotOptimize(results);
    }
  });
}

BENCHMARK(BM_AssignToLeadersTopK1)
    ->Apply(BenchmarkArgsK1)
    ->ArgNames({"dim", "leaders"});

BENCHMARK(BM_AssignToLeadersTopK3)
    ->Apply(BenchmarkArgsK3K10)
    ->ArgNames({"dim", "leaders"});

BENCHMARK(BM_AssignToLeadersConsecutiveTopK10)
    ->Apply(BenchmarkArgsK3K10)
    ->ArgNames({"dim", "leaders"});

BENCHMARK(BM_ManyToManyTopK1)
    ->Apply(BenchmarkArgsK1)
    ->ArgNames({"dim", "leaders"});

BENCHMARK(BM_ManyToManyTopK3)
    ->Apply(BenchmarkArgsK3K10)
    ->ArgNames({"dim", "leaders"});

BENCHMARK(BM_ManyToManyConsecutiveTopK10)
    ->Apply(BenchmarkArgsK3K10)
    ->ArgNames({"dim", "leaders"});

}  // namespace
}  // namespace ginger

BENCHMARK_MAIN();
