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
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/types/span.h"
#include "ginger/turboquant/many_to_many/benchmark_util.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/many_to_many_topk.h"
#include "ginger/turboquant/many_to_many/many_to_many_turboquant.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "hwy/targets.h"

namespace ginger {
namespace {

// No-op callback baseline: measures scoring cost without top-k tracking.
void BM_ManyToManyNoOp(benchmark::State& state) {
  hwy::GetChosenTarget().Update(hwy::SupportedTargets());
  const size_t dim = state.range(0);
  const size_t num_points = state.range(1);
  auto* data = GetSharedData(dim, num_points);
  for (auto _ : state) {
    ManyToManyTurboQuant(*data->db_points, *data->db_points,
                         IndexSubset::Contiguous(0, num_points),
                         IndexSubset::Contiguous(0, num_points),
                         [](size_t, absl::Span<const float> scores, size_t) {
                           benchmark::DoNotOptimize(scores);
                         });
  }
  SetCounters(state, num_points, dim, /*bytes_per_dot_product=*/dim / 2);
}

void Top1Args(benchmark::Benchmark* b) {
  for (bool pretranspose : {false, true}) {
    for (int dim : {128, 256, 768, 1536}) {
      for (int num_db : {500, 2000, 10000}) {
        b->Args({pretranspose, dim, num_db, num_db});
      }
    }
    for (int dim : {256, 768}) {
      for (int num_db : {10000}) {
        for (int num_q : {100, 1000, 5000, 20000, 50000}) {
          b->Args({pretranspose, dim, num_db, num_q});
        }
      }
    }
  }
}

void BM_ManyToManyTop1(benchmark::State& state) {
  hwy::GetChosenTarget().Update(hwy::SupportedTargets());
  const bool pretranspose = state.range(0);
  const size_t dim = state.range(1);
  const size_t num_db = state.range(2);
  const size_t num_queries = state.range(3);
  const bool is_symmetric = (num_queries == num_db);

  auto* data = GetSharedData(dim, num_db, num_queries);

  if (pretranspose) {
    auto pretransposed =
        PreTransposedTurboQuantDataset::Build(*data->db_points, *data->tq);
    for (auto _ : state) {
      auto results = ManyToManyTurboQuantTopK(
          is_symmetric ? *data->db_points : *data->query_points, pretransposed,
          IndexSubset::Contiguous(0, num_queries), /*k=*/1);
      benchmark::DoNotOptimize(results);
    }
  } else {
    for (auto _ : state) {
      auto results = ManyToManyTurboQuantTopK(
          is_symmetric ? *data->db_points : *data->query_points,
          *data->db_points, IndexSubset::Contiguous(0, num_queries),
          IndexSubset::Contiguous(0, num_db), /*k=*/1);
      benchmark::DoNotOptimize(results);
    }
  }

  state.counters["num_queries"] = num_queries;
  state.counters["num_db"] = num_db;
  state.counters["MulAccOps/sec"] =
      benchmark::Counter(num_queries * num_db * dim,
                         benchmark::Counter::kIsIterationInvariantRate);
  state.counters["DotProducts/sec"] = benchmark::Counter(
      num_queries * num_db, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["BytesProcessed/sec"] =
      benchmark::Counter(num_queries * num_db * (dim / 2),
                         benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_ManyToManyNoOp)
    ->Apply(CustomArgs)
    ->ArgNames({"dim", "num_queries"});

BENCHMARK(BM_ManyToManyTop1)
    ->Apply(Top1Args)
    ->ArgNames({"pretranspose", "dim", "num_db", "num_queries"});

// Benchmark for ManyToManyTurboQuantTopK with arbitrary k.
void BM_ManyToManyTopK(benchmark::State& state) {
  hwy::GetChosenTarget().Update(hwy::SupportedTargets());
  const size_t dim = state.range(0);
  const size_t num_db = state.range(1);
  const size_t num_queries = state.range(2);
  const int k = state.range(3);
  const bool is_symmetric = (num_queries == num_db);

  auto* data = GetSharedData(dim, num_db, num_queries);

  for (auto _ : state) {
    auto results = ManyToManyTurboQuantTopK(
        is_symmetric ? *data->db_points : *data->query_points, *data->db_points,
        IndexSubset::Contiguous(0, num_queries),
        IndexSubset::Contiguous(0, num_db), k);
    benchmark::DoNotOptimize(results);
  }

  state.counters["num_queries"] = num_queries;
  state.counters["num_db"] = num_db;
  state.counters["k"] = k;
  state.counters["MulAccOps/sec"] =
      benchmark::Counter(num_queries * num_db * dim,
                         benchmark::Counter::kIsIterationInvariantRate);
  state.counters["DotProducts/sec"] = benchmark::Counter(
      num_queries * num_db, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["BytesProcessed/sec"] =
      benchmark::Counter(num_queries * num_db * (dim / 2),
                         benchmark::Counter::kIsIterationInvariantRate);
}

// Argument generator for TopK benchmarks.
void TopKArgs(benchmark::Benchmark* b) {
  for (int dim : {256, 768}) {  // Common dimensions
    // 1. Varying k, fixed large DB and medium queries
    int num_db = 10000;
    int num_q = 1000;
    for (int k : {1, 10, 50, 100, 500, 1000}) {
      b->Args({dim, num_db, num_q, k});
    }

    // 2. Varying queries, fixed DB and k
    int k = 100;
    num_db = 10000;
    for (int num_q : {100, 1000, 10000}) {
      b->Args({dim, num_db, num_q, k});
    }

    // 3. Varying DB size, fixed queries and k
    num_q = 1000;
    k = 100;
    for (int num_db : {2000, 10000, 50000}) {
      b->Args({dim, num_db, num_q, k});
    }

    // 4. Symmetric case (num_q == num_db)
    k = 100;
    for (int num_db : {2000, 10000}) {
      b->Args({dim, num_db, num_db, k});
    }

    // 5. Large database scale case
    b->Args({dim, 1000000, 1000, 50});

    // 6. Small workloads with k = 20
    for (int n : {500, 2000}) {
      b->Args({dim, n, n, 20});
    }
  }
}

BENCHMARK(BM_ManyToManyTopK)
    ->Apply(TopKArgs)
    ->ArgNames({"dim", "num_db", "num_queries", "k"});

}  // namespace
}  // namespace ginger

BENCHMARK_MAIN();
