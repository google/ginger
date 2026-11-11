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

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/turboquant/many_to_many/benchmark_util.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/many_to_many_turboquant.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "hwy/targets.h"

namespace ginger {
namespace {

void TQArgs(benchmark::Benchmark* b) {
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

void BM_TurboQuantManyToMany(benchmark::State& state) {
  hwy::GetChosenTarget().Update(hwy::SupportedTargets());
  const bool pretranspose = state.range(0);
  const size_t dim = state.range(1);
  const size_t num_db = state.range(2);
  const size_t num_queries = state.range(3);
  const bool is_symmetric = (num_queries == num_db);

  auto* data = GetSharedData(dim, num_db, num_queries);
  auto callback = [](size_t, absl::Span<const float> scores, size_t) {
    benchmark::DoNotOptimize(scores);
  };

  if (pretranspose) {
    auto pretransposed =
        PreTransposedTurboQuantDataset::Build(*data->db_points, *data->tq);
    for (auto _ : state) {
      ManyToManyTurboQuant(
          is_symmetric ? *data->db_points : *data->query_points, pretransposed,
          IndexSubset::Contiguous(0, num_queries), callback);
    }
  } else {
    for (auto _ : state) {
      ManyToManyTurboQuant(
          is_symmetric ? *data->db_points : *data->query_points,
          *data->db_points, IndexSubset::Contiguous(0, num_queries),
          IndexSubset::Contiguous(0, num_db), callback);
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

void BM_EigenMatMul(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t num_points = state.range(1);
  auto* data = GetSharedData(dim, num_points);
  Eigen::Map<const Eigen::MatrixXf> mat(data->db_floats.data(), dim,
                                        num_points);
  Eigen::MatrixXf result(num_points, num_points);
  for (auto _ : state) {
    result.noalias() = mat.transpose() * mat;
    benchmark::DoNotOptimize(result.data());
  }
  SetCounters(state, num_points, dim,
              /*bytes_per_dot_product=*/dim * sizeof(float));
}

BENCHMARK(BM_TurboQuantManyToMany)
    ->Apply(TQArgs)
    ->ArgNames({"pretranspose", "dim", "num_db", "num_queries"});
BENCHMARK(BM_EigenMatMul)->Apply(CustomArgs);

}  // namespace
}  // namespace ginger

BENCHMARK_MAIN();
