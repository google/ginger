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

#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/projection/pca_float.h"
#include "ginger/projection/pca_turboquant.h"

namespace ginger {
namespace {

struct BenchmarkData {
  int32_t input_dim;
  int32_t output_dim;
  std::vector<float> query;
  std::vector<float> output;
  Eigen::MatrixXf projection_matrix;

  std::unique_ptr<TurboQuantPcaProjection> tq_projection;
  std::unique_ptr<FloatPcaProjection> float_projection;

  BenchmarkData(int32_t input_dim, int32_t output_dim)
      : input_dim(input_dim),
        output_dim(output_dim),
        query(((input_dim + 127) / 128) * 128 * 2, 0.0f),
        output(output_dim, 0.0f),
        projection_matrix(output_dim, input_dim) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int32_t i = 0; i < input_dim; ++i) {
      query[i] = dist(gen);
    }
    for (int32_t i = 0; i < output_dim; ++i) {
      for (int32_t j = 0; j < input_dim; ++j) {
        projection_matrix(i, j) = dist(gen);
      }
    }

    auto float_proj_or = FloatPcaProjection::Create(projection_matrix);
    if (float_proj_or.ok()) {
      float_projection = std::move(*float_proj_or);
    }

    auto tq_proj_or = TurboQuantPcaProjection::Create(projection_matrix);
    if (tq_proj_or.ok()) {
      tq_projection = std::move(*tq_proj_or);
    }
  }
};

void BM_FloatPca(benchmark::State& state) {
  const int32_t input_dim = state.range(0);
  const int32_t output_dim = state.range(1);
  BenchmarkData data(input_dim, output_dim);
  alignas(64) float local_output[1024];

  for (auto _ : state) {
    auto status = data.float_projection->Project(
        absl::MakeSpan(data.query).first(input_dim),
        absl::MakeSpan(local_output, output_dim));
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(local_output);
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_TurboQuantPca(benchmark::State& state) {
  const int32_t input_dim = state.range(0);
  const int32_t output_dim = state.range(1);
  BenchmarkData data(input_dim, output_dim);
  alignas(64) float local_output[1024];

  for (auto _ : state) {
    auto status =
        data.tq_projection->Project(absl::MakeSpan(data.query).first(input_dim),
                                    absl::MakeSpan(local_output, output_dim));
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(local_output);
  }
  state.SetItemsProcessed(state.iterations());
}

static void BenchmarkArguments(benchmark::Benchmark* b) {
  b->Args({3072, 768});
}

BENCHMARK(BM_FloatPca)->Apply(BenchmarkArguments);
BENCHMARK(BM_TurboQuantPca)->Apply(BenchmarkArguments);

}  // namespace
}  // namespace ginger

BENCHMARK_MAIN();
