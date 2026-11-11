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
#include <type_traits>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/distance/distance_bytes.h"
#include "ginger/types.h"

namespace ginger {
namespace {

template <typename T>
struct BenchmarkData {
  SpanDataset dataset;
  std::vector<T> query;
  std::vector<DistanceAndId> neighbors;

  BenchmarkData(size_t dim, size_t num_neighbors)
      : dataset(num_neighbors + 1, dim, CoordinateTypeFor<T>()),
        query(dim),
        neighbors(num_neighbors) {
    absl::BitGen bg;
    const auto random_value = [&bg]() -> T {
      if constexpr (std::is_signed_v<T>) {
        return static_cast<T>(absl::Uniform<int16_t>(bg, -128, 127));
      } else {
        return static_cast<T>(absl::Uniform<uint16_t>(bg, 0, 255));
      }
    };

    for (size_t i = 0; i < dim; ++i) {
      query[i] = random_value();
    }

    for (size_t i = 0; i < num_neighbors; ++i) {
      std::vector<T> point(dim);
      for (size_t j = 0; j < dim; ++j) {
        point[j] = random_value();
      }
      dataset.Set<T>(i, point);
      neighbors[i].id = static_cast<GingerNodeId>(i);
    }
  }
};

// =============================================================================
// Int8 Benchmarks
// =============================================================================

void BM_DotProductInt8_Highway_ByteQuery(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t num_neighbors = 50;
  BenchmarkData<int8_t> data(dim, num_neighbors);

  for (auto _ : state) {
    DotProductDistance(absl::MakeConstSpan(data.query),
                       absl::MakeSpan(data.neighbors), data.dataset);
    benchmark::DoNotOptimize(data.neighbors);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors * dim * sizeof(int8_t));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors);
}

void BM_SquaredL2Int8_Highway_ByteQuery(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t num_neighbors = 50;
  BenchmarkData<int8_t> data(dim, num_neighbors);

  for (auto _ : state) {
    SquaredL2Distance(absl::MakeConstSpan(data.query),
                      absl::MakeSpan(data.neighbors), data.dataset);
    benchmark::DoNotOptimize(data.neighbors);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors * dim * sizeof(int8_t));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors);
}

// =============================================================================
// Uint8 Benchmarks
// =============================================================================

void BM_DotProductUint8_Highway_ByteQuery(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t num_neighbors = 50;
  BenchmarkData<uint8_t> data(dim, num_neighbors);

  for (auto _ : state) {
    DotProductDistance(absl::MakeConstSpan(data.query),
                       absl::MakeSpan(data.neighbors), data.dataset);
    benchmark::DoNotOptimize(data.neighbors);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors * dim * sizeof(uint8_t));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors);
}

void BM_SquaredL2Uint8_Highway_ByteQuery(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t num_neighbors = 50;
  BenchmarkData<uint8_t> data(dim, num_neighbors);

  for (auto _ : state) {
    SquaredL2Distance(absl::MakeConstSpan(data.query),
                      absl::MakeSpan(data.neighbors), data.dataset);
    benchmark::DoNotOptimize(data.neighbors);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors * dim * sizeof(uint8_t));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_neighbors);
}

// =============================================================================
// Benchmark Registrations
// =============================================================================

BENCHMARK(BM_DotProductInt8_Highway_ByteQuery)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_SquaredL2Int8_Highway_ByteQuery)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_DotProductUint8_Highway_ByteQuery)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_SquaredL2Uint8_Highway_ByteQuery)
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
