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
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/pipnn/process_leaf.h"
#include "ginger/internal/scheduler_reference_stub.h"
#include "hwy/targets.h"
#include "parlay/parallel.h"

namespace ginger {
namespace {

// Gather benchmark: leaf indices are random samples from a large (1M) dataset.
// This stresses gather/decode from non-contiguous memory, matching real usage.
constexpr size_t kLargeDatasetSize = 1'000'000;

struct SharedData {
  size_t dimensionality = 0;
  size_t num_points = 0;

  std::vector<float> float_data;
  std::unique_ptr<TurboQuantCodec> tq;
  std::unique_ptr<TurboQuantDatasetSoA> tq_base_points;
};

// Returns cached data for (dimensionality, num_points). Constructs in parallel
// for large datasets. Thread-safe: only called from the main thread before
// benchmark iterations.
const SharedData* GetSharedData(size_t dimensionality, size_t num_points) {
  static absl::NoDestructor<SharedData> cache;

  if (cache->dimensionality == dimensionality &&
      cache->num_points == num_points) {
    return &*cache;
  }

  cache->dimensionality = dimensionality;
  cache->num_points = num_points;

  cache->float_data.resize(num_points * dimensionality);
  constexpr size_t kBlockSize = 1024;
  const size_t num_blocks = (num_points + kBlockSize - 1) / kBlockSize;
  ginger::internal::ParallelSchedulerReference scheduler;
  parlay::parallel_for(
      0, num_blocks,
      [&](size_t block) {
        const size_t begin = block * kBlockSize;
        const size_t end = std::min(begin + kBlockSize, num_points);
        // Per-block RNG seeded deterministically from block index.
        std::mt19937 rng(42 + block);
        std::normal_distribution<float> dist(0.0f,
                                             1.0f / std::sqrt(dimensionality));
        for (size_t i = begin; i < end; ++i) {
          for (size_t d = 0; d < dimensionality; ++d) {
            cache->float_data[i * dimensionality + d] = dist(rng);
          }
        }
      },
      /*granularity=*/1);

  std::vector<absl::Span<const float>> datapoints;
  datapoints.reserve(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    datapoints.push_back(absl::MakeConstSpan(
        cache->float_data.data() + i * dimensionality, dimensionality));
  }
  cache->tq = std::make_unique<TurboQuantCodec>(dimensionality, 42);
  auto bp = TurboQuantDatasetSoA::Create(
      *cache->tq, datapoints,
      TurboQuantDatasetOptions{
          .pad_to_cache_line_size = false,
          .quantization_bits = 4,
          .store_query_byte_sums = false,
          .normalized_datapoints = false,
          .scaling_storage_mode = ScalingStorageMode::kSeparateArray});
  ABSL_CHECK_OK(bp);
  cache->tq_base_points =
      std::make_unique<TurboQuantDatasetSoA>(*std::move(bp));

  return &*cache;
}

void GatherBenchmarkArgs(benchmark::Benchmark* b) {
  for (int dim : {96, 128, 256, 512, 768, 1536}) {
    for (int leaf_size : {100, 200, 300, 500, 2000}) {
      b->Args({dim, leaf_size});
    }
  }
}

void BM_ProcessLeafTop2Gather(benchmark::State& state) {
  hwy::GetChosenTarget().Update(hwy::SupportedTargets());
  const size_t dim = state.range(0);
  const size_t leaf_size = state.range(1);

  const SharedData* shared = GetSharedData(dim, kLargeDatasetSize);

  // Sample leaf_size random indices from [0, kLargeDatasetSize).
  std::mt19937 rng(123);
  std::vector<int> indices(leaf_size);
  std::uniform_int_distribution<int> index_dist(0, kLargeDatasetSize - 1);
  for (auto& idx : indices) idx = index_dist(rng);
  // Deduplicate (ProcessLeaf requires distinct indices for self-exclusion).
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

  const size_t actual_leaf = indices.size();
  const size_t num_pairs = actual_leaf * actual_leaf;

  ProcessLeafWorkspace workspace;
  // Warm up.
  ProcessLeafTop2(*shared->tq, *shared->tq_base_points, absl::MakeSpan(indices),
                  workspace);

  for (auto _ : state) {
    ProcessLeafTop2(*shared->tq, *shared->tq_base_points,
                    absl::MakeSpan(indices), workspace);
    benchmark::DoNotOptimize(workspace.top2_results.data());
  }

  state.counters["items/sec"] = benchmark::Counter(
      actual_leaf, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["ops/sec"] = benchmark::Counter(
      num_pairs * dim, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_ProcessLeafTop2Gather)
    ->Apply(GatherBenchmarkArgs)
    ->ArgNames({"dim", "leafsize"});

}  // namespace
}  // namespace ginger

BENCHMARK_MAIN();
