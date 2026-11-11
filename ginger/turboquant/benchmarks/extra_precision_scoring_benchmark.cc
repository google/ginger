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
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/internal/scheduler_reference_stub.h"
#include "hwy/targets.h"
#include "parlay/parallel.h"

namespace ginger {
namespace {

using ::benchmark::Benchmark;

constexpr int kNumDatapoints = 1'000'000;
constexpr int kNumQueries = 20;

constexpr int kDimensionalities[] = {
    96,  100,             // Good indicators for messing up the prefetching.
    128, 256, 768, 1536,  // Common embedding dimensionalities.
    3072                  // Large embedding models.
};

constexpr int kIndexLengths[] = {20, 50, 100, 150, 200, 500};

struct CachedDataset {
  size_t dimensionality = 0;
  int quantization_bits = 0;
  TurboQuantDistanceMetric metric =
      TurboQuantDistanceMetric::kDotProductDistance;
  ScalingStorageMode mode = ScalingStorageMode::kSeparateArray;

  std::unique_ptr<TurboQuantDatasetSoA> dataset;
  std::unique_ptr<TurboQuantCodec> turboquant;

  bool operator==(const CachedDataset& rhs) const {
    return dimensionality == rhs.dimensionality &&
           quantization_bits == rhs.quantization_bits && metric == rhs.metric &&
           mode == rhs.mode;
  }
};

void PopulateDatasetRandomly(const TurboQuantCodec& turboquant,
                             size_t num_datapoints,
                             TurboQuantDatasetSoA* dataset) {
  const size_t dimensionality = turboquant.dimensionality();
  const int quantization_bits = dataset->quantization_bits();
  constexpr size_t kPoolSize = 4096;

  std::vector<TurboQuantBasePoint> pool;
  pool.reserve(kPoolSize);
  {
    absl::BitGen bitgen;
    std::normal_distribution<float> dist(0.0f,
                                         1.0f / std::sqrt(dimensionality));
    std::vector<float> temp_coords(dimensionality);

    for (size_t i = 0; i < kPoolSize; ++i) {
      for (float& val : temp_coords) {
        val = dist(bitgen);
      }
      auto result = EncodeDatapoint(turboquant, temp_coords, quantization_bits,
                                    dataset->has_3bit_prefix());
      ABSL_CHECK_OK(result);
      pool.push_back(std::move(*result));
    }
  }

  ginger::internal::ParallelSchedulerReference scheduler;
  const size_t block_size = 1 << 12;
  parlay::parallel_for(
      0, (num_datapoints + block_size - 1) / block_size, [&](size_t block_idx) {
        size_t start = block_idx * block_size;
        size_t end = std::min((block_idx + 1) * block_size, num_datapoints);
        for (size_t i = start; i < end; ++i) {
          ABSL_CHECK_OK(dataset->UpdatePoint(pool[i % kPoolSize].view(), i));
        }
      });
}

const CachedDataset* GetCachedDataset(size_t dimensionality,
                                      int quantization_bits,
                                      TurboQuantDistanceMetric metric,
                                      ScalingStorageMode mode) {
  static absl::NoDestructor<CachedDataset> cache;

  CachedDataset cache_entry = {
      .dimensionality = dimensionality,
      .quantization_bits = quantization_bits,
      .metric = metric,
      .mode = mode,
  };

  if (cache->dataset != nullptr && *cache == cache_entry) {
    return &*cache;
  }

  cache->dataset.reset();
  cache->turboquant.reset();

  cache_entry.turboquant =
      std::make_unique<TurboQuantCodec>(dimensionality, /*seed=*/555, metric);

  auto dataset_or = TurboQuantDatasetSoA::Create(
      *cache_entry.turboquant, kNumDatapoints,
      TurboQuantDatasetOptions{.pad_to_cache_line_size = false,
                               .quantization_bits = quantization_bits,
                               .store_query_byte_sums = false,
                               .normalized_datapoints = false,
                               .scaling_storage_mode = mode});
  ABSL_CHECK_OK(dataset_or);
  auto dataset = *std::move(dataset_or);

  PopulateDatasetRandomly(*cache_entry.turboquant, kNumDatapoints, &dataset);

  cache_entry.dataset =
      std::make_unique<TurboQuantDatasetSoA>(std::move(dataset));

  *cache = std::move(cache_entry);
  return &*cache;
}

void BM_ExtraPrecisionScoring(benchmark::State& state) {
  hwy::GetChosenTarget().Update(hwy::SupportedTargets());
  absl::BitGen bitgen;

  const size_t dimensionality = state.range(0);
  const int quantization_bits = state.range(1);
  const int num_datapoints_scored = state.range(2);
  const TurboQuantDistanceMetric metric =
      static_cast<TurboQuantDistanceMetric>(state.range(3));
  const ScalingStorageMode mode =
      static_cast<ScalingStorageMode>(state.range(4));

  auto* cache =
      GetCachedDataset(dimensionality, quantization_bits, metric, mode);
  TurboQuantDatasetSoA* base_points = cache->dataset.get();
  TurboQuantCodec* turboquant = cache->turboquant.get();

  // Generate queries.
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dimensionality));
  std::vector<TurboQuantQuery> queries;
  queries.reserve(kNumQueries);
  std::vector<float> query_storage(kNumQueries * dimensionality);
  for (float& x : query_storage) {
    x = dist(bitgen);
  }
  for (int i = 0; i < kNumQueries; ++i) {
    auto query = TurboQuantQuery::Create(
        *turboquant,
        absl::MakeConstSpan(query_storage)
            .subspan(i * dimensionality, dimensionality),
        /*enable_higher_precision_scoring=*/true);
    ABSL_CHECK_OK(query);
    queries.push_back(*std::move(query));
  }

  // Generate random indices for each query.
  std::vector<std::vector<int32_t>> indices_storage(kNumQueries);
  for (int q = 0; q < kNumQueries; ++q) {
    indices_storage[q].resize(num_datapoints_scored);
    for (int& idx : indices_storage[q]) {
      idx = absl::Uniform<int32_t>(bitgen, 0, kNumDatapoints);
    }
  }

  std::string label = absl::StrCat(
      (metric == TurboQuantDistanceMetric::kDotProductDistance) ? "DotProduct"
                                                                : "SquaredL2",
      ", SoA, ", quantization_bits,
      " bits, Random, idx=", num_datapoints_scored,
      mode == ScalingStorageMode::kIn3BitPrefix ? ", 3-bit prefix" : "");
  state.SetLabel(label);

  std::vector<float> scores_storage(num_datapoints_scored);
  for (auto _ : state) {
    for (int q = 0; q < kNumQueries; ++q) {
      auto accessor = ScoresAndIndicesPairOfSpansView(
          absl::MakeConstSpan(indices_storage[q]),
          absl::MakeSpan(scores_storage));

      base_points->UnsafeComputeDistancesExtraPrecision(queries[q], accessor);

      benchmark::DoNotOptimize(scores_storage);
    }
  }

  const size_t items_per_iter = kNumQueries * num_datapoints_scored;
  const size_t bytes_per_iter =
      (base_points->stride() + base_points->extra_precision_stride() +
       (metric == TurboQuantDistanceMetric::kSquaredL2Distance ? sizeof(float)
                                                               : 0)) *
      items_per_iter;
  state.SetBytesProcessed(state.iterations() * bytes_per_iter);
  state.SetItemsProcessed(state.iterations() * items_per_iter);
}

BENCHMARK(BM_ExtraPrecisionScoring)
    ->Apply([](Benchmark* b) {
      for (int dim : kDimensionalities) {
        for (int bits : {5, 6, 7, 8, 10, 12}) {
          for (TurboQuantDistanceMetric metric :
               {TurboQuantDistanceMetric::kDotProductDistance,
                TurboQuantDistanceMetric::kSquaredL2Distance}) {
            for (ScalingStorageMode mode :
                 {ScalingStorageMode::kSeparateArray,
                  ScalingStorageMode::kIn3BitPrefix}) {
              // The 3-bit prefix requires the kScalingOnly norm layout.
              if (mode == ScalingStorageMode::kIn3BitPrefix &&
                  metric != TurboQuantDistanceMetric::kDotProductDistance) {
                continue;
              }
              for (int idx_len : kIndexLengths) {
                b->Args({dim, bits, idx_len, static_cast<int>(metric),
                         static_cast<int>(mode)});
              }
            }
          }
        }
      }
      b->ArgNames({"dim", "bits", "idx_len", "metric", "mode"});
    })
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace ginger

BENCHMARK_MAIN();
