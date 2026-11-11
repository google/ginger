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

#ifndef GINGER_TURBOQUANT_MANY_TO_MANY_BENCHMARK_UTIL_H_
#define GINGER_TURBOQUANT_MANY_TO_MANY_BENCHMARK_UTIL_H_

#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/base/const_init.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"

namespace ginger {

// Lazily-initialized, mutex-protected cache of TurboQuant-encoded random data.
// Shared across benchmark functions that use the same (dim, num_points) pair.
struct SharedData {
  size_t dimensionality = 0;
  size_t num_db = 0;
  size_t num_queries = 0;
  std::vector<float> db_floats;
  std::unique_ptr<TurboQuantCodec> tq;
  std::unique_ptr<TurboQuantDatasetSoA> db_points;
  std::unique_ptr<TurboQuantDatasetSoA> query_points;
};

inline SharedData* GetSharedData(size_t dimensionality, size_t num_db,
                                 size_t num_queries = 0) {
  if (num_queries == 0) {
    num_queries = num_db;
  }
  static absl::Mutex mutex(absl::kConstInit);
  absl::MutexLock lock(mutex);
  static SharedData* cache = new SharedData();
  if (cache->dimensionality == dimensionality && cache->num_db == num_db &&
      cache->num_queries == num_queries) {
    return cache;
  }
  cache->dimensionality = dimensionality;
  cache->num_db = num_db;
  cache->num_queries = num_queries;

  absl::BitGen bitgen;
  std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(dimensionality));

  cache->db_floats.resize(num_db * dimensionality);
  for (float& x : cache->db_floats) x = dist(bitgen);
  std::vector<absl::Span<const float>> db_datapoints;
  db_datapoints.reserve(num_db);
  for (size_t i = 0; i < num_db; ++i) {
    db_datapoints.push_back(absl::MakeConstSpan(
        cache->db_floats.data() + i * dimensionality, dimensionality));
  }

  std::vector<float> q_floats(num_queries * dimensionality);
  for (float& x : q_floats) x = dist(bitgen);
  std::vector<absl::Span<const float>> q_datapoints;
  q_datapoints.reserve(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    q_datapoints.push_back(absl::MakeConstSpan(
        q_floats.data() + i * dimensionality, dimensionality));
  }

  cache->tq = std::make_unique<TurboQuantCodec>(dimensionality, 42);

  auto db_bp = TurboQuantDatasetSoA::Create(*cache->tq, db_datapoints);
  ABSL_CHECK_OK(db_bp);
  cache->db_points = std::make_unique<TurboQuantDatasetSoA>(*std::move(db_bp));

  auto q_bp = TurboQuantDatasetSoA::Create(*cache->tq, q_datapoints);
  ABSL_CHECK_OK(q_bp);
  cache->query_points =
      std::make_unique<TurboQuantDatasetSoA>(*std::move(q_bp));

  return cache;
}

inline void SetCounters(benchmark::State& state, size_t num_points, size_t dim,
                        size_t bytes_per_dot_product) {
  state.counters["num_queries"] = num_points;
  state.counters["num_db"] = num_points;
  state.counters["MulAccOps/sec"] =
      benchmark::Counter(num_points * num_points * dim,
                         benchmark::Counter::kIsIterationInvariantRate);
  state.counters["DotProducts/sec"] = benchmark::Counter(
      num_points * num_points, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["BytesProcessed/sec"] =
      benchmark::Counter(num_points * num_points * bytes_per_dot_product,
                         benchmark::Counter::kIsIterationInvariantRate);
}

inline void CustomArgs(benchmark::Benchmark* b) {
  for (int dim : {128, 256, 768, 1536}) {
    for (int n : {500, 2000, 10000}) {
      b->Args({dim, n});
    }
  }
}

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_BENCHMARK_UTIL_H_
