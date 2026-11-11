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

#ifndef GINGER_TURBOQUANT_MANY_TO_MANY_MANY_TO_MANY_TOPK_H_
#define GINGER_TURBOQUANT_MANY_TO_MANY_MANY_TO_MANY_TOPK_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/types/span.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/many_to_many_turboquant.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"

namespace ginger {

struct TopKResult {
  // Position of the database point within the `db_subset` passed to
  // ManyToManyTurboQuantTopK, in `[0, db_subset.size())`, not an index into
  // the `database` dataset. Use `db_subset[db_idx]` to map it back to a
  // dataset index. For the PreTransposedTurboQuantDataset overloads, this is
  // the position in the pretransposed dataset.
  uint32_t db_idx;
  float distance;

  bool operator==(const TopKResult&) const = default;
};

namespace many_to_many_topk_internal {

// Working memory for many-to-many Top-K operations.
struct TopKState {
  // Pre-allocated buffer storing intermediate results per query.
  // For query `i`, the sub-array starts at `items[i * buffer_capacity]`.
  // Format: 32-bit ordered score (upper) | 32-bit database index (lower).
  std::vector<uint64_t> items;

  // Current number of collected candidates for each query.
  std::vector<size_t> sizes;

  // Cutoff threshold for each query. Database points with a distance
  // strictly above this threshold are skipped.
  std::vector<float> thresholds;

  // Final results, `k` per query.
  std::vector<TopKResult> results;

  // Reusable scratch buffers for GEMM/decoding.
  ManyToManyTurboQuantWorkspace workspace;

  // Fixed configuration for this execution.
  size_t num_queries = 0;
  size_t k = 0;
  size_t buffer_capacity = 0;
};

}  // namespace many_to_many_topk_internal

// Working memory for the `ManyToManyTopKState&` overloads of
// `ManyToManyTurboQuantTopK` below. Reusing one instance across calls (e.g.,
// one per thread) avoids reallocating its buffers.
class ManyToManyTopKState {
 private:
  friend struct ManyToManyTopKStateAccess;
  friend class ManyToManyTopKStatePeer;

  many_to_many_topk_internal::TopKState state_;
};

// Computes approximate distances between queries and database points
// using ManyToManyTurboQuant, and retains the top-k nearest database points
// per query in query_subset.
//
// Returns a vector of size query_subset.size() * k. The results for the query
// at position i of query_subset are stored in indices [i * k, (i + 1) * k).
// `TopKResult::db_idx` is a position within `db_subset` (see above).
//
// `initial_threshold` sets the starting distance bound per query. Only scores
// below this threshold are collected. Use this to implement edge weight
// filtering: e.g., for similarity_lower_bound s, pass -s as the threshold
// (since TurboQuant distances are negated dot products).
std::vector<TopKResult> ManyToManyTurboQuantTopK(
    const TurboQuantDatasetSoA& queries, const TurboQuantDatasetSoA& database,
    IndexSubset query_subset, IndexSubset db_subset, int k,
    float initial_threshold = std::numeric_limits<float>::max());

// Pretransposed database overload for TopK.
std::vector<TopKResult> ManyToManyTurboQuantTopK(
    const TurboQuantDatasetSoA& queries,
    const PreTransposedTurboQuantDataset& database, IndexSubset query_subset,
    int k, float initial_threshold = std::numeric_limits<float>::max());

// Same as the overloads above, but reuses the memory in `state` across calls.
// The returned span points into `state` and remains valid until `state` is
// used again or destroyed.
//
// If non-empty, `initial_thresholds[i]` is the initial threshold (see above) of
// the i-th query of `query_subset`. CHECK-fails unless
// `initial_thresholds.size() == query_subset.size()`.
absl::Span<const TopKResult> ManyToManyTurboQuantTopK(
    const TurboQuantDatasetSoA& queries, const TurboQuantDatasetSoA& database,
    IndexSubset query_subset, IndexSubset db_subset, int k,
    ManyToManyTopKState& state,
    absl::Span<const float> initial_thresholds = {});

// Pretransposed database overload of the above.
absl::Span<const TopKResult> ManyToManyTurboQuantTopK(
    const TurboQuantDatasetSoA& queries,
    const PreTransposedTurboQuantDataset& database, IndexSubset query_subset,
    int k, ManyToManyTopKState& state,
    absl::Span<const float> initial_thresholds = {});

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_MANY_TO_MANY_TOPK_H_
