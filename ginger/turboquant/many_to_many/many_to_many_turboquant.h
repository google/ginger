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

#ifndef GINGER_TURBOQUANT_MANY_TO_MANY_MANY_TO_MANY_TURBOQUANT_H_
#define GINGER_TURBOQUANT_MANY_TO_MANY_MANY_TO_MANY_TURBOQUANT_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// Reusable scratch buffers for ManyToManyTurboQuant to avoid heap allocations
// across repeated invocations. Buffers grow on demand and are never shrunk.
struct ManyToManyTurboQuantWorkspace {
  // Decoded and block-transposed database panels for the current L3 block
  // (used by the on-the-fly database decode path).
  hwy::AlignedVector<uint8_t> db_panels;
  // Per-database-point quantization scale factors for the current L3 block.
  hwy::AlignedVector<float> db_norm_scaling_factors;
  // Per-database-point squared L2 norms for the current L3 block (used only
  // when metric == kSquaredL2Distance).
  hwy::AlignedVector<float> db_sqr_norms;
  // Decoded int8 query coordinates for the current L2 query batch.
  hwy::AlignedVector<int8_t> q_decoded;
  // Sum of decoded int8 coordinates per query for zero-point offset correction.
  std::vector<int32_t> q_byte_sums;
  // Per-query quantization scale factors for the current L2 query batch.
  std::vector<float> q_norms;
  // Per-query squared L2 norms for the current L2 query batch (used only when
  // metric == kSquaredL2Distance).
  std::vector<float> q_sqr_norms;
  // Intermediate distance output buffer for a micro-batch of queries against
  // an L2 database sub-block.
  hwy::AlignedVector<float> score_buf;
};

// Callback invoked when a batch of scores is ready for a specific query.
//
// All indices passed to the callback are local to the `IndexSubset`s passed
// to ManyToManyTurboQuant, not indices into the `queries` or `database`
// datasets. Use `query_subset[query_idx]` and `db_subset[first_db_idx + i]` to
// map them back to dataset indices. For the PreTransposedTurboQuantDataset
// overloads, database indices are positions in the pretransposed dataset.
//
// Parameters:
//   query_idx: The position of the query within `query_subset`, in
//              `[0, query_subset.size())`.
//   scores: A span of computed scores. `scores[i]` corresponds to the distance
//           between the query at position `query_idx` of `query_subset` and
//           the database point at position `first_db_idx + i` of `db_subset`.
//           By default, these are negative dot products (lower is better).
//   first_db_idx: The position of the first database point of this batch
//                 within `db_subset`, in `[0, db_subset.size())`.
using ManyToManyCallback = absl::FunctionRef<void(
    size_t query_idx, absl::Span<const float> scores, size_t first_db_idx)>;

// Computes approximate distances between all queries and database points
// using TurboQuant 4-bit quantization.
//
// This function implements a high-performance, cache-tiled matrix
// multiplication optimized for SIMD (using Google Highway). It tiles database
// points to fit into L2 cache and processes them in batches. For each database
// batch, it scores a subset of queries.
//
// Requirement:  `queries` and `database` must have the same dimensionality and
// must have been encoded with the same `TurboQuantCodec` quantizer (seed,
// dimensionality, distance metric, etc.).  Dimensionality and metric mismatches
// cause a CHECK-failure.  WARNING: the seed is **not** verified at runtime;
// the caller is responsible for using the same seed for both datasets.
//
// Supported metrics:
// - kDotProductDistance: scores are negative dot products (lower is better).
// - kSquaredL2Distance: scores are squared L2 distances.
//
// Output stream:
// - Results are streamed sequentially via the `callback`. The callback will be
// invoked multiple times per query as database batches are processed.
void ManyToManyTurboQuant(const TurboQuantDatasetSoA& queries,
                          const TurboQuantDatasetSoA& database,
                          IndexSubset query_subset, IndexSubset db_subset,
                          ManyToManyCallback callback);

// Overload that reuses pre-allocated buffers in `workspace` (growing them only
// when necessary) to avoid heap allocations across repeated calls.
void ManyToManyTurboQuant(const TurboQuantDatasetSoA& queries,
                          const TurboQuantDatasetSoA& database,
                          IndexSubset query_subset, IndexSubset db_subset,
                          ManyToManyCallback callback,
                          ManyToManyTurboQuantWorkspace& workspace);

// Overload that operates on a PreTransposedTurboQuantDataset for the database.
//
// Use this when the database has been pre-decoded and block-transposed via
// PreTransposedTurboQuantDataset::Build(). This avoids the per-batch
// decode+transpose cost, which is beneficial when the same database is scored
// against multiple query batches (e.g., repeated ManyToMany calls with
// different query sets against the same database).
//
// The queries are still decoded on the fly from nibble-packed representation.
//
// Requirements (CHECK-fail on violation):
// - `queries` must have the same dimensionality as the pretransposed database.
// - Both must use the same TurboQuantDistanceMetric (kDotProductDistance or
//   kSquaredL2Distance). For L2, the database must have been built with
//   non-empty sqr_norms (i.e., from a kSquaredL2Distance quantizer).
void ManyToManyTurboQuant(const TurboQuantDatasetSoA& queries,
                          const PreTransposedTurboQuantDataset& database,
                          IndexSubset query_subset,
                          ManyToManyCallback callback);

// Overload for PreTransposedTurboQuantDataset that reuses `workspace` buffers
// across calls to avoid per-invocation heap allocations.
void ManyToManyTurboQuant(const TurboQuantDatasetSoA& queries,
                          const PreTransposedTurboQuantDataset& database,
                          IndexSubset query_subset, ManyToManyCallback callback,
                          ManyToManyTurboQuantWorkspace& workspace);

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_MANY_TO_MANY_TURBOQUANT_H_
