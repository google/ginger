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

#ifndef GINGER_CANDIDATE_FILTERING_JL_H_
#define GINGER_CANDIDATE_FILTERING_JL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/searcher.pb.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// Packed 1-bit Johnson-Lindenstrauss (JL) representation for a query vector.
// Holds packed 64-bit words representing sign bits from the sparse JL
// projection, aligned using hwy::AlignedVector.
class JlQuery {
 public:
  JlQuery() = default;
  explicit JlQuery(size_t num_words) : words_(num_words) {}

  absl::Span<uint64_t> mutable_data() { return absl::MakeSpan(words_); }
  absl::Span<const uint64_t> data() const {
    return absl::MakeConstSpan(words_);
  }
  size_t num_words() const { return words_.size(); }

 private:
  hwy::AlignedVector<uint64_t> words_;
};

// Manages sparse Johnson-Lindenstrauss random projection, 1-bit quantization,
// and contiguous packed bitset storage for dataset points.
class JlDataset {
 public:
  JlDataset(const JlDataset&) = delete;
  JlDataset& operator=(const JlDataset&) = delete;
  JlDataset(JlDataset&&) = default;
  JlDataset& operator=(JlDataset&&) = default;

  // Creates a JlDataset and quantizes all points from the input SpanDataset.
  //
  // Preconditions:
  // - `jl_dim` must be positive and a multiple of 64.
  // - `metric` must be `DOT_PRODUCT` or `SQUARED_L2`.
  static absl::StatusOr<std::unique_ptr<JlDataset>> Create(
      int32_t jl_dim, int32_t seed, DistanceMetric metric,
      const SpanDataset& dataset);

  // Creates an empty JlDataset with allocated storage for `num_points`.
  //
  // Preconditions:
  // - `source_dim > 0`, `jl_dim > 0` (multiple of 64), `seed >= 0`.
  // - `metric` must be `DOT_PRODUCT` or `SQUARED_L2`.
  static absl::StatusOr<std::unique_ptr<JlDataset>> Create(
      int32_t source_dim, int32_t jl_dim, int32_t seed, DistanceMetric metric,
      size_t num_points);

  // Creates a JlDataset from pre-populated quantized storage (e.g., for
  // deserialization).
  //
  // Preconditions:
  // - Same as above, and `data.size()` must equal `num_points * (jl_dim / 64)`.
  static absl::StatusOr<std::unique_ptr<JlDataset>> Create(
      int32_t source_dim, int32_t jl_dim, int32_t seed, DistanceMetric metric,
      size_t num_points, hwy::AlignedVector<uint64_t> data);

  // Quantizes a single query vector into a packed JlQuery bitset.
  JlQuery QuantizeQuery(absl::Span<const float> query) const;

  // Quantizes a single point vector into packed uint64_t words using a sparse
  // Johnson-Lindenstrauss random projection.
  //
  // Each output bit is computed as the sign of a sparse dot product over a
  // fixed small subset of random coordinates (5 or 6 non-zeros per bit
  // depending on the metric).
  //
  // `output.size()` must equal `words_per_point()`.
  void QuantizePoint(absl::Span<const float> point,
                     absl::Span<uint64_t> output) const;

  // Updates the JL quantized representation of a single datapoint.
  void UpdatePoint(GingerNodeId id, absl::Span<const float> point);

  // Gets the packed uint64_t slice for a specific point.
  absl::Span<const uint64_t> GetPoint(GingerNodeId id) const;

  // Computes Hamming distances between `query` and all `candidates` in place.
  // `candidate.distance` is updated to the Hamming distance (cast to float).
  void UnsafeComputeDistances(const JlQuery& query,
                              absl::Span<DistanceAndId> candidates) const;

  // Permutes datapoints in place according to `permutation`.
  void ReorderInPlace(absl::Span<const GingerNodeId> permutation);

  // Returns a new JlDataset with permuted datapoints.
  std::unique_ptr<JlDataset> Reorder(
      absl::Span<const GingerNodeId> permutation) const;

  size_t size() const { return num_points_; }
  size_t words_per_point() const { return words_per_point_; }
  int32_t source_dim() const { return source_dim_; }
  DistanceMetric metric() const { return metric_; }
  int32_t seed() const { return seed_; }
  int32_t jl_dim() const { return static_cast<int32_t>(words_per_point_ * 64); }
  absl::Span<const uint64_t> data() const { return absl::MakeConstSpan(data_); }

 private:
  JlDataset(int32_t source_dim, int32_t seed, size_t words_per_point,
            DistanceMetric metric, std::vector<int32_t> indices,
            std::vector<int8_t> signs, size_t num_points,
            hwy::AlignedVector<uint64_t> data);

  int32_t source_dim_ = 0;
  int32_t seed_ = 0;
  DistanceMetric metric_ = UNDEFINED_DISTANCE;
  size_t words_per_point_ = 0;
  int32_t non_zeros_ = 0;
  std::vector<int32_t> indices_;
  std::vector<int8_t> signs_;

  size_t num_points_ = 0;
  hwy::AlignedVector<uint64_t> data_;
};

// Functor for search-time JL distance scoring, compatible with CandidateFilter.
struct JlSearchScorer {
  const JlDataset* dataset;
  const JlQuery* query;

  void operator()(absl::Span<DistanceAndId> candidates) const {
    dataset->UnsafeComputeDistances(*query, candidates);
  }
};

}  // namespace ginger

#endif  // GINGER_CANDIDATE_FILTERING_JL_H_
