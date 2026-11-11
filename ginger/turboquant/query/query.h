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

#ifndef GINGER_TURBOQUANT_QUERY_QUERY_H_
#define GINGER_TURBOQUANT_QUERY_QUERY_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/datapoint/encode_datapoint.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// A query prepared for TurboQuant scoring. Contains the encoded int8 query
// vector and associated metadata needed by the SIMD scoring kernels.
//
// Create via the static factory methods which require a TurboQuantCodec
// to perform the encoding.
class TurboQuantQuery {
 public:
  const int8_t* query_data() const { return query_.data(); }
  absl::Span<const float> float_query_data() const { return float_query_; }
  size_t dimensionality() const { return dimensionality_; }
  size_t num_bytes_per_query() const { return query_.size(); }
  int32_t query_byte_sum() const { return query_byte_sum_; }

  // The norm scaling factor (norm(unquantized) / norm(quantized))
  float norm_scaling_factor() const { return norm_scaling_factor_; }
  float norm_scaling_factor_turboslice() const {
    return norm_scaling_factor_turboslice_;
  }

  // The square norm of the unquantized vector.
  float unquantized_squared_norm() const { return unquantized_squared_norm_; }
  TurboQuantDistanceMetric metric() const { return metric_; }

  // Set `enable_higher_precision_scoring` to true to preserve float query data
  // required for 4+k bit extra precision scoring.
  static absl::StatusOr<TurboQuantQuery> Create(
      const TurboQuantCodec& turboquant, absl::Span<const float> query,
      bool enable_higher_precision_scoring = false);

  // Creates a query prepared solely for 4+k bit extra-precision scoring
  // (`UnsafeComputeDistancesExtraPrecision`), skipping int8 query quantization
  // and allocation.
  static absl::StatusOr<TurboQuantQuery> CreateForExtraPrecision(
      const TurboQuantCodec& turboquant, absl::Span<const float> query);

  static absl::StatusOr<TurboQuantQuery> CreateTurboSliceQuery(
      const TurboQuantCodec& turboquant, absl::Span<const float> query);

  static absl::StatusOr<TurboQuantQuery> CreateTurboSliceQuery(
      const TurboQuantCodec& turboquant, const TurboQuantQuery& query);

  // Creates a query from a quantized datapoint.
  static absl::StatusOr<TurboQuantQuery> CreateFromQuantizedDataPoint(
      const TurboQuantCodec& turboquant,
      const TurboQuantBasePointView& datapoint);

  // Like `CreateFromQuantizedDataPoint`, but also dequantizes the datapoint's
  // 4+k-bit representation into rotated float space so that the query can be
  // used for extra-precision scoring (`UnsafeComputeDistancesExtraPrecision`).
  // Returns an error if the extra precision data is missing or invalid.
  static absl::StatusOr<TurboQuantQuery>
  CreateFromQuantizedDataPointWithExtraPrecision(
      const TurboQuantCodec& turboquant,
      const TurboQuantBasePointView& datapoint);

  // Resets the query from a quantized datapoint, reusing the existing buffers.
  // Clears any float query data.
  absl::Status ResetFromQuantizedDataPoint(
      const TurboQuantCodec& turboquant,
      const TurboQuantBasePointView& datapoint);

  // Like `ResetFromQuantizedDataPoint`, but also populates the float query data
  // for extra-precision scoring (see
  // `CreateFromQuantizedDataPointWithExtraPrecision`). Returns an error if the
  // extra precision data is missing or invalid.
  absl::Status ResetFromQuantizedDataPointWithExtraPrecision(
      const TurboQuantCodec& turboquant,
      const TurboQuantBasePointView& datapoint);

 private:
  TurboQuantQuery() = default;
  TurboQuantQuery(size_t dimensionality, size_t padded_dimensionality)
      : dimensionality_(dimensionality), query_(padded_dimensionality) {}
  // Sets the common metadata fields. Called by all Create*/Reset* methods
  // after encoding the query data.
  void SetMetadata(TurboQuantDistanceMetric metric, float norm_scaling_factor,
                   float unquantized_squared_norm,
                   int32_t query_byte_sum = kQueryByteSumNotSet);
  absl::Status Reset4BitFromQuantizedDataPoint(
      const TurboQuantCodec& turboquant,
      const TurboQuantBasePointView& datapoint);

  size_t dimensionality_ = 0;
  std::vector<int8_t, hwy::AlignedAllocator<int8_t>> query_;
  std::vector<float> float_query_;

  float norm_scaling_factor_ = 0.0f;
  float norm_scaling_factor_turboslice_ = 0.0f;
  float unquantized_squared_norm_ = 0.0f;
  TurboQuantDistanceMetric metric_ =
      TurboQuantDistanceMetric::kDotProductDistance;
  int32_t query_byte_sum_ = 0;
};

// Computes the query byte sum for an encoded int8 query.
int32_t ComputeQueryByteSum(absl::Span<const int8_t> query);

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_QUERY_QUERY_H_
