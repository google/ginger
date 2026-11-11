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

#ifndef GINGER_TURBOQUANT_LAYOUT_DATASET_H_
#define GINGER_TURBOQUANT_LAYOUT_DATASET_H_

// TurboQuant is a quantization scheme with good approximation properties for
// the dot product of dense vectors. See https://arxiv.org/abs/2504.19874 for
// details. This version is a SIMD-optimized one-to-many dot-product scoring
// engine. It uses 4 bits per dimension. It supports an optional additional
// (1, 2, 3, 4, 6, or 8) bits per dimension to improve the accuracy during
// rescoring by having a more refined quantization codebook. Important info:
// rescoring is optional and uses separately stored data, so doesn't slow down
// 4-bit scoring.

// The quantization algorithm works as follows:
// 1) Multiply each coordinate by a random sign, apply a rotation to the
//  coordinates, then normalize and remember the norm.
// 2) Each coordinate is quantized using a fixed codebook of size 16, leading to
// a 4 bit representation per dimension. For each coordinate we snap to the
// closest codebook entry and encode the coordinate as the index in the
// codebook.
// 3) If quantization_bits > 4, then we refine the quantization using a
// codebook with 2^(quantization_bits - 4) entries, and we
// store the index in the codebook in
// output_extra_precision_data.

// Use the method as follows.
// 1) Create a TurboQuantCodec object `tq`.
// 2) Hand `tq` to TurboQuantDatasetSoA::Create() to create a
// collection of quantized datapoints. Using either the Create() function with
// input datapoints or by calling Create() with the number of data points and
// then calling UpdatePoint() on the required indices.
// 3) TurboQuantCodec::EncodeQuery to prepare a query vector for scoring.
// This operation is not cheap and thus should be done once per query.
// 4) TurboQuantDataset::ComputeDistances to compute distances
// (either `-dot product` or `squared L2 distance`) between the query and a
// subset of datapoints. Note that there is also a UnsafeComputeDistances
// function that skips input validation for performance reasons.

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/datapoint/encode_datapoint.h"
#include "ginger/turboquant/decoding/dequantize.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/turboquant/types/quantization_bits.h"
#include "ginger/types.h"

namespace ginger {

struct TurboQuantDatasetParams {
  // The dimensionality of the datapoints.
  // Must be positive.
  size_t dimensionality = 0;

  // The dimensionality of the TurboSlice prefix. See TurboQuantCodec in
  // codec/codec.h for full description of TurboSlice concepts and usage.
  // If set to 0 (default), TurboSlice is disabled.
  size_t turboslice_dimensionality = 0;

  // The number of extra bits per dimension. If 0, the dataset only stores
  // 4-bit quantization data. If > 0, the dataset stores 4-bit quantization
  // data and extra precision quantization data.
  // Valid values are 0, 1, 2, 3, 4, 6, and 8.
  // TODO: Switch this to quantization_bits and update callers.
  int extra_bits_per_dim = 0;

  // The metric to use for scoring.
  TurboQuantDistanceMetric metric =
      TurboQuantDistanceMetric::kDotProductDistance;

  // If true, indicates that datapoints are normalized (||y||^2 = 1).
  // When metric is kSquaredL2Distance and this is true, we use the same memory
  // layout as kDotProductDistance (skipping squared norm storage) and skip
  // squared norm memory reads during scoring.
  bool normalized_datapoints = false;

  // If true, the dataset also needs to store the norm scaling factor for 4-bit
  // precision only scoring, even if extra_bits_per_dim > 0.
  // This could be used for example in approximate nearest neighbor search where
  // we want to filter the datapoints using the cheap 4-bit scores against a
  // threshold, before we decide to use the more expensive extra precision
  // scoring on a smaller subset of the datapoints.
  bool always_support_4bit_precision = false;

  // TODO: Add options to allow less precision norm storage to
  // save memory. e.g., use 1 or 2 bytes for the norm, etc.
  absl::Status Validate() const {
    if (dimensionality == 0) {
      return absl::InvalidArgumentError("Dimensionality must be positive.");
    }
    if (turboslice_dimensionality > dimensionality) {
      return absl::InvalidArgumentError(
          "TurboSlice dimensionality cannot exceed dimensionality.");
    }
    ABSL_RETURN_IF_ERROR(ValidateTurboQuantExtraBits(extra_bits_per_dim));
    return absl::OkStatus();
  }

  // Returns the number of bytes required per datapoint for the 4-bit
  // quantization data.
  // TODO: rename 'num_bytes_per_datapoint' to
  // 'base_bytes_per_datapoint' across the codebase for clarity.
  size_t base_bytes_per_datapoint() const { return (dimensionality + 1) / 2; }

  size_t turboslice_bytes_per_datapoint() const {
    return (turboslice_dimensionality + 1) / 2;
  }

  // Returns the number of bytes required per datapoint for the extra precision
  // data. Excludes the `kPrefixLowBitBytes` of 4+k datasets with a 3-bit
  // prefix; see `TurboQuantDataset::extra_precision_bytes()`.
  size_t extra_bytes_per_datapoint() const {
    return TurboQuantCodec::GetExtraPrecisionBytes(dimensionality,
                                                   extra_bits_per_dim);
  }

  // Returns the total number of bytes required per datapoint.
  // Note that this is only the minimum number of bytes required, the actual
  // usage might be larger depending the dataset data-structure and memory
  // layout design. Like `extra_bytes_per_datapoint()`, it excludes the
  // `kPrefixLowBitBytes` of 4+k datasets with a 3-bit prefix.
  size_t total_bytes_per_datapoint() const {
    int num_norms = 1;
    if (extra_bits_per_dim > 0 && always_support_4bit_precision) {
      num_norms++;
    }
    if (metric == TurboQuantDistanceMetric::kSquaredL2Distance &&
        !normalized_datapoints) {
      num_norms++;
    }
    // Norms are counted at float width: this bound is the AoS row layout
    // contract (see AosBufferSpec), which stores each norm as a float.
    return base_bytes_per_datapoint() + extra_bytes_per_datapoint() +
           num_norms * sizeof(float);
  }

  bool operator==(const TurboQuantDatasetParams&) const = default;

  // Creates a dataset parameters from a quantizer.
  static absl::StatusOr<TurboQuantDatasetParams> FromQuantizer(
      const TurboQuantCodec& turboquant, int quantization_bits,
      bool always_support_4bit_precision, bool normalized_datapoints = false) {
    auto params = TurboQuantDatasetParams{
        .dimensionality = turboquant.dimensionality(),
        .turboslice_dimensionality = turboquant.turboslice_dimensionality(),
        .extra_bits_per_dim = quantization_bits - 4,
        .metric = turboquant.metric(),
        .normalized_datapoints = normalized_datapoints,
        .always_support_4bit_precision = always_support_4bit_precision};
    ABSL_RETURN_IF_ERROR(params.Validate());
    return params;
  }
};

// Represents a collection of quantized datapoints.
class TurboQuantDataset {
 public:
  explicit TurboQuantDataset(const TurboQuantDatasetParams& params)
      : params_(params) {
    ABSL_DCHECK_OK(params_.Validate());
  }

  virtual ~TurboQuantDataset() = default;

  // This function is thread-safe as long as two concurrent calls do not modify
  // the same index. coordinates must have exactly `dimensionality_` entries.
  // Quantizes the datapoints and stores the encoding at the given index.
  // TODO: Add TurboQuantCodec as a member variable, instead
  // of being passed around everywhere, to ensure that the codec is compatible
  // with the dataset.
  virtual absl::Status UpdatePoint(const TurboQuantCodec& turboquant,
                                   absl::Span<const float> coordinates,
                                   size_t index) = 0;

  // Thread-safe as long as two concurrent calls do not
  // modify the same index. Copies quantized datapoint to the given index. The
  // quantized datapoint must have size `num_bytes_per_datapoint`.
  virtual absl::Status UpdatePoint(const TurboQuantBasePointView& datapoint,
                                   size_t index) = 0;

  // Appends a quantized datapoint to the collection. `datapoint` must not
  // point into this dataset's own storage.
  virtual absl::Status AppendPoint(
      const TurboQuantBasePointView& datapoint) = 0;

  // TODO: Remove the codec argument.
  // Appends a raw datapoint to the collection.
  virtual absl::Status AppendPoint(const TurboQuantCodec& turboquant,
                                   absl::Span<const float> coordinates) = 0;

  // Computes distances for the datapoints specified in `indices` and stores
  // them in `scores`. `indices` and `scores` must have the same size. Returns
  // an error if the inputs are invalid. `UnsafeComputeDistances` skips these
  // checks for performance.
  absl::Status ComputeDistances(const TurboQuantQuery& query,
                                absl::Span<const int32_t> indices,
                                absl::Span<float> scores) const {
    if (query.dimensionality() != dimensionality()) {
      return absl::InvalidArgumentError("Query dimensionality does not match.");
    }
    if (query.num_bytes_per_query() == 0) {
      return absl::InvalidArgumentError(
          "Query does not contain 4-bit quantized data.");
    }
    ABSL_ASSIGN_OR_RETURN(
        auto accessor,
        CreateScoringAccessor(indices, scores, num_datapoints()));

    UnsafeComputeDistances(query, accessor);
    return absl::OkStatus();
  }

  absl::Status ComputeDistances(const TurboQuantQuery& query,
                                absl::Span<const uint32_t> indices,
                                absl::Span<float> scores) const {
    if (query.dimensionality() != dimensionality()) {
      return absl::InvalidArgumentError("Query dimensionality does not match.");
    }
    if (query.num_bytes_per_query() == 0) {
      return absl::InvalidArgumentError(
          "Query does not contain 4-bit quantized data.");
    }
    ABSL_ASSIGN_OR_RETURN(
        auto accessor,
        CreateScoringAccessor(indices, scores, num_datapoints()));

    UnsafeComputeDistances(query, accessor);
    return absl::OkStatus();
  }

  absl::Status ComputeDistances(
      const TurboQuantQuery& query,
      absl::Span<std::pair<float, int32_t>> scores_and_indices) const {
    if (query.dimensionality() != dimensionality()) {
      return absl::InvalidArgumentError("Query dimensionality does not match.");
    }
    if (query.num_bytes_per_query() == 0) {
      return absl::InvalidArgumentError(
          "Query does not contain 4-bit quantized data.");
    }
    ABSL_ASSIGN_OR_RETURN(
        auto accessor,
        CreateScoringAccessor(scores_and_indices, num_datapoints()));

    UnsafeComputeDistances(query, accessor);
    return absl::OkStatus();
  }

  absl::Status ComputeDistances(
      const TurboQuantQuery& query,
      absl::Span<::ginger::DistanceAndId> scores_and_indices) const {
    if (query.dimensionality() != dimensionality()) {
      return absl::InvalidArgumentError("Query dimensionality does not match.");
    }
    if (query.num_bytes_per_query() == 0) {
      return absl::InvalidArgumentError(
          "Query does not contain 4-bit quantized data.");
    }
    ABSL_ASSIGN_OR_RETURN(
        auto accessor,
        CreateScoringAccessor(scores_and_indices, num_datapoints()));

    UnsafeComputeDistances(query, accessor);
    return absl::OkStatus();
  }

  // Computes distances for all datapoints.
  absl::Status ComputeAllDatapointsDistances(const TurboQuantQuery& query,
                                             absl::Span<float> scores) const {
    if (query.dimensionality() != dimensionality()) {
      return absl::InvalidArgumentError("Query dimensionality does not match.");
    }
    if (query.num_bytes_per_query() == 0) {
      return absl::InvalidArgumentError(
          "Query does not contain 4-bit quantized data.");
    }
    const size_t num = num_datapoints();
    if (scores.size() != num) {
      return absl::InvalidArgumentError(
          "`scores` must have the same size as the number of datapoints.");
    }

    ScoresAndIndicesContiguousRangeView accessor(0, num, scores);
    UnsafeComputeDistances(query, ScoringAccessor(accessor));
    return absl::OkStatus();
  }

  // Same as ComputeAllDatapointsDistances, but uses extra precision if
  // available.
  absl::Status ComputeAllDatapointsDistancesExtraPrecision(
      const TurboQuantQuery& query, absl::Span<float> scores) const {
    if (quantization_bits() == 4) {
      return ComputeAllDatapointsDistances(query, scores);
    }
    if (query.dimensionality() != dimensionality()) {
      return absl::InvalidArgumentError("Query dimensionality does not match.");
    }
    const size_t num = num_datapoints();
    if (scores.size() != num) {
      return absl::InvalidArgumentError(
          "`scores` must have the same size as the number of datapoints.");
    }
    ScoresAndIndicesContiguousRangeView accessor(0, num, scores);
    UnsafeComputeDistancesExtraPrecision(query, ScoringAccessor(accessor));
    return absl::OkStatus();
  }

  virtual TurboQuantBasePointView DataPointView(size_t index) const = 0;

  // Whether rows carry the merged 3-bit codebook on the odd dimensions of
  // their first `kScaleBitBytes` bytes.
  virtual bool has_3bit_prefix() const { return false; }

  // Dequantizes a datapoint at the given index to approximate float
  // coordinates.
  void DequantizeToRotatedSpace(size_t index, absl::Span<float> output) const {
    ::ginger::DequantizeToRotatedSpace(dimensionality(), extra_bits_per_dim(),
                                       DataPointView(index), output);
  }

  // Extra-precision bytes per datapoint, including the `kPrefixLowBitBytes` of
  // 4+k datasets with a 3-bit prefix.
  size_t extra_precision_bytes() const {
    return GetExtraPrecisionBytes(dimensionality(), extra_bits_per_dim(),
                                  has_3bit_prefix());
  }

  // Offset of the `kPrefixLowBitBytes` within a datapoint's extra-precision
  // data when `has_3bit_prefix()`.
  size_t prefix_low_bits_offset() const {
    ABSL_DCHECK(has_3bit_prefix() && extra_bits_per_dim() > 0);
    return params_.extra_bytes_per_datapoint();
  }

  // Returns the number of datapoints in the collection.
  virtual size_t num_datapoints() const = 0;

  const TurboQuantDatasetParams& params() const { return params_; }

  size_t num_bytes_per_datapoint() const {
    return params_.base_bytes_per_datapoint();
  }
  size_t dimensionality() const { return params_.dimensionality; }
  int extra_bits_per_dim() const { return params_.extra_bits_per_dim; }
  int quantization_bits() const { return params_.extra_bits_per_dim + 4; }
  TurboQuantDistanceMetric metric() const { return params_.metric; }
  bool normalized_datapoints() const { return params_.normalized_datapoints; }

  // The virtual methods to be overridden by the derived classes.
  // The implementations are encouraged to use `DISPATCH_SCORING_ACCESSOR` to
  // dispatch to the correct scoring function based on the accessor type for
  // maximum compiler optimizations opportunities.
  virtual void UnsafeComputeDistances(const TurboQuantQuery& query,
                                      ScoringAccessor accessor) const = 0;

  virtual void UnsafeComputeDistancesTurboSlice(
      const TurboQuantQuery& /*query*/, ScoringAccessor /*accessor*/) const {
    ABSL_LOG(FATAL) << "TurboSlice is not implemented for this dataset layout.";
  }

  virtual void UnsafeComputeDistancesExtraPrecision(
      const TurboQuantQuery& query, ScoringAccessor accessor) const = 0;

 private:
  TurboQuantDatasetParams params_;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_LAYOUT_DATASET_H_
