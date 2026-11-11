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

#ifndef GINGER_TURBOQUANT_CODEC_CODEC_H_
#define GINGER_TURBOQUANT_CODEC_CODEC_H_

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/datapoint/encode_datapoint.h"
#include "ginger/turboquant/query/encode_query.h"
#include "ginger/turboquant/rotation/rotation.h"
#include "ginger/turboquant/types/distance_metric.h"

namespace ginger {

class TurboQuantCodec {
 public:
  using EncodeQueryResult = ::ginger::EncodeQueryResult;
  using EncodeDatapointResult = ::ginger::EncodeDatapointResult;

  // Constructs a TurboQuant codec for the given dimensionality and random seed.
  //
  // `turboslice_dimensionality > 0` enables TurboSlice, a two-phase scoring
  // optimization. When enabled
  // (turboslice_dimensionality > 0), it allows computing approximate distances
  // using only a prefix of the quantized dimensions, which is cheaper than
  // scoring with all dimensions. This is useful for quickly pruning candidates
  // before computing full-precision distances on the survivors.
  // Note: TurboSlice distances are rescaled so that their expected value range
  // matches the full-dimensional distance, allowing direct comparison. However,
  // because scores are estimated from only a small prefix of dimensions, they
  // exhibit higher approximation variance than full-dimensional scores. Use
  // with caution!
  //
  // Usage:
  //   1. Pass a non-zero `turboslice_dimensionality` here to enable TurboSlice.
  //      Datasets created from this codec via TurboQuantDatasetSoA::Create
  //      will automatically store TurboSlice norm data.
  //   2. Create TurboSlice queries via
  //      TurboQuantQuery::CreateTurboSliceQuery.
  //   3. Score with
  //      TurboQuantDataset::UnsafeComputeDistancesTurboSlice.
  //
  // `turboslice_dimensionality` must be <= `dimensionality`. Set to 0 (default)
  // to disable TurboSlice.
  TurboQuantCodec(size_t dimensionality, uint32_t seed,
                  TurboQuantDistanceMetric metric =
                      TurboQuantDistanceMetric::kDotProductDistance,
                  size_t turboslice_dimensionality = 0);

  // `coordinates` must have exactly `dimensionality_` entries.
  // `output_coordinates` must have exactly `num_bytes_per_datapoint()` entries.
  // The encoded datapoint is stored in output_coordinates,
  // and the squared norm of the original coordinates, the ratio of the original
  // norm and the 4 bit quantized norm and the ratio of the original norm and
  // the 4+k bit quantized norm is returned in the EncodeDatapointResult struct.
  //
  // quantization_bits parameter below is the total number of bits per dimension
  // used for quantization. The extra bits k are stored in
  // output_extra_precision_data. It defaults to 4 and currently setting it to
  // 4, 5, 6, 7, 8, 10, or 12 is supported.
  // Set `has_3bit_prefix` to true when encoding for a dataset that uses
  // `ScalingStorageMode::kIn3BitPrefix` (i.e., `dataset.has_3bit_prefix()`).
  absl::StatusOr<EncodeDatapointResult> EncodeDatapointStructured(
      absl::Span<const float> coordinates,
      absl::Span<uint8_t> output_coordinates, int quantization_bits = 4,
      absl::Span<uint8_t> output_extra_precision_data = {},
      bool has_3bit_prefix = false) const {
    return ginger::internal::EncodeDatapointStructured(
        coordinates, output_coordinates, quantization_bits,
        output_extra_precision_data, dimensionality_, num_bytes_per_datapoint_,
        turboslice_dimensionality_, rotation_, has_3bit_prefix);
  }

  // Prepares the query vector for scoring. Rotates and scales the vector, and
  // reorders coordinates to the format expected by the scoring loop.
  // `query` must have length `dimensionality_`. `encoded_query` must have
  // length `num_bytes_per_query()`. This is called from
  // TurboQuantQuery::Create. Returns norm multipliers and the squared
  // norm of the query.
  absl::StatusOr<EncodeQueryResult> EncodeQuery(
      absl::Span<const float> query, absl::Span<int8_t> encoded_query,
      absl::Span<float> rotated_float_query = {}) const {
    return ginger::internal::EncodeQuery(
        query, encoded_query, rotated_float_query, dimensionality_,
        num_bytes_per_query(), num_bytes_per_datapoint_, rotation_,
        turboslice_dimensionality_);
  }

  // Prepares the query vector solely for extra-precision (4+k bit) scoring,
  // skipping int8 query quantization and deinterleaving.
  // `query` and `rotated_float_query` must have length `dimensionality_`.
  absl::StatusOr<EncodeQueryResult> EncodeQueryForExtraPrecision(
      absl::Span<const float> query,
      absl::Span<float> rotated_float_query) const {
    return ginger::internal::EncodeQueryForExtraPrecision(
        query, rotated_float_query, dimensionality_, rotation_);
  }

  // Prepares the TurboSlice query vector for scoring. Rotates and scales the
  // vector using the full vector dimensionality, and quantizes the prefix of
  // `turboslice_dimensionality()` coordinates.
  // `query` must have length `dimensionality()`.
  // `encoded_turboslice_query` must have length
  // `num_bytes_per_turboslice_query()`.
  absl::StatusOr<EncodeQueryResult> EncodeTurboSliceQuery(
      absl::Span<const float> query,
      absl::Span<int8_t> encoded_turboslice_query) const {
    return ginger::internal::EncodeTurboSliceQuery(
        query, encoded_turboslice_query, dimensionality_,
        turboslice_dimensionality_, num_bytes_per_turboslice_query(),
        turboslice_bytes_per_datapoint_, rotation_);
  }

  // Encodes a quantized datapoint as a query vector.
  // This is useful to implement symmetric scoring (base against base point)
  // using asymmetric scoring. Notably, this is as efficient as a custom
  // implementation. `encoded_query` must have length `num_bytes_per_query()`.
  absl::Status EncodeQuantizedDataPointAsQuery(
      absl::Span<const uint8_t> quantized_point,
      absl::Span<int8_t> encoded_query, bool has_3bit_prefix) const {
    return ginger::internal::EncodeQuantizedDataPointAsQuery(
        quantized_point, encoded_query, num_bytes_per_datapoint_,
        num_bytes_per_query(), dimensionality_, has_3bit_prefix);
  }

  // Multiplies each coordinate by a random sign, normalizes and rotates the
  // coordinates. Expects the norm of the input coordinates as input,
  // since this is computed before calling this function for both query and base
  // points, but the error handling differs.
  absl::Status FlipRandomSignsNormalizeAndRotate(
      absl::Span<const float> coordinates, float norm,
      absl::Span<float> rotated_coordinates) const {
    return rotation_.FlipRandomSignsNormalizeAndRotate(coordinates, norm,
                                                       rotated_coordinates);
  }

  size_t dimensionality() const { return dimensionality_; }
  size_t turboslice_dimensionality() const {
    return turboslice_dimensionality_;
  }
  uint32_t seed() const { return seed_; }
  size_t num_bytes_per_datapoint() const { return num_bytes_per_datapoint_; }
  size_t turboslice_bytes_per_datapoint() const {
    return turboslice_bytes_per_datapoint_;
  }

  // Returns the number of bytes required for the `encoded_query` span in
  // `EncodeQuery` and `EncodeQuantizedDataPointAsQuery`.
  size_t num_bytes_per_query() const;
  size_t num_bytes_per_turboslice_query() const;
  TurboQuantDistanceMetric metric() const { return metric_; }

  // Returns the padding used for query coordinates.
  static size_t QueryPadding();

  // Returns the number of bytes required for `extra_bits_per_dim` extra
  // precision bits.
  static size_t GetExtraPrecisionBytes(size_t dimensionality,
                                       int extra_bits_per_dim);

  // Infer the number of bytes per datapoint for the base 4-bit quantization and
  // the additional precision data, given extra bits per dimension and total
  // stored bytes. Supports extra_bits_per_dim in {0, 1, 2, 3, 4, 6, 8}.
  // Check-fails for other values. Assumes no 3-bit prefix, i.e., no
  // `kPrefixLowBitBytes` in the additional precision data.
  static std::pair<size_t, size_t> InferBaseAndAdditionalPrecisionBytes(
      size_t total_bytes, int extra_bits_per_dim);

 protected:
  size_t dimensionality_;
  size_t turboslice_dimensionality_;
  uint32_t seed_;
  size_t num_bytes_per_datapoint_;
  size_t turboslice_bytes_per_datapoint_;
  HadamardAndMatrixRotation rotation_;
  TurboQuantDistanceMetric metric_;
};

// Convenience method intended for encoding one datapoint at a time in
// streams. Allocates the output buffer internally and returns an owning
// TurboQuantBasePoint.
// `quantization_bits` can be in {4, 5, 6, 7, 8, 10, 12}.
absl::StatusOr<TurboQuantBasePoint> EncodeDatapoint(
    const TurboQuantCodec& turboquant, absl::Span<const float> coordinates,
    int quantization_bits, bool has_3bit_prefix);

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_CODEC_CODEC_H_
