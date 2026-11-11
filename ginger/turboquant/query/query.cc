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

#include "ginger/turboquant/query/query.h"

#include <cstdint>
#include <cstring>
#include <numeric>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/datapoint/encode_datapoint.h"
#include "ginger/turboquant/decoding/dequantize.h"
#include "ginger/turboquant/query/encode_query.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/turboquant/types/quantization_bits.h"
#include "hwy/base.h"

namespace ginger {

int32_t ComputeQueryByteSum(absl::Span<const int8_t> query) {
  return std::accumulate(query.begin(), query.end(), int32_t{0});
}

void TurboQuantQuery::SetMetadata(TurboQuantDistanceMetric metric,
                                  float norm_scaling_factor,
                                  float unquantized_squared_norm,
                                  int32_t query_byte_sum) {
  metric_ = metric;
  norm_scaling_factor_ = norm_scaling_factor;
  unquantized_squared_norm_ = unquantized_squared_norm;
  query_byte_sum_ = (query_byte_sum == kQueryByteSumNotSet)
                        ? ComputeQueryByteSum(query_)
                        : query_byte_sum;
}

absl::StatusOr<TurboQuantQuery> TurboQuantQuery::Create(
    const TurboQuantCodec& turboquant, absl::Span<const float> query,
    bool enable_higher_precision_scoring) {
  TurboQuantQuery q(turboquant.dimensionality(),
                    turboquant.num_bytes_per_query());
  absl::Span<float> float_query_span = {};
  if (enable_higher_precision_scoring) {
    q.float_query_.resize(turboquant.dimensionality());
    float_query_span = absl::MakeSpan(q.float_query_);
  }
  ABSL_ASSIGN_OR_RETURN(auto encode_query_result,
                        turboquant.EncodeQuery(query, absl::MakeSpan(q.query_),
                                               float_query_span));
  q.SetMetadata(turboquant.metric(), encode_query_result.norm_scaling_factor,
                encode_query_result.unquantized_squared_norm);
  q.norm_scaling_factor_turboslice_ =
      encode_query_result.norm_scaling_factor_turboslice;
  return q;
}

absl::StatusOr<TurboQuantQuery> TurboQuantQuery::CreateForExtraPrecision(
    const TurboQuantCodec& turboquant, absl::Span<const float> query) {
  TurboQuantQuery q;
  q.dimensionality_ = turboquant.dimensionality();
  q.float_query_.resize(turboquant.dimensionality());
  ABSL_ASSIGN_OR_RETURN(auto encode_query_result,
                        turboquant.EncodeQueryForExtraPrecision(
                            query, absl::MakeSpan(q.float_query_)));
  q.SetMetadata(turboquant.metric(), encode_query_result.norm_scaling_factor,
                encode_query_result.unquantized_squared_norm,
                /*query_byte_sum=*/0);
  return q;
}

absl::StatusOr<TurboQuantQuery> TurboQuantQuery::CreateTurboSliceQuery(
    const TurboQuantCodec& turboquant, absl::Span<const float> query) {
  TurboQuantQuery q(turboquant.turboslice_dimensionality(),
                    turboquant.num_bytes_per_turboslice_query());
  ABSL_ASSIGN_OR_RETURN(
      auto encode_query_result,
      turboquant.EncodeTurboSliceQuery(query, absl::MakeSpan(q.query_)));
  q.SetMetadata(turboquant.metric(),
                encode_query_result.norm_scaling_factor_turboslice,
                encode_query_result.unquantized_squared_norm);
  return q;
}

absl::StatusOr<TurboQuantQuery> TurboQuantQuery::CreateTurboSliceQuery(
    const TurboQuantCodec& turboquant, const TurboQuantQuery& query) {
  if (turboquant.turboslice_dimensionality() == 0) {
    return absl::InvalidArgumentError(
        "TurboSlice dimensionality not configured.");
  }
  if (query.dimensionality() != turboquant.dimensionality()) {
    return absl::InvalidArgumentError(
        "Query dimensionality does not match TurboQuant.");
  }
  if (query.num_bytes_per_query() == 0) {
    return absl::InvalidArgumentError(
        "Query does not contain 4-bit quantized data.");
  }
  TurboQuantQuery q(turboquant.turboslice_dimensionality(),
                    turboquant.num_bytes_per_turboslice_query());
  // Even dims are stored in the first half of the query, odd dims in the
  // second half, one byte each.
  const size_t ts_dim = turboquant.turboslice_dimensionality();
  const size_t num_even_dims = (ts_dim + 1) / 2;
  const size_t num_odd_dims = ts_dim / 2;
  const size_t ts_odd_offset = q.num_bytes_per_query() / 2;
  std::memcpy(q.query_.data(), query.query_data(), num_even_dims);
  std::memcpy(q.query_.data() + ts_odd_offset,
              query.query_data() + query.num_bytes_per_query() / 2,
              num_odd_dims);
  q.SetMetadata(query.metric(), query.norm_scaling_factor_turboslice(),
                query.unquantized_squared_norm());
  return q;
}

absl::StatusOr<TurboQuantQuery> TurboQuantQuery::CreateFromQuantizedDataPoint(
    const TurboQuantCodec& turboquant,
    const TurboQuantBasePointView& datapoint) {
  TurboQuantQuery q(turboquant.dimensionality(),
                    turboquant.num_bytes_per_query());
  ABSL_RETURN_IF_ERROR(q.ResetFromQuantizedDataPoint(turboquant, datapoint));
  return q;
}

absl::StatusOr<TurboQuantQuery>
TurboQuantQuery::CreateFromQuantizedDataPointWithExtraPrecision(
    const TurboQuantCodec& turboquant,
    const TurboQuantBasePointView& datapoint) {
  TurboQuantQuery q(turboquant.dimensionality(),
                    turboquant.num_bytes_per_query());
  ABSL_RETURN_IF_ERROR(
      q.ResetFromQuantizedDataPointWithExtraPrecision(turboquant, datapoint));
  return q;
}

absl::Status TurboQuantQuery::Reset4BitFromQuantizedDataPoint(
    const TurboQuantCodec& turboquant,
    const TurboQuantBasePointView& datapoint) {
  // The query may be reused with a different codec.
  dimensionality_ = turboquant.dimensionality();
  query_.resize(turboquant.num_bytes_per_query());
  ABSL_RETURN_IF_ERROR(turboquant.EncodeQuantizedDataPointAsQuery(
      datapoint.quantized_data, absl::MakeSpan(query_),
      datapoint.has_3bit_prefix));
  SetMetadata(turboquant.metric(), datapoint.norm_scaling_factor_4bit,
              datapoint.unquantized_squared_norm, datapoint.query_byte_sum);
  norm_scaling_factor_turboslice_ = datapoint.norm_scaling_factor_turboslice;
  return absl::OkStatus();
}

absl::Status TurboQuantQuery::ResetFromQuantizedDataPoint(
    const TurboQuantCodec& turboquant,
    const TurboQuantBasePointView& datapoint) {
  ABSL_RETURN_IF_ERROR(Reset4BitFromQuantizedDataPoint(turboquant, datapoint));
  float_query_.clear();
  return absl::OkStatus();
}

absl::Status TurboQuantQuery::ResetFromQuantizedDataPointWithExtraPrecision(
    const TurboQuantCodec& turboquant,
    const TurboQuantBasePointView& datapoint) {
  if (datapoint.extra_bits_per_dim <= 0 ||
      datapoint.extra_precision_data.empty()) {
    return absl::InvalidArgumentError("Datapoint has no extra precision data.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateTurboQuantExtraBits(datapoint.extra_bits_per_dim));
  if (datapoint.extra_precision_data.size() !=
      GetExtraPrecisionBytes(turboquant.dimensionality(),
                             datapoint.extra_bits_per_dim,
                             datapoint.has_3bit_prefix)) {
    return absl::InvalidArgumentError(
        "Extra precision data size does not match dimensionality and "
        "extra_bits_per_dim.");
  }
  ABSL_RETURN_IF_ERROR(Reset4BitFromQuantizedDataPoint(turboquant, datapoint));
  float_query_.resize(turboquant.dimensionality());
  DequantizeToRotatedSpace(turboquant.dimensionality(),
                           datapoint.extra_bits_per_dim, datapoint,
                           absl::MakeSpan(float_query_));
  return absl::OkStatus();
}

}  // namespace ginger
