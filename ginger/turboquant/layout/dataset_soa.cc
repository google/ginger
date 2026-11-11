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

#include "ginger/turboquant/layout/dataset_soa.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/internal/parallel_for_with_status.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/datapoint/encode_datapoint.h"
#include "ginger/turboquant/layout/dataset.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/score_4bit_soa.h"
#include "ginger/turboquant/scoring/score_extra_precision_soa.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/turboquant/types/quantization_bits.h"
#include "hwy/base.h"

namespace ginger {
namespace {

// The last row is extended to a full cache line so that SIMD loads may read
// past the coded bytes.
size_t QuantizedDataBytes(size_t num_datapoints, size_t stride,
                          size_t coded_bytes) {
  if (num_datapoints == 0) return 0;
  return (num_datapoints - 1) * stride +
         std::max(stride,
                  hwy::RoundUpTo(coded_bytes,
                                 static_cast<size_t>(ABSL_CACHELINE_SIZE)));
}

}  // namespace

void TurboQuantDatasetSoA::WritePointNorms(size_t index,
                                           float norm_scaling_factor_4bit,
                                           float norm_scaling_factor_turboslice,
                                           float unquantized_squared_norm) {
  // Zero the row padding (and the last row's SIMD guard bytes) first, so that
  // an embedded norm entry written into the padding is not overwritten.
  uint8_t* row = quantized_data_.data() + index * stride_;
  const size_t coded_bytes = num_bytes_per_datapoint();
  const size_t row_end = (index + 1 >= num_datapoints_)
                             ? quantized_data_.size() - index * stride_
                             : stride_;
  std::memset(row + coded_bytes, 0, row_end - coded_bytes);
  if (scaling_storage_mode_ == ScalingStorageMode::kRowTail) {
    WriteNormEntry(row + stride_ - NormEntryBytes(norm_layout_), norm_layout_,
                   norm_scaling_factor_4bit, unquantized_squared_norm);
  } else if (scaling_storage_mode_ == ScalingStorageMode::kIn3BitPrefix) {
    const float embedded_scale = IsTurboSliceEnabled()
                                     ? norm_scaling_factor_turboslice
                                     : norm_scaling_factor_4bit;
    PackScaleBits(row, hwy::F16FromF32(embedded_scale));
  }
  if (scaling_storage_mode_4bit() == ScalingStorageMode::kSeparateArray) {
    norm_data_->Set(index, norm_scaling_factor_4bit, unquantized_squared_norm);
  }
  if (IsTurboSliceEnabled() &&
      scaling_storage_mode_turboslice() == ScalingStorageMode::kSeparateArray) {
    turboslice_norm_data_->Set(index, norm_scaling_factor_turboslice,
                               unquantized_squared_norm);
  }
}

TurboQuantDatasetSoA::TurboQuantDatasetSoA(
    const TurboQuantDatasetParams& params, size_t num_datapoints, size_t stride,
    ScalingStorageMode scaling_storage_mode, bool store_query_byte_sums)
    : TurboQuantDataset(params),
      stride_(stride),
      norm_layout_(NormLayoutFor(params.metric, params.normalized_datapoints)),
      scaling_storage_mode_(scaling_storage_mode),
      store_query_byte_sums_(store_query_byte_sums),
      num_datapoints_(num_datapoints),
      quantized_data_(QuantizedDataBytes(num_datapoints, stride_,
                                         params.base_bytes_per_datapoint())),
      norm_data_(std::make_unique<NormStorage>(
          scaling_storage_mode_4bit() == ScalingStorageMode::kSeparateArray
              ? num_datapoints
              : 0,
          norm_layout_)),
      query_byte_sums_(store_query_byte_sums ? num_datapoints : 0,
                       kQueryByteSumNotSet) {
  if (IsTurboSliceEnabled()) {
    turboslice_norm_data_ = std::make_unique<NormStorage>(
        scaling_storage_mode_turboslice() == ScalingStorageMode::kSeparateArray
            ? num_datapoints
            : 0,
        norm_layout_);
  }
  if (quantization_bits() > 4) {
    extra_precision_data_.resize(num_datapoints * extra_precision_stride());
  }
}

TurboQuantDatasetSoA::~TurboQuantDatasetSoA() = default;
TurboQuantDatasetSoA::TurboQuantDatasetSoA(TurboQuantDatasetSoA&&) = default;
TurboQuantDatasetSoA& TurboQuantDatasetSoA::operator=(TurboQuantDatasetSoA&&) =
    default;

void TurboQuantDatasetSoA::EnableQueryByteSums() {
  if (store_query_byte_sums_) {
    return;
  }
  store_query_byte_sums_ = true;
  query_byte_sums_.resize(num_datapoints(), kQueryByteSumNotSet);
  for (size_t i = 0; i < num_datapoints(); ++i) {
    query_byte_sums_[i] = ComputeQueryByteSum(
        dimensionality(),
        absl::MakeConstSpan(quantized_data_)
            .subspan(i * stride_, num_bytes_per_datapoint()),
        has_3bit_prefix());
  }
}

void TurboQuantDatasetSoA::DropQueryByteSums() {
  store_query_byte_sums_ = false;
  std::vector<int32_t>().swap(query_byte_sums_);
}

TurboQuantBasePointView TurboQuantDatasetSoA::DataPointView(
    size_t index) const {
  ABSL_CHECK_LT(index, num_datapoints());
  const size_t extra_stride = extra_precision_stride();
  const size_t extra_bytes = extra_precision_bytes();
  const float scale_4bit = norm_scaling_factor(index);
  return TurboQuantBasePointView(
      {.quantized_data =
           absl::MakeConstSpan(quantized_data_)
               .subspan(index * stride_, num_bytes_per_datapoint()),
       .extra_precision_data =
           quantization_bits() > 4
               ? absl::MakeConstSpan(extra_precision_data_)
                     .subspan(index * extra_stride, extra_bytes)
               : absl::Span<const uint8_t>(),
       .norm_scaling_factor_4bit = scale_4bit,
       .norm_scaling_factor_4plusk = quantization_bits() > 4
                                         ? norm_scaling_factor_4plusk(index)
                                         : scale_4bit,
       // unquantized_squared_norm returns 1 for normalized datasets, which
       // store none. Dot product has no squared-norm term, and reports 0.
       .unquantized_squared_norm =
           metric() == TurboQuantDistanceMetric::kSquaredL2Distance
               ? unquantized_squared_norm(index)
               : 0.0f,
       .norm_scaling_factor_turboslice = norm_scaling_factor_turboslice(index),
       .query_byte_sum = store_query_byte_sums_ ? query_byte_sums_[index]
                                                : kQueryByteSumNotSet,
       .has_3bit_prefix = has_3bit_prefix(),
       .extra_bits_per_dim = extra_bits_per_dim()});
}

void TurboQuantDatasetSoA::UnsafeComputeDistances(
    const TurboQuantQuery& query, ScoringAccessor accessor) const {
  ComputeDistances4BitSoA(*this, query, accessor,
                          /*is_turboslice_scoring=*/false);
}

void TurboQuantDatasetSoA::UnsafeComputeDistancesTurboSlice(
    const TurboQuantQuery& query, ScoringAccessor accessor) const {
  ABSL_CHECK(IsTurboSliceEnabled()) << "TurboSlice not enabled on dataset.";
  ComputeDistances4BitSoA(*this, query, accessor,
                          /*is_turboslice_scoring=*/true);
}

void TurboQuantDatasetSoA::UnsafeComputeDistancesExtraPrecision(
    const TurboQuantQuery& query, ScoringAccessor accessor) const {
  if (quantization_bits() == 4) {
    UnsafeComputeDistances(query, accessor);
    return;
  }

  ComputeDistancesExtraPrecisionSoA(*this, query, accessor);
}

absl::StatusOr<TurboQuantDatasetSoA> TurboQuantDatasetSoA::Create(
    const TurboQuantCodec& turboquant,
    absl::Span<const absl::Span<const float>> dense_datapoints) {
  return Create(turboquant, dense_datapoints, TurboQuantDatasetOptions{});
}

absl::StatusOr<TurboQuantDatasetSoA> TurboQuantDatasetSoA::Create(
    const TurboQuantCodec& turboquant,
    absl::Span<const absl::Span<const float>> dense_datapoints,
    const TurboQuantDatasetOptions& options) {
  if (dense_datapoints.empty()) {
    return absl::InvalidArgumentError("No datapoints provided.");
  }
  ABSL_ASSIGN_OR_RETURN(auto base_points,
                        TurboQuantDatasetSoA::Create(
                            turboquant, dense_datapoints.size(), options));
  ABSL_RETURN_IF_ERROR(::ginger::parallel_for_with_status(
      0, dense_datapoints.size(), [&, dense_datapoints](size_t i) {
        return base_points.UpdatePoint(turboquant, dense_datapoints[i], i);
      }));
  return base_points;
}

absl::StatusOr<TurboQuantDatasetSoA> TurboQuantDatasetSoA::Create(
    const TurboQuantCodec& turboquant, size_t num_datapoints) {
  return Create(turboquant, num_datapoints, TurboQuantDatasetOptions{});
}

absl::StatusOr<TurboQuantDatasetSoA> TurboQuantDatasetSoA::Create(
    const TurboQuantCodec& turboquant, size_t num_datapoints,
    const TurboQuantDatasetOptions& options) {
  ABSL_RETURN_IF_ERROR(ValidateTurboQuantBits(options.quantization_bits));
  ABSL_ASSIGN_OR_RETURN(auto params,
                        TurboQuantDatasetParams::FromQuantizer(
                            turboquant, options.quantization_bits,
                            /*always_support_4bit_precision=*/false,
                            options.normalized_datapoints));
  const NormLayout norm_layout =
      NormLayoutFor(params.metric, params.normalized_datapoints);
  if (options.scaling_storage_mode == ScalingStorageMode::kIn3BitPrefix) {
    if (!CanUse3BitPrefix(norm_layout, params.dimensionality)) {
      return absl::InvalidArgumentError(
          "kIn3BitPrefix requires the kScalingOnly norm layout and at least 32 "
          "dimensions.");
    }
    if (params.turboslice_dimensionality > 0 &&
        params.turboslice_dimensionality < kScaleBitDims) {
      return absl::InvalidArgumentError(
          "kIn3BitPrefix with TurboSlice requires at least 32 TurboSlice "
          "dimensions.");
    }
  }
  const size_t stride = RowStrideFor(
      params.base_bytes_per_datapoint(), norm_layout,
      options.pad_to_cache_line_size, options.scaling_storage_mode);
  return TurboQuantDatasetSoA(params, num_datapoints, stride,
                              options.scaling_storage_mode,
                              options.store_query_byte_sums);
}

TurboQuantDatasetSoA TurboQuantDatasetSoA::UninitializedLike(
    const TurboQuantDatasetSoA& other) {
  return TurboQuantDatasetSoA(other.params(), other.num_datapoints(),
                              other.stride(), other.scaling_storage_mode(),
                              other.store_query_byte_sums());
}

void TurboQuantDatasetSoA::GrowStorageForAppend() {
  const size_t num_rows = num_datapoints() + 1;
  quantized_data_.resize(
      QuantizedDataBytes(num_rows, stride_, num_bytes_per_datapoint()), 0);
  if (scaling_storage_mode_4bit() == ScalingStorageMode::kSeparateArray) {
    norm_data_->Resize(num_rows);
  }
  if (IsTurboSliceEnabled() &&
      scaling_storage_mode_turboslice() == ScalingStorageMode::kSeparateArray) {
    turboslice_norm_data_->Resize(num_rows);
  }
  if (quantization_bits() > 4) {
    extra_precision_data_.resize(num_rows * extra_precision_stride(), 0);
  }
  if (store_query_byte_sums_) {
    query_byte_sums_.resize(num_rows, kQueryByteSumNotSet);
  }
}

namespace {
bool IsPointerInBuffer(absl::Span<const uint8_t> span,
                       absl::Span<const uint8_t> buffer) {
  if (span.empty() || buffer.empty()) return false;
  std::less<const uint8_t*> less;
  return !less(span.data(), buffer.data()) &&
         less(span.data(), buffer.data() + buffer.size());
}
}  // namespace

absl::Status TurboQuantDatasetSoA::AppendPoint(
    const TurboQuantBasePointView& datapoint) {
  ABSL_DCHECK(!IsPointerInBuffer(datapoint.quantized_data, quantized_data_));
  ABSL_DCHECK(!IsPointerInBuffer(datapoint.extra_precision_data,
                                 extra_precision_data_));
  GrowStorageForAppend();
  ABSL_RETURN_IF_ERROR(WritePoint(datapoint, num_datapoints_));
  ++num_datapoints_;
  return absl::OkStatus();
}

absl::Status TurboQuantDatasetSoA::AppendPoint(
    const TurboQuantCodec& turboquant, absl::Span<const float> coordinates) {
  GrowStorageForAppend();
  ABSL_RETURN_IF_ERROR(WritePoint(turboquant, coordinates, num_datapoints_));
  ++num_datapoints_;
  return absl::OkStatus();
}

absl::Status TurboQuantDatasetSoA::UpdatePoint(
    const TurboQuantCodec& turboquant, absl::Span<const float> coordinates,
    size_t index) {
  if (index >= num_datapoints()) {
    return absl::InvalidArgumentError(
        "Index is larger than the number of datapoints.");
  }
  return WritePoint(turboquant, coordinates, index);
}

absl::Status TurboQuantDatasetSoA::WritePoint(
    const TurboQuantCodec& turboquant, absl::Span<const float> coordinates,
    size_t index) {
  const size_t extra_stride = extra_precision_stride();
  const size_t extra_bytes = extra_precision_bytes();
  absl::Span<uint8_t> extra_precision_span = {};
  if (quantization_bits() > 4) {
    extra_precision_span = absl::MakeSpan(extra_precision_data_)
                               .subspan(index * extra_stride, extra_bytes);
  }
  absl::Span<uint8_t> quantized_point_span =
      absl::MakeSpan(quantized_data_)
          .subspan(index * stride_, num_bytes_per_datapoint());
  ABSL_ASSIGN_OR_RETURN(
      auto res, turboquant.EncodeDatapointStructured(
                    coordinates, quantized_point_span, quantization_bits(),
                    extra_precision_span, has_3bit_prefix()));
  WritePointNorms(index, res.norm_scaling_factor_4bit,
                  res.norm_scaling_factor_turboslice,
                  res.unquantized_squared_norm);
  if (quantization_bits() > 4) {
    std::memcpy(
        extra_precision_data_.data() + index * extra_stride + extra_bytes,
        &res.norm_scaling_factor_4plusk, sizeof(float));
  }
  if (store_query_byte_sums_) {
    query_byte_sums_[index] = ComputeQueryByteSum(
        turboquant.dimensionality(), quantized_point_span, has_3bit_prefix());
  }
  return absl::OkStatus();
}

absl::Status TurboQuantDatasetSoA::UpdatePoint(
    const TurboQuantBasePointView& datapoint, size_t index) {
  if (index >= num_datapoints()) {
    return absl::InvalidArgumentError(
        "Index is larger than the number of datapoints.");
  }
  return WritePoint(datapoint, index);
}

absl::Status TurboQuantDatasetSoA::WritePoint(
    const TurboQuantBasePointView& datapoint, size_t index) {
  if (datapoint.quantized_data.size() != num_bytes_per_datapoint()) {
    return absl::InvalidArgumentError(
        "Quantized datapoint size does not match number of bytes per "
        "datapoint.");
  }
  if (datapoint.has_3bit_prefix != has_3bit_prefix()) {
    return absl::InvalidArgumentError(
        "Datapoint has_3bit_prefix does not match dataset has_3bit_prefix().");
  }
  if (quantization_bits() > 4 &&
      datapoint.extra_precision_data.size() != extra_precision_bytes()) {
    return absl::InvalidArgumentError(
        "Extra precision data size does not match number of bytes per "
        "datapoint.");
  }
  std::memcpy(quantized_data_.data() + index * stride_,
              datapoint.quantized_data.data(), num_bytes_per_datapoint());
  WritePointNorms(index, datapoint.norm_scaling_factor_4bit,
                  datapoint.norm_scaling_factor_turboslice,
                  datapoint.unquantized_squared_norm);
  if (quantization_bits() > 4) {
    const size_t extra_stride = extra_precision_stride();
    const size_t extra_bytes = extra_precision_bytes();
    uint8_t* dst = extra_precision_data_.data() + index * extra_stride;
    std::memcpy(dst, datapoint.extra_precision_data.data(), extra_bytes);
    std::memcpy(dst + extra_bytes, &datapoint.norm_scaling_factor_4plusk,
                sizeof(float));
  }
  if (store_query_byte_sums_) {
    if (datapoint.query_byte_sum == kQueryByteSumNotSet) {
      query_byte_sums_[index] = ComputeQueryByteSum(
          dimensionality(), datapoint.quantized_data, has_3bit_prefix());
    } else {
      query_byte_sums_[index] = datapoint.query_byte_sum;
    }
  }
  return absl::OkStatus();
}

}  // namespace ginger
