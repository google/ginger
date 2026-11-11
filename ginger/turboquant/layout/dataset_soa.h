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

#ifndef GINGER_TURBOQUANT_LAYOUT_DATASET_SOA_H_
#define GINGER_TURBOQUANT_LAYOUT_DATASET_SOA_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/dataset.h"
#include "ginger/turboquant/layout/no_init_allocator.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/types/distance_metric.h"

namespace ginger {

// Options for creating a `TurboQuantDatasetSoA`. These are set by the user;
// the internal `TurboQuantDatasetParams` are derived from them and the codec.
struct TurboQuantDatasetOptions {
  // If true, each datapoint is padded and aligned to multiples of
  // `ABSL_CACHELINE_SIZE` bytes. This prevents straddled SIMD loads, but
  // increases memory footprint. When the dimensionality is a multiple of the
  // cache line size (typically 64 bytes), this is a no-op.
  bool pad_to_cache_line_size = false;

  // Total number of bits per dimension. For `quantization_bits` = 4 + k, the
  // extra k bits are stored separately for reranking. Valid values are 4, 5,
  // 6, 7, 8, 10, and 12.
  int quantization_bits = 4;

  // If true, the sum of the quantized byte values is precalculated and stored
  // for each datapoint. This speeds up constructing queries from datapoints in
  // this dataset.
  bool store_query_byte_sums = false;

  // Whether the datapoints are unit-normalized, which lets SQUARED_L2 scoring
  // skip storing squared norms.
  bool normalized_datapoints = false;

  // Where the 4-bit norm scaling factor (and the squared norm, if stored) is
  // kept; see `ScalingStorageMode`.
  ScalingStorageMode scaling_storage_mode = ScalingStorageMode::kSeparateArray;
};

// The user is responsible for ensuring that requested datapoints are encoded by
// either calling the Create() function with input datapoints or by calling
// Create() with the number of data points and then calling UpdatePoint() on the
// required indices.
class TurboQuantDatasetSoA final : public TurboQuantDataset {
 public:
  ~TurboQuantDatasetSoA() final;
  TurboQuantDatasetSoA(TurboQuantDatasetSoA&&);
  TurboQuantDatasetSoA& operator=(TurboQuantDatasetSoA&&);

  // Creates a collection of quantized datapoints from `dense_datapoints`.
  static absl::StatusOr<TurboQuantDatasetSoA> Create(
      const TurboQuantCodec& turboquant,
      absl::Span<const absl::Span<const float>> dense_datapoints);
  static absl::StatusOr<TurboQuantDatasetSoA> Create(
      const TurboQuantCodec& turboquant,
      absl::Span<const absl::Span<const float>> dense_datapoints,
      const TurboQuantDatasetOptions& options);

  // Creates an empty collection for `num_datapoints`. Populate it via
  // UpdatePoint().
  static absl::StatusOr<TurboQuantDatasetSoA> Create(
      const TurboQuantCodec& turboquant, size_t num_datapoints);
  static absl::StatusOr<TurboQuantDatasetSoA> Create(
      const TurboQuantCodec& turboquant, size_t num_datapoints,
      const TurboQuantDatasetOptions& options);

  // Creates a collection with the same layout and number of datapoints as
  // `other`, with zeroed contents. Populate it via UpdatePoint().
  static TurboQuantDatasetSoA UninitializedLike(
      const TurboQuantDatasetSoA& other);

  absl::Status UpdatePoint(const TurboQuantCodec& turboquant,
                           absl::Span<const float> coordinates,
                           size_t index) final;

  absl::Status UpdatePoint(const TurboQuantBasePointView& datapoint,
                           size_t index) final;

  absl::Status AppendPoint(const TurboQuantBasePointView& datapoint) final;

  absl::Status AppendPoint(const TurboQuantCodec& turboquant,
                           absl::Span<const float> coordinates) final;

  size_t stride() const { return stride_; }
  // Row stride of `extra_precision_data_`: `extra_precision_bytes()` plus the
  // trailing `float` 4+k norm scaling factor when `extra_bits_per_dim() > 0`.
  size_t extra_precision_stride() const {
    return extra_bits_per_dim() > 0 ? extra_precision_bytes() + sizeof(float)
                                    : 0;
  }
  ScalingStorageMode scaling_storage_mode() const {
    return scaling_storage_mode_;
  }
  bool store_query_byte_sums() const { return store_query_byte_sums_; }
  const uint8_t* quantized_data() const { return quantized_data_.data(); }
  const uint8_t* extra_precision_data() const {
    return extra_precision_data_.data();
  }

  bool has_3bit_prefix() const final {
    return scaling_storage_mode_ == ScalingStorageMode::kIn3BitPrefix;
  }

  // `kRowTail` sits at the end of the full 4-bit row, so full 4-bit scoring
  // always uses it while TurboSlice reads `turboslice_norm_data_`.
  // `kIn3BitPrefix` sits in the first 16 bytes of the row, so when TurboSlice
  // is enabled it holds the TurboSlice factor and full 4-bit scoring reads
  // `norm_data_`.
  ScalingStorageMode scaling_storage_mode_4bit() const {
    return (IsTurboSliceEnabled() &&
            scaling_storage_mode_ == ScalingStorageMode::kIn3BitPrefix)
               ? ScalingStorageMode::kSeparateArray
               : scaling_storage_mode_;
  }

  ScalingStorageMode scaling_storage_mode_turboslice() const {
    return scaling_storage_mode_ == ScalingStorageMode::kIn3BitPrefix
               ? ScalingStorageMode::kIn3BitPrefix
               : ScalingStorageMode::kSeparateArray;
  }

  // Enables precalculating query byte sums on an already built dataset.
  void EnableQueryByteSums();

  // Releases memory used by precalculated query byte sums and disables storing
  // them for future points.
  void DropQueryByteSums();

  NormStorageReader NormReader() const {
    const ScalingStorageMode mode = scaling_storage_mode_4bit();
    if (mode == ScalingStorageMode::kRowTail) {
      const uint8_t* norm_base =
          quantized_data_.empty()
              ? nullptr
              : quantized_data_.data() + stride_ - NormEntryBytes(norm_layout_);
      return NormStorageReader(norm_layout_, norm_base, stride_,
                               ScalingStorageMode::kRowTail);
    }
    if (mode == ScalingStorageMode::kIn3BitPrefix) {
      return NormStorageReader(norm_layout_, quantized_data_.data(), stride_,
                               ScalingStorageMode::kIn3BitPrefix);
    }
    return NormStorageReader(*norm_data_);
  }

  NormStorageReader TurboSliceNormReader() const {
    ABSL_DCHECK(IsTurboSliceEnabled());
    if (scaling_storage_mode_turboslice() ==
        ScalingStorageMode::kIn3BitPrefix) {
      return NormStorageReader(norm_layout_, quantized_data_.data(), stride_,
                               ScalingStorageMode::kIn3BitPrefix);
    }
    return NormStorageReader(*turboslice_norm_data_);
  }

  float norm_scaling_factor(size_t index) const {
    return NormReader().GetNormScalingFactor(index);
  }
  float norm_scaling_factor_turboslice(size_t index) const {
    return IsTurboSliceEnabled()
               ? TurboSliceNormReader().GetNormScalingFactor(index)
               : 0.0f;
  }
  float norm_scaling_factor_4plusk(size_t index) const {
    if (quantization_bits() <= 4) {
      return norm_scaling_factor(index);
    }
    float scale;
    std::memcpy(&scale,
                extra_precision_data_.data() +
                    index * extra_precision_stride() + extra_precision_bytes(),
                sizeof(float));
    return scale;
  }
  const NormStorage& norm_data() const { return *norm_data_; }
  NormLayout norm_layout() const { return norm_layout_; }

  // For testing only!
  void TestingOnlySetNorm(size_t index, float norm) {
    WritePointNorms(index, norm, norm_scaling_factor_turboslice(index),
                    metric() == TurboQuantDistanceMetric::kSquaredL2Distance
                        ? unquantized_squared_norm(index)
                        : 0.0f);
  }
  size_t TestingOnlyGetQuantizedDataSize() const {
    return quantized_data_.size();
  }
  float unquantized_squared_norm(size_t index) const {
    ABSL_DCHECK_EQ(metric(), TurboQuantDistanceMetric::kSquaredL2Distance);
    return NormReader().GetSquaredNorm(index);
  }

  TurboQuantBasePointView DataPointView(size_t index) const final;

  size_t num_datapoints() const final { return num_datapoints_; }

  void UnsafeComputeDistances(const TurboQuantQuery& query,
                              ScoringAccessor accessor) const final;

  void UnsafeComputeDistancesExtraPrecision(
      const TurboQuantQuery& query, ScoringAccessor accessor) const final;

  bool IsTurboSliceEnabled() const {
    return params().turboslice_dimensionality > 0;
  }

  void UnsafeComputeDistancesTurboSlice(const TurboQuantQuery& query,
                                        ScoringAccessor accessor) const final;

 private:
  size_t stride_ = 0;
  // How the norm data is stored, both in norm_data_ and in the tail of each
  // row.
  NormLayout norm_layout_ = NormLayout::kScalingOnly;
  ScalingStorageMode scaling_storage_mode_ = ScalingStorageMode::kSeparateArray;
  bool store_query_byte_sums_ = false;
  size_t num_datapoints_ = 0;
  std::vector<uint8_t, NoInitAlignedAllocator<uint8_t>> quantized_data_;
  std::vector<uint8_t, NoInitAlignedAllocator<uint8_t>> extra_precision_data_;
  // Stores the norm scaling factors and, for the kScalingAndSquaredNorm
  // layout, also the squared norms of the unquantized datapoints when not
  // embedded alongside the codes.
  std::unique_ptr<NormStorage> norm_data_;
  std::unique_ptr<NormStorage> turboslice_norm_data_;
  std::vector<int32_t> query_byte_sums_;

  TurboQuantDatasetSoA(const TurboQuantDatasetParams& params,
                       size_t num_datapoints, size_t stride,
                       ScalingStorageMode scaling_storage_mode,
                       bool store_query_byte_sums);

  void WritePointNorms(size_t index, float norm_scaling_factor_4bit,
                       float norm_scaling_factor_turboslice,
                       float unquantized_squared_norm);
  // Makes room for row num_datapoints() without counting it, so a failed
  // write leaves the dataset unchanged. Calling it again reuses the same row.
  void GrowStorageForAppend();
  // Writes a point to row `index`, which may be the uncounted row made by
  // GrowStorageForAppend().
  absl::Status WritePoint(const TurboQuantCodec& turboquant,
                          absl::Span<const float> coordinates, size_t index);
  absl::Status WritePoint(const TurboQuantBasePointView& datapoint,
                          size_t index);
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_LAYOUT_DATASET_SOA_H_
