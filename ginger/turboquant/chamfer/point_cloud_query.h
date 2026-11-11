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

#ifndef GINGER_TURBOQUANT_CHAMFER_POINT_CLOUD_QUERY_H_
#define GINGER_TURBOQUANT_CHAMFER_POINT_CLOUD_QUERY_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "hwy/aligned_allocator.h"

namespace ginger {

// Represents a multi-token point cloud query encoded as int8 vectors using the
// TurboQuant query infrastructure. Prepares contiguous metadata arrays for
// zero-cost consumption by SIMD scoring loops without runtime int4->int8
// decoding.
class TurboQuantPointCloudQuery {
 public:
  // Creates a point cloud query directly from unquantized float vectors.
  // raw_query must have size divisible by codec.dimensionality().
  static absl::StatusOr<TurboQuantPointCloudQuery> Create(
      const ginger::TurboQuantCodec& codec, absl::Span<const float> raw_query);

  // Creates a point cloud query by converting a subset of a 4-bit dataset into
  // int8 query format.
  static absl::StatusOr<TurboQuantPointCloudQuery> CreateFromQuantizedDataset(
      const ginger::TurboQuantDatasetSoA& dataset,
      ginger::IndexSubset query_subset);

  TurboQuantPointCloudQuery() = default;

  TurboQuantPointCloudQuery(const TurboQuantPointCloudQuery&) = delete;
  TurboQuantPointCloudQuery& operator=(const TurboQuantPointCloudQuery&) =
      delete;
  TurboQuantPointCloudQuery(TurboQuantPointCloudQuery&&) noexcept = default;
  TurboQuantPointCloudQuery& operator=(TurboQuantPointCloudQuery&&) noexcept =
      default;

  TurboQuantPointCloudQuery(hwy::AlignedVector<int8_t> decoded_data,
                            std::vector<int32_t> byte_sums,
                            std::vector<float> norm_scaling_factors,
                            size_t num_tokens, size_t decoded_dimension)
      : num_tokens_(num_tokens),
        decoded_dimension_(decoded_dimension),
        decoded_data_(std::move(decoded_data)),
        byte_sums_(std::move(byte_sums)),
        norm_scaling_factors_(std::move(norm_scaling_factors)) {}

  size_t num_tokens() const { return num_tokens_; }
  size_t decoded_dimension() const { return decoded_dimension_; }
  const int8_t* decoded_data() const { return decoded_data_.data(); }
  const int32_t* byte_sums() const { return byte_sums_.data(); }
  const float* norm_scaling_factors() const {
    return norm_scaling_factors_.data();
  }

 private:
  size_t num_tokens_ = 0;
  size_t decoded_dimension_ = 0;
  hwy::AlignedVector<int8_t> decoded_data_;
  std::vector<int32_t> byte_sums_;
  std::vector<float> norm_scaling_factors_;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_CHAMFER_POINT_CLOUD_QUERY_H_
