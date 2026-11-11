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

#ifndef GINGER_TURBOQUANT_SCORING_SCORING_SCALAR_REFERENCE_H_
#define GINGER_TURBOQUANT_SCORING_SCORING_SCALAR_REFERENCE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codebooks/centroid_tables.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/types/distance_metric.h"

namespace ginger {

inline void TurboQuantScoringInt8ScalarReference(
    const TurboQuantDatasetSoA& base_points, const TurboQuantQuery& query,
    absl::Span<float> scores, absl::Span<const int32_t> indices,
    bool is_turboslice_scoring = false) {
  using internal::kTurboQuantCentroidsInt8;
  const size_t dim = query.dimensionality();
  const size_t num_bytes = (dim + 1) / 2;
  const size_t padding = TurboQuantCodec::QueryPadding();
  const size_t deinterleaved_end =
      ((num_bytes + padding - 1) / padding) * padding;

  // With a 3-bit prefix, the odd dimensions of the first `kScaleBitBytes` bytes
  // use the merged 3-bit codebook.
  const bool has_3bit_prefix = base_points.has_3bit_prefix();

  for (size_t i = 0; i < scores.size(); ++i) {
    int32_t dot = 0;
    const int32_t base_point_index = indices[i];
    const uint8_t* p =
        base_points.quantized_data() + base_point_index * base_points.stride();

    for (size_t j = 0; j < dim / 2; ++j) {
      const uint8_t b1 = p[j] & 0xF;
      const uint8_t b2 = p[j] >> 4;
      const int8_t odd_centroid =
          (has_3bit_prefix && j < kScaleBitBytes)
              ? turboquant_codebooks::kTurboQuant3BitCentroidsInt8[b2]
              : kTurboQuantCentroidsInt8[b2];
      dot += static_cast<int32_t>(kTurboQuantCentroidsInt8[b1]) *
             query.query_data()[j];
      dot += static_cast<int32_t>(odd_centroid) *
             query.query_data()[deinterleaved_end + j];
    }

    if (dim % 2 != 0) {
      const size_t j = dim / 2;
      const uint8_t b1 = p[j] & 0xF;
      dot += static_cast<int32_t>(kTurboQuantCentroidsInt8[b1]) *
             query.query_data()[j];
    }

    auto dp_view = base_points.DataPointView(base_point_index);
    const float norm_scaling = is_turboslice_scoring
                                   ? dp_view.norm_scaling_factor_turboslice
                                   : dp_view.norm_scaling_factor_4bit;

    const float neg_dot_product_approx =
        -static_cast<float>(dot) * norm_scaling * query.norm_scaling_factor();

    if (query.metric() == TurboQuantDistanceMetric::kSquaredL2Distance) {
      scores[i] = base_points.unquantized_squared_norm(base_point_index) +
                  2 * neg_dot_product_approx + query.unquantized_squared_norm();
    } else {
      scores[i] = neg_dot_product_approx;
    }
  }
}

// Scalar reference implementation for extra precision scoring (4+k bit).
// This uses float centroid lookups and float query data instead of int8.
// Used for testing the SIMD implementation.
//
// All extra bit widths (1, 2, 3, 4, 6, 8) use the same contiguous bit-packing
// scheme: each dimension's `extra_bits_per_dim` bits are stored consecutively
// in `dp_extra`, so the bit offset for dimension `d` is `d *
// extra_bits_per_dim`. Values may cross byte boundaries, so we always read 2
// bytes and shift. The only thing that changes between bit widths is the
// codebook (via SignedCentroidTable, which also bakes in the sign).
inline void TurboQuantScoringExtraPrecisionScalarReference(
    const TurboQuantDatasetSoA& base_points, const TurboQuantQuery& query,
    absl::Span<float> scores, absl::Span<const int32_t> indices) {
  const size_t dim = base_points.dimensionality();
  const int extra_bits_per_dim = base_points.extra_bits_per_dim();
  const float* query_data = query.float_query_data().data();

  // Select the signed centroid table for the given bit width.
  const int num_sub_centroids = 1 << extra_bits_per_dim;
  const uint16_t extra_mask = num_sub_centroids - 1;
  const float* centroid_table = nullptr;
  switch (extra_bits_per_dim) {
    case 1: {
      static constexpr turboquant_internal::SignedCentroidTable<1> table{};
      centroid_table = table.data();
      break;
    }
    case 2: {
      static constexpr turboquant_internal::SignedCentroidTable<2> table{};
      centroid_table = table.data();
      break;
    }
    case 3: {
      static constexpr turboquant_internal::SignedCentroidTable<3> table{};
      centroid_table = table.data();
      break;
    }
    case 4: {
      static constexpr turboquant_internal::SignedCentroidTable<4> table{};
      centroid_table = table.data();
      break;
    }
    case 6: {
      static constexpr turboquant_internal::SignedCentroidTable<6> table{};
      centroid_table = table.data();
      break;
    }
    case 8: {
      static constexpr turboquant_internal::SignedCentroidTable<8> table{};
      centroid_table = table.data();
      break;
    }
    default:
      ABSL_LOG(FATAL) << "Unsupported extra_bits_per_dim: "
                      << extra_bits_per_dim
                      << ". Supported values are 1, 2, 3, 4, 6, 8.";
  }

  for (size_t i = 0; i < scores.size(); ++i) {
    const int32_t dp_idx = indices[i];
    auto dp = base_points.DataPointView(dp_idx);
    const uint8_t* dp_quant = dp.quantized_data.data();
    const uint8_t* dp_extra = dp.extra_precision_data.data();
    uint8_t prefix_codes[kScaleBitBytes];
    if (base_points.has_3bit_prefix()) {
      RestorePrefixCodes(
          dp_quant, dp.extra_precision_data.last(kPrefixLowBitBytes).data(),
          prefix_codes);
    }

    float sum = 0.0f;
    int current_bit = 0;

    for (size_t d = 0; d < dim; ++d) {
      const uint8_t byte = (base_points.has_3bit_prefix() && d < kScaleBitDims)
                               ? prefix_codes[d / 2]
                               : dp_quant[d / 2];
      const uint8_t base_code = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);

      // Contiguous bit-packing: read `extra_bits_per_dim` bits starting at
      // bit offset `current_bit`. Values may cross byte boundaries.
      uint16_t two_bytes;
      memcpy(&two_bytes, dp_extra + (current_bit / 8), sizeof(two_bytes));
      const uint16_t extra = (two_bytes >> (current_bit % 8)) & extra_mask;
      current_bit += extra_bits_per_dim;

      // SignedCentroidTable layout: table[base_code * num_sub_centroids + sub].
      sum +=
          query_data[d] * centroid_table[base_code * num_sub_centroids + extra];
    }

    float score = sum * dp.norm_scaling_factor_4plusk;

    if (query.metric() == TurboQuantDistanceMetric::kSquaredL2Distance) {
      scores[i] = dp.unquantized_squared_norm +
                  query.unquantized_squared_norm() - 2 * score;
    } else {
      scores[i] = -score;
    }
  }
}

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_SCORING_SCORING_SCALAR_REFERENCE_H_
