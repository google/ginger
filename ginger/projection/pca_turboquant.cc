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

// Implements PCA projection with a 4-bit TurboQuant-quantized projection
// matrix.
//
// Motivation:
// PCA projection during vector search is typically memory-bandwidth
// bound because worker threads repeatedly stream the projection matrix from
// memory. Quantizing the projection matrix from FP32 to 4-bit reduces its
// memory footprint by ~8x, significantly speeding up matrix-vector projection
// under concurrent query loads.
//
// Quantization Design & Why No Rotation:
// Standard TurboQuant applies a random orthogonal rotation prior to
// quantization to make dimensions Gaussian, then snaps each entry to the
// nearest centroid of a pretrained codebook. For PCA, the projection matrix
// rows are already orthogonal eigenvectors whose components are
// well-distributed across dimensions. Omitting rotation avoids query-side
// rotation overhead while empirically preserving search recall.
//
// `Create`: Matrix rows are individually scaled and quantized to 4-bit indices
//   using the TurboQuant codebook, then packed into split nibbles for SIMD
//   loads.
// `Project`: Queries are dynamically quantized to int8, and projection outputs
//   are computed via SIMD dot products and scaled by row/query inverse scales.

#include "ginger/projection/pca_turboquant.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/projection/config.pb.h"
#include "ginger/projection/pca.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "hwy/aligned_allocator.h"

#ifndef GINGER_PROJECTION_PCA_TURBOQUANT_TYPES_
#define GINGER_PROJECTION_PCA_TURBOQUANT_TYPES_
namespace ginger {

struct QuantizationScale {
  float scale;
  float inv_scale;

  static QuantizationScale FromMaxAbs(float max_abs) {
    if (max_abs > 1e-12f) {
      return QuantizationScale{.scale = 127.0f / max_abs,
                               .inv_scale = max_abs / 127.0f};
    }
    return QuantizationScale{.scale = 0.0f, .inv_scale = 1.0f};
  }
};

struct QuantizedQuery {
  hwy::AlignedVector<int8_t> data;
  int32_t sum;
};

}  // namespace ginger
#endif  // GINGER_PROJECTION_PCA_TURBOQUANT_TYPES_

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ginger/projection/pca_turboquant.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
#include "ginger/turboquant/simd/vec_group-inl.h"

namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = ::hwy::HWY_NAMESPACE;
namespace rg = HWY_NAMESPACE;

// Computes the int8 quantization scale and inverse scale for `values` using
// SIMD instructions.
HWY_INLINE QuantizationScale
ComputeQuantizationScale(absl::Span<const float> values) {
  const hn::ScalableTag<float> df;
  const size_t N = hn::Lanes(df);
  const size_t size = values.size();
  const float* HWY_RESTRICT ptr = values.data();

  auto max0 = hn::Zero(df);
  auto max1 = hn::Zero(df);
  auto max2 = hn::Zero(df);
  auto max3 = hn::Zero(df);

  const auto accumulate_max = [&](auto& max_v, size_t offset) {
    max_v = hn::Max(max_v, hn::Abs(hn::LoadU(df, ptr + offset)));
  };

  size_t i = 0;
  for (; i + 4 * N <= size; i += 4 * N) {
    accumulate_max(max0, i);
    accumulate_max(max1, i + N);
    accumulate_max(max2, i + 2 * N);
    accumulate_max(max3, i + 3 * N);
  }
  if (i + 2 * N <= size) {
    accumulate_max(max0, i);
    accumulate_max(max1, i + N);
    i += 2 * N;
  }
  if (i + N <= size) {
    accumulate_max(max2, i);
    i += N;
  }

  const auto max_all = hn::Max(hn::Max(max0, max1), hn::Max(max2, max3));
  float max_abs = hn::ReduceMax(df, max_all);
  for (; i < size; ++i) {
    max_abs = std::max(max_abs, std::abs(ptr[i]));
  }
  return QuantizationScale::FromMaxAbs(max_abs);
}

// Quantizes the query vector to int8 values in [-127, 127] and computes their
// sum using SIMD instructions.
HWY_INLINE QuantizedQuery QuantizeQuery(absl::Span<const float> query,
                                        float scale, size_t padded_dim) {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const hn::Rebind<int8_t, hn::ScalableTag<float>> di8;
  const size_t N = hn::Lanes(df);
  const size_t input_dim = query.size();
  const float* HWY_RESTRICT query_ptr = query.data();

  hwy::AlignedVector<int8_t> quantized_query(padded_dim, 0);
  int8_t* HWY_RESTRICT dst_ptr = quantized_query.data();

  const auto scale_v = hn::Set(df, scale);
  const auto min_v = hn::Set(di32, -127);
  const auto max_v = hn::Set(di32, 127);
  auto sum0 = hn::Zero(di32);
  auto sum1 = hn::Zero(di32);
  auto sum2 = hn::Zero(di32);
  auto sum3 = hn::Zero(di32);

  const auto quantize_and_accumulate = [&](auto& sum_v, size_t offset) {
    const auto qf = hn::Mul(hn::LoadU(df, query_ptr + offset), scale_v);
    const auto qi = hn::Clamp(hn::NearestInt(qf), min_v, max_v);
    sum_v = hn::Add(sum_v, qi);
    hn::StoreU(hn::DemoteTo(di8, qi), di8, dst_ptr + offset);
  };

  size_t i = 0;
  for (; i + 4 * N <= input_dim; i += 4 * N) {
    quantize_and_accumulate(sum0, i);
    quantize_and_accumulate(sum1, i + N);
    quantize_and_accumulate(sum2, i + 2 * N);
    quantize_and_accumulate(sum3, i + 3 * N);
  }
  if (i + 2 * N <= input_dim) {
    quantize_and_accumulate(sum0, i);
    quantize_and_accumulate(sum1, i + N);
    i += 2 * N;
  }
  if (i + N <= input_dim) {
    quantize_and_accumulate(sum2, i);
    i += N;
  }

  const auto sum_all = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
  int32_t query_sum = hn::ReduceSum(di32, sum_all);
  for (; i < input_dim; ++i) {
    const int32_t q = std::clamp<int32_t>(
        static_cast<int32_t>(std::nearbyint(query_ptr[i] * scale)), -127, 127);
    dst_ptr[i] = static_cast<int8_t>(q);
    query_sum += q;
  }
  return QuantizedQuery{.data = std::move(quantized_query), .sum = query_sum};
}

// TQ4 Project Kernel using block-split query nibble unpacking and dual
// 8-row panel accumulators.
HWY_INLINE void TQ4ProjectKernel(const int8_t* HWY_RESTRICT query,
                                 size_t num_packed_bytes,
                                 const uint8_t* HWY_RESTRICT weights,
                                 size_t output_dim, size_t padded_output_dim,
                                 const float* HWY_RESTRICT inv_scales,
                                 float inv_query_scale, int32_t query_sum,
                                 float* HWY_RESTRICT output) {
  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int32 = hn::ScalableTag<int32_t>;

  const Int8 di8;
  const Uint8 du8;
  const Int32 di32;
  const size_t N = hn::Lanes(du8);

  using VecI8 = hn::Vec<Int8>;
  using VecU8 = hn::Vec<Uint8>;
  using VecI32 = hn::Vec<Int32>;

  const VecU8 mask = hn::Set(du8, 0x0F);
  const VecU8 centroids_u8 = hn::LoadDup128(
      du8, turboquant_codebooks::kTurboQuantCentroidsUint8.data());

  for (size_t row_block = 0; row_block < padded_output_dim; row_block += 8) {
    // Defines 8 vector variables (acc_lo_group0..acc_lo_group7 and
    // acc_hi_group0..acc_hi_group7) to accumulate products for 8 rows in
    // parallel without declaring arrays of sizeless vector types.
    // They can be accessed individually using VEC_GROUP_ACCESSOR.
    VEC_GROUP_DEFINE(di32, acc_lo_group);
    VEC_GROUP_DEFINE(di32, acc_hi_group);
    const uint8_t* HWY_RESTRICT row_ptrs[8];
    rg::ForIndex<8>([&](auto r) {
      constexpr size_t row_offset = decltype(r)::value;
      row_ptrs[r] = weights + (row_block + row_offset) * num_packed_bytes;
      VEC_GROUP_ACCESSOR(acc_lo_group)(r) = hn::Zero(di32);
      VEC_GROUP_ACCESSOR(acc_hi_group)(r) = hn::Zero(di32);
    });

    for (size_t byte_idx = 0; byte_idx < num_packed_bytes; byte_idx += N) {
      const VecI8 query_lo = hn::Load(di8, query + byte_idx);
      const VecI8 query_hi = hn::Load(di8, query + num_packed_bytes + byte_idx);

      rg::ForIndex<8>([&](auto r) {
        const VecU8 packed_w = hn::Load(du8, row_ptrs[r] + byte_idx);
        const VecU8 lo_indices = hn::And(packed_w, mask);
        const VecU8 hi_indices = hn::ShiftRight<4>(packed_w);
        const VecU8 lo_weights = hn::TableLookupBytes(centroids_u8, lo_indices);
        const VecU8 hi_weights = hn::TableLookupBytes(centroids_u8, hi_indices);

        VecI32& acc_lo = VEC_GROUP_ACCESSOR(acc_lo_group)(r);
        VecI32& acc_hi = VEC_GROUP_ACCESSOR(acc_hi_group)(r);
        acc_lo = hn::SumOfMulQuadAccumulate(di32, lo_weights, query_lo, acc_lo);
        acc_hi = hn::SumOfMulQuadAccumulate(di32, hi_weights, query_hi, acc_hi);
      });
    }

    rg::ForIndex<8>([&](auto r) {
      constexpr size_t row_offset = decltype(r)::value;
      const size_t row_idx = row_block + row_offset;
      if (row_idx < output_dim) {
        const int32_t sum_lo =
            hn::ReduceSum(di32, VEC_GROUP_ACCESSOR(acc_lo_group)(r));
        const int32_t sum_hi =
            hn::ReduceSum(di32, VEC_GROUP_ACCESSOR(acc_hi_group)(r));
        const int32_t dot_product_sum = sum_lo + sum_hi;
        output[row_idx] =
            static_cast<float>(dot_product_sum - 128 * query_sum) *
            (inv_scales[row_idx] * inv_query_scale);
      }
    });
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace ginger {

HWY_EXPORT(ComputeQuantizationScale);
HWY_EXPORT(QuantizeQuery);
HWY_EXPORT(TQ4ProjectKernel);

namespace {

// Finds the nearest 4-bit TurboQuant centroid index for a given scaled value.
uint8_t ComputeNearestCentroidIdx(float val) {
  const std::array<int8_t, 16>& centroids =
      turboquant_codebooks::kTurboQuantCentroidsInt8;
  float min_diff = std::numeric_limits<float>::max();
  uint8_t best_c = 0;
  for (size_t c = 0; c < centroids.size(); ++c) {
    const float diff = std::abs(val - static_cast<float>(centroids[c]));
    if (diff < min_diff) {
      min_diff = diff;
      best_c = static_cast<uint8_t>(c);
    }
  }
  return best_c;
}

struct QuantizationResult {
  hwy::AlignedVector<uint8_t> quantized_indices;
  hwy::AlignedVector<float> inv_scales;
};

// Quantizes the projection matrix into 4-bit TurboQuant centroid indices,
// storing one index per byte in a padded buffer of size
// `padded_output_dim * padded_input_dim`.
QuantizationResult QuantizeWeights(const Eigen::MatrixXf& projection_matrix,
                                   size_t padded_input_dim,
                                   size_t padded_output_dim) {
  const size_t output_dim = projection_matrix.rows();
  const size_t input_dim = projection_matrix.cols();
  hwy::AlignedVector<uint8_t> quantized_indices(
      padded_output_dim * padded_input_dim, 0);
  hwy::AlignedVector<float> padded_inv_scales(padded_output_dim, 0.0f);

  for (size_t i = 0; i < output_dim; ++i) {
    const QuantizationScale q_scale = QuantizationScale::FromMaxAbs(
        projection_matrix.row(i).cwiseAbs().maxCoeff());
    padded_inv_scales[i] = q_scale.inv_scale;

    for (size_t j = 0; j < input_dim; ++j) {
      quantized_indices[i * padded_input_dim + j] =
          ComputeNearestCentroidIdx(projection_matrix(i, j) * q_scale.scale);
    }
  }

  return QuantizationResult{
      .quantized_indices = std::move(quantized_indices),
      .inv_scales = std::move(padded_inv_scales),
  };
}

// Packs pairs of 4-bit centroid indices into bytes.
//
// For each row, weight `k` and weight `k + half` (where half = padded_input_dim
// / 2) are paired into the low and high nibbles of byte `k`:
// +-------------------+-------------------+-------+---------------------------+
// |      Byte 0       |      Byte 1       |  ...  |        Byte half-1        |
// | [ W_half | W0 ]   | [ W_half+1 | W1 ] |  ...  | [ W_2*half-1 | W_half-1 ] |
// +-------------------+-------------------+-------+---------------------------+
//      hi_nib   lo_nib
//
// In `TQ4ProjectKernel`, loading N packed bytes yields 2N weights. Masking low
// nibbles (& 0x0F) yields weights [0..N-1], and shifting high nibbles (>> 4)
// yields weights [half..half+N-1]. This aligns with loads from the query,
// avoiding unzipping.
hwy::AlignedVector<uint8_t> PackTurboQuantWeights(
    size_t padded_input_dim, size_t padded_output_dim,
    absl::Span<const uint8_t> quantized_indices) {
  const size_t num_packed_bytes = padded_input_dim / 2;
  hwy::AlignedVector<uint8_t> packed_weights(
      padded_output_dim * num_packed_bytes, 0);

  for (size_t row_idx = 0; row_idx < padded_output_dim; ++row_idx) {
    const size_t half = num_packed_bytes;
    const uint8_t* row_indices =
        quantized_indices.data() + row_idx * padded_input_dim;
    uint8_t* row_packed = packed_weights.data() + row_idx * num_packed_bytes;

    for (size_t k = 0; k < half; ++k) {
      row_packed[k] = static_cast<uint8_t>(
          (row_indices[k] & 0x0F) | ((row_indices[k + half] & 0x0F) << 4));
    }
  }

  return packed_weights;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TurboQuantPcaProjection>>
TurboQuantPcaProjection::Create(const Eigen::MatrixXf& projection_matrix) {
  const int32_t output_dim = projection_matrix.rows();
  const int32_t input_dim = projection_matrix.cols();

  if (output_dim == 0 || input_dim == 0) {
    return absl::InvalidArgumentError(
        "Projection matrix dimensions must be strictly positive.");
  }

  const size_t padded_input_dim = hwy::RoundUpTo(input_dim, 128);
  const size_t padded_output_dim = hwy::RoundUpTo(output_dim, 8);

  QuantizationResult quant_result =
      QuantizeWeights(projection_matrix, padded_input_dim, padded_output_dim);

  hwy::AlignedVector<uint8_t> packed_weights = PackTurboQuantWeights(
      padded_input_dim, padded_output_dim,
      absl::MakeConstSpan(quant_result.quantized_indices));

  return Create(input_dim, output_dim, std::move(packed_weights),
                std::move(quant_result.inv_scales));
}

absl::StatusOr<std::unique_ptr<TurboQuantPcaProjection>>
TurboQuantPcaProjection::Create(int32_t input_dim, int32_t output_dim,
                                hwy::AlignedVector<uint8_t> packed_weights,
                                hwy::AlignedVector<float> row_inv_scales) {
  if (output_dim <= 0 || input_dim <= 0) {
    return absl::InvalidArgumentError(
        "Projection dimensions must be strictly positive.");
  }
  if (output_dim > input_dim) {
    return absl::InvalidArgumentError(
        "Output dimension must be smaller than or equal to input dimension.");
  }
  const size_t padded_input_dim = hwy::RoundUpTo(input_dim, 128);
  const size_t padded_output_dim = hwy::RoundUpTo(output_dim, 8);
  const size_t expected_weights_size =
      padded_output_dim * (padded_input_dim / 2);
  if (packed_weights.size() != expected_weights_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Packed weights size mismatch: got %d, expected %d.",
                        packed_weights.size(), expected_weights_size));
  }
  if (row_inv_scales.size() != padded_output_dim) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Row inverse scales size mismatch: got %d, expected %d.",
        row_inv_scales.size(), padded_output_dim));
  }

  return absl::WrapUnique(new TurboQuantPcaProjection(
      input_dim, output_dim, std::move(packed_weights),
      std::move(row_inv_scales)));
}

TurboQuantPcaProjection::TurboQuantPcaProjection(
    int32_t input_dim, int32_t output_dim,
    hwy::AlignedVector<uint8_t> packed_weights,
    hwy::AlignedVector<float> row_inv_scales)
    : PcaProjection(input_dim, output_dim, MATRIX_REPRESENTATION_TURBOQUANT),
      packed_weights_(std::move(packed_weights)),
      row_inv_scales_(std::move(row_inv_scales)) {}

absl::Status TurboQuantPcaProjection::Project(absl::Span<const float> query,
                                              absl::Span<float> output) const {
  if (query.size() != static_cast<size_t>(input_dim())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Query dimensionality (%d) does not match PCA input "
                        "dimensionality (%d).",
                        query.size(), input_dim()));
  }
  if (output.size() != output_dim()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Output dimensionality (%d) does not match PCA output "
                        "dimensionality (%d).",
                        output.size(), output_dim()));
  }

  const size_t padded_input_dim = hwy::RoundUpTo(input_dim(), 128);
  const size_t num_packed_bytes = padded_input_dim / 2;
  const size_t padded_output_dim = hwy::RoundUpTo(output_dim(), 8);

  const QuantizationScale query_scale =
      HWY_DYNAMIC_DISPATCH(ComputeQuantizationScale)(query);
  const QuantizedQuery quantized_query = HWY_DYNAMIC_DISPATCH(QuantizeQuery)(
      query, query_scale.scale, padded_input_dim);

  HWY_DYNAMIC_DISPATCH(TQ4ProjectKernel)
  (quantized_query.data.data(), num_packed_bytes, packed_weights_.data(),
   output_dim(), padded_output_dim, row_inv_scales_.data(),
   query_scale.inv_scale, quantized_query.sum, output.data());

  return absl::OkStatus();
}

}  // namespace ginger
#endif  // HWY_ONCE
