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

// Queries-specific SIMD helpers for TurboQuant Decoding.
//
// This is a Highway per-target `-inl.h` header using the standard toggle-guard
// pattern (see third_party/highway/hwy/contrib/dot/dot-inl.h for a reference).
//
// Layer 1 (outer #ifndef): Runs once per translation unit. Includes
//   non-per-target headers (e.g. absl/types/span.h) that must not be
//   re-included for every target.
// Layer 2 (toggle guard): Re-includes the SIMD implementation once per Highway
//   target. The toggle macro flips on/off in sync with HWY_TARGET_TOGGLE,
//   which highway.h defines/undefines for each target.
//
// Include after foreach_target.h and highway.h.

#ifndef GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_QUERIES_INL_H_
#define GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_QUERIES_INL_H_

#include <cstddef>
#include <cstdint>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"

namespace ginger {

// Non-owning view of decoded query data. Bundles the source pointers and
// metadata needed by PrepareQueryBatch.
// The caller is responsible for the lifetime of the underlying buffers.
struct DecodedQueryView {
  const int8_t* decoded_data = nullptr;
  const int32_t* byte_sums = nullptr;
  const float* norm_scaling_factors = nullptr;
  const float* squared_norms = nullptr;  // nullptr when metric is dot product.
  size_t decoded_dimension = 0;
  bool is_l2 = false;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_QUERIES_INL_H_

// clang-format off
#if defined(GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_QUERIES_TOGGLE) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_QUERIES_TOGGLE
#undef GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_QUERIES_TOGGLE
#else
#define GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_QUERIES_TOGGLE
#endif
// clang-format on

#include "hwy/highway.h"

namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using ::ginger::turboquant_codebooks::kTurboQuantCentroidsInt8;

// Dequantizes packed 4-bit nibbles into two 8-bit vectors (signed).
//
// Unpacks each byte in `packed_nibbles` into its even (low 4 bits) and odd
// (high 4 bits) nibbles. It then uses `TableLookupBytes` with the `centroids`
// codebook to map indices to their actual values.
//
// The even/odd split is an artifact of the nibble encoding: each input byte
// stores two quantization indices (one per 4 bits). This parity distinction
// propagates through the data layout but carries no semantic meaning —
// downstream kernels treat both parities identically.
//
// Parameters:
//   i8:              Highway tag for int8_t; used for BitCast.
//   mask:            Pre-computed Set(u8, 0x0F) to isolate the low nibble.
//   centroids:       16-entry int8_t codebook loaded via LoadDup128.
//   packed_nibbles:  Vector of raw packed bytes (two 4-bit indices each).
//   even_vals:       [out] Decoded values for even (low-nibble) dimensions.
//   odd_vals:        [out] Decoded values for odd  (high-nibble) dimensions.
template <typename Int8, typename Uint8>
HWY_INLINE void DequantizeNibbles(const Int8 i8, const hn::Vec<Uint8>& mask,
                                  const hn::Vec<Int8>& centroids,
                                  const hn::Vec<Uint8>& packed_nibbles,
                                  hn::Vec<Int8>& even_vals,
                                  hn::Vec<Int8>& odd_vals) {
  const hn::Vec<Uint8> even_idxs = hn::And(packed_nibbles, mask);
  const hn::Vec<Uint8> odd_idxs = hn::ShiftRight<4>(packed_nibbles);
  even_vals = hn::BitCast(i8, hn::TableLookupBytes(centroids, even_idxs));
  odd_vals = hn::BitCast(i8, hn::TableLookupBytes(centroids, odd_idxs));
}

// Returns the per-query stride (in int8_t elements) of the decoded output
// buffer. This is the number of elements to allocate per query for
// DecodeQueries and ComputeQueryByteSums.
HWY_INLINE size_t
DecodedQueryStride(const ::ginger::TurboQuantDatasetSoA& queries) {
  const size_t N = hn::Lanes(hn::ScalableTag<uint8_t>());
  const size_t num_dim_chunks =
      hwy::DivCeil(queries.num_bytes_per_datapoint(), N);
  return num_dim_chunks * 2 * N;
}

// Decodes nibble-packed query data into a row-major int8_t format.
//
// Broad usage in many-to-many search: Queries are decoded upfront (or in
// batches) into a temporary buffer. This amortizes the decoding cost, as the
// decoded queries can be reused across many database panels.
//
// Each input byte (two 4-bit indices) is expanded into two 8-bit values via
// codebook lookup. The output represents dimensions in full bytes, padded
// if necessary for alignment or odd dimensions.

// Decodes queries specified by IndexSubset into all_q_decoded_span.
//
// Example usage:
//   TurboQuantDatasetSoA queries = ...;
//   size_t stride = DecodedQueryStride(queries);
//   std::vector<int8_t> decoded(num_queries * stride);
//   DecodeQueries(queries, IndexSubset::Contiguous(0, num_queries),
//   absl::MakeSpan(decoded));
HWY_INLINE void DecodeQueries(const ::ginger::TurboQuantDatasetSoA& queries,
                              IndexSubset query_indices,
                              absl::Span<int8_t> all_q_decoded_span) {
  // TODO: Support 3-bit prefix in DecodeQueries.
  ABSL_CHECK(!queries.has_3bit_prefix());
  const size_t num_queries = query_indices.size();
  const size_t decoded_dim = DecodedQueryStride(queries);
  ABSL_CHECK_EQ(all_q_decoded_span.size(), num_queries * decoded_dim);
  int8_t* HWY_RESTRICT all_q_decoded = all_q_decoded_span.data();
  const size_t dimensionality = queries.dimensionality();
  if (dimensionality == 0) return;
  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  const Int8 i8;
  const Uint8 u8;
  const size_t N = hn::Lanes(u8);
  const size_t bytes_per_point = queries.num_bytes_per_datapoint();
  const size_t q_stride = queries.stride();
  const uint8_t* q_data = queries.quantized_data();

  const hn::Vec<Uint8> mask = hn::Set(u8, 0x0F);
  const hn::Vec<Int8> centroids =
      hn::LoadDup128(i8, kTurboQuantCentroidsInt8.data());
  const size_t num_full_chunks = bytes_per_point / N;
  const size_t remainder = bytes_per_point % N;

  const auto valid_mask = remainder > 0
                              ? hn::RebindMask(i8, hn::FirstN(u8, remainder))
                              : hn::FirstN(i8, N) /* unused */;

  size_t qi = 0;
  for (; qi + 4 <= num_queries; qi += 4) {
    const uint8_t* q_base[4];
    int8_t* q_out[4];
    for (size_t k = 0; k < 4; ++k) {
      q_base[k] = q_data + query_indices[qi + k] * q_stride;
      q_out[k] = all_q_decoded + (qi + k) * decoded_dim;
    }
    for (size_t c = 0; c < num_full_chunks; ++c) {
      for (size_t k = 0; k < 4; ++k) {
        const auto nibbles = hn::LoadU(u8, q_base[k] + c * N);
        hn::Vec<Int8> q_even, q_odd;
        DequantizeNibbles<Int8, Uint8>(i8, mask, centroids, nibbles, q_even,
                                       q_odd);
        hn::StoreU(q_even, i8, q_out[k] + c * 2 * N);
        hn::StoreU(q_odd, i8, q_out[k] + c * 2 * N + N);
      }
    }
    if (remainder > 0) {
      const size_t c = num_full_chunks;
      for (size_t k = 0; k < 4; ++k) {
        const auto nibbles = hn::LoadU(u8, q_base[k] + c * N);
        hn::Vec<Int8> q_even, q_odd;
        DequantizeNibbles<Int8, Uint8>(i8, mask, centroids, nibbles, q_even,
                                       q_odd);
        hn::StoreU(hn::IfThenElseZero(valid_mask, q_even), i8,
                   q_out[k] + c * 2 * N);
        hn::StoreU(hn::IfThenElseZero(valid_mask, q_odd), i8,
                   q_out[k] + c * 2 * N + N);
      }
    }
  }
  for (; qi < num_queries; ++qi) {
    const uint8_t* q_base = q_data + query_indices[qi] * q_stride;
    int8_t* q_out = all_q_decoded + qi * decoded_dim;
    for (size_t c = 0; c < num_full_chunks; ++c) {
      const auto nibbles = hn::LoadU(u8, q_base + c * N);
      hn::Vec<Int8> q_even, q_odd;
      DequantizeNibbles<Int8, Uint8>(i8, mask, centroids, nibbles, q_even,
                                     q_odd);
      hn::StoreU(q_even, i8, q_out + c * 2 * N);
      hn::StoreU(q_odd, i8, q_out + c * 2 * N + N);
    }
    if (remainder > 0) {
      const size_t c = num_full_chunks;
      const auto nibbles = hn::LoadU(u8, q_base + c * N);
      hn::Vec<Int8> q_even, q_odd;
      DequantizeNibbles<Int8, Uint8>(i8, mask, centroids, nibbles, q_even,
                                     q_odd);
      hn::StoreU(hn::IfThenElseZero(valid_mask, q_even), i8, q_out + c * 2 * N);
      hn::StoreU(hn::IfThenElseZero(valid_mask, q_odd), i8,
                 q_out + c * 2 * N + N);
    }
  }

  if (dimensionality > 0 && dimensionality % 2 != 0) {
    const size_t last_byte = bytes_per_point - 1;
    const size_t chunk = last_byte / N;
    const size_t lane = last_byte % N;
    const size_t off = chunk * 2 * N + N + lane;
    for (size_t q = 0; q < num_queries; ++q) {
      all_q_decoded[q * decoded_dim + off] = 0;
    }
  }
}

// Computes the sum of decoded query byte values for each query.
//
// This is used by the score correction step (see README.md § Score Correction):
// the asymmetric int8 × uint8 layout requires subtracting 128 * sum(q_i) from
// each raw dot product. This function computes sum(q_i) upfront so the
// correction is a cheap vector subtraction in the scoring epilogue.
HWY_INLINE void ComputeQueryByteSums(
    const ::ginger::TurboQuantDatasetSoA& queries,
    absl::Span<const int8_t> all_q_decoded_span, size_t num_queries,
    absl::Span<int32_t> all_q_byte_sums_span) {
  const size_t decoded_dim = DecodedQueryStride(queries);
  ABSL_CHECK_EQ(all_q_decoded_span.size(), num_queries * decoded_dim);
  ABSL_CHECK_EQ(all_q_byte_sums_span.size(), num_queries);
  const int8_t* HWY_RESTRICT all_q_decoded = all_q_decoded_span.data();
  int32_t* HWY_RESTRICT all_q_byte_sums = all_q_byte_sums_span.data();
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int8 = hn::ScalableTag<int8_t>;
  using Int32 = hn::ScalableTag<int32_t>;
  const Uint8 u8;
  const Int8 i8;
  const Int32 i32;
  const size_t N = hn::Lanes(u8);
  const size_t num_dim_chunks =
      hwy::DivCeil(queries.num_bytes_per_datapoint(), N);
  const auto ones = hn::Set(u8, uint8_t{1});
  for (size_t qi = 0; qi < num_queries; ++qi) {
    const int8_t* q_ptr = all_q_decoded + qi * decoded_dim;
    hn::Vec<Int32> accum = hn::Zero(i32);
    for (size_t j = 0; j < num_dim_chunks * 2; ++j) {
      const auto v = hn::Load(i8, q_ptr + j * N);
      accum = hn::SumOfMulQuadAccumulate(i32, ones, v, accum);
    }
    all_q_byte_sums[qi] = hn::ReduceSum(i32, accum);
  }
}

// Decodes queries, computes byte sums, and fetches norm scaling factors
// (and unquantized squared norms for L2 metric). Dispatches to either the
// gathered or contiguous decode implementation based on whether query_indices
// is contiguous or gathered.
//
// Parameters:
//   queries: Base points containing the quantized queries.
//   query_indices: Subset of queries to decode (either contiguous range or
//     gathered indices).
//   decoded: Output buffer where decoded queries are written as a flattened 2D
//     array of shape [query_indices.size(), decoded_dimension].
//   byte_sums: Output buffer for query byte sums.
//   norms: Output buffer for norm scaling factors.
//   sqr_norms: Output buffer for unquantized squared norms (used for L2).
//
// Returns a DecodedQueryView bundling the output buffers.
HWY_INLINE DecodedQueryView DecodeQueriesAndMetadata(
    const ::ginger::TurboQuantDatasetSoA& queries, IndexSubset query_indices,
    absl::Span<int8_t> decoded, absl::Span<int32_t> byte_sums,
    absl::Span<float> norms, absl::Span<float> sqr_norms) {
  const size_t num_queries = query_indices.size();
  const size_t decoded_dim = DecodedQueryStride(queries);
  ABSL_CHECK_GE(decoded.size(), num_queries * decoded_dim);
  ABSL_CHECK_GE(byte_sums.size(), num_queries);
  ABSL_CHECK_GE(norms.size(), num_queries);
  const bool is_l2 =
      queries.metric() == TurboQuantDistanceMetric::kSquaredL2Distance;
  if (is_l2) {
    ABSL_CHECK_GE(sqr_norms.size(), num_queries);
  }

  const absl::Span<int8_t> decoded_subspan =
      decoded.subspan(0, num_queries * decoded_dim);
  DecodeQueries(queries, query_indices, decoded_subspan);

  for (size_t i = 0; i < num_queries; ++i) {
    const size_t src_idx = query_indices[i];
    norms[i] = queries.norm_scaling_factor(src_idx);
    if (is_l2) {
      sqr_norms[i] = queries.unquantized_squared_norm(src_idx);
    }
  }

  ComputeQueryByteSums(queries, absl::MakeConstSpan(decoded_subspan),
                       num_queries, byte_sums.subspan(0, num_queries));

  return {.decoded_data = decoded_subspan.data(),
          .byte_sums = byte_sums.data(),
          .norm_scaling_factors = norms.data(),
          .squared_norms = is_l2 ? sqr_norms.data() : nullptr,
          .decoded_dimension = decoded_dim,
          .is_l2 = is_l2};
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger

// clang-format off
#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_QUERIES_TOGGLE // NOLINT
// clang-format on
