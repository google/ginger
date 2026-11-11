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

// Unit tests for decode_queries-inl.h: nibble decoding, query decode.
//
// Each SIMD function is validated against a scalar reference implementation
// that performs the equivalent computation using plain C++ loops.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/many_to_many/decode_queries_test.cc"
// clang-format on
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

#ifndef DECODE_QUERIES_TEST_SCALAR_REF_DEFINED_
#define DECODE_QUERIES_TEST_SCALAR_REF_DEFINED_
namespace scalar_ref {

using ginger::turboquant_codebooks::kTurboQuantCentroidsInt8;

inline int8_t DecodeNibbleSigned(uint8_t packed_byte, int parity) {
  uint8_t idx = (parity == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
  return kTurboQuantCentroidsInt8[idx];
}

// Scalar reference for DecodeQueries.
inline void ScalarDecodeQueries(const uint8_t* q_data, size_t q_stride,
                                size_t num_queries, size_t bytes_per_point,
                                size_t num_dim_chunks, size_t N,
                                int8_t* all_q_decoded,
                                absl::Span<const int> indices,
                                size_t dimensionality) {
  if (dimensionality == 0) return;
  const size_t decoded_dim = num_dim_chunks * 2 * N;
  for (size_t qi = 0; qi < num_queries; ++qi) {
    const size_t src = indices.empty() ? qi : indices[qi];
    const uint8_t* q_row = q_data + src * q_stride;
    int8_t* q_out = all_q_decoded + qi * decoded_dim;
    std::memset(q_out, 0, decoded_dim);

    for (size_t byte_idx = 0; byte_idx < bytes_per_point; ++byte_idx) {
      size_t chunk = byte_idx / N;
      size_t lane = byte_idx % N;
      int8_t even = DecodeNibbleSigned(q_row[byte_idx], 0);
      int8_t odd = DecodeNibbleSigned(q_row[byte_idx], 1);
      q_out[chunk * 2 * N + lane] = even;
      q_out[chunk * 2 * N + N + lane] = odd;
    }
    if (dimensionality > 0 && dimensionality % 2 != 0) {
      size_t last_byte = bytes_per_point - 1;
      size_t chunk = last_byte / N;
      size_t lane = last_byte % N;
      q_out[chunk * 2 * N + N + lane] = 0;
    }
  }
}

// Scalar reference for ComputeQueryByteSums.
inline void ScalarComputeQueryByteSums(const int8_t* all_q_decoded,
                                       size_t num_queries,
                                       size_t num_dim_chunks, size_t N,
                                       absl::Span<int32_t> all_q_byte_sums) {
  const size_t decoded_dim = num_dim_chunks * 2 * N;
  for (size_t qi = 0; qi < num_queries; ++qi) {
    const int8_t* q_ptr = all_q_decoded + qi * decoded_dim;
    int32_t sum = 0;
    for (size_t i = 0; i < decoded_dim; ++i) {
      sum += static_cast<int32_t>(q_ptr[i]);
    }
    all_q_byte_sums[qi] = sum;
  }
}

}  // namespace scalar_ref
#endif  // DECODE_QUERIES_TEST_SCALAR_REF_DEFINED_

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/decode_queries-inl.h"
#include "ginger/turboquant/many_to_many/index_subset.h"

namespace ginger {
namespace HWY_NAMESPACE {
namespace {}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

HWY_NOINLINE void TestDecodeQueriesVsScalar() {
  const size_t N = hn::Lanes(hn::ScalableTag<int8_t>());

  for (size_t dim = 1; dim <= 128; ++dim) {
    const size_t bytes_per_point = (dim + 1) / 2;
    const size_t num_dim_chunks =
        std::max(size_t{1}, (bytes_per_point + N - 1) / N);
    const size_t decoded_dim = num_dim_chunks * 2 * N;

    for (size_t num_queries : {1, 3, 4, 5, 8}) {
      const size_t q_stride = bytes_per_point;
      const size_t q_packed_size = num_queries * num_dim_chunks * N;

      hwy::AlignedVector<uint8_t> q_packed(q_packed_size, 0);
      for (size_t q = 0; q < num_queries; ++q) {
        for (size_t i = 0; i < bytes_per_point; ++i) {
          q_packed[q * q_stride + i] =
              static_cast<uint8_t>((i * 3 + q * 7 + 1) % 256);
        }
      }

      ginger::TurboQuantCodec tq(dim, 42);
      auto base_points_or =
          ginger::TurboQuantDatasetSoA::Create(tq, num_queries);
      ASSERT_TRUE(base_points_or.ok());
      auto& base_points = *base_points_or;

      for (size_t q = 0; q < num_queries; ++q) {
        ginger::TurboQuantBasePointView view{
            .quantized_data =
                absl::MakeConstSpan(&q_packed[q * q_stride], bytes_per_point)};
        ASSERT_TRUE(base_points.UpdatePoint(view, q).ok());
      }

      hwy::AlignedVector<int8_t> simd_decoded(num_queries * decoded_dim, 0x7F);
      ginger::HWY_NAMESPACE::DecodeQueries(
          base_points, ginger::IndexSubset::Contiguous(0, num_queries),
          absl::Span<int8_t>(simd_decoded.data(), simd_decoded.size()));

      hwy::AlignedVector<int8_t> scalar_decoded(num_queries * decoded_dim,
                                                0x7F);
      scalar_ref::ScalarDecodeQueries(q_packed.data(), q_stride, num_queries,
                                      bytes_per_point, num_dim_chunks, N,
                                      scalar_decoded.data(), {}, dim);

      EXPECT_EQ(simd_decoded, scalar_decoded);
    }
  }
}

HWY_NOINLINE void TestDecodeQueriesWithIndices() {
  const size_t N = hn::Lanes(hn::ScalableTag<int8_t>());
  const size_t dim = 32;
  const size_t bytes_per_point = 16;
  const size_t num_dim_chunks = (bytes_per_point + N - 1) / N;
  const size_t decoded_dim = num_dim_chunks * 2 * N;
  const size_t num_total = 8;
  const size_t num_queries = 4;

  hwy::AlignedVector<uint8_t> q_packed(num_total * num_dim_chunks * N, 0);
  for (size_t q = 0; q < num_total; ++q) {
    for (size_t i = 0; i < bytes_per_point; ++i) {
      q_packed[q * bytes_per_point + i] =
          static_cast<uint8_t>((q * 17 + i * 5) % 256);
    }
  }

  ginger::TurboQuantCodec tq(dim, 42);
  auto base_points_or = ginger::TurboQuantDatasetSoA::Create(tq, num_total);
  ASSERT_TRUE(base_points_or.ok());
  auto& base_points = *base_points_or;

  for (size_t q = 0; q < num_total; ++q) {
    ginger::TurboQuantBasePointView view{
        .quantized_data = absl::MakeConstSpan(&q_packed[q * bytes_per_point],
                                              bytes_per_point)};
    ASSERT_TRUE(base_points.UpdatePoint(view, q).ok());
  }

  std::vector<int> indices = {7, 5, 2, 0};

  hwy::AlignedVector<int8_t> simd_decoded(num_queries * decoded_dim, 0);
  ginger::HWY_NAMESPACE::DecodeQueries(
      base_points, ginger::IndexSubset::Gathered(indices),
      absl::Span<int8_t>(simd_decoded.data(), simd_decoded.size()));

  hwy::AlignedVector<int8_t> scalar_decoded(num_queries * decoded_dim, 0);
  scalar_ref::ScalarDecodeQueries(q_packed.data(), bytes_per_point, num_queries,
                                  bytes_per_point, num_dim_chunks, N,
                                  scalar_decoded.data(), indices, dim);

  EXPECT_EQ(simd_decoded, scalar_decoded);
}

HWY_NOINLINE void TestComputeQueryByteSumsVsScalar() {
  const size_t N = hn::Lanes(hn::ScalableTag<int8_t>());

  for (size_t dim = 1; dim <= 128; ++dim) {
    const size_t bytes_per_point = (dim + 1) / 2;
    const size_t num_dim_chunks =
        std::max(size_t{1}, (bytes_per_point + N - 1) / N);
    const size_t decoded_dim = num_dim_chunks * 2 * N;

    for (size_t num_queries : {1, 3, 4, 7}) {
      hwy::AlignedVector<uint8_t> q_packed(num_queries * num_dim_chunks * N, 0);
      for (size_t q = 0; q < num_queries; ++q) {
        for (size_t i = 0; i < bytes_per_point; ++i) {
          q_packed[q * bytes_per_point + i] =
              static_cast<uint8_t>((i * 3 + q * 11) % 256);
        }
      }

      ginger::TurboQuantCodec tq(dim, 42);
      auto base_points_or =
          ginger::TurboQuantDatasetSoA::Create(tq, num_queries);
      ASSERT_TRUE(base_points_or.ok());
      auto& base_points = *base_points_or;

      for (size_t q = 0; q < num_queries; ++q) {
        ginger::TurboQuantBasePointView view{
            .quantized_data = absl::MakeConstSpan(
                &q_packed[q * bytes_per_point], bytes_per_point)};
        ASSERT_TRUE(base_points.UpdatePoint(view, q).ok());
      }

      hwy::AlignedVector<int8_t> q_decoded(num_queries * decoded_dim, 0);

      ginger::HWY_NAMESPACE::DecodeQueries(
          base_points, ginger::IndexSubset::Contiguous(0, num_queries),
          absl::Span<int8_t>(q_decoded.data(), q_decoded.size()));

      hwy::AlignedVector<int32_t> simd_sums(num_queries, 0);
      ginger::HWY_NAMESPACE::ComputeQueryByteSums(
          base_points,
          absl::Span<const int8_t>(q_decoded.data(), q_decoded.size()),
          num_queries, absl::Span<int32_t>(simd_sums.data(), simd_sums.size()));

      hwy::AlignedVector<int32_t> scalar_sums(num_queries, 0);
      scalar_ref::ScalarComputeQueryByteSums(q_decoded.data(), num_queries,
                                             num_dim_chunks, N,
                                             absl::MakeSpan(scalar_sums));

      EXPECT_EQ(simd_sums, scalar_sums);
    }
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(DecodeQueriesTest);
HWY_EXPORT_AND_TEST_P(DecodeQueriesTest, TestDecodeQueriesVsScalar);
HWY_EXPORT_AND_TEST_P(DecodeQueriesTest, TestDecodeQueriesWithIndices);
HWY_EXPORT_AND_TEST_P(DecodeQueriesTest, TestComputeQueryByteSumsVsScalar);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif
