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

#include "ginger/distance/distance_bytes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/types.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ginger/distance/distance_bytes.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// =============================================================================
// Helper Functions
// =============================================================================

// Loads N 8-bit integers from ptr, promoting each 8-bit element into a 16-bit
// signed integer lane of the returned SIMD vector.
HWY_INLINE hn::Vec<hn::ScalableTag<int16_t>> LoadAndPromoteToInt16(
    hn::ScalableTag<int16_t> d16, const int8_t* HWY_RESTRICT ptr) {
  const hn::Rebind<int8_t, decltype(d16)> d8;
  return hn::PromoteTo(d16, hn::LoadU(d8, ptr));
}
HWY_INLINE hn::Vec<hn::ScalableTag<int16_t>> LoadAndPromoteToInt16(
    hn::ScalableTag<int16_t> d16, const uint8_t* HWY_RESTRICT ptr) {
  const hn::Rebind<uint8_t, decltype(d16)> du8;
  const hn::Rebind<uint16_t, decltype(d16)> du16;
  return hn::BitCast(d16, hn::PromoteTo(du16, hn::LoadU(du8, ptr)));
}

// Prefetches up to the first `kMaxPrefetchBytes` of candidate vector data of
// dimension `dim` from `ptr` into cache. Capping the prefetch per vector
// avoids saturating Line Fill Buffers on high-dimensional vectors while
// triggering the hardware streamer for any remaining sequential cache lines.
template <typename ByteT>
HWY_INLINE void PrefetchPoint(const ByteT* HWY_RESTRICT ptr, size_t dim) {
  constexpr size_t kMaxPrefetchBytes = 256;
  const char* data = reinterpret_cast<const char*>(ptr);
  const size_t bytes = std::min(dim * sizeof(ByteT), kMaxPrefetchBytes);
  for (size_t offset = 0; offset < bytes; offset += 64) {
    absl::PrefetchToLocalCache(data + offset);
  }
}

// Number of candidate points that the batch kernels prefetch ahead of use. A
// core can only track a limited number of outstanding L1 misses (one per line
// fill buffer); further prefetches wait in the load buffer until one frees up.
// Prefetching this far ahead keeps all fill buffers busy on DRAM-resident
// datasets, which a distance of 8 did not.
constexpr size_t kBatchPrefetchDistance = 20;

// Returns a pointer to the coordinates of point `id` in `dataset`. Unlike
// `SpanDataset::get`, this does not check the coordinate type on every call, so
// callers must check `dataset.coordinate_type()` once before using it.
template <typename ByteT>
HWY_INLINE const ByteT* UncheckedPointPtr(const SpanDataset& dataset,
                                          GingerNodeId id) {
  return static_cast<const ByteT*>(dataset[id]);
}

// =============================================================================
// Dot Product Functions
// =============================================================================

// Computes the dot product between a byte query vector `a` (int8_t or uint8_t)
// and a dataset point vector `b` (int8_t or uint8_t).
template <typename ByteT>
HWY_INLINE float DotProductSingleByteQuerySimd(const ByteT* HWY_RESTRICT a,
                                               const ByteT* HWY_RESTRICT b,
                                               const size_t dim) {
  using WideT = std::conditional_t<std::is_signed_v<ByteT>, int32_t, uint32_t>;
  const hn::ScalableTag<WideT> d32;
  const hn::Repartition<ByteT, decltype(d32)> d8;
  const size_t N = hn::Lanes(d8);

  auto sum0 = hn::Zero(d32);
  auto sum1 = hn::Zero(d32);
  auto sum2 = hn::Zero(d32);
  auto sum3 = hn::Zero(d32);

  // Loads a SIMD lane from each vector and accumulates the 4-byte dot products
  // using SumOfMulQuadAccumulate into the target 32-bit sum accumulator.
  const auto accumulate = [&](auto& sum, size_t offset) {
    const auto qa = hn::LoadU(d8, a + offset);
    const auto qb = hn::LoadU(d8, b + offset);
    sum = hn::SumOfMulQuadAccumulate(d32, qa, qb, sum);
  };

  // Processes vector dimensions in SIMD chunks using hierarchical unrolling
  // of 4N, 2N, and N lanes.
  size_t i = 0;
  for (; i + 4 * N <= dim; i += 4 * N) {
    accumulate(sum0, i);
    accumulate(sum1, i + N);
    accumulate(sum2, i + 2 * N);
    accumulate(sum3, i + 3 * N);
  }
  if (i + 2 * N <= dim) {
    accumulate(sum0, i);
    accumulate(sum1, i + N);
    i += 2 * N;
  }
  if (i + N <= dim) {
    accumulate(sum2, i);
    i += N;
  }

  // Combines the sum of the four partial SIMD accumulators and handle any
  // remaining scalar tail elements that do not fill a SIMD lane.
  auto sum = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
  WideT total = hn::ReduceSum(d32, sum);

  for (; i < dim; ++i) {
    total += static_cast<WideT>(a[i]) * static_cast<WideT>(b[i]);
  }
  return static_cast<float>(total);
}

// Computes negative dot product distances between a byte query vector `query`
// (int8_t or uint8_t) and dataset point vectors (int8_t or uint8_t) for all
// neighbors.
template <typename ByteT>
void DotProductBatchByteQuerySimdImpl(absl::Span<const ByteT> query,
                                      absl::Span<DistanceAndId> neighbors,
                                      const SpanDataset& dataset) {
  using WideT = std::conditional_t<std::is_signed_v<ByteT>, int32_t, uint32_t>;
  const hn::ScalableTag<WideT> d32;
  const hn::Repartition<ByteT, decltype(d32)> d8;
  const size_t N = hn::Lanes(d8);

  const size_t num_neighbors = neighbors.size();
  if (num_neighbors == 0) {
    return;
  }
  ABSL_CHECK_EQ(dataset.coordinate_type(), CoordinateTypeFor<ByteT>());
  const size_t dim = query.size();
  const ByteT* HWY_RESTRICT q_ptr = query.data();
  DistanceAndId* nbr_ptr = neighbors.data();

  for (size_t p = 0; p < kBatchPrefetchDistance && p < num_neighbors; ++p) {
    PrefetchPoint(UncheckedPointPtr<ByteT>(dataset, nbr_ptr[p].id), dim);
  }

  size_t j = 0;
  for (; j + 4 <= num_neighbors; j += 4) {
    const size_t prefetch_end =
        std::min(j + kBatchPrefetchDistance + 4, num_neighbors);
    for (size_t p = j + kBatchPrefetchDistance; p < prefetch_end; ++p) {
      PrefetchPoint(UncheckedPointPtr<ByteT>(dataset, nbr_ptr[p].id), dim);
    }

    const ByteT* c0 = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j].id);
    const ByteT* c1 = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j + 1].id);
    const ByteT* c2 = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j + 2].id);
    const ByteT* c3 = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j + 3].id);

    auto acc0 = hn::Zero(d32);
    auto acc1 = hn::Zero(d32);
    auto acc2 = hn::Zero(d32);
    auto acc3 = hn::Zero(d32);

    size_t i = 0;
    for (; i + N <= dim; i += N) {
      const auto q = hn::LoadU(d8, q_ptr + i);
      acc0 = hn::SumOfMulQuadAccumulate(d32, q, hn::LoadU(d8, c0 + i), acc0);
      acc1 = hn::SumOfMulQuadAccumulate(d32, q, hn::LoadU(d8, c1 + i), acc1);
      acc2 = hn::SumOfMulQuadAccumulate(d32, q, hn::LoadU(d8, c2 + i), acc2);
      acc3 = hn::SumOfMulQuadAccumulate(d32, q, hn::LoadU(d8, c3 + i), acc3);
    }

    WideT dot0 = hn::ReduceSum(d32, acc0);
    WideT dot1 = hn::ReduceSum(d32, acc1);
    WideT dot2 = hn::ReduceSum(d32, acc2);
    WideT dot3 = hn::ReduceSum(d32, acc3);

    for (; i < dim; ++i) {
      dot0 += static_cast<WideT>(q_ptr[i]) * static_cast<WideT>(c0[i]);
      dot1 += static_cast<WideT>(q_ptr[i]) * static_cast<WideT>(c1[i]);
      dot2 += static_cast<WideT>(q_ptr[i]) * static_cast<WideT>(c2[i]);
      dot3 += static_cast<WideT>(q_ptr[i]) * static_cast<WideT>(c3[i]);
    }

    nbr_ptr[j].distance = -static_cast<float>(dot0);
    nbr_ptr[j + 1].distance = -static_cast<float>(dot1);
    nbr_ptr[j + 2].distance = -static_cast<float>(dot2);
    nbr_ptr[j + 3].distance = -static_cast<float>(dot3);
  }

  for (; j < num_neighbors; ++j) {
    const ByteT* c = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j].id);
    nbr_ptr[j].distance = -DotProductSingleByteQuerySimd(q_ptr, c, dim);
  }
}

// =============================================================================
// Squared L2 Functions
// =============================================================================

// Computes the squared L2 distance between a byte query vector `a` (int8_t or
// uint8_t) and a dataset point vector `b` (int8_t or uint8_t).
template <typename ByteT>
HWY_INLINE float SquaredL2SingleByteQuerySimd(const ByteT* HWY_RESTRICT a,
                                              const ByteT* HWY_RESTRICT b,
                                              const size_t dim) {
  const hn::ScalableTag<int32_t> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  const size_t N = hn::Lanes(d16);

  auto sum0 = hn::Zero(d32);
  auto sum1 = hn::Zero(d32);
  auto sum2 = hn::Zero(d32);
  auto sum3 = hn::Zero(d32);

  auto sum0_extra = hn::Zero(d32);
  auto sum1_extra = hn::Zero(d32);
  auto sum2_extra = hn::Zero(d32);
  auto sum3_extra = hn::Zero(d32);

  // Loads a SIMD lane from each vector, promotes byte values to int16,
  // computes their difference, and accumulates the squared difference using
  // ReorderWidenMulAccumulate into the target 32-bit sum accumulators.
  const auto accumulate = [&](auto& sum, auto& sum_extra, size_t offset) {
    const auto qa = LoadAndPromoteToInt16(d16, a + offset);
    const auto qb = LoadAndPromoteToInt16(d16, b + offset);
    const auto diff = hn::Sub(qa, qb);
    sum = hn::ReorderWidenMulAccumulate(d32, diff, diff, sum, sum_extra);
  };

  // Processes vector dimensions in SIMD chunks using hierarchical unrolling
  // of 4N, 2N, and N lanes.
  size_t i = 0;
  for (; i + 4 * N <= dim; i += 4 * N) {
    accumulate(sum0, sum0_extra, i);
    accumulate(sum1, sum1_extra, i + N);
    accumulate(sum2, sum2_extra, i + 2 * N);
    accumulate(sum3, sum3_extra, i + 3 * N);
  }
  if (i + 2 * N <= dim) {
    accumulate(sum0, sum0_extra, i);
    accumulate(sum1, sum1_extra, i + N);
    i += 2 * N;
  }
  if (i + N <= dim) {
    accumulate(sum2, sum2_extra, i);
    i += N;
  }

  // Combines the sum of the four partial SIMD accumulators and handle any
  // remaining scalar tail elements that do not fill a SIMD lane.
  auto s0 = hn::Add(sum0, sum0_extra);
  auto s1 = hn::Add(sum1, sum1_extra);
  auto s2 = hn::Add(sum2, sum2_extra);
  auto s3 = hn::Add(sum3, sum3_extra);
  auto sum = hn::Add(hn::Add(s0, s1), hn::Add(s2, s3));
  int32_t total = hn::ReduceSum(d32, sum);

  for (; i < dim; ++i) {
    int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
    total += diff * diff;
  }
  return static_cast<float>(total);
}

// Computes squared L2 distances between a byte query vector `query` (int8_t or
// uint8_t) and dataset point vectors (int8_t or uint8_t) for all neighbors.
template <typename ByteT>
void SquaredL2BatchByteQuerySimdImpl(absl::Span<const ByteT> query,
                                     absl::Span<DistanceAndId> neighbors,
                                     const SpanDataset& dataset) {
  const hn::ScalableTag<int32_t> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  const size_t N = hn::Lanes(d16);

  const size_t num_neighbors = neighbors.size();
  if (num_neighbors == 0) {
    return;
  }
  ABSL_CHECK_EQ(dataset.coordinate_type(), CoordinateTypeFor<ByteT>());
  const size_t dim = query.size();
  const ByteT* HWY_RESTRICT q_ptr = query.data();
  DistanceAndId* nbr_ptr = neighbors.data();

  for (size_t p = 0; p < kBatchPrefetchDistance && p < num_neighbors; ++p) {
    PrefetchPoint(UncheckedPointPtr<ByteT>(dataset, nbr_ptr[p].id), dim);
  }

  size_t j = 0;
  for (; j + 4 <= num_neighbors; j += 4) {
    const size_t prefetch_end =
        std::min(j + kBatchPrefetchDistance + 4, num_neighbors);
    for (size_t p = j + kBatchPrefetchDistance; p < prefetch_end; ++p) {
      PrefetchPoint(UncheckedPointPtr<ByteT>(dataset, nbr_ptr[p].id), dim);
    }

    const ByteT* c0 = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j].id);
    const ByteT* c1 = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j + 1].id);
    const ByteT* c2 = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j + 2].id);
    const ByteT* c3 = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j + 3].id);

    auto acc0 = hn::Zero(d32);
    auto acc1 = hn::Zero(d32);
    auto acc2 = hn::Zero(d32);
    auto acc3 = hn::Zero(d32);

    auto acc0_extra = hn::Zero(d32);
    auto acc1_extra = hn::Zero(d32);
    auto acc2_extra = hn::Zero(d32);
    auto acc3_extra = hn::Zero(d32);

    size_t i = 0;
    for (; i + N <= dim; i += N) {
      const auto q = LoadAndPromoteToInt16(d16, q_ptr + i);
      const auto diff0 = hn::Sub(q, LoadAndPromoteToInt16(d16, c0 + i));
      const auto diff1 = hn::Sub(q, LoadAndPromoteToInt16(d16, c1 + i));
      const auto diff2 = hn::Sub(q, LoadAndPromoteToInt16(d16, c2 + i));
      const auto diff3 = hn::Sub(q, LoadAndPromoteToInt16(d16, c3 + i));

      acc0 = hn::ReorderWidenMulAccumulate(d32, diff0, diff0, acc0, acc0_extra);
      acc1 = hn::ReorderWidenMulAccumulate(d32, diff1, diff1, acc1, acc1_extra);
      acc2 = hn::ReorderWidenMulAccumulate(d32, diff2, diff2, acc2, acc2_extra);
      acc3 = hn::ReorderWidenMulAccumulate(d32, diff3, diff3, acc3, acc3_extra);
    }

    int32_t dist0 = hn::ReduceSum(d32, hn::Add(acc0, acc0_extra));
    int32_t dist1 = hn::ReduceSum(d32, hn::Add(acc1, acc1_extra));
    int32_t dist2 = hn::ReduceSum(d32, hn::Add(acc2, acc2_extra));
    int32_t dist3 = hn::ReduceSum(d32, hn::Add(acc3, acc3_extra));

    for (; i < dim; ++i) {
      const int32_t diff0 =
          static_cast<int32_t>(q_ptr[i]) - static_cast<int32_t>(c0[i]);
      const int32_t diff1 =
          static_cast<int32_t>(q_ptr[i]) - static_cast<int32_t>(c1[i]);
      const int32_t diff2 =
          static_cast<int32_t>(q_ptr[i]) - static_cast<int32_t>(c2[i]);
      const int32_t diff3 =
          static_cast<int32_t>(q_ptr[i]) - static_cast<int32_t>(c3[i]);
      dist0 += diff0 * diff0;
      dist1 += diff1 * diff1;
      dist2 += diff2 * diff2;
      dist3 += diff3 * diff3;
    }

    nbr_ptr[j].distance = static_cast<float>(dist0);
    nbr_ptr[j + 1].distance = static_cast<float>(dist1);
    nbr_ptr[j + 2].distance = static_cast<float>(dist2);
    nbr_ptr[j + 3].distance = static_cast<float>(dist3);
  }

  for (; j < num_neighbors; ++j) {
    const ByteT* c = UncheckedPointPtr<ByteT>(dataset, nbr_ptr[j].id);
    nbr_ptr[j].distance = SquaredL2SingleByteQuerySimd(q_ptr, c, dim);
  }
}
// =============================================================================
// Dispatch Templates
// =============================================================================

template <typename ByteT>
void DotProductBatchByteQuerySimd(absl::Span<const ByteT> query,
                                  absl::Span<DistanceAndId> neighbors,
                                  const SpanDataset& dataset) {
  DotProductBatchByteQuerySimdImpl<ByteT>(query, neighbors, dataset);
}

template <typename ByteT>
void SquaredL2BatchByteQuerySimd(absl::Span<const ByteT> query,
                                 absl::Span<DistanceAndId> neighbors,
                                 const SpanDataset& dataset) {
  SquaredL2BatchByteQuerySimdImpl<ByteT>(query, neighbors, dataset);
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {

void DotProductDistance(absl::Span<const int8_t> query,
                        absl::Span<DistanceAndId> neighbors,
                        const SpanDataset& dataset) {
  ABSL_DCHECK_EQ(query.size(), dataset.dimensionality());
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(DotProductBatchByteQuerySimd<int8_t>)(
      query, neighbors, dataset);
}

void SquaredL2Distance(absl::Span<const int8_t> query,
                       absl::Span<DistanceAndId> neighbors,
                       const SpanDataset& dataset) {
  ABSL_DCHECK_EQ(query.size(), dataset.dimensionality());
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(SquaredL2BatchByteQuerySimd<int8_t>)(
      query, neighbors, dataset);
}

void DotProductDistance(absl::Span<const uint8_t> query,
                        absl::Span<DistanceAndId> neighbors,
                        const SpanDataset& dataset) {
  ABSL_DCHECK_EQ(query.size(), dataset.dimensionality());
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(DotProductBatchByteQuerySimd<uint8_t>)(
      query, neighbors, dataset);
}

void SquaredL2Distance(absl::Span<const uint8_t> query,
                       absl::Span<DistanceAndId> neighbors,
                       const SpanDataset& dataset) {
  ABSL_DCHECK_EQ(query.size(), dataset.dimensionality());
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(SquaredL2BatchByteQuerySimd<uint8_t>)(
      query, neighbors, dataset);
}

}  // namespace ginger
#endif  // HWY_ONCE
