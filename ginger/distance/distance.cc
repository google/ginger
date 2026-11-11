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

#include "ginger/distance/distance.h"

#include <cstddef>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/types.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ginger/distance/distance.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Single dot product with hierarchical unrolling (4N -> 2N -> N)
template <class D>
HWY_INLINE float DotProductSingleSimd(D d, const float* HWY_RESTRICT a,
                                      const float* HWY_RESTRICT b, size_t dim) {
  const size_t N = hn::Lanes(d);
  auto sum0 = hn::Zero(d);
  auto sum1 = hn::Zero(d);
  auto sum2 = hn::Zero(d);
  auto sum3 = hn::Zero(d);

  size_t i = 0;
  for (; i + 4 * N <= dim; i += 4 * N) {
    sum0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum0);
    sum1 = hn::MulAdd(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N), sum1);
    sum2 = hn::MulAdd(hn::LoadU(d, a + i + 2 * N), hn::LoadU(d, b + i + 2 * N),
                      sum2);
    sum3 = hn::MulAdd(hn::LoadU(d, a + i + 3 * N), hn::LoadU(d, b + i + 3 * N),
                      sum3);
  }
  if (i + 2 * N <= dim) {
    sum0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum0);
    sum1 = hn::MulAdd(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N), sum1);
    i += 2 * N;
  }
  if (i + N <= dim) {
    sum2 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum2);
    i += N;
  }

  auto sum = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
  float total = hn::ReduceSum(d, sum);

  for (; i < dim; ++i) {
    total += a[i] * b[i];
  }
  return total;
}

// Single squared L2 distance with hierarchical unrolling (4N -> 2N -> N)
template <class D>
HWY_INLINE float SquaredL2SingleSimd(D d, const float* HWY_RESTRICT a,
                                     const float* HWY_RESTRICT b, size_t dim) {
  const size_t N = hn::Lanes(d);
  auto sum0 = hn::Zero(d);
  auto sum1 = hn::Zero(d);
  auto sum2 = hn::Zero(d);
  auto sum3 = hn::Zero(d);

  size_t i = 0;
  for (; i + 4 * N <= dim; i += 4 * N) {
    const auto diff0 = hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
    const auto diff1 =
        hn::Sub(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N));
    const auto diff2 =
        hn::Sub(hn::LoadU(d, a + i + 2 * N), hn::LoadU(d, b + i + 2 * N));
    const auto diff3 =
        hn::Sub(hn::LoadU(d, a + i + 3 * N), hn::LoadU(d, b + i + 3 * N));
    sum0 = hn::MulAdd(diff0, diff0, sum0);
    sum1 = hn::MulAdd(diff1, diff1, sum1);
    sum2 = hn::MulAdd(diff2, diff2, sum2);
    sum3 = hn::MulAdd(diff3, diff3, sum3);
  }
  if (i + 2 * N <= dim) {
    const auto diff0 = hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
    const auto diff1 =
        hn::Sub(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N));
    sum0 = hn::MulAdd(diff0, diff0, sum0);
    sum1 = hn::MulAdd(diff1, diff1, sum1);
    i += 2 * N;
  }
  if (i + N <= dim) {
    const auto diff0 = hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
    sum2 = hn::MulAdd(diff0, diff0, sum2);
    i += N;
  }

  auto sum = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
  float total = hn::ReduceSum(d, sum);

  for (; i < dim; ++i) {
    const float diff = a[i] - b[i];
    total += diff * diff;
  }
  return total;
}

template <typename Dataset>
void DotProductDistanceSimd(absl::Span<const float> query,
                            absl::Span<DistanceAndId> neighbors,
                            const Dataset& dataset) {
  const hn::ScalableTag<float> d;
  const size_t N = hn::Lanes(d);
  const size_t dim = query.size();
  const float* HWY_RESTRICT q_ptr = query.data();

  size_t j = 0;
  const size_t num_neighbors = neighbors.size();
  for (; j + 4 <= num_neighbors; j += 4) {
    const float* c0 = dataset.template get<float>(neighbors[j].id).data();
    const float* c1 = dataset.template get<float>(neighbors[j + 1].id).data();
    const float* c2 = dataset.template get<float>(neighbors[j + 2].id).data();
    const float* c3 = dataset.template get<float>(neighbors[j + 3].id).data();

    auto acc0 = hn::Zero(d);
    auto acc1 = hn::Zero(d);
    auto acc2 = hn::Zero(d);
    auto acc3 = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= dim; i += N) {
      const auto q = hn::LoadU(d, q_ptr + i);
      acc0 = hn::MulAdd(q, hn::LoadU(d, c0 + i), acc0);
      acc1 = hn::MulAdd(q, hn::LoadU(d, c1 + i), acc1);
      acc2 = hn::MulAdd(q, hn::LoadU(d, c2 + i), acc2);
      acc3 = hn::MulAdd(q, hn::LoadU(d, c3 + i), acc3);
    }

    float dot0 = hn::ReduceSum(d, acc0);
    float dot1 = hn::ReduceSum(d, acc1);
    float dot2 = hn::ReduceSum(d, acc2);
    float dot3 = hn::ReduceSum(d, acc3);

    for (; i < dim; ++i) {
      const float q = q_ptr[i];
      dot0 += q * c0[i];
      dot1 += q * c1[i];
      dot2 += q * c2[i];
      dot3 += q * c3[i];
    }

    neighbors[j].distance = -dot0;
    neighbors[j + 1].distance = -dot1;
    neighbors[j + 2].distance = -dot2;
    neighbors[j + 3].distance = -dot3;
  }

  for (; j < num_neighbors; ++j) {
    const float* c = dataset.template get<float>(neighbors[j].id).data();
    neighbors[j].distance = -DotProductSingleSimd(d, q_ptr, c, dim);
  }
}

template <typename Dataset>
void SquaredL2DistanceSimd(absl::Span<const float> query,
                           absl::Span<DistanceAndId> neighbors,
                           const Dataset& dataset) {
  const hn::ScalableTag<float> d;
  const size_t N = hn::Lanes(d);
  const size_t dim = query.size();
  const float* HWY_RESTRICT q_ptr = query.data();

  size_t j = 0;
  const size_t num_neighbors = neighbors.size();
  for (; j + 4 <= num_neighbors; j += 4) {
    const float* c0 = dataset.template get<float>(neighbors[j].id).data();
    const float* c1 = dataset.template get<float>(neighbors[j + 1].id).data();
    const float* c2 = dataset.template get<float>(neighbors[j + 2].id).data();
    const float* c3 = dataset.template get<float>(neighbors[j + 3].id).data();

    auto acc0 = hn::Zero(d);
    auto acc1 = hn::Zero(d);
    auto acc2 = hn::Zero(d);
    auto acc3 = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= dim; i += N) {
      const auto q = hn::LoadU(d, q_ptr + i);
      const auto diff0 = hn::Sub(q, hn::LoadU(d, c0 + i));
      const auto diff1 = hn::Sub(q, hn::LoadU(d, c1 + i));
      const auto diff2 = hn::Sub(q, hn::LoadU(d, c2 + i));
      const auto diff3 = hn::Sub(q, hn::LoadU(d, c3 + i));
      acc0 = hn::MulAdd(diff0, diff0, acc0);
      acc1 = hn::MulAdd(diff1, diff1, acc1);
      acc2 = hn::MulAdd(diff2, diff2, acc2);
      acc3 = hn::MulAdd(diff3, diff3, acc3);
    }

    float dist0 = hn::ReduceSum(d, acc0);
    float dist1 = hn::ReduceSum(d, acc1);
    float dist2 = hn::ReduceSum(d, acc2);
    float dist3 = hn::ReduceSum(d, acc3);

    for (; i < dim; ++i) {
      const float q = q_ptr[i];
      const float diff0 = q - c0[i];
      const float diff1 = q - c1[i];
      const float diff2 = q - c2[i];
      const float diff3 = q - c3[i];
      dist0 += diff0 * diff0;
      dist1 += diff1 * diff1;
      dist2 += diff2 * diff2;
      dist3 += diff3 * diff3;
    }

    neighbors[j].distance = dist0;
    neighbors[j + 1].distance = dist1;
    neighbors[j + 2].distance = dist2;
    neighbors[j + 3].distance = dist3;
  }

  for (; j < num_neighbors; ++j) {
    const float* c = dataset.template get<float>(neighbors[j].id).data();
    neighbors[j].distance = SquaredL2SingleSimd(d, q_ptr, c, dim);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {

void DotProductDistance(absl::Span<const float> query,
                        absl::Span<DistanceAndId> neighbors,
                        const SpanDataset& dataset) {
  ABSL_DCHECK_EQ(query.size(), dataset.dimensionality());
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(DotProductDistanceSimd<SpanDataset>)(
      query, neighbors, dataset);
}

void SquaredL2Distance(absl::Span<const float> query,
                       absl::Span<DistanceAndId> neighbors,
                       const SpanDataset& dataset) {
  ABSL_DCHECK_EQ(query.size(), dataset.dimensionality());
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(SquaredL2DistanceSimd<SpanDataset>)(
      query, neighbors, dataset);
}

}  // namespace ginger
#endif  // HWY_ONCE
