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

#include "ginger/turboquant/chamfer/point_cloud_query.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/query/query.h"

#undef HWY_TARGET_INCLUDE

// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/chamfer/point_cloud_query.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/decode_queries-inl.h"

namespace ginger {


namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;
using ::ginger::HWY_NAMESPACE::DecodedQueryStride;
using ::ginger::HWY_NAMESPACE::DecodeQueriesAndMetadata;

absl::StatusOr<TurboQuantPointCloudQuery> CreateImpl(
    const TurboQuantCodec& codec, absl::Span<const float> raw_query) {
  const size_t dim = codec.dimensionality();
  if (dim == 0 || raw_query.size() % dim != 0) {
    return absl::InvalidArgumentError(
        "Raw query size must be a multiple of dimensionality.");
  }
  const size_t num_tokens = raw_query.size() / dim;
  if (num_tokens == 0) return TurboQuantPointCloudQuery();

  const size_t N = hn::Lanes(hn::ScalableTag<uint8_t>());
  const size_t num_dim_chunks =
      hwy::DivCeil(codec.num_bytes_per_datapoint(), N);
  const size_t decoded_dim = num_dim_chunks * 2 * N;
  hwy::AlignedVector<int8_t> decoded_data(num_tokens * decoded_dim, 0);
  std::vector<int32_t> byte_sums(num_tokens);
  std::vector<float> norm_scaling_factors(num_tokens);

  for (size_t i = 0; i < num_tokens; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        TurboQuantQuery q,
        TurboQuantQuery::Create(codec, raw_query.subspan(i * dim, dim), false));
    byte_sums[i] = q.query_byte_sum();
    norm_scaling_factors[i] = q.norm_scaling_factor();

    const int8_t* HWY_RESTRICT global_q = q.query_data();
    const size_t half_bytes = q.num_bytes_per_query() / 2;
    int8_t* HWY_RESTRICT out = decoded_data.data() + i * decoded_dim;
    for (size_t c = 0; c < num_dim_chunks; ++c) {
      const size_t src_offset = c * N;
      if (src_offset < half_bytes) {
        const size_t copy_len = std::min(N, half_bytes - src_offset);
        std::copy_n(global_q + src_offset, copy_len, out + c * 2 * N);
        std::copy_n(global_q + half_bytes + src_offset, copy_len,
                    out + c * 2 * N + N);
      }
    }
  }
  return TurboQuantPointCloudQuery(
      std::move(decoded_data), std::move(byte_sums),
      std::move(norm_scaling_factors), num_tokens, decoded_dim);
}

absl::StatusOr<TurboQuantPointCloudQuery> CreateFromQuantizedDatasetImpl(
    const TurboQuantDatasetSoA& dataset, IndexSubset query_subset) {
  const size_t num_tokens = query_subset.size();
  if (num_tokens == 0) return TurboQuantPointCloudQuery();

  const size_t decoded_dim = DecodedQueryStride(dataset);
  hwy::AlignedVector<int8_t> decoded_data(num_tokens * decoded_dim);
  std::vector<int32_t> byte_sums(num_tokens);
  std::vector<float> norm_scaling_factors(num_tokens);

  DecodeQueriesAndMetadata(dataset, query_subset, absl::MakeSpan(decoded_data),
                           absl::MakeSpan(byte_sums),
                           absl::MakeSpan(norm_scaling_factors),
                           /*sqr_norms=*/{});
  return TurboQuantPointCloudQuery(
      std::move(decoded_data), std::move(byte_sums),
      std::move(norm_scaling_factors), num_tokens, decoded_dim);
}

}  // namespace
}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
HWY_EXPORT(CreateImpl);
HWY_EXPORT(CreateFromQuantizedDatasetImpl);

absl::StatusOr<TurboQuantPointCloudQuery> TurboQuantPointCloudQuery::Create(
    const ginger::TurboQuantCodec& codec, absl::Span<const float> raw_query) {
  return HWY_DYNAMIC_DISPATCH(CreateImpl)(codec, raw_query);
}

absl::StatusOr<TurboQuantPointCloudQuery>
TurboQuantPointCloudQuery::CreateFromQuantizedDataset(
    const ginger::TurboQuantDatasetSoA& dataset,
    ginger::IndexSubset query_subset) {
  return HWY_DYNAMIC_DISPATCH(CreateFromQuantizedDatasetImpl)(dataset,
                                                              query_subset);
}
#endif  // HWY_ONCE

}  // namespace ginger
