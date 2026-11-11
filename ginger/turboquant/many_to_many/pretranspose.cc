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

#include "ginger/turboquant/many_to_many/pretranspose.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE

// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/many_to_many/pretranspose.cc"
// clang-format on

#include "hwy/cache_control.h"   // IWYU pragma: keep
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/decode_db-inl.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/vec_group-inl.h"

namespace ginger {

namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

PreTransposedTurboQuantDataset BuildPreTransposed(
    const TurboQuantDatasetSoA& base_points, IndexSubset subset,
    const TurboQuantCodec& /*quantizer*/) {
  using Uint8 = hn::ScalableTag<uint8_t>;
  const Uint8 u8;
  const size_t N = hn::Lanes(u8);

  PreTransposedTurboQuantDataset result;
  result.num_points = subset.size();
  result.dimensionality = base_points.dimensionality();
  result.bytes_per_point = base_points.num_bytes_per_datapoint();
  result.points_per_panel = N / 4;

  const size_t num_dim_chunks = hwy::DivCeil(result.bytes_per_point, N);
  const size_t tiles_per_chunk = N / 2;
  result.total_tiles = num_dim_chunks * tiles_per_chunk;
  result.panel_bytes = result.total_tiles * N;
  result.num_panels = hwy::DivCeil(result.num_points, result.points_per_panel);

  result.panel_data.resize(result.num_panels * result.panel_bytes);
  for (size_t p = 0; p < result.num_panels; ++p) {
    const size_t start = p * result.points_per_panel;
    const size_t size =
        std::min(result.points_per_panel, result.num_points - start);
    DecodeDbPanel(
        base_points, subset.subspan(start, size),
        absl::Span<uint8_t>(result.panel_data.data() + p * result.panel_bytes,
                            result.panel_bytes));
  }

  const size_t padded_count = result.num_panels * result.points_per_panel;
  result.norm_scaling_factors.assign(padded_count,
                                     std::numeric_limits<float>::infinity());
  for (size_t i = 0; i < result.num_points; ++i) {
    result.norm_scaling_factors[i] = base_points.norm_scaling_factor(subset[i]);
  }

  // Populate squared norms for L2 distance metric.
  if (base_points.metric() == TurboQuantDistanceMetric::kSquaredL2Distance) {
    result.sqr_norms.assign(padded_count, std::numeric_limits<float>::max());
    for (size_t i = 0; i < result.num_points; ++i) {
      result.sqr_norms[i] = base_points.unquantized_squared_norm(subset[i]);
    }
  }

  return result;
}

}  // namespace
}  // namespace HWY_NAMESPACE

}  // namespace ginger

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace ginger {

HWY_EXPORT(BuildPreTransposed);

PreTransposedTurboQuantDataset PreTransposedTurboQuantDataset::Build(
    const TurboQuantDatasetSoA& base_points, absl::Span<const int> indices,
    const TurboQuantCodec& quantizer) {
  return HWY_DYNAMIC_DISPATCH(BuildPreTransposed)(
      base_points, IndexSubset::Gathered(indices), quantizer);
}

PreTransposedTurboQuantDataset PreTransposedTurboQuantDataset::Build(
    const TurboQuantDatasetSoA& base_points, const TurboQuantCodec& quantizer) {
  return HWY_DYNAMIC_DISPATCH(BuildPreTransposed)(
      base_points, IndexSubset::Contiguous(0, base_points.num_datapoints()),
      quantizer);
}

}  // namespace ginger

#endif  // HWY_ONCE
