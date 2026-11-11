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

#include "ginger/pipnn/simhash.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/pipnn/structured_random_matrix.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"
#include "parlay/parallel.h"

namespace ginger {

SimHashProjections::SimHashProjections(
    absl::Span<const absl::Span<const float>> float_data, size_t dimensionality,
    int num_projections)
    : num_projections_(num_projections),
      num_points_(float_data.size()),
      projections_(num_points_ * num_projections_, 0.0f) {
  ABSL_CHECK_LE(num_projections_, 16);
  StructuredRandomProjections projector(num_projections_, dimensionality);
  parlay::parallel_for(0, float_data.size(), [&](size_t i) {
    Eigen::VectorXf projected =
        projector.Project(Eigen::Map<const Eigen::VectorXf>(
            float_data[i].data(), dimensionality));
    std::copy_n(projected.data(), num_projections_,
                projections_.data() + i * num_projections_);
  });
}

SimHashProjections::SimHashProjections(const TurboQuantDatasetSoA& points,
                                       int num_projections)
    : num_projections_(num_projections),
      num_points_(points.num_datapoints()),
      projections_(num_points_ * num_projections_, 0.0f) {
  ABSL_CHECK_LE(num_projections_, 16);
  const size_t dim = points.dimensionality();
  StructuredRandomProjections projector(num_projections_, dim);

  // Decode TQ nibbles to approximate float vectors and project them.
  parlay::parallel_for(0, num_points_, [&](size_t i) {
    Eigen::VectorXf decoded(dim);
    points.DequantizeToRotatedSpace(i, absl::MakeSpan(decoded.data(), dim));
    Eigen::VectorXf projected = projector.Project(decoded);
    std::copy_n(projected.data(), num_projections_,
                projections_.data() + i * num_projections_);
  });
}

uint16_t SimHashProjections::RelativeHash(GingerNodeId u,
                                          GingerNodeId v) const {
  uint16_t hash = 0;
  const float* hu = &projections_[static_cast<size_t>(u) * num_projections_];
  const float* hv = &projections_[static_cast<size_t>(v) * num_projections_];
  for (int j = 0; j < num_projections_; ++j) {
    hash = (hash << 1) | (hu[j] > hv[j]);
  }
  return hash;
}

}  // namespace ginger
