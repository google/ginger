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

#ifndef GINGER_PIPNN_SIMHASH_H_
#define GINGER_PIPNN_SIMHASH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"

namespace ginger {

class SimHashProjections {
 public:
  SimHashProjections() = default;

  // Build from float data + structured random projections.
  SimHashProjections(absl::Span<const absl::Span<const float>> float_data,
                     size_t dimensionality, int num_projections);

  // Build from TurboQuant base points. Decodes quantized nibbles to
  // approximate float vectors (using the int8 centroid table) and
  // projects those. Norm scaling does not affect projection signs.
  SimHashProjections(const TurboQuantDatasetSoA& points, int num_projections);

  // SimHash sketch of (u-v).
  uint16_t RelativeHash(GingerNodeId u, GingerNodeId v) const;

  int num_projections() const { return num_projections_; }
  size_t num_points() const { return num_points_; }
  absl::Span<const float> projections() const { return projections_; }

 private:
  int num_projections_ = 0;
  size_t num_points_ = 0;
  // TODO: Try if  bf16 or int8 suffices for quality.
  std::vector<float> projections_;
};

}  // namespace ginger

#endif  // GINGER_PIPNN_SIMHASH_H_
