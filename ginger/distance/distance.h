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

#ifndef GINGER_DISTANCE_DISTANCE_H_
#define GINGER_DISTANCE_DISTANCE_H_

#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/types.h"

namespace ginger {

// Computes the dot product distance between `query` and all `neighbors`. The
// distance is defined as the negative of the dot product, so that smaller
// values (more negative) correspond to higher dot product.
void DotProductDistance(absl::Span<const float> query,
                        absl::Span<DistanceAndId> neighbors,
                        const SpanDataset& dataset);

// See above, using dataset[node_id] as query.
inline void DotProductDistance(GingerNodeId node_id,
                               absl::Span<DistanceAndId> neighbors,
                               const SpanDataset& dataset) {
  DotProductDistance(dataset.get<float>(node_id), neighbors, dataset);
}

// Computes the squared L2 distance between `query` and all `neighbors`.
void SquaredL2Distance(absl::Span<const float> query,
                       absl::Span<DistanceAndId> neighbors,
                       const SpanDataset& dataset);

// See above, using dataset[node_id] as query.
inline void SquaredL2Distance(GingerNodeId node_id,
                              absl::Span<DistanceAndId> neighbors,
                              const SpanDataset& dataset) {
  SquaredL2Distance(dataset.get<float>(node_id), neighbors, dataset);
}

}  // namespace ginger

#endif  // GINGER_DISTANCE_DISTANCE_H_
