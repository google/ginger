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

#ifndef GINGER_DISTANCE_DISTANCE_BYTES_H_
#define GINGER_DISTANCE_DISTANCE_BYTES_H_

#include <cstdint>

#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/types.h"

namespace ginger {

// ============================================================================
// int8_t Distance Functions
//
// Computes distance metrics between an int8_t query and points in an int8_t
// dataset. Note that the dot product distance is defined as the negative of the
// dot product (smaller values correspond to higher dot product).
// ============================================================================

// Computes the dot product distance between an int8_t `query` and all
// `neighbors` in an int8_t dataset.
void DotProductDistance(absl::Span<const int8_t> query,
                        absl::Span<DistanceAndId> neighbors,
                        const SpanDataset& dataset);

// Computes the dot product distance between dataset[node_id] and all
// `neighbors` in an int8_t dataset.
inline void DotProductDistanceInt8(GingerNodeId node_id,
                                   absl::Span<DistanceAndId> neighbors,
                                   const SpanDataset& dataset) {
  DotProductDistance(dataset.get<int8_t>(node_id), neighbors, dataset);
}

// Computes the squared L2 distance between an int8_t `query` and all
// `neighbors` in an int8_t dataset.
void SquaredL2Distance(absl::Span<const int8_t> query,
                       absl::Span<DistanceAndId> neighbors,
                       const SpanDataset& dataset);

// Computes the squared L2 distance between dataset[node_id] and all `neighbors`
// in an int8_t dataset.
inline void SquaredL2DistanceInt8(GingerNodeId node_id,
                                  absl::Span<DistanceAndId> neighbors,
                                  const SpanDataset& dataset) {
  SquaredL2Distance(dataset.get<int8_t>(node_id), neighbors, dataset);
}

// ============================================================================
// uint8_t Distance Functions
//
// Computes distance metrics between a uint8_t query and points in a uint8_t
// dataset. Note that the dot product distance is defined as the negative of the
// dot product (smaller values correspond to higher dot product).
// ============================================================================

// Computes the dot product distance between a uint8_t `query` and all
// `neighbors` in a uint8_t dataset.
void DotProductDistance(absl::Span<const uint8_t> query,
                        absl::Span<DistanceAndId> neighbors,
                        const SpanDataset& dataset);

// Computes the dot product distance between dataset[node_id] and all
// `neighbors` in a uint8_t dataset.
inline void DotProductDistanceUint8(GingerNodeId node_id,
                                    absl::Span<DistanceAndId> neighbors,
                                    const SpanDataset& dataset) {
  DotProductDistance(dataset.get<uint8_t>(node_id), neighbors, dataset);
}

// Computes the squared L2 distance between a uint8_t `query` and all
// `neighbors` in a uint8_t dataset.
void SquaredL2Distance(absl::Span<const uint8_t> query,
                       absl::Span<DistanceAndId> neighbors,
                       const SpanDataset& dataset);

// Computes the squared L2 distance between dataset[node_id] and all `neighbors`
// in a uint8_t dataset.
inline void SquaredL2DistanceUint8(GingerNodeId node_id,
                                   absl::Span<DistanceAndId> neighbors,
                                   const SpanDataset& dataset) {
  SquaredL2Distance(dataset.get<uint8_t>(node_id), neighbors, dataset);
}

}  // namespace ginger

#endif  // GINGER_DISTANCE_DISTANCE_BYTES_H_
