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

#ifndef GINGER_DISTANCE_HAMMING_H_
#define GINGER_DISTANCE_HAMMING_H_

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {

// Computes the bitwise Hamming distance (the total number of differing bits)
// between a bit-packed `query` and each candidate point in `candidates` from
// `dataset`. Each dataset point is stored as a contiguous sequence of
// `words_per_point` packed 64-bit words (indexed at `dataset + candidate.id *
// words_per_point`). The resulting bit count is written into each candidate's
// `distance` field as a float.
void HammingDistance(absl::Span<const uint64_t> query,
                     absl::Span<DistanceAndId> candidates,
                     const uint64_t* dataset, size_t words_per_point);

}  // namespace ginger

#endif  // GINGER_DISTANCE_HAMMING_H_
