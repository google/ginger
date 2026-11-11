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

#include "ginger/search/beam_search.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger::internal {

void MergeBeamAndCandidates(std::vector<BeamEntry>& beam,
                            absl::Span<const DistanceAndId> candidates,
                            size_t max_beam_size, size_t& next_unexplored) {
  if (candidates.empty()) return;
  if (beam.empty()) {
    size_t count = std::min(candidates.size(), max_beam_size);
    beam.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      beam.push_back(BeamEntry(candidates[i]));
    }
    return;
  }

  // Iterate through the union of the two sorted arrays in decreasing order by
  // advancing `candidates_it` and `beam_it` (backwards) as needed. We use
  // `write_it` to write the merged array.
  int32_t beam_it = static_cast<int32_t>(beam.size()) - 1;
  int32_t candidates_it = static_cast<int32_t>(candidates.size()) - 1;

  int32_t write_it = beam_it + candidates_it + 1;
  // We assume that `candidates` and `beam` are disjoint, and so we already know
  // what the new beam size will be.
  int32_t new_beam_size =
      std::min(max_beam_size, beam.size() + candidates.size());
  beam.resize(new_beam_size);

  // Invariant: write_it = beam_it + candidates_it + 1.
  // Once candidates_it == -1, we have beam_it == write_it and we can stop.
  while (candidates_it >= 0) {
    BeamEntry next_entry;

    if (beam_it < 0 ||
        candidates[candidates_it].distance > beam[beam_it].distance) {
      next_entry = candidates[candidates_it--];
    } else {
      next_entry = beam[beam_it--];
    }

    if (write_it < beam.size()) {
      beam[write_it] = next_entry;
    }

    --write_it;
  }

  if (write_it < static_cast<int32_t>(next_unexplored)) {
    next_unexplored = write_it + 1;
  }
}

}  // namespace ginger::internal
