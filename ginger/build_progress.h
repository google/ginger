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

#ifndef GINGER_BUILD_PROGRESS_H_
#define GINGER_BUILD_PROGRESS_H_

#include <cstddef>

#include "absl/time/time.h"
#include "ginger/types.h"

namespace ginger {

// Progress of an index build, reported after each batch of inserted points.
struct IndexBuildProgress {
  // Number of points in the index.
  GingerNodeId num_points = 0;
  // Number of insertion passes over all points.
  int total_passes = 0;
  // Zero-based index of the pass that is running.
  int current_pass = 0;
  // Points inserted so far, summed over all passes. The build is done when
  // this reaches `num_points * total_passes`.
  GingerNodeId total_completed_points = 0;
  // Wall time since the first pass started.
  absl::Duration elapsed_time = absl::ZeroDuration();
  // Wall time and number of points of the batch that just finished.
  absl::Duration last_batch_duration = absl::ZeroDuration();
  size_t last_batch_size = 0;
};

}  // namespace ginger

#endif  // GINGER_BUILD_PROGRESS_H_
