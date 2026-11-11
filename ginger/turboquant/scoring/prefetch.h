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

#ifndef GINGER_TURBOQUANT_SCORING_PREFETCH_H_
#define GINGER_TURBOQUANT_SCORING_PREFETCH_H_

#include <cstddef>
#include <cstdint>

#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"

namespace ginger {

// Helper function to iterate over cachelines of a memory range and invoke a
// prefetch functor.
template <typename F>
inline void PrefetchByCachelines(const uint8_t* ptr, const size_t size,
                                 F&& prefetch_fn) {
  if (size == 0) return;
  const uint8_t* last_byte = ptr + size - 1;
  const uintptr_t first_line =
      reinterpret_cast<uintptr_t>(ptr) &
      ~(static_cast<uintptr_t>(ABSL_CACHELINE_SIZE) - 1);
  for (uintptr_t line = first_line;
       line <= reinterpret_cast<uintptr_t>(last_byte);
       line += ABSL_CACHELINE_SIZE) {
    prefetch_fn(reinterpret_cast<const void*>(line));
  }
}

// Prefetches the first `num_bytes_to_prefetch` bytes of the datapoint at
// `quantized_datapoint_ptr`.
inline void PrefetchDatapoint(const uint8_t* quantized_datapoint_ptr,
                              const size_t num_bytes_to_prefetch) {
  PrefetchByCachelines(quantized_datapoint_ptr, num_bytes_to_prefetch,
                       [](const void* p) { absl::PrefetchToLocalCacheNta(p); });
}

// Same as above but allows specifying the prefetch hint.
template <int Hint>
inline void PrefetchDatapointHint(const uint8_t* quantized_datapoint_ptr,
                                  const size_t num_bytes_to_prefetch) {
  PrefetchByCachelines(quantized_datapoint_ptr, num_bytes_to_prefetch,
                       [](const void* p) {
#if defined(__GNUC__) || defined(__clang__)
                         __builtin_prefetch(p, 0, Hint);
#endif
                       });
}

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_SCORING_PREFETCH_H_
