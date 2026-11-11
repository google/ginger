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

#ifndef GINGER_INTERNAL_PARALLEL_FOR_WITH_STATUS_H_
#define GINGER_INTERNAL_PARALLEL_FOR_WITH_STATUS_H_

#include <atomic>
#include <cstddef>
#include <type_traits>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "parlay/parallel.h"

namespace ginger {

// Similar to `parlay::parallel_for`, with the following differences:
// - `f` is expected to return an `absl::Status` instead of `void`.
// - `parallel_for_with_status` returns an `absl::Status` instead of `void`. The
//   returned status is one of the error statuses produced by the calls to `f`,
//   if any, or an Ok status otherwise.
template <typename F>
absl::Status parallel_for_with_status(
    size_t start, size_t end, F&& f,
    // We use `long` for consistency with `parlay::parallel_for`.
    // NOLINTBEGIN(runtime/int)
    // NOLINTBEGIN(google-runtime-int)
    long granularity = 0,
    // NOLINTEND(google-runtime-int)
    // NOLINTEND(runtime/int)
    bool conservative = false) {
  static_assert(std::is_invocable_v<F&, size_t>);
  absl::Status overall_status;
  std::atomic<bool> error_found = false;
  parlay::parallel_for(
      start, end,
      [&](size_t i) {
        if (absl::Status status = f(i); !status.ok()) [[unlikely]] {
          bool expected = false;
          if (error_found.compare_exchange_strong(expected, true)) {
            overall_status = status;
          }
        }
      },
      granularity, conservative);
  return overall_status;
}

template <typename... Tasks>
absl::Status par_do_with_status(Tasks&&... tasks) {
  const absl::FunctionRef<absl::Status()> fns[] = {tasks...};
  return parallel_for_with_status(
      0, sizeof...(Tasks), [&](size_t i) { return fns[i](); },
      /*granularity=*/1);
}

}  // namespace ginger

#endif  // GINGER_INTERNAL_PARALLEL_FOR_WITH_STATUS_H_
