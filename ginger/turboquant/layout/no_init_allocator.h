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

#ifndef GINGER_TURBOQUANT_LAYOUT_NO_INIT_ALLOCATOR_H_
#define GINGER_TURBOQUANT_LAYOUT_NO_INIT_ALLOCATOR_H_

#include <new>
#include <type_traits>
#include <utility>

#include "hwy/aligned_allocator.h"

namespace ginger {

// Aligned allocator that performs no initialization (skips zero-filling) for
// trivially default constructible types on default construction. This avoids
// expensive sequential zero-filling of multi-GB quantization buffers during
// allocation / resizing.
template <class T>
struct NoInitAlignedAllocator : public hwy::AlignedAllocator<T> {
  using value_type = T;

  NoInitAlignedAllocator() = default;

  template <class V>
  explicit NoInitAlignedAllocator(const NoInitAlignedAllocator<V>&) noexcept {}

  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    if constexpr (sizeof...(args) == 0 &&
                  std::is_trivially_default_constructible_v<U>) {
      // Skip value-initialization (no zero-fill for trivially default
      // constructible types).
      return;
    } else {
      ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
  }
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_LAYOUT_NO_INIT_ALLOCATOR_H_
