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

// Sizeless-compatible Vec "array" utilities for TurboQuant.
//
// On arm, hn::Vec<D> is a sizeless type: it cannot appear in
// C arrays, as a struct/class member, or be used with pointer arithmetic. This
// header provides macros and helpers that give array-like ergonomics while
// keeping every Vec as a separate named local variable.
//
// Core components:
//   ForIndex<N>(f)                – compile-time unrolled loop [0, N)
//   VEC_GROUP_DEFINE(tag, name)   – defines 8 named Vec variables
//   VEC_GROUP_ACCESSOR(name)      – creates a lambda for indexed access
//
// Include after foreach_target.h and highway.h. The toggle guard handles
// per-target re-inclusion.

#ifndef GINGER_TURBOQUANT_SIMD_VEC_GROUP_INL_H_
#define GINGER_TURBOQUANT_SIMD_VEC_GROUP_INL_H_

#include <cstddef>
#include <type_traits>
#include <utility>

#include "hwy/highway.h"

namespace ginger {

// ForIndex<N>(f): compile-time unrolled loop
//
// Calls f(integral_constant<size_t, 0>{}), f(integral_constant<size_t, 1>{}),
// ..., f(integral_constant<size_t, N-1>{}) in order.
//
// The argument to f is a std::integral_constant, so its value is available at
// compile time via decltype(arg)::value or arg.value. This enables constexpr
// dispatch (e.g., if-constexpr chains in VEC_GROUP_ACCESSOR).
//
// Usage:
//   ForIndex<4>([&](auto q) {
//     acc(q) = hn::Zero(i32);  // q.value is constexpr 0, 1, 2, or 3
//   });

}  // namespace ginger
#endif  // GINGER_TURBOQUANT_SIMD_VEC_GROUP_INL_H_

// clang-format off
#if defined(GINGER_TURBOQUANT_SIMD_VEC_GROUP_TOGGLE) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef GINGER_TURBOQUANT_SIMD_VEC_GROUP_TOGGLE
#undef GINGER_TURBOQUANT_SIMD_VEC_GROUP_TOGGLE
#else
#define GINGER_TURBOQUANT_SIMD_VEC_GROUP_TOGGLE
#endif
// clang-format on

namespace ginger {
namespace HWY_NAMESPACE {

namespace vec_group_internal {
template <size_t... Is, class F>
HWY_INLINE void ForIndexImpl(std::index_sequence<Is...>, F&& f) {
  (f(std::integral_constant<size_t, Is>{}), ...);
}
}  // namespace vec_group_internal

template <size_t N, class F>
HWY_INLINE void ForIndex(F&& f) {
  vec_group_internal::ForIndexImpl(std::make_index_sequence<N>{},
                                   std::forward<F>(f));
}

// VEC_GROUP macros: named Vec variables with indexed access

// Defines 8 named Vec<decltype(tag)> variables: name0, name1, ..., name7.
// All are uninitialized. Only the first Mq will be accessed (the rest are
// unused and may be optimized away).
//
// Usage:
//   VEC_GROUP_DEFINE(i32, acc);  // defines acc0..acc7
#define VEC_GROUP_DEFINE(tag, name)                                            \
  hn::Vec<decltype(tag)> name##0, name##1, name##2, name##3, name##4, name##5, \
      name##6, name##7

// Creates a generic lambda providing compile-time indexed access to the
// 8 named Vec variables defined by VEC_GROUP_DEFINE.
//
// The lambda takes a std::integral_constant<size_t, I> and returns a
// reference to the I-th variable.
//
// Usage:
//   VEC_GROUP_DEFINE(i32, acc);
//   ForIndex<kMq>([&](auto q) {
//     VEC_GROUP_ACCESSOR(acc)(q) = hn::Zero(i32);
//   });
//   MicroKernelAccumulate<kMq>(..., VEC_GROUP_ACCESSOR(acc));
#define VEC_GROUP_ACCESSOR(name)                                        \
  [&](auto _vg_idx) -> auto& {                                          \
    constexpr size_t _vg_i = decltype(_vg_idx)::value;                  \
    static_assert(_vg_i < 8, "VEC_GROUP_ACCESSOR: index out of range"); \
    if constexpr (_vg_i == 0)                                           \
      return name##0;                                                   \
    else if constexpr (_vg_i == 1)                                      \
      return name##1;                                                   \
    else if constexpr (_vg_i == 2)                                      \
      return name##2;                                                   \
    else if constexpr (_vg_i == 3)                                      \
      return name##3;                                                   \
    else if constexpr (_vg_i == 4)                                      \
      return name##4;                                                   \
    else if constexpr (_vg_i == 5)                                      \
      return name##5;                                                   \
    else if constexpr (_vg_i == 6)                                      \
      return name##6;                                                   \
    else if constexpr (_vg_i == 7)                                      \
      return name##7;                                                   \
    else                                                                \
      static_assert(false);                                             \
  }

}  // namespace HWY_NAMESPACE
}  // namespace ginger

// clang-format off
#endif  // GINGER_TURBOQUANT_SIMD_VEC_GROUP_TOGGLE // NOLINT
// clang-format on
