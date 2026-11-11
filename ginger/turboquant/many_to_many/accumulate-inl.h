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

// SIMD micro-kernel for TurboQuant Accumulation.
//
// Include after foreach_target.h, highway.h. The toggle
// guard handles per-target re-inclusion.

#ifndef GINGER_TURBOQUANT_MANY_TO_MANY_ACCUMULATE_INL_H_
#define GINGER_TURBOQUANT_MANY_TO_MANY_ACCUMULATE_INL_H_
#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_ACCUMULATE_INL_H_

// clang-format off
#if defined(GINGER_TURBOQUANT_MANY_TO_MANY_ACCUMULATE_TOGGLE) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef GINGER_TURBOQUANT_MANY_TO_MANY_ACCUMULATE_TOGGLE
#undef GINGER_TURBOQUANT_MANY_TO_MANY_ACCUMULATE_TOGGLE
#else
#define GINGER_TURBOQUANT_MANY_TO_MANY_ACCUMULATE_TOGGLE
#endif
// clang-format on

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "ginger/turboquant/many_to_many/vec_group-inl.h"
#include "hwy/highway.h"

namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Unified micro-kernel (Inner-loop GEMM Accumulation)
//
// Computes dot products between 'Mq' queries and an arbitrary number of
// database panels.
//
// This function performs the core multiply-accumulate operations in the
// ManyToManyTurboQuant pipeline. It leverages Highway's SumOfMulQuadAccumulate
// to achieve high throughput by broadcasting query dimensions over database
// points.
//
// Layout Invariants:
// - Queries: Pre-decoded into row-major int8_t format. Each query contains
//   'total_tiles' groups of 4 bytes (matching the granularity of
//   SumOfMulQuadAccumulate).
// - Database: Packed into a block-transposed layout where each vector-sized
//   tile contains data for multiple points.
//
// Operation for each tile 't' (dimension group):
// 1. Load a vector-sized tile of database data (uint8_t) for each panel.
// 2. Load 4 bytes of query data for the current dimension group.
// 3. Broadcast these 4 query bytes to all lanes of a vector.
// 4. Use SumOfMulQuadAccumulate to multiply the 4-byte query pattern with every
//    4-byte group in each panel's tile.
//
// The number of panels is deduced from the number of accumulator accessor
// arguments. Each accessor is a lambda created by VEC_GROUP_ACCESSOR(name) that
// provides compile-time indexed access to named Vec variables — this avoids
// illegal arrays of sizeless Vec types on SVE/RISC-V.
//
// Parameters:
//   Mq          – number of queries (compile-time)
//   query_ptrs  – array of Mq pointers to decoded query bytes
//   blocks      – array of NumPanels pointers to panel data
//   total_tiles – number of 4-byte tile iterations
//   N           – number of SIMD lanes (bytes per tile)
//   accs...     – one VEC_GROUP_ACCESSOR per panel (deduces NumPanels)

template <size_t Mq, class... Accumulators>
HWY_INLINE void MicroKernelAccumulate(
    const hn::ScalableTag<uint8_t> u8, const hn::ScalableTag<int8_t> i8,
    const hn::ScalableTag<int32_t> i32,
    const int8_t* HWY_RESTRICT const* query_ptrs,
    const uint8_t* HWY_RESTRICT const* blocks, size_t total_tiles, size_t N,
    Accumulators... accs) {
  constexpr size_t kNumPanels = sizeof...(Accumulators);
  auto acc_fns = std::make_tuple(accs...);

  // Zero all accumulators for all panels.
  ForIndex<Mq>([&](auto q) {
    ForIndex<kNumPanels>(
        [&](auto p) { std::get<p.value>(acc_fns)(q) = hn::Zero(i32); });
  });

  for (size_t t = 0; t < total_tiles; ++t) {
    ForIndex<Mq>([&](auto q) {
      int32_t q32 = 0;
      hwy::CopyBytes(query_ptrs[q.value] + t * sizeof(int32_t), &q32,
                     sizeof(int32_t));
      const auto qv = hn::BitCast(i8, hn::Set(i32, q32));
      ForIndex<kNumPanels>([&](auto p) {
        // The block Load depends only on (t, p), not on q. After full
        // compile-time unrolling of ForIndex<Mq>, the compiler will eliminate
        // identical loads across queries.
        const auto b = hn::Load(u8, blocks[p.value] + t * N);
        std::get<p.value>(acc_fns)(q) = hn::SumOfMulQuadAccumulate(
            i32, b, qv, std::get<p.value>(acc_fns)(q));
      });
    });
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace ginger

// clang-format off
#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_ACCUMULATE_TOGGLE // NOLINT
// clang-format on
