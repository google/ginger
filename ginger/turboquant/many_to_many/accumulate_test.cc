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

// Unit tests for accumulate-inl.h: SIMD micro-kernel dot-product accumulation.
//
// Each SIMD micro-kernel is validated against a scalar reference that computes
// SumOfMulQuad: for each group of 4 bytes, multiply uint8 * int8 and sum.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ginger/internal/gunit_support.h"
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/many_to_many/accumulate_test.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

#ifndef ACCUMULATE_TEST_SCALAR_REF_DEFINED_
#define ACCUMULATE_TEST_SCALAR_REF_DEFINED_
namespace scalar_ref {

// Scalar SumOfMulQuad: for each group of 4 bytes at position [4*i..4*i+3],
// compute sum(uint8[j] * int8[j]) and add to accumulator.
// This matches Highway's SumOfMulQuadAccumulate semantics.
inline int32_t ScalarSumOfMulQuad(const uint8_t* u8_data, const int8_t* i8_data,
                                  size_t N) {
  int32_t acc = 0;
  for (size_t i = 0; i < N; i += 4) {
    int32_t quad = 0;
    for (size_t j = 0; j < 4 && (i + j) < N; ++j) {
      quad += static_cast<int32_t>(u8_data[i + j]) *
              static_cast<int32_t>(i8_data[i + j]);
    }
    acc += quad;
  }
  return acc;
}

// Scalar reference for a single panel.
// query_ptrs[q] points to int8 data reinterpreted as int32 tiles.
// block[t*N .. t*N+N-1] is uint8 data for tile t.
// For each tile t, the query broadcasts query_ptrs[q][t] (an int32 = 4 int8s)
// to all N/4 groups, and dot-products with block data.
inline void ScalarAccumulate1Panel(size_t Mq, const int8_t* const* query_ptrs,
                                   const uint8_t* block, size_t total_tiles,
                                   size_t N, absl::Span<int32_t> acc) {
  absl::c_fill(acc, 0);
  for (size_t t = 0; t < total_tiles; ++t) {
    for (size_t q = 0; q < Mq; ++q) {
      // The SIMD kernel broadcasts query_ptrs[q][t] (as int32) across the
      // vector, meaning 4 bytes are replicated N/4 times. Then it dot-products
      // with the block.
      int8_t qbytes[4];
      std::memcpy(qbytes, query_ptrs[q] + t * 4, 4);

      // Build a full N-byte query vector by replicating the 4 bytes.
      std::vector<int8_t> q_expanded(N);
      for (size_t i = 0; i < N; ++i) {
        q_expanded[i] = qbytes[i % 4];
      }

      acc[q] += ScalarSumOfMulQuad(block + t * N, q_expanded.data(), N);
    }
  }
}

// Per-lane scalar reference: returns one result per i32 lane (N/4 values).
// Each lane k accumulates independently over tiles, matching the SIMD kernel's
// per-lane semantics. Use to verify individual lanes rather than their sum.
inline void ScalarAccumulate1PanelPerLane(const int8_t* query_ptr,
                                          const uint8_t* block,
                                          size_t total_tiles, size_t N,
                                          absl::Span<int32_t> per_lane_acc) {
  const size_t NI32 = N / 4;
  absl::c_fill(per_lane_acc, 0);
  for (size_t t = 0; t < total_tiles; ++t) {
    int8_t qbytes[4];
    std::memcpy(qbytes, query_ptr + t * 4, 4);
    for (size_t k = 0; k < NI32; ++k) {
      for (size_t j = 0; j < 4; ++j) {
        per_lane_acc[k] += static_cast<int32_t>(block[t * N + k * 4 + j]) *
                           static_cast<int32_t>(qbytes[j]);
      }
    }
  }
}

}  // namespace scalar_ref
#endif  // ACCUMULATE_TEST_SCALAR_REF_DEFINED_

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/accumulate-inl.h"  // NOLINT

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

template <class V>
int32_t SumAllLanes(hn::ScalableTag<int32_t> i32, V v) {
  const size_t NI32 = hn::Lanes(i32);
  hwy::AlignedVector<int32_t> buf(NI32, 0);
  hn::StoreU(v, i32, buf.data());
  int32_t s = 0;
  for (size_t i = 0; i < NI32; ++i) s += buf[i];
  return s;
}

HWY_NOINLINE void TestAccumulate1PanelMq1() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);

  for (size_t total_tiles : {1, 2, 4, 8, 16}) {
    hwy::AlignedVector<uint8_t> block(total_tiles * N, 0);
    for (size_t i = 0; i < block.size(); ++i) {
      block[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
    }

    hwy::AlignedVector<int8_t> query(total_tiles * 4, 0);
    for (size_t i = 0; i < query.size(); ++i) {
      query[i] = static_cast<int8_t>(((i * 11 + 5) % 256) - 128);
    }

    const int8_t* query_ptrs[1] = {query.data()};

    VEC_GROUP_DEFINE(i32, acc);
    const uint8_t* blks[] = {block.data()};
    ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
        u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc));

    std::vector<int32_t> scalar_acc(1, 0);
    scalar_ref::ScalarAccumulate1Panel(1, query_ptrs, block.data(), total_tiles,
                                       N, absl::MakeSpan(scalar_acc));

    EXPECT_EQ(SumAllLanes(i32, acc0), scalar_acc[0])
        << "total_tiles=" << total_tiles;
  }
}

HWY_NOINLINE void TestAccumulate1PanelMq4() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);

  const size_t total_tiles = 8;

  hwy::AlignedVector<uint8_t> block(total_tiles * N, 0);
  for (size_t i = 0; i < block.size(); ++i) {
    block[i] = static_cast<uint8_t>((i * 3 + 1) % 256);
  }

  hwy::AlignedVector<int8_t> queries[4];
  const int8_t* query_ptrs[4];
  for (size_t q = 0; q < 4; ++q) {
    queries[q].resize(total_tiles * 4);
    for (size_t i = 0; i < queries[q].size(); ++i) {
      queries[q][i] = static_cast<int8_t>(((i * (q + 3) + q * 17) % 256) - 128);
    }
    query_ptrs[q] = queries[q].data();
  }

  VEC_GROUP_DEFINE(i32, acc);
  const uint8_t* blks[] = {block.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<4>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc));

  std::vector<int32_t> scalar_acc(4, 0);
  scalar_ref::ScalarAccumulate1Panel(4, query_ptrs, block.data(), total_tiles,
                                     N, absl::MakeSpan(scalar_acc));

  EXPECT_EQ(SumAllLanes(i32, acc0), scalar_acc[0]) << "q=0";
  EXPECT_EQ(SumAllLanes(i32, acc1), scalar_acc[1]) << "q=1";
  EXPECT_EQ(SumAllLanes(i32, acc2), scalar_acc[2]) << "q=2";
  EXPECT_EQ(SumAllLanes(i32, acc3), scalar_acc[3]) << "q=3";
}

HWY_NOINLINE void TestAccumulate2Panels() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);

  const size_t total_tiles = 8;
  constexpr size_t Mq = 2;

  hwy::AlignedVector<uint8_t> block0(total_tiles * N, 0);
  hwy::AlignedVector<uint8_t> block1(total_tiles * N, 0);
  for (size_t i = 0; i < block0.size(); ++i) {
    block0[i] = static_cast<uint8_t>((i * 5 + 7) % 256);
    block1[i] = static_cast<uint8_t>((i * 11 + 13) % 256);
  }

  hwy::AlignedVector<int8_t> queries[Mq];
  const int8_t* query_ptrs[Mq];
  for (size_t q = 0; q < Mq; ++q) {
    queries[q].resize(total_tiles * 4);
    for (size_t i = 0; i < queries[q].size(); ++i) {
      queries[q][i] = static_cast<int8_t>(((i * (q + 7) + q * 23) % 256) - 128);
    }
    query_ptrs[q] = queries[q].data();
  }

  VEC_GROUP_DEFINE(i32, a0);
  VEC_GROUP_DEFINE(i32, a1);
  const uint8_t* blks[] = {block0.data(), block1.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<Mq>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(a0),
      VEC_GROUP_ACCESSOR(a1));

  std::vector<int32_t> scalar_acc0(Mq, 0), scalar_acc1(Mq, 0);
  scalar_ref::ScalarAccumulate1Panel(Mq, query_ptrs, block0.data(), total_tiles,
                                     N, absl::MakeSpan(scalar_acc0));
  scalar_ref::ScalarAccumulate1Panel(Mq, query_ptrs, block1.data(), total_tiles,
                                     N, absl::MakeSpan(scalar_acc1));

  EXPECT_EQ(SumAllLanes(i32, a00), scalar_acc0[0]) << "panel0 q=0";
  EXPECT_EQ(SumAllLanes(i32, a01), scalar_acc0[1]) << "panel0 q=1";
  EXPECT_EQ(SumAllLanes(i32, a10), scalar_acc1[0]) << "panel1 q=0";
  EXPECT_EQ(SumAllLanes(i32, a11), scalar_acc1[1]) << "panel1 q=1";
}

HWY_NOINLINE void TestAccumulate4Panels() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);

  const size_t total_tiles = 4;
  constexpr size_t Mq = 2;

  hwy::AlignedVector<uint8_t> blocks[4];
  for (int p = 0; p < 4; ++p) {
    blocks[p].resize(total_tiles * N);
    for (size_t i = 0; i < blocks[p].size(); ++i) {
      blocks[p][i] = static_cast<uint8_t>((i * (p + 3) + p * 19) % 256);
    }
  }

  hwy::AlignedVector<int8_t> queries[Mq];
  const int8_t* query_ptrs[Mq];
  for (size_t q = 0; q < Mq; ++q) {
    queries[q].resize(total_tiles * 4);
    for (size_t i = 0; i < queries[q].size(); ++i) {
      queries[q][i] = static_cast<int8_t>(((i * (q + 5) + q * 31) % 256) - 128);
    }
    query_ptrs[q] = queries[q].data();
  }

  VEC_GROUP_DEFINE(i32, p0);
  VEC_GROUP_DEFINE(i32, p1);
  VEC_GROUP_DEFINE(i32, p2);
  VEC_GROUP_DEFINE(i32, p3);
  const uint8_t* blks[] = {blocks[0].data(), blocks[1].data(), blocks[2].data(),
                           blocks[3].data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<Mq>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(p0),
      VEC_GROUP_ACCESSOR(p1), VEC_GROUP_ACCESSOR(p2), VEC_GROUP_ACCESSOR(p3));

  std::vector<int32_t> s0(Mq), s1(Mq), s2(Mq), s3(Mq);
  scalar_ref::ScalarAccumulate1Panel(Mq, query_ptrs, blocks[0].data(),
                                     total_tiles, N, absl::MakeSpan(s0));
  scalar_ref::ScalarAccumulate1Panel(Mq, query_ptrs, blocks[1].data(),
                                     total_tiles, N, absl::MakeSpan(s1));
  scalar_ref::ScalarAccumulate1Panel(Mq, query_ptrs, blocks[2].data(),
                                     total_tiles, N, absl::MakeSpan(s2));
  scalar_ref::ScalarAccumulate1Panel(Mq, query_ptrs, blocks[3].data(),
                                     total_tiles, N, absl::MakeSpan(s3));

  EXPECT_EQ(SumAllLanes(i32, p00), s0[0]) << "p0 q=0";
  EXPECT_EQ(SumAllLanes(i32, p01), s0[1]) << "p0 q=1";
  EXPECT_EQ(SumAllLanes(i32, p10), s1[0]) << "p1 q=0";
  EXPECT_EQ(SumAllLanes(i32, p11), s1[1]) << "p1 q=1";
  EXPECT_EQ(SumAllLanes(i32, p20), s2[0]) << "p2 q=0";
  EXPECT_EQ(SumAllLanes(i32, p21), s2[1]) << "p2 q=1";
  EXPECT_EQ(SumAllLanes(i32, p30), s3[0]) << "p3 q=0";
  EXPECT_EQ(SumAllLanes(i32, p31), s3[1]) << "p3 q=1";
}

// Test with zero inputs (identity check).
HWY_NOINLINE void TestAccumulateZeroInput() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  const size_t total_tiles = 4;
  hwy::AlignedVector<uint8_t> block(total_tiles * N, 0);
  hwy::AlignedVector<int8_t> query(total_tiles * 4, 0);
  const int8_t* query_ptrs[1] = {query.data()};

  VEC_GROUP_DEFINE(i32, acc);
  const uint8_t* blks[] = {block.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc));

  hwy::AlignedVector<int32_t> result(NI32, -1);
  hn::StoreU(acc0, i32, result.data());
  for (size_t i = 0; i < NI32; ++i) {
    EXPECT_EQ(result[i], 0) << "lane=" << i;
  }
}

// Test with known simple values where result is manually calculable.
HWY_NOINLINE void TestAccumulate1PanelKnownValues() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  // 1 tile, all bytes = 1. Each quad: 1*1 * 4 = 4 per i32 lane.
  const size_t total_tiles = 1;
  hwy::AlignedVector<uint8_t> block(N, 1);
  hwy::AlignedVector<int8_t> query(4, 1);  // 1 tile = 4 bytes of query
  const int8_t* query_ptrs[1] = {query.data()};

  VEC_GROUP_DEFINE(i32, acc);
  const uint8_t* blks[] = {block.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc));

  hwy::AlignedVector<int32_t> result(NI32, 0);
  hn::StoreU(acc0, i32, result.data());

  // Each i32 lane gets one quad: 1*1 + 1*1 + 1*1 + 1*1 = 4.
  for (size_t i = 0; i < NI32; ++i) {
    EXPECT_EQ(result[i], 4) << "lane=" << i;
  }
}

// Verify extreme int8/uint8 values: uint8=255 × int8=-128.
// Catches sign-extension bugs in SumOfMulQuadAccumulate emulation paths.
HWY_NOINLINE void TestAccumulateSignBoundary() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  const size_t total_tiles = 2;
  hwy::AlignedVector<uint8_t> block(total_tiles * N, 255);
  hwy::AlignedVector<int8_t> query(total_tiles * 4, -128);
  const int8_t* query_ptrs[1] = {query.data()};

  VEC_GROUP_DEFINE(i32, acc);
  const uint8_t* blks[] = {block.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc));

  hwy::AlignedVector<int32_t> result(NI32, 0);
  hn::StoreU(acc0, i32, result.data());

  // Each quad per tile: 255 * (-128) * 4 = -130560.
  const int32_t expected = -130560 * static_cast<int32_t>(total_tiles);
  for (size_t i = 0; i < NI32; ++i) {
    EXPECT_EQ(result[i], expected) << "lane=" << i;
  }
}

// Single tile with Mq=4 and 2 panels: exercises the edge where the tile loop
// body runs exactly once (no CSE opportunity across tile iterations).
HWY_NOINLINE void TestAccumulate1TileMq4TwoPanels() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);

  const size_t total_tiles = 1;
  constexpr size_t Mq = 4;

  hwy::AlignedVector<uint8_t> block0(total_tiles * N, 0);
  hwy::AlignedVector<uint8_t> block1(total_tiles * N, 0);
  for (size_t i = 0; i < block0.size(); ++i) {
    block0[i] = static_cast<uint8_t>((i * 13 + 7) % 256);
    block1[i] = static_cast<uint8_t>((i * 17 + 11) % 256);
  }

  hwy::AlignedVector<int8_t> queries[Mq];
  const int8_t* query_ptrs[Mq];
  for (size_t q = 0; q < Mq; ++q) {
    queries[q].resize(total_tiles * 4);
    for (size_t i = 0; i < queries[q].size(); ++i) {
      queries[q][i] = static_cast<int8_t>(((i * (q + 3) + q * 29) % 256) - 128);
    }
    query_ptrs[q] = queries[q].data();
  }

  VEC_GROUP_DEFINE(i32, a0);
  VEC_GROUP_DEFINE(i32, a1);
  const uint8_t* blks[] = {block0.data(), block1.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<Mq>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(a0),
      VEC_GROUP_ACCESSOR(a1));

  std::vector<int32_t> s0(Mq, 0), s1(Mq, 0);
  scalar_ref::ScalarAccumulate1Panel(Mq, query_ptrs, block0.data(), total_tiles,
                                     N, absl::MakeSpan(s0));
  scalar_ref::ScalarAccumulate1Panel(Mq, query_ptrs, block1.data(), total_tiles,
                                     N, absl::MakeSpan(s1));

  EXPECT_EQ(SumAllLanes(i32, a00), s0[0]) << "panel0 q=0";
  EXPECT_EQ(SumAllLanes(i32, a01), s0[1]) << "panel0 q=1";
  EXPECT_EQ(SumAllLanes(i32, a02), s0[2]) << "panel0 q=2";
  EXPECT_EQ(SumAllLanes(i32, a03), s0[3]) << "panel0 q=3";
  EXPECT_EQ(SumAllLanes(i32, a10), s1[0]) << "panel1 q=0";
  EXPECT_EQ(SumAllLanes(i32, a11), s1[1]) << "panel1 q=1";
  EXPECT_EQ(SumAllLanes(i32, a12), s1[2]) << "panel1 q=2";
  EXPECT_EQ(SumAllLanes(i32, a13), s1[3]) << "panel1 q=3";
}

// Large tile counts (32, 64) — exercises behavior beyond the existing max of 16
// and stresses accumulator magnitude.
HWY_NOINLINE void TestAccumulateLargeTileCount() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);

  for (size_t total_tiles : {32, 64}) {
    hwy::AlignedVector<uint8_t> block(total_tiles * N, 0);
    for (size_t i = 0; i < block.size(); ++i) {
      block[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
    }

    hwy::AlignedVector<int8_t> query(total_tiles * 4, 0);
    for (size_t i = 0; i < query.size(); ++i) {
      query[i] = static_cast<int8_t>(((i * 11 + 5) % 256) - 128);
    }

    const int8_t* query_ptrs[1] = {query.data()};

    VEC_GROUP_DEFINE(i32, acc);
    const uint8_t* blks[] = {block.data()};
    ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
        u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc));

    std::vector<int32_t> scalar_acc(1, 0);
    scalar_ref::ScalarAccumulate1Panel(1, query_ptrs, block.data(), total_tiles,
                                       N, absl::MakeSpan(scalar_acc));

    EXPECT_EQ(SumAllLanes(i32, acc0), scalar_acc[0])
        << "total_tiles=" << total_tiles;
  }
}

// Mq=8: maximum VEC_GROUP size. Exercises all 8 accessor branches.
HWY_NOINLINE void TestAccumulateMq8() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);

  const size_t total_tiles = 4;
  constexpr size_t Mq = 8;

  hwy::AlignedVector<uint8_t> block(total_tiles * N, 0);
  for (size_t i = 0; i < block.size(); ++i) {
    block[i] = static_cast<uint8_t>((i * 13 + 7) % 256);
  }

  hwy::AlignedVector<int8_t> queries[Mq];
  const int8_t* query_ptrs[Mq];
  for (size_t q = 0; q < Mq; ++q) {
    queries[q].resize(total_tiles * 4);
    for (size_t i = 0; i < queries[q].size(); ++i) {
      queries[q][i] = static_cast<int8_t>(((i * (q + 2) + q * 37) % 256) - 128);
    }
    query_ptrs[q] = queries[q].data();
  }

  VEC_GROUP_DEFINE(i32, acc);
  const uint8_t* blks[] = {block.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<Mq>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc));

  std::vector<int32_t> scalar_acc(Mq, 0);
  scalar_ref::ScalarAccumulate1Panel(Mq, query_ptrs, block.data(), total_tiles,
                                     N, absl::MakeSpan(scalar_acc));

  EXPECT_EQ(SumAllLanes(i32, acc0), scalar_acc[0]) << "q=0";
  EXPECT_EQ(SumAllLanes(i32, acc1), scalar_acc[1]) << "q=1";
  EXPECT_EQ(SumAllLanes(i32, acc2), scalar_acc[2]) << "q=2";
  EXPECT_EQ(SumAllLanes(i32, acc3), scalar_acc[3]) << "q=3";
  EXPECT_EQ(SumAllLanes(i32, acc4), scalar_acc[4]) << "q=4";
  EXPECT_EQ(SumAllLanes(i32, acc5), scalar_acc[5]) << "q=5";
  EXPECT_EQ(SumAllLanes(i32, acc6), scalar_acc[6]) << "q=6";
  EXPECT_EQ(SumAllLanes(i32, acc7), scalar_acc[7]) << "q=7";
}

// Near-saturation: accumulator values approach INT32_MAX without overflow.
// uint8=255, int8=127 → each quad = 129540/tile, 16000 tiles → 2,072,640,000
// (96.5% of INT32_MAX).
HWY_NOINLINE void TestAccumulateNearSaturation() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  const size_t total_tiles = 16000;
  hwy::AlignedVector<uint8_t> block(total_tiles * N, 255);
  hwy::AlignedVector<int8_t> query(total_tiles * 4, 127);
  const int8_t* query_ptrs[1] = {query.data()};

  VEC_GROUP_DEFINE(i32, acc);
  const uint8_t* blks[] = {block.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc));

  hwy::AlignedVector<int32_t> result(NI32, 0);
  hn::StoreU(acc0, i32, result.data());

  // 255 * 127 * 4 = 129540 per lane per tile.
  const int32_t expected = 129540 * static_cast<int32_t>(total_tiles);
  for (size_t i = 0; i < NI32; ++i) {
    EXPECT_EQ(result[i], expected) << "lane=" << i;
  }
}

// Per-lane verification for 2 panels × 2 queries. Stores and checks every i32
// lane individually, catching lane-swap or shuffle bugs that SumAllLanes masks.
HWY_NOINLINE void TestAccumulatePerLaneMultiPanel() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  const size_t total_tiles = 4;
  constexpr size_t Mq = 2;

  hwy::AlignedVector<uint8_t> block0(total_tiles * N, 0);
  hwy::AlignedVector<uint8_t> block1(total_tiles * N, 0);
  for (size_t i = 0; i < block0.size(); ++i) {
    block0[i] = static_cast<uint8_t>((i * 5 + 7) % 256);
    block1[i] = static_cast<uint8_t>((i * 11 + 13) % 256);
  }

  hwy::AlignedVector<int8_t> queries[Mq];
  const int8_t* query_ptrs[Mq];
  for (size_t q = 0; q < Mq; ++q) {
    queries[q].resize(total_tiles * 4);
    for (size_t i = 0; i < queries[q].size(); ++i) {
      queries[q][i] = static_cast<int8_t>(((i * (q + 7) + q * 23) % 256) - 128);
    }
    query_ptrs[q] = queries[q].data();
  }

  VEC_GROUP_DEFINE(i32, a0);
  VEC_GROUP_DEFINE(i32, a1);
  const uint8_t* blks[] = {block0.data(), block1.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<Mq>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(a0),
      VEC_GROUP_ACCESSOR(a1));

  hwy::AlignedVector<int32_t> r00(NI32), r01(NI32), r10(NI32), r11(NI32);
  hn::StoreU(a00, i32, r00.data());
  hn::StoreU(a01, i32, r01.data());
  hn::StoreU(a10, i32, r10.data());
  hn::StoreU(a11, i32, r11.data());

  std::vector<int32_t> s00(NI32), s01(NI32), s10(NI32), s11(NI32);
  scalar_ref::ScalarAccumulate1PanelPerLane(
      query_ptrs[0], block0.data(), total_tiles, N, absl::MakeSpan(s00));
  scalar_ref::ScalarAccumulate1PanelPerLane(
      query_ptrs[1], block0.data(), total_tiles, N, absl::MakeSpan(s01));
  scalar_ref::ScalarAccumulate1PanelPerLane(
      query_ptrs[0], block1.data(), total_tiles, N, absl::MakeSpan(s10));
  scalar_ref::ScalarAccumulate1PanelPerLane(
      query_ptrs[1], block1.data(), total_tiles, N, absl::MakeSpan(s11));

  for (size_t k = 0; k < NI32; ++k) {
    EXPECT_EQ(r00[k], s00[k]) << "panel0 q=0 lane=" << k;
    EXPECT_EQ(r01[k], s01[k]) << "panel0 q=1 lane=" << k;
    EXPECT_EQ(r10[k], s10[k]) << "panel1 q=0 lane=" << k;
    EXPECT_EQ(r11[k], s11[k]) << "panel1 q=1 lane=" << k;
  }
}

// ============================================================================
// Algebraic property tests (no scalar reference)
// ============================================================================

// Linearity: doubling all query bytes should double every accumulator lane.
HWY_NOINLINE void TestAccumulateLinearity() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  const size_t total_tiles = 4;

  hwy::AlignedVector<uint8_t> block(total_tiles * N, 0);
  for (size_t i = 0; i < block.size(); ++i) {
    block[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
  }

  // Query with values in [-64, 63] so doubling stays in int8 range.
  hwy::AlignedVector<int8_t> query1x(total_tiles * 4, 0);
  hwy::AlignedVector<int8_t> query2x(total_tiles * 4, 0);
  for (size_t i = 0; i < query1x.size(); ++i) {
    int8_t v = static_cast<int8_t>(((i * 11 + 5) % 128) - 64);
    query1x[i] = v;
    query2x[i] = static_cast<int8_t>(v * 2);
  }

  const int8_t* ptrs1x[1] = {query1x.data()};
  const int8_t* ptrs2x[1] = {query2x.data()};
  const uint8_t* blks[] = {block.data()};

  VEC_GROUP_DEFINE(i32, acc1);
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, ptrs1x, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc1));

  VEC_GROUP_DEFINE(i32, acc2);
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, ptrs2x, blks, total_tiles, N, VEC_GROUP_ACCESSOR(acc2));

  hwy::AlignedVector<int32_t> r1(NI32), r2(NI32);
  hn::StoreU(acc10, i32, r1.data());
  hn::StoreU(acc20, i32, r2.data());

  for (size_t i = 0; i < NI32; ++i) {
    EXPECT_EQ(r2[i], 2 * r1[i]) << "lane=" << i;
  }
}

// Additivity over tiles: accumulating [0..A+B) equals the sum of
// accumulating [0..A) and [A..A+B) separately.
HWY_NOINLINE void TestAccumulateAdditivity() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  const size_t tiles_a = 3;
  const size_t tiles_b = 5;
  const size_t total_tiles = tiles_a + tiles_b;

  hwy::AlignedVector<uint8_t> block(total_tiles * N, 0);
  for (size_t i = 0; i < block.size(); ++i) {
    block[i] = static_cast<uint8_t>((i * 13 + 7) % 256);
  }

  hwy::AlignedVector<int8_t> query(total_tiles * 4, 0);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = static_cast<int8_t>(((i * 11 + 5) % 256) - 128);
  }

  // Full run: tiles [0, total_tiles).
  const int8_t* ptrs_full[1] = {query.data()};
  const uint8_t* blks_full[] = {block.data()};
  VEC_GROUP_DEFINE(i32, full);
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(u8, i8, i32, ptrs_full,
                                                  blks_full, total_tiles, N,
                                                  VEC_GROUP_ACCESSOR(full));

  // Part A: tiles [0, tiles_a).
  const int8_t* ptrs_a[1] = {query.data()};
  const uint8_t* blks_a[] = {block.data()};
  VEC_GROUP_DEFINE(i32, partA);
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, ptrs_a, blks_a, tiles_a, N, VEC_GROUP_ACCESSOR(partA));

  // Part B: tiles [tiles_a, total_tiles). Shift both block and query pointers.
  const int8_t* query_b_ptr = query.data() + tiles_a * 4;
  const int8_t* ptrs_b[1] = {query_b_ptr};
  const uint8_t* blks_b[] = {block.data() + tiles_a * N};
  VEC_GROUP_DEFINE(i32, partB);
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, ptrs_b, blks_b, tiles_b, N, VEC_GROUP_ACCESSOR(partB));

  hwy::AlignedVector<int32_t> rf(NI32), ra(NI32), rb(NI32);
  hn::StoreU(full0, i32, rf.data());
  hn::StoreU(partA0, i32, ra.data());
  hn::StoreU(partB0, i32, rb.data());

  for (size_t i = 0; i < NI32; ++i) {
    EXPECT_EQ(rf[i], ra[i] + rb[i]) << "lane=" << i;
  }
}

// Query permutation: swapping q=0 and q=1 swaps the accumulator outputs.
HWY_NOINLINE void TestAccumulateQueryPermutation() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  const size_t total_tiles = 4;

  hwy::AlignedVector<uint8_t> block(total_tiles * N, 0);
  for (size_t i = 0; i < block.size(); ++i) {
    block[i] = static_cast<uint8_t>((i * 5 + 7) % 256);
  }

  // Two distinct queries.
  hwy::AlignedVector<int8_t> qA(total_tiles * 4), qB(total_tiles * 4);
  for (size_t i = 0; i < qA.size(); ++i) {
    qA[i] = static_cast<int8_t>(((i * 3 + 1) % 256) - 128);
    qB[i] = static_cast<int8_t>(((i * 7 + 13) % 256) - 128);
  }

  // Order 1: [A, B]
  const int8_t* ptrs_ab[2] = {qA.data(), qB.data()};
  const uint8_t* blks[] = {block.data()};
  VEC_GROUP_DEFINE(i32, ab);
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<2>(
      u8, i8, i32, ptrs_ab, blks, total_tiles, N, VEC_GROUP_ACCESSOR(ab));

  // Order 2: [B, A]
  const int8_t* ptrs_ba[2] = {qB.data(), qA.data()};
  VEC_GROUP_DEFINE(i32, ba);
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<2>(
      u8, i8, i32, ptrs_ba, blks, total_tiles, N, VEC_GROUP_ACCESSOR(ba));

  hwy::AlignedVector<int32_t> ab0v(NI32), ab1v(NI32), ba0v(NI32), ba1v(NI32);
  hn::StoreU(ab0, i32, ab0v.data());
  hn::StoreU(ab1, i32, ab1v.data());
  hn::StoreU(ba0, i32, ba0v.data());
  hn::StoreU(ba1, i32, ba1v.data());

  for (size_t i = 0; i < NI32; ++i) {
    EXPECT_EQ(ab0v[i], ba1v[i]) << "A result mismatch lane=" << i;
    EXPECT_EQ(ab1v[i], ba0v[i]) << "B result mismatch lane=" << i;
  }
}

// Panel independence: panel 0's accumulator is identical whether called with
// 1 panel or 2 panels.
HWY_NOINLINE void TestAccumulatePanelIndependence() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  const size_t total_tiles = 4;

  hwy::AlignedVector<uint8_t> block0(total_tiles * N, 0);
  hwy::AlignedVector<uint8_t> block1(total_tiles * N, 0);
  for (size_t i = 0; i < block0.size(); ++i) {
    block0[i] = static_cast<uint8_t>((i * 5 + 7) % 256);
    block1[i] = static_cast<uint8_t>((i * 11 + 13) % 256);
  }

  hwy::AlignedVector<int8_t> query(total_tiles * 4, 0);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = static_cast<int8_t>(((i * 3 + 1) % 256) - 128);
  }
  const int8_t* query_ptrs[1] = {query.data()};

  // Run with 1 panel.
  VEC_GROUP_DEFINE(i32, solo);
  const uint8_t* blks1[] = {block0.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, query_ptrs, blks1, total_tiles, N, VEC_GROUP_ACCESSOR(solo));

  // Run with 2 panels.
  VEC_GROUP_DEFINE(i32, p0);
  VEC_GROUP_DEFINE(i32, p1);
  const uint8_t* blks2[] = {block0.data(), block1.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, query_ptrs, blks2, total_tiles, N, VEC_GROUP_ACCESSOR(p0),
      VEC_GROUP_ACCESSOR(p1));

  hwy::AlignedVector<int32_t> rs(NI32), rp(NI32);
  hn::StoreU(solo0, i32, rs.data());
  hn::StoreU(p00, i32, rp.data());

  for (size_t i = 0; i < NI32; ++i) {
    EXPECT_EQ(rs[i], rp[i]) << "lane=" << i;
  }
}

// Identical panels: when all panels share the same data, their accumulators
// must be equal.
HWY_NOINLINE void TestAccumulateIdenticalPanels() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const size_t NI32 = hn::Lanes(i32);

  const size_t total_tiles = 4;

  hwy::AlignedVector<uint8_t> block(total_tiles * N, 0);
  for (size_t i = 0; i < block.size(); ++i) {
    block[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
  }

  hwy::AlignedVector<int8_t> query(total_tiles * 4, 0);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = static_cast<int8_t>(((i * 11 + 5) % 256) - 128);
  }
  const int8_t* query_ptrs[1] = {query.data()};

  // All 3 panels point to the same block data.
  VEC_GROUP_DEFINE(i32, p0);
  VEC_GROUP_DEFINE(i32, p1);
  VEC_GROUP_DEFINE(i32, p2);
  const uint8_t* blks[] = {block.data(), block.data(), block.data()};
  ginger::HWY_NAMESPACE::MicroKernelAccumulate<1>(
      u8, i8, i32, query_ptrs, blks, total_tiles, N, VEC_GROUP_ACCESSOR(p0),
      VEC_GROUP_ACCESSOR(p1), VEC_GROUP_ACCESSOR(p2));

  hwy::AlignedVector<int32_t> r0(NI32), r1(NI32), r2(NI32);
  hn::StoreU(p00, i32, r0.data());
  hn::StoreU(p10, i32, r1.data());
  hn::StoreU(p20, i32, r2.data());

  for (size_t i = 0; i < NI32; ++i) {
    EXPECT_EQ(r0[i], r1[i]) << "p0 vs p1 lane=" << i;
    EXPECT_EQ(r0[i], r2[i]) << "p0 vs p2 lane=" << i;
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(AccumulateTest);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulate1PanelMq1);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulate1PanelMq4);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulate2Panels);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulate4Panels);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulateZeroInput);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulate1PanelKnownValues);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulateSignBoundary);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulate1TileMq4TwoPanels);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulateLargeTileCount);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulateMq8);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulateNearSaturation);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulatePerLaneMultiPanel);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulateLinearity);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulateAdditivity);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulateQueryPermutation);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulatePanelIndependence);
HWY_EXPORT_AND_TEST_P(AccumulateTest, TestAccumulateIdenticalPanels);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif
