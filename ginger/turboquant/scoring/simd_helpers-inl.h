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

// Per-target SIMD helpers for TurboQuant bit extraction.
//
// Include after foreach_target.h and highway.h. The toggle guard handles
// per-target re-inclusion.

// Dummy include guard to satisfy the presubmit check.
#ifndef GINGER_TURBOQUANT_SCORING_SIMD_HELPERS_INL_H_
#define GINGER_TURBOQUANT_SCORING_SIMD_HELPERS_INL_H_
#include "absl/log/absl_check.h"
#include "ginger/turboquant/codebooks/piecewise_cubic.h"
#endif  // GINGER_TURBOQUANT_SCORING_SIMD_HELPERS_INL_H_

// Highway's reinclude guard toggle.
// clang-format off
#if defined(GINGER_TURBOQUANT_SCORING_SIMD_HELPERS_TOGGLE) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef GINGER_TURBOQUANT_SCORING_SIMD_HELPERS_TOGGLE
#undef GINGER_TURBOQUANT_SCORING_SIMD_HELPERS_TOGGLE
#else
#define GINGER_TURBOQUANT_SCORING_SIMD_HELPERS_TOGGLE
#endif
// clang-format on

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hwy/highway.h"

namespace ginger {
namespace HWY_NAMESPACE {
namespace turboquant_internal {

namespace hn = hwy::HWY_NAMESPACE;

// LoadPackedBits: extract N contiguous k-bit fields from a packed bitstream
// into uint32 lanes.
//
// Field `i` of the result vector = bits [d*k + i*k .. d*k + (i+1)*k - 1]
// of `data`.
//
// Strategy (compile-time + runtime):
//   k == 8:  Byte-aligned PromoteTo.
//   k == 4, N*4 <= 32:  Broadcast uint32, per-lane shift, mask.
//   k == 4, N*4 > 32, HWY_TARGET <= HWY_AVX3:
//           TableLookupBytes shuffle (AVX-512 optimized).
//   k == 6, N == 16, HWY_TARGET <= HWY_AVX3:
//           TableLookupBytes shuffle (AVX-512 optimized).
//   k in {1,2,3,6}, N*k <= 32:  Broadcast uint32, per-lane shift, mask.
//   N*k in (32, 64]:  Half-split, two uint32 broadcasts.
//   N*k > 64:  Half-split, two uint64 broadcasts + TruncateTo.
template <int k, typename UInt32>
HWY_INLINE hn::Vec<UInt32> LoadPackedBits(const UInt32 du,
                                          const uint8_t* HWY_RESTRICT data,
                                          size_t d) {
  static_assert(k >= 1 && (k <= 4 || k == 6 || k == 8));
  static_assert(HWY_IS_LITTLE_ENDIAN);
  const size_t N = hn::Lanes(du);
  constexpr uint32_t kMask = (1u << k) - 1;

  // Case 1: k=8, byte-aligned. Direct load + widen.
  if constexpr (k == 8) {
    const hn::Rebind<uint8_t, UInt32> du8;
    return hn::PromoteTo(du, hn::LoadU(du8, data + d));
  }
  const size_t bit_offset = d * k;
  const size_t byte_offset = bit_offset >> 3;
  const size_t intra_byte = bit_offset & 7;

  // Case 2: All N fields fit in a single uint32.
  if (N * k <= 32) {
    uint32_t word = 0;
    memcpy(&word, data + byte_offset,
           std::min<size_t>(hwy::DivCeil(N * k + intra_byte, 8), sizeof(word)));
    word >>= intra_byte;
    const auto shifts = hn::Mul(hn::Iota(du, 0), hn::Set(du, uint32_t{k}));
    return hn::And(hn::Shr(hn::Set(du, word), shifts), hn::Set(du, kMask));
  }

  // Cases below: N*k > 32, guaranteed byte-aligned.
  ABSL_DCHECK_EQ(intra_byte, 0u);

#if HWY_TARGET <= HWY_AVX3
  // Case 3: Optimized nibble shuffle for k=4 on AVX-512 wide vectors (N=16).
  //
  // For N=16, 16 nibbles = 64 bits (8 bytes) of packed data [B0..B7].
  // We load the 8 bytes as a uint64 and broadcast to every 64-bit lane, so
  // each 128-bit sub-lane (16 bytes) sees: [B0..B7, B0..B7].
  // Shuffle indices (`shuf` = j/8 + j%4) select the byte pair for each lane:
  //   Lane  0 (even): [B0,B1,B2,B3] -> nibble 0 is low nibble of B0
  //   Lane  1 (odd):  [B0,B1,B2,B3] -> nibble 1 is high nibble of B0
  //   Lane  2 (even): [B1,B2,B3,B4] -> nibble 2 is low nibble of B1
  //   Lane  3 (odd):  [B1,B2,B3,B4] -> nibble 3 is high nibble of B1
  // Even lanes shift right by 0, odd lanes by 4, then mask with 0x0F.
  if constexpr (k == 4) {
    const hn::Repartition<uint8_t, UInt32> du8;
    // Load N/2 bytes as uint64, broadcast into every 64-bit lane.
    uint64_t word64 = 0;
    memcpy(&word64, data + byte_offset,
           std::min<size_t>(hwy::DivCeil(N * k, 8), sizeof(word64)));
    const auto dup =
        hn::BitCast(du8, hn::Set(hn::Repartition<uint64_t, UInt32>{}, word64));
    // Shuffle: byte lane j should hold byte j/8 + j%4, so each uint32
    // lane gets the byte pair containing its nibble.
    const auto iota8 = hn::Iota(du8, 0);
    const auto shuf =
        hn::Add(hn::ShiftRight<3>(iota8), hn::And(iota8, hn::Set(du8, 3)));
    const auto shuffled = hn::BitCast(du, hn::TableLookupBytes(dup, shuf));
    // Even lanes need low nibble (shift 0), odd lanes need high (shift 4).
    // nibble_shifts = [0, 4, 0, 4, ...]
    const auto nibble_shifts =
        hn::ShiftLeft<2>(hn::And(hn::Iota(du, 0), hn::Set(du, 1u)));
    return hn::And(hn::Shr(shuffled, nibble_shifts), hn::Set(du, kMask));
  }
  // Case 3b: Optimized 6-bit shuffle for k=6 on AVX-512 wide vectors (N=16).
  //
  // For N=16, 16 6-bit values = 96 bits (12 bytes) of packed data [B0..B11],
  // consisting of 4 groups of 4 values (3 bytes = 24 bits per 128-bit block).
  // We broadcast bytes B0..B7 (uint64) and B8..B11 (uint32) and interleave them
  // so every 128-bit block holds [B0..B11, B8..B11]. Within block b (0..3),
  // all 4 uint32 lanes select the 4 bytes starting at 3*b, shift right by
  // [0, 6, 12, 18], and mask with 0x3F.
  if constexpr (k == 6) {
    if (N == 16) {
      const hn::Repartition<uint8_t, UInt32> du8;
      const hn::Repartition<uint64_t, UInt32> du64;
      uint64_t lo64 = 0;
      uint32_t hi32 = 0;
      memcpy(&lo64, data + byte_offset, sizeof(lo64));
      memcpy(&hi32, data + byte_offset + sizeof(lo64), sizeof(hi32));
      const auto dup = hn::BitCast(
          du8, hn::InterleaveLower(du64, hn::Set(du64, lo64),
                                   hn::BitCast(du64, hn::Set(du, hi32))));
      const auto iota8 = hn::Iota(du8, 0);
      const auto block = hn::ShiftRight<4>(iota8);
      const auto block3 = hn::Add(block, hn::ShiftLeft<1>(block));
      const auto shuf = hn::Add(block3, hn::And(iota8, hn::Set(du8, 3)));
      const auto shuffled = hn::BitCast(du, hn::TableLookupBytes(dup, shuf));
      const auto lane_in_block = hn::And(hn::Iota(du, 0), hn::Set(du, 3u));
      const auto shifts = hn::Mul(lane_in_block, hn::Set(du, uint32_t{k}));
      return hn::And(hn::Shr(shuffled, shifts), hn::Set(du, kMask));
    }
  }
#endif  // HWY_TARGET <= HWY_AVX3

  // Case 4: Half-split for N*k in (32, 64]. Each half fits in uint32.
  if (N * k <= 64) {
    const hn::Half<UInt32> dh;
    const auto half_shifts = hn::Mul(hn::Iota(dh, 0), hn::Set(dh, uint32_t{k}));
    const auto half_mask = hn::Set(dh, kMask);
    uint32_t lo_word = 0, hi_word = 0;
    const size_t half_offset = hwy::DivCeil((N / 2) * k, 8);
    const size_t half_bytes = std::min<size_t>(half_offset, sizeof(uint32_t));
    memcpy(&lo_word, data + byte_offset, half_bytes);
    memcpy(&hi_word, data + byte_offset + half_offset, half_bytes);
    return hn::Combine(
        du, hn::And(hn::Shr(hn::Set(dh, hi_word), half_shifts), half_mask),
        hn::And(hn::Shr(hn::Set(dh, lo_word), half_shifts), half_mask));
  }

  // Case 5: Half-split using uint64 for N*k > 64.
  {
    const hn::Repartition<uint64_t, UInt32> du64;
    const hn::Half<UInt32> dh;
    uint64_t lo_word = 0, hi_word = 0;
    const size_t half_offset = hwy::DivCeil((N / 2) * k, 8);
    const size_t half_bytes = std::min<size_t>(half_offset, sizeof(uint64_t));
    memcpy(&lo_word, data + byte_offset, half_bytes);
    memcpy(&hi_word, data + byte_offset + half_offset, half_bytes);
    const auto shifts64 =
        hn::Mul(hn::Iota(du64, 0), hn::Set(du64, uint64_t{k}));
    const auto mask64 = hn::Set(du64, uint64_t{kMask});
    return hn::Combine(
        du,
        hn::TruncateTo(
            dh, hn::And(hn::Shr(hn::Set(du64, hi_word), shifts64), mask64)),
        hn::TruncateTo(
            dh, hn::And(hn::Shr(hn::Set(du64, lo_word), shifts64), mask64)));
  }
}

// PiecewiseCubicCentroid: evaluates the 4+k (k = 6, 8) centroid as
// c0 + t * (c1 + t * (c2 + t * c3)) with per-segment coefficients looked up in
// registers (see piecewise_cubic.h). Requires 16 lanes.
template <int k, class Float, class UInt32, class Int32>
HWY_INLINE hn::Vec<Float> PiecewiseCubicCentroid(Float df, UInt32 du, Int32 di,
                                                 hn::Vec<UInt32> codes,
                                                 hn::Vec<UInt32> extra) {
  namespace cb = ::ginger::turboquant_codebooks;
  constexpr int kShift = k - cb::kTailSegmentBits<k>;
  const auto& coef = cb::kPiecewiseCubic<k>.coef;

  // 4-bit codes: bits 0..2 are the bucket, bit 3 is the sign. ShiftLeft<28>
  // moves the sign bit to the float sign position (bit 31).
  const auto bucket = hn::And(codes, hn::Set(du, 7u));
  const auto sign = hn::ShiftLeft<28>(hn::And(codes, hn::Set(du, 8u)));

  // For buckets 0..6, extra is directly t within the bucket's segment. For the
  // tail bucket 7, the upper bits of extra select a sub-segment and the lower
  // kShift bits are t.
  const auto is_tail = hn::Eq(bucket, hn::Set(du, 7u));
  const auto seg = hn::Add(
      bucket, hn::IfThenElseZero(is_tail, hn::ShiftRight<kShift>(extra)));
  const auto t = hn::IfThenElse(
      is_tail, hn::And(extra, hn::Set(du, (1u << kShift) - 1)), extra);
  const auto tf = hn::ConvertTo(df, hn::BitCast(di, t));

  // Per-lane lookup of the coefficients coef[j][seg].
  const auto idx = hn::IndicesFromVec(df, hn::BitCast(di, seg));
  auto lookup = [&](int j) HWY_ATTR {
    if constexpr (cb::kNumCubicSegments<k> <= 16) {
      return hn::TableLookupLanes(hn::Load(df, coef[j]), idx);
    } else {
      return hn::TwoTablesLookupLanes(df, hn::Load(df, coef[j]),
                                      hn::Load(df, coef[j] + 16), idx);
    }
  };

  // Horner's method: c0 + t * (c1 + t * (c2 + t * c3)).
  auto v = hn::MulAdd(lookup(3), tf, lookup(2));
  v = hn::MulAdd(v, tf, lookup(1));
  v = hn::MulAdd(v, tf, lookup(0));
  return hn::BitCast(df, hn::Xor(hn::BitCast(du, v), sign));
}

// LookupCentroid: looks up float centroid values given 4-bit base codes and
// k-bit extra sub-indices.
//
// Encapsulates the table-lookup strategy:
//   k=1,2: positive centroids table with TableLookupLanes /
//          TwoTablesLookupLanes / blend tree, plus XOR sign bit from codes.
//   k=3:   signed centroids table with 3-level blend tree on N>=16, or
//          GatherIndex on N<16. Sign is baked into the table.
//   k=4:   signed centroids table with GatherIndex. Sign is baked in.
//   k=6,8: piecewise-cubic evaluation on N=16, otherwise GatherIndex on the
//          signed centroids table (same values).
template <int k, class Float, class UInt32, class Int32>
HWY_INLINE hn::Vec<Float> LookupCentroid(
    Float df, UInt32 du, Int32 di, hn::Vec<UInt32> codes, hn::Vec<UInt32> extra,
    const float* HWY_RESTRICT pos_table,
    const float* HWY_RESTRICT signed_table) {
  static_assert(k >= 1 && (k <= 4 || k == 6 || k == 8));
  const size_t N = hn::Lanes(df);
  if constexpr (k == 1) {
    const auto bucket_mask = hn::Set(du, 7u);
    const auto sign_raw_mask = hn::Set(du, 8u);
    const auto c0 = hn::Load(df, pos_table);
    const auto bucket = hn::And(codes, bucket_mask);
    const auto lookup_idx = hn::Or(hn::ShiftLeft<1>(bucket), extra);
    hn::Vec<Float> centroid_v;
    if (N >= 16) {
      const auto indices = hn::IndicesFromVec(df, lookup_idx);
      centroid_v = hn::TableLookupLanes(c0, indices);
    } else if (N == 8) {
      const auto c1 = hn::Load(df, pos_table + N);
      const auto indices = hn::IndicesFromVec(df, lookup_idx);
      centroid_v = hn::TwoTablesLookupLanes(df, c0, c1, indices);
    } else {
      centroid_v = hn::GatherIndex(df, pos_table, hn::BitCast(di, lookup_idx));
    }
    const auto signs = hn::ShiftLeft<28>(hn::And(codes, sign_raw_mask));
    return hn::BitCast(df, hn::Xor(hn::BitCast(du, centroid_v), signs));
  } else if constexpr (k == 2) {
    const auto bucket_mask = hn::Set(du, 7u);
    const auto sign_raw_mask = hn::Set(du, 8u);
    const auto c0 = hn::Load(df, pos_table);
    const auto bucket = hn::And(codes, bucket_mask);
    const auto lookup_idx = hn::Or(hn::ShiftLeft<2>(bucket), extra);
    hn::Vec<Float> centroid_v;
    if (N >= 16) {
      const auto c1 = hn::Load(df, pos_table + N);
      const auto indices = hn::IndicesFromVec(df, lookup_idx);
      centroid_v = hn::TwoTablesLookupLanes(df, c0, c1, indices);
    } else if (N == 8) {
      const auto c1 = hn::Load(df, pos_table + N);
      const auto c2 = hn::Load(df, pos_table + 2 * N);
      const auto c3 = hn::Load(df, pos_table + 3 * N);
      const auto m8 = hn::RebindMask(
          df, hn::Ne(hn::And(lookup_idx, hn::Set(du, 8u)), hn::Zero(du)));
      const auto m16 = hn::RebindMask(
          df, hn::Ne(hn::And(lookup_idx, hn::Set(du, 16u)), hn::Zero(du)));
      const auto indices =
          hn::IndicesFromVec(df, hn::And(lookup_idx, hn::Set(du, 7u)));
      const auto res01 = hn::IfThenElse(m8, hn::TableLookupLanes(c1, indices),
                                        hn::TableLookupLanes(c0, indices));
      const auto res23 = hn::IfThenElse(m8, hn::TableLookupLanes(c3, indices),
                                        hn::TableLookupLanes(c2, indices));
      centroid_v = hn::IfThenElse(m16, res23, res01);
    } else {
      centroid_v = hn::GatherIndex(df, pos_table, hn::BitCast(di, lookup_idx));
    }
    const auto signs = hn::ShiftLeft<28>(hn::And(codes, sign_raw_mask));
    return hn::BitCast(df, hn::Xor(hn::BitCast(du, centroid_v), signs));
  } else if constexpr (k == 3) {
    const auto full_idx = hn::Or(hn::ShiftLeft<3>(codes), extra);
    if (N >= 16) {
      const auto c0 = hn::Load(df, signed_table);
      const auto c1 = hn::Load(df, signed_table + N);
      const auto c2 = hn::Load(df, signed_table + 2 * N);
      const auto c3 = hn::Load(df, signed_table + 3 * N);
      const auto c4 = hn::Load(df, signed_table + 4 * N);
      const auto c5 = hn::Load(df, signed_table + 5 * N);
      const auto c6 = hn::Load(df, signed_table + 6 * N);
      const auto c7 = hn::Load(df, signed_table + 7 * N);
      const auto m16 = hn::RebindMask(
          df, hn::Ne(hn::And(full_idx, hn::Set(du, 16u)), hn::Zero(du)));
      const auto m32 = hn::RebindMask(
          df, hn::Ne(hn::And(full_idx, hn::Set(du, 32u)), hn::Zero(du)));
      const auto m64 = hn::RebindMask(
          df, hn::Ne(hn::And(full_idx, hn::Set(du, 64u)), hn::Zero(du)));
      const auto indices =
          hn::IndicesFromVec(df, hn::And(full_idx, hn::Set(du, 15u)));
      const auto res01 = hn::IfThenElse(m16, hn::TableLookupLanes(c1, indices),
                                        hn::TableLookupLanes(c0, indices));
      const auto res23 = hn::IfThenElse(m16, hn::TableLookupLanes(c3, indices),
                                        hn::TableLookupLanes(c2, indices));
      const auto res03 = hn::IfThenElse(m32, res23, res01);
      const auto res45 = hn::IfThenElse(m16, hn::TableLookupLanes(c5, indices),
                                        hn::TableLookupLanes(c4, indices));
      const auto res67 = hn::IfThenElse(m16, hn::TableLookupLanes(c7, indices),
                                        hn::TableLookupLanes(c6, indices));
      const auto res47 = hn::IfThenElse(m32, res67, res45);
      return hn::IfThenElse(m64, res47, res03);
    } else {
      return hn::GatherIndex(df, signed_table, hn::BitCast(di, full_idx));
    }
  } else {
    if constexpr (k == 6 || k == 8) {
      if (N == 16) {
        return PiecewiseCubicCentroid<k>(df, du, di, codes, extra);
      }
    }
    const auto full_idx = hn::Or(hn::ShiftLeft<k>(codes), extra);
    return hn::GatherIndex(df, signed_table, hn::BitCast(di, full_idx));
  }
}

}  // namespace turboquant_internal
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace ginger

// clang-format off
#endif  // GINGER_TURBOQUANT_SCORING_SIMD_HELPERS_TOGGLE // NOLINT
// clang-format on
