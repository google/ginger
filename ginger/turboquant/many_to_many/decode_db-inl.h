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

// Database-specific SIMD helpers for TurboQuant decoding.
//
// Include after foreach_target.h and highway.h. The toggle guard handles
// per-target re-inclusion.
//
// Decodes database data into a block-transposed layout optimized for fast
// VNNI/SumOfMulQuadAccumulate scoring. Refer to README.md for more details.

#ifndef GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_DB_INL_H_
#define GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_DB_INL_H_
#include <algorithm>

#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_DB_INL_H_

// clang-format off
#if defined(GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_DB_TOGGLE) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_DB_TOGGLE
#undef GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_DB_TOGGLE
#else
#define GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_DB_TOGGLE
#endif
// clang-format on

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/vec_group-inl.h"
#include "hwy/highway.h"

namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using ::ginger::turboquant_codebooks::kTurboQuantCentroidsUint8;

// Dequantizes packed 4-bit nibbles into two 8-bit vectors (unsigned).
// Unpacks each byte in `packed_nibbles` into its even (low 4 bits) and odd
// (high 4 bits) nibbles. It then uses `TableLookupBytes` with the
// `unsigned_centroids` codebook to map indices to their actual values.
template <typename Uint8>
HWY_INLINE void DequantizeNibblesUnsigned(
    const hn::Vec<Uint8>& mask, const hn::Vec<Uint8>& unsigned_centroids,
    const hn::Vec<Uint8>& packed_nibbles, hn::Vec<Uint8>& even_vals,
    hn::Vec<Uint8>& odd_vals) {
  const hn::Vec<Uint8> even_idxs = hn::And(packed_nibbles, mask);
  const hn::Vec<Uint8> odd_idxs = hn::ShiftRight<4>(packed_nibbles);
  even_vals = hn::TableLookupBytes(unsigned_centroids, even_idxs);
  odd_vals = hn::TableLookupBytes(unsigned_centroids, odd_idxs);
}

// Transposes 4 decoded u8 vectors (one per point) and stores them into tile
// rows. Uses Highway's StoreInterleaved4 to interleave at the u32 level,
// producing contiguous groups [P0_g, P1_g, P2_g, P3_g], then scatters each
// 16-byte group to its tile row. Portable across all targets (x86, NEON, SVE).
HWY_INLINE void TransposeToTiles(const hn::ScalableTag<uint32_t>& u32,
                                 hn::Vec<hn::ScalableTag<uint8_t>> v0,
                                 hn::Vec<hn::ScalableTag<uint8_t>> v1,
                                 hn::Vec<hn::ScalableTag<uint8_t>> v2,
                                 hn::Vec<hn::ScalableTag<uint8_t>> v3,
                                 uint8_t* panel, size_t tile_base, size_t N,
                                 size_t pi) {
  const auto p0 = hn::BitCast(u32, v0);
  const auto p1 = hn::BitCast(u32, v1);
  const auto p2 = hn::BitCast(u32, v2);
  const auto p3 = hn::BitCast(u32, v3);

  HWY_ALIGN uint32_t buf[HWY_MAX_LANES_D(hn::ScalableTag<uint32_t>) * 4];
  hn::StoreInterleaved4(p0, p1, p2, p3, u32, buf);

  const auto* src = reinterpret_cast<const uint8_t*>(buf);
  for (size_t g = 0; g < N / 4; ++g) {
    std::memcpy(panel + tile_base + g * N + pi * 4, src + g * 16, 16);
  }
}

// Returns the number of bytes per decoded DB panel.
HWY_INLINE size_t DbPanelBytes(const ::ginger::TurboQuantDatasetSoA& db) {
  const size_t N = hn::Lanes(hn::ScalableTag<uint8_t>());
  const size_t num_dim_chunks = hwy::DivCeil(db.num_bytes_per_datapoint(), N);
  const size_t tiles_per_chunk = N / 2;
  return num_dim_chunks * tiles_per_chunk * N;
}

// Returns the number of database points per panel (i.e., points_per_panel =
// N/4).
HWY_INLINE size_t DbPanelNumPoints() {
  return hn::Lanes(hn::ScalableTag<int32_t>());
}

// Decodes database points specified by IndexSubset into a block-transposed
// panel.
//
// Example usage:
//   TurboQuantDatasetSoA database = ...;
//   size_t panel_bytes = DbPanelBytes(database);
//   size_t num_db_points = DbPanelNumPoints();
//   std::vector<uint8_t> panel(panel_bytes);
//   DecodeDbPanel(database, IndexSubset::Contiguous(start_idx, num_db_points),
//   absl::MakeSpan(panel));
HWY_INLINE void DecodeDbPanel(const ::ginger::TurboQuantDatasetSoA& db,
                              IndexSubset db_indices,
                              absl::Span<uint8_t> panel_span) {
  // TODO: Support 3-bit prefix in DecodeDbPanel.
  ABSL_CHECK(!db.has_3bit_prefix());
  const size_t num_db_points = db_indices.size();
  uint8_t* HWY_RESTRICT panel = panel_span.data();
  using Uint8 = hn::ScalableTag<uint8_t>;
  const Uint8 u8;
  const size_t N = hn::Lanes(u8);
  const size_t num_dim_chunks = hwy::DivCeil(db.num_bytes_per_datapoint(), N);
  if (num_dim_chunks == 0) return;

  const size_t db_stride = db.stride();
  const uint8_t* db_data = db.quantized_data();
  const size_t points_per_panel = N / 4;
  const size_t tiles_per_chunk = N / 2;

  const hn::Vec<Uint8> mask = hn::Set(u8, 0x0F);
  const hn::Vec<Uint8> unsigned_centroids =
      hn::LoadDup128(u8, kTurboQuantCentroidsUint8.data());
  const size_t panel_bytes = num_dim_chunks * tiles_per_chunk * N;
  if (num_db_points < points_per_panel) std::memset(panel, 0, panel_bytes);

  const hn::ScalableTag<uint32_t> u32;

  for (size_t pi = 0; pi < num_db_points; pi += 4) {
    const size_t count = std::min(size_t{4}, num_db_points - pi);

    for (size_t j = pi + 1; j < std::min(pi + 4, num_db_points); ++j) {
      absl::PrefetchToLocalCache(db_data + db_indices[j] * db_stride);
    }

    for (size_t c = 0; c < num_dim_chunks; ++c) {
      VEC_GROUP_DEFINE(u8, d_even);
      VEC_GROUP_DEFINE(u8, d_odd);
      ForIndex<4>([&](auto i) {
        if (i.value < count) {
          const uint8_t* d_ptr = db_data + db_indices[pi + i.value] * db_stride;
          const auto packed_nibbles = hn::LoadU(u8, d_ptr + c * N);
          DequantizeNibblesUnsigned<Uint8>(
              mask, unsigned_centroids, packed_nibbles,
              VEC_GROUP_ACCESSOR(d_even)(i), VEC_GROUP_ACCESSOR(d_odd)(i));
        } else {
          VEC_GROUP_ACCESSOR(d_even)(i) = hn::Zero(u8);
          VEC_GROUP_ACCESSOR(d_odd)(i) = hn::Zero(u8);
        }
      });

      const size_t co = c * tiles_per_chunk * N;

      TransposeToTiles(u32, d_even0, d_even1, d_even2, d_even3, panel, co, N,
                       pi);
      TransposeToTiles(u32, d_odd0, d_odd1, d_odd2, d_odd3, panel,
                       co + points_per_panel * N, N, pi);
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger

#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_DECODE_DB_TOGGLE
        // // NOLINT
