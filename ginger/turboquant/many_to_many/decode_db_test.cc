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

// Unit tests for decode_db-inl.h: DB panel decode.
//
// Each SIMD function is validated against a scalar reference implementation
// that performs the equivalent computation using plain C++ loops.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/many_to_many/decode_db_test.cc"
// clang-format on
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "hwy/cache_control.h"   // IWYU pragma: keep
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

#ifndef DECODE_DB_TEST_SCALAR_REF_DEFINED_
#define DECODE_DB_TEST_SCALAR_REF_DEFINED_
namespace scalar_ref {

using ginger::turboquant_codebooks::kTurboQuantCentroidsUint8;

// Decode a single nibble to an unsigned uint8 centroid value.
inline uint8_t DecodeNibbleUnsigned(uint8_t packed_byte, int parity) {
  uint8_t idx = (parity == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
  return kTurboQuantCentroidsUint8[idx];
}

// Scalar reference for DecodeDbPanel.
inline void ScalarDecodeDbPanel(const uint8_t* db_data, size_t db_stride,
                                size_t db_start, size_t pi_start, size_t kP,
                                size_t points_per_panel, size_t num_dim_chunks,
                                size_t tiles_per_chunk, size_t N,
                                uint8_t* panel, absl::Span<const int> indices) {
  if (num_dim_chunks == 0) return;
  const size_t panel_bytes = num_dim_chunks * tiles_per_chunk * N;
  std::memset(panel, 0, panel_bytes);

  for (size_t pi = 0; pi < kP; ++pi) {
    const size_t cur = db_start + pi_start + pi;
    const uint8_t* d_ptr =
        db_data + (!indices.empty() ? indices[cur] : cur) * db_stride;

    for (size_t c = 0; c < num_dim_chunks; ++c) {
      const size_t co = c * tiles_per_chunk * N;
      for (size_t lane = 0; lane < N; ++lane) {
        uint8_t packed = d_ptr[c * N + lane];
        uint8_t even_val = DecodeNibbleUnsigned(packed, 0);
        uint8_t odd_val = DecodeNibbleUnsigned(packed, 1);

        size_t g = lane / 4;
        size_t byte_in_u32 = lane % 4;

        panel[co + (0 * points_per_panel + g) * N + pi * 4 + byte_in_u32] =
            even_val;
        panel[co + (1 * points_per_panel + g) * N + pi * 4 + byte_in_u32] =
            odd_val;
      }
    }
  }
}

}  // namespace scalar_ref
#endif  // DECODE_DB_TEST_SCALAR_REF_DEFINED_

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/decode_db-inl.h"
#include "ginger/turboquant/many_to_many/vec_group-inl.h"  // NOLINT

namespace ginger {
namespace HWY_NAMESPACE {
namespace {}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

HWY_NOINLINE void TestDecodeDbPanelVsScalar() {
  const size_t N = hn::Lanes(hn::ScalableTag<uint8_t>());
  const size_t points_per_panel = hn::Lanes(hn::ScalableTag<int32_t>());
  const size_t tiles_per_chunk = N / 2;

  for (size_t dim = 1; dim <= 128; ++dim) {
    const size_t bytes_per_point = (dim + 1) / 2;
    const size_t num_dim_chunks = (bytes_per_point + N - 1) / N;
    const size_t panel_bytes = num_dim_chunks * tiles_per_chunk * N;

    for (size_t kP :
         {size_t{1}, size_t{3}, std::min(size_t{4}, points_per_panel),
          points_per_panel}) {
      const size_t db_stride = bytes_per_point;
      hwy::AlignedVector<uint8_t> db_packed(kP * num_dim_chunks * N, 0);
      for (size_t p = 0; p < kP; ++p) {
        for (size_t i = 0; i < bytes_per_point; ++i) {
          db_packed[p * db_stride + i] =
              static_cast<uint8_t>((p * 13 + i * 7 + 3) % 256);
        }
      }

      ginger::TurboQuantCodec tq(dim, 42);
      auto base_points_or = ginger::TurboQuantDatasetSoA::Create(tq, kP);
      ASSERT_TRUE(base_points_or.ok());
      auto& base_points = *base_points_or;

      for (size_t p = 0; p < kP; ++p) {
        ginger::TurboQuantBasePointView view{
            .quantized_data = absl::MakeConstSpan(&db_packed[p * db_stride],
                                                  bytes_per_point)};
        ASSERT_TRUE(base_points.UpdatePoint(view, p).ok());
      }

      hwy::AlignedVector<uint8_t> simd_panel(panel_bytes, 0xCC);

      ginger::HWY_NAMESPACE::DecodeDbPanel(
          base_points, ginger::IndexSubset::Contiguous(0, kP),
          absl::Span<uint8_t>(simd_panel.data(), simd_panel.size()));

      hwy::AlignedVector<uint8_t> scalar_panel(panel_bytes, 0xCC);
      scalar_ref::ScalarDecodeDbPanel(
          db_packed.data(), db_stride,
          /*db_start=*/0, /*pi_start=*/0, kP, points_per_panel, num_dim_chunks,
          tiles_per_chunk, N, scalar_panel.data(), {});

      EXPECT_EQ(simd_panel, scalar_panel);
    }
  }
}

HWY_NOINLINE void TestDecodeDbPanelWithIndices() {
  const size_t N = hn::Lanes(hn::ScalableTag<uint8_t>());
  const size_t points_per_panel = hn::Lanes(hn::ScalableTag<int32_t>());
  const size_t tiles_per_chunk = N / 2;
  const size_t bytes_per_point = 32;
  const size_t num_dim_chunks = (bytes_per_point + N - 1) / N;
  const size_t panel_bytes = num_dim_chunks * tiles_per_chunk * N;
  const size_t num_total = 20;
  const size_t kP = 4;

  hwy::AlignedVector<uint8_t> db_packed(num_total * num_dim_chunks * N, 0);
  for (size_t p = 0; p < num_total; ++p) {
    for (size_t i = 0; i < bytes_per_point; ++i) {
      db_packed[p * bytes_per_point + i] =
          static_cast<uint8_t>((p * 31 + i * 11) % 256);
    }
  }

  std::vector<int> indices = {15, 3, 7, 0, 12, 9, 1,  5,
                              14, 6, 2, 8, 11, 4, 10, 13};

  size_t dim = bytes_per_point * 2;
  ginger::TurboQuantCodec tq(dim, 42);
  auto base_points_or = ginger::TurboQuantDatasetSoA::Create(tq, num_total);
  ASSERT_TRUE(base_points_or.ok());
  auto& base_points = *base_points_or;

  for (size_t p = 0; p < num_total; ++p) {
    ginger::TurboQuantBasePointView view{
        .quantized_data = absl::MakeConstSpan(&db_packed[p * bytes_per_point],
                                              bytes_per_point)};
    ASSERT_TRUE(base_points.UpdatePoint(view, p).ok());
  }

  hwy::AlignedVector<uint8_t> simd_panel(panel_bytes, 0);
  ginger::HWY_NAMESPACE::DecodeDbPanel(
      base_points,
      ginger::IndexSubset::Gathered(
          absl::MakeConstSpan(indices).subspan(0, kP)),
      absl::Span<uint8_t>(simd_panel.data(), simd_panel.size()));

  hwy::AlignedVector<uint8_t> scalar_panel(panel_bytes, 0);
  scalar_ref::ScalarDecodeDbPanel(
      db_packed.data(), bytes_per_point,
      /*db_start=*/0, /*pi_start=*/0, kP, points_per_panel, num_dim_chunks,
      tiles_per_chunk, N, scalar_panel.data(), indices);

  EXPECT_EQ(simd_panel, scalar_panel);
}

HWY_NOINLINE void TestDecodeDbPanelWithPiStart() {
  const size_t N = hn::Lanes(hn::ScalableTag<uint8_t>());
  const size_t points_per_panel = hn::Lanes(hn::ScalableTag<int32_t>());
  const size_t tiles_per_chunk = N / 2;
  const size_t bytes_per_point = 16;
  const size_t num_dim_chunks =
      std::max(size_t{1}, (bytes_per_point + N - 1) / N);
  const size_t panel_bytes = num_dim_chunks * tiles_per_chunk * N;
  const size_t db_start = 2;
  const size_t pi_start = 0;
  const size_t kP = 4;
  const size_t num_total = db_start + kP + 4;

  std::vector<uint8_t> db_packed(num_total * bytes_per_point, 0);
  for (size_t p = 0; p < num_total; ++p) {
    for (size_t i = 0; i < bytes_per_point; ++i) {
      db_packed[p * bytes_per_point + i] =
          static_cast<uint8_t>((p * 23 + i * 3) % 256);
    }
  }

  ginger::TurboQuantCodec tq((bytes_per_point * 2), 42);
  auto base_points_or = ginger::TurboQuantDatasetSoA::Create(tq, num_total);
  ASSERT_TRUE(base_points_or.ok());
  auto& base_points = *base_points_or;

  for (size_t p = 0; p < num_total; ++p) {
    ginger::TurboQuantBasePointView view{
        .quantized_data = absl::MakeConstSpan(&db_packed[p * bytes_per_point],
                                              bytes_per_point)};
    ASSERT_TRUE(base_points.UpdatePoint(view, p).ok());
  }

  hwy::AlignedVector<uint8_t> simd_panel(panel_bytes, 0);
  ginger::HWY_NAMESPACE::DecodeDbPanel(
      base_points, ginger::IndexSubset::Contiguous(db_start + pi_start, kP),
      absl::Span<uint8_t>(simd_panel.data(), simd_panel.size()));

  hwy::AlignedVector<uint8_t> scalar_panel(panel_bytes, 0);
  scalar_ref::ScalarDecodeDbPanel(db_packed.data(), bytes_per_point, db_start,
                                  pi_start, kP, points_per_panel,
                                  num_dim_chunks, tiles_per_chunk, N,
                                  scalar_panel.data(), {});

  EXPECT_EQ(simd_panel, scalar_panel);
}

HWY_NOINLINE void TestTransposeToTiles() {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<uint32_t> u32;
  const size_t N = hn::Lanes(u8);
  const size_t tiles_per_chunk = N / 2;
  const size_t num_dim_chunks = 1;
  const size_t panel_bytes = num_dim_chunks * tiles_per_chunk * N;

  hwy::AlignedVector<uint8_t> panel(panel_bytes, 0xCC);

  HWY_ALIGN uint8_t p_data[4 * HWY_MAX_BYTES] = {};
  for (size_t p = 0; p < 4; ++p) {
    for (size_t i = 0; i < N; ++i) {
      p_data[p * N + i] = (p * 13 + i * 7) % 256;
    }
  }

  auto v0 = hn::LoadU(u8, &p_data[0 * N]);
  auto v1 = hn::LoadU(u8, &p_data[1 * N]);
  auto v2 = hn::LoadU(u8, &p_data[2 * N]);
  auto v3 = hn::LoadU(u8, &p_data[3 * N]);

  // Test even parity (tile_base = 0).
  ginger::HWY_NAMESPACE::TransposeToTiles(u32, v0, v1, v2, v3, panel.data(), 0,
                                          N, 0);

  hwy::AlignedVector<uint8_t> expected(panel_bytes, 0xCC);
  for (size_t pi = 0; pi < 4; ++pi) {
    for (size_t lane = 0; lane < N; ++lane) {
      uint8_t val = p_data[pi * N + lane];
      size_t g = lane / 4;
      size_t b = lane % 4;
      expected[g * N + pi * 4 + b] = val;
    }
  }

  for (size_t i = 0; i < N / 4; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      EXPECT_EQ(panel[i * N + j], expected[i * N + j])
          << "Mismatch at group=" << i << " byte=" << j;
    }
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(DecodeDbTest);
HWY_EXPORT_AND_TEST_P(DecodeDbTest, TestDecodeDbPanelVsScalar);
HWY_EXPORT_AND_TEST_P(DecodeDbTest, TestDecodeDbPanelWithIndices);
HWY_EXPORT_AND_TEST_P(DecodeDbTest, TestDecodeDbPanelWithPiStart);
HWY_EXPORT_AND_TEST_P(DecodeDbTest, TestTransposeToTiles);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif
