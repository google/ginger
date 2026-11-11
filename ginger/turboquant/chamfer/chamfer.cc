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

#include "ginger/turboquant/chamfer/chamfer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/chamfer/point_cloud_query.h"
#include "ginger/turboquant/codebooks/codebooks.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/chamfer/chamfer.cc"
// clang-format on
#include "hwy/aligned_allocator.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

// Reinclude guard to prevent redefining ChamferWorkspace.
#ifndef GINGER_TURBOQUANT_CHAMFER_CHAMFER_CC_WORKSPACE_
#define GINGER_TURBOQUANT_CHAMFER_CHAMFER_CC_WORKSPACE_

namespace ginger {

// Reusable thread-local memory workspace used during Chamfer scoring.
struct ChamferWorkspace {
  // Candidate DB panel metadata for the current batch of neighbor documents.
  // Let `N = hn::Lanes(uint8)`. A panel is a group of `N/4` points. The GEMM
  // uses 4 dimensions per point in a panel because `SumOfMulQuadAccumulate`
  // takes N bytes and outputs N/4 int32 scores.
  std::vector<size_t> panel_indices;
  std::vector<uint32_t> panel_to_doc;
  std::vector<uint8_t> panel_valid_len;

  // Buffers for unpacking 4-bit indices into 8-bit codebook entries.
  hwy::AlignedVector<uint8_t> panel_buf;
  hwy::AlignedVector<float> block_norms;

  // Row-major matrix of current best minimum distance per document per query
  // for the current batch: [batch_neighbors x num_queries].
  std::vector<float> doc_min_dists;

  // Resizes per-batch metadata and initializes doc_min_dists to +inf.
  void PrepareForBatch(size_t total_panels, size_t num_neighbors,
                       size_t num_query_tokens) {
    panel_indices.resize(total_panels);
    panel_to_doc.resize(total_panels);
    panel_valid_len.resize(total_panels);
    doc_min_dists.assign(num_neighbors * num_query_tokens,
                         std::numeric_limits<float>::max());
  }
};

// Number of panels unpacked into scratch buffers per GEMM block.
constexpr size_t kPanelsPerBlock = 8;

// Result struct for PrepareTransposedPanelsImpl.
struct TransposedPanelData {
  size_t panel_bytes = 0;
  size_t points_per_panel = 0;
  std::vector<size_t> doc_panel_offsets;
  hwy::AlignedVector<uint8_t> transposed_panels;
  hwy::AlignedVector<float> panel_norms;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_CHAMFER_CHAMFER_CC_WORKSPACE_

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/decode_db-inl.h"
#include "ginger/turboquant/many_to_many/decode_queries-inl.h"
#include "ginger/turboquant/many_to_many/gemm_loop-inl.h"

namespace ginger {


namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;
namespace rg = ::ginger::HWY_NAMESPACE;
using ::ginger::turboquant_codebooks::kTurboQuantCentroidsUint8;
using rg::DbPanelBytes;
using rg::DbPanelNumPoints;
using rg::ForIndex;
using rg::RunQueryLoop;
using rg::TransposeToTiles;

// Transposes a batch of database tokens (up to `points_per_panel`) from
// row-major layout into interleaved tiles in `temp_panel`, then packs pairs of
// 4-bit index tiles into single byte tiles in `dst_panel`.
HWY_INLINE void TransposeAndPackDbPanel(const TurboQuantDatasetSoA& db,
                                        size_t start_token, size_t valid_len,
                                        uint8_t* HWY_RESTRICT temp_panel,
                                        uint8_t* HWY_RESTRICT dst_panel) {
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
  const size_t panel_bytes = num_dim_chunks * tiles_per_chunk * N;
  if (valid_len < points_per_panel) std::memset(temp_panel, 0, panel_bytes);

  const hn::ScalableTag<uint32_t> u32;

  for (size_t point_idx = 0; point_idx < valid_len; point_idx += 4) {
    const size_t num_points = std::min(size_t{4}, valid_len - point_idx);
    for (size_t chunk_idx = 0; chunk_idx < num_dim_chunks; ++chunk_idx) {
      VEC_GROUP_DEFINE(u8, even_nibbles);
      VEC_GROUP_DEFINE(u8, odd_nibbles);
      ForIndex<4>([&](auto i) {
        if (i.value < num_points) {
          const uint8_t* token_ptr =
              db_data + (start_token + point_idx + i.value) * db_stride;
          const hn::Vec<Uint8> packed_nibbles =
              hn::LoadU(u8, token_ptr + chunk_idx * N);
          VEC_GROUP_ACCESSOR(even_nibbles)(i) = hn::And(packed_nibbles, mask);
          VEC_GROUP_ACCESSOR(odd_nibbles)(i) =
              hn::ShiftRight<4>(packed_nibbles);
        } else {
          VEC_GROUP_ACCESSOR(even_nibbles)(i) = hn::Zero(u8);
          VEC_GROUP_ACCESSOR(odd_nibbles)(i) = hn::Zero(u8);
        }
      });
      const size_t chunk_byte_offset = chunk_idx * tiles_per_chunk * N;
      TransposeToTiles(u32, even_nibbles0, even_nibbles1, even_nibbles2,
                       even_nibbles3, temp_panel, chunk_byte_offset, N,
                       point_idx);
      TransposeToTiles(u32, odd_nibbles0, odd_nibbles1, odd_nibbles2,
                       odd_nibbles3, temp_panel,
                       chunk_byte_offset + points_per_panel * N, N, point_idx);
    }
  }

  // Pack consecutive pairs of 4-bit index tiles into single byte tiles.
  const size_t num_tile_pairs = (num_dim_chunks * tiles_per_chunk) / 2;
  for (size_t pair_idx = 0; pair_idx < num_tile_pairs; ++pair_idx) {
    const hn::Vec<Uint8> even_tile =
        hn::LoadU(u8, temp_panel + (2 * pair_idx) * N);
    const hn::Vec<Uint8> odd_tile =
        hn::LoadU(u8, temp_panel + (2 * pair_idx + 1) * N);
    hn::StoreU(hn::Or(even_tile, hn::ShiftLeft<4>(odd_tile)), u8,
               dst_panel + pair_idx * N);
  }
}

// Converts raw document token embeddings into pre-transposed, packed int4
// panels and extracts norm scaling factors during dataset initialization.
TransposedPanelData PrepareTransposedPanelsImpl(
    const TurboQuantDatasetSoA& tokens, absl::Span<const size_t> doc_offsets) {
  TransposedPanelData result;
  const size_t num_docs = doc_offsets.empty() ? 0 : doc_offsets.size() - 1;
  const size_t full_panel_bytes = DbPanelBytes(tokens);
  const size_t N = hn::Lanes(hn::ScalableTag<uint8_t>());
  ABSL_CHECK_EQ((full_panel_bytes / N) % 2, 0);

  result.panel_bytes = full_panel_bytes / 2;
  result.points_per_panel = DbPanelNumPoints();
  result.doc_panel_offsets.resize(num_docs + 1);

  size_t total_panels = 0;
  for (size_t doc_idx = 0; doc_idx < num_docs; ++doc_idx) {
    result.doc_panel_offsets[doc_idx] = total_panels;
    const size_t num_doc_tokens =
        doc_offsets[doc_idx + 1] - doc_offsets[doc_idx];
    total_panels += hwy::DivCeil(num_doc_tokens, result.points_per_panel);
  }
  result.doc_panel_offsets[num_docs] = total_panels;

  result.transposed_panels.resize(total_panels * result.panel_bytes);
  result.panel_norms.resize(total_panels * result.points_per_panel, 0.0f);

  hwy::AlignedVector<uint8_t> temp_panel(full_panel_bytes);
  size_t panel_idx = 0;
  for (size_t doc_idx = 0; doc_idx < num_docs; ++doc_idx) {
    const size_t doc_start_token = doc_offsets[doc_idx];
    const size_t num_doc_tokens = doc_offsets[doc_idx + 1] - doc_start_token;
    const size_t num_doc_panels =
        hwy::DivCeil(num_doc_tokens, result.points_per_panel);
    for (size_t panel_in_doc = 0; panel_in_doc < num_doc_panels;
         ++panel_in_doc, ++panel_idx) {
      const size_t panel_start_token =
          doc_start_token + panel_in_doc * result.points_per_panel;
      const size_t valid_len =
          std::min(result.points_per_panel,
                   num_doc_tokens - panel_in_doc * result.points_per_panel);
      TransposeAndPackDbPanel(
          tokens, panel_start_token, valid_len, temp_panel.data(),
          result.transposed_panels.data() + panel_idx * result.panel_bytes);
      float* panel_norms =
          result.panel_norms.data() + panel_idx * result.points_per_panel;
      for (size_t t = 0; t < valid_len; ++t) {
        panel_norms[t] = tokens.norm_scaling_factor(panel_start_token + t);
      }
    }
  }
  return result;
}

// Batched horizontal reduction of 4 full-width accumulator vectors into a
// single quarter-width SIMD register containing the 4 scalar minimums.
// Requires >= 16 lanes (AVX-512) so that quarter-width vectors have 4 lanes.
//
// Reduction tree algebra:
//   Step 1 (Fold in half):      h_i = Min(LowerHalf(v_i), UpperHalf(v_i))
//   Step 2 (Fold to quarter):   q_i = Min(LowerHalf(h_i), UpperHalf(h_i))
//   Step 3 (Interleave & Min):  s02 = Min(InterleaveLo/Hi(q0, q2))
//                               s13 = Min(InterleaveLo/Hi(q1, q3))
//   Step 4 (Final Packing):     res = Min(InterleaveLo(s02, s13),
//                                         InterleaveHi(s02, s13))
//
// Result: Lane indices [0, 1, 2, 3] directly hold min(v0), min(v1), min(v2),
// min(v3).
template <class D, class DH, class DQ>
HWY_INLINE hn::Vec<DQ> Reduce4VectorsMin(D d_tag, DH dh, DQ dq, hn::Vec<D> v0,
                                         hn::Vec<D> v1, hn::Vec<D> v2,
                                         hn::Vec<D> v3) {
  const hn::Vec<DH> half0 =
      hn::Min(hn::LowerHalf(dh, v0), hn::UpperHalf(dh, v0));
  const hn::Vec<DH> half1 =
      hn::Min(hn::LowerHalf(dh, v1), hn::UpperHalf(dh, v1));
  const hn::Vec<DH> half2 =
      hn::Min(hn::LowerHalf(dh, v2), hn::UpperHalf(dh, v2));
  const hn::Vec<DH> half3 =
      hn::Min(hn::LowerHalf(dh, v3), hn::UpperHalf(dh, v3));

  const hn::Vec<DQ> quarter0 =
      hn::Min(hn::LowerHalf(dq, half0), hn::UpperHalf(dq, half0));
  const hn::Vec<DQ> quarter1 =
      hn::Min(hn::LowerHalf(dq, half1), hn::UpperHalf(dq, half1));
  const hn::Vec<DQ> quarter2 =
      hn::Min(hn::LowerHalf(dq, half2), hn::UpperHalf(dq, half2));
  const hn::Vec<DQ> quarter3 =
      hn::Min(hn::LowerHalf(dq, half3), hn::UpperHalf(dq, half3));

  const hn::Vec<DQ> min_quarter_02 =
      hn::Min(hn::InterleaveLower(dq, quarter0, quarter2),
              hn::InterleaveUpper(dq, quarter0, quarter2));
  const hn::Vec<DQ> min_quarter_13 =
      hn::Min(hn::InterleaveLower(dq, quarter1, quarter3),
              hn::InterleaveUpper(dq, quarter1, quarter3));

  const hn::Vec<DQ> interleaved_lo =
      hn::InterleaveLower(dq, min_quarter_02, min_quarter_13);
  const hn::Vec<DQ> interleaved_hi =
      hn::InterleaveUpper(dq, min_quarter_02, min_quarter_13);
  return hn::Min(interleaved_lo, interleaved_hi);
}

// Unpacks up to `num_block_panels` pre-transposed 4-bit panels starting at
// `panel_block_start` into 8-bit quantized centroid values in
// `workspace.panel_buf` and copies corresponding panel norms into
// `workspace.block_norms`.
void UnpackPanelBlock(const TurboQuantChamferDataset& dataset,
                      size_t panel_block_start, size_t num_block_panels,
                      ChamferWorkspace& workspace) {
  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  const Int8 di8;
  const Uint8 du8;
  const size_t N = hn::Lanes(di8);
  const size_t points_per_panel = dataset.points_per_panel();
  const size_t panel_bytes = dataset.panel_bytes();
  const size_t full_panel_bytes = panel_bytes * 2;
  const size_t num_tile_pairs = full_panel_bytes / (2 * N);

  const uint8_t* HWY_RESTRICT all_panels = dataset.transposed_panels();
  const float* HWY_RESTRICT all_norms = dataset.panel_norms();
  const hn::Vec<Uint8> mask = hn::Set(du8, 0x0F);
  const hn::Vec<Uint8> centroids =
      hn::LoadDup128(du8, kTurboQuantCentroidsUint8.data());

  for (size_t panel_in_block = 0; panel_in_block < num_block_panels;
       ++panel_in_block) {
    const size_t panel_idx =
        workspace.panel_indices[panel_block_start + panel_in_block];
    const uint8_t* HWY_RESTRICT src = all_panels + panel_idx * panel_bytes;
    uint8_t* HWY_RESTRICT dst =
        workspace.panel_buf.data() + panel_in_block * full_panel_bytes;
    for (size_t pair_idx = 0; pair_idx < num_tile_pairs;
         ++pair_idx, src += N, dst += 2 * N) {
      const hn::Vec<Uint8> packed_indices = hn::LoadU(du8, src);
      hn::StoreU(hn::TableLookupBytes(centroids, hn::And(packed_indices, mask)),
                 du8, dst);
      hn::StoreU(
          hn::TableLookupBytes(centroids, hn::ShiftRight<4>(packed_indices)),
          du8, dst + N);
    }
    // Gather norm scaling factors for this panel.
    std::copy_n(
        all_norms + panel_idx * points_per_panel, points_per_panel,
        workspace.block_norms.begin() + panel_in_block * points_per_panel);
  }
}

// Reduces panel similarity/distance scores across SIMD lanes into per-query
// document minimums (`doc_min_row`), optionally masking out invalid lanes when
// `kMasked` is true.
template <size_t kQueriesPerBatch, bool kMasked, class ScoresAccessor>
void ReducePanelScores(ScoresAccessor scores_accessor,
                       float* HWY_RESTRICT doc_min_row, size_t query_start_idx,
                       size_t num_active_queries, size_t valid_len) {
  using Float = hn::ScalableTag<float>;
  const Float df;
  const hn::Mask<Float> mask = hn::FirstN(df, valid_len);
  const hn::Vec<Float> inf = hn::Set(df, std::numeric_limits<float>::max());

  auto get_score = [&](auto query_idx_tag) HWY_ATTR -> hn::Vec<Float> {
    const hn::Vec<Float> raw_scores = scores_accessor(query_idx_tag);
    if constexpr (kMasked) {
      return hn::IfThenElse(mask, raw_scores, inf);
    } else {
      return raw_scores;
    }
  };

  auto reduce_single_query = [&](auto query_idx_tag) HWY_ATTR {
    constexpr size_t batch_offset = decltype(query_idx_tag)::value;
    if (batch_offset < num_active_queries) {
      const float panel_min_score =
          hn::GetLane(hn::MinOfLanes(df, get_score(query_idx_tag)));
      doc_min_row[query_start_idx + batch_offset] = std::min(
          doc_min_row[query_start_idx + batch_offset], panel_min_score);
    }
  };

  if constexpr (kQueriesPerBatch >= 4) {
    if (hn::Lanes(df) == 16 && num_active_queries >= 4) {
      const hn::Half<decltype(df)> dh;
      const hn::Half<decltype(dh)> dq;
      using DQ = decltype(dq);
      const hn::Vec<DQ> batch_min_scores = Reduce4VectorsMin(
          df, dh, dq, get_score(std::integral_constant<size_t, 0>{}),
          get_score(std::integral_constant<size_t, 1>{}),
          get_score(std::integral_constant<size_t, 2>{}),
          get_score(std::integral_constant<size_t, 3>{}));
      const hn::Vec<DQ> cur_min_scores =
          hn::LoadU(dq, doc_min_row + query_start_idx);
      hn::StoreU(hn::Min(cur_min_scores, batch_min_scores), dq,
                 doc_min_row + query_start_idx);

      ForIndex<kQueriesPerBatch - 4>([&](auto tail_idx_tag) HWY_ATTR {
        constexpr size_t tail_idx = decltype(tail_idx_tag)::value;
        reduce_single_query(std::integral_constant<size_t, 4 + tail_idx>{});
      });
      return;
    }
  }

  ForIndex<kQueriesPerBatch>(reduce_single_query);
}

void ComputeMinDistances(const TurboQuantPointCloudQuery& query,
                         const TurboQuantChamferDataset& dataset,
                         ChamferWorkspace& workspace) {
  using Int8 = hn::ScalableTag<int8_t>;
  const Int8 di8;
  const size_t N = hn::Lanes(di8);
  const size_t points_per_panel = dataset.points_per_panel();
  const size_t panel_bytes = dataset.panel_bytes();
  const size_t num_queries = query.num_tokens();
  const size_t num_panels = workspace.panel_indices.size();
  if (num_queries == 0 || num_panels == 0) return;

  const size_t full_panel_bytes = panel_bytes * 2;
  const size_t total_tiles = full_panel_bytes / N;

  const DecodedQueryView decoded_query_view{
      .decoded_data = query.decoded_data(),
      .byte_sums = query.byte_sums(),
      .norm_scaling_factors = query.norm_scaling_factors(),
      .squared_norms = nullptr,
      .decoded_dimension = query.decoded_dimension(),
      .is_l2 = false,
  };

  for (size_t panel_block_start = 0; panel_block_start < num_panels;
       panel_block_start += kPanelsPerBlock) {
    const size_t num_block_panels =
        std::min(kPanelsPerBlock, num_panels - panel_block_start);
    UnpackPanelBlock(dataset, panel_block_start, num_block_panels, workspace);

    auto panel_ptr = [&](size_t panel_in_block) {
      return workspace.panel_buf.data() + panel_in_block * full_panel_bytes;
    };

    constexpr int kQueriesPerBatch = 4;
    auto score_consumer = [&](auto scores_accessor, size_t query_start_idx,
                              size_t num_active_queries,
                              size_t local_panel_base_idx) HWY_ATTR {
      const size_t global_panel_idx =
          panel_block_start + local_panel_base_idx / points_per_panel;
      const size_t doc_idx = workspace.panel_to_doc[global_panel_idx];
      const size_t valid_len = workspace.panel_valid_len[global_panel_idx];
      float* HWY_RESTRICT doc_min_row =
          workspace.doc_min_dists.data() + doc_idx * num_queries;

      if (valid_len == points_per_panel) {
        ReducePanelScores<kQueriesPerBatch, /*kMasked=*/false>(
            scores_accessor, doc_min_row, query_start_idx, num_active_queries,
            valid_len);
      } else {
        ReducePanelScores<kQueriesPerBatch, /*kMasked=*/true>(
            scores_accessor, doc_min_row, query_start_idx, num_active_queries,
            valid_len);
      }
    };
    RunQueryLoop<kQueriesPerBatch>(
        decoded_query_view, num_queries, num_block_panels, total_tiles,
        workspace.block_norms.data(), nullptr, false, panel_ptr,
        [](size_t, size_t, const QueryBatch<kQueriesPerBatch>&) {},
        score_consumer, [](size_t, const QueryBatch<kQueriesPerBatch>&) {});
  }
}

// Gathers all candidate document panels for `neighbors`, populates the panel
// mapping tables (`panel_indices`, `panel_to_doc`, `panel_valid_len`), and
// initializes `doc_min_dists` to +inf.
void PrepareCandidatePanels(const TurboQuantChamferDataset& dataset,
                            absl::Span<DistanceAndId> neighbors,
                            size_t num_query_tokens,
                            ChamferWorkspace& workspace) {
  const auto doc_panel_offsets = dataset.doc_panel_offsets();
  const auto doc_offsets = dataset.doc_offsets();

  size_t total_panels = 0;
  for (const auto& neighbor : neighbors) {
    const size_t doc_id = neighbor.id;
    total_panels += doc_panel_offsets[doc_id + 1] - doc_panel_offsets[doc_id];
  }

  const size_t points_per_panel = dataset.points_per_panel();
  workspace.PrepareForBatch(total_panels, neighbors.size(), num_query_tokens);

  size_t curr_panel = 0;
  for (size_t i = 0; i < neighbors.size(); ++i) {
    const size_t doc_id = neighbors[i].id;
    const size_t num_doc_tokens = doc_offsets[doc_id + 1] - doc_offsets[doc_id];
    const size_t panel_start = doc_panel_offsets[doc_id];
    const size_t panel_end = doc_panel_offsets[doc_id + 1];
    for (size_t panel_idx = panel_start; panel_idx < panel_end; ++panel_idx) {
      const size_t panel_in_doc = panel_idx - panel_start;
      workspace.panel_indices[curr_panel] = panel_idx;
      workspace.panel_to_doc[curr_panel] = static_cast<uint32_t>(i);
      // Safe cast: points_per_panel = SIMD_width / 4 <= 64 (even on SVE-2048),
      // so valid_len always fits in uint8_t.
      workspace.panel_valid_len[curr_panel] = static_cast<uint8_t>(std::min(
          points_per_panel, num_doc_tokens - panel_in_doc * points_per_panel));
      curr_panel++;
    }
  }
}

// Aggregates per-query minimum distances into final Chamfer distances for each
// candidate neighbor document.
void FinalizeChamferDistances(const TurboQuantChamferDataset& dataset,
                              size_t num_query_tokens,
                              const ChamferWorkspace& workspace,
                              absl::Span<DistanceAndId> neighbors) {
  const auto doc_offsets = dataset.doc_offsets();
  const float inv_num_query_tokens =
      1.0f / static_cast<float>(num_query_tokens);
  for (size_t neighbor_idx = 0; neighbor_idx < neighbors.size();
       ++neighbor_idx) {
    const size_t doc_id = neighbors[neighbor_idx].id;
    if (doc_offsets[doc_id + 1] == doc_offsets[doc_id]) {
      neighbors[neighbor_idx].distance = 0.0f;
      continue;
    }
    const float* doc_min_row =
        workspace.doc_min_dists.data() + neighbor_idx * num_query_tokens;
    const float dist_sum =
        std::accumulate(doc_min_row, doc_min_row + num_query_tokens, 0.0f);
    neighbors[neighbor_idx].distance = dist_sum * inv_num_query_tokens;
  }
}

void ChamferKernelImpl(const TurboQuantPointCloudQuery& query,
                       const TurboQuantChamferDataset& dataset,
                       absl::Span<DistanceAndId> neighbors,
                       ChamferWorkspace& workspace) {
  ABSL_DCHECK(!neighbors.empty());
  const size_t num_query_tokens = query.num_tokens();
  ABSL_DCHECK_GT(num_query_tokens, 0);

  const size_t full_panel_bytes = dataset.panel_bytes() * 2;
  workspace.panel_buf.resize(kPanelsPerBlock * full_panel_bytes);
  workspace.block_norms.resize(kPanelsPerBlock * dataset.points_per_panel());

  // Process neighbors in bounded batches to keep `doc_min_dists` and panel
  // metadata small.
  constexpr size_t kMaxNeighborsPerBatch = 256;
  for (size_t offset = 0; offset < neighbors.size();
       offset += kMaxNeighborsPerBatch) {
    const size_t batch_size =
        std::min(kMaxNeighborsPerBatch, neighbors.size() - offset);
    absl::Span<DistanceAndId> batch_neighbors =
        neighbors.subspan(offset, batch_size);
    PrepareCandidatePanels(dataset, batch_neighbors, num_query_tokens,
                           workspace);
    ComputeMinDistances(query, dataset, workspace);
    FinalizeChamferDistances(dataset, num_query_tokens, workspace,
                             batch_neighbors);
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {

HWY_EXPORT(PrepareTransposedPanelsImpl);
HWY_EXPORT(ChamferKernelImpl);

TurboQuantChamferDataset::TurboQuantChamferDataset(
    const TurboQuantDatasetSoA& tokens, std::vector<size_t> doc_offsets)
    : dimensionality_(tokens.dimensionality()),
      size_(doc_offsets.empty()
                ? 0
                : static_cast<GingerNodeId>(doc_offsets.size() - 1)),
      doc_offsets_(std::move(doc_offsets)) {
  ABSL_CHECK(tokens.metric() ==
             ::ginger::TurboQuantDistanceMetric::kDotProductDistance);
  ABSL_CHECK(!tokens.has_3bit_prefix());
  if (!doc_offsets_.empty()) {
    ABSL_CHECK_EQ(doc_offsets_.front(), 0);
    ABSL_CHECK_EQ(doc_offsets_.back(), tokens.num_datapoints());
  }
  ABSL_DCHECK(std::is_sorted(doc_offsets_.begin(), doc_offsets_.end()));
  if (size_ > 0) {
    TransposedPanelData panels =
        HWY_DYNAMIC_DISPATCH(PrepareTransposedPanelsImpl)(tokens, doc_offsets_);
    panel_bytes_ = panels.panel_bytes;
    points_per_panel_ = panels.points_per_panel;
    doc_panel_offsets_ = std::move(panels.doc_panel_offsets);
    transposed_panels_ = std::move(panels.transposed_panels);
    panel_norms_ = std::move(panels.panel_norms);
  }
}

void TurboQuantChamferDataset::ChamferDistance(
    const TurboQuantPointCloudQuery& query,
    absl::Span<DistanceAndId> neighbors) const {
  if (neighbors.empty()) return;
  if (query.num_tokens() == 0) {
    for (DistanceAndId& neighbor : neighbors) neighbor.distance = 0.0f;
    return;
  }
  thread_local ChamferWorkspace workspace;
  HWY_DYNAMIC_DISPATCH(ChamferKernelImpl)(query, *this, neighbors, workspace);
}

}  // namespace ginger
#endif
