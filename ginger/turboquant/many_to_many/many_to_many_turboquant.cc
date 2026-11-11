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

// Many-to-many approximate distance computation using TurboQuant 4-bit
// quantization. Supports two database representations:
//
//   1. On-the-fly decode:  raw nibble-packed database → panels decoded per
//      L3 block via DecodeDbPanel.
//   2. Pretransposed:      panels decoded once upfront into a
//      PreTransposedTurboQuantDataset → sliced by pointer per L3 block.
//
// Both paths share a single cache-tiled loop (ManyToManyLoop) that tiles in
// three levels:
//
//   L3 block  (~20 MB)  — DB panels resident across all query batches.
//   L2 query  (~192 KB) — decoded query vectors.
//   L2 DB sub-block (~192 KB) — a chunk of the L3 block sized so that the
//                     score buffer (kMq × db_sub_block floats) stays
//                     L2-resident.
#include "ginger/turboquant/many_to_many/many_to_many_turboquant.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE

// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/many_to_many/many_to_many_turboquant.cc"
// clang-format on

#include "hwy/cache_control.h"   // IWYU pragma: keep
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/many_to_many/decode_db-inl.h"
#include "ginger/turboquant/many_to_many/decode_queries-inl.h"
#include "ginger/turboquant/many_to_many/gemm_loop-inl.h"

namespace ginger {

namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

// Number of queries processed per micro-kernel invocation.
static constexpr size_t kMq = 6;

// ---- Cache-tiling parameter helpers ----------------------------------------
//
// The three-level tiling scheme is laid out as:
//
//   L3 outer loop (db_batch_l3 datapoints) — panels decoded/pinned once,
//                                     reused for all query batches.  ~20 MB.
//   L2 query loop (query_batch queries) — decoded queries.  ~192 KB budget.
//   L2 DB sub-block loop (db_sub_batch) — score_buf (kMq × db_sub_batch
//                                     floats) stays L2-resident.  ~192 KB.
// The budgets are chosen to stay moderately below typical cache sizes starting
// with icelake.
//
// All batch sizes are rounded to a multiple of points_per_panel (SIMD lane
// count for i32, typically 4–16) so the panel/scoring loops have no remainders.

// Max DB datapoints whose decoded panels fit in L3. Panels are decoded once
// per L3 block and reused across all query batches in Loop 2.
// NOMUTANTS -- changing these values does not impact correctness.
size_t NumDbPointsInL3Cache(size_t panel_bytes, size_t points_per_panel) {
  constexpr size_t kL3Budget = 20 * 1024 * 1024;
  constexpr size_t kMaxDbBatch = 65536;
  constexpr size_t kMinDbBatch = 32;
  const size_t panels_in_budget = kL3Budget / panel_bytes;
  size_t db_batch = std::min(kMaxDbBatch, panels_in_budget * points_per_panel);
  return std::max(kMinDbBatch,
                  (db_batch / points_per_panel) * points_per_panel);
}

// Max DB datapoints per sub-block such that score_buf (kMq × db_sub_batch
// floats) stays L2-resident. Each sub-block is scored against the current
// query batch before advancing to the next sub-block.
// NOMUTANTS -- changing these values does not impact correctness.
size_t NumDbPointsInL2Cache(size_t panel_bytes, size_t points_per_panel) {
  constexpr size_t kL2Budget = 192 * 1024;
  constexpr size_t kMaxDbBatch = 1024;
  constexpr size_t kMinDbBatch = 32;
  const size_t panels_in_budget = kL2Budget / panel_bytes;
  size_t db_batch = std::min(kMaxDbBatch, panels_in_budget * points_per_panel);
  return std::max(kMinDbBatch,
                  (db_batch / points_per_panel) * points_per_panel);
}

// Max queries per L2 batch. Decoded query vectors are int8, one per
// decoded_dim bytes. Rounded down to kMq so the micro-kernel inner loop
// has no partial-batch remainder.
// NOMUTANTS -- changing these values does not impact correctness.
size_t NumQueriesInL2Cache(size_t decoded_dim) {
  constexpr size_t kL2Budget = 192 * 1024;
  constexpr size_t kMaxQueryBatch = 4096;
  size_t qb = (kL2Budget / decoded_dim / kMq) * kMq;
  return std::min(kMaxQueryBatch, std::max(kMq, qb));
}

// ---- Micro-kernel scoring ---------------------------------------------------

// Read-only view of panel data and norm scalings for one L3-sized DB block
// (or an L2 sub-block). Pointers reference either local decode buffers
// (on-the-fly path) or the pretransposed dataset.
// See the README for the panel memory layout.
struct L3BlockView {
  const uint8_t* panels;
  const float* norm_scaling_factors;
  const float* sqr_norms;  // nullptr when metric is dot product.
};

// Scores already-decoded queries against already-materialized panels and
// streams results via the callback. Operates on one L2 sub-block at a time.
// See the README for the panel memory layout.
//
// Parameters:
//   dq               – view of decoded int8 query vectors and metadata.
//   num_queries      – number of decoded queries in this batch.
//   db_sub_block     – view of panel data and norm scalings for this sub-block.
//   num_panels       – number of panels in this sub-block.
//   num_db_points    – number of DB points in this sub-block.
//   total_tiles      – SIMD tiles per panel (dimension loop count).
//   panel_bytes      – byte size of one decoded panel.
//   db_start         – global DB index of the first point in this sub-block.
//   q_offset         – offset added to query indices in the callback.
//   score_buf        – temporary buffer to store scores before callback.
//   score_stride     – stride between query rows in score_buf.
//   callback         – invoked per kMq-query micro-batch with scored results.
void ScoreQueriesAgainstPanels(const DecodedQueryView& dq, size_t num_queries,
                               const L3BlockView& db_sub_block,
                               size_t num_panels, size_t num_db_points,
                               size_t total_tiles, size_t panel_bytes,
                               size_t db_start, size_t q_offset,
                               float* HWY_RESTRICT score_buf,
                               size_t score_stride,
                               ManyToManyCallback callback) {
  using Float = hn::ScalableTag<float>;
  const Float df;

  auto panel_ptr = [panels = db_sub_block.panels,
                    panel_bytes](size_t p) -> const uint8_t* {
    return panels + p * panel_bytes;
  };

  RunQueryLoop<kMq>(
      dq, num_queries, num_panels, total_tiles,
      db_sub_block.norm_scaling_factors, db_sub_block.sqr_norms,
      /*lower_triangle=*/false, panel_ptr,
      // init_batch: no-op (QueryBatch already populated by DecodedQueryView
      // overload)
      [](size_t /*q_start*/, size_t /*num_active*/,
         const QueryBatch<kMq>& /*qb*/) {},
      // score_consumer: store to score_buf
      [&](auto scores_accessor, size_t /*q_start*/, size_t num_active,
          size_t panel_base) HWY_ATTR {
        ForIndex<kMq>([&](auto q_idx) HWY_ATTR {
          constexpr size_t q = decltype(q_idx)::value;
          if (q < num_active) {
            hn::StoreU(scores_accessor(q_idx), df,
                       score_buf + q * score_stride + panel_base);
          }
        });
      },
      // reduce_batch: invoke callback for actual queries
      [&](size_t q_start, const QueryBatch<kMq>& qb) {
        // TODO: reduce callback overhead by calling for multiple
        // queries at once.
        for (size_t q = 0; q < qb.num_active_queries; ++q) {
          callback(
              q_offset + q_start + q,
              absl::MakeConstSpan(score_buf + q * score_stride, num_db_points),
              db_start);
        }
      });
}

// ---- Unified cache-tiled main loop -----------------------------------------

// Main loop shared by both data paths. The caller supplies a
// `prepare_l3_block(db_L3, nb_L3) -> L3BlockView` callable that materializes
// panels and norm scalings for the current L3 block — either by decoding into
// local buffers or by returning pointers into the pretransposed dataset.
//
// Query subsetting behavior:
//   - `query_subset`: Specifies the active subset of queries to process, which
//     may be either a contiguous range of indices or an explicit gathered
//     subspan.
//   - Callback indexing: The `query_idx` passed to `callback` is always in the
//     range `[0, query_subset.size())`, representing the offset within the
//     active subset rather than the absolute query index in `queries`.
//
// Loop structure (5 nested levels):
//   1. L3 DB outer      — iterate DB in nc-point blocks, call prepare_l3_block
//   2. L2 query         — iterate queries in mc-point batches, decode once
//   3. L2 DB sub-block  — iterate L3 block in nc_sub-point sub-blocks
//   4-5. Micro-kernel   — kMq queries × 2-panel unroll × tiles  (VPDPBUSD)
using PrepareL3Block =
    absl::FunctionRef<L3BlockView(size_t db_L3, size_t nb_L3)>;

void ManyToManyLoop(const TurboQuantDatasetSoA& queries,
                    IndexSubset query_subset, size_t num_db,
                    size_t points_per_panel, size_t panel_bytes,
                    size_t total_tiles, bool is_l2,
                    PrepareL3Block prepare_l3_block,
                    ManyToManyCallback callback,
                    ManyToManyTurboQuantWorkspace& workspace) {
  const size_t num_queries = query_subset.size();
  const size_t decoded_dim = DecodedQueryStride(queries);

  // Tiling parameters (see helpers at top of file).
  size_t db_batch_l3 = NumDbPointsInL3Cache(panel_bytes, points_per_panel);
  size_t db_sub_batch = NumDbPointsInL2Cache(panel_bytes, points_per_panel);
  size_t query_batch = NumQueriesInL2Cache(decoded_dim);
  const size_t actual_db_batch_l3 = std::min(db_batch_l3, num_db);

  // Query decode buffers — filled once per L2 query batch (Loop 2), reused
  // across all L2 DB sub-blocks (Loop 3) within the same L3 block.
  // NOMUTANTS -- exact buffer size is correct by construction.
  const size_t q_decoded_size = query_batch * decoded_dim;
  if (workspace.q_decoded.size() < q_decoded_size) {
    workspace.q_decoded.resize(q_decoded_size);
  }
  if (workspace.q_byte_sums.size() < query_batch) {
    workspace.q_byte_sums.resize(query_batch);
  }
  if (workspace.q_norms.size() < query_batch) {
    workspace.q_norms.resize(query_batch);
  }
  if (is_l2 && workspace.q_sqr_norms.size() < query_batch) {
    workspace.q_sqr_norms.resize(query_batch);
  }

  // Score buffer: kMq rows × score_stride columns. Sized for the max DB
  // sub-batch and reused across all L2 DB sub-blocks (Loop 3) to avoid
  // inner-loop heap allocations. NOTE: `score_buf` rows must be sized to a
  // multiple of `hn::Lanes(i8) / 4`, even for partial panels. The epilogue uses
  // unaligned `StoreU` operations that write full vector widths, overwriting
  // trailing elements with garbage. NOMUTANTS -- exact buffer size is correct
  // by construction.
  const size_t score_stride =
      hwy::RoundUpTo(db_sub_batch, hn::Lanes(hn::ScalableTag<int32_t>()));
  const size_t score_buf_size = kMq * score_stride;  // NOMUTANTS
  if (workspace.score_buf.size() < score_buf_size) {
    workspace.score_buf.resize(score_buf_size);
  }

  // --- Loop 1: L3-tiled DB outer loop ---
  for (size_t db_L3 = 0; db_L3 < num_db; db_L3 += actual_db_batch_l3) {
    const size_t nb_L3 = std::min(actual_db_batch_l3, num_db - db_L3);
    ABSL_DCHECK_LE(db_L3 + nb_L3,
              num_db);  // Loop invariant: block never extends past the DB end.

    const L3BlockView block = prepare_l3_block(db_L3, nb_L3);

    // --- Loop 2: L2-tiled query loop ---
    for (size_t q_start = 0; q_start < num_queries; q_start += query_batch) {
      const size_t qb = std::min(query_batch, num_queries - q_start);

      // Decode queries for this batch.
      const DecodedQueryView dq = DecodeQueriesAndMetadata(
          queries, query_subset.subspan(q_start, qb),
          absl::MakeSpan(workspace.q_decoded).first(qb * decoded_dim),
          absl::MakeSpan(workspace.q_byte_sums).first(qb),
          absl::MakeSpan(workspace.q_norms).first(qb),
          absl::MakeSpan(workspace.q_sqr_norms).first(is_l2 ? qb : 0));

      // --- Loop 3: L2-tiled DB sub-block loop ---

      for (size_t db_sub = 0; db_sub < nb_L3; db_sub += db_sub_batch) {
        const size_t nb_sub = std::min(db_sub_batch, nb_L3 - db_sub);
        const size_t sub_panel_start = db_sub / points_per_panel;
        const size_t num_sub_panels = hwy::DivCeil(nb_sub, points_per_panel);

        const L3BlockView sub_block = {
            .panels = block.panels + sub_panel_start * panel_bytes,
            .norm_scaling_factors = block.norm_scaling_factors + db_sub,
            .sqr_norms = is_l2 ? block.sqr_norms + db_sub : nullptr};

        ScoreQueriesAgainstPanels(
            dq, qb, sub_block, num_sub_panels, nb_sub, total_tiles, panel_bytes,
            db_L3 + db_sub, /*q_offset=*/q_start, workspace.score_buf.data(),
            score_stride, callback);
      }
    }
  }
}

// ---- Impl wrappers (one per data path) ------------------------------------

// On-the-fly decode path: decodes nibble-packed DB panels into local buffers
// for each L3 block. Supports gathered db_indices.
void ManyToManyTurboQuantImpl(const TurboQuantDatasetSoA& queries,
                              const TurboQuantDatasetSoA& database,
                              IndexSubset query_subset, IndexSubset db_subset,
                              ManyToManyCallback callback,
                              ManyToManyTurboQuantWorkspace& workspace) {
  const size_t points_per_panel = DbPanelNumPoints();
  const size_t panel_bytes = DbPanelBytes(database);
  const size_t N = hn::Lanes(hn::ScalableTag<int8_t>());
  const size_t total_tiles = panel_bytes / N;

  const size_t num_db = db_subset.size();
  const bool is_l2 =
      queries.metric() == TurboQuantDistanceMetric::kSquaredL2Distance;

  // Pre-allocate decode buffers sized for one L3 block.
  const size_t actual_nc =
      std::min(NumDbPointsInL3Cache(panel_bytes, points_per_panel), num_db);
  const size_t max_panels = hwy::DivCeil(actual_nc, points_per_panel);
  const size_t db_panels_size = max_panels * panel_bytes;  // NOMUTANTS
  if (workspace.db_panels.size() < db_panels_size) {
    workspace.db_panels.resize(db_panels_size);
  }
  // NOMUTANTS -- points_per_panel padding ensures full-width SIMD loads on the
  // last panel never read past the buffer or touch uninitialized memory.
  const size_t norms_size = actual_nc + points_per_panel;
  if (workspace.db_norm_scaling_factors.size() < norms_size) {
    workspace.db_norm_scaling_factors.resize(norms_size);
  }
  if (is_l2 && workspace.db_sqr_norms.size() < norms_size) {
    workspace.db_sqr_norms.resize(norms_size);
  }

  ManyToManyLoop(
      queries, query_subset, num_db, points_per_panel, panel_bytes, total_tiles,
      is_l2,
      /*prepare_l3_block=*/
      [&](size_t db_L3, size_t nb_L3) -> L3BlockView {
        const size_t np = hwy::DivCeil(nb_L3, points_per_panel);
        // Fill norm scalings for the `nb_L3` active points in this block.
        // Trailing lanes `[nb_L3, np * points_per_panel)` of a partial last
        // panel are not overwritten and may hold stale values from a previous
        // L3 block or call. This is harmless: scores computed in those lanes
        // are written past `num_db_points` in `score_buf` and sliced off
        // before `callback`.
        for (size_t di = 0; di < nb_L3; ++di) {
          const size_t src = db_subset[db_L3 + di];
          workspace.db_norm_scaling_factors[di] =
              database.norm_scaling_factor(src);
          if (is_l2) {
            workspace.db_sqr_norms[di] = database.unquantized_squared_norm(src);
          }
        }
        // Decode panels.
        for (size_t p = 0; p < np; ++p) {
          const size_t start = p * points_per_panel;
          // NOMUTANTS -- off-by-one in size only over-requests decoding for a
          // padding lane whose score is sliced off before `callback` (which
          // covers only `nb_L3` valid points).
          const size_t size = std::min(points_per_panel, nb_L3 - start);
          absl::Span<uint8_t> panel_span(
              workspace.db_panels.data() + p * panel_bytes, panel_bytes);
          DecodeDbPanel(database, db_subset.subspan(db_L3 + start, size),
                        panel_span);
        }
        return {
            .panels = workspace.db_panels.data(),
            .norm_scaling_factors = workspace.db_norm_scaling_factors.data(),
            .sqr_norms = is_l2 ? workspace.db_sqr_norms.data() : nullptr};
      },
      callback, workspace);
}

// Pretransposed path: database panels are already dequantized and
// transposed in memory within `database`. The `prepare_l3_block` lambda
// passed to `ManyToManyLoop` only sets up pointers to the pretransposed data.
void ManyToManyPreTransposedImpl(const TurboQuantDatasetSoA& queries,
                                 const PreTransposedTurboQuantDataset& database,
                                 IndexSubset query_subset,
                                 ManyToManyCallback callback,
                                 ManyToManyTurboQuantWorkspace& workspace) {
  const bool is_l2 =
      queries.metric() == TurboQuantDistanceMetric::kSquaredL2Distance;
  ABSL_CHECK(!is_l2 || !database.sqr_norms.empty());

  ManyToManyLoop(
      queries, query_subset, database.num_points, database.points_per_panel,
      database.panel_bytes, database.total_tiles, is_l2,
      /*prepare_l3_block=*/
      [&](size_t db_L3, size_t /*nb_L3*/) -> L3BlockView {
        const size_t start = db_L3 / database.points_per_panel;
        return {
            .panels = database.panel_data.data() + start * database.panel_bytes,
            .norm_scaling_factors =
                database.norm_scaling_factors.data() + db_L3,
            .sqr_norms = is_l2 ? database.sqr_norms.data() + db_L3 : nullptr};
      },
      callback, workspace);
}
}  // namespace
}  // namespace HWY_NAMESPACE

}  // namespace ginger

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace ginger {

HWY_EXPORT(ManyToManyTurboQuantImpl);
HWY_EXPORT(ManyToManyPreTransposedImpl);

void ManyToManyTurboQuant(const TurboQuantDatasetSoA& queries,
                          const TurboQuantDatasetSoA& database,
                          IndexSubset query_subset, IndexSubset db_subset,
                          ManyToManyCallback callback,
                          ManyToManyTurboQuantWorkspace& workspace) {
  if (query_subset.empty() || db_subset.empty()) return;
  ABSL_CHECK_EQ(queries.dimensionality(), database.dimensionality());
  ABSL_CHECK(queries.metric() == database.metric());
  HWY_DYNAMIC_DISPATCH(ManyToManyTurboQuantImpl)(
      queries, database, query_subset, db_subset, callback, workspace);
}

void ManyToManyTurboQuant(const TurboQuantDatasetSoA& queries,
                          const TurboQuantDatasetSoA& database,
                          IndexSubset query_subset, IndexSubset db_subset,
                          ManyToManyCallback callback) {
  ManyToManyTurboQuantWorkspace workspace;
  ManyToManyTurboQuant(queries, database, query_subset, db_subset, callback,
                       workspace);
}

void ManyToManyTurboQuant(const TurboQuantDatasetSoA& queries,
                          const PreTransposedTurboQuantDataset& database,
                          IndexSubset query_subset, ManyToManyCallback callback,
                          ManyToManyTurboQuantWorkspace& workspace) {
  if (query_subset.empty() || database.num_points == 0) return;
  ABSL_CHECK_EQ(queries.dimensionality(), database.dimensionality);
  if (queries.metric() == TurboQuantDistanceMetric::kSquaredL2Distance) {
    ABSL_CHECK(!database.sqr_norms.empty())
        << "Pretransposed database must have sqr_norms for L2 metric. "
           "Build with a kSquaredL2Distance quantizer.";
  }
  HWY_DYNAMIC_DISPATCH(ManyToManyPreTransposedImpl)(
      queries, database, query_subset, callback, workspace);
}

void ManyToManyTurboQuant(const TurboQuantDatasetSoA& queries,
                          const PreTransposedTurboQuantDataset& database,
                          IndexSubset query_subset,
                          ManyToManyCallback callback) {
  ManyToManyTurboQuantWorkspace workspace;
  ManyToManyTurboQuant(queries, database, query_subset, callback, workspace);
}

}  // namespace ginger

#endif  // HWY_ONCE
