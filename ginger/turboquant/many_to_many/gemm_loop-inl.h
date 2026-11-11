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

// Shared SIMD primitives for TurboQuant GEMM loops.
//
// Provides the internals of a GEMM implementation as reusable building blocks.
// Kernels such as ManyToManyTurboQuant or PiPNN components like ProcessLeaf,
// or AssignLeaders can compose the primitives and fuse/hook in their own logic.
//
// Components:
//   QueryBatch<kQueriesPerBatch>        – lightweight view of one query batch.
//   DecodedQueryView                    – non-owning view of decoded query
//                                         arrays.
//   PrepareQueryBatch<kQueriesPerBatch> – fills a QueryBatch from a
//                                         DecodedQueryView.
//   RunPanelLoop<kQueriesPerBatch>      – executes the panel loop for a batch
//                                         of kQueriesPerBatch queries using an
//                                         unrolled micro-kernel.
//   ComputePanelScores                  – bias correction + norm scaling
//                                         epilogue.
//   RunQueryLoop<kQueriesPerBatch>      – full outer query-batch loop
//                                         orchestration.
// Notation & Terminology:
//   - Database Panel:
//     A block of database vectors packed together. To maximize SIMD efficiency,
//     the vectors are transposed and interleaved. A panel contains exactly
//     `points_per_panel` points, which corresponds to the number of lanes in a
//     float SIMD vector: `NF = hn::Lanes(ScalableTag<float>())` (e.g., 8 points
//     for AVX2, 16 points for AVX-512).
//   - Query Batch:
//     A group of `kQueriesPerBatch` queries processed in parallel to amortize
//     the cost of loading database panel data into registers.
//   - kQueriesPerBatch:
//     The unrolling factor for the query loop. It determines the number of
//     queries processed concurrently by the accumulator micro-kernel.
//   - Tile:
//     A block of database dimensions within a panel. `tiles_per_vector`
//     is the number of SIMD tiles per database vector along the dimension loop.
//
// Detailed GEMM execution flow as implemented by RunQueryLoop:
// 1. Outer Query Loop:
//    Iterate over queries in blocks of `kQueriesPerBatch`. For each block:
//    a. Prepare QueryBatch:
//       Call `PrepareQueryBatch` to populate a `QueryBatch` with pointers to
//       the decoded query data and metadata (norm scaling, squared norms, byte
//       sums). If the remaining queries are fewer than `kQueriesPerBatch`, the
//       batch is padded by aliasing the last valid query and zeroing out its
//       metadata.
//    b. Precompute Bias Correction:
//       Compute the bias correction vector for each query based on its byte
//       sums (adjusting for uint8 database points vs. int8 queries).
// 2. Inner Panel Loop (RunPanelLoop):
//    Iterate over database panels (optionally starting at `panel_start` for
//    lower-triangle computations). The loop is unrolled by 4, 2, and 1 panels
//    to maximize pipelining. For each panel block:
//    a. MicroKernelAccumulate:
//       Compute dot products between the `kQueriesPerBatch` queries and the
//       `points_per_panel` database points in each panel via
//       Multiply-Accumulate. This updates `kQueriesPerBatch` accumulator
//       registers per panel.
//    b. Epilogue (panel_epilogue_fn):
//       For each panel in the block, load its database norm scaling factors and
//       squared norms (if using L2 distance). Then, for each query in the
//       batch:
//       i. Call `ComputePanelScores` to:
//          - Subtract the bias correction from the raw accumulator.
//          - Convert the int32 result to float.
//          - Apply query norm scaling and database norm scaling.
//          - If computing L2 distance: combine with query and database squared
//            norms using: `||a-b||² = ||a||² + ||b||² - 2·dot(a,b)`.
//       ii. Call user-provided callable `panel_scores_consumer_fn` with an
//       accessor to the VEC_GROUP of float score vectors for all
//       `kQueriesPerBatch` queries, along with `query_start_idx`,
//       `num_active_queries`, and the panel's base index (`panel_idx *
//       points_per_panel`).
// 3. Batch Reduction:
//    Call `reduce_batch_fn` at the end of the batch to consume or reduce
//    the accumulated query results (e.g. updating top-K lists across lanes).

// Include after foreach_target.h, highway.h. The toggle guard handles
// per-target re-inclusion.

#ifndef GINGER_TURBOQUANT_MANY_TO_MANY_GEMM_LOOP_INL_H_
#define GINGER_TURBOQUANT_MANY_TO_MANY_GEMM_LOOP_INL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/log/absl_check.h"
#include "ginger/turboquant/many_to_many/decode_queries-inl.h"

namespace ginger {

// QueryBatch: bundles query data pointers and metadata for a kernel batch.
// Inactive query slots (when num_active_queries < kQueriesPerBatch) should
// alias the last valid query pointer and zero the scalar fields, so that the
// micro-kernel reads valid memory but produces zeroed-out scores.
template <size_t kQueriesPerBatch>
struct QueryBatch {
  const int8_t* query_ptrs[kQueriesPerBatch];
  float norm_scaling_factors[kQueriesPerBatch];
  float squared_norms[kQueriesPerBatch];
  int32_t byte_sums[kQueriesPerBatch];
  size_t num_active_queries;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_GEMM_LOOP_INL_H_

// clang-format off
#if defined(GINGER_TURBOQUANT_MANY_TO_MANY_GEMM_LOOP_TOGGLE) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef GINGER_TURBOQUANT_MANY_TO_MANY_GEMM_LOOP_TOGGLE
#undef GINGER_TURBOQUANT_MANY_TO_MANY_GEMM_LOOP_TOGGLE
#else
#define GINGER_TURBOQUANT_MANY_TO_MANY_GEMM_LOOP_TOGGLE
#endif
// clang-format on

#include "ginger/turboquant/many_to_many/accumulate-inl.h"
#include "ginger/turboquant/many_to_many/vec_group-inl.h"
#include "hwy/highway.h"

namespace ginger {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using Float = hn::ScalableTag<float>;
using Int = hn::ScalableTag<int32_t>;

namespace internal {

// Fills a QueryBatch from a DecodedQueryView.
//
// Parameters:
//   decoded_view: View of the pre-decoded query data arrays.
//   query_offset: Index of the first query in this batch, relative
//     to the start of the decoded_view arrays.
//   num_active_queries: Number of valid queries in this batch (≤
//     kQueriesPerBatch).
//   query_batch: Output QueryBatch to be populated.
template <size_t kQueriesPerBatch>
HWY_INLINE void PrepareQueryBatch(const DecodedQueryView& decoded_view,
                                  size_t query_offset,
                                  size_t num_active_queries,
                                  QueryBatch<kQueriesPerBatch>& query_batch) {
  if (num_active_queries == 0) return;
  ABSL_QCHECK(!decoded_view.is_l2 || decoded_view.squared_norms != nullptr);
  query_batch.num_active_queries = num_active_queries;
  ForIndex<kQueriesPerBatch>([&](auto query_idx) HWY_ATTR {
    constexpr size_t q = decltype(query_idx)::value;
    if (q < num_active_queries) {
      query_batch.query_ptrs[q] =
          decoded_view.decoded_data +
          (query_offset + q) * decoded_view.decoded_dimension;
      query_batch.norm_scaling_factors[q] =
          decoded_view.norm_scaling_factors[query_offset + q];
      query_batch.squared_norms[q] =
          decoded_view.is_l2 ? decoded_view.squared_norms[query_offset + q]
                             : 0.0f;
      query_batch.byte_sums[q] = decoded_view.byte_sums[query_offset + q];
    } else {
      // Pad: alias last valid pointer, zero scalars so scores are inert.
      query_batch.query_ptrs[q] =
          query_batch.query_ptrs[num_active_queries - 1];
      query_batch.norm_scaling_factors[q] = 0.0f;
      query_batch.squared_norms[q] = 0.0f;
      query_batch.byte_sums[q] = 0;
    }
  });
}

// Runs the GEMM inner loop, calling `panel_epilogue_fn`
// after each panel's accumulators are ready. Unrolled by 4 panels for
// performance.
//
// Parameters:
//   query_ptrs: Array of kQueriesPerBatch decoded int8 query pointers.
//   num_panels: Total number of database panels.
//   tiles_per_vector: SIMD tiles per panel (dimension loop count).
//   panel_start: First panel to process (0 for flat, or
//     query_start_idx / points_per_panel for lower-triangle).
//   panel_pointer_fn: Callable `(size_t p) -> const uint8_t*`.
//   panel_epilogue_fn: Callable `(auto acc_accessor, size_t panel_idx)`.
template <size_t kQueriesPerBatch, typename PanelPtrFn,
          typename PanelEpilogueFn>
HWY_INLINE void RunPanelLoop(const int8_t* HWY_RESTRICT const* query_ptrs,
                             size_t num_panels, size_t tiles_per_vector,
                             size_t panel_start, PanelPtrFn panel_pointer_fn,
                             PanelEpilogueFn panel_epilogue_fn) {
  hn::ScalableTag<uint8_t> du8;
  hn::ScalableTag<int8_t> di8;
  Int di32;
  const size_t N = hn::Lanes(di8);
  VEC_GROUP_DEFINE(di32, a0);
  VEC_GROUP_DEFINE(di32, a1);
  VEC_GROUP_DEFINE(di32, a2);
  VEC_GROUP_DEFINE(di32, a3);

  size_t p = panel_start;
  for (; p + 4 <= num_panels; p += 4) {
    const uint8_t* panel_blocks[] = {
        panel_pointer_fn(p), panel_pointer_fn(p + 1), panel_pointer_fn(p + 2),
        panel_pointer_fn(p + 3)};
    MicroKernelAccumulate<kQueriesPerBatch>(
        du8, di8, di32, query_ptrs, panel_blocks, tiles_per_vector, N,
        VEC_GROUP_ACCESSOR(a0), VEC_GROUP_ACCESSOR(a1), VEC_GROUP_ACCESSOR(a2),
        VEC_GROUP_ACCESSOR(a3));
    panel_epilogue_fn(VEC_GROUP_ACCESSOR(a0), p);
    panel_epilogue_fn(VEC_GROUP_ACCESSOR(a1), p + 1);
    panel_epilogue_fn(VEC_GROUP_ACCESSOR(a2), p + 2);
    panel_epilogue_fn(VEC_GROUP_ACCESSOR(a3), p + 3);
  }
  for (; p + 2 <= num_panels; p += 2) {
    const uint8_t* panel_blocks[] = {panel_pointer_fn(p),
                                     panel_pointer_fn(p + 1)};
    MicroKernelAccumulate<kQueriesPerBatch>(
        du8, di8, di32, query_ptrs, panel_blocks, tiles_per_vector, N,
        VEC_GROUP_ACCESSOR(a0), VEC_GROUP_ACCESSOR(a1));
    panel_epilogue_fn(VEC_GROUP_ACCESSOR(a0), p);
    panel_epilogue_fn(VEC_GROUP_ACCESSOR(a1), p + 1);
  }
  for (; p < num_panels; ++p) {
    const uint8_t* panel_blocks[] = {panel_pointer_fn(p)};
    MicroKernelAccumulate<kQueriesPerBatch>(du8, di8, di32, query_ptrs,
                                            panel_blocks, tiles_per_vector, N,
                                            VEC_GROUP_ACCESSOR(a0));
    panel_epilogue_fn(VEC_GROUP_ACCESSOR(a0), p);
  }
}

// Resolves raw int32 GEMM accumulators into float scores for all
// kQueriesPerBatch queries against one panel.
template <size_t kQueriesPerBatch, typename AccAccessor, typename BiasAccessor,
          typename ScoresAccessor>
HWY_INLINE void ComputePanelScores(
    AccAccessor acc_accessor, BiasAccessor bias_accessor,
    const QueryBatch<kQueriesPerBatch>& query_batch,
    const hn::Vec<Float>& negated_db_norm_scaling,
    const hn::Vec<Float>& db_sqr_norms, bool is_l2,
    ScoresAccessor scores_accessor) {
  Float df;
  ForIndex<kQueriesPerBatch>([&](auto query_idx) HWY_ATTR {
    constexpr size_t q = decltype(query_idx)::value;
    // Step 1: Bias correction in int32 (uint8→int8 offset), precomputed.
    const hn::Vec<Int> corrected_accumulator =
        hn::Sub(acc_accessor(query_idx), bias_accessor(query_idx));
    // Step 2: Convert to float and scale by database and query norms.
    const hn::Vec<Float> scaled_dot_product =
        hn::Mul(hn::Mul(hn::ConvertTo(df, corrected_accumulator),
                        negated_db_norm_scaling),
                hn::Set(df, query_batch.norm_scaling_factors[q]));
    // Step 3: Apply distance metric formulation.
    if (is_l2) {  // ||a-b||² = ||a||² + ||b||² - 2·dot(a,b)
      scores_accessor(query_idx) = hn::Add(
          hn::MulAdd(scaled_dot_product, hn::Set(df, 2.0f), db_sqr_norms),
          hn::Set(df, query_batch.squared_norms[q]));
    } else {
      scores_accessor(query_idx) = scaled_dot_product;
    }
  });
}

}  // namespace internal

// Drives the query-batch loop: iterates in kQueriesPerBatch steps, prepares
// each query batch (QueryBatch + bias correction), runs the panel-unrolled
// GEMM with ComputePanelScores, and calls kernel-provided callbacks for
// initialization, score consumption, and reduction.
//
// Parameters:
//
//   num_queries       - Total number of queries to process.
//
//   is_l2             - If true, computes L2 squared distance; if false,
//                       computes dot product.
//
//   num_panels        - Total number of database panels.
//
//   tiles_per_vector  - Number of SIMD dimension tiles per database vector.
//
//   db_norm_scaling_factors - Array of database norm scaling factors (floats).
//
//   db_sqr_norms      - Array of database squared norms (floats), used only
//                       when is_l2 is true.
//
//   lower_triangle    - If true, limits panel loop to lower-triangle
//                       (query_start_idx / NF).
//
//   panel_pointer_fn  - Callable `(size_t p) -> const uint8_t*` returning the
//                        pointer to the p-th panel.
//
//   init_batch_fn     - Callback called before scoring a batch.
//
//   panel_scores_consumer_fn - Callback called to consume a batch of scores for
//                        all active queries against a panel.
//
//   reduce_batch_fn   - Callback called after scoring a batch.
//
// Callback signatures:
//
//   init_batch_fn(size_t query_start_idx, size_t num_active_queries,
//                 QueryBatch<kQueriesPerBatch>& query_batch)
//     Called once per batch, before the panel loop. Use to populate
//     the QueryBatch (e.g., on-the-fly decode) and initialize per-batch state
//     (e.g., top-K registers).
//
//   panel_scores_consumer_fn(auto scores_accessor, size_t query_start_idx,
//                            size_t num_active_queries, size_t panel_base_idx)
//     Called once per panel inside the GEMM loop. scores_accessor allows
//     indexing into the VEC_GROUP of float score vectors for all
//     kQueriesPerBatch queries. The base index is computed as panel_base_idx =
//     panel_idx * points_per_panel.
//
//   reduce_batch_fn(size_t query_start_idx,
//                   const QueryBatch<kQueriesPerBatch>& query_batch)
//     Called once per batch, after the panel loop. Use to reduce
//     per-batch state (e.g., cross-lane top-K reduction).
template <size_t kQueriesPerBatch, class PanelPtrFn, class InitBatchFn,
          class PanelScoresConsumerFn, class ReduceBatchFn>
void RunQueryLoop(size_t num_queries, bool is_l2, size_t num_panels,
                  size_t tiles_per_vector, const float* db_norm_scaling_factors,
                  const float* db_sqr_norms, bool lower_triangle,
                  PanelPtrFn panel_pointer_fn, InitBatchFn init_batch_fn,
                  PanelScoresConsumerFn panel_scores_consumer_fn,
                  ReduceBatchFn reduce_batch_fn) {
  ABSL_QCHECK(!is_l2 || db_sqr_norms != nullptr);
  Int di32;
  Float df;
  const size_t NF = hn::Lanes(df);

  for (size_t query_start_idx = 0; query_start_idx < num_queries;
       query_start_idx += kQueriesPerBatch) {
    const size_t num_active_queries =
        std::min(kQueriesPerBatch, num_queries - query_start_idx);

    QueryBatch<kQueriesPerBatch> query_batch;
    init_batch_fn(query_start_idx, num_active_queries, query_batch);
    VEC_GROUP_DEFINE(di32, bias_correction);
    ForIndex<kQueriesPerBatch>([&](auto query_idx) HWY_ATTR {
      constexpr size_t q = decltype(query_idx)::value;
      VEC_GROUP_ACCESSOR(bias_correction)(query_idx) =
          hn::Set(di32, 128 * query_batch.byte_sums[q]);
    });

    const size_t panel_start = lower_triangle ? (query_start_idx / NF) : 0;

    // The outer GEMM loop processes queries in blocks of `kQueriesPerBatch` to
    // amortize the cost of loading database points across multiple queries.
    // When `MicroKernelAccumulate` finishes a panel, it holds
    // `kQueriesPerBatch` accumulators. This epilogue function matches that
    // unrolling, processing the accumulators and streaming out scores for all
    // `kQueriesPerBatch` queries.
    auto panel_epilogue_fn = [&](auto accumulator_accessor,
                                 size_t panel_idx) HWY_ATTR {
      const size_t panel_base_idx = panel_idx * NF;
      const hn::Vec<Float> negated_db_norm_scaling =
          hn::Neg(hn::Load(df, db_norm_scaling_factors + panel_base_idx));
      const hn::Vec<Float> db_sqr_norms_vec =
          db_sqr_norms != nullptr ? hn::Load(df, db_sqr_norms + panel_base_idx)
                                  : hn::Zero(df);
      VEC_GROUP_DEFINE(df, scores);
      internal::ComputePanelScores<kQueriesPerBatch>(
          accumulator_accessor, VEC_GROUP_ACCESSOR(bias_correction),
          query_batch, negated_db_norm_scaling, db_sqr_norms_vec, is_l2,
          VEC_GROUP_ACCESSOR(scores));
      panel_scores_consumer_fn(VEC_GROUP_ACCESSOR(scores), query_start_idx,
                               num_active_queries, panel_base_idx);
    };
    internal::RunPanelLoop<kQueriesPerBatch>(
        query_batch.query_ptrs, num_panels, tiles_per_vector, panel_start,
        panel_pointer_fn, panel_epilogue_fn);

    reduce_batch_fn(query_start_idx, query_batch);
  }
}

// Convenience overload of RunQueryLoop for pre-decoded queries.
//
// QueryBatch is automatically populated using `PrepareQueryBatch` from the
// `decoded_view` before `init_batch_fn` is invoked.
template <size_t kQueriesPerBatch, class PanelPtrFn, class InitBatchFn,
          class PanelScoresConsumerFn, class ReduceBatchFn>
void RunQueryLoop(const DecodedQueryView& decoded_view, size_t num_queries,
                  size_t num_panels, size_t tiles_per_vector,
                  const float* db_norm_scaling_factors,
                  const float* db_sqr_norms, bool lower_triangle,
                  PanelPtrFn panel_pointer_fn, InitBatchFn init_batch_fn,
                  PanelScoresConsumerFn panel_scores_consumer_fn,
                  ReduceBatchFn reduce_batch_fn) {
  RunQueryLoop<kQueriesPerBatch>(
      num_queries, decoded_view.is_l2, num_panels, tiles_per_vector,
      db_norm_scaling_factors, db_sqr_norms, lower_triangle,
      std::move(panel_pointer_fn),
      [&decoded_view, &init_batch_fn](
          size_t query_start_idx, size_t num_active_queries,
          QueryBatch<kQueriesPerBatch>& query_batch) {
        internal::PrepareQueryBatch(decoded_view, query_start_idx,
                                    num_active_queries, query_batch);
        init_batch_fn(query_start_idx, num_active_queries, query_batch);
      },
      std::move(panel_scores_consumer_fn), std::move(reduce_batch_fn));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace ginger

// clang-format off
#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_GEMM_LOOP_TOGGLE // NOLINT
// clang-format on
