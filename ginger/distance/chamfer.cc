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

#include "ginger/distance/chamfer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/types.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ginger/distance/chamfer.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#ifndef GINGER_DISTANCE_CHAMFER_CC_
#define GINGER_DISTANCE_CHAMFER_CC_

namespace ginger {
constexpr size_t kBlockSize = 16;
constexpr size_t kBlockSize8 = 8;
constexpr size_t kMaxQueries = 16;

namespace {
absl::Span<const float> DocSpan(const ChamferDataset& dataset,
                                GingerNodeId id) {
  const float* begin =
      static_cast<const float*>(dataset.tokens()[dataset.doc_offsets()[id]]);
  const float* end = static_cast<const float*>(
      dataset.tokens()[dataset.doc_offsets()[id + 1]]);
  return absl::MakeConstSpan(begin, end);
}
}  // namespace
}  // namespace ginger

#endif  // GINGER_DISTANCE_CHAMFER_CC_

HWY_BEFORE_NAMESPACE();
#include "ginger/turboquant/simd/vec_group-inl.h"
namespace ginger {
namespace HWY_NAMESPACE {

namespace {

namespace hn = hwy::HWY_NAMESPACE;
using HWY_NAMESPACE::ForIndex;

// Extend VEC_GROUP to 16 vectors.
#define VEC_GROUP_DEFINE_16(tag, name) \
  VEC_GROUP_DEFINE(tag, name##_lo);    \
  VEC_GROUP_DEFINE(tag, name##_hi)

#define VEC_GROUP_ACCESSOR_16(name)                                   \
  [&](auto _vg_idx) -> auto& {                                        \
    constexpr size_t _i = decltype(_vg_idx)::value;                   \
    static_assert(_i < 16, "Index out of range for 16-vector group"); \
    auto _acc_lo = VEC_GROUP_ACCESSOR(name##_lo);                     \
    auto _acc_hi = VEC_GROUP_ACCESSOR(name##_hi);                     \
    if constexpr (_i < 8)                                             \
      return _acc_lo(std::integral_constant<size_t, _i>{});           \
    else                                                              \
      return _acc_hi(std::integral_constant<size_t, _i - 8>{});       \
  }

// Computes query-document dot products for a tile of (NumBlocks x NumQueries)
// candidate/query pairs and writes the per-query max-over-lanes into
// out_scores.
//
// Each "block" is a BlockSize x dim matrix of document token embeddings
// stored in transposed layout (b_ptrs[b] points to BlockSize floats for
// each of the dim dimensions in turn).
//
// out_scores layout: out_scores[b * nq + (q_tile_start + q)] is the max dot
// product between query q and any token in block b.
//
// Instantiated with BlockSize=kBlockSize (16) for full blocks and
// BlockSize=kBlockSize8 (8) for micro-blocks.
template <size_t NumBlocks, size_t NumQueries, size_t BlockSize, class D>
HWY_INLINE void ScoreTransposedTile(D d_tag, absl::Span<const float> query,
                                    size_t dimensions,
                                    const float* const* HWY_RESTRICT block_ptrs,
                                    size_t query_tile_start,
                                    float* HWY_RESTRICT out_scores) {
  static_assert(NumBlocks <= 4 && NumQueries <= 4,
                "Tile dimensions must be <= 4");
  const size_t num_queries = query.size() / dimensions;
  const float* HWY_RESTRICT queries_ptr = query.data();
  const size_t lanes = hn::Lanes(d_tag);
  const auto neg_inf = hn::Set(d_tag, -std::numeric_limits<float>::infinity());

  // max_dot[b][q] = max over all token positions of dot(query_q, doc_token).
  // Indexed as b * NumQueries + q, reduced across SIMD lanes at the end.
  VEC_GROUP_DEFINE_16(d_tag, max_dot);
  auto MaxDot = VEC_GROUP_ACCESSOR_16(max_dot);
  ForIndex<NumBlocks * NumQueries>([&](auto i)
                                       HWY_ATTR { MaxDot(i) = neg_inf; });

  // Outer loop: SIMD-width-sized slices of the BlockSize token dimension.
  for (size_t token_slice_start = 0; token_slice_start < BlockSize;
       token_slice_start += lanes) {
    // Accumulate dot products for this lane slice over all embedding dims.
    VEC_GROUP_DEFINE_16(d_tag, dot);
    auto Dot = VEC_GROUP_ACCESSOR_16(dot);
    ForIndex<NumBlocks * NumQueries>(
        [&](auto i) HWY_ATTR { Dot(i) = hn::Zero(d_tag); });

    for (size_t dim_idx = 0; dim_idx < dimensions; ++dim_idx) {
      // Broadcast each query's dim_idx-th embedding scalar across all lanes.
      VEC_GROUP_DEFINE(d_tag, q_scalars);
      auto Q = VEC_GROUP_ACCESSOR(q_scalars);
      ForIndex<NumQueries>([&](auto q) HWY_ATTR {
        Q(q) = hn::Set(
            d_tag,
            queries_ptr[(query_tile_start + q.value) * dimensions + dim_idx]);
      });

      // Load BlockSize document token values at embedding dim dim_idx.
      VEC_GROUP_DEFINE(d_tag, doc_slice);
      auto B = VEC_GROUP_ACCESSOR(doc_slice);
      ForIndex<NumBlocks>([&](auto b) HWY_ATTR {
        B(b) = hn::LoadU(d_tag, block_ptrs[b.value] + dim_idx * BlockSize +
                                    token_slice_start);
      });

      // FMA: dot[b][q] += Q[q] * B[b]  for all (b, q) pairs.
      ForIndex<NumBlocks>([&](auto b) HWY_ATTR {
        ForIndex<NumQueries>([&](auto q) HWY_ATTR {
          constexpr size_t i = b.value * NumQueries + q.value;
          Dot(std::integral_constant<size_t, i>{}) =
              hn::MulAdd(Q(q), B(b), Dot(std::integral_constant<size_t, i>{}));
        });
      });
    }

    ForIndex<NumBlocks * NumQueries>(
        [&](auto i) HWY_ATTR { MaxDot(i) = hn::Max(MaxDot(i), Dot(i)); });
  }

  // Reduce across SIMD lanes and write results.
  ForIndex<NumBlocks>([&](auto b) HWY_ATTR {
    ForIndex<NumQueries>([&](auto q) HWY_ATTR {
      constexpr size_t i = b.value * NumQueries + q.value;
      out_scores[b.value * num_queries + (query_tile_start + q.value)] =
          hn::ReduceMax(d_tag, MaxDot(std::integral_constant<size_t, i>{}));
    });
  });
}

// Iterates over query tiles of size 4 and 1 to cover all num_queries.
template <size_t NumBlocks, size_t BlockSize, class D>
HWY_INLINE void ScoreBlocks(D d_tag, absl::Span<const float> query,
                            size_t dimensions,
                            const float* const* HWY_RESTRICT block_ptrs,
                            float* HWY_RESTRICT out_scores) {
  const size_t num_queries = query.size() / dimensions;
  size_t query_idx = 0;
  for (; query_idx + 4 <= num_queries; query_idx += 4) {
    ScoreTransposedTile<NumBlocks, 4, BlockSize>(
        d_tag, query, dimensions, block_ptrs, query_idx, out_scores);
  }
  for (; query_idx < num_queries; ++query_idx) {
    ScoreTransposedTile<NumBlocks, 1, BlockSize>(
        d_tag, query, dimensions, block_ptrs, query_idx, out_scores);
  }
}

// Resolves the NumBlocks template from the runtime num_blocks value (1..4).
template <size_t BlockSize, class D>
HWY_INLINE void ScoreBlocksDispatch(size_t num_blocks, D d_tag,
                                    absl::Span<const float> query,
                                    size_t dimensions,
                                    const float* const* HWY_RESTRICT block_ptrs,
                                    float* HWY_RESTRICT out_scores) {
  switch (num_blocks) {
    case 4:
      return ScoreBlocks<4, BlockSize>(d_tag, query, dimensions, block_ptrs,
                                       out_scores);
    case 3:
      return ScoreBlocks<3, BlockSize>(d_tag, query, dimensions, block_ptrs,
                                       out_scores);
    case 2:
      return ScoreBlocks<2, BlockSize>(d_tag, query, dimensions, block_ptrs,
                                       out_scores);
    default:
      return ScoreBlocks<1, BlockSize>(d_tag, query, dimensions, block_ptrs,
                                       out_scores);
  }
}

// Batched horizontal reduction of 4 full-width accumulator vectors into a
// single quarter-width SIMD register containing the 4 scalar sums.
// Requires >= 16 lanes (AVX-512) so that quarter-width has 4 lanes.
//
// Reduction tree:
//   Step 1 (Fold in half):       h_i = LowerHalf(v_i) + UpperHalf(v_i)
//   Step 2 (Fold to quarter):    q_i = LowerHalf(h_i) + UpperHalf(h_i)
//   Step 3 (Interleave & Add):   s01 = InterleaveLo/Hi(q0, q1) -> add
//                                s23 = InterleaveLo/Hi(q2, q3) -> add
//   Step 4 (Final Packing):      res = InterleaveLo(s01, s23) +
//                                      InterleaveHi(s01, s23)
//
// Result: Lane indices [0, 1, 2, 3] hold sum(v0), sum(v2), sum(v1), sum(v3).
// (Permuted channel order is unpacked explicitly in ScoreTileRowMajor4.)
template <class D, class DH, class DQ>
HWY_INLINE hn::Vec<DQ> Reduce4Vectors(D d_tag, DH dh, DQ dq, hn::Vec<D> sum0,
                                      hn::Vec<D> sum1, hn::Vec<D> sum2,
                                      hn::Vec<D> sum3) {
  auto h0 = hn::Add(hn::LowerHalf(dh, sum0), hn::UpperHalf(dh, sum0));
  auto h1 = hn::Add(hn::LowerHalf(dh, sum1), hn::UpperHalf(dh, sum1));
  auto h2 = hn::Add(hn::LowerHalf(dh, sum2), hn::UpperHalf(dh, sum2));
  auto h3 = hn::Add(hn::LowerHalf(dh, sum3), hn::UpperHalf(dh, sum3));

  auto q0 = hn::Add(hn::LowerHalf(dq, h0), hn::UpperHalf(dq, h0));
  auto q1 = hn::Add(hn::LowerHalf(dq, h1), hn::UpperHalf(dq, h1));
  auto q2 = hn::Add(hn::LowerHalf(dq, h2), hn::UpperHalf(dq, h2));
  auto q3 = hn::Add(hn::LowerHalf(dq, h3), hn::UpperHalf(dq, h3));

  auto s01 =
      hn::Add(hn::InterleaveLower(dq, q0, q1), hn::InterleaveUpper(dq, q0, q1));
  auto s23 =
      hn::Add(hn::InterleaveLower(dq, q2, q3), hn::InterleaveUpper(dq, q2, q3));

  auto lo = hn::InterleaveLower(dq, s01, s23);
  auto hi = hn::InterleaveUpper(dq, s01, s23);
  return hn::Add(lo, hi);
}

// Scores all doc tokens against a single query using SIMD dot products.
// Maintains a running max over all token dot products.
template <class D>
HWY_INLINE void ScoreTileRowMajor1(D d_tag, const float* HWY_RESTRICT query_ptr,
                                   size_t dimensions,
                                   absl::Span<const float> doc,
                                   float* HWY_RESTRICT out_max) {
  const size_t num_doc_tokens = doc.size() / dimensions;
  const float* HWY_RESTRICT doc_ptr = doc.data();
  const size_t lanes = hn::Lanes(d_tag);
  float max_score = out_max[0];

  for (size_t doc_token_idx = 0; doc_token_idx < num_doc_tokens;
       ++doc_token_idx) {
    const float* doc_tok = doc_ptr + doc_token_idx * dimensions;
    hn::Vec<D> sum0 = hn::Zero(d_tag);
    size_t dim_idx = 0;
    for (; dim_idx + lanes <= dimensions; dim_idx += lanes) {
      sum0 = hn::MulAdd(hn::LoadU(d_tag, query_ptr + dim_idx),
                        hn::LoadU(d_tag, doc_tok + dim_idx), sum0);
    }
    float dot0 = hn::ReduceSum(d_tag, sum0);
    for (; dim_idx < dimensions; ++dim_idx) {
      dot0 += query_ptr[dim_idx] * doc_tok[dim_idx];
    }
    max_score = std::max(max_score, dot0);
  }
  out_max[0] = max_score;
}

// Scores doc tokens against 4 queries simultaneously. Updates
// out_max[0..3] with the maximum dot product per query. Uses a 4-query x
// 4-doc GEMM tile for the inner loop to maximize FMA throughput, then falls
// back to 1-doc-at-a-time for the 0..3 remainder.
//
// When SIMD width >= 16 lanes (AVX-512), uses Reduce4Vectors for batched
// horizontal reduction (4 reductions in ~12 instructions instead of 16
// individual ReduceSum calls). Falls through to ScoreTileRowMajor1 on
// narrower targets (NEON, SSE, AVX2).
template <class D>
HWY_INLINE void ScoreTileRowMajor4(D d_tag,
                                   const float* HWY_RESTRICT queries_ptr,
                                   size_t dimensions,
                                   absl::Span<const float> doc,
                                   float* HWY_RESTRICT out_max) {
  const size_t num_doc_tokens = doc.size() / dimensions;
  const float* HWY_RESTRICT doc_ptr = doc.data();
  const size_t lanes = hn::Lanes(d_tag);
  const float* HWY_RESTRICT query_ptrs[4] = {
      queries_ptr + 0 * dimensions,
      queries_ptr + 1 * dimensions,
      queries_ptr + 2 * dimensions,
      queries_ptr + 3 * dimensions,
  };

  if (lanes == 16 && (dimensions % lanes == 0)) {
    const hn::Half<D> dh;
    const hn::Half<decltype(dh)> dq;
    auto max_perm = hn::Set(dq, -std::numeric_limits<float>::infinity());

    size_t doc_token_idx = 0;
    // Process 4 doc tokens at a time via 4 queries x 4 doc tokens GEMM tile.
    for (; doc_token_idx + 4 <= num_doc_tokens; doc_token_idx += 4) {
      const float* HWY_RESTRICT doc_ptrs[4] = {
          doc_ptr + (doc_token_idx + 0) * dimensions,
          doc_ptr + (doc_token_idx + 1) * dimensions,
          doc_ptr + (doc_token_idx + 2) * dimensions,
          doc_ptr + (doc_token_idx + 3) * dimensions,
      };

      VEC_GROUP_DEFINE_16(d_tag, acc);
      auto Acc = VEC_GROUP_ACCESSOR_16(acc);
      ForIndex<16>([&](auto i) HWY_ATTR { Acc(i) = hn::Zero(d_tag); });

      for (size_t dim_idx = 0; dim_idx < dimensions; dim_idx += lanes) {
        VEC_GROUP_DEFINE(d_tag, q_v);
        auto Q = VEC_GROUP_ACCESSOR(q_v);
        ForIndex<4>([&](auto q) HWY_ATTR {
          Q(q) = hn::LoadU(d_tag, query_ptrs[q.value] + dim_idx);
        });

        VEC_GROUP_DEFINE(d_tag, d_v);
        auto D_vec = VEC_GROUP_ACCESSOR(d_v);
        ForIndex<4>([&](auto d) HWY_ATTR {
          D_vec(d) = hn::LoadU(d_tag, doc_ptrs[d.value] + dim_idx);
        });

        ForIndex<4>([&](auto d) HWY_ATTR {
          ForIndex<4>([&](auto q) HWY_ATTR {
            constexpr size_t i = d.value * 4 + q.value;
            Acc(std::integral_constant<size_t, i>{}) = hn::MulAdd(
                Q(q), D_vec(d), Acc(std::integral_constant<size_t, i>{}));
          });
        });
      }

      ForIndex<4>([&](auto d) HWY_ATTR {
        constexpr size_t base = d.value * 4;
        auto dots = Reduce4Vectors(
            d_tag, dh, dq, Acc(std::integral_constant<size_t, base + 0>{}),
            Acc(std::integral_constant<size_t, base + 1>{}),
            Acc(std::integral_constant<size_t, base + 2>{}),
            Acc(std::integral_constant<size_t, base + 3>{}));
        max_perm = hn::Max(max_perm, dots);
      });
    }

    // Trailing 0..3 doc tokens.
    for (; doc_token_idx < num_doc_tokens; ++doc_token_idx) {
      const float* doc_tok = doc_ptr + doc_token_idx * dimensions;
      VEC_GROUP_DEFINE(d_tag, sum_v);
      auto Sum = VEC_GROUP_ACCESSOR(sum_v);
      ForIndex<4>([&](auto q) HWY_ATTR { Sum(q) = hn::Zero(d_tag); });

      for (size_t dim_idx = 0; dim_idx < dimensions; dim_idx += lanes) {
        const hn::Vec<D> d0 = hn::LoadU(d_tag, doc_tok + dim_idx);
        ForIndex<4>([&](auto q) HWY_ATTR {
          Sum(q) = hn::MulAdd(hn::LoadU(d_tag, query_ptrs[q.value] + dim_idx),
                              d0, Sum(q));
        });
      }

      auto dots = Reduce4Vectors(d_tag, dh, dq,
                                 Sum(std::integral_constant<size_t, 0>{}),
                                 Sum(std::integral_constant<size_t, 1>{}),
                                 Sum(std::integral_constant<size_t, 2>{}),
                                 Sum(std::integral_constant<size_t, 3>{}));
      max_perm = hn::Max(max_perm, dots);
    }

    alignas(16) float max_buf[4];
    hn::StoreU(max_perm, dq, max_buf);
    // Reduce4Vectors yields horizontal sums in permuted channel order [0, 2, 1,
    // 3].
    out_max[0] = std::max(out_max[0], max_buf[0]);  // Query 0
    out_max[1] = std::max(out_max[1], max_buf[2]);  // Query 1
    out_max[2] = std::max(out_max[2], max_buf[1]);  // Query 2
    out_max[3] = std::max(out_max[3], max_buf[3]);  // Query 3
    return;
  }

  // Fallback for smaller vector widths or non-multiple dimensions.
  for (size_t q = 0; q < 4; ++q) {
    ScoreTileRowMajor1(d_tag, query_ptrs[q], dimensions, doc, out_max + q);
  }
}

template <class D>
HWY_INLINE void ScoreRowMajorBlocks(D d_tag, absl::Span<const float> query,
                                    size_t dimensions,
                                    absl::Span<const float> doc,
                                    float* HWY_RESTRICT out_max) {
  const size_t num_queries = query.size() / dimensions;
  const float* HWY_RESTRICT queries_ptr = query.data();
  size_t query_idx = 0;
  for (; query_idx + 4 <= num_queries; query_idx += 4) {
    ScoreTileRowMajor4(d_tag, queries_ptr + query_idx * dimensions, dimensions,
                       doc, out_max + query_idx);
  }
  for (; query_idx < num_queries; ++query_idx) {
    ScoreTileRowMajor1(d_tag, queries_ptr + query_idx * dimensions, dimensions,
                       doc, out_max + query_idx);
  }
}

HWY_INLINE float AverageSimilarity(const float* HWY_RESTRICT scores,
                                   size_t num_queries, float inv_num_queries) {
  float total = 0.0f;
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    total += scores[query_idx];
  }
  return total * inv_num_queries;
}

HWY_INLINE void MergeScores(float* HWY_RESTRICT accumulated_max_scores,
                            const float* HWY_RESTRICT new_scores,
                            size_t num_queries) {
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    accumulated_max_scores[query_idx] =
        std::max(accumulated_max_scores[query_idx], new_scores[query_idx]);
  }
}

// Maximum number of query tokens supported by DocTracker stack buffers.

template <class D>
HWY_INLINE float ScoreCandidateRowMajor(D d_tag, absl::Span<const float> query,
                                        size_t dimensions,
                                        absl::Span<const float> doc,
                                        float inv_num_queries) {
  const size_t num_queries = query.size() / dimensions;
  float max_score_per_query[kMaxQueries];
  std::fill_n(max_score_per_query, num_queries,
              -std::numeric_limits<float>::infinity());
  ScoreRowMajorBlocks(d_tag, query, dimensions, doc, max_score_per_query);
  return AverageSimilarity(max_score_per_query, num_queries, inv_num_queries);
}

// Issues cache prefetches across a document's token storage.
HWY_INLINE void PrefetchCandidate(const ChamferDataset& dataset,
                                  GingerNodeId id, size_t block_stride) {
  absl::Span<const float> doc = DocSpan(dataset, id);
  for (const float* ptr = doc.begin(); ptr < doc.end(); ptr += block_stride) {
    absl::PrefetchToLocalCache(ptr);
  }
}

// Maximum number of documents simultaneously in-flight during cross-doc
// batching. Each pending buffer holds up to 3 entries between flushes (the 4th
// triggers a flush). With two independent buffers (size-16 and size-8 blocks),
// up to 6 distinct docs can be pending. One additional slot is needed for the
// doc currently being processed before its first push triggers a flush.
static constexpr size_t kMaxInFlight = 7;

// Accumulates blocks from up to 4 documents for batched scoring.
struct PendingBuffer {
  const float* ptrs[4];
  uint8_t slots[4];
  uint8_t count = 0;

  void Push(const float* ptr, uint8_t slot) {
    ptrs[count] = ptr;
    slots[count] = slot;
    ++count;
  }
  bool IsFull() const { return count == 4; }
};

// Manages in-flight document state for cross-doc block batching.
// Owns all scratch buffers (stack-allocated, bounded by kMaxQueries and
// kMaxInFlight) and provides methods to allocate/free slots, push blocks,
// and flush pending batches.
struct DocTracker {
  // Per-slot state.
  struct Slot {
    GingerNodeId neighbor_idx;
    uint16_t pending_count;
  };

  // Context (constant for the entire scoring pass).
  absl::Span<const float> query;
  size_t dimensions;
  size_t num_queries;
  absl::Span<DistanceAndId> neighbors;
  float inv_num_queries;
  bool accumulate;
  hn::CappedTag<float, 16> d_tag;
  hn::CappedTag<float, 8> d8_tag;

  // Scratch buffers.
  HWY_ALIGN float scores[4 * kMaxQueries];
  HWY_ALIGN float doc_max[kMaxInFlight * kMaxQueries];

  // Slot management.
  Slot slots[kMaxInFlight];
  uint8_t free_stack[kMaxInFlight];
  uint8_t free_top;

  // Pending block buffers for cross-doc batching.
  PendingBuffer pending_16;
  PendingBuffer pending_8;

  void Init(absl::Span<const float> q, size_t dims,
            absl::Span<DistanceAndId> nbrs, float inv_total_queries,
            bool accum = false) {
    query = q;
    dimensions = dims;
    num_queries = q.size() / dims;
    neighbors = nbrs;
    inv_num_queries = inv_total_queries;
    accumulate = accum;
    free_top = kMaxInFlight;
    for (size_t i = 0; i < kMaxInFlight; ++i) {
      free_stack[i] = kMaxInFlight - 1 - i;
    }
  }

  uint8_t AllocSlot(GingerNodeId neighbor_idx, uint16_t expected_blocks) {
    ABSL_DCHECK_GT(free_top, 0);
    const uint8_t slot = free_stack[--free_top];
    slots[slot] = {neighbor_idx, expected_blocks};
    std::fill_n(doc_max + slot * num_queries, num_queries,
                -std::numeric_limits<float>::infinity());
    return slot;
  }

  float* SlotScores(uint8_t slot) { return doc_max + slot * num_queries; }

  // Scores a small doc (< 8 tokens) entirely in row-major layout.
  float ScoreSmallDoc(absl::Span<const float> doc) {
    return ScoreCandidateRowMajor(d_tag, query, dimensions, doc,
                                  inv_num_queries);
  }

  // Scores remainder tokens (< 8) and merges into the slot's max scores.
  void ScoreRemainder(absl::Span<const float> doc, uint8_t slot) {
    ScoreRowMajorBlocks(d_tag, query, dimensions, doc, SlotScores(slot));
  }

  void PushFullBlock(const float* ptr, uint8_t slot) {
    pending_16.Push(ptr, slot);
    if (pending_16.IsFull()) {
      Flush<kBlockSize>(d_tag, pending_16);
    }
  }

  void PushTailBlock(const float* ptr, uint8_t slot) {
    pending_8.Push(ptr, slot);
    if (pending_8.IsFull()) {
      Flush<kBlockSize8>(d8_tag, pending_8);
    }
  }

  void FlushAll() {
    Flush<kBlockSize>(d_tag, pending_16);
    Flush<kBlockSize8>(d8_tag, pending_8);
  }

 private:
  void FinalizeSlot(uint8_t slot) {
    const float sim =
        AverageSimilarity(SlotScores(slot), num_queries, inv_num_queries);
    if (accumulate) {
      neighbors[slots[slot].neighbor_idx].distance += sim;
    } else {
      neighbors[slots[slot].neighbor_idx].distance = sim;
    }
    free_stack[free_top++] = slot;
  }

  template <size_t BlockSize, class D>
  void Flush(D d, PendingBuffer& pending) {
    if (pending.count == 0) return;
    ScoreBlocksDispatch<BlockSize>(pending.count, d, query, dimensions,
                                   pending.ptrs, scores);
    for (size_t i = 0; i < pending.count; ++i) {
      const uint8_t slot = pending.slots[i];
      MergeScores(SlotScores(slot), scores + i * num_queries, num_queries);
      if (--slots[slot].pending_count == 0) {
        FinalizeSlot(slot);
      }
    }
    pending.count = 0;
  }
};

}  // namespace

void ChamferDistanceSimd(absl::Span<const float> query,
                         absl::Span<DistanceAndId> neighbors,
                         const ChamferDataset& dataset, float inv_total_queries,
                         bool accumulate = false) {
  const size_t dimensions = dataset.dimensionality();
  ABSL_QCHECK_LE(query.size() / dimensions, kMaxQueries);

  DocTracker tracker;
  tracker.Init(query, dimensions, neighbors, inv_total_queries, accumulate);

  const size_t block_stride = kBlockSize * dimensions;

  constexpr size_t kPrefetchDistance = 8;
  for (size_t p = 0; p < kPrefetchDistance && p < neighbors.size(); ++p) {
    PrefetchCandidate(dataset, neighbors[p].id, block_stride);
  }

  for (size_t ni = 0; ni < neighbors.size(); ++ni) {
    if (ni + kPrefetchDistance < neighbors.size()) {
      PrefetchCandidate(dataset, neighbors[ni + kPrefetchDistance].id,
                        block_stride);
    }
    const GingerNodeId id = neighbors[ni].id;
    const size_t num_tokens = dataset.num_tokens(id);
    if (num_tokens == 0) {
      if (!accumulate) {
        neighbors[ni].distance = 0.0f;
      }
      continue;
    }

    absl::Span<const float> doc = DocSpan(dataset, id);

    if (num_tokens < kBlockSize8) {
      const float sim = tracker.ScoreSmallDoc(doc);
      if (accumulate) {
        neighbors[ni].distance += sim;
      } else {
        neighbors[ni].distance = sim;
      }
      continue;
    }

    const size_t num_full_blocks = num_tokens / kBlockSize;
    const bool has_tail = (num_tokens % kBlockSize) >= kBlockSize8;
    const uint16_t expected_blocks =
        static_cast<uint16_t>(num_full_blocks + (has_tail ? 1 : 0));

    const uint8_t slot =
        tracker.AllocSlot(static_cast<GingerNodeId>(ni), expected_blocks);

    // Score remainder, before pushing transposed blocks to pending.
    const size_t num_remainder = num_tokens % kBlockSize8;
    if (num_remainder > 0) {
      const size_t remainder_size = num_remainder * dimensions;
      tracker.ScoreRemainder(
          absl::MakeConstSpan(doc.end() - remainder_size, remainder_size),
          slot);
    }

    const float* doc_ptr = doc.data();
    for (size_t b = 0; b < num_full_blocks; ++b) {
      tracker.PushFullBlock(doc_ptr, slot);
      doc_ptr += block_stride;
    }

    if (has_tail) {
      tracker.PushTailBlock(doc_ptr, slot);
    }
  }

  tracker.FlushAll();
}

#undef VEC_GROUP_DEFINE_16
#undef VEC_GROUP_ACCESSOR_16

}  // namespace HWY_NAMESPACE
}  // namespace ginger

HWY_AFTER_NAMESPACE();
#if HWY_ONCE
namespace ginger {

HWY_EXPORT(ChamferDistanceSimd);

namespace {

void TransposeBlock(const float* HWY_RESTRICT src, float* HWY_RESTRICT dst,
                    size_t block_size, size_t dimensions) {
  for (size_t dim_idx = 0; dim_idx < dimensions; ++dim_idx) {
    for (size_t token_idx = 0; token_idx < block_size; ++token_idx) {
      dst[dim_idx * block_size + token_idx] =
          src[token_idx * dimensions + dim_idx];
    }
  }
}

}  // namespace

void ChamferDataset::SetDoc(GingerNodeId id,
                            absl::Span<const float> doc_tokens) {
  const size_t num_doc_tokens = num_tokens(id);
  const size_t dimensions = dimensionality();
  ABSL_DCHECK_EQ(doc_tokens.size(), num_doc_tokens * dimensions);
  float* dst = static_cast<float*>(tokens_[doc_offsets_[id]]);
  if (num_doc_tokens < kBlockSize8) {
    std::copy_n(doc_tokens.begin(), doc_tokens.size(), dst);
    return;
  }

  const float* src = doc_tokens.data();
  const size_t full_stride = kBlockSize * dimensions;

  for (size_t b = 0; b < num_doc_tokens / kBlockSize; ++b) {
    TransposeBlock(src, dst, kBlockSize, dimensions);
    src += full_stride;
    dst += full_stride;
  }

  if ((num_doc_tokens % kBlockSize) >= kBlockSize8) {
    const size_t tail_stride = kBlockSize8 * dimensions;
    TransposeBlock(src, dst, kBlockSize8, dimensions);
    src += tail_stride;
    dst += tail_stride;
  }

  const size_t remainder_tokens = num_doc_tokens % kBlockSize8;
  if (remainder_tokens > 0) {
    std::copy_n(src, remainder_tokens * dimensions, dst);
  }
}

void ChamferDataset::TransposeDoc(GingerNodeId id, absl::Span<float> scratch) {
  const size_t num_doc_tokens = num_tokens(id);
  if (num_doc_tokens < kBlockSize8) return;
  const size_t dimensions = dimensionality();
  const size_t full_stride = kBlockSize * dimensions;
  std::vector<float> owned_scratch;
  if (scratch.empty()) {
    owned_scratch.resize(full_stride);
    scratch = absl::MakeSpan(owned_scratch);
  }
  ABSL_DCHECK_GE(scratch.size(), full_stride);

  auto transpose_block = [&](float* block, size_t block_size) {
    const size_t stride = block_size * dimensions;
    TransposeBlock(block, scratch.data(), block_size, dimensions);
    std::copy_n(scratch.data(), stride, block);
  };

  float* ptr = static_cast<float*>(tokens_[doc_offsets_[id]]);
  for (size_t block_idx = 0; block_idx < num_doc_tokens / kBlockSize;
       ++block_idx) {
    transpose_block(ptr, kBlockSize);
    ptr += full_stride;
  }
  if ((num_doc_tokens % kBlockSize) >= kBlockSize8) {
    transpose_block(ptr, kBlockSize8);
  }
}

void ChamferDataset::PrepareTransposedBlocks() {
  const GingerNodeId num_docs = size();
  const size_t block_stride = kBlockSize * dimensionality();
  std::vector<float> scratch(block_stride);
  for (GingerNodeId id = 0; id < num_docs; ++id) {
    TransposeDoc(id, absl::MakeSpan(scratch));
  }
}

void ChamferDataset::ChamferDistance(
    absl::Span<const float> query, absl::Span<DistanceAndId> neighbors) const {
  if (neighbors.empty()) return;
  const size_t dimensions = dimensionality();
  const size_t num_queries =
      (dimensions == 0) ? 0 : (query.size() / dimensions);
  if (num_queries == 0) {
    for (DistanceAndId& neighbor : neighbors) neighbor.distance = 0.0f;
    return;
  }
  const float inv_total_queries = -1.0f / static_cast<float>(num_queries);
  if (num_queries <= kMaxQueries) {
    HWY_DYNAMIC_DISPATCH(ChamferDistanceSimd)(query, neighbors, *this,
                                              inv_total_queries,
                                              /*accumulate=*/false);
    return;
  }

  // When num_queries > kMaxQueries, zero neighbors' distances first and then
  // accumulate each chunk's contribution directly in-place without allocation.
  for (DistanceAndId& neighbor : neighbors) {
    neighbor.distance = 0.0f;
  }
  for (size_t q_start = 0; q_start < num_queries; q_start += kMaxQueries) {
    const size_t chunk_queries = std::min(kMaxQueries, num_queries - q_start);
    absl::Span<const float> sub_query =
        query.subspan(q_start * dimensions, chunk_queries * dimensions);
    HWY_DYNAMIC_DISPATCH(ChamferDistanceSimd)(sub_query, neighbors, *this,
                                              inv_total_queries, true);
  }
}

}  // namespace ginger
#endif  // HWY_ONCE
