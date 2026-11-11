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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "ginger/internal/gunit_support.h"
#include "ginger/turboquant/pipnn/topk_result.h"

// foreach_target.h re-includes this file once per enabled SIMD target to
// compile target-specific code into separate namespaces for runtime dispatch.
#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/pipnn/topk_tracker_test.cc"
// clang-format on
// clang-format off

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "ginger/turboquant/pipnn/topk_tracker-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();

namespace ginger {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

HWY_NOINLINE void TestInsertTopKBasic() {
  FixedTopKResult<3> r;
  r.Insert(-10.0f, 42);
  EXPECT_FLOAT_EQ(r.Get(0).score, -10.0f);
  EXPECT_EQ(r.Get(0).index, 42);

  r.Insert(-5.0f, 7);
  EXPECT_FLOAT_EQ(r.Get(0).score, -10.0f);
  EXPECT_FLOAT_EQ(r.Get(1).score, -5.0f);
  EXPECT_EQ(r.Get(1).index, 7);

  r.Insert(-8.0f, 99);
  EXPECT_FLOAT_EQ(r.Get(0).score, -10.0f);
  EXPECT_FLOAT_EQ(r.Get(1).score, -8.0f);
  EXPECT_FLOAT_EQ(r.Get(2).score, -5.0f);
  EXPECT_EQ(r.Get(0).index, 42);
  EXPECT_EQ(r.Get(1).index, 99);
  EXPECT_EQ(r.Get(2).index, 7);
}

HWY_NOINLINE void TestInsertTopKRejection() {
  FixedTopKResult<2> r;
  r.Insert(-3.0f, 1);
  r.Insert(-7.0f, 2);
  // -1.0 is worse than both, should be rejected.
  r.Insert(-1.0f, 3);
  EXPECT_FLOAT_EQ(r.Get(0).score, -7.0f);
  EXPECT_FLOAT_EQ(r.Get(1).score, -3.0f);
  EXPECT_EQ(r.Get(0).index, 2);
  EXPECT_EQ(r.Get(1).index, 1);
}

HWY_NOINLINE void TestInsertTopKDisplacement() {
  // Insert a new best that displaces everything down.
  FixedTopKResult<3> r;
  r.Insert(5.0f, 10);
  r.Insert(3.0f, 20);
  r.Insert(1.0f, 30);
  // Now insert a new global minimum.
  r.Insert(-2.0f, 40);
  EXPECT_FLOAT_EQ(r.Get(0).score, -2.0f);
  EXPECT_FLOAT_EQ(r.Get(1).score, 1.0f);
  EXPECT_FLOAT_EQ(r.Get(2).score, 3.0f);
  EXPECT_EQ(r.Get(0).index, 40);
  EXPECT_EQ(r.Get(1).index, 30);
  EXPECT_EQ(r.Get(2).index, 20);
}

HWY_NOINLINE void TestInsertTopK1() {
  FixedTopKResult<1> r;
  r.Insert(5.0f, 1);
  EXPECT_FLOAT_EQ(r.Get(0).score, 5.0f);
  r.Insert(3.0f, 2);
  EXPECT_FLOAT_EQ(r.Get(0).score, 3.0f);
  EXPECT_EQ(r.Get(0).index, 2);
  // Worse score rejected.
  r.Insert(10.0f, 3);
  EXPECT_FLOAT_EQ(r.Get(0).score, 3.0f);
}

HWY_NOINLINE void TestUpdateTop1SinglePanel() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  const float inf = std::numeric_limits<float>::max();

  // Create scores where lane 2 (or last lane if NF<3) has the minimum.
  HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) scores[i] = 100.0f;
  const size_t best_lane = NF < 3 ? NF - 1 : 2;
  scores[best_lane] = -50.0f;

  HWY_ALIGN float top_v_buf[HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t top_idx_buf[HWY_MAX_BYTES / sizeof(int32_t)];
  hn::Store(hn::Set(df, inf), df, top_v_buf);
  hn::Store(hn::Set(di32, -1), di32, top_idx_buf);

  UpdateTopK<1>(hn::Load(df, scores), /*panel_base=*/10,
                /*num_points=*/10 + NF, top_v_buf, top_idx_buf);

  FixedTopKResult<1> result;
  ReduceTopK<1>(top_v_buf, top_idx_buf, result);
  EXPECT_FLOAT_EQ(result.Get(0).score, -50.0f);
  EXPECT_EQ(result.Get(0).index, static_cast<int32_t>(10 + best_lane));
}

HWY_NOINLINE void TestUpdateTop1MultiplePanels() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  const size_t num_points = 3 * NF;
  const float inf = std::numeric_limits<float>::max();

  HWY_ALIGN float top_v_buf[HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t top_idx_buf[HWY_MAX_BYTES / sizeof(int32_t)];
  hn::Store(hn::Set(df, inf), df, top_v_buf);
  hn::Store(hn::Set(di32, -1), di32, top_idx_buf);

  // Panel 0: min at lane 0 = 10.0
  HWY_ALIGN float s0[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) s0[i] = 100.0f;
  s0[0] = 10.0f;
  UpdateTopK<1>(hn::Load(df, s0), 0, num_points, top_v_buf, top_idx_buf);

  // Panel 1: min at lane 1 = 5.0 (global best so far)
  HWY_ALIGN float s1[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) s1[i] = 100.0f;
  size_t lane1 = NF < 2 ? 0 : 1;
  s1[lane1] = 5.0f;
  UpdateTopK<1>(hn::Load(df, s1), NF, num_points, top_v_buf, top_idx_buf);

  // Panel 2: min at lane 0 = -3.0 (new global best)
  HWY_ALIGN float s2[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) s2[i] = 100.0f;
  s2[0] = -3.0f;
  UpdateTopK<1>(hn::Load(df, s2), 2 * NF, num_points, top_v_buf, top_idx_buf);

  FixedTopKResult<1> result;
  ReduceTopK<1>(top_v_buf, top_idx_buf, result);
  EXPECT_FLOAT_EQ(result.Get(0).score, -3.0f);
  EXPECT_EQ(result.Get(0).index, static_cast<int32_t>(2 * NF));
}

HWY_NOINLINE void TestUpdateTop1PaddingMasked() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  if (NF < 2) return;
  const float inf = std::numeric_limits<float>::max();

  // Only 1 valid point in the panel, rest are padding.
  // Padding lanes have the best score but should be masked to +inf.
  HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
  scores[0] = 50.0f;
  for (size_t i = 1; i < NF; ++i) scores[i] = -999.0f;

  HWY_ALIGN float top_v_buf[HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t top_idx_buf[HWY_MAX_BYTES / sizeof(int32_t)];
  hn::Store(hn::Set(df, inf), df, top_v_buf);
  hn::Store(hn::Set(di32, -1), di32, top_idx_buf);

  // num_points=1 means only lane 0 (base_v=0) is valid.
  UpdateTopK<1>(hn::Load(df, scores), /*panel_base=*/0,
                /*num_points=*/1, top_v_buf, top_idx_buf);

  FixedTopKResult<1> result;
  ReduceTopK<1>(top_v_buf, top_idx_buf, result);
  EXPECT_FLOAT_EQ(result.Get(0).score, 50.0f);
  EXPECT_EQ(result.Get(0).index, 0);
}

HWY_NOINLINE void TestUpdateTopKBasic() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  constexpr size_t K = 3;
  const size_t num_points = 4 * NF;
  const float inf = std::numeric_limits<float>::max();

  HWY_ALIGN float top_v_buf[K * HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t top_idx_buf[K * HWY_MAX_BYTES / sizeof(int32_t)];
  for (size_t k = 0; k < K; ++k) {
    hn::Store(hn::Set(df, inf), df, top_v_buf + k * NF);
    hn::Store(hn::Set(di32, -1), di32, top_idx_buf + k * NF);
  }

  // Insert 4 panels with decreasing minima across different lanes.
  for (size_t p = 0; p < 4; ++p) {
    HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
    for (size_t i = 0; i < NF; ++i) {
      scores[i] = 100.0f - static_cast<float>(p * NF + i) * 0.5f;
    }
    UpdateTopK<K>(hn::Load(df, scores), p * NF, num_points, top_v_buf,
                  top_idx_buf);
  }

  FixedTopKResult<K> result;
  ReduceTopK<K>(top_v_buf, top_idx_buf, result);

  // Verify sorted ascending (best = most negative first).
  for (size_t k = 1; k < K; ++k) {
    EXPECT_LE(result.Get(k - 1).score, result.Get(k).score) << "k=" << k;
  }
  // All indices should be valid.
  for (size_t k = 0; k < K; ++k) {
    EXPECT_GE(result.Get(k).index, 0) << "k=" << k;
    EXPECT_LT(result.Get(k).index, static_cast<int32_t>(num_points))
        << "k=" << k;
  }
  // Best should be last element (100 - (4*NF-1)*0.5).
  float expected_best = 100.0f - static_cast<float>(4 * NF - 1) * 0.5f;
  EXPECT_FLOAT_EQ(result.Get(0).score, expected_best);
}

HWY_NOINLINE void TestUpdateTopKPaddingMasked() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  if (NF < 2) return;
  constexpr size_t K = 2;
  const float inf = std::numeric_limits<float>::max();

  HWY_ALIGN float top_v_buf[K * HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t top_idx_buf[K * HWY_MAX_BYTES / sizeof(int32_t)];
  for (size_t k = 0; k < K; ++k) {
    hn::Store(hn::Set(df, inf), df, top_v_buf + k * NF);
    hn::Store(hn::Set(di32, -1), di32, top_idx_buf + k * NF);
  }

  // 1 valid point out of NF lanes. Padding has best scores.
  HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
  scores[0] = 42.0f;
  for (size_t i = 1; i < NF; ++i) scores[i] = -999.0f;

  UpdateTopK<K>(hn::Load(df, scores), /*panel_base=*/0,
                /*num_points=*/1, top_v_buf, top_idx_buf);

  FixedTopKResult<K> result;
  ReduceTopK<K>(top_v_buf, top_idx_buf, result);
  EXPECT_FLOAT_EQ(result.Get(0).score, 42.0f);
  EXPECT_EQ(result.Get(0).index, 0);
  // Second result should remain at infinity (unfilled).
  EXPECT_EQ(result.Get(1).score, inf);
}

HWY_NOINLINE void TestReduceTop1AllInf() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const float inf = std::numeric_limits<float>::max();

  HWY_ALIGN float top_v_buf[HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t top_idx_buf[HWY_MAX_BYTES / sizeof(int32_t)];
  hn::Store(hn::Set(df, inf), df, top_v_buf);
  hn::Store(hn::Set(di32, -1), di32, top_idx_buf);

  FixedTopKResult<1> result;
  ReduceTopK<1>(top_v_buf, top_idx_buf, result);
  EXPECT_EQ(result.Get(0).score, inf);
  EXPECT_EQ(result.Get(0).index, -1);
}

HWY_NOINLINE void TestReduceTopKPartialFill() {
  // Only insert enough values to fill K-1 of the K slots.
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  constexpr size_t K = 3;
  const float inf = std::numeric_limits<float>::max();

  HWY_ALIGN float top_v_buf[K * HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t top_idx_buf[K * HWY_MAX_BYTES / sizeof(int32_t)];
  for (size_t k = 0; k < K; ++k) {
    hn::Store(hn::Set(df, inf), df, top_v_buf + k * NF);
    hn::Store(hn::Set(di32, -1), di32, top_idx_buf + k * NF);
  }

  // Insert a single panel with 2 distinct values (< K=3).
  HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) scores[i] = inf;
  scores[0] = 1.0f;
  if (NF > 1) scores[1] = 2.0f;

  size_t num_valid = NF > 1 ? 2 : 1;
  UpdateTopK<K>(hn::Load(df, scores), 0, num_valid, top_v_buf, top_idx_buf);

  FixedTopKResult<K> result;
  ReduceTopK<K>(top_v_buf, top_idx_buf, result);
  EXPECT_FLOAT_EQ(result.Get(0).score, 1.0f);
  if (NF > 1) {
    EXPECT_FLOAT_EQ(result.Get(1).score, 2.0f);
    EXPECT_EQ(result.Get(2).score, inf);
  }
}

HWY_NOINLINE void TestRegTop1Basic() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  const float inf = std::numeric_limits<float>::max();

  auto best_score = hn::Set(df, inf);
  auto best_idx = hn::Set(di32, -1);

  // Panel 0: min at lane 0 = 15.0f
  HWY_ALIGN float s0[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) s0[i] = 100.0f;
  s0[0] = 15.0f;
  UpdateTopK1InRegisters(hn::Load(df, s0), 0, 2 * NF, best_score, best_idx);

  // Panel 1: min at last lane = -5.0f
  HWY_ALIGN float s1[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) s1[i] = 100.0f;
  s1[NF - 1] = -5.0f;
  UpdateTopK1InRegisters(hn::Load(df, s1), NF, 2 * NF, best_score, best_idx);

  FixedTopKResult<1> result;
  HorizontalReduceTopK1FromRegisters(best_score, best_idx, result);
  EXPECT_FLOAT_EQ(result.Get(0).score, -5.0f);
  EXPECT_EQ(result.Get(0).index, static_cast<int32_t>(NF + NF - 1));
}

HWY_NOINLINE void TestRegTop3Basic() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  const float inf = std::numeric_limits<float>::max();

  auto score0 = hn::Set(df, inf);
  auto score1 = hn::Set(df, inf);
  auto score2 = hn::Set(df, inf);
  auto idx0 = hn::Set(di32, -1);
  auto idx1 = hn::Set(di32, -1);
  auto idx2 = hn::Set(di32, -1);

  // Feed 4 panels with distinct decreasing scores to test displacement cascade
  for (size_t p = 0; p < 4; ++p) {
    HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
    for (size_t i = 0; i < NF; ++i) {
      scores[i] = 100.0f - static_cast<float>(p * NF + i);
    }
    UpdateTopK3InRegisters(hn::Load(df, scores), p * NF, 4 * NF, score0, score1,
                           score2, idx0, idx1, idx2);
  }

  FixedTopKResult<3> result;
  HorizontalReduceTopK3FromRegisters(score0, score1, score2, idx0, idx1, idx2,
                                     result);

  // The 3 absolute minimums should be at the very end of the 4*NF points
  float best0 = 100.0f - static_cast<float>(4 * NF - 1);
  float best1 = 100.0f - static_cast<float>(4 * NF - 2);
  float best2 = 100.0f - static_cast<float>(4 * NF - 3);

  EXPECT_FLOAT_EQ(result.Get(0).score, best0);
  EXPECT_FLOAT_EQ(result.Get(1).score, best1);
  EXPECT_FLOAT_EQ(result.Get(2).score, best2);
  EXPECT_EQ(result.Get(0).index, static_cast<int32_t>(4 * NF - 1));
  EXPECT_EQ(result.Get(1).index, static_cast<int32_t>(4 * NF - 2));
  EXPECT_EQ(result.Get(2).index, static_cast<int32_t>(4 * NF - 3));
}

HWY_NOINLINE void TestRegTop3PaddingMasked() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  if (NF < 2) return;
  const float inf = std::numeric_limits<float>::max();

  auto score0 = hn::Set(df, inf);
  auto score1 = hn::Set(df, inf);
  auto score2 = hn::Set(df, inf);
  auto idx0 = hn::Set(di32, -1);
  auto idx1 = hn::Set(di32, -1);
  auto idx2 = hn::Set(di32, -1);

  // 1 valid point out of NF lanes. Padding lanes have deceptive -inf scores.
  HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
  scores[0] = 10.0f;
  for (size_t i = 1; i < NF; ++i) scores[i] = -INFINITY;

  UpdateTopK3InRegisters(hn::Load(df, scores), 0, 1, score0, score1, score2,
                         idx0, idx1, idx2);

  FixedTopKResult<3> result;
  HorizontalReduceTopK3FromRegisters(score0, score1, score2, idx0, idx1, idx2,
                                     result);

  EXPECT_FLOAT_EQ(result.Get(0).score, 10.0f);
  EXPECT_EQ(result.Get(0).index, 0);
  EXPECT_EQ(result.Get(1).score, inf);
  EXPECT_EQ(result.Get(2).score, inf);
}

HWY_NOINLINE void TestAdversarialNaNPoisoning() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  const float inf = std::numeric_limits<float>::max();

  auto score0 = hn::Set(df, inf);
  auto score1 = hn::Set(df, inf);
  auto score2 = hn::Set(df, inf);
  auto idx0 = hn::Set(di32, -1);
  auto idx1 = hn::Set(di32, -1);
  auto idx2 = hn::Set(di32, -1);

  // Panel 0: Valid scores
  HWY_ALIGN float s0[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) s0[i] = static_cast<float>(i + 1);
  UpdateTopK3InRegisters(hn::Load(df, s0), 0, 2 * NF, score0, score1, score2,
                         idx0, idx1, idx2);

  // Panel 1: All NaNs (should be completely ignored and not infect state)
  HWY_ALIGN float s1[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) s1[i] = NAN;
  UpdateTopK3InRegisters(hn::Load(df, s1), NF, 2 * NF, score0, score1, score2,
                         idx0, idx1, idx2);

  FixedTopKResult<3> result;
  HorizontalReduceTopK3FromRegisters(score0, score1, score2, idx0, idx1, idx2,
                                     result);

  EXPECT_FLOAT_EQ(result.Get(0).score, 1.0f);
  EXPECT_EQ(result.Get(0).index, 0);
  if (NF >= 3) {
    EXPECT_FLOAT_EQ(result.Get(1).score, 2.0f);
    EXPECT_FLOAT_EQ(result.Get(2).score, 3.0f);
  }
}

HWY_NOINLINE void TestAdversarialMatrixTie() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  const float inf = std::numeric_limits<float>::max();

  auto score0 = hn::Set(df, inf);
  auto score1 = hn::Set(df, inf);
  auto score2 = hn::Set(df, inf);
  auto idx0 = hn::Set(di32, -1);
  auto idx1 = hn::Set(di32, -1);
  auto idx2 = hn::Set(di32, -1);

  // Feed 4 panels with the exact same score (0.0f) across all lanes.
  // This forces horizontal reduction to mask out winners one by one without
  // duplication or infinite looping.
  for (size_t p = 0; p < 4; ++p) {
    HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
    for (size_t i = 0; i < NF; ++i) scores[i] = 0.0f;
    UpdateTopK3InRegisters(hn::Load(df, scores), p * NF, 4 * NF, score0, score1,
                           score2, idx0, idx1, idx2);
  }

  FixedTopKResult<3> result;
  HorizontalReduceTopK3FromRegisters(score0, score1, score2, idx0, idx1, idx2,
                                     result);

  EXPECT_FLOAT_EQ(result.Get(0).score, 0.0f);
  EXPECT_FLOAT_EQ(result.Get(1).score, 0.0f);
  EXPECT_FLOAT_EQ(result.Get(2).score, 0.0f);
  // Indices should be valid and distinct
  EXPECT_NE(result.Get(0).index, result.Get(1).index);
  EXPECT_NE(result.Get(0).index, result.Get(2).index);
  EXPECT_NE(result.Get(1).index, result.Get(2).index);
}

HWY_NOINLINE void TestAdversarialSequenceBestFirst() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  constexpr size_t K = 3;
  const size_t num_points = 4 * NF;
  const float inf = std::numeric_limits<float>::max();

  HWY_ALIGN float top_v_buf[K * HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t top_idx_buf[K * HWY_MAX_BYTES / sizeof(int32_t)];
  for (size_t k = 0; k < K; ++k) {
    hn::Store(hn::Set(df, inf), df, top_v_buf + k * NF);
    hn::Store(hn::Set(di32, -1), di32, top_idx_buf + k * NF);
  }

  // Panel 0 has absolute best scores (-100.0f)
  HWY_ALIGN float s0[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) s0[i] = -100.0f + static_cast<float>(i);
  UpdateTopK<K>(hn::Load(df, s0), 0, num_points, top_v_buf, top_idx_buf);

  // Panels 1, 2, 3 have progressively worse scores (should be rejected early by
  // AllTrue(Ge(...)))
  for (size_t p = 1; p < 4; ++p) {
    HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
    for (size_t i = 0; i < NF; ++i) scores[i] = static_cast<float>(p * 100 + i);
    UpdateTopK<K>(hn::Load(df, scores), p * NF, num_points, top_v_buf,
                  top_idx_buf);
  }

  FixedTopKResult<K> result;
  ReduceTopK<K>(top_v_buf, top_idx_buf, result);

  EXPECT_FLOAT_EQ(result.Get(0).score, -100.0f);
  if (NF >= 3) {
    EXPECT_FLOAT_EQ(result.Get(1).score, -99.0f);
    EXPECT_FLOAT_EQ(result.Get(2).score, -98.0f);
    EXPECT_EQ(result.Get(0).index, 0);
    EXPECT_EQ(result.Get(1).index, 1);
    EXPECT_EQ(result.Get(2).index, 2);
  }
}

HWY_NOINLINE void TestUpdateTop10LargeK() {
  const hn::ScalableTag<float> df;
  const hn::ScalableTag<int32_t> di32;
  const size_t NF = hn::Lanes(df);
  constexpr size_t K = 10;
  const size_t num_points = 20 * NF;
  const float inf = std::numeric_limits<float>::max();

  HWY_ALIGN float top_v_buf[K * HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t top_idx_buf[K * HWY_MAX_BYTES / sizeof(int32_t)];
  for (size_t k = 0; k < K; ++k) {
    hn::Store(hn::Set(df, inf), df, top_v_buf + k * NF);
    hn::Store(hn::Set(di32, -1), di32, top_idx_buf + k * NF);
  }

  // Insert 20 panels with decreasing scores
  for (size_t p = 0; p < 20; ++p) {
    HWY_ALIGN float scores[HWY_MAX_BYTES / sizeof(float)];
    for (size_t i = 0; i < NF; ++i) {
      scores[i] = 1000.0f - static_cast<float>(p * NF + i);
    }
    UpdateTopK<K>(hn::Load(df, scores), p * NF, num_points, top_v_buf,
                  top_idx_buf);
  }

  FixedTopKResult<K> result;
  ReduceTopK<K>(top_v_buf, top_idx_buf, result);

  for (size_t k = 0; k < K; ++k) {
    float expected = 1000.0f - static_cast<float>(20 * NF - 1 - k);
    EXPECT_FLOAT_EQ(result.Get(k).score, expected) << "k=" << k;
    EXPECT_EQ(result.Get(k).index, static_cast<int32_t>(20 * NF - 1 - k))
        << "k=" << k;
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {
namespace {
HWY_BEFORE_TEST(TopkTrackerTest);

HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestInsertTopKBasic);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestInsertTopKRejection);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestInsertTopKDisplacement);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestInsertTopK1);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestUpdateTop1SinglePanel);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestUpdateTop1MultiplePanels);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestUpdateTop1PaddingMasked);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestUpdateTopKBasic);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestUpdateTopKPaddingMasked);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestReduceTop1AllInf);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestReduceTopKPartialFill);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestRegTop1Basic);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestRegTop3Basic);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestRegTop3PaddingMasked);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestAdversarialNaNPoisoning);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestAdversarialMatrixTie);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestAdversarialSequenceBestFirst);
HWY_EXPORT_AND_TEST_P(TopkTrackerTest, TestUpdateTop10LargeK);
HWY_AFTER_TEST();
}  // namespace
}  // namespace ginger
HWY_TEST_MAIN();
#endif
