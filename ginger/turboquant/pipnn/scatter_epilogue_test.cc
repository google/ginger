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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "ginger/internal/gunit_support.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/pipnn/scatter_epilogue_test.cc"
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
#include "ginger/turboquant/pipnn/scatter_epilogue-inl.h"  // NOLINT

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

using ginger::HWY_NAMESPACE::ScatterSingleQuery;

HWY_NOINLINE void TestScatterSingleQueryBasic() {
  using F32 = hn::ScalableTag<float>;
  using I32 = hn::ScalableTag<int32_t>;
  const F32 df;
  const I32 di32;
  const size_t NF = hn::Lanes(df);
  const float inf = std::numeric_limits<float>::max();

  // Scores: simple distinct values.
  HWY_ALIGN float score_data[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) {
    score_data[i] = 10.0f + static_cast<float>(i) * 2.0f;
  }
  hn::Vec<F32> scores_v = hn::Load(df, score_data);

  size_t query_index = 0;
  size_t base = 999;
  size_t num_points = 100000;
  hn::Vec<I32> lane_idx = hn::Iota(di32, 0);
  hn::Vec<F32> inf_v = hn::Set(df, inf);

  // Query-side trackers.
  hn::Vec<F32> min1_v = hn::Set(df, inf);
  hn::Vec<F32> min2_v = hn::Set(df, inf);
  hn::Vec<I32> min1_idx_v = hn::Set(di32, -1);
  hn::Vec<I32> min2_idx_v = hn::Set(di32, -1);

  // DB-side trackers.
  hn::Vec<F32> dm1 = hn::Set(df, inf);
  hn::Vec<F32> dm2 = hn::Set(df, inf);
  hn::Vec<I32> di1 = hn::Set(di32, -1);
  hn::Vec<I32> di2 = hn::Set(di32, -1);

  hn::Vec<I32> base_indices =
      hn::Add(hn::Set(di32, static_cast<int32_t>(base)), lane_idx);
  ScatterSingleQuery(df, di32, scores_v, query_index, base, num_points,
                     base_indices, inf_v, min1_v, min2_v, min1_idx_v,
                     min2_idx_v, dm1, dm2, di1, di2);

  // Query-side and DB-side get the same symmetric score.
  HWY_ALIGN float m1_out[HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t m1_idx_out[HWY_MAX_BYTES / sizeof(int32_t)];
  hn::Store(min1_v, df, m1_out);
  hn::Store(min1_idx_v, di32, m1_idx_out);
  for (size_t i = 0; i < NF; ++i) {
    EXPECT_FLOAT_EQ(m1_out[i], score_data[i]) << "lane=" << i;
    EXPECT_EQ(m1_idx_out[i], static_cast<int32_t>(base + i)) << "lane=" << i;
  }

  HWY_ALIGN float dm1_out[HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t dm1_idx_out[HWY_MAX_BYTES / sizeof(int32_t)];
  hn::Store(dm1, df, dm1_out);
  hn::Store(di1, di32, dm1_idx_out);
  for (size_t i = 0; i < NF; ++i) {
    EXPECT_FLOAT_EQ(dm1_out[i], score_data[i]) << "lane=" << i;
    EXPECT_EQ(dm1_idx_out[i], static_cast<int32_t>(query_index))
        << "lane=" << i;
  }
}

HWY_NOINLINE void TestScatterEpilogueSelfMask() {
  using F32 = hn::ScalableTag<float>;
  using I32 = hn::ScalableTag<int32_t>;
  const F32 df;
  const I32 di32;
  const size_t NF = hn::Lanes(df);
  const float inf = std::numeric_limits<float>::max();

  hn::Vec<F32> scores_v = hn::Set(df, 5.0f);

  // Query index is at base + 0, so lane 0 should be self-masked.
  size_t query_index = 0;
  size_t base = 0;
  size_t num_points = NF;
  hn::Vec<I32> lane_idx = hn::Iota(di32, 0);
  hn::Vec<F32> inf_v = hn::Set(df, inf);

  hn::Vec<F32> min1_v = hn::Set(df, inf);
  hn::Vec<F32> min2_v = hn::Set(df, inf);
  hn::Vec<I32> min1_idx_v = hn::Set(di32, -1);
  hn::Vec<I32> min2_idx_v = hn::Set(di32, -1);
  hn::Vec<F32> dm1 = hn::Set(df, inf);
  hn::Vec<F32> dm2 = hn::Set(df, inf);
  hn::Vec<I32> di1 = hn::Set(di32, -1);
  hn::Vec<I32> di2 = hn::Set(di32, -1);

  hn::Vec<I32> base_indices =
      hn::Add(hn::Set(di32, static_cast<int32_t>(base)), lane_idx);
  ScatterSingleQuery(df, di32, scores_v, query_index, base, num_points,
                     base_indices, inf_v, min1_v, min2_v, min1_idx_v,
                     min2_idx_v, dm1, dm2, di1, di2);

  // Lane 0 should be +inf on both query and DB side.
  HWY_ALIGN float m1_out[HWY_MAX_BYTES / sizeof(float)];
  hn::Store(min1_v, df, m1_out);
  // The tracker is per-lane: lane 0 only saw its own self-masked +inf.
  EXPECT_EQ(m1_out[0], inf) << "Self-masked lane should remain +inf";
  if (NF > 1) {
    EXPECT_NE(m1_out[1], inf) << "Non-self lane should have a real score";
  }

  // DB-side lane 0 should also be self-masked.
  HWY_ALIGN float dm1_out[HWY_MAX_BYTES / sizeof(float)];
  hn::Store(dm1, df, dm1_out);
  EXPECT_EQ(dm1_out[0], inf) << "Self-distance not masked on DB side";
  if (NF > 1) {
    EXPECT_NE(dm1_out[1], inf) << "Non-self lane wrongly masked";
  }
}

HWY_NOINLINE void TestScatterEpilogueQueryMin2Tracker() {
  using F32 = hn::ScalableTag<float>;
  using I32 = hn::ScalableTag<int32_t>;
  const F32 df;
  const I32 di32;
  const size_t NF = hn::Lanes(df);
  if (NF < 2) return;
  const float inf = std::numeric_limits<float>::max();

  // Put distinct scores in lanes so we can verify min-2 tracking.
  HWY_ALIGN float score_data[HWY_MAX_BYTES / sizeof(float)];
  for (size_t i = 0; i < NF; ++i) {
    score_data[i] = -100.0f - static_cast<float>(i) * 100.0f;
  }
  hn::Vec<F32> scores_v = hn::Load(df, score_data);

  size_t query_index = 0;
  size_t base = 999;
  size_t num_points = 100000;
  hn::Vec<I32> lane_idx = hn::Iota(di32, 0);
  hn::Vec<F32> inf_v = hn::Set(df, inf);

  hn::Vec<F32> min1_v = hn::Set(df, inf);
  hn::Vec<F32> min2_v = hn::Set(df, inf);
  hn::Vec<I32> min1_idx_v = hn::Set(di32, -1);
  hn::Vec<I32> min2_idx_v = hn::Set(di32, -1);
  hn::Vec<F32> dm1 = hn::Set(df, inf);
  hn::Vec<F32> dm2 = hn::Set(df, inf);
  hn::Vec<I32> di1 = hn::Set(di32, -1);
  hn::Vec<I32> di2 = hn::Set(di32, -1);

  hn::Vec<I32> base_indices =
      hn::Add(hn::Set(di32, static_cast<int32_t>(base)), lane_idx);
  ScatterSingleQuery(df, di32, scores_v, query_index, base, num_points,
                     base_indices, inf_v, min1_v, min2_v, min1_idx_v,
                     min2_idx_v, dm1, dm2, di1, di2);

  HWY_ALIGN float m1_out[HWY_MAX_BYTES / sizeof(float)];
  HWY_ALIGN int32_t m1_idx_out[HWY_MAX_BYTES / sizeof(int32_t)];
  hn::Store(min1_v, df, m1_out);
  hn::Store(min1_idx_v, di32, m1_idx_out);

  for (size_t i = 0; i < NF; ++i) {
    EXPECT_FLOAT_EQ(m1_out[i], score_data[i]) << "lane=" << i;
    EXPECT_EQ(m1_idx_out[i], static_cast<int32_t>(base + i)) << "lane=" << i;
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(ScatterEpilogueTest);

HWY_EXPORT_AND_TEST_P(ScatterEpilogueTest, TestScatterSingleQueryBasic);
HWY_EXPORT_AND_TEST_P(ScatterEpilogueTest, TestScatterEpilogueSelfMask);
HWY_EXPORT_AND_TEST_P(ScatterEpilogueTest, TestScatterEpilogueQueryMin2Tracker);

HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif
