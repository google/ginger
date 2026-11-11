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

#include "ginger/distance/hamming.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "ginger/internal/gunit_support.h"
#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "ginger/types.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE \
  "ginger/distance/hamming_test.cc"
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

using ::ginger::DistanceAndId;
using ::ginger::GingerNodeId;
using ::ginger::HammingDistance;

// Verifies that an empty candidates span is a no-op.
HWY_NOINLINE void EmptyCandidatesDoesNotCrash() {
  const std::vector<uint64_t> query = {1, 2};
  const std::vector<uint64_t> dataset = {1, 2};
  std::vector<DistanceAndId> candidates;

  HammingDistance(absl::MakeConstSpan(query), absl::MakeSpan(candidates),
                  dataset.data(), /*words_per_point=*/2);
  EXPECT_TRUE(candidates.empty());
}

// Verifies that words_per_point = 0 evaluates to distance 0.0f.
HWY_NOINLINE void ZeroWordsPerPointDoesNotCrash() {
  const std::vector<uint64_t> query = {};
  const std::vector<uint64_t> dataset = {};
  std::vector<DistanceAndId> candidates = {
      {.distance = -1.0f, .id = 0}, {.distance = -1.0f, .id = 1},
      {.distance = -1.0f, .id = 2}, {.distance = -1.0f, .id = 3},
      {.distance = -1.0f, .id = 4},
  };

  HammingDistance(absl::MakeConstSpan(query), absl::MakeSpan(candidates),
                  dataset.data(), /*words_per_point=*/0);
  for (size_t i = 0; i < candidates.size(); ++i) {
    EXPECT_EQ(candidates[i].id, static_cast<GingerNodeId>(i));
    EXPECT_FLOAT_EQ(candidates[i].distance, 0.0f)
        << "Mismatch at candidate index " << i;
  }
}

// Tests non-consecutive, out-of-order, and duplicate candidate IDs across both
// the 4-point batch loop and the single-point remainder loop.
HWY_NOINLINE void NonConsecutiveAndOutOfOrderCandidateIds() {
  constexpr size_t kWordsPerPoint = 4;
  constexpr size_t kTotalDatasetPoints = 100;
  std::mt19937_64 rng(12345);

  std::vector<uint64_t> query(kWordsPerPoint);
  for (size_t w = 0; w < kWordsPerPoint; ++w) {
    query[w] = rng();
  }

  std::vector<uint64_t> dataset(kTotalDatasetPoints * kWordsPerPoint);
  for (size_t i = 0; i < dataset.size(); ++i) {
    dataset[i] = rng();
  }

  const std::vector<GingerNodeId> test_ids = {42, 0, 99, 5, 50, 42, 1, 77, 13};
  std::vector<DistanceAndId> candidates(test_ids.size());
  for (size_t i = 0; i < test_ids.size(); ++i) {
    candidates[i].id = test_ids[i];
    candidates[i].distance = -1.0f;
  }

  HammingDistance(absl::MakeConstSpan(query), absl::MakeSpan(candidates),
                  dataset.data(), kWordsPerPoint);

  for (size_t i = 0; i < candidates.size(); ++i) {
    EXPECT_EQ(candidates[i].id, test_ids[i]);
    const uint64_t* const pt =
        dataset.data() + candidates[i].id * kWordsPerPoint;
    uint64_t expected_dist = 0;
    for (size_t w = 0; w < kWordsPerPoint; ++w) {
      expected_dist += absl::popcount(query[w] ^ pt[w]);
    }
    EXPECT_FLOAT_EQ(candidates[i].distance, static_cast<float>(expected_dist))
        << "Mismatch at candidate index " << i << " with ID "
        << candidates[i].id;
  }
}

// Tests extreme bit patterns (all-zeros, all-ones, alternating 0xAA/0x55)
// across multiple words per point.
HWY_NOINLINE void ExtremeBitPatterns() {
  for (size_t words_per_point : {1, 3, 4, 7, 8, 9, 16}) {
    const std::vector<uint64_t> all_zeros_query(words_per_point, 0ULL);

    std::vector<uint64_t> dataset(5 * words_per_point);
    for (size_t w = 0; w < words_per_point; ++w) {
      dataset[0 * words_per_point + w] = 0ULL;
      dataset[1 * words_per_point + w] = ~0ULL;
      dataset[2 * words_per_point + w] = 0xAAAAAAAAAAAAAAAAULL;
      dataset[3 * words_per_point + w] = 0x5555555555555555ULL;
      dataset[4 * words_per_point + w] = 1ULL;
    }

    std::vector<DistanceAndId> candidates = {
        {.distance = -1.0f, .id = 0}, {.distance = -1.0f, .id = 1},
        {.distance = -1.0f, .id = 2}, {.distance = -1.0f, .id = 3},
        {.distance = -1.0f, .id = 4},
    };

    HammingDistance(absl::MakeConstSpan(all_zeros_query),
                    absl::MakeSpan(candidates), dataset.data(),
                    words_per_point);

    for (size_t i = 0; i < candidates.size(); ++i) {
      EXPECT_EQ(candidates[i].id, static_cast<GingerNodeId>(i));
    }
    EXPECT_FLOAT_EQ(candidates[0].distance, 0.0f);
    EXPECT_FLOAT_EQ(candidates[1].distance,
                    static_cast<float>(64 * words_per_point));
    EXPECT_FLOAT_EQ(candidates[2].distance,
                    static_cast<float>(32 * words_per_point));
    EXPECT_FLOAT_EQ(candidates[3].distance,
                    static_cast<float>(32 * words_per_point));
    EXPECT_FLOAT_EQ(candidates[4].distance,
                    static_cast<float>(1 * words_per_point));
  }
}

// Tests varied word counts (covering SIMD lane widths, scalar tails, and the
// 256-byte / 32-word prefetch cap) and candidate counts (covering 4-point
// batches, single-point tails, and prefetch distance boundaries).
HWY_NOINLINE void VariedDimensionsAndCandidatesCoverAllUnrollingBranches() {
  const std::vector<size_t> test_words_per_point = {
      1, 2, 3, 4, 5, 7, 8, 9, 13, 15, 16, 17, 31, 32, 33, 64};
  const std::vector<size_t> test_candidate_counts = {1, 2,  3,  4,  5, 8,
                                                     9, 11, 12, 16, 55};

  std::mt19937_64 rng(42);

  for (size_t words_per_point : test_words_per_point) {
    for (size_t num_candidates : test_candidate_counts) {
      std::vector<uint64_t> query(words_per_point);
      for (size_t w = 0; w < words_per_point; ++w) {
        query[w] = rng();
      }

      std::vector<uint64_t> dataset(num_candidates * words_per_point);
      for (size_t i = 0; i < dataset.size(); ++i) {
        dataset[i] = rng();
      }

      std::vector<DistanceAndId> candidates(num_candidates);
      for (size_t i = 0; i < num_candidates; ++i) {
        candidates[i].id = static_cast<GingerNodeId>(i);
        candidates[i].distance = -1.0f;
      }

      HammingDistance(absl::MakeConstSpan(query), absl::MakeSpan(candidates),
                      dataset.data(), words_per_point);

      for (size_t i = 0; i < num_candidates; ++i) {
        EXPECT_EQ(candidates[i].id, static_cast<GingerNodeId>(i));
        const uint64_t* const pt = dataset.data() + i * words_per_point;
        uint64_t expected_dist = 0;
        for (size_t w = 0; w < words_per_point; ++w) {
          expected_dist += absl::popcount(query[w] ^ pt[w]);
        }
        EXPECT_FLOAT_EQ(candidates[i].distance,
                        static_cast<float>(expected_dist))
            << "Mismatch at words_per_point=" << words_per_point
            << ", num_candidates=" << num_candidates
            << ", candidate_index=" << i;
      }
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

HWY_BEFORE_TEST(DistanceHammingTest);
HWY_EXPORT_AND_TEST_P(DistanceHammingTest, EmptyCandidatesDoesNotCrash);
HWY_EXPORT_AND_TEST_P(DistanceHammingTest, ZeroWordsPerPointDoesNotCrash);
HWY_EXPORT_AND_TEST_P(DistanceHammingTest,
                      NonConsecutiveAndOutOfOrderCandidateIds);
HWY_EXPORT_AND_TEST_P(DistanceHammingTest, ExtremeBitPatterns);
HWY_EXPORT_AND_TEST_P(DistanceHammingTest,
                      VariedDimensionsAndCandidatesCoverAllUnrollingBranches);
HWY_AFTER_TEST();

TEST(DistanceHammingDeathTest, QueryWordCountMismatch) {
  const std::vector<uint64_t> query = {1, 2, 3};
  const std::vector<uint64_t> dataset = {1, 2};
  std::vector<ginger::DistanceAndId> candidates = {{.id = 0}};
  EXPECT_DEBUG_DEATH(
      ginger::HammingDistance(absl::MakeConstSpan(query),
                              absl::MakeSpan(candidates), dataset.data(),
                              /*words_per_point=*/2),
      "");
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
#endif
