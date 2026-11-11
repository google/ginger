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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"
#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "ginger/types.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ginger/distance/hamming.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ginger {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Prefetches up to the first `kMaxPrefetchBytes` of candidate vector data of
// `words_per_point` 64-bit words from `ptr` into cache. Capping the prefetch
// per vector avoids saturating Line Fill Buffers on high-dimensional vectors
// while triggering the hardware streamer for any remaining sequential cache
// lines.
HWY_INLINE void PrefetchPoint(const uint64_t* HWY_RESTRICT ptr,
                              size_t words_per_point) {
  constexpr size_t kMaxPrefetchBytes = 256;
  const char* data = reinterpret_cast<const char*>(ptr);
  const size_t bytes =
      std::min(words_per_point * sizeof(uint64_t), kMaxPrefetchBytes);
  for (size_t offset = 0; offset < bytes; offset += 64) {
    absl::PrefetchToLocalCache(data + offset);
  }
}

// Computes the bitwise Hamming distance (the total number of differing bits)
// between a bit-packed `query` and each candidate point in `candidates` from
// `dataset` across `words_per_point` packed 64-bit words, writing the computed
// distances into the `candidates` span.
//
// On x86 targets without native 64-bit vector population count (SSE2 through
// AVX3/Skylake-X), Highway's 64-bit vector popcount falls back to an emulated
// multi-step shift/mask/add widening tree. On those targets, we reinterpret the
// 64-bit XOR result as 8-bit bytes, compute byte-level popcounts, and reduce
// groups of 8 adjacent byte counts into 64-bit lanes via `hn::SumsOf8`. On
// targets with native 64-bit vector popcount support, we use 64-bit
// `hn::PopulationCount` directly.
void ComputeHammingDistancesSimd(const uint64_t* HWY_RESTRICT query,
                                 const uint64_t* HWY_RESTRICT dataset,
                                 size_t words_per_point,
                                 absl::Span<DistanceAndId> candidates) {
  const hn::ScalableTag<uint64_t> d64;
  using VecU64 = hn::Vec<decltype(d64)>;
  const size_t N = hn::Lanes(d64);

  const size_t num_candidates = candidates.size();
  const auto get_point = [&](size_t idx) {
    return dataset + candidates[idx].id * words_per_point;
  };

  const auto accumulate = [&](VecU64& sum, const VecU64& q,
                              const uint64_t* HWY_RESTRICT p, size_t offset) {
    const VecU64 pt = hn::LoadU(d64, p + offset);
    const VecU64 diff = hn::Xor(q, pt);
#if HWY_TARGET >= HWY_AVX3 && HWY_TARGET <= HWY_SSE2
    const hn::Repartition<uint8_t, decltype(d64)> d8;
    const auto popcnt = hn::PopulationCount(hn::BitCast(d8, diff));
    sum = hn::Add(sum, hn::SumsOf8(popcnt));
#else
    sum = hn::Add(sum, hn::PopulationCount(diff));
#endif
  };

  constexpr size_t kPrefetchDistance = 8;
  for (size_t p = 0; p < kPrefetchDistance && p < num_candidates; ++p) {
    PrefetchPoint(get_point(p), words_per_point);
  }

  size_t j = 0;
  for (; j + 4 <= num_candidates; j += 4) {
    const size_t prefetch_end =
        std::min(j + kPrefetchDistance + 4, num_candidates);
    for (size_t p = j + kPrefetchDistance; p < prefetch_end; ++p) {
      PrefetchPoint(get_point(p), words_per_point);
    }

    const uint64_t* const HWY_RESTRICT p0 = get_point(j);
    const uint64_t* const HWY_RESTRICT p1 = get_point(j + 1);
    const uint64_t* const HWY_RESTRICT p2 = get_point(j + 2);
    const uint64_t* const HWY_RESTRICT p3 = get_point(j + 3);

    VecU64 sum0 = hn::Zero(d64);
    VecU64 sum1 = hn::Zero(d64);
    VecU64 sum2 = hn::Zero(d64);
    VecU64 sum3 = hn::Zero(d64);

    size_t w = 0;
    for (; w + N <= words_per_point; w += N) {
      const VecU64 q = hn::LoadU(d64, query + w);
      accumulate(sum0, q, p0, w);
      accumulate(sum1, q, p1, w);
      accumulate(sum2, q, p2, w);
      accumulate(sum3, q, p3, w);
    }

    uint64_t dist0 = hn::ReduceSum(d64, sum0);
    uint64_t dist1 = hn::ReduceSum(d64, sum1);
    uint64_t dist2 = hn::ReduceSum(d64, sum2);
    uint64_t dist3 = hn::ReduceSum(d64, sum3);

    for (; w < words_per_point; ++w) {
      const uint64_t qw = query[w];
      dist0 += absl::popcount(qw ^ p0[w]);
      dist1 += absl::popcount(qw ^ p1[w]);
      dist2 += absl::popcount(qw ^ p2[w]);
      dist3 += absl::popcount(qw ^ p3[w]);
    }

    candidates[j].distance = static_cast<float>(dist0);
    candidates[j + 1].distance = static_cast<float>(dist1);
    candidates[j + 2].distance = static_cast<float>(dist2);
    candidates[j + 3].distance = static_cast<float>(dist3);
  }

  for (; j < num_candidates; ++j) {
    const uint64_t* const HWY_RESTRICT pt_words = get_point(j);
    size_t w = 0;

    VecU64 sum = hn::Zero(d64);
    for (; w + N <= words_per_point; w += N) {
      const VecU64 q = hn::LoadU(d64, query + w);
      accumulate(sum, q, pt_words, w);
    }

    uint64_t dist = hn::ReduceSum(d64, sum);
    for (; w < words_per_point; ++w) {
      dist += absl::popcount(query[w] ^ pt_words[w]);
    }

    candidates[j].distance = static_cast<float>(dist);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {

HWY_EXPORT(ComputeHammingDistancesSimd);

void HammingDistance(absl::Span<const uint64_t> query,
                     absl::Span<DistanceAndId> candidates,
                     const uint64_t* dataset, size_t words_per_point) {
  ABSL_DCHECK_EQ(query.size(), words_per_point);
  HWY_DYNAMIC_DISPATCH(ComputeHammingDistancesSimd)(
      query.data(), dataset, words_per_point, candidates);
}

}  // namespace ginger

#endif  // HWY_ONCE
