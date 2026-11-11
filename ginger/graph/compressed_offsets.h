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

#ifndef GINGER_GRAPH_COMPRESSED_OFFSETS_H_
#define GINGER_GRAPH_COMPRESSED_OFFSETS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/prefetch.h"
#include "absl/log/absl_check.h"

namespace ginger {

// Compact storage for monotonically nondecreasing CSR graph offsets.
//
// Precondition: Input offsets must be nonnegative and monotonically
// nondecreasing (`0 <= offsets[0] <= offsets[1] <= ...`), as in a CSR graph.
//
// Automatically chooses the most compact valid physical representation:
// - `Mode::kBlock16`: One 64-byte cache-line-aligned block per 28 entries,
//   containing a 64-bit base plus 28 16-bit relative displacements. Uses
//   ~2.286 bytes/entry and guarantees that every neighbor range access touches
//   strictly a single 64-byte CPU cache line (including block boundaries).
// - `Mode::kFlat32`: Flat 32-bit unsigned offsets (`std::vector<uint32_t>`).
// - `Mode::kFlat64`: Flat 64-bit signed offsets (`std::vector<int64_t>`).
class CompressedOffsets {
 public:
  static constexpr size_t kBlockSize = 28;

  enum class Mode : uint8_t {
    kBlock16 = 0,
    kFlat32 = 1,
    kFlat64 = 2,
  };

  struct alignas(64) CacheLineBlock {
    int64_t base;
    uint16_t rel[kBlockSize];

    bool operator==(const CacheLineBlock& other) const = default;
  };
  static_assert(sizeof(CacheLineBlock) == 64);

  explicit CompressedOffsets(std::vector<int64_t> offsets)
      : size_(offsets.size()) {
    ABSL_DCHECK(std::is_sorted(offsets.begin(), offsets.end()));
    ABSL_DCHECK(offsets.empty() || offsets.front() >= 0);

    const bool fits32 =
        offsets.empty() || static_cast<uint64_t>(offsets.back()) <=
                               std::numeric_limits<uint32_t>::max();

    const size_t num_blocks = size_ == 0 ? 0 : ((size_ - 1) / kBlockSize + 1);
    const size_t block16_bytes = num_blocks * sizeof(CacheLineBlock);
    const size_t flat_bytes =
        size_ * (fits32 ? sizeof(uint32_t) : sizeof(int64_t));

    if (block16_bytes < flat_bytes) {
      // Because input offsets are monotonically nondecreasing, checking
      // last - base per block suffices to verify whether all entries in the
      // block fit in uint16_t.
      bool fits = true;
      for (size_t b = 0; b < num_blocks; ++b) {
        const size_t block_start = b * kBlockSize;
        const size_t block_last_idx = std::min((b + 1) * kBlockSize, size_ - 1);
        const int64_t base = offsets[block_start];
        const int64_t last = offsets[block_last_idx];
        if (static_cast<uint64_t>(last - base) >
            std::numeric_limits<uint16_t>::max()) {
          fits = false;
          break;
        }
      }
      if (fits) {
        std::vector<CacheLineBlock> blocks(num_blocks);
        for (size_t b = 0; b < num_blocks; ++b) {
          const size_t block_start = b * kBlockSize;
          const int64_t base = offsets[block_start];
          blocks[b].base = base;
          for (size_t j = 0; j < kBlockSize; ++j) {
            const size_t idx = std::min(block_start + 1 + j, size_ - 1);
            blocks[b].rel[j] = static_cast<uint16_t>(offsets[idx] - base);
          }
        }
        storage_ = std::move(blocks);
        return;
      }
    }

    if (fits32) {
      storage_ = std::vector<uint32_t>(offsets.begin(), offsets.end());
    } else {
      storage_ = std::move(offsets);
    }
  }

  int64_t operator[](size_t i) const {
    ABSL_DCHECK_LT(i, size_);
    if (const auto* blocks =
            std::get_if<std::vector<CacheLineBlock>>(&storage_)) {
      const size_t b = i / kBlockSize;
      const size_t k = i % kBlockSize;
      const CacheLineBlock& block = (*blocks)[b];
      return block.base + (k == 0 ? 0 : block.rel[k - 1]);
    }
    if (const auto* narrow = std::get_if<std::vector<uint32_t>>(&storage_)) {
      return (*narrow)[i];
    }
    return (*std::get_if<std::vector<int64_t>>(&storage_))[i];
  }

  // Returns `{offsets[i], offsets[i + 1]}` from a single 64-byte cache line
  // without cross-cache-line boundary crossings or secondary base lookups.
  std::pair<int64_t, int64_t> GetRange(size_t i) const {
    ABSL_DCHECK_LT(i, size_);
    ABSL_DCHECK_LT(i + 1, size_);
    if (const auto* blocks =
            std::get_if<std::vector<CacheLineBlock>>(&storage_)) {
      const size_t b = i / kBlockSize;
      const size_t k = i % kBlockSize;
      const CacheLineBlock& block = (*blocks)[b];
      const int64_t start = block.base + (k == 0 ? 0 : block.rel[k - 1]);
      const int64_t end = block.base + block.rel[k];
      return {start, end};
    }
    if (const auto* narrow = std::get_if<std::vector<uint32_t>>(&storage_)) {
      return {static_cast<int64_t>((*narrow)[i]),
              static_cast<int64_t>((*narrow)[i + 1])};
    }
    const auto& wide = *std::get_if<std::vector<int64_t>>(&storage_);
    return {wide[i], wide[i + 1]};
  }

  // Prefetches the cache line containing the offset range for index `i`.
  void PrefetchRange(size_t i) const {
    ABSL_DCHECK_LT(i + 1, size_);

    if (const auto* blocks =
            std::get_if<std::vector<CacheLineBlock>>(&storage_)) {
      absl::PrefetchToLocalCache(&(*blocks)[i / kBlockSize]);
      return;
    }

    // In flat modes, offsets[i] and offsets[i + 1] straddle cache lines every
    // 16th (kFlat32) or 8th (kFlat64) entry. We intentionally do not prefetch
    // i + 1 as well because it provides no measurable benefit on most graphs
    // and can cause performance degradations on some datasets.
    if (const auto* narrow = std::get_if<std::vector<uint32_t>>(&storage_)) {
      absl::PrefetchToLocalCache(&(*narrow)[i]);
      return;
    }

    const auto& wide = *std::get_if<std::vector<int64_t>>(&storage_);
    absl::PrefetchToLocalCache(&wide[i]);
  }

  size_t size() const { return size_; }
  Mode mode() const { return static_cast<Mode>(storage_.index()); }

  size_t NumBytes() const {
    if (const auto* blocks =
            std::get_if<std::vector<CacheLineBlock>>(&storage_)) {
      return blocks->size() * sizeof(CacheLineBlock);
    }
    if (const auto* narrow = std::get_if<std::vector<uint32_t>>(&storage_)) {
      return narrow->size() * sizeof(uint32_t);
    }
    return std::get_if<std::vector<int64_t>>(&storage_)->size() *
           sizeof(int64_t);
  }

  std::vector<int64_t> ToVector() const {
    if (const auto* wide = std::get_if<std::vector<int64_t>>(&storage_)) {
      return *wide;
    }
    if (const auto* narrow = std::get_if<std::vector<uint32_t>>(&storage_)) {
      return std::vector<int64_t>(narrow->begin(), narrow->end());
    }
    std::vector<int64_t> out(size_);
    for (size_t i = 0; i < size_; ++i) {
      out[i] = (*this)[i];
    }
    return out;
  }

  bool operator==(const CompressedOffsets& other) const = default;
  bool operator!=(const CompressedOffsets& other) const = default;

 private:
  size_t size_;
  std::variant<std::vector<CacheLineBlock>, std::vector<uint32_t>,
               std::vector<int64_t>>
      storage_;
};

}  // namespace ginger

#endif  // GINGER_GRAPH_COMPRESSED_OFFSETS_H_
