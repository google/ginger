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

#ifndef GINGER_SEARCH_FAST_VISITED_TABLE_H_
#define GINGER_SEARCH_FAST_VISITED_TABLE_H_

#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"

namespace ginger {
namespace internal {

// A fast visited set implementation using a custom linear probing hash table.
// Supports fast reset by incrementing a generation counter, avoiding the need
// to clear memory on every search. Table capacity is always a power of two.
//
// Example usage:
//   FastVisitedTable visited_table;
//   if (visited_table.Insert(node_id)) {
//     // First time visiting `node_id` during this search.
//     ProcessNode(node_id);
//   }
//
//   // Prepare for a new search by fast-resetting the generation counter:
//   visited_table.Reset(min_capacity_for_next_search);
class FastVisitedTable {
 public:
  struct Slot {
    GingerNodeId id;
    uint32_t generation;
  };

  static constexpr size_t kDefaultCapacity = 1 << 14;

  // Initializes the visited table with the given initial capacity rounded up to
  // the next power of two.
  explicit FastVisitedTable(size_t initial_capacity = kDefaultCapacity) {
    Init(initial_capacity);
  }

  // Resets the visited table for a new search. Increments the generation
  // counter to invalidate previous entries. If `min_capacity` exceeds the
  // current table capacity, the table is reinitialized to at least that size
  // rounded up to the next power of two.
  void Reset(size_t min_capacity = 0) {
    if (table_.size() < min_capacity) {
      Init(min_capacity);
      return;
    }

    ++current_generation_;
    num_inserted_this_generation_ = 0;
    if (current_generation_ == 0) {
      // Handle overflow by explicitly resetting all slots
      for (Slot& slot : table_) {
        slot.id = -1;
        slot.generation = 0;
      }
      current_generation_ = 1;
    }
  }

  // Inserts `id` into the visited table. Returns true if `id` was not already
  // present in the current generation, false otherwise.
  bool Insert(GingerNodeId id) {
    uint64_t idx = GetIndex(id);
    if (table_[idx].generation == current_generation_) {
      if (table_[idx].id == id) {
        return false;
      }
      idx = (idx + 1) & mask_;
      while (table_[idx].generation == current_generation_) {
        if (table_[idx].id == id) {
          return false;
        }
        idx = (idx + 1) & mask_;
      }
    }
    table_[idx] = {id, current_generation_};
    ++num_inserted_this_generation_;

    if (num_inserted_this_generation_ > GrowthThreshold()) {
      Resize(capacity() * 2);
    }
    return true;
  }

  size_t capacity() const { return table_.size(); }
  uint32_t current_generation() const { return current_generation_; }
  void SetCurrentGenerationForTesting(uint32_t gen) {
    current_generation_ = gen;
  }
  absl::Span<const Slot> table() const { return table_; }

 private:
  // Magic hash constant derived from fractional part of the golden ratio to
  // scatter sequential keys as evenly and far apart as possible. floor(2^32 *
  // (sqrt(5) - 1) / 2) = 0x9E3779B9.
  static constexpr uint32_t kHashMultiplier = 0x9E3779B9U;

  void Init(size_t capacity = kDefaultCapacity) {
    capacity = std::bit_ceil(capacity);
    table_.assign(capacity, {-1, 0});
    shift_ = 32 - std::countr_zero(capacity);
    mask_ = capacity - 1;
    current_generation_ = 1;
    num_inserted_this_generation_ = 0;
  }

  uint64_t GetIndex(GingerNodeId id) const {
    return (static_cast<uint32_t>(id) * kHashMultiplier) >> shift_;
  }

  void Resize(size_t new_capacity) {
    ABSL_DCHECK(std::has_single_bit(new_capacity));
    hwy::AlignedVector<Slot> old_table = std::move(table_);
    table_.assign(new_capacity, {-1, 0});
    shift_ = 32 - std::countr_zero(new_capacity);
    mask_ = new_capacity - 1;
    for (const Slot& slot : old_table) {
      if (slot.generation == current_generation_) {
        uint64_t idx = GetIndex(slot.id);
        while (table_[idx].generation == current_generation_) {
          idx = (idx + 1) & mask_;
        }
        table_[idx] = slot;
      }
    }
  }

  // We maintain at least 50% occupancy. Since capacity (`mask_ + 1`) is a
  // power of two, we can derive the resize threshold directly from `mask_`
  // using `mask_ >> 1`.
  size_t GrowthThreshold() const { return mask_ >> 1; }

  hwy::AlignedVector<Slot> table_;
  // Right-shift count used in GetIndex() for multiplicative
  // hashing to map a 32-bit hash into [0, capacity - 1] (i.e., 32 - log2(cap)).
  uint32_t shift_;
  // Bitmask equal to (capacity - 1) used for fast bitwise modulo operations
  // during linear probing (since capacity is always a power of two).
  uint32_t mask_;
  uint32_t current_generation_;
  uint32_t num_inserted_this_generation_;
};

}  // namespace internal
}  // namespace ginger

#endif  // GINGER_SEARCH_FAST_VISITED_TABLE_H_
