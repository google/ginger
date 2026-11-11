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

#ifndef GINGER_TURBOQUANT_MANY_TO_MANY_INDEX_SUBSET_H_
#define GINGER_TURBOQUANT_MANY_TO_MANY_INDEX_SUBSET_H_

#include <cstddef>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"

namespace ginger {

// Encapsulates either a contiguous range of indices [start, start + count)
// or an explicit gathered subspan of indices. This provides a unified view
// for query and database indexing, avoiding redundant parameters and branches.
class IndexSubset {
 public:
  IndexSubset() = default;

  // Constructs a contiguous range of indices [start, start + count).
  static IndexSubset Contiguous(size_t start, size_t count) {
    IndexSubset subset;
    subset.start_ = start;
    subset.count_ = count;
    return subset;
  }

  // Constructs a gathered view from an explicit span of indices.
  static IndexSubset Gathered(absl::Span<const int> indices) {
    IndexSubset subset;
    subset.indices_ = indices;
    return subset;
  }

  // Returns true if this subset represents a contiguous range.
  bool is_contiguous() const { return indices_.empty(); }

  // Returns the number of indices in this subset.
  size_t size() const { return is_contiguous() ? count_ : indices_.size(); }

  // Returns true if the subset is empty.
  bool empty() const { return size() == 0; }

  // Accesses the i-th index in the subset.
  size_t operator[](size_t i) const {
    ABSL_DCHECK_LT(i, size());
    return is_contiguous() ? start_ + i : static_cast<size_t>(indices_[i]);
  }

  // Returns a subspan of this IndexSubset starting at `offset` with length
  // `len`.
  IndexSubset subspan(size_t offset, size_t len) const {
    ABSL_CHECK_LE(offset + len, size());
    if (is_contiguous()) {
      return IndexSubset::Contiguous(start_ + offset, len);
    } else {
      return IndexSubset::Gathered(indices_.subspan(offset, len));
    }
  }

  // Returns the underlying span of indices (empty if contiguous).
  absl::Span<const int> indices() const { return indices_; }

  // Returns the starting index (only valid if contiguous).
  size_t start() const {
    ABSL_DCHECK(is_contiguous());
    return start_;
  }

 private:
  absl::Span<const int> indices_;
  size_t start_ = 0;
  size_t count_ = 0;
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_MANY_TO_MANY_INDEX_SUBSET_H_
