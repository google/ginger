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

#ifndef GINGER_DISTANCE_CHAMFER_H_
#define GINGER_DISTANCE_CHAMFER_H_

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/types.h"

namespace ginger {

// A dataset of multi-vector point clouds for Chamfer distance computation.
//
// Each document is a point cloud: a variable-length sequence of token
// embeddings of the same dimensionality. Internally, documents are stored in a
// transposed block layout for better scoring performance.
class ChamferDataset {
 public:
  ChamferDataset(size_t dimensionality, std::vector<size_t> doc_offsets)
      : tokens_(doc_offsets.empty() ? 0 : doc_offsets.back(), dimensionality),
        size_(doc_offsets.empty()
                  ? 0
                  : static_cast<GingerNodeId>(doc_offsets.size() - 1)),
        doc_offsets_(std::move(doc_offsets)) {
    if (!doc_offsets_.empty()) {
      ABSL_CHECK_EQ(doc_offsets_.front(), 0);
    }
    ABSL_DCHECK(std::is_sorted(doc_offsets_.begin(), doc_offsets_.end()));
  }

  ChamferDataset(SpanDataset tokens, std::vector<size_t> doc_offsets)
      : tokens_(std::move(tokens)),
        size_(doc_offsets.empty()
                  ? 0
                  : static_cast<GingerNodeId>(doc_offsets.size() - 1)),
        doc_offsets_(std::move(doc_offsets)) {
    ABSL_CHECK_EQ(tokens_.coordinate_type(), CoordinateType::kFloat);
    ABSL_DCHECK(std::is_sorted(doc_offsets_.begin(), doc_offsets_.end()));
    if (!doc_offsets_.empty()) {
      ABSL_CHECK_EQ(doc_offsets_.front(), 0);
      ABSL_CHECK_EQ(doc_offsets_.back(), tokens_.size());
    }
    PrepareTransposedBlocks();
  }

  ChamferDataset(ChamferDataset&&) noexcept = default;
  ChamferDataset& operator=(ChamferDataset&&) noexcept = default;

  GingerNodeId size() const { return size_; }

  size_t dimensionality() const { return tokens_.dimensionality(); }

  // Number of tokens in document `id`.
  size_t num_tokens(GingerNodeId id) const {
    ABSL_DCHECK_GE(id, 0);
    ABSL_DCHECK_LT(id, size_);
    return doc_offsets_[id + 1] - doc_offsets_[id];
  }

  // Sets all token embeddings for document `id` from a flat float span of
  // length num_tokens(id) * dimensionality().
  void SetDoc(GingerNodeId id, absl::Span<const float> doc_tokens);

  // Const access to the underlying token storage. Note: for documents with
  // >= 8 tokens, blocks of 8 or 16 tokens are stored in transposed layout.
  const SpanDataset& tokens() const { return tokens_; }

  absl::Span<const size_t> doc_offsets() const { return doc_offsets_; }

  // Computes the Chamfer distance from `query` to each neighbor's document,
  // writing results into neighbors[i].distance.
  // The Chamfer similarity between query Q = {q_1..q_nq} and document
  // D = {d_1..d_nd} is:
  //   sim(Q, D) = (1 / |Q|) * sum_{q in Q} max_{d in D} dot(q, d)
  // The Chamfer distance is its negation (smaller = more similar).
  void ChamferDistance(absl::Span<const float> query,
                       absl::Span<DistanceAndId> neighbors) const;

 private:
  void PrepareTransposedBlocks();

  // Transposes full blocks of document `id` in-place from [token, dim] to
  // [dim, token] layout within tokens_. No-op if num_tokens(id) < 8.
  void TransposeDoc(GingerNodeId id, absl::Span<float> scratch = {});

  SpanDataset tokens_;
  GingerNodeId size_ = 0;
  std::vector<size_t> doc_offsets_;
};

}  // namespace ginger

#endif  // GINGER_DISTANCE_CHAMFER_H_
