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

#ifndef GINGER_TURBOQUANT_SCORING_SCORING_ACCESSORS_H_
#define GINGER_TURBOQUANT_SCORING_SCORING_ACCESSORS_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <variant>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {

// WARNING: Please don't add new accessor types to this file without running
// rigorous performance analysis, as it could lead to code bloat and performance
// regressions.
//
// SAFETY & VALIDITY DATA CONTRACT:
// - These accessors assume absolute pointer stability and index bounds
//   validity.
// - They are strictly intended for internal TurboQuant hot scoring loops.
// - WARNING: If you use these classes in other scenarios or outside the
//   quantizer's validated scopes, you MUST enforce explicit bounds checks!
class ScoresAndIndicesSpanOfPairsView {
 public:
  ScoresAndIndicesSpanOfPairsView() = default;
  explicit ScoresAndIndicesSpanOfPairsView(
      absl::Span<std::pair<float, int32_t>> s)
      : size_(static_cast<uint32_t>(s.size())), raw_data_(s.data()) {
    ABSL_DCHECK_LE(s.size(), std::numeric_limits<uint32_t>::max());
  }

  float GetScore(size_t i) const { return raw_data_[i].first; }
  void SetScore(size_t i, float score) { raw_data_[i].first = score; }
  void IncScore(size_t i, float inc) { raw_data_[i].first += inc; }
  size_t Index(size_t i) const { return raw_data_[i].second; }
  size_t size() const { return size_; }

  absl::Span<std::pair<float, int32_t>> scores_and_indices() const {
    return absl::MakeSpan(raw_data_, size_);
  }

  absl::Status Validate(size_t max_index) const {
    for (size_t i = 0; i < size_; ++i) {
      if (raw_data_[i].second < 0 ||
          static_cast<size_t>(raw_data_[i].second) >= max_index) {
        return absl::InvalidArgumentError("Index out of bounds.");
      }
    }
    return absl::OkStatus();
  }

 private:
  uint32_t size_ = 0;
  std::pair<float, int32_t>* raw_data_ = nullptr;
};

class ScoresAndIndicesDistanceAndIdView {
 public:
  ScoresAndIndicesDistanceAndIdView() = default;
  explicit ScoresAndIndicesDistanceAndIdView(
      absl::Span<::ginger::DistanceAndId> s)
      : size_(static_cast<uint32_t>(s.size())), raw_data_(s.data()) {
    ABSL_DCHECK_LE(s.size(), std::numeric_limits<uint32_t>::max());
  }

  float GetScore(size_t i) const { return raw_data_[i].distance; }
  void SetScore(size_t i, float score) { raw_data_[i].distance = score; }
  void IncScore(size_t i, float inc) { raw_data_[i].distance += inc; }
  size_t Index(size_t i) const { return raw_data_[i].id; }
  size_t size() const { return size_; }

  absl::Span<::ginger::DistanceAndId> scores_and_indices() const {
    return absl::MakeSpan(raw_data_, size_);
  }

  absl::Status Validate(size_t max_index) const {
    for (size_t i = 0; i < size_; ++i) {
      if (raw_data_[i].id < 0 ||
          static_cast<size_t>(raw_data_[i].id) >= max_index) {
        return absl::InvalidArgumentError("Index out of bounds.");
      }
    }
    return absl::OkStatus();
  }

 private:
  uint32_t size_ = 0;
  ::ginger::DistanceAndId* raw_data_ = nullptr;
};

template <typename IndexType>
class ScoresAndIndicesPairOfSpansViewBase {
 public:
  ScoresAndIndicesPairOfSpansViewBase() = default;
  float GetScore(size_t i) const { return raw_scores_[i]; }
  void SetScore(size_t i, float score) { raw_scores_[i] = score; }
  void IncScore(size_t i, float inc) { raw_scores_[i] += inc; }
  size_t Index(size_t i) const { return raw_indices_[i]; }
  size_t size() const { return size_; }

  absl::Span<float> scores() const {
    return absl::MakeSpan(raw_scores_, size_);
  }
  absl::Span<const IndexType> indices() const {
    return absl::MakeConstSpan(raw_indices_, size_);
  }

  absl::Status Validate(size_t max_index) const {
    for (size_t i = 0; i < size_; ++i) {
      if (raw_indices_[i] < 0 ||
          static_cast<size_t>(raw_indices_[i]) >= max_index) {
        return absl::InvalidArgumentError("Index out of bounds.");
      }
    }
    return absl::OkStatus();
  }

 protected:
  ScoresAndIndicesPairOfSpansViewBase(absl::Span<const IndexType> idxs,
                                      absl::Span<float> scrs)
      : size_(static_cast<uint32_t>(scrs.size())),
        raw_scores_(scrs.data()),
        raw_indices_(idxs.data()) {
    ABSL_DCHECK_EQ(idxs.size(), scrs.size());
    ABSL_DCHECK_LE(scrs.size(), std::numeric_limits<uint32_t>::max());
    ABSL_DCHECK_LE(idxs.size(), std::numeric_limits<uint32_t>::max());
  }

 private:
  uint32_t size_ = 0;
  float* raw_scores_ = nullptr;
  const IndexType* raw_indices_ = nullptr;
};

class ScoresAndIndicesPairOfSpansView
    : public ScoresAndIndicesPairOfSpansViewBase<int32_t> {
 public:
  ScoresAndIndicesPairOfSpansView() = default;
  ScoresAndIndicesPairOfSpansView(absl::Span<const int32_t> idxs,
                                  absl::Span<float> scrs)
      : ScoresAndIndicesPairOfSpansViewBase<int32_t>(idxs, scrs) {}
};

class ScoresAndU32IndicesPairOfSpansView
    : public ScoresAndIndicesPairOfSpansViewBase<uint32_t> {
 public:
  ScoresAndU32IndicesPairOfSpansView() = default;
  ScoresAndU32IndicesPairOfSpansView(absl::Span<const uint32_t> idxs,
                                     absl::Span<float> scrs)
      : ScoresAndIndicesPairOfSpansViewBase<uint32_t>(idxs, scrs) {}
};

class ScoresAndIndicesContiguousRangeView {
 public:
  ScoresAndIndicesContiguousRangeView() = default;
  ScoresAndIndicesContiguousRangeView(size_t start_index, size_t end_index,
                                      absl::Span<float> scores)
      : start_index_(static_cast<uint32_t>(start_index)),
        end_index_(static_cast<uint32_t>(end_index)),
        raw_scores_(scores.data()) {
    ABSL_DCHECK_EQ(end_index - start_index, scores.size());
    ABSL_DCHECK_LE(start_index, std::numeric_limits<uint32_t>::max());
    ABSL_DCHECK_LE(end_index, std::numeric_limits<uint32_t>::max());
  }

  float GetScore(size_t i) const { return raw_scores_[i]; }
  void SetScore(size_t i, float score) { raw_scores_[i] = score; }
  void IncScore(size_t i, float inc) { raw_scores_[i] += inc; }
  size_t Index(size_t i) const { return start_index_ + i; }
  size_t size() const { return end_index_ - start_index_; }

  absl::Span<float> scores() const {
    return absl::MakeSpan(raw_scores_, size_t(end_index_ - start_index_));
  }

  absl::Status Validate(size_t max_index) const {
    if (start_index_ > end_index_) {
      return absl::InvalidArgumentError(
          "Start index must be less than or equal to end index.");
    }
    if (end_index_ > max_index) {
      return absl::InvalidArgumentError("Index out of bounds.");
    }
    return absl::OkStatus();
  }

 private:
  uint32_t start_index_ = 0;
  uint32_t end_index_ = 0;
  float* raw_scores_ = nullptr;
};

using ScoringAccessor = ::std::variant<
    ScoresAndIndicesSpanOfPairsView, ScoresAndIndicesDistanceAndIdView,
    ScoresAndIndicesPairOfSpansView, ScoresAndU32IndicesPairOfSpansView,
    ScoresAndIndicesContiguousRangeView>;

inline absl::StatusOr<ScoringAccessor> CreateScoringAccessor(
    absl::Span<const int32_t> idxs, absl::Span<float> scrs, size_t max_index) {
  if (idxs.size() != scrs.size()) {
    return absl::InvalidArgumentError(
        "`indices` and `scores` must have the same size.");
  }
  auto view = ScoresAndIndicesPairOfSpansView(idxs, scrs);
  ABSL_RETURN_IF_ERROR(view.Validate(max_index));
  return view;
}

inline absl::StatusOr<ScoringAccessor> CreateScoringAccessor(
    absl::Span<const uint32_t> idxs, absl::Span<float> scrs, size_t max_index) {
  if (idxs.size() != scrs.size()) {
    return absl::InvalidArgumentError(
        "`indices` and `scores` must have the same size.");
  }
  auto view = ScoresAndU32IndicesPairOfSpansView(idxs, scrs);
  ABSL_RETURN_IF_ERROR(view.Validate(max_index));
  return view;
}

inline absl::StatusOr<ScoringAccessor> CreateScoringAccessor(
    absl::Span<std::pair<float, int32_t>> s, size_t max_index) {
  auto view = ScoresAndIndicesSpanOfPairsView(s);
  ABSL_RETURN_IF_ERROR(view.Validate(max_index));
  return view;
}

inline absl::StatusOr<ScoringAccessor> CreateScoringAccessor(
    absl::Span<::ginger::DistanceAndId> s, size_t max_index) {
  auto view = ScoresAndIndicesDistanceAndIdView(s);
  ABSL_RETURN_IF_ERROR(view.Validate(max_index));
  return view;
}

inline absl::StatusOr<ScoringAccessor> CreateScoringAccessor(
    size_t start_index, size_t end_index, absl::Span<float> scores,
    size_t max_index) {
  if (start_index > end_index) {
    return absl::InvalidArgumentError(
        "Start index must be less than or equal to end index.");
  }
  if (end_index - start_index != scores.size()) {
    return absl::InvalidArgumentError("Size mismatch.");
  }
  auto view =
      ScoresAndIndicesContiguousRangeView(start_index, end_index, scores);
  ABSL_RETURN_IF_ERROR(view.Validate(max_index));
  return view;
}

// Non-validating factory functions for creating ScoringAccessors.
// Use these for unsafe callsites that already trust their data.
inline ScoringAccessor CreateScoringAccessor(absl::Span<const int32_t> indices,
                                             absl::Span<float> scores) {
  return ScoresAndIndicesPairOfSpansView(indices, scores);
}

inline ScoringAccessor CreateScoringAccessor(absl::Span<const uint32_t> indices,
                                             absl::Span<float> scores) {
  return ScoresAndU32IndicesPairOfSpansView(indices, scores);
}

inline ScoringAccessor CreateScoringAccessor(
    absl::Span<std::pair<float, int32_t>> scores_and_indices) {
  return ScoresAndIndicesSpanOfPairsView(scores_and_indices);
}

inline ScoringAccessor CreateScoringAccessor(
    absl::Span<::ginger::DistanceAndId> scores_and_indices) {
  return ScoresAndIndicesDistanceAndIdView(scores_and_indices);
}

// Dispatches to the correct scoring function based on the accessor type.
#define DISPATCH_SCORING_ACCESSOR(FunctionName, accessor_variant, ...)     \
  ::std::visit(                                                            \
      [&](auto&& concrete_accessor) {                                      \
        return FunctionName(__VA_ARGS__ __VA_OPT__(, ) concrete_accessor); \
      },                                                                   \
      accessor_variant)

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_SCORING_SCORING_ACCESSORS_H_
