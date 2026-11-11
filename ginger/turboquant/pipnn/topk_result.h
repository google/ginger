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

#ifndef GINGER_TURBOQUANT_PIPNN_TOPK_RESULT_H_
#define GINGER_TURBOQUANT_PIPNN_TOPK_RESULT_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>

namespace ginger {

// Per-query result of fused top-K neighbor selection.
template <size_t K>
class FixedTopKResult {
 public:
  struct Entry {
    float score;
    int32_t index;
  };

  FixedTopKResult() {
    static_assert(K > 0, "K must be greater than 0");
    for (size_t i = 0; i < K; ++i) {
      scores_[i] = std::numeric_limits<float>::max();
      indices_[i] = -1;
    }
  }

  // Scalar sorted insertion. Maintains scores in ascending order. Breaks score
  // ties deterministically using the leader index to ensure invariance to SIMD
  // lane reduction order.
  void Insert(float score, int32_t idx) {
    if (std::tie(score, idx) >= std::tie(scores_[K - 1], indices_[K - 1]))
      return;
    size_t pos = K - 1;
    while (pos > 0 && std::tie(score, idx) <
                          std::tie(scores_[pos - 1], indices_[pos - 1])) {
      scores_[pos] = scores_[pos - 1];
      indices_[pos] = indices_[pos - 1];
      --pos;
    }
    scores_[pos] = score;
    indices_[pos] = idx;
  }

  void Set(size_t i, float score, int32_t idx) {
    scores_[i] = score;
    indices_[i] = idx;
  }

  Entry Get(size_t i) const { return {scores_[i], indices_[i]}; }

 private:
  float scores_[K];
  int32_t indices_[K];
};

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_PIPNN_TOPK_RESULT_H_
