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

#ifndef GINGER_INTERNAL_PORTABLE_RANDOM_H_
#define GINGER_INTERNAL_PORTABLE_RANDOM_H_

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"

// Random distributions with a fixed algorithm.
//
// The C++ standard fixes the output of `std::mt19937`, but not the algorithms
// behind `std::*_distribution` or `std::shuffle`, and standard libraries
// differ. Rotations and projections are regenerated from their seed, and the
// resulting encodings are stored and exchanged, so they must not depend on the
// standard library. These functions reproduce the algorithms of LLVM's libc++,
// which generated all existing encodings, so the same seed gives the same
// values on every platform.
namespace ginger::internal::portable_random {

// Returns a uniform integer in [0, bound) like libc++'s
// `std::uniform_int_distribution`: the lowest ceil(log2(bound)) bits of one
// engine output, retried while the value is out of range. `bound` must be
// positive and fit into the engine's output bits.
template <typename Engine>
uint64_t UniformInt(Engine& gen, uint64_t bound) {
  static_assert(Engine::min() == 0);
  static_assert(Engine::max() == UINT32_MAX || Engine::max() == UINT64_MAX);
  ABSL_DCHECK_GT(bound, 0);
  if (bound == 1) return 0;
  const int bits = std::bit_width(bound - 1);
  ABSL_DCHECK_LE(bits, std::bit_width(uint64_t{Engine::max()}));
  const uint64_t mask = bits == 64 ? UINT64_MAX : (uint64_t{1} << bits) - 1;
  uint64_t value;
  do {
    value = static_cast<uint64_t>(gen()) & mask;
  } while (value >= bound);
  return value;
}

// Returns true with probability `p` like libc++'s `std::bernoulli_distribution`
// on a 64-bit engine.
inline bool Bernoulli(std::mt19937_64& gen, double p) {
  ABSL_DCHECK(p >= 0.0 && p <= 1.0);
  return static_cast<double>(gen()) / 0x1p64 < p;
}

// Standard normal samples like libc++'s `std::normal_distribution<float>`:
// the Marsaglia polar method on uniforms from `std::generate_canonical`, which
// returns two samples per accepted pair and caches the second one.
class NormalDistribution {
 public:
  float operator()(std::mt19937& gen) {
    // A fused multiply-add in `s` would change which pairs are rejected, and
    // with them every later sample.
#pragma STDC FP_CONTRACT OFF
    if (has_cached_) {
      has_cached_ = false;
      return cached_;
    }
    float u;
    float v;
    float s;
    do {
      u = Uniform(gen);
      v = Uniform(gen);
      const float uu = u * u;
      const float vv = v * v;
      s = uu + vv;
    } while (s > 1 || s == 0);
    const float factor = std::sqrt(-2 * std::log(s) / s);
    cached_ = v * factor;
    has_cached_ = true;
    return u * factor;
  }

 private:
  // Uniform in [-1, 1], like `std::uniform_real_distribution<float>(-1, 1)`.
  static float Uniform(std::mt19937& gen) {
    const float canonical = static_cast<float>(gen()) / 4294967296.0f;
    return 2.0f * canonical + -1.0f;
  }

  bool has_cached_ = false;
  float cached_ = 0.0f;
};

// Shuffles `values` like libc++'s `std::shuffle`: forward Fisher-Yates that
// swaps each position with a uniform pick from itself and the positions after
// it.
template <typename T>
void Shuffle(absl::Span<T> values, std::mt19937& gen) {
  using std::swap;
  for (size_t i = 0; i + 1 < values.size(); ++i) {
    const uint64_t offset = UniformInt(gen, values.size() - i);
    if (offset != 0) swap(values[i], values[i + offset]);
  }
}

}  // namespace ginger::internal::portable_random

#endif  // GINGER_INTERNAL_PORTABLE_RANDOM_H_
