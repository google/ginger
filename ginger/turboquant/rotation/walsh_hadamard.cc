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

#include "ginger/turboquant/rotation/walsh_hadamard.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/types/span.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE "ginger/turboquant/rotation/walsh_hadamard.cc"
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

#include "ginger/turboquant/simd/vec_group-inl.h"

namespace ginger {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

// Applies a butterfly stage across vector pairs at distance `Stride`:
//   (v[a], v[a + Stride]) -> (v[a] + v[a + Stride], v[a] - v[a + Stride])
template <size_t Stride, size_t NumVecs, class D, class Acc>
HWY_INLINE void AcrossVectorButterfly(D, Acc&& v_acc) {
  static_assert(Stride < NumVecs && (NumVecs % (2 * Stride) == 0));
  ForIndex<NumVecs / 2>([&](auto i) {
    // Map pair index `i` to its first vector `idx_a`:
    // Vectors are grouped into blocks of size `2 * Stride`. For pair `i`:
    //   - (i / Stride) * (2 * Stride) is the start index of block (i / Stride).
    //   - (i % Stride) is the offset within the first half of the block.
    constexpr size_t idx_a =
        ((i.value / Stride) * (2 * Stride)) + (i.value % Stride);
    constexpr size_t idx_b = idx_a + Stride;
    const hn::Vec<D> a = v_acc(std::integral_constant<size_t, idx_a>{});
    const hn::Vec<D> b = v_acc(std::integral_constant<size_t, idx_b>{});
    v_acc(std::integral_constant<size_t, idx_a>{}) = hn::Add(a, b);
    v_acc(std::integral_constant<size_t, idx_b>{}) = hn::Sub(a, b);
  });
}

// Applies log2(NumVecs) consecutive butterfly stages across `NumVecs`
// registers, computing an in-place NumVecs-point Hadamard transform.
template <size_t NumVecs, class D, class Acc>
HWY_INLINE void AcrossVectorHadamard(D d, Acc&& v_acc) {
  if constexpr (NumVecs >= 2) AcrossVectorButterfly<1, NumVecs>(d, v_acc);
  if constexpr (NumVecs >= 4) AcrossVectorButterfly<2, NumVecs>(d, v_acc);
  if constexpr (NumVecs >= 8) AcrossVectorButterfly<4, NumVecs>(d, v_acc);
}

// Fuses K consecutive Hadamard stages (K = 1, 2, or 3) into a single pass
// over `data`. Groups the array into blocks of 2^K * L elements, loads 2^K
// sub-blocks of stride L into registers, applies an in-register 2^K-point
// Hadamard transform, and stores the results.
template <size_t K>
HWY_INLINE void ApplyStridedPass(float* HWY_RESTRICT data, size_t dim,
                                 size_t L) {
  constexpr size_t kNumVecs = 1ULL << K;
  const hn::ScalableTag<float> f32;
  const size_t N = hn::Lanes(f32);
  for (size_t i = 0; i < dim; i += kNumVecs * L) {
    for (size_t j = 0; j < L; j += N) {
      VEC_GROUP_DEFINE(f32, v);
      auto v_acc = VEC_GROUP_ACCESSOR(v);
      ForIndex<kNumVecs>([&](auto k) {
        v_acc(k) = hn::LoadU(f32, data + i + k.value * L + j);
      });
      AcrossVectorHadamard<kNumVecs>(f32, v_acc);
      ForIndex<kNumVecs>([&](auto k) {
        hn::StoreU(v_acc(k), f32, data + i + k.value * L + j);
      });
    }
  }
}

// Dispatches remaining Hadamard stages from stride `len` up to `dim`,
// greedily fusing 3, 2, or 1 stages per strided pass.
HWY_INLINE void ApplyStridedStages(float* HWY_RESTRICT data, size_t dim,
                                   size_t len) {
  while (len < dim) {
    if (len * 8 <= dim) {
      ApplyStridedPass<3>(data, dim, len);
      len *= 8;
    } else if (len * 4 <= dim) {
      ApplyStridedPass<2>(data, dim, len);
      len *= 4;
    } else {
      ApplyStridedPass<1>(data, dim, len);
      len *= 2;
    }
  }
}

#if HWY_MAX_BYTES >= 16

// Fuses all log2(Dim) Hadamard stages for a Dim-float block in registers:
//   - Strides 1, 2, [4] (in-lane): Single-register shuffles and FMA.
//   - Strides 4, 8, ... (inter-register): AcrossVectorHadamard across
//   registers.
template <size_t Dim, size_t Lanes, class D>
HWY_INLINE void ApplyInRegisterBlock(float* HWY_RESTRICT p, D d) {
  constexpr size_t kNumVecs = Dim / Lanes;
  static_assert(kNumVecs == 4 || kNumVecs == 8);

  using U64 = hn::Repartition<uint64_t, D>;
  using Vec = hn::Vec<D>;
  const Vec one = hn::Set(d, 1.0f);
  const Vec neg = hn::Set(d, -1.0f);
  const Vec signs1 = hn::OddEven(neg, one);
  const Vec signs2 = hn::BitCast(
      d, hn::OddEven(hn::BitCast(U64(), neg), hn::BitCast(U64(), one)));

  VEC_GROUP_DEFINE(d, v);
  auto v_acc = VEC_GROUP_ACCESSOR(v);
  ForIndex<kNumVecs>(
      [&](auto i) { v_acc(i) = hn::LoadU(d, p + Lanes * i.value); });

  // In-lane stages: Stride 1 (Reverse2) and Stride 2 (Shuffle1032).
  ForIndex<kNumVecs>([&](auto i) {
    v_acc(i) = hn::MulAdd(v_acc(i), signs1, hn::Reverse2(d, v_acc(i)));
  });
  ForIndex<kNumVecs>([&](auto i) {
    v_acc(i) = hn::MulAdd(v_acc(i), signs2, hn::Shuffle1032(v_acc(i)));
  });
  // In-lane stage: Stride 4 (SwapAdjacentBlocks, 256-bit+ vectors only).
  if constexpr (Lanes >= 8) {
    const Vec signs4 = hn::OddEvenBlocks(neg, one);
    ForIndex<kNumVecs>([&](auto i) {
      v_acc(i) = hn::MulAdd(v_acc(i), signs4, hn::SwapAdjacentBlocks(v_acc(i)));
    });
  }

  // Inter-register stages:
  //   - When Lanes == 4 (128-bit): Strides 4, 8, [16] across kNumVecs
  //   registers.
  //   - When Lanes == 8 (256-bit): Strides 8, 16, 32 across kNumVecs registers.
  AcrossVectorHadamard<kNumVecs>(d, v_acc);

  ForIndex<kNumVecs>(
      [&](auto i) { hn::StoreU(v_acc(i), d, p + Lanes * i.value); });
}

#endif  // HWY_MAX_BYTES >= 16

void WalshHadamardImpl(float* HWY_RESTRICT data, size_t dim) {
  const size_t N = hn::Lanes(hn::ScalableTag<float>());
  // Fast path: use the in-register base case for dim >= 64 on 256-bit+ SIMD.
#if HWY_MAX_BYTES >= 32
  if (N >= 8 && dim >= 64) {
    const hn::FixedTag<float, 8> d8;
    for (size_t i = 0; i < dim; i += 64) {
      ApplyInRegisterBlock<64, 8>(data + i, d8);
    }
    return ApplyStridedStages(data, dim, 64);
  }
#endif  // HWY_MAX_BYTES >= 32

#if HWY_MAX_BYTES >= 16
  if (N >= 4 && dim == 16) {
    return ApplyInRegisterBlock<16, 4>(data, hn::FixedTag<float, 4>());
  }
  if (N >= 4 && dim >= 32) {
    const hn::FixedTag<float, 4> d4;
    for (size_t i = 0; i < dim; i += 32) {
      ApplyInRegisterBlock<32, 4>(data + i, d4);
    }
    return ApplyStridedStages(data, dim, 32);
  }
#endif  // HWY_MAX_BYTES >= 16

  // Scalar fallback.
  // The strided stages require `len` >= num lanes.
  size_t len = 1;
  while (len < N && len < dim) {
    for (size_t i = 0; i < dim; i += 2 * len) {
      for (size_t j = 0; j < len; ++j) {
        const float a = data[i + j];
        const float b = data[i + len + j];
        data[i + j] = a + b;
        data[i + len + j] = a - b;
      }
    }
    len *= 2;
  }
  ApplyStridedStages(data, dim, len);
}
}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace ginger
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ginger {
HWY_EXPORT(WalshHadamardImpl);

absl::Status WalshHadamardTransform(absl::Span<float> data) {
  if (data.data() == nullptr && !data.empty()) {
    return absl::InvalidArgumentError("data pointer cannot be null.");
  }
  const size_t dim = data.size();
  if (dim == 0 || !std::has_single_bit(dim)) {
    return absl::InvalidArgumentError(
        "dim must be a positive power of 2 for Walsh-Hadamard transform.");
  }
  if (dim == 1) {
    return absl::OkStatus();
  }
  HWY_DYNAMIC_DISPATCH(WalshHadamardImpl)(data.data(), dim);
  return absl::OkStatus();
}

}  // namespace ginger
#endif  // HWY_ONCE
