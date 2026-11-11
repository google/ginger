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

#include <algorithm>
#include <bit>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "ginger/turboquant/rotation/rotation.h"
#include "ginger/turboquant/rotation/walsh_hadamard.h"

namespace ginger {
namespace {


constexpr int kNumVectors = 100;
constexpr int kDimensionalities[] = {
    16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072,
};

void RotationArgs(benchmark::Benchmark* b) {
  for (int dim : kDimensionalities) {
    for (int permuted : {0, 1}) {
      b->Args({dim, permuted});
    }
  }
  b->ArgNames({"dim", "permuted"});
}

void HadamardArgs(benchmark::Benchmark* b) {
  for (int dim : kDimensionalities) {
    if (std::has_single_bit(static_cast<unsigned>(dim))) {
      b->Args({dim, 0});
    }
  }
  b->ArgNames({"dim", "permuted"});
}

class RotationBenchmarkFixture : public benchmark::Fixture {
 protected:
  void SetUp(const benchmark::State& state) override {
    dim_ = state.range(0);
    num_subspaces_ = dim_;
    while (num_subspaces_ > 1 && num_subspaces_ % 2 == 0) {
      num_subspaces_ /= 2;
    }
    subspace_dim_ = dim_ / num_subspaces_;
    rotator_ = std::make_unique<HadamardAndMatrixRotation>(
        dim_, /*seed=*/1000,
        /*permute_coordinates=*/state.range(1) != 0);
    vectors_.assign(kNumVectors, std::vector<float>(dim_));
    norms_.resize(kNumVectors);
    output_.resize(dim_);
    absl::BitGen bitgen;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < kNumVectors; ++i) {
      float sum_sq = 0.0f;
      for (int d = 0; d < dim_; ++d) {
        vectors_[i][d] = dist(bitgen);
        sum_sq += vectors_[i][d] * vectors_[i][d];
      }
      norms_[i] = std::sqrt(sum_sq);
    }
  }

  void RecordCounters(benchmark::State& state) const {
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * dim_ * sizeof(float));
    state.counters["subspaces"] = num_subspaces_;
    state.counters["subspace_dim"] = subspace_dim_;
  }

  int dim_ = 0;
  int num_subspaces_ = 0;
  int subspace_dim_ = 0;
  std::unique_ptr<HadamardAndMatrixRotation> rotator_;
  std::vector<std::vector<float>> vectors_;
  std::vector<float> norms_;
  std::vector<float> output_;
};

BENCHMARK_DEFINE_F(RotationBenchmarkFixture, BM_Rotate)
(benchmark::State& state) {
  int idx = 0;
  for (auto _ : state) {
    auto& input = vectors_[idx];
    auto status =
        rotator_->Rotate(absl::MakeSpan(input), absl::MakeSpan(output_));
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(output_.data());
    idx = (idx + 1) % kNumVectors;
  }
  RecordCounters(state);
}

BENCHMARK_DEFINE_F(RotationBenchmarkFixture,
                   BM_FlipRandomSignsNormalizeAndRotate)
(benchmark::State& state) {
  int idx = 0;
  for (auto _ : state) {
    const auto& input = vectors_[idx];
    const float norm = norms_[idx];
    auto status = rotator_->FlipRandomSignsNormalizeAndRotate(
        absl::MakeSpan(input), norm, absl::MakeSpan(output_));
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(output_.data());
    idx = (idx + 1) % kNumVectors;
  }
  RecordCounters(state);
}

BENCHMARK_DEFINE_F(RotationBenchmarkFixture, BM_InverseRotateInPlace)
(benchmark::State& state) {
  int idx = 0;
  for (auto _ : state) {
    // Copy first since repeated in-place inverse rotations shrink the values
    // towards denormals.
    std::copy(vectors_[idx].begin(), vectors_[idx].end(), output_.begin());
    auto status = rotator_->InverseRotate(absl::MakeSpan(output_));
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(output_.data());
    idx = (idx + 1) % kNumVectors;
  }
  RecordCounters(state);
}

BENCHMARK_DEFINE_F(RotationBenchmarkFixture, BM_WalshHadamardTransformOnly)
(benchmark::State& state) {
  int idx = 0;
  for (auto _ : state) {
    auto& input = vectors_[idx];
    auto status = WalshHadamardTransform(absl::MakeSpan(input));
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(input.data());
    idx = (idx + 1) % kNumVectors;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * dim_ * sizeof(float));
}

BENCHMARK_REGISTER_F(RotationBenchmarkFixture, BM_Rotate)->Apply(RotationArgs);

BENCHMARK_REGISTER_F(RotationBenchmarkFixture,
                     BM_FlipRandomSignsNormalizeAndRotate)
    ->Apply(RotationArgs);

BENCHMARK_REGISTER_F(RotationBenchmarkFixture, BM_InverseRotateInPlace)
    ->Apply(RotationArgs);

BENCHMARK_REGISTER_F(RotationBenchmarkFixture, BM_WalshHadamardTransformOnly)
    ->Apply(HadamardArgs);

}  // namespace
}  // namespace ginger

BENCHMARK_MAIN();
