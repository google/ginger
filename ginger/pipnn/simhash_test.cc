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

#include "ginger/pipnn/simhash.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/types/distance_metric.h"

namespace ginger {
namespace {

using ::absl_testing::IsOk;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::Gt;
using ::testing::Lt;
using ::testing::Not;

TEST(SimHashProjectionsTest, BuildAndRelativeHash) {
  std::vector<float> p0 = {1.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {0.0f, 1.0f, 0.0f};
  std::vector<float> p2 = {1.0f, 0.1f, 0.0f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2};

  const int kNumProjections = 8;
  const size_t kNumPoints = 3;
  SimHashProjections simhash_projections(data, kNumPoints, kNumProjections);

  EXPECT_EQ(simhash_projections.num_projections(), kNumProjections);
  EXPECT_EQ(simhash_projections.num_points(), kNumPoints);
  EXPECT_EQ(simhash_projections.projections().size(),
            kNumPoints * kNumProjections);

  // Rigorous per-point projection validation.
  for (size_t i = 0; i < kNumPoints; ++i) {
    auto pt_projections = absl::MakeConstSpan(
        &simhash_projections.projections()[i * kNumProjections],
        kNumProjections);
    EXPECT_THAT(pt_projections, Not(Each(0.0f))) << "i: " << i;
  }

  // Orthogonal points p0 and p1 must produce distinct projection vectors.
  auto proj0 = absl::MakeConstSpan(&simhash_projections.projections()[0],
                                   kNumProjections);
  auto proj1 = absl::MakeConstSpan(
      &simhash_projections.projections()[kNumProjections], kNumProjections);
  EXPECT_NE(proj0, proj1);

  uint16_t hash_0_1 = simhash_projections.RelativeHash(0, 1);
  uint16_t hash_0_2 = simhash_projections.RelativeHash(0, 2);
  uint16_t hash_1_2 = simhash_projections.RelativeHash(1, 2);

  EXPECT_THAT(hash_0_1, AllOf(Gt(0), Lt(1 << kNumProjections)));
  EXPECT_THAT(hash_0_2, AllOf(Gt(0), Lt(1 << kNumProjections)));
  EXPECT_THAT(hash_1_2, AllOf(Gt(0), Lt(1 << kNumProjections)));
  EXPECT_EQ(simhash_projections.RelativeHash(0, 0), 0);
}

TEST(SimHashProjectionsTest, MaxProjectionsSupported) {
  std::vector<float> p0 = {1.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {0.0f, 1.0f, 0.0f};
  std::vector<absl::Span<const float>> data = {p0, p1};

  SimHashProjections simhash_projections(data, 3, 16);

  EXPECT_EQ(simhash_projections.num_projections(), 16);
  EXPECT_EQ(simhash_projections.num_points(), 2);
}

TEST(SimHashProjectionsTest, ExceedsMaxProjectionsDies) {
  std::vector<float> p0 = {1.0f, 0.0f, 0.0f};
  std::vector<absl::Span<const float>> data = {p0};

  EXPECT_DEATH(SimHashProjections(data, 3, 17), "");
}

TEST(SimHashProjectionsTest, AntiSymmetryOfRelativeHash) {
  std::vector<float> p0 = {1.0f, 0.2f, 0.5f};
  std::vector<float> p1 = {0.1f, 0.9f, 0.3f};
  std::vector<absl::Span<const float>> data = {p0, p1};

  const int kNumProjections = 8;
  SimHashProjections simhash(data, 3, kNumProjections);

  uint16_t hash_0_1 = simhash.RelativeHash(0, 1);
  uint16_t hash_1_0 = simhash.RelativeHash(1, 0);

  // RelativeHash(0, 1) should be the bitwise complement of RelativeHash(1, 0)
  // masked to kNumProjections bits (assuming no exact projection ties).
  uint16_t mask = (1 << kNumProjections) - 1;
  EXPECT_EQ(hash_0_1, (~hash_1_0) & mask);
}

TEST(SimHashProjectionsTest, SelfRelativeHashIsZero) {
  std::vector<float> p0 = {1.0f, 0.5f, 0.2f};
  std::vector<absl::Span<const float>> data = {p0};

  SimHashProjections simhash(data, 3, 8);

  EXPECT_EQ(simhash.RelativeHash(0, 0), 0);
}

TEST(SimHashProjectionsTest, BuildWithTurboQuant) {
  std::vector<float> p0 = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> p1 = {0.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> p2 = {0.0f, 0.0f, 1.0f, 0.0f};
  std::vector<absl::Span<const float>> data = {p0, p1, p2};
  const size_t dim = 4;
  const size_t num_points = 3;

  TurboQuantCodec quantizer(dim, 123,
                            TurboQuantDistanceMetric::kDotProductDistance);
  ASSERT_OK_AND_ASSIGN(auto base_points,
                       TurboQuantDatasetSoA::Create(quantizer, num_points));

  for (size_t i = 0; i < num_points; ++i) {
    EXPECT_THAT(base_points.UpdatePoint(quantizer, data[i], i), IsOk())
        << "i: " << i;
  }

  const int kNumProjections = 8;
  SimHashProjections simhash(base_points, kNumProjections);

  EXPECT_EQ(simhash.num_projections(), kNumProjections);
  EXPECT_EQ(simhash.num_points(), num_points);
  EXPECT_EQ(simhash.projections().size(), num_points * kNumProjections);

  for (size_t i = 0; i < num_points; ++i) {
    auto pt_projections = absl::MakeConstSpan(
        &simhash.projections()[i * kNumProjections], kNumProjections);
    EXPECT_THAT(pt_projections, Not(Each(0.0f))) << "i: " << i;
  }

  // Orthogonal points p0 and p1 must produce distinct projection vectors.
  auto proj0 = absl::MakeConstSpan(&simhash.projections()[0], kNumProjections);
  auto proj1 = absl::MakeConstSpan(&simhash.projections()[kNumProjections],
                                   kNumProjections);
  EXPECT_NE(proj0, proj1);

  uint16_t hash_0_1 = simhash.RelativeHash(0, 1);
  uint16_t hash_0_2 = simhash.RelativeHash(0, 2);
  uint16_t hash_1_2 = simhash.RelativeHash(1, 2);

  EXPECT_THAT(hash_0_1, AllOf(Gt(0), Lt(1 << kNumProjections)));
  EXPECT_THAT(hash_0_2, AllOf(Gt(0), Lt(1 << kNumProjections)));
  EXPECT_THAT(hash_1_2, AllOf(Gt(0), Lt(1 << kNumProjections)));
  EXPECT_EQ(simhash.RelativeHash(0, 0), 0);
}

}  // namespace
}  // namespace ginger
