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

#include "ginger/turboquant/types/distance_metric.h"

#include "ginger/internal/gunit_support.h"
#include "ginger/searcher.pb.h"

namespace ginger {
namespace {

TEST(DistanceMetricTest, ToTurboQuantDistanceMetric) {
  EXPECT_EQ(ToTurboQuantDistanceMetric(::ginger::DOT_PRODUCT),
            TurboQuantDistanceMetric::kDotProductDistance);
  EXPECT_EQ(ToTurboQuantDistanceMetric(::ginger::SQUARED_L2),
            TurboQuantDistanceMetric::kSquaredL2Distance);
  EXPECT_DEATH(ToTurboQuantDistanceMetric(::ginger::UNDEFINED_DISTANCE),
               "Unsupported distance metric for TurboQuant");
}

TEST(DistanceMetricTest, ToGingerDistanceMetric) {
  EXPECT_EQ(
      ToGingerDistanceMetric(TurboQuantDistanceMetric::kDotProductDistance),
      ::ginger::DOT_PRODUCT);
  EXPECT_EQ(
      ToGingerDistanceMetric(TurboQuantDistanceMetric::kSquaredL2Distance),
      ::ginger::SQUARED_L2);
  EXPECT_DEATH(
      ToGingerDistanceMetric(static_cast<TurboQuantDistanceMetric>(99)),
      "Unsupported distance metric");
}

}  // namespace
}  // namespace ginger
