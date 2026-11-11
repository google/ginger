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

#ifndef GINGER_TURBOQUANT_TYPES_DISTANCE_METRIC_H_
#define GINGER_TURBOQUANT_TYPES_DISTANCE_METRIC_H_

#include "absl/log/absl_log.h"
#include "ginger/searcher.pb.h"

namespace ginger {

// The distance metric to use. The default is kDotProductDistance which is the
// negative of the dot product.
enum class TurboQuantDistanceMetric {
  kDotProductDistance,
  kSquaredL2Distance,
};

// Converts ginger searcher proto DistanceMetric to TurboQuantDistanceMetric.
// Fails with FATAL log if metric is not DOT_PRODUCT or SQUARED_L2.
inline TurboQuantDistanceMetric ToTurboQuantDistanceMetric(
    ::ginger::DistanceMetric metric) {
  switch (metric) {
    case ::ginger::DOT_PRODUCT:
      return TurboQuantDistanceMetric::kDotProductDistance;
    case ::ginger::SQUARED_L2:
      return TurboQuantDistanceMetric::kSquaredL2Distance;
    default:
      ABSL_LOG(FATAL) << "Unsupported distance metric for TurboQuant: "
                      << metric;
  }
}

// Converts TurboQuantDistanceMetric to ginger searcher proto DistanceMetric.
// Fails with FATAL log if metric is not kDotProductDistance or
// kSquaredL2Distance.
inline ::ginger::DistanceMetric ToGingerDistanceMetric(
    TurboQuantDistanceMetric metric) {
  switch (metric) {
    case TurboQuantDistanceMetric::kDotProductDistance:
      return ::ginger::DOT_PRODUCT;
    case TurboQuantDistanceMetric::kSquaredL2Distance:
      return ::ginger::SQUARED_L2;
    default:
      ABSL_LOG(FATAL) << "Unsupported distance metric: "
                      << static_cast<int>(metric);
  }
}

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_TYPES_DISTANCE_METRIC_H_
