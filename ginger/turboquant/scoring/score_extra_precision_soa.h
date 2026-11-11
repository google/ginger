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

#ifndef GINGER_TURBOQUANT_SCORING_SCORE_EXTRA_PRECISION_SOA_H_
#define GINGER_TURBOQUANT_SCORING_SCORE_EXTRA_PRECISION_SOA_H_

#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"

namespace ginger {

// Computes extra-precision (5/6/7/8/10/12-bit) distances for a batch of
// datapoints in a SoA dataset against the given query. Dispatches to the best
// SIMD target at runtime via Highway dynamic dispatch.
void ComputeDistancesExtraPrecisionSoA(const TurboQuantDatasetSoA& base_points,
                                       const TurboQuantQuery& query,
                                       ScoringAccessor accessor);

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_SCORING_SCORE_EXTRA_PRECISION_SOA_H_
