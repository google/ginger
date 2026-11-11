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

#ifndef GINGER_TURBOQUANT_PIPNN_ASSIGN_LEADERS_H_
#define GINGER_TURBOQUANT_PIPNN_ASSIGN_LEADERS_H_

#include <vector>

#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/many_to_many/index_subset.h"
#include "ginger/turboquant/many_to_many/pretranspose.h"
#include "ginger/turboquant/pipnn/topk_result.h"

namespace ginger {

// Fused GEMM + top-K leader assignment for assigning points to their nearest
// leaders in a pre-transposed dataset. Specialized for K=1, 3, and 10 to
// support the dataset partitioning and tree routing steps in PiPNN.
//
// The output entries in FixedTopKResult are sorted in ascending order (most
// similar leader first).
//
// Common Args:
//   quantized_points: The dataset of query points in TurboQuantCodec
//     format.
//   query_subset: The subset of queries to process (gathered or contiguous).
//   leaders: The dataset of potential leaders in pre-transposed TurboQuant
//     format.

// Finds the single nearest leader (K=1) for a subset of queries.
// Uses optimized in-register top-1 tracking.
std::vector<FixedTopKResult<1>> AssignToLeadersTopK1(
    const TurboQuantDatasetSoA& quantized_points, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& leaders);

// Finds the 3 nearest leaders (K=3) for a subset of queries.
// Uses optimized in-register top-3 tracking.
std::vector<FixedTopKResult<3>> AssignToLeadersTopK3(
    const TurboQuantDatasetSoA& quantized_points, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& leaders);

// Finds the 10 nearest leaders (K=10) for a subset of queries.
// Uses buffer-backed top-10 tracking.
std::vector<FixedTopKResult<10>> AssignToLeadersTopK10(
    const TurboQuantDatasetSoA& quantized_points, IndexSubset query_subset,
    const PreTransposedTurboQuantDataset& leaders);

}  // namespace ginger

#endif  // GINGER_TURBOQUANT_PIPNN_ASSIGN_LEADERS_H_
