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

#ifndef GINGER_PIPNN_PIPNN_H_
#define GINGER_PIPNN_PIPNN_H_

#include <functional>

#include "absl/status/statusor.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"

namespace ginger {

// Extracts the robust pruning alpha from `config`, returning an error if both
// the deprecated `robust_prune_alpha` and `prune_config` are set.
// TODO: Remove once config.robust_prune_alpha is deleted.
absl::StatusOr<float> GetRobustPruneAlpha(const PipnnConfig& config);

// Builds a PiPNN approximate nearest neighbor graph index, see
// https://arxiv.org/abs/2602.21247.
absl::StatusOr<FlatGraph> BuildPipnnIndex(
    const TurboQuantDatasetSoA& points, const TurboQuantCodec& quantizer,
    const PipnnConfig& config,
    const std::function<bool()>& is_cancelled = []() { return false; });

}  // namespace ginger

#endif  // GINGER_PIPNN_PIPNN_H_
