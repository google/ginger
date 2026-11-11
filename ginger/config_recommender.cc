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

#include "ginger/config_recommender.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/projection/config.pb.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/searcher.pb.h"

namespace ginger {

absl::StatusOr<GingerSearcherConfig> RecommendGingerConfig(
    int dimensionality, DistanceMetric metric, float robust_prune_alpha) {
  if (dimensionality <= 0) {
    return absl::InvalidArgumentError("dimensionality must be positive.");
  }
  if (metric == UNDEFINED_DISTANCE) {
    return absl::InvalidArgumentError("Distance metric is not set.");
  }
  if (!(robust_prune_alpha >= 1.0f)) {
    return absl::InvalidArgumentError("robust_prune_alpha must be >= 1.0.");
  }

  GingerSearcherConfig config;
  config.set_distance_metric(metric);

  PipnnConfig& pipnn = *config.mutable_pipnn_config();
  pipnn.set_max_degree(64);
  pipnn.set_num_passes(2);
  pipnn.set_pruning_strategy(
      PipnnConfig::PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY);
  pipnn.mutable_prune_config()->set_alpha(robust_prune_alpha);
  for (const int fanout : {10, 3}) {
    pipnn.mutable_leader_selection()->add_fanout_sequence(fanout);
  }

  // `scaling_storage_mode` is deliberately left unspecified. This makes
  // `GingerSearcherBuilder` pick the mode via `RecommendedScalingStorageMode`
  // (see `internal::ResolveScalingStorageMode`) once the dimensionality and
  // norm layout are known. With `allow_three_bit_prefix`, that recommendation
  // may choose the 3-bit prefix.
  config.mutable_turbo_quant()->set_extra_bits(8);
  config.mutable_turbo_quant()->set_padding(false);
  config.mutable_turbo_quant()->set_allow_three_bit_prefix(true);

  if (dimensionality >= 512) {
    PcaProjectionConfig& pca = *config.mutable_pca();
    pca.set_matrix_representation(MATRIX_REPRESENTATION_TURBOQUANT);
    PcaProjectionConfig::Truncation& truncation = *pca.mutable_truncation();
    truncation.set_significance_threshold(0.9f);
    truncation.set_truncation_threshold(0.5f);
    // Without padding, the scaling factor goes at the end of each 4-bit row
    // (`kRowTail`). PCA-projected points are not normalized, so the norm entry
    // takes 8 bytes (16 dimensions) for SQUARED_L2 and 2 bytes (4 dimensions)
    // for DOT_PRODUCT. Rounding to `128 * k - tail` dimensions makes each row
    // fill exactly `k` cache lines.
    truncation.set_alignment(128);
    truncation.set_alignment_tail(metric == SQUARED_L2 ? 16 : 4);
  }

  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);

  // Reranking uses 12-bit TurboQuant codes, so the float dataset is not
  // needed at search time.
  config.set_drop_dataset(true);
  return config;
}

}  // namespace ginger
