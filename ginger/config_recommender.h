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

#ifndef GINGER_CONFIG_RECOMMENDER_H_
#define GINGER_CONFIG_RECOMMENDER_H_

#include "absl/status/statusor.h"
#include "ginger/searcher.pb.h"

namespace ginger {

// Recommends a searcher configuration for a float dataset with
// `dimensionality` dimensions. The distance metric and the RobustPrune alpha
// depend on the data and are not inferred. Byte (int8/uint8) datasets are not
// supported: PCA rejects them, and dropping the dataset loses the coordinate
// type of the queries.
//
// The recommendation builds a PiPNN graph with on-the-fly RobustPrune, searches
// over 4-bit TurboQuant codes, and reranks with 12-bit TurboQuant codes. The
// graph is reordered and compressed. PCA is enabled for high-dimensional data.
// The float dataset is dropped after the build. Use
// `GingerSearcher::DefaultSearchOptions` to get matching search options.
//
// `normalized_datapoints` is left unset, so the builder autodetects unit-norm
// datasets. Set it explicitly to override the autodetection, e.g. to `false`
// for SQUARED_L2 data whose norms are close to, but not exactly, 1.
absl::StatusOr<GingerSearcherConfig> RecommendGingerConfig(
    int dimensionality, DistanceMetric metric, float robust_prune_alpha);

}  // namespace ginger

#endif  // GINGER_CONFIG_RECOMMENDER_H_
