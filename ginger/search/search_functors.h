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

#ifndef GINGER_SEARCH_SEARCH_FUNCTORS_H_
#define GINGER_SEARCH_SEARCH_FUNCTORS_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/distance/distance.h"
#include "ginger/distance/distance_bytes.h"
#include "ginger/searcher.pb.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"

namespace ginger {

// SearchDistanceScorer requires a search-time scoring callable with or without
// beam context:
//   void operator()(absl::Span<DistanceAndId> candidates) const;
// or
//   [void|absl::Span<DistanceAndId>] operator()(
//       absl::Span<DistanceAndId> candidates,
//       absl::Span<const BeamEntry> beam) const;
// Computes distances between a pre-configured query and the candidates,
// updating candidate.distance in place. May return a span of filtered
// surviving candidates when candidate filtering is enabled.
template <typename T>
concept SearchDistanceScorerWithoutBeam =
    requires(const T& t, absl::Span<DistanceAndId> candidates) {
      { t(candidates) };
    };

template <typename T>
concept SearchDistanceScorerWithBeam =
    requires(const T& t, absl::Span<DistanceAndId> candidates,
             absl::Span<const BeamEntry> beam) {
      { t(candidates, beam) };
    };

template <typename T>
concept SearchDistanceScorer =
    SearchDistanceScorerWithoutBeam<T> || SearchDistanceScorerWithBeam<T>;

// BuildDistanceScorer requires a build-time scoring callable:
//   [void|absl::Status] operator()(GingerNodeId query_id,
//                                  absl::Span<DistanceAndId> candidates)
//                                  const;
//   Computes distances between the dataset point `query_id` and the
//   candidates. May return absl::Status if accessing the query can fail
//   (e.g. query unpacking in TurboQuant).
template <typename T>
concept BuildDistanceScorer =
    requires(T t, GingerNodeId id, absl::Span<DistanceAndId> candidates) {
      { t(id, candidates) };
    };

// ============================================================================
// Float distance scoring functors
// ============================================================================

struct FloatSquaredL2 {
  const SpanDataset* dataset;
  absl::Span<const float> query = {};

  void operator()(absl::Span<DistanceAndId> candidates) const {
    SquaredL2Distance(query, candidates, *dataset);
  }

  void operator()(GingerNodeId query_id,
                  absl::Span<DistanceAndId> candidates) const {
    SquaredL2Distance(query_id, candidates, *dataset);
  }
};

struct FloatDotProduct {
  const SpanDataset* dataset;
  absl::Span<const float> query = {};

  void operator()(absl::Span<DistanceAndId> candidates) const {
    DotProductDistance(query, candidates, *dataset);
  }

  void operator()(GingerNodeId query_id,
                  absl::Span<DistanceAndId> candidates) const {
    DotProductDistance(query_id, candidates, *dataset);
  }
};

// ============================================================================
// Int8 distance scoring functors
// ============================================================================

struct Int8SquaredL2 {
  const SpanDataset* dataset;
  absl::Span<const int8_t> query = {};

  void operator()(absl::Span<DistanceAndId> candidates) const {
    SquaredL2Distance(query, candidates, *dataset);
  }

  void operator()(GingerNodeId query_id,
                  absl::Span<DistanceAndId> candidates) const {
    SquaredL2DistanceInt8(query_id, candidates, *dataset);
  }
};

struct Int8DotProduct {
  const SpanDataset* dataset;
  absl::Span<const int8_t> query = {};

  void operator()(absl::Span<DistanceAndId> candidates) const {
    DotProductDistance(query, candidates, *dataset);
  }

  void operator()(GingerNodeId query_id,
                  absl::Span<DistanceAndId> candidates) const {
    DotProductDistanceInt8(query_id, candidates, *dataset);
  }
};

// ============================================================================
// Uint8 distance scoring functors
// ============================================================================

struct Uint8SquaredL2 {
  const SpanDataset* dataset;
  absl::Span<const uint8_t> query = {};

  void operator()(absl::Span<DistanceAndId> candidates) const {
    SquaredL2Distance(query, candidates, *dataset);
  }

  void operator()(GingerNodeId query_id,
                  absl::Span<DistanceAndId> candidates) const {
    SquaredL2DistanceUint8(query_id, candidates, *dataset);
  }
};

struct Uint8DotProduct {
  const SpanDataset* dataset;
  absl::Span<const uint8_t> query = {};

  void operator()(absl::Span<DistanceAndId> candidates) const {
    DotProductDistance(query, candidates, *dataset);
  }

  void operator()(GingerNodeId query_id,
                  absl::Span<DistanceAndId> candidates) const {
    DotProductDistanceUint8(query_id, candidates, *dataset);
  }
};

// ============================================================================
// TurboQuant distance scoring functors
//
// Each functor supports two modes:
//   - Search: fixed_query is set, call operator()(candidates).
//   - Build:  quantizer is set, call operator()(query_id, candidates).
//             Uses a thread_local query cache for zero-allocation switching.
// ============================================================================

// Post-processes TurboQuant distance estimations, clamping small negative
// values that may arise due to quantization error under Squared L2 distance.
inline void PostProcessTurboQuantDistances(
    absl::Span<DistanceAndId> candidates) {
  for (auto& cand : candidates) {
    cand.distance = std::max(0.0f, cand.distance);
  }
}

// Computes TurboQuant distances from a dataset point to a set of candidates,
// reusing a thread_local query buffer to avoid per-call memory allocation.
inline absl::Status ComputeTurboQuantBuildDistances(
    const TurboQuantCodec& quantizer, const TurboQuantDatasetSoA& dataset,
    GingerNodeId query_id, absl::Span<DistanceAndId> candidates) {
  thread_local std::optional<TurboQuantQuery> local_query;
  if (!local_query.has_value() ||
      local_query->num_bytes_per_query() != quantizer.num_bytes_per_query()) {
    ABSL_ASSIGN_OR_RETURN(local_query,
                          TurboQuantQuery::CreateFromQuantizedDataPoint(
                              quantizer, dataset.DataPointView(query_id)));
  } else {
    ABSL_RETURN_IF_ERROR(local_query->ResetFromQuantizedDataPoint(
        quantizer, dataset.DataPointView(query_id)));
  }
  dataset.UnsafeComputeDistances(*local_query,
                                 CreateScoringAccessor(candidates));
  if (quantizer.metric() == TurboQuantDistanceMetric::kSquaredL2Distance) {
    PostProcessTurboQuantDistances(candidates);
  }
  return absl::OkStatus();
}

// Search-time TurboQuant scorer with a fixed pre-quantized query.
struct TurboQuantSearchScorer {
  const TurboQuantDatasetSoA* dataset;
  const TurboQuantQuery* fixed_query;

  void operator()(absl::Span<DistanceAndId> candidates) const {
    dataset->UnsafeComputeDistances(*fixed_query,
                                    CreateScoringAccessor(candidates));
  }
};

// Search-time TurboSlice scorer using a pre-quantized TurboSlice query prefix.
struct TurboSliceSearchScorer {
  const TurboQuantDatasetSoA* dataset;
  const TurboQuantQuery* turboslice_query;

  void operator()(absl::Span<DistanceAndId> candidates) const {
    ScoresAndIndicesDistanceAndIdView accessor(candidates);
    dataset->UnsafeComputeDistancesTurboSlice(*turboslice_query, accessor);
  }
};

// Build-time TurboQuant scorer that creates queries from dataset points.
struct TurboQuantBuildScorer {
  const TurboQuantDatasetSoA* dataset;
  const TurboQuantCodec* quantizer;

  absl::Status operator()(GingerNodeId query_id,
                          absl::Span<DistanceAndId> candidates) const {
    return ComputeTurboQuantBuildDistances(*quantizer, *dataset, query_id,
                                           candidates);
  }
};

// ============================================================================
// Dataset distance scorer helpers
// ============================================================================

using DatasetBuildScorer =
    std::function<void(GingerNodeId, absl::Span<DistanceAndId>)>;

using DatasetSearchScorer =
    std::variant<FloatSquaredL2, FloatDotProduct, Int8SquaredL2, Int8DotProduct,
                 Uint8SquaredL2, Uint8DotProduct>;

// Returns the build-time distance scoring function for a given coordinate type
// and distance metric.
inline absl::StatusOr<DatasetBuildScorer> GetDatasetBuildScorer(
    CoordinateType coordinate_type, DistanceMetric metric,
    const SpanDataset* dataset) {
  if (metric != SQUARED_L2 && metric != DOT_PRODUCT) {
    return absl::InvalidArgumentError("Unsupported distance metric.");
  }

  const bool is_squared_l2 = metric == SQUARED_L2;
  switch (coordinate_type) {
    case CoordinateType::kFloat:
      return is_squared_l2 ? DatasetBuildScorer(FloatSquaredL2{dataset})
                           : DatasetBuildScorer(FloatDotProduct{dataset});
    case CoordinateType::kInt8:
      return is_squared_l2 ? DatasetBuildScorer(Int8SquaredL2{dataset})
                           : DatasetBuildScorer(Int8DotProduct{dataset});
    case CoordinateType::kUint8:
      return is_squared_l2 ? DatasetBuildScorer(Uint8SquaredL2{dataset})
                           : DatasetBuildScorer(Uint8DotProduct{dataset});
    default:
      return absl::InvalidArgumentError("Unsupported coordinate type.");
  }
}

// Returns the search-time distance scoring function for a given coordinate type
// and distance metric.
inline absl::StatusOr<DatasetSearchScorer> GetDatasetSearchScorer(
    CoordinateType coordinate_type, DistanceMetric metric,
    const SpanDataset* dataset, absl::Span<const float> float_query = {},
    absl::Span<const int8_t> int8_query = {},
    absl::Span<const uint8_t> uint8_query = {}) {
  if (metric != SQUARED_L2 && metric != DOT_PRODUCT) {
    return absl::InvalidArgumentError("Unsupported distance metric.");
  }

  const bool is_squared_l2 = metric == SQUARED_L2;
  switch (coordinate_type) {
    case CoordinateType::kFloat:
      return is_squared_l2
                 ? DatasetSearchScorer(FloatSquaredL2{dataset, float_query})
                 : DatasetSearchScorer(FloatDotProduct{dataset, float_query});
    case CoordinateType::kInt8:
      return is_squared_l2
                 ? DatasetSearchScorer(Int8SquaredL2{dataset, int8_query})
                 : DatasetSearchScorer(Int8DotProduct{dataset, int8_query});
    case CoordinateType::kUint8:
      return is_squared_l2
                 ? DatasetSearchScorer(Uint8SquaredL2{dataset, uint8_query})
                 : DatasetSearchScorer(Uint8DotProduct{dataset, uint8_query});
    default:
      return absl::InvalidArgumentError("Unsupported coordinate type.");
  }
}

}  // namespace ginger

#endif  // GINGER_SEARCH_SEARCH_FUNCTORS_H_
