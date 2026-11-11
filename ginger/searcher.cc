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

#include "ginger/searcher.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/functional/overload.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ginger/build_progress.h"
#include "ginger/candidate_filtering/candidate_filter.h"
#include "ginger/candidate_filtering/jl.h"
#include "ginger/dataset.h"
#include "ginger/distance/distance.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/projection/pca.h"
#include "ginger/query.h"
#include "ginger/search/beam_search.h"
#include "ginger/search/search_functors.h"
#include "ginger/searcher.pb.h"
#include "ginger/searcher_builder.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/scoring/scoring_accessors.h"
#include "ginger/types.h"

namespace ginger {

namespace {

// Truncates `results` to at most `num_neighbors` entries with the smallest
// distances and sorts them in ascending order of distance.
void FilterTopK(std::vector<DistanceAndId>& results, int num_neighbors) {
  if (results.size() > num_neighbors) {
    absl::c_nth_element(results, results.begin() + num_neighbors);
    results.resize(num_neighbors);
  }
  absl::c_sort(results);
}

// Runs beam search on `index` starting from `entry_point` using `compute_dist`
// to score candidate distances. Returns the final beam of candidate nodes and
// their distances.
template <SearchDistanceScorer ComputeDistancesFn>
absl::StatusOr<std::vector<DistanceAndId>> DispatchBeamSearch(
    GingerNodeId entry_point, const GingerSearcher::IndexVariant& index,
    ComputeDistancesFn compute_dist, int beam_size,
    BeamSearchResult::Statistics* stats) {
  auto nav_scorer =
      [&](absl::Span<DistanceAndId> candidates,
          absl::Span<const BeamEntry> beam) -> absl::Span<DistanceAndId> {
    if constexpr (SearchDistanceScorerWithBeam<ComputeDistancesFn>) {
      return compute_dist(candidates, beam);
    } else {
      compute_dist(candidates);
      return candidates;
    }
  };

  auto run_search =
      [&](const auto& index_ptr) -> absl::StatusOr<std::vector<DistanceAndId>> {
    using T = std::decay_t<decltype(index_ptr)>;
    // the monostate check is required by std::visit.
    if constexpr (!std::is_same_v<T, std::monostate>) {
      const auto& graph = *index_ptr;

      BeamSearchResult res =
          NavigationBeamSearch(graph, nav_scorer, {entry_point}, beam_size);
      if (stats != nullptr) {
        stats->num_candidates_considered += res.stats.num_candidates_considered;
        stats->num_distance_computations += res.stats.num_distance_computations;
        stats->num_nodes_visited += res.stats.num_nodes_visited;
      }

      return std::move(res.final_beam);
    }

    ABSL_UNREACHABLE();
  };

  return std::visit(run_search, index);
}

// Runs beam search on `index` using `filter_scorer` to filter candidates before
// scoring survivors with `main_scorer`. Returns the final beam of candidate
// nodes and their distances.
template <SearchDistanceScorer FilterScorer, SearchDistanceScorer MainScorer>
absl::StatusOr<std::vector<DistanceAndId>>
DispatchBeamSearchWithCandidateFiltering(
    GingerNodeId entry_point, const GingerSearcher::IndexVariant& index,
    FilterScorer filter_scorer, MainScorer main_scorer, int beam_size,
    int beam_mean_width, BeamSearchResult::Statistics* stats) {
  CandidateFilter candidate_filter(
      std::move(filter_scorer),
      BeamMeanThresholdPolicy(beam_size, beam_mean_width));
  auto two_stage_scorer = [&](absl::Span<DistanceAndId> candidates,
                              absl::Span<const BeamEntry> beam) {
    absl::Span<DistanceAndId> survivors = candidate_filter(candidates, beam);
    main_scorer(survivors);
    return survivors;
  };

  auto results = DispatchBeamSearch(entry_point, index, two_stage_scorer,
                                    beam_size, stats);
  if (stats != nullptr) {
    stats->num_filter_computations +=
        candidate_filter.num_filter_computations();
  }

  return results;
}

}  // namespace

// Internal container storing the preprocessed query required for search.
//
// Unlike the user-facing `SearchQuery`, `PreparedQuery` holds validated
// coordinate views, owns intermediate buffers (e.g., used for PCA projection or
// byte-to-float conversions), and owns quantized query representations
// (TurboQuant, TurboSlice, JL).
struct GingerSearcher::PreparedQuery {
  // Buffer for transformed float coordinates (e.g., PCA projection or
  // byte-to-float conversion) that must remain valid during query execution.
  std::vector<float> float_buffer;

  // Non-owning views of the raw input coordinates.
  absl::Span<const float> float_data;
  absl::Span<const int8_t> int8_data;
  absl::Span<const uint8_t> uint8_data;

  // TurboQuant representation, either quantized on-the-fly (owned) or
  // referenced from a pre-quantized query (unowned).
  std::optional<TurboQuantQuery> owned_turboquant;
  const TurboQuantQuery* unowned_turboquant = nullptr;

  // Owned query representations used for filtering candidates.
  std::optional<TurboQuantQuery> owned_turboslice;
  std::optional<JlQuery> owned_jl;

  // Full-dimensional TurboQuant query for extra-bits reranking on the
  // searcher's separate rerank dataset (only used when PCA is enabled).
  std::optional<TurboQuantQuery> owned_rerank_turboquant;

  template <typename T>
  void SetData(absl::Span<const T> span) {
    if constexpr (std::is_same_v<T, float>) {
      float_data = span;
    } else if constexpr (std::is_same_v<T, int8_t>) {
      int8_data = span;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      uint8_data = span;
    } else {
      static_assert(sizeof(T) == 0, "Unsupported query data type.");
    }
  }

  const TurboQuantQuery* turboquant() const {
    return owned_turboquant.has_value() ? &*owned_turboquant
                                        : unowned_turboquant;
  }
  const TurboQuantQuery* turboslice() const {
    return owned_turboslice.has_value() ? &*owned_turboslice : nullptr;
  }
  const JlQuery* jl() const {
    return owned_jl.has_value() ? &*owned_jl : nullptr;
  }
  const TurboQuantQuery* rerank_turboquant() const {
    return owned_rerank_turboquant.has_value() ? &*owned_rerank_turboquant
                                               : nullptr;
  }
};

absl::StatusOr<std::unique_ptr<GingerSearcher>> GingerSearcher::Create(
    const GingerSearcherConfig& config, std::shared_ptr<SpanDataset> dataset,
    std::function<void(const IndexBuildProgress&)> progress_callback) {
  return GingerSearcherBuilder(config)
      .AddMutableDataset(std::move(dataset))
      .SetProgressCallback(std::move(progress_callback))
      .Build();
}

GingerSearcher::GingerSearcher(
    GingerSearcherConfig config, GingerNodeId num_points, int dimensionality,
    std::shared_ptr<const SpanDataset> dataset,
    std::shared_ptr<const TurboQuantDatasetSoA> quantized_dataset,
    std::shared_ptr<const TurboQuantCodec> quantizer,
    std::shared_ptr<const JlDataset> jl_dataset, GingerNodeId entry_point,
    IndexVariant index, std::unique_ptr<const PcaProjection> projection,
    std::vector<GingerNodeId> remapping,
    std::shared_ptr<const TurboQuantDatasetSoA> rerank_quantized_dataset,
    std::shared_ptr<const TurboQuantCodec> rerank_quantizer,
    std::optional<uint32_t> dataset_fingerprint)
    : config_(std::move(config)),
      num_points_(num_points),
      dimensionality_(dimensionality),
      dataset_(std::move(dataset)),
      quantized_dataset_(std::move(quantized_dataset)),
      quantizer_(std::move(quantizer)),
      jl_dataset_(std::move(jl_dataset)),
      entry_point_(entry_point),
      index_(std::move(index)),
      projection_(std::move(projection)),
      remapping_(std::move(remapping)),
      rerank_quantized_dataset_(std::move(rerank_quantized_dataset)),
      rerank_quantizer_(std::move(rerank_quantizer)),
      dataset_fingerprint_(dataset_fingerprint) {
  ABSL_CHECK_GE(num_points_, 0);
  ABSL_CHECK_GE(dimensionality_, 0);
  ABSL_CHECK_EQ(quantized_dataset_ == nullptr, quantizer_ == nullptr);
  ABSL_CHECK_EQ(rerank_quantized_dataset_ == nullptr,
                rerank_quantizer_ == nullptr);
  ABSL_CHECK_GE(entry_point_, 0);
  ABSL_CHECK_LT(entry_point_, num_points);
  ABSL_CHECK(!std::holds_alternative<std::monostate>(index_));
}

bool GingerSearcher::operator==(const GingerSearcher& other) const {
  if (num_points_ != other.num_points_ ||
      dimensionality_ != other.dimensionality_ ||
      entry_point_ != other.entry_point_ || remapping_ != other.remapping_) {
    return false;
  }

  return std::visit(
      [](const auto& index1, const auto& index2) -> bool {
        using T1 = std::decay_t<decltype(index1)>;
        using T2 = std::decay_t<decltype(index2)>;

        if constexpr (std::is_same_v<T1, T2>) {
          if constexpr (std::is_same_v<T1, std::monostate>) {
            return true;
          } else {
            if ((index1 == nullptr) != (index2 == nullptr)) return false;
            if (index1 == nullptr) return true;
            return *index1 == *index2;
          }
        }

        return false;
      },
      index_, other.index_);
}

absl::StatusOr<std::vector<DistanceAndId>> GingerSearcher::FindNeighbors(
    SearchQuery search_query, const SearchOptions& options,
    BeamSearchResult::Statistics* stats) const {
  ABSL_RETURN_IF_ERROR(ValidateSearchOptions(search_query, options));

  ABSL_ASSIGN_OR_RETURN(PreparedQuery query,
                        PrepareQuery(search_query, options));
  ABSL_ASSIGN_OR_RETURN(std::vector<DistanceAndId> results,
                        DispatchSearch(query, options, stats));
  DispatchRerank(results, query, options, stats);

  FilterTopK(results, options.num_neighbors());
  RemapCandidateIds(results);
  return results;
}

SearchOptions GingerSearcher::DefaultSearchOptions(int num_neighbors,
                                                   int beam_size) const {
  SearchOptions options;
  options.set_num_neighbors(num_neighbors);
  options.set_beam_size(beam_size);
  const bool use_tq = quantizer_ != nullptr;
  options.set_search_space(use_tq ? SEARCH_SPACE_TURBO_QUANT
                                  : SEARCH_SPACE_FLOAT);

  const TurboQuantDatasetSoA* extra_bits_dataset =
      rerank_quantized_dataset_ != nullptr ? rerank_quantized_dataset_.get()
                                           : quantized_dataset_.get();
  if (extra_bits_dataset != nullptr &&
      extra_bits_dataset->extra_bits_per_dim() > 0) {
    options.set_reranking_strategy(RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS);
  } else if (use_tq && dataset_ != nullptr &&
             dataset_->coordinate_type() == CoordinateType::kFloat) {
    options.set_reranking_strategy(RERANKING_STRATEGY_FLOAT);
  } else {
    options.set_reranking_strategy(RERANKING_STRATEGY_NONE);
    return options;
  }
  options.set_rerank_factor(6.0f);
  return options;
}

absl::StatusOr<GingerSearcher::PreparedQuery> GingerSearcher::PrepareQuery(
    const SearchQuery& search_query, const SearchOptions& options) const {
  const bool use_tq = options.search_space() == SEARCH_SPACE_TURBO_QUANT;
  const bool use_tq_extra_bits =
      options.reranking_strategy() == RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS;
  // With a separate rerank dataset, extra-bits reranking uses its own query,
  // so the search query does not need higher precision.
  const bool use_rerank_dataset =
      use_tq_extra_bits && rerank_quantized_dataset_ != nullptr;
  // Without a separate rerank dataset, extra-bits reranking scores against
  // `quantized_dataset_` with the TurboQuant query, so the query is needed
  // even when searching in float space.
  const bool needs_turboquant_query =
      use_tq || (use_tq_extra_bits && !use_rerank_dataset);

  // Unpacks the underlying representation from `search_query` and initializes
  // `query`. For non-quantized queries, it validates type compatibility with
  // the dataset and prepares a TurboQuant query if searching in quantized
  // space or reranking with extra bits from `quantized_dataset_`. For
  // pre-quantized inputs, it stores the TurboQuant query directly.
  PreparedQuery query;
  auto visitor = absl::Overload{
      [&]<typename T>(absl::Span<const T> span) -> absl::Status {
        if (coordinate_type() != CoordinateTypeFor<T>()) {
          return absl::InvalidArgumentError(absl::StrCat(
              CoordinateTypeName<T>(), " query is only supported for ",
              CoordinateTypeName<T>(), " datasets."));
        }
        if (span.size() != static_cast<size_t>(dimensionality_)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Query has ", span.size(), " dimensions, but the searcher has ",
              dimensionality_, "."));
        }
        query.SetData(span);

        if (needs_turboquant_query) {
          ABSL_CHECK(quantizer_ != nullptr);
          absl::Span<const float> query_span;

          // Prepares the float coordinates for TurboQuant quantization. For
          // float queries, applies PCA projection if configured. For byte
          // queries (int8/uint8), converts the coordinates to floats without
          // projection.
          if constexpr (std::is_same_v<T, float>) {
            if (projection_ != nullptr) {
              ABSL_ASSIGN_OR_RETURN(query.float_buffer,
                                    projection_->Project(span));
              query_span = query.float_buffer;
            } else {
              query_span = span;
            }
          } else {
            ABSL_DCHECK(projection_ == nullptr);
            query.float_buffer.assign(span.begin(), span.end());
            query_span = query.float_buffer;
          }

          ABSL_ASSIGN_OR_RETURN(query.owned_turboquant,
                                TurboQuantQuery::Create(
                                    *quantizer_, query_span,
                                    use_tq_extra_bits && !use_rerank_dataset));
        }

        return absl::OkStatus();
      },
      [&](std::reference_wrapper<const TurboQuantQuery> ref) -> absl::Status {
        ABSL_CHECK(quantizer_ != nullptr);
        const TurboQuantQuery& tq_query = ref.get();
        if (tq_query.dimensionality() != quantizer_->dimensionality()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "TurboQuant query has ", tq_query.dimensionality(),
              " dimensions, but the searcher's TurboQuant codec has ",
              quantizer_->dimensionality(), "."));
        }
        if (use_tq_extra_bits && tq_query.float_query_data().empty()) {
          return absl::InvalidArgumentError(
              "TurboQuant query is missing float query data required for extra "
              "bits reranking.");
        }
        query.unowned_turboquant = &tq_query;
        return absl::OkStatus();
      },
      [&](const TurboQuantBasePointView& view) {
        ABSL_CHECK(quantizer_ != nullptr);
        ABSL_ASSIGN_OR_RETURN(
            query.owned_turboquant,
            use_tq_extra_bits
                ? TurboQuantQuery::
                      CreateFromQuantizedDataPointWithExtraPrecision(
                          *quantizer_, view)
                : TurboQuantQuery::CreateFromQuantizedDataPoint(*quantizer_,
                                                                view));
        return absl::OkStatus();
      },
  };
  ABSL_RETURN_IF_ERROR(std::visit(visitor, search_query.data()));

  if (use_rerank_dataset) {
    ABSL_ASSIGN_OR_RETURN(query.owned_rerank_turboquant,
                          TurboQuantQuery::CreateForExtraPrecision(
                              *rerank_quantizer_, query.float_data));
  }

  if (options.candidate_filter().has_turbo_slice()) {
    ABSL_CHECK(quantizer_ != nullptr);
    ABSL_CHECK(query.turboquant() != nullptr);
    ABSL_ASSIGN_OR_RETURN(query.owned_turboslice,
                          TurboQuantQuery::CreateTurboSliceQuery(
                              *quantizer_, *query.turboquant()));
  }

  if (options.candidate_filter().has_jl()) {
    ABSL_CHECK(jl_dataset_ != nullptr);
    ABSL_CHECK(!query.float_data.empty());
    query.owned_jl = jl_dataset_->QuantizeQuery(query.float_data);
  }

  return query;
}

absl::StatusOr<std::vector<DistanceAndId>> GingerSearcher::DispatchSearch(
    const PreparedQuery& query, const SearchOptions& options,
    BeamSearchResult::Statistics* stats) const {
  const bool use_tq = options.search_space() == SEARCH_SPACE_TURBO_QUANT;
  const int beam_size = options.beam_size();

  auto run_search = [&](const auto& main_scorer) {
    const int beam_mean_width = options.candidate_filter().beam_mean_width();

    if (options.candidate_filter().has_turbo_slice()) {
      ABSL_CHECK(query.turboslice() != nullptr);
      TurboSliceSearchScorer turboslice_scorer{quantized_dataset_.get(),
                                               query.turboslice()};
      return DispatchBeamSearchWithCandidateFiltering(
          entry_point_, index_, turboslice_scorer, main_scorer, beam_size,
          beam_mean_width, stats);
    }
    if (options.candidate_filter().has_jl()) {
      ABSL_CHECK(query.jl() != nullptr);
      ABSL_CHECK(jl_dataset_ != nullptr);
      JlSearchScorer jl_scorer{jl_dataset_.get(), query.jl()};
      return DispatchBeamSearchWithCandidateFiltering(
          entry_point_, index_, jl_scorer, main_scorer, beam_size,
          beam_mean_width, stats);
    }

    return DispatchBeamSearch(entry_point_, index_, main_scorer, beam_size,
                              stats);
  };

  if (use_tq) {
    ABSL_CHECK(query.turboquant() != nullptr);
    return run_search(TurboQuantSearchScorer{
        quantized_dataset_.get(),
        query.turboquant(),
    });
  } else {
    ABSL_ASSIGN_OR_RETURN(
        DatasetSearchScorer scorer,
        GetDatasetSearchScorer(dataset_->coordinate_type(),
                               config_.distance_metric(), dataset_.get(),
                               query.float_data, query.int8_data,
                               query.uint8_data));
    return std::visit(run_search, scorer);
  }
}

void GingerSearcher::DispatchRerank(std::vector<DistanceAndId>& results,
                                    const PreparedQuery& query,
                                    const SearchOptions& options,
                                    BeamSearchResult::Statistics* stats) const {
  const bool float_vector_rerank =
      options.reranking_strategy() == RERANKING_STRATEGY_FLOAT;
  const bool rerank_extra_bits =
      options.reranking_strategy() == RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS;

  if (options.has_rerank_factor()) {
    // We limit the number of candidates passed to the reranking stage to avoid
    // computing distance on candidates that are unlikely to be in the final top
    // `num_neighbors`.
    const double max_to_rerank =
        std::max<double>(options.num_neighbors(),
                         options.num_neighbors() * options.rerank_factor());
    if (max_to_rerank < results.size()) {
      results.resize(static_cast<size_t>(max_to_rerank));
    }
  }

  if (float_vector_rerank) {
    ABSL_CHECK(dataset_ != nullptr);
    ABSL_CHECK(dataset_->coordinate_type() == CoordinateType::kFloat);
    ABSL_CHECK(!query.float_data.empty());
    if (config_.distance_metric() == SQUARED_L2) {
      SquaredL2Distance(query.float_data, absl::MakeSpan(results), *dataset_);
    } else {
      DotProductDistance(query.float_data, absl::MakeSpan(results), *dataset_);
    }
    if (stats != nullptr) stats->num_reranked += results.size();
    return;
  }

  if (rerank_extra_bits) {
    const TurboQuantDatasetSoA* rerank_dataset = quantized_dataset_.get();
    const TurboQuantQuery* rerank_query = query.turboquant();
    if (rerank_quantized_dataset_ != nullptr) {
      rerank_dataset = rerank_quantized_dataset_.get();
      rerank_query = query.rerank_turboquant();
    }
    ABSL_CHECK(rerank_dataset != nullptr);
    ABSL_CHECK(rerank_query != nullptr);
    rerank_dataset->UnsafeComputeDistancesExtraPrecision(
        *rerank_query, CreateScoringAccessor(absl::MakeSpan(results)));
    if (stats != nullptr) stats->num_reranked += results.size();
  }

  // The final distances are TurboQuant estimates if they were computed by
  // extra-bits reranking or, without reranking, by TurboQuant search.
  const bool turboquant_distances =
      rerank_extra_bits || options.search_space() == SEARCH_SPACE_TURBO_QUANT;
  if (turboquant_distances && config_.distance_metric() == SQUARED_L2) {
    PostProcessTurboQuantDistances(absl::MakeSpan(results));
  }
}

void GingerSearcher::RemapCandidateIds(
    std::vector<DistanceAndId>& candidates) const {
  if (remapping_.empty()) return;

  for (DistanceAndId& result : candidates) {
    result.id = remapping_[result.id];
  }
}

absl::Status GingerSearcher::ValidateSearchOptions(
    const SearchQuery& search_query, const SearchOptions& options) const {
  if (options.num_neighbors() <= 0) {
    return absl::InvalidArgumentError(
        "num_neighbors must be positive in SearchOptions.");
  }
  if (options.beam_size() <= 0) {
    return absl::InvalidArgumentError(
        "beam_size must be positive in SearchOptions.");
  }
  if (options.has_rerank_factor() &&
      !(options.rerank_factor() > 0 &&
        std::isfinite(options.rerank_factor()))) {
    return absl::InvalidArgumentError(
        "rerank_factor must be positive and finite when set.");
  }
  if (options.search_space() == SEARCH_SPACE_UNSPECIFIED) {
    return absl::InvalidArgumentError(
        "search_space must be specified in SearchOptions.");
  }
  if (options.reranking_strategy() == RERANKING_STRATEGY_UNSPECIFIED) {
    return absl::InvalidArgumentError(
        "reranking_strategy must be specified in SearchOptions.");
  }

  const bool quantized_only =
      std::holds_alternative<std::reference_wrapper<const TurboQuantQuery>>(
          search_query.data()) ||
      std::holds_alternative<TurboQuantBasePointView>(search_query.data());
  if (quantized_only) {
    if (options.search_space() == SEARCH_SPACE_FLOAT) {
      return absl::InvalidArgumentError(
          "Float search space is not available for pre-quantized queries.");
    }
    if (options.reranking_strategy() == RERANKING_STRATEGY_FLOAT) {
      return absl::InvalidArgumentError(
          "Float reranking is not available for pre-quantized queries.");
    }
    if (options.search_space() == SEARCH_SPACE_TURBO_QUANT &&
        projection_ != nullptr) {
      return absl::InvalidArgumentError(
          "PCA projection is not supported for pre-quantized queries.");
    }
  }

  if (options.search_space() == SEARCH_SPACE_TURBO_QUANT) {
    if (quantizer_ == nullptr) {
      return absl::FailedPreconditionError(
          "TurboQuant coordinates not available.");
    }
  } else if (options.search_space() == SEARCH_SPACE_FLOAT) {
    if (dataset_ == nullptr) {
      return absl::FailedPreconditionError("Float dataset not available.");
    }
  } else {
    return absl::FailedPreconditionError(
        "No coordinates available for search.");
  }

  if (options.reranking_strategy() == RERANKING_STRATEGY_FLOAT) {
    if (dataset_ == nullptr ||
        dataset_->coordinate_type() != CoordinateType::kFloat) {
      return absl::FailedPreconditionError("Reranking requires float dataset.");
    }
  }
  if (options.reranking_strategy() ==
      RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS) {
    if (rerank_quantized_dataset_ != nullptr &&
        !std::holds_alternative<absl::Span<const float>>(search_query.data())) {
      return absl::InvalidArgumentError(
          "Extra bits reranking with a rerank dataset requires float query "
          "vectors.");
    }
    const TurboQuantDatasetSoA* extra_bits_dataset =
        rerank_quantized_dataset_ != nullptr ? rerank_quantized_dataset_.get()
                                             : quantized_dataset_.get();
    if (extra_bits_dataset == nullptr ||
        extra_bits_dataset->extra_bits_per_dim() <= 0) {
      return absl::FailedPreconditionError(
          "Extra bits reranking requested but quantized dataset has no extra "
          "bits.");
    }
  }

  if (options.candidate_filter().beam_mean_width() < 0) {
    return absl::InvalidArgumentError(
        "candidate_filter.beam_mean_width must not be negative.");
  }
  if (options.candidate_filter().has_turbo_slice()) {
    if (options.search_space() != SEARCH_SPACE_TURBO_QUANT) {
      return absl::InvalidArgumentError(
          "TurboSlice candidate filtering is only supported for "
          "SEARCH_SPACE_TURBO_QUANT.");
    }
    if (quantizer_ == nullptr || quantizer_->turboslice_dimensionality() == 0 ||
        quantized_dataset_ == nullptr ||
        !quantized_dataset_->IsTurboSliceEnabled()) {
      return absl::FailedPreconditionError(
          "TurboSlice candidate filtering requested, but TurboSlice is not "
          "enabled on the quantizer or dataset.");
    }
  }

  if (options.candidate_filter().has_jl()) {
    const bool is_float_query =
        std::holds_alternative<absl::Span<const float>>(search_query.data());
    if (!is_float_query) {
      return absl::InvalidArgumentError(
          "JL candidate filtering requires float query vectors.");
    }
    if (jl_dataset_ == nullptr) {
      return absl::FailedPreconditionError(
          "JL candidate filtering requested, but JL dataset is not available "
          "on the searcher.");
    }
  }

  return absl::OkStatus();
}

}  // namespace ginger
