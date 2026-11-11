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

#ifndef GINGER_SEARCHER_H_
#define GINGER_SEARCHER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/build_progress.h"
#include "ginger/candidate_filtering/jl.h"
#include "ginger/dataset.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/projection/pca.h"
#include "ginger/query.h"
#include "ginger/searcher.pb.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"

// A static approximate nearest neighbor search index that supports L2 and dot
// product distances.
//
// This index is used to efficiently retrieve the top-k nearest neighbors for
// vector queries against large-scale datasets. It navigates an underlying
// proximity graph (such as Vamana or PiPNN) starting from a designated entry
// point using beam search. Searches can execute either on full-precision
// coordinates or on quantized coordinates compressed via TurboQuant, with
// optional reranking using full-precision vectors or TurboQuant extra bits. To
// accelerate queries on high-dimensional datasets, the searcher also supports
// dimensionality reduction via PCA as well as two-stage candidate filtering
// using JL projections or TurboSlice.
//
// Instances of `GingerSearcher` can be constructed either via the static
// `GingerSearcher::Create()` method or the `GingerSearcherBuilder` factory.
// At query time, callers specify search parameters via `SearchOptions` (such as
// target `num_neighbors`, `beam_size`, search space, and reranking strategy)
// and invoke `FindNeighbors()` with a query vector to receive the nearest
// neighbors. The following is an example usage:
//
//   ```cpp
//   // 1. Create the search index.
//   GingerSearcherConfig config;
//   config.set_distance_metric(DOT_PRODUCT);
//   // Configure graph construction parameters (e.g. Vamana config)...
//
//   ABSL_ASSIGN_OR_RETURN(std::unique_ptr<GingerSearcher> searcher,
//                         GingerSearcher::Create(config, dataset));
//
//   // 2. Perform the search on the index.
//   SearchOptions options;
//   options.set_num_neighbors(10);
//   options.set_beam_size(50);
//   options.set_search_space(SEARCH_SPACE_TURBO_QUANT);
//   options.set_reranking_strategy(RERANKING_STRATEGY_FLOAT);
//   options.set_rerank_factor(5);
//
//   ABSL_ASSIGN_OR_RETURN(std::vector<DistanceAndId> neighbors,
//                         searcher->FindNeighbors(query_span, options));
//   ```
namespace ginger {

class GingerSearcher {
 public:
  using IndexVariant =
      std::variant<std::monostate, std::unique_ptr<const FlatGraph>,
                   std::unique_ptr<const CSRGraph>,
                   std::unique_ptr<const CompressedCSRGraph<>>>;

  // Creates a `GingerSearcher` instance from a configuration and a dataset.
  //
  // For more advanced build controls (such as supplying pre-quantized datasets,
  // pre-built indices, etc.), use `GingerSearcherBuilder`.
  //
  // Args:
  //   - config: Configuration specifying distance metrics, Vamana/PiPNN graph
  //     parameters, TurboQuant quantization settings, and index build options.
  //   - dataset: The dataset containing vectors to index.
  //   - progress_callback: Optional callback invoked periodically with
  //     `IndexBuildProgress` during long-running index construction steps.
  //
  // Returns:
  //   - A `std::unique_ptr` to the initialized `GingerSearcher`, or an error
  //     status if the configuration is invalid or index construction fails.
  static absl::StatusOr<std::unique_ptr<GingerSearcher>> Create(
      const GingerSearcherConfig& config, std::shared_ptr<SpanDataset> dataset,
      std::function<void(const IndexBuildProgress&)> progress_callback =
          nullptr);

  GingerSearcher(const GingerSearcher&) = delete;
  GingerSearcher& operator=(const GingerSearcher&) = delete;

  GingerSearcher(GingerSearcher&&) = default;
  GingerSearcher& operator=(GingerSearcher&&) = default;

  // Compares two searchers for equality. Two searchers are equal if they have
  // the same number of points, dimensionality, graph entry point, ID mapping,
  // and underlying graphs. Attached datasets and quantizers are not compared.
  bool operator==(const GingerSearcher& other) const;
  bool operator!=(const GingerSearcher& other) const {
    return !(*this == other);
  }

  // Returns the configuration used to build the searcher.
  const GingerSearcherConfig& config() const { return config_; }

  // Returns the total number of points indexed by the searcher.
  GingerNodeId size() const { return num_points_; }

  // Returns the dimensionality of the vectors indexed by the searcher. Queries
  // accepted by the searcher must have this dimensionality.
  int dimensionality() const { return dimensionality_; }

  // Returns the coordinate data type of the underlying dataset (e.g. float,
  // int8, uint8). Queries accepted by the searcher must match this coordinate
  // type. Returns `CoordinateType::kFloat` if only quantized data is present.
  CoordinateType coordinate_type() const {
    if (dataset_ != nullptr) return dataset_->coordinate_type();
    return CoordinateType::kFloat;
  }

  // Returns a non-mutable reference to the underlying graph index. Depending on
  // the construction, it can be represented as a flat graph (`FlatGraph`), CSR
  // graph (`CSRGraph`), or compressed graph (`CompressedCSRGraph`).
  const IndexVariant& index() const { return index_; }

  // Returns the entry point of the underlying graph index.
  GingerNodeId entry_point() const { return entry_point_; }

  // Returns the node ID remapping table (reordered node ID -> original point
  // ID) used when index reordering is applied. Returns an empty span if no
  // reordering was applied.
  absl::Span<const GingerNodeId> remapping() const { return remapping_; }

  // Searches for the nearest neighbors of a query point.
  //
  // The query point can be supplied as a full-precision vector (with float,
  // int8_t, or uint8_t coordinates) or as a pre-quantized TurboQuant query. The
  // search runs a beam search on the graph index according to the configuration
  // in `options`, including candidate filtering (TurboSlice, JL) and
  // post-search reranking (float or TurboQuant extra precision).
  //
  // Results are returned as a vector of `DistanceAndId` representing the top
  // `options.num_neighbors()` matches, sorted in non-decreasing order of
  // distance (with ties broken by neighbor ID). If index remapping was applied
  // during graph construction, the returned IDs reflect the original dataset
  // node IDs.
  //
  // Args:
  //   - query: The query vector or pre-quantized query point. Callers can pass
  //     an `absl::Span`, `std::vector`, or `TurboQuantQuery` directly as
  //     `SearchQuery` is implicitly constructible.
  //   - options: Search options specifying beam size, number of neighbors,
  //     search space, and reranking/filtering strategy.
  //   - stats: Optional output pointer to collect search diagnostics (number of
  //     distance computations, nodes visited, candidates considered, filter
  //     computations, and reranked items). If nullptr, statistics tracking is
  //     omitted.
  //
  // Returns:
  //   - A sorted vector of nearest neighbors, or an error status if `options`
  //     are invalid or required dataset components are missing.
  absl::StatusOr<std::vector<DistanceAndId>> FindNeighbors(
      SearchQuery query, const SearchOptions& options,
      BeamSearchResult::Statistics* stats = nullptr) const;

  // Returns search options with `num_neighbors` and `beam_size` and defaults
  // for the remaining fields that fit this searcher's datasets. Searches over
  // TurboQuant codes if available, otherwise over floats. Reranks with
  // TurboQuant extra bits if available, otherwise with floats if the search is
  // quantized and the float dataset was kept.
  SearchOptions DefaultSearchOptions(int num_neighbors, int beam_size) const;

 private:
  struct PreparedQuery;
  friend class GingerSearcherBuilder;
  friend struct GingerSearcherAccess;

  GingerSearcher(
      GingerSearcherConfig config, GingerNodeId num_points, int dimensionality,
      std::shared_ptr<const SpanDataset> dataset,
      std::shared_ptr<const TurboQuantDatasetSoA> quantized_dataset,
      std::shared_ptr<const TurboQuantCodec> quantizer,
      std::shared_ptr<const JlDataset> jl_dataset, GingerNodeId entry_point,
      IndexVariant index, std::unique_ptr<const PcaProjection> projection,
      std::vector<GingerNodeId> remapping,
      std::shared_ptr<const TurboQuantDatasetSoA> rerank_quantized_dataset,
      std::shared_ptr<const TurboQuantCodec> rerank_quantizer,
      std::optional<uint32_t> dataset_fingerprint);

  absl::Status ValidateSearchOptions(const SearchQuery& search_query,
                                     const SearchOptions& options) const;

  absl::StatusOr<PreparedQuery> PrepareQuery(
      const SearchQuery& search_query, const SearchOptions& options) const;

  absl::StatusOr<std::vector<DistanceAndId>> DispatchSearch(
      const PreparedQuery& query, const SearchOptions& options,
      BeamSearchResult::Statistics* stats) const;

  void DispatchRerank(std::vector<DistanceAndId>& results,
                      const PreparedQuery& query, const SearchOptions& options,
                      BeamSearchResult::Statistics* stats) const;

  void RemapCandidateIds(std::vector<DistanceAndId>& candidates) const;

  GingerSearcherConfig config_;
  GingerNodeId num_points_;
  int dimensionality_;

  std::shared_ptr<const SpanDataset> dataset_;
  std::shared_ptr<const TurboQuantDatasetSoA> quantized_dataset_;
  std::shared_ptr<const TurboQuantCodec> quantizer_;
  std::shared_ptr<const JlDataset> jl_dataset_;

  GingerNodeId entry_point_;
  IndexVariant index_;
  std::unique_ptr<const PcaProjection> projection_;
  std::vector<GingerNodeId> remapping_;

  // Full-dimensional TurboQuant dataset with extra bits, used instead of
  // `quantized_dataset_` for extra-bits reranking when PCA is enabled. Null
  // otherwise.
  std::shared_ptr<const TurboQuantDatasetSoA> rerank_quantized_dataset_;
  std::shared_ptr<const TurboQuantCodec> rerank_quantizer_;

  // Fingerprint of the float dataset in its original point order, if the
  // searcher was built from one.
  std::optional<uint32_t> dataset_fingerprint_;
};

}  // namespace ginger

#endif  // GINGER_SEARCHER_H_
