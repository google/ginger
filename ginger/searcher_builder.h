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

#ifndef GINGER_SEARCHER_BUILDER_H_
#define GINGER_SEARCHER_BUILDER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "ginger/build_progress.h"
#include "ginger/candidate_filtering/jl.h"
#include "ginger/dataset.h"
#include "ginger/projection/pca.h"
#include "ginger/searcher.h"
#include "ginger/searcher.pb.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/types.h"
#include "ginger/vamana/index_builder.h"

namespace ginger {

// Builder for constructing `GingerSearcher` instances.
//
// This class simplifies the construction of a `GingerSearcher` by providing a
// flexible interface. It can be initialized with a `GingerSearcherConfig` proto
// or configured manually via setter methods. At a minimum, the user must
// provide:
// - Dataset: A full-precision dataset, a quantized dataset, or both.
// - Index: A pre-built index or a configuration to build one.
//
// This is a single-use builder. Calling `Build()` consumes internal state. It
// must not be called more than once on the same builder instance.
//
// Example usage:
//
// 1. Build using a configuration proto and a dataset:
//
// ```cpp
// GingerSearcherConfig searcher_config = ...;
// ABSL_ASSIGN_OR_RETURN(std::unique_ptr<GingerSearcher> searcher,
//                       GingerSearcherBuilder(searcher_config)
//                           .AddDataset(dataset)
//                           .Build());
// ```
//
// 2. Build from scratch: The builder automatically handles quantization and
//    index construction using the provided configuration and raw dataset.
//
// ```cpp
// ABSL_ASSIGN_OR_RETURN(std::unique_ptr<GingerSearcher> searcher,
//                       GingerSearcherBuilder()
//                           .AddDataset(dataset)
//                           .SetVamanaConfig(vamana_config)
//                           .SetTurboQuantConfig(tq_config)
//                           .SetDistanceMetric(SQUARED_L2)
//                           .Build());
// ```
//
// 3. Use pre-built components: Providing a pre-computed index and quantized
//    dataset skips the build steps.
//
// ```cpp
// ABSL_ASSIGN_OR_RETURN(std::unique_ptr<GingerSearcher> searcher,
//                       GingerSearcherBuilder()
//                           .AddDataset(dataset)
//                           .AddQuantizedDataset(quantized_dataset, quantizer)
//                           .AddIndex(prebuilt_index)
//                           .SetDistanceMetric(SQUARED_L2)
//                           .Build());
// ```
//
class GingerSearcherBuilder {
 public:
  using ProgressCallback = std::function<void(const IndexBuildProgress&)>;

  // Statistics of the build process, containing the time spent on computing
  // the PCA projection, quantization, index construction, and graph
  // compression.
  struct BuildStats {
    absl::Duration projection_time = absl::ZeroDuration();
    absl::Duration quantization_time = absl::ZeroDuration();
    absl::Duration jl_quantization_time = absl::ZeroDuration();
    absl::Duration construction_time = absl::ZeroDuration();

    absl::Duration compute_reordering_time = absl::ZeroDuration();
    absl::Duration apply_reordering_time = absl::ZeroDuration();
    absl::Duration graph_compression_time = absl::ZeroDuration();
  };

  GingerSearcherBuilder() = default;
  explicit GingerSearcherBuilder(GingerSearcherConfig config);

  // Adds a dataset to the searcher. This dataset is required if you intend to
  // use TurboQuant quantization or PCA projection and do not provide a
  // pre-quantized dataset.
  GingerSearcherBuilder& AddDataset(std::shared_ptr<const SpanDataset> dataset);

  // Has the same interface as the method above, but allows for reordering the
  // dataset in-place when reordering is enabled to reduce the memory overhead
  // of copying the dataset.
  GingerSearcherBuilder& AddMutableDataset(
      std::shared_ptr<SpanDataset> dataset);

  // Adds a quantized dataset and its corresponding quantizer to the searcher.
  // Requirements for the quantized dataset and quantizer:
  //  - The quantizer metric must match the specified metric in the
  //    `GingerSearcherConfig` or the one set by `SetDistanceMetric`.
  //  - The quantizer must have the same parameters as the quantizer used to
  //    quantize the dataset (dimensionality and random seed).
  //  - If a dataset is provided, the number of datapoints and the
  //    dimensionality in the quantized dataset must match the dataset.
  GingerSearcherBuilder& AddQuantizedDataset(
      std::shared_ptr<const TurboQuantDatasetSoA> quantized_dataset,
      std::shared_ptr<const TurboQuantCodec> quantizer);

  // Has the same interface as the method above, but allows for reordering the
  // quantized dataset in-place when reordering is enabled to reduce the memory
  // overhead of copying the dataset.
  GingerSearcherBuilder& AddMutableQuantizedDataset(
      std::shared_ptr<TurboQuantDatasetSoA> quantized_dataset,
      std::shared_ptr<const TurboQuantCodec> quantizer);

  // Adds a JL (Johnson-Lindenstrauss) dataset to the searcher.
  GingerSearcherBuilder& AddJlDataset(
      std::shared_ptr<const JlDataset> jl_dataset);

  // Has the same interface as the method above, but allows for reordering the
  // JL dataset in-place when reordering is enabled.
  GingerSearcherBuilder& AddMutableJlDataset(
      std::shared_ptr<JlDataset> jl_dataset);

  // Adds a pre-computed PCA projection to the searcher.
  GingerSearcherBuilder& AddProjection(
      std::unique_ptr<const PcaProjection> projection);

  // Adds a pre-built index to the searcher. The index must have the same size
  // as the dataset.
  GingerSearcherBuilder& AddIndex(GingerSearcher::IndexVariant index);

  // Adds a pre-computed reordering for the points in the index. Reordering
  // permutes the layout of the points in the dataset and the index graph to
  // improve locality of nodes in the graph.
  //
  // Note that this is done in-place when a mutable dataset is provided.
  // Otherwise, a copy of the dataset is done, which can result in a large
  // memory overhead.
  //
  // Requirements for the reordering:
  //   - The reordering must contain all points of the index.
  //   - The reordering must be a permutation of those points.
  GingerSearcherBuilder& AddReordering(std::vector<GingerNodeId> reordering);

  // Sets the search entry-point node ID in the index. When not set, defaults
  // to `kVamanaEntryPoint`. If `Build()` reorders the index, this entry point
  // is remapped to `reordering[entry_point]`.
  GingerSearcherBuilder& SetEntryPoint(GingerNodeId entry_point);

  // Adds a pre-computed inverse reordering (`reordered_id -> original_id`)
  // for an index and dataset that have already been reordered (e.g., when
  // loading a serialized searcher). Unlike `AddReordering`
  // (`original_id -> reordered_id`), this method does not reorder the dataset
  // or graph; `GingerSearcher::FindNeighbors()` uses it to map internal graph
  // node IDs back to original dataset point IDs.
  //
  // `Build()` returns `absl::InvalidArgumentError` if this is set when
  // reordering is enabled.
  GingerSearcherBuilder& AddInverseReordering(
      std::vector<GingerNodeId> inverse_reordering);

  // Sets the distance metric to be used for construction and search.
  GingerSearcherBuilder& SetDistanceMetric(DistanceMetric metric);

  // Configures the Vamana index construction parameters. Note that this config
  // is only used when the user does not provide a pre-built index.
  GingerSearcherBuilder& SetVamanaConfig(VamanaConfig config);

  // Configures the PiPNN index construction parameters. Note that this config
  // is only used when the user does not provide a pre-built index.
  GingerSearcherBuilder& SetPipnnConfig(PipnnConfig config);

  // Configures the TurboQuant quantization parameters. If set, the dataset will
  // be quantized using these parameters. Note that this config is only used
  // when the user does not provide a pre-quantized dataset.
  GingerSearcherBuilder& SetTurboQuantConfig(TurboQuantConfig config);

  // Configures the JL random projection parameters.
  GingerSearcherBuilder& SetJlConfig(JlConfig config);

  // Configures the graph reordering algorithm to use for reordering the
  // dataset and graph.
  GingerSearcherBuilder& SetReorderingAlgorithm(
      ReorderingAlgorithm reordering_algorithm);

  // Configures whether to use graph compression to compress the graph index.
  GingerSearcherBuilder& SetUseGraphCompression(bool use_graph_compression);

  // Configures the PCA projection parameters. If set, the dataset will be
  // projected onto a lower-dimensional subspace. This is only a search
  // optimization. Index build happens without projection. This config is only
  // used when the user does not provide a pre-quantized dataset and the
  // TurboQuant config is provided.
  GingerSearcherBuilder& SetPcaProjectionConfig(PcaProjectionConfig config);

  // Configures whether dataset reordering should be performed in-place.
  // Defaults to false to allow for parallel execution. When set to true,
  // mutable datasets are reordered in-place to save memory.
  GingerSearcherBuilder& SetReorderInPlace(bool reorder_in_place);

  // Configures whether to drop the full-precision dataset after quantization
  // and index construction. When set to true, the dataset is released before
  // constructing the searcher.
  GingerSearcherBuilder& SetDropDataset(bool drop_dataset);

  // Sets the progress callback function used during the index construction.
  //
  // The callback function is repeatedly called during the build process with an
  // `IndexBuildProgress` object that contains information about its current
  // state to allow for progress reporting.
  GingerSearcherBuilder& SetProgressCallback(
      ProgressCallback progress_callback);

  // Sets a predicate that index construction polls between batches. Once it
  // returns true, `Build()` stops and returns a `CancelledError`. The predicate
  // may be called from worker threads concurrently.
  GingerSearcherBuilder& SetCancellationCheck(
      std::function<bool()> is_cancelled);

  // Builds and returns the constructed `GingerSearcher` instance.
  //
  // This method consumes internal state and can only be called once on a given
  // builder instance. Calling `Build()` a second time returns an error.
  //
  // During the build process:
  //   - If no pre-quantized dataset is provided but a `TurboQuantConfig` is
  //     present, the dataset is quantized.
  //   - If no pre-built index is provided, the index is constructed using the
  //     configured algorithm: Vamana (via `VamanaConfig`) or PiPNN
  //     (via `PipnnConfig`), depending on which is set in the config.
  //   - If a `BuildStats` object is provided, it is populated with the
  //     statistics of the build process.
  //
  // Preconditions:
  //   - At least one dataset (full-precision or quantized) must be provided.
  //     Both can be provided in which case they must have the same size and
  //     dimensionality.
  //   - If a quantized dataset is provided, a quantizer must also be provided,
  //     and the quantizer metric must match the specified metric.
  //   - If a pre-built index is provided, its size must match the size of the
  //     dataset.
  //
  // Returns:
  //   - A `GingerSearcher` instance if the build process is successful.
  //   - A `absl::InvalidArgumentError` if any precondition is violated.
  absl::StatusOr<std::unique_ptr<GingerSearcher>> Build(
      BuildStats* stats = nullptr);

 private:
  friend struct GingerSearcherBuilderAccess;

  absl::Status ValidateConfig();
  absl::Status ValidateDatasets();
  absl::Status ValidateQuantizer();
  absl::Status ValidateIndex();
  absl::Status ValidateReordering();
  // Returns the fingerprint of the float dataset in its original point order,
  // checked against `expected_dataset_fingerprint_`, or the expected
  // fingerprint if there is no such dataset.
  absl::StatusOr<std::optional<uint32_t>> ComputeDatasetFingerprint() const;

  // Builds the PCA projection. It requires a float dataset and a PCA config.
  absl::Status BuildProjection(BuildStats* stats);
  // Builds the quantized datasets used for navigation and reranking.
  absl::Status BuildQuantizedDatasetsForSearch(BuildStats* stats);
  // Builds the quantized dataset. It requires a dataset and a TurboQuant
  // config. If `apply_pca` is true and PCA is enabled, the dataset is
  // projected using PCA before quantization.
  absl::Status BuildQuantizedDataset(BuildStats* stats,
                                     const TurboQuantConfig& tq_config,
                                     bool apply_pca);
  // Builds the JL dataset. It requires a float dataset and a JL config.
  absl::Status BuildJlDataset(BuildStats* stats);

  absl::Status BuildVamanaIndex(BuildStats* stats);
  absl::Status BuildPipnnIndex(BuildStats* stats);

  absl::Status ComputeReordering(BuildStats* stats);
  absl::Status ApplyReordering(BuildStats* stats);
  absl::Status ReorderAddedDatasets(BuildStats* stats);
  absl::Status BuildCompressedGraph(BuildStats* stats);

  bool WillReorder() const;
  bool WillCompressGraph() const;
  std::vector<GingerNodeId> ComputeInverseReordering() const;

  void DropQuantizedDataset();

  void MarkAttachedDatasetsUnordered() {
    if (!inverse_reordering_.empty()) {
      datasets_to_reorder_.index_already_reordered = true;
    }
  }

  // Tracks which caller-attached datasets need to be permuted to match a loaded
  // pre-reordered index.
  struct DatasetsToReorder {
    bool index_already_reordered = false;
    bool float_dataset = false;
    bool quantized_dataset = false;
    bool jl_dataset = false;
  };

  GingerSearcherConfig config_;
  ProgressCallback progress_callback_;
  std::function<bool()> is_cancelled_ = [] { return false; };
  bool built_ = false;
  bool reorder_in_place_ = false;
  DatasetsToReorder datasets_to_reorder_;

  std::shared_ptr<const SpanDataset> dataset_;
  std::shared_ptr<const TurboQuantDatasetSoA> quantized_dataset_;
  std::shared_ptr<const TurboQuantCodec> quantizer_;
  std::shared_ptr<const JlDataset> jl_dataset_;

  std::shared_ptr<SpanDataset> mutable_dataset_;
  std::shared_ptr<TurboQuantDatasetSoA> mutable_quantized_dataset_;
  std::shared_ptr<JlDataset> mutable_jl_dataset_;

  // Full-dimensional 4+k TurboQuant dataset for extra-bits reranking. Only
  // built when both PCA and extra bits are enabled.
  std::shared_ptr<TurboQuantDatasetSoA> mutable_rerank_quantized_dataset_;
  std::shared_ptr<const TurboQuantDatasetSoA> rerank_quantized_dataset_;
  std::shared_ptr<const TurboQuantCodec> rerank_quantizer_;

  GingerSearcher::IndexVariant index_;
  std::unique_ptr<const PcaProjection> projection_;
  GingerNodeId entry_point_ = kVamanaEntryPoint;
  std::vector<GingerNodeId> reordering_;
  std::vector<GingerNodeId> inverse_reordering_;

  // Fingerprint of the original float dataset. Set by the loader, and checked
  // against a float dataset that the caller attaches.
  std::optional<uint32_t> expected_dataset_fingerprint_;
};

namespace internal {

// Returns the `ScalingStorageMode` that `GingerSearcherBuilder` uses for a
// TurboQuant dataset with `dimensionality` and `layout`.
ScalingStorageMode ResolveScalingStorageMode(const TurboQuantConfig& tq_config,
                                             size_t dimensionality,
                                             NormLayout layout);

}  // namespace internal

}  // namespace ginger

#endif  // GINGER_SEARCHER_BUILDER_H_
