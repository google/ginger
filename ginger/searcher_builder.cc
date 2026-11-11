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

#include "ginger/searcher_builder.h"

#include <algorithm>
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
#include "absl/cleanup/cleanup.h"
#include "absl/crc/crc32c.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/candidate_filtering/jl.h"
#include "ginger/dataset.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/compressed_graph_builder.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/graph/graph_stats.h"
#include "ginger/internal/parallel_for_with_status.h"
#include "ginger/pipnn/pipnn.h"
#include "ginger/projection/pca.h"
#include "ginger/reordering/recursive_graph_bisection.h"
#include "ginger/reordering/recursive_graph_contraction.h"
#include "ginger/reordering/recursive_graph_partitioning.h"
#include "ginger/reordering/reordering_utils.h"
#include "ginger/robust_prune_config.pb.h"
#include "ginger/search/search_functors.h"
#include "ginger/searcher.h"
#include "ginger/searcher.pb.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/turboquant/types/quantization_bits.h"
#include "ginger/types.h"
#include "ginger/vamana/index_builder.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace ginger {
namespace {

// Returns a CRC32C over the size, dimensionality and coordinate type of
// `dataset` and up to `kMaxSampledPoints` evenly spaced points, which is enough
// to catch a different or reordered dataset without reading all of it.
uint32_t DatasetFingerprint(const SpanDataset& dataset) {
  constexpr size_t kMaxSampledPoints = 1024;
  // kMaxSampledPoints is hashed too, so that changing it cannot silently
  // reject existing indexes.
  const uint64_t shape[4] = {
      static_cast<uint64_t>(dataset.size()), dataset.dimensionality(),
      static_cast<uint64_t>(dataset.coordinate_type()), kMaxSampledPoints};
  absl::crc32c_t crc = absl::ComputeCrc32c(
      absl::string_view(reinterpret_cast<const char*>(shape), sizeof(shape)));
  const size_t num_points = dataset.size();
  const size_t num_samples = std::min(num_points, kMaxSampledPoints);
  for (size_t k = 0; k < num_samples; ++k) {
    const absl::Span<const char> point =
        dataset.get_bytes(k * num_points / num_samples);
    crc =
        absl::ExtendCrc32c(crc, absl::string_view(point.data(), point.size()));
  }
  return static_cast<uint32_t>(crc);
}

absl::StatusOr<TurboQuantDistanceMetric> MapMetric(DistanceMetric metric) {
  switch (metric) {
    case DOT_PRODUCT:
      return TurboQuantDistanceMetric::kDotProductDistance;
    case SQUARED_L2:
      return TurboQuantDistanceMetric::kSquaredL2Distance;
    default:
      return absl::InvalidArgumentError("Unsupported distance metric.");
  }
}

bool AreAllDatapointsUnitNorm(const SpanDataset& dataset) {
  if (dataset.size() == 0 ||
      dataset.coordinate_type() != CoordinateType::kFloat) {
    return false;
  }
  constexpr float kMinSqNorm = 0.999f * 0.999f;
  constexpr float kMaxSqNorm = 1.001f * 1.001f;
  const int dim = dataset.dimensionality();

  return parlay::all_of(
      parlay::delayed_tabulate(
          dataset.size(),
          [&](const size_t i) {
            absl::Span<const float> pt = dataset.get<float>(i);
            const float sq_norm =
                Eigen::Map<const Eigen::VectorXf>(pt.data(), dim).squaredNorm();
            return sq_norm >= kMinSqNorm && sq_norm <= kMaxSqNorm;
          }),
      [](bool is_unit_norm) { return is_unit_norm; });
}

}  // namespace

namespace internal {

ScalingStorageMode ResolveScalingStorageMode(const TurboQuantConfig& tq_config,
                                             size_t dimensionality,
                                             NormLayout layout) {
  switch (tq_config.scaling_storage_mode()) {
    case TurboQuantConfig::SCALING_SEPARATE_ARRAY:
      return ScalingStorageMode::kSeparateArray;
    case TurboQuantConfig::SCALING_ROW_TAIL:
      return ScalingStorageMode::kRowTail;
    case TurboQuantConfig::SCALING_IN_THREE_BIT_PREFIX:
      return ScalingStorageMode::kIn3BitPrefix;
    case TurboQuantConfig::SCALING_STORAGE_MODE_UNSPECIFIED:
    default:
      return RecommendedScalingStorageMode(dimensionality, layout,
                                           tq_config.padding(),
                                           tq_config.allow_three_bit_prefix());
  }
}

}  // namespace internal

GingerSearcherBuilder::GingerSearcherBuilder(GingerSearcherConfig config)
    : config_(std::move(config)) {}

GingerSearcherBuilder& GingerSearcherBuilder::AddDataset(
    std::shared_ptr<const SpanDataset> dataset) {
  mutable_dataset_ = nullptr;
  dataset_ = std::move(dataset);
  datasets_to_reorder_.float_dataset =
      datasets_to_reorder_.index_already_reordered;
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::AddMutableDataset(
    std::shared_ptr<SpanDataset> dataset) {
  mutable_dataset_ = dataset;
  dataset_ = std::move(dataset);
  datasets_to_reorder_.float_dataset =
      datasets_to_reorder_.index_already_reordered;
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::AddQuantizedDataset(
    std::shared_ptr<const TurboQuantDatasetSoA> quantized_dataset,
    std::shared_ptr<const TurboQuantCodec> quantizer) {
  mutable_quantized_dataset_ = nullptr;
  quantized_dataset_ = std::move(quantized_dataset);
  quantizer_ = std::move(quantizer);
  datasets_to_reorder_.quantized_dataset =
      datasets_to_reorder_.index_already_reordered;
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::AddMutableQuantizedDataset(
    std::shared_ptr<TurboQuantDatasetSoA> quantized_dataset,
    std::shared_ptr<const TurboQuantCodec> quantizer) {
  mutable_quantized_dataset_ = quantized_dataset;
  quantized_dataset_ = std::move(quantized_dataset);
  quantizer_ = std::move(quantizer);
  datasets_to_reorder_.quantized_dataset =
      datasets_to_reorder_.index_already_reordered;
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::AddJlDataset(
    std::shared_ptr<const JlDataset> jl_dataset) {
  mutable_jl_dataset_ = nullptr;
  jl_dataset_ = std::move(jl_dataset);
  datasets_to_reorder_.jl_dataset =
      datasets_to_reorder_.index_already_reordered;
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::AddMutableJlDataset(
    std::shared_ptr<JlDataset> jl_dataset) {
  mutable_jl_dataset_ = jl_dataset;
  jl_dataset_ = std::move(jl_dataset);
  datasets_to_reorder_.jl_dataset =
      datasets_to_reorder_.index_already_reordered;
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::AddProjection(
    std::unique_ptr<const PcaProjection> projection) {
  projection_ = std::move(projection);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::AddIndex(
    GingerSearcher::IndexVariant index) {
  index_ = std::move(index);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::AddReordering(
    std::vector<GingerNodeId> reordering) {
  reordering_ = std::move(reordering);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetEntryPoint(
    GingerNodeId entry_point) {
  entry_point_ = entry_point;
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::AddInverseReordering(
    std::vector<GingerNodeId> inverse_reordering) {
  inverse_reordering_ = std::move(inverse_reordering);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetDistanceMetric(
    DistanceMetric metric) {
  config_.set_distance_metric(metric);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetVamanaConfig(
    VamanaConfig config) {
  *config_.mutable_vamana_config() = std::move(config);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetPipnnConfig(
    PipnnConfig config) {
  *config_.mutable_pipnn_config() = std::move(config);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetTurboQuantConfig(
    TurboQuantConfig config) {
  *config_.mutable_turbo_quant() = std::move(config);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetJlConfig(JlConfig config) {
  *config_.mutable_jl() = std::move(config);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetReorderingAlgorithm(
    ReorderingAlgorithm reordering_algorithm) {
  config_.set_reordering_algorithm(reordering_algorithm);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetUseGraphCompression(
    bool use_graph_compression) {
  config_.set_use_graph_compression(use_graph_compression);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetPcaProjectionConfig(
    PcaProjectionConfig config) {
  *config_.mutable_pca() = std::move(config);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetDropDataset(
    bool drop_dataset) {
  config_.set_drop_dataset(drop_dataset);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetReorderInPlace(
    bool reorder_in_place) {
  reorder_in_place_ = reorder_in_place;
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetProgressCallback(
    ProgressCallback progress_callback) {
  progress_callback_ = std::move(progress_callback);
  return *this;
}

GingerSearcherBuilder& GingerSearcherBuilder::SetCancellationCheck(
    std::function<bool()> is_cancelled) {
  is_cancelled_ = std::move(is_cancelled);
  return *this;
}

absl::StatusOr<std::unique_ptr<GingerSearcher>> GingerSearcherBuilder::Build(
    BuildStats* stats) {
  if (built_) {
    return absl::FailedPreconditionError(
        "GingerSearcherBuilder is single-use and cannot be reused.");
  }
  built_ = true;

  ABSL_RETURN_IF_ERROR(ValidateConfig());
  ABSL_RETURN_IF_ERROR(ValidateDatasets());
  ABSL_ASSIGN_OR_RETURN(const std::optional<uint32_t> dataset_fingerprint,
                        ComputeDatasetFingerprint());

  if (!config_.has_normalized_datapoints()) {
    if (quantized_dataset_ != nullptr &&
        quantized_dataset_->normalized_datapoints()) {
      config_.set_normalized_datapoints(true);
    } else if (dataset_ != nullptr && AreAllDatapointsUnitNorm(*dataset_)) {
      ABSL_LOG(WARNING) << "All dataset points have (near) unit norm. Setting "
                           "normalized_datapoints = true.";
      config_.set_normalized_datapoints(true);
    }
  }

  ABSL_RETURN_IF_ERROR(ValidateQuantizer());
  ABSL_RETURN_IF_ERROR(ValidateIndex());
  ABSL_RETURN_IF_ERROR(ValidateReordering());
  if (stats != nullptr) *stats = BuildStats{};
  ABSL_RETURN_IF_ERROR(ReorderAddedDatasets(stats));

  const GingerNodeId num_points = dataset_ != nullptr
                                      ? dataset_->size()
                                      : quantized_dataset_->num_datapoints();
  int dimensionality = 0;
  if (dataset_ != nullptr) {
    dimensionality = dataset_->dimensionality();
  } else if (projection_ != nullptr) {
    dimensionality = projection_->input_dim();
  } else {
    dimensionality = quantized_dataset_->dimensionality();
  }

  const bool build_index = std::holds_alternative<std::monostate>(index_);
  const bool build_quantized_dataset =
      quantized_dataset_ == nullptr && config_.has_turbo_quant();
  const bool build_jl_dataset = jl_dataset_ == nullptr && config_.has_jl();
  const bool build_pca_projection =
      projection_ == nullptr && config_.has_pca() && config_.has_turbo_quant();

  if (build_index) {
    // Quantized vectors are used during index building to speed up graph
    // construction. However, index building always operates in the
    // full-dimensional space to produce higher-quality graphs, so PCA is not
    // applied here. If PCA is enabled, the dataset is re-quantized for search
    // after index building.
    //
    // PiPNN index construction does not support 3-bit prefix datasets. If the
    // dataset would use the prefix, we build the index on a dataset without it
    // and re-quantize for search with the configured scaling storage mode.
    bool index_without_3bit_prefix = false;
    if (build_quantized_dataset) {
      TurboQuantConfig index_tq_config = config_.turbo_quant();
      if (config_.has_pipnn_config()) {
        ABSL_ASSIGN_OR_RETURN(const TurboQuantDistanceMetric metric,
                              MapMetric(config_.distance_metric()));
        if (internal::ResolveScalingStorageMode(
                index_tq_config, dimensionality,
                NormLayoutFor(metric, config_.normalized_datapoints())) ==
            ScalingStorageMode::kIn3BitPrefix) {
          index_tq_config.set_scaling_storage_mode(
              TurboQuantConfig::SCALING_STORAGE_MODE_UNSPECIFIED);
          index_tq_config.set_allow_three_bit_prefix(false);
          index_without_3bit_prefix = true;
        }
      }
      ABSL_RETURN_IF_ERROR(BuildQuantizedDataset(stats, index_tq_config,
                                                 /*apply_pca=*/false));
    }

    if (config_.has_vamana_config()) {
      ABSL_RETURN_IF_ERROR(BuildVamanaIndex(stats));
    } else {
      ABSL_RETURN_IF_ERROR(BuildPipnnIndex(stats));
    }

    // Drop the dataset without the prefix, so that the search datasets are
    // quantized from scratch below.
    if (index_without_3bit_prefix) DropQuantizedDataset();
  }

  if (WillReorder()) {
    if (reordering_.empty()) {
      ABSL_RETURN_IF_ERROR(ComputeReordering(stats));
    }
    if (inverse_reordering_.empty()) {
      inverse_reordering_ = ComputeInverseReordering();
    }
    ABSL_RETURN_IF_ERROR(ApplyReordering(stats));
    entry_point_ = reordering_[entry_point_];
  }

  if (WillCompressGraph()) {
    ABSL_RETURN_IF_ERROR(BuildCompressedGraph(stats));
  }

  if (build_pca_projection) {
    ABSL_RETURN_IF_ERROR(BuildProjection(stats));
  }
  if (build_quantized_dataset) {
    ABSL_RETURN_IF_ERROR(BuildQuantizedDatasetsForSearch(stats));
  }
  if (build_jl_dataset) {
    ABSL_RETURN_IF_ERROR(BuildJlDataset(stats));
  }

  if (config_.drop_dataset()) {
    dataset_ = nullptr;
    mutable_dataset_ = nullptr;
  }

  return absl::WrapUnique(new GingerSearcher(
      std::move(config_), num_points, dimensionality, std::move(dataset_),
      std::move(quantized_dataset_), std::move(quantizer_),
      std::move(jl_dataset_), entry_point_, std::move(index_),
      std::move(projection_), std::move(inverse_reordering_),
      std::move(rerank_quantized_dataset_), std::move(rerank_quantizer_),
      dataset_fingerprint));
}

absl::StatusOr<std::optional<uint32_t>>
GingerSearcherBuilder::ComputeDatasetFingerprint() const {
  // A float dataset is in its original point order unless it belongs to a
  // loaded, already reordered index.
  const bool in_original_order =
      dataset_ != nullptr &&
      (inverse_reordering_.empty() || datasets_to_reorder_.float_dataset);
  if (!in_original_order) return expected_dataset_fingerprint_;
  const uint32_t fingerprint = DatasetFingerprint(*dataset_);
  if (expected_dataset_fingerprint_.has_value() &&
      *expected_dataset_fingerprint_ != fingerprint) {
    return absl::FailedPreconditionError(
        "The float dataset does not match the dataset the index was built "
        "from. Attach the same points in their original order.");
  }
  return fingerprint;
}

absl::Status GingerSearcherBuilder::ValidateConfig() {
  if (config_.distance_metric() == UNDEFINED_DISTANCE) {
    return absl::InvalidArgumentError("Distance metric is not set.");
  }

  if (quantizer_ == nullptr && config_.has_turbo_quant()) {
    if (config_.turbo_quant().seed() < 0) {
      return absl::InvalidArgumentError(
          "TurboQuant seed must be non-negative.");
    }

    const int32_t extra_bits = config_.turbo_quant().extra_bits();
    ABSL_RETURN_IF_ERROR(ValidateTurboQuantExtraBits(extra_bits));

    const int32_t turboslice_dimensionality =
        config_.turbo_quant().turboslice_dimensionality();
    if (turboslice_dimensionality > 0) {
      if (dataset_ != nullptr) {
        if (turboslice_dimensionality > dataset_->dimensionality()) {
          return absl::InvalidArgumentError(
              "TurboSlice dimensionality cannot be greater than vector "
              "dimensionality.");
        }
      }
    }
    if (turboslice_dimensionality < 0) {
      return absl::InvalidArgumentError(
          "TurboQuant turboslice_dimensionality must be non-negative.");
    }
  }

  if (config_.has_jl()) {
    if (config_.jl().seed() < 0) {
      return absl::InvalidArgumentError("JL seed must be non-negative.");
    }
    if (config_.jl().jl_dimensionality() <= 0 ||
        (config_.jl().jl_dimensionality() % 64 != 0)) {
      return absl::InvalidArgumentError(
          "JL dimensionality must be positive and a multiple of 64.");
    }
  }

  if (std::holds_alternative<std::monostate>(index_)) {
    if (!config_.has_vamana_config() && !config_.has_pipnn_config()) {
      return absl::InvalidArgumentError(
          "Must provide either Vamana or PiPNN config to build an index.");
    }

    auto validate_alpha = [](float alpha) -> absl::Status {
      if (alpha < 1.0f) {
        return absl::InvalidArgumentError("robust_prune alpha must be >= 1.0.");
      }
      return absl::OkStatus();
    };

    if (config_.has_vamana_config()) {
      for (const auto& pass : config_.vamana_config().passes()) {
        ABSL_RETURN_IF_ERROR(validate_alpha(pass.prune_config().alpha()));
      }
    }
    if (config_.has_pipnn_config()) {
      ABSL_ASSIGN_OR_RETURN(const float alpha,
                            GetRobustPruneAlpha(config_.pipnn_config()));
      ABSL_RETURN_IF_ERROR(validate_alpha(alpha));
    }
  }

  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::ValidateDatasets() {
  if (dataset_ == nullptr && quantized_dataset_ == nullptr) {
    return absl::InvalidArgumentError("No dataset provided.");
  }
  if (config_.has_jl() && jl_dataset_ == nullptr && dataset_ == nullptr) {
    return absl::FailedPreconditionError(
        "Building JL dataset requires a float dataset.");
  }
  if (config_.has_pca() && projection_ == nullptr && dataset_ == nullptr) {
    return absl::FailedPreconditionError(
        "Building PCA projection requires a float dataset.");
  }

  if (dataset_ != nullptr && dataset_->size() <= 0) {
    return absl::InvalidArgumentError("Empty dataset.");
  }
  if (quantized_dataset_ != nullptr &&
      quantized_dataset_->num_datapoints() <= 0) {
    return absl::InvalidArgumentError("Empty dataset.");
  }

  if (dataset_ != nullptr &&
      dataset_->coordinate_type() != CoordinateType::kFloat) {
    if (config_.has_pca()) {
      return absl::InvalidArgumentError(
          "PCA projection is not supported for byte coordinates.");
    }
    if (config_.has_jl() && jl_dataset_ == nullptr) {
      return absl::InvalidArgumentError(
          "JL candidate filtering requires float coordinates.");
    }
    if (config_.normalized_datapoints()) {
      return absl::InvalidArgumentError(
          "normalized_datapoints is only supported with float coordinates.");
    }
  }

  if ((config_.has_pca() || projection_ != nullptr) &&
      quantized_dataset_ == nullptr && !config_.has_turbo_quant()) {
    return absl::InvalidArgumentError(
        "PCA projection is only supported with TurboQuant.");
  }

  if (dataset_ != nullptr && quantized_dataset_ != nullptr) {
    if (dataset_->size() != quantized_dataset_->num_datapoints()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Dataset size mismatch: dataset has %d points, "
          "quantized dataset has %d points",
          dataset_->size(), quantized_dataset_->num_datapoints()));
    }

    if (projection_ == nullptr &&
        dataset_->dimensionality() != quantized_dataset_->dimensionality()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Dimensionality mismatch: expected quantized dataset dimensionality "
          "to be %d, but got %d",
          dataset_->dimensionality(), quantized_dataset_->dimensionality()));
    }
  }

  if (projection_ != nullptr && dataset_ != nullptr) {
    if (projection_->input_dim() != dataset_->dimensionality()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Dimensionality mismatch: dataset has %d dimensions, "
          "projection input has %d dimensions",
          dataset_->dimensionality(), projection_->input_dim()));
    }
  }

  if (projection_ != nullptr && quantized_dataset_ != nullptr) {
    if (projection_->output_dim() != quantized_dataset_->dimensionality()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Dimensionality mismatch: expected quantized dataset "
          "dimensionality to be %d, but got %d",
          projection_->output_dim(), quantized_dataset_->dimensionality()));
    }
  }

  // TODO: Support providing pre-quantized search and rerank
  // datasets when PCA is enabled.
  if (config_.has_pca() && quantized_dataset_ != nullptr) {
    if (projection_ == nullptr) {
      return absl::InvalidArgumentError(
          "Providing a pre-quantized dataset with PCA enabled requires "
          "providing a pre-built PCA projection.");
    }
    if (config_.turbo_quant().extra_bits() > 0 &&
        rerank_quantized_dataset_ == nullptr) {
      return absl::InvalidArgumentError(
          "Extra-bits reranking with PCA is not supported when providing a "
          "pre-quantized dataset.");
    }
  }

  if (jl_dataset_ != nullptr) {
    if (jl_dataset_->size() <= 0) {
      return absl::InvalidArgumentError("Empty JL dataset.");
    }
    const GingerNodeId num_points = dataset_ != nullptr
                                        ? dataset_->size()
                                        : quantized_dataset_->num_datapoints();
    if (jl_dataset_->size() != num_points) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Dataset size mismatch: JL dataset has %d points, "
                          "expected %d points",
                          jl_dataset_->size(), num_points));
    }
    int source_dim = 0;
    if (dataset_ != nullptr) {
      source_dim = dataset_->dimensionality();
    } else if (projection_ != nullptr) {
      source_dim = projection_->input_dim();
    } else {
      source_dim = quantized_dataset_->dimensionality();
    }
    if (jl_dataset_->source_dim() != source_dim) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Dimensionality mismatch: JL source dimensionality "
                          "has %d dimensions, expected %d dimensions",
                          jl_dataset_->source_dim(), source_dim));
    }
    if (jl_dataset_->metric() != config_.distance_metric()) {
      return absl::InvalidArgumentError(
          "JL dataset metric does not match the metric specified in the "
          "config.");
    }
  }

  if (config_.drop_dataset()) {
    if (quantized_dataset_ == nullptr && !config_.has_turbo_quant()) {
      return absl::InvalidArgumentError(
          "Cannot drop full-precision dataset without a quantized dataset "
          "provided or configured.");
    }
  }

  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::ValidateQuantizer() {
  if (quantized_dataset_ != nullptr && quantizer_ == nullptr) {
    return absl::InvalidArgumentError(
        "Quantizer must be provided if a quantized dataset is provided.");
  }

  if (quantizer_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(const TurboQuantDistanceMetric metric,
                          MapMetric(config_.distance_metric()));

    if (quantizer_->metric() != metric) {
      return absl::InvalidArgumentError(
          "Quantizer metric does not match the metric specified in the "
          "config.");
    }

    if (quantized_dataset_ == nullptr) {
      return absl::InvalidArgumentError(
          "Quantized dataset must be provided if a quantizer is provided.");
    }

    if (quantizer_->dimensionality() != quantized_dataset_->dimensionality()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Dimensionality mismatch: quantizer has %d dimensions, quantized "
          "dataset has %d dimensions",
          quantizer_->dimensionality(), quantized_dataset_->dimensionality()));
    }

    if (config_.has_turbo_quant()) {
      if (config_.turbo_quant().turboslice_dimensionality() !=
          quantizer_->turboslice_dimensionality()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Quantizer TurboSlice dimensionality mismatch: quantizer has %d "
            "dimensions, config specifies %d dimensions",
            quantizer_->turboslice_dimensionality(),
            config_.turbo_quant().turboslice_dimensionality()));
      }
    }

    if (quantizer_->turboslice_dimensionality() !=
        quantized_dataset_->params().turboslice_dimensionality) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Quantizer TurboSlice dimensionality mismatch: quantizer has %d "
          "dimensions, quantized dataset has %d dimensions",
          quantizer_->turboslice_dimensionality(),
          quantized_dataset_->params().turboslice_dimensionality));
    }
  }

  // Reject an explicit 3-bit prefix up front rather than after building the
  // graph, e.g. when only the PCA re-quantization is ineligible.
  if (config_.has_turbo_quant() && quantized_dataset_ == nullptr &&
      config_.turbo_quant().scaling_storage_mode() ==
          TurboQuantConfig::SCALING_IN_THREE_BIT_PREFIX) {
    ABSL_ASSIGN_OR_RETURN(const TurboQuantDistanceMetric metric,
                          MapMetric(config_.distance_metric()));
    const int32_t turboslice_dim =
        config_.turbo_quant().turboslice_dimensionality();
    if (turboslice_dim > 0 && turboslice_dim < kScaleBitDims) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "SCALING_IN_THREE_BIT_PREFIX requires turboslice_dimensionality to "
          "be 0 or at least %d, got %d.",
          kScaleBitDims, turboslice_dim));
    }
    auto check_eligible = [&](int32_t dim, bool normalized) -> absl::Status {
      if (!CanUse3BitPrefix(NormLayoutFor(metric, normalized), dim)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "SCALING_IN_THREE_BIT_PREFIX requires dot product or normalized "
            "squared L2 without PCA, and at least %d dimensions (got %d).",
            kScaleBitDims, dim));
      }
      return absl::OkStatus();
    };
    // Mirrors Build(): without a pre-built index, we first quantize without
    // PCA; with PCA, we (re-)quantize the projected, unnormalized points.
    const bool pca_enabled = projection_ != nullptr || config_.has_pca();
    if (std::holds_alternative<std::monostate>(index_) || !pca_enabled) {
      ABSL_RETURN_IF_ERROR(check_eligible(dataset_->dimensionality(),
                                          config_.normalized_datapoints()));
    }
    if (pca_enabled) {
      // A truncated PCA dimension is only known after training.
      int32_t pca_dim = kScaleBitDims;
      if (projection_ != nullptr) {
        pca_dim = projection_->output_dim();
      } else if (config_.pca().has_dimension()) {
        pca_dim = config_.pca().dimension();
      }
      ABSL_RETURN_IF_ERROR(check_eligible(pca_dim, /*normalized=*/false));
    }
  }

  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::ValidateIndex() {
  if (!std::holds_alternative<std::monostate>(index_)) {
    const GingerNodeId index_num_nodes = std::visit(
        [](const auto& index_ptr) -> GingerNodeId {
          using T = std::decay_t<decltype(index_ptr)>;

          if constexpr (!std::is_same_v<T, std::monostate>) {
            return index_ptr->NumNodes();
          }

          ABSL_UNREACHABLE();
        },
        index_);

    const GingerNodeId num_points = dataset_ != nullptr
                                        ? dataset_->size()
                                        : quantized_dataset_->num_datapoints();

    if (index_num_nodes != num_points) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Index size mismatch: index has %d nodes, dataset has %d points",
          index_num_nodes, num_points));
    }
  }

  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::ValidateReordering() {
  const GingerNodeId num_points = dataset_ != nullptr
                                      ? dataset_->size()
                                      : quantized_dataset_->num_datapoints();
  if (entry_point_ < 0 || entry_point_ >= num_points) {
    return absl::InvalidArgumentError("Entry point is out of bounds.");
  }

  const bool has_compressed_index =
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          index_);
  if (has_compressed_index && WillReorder()) {
    return absl::InvalidArgumentError(
        "Cannot reorder an already compressed index.");
  }
  if (!inverse_reordering_.empty() && WillReorder()) {
    return absl::InvalidArgumentError(
        "Cannot specify inverse reordering when reordering is enabled.");
  }
  if (WillReorder() && reordering_.empty() &&
      config_.reordering_algorithm() == REORDERING_ALGORITHM_UNSPECIFIED) {
    return absl::InvalidArgumentError(
        "Reordering algorithm must be specified when reordering is enabled.");
  }

  auto validate_permutation = [&](absl::Span<const GingerNodeId> perm,
                                  absl::string_view name) -> absl::Status {
    if (perm.empty()) return absl::OkStatus();
    if (perm.size() != num_points) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s size mismatch: got %d points, dataset has %d points", name,
          perm.size(), num_points));
    }

    std::vector<bool> visited(num_points, false);
    for (GingerNodeId id : perm) {
      if (id < 0 || id >= num_points) {
        return absl::InvalidArgumentError(
            absl::StrFormat("%s contains out-of-bounds indices.", name));
      }
      if (visited[id]) {
        return absl::InvalidArgumentError(
            absl::StrFormat("%s contains duplicate indices.", name));
      }
      visited[id] = true;
    }
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(validate_permutation(reordering_, "Reordering"));
  ABSL_RETURN_IF_ERROR(
      validate_permutation(inverse_reordering_, "Inverse reordering"));

  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::BuildProjection(BuildStats* stats) {
  ABSL_QCHECK(dataset_ != nullptr);
  ABSL_QCHECK_EQ(dataset_->coordinate_type(), CoordinateType::kFloat);
  ABSL_QCHECK(config_.has_pca());
  absl::Time start_time = absl::Now();

  auto projection_or = PcaProjection::Create(config_.pca(), *dataset_);
  const auto time_elapsed = absl::Now() - start_time;
  if (stats != nullptr) stats->projection_time += time_elapsed;

  if (!projection_or.ok()) {
    if (absl::IsFailedPrecondition(projection_or.status())) {
      ABSL_LOG(WARNING) << "PCA projection skipped: "
                        << projection_or.status().message();
      config_.clear_pca();
      return absl::OkStatus();
    }
    return projection_or.status();
  }

  projection_ = std::move(projection_or.value());

  ABSL_LOG(INFO) << "Projection time: " << time_elapsed;
  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::BuildQuantizedDatasetsForSearch(
    BuildStats* stats) {
  // Reranking in the PCA-projected subspace suffers from projection error
  // (discarded dimensions) that additional quantization bits cannot recover.
  // Therefore, when both PCA and extra bits are enabled, we keep a separate
  // full-dimensional 4+k dataset for reranking, while the PCA-projected dataset
  // used for graph navigation only requires 4-bit codes.
  if (projection_ != nullptr && config_.turbo_quant().extra_bits() > 0) {
    if (quantized_dataset_ == nullptr) {
      ABSL_RETURN_IF_ERROR(BuildQuantizedDataset(stats, config_.turbo_quant(),
                                                 /*apply_pca=*/false));
    }
    rerank_quantized_dataset_ = std::move(quantized_dataset_);
    mutable_rerank_quantized_dataset_ = std::move(mutable_quantized_dataset_);
    rerank_quantizer_ = std::move(quantizer_);
  }

  // Build the quantized dataset for search if:
  // 1. The index was pre-built, so no quantized dataset has been constructed
  //    yet (quantized_dataset_ == nullptr).
  // 2. PCA is enabled (projection_ != nullptr), because index construction
  //    used unprojected vectors and search requires the PCA-projected dataset.
  // Otherwise, the dataset was already quantized and reordered during index
  // construction, so no re-quantization is needed.
  if (quantized_dataset_ == nullptr || projection_ != nullptr) {
    TurboQuantConfig tq_config = config_.turbo_quant();

    // When a dedicated full-dimensional rerank dataset is available, the
    // PCA-projected dataset is only used for graph navigation during beam
    // search. Setting extra_bits to 0 saves memory, as additional precision
    // bits are only needed during the reranking stage.
    if (rerank_quantized_dataset_ != nullptr) {
      tq_config.set_extra_bits(0);
    }

    return BuildQuantizedDataset(stats, tq_config, /*apply_pca=*/true);
  }

  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::BuildQuantizedDataset(
    BuildStats* stats, const TurboQuantConfig& tq_config, bool apply_pca) {
  ABSL_QCHECK(dataset_ != nullptr);
  absl::Time start_time = absl::Now();

  // Drop any existing quantized dataset to ensure we do not have two copies in
  // memory during quantization.
  DropQuantizedDataset();

  int32_t tq_dim = dataset_->dimensionality();
  // If PCA is enabled, the we quantize the projected vectors, so the quantizer
  // uses the PCA output dimension instead of the original dimension.
  const PcaProjection* pca = apply_pca ? projection_.get() : nullptr;
  if (pca != nullptr) {
    tq_dim = pca->output_dim();
  }

  const int32_t turboslice_dim = tq_config.turboslice_dimensionality();
  if (turboslice_dim < 0 || turboslice_dim > tq_dim) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "turbo_quant.turboslice_dimensionality is %d, but must be in [0, %d]. "
        "%d is the dimensionality of the quantized vectors: the PCA output "
        "dimension if PCA is enabled, otherwise the dataset dimensionality.",
        turboslice_dim, tq_dim, tq_dim));
  }
  ABSL_ASSIGN_OR_RETURN(const TurboQuantDistanceMetric distance_metric,
                        MapMetric(config_.distance_metric()));
  quantizer_ = std::make_shared<TurboQuantCodec>(
      tq_dim, tq_config.seed(), distance_metric,
      tq_config.turboslice_dimensionality());

  // Even if input points are normalized, PCA projection does not preserve
  // norms.
  const bool normalized_datapoints =
      config_.normalized_datapoints() && pca == nullptr;
  const ScalingStorageMode scaling_mode = internal::ResolveScalingStorageMode(
      tq_config, tq_dim, NormLayoutFor(distance_metric, normalized_datapoints));
  // TODO: consider turning on store_query_byte_sums.
  ABSL_ASSIGN_OR_RETURN(auto quantized_points,
                        TurboQuantDatasetSoA::Create(
                            *quantizer_, dataset_->size(),
                            TurboQuantDatasetOptions{
                                .pad_to_cache_line_size = tq_config.padding(),
                                .quantization_bits = 4 + tq_config.extra_bits(),
                                .store_query_byte_sums = false,
                                .normalized_datapoints = normalized_datapoints,
                                .scaling_storage_mode = scaling_mode}));

  // When drop_dataset is enabled, dataset_ was not reordered, so we gather via
  // inverse_reordering_. Checking !empty() skips this during index building,
  // which runs before ComputeReordering and reads sequentially.
  const bool has_reordering =
      config_.drop_dataset() &&
      (WillReorder() || datasets_to_reorder_.float_dataset) &&
      !inverse_reordering_.empty();
  ABSL_RETURN_IF_ERROR(parallel_for_with_status(
      0, dataset_->size(), [&](const size_t i) -> absl::Status {
        const size_t pos = has_reordering ? inverse_reordering_[i] : i;
        absl::Span<const float> point;
        std::vector<float> cast_point;
        switch (dataset_->coordinate_type()) {
          case CoordinateType::kFloat:
            point = dataset_->get<float>(pos);
            break;
          case CoordinateType::kInt8: {
            auto raw = dataset_->get<int8_t>(pos);
            cast_point.assign(raw.begin(), raw.end());
            point = cast_point;
            break;
          }
          case CoordinateType::kUint8: {
            auto raw = dataset_->get<uint8_t>(pos);
            cast_point.assign(raw.begin(), raw.end());
            point = cast_point;
            break;
          }
        }

        // If PCA is enabled, we project the float point into the
        // lower-dimensional PCA subspace before quantizing it.
        if (pca != nullptr) {
          ABSL_ASSIGN_OR_RETURN(auto projected_point, pca->Project(point));
          return quantized_points.UpdatePoint(*quantizer_, projected_point, i);
        } else {
          return quantized_points.UpdatePoint(*quantizer_, point, i);
        }
      }));

  mutable_quantized_dataset_ =
      std::make_shared<TurboQuantDatasetSoA>(std::move(quantized_points));
  quantized_dataset_ = mutable_quantized_dataset_;

  const absl::Duration time_elapsed = absl::Now() - start_time;
  if (stats != nullptr) stats->quantization_time += time_elapsed;

  ABSL_LOG(INFO) << "TurboQuant quantization took " << time_elapsed;
  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::BuildJlDataset(BuildStats* stats) {
  ABSL_QCHECK(dataset_ != nullptr);
  ABSL_QCHECK_EQ(dataset_->coordinate_type(), CoordinateType::kFloat);
  ABSL_QCHECK(config_.has_jl());
  absl::Time start_time = absl::Now();

  ABSL_ASSIGN_OR_RETURN(
      auto jl_dataset,
      JlDataset::Create(dataset_->dimensionality(),
                        config_.jl().jl_dimensionality(), config_.jl().seed(),
                        config_.distance_metric(), dataset_->size()));

  // When drop_dataset is enabled, dataset_ was not reordered, so we
  // gather points directly using inverse_reordering_.
  const bool has_reordering =
      config_.drop_dataset() &&
      (WillReorder() || datasets_to_reorder_.float_dataset) &&
      !inverse_reordering_.empty();
  parlay::parallel_for(0, dataset_->size(), [&](const size_t i) {
    const size_t pos = has_reordering ? inverse_reordering_[i] : i;
    jl_dataset->UpdatePoint(i, dataset_->get<float>(pos));
  });

  mutable_jl_dataset_ = std::move(jl_dataset);
  jl_dataset_ = mutable_jl_dataset_;

  const auto time_elapsed = absl::Now() - start_time;
  if (stats != nullptr) stats->jl_quantization_time += time_elapsed;
  ABSL_LOG(INFO) << "JL quantization took " << time_elapsed;
  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::BuildVamanaIndex(BuildStats* stats) {
  ABSL_QCHECK(dataset_ != nullptr || quantized_dataset_ != nullptr);
  ABSL_QCHECK(config_.has_vamana_config());
  absl::Time start_time = absl::Now();

  DatasetBuildScorer distance_fn;
  if (quantized_dataset_) {
    // TODO: Propagate absl::Status through BuildVamanaIndex and
    // RobustPrune so that errors from ComputeTurboQuantBuildDistances are not
    // silently swallowed during index construction.
    distance_fn = TurboQuantBuildScorer{
        quantized_dataset_.get(),
        quantizer_.get(),
    };
  } else {
    ABSL_ASSIGN_OR_RETURN(
        distance_fn,
        GetDatasetBuildScorer(dataset_->coordinate_type(),
                              config_.distance_metric(), dataset_.get()));
  }

  const bool normalized_dot_product =
      config_.distance_metric() == DOT_PRODUCT &&
      config_.normalized_datapoints();
  if (normalized_dot_product) {
    distance_fn = [base_fn = std::move(distance_fn)](
                      GingerNodeId query_id,
                      absl::Span<DistanceAndId> candidates) {
      base_fn(query_id, candidates);
      for (DistanceAndId& cand : candidates) {
        cand.distance = std::max(0.0f, 1.0f + cand.distance);
      }
    };
  }

  const bool nonpositive_distances =
      config_.distance_metric() == DOT_PRODUCT && !normalized_dot_product;
  VamanaConfig v_config = config_.vamana_config();
  for (int i = 0; i < v_config.passes_size(); ++i) {
    v_config.mutable_passes(i)
        ->mutable_prune_config()
        ->set_nonpositive_distances(nonpositive_distances);
  }
  const GingerNodeId num_points = dataset_ != nullptr
                                      ? dataset_->size()
                                      : quantized_dataset_->num_datapoints();
  ABSL_ASSIGN_OR_RETURN(
      FlatGraph vamana_index,
      ginger::BuildVamanaIndex(num_points, v_config, std::move(distance_fn),
                               progress_callback_, is_cancelled_));
  index_ = std::make_unique<const FlatGraph>(std::move(vamana_index));

  const absl::Duration time_elapsed = absl::Now() - start_time;
  if (stats != nullptr) stats->construction_time += time_elapsed;

  ABSL_LOG(INFO) << "Vamana index construction took " << time_elapsed;
  PrintGraphStats(*std::get<std::unique_ptr<const FlatGraph>>(index_), "Vamana",
                  /*start_node=*/kVamanaEntryPoint);
  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::BuildPipnnIndex(BuildStats* stats) {
  ABSL_QCHECK(config_.has_pipnn_config());

  // PiPNN index construction requires a TurboQuant dataset. If the caller did
  // not configure TurboQuant, build a temporary TQ dataset only used for
  // constructing the PiPNN index and release it immediately afterwards.
  const bool needs_quantization = quantized_dataset_ == nullptr;
  if (needs_quantization) {
    TurboQuantConfig tq_config;
    tq_config.set_padding(true);
    tq_config.set_seed(config_.pipnn_config().seed());
    ABSL_RETURN_IF_ERROR(
        BuildQuantizedDataset(stats, tq_config, /*apply_pca=*/false));
  }
  absl::Cleanup cleanup_tq = [&] {
    if (needs_quantization) DropQuantizedDataset();
  };

  // TODO: Support 3-bit prefix in PiPNN index construction.
  if (quantized_dataset_->has_3bit_prefix()) {
    return absl::InvalidArgumentError(
        "PiPNN index construction does not yet support 3-bit prefix datasets.");
  }

  absl::Time start_time = absl::Now();

  ABSL_ASSIGN_OR_RETURN(
      FlatGraph pipnn_graph,
      ginger::BuildPipnnIndex(*quantized_dataset_, *quantizer_,
                              config_.pipnn_config(), is_cancelled_));
  index_ = std::make_unique<const FlatGraph>(std::move(pipnn_graph));

  const absl::Duration time_elapsed = absl::Now() - start_time;
  if (stats != nullptr) stats->construction_time += time_elapsed;

  ABSL_LOG(INFO) << "PiPNN index construction took " << time_elapsed;
  PrintGraphStats(*std::get<std::unique_ptr<const FlatGraph>>(index_), "PiPNN",
                  /*start_node=*/0);
  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::ComputeReordering(BuildStats* stats) {
  ABSL_QCHECK(!std::holds_alternative<std::monostate>(index_));
  absl::Time start_time = absl::Now();

  auto compute = [&](const auto& graph_ptr) -> absl::Status {
    using T = std::decay_t<decltype(graph_ptr)>;

    if constexpr (std::is_same_v<T, std::unique_ptr<const CSRGraph>> ||
                  std::is_same_v<T, std::unique_ptr<const FlatGraph>>) {
      const auto& graph = *graph_ptr;
      switch (config_.reordering_algorithm()) {
        case RECURSIVE_GRAPH_BISECTION:
          reordering_ = ComputeRecursiveGraphBisectionReordering(graph);
          break;
        case RECURSIVE_GRAPH_CONTRACTION:
          reordering_ = ComputeRecursiveGraphContractionReordering(graph);
          break;
        case RECURSIVE_GRAPH_PARTITIONING:
          reordering_ = ComputeRecursiveGraphPartitioningReordering(graph);
          break;
        default:
          return absl::InvalidArgumentError(
              absl::StrFormat("Unsupported reordering algorithm: %d",
                              config_.reordering_algorithm()));
      }
      return absl::OkStatus();
    }

    return absl::InvalidArgumentError(
        "Reordering can only be computed on CSRGraph or FlatGraph.");
  };
  ABSL_RETURN_IF_ERROR(std::visit(compute, index_));

  const absl::Duration time_elapsed = absl::Now() - start_time;
  if (stats != nullptr) stats->compute_reordering_time += time_elapsed;

  ABSL_LOG(INFO) << "Computing reordering took " << time_elapsed;
  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::ApplyReordering(BuildStats* stats) {
  ABSL_QCHECK(!reordering_.empty());
  ABSL_QCHECK(!inverse_reordering_.empty());
  ABSL_QCHECK(!std::holds_alternative<std::monostate>(index_));
  absl::Time start_time = absl::Now();

  // If drop_dataset is true, we skip reordering the float dataset to avoid an
  // unnecessary copy or in-place scramble. This is safe because the subsequent
  // BuildProjection step computes PCA covariance, which is order-independent.
  // BuildQuantizedDataset and BuildJlDataset will then gather points in the
  // reordered graph order directly via inverse_reordering_. Any existing
  // quantized and JL datasets are kept for serving, so they must be physically
  // shuffled to align with the graph.
  if (dataset_ != nullptr && !config_.drop_dataset()) {
    if (reorder_in_place_ && mutable_dataset_ != nullptr) {
      ReorderDatasetInPlace(*mutable_dataset_, reordering_);
    } else {
      dataset_ = ReorderDataset(*dataset_, reordering_);
      mutable_dataset_ = nullptr;
    }
  }

  if (quantized_dataset_ != nullptr) {
    if (reorder_in_place_ && mutable_quantized_dataset_ != nullptr) {
      ReorderQuantizedDatasetInPlace(*mutable_quantized_dataset_, reordering_);
    } else {
      ABSL_ASSIGN_OR_RETURN(
          quantized_dataset_,
          ReorderQuantizedDataset(*quantized_dataset_, reordering_));
      mutable_quantized_dataset_ = nullptr;
    }
  }

  if (rerank_quantized_dataset_ != nullptr) {
    if (reorder_in_place_ && mutable_rerank_quantized_dataset_ != nullptr) {
      ReorderQuantizedDatasetInPlace(*mutable_rerank_quantized_dataset_,
                                     reordering_);
    } else {
      ABSL_ASSIGN_OR_RETURN(
          rerank_quantized_dataset_,
          ReorderQuantizedDataset(*rerank_quantized_dataset_, reordering_));
      mutable_rerank_quantized_dataset_ = nullptr;
    }
  }

  // ReorderDataset and ReorderQuantizedDataset scatter points to their new
  // positions using reordering_ (old ID to new ID). JlDataset::Reorder gathers
  // points from their old positions using inverse_reordering_ (new ID to old
  // ID).
  if (jl_dataset_ != nullptr) {
    if (reorder_in_place_ && mutable_jl_dataset_ != nullptr) {
      mutable_jl_dataset_->ReorderInPlace(inverse_reordering_);
    } else {
      jl_dataset_ = jl_dataset_->Reorder(inverse_reordering_);
      mutable_jl_dataset_ = nullptr;
    }
  }

  index_ = std::visit(
      [&](const auto& graph_ptr) -> GingerSearcher::IndexVariant {
        using T = std::decay_t<decltype(graph_ptr)>;
        if constexpr (std::is_same_v<T, std::unique_ptr<const FlatGraph>>) {
          return std::make_unique<const FlatGraph>(
              ReorderFlatGraph(*graph_ptr, reordering_));
        } else if constexpr (std::is_same_v<T,
                                            std::unique_ptr<const CSRGraph>>) {
          return std::make_unique<const CSRGraph>(
              ReorderGraph(*graph_ptr, reordering_));
        }
        ABSL_UNREACHABLE();
      },
      index_);

  const absl::Duration time_elapsed = absl::Now() - start_time;
  if (stats != nullptr) stats->apply_reordering_time += time_elapsed;

  ABSL_LOG(INFO) << "Applying reordering took " << time_elapsed;
  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::ReorderAddedDatasets(BuildStats* stats) {
  // This runs when a caller attaches unreordered datasets to a pre-reordered
  // index loaded via LoadGingerSearcherBuilder.
  //
  // If drop_dataset is true, we skip reordering the float dataset. This avoids
  // allocating a large copy (or scrambling a mutable dataset in-place) just to
  // drop it after quantization. Skipping it is safe because the loaded searcher
  // already contains the index and PCA projection. Those build steps are
  // bypassed. Only BuildQuantizedDataset and BuildJlDataset will read the
  // floats. They gather points in the reordered graph order directly via
  // inverse_reordering_.
  //
  // drop_dataset only discards the float dataset. Quantized and JL datasets are
  // kept for serving, so they must be physically shuffled to align with the
  // graph.
  const bool need_float = datasets_to_reorder_.float_dataset &&
                          dataset_ != nullptr && !config_.drop_dataset();
  const bool need_quantized =
      datasets_to_reorder_.quantized_dataset && quantized_dataset_ != nullptr;
  const bool need_jl =
      datasets_to_reorder_.jl_dataset && jl_dataset_ != nullptr;
  if (inverse_reordering_.empty() ||
      (!need_float && !need_quantized && !need_jl)) {
    return absl::OkStatus();
  }
  const absl::Time start_time = absl::Now();
  if (need_float || need_quantized) {
    std::vector<GingerNodeId> reordering(inverse_reordering_.size());
    parlay::parallel_for(0, inverse_reordering_.size(), [&](GingerNodeId i) {
      reordering[inverse_reordering_[i]] = i;
    });
    if (need_float) {
      if (reorder_in_place_ && mutable_dataset_ != nullptr) {
        ReorderDatasetInPlace(*mutable_dataset_, reordering);
      } else {
        dataset_ = ReorderDataset(*dataset_, reordering);
        mutable_dataset_ = nullptr;
      }
      datasets_to_reorder_.float_dataset = false;
    }
    if (need_quantized) {
      if (reorder_in_place_ && mutable_quantized_dataset_ != nullptr) {
        ReorderQuantizedDatasetInPlace(*mutable_quantized_dataset_, reordering);
      } else {
        ABSL_ASSIGN_OR_RETURN(
            quantized_dataset_,
            ReorderQuantizedDataset(*quantized_dataset_, reordering));
        mutable_quantized_dataset_ = nullptr;
      }
      datasets_to_reorder_.quantized_dataset = false;
    }
  }
  // SpanDataset and TurboQuantDatasetSoA scatter via reordering. JlDataset
  // gathers via inverse_reordering_.
  if (need_jl) {
    if (reorder_in_place_ && mutable_jl_dataset_ != nullptr) {
      mutable_jl_dataset_->ReorderInPlace(inverse_reordering_);
    } else {
      jl_dataset_ = jl_dataset_->Reorder(inverse_reordering_);
      mutable_jl_dataset_ = nullptr;
    }
    datasets_to_reorder_.jl_dataset = false;
  }
  if (stats != nullptr) {
    stats->apply_reordering_time += absl::Now() - start_time;
  }
  return absl::OkStatus();
}

absl::Status GingerSearcherBuilder::BuildCompressedGraph(BuildStats* stats) {
  absl::Time start_time = absl::Now();

  int64_t uncompressed_size = 0;
  CompressedCSRGraph<> compressed_graph = std::visit(
      [&](const auto& graph_ptr) -> CompressedCSRGraph<> {
        using T = std::decay_t<decltype(graph_ptr)>;

        if constexpr (std::is_same_v<T, std::unique_ptr<const CSRGraph>> ||
                      std::is_same_v<T, std::unique_ptr<const FlatGraph>>) {
          uncompressed_size = graph_ptr->NumBytes();
          return CompressGraph(*graph_ptr);
        }

        ABSL_UNREACHABLE();
      },
      index_);

  const double compression_ratio =
      static_cast<double>(uncompressed_size) / compressed_graph.NumBytes();
  const double bytes_per_edge =
      static_cast<double>(compressed_graph.CompressedTargets().size()) /
      compressed_graph.NumArcs();

  index_ =
      std::make_unique<const CompressedCSRGraph<>>(std::move(compressed_graph));

  const absl::Duration time_elapsed = absl::Now() - start_time;
  if (stats != nullptr) stats->graph_compression_time += time_elapsed;

  ABSL_LOG(INFO) << "Graph compression took " << time_elapsed
                 << ". Compression ratio: " << compression_ratio
                 << ", bytes per edge: " << bytes_per_edge;
  return absl::OkStatus();
}

bool GingerSearcherBuilder::WillReorder() const {
  const bool has_reordering_algorithm =
      config_.has_reordering_algorithm() &&
      config_.reordering_algorithm() != REORDERING_ALGORITHM_NONE;
  return !reordering_.empty() || has_reordering_algorithm;
}

bool GingerSearcherBuilder::WillCompressGraph() const {
  const bool has_compressed_index =
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          index_);
  return config_.use_graph_compression() && !has_compressed_index;
}

std::vector<GingerNodeId> GingerSearcherBuilder::ComputeInverseReordering()
    const {
  ABSL_QCHECK(!reordering_.empty());

  std::vector<GingerNodeId> inverse_reordering(reordering_.size());
  parlay::parallel_for(0, reordering_.size(), [&](const GingerNodeId i) {
    inverse_reordering[reordering_[i]] = i;
  });

  return inverse_reordering;
}

void GingerSearcherBuilder::DropQuantizedDataset() {
  quantizer_ = nullptr;
  quantized_dataset_ = nullptr;
  mutable_quantized_dataset_ = nullptr;
}

}  // namespace ginger
