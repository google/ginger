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

#include "ginger/io/index_io.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/candidate_filtering/jl.h"
#include "ginger/dataset.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/compressed_offsets.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/internal/array_io.h"
#include "ginger/internal/parallel_for_with_status.h"
#include "ginger/internal/searcher_access.h"
#include "ginger/io/io.pb.h"
#include "ginger/io/serialization.h"
#include "ginger/projection/config.pb.h"
#include "ginger/projection/pca.h"
#include "ginger/projection/pca_float.h"
#include "ginger/projection/pca_turboquant.h"
#include "ginger/searcher.h"
#include "ginger/searcher.pb.h"
#include "ginger/searcher_builder.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/datapoint/encode_datapoint.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

namespace ginger::io::internal {

namespace {

// These entry names are part of the on-disk format and must not be changed, as
// doing so would break backwards compatibility.
constexpr absl::string_view kFlatGraphHeader = "ginger_flat_graph.header.pbtxt";
constexpr absl::string_view kFlatGraphData = "ginger_flat_graph";

constexpr absl::string_view kCompressedGraphHeader =
    "ginger_compressed_graph.header.pbtxt";
constexpr absl::string_view kCompressedGraphOffsets =
    "ginger_compressed_graph.offsets";
constexpr absl::string_view kCompressedGraphTargets =
    "ginger_compressed_graph.targets";

constexpr absl::string_view kFloatDatasetHeader = "float_dataset.header.pbtxt";
constexpr absl::string_view kFloatDatasetData = "float_dataset";

constexpr absl::string_view kQuantizedDatasetHeader =
    "turboquant_dataset.header.pbtxt";
constexpr absl::string_view kQuantizedDatasetData = "turboquant_dataset";

// Full-dimensional extra-bits dataset used for reranking when PCA is enabled.
constexpr absl::string_view kRerankDatasetHeader =
    "rerank_turboquant_dataset.header.pbtxt";
constexpr absl::string_view kRerankDatasetData = "rerank_turboquant_dataset";

// TurboSlice scales of the quantized dataset stored in entry `data_name`.
std::string TurboSliceScalesName(absl::string_view data_name) {
  return absl::StrCat(data_name, ".turboslice_scales");
}

constexpr absl::string_view kJlDatasetHeader = "jl_dataset.header.pbtxt";
constexpr absl::string_view kJlDatasetData = "jl_dataset";

constexpr absl::string_view kPcaHeader = "pca_matrix.header.pbtxt";
constexpr absl::string_view kPcaMatrix = "pca_matrix";
constexpr absl::string_view kPcaPackedWeights = "pca_matrix.packed_weights";
constexpr absl::string_view kPcaRowInvScales = "pca_matrix.row_inv_scales";

constexpr absl::string_view kSearcherHeader = "ginger_searcher.header.pbtxt";
constexpr absl::string_view kSearcherInverseReordering =
    "ginger_searcher.inverse_reordering";

}  // namespace

absl::Status WriteFlatGraph(array_io::Writer& writer, const FlatGraph& graph) {
  FlatGraphHeader header;
  header.set_num_nodes(graph.NumNodes());
  header.set_max_degree(graph.MaxDegree());
  ABSL_RETURN_IF_ERROR(
      writer.WriteArray<GingerNodeId>(kFlatGraphData, graph.GetData()));
  return writer.WriteTextProto(kFlatGraphHeader, header);
}

absl::StatusOr<FlatGraph> ReadFlatGraph(const array_io::Reader& reader) {
  FlatGraphHeader header;
  ABSL_RETURN_IF_ERROR(reader.ReadTextProto(kFlatGraphHeader, header));
  if (header.num_nodes() < 0 || header.max_degree() < 0) {
    return absl::InvalidArgumentError(
        "num_nodes and max_degree must be non-negative");
  }
  ABSL_RETURN_IF_ERROR(reader.CheckArraySize<GingerNodeId>(
      kFlatGraphData, {static_cast<uint64_t>(header.num_nodes()),
                       static_cast<uint64_t>(header.max_degree()) + 1}));
  FlatGraph graph(header.num_nodes(), header.max_degree());
  ABSL_RETURN_IF_ERROR(
      reader.ReadArray<GingerNodeId>(kFlatGraphData, graph.MutableData()));
  ABSL_RETURN_IF_ERROR(parallel_for_with_status(
      0, graph.NumNodes(), [&](size_t u) -> absl::Status {
        const GingerNodeId degree = graph.Degree(u);
        if (degree < 0 || degree > graph.MaxDegree()) {
          return absl::DataLossError(
              absl::StrCat("Node ", u, " has degree ", degree,
                           ", but max_degree is ", graph.MaxDegree(), "."));
        }
        for (const GingerNodeId v : graph.Neighbors(u)) {
          if (v < 0 || v >= graph.NumNodes()) {
            return absl::DataLossError(absl::StrCat(
                "Node ", u, " has neighbor ", v, ", but the graph has ",
                graph.NumNodes(), " nodes."));
          }
        }
        return absl::OkStatus();
      }));
  return graph;
}

template <typename Config>
absl::Status WriteCompressedGraph(array_io::Writer& writer,
                                  const CompressedCSRGraph<Config>& graph) {
  CompressedGraphHeader header;
  header.set_num_nodes(graph.NumNodes());
  header.set_num_arcs(graph.NumArcs());
  header.set_num_compressed_bytes(graph.CompressedTargets().size());

  const CompressedOffsets& offsets = graph.Offsets();
  ABSL_RETURN_IF_ERROR(par_do_with_status(
      [&] {
        return writer.WriteRecords(
            kCompressedGraphOffsets, offsets.size(), sizeof(int64_t),
            [&](size_t begin, size_t end, absl::Span<char> out) {
              for (size_t i = begin; i < end; ++i) {
                const int64_t v = offsets[i];
                std::memcpy(out.data() + (i - begin) * sizeof(int64_t), &v,
                            sizeof(int64_t));
              }
              return absl::OkStatus();
            });
      },
      [&] {
        return writer.WriteArray<uint8_t>(kCompressedGraphTargets,
                                          graph.CompressedTargets());
      }));
  return writer.WriteTextProto(kCompressedGraphHeader, header);
}

template <typename Config>
absl::StatusOr<CompressedCSRGraph<Config>> ReadCompressedGraph(
    const array_io::Reader& reader) {
  CompressedGraphHeader header;
  ABSL_RETURN_IF_ERROR(reader.ReadTextProto(kCompressedGraphHeader, header));
  if (header.num_nodes() < 0 || header.num_arcs() < 0 ||
      header.num_compressed_bytes() < 0) {
    return absl::InvalidArgumentError(
        "num_nodes, num_arcs, and num_compressed_bytes must be non-negative");
  }
  const int64_t num_nodes = header.num_nodes();
  const int64_t num_arcs = header.num_arcs();
  const int64_t num_compressed_bytes = header.num_compressed_bytes();

  ABSL_RETURN_IF_ERROR(reader.CheckArraySize<int64_t>(
      kCompressedGraphOffsets, {static_cast<uint64_t>(num_nodes) + 1}));
  ABSL_RETURN_IF_ERROR(reader.CheckArraySize<uint8_t>(
      kCompressedGraphTargets, {static_cast<uint64_t>(num_compressed_bytes)}));
  std::vector<int64_t> offsets(num_nodes + 1);
  std::vector<uint8_t> compressed_targets(num_compressed_bytes);

  ABSL_RETURN_IF_ERROR(par_do_with_status(
      [&] {
        // Offsets are streamed out with `WriteRecords` to avoid materializing a
        // full vector during save, but read directly with `ReadArray` to avoid
        // a block-buffer memcpy.
        // The `array_io.h` contract allows this for `FileWriter`.
        return reader.ReadArray<int64_t>(kCompressedGraphOffsets,
                                         absl::MakeSpan(offsets));
      },
      [&] {
        return reader.ReadArray<uint8_t>(kCompressedGraphTargets,
                                         absl::MakeSpan(compressed_targets));
      }));

  // Neighbor lists are not decoded here, so their contents are trusted.
  if (offsets.front() != 0 || offsets.back() > num_compressed_bytes) {
    return absl::DataLossError(
        "Compressed graph offsets must start at 0 and end within the "
        "compressed targets.");
  }
  for (size_t u = 0; u + 1 < offsets.size(); ++u) {
    if (offsets[u] > offsets[u + 1]) {
      return absl::DataLossError(
          absl::StrCat("Compressed graph offsets decrease at node ", u, "."));
    }
  }
  return CompressedCSRGraph<Config>(std::move(offsets),
                                    std::move(compressed_targets), num_arcs);
}

template absl::Status WriteCompressedGraph<CompressedGraphConfig>(
    array_io::Writer&, const CompressedCSRGraph<CompressedGraphConfig>&);
template absl::Status WriteCompressedGraph<AdaptiveCompressedGraphConfig>(
    array_io::Writer&,
    const CompressedCSRGraph<AdaptiveCompressedGraphConfig>&);
template absl::StatusOr<CompressedCSRGraph<CompressedGraphConfig>>
ReadCompressedGraph<CompressedGraphConfig>(const array_io::Reader&);
template absl::StatusOr<CompressedCSRGraph<AdaptiveCompressedGraphConfig>>
ReadCompressedGraph<AdaptiveCompressedGraphConfig>(const array_io::Reader&);

namespace {

absl::Status WriteFloatDataset(array_io::Writer& writer,
                               const SpanDataset& dataset) {
  if (dataset.coordinate_type() != CoordinateType::kFloat) {
    return absl::UnimplementedError(
        "Only float datasets are supported for serialization.");
  }
  DatasetHeader header;
  header.set_num_points(dataset.size());
  header.set_dimensionality(dataset.dimensionality());

  const size_t total =
      static_cast<size_t>(dataset.size()) * dataset.dimensionality();
  const absl::Span<const float> all_data =
      total == 0 ? absl::Span<const float>()
                 : absl::Span<const float>(
                       static_cast<const float*>(dataset[0]), total);
  ABSL_RETURN_IF_ERROR(writer.WriteArray<float>(kFloatDatasetData, all_data));
  return writer.WriteTextProto(kFloatDatasetHeader, header);
}

absl::StatusOr<std::unique_ptr<SpanDataset>> ReadFloatDataset(
    const array_io::Reader& reader) {
  DatasetHeader header;
  ABSL_RETURN_IF_ERROR(reader.ReadTextProto(kFloatDatasetHeader, header));

  if (header.num_points() < 0 || header.dimensionality() <= 0) {
    return absl::InvalidArgumentError(
        "num_points must be non-negative and dimensionality must be positive");
  }

  ABSL_RETURN_IF_ERROR(reader.CheckArraySize<float>(
      kFloatDatasetData, {static_cast<uint64_t>(header.num_points()),
                          static_cast<uint64_t>(header.dimensionality())}));
  auto dataset = std::make_unique<SpanDataset>(header.num_points(),
                                               header.dimensionality());
  const size_t total =
      static_cast<size_t>(dataset->size()) * dataset->dimensionality();
  const absl::Span<float> all_data =
      total == 0 ? absl::Span<float>()
                 : absl::Span<float>(static_cast<float*>((*dataset)[0]), total);
  ABSL_RETURN_IF_ERROR(reader.ReadArray<float>(kFloatDatasetData, all_data));
  return dataset;
}

absl::Status WriteQuantizedDataset(array_io::Writer& writer,
                                   const TurboQuantDatasetSoA& dataset,
                                   int random_seed,
                                   absl::string_view header_name,
                                   absl::string_view data_name) {
  if (dataset.num_datapoints() == 0) {
    return absl::InvalidArgumentError("Cannot save empty dataset.");
  }
  QuantizedDatasetHeader header;
  header.set_num_datapoints(dataset.num_datapoints());
  header.set_dimensionality(dataset.dimensionality());
  header.set_random_seed(random_seed);
  header.set_quantization_bits(dataset.quantization_bits());
  header.set_normalized_datapoints(dataset.normalized_datapoints());
  header.set_turboslice_dimensionality(
      dataset.params().turboslice_dimensionality);
  switch (dataset.scaling_storage_mode()) {
    case ScalingStorageMode::kSeparateArray:
      header.set_scaling_storage_mode(TurboQuantConfig::SCALING_SEPARATE_ARRAY);
      break;
    case ScalingStorageMode::kRowTail:
      header.set_scaling_storage_mode(TurboQuantConfig::SCALING_ROW_TAIL);
      break;
    case ScalingStorageMode::kIn3BitPrefix:
      header.set_scaling_storage_mode(
          TurboQuantConfig::SCALING_IN_THREE_BIT_PREFIX);
      break;
  }
  const size_t point_size = SerializedPointSize(dataset.DataPointView(0));
  header.set_serialized_point_size(point_size);
  switch (dataset.metric()) {
    case TurboQuantDistanceMetric::kDotProductDistance:
      header.set_metric(QuantizedDatasetHeader::DEFAULT_DOT_PRODUCT);
      break;
    case TurboQuantDistanceMetric::kSquaredL2Distance:
      header.set_metric(QuantizedDatasetHeader::SQUARED_L2);
      break;
  }

  ABSL_RETURN_IF_ERROR(writer.WriteRecords(
      data_name, dataset.num_datapoints(), point_size,
      [&](size_t begin, size_t end, absl::Span<char> out) -> absl::Status {
        for (size_t i = begin; i < end; ++i) {
          SerializePoint(dataset.DataPointView(i),
                         out.subspan((i - begin) * point_size, point_size));
        }
        return absl::OkStatus();
      }));
  if (dataset.IsTurboSliceEnabled()) {
    std::vector<float> scales(dataset.num_datapoints());
    for (size_t i = 0; i < scales.size(); ++i) {
      scales[i] = dataset.norm_scaling_factor_turboslice(i);
    }
    ABSL_RETURN_IF_ERROR(writer.WriteArray<float>(
        TurboSliceScalesName(data_name), absl::MakeConstSpan(scales)));
  }
  return writer.WriteTextProto(header_name, header);
}

struct LoadedQuantizedDataset {
  TurboQuantDatasetSoA dataset;
  std::shared_ptr<const TurboQuantCodec> quantizer;
};

// `config` is the saved searcher config. Its `turbo_quant.padding` setting
// chooses the row padding, which the file format does not depend on.
absl::StatusOr<LoadedQuantizedDataset> ReadQuantizedDatasetAndCodec(
    const array_io::Reader& reader, const GingerSearcherConfig& config,
    absl::string_view header_name, absl::string_view data_name) {
  QuantizedDatasetHeader header;
  ABSL_RETURN_IF_ERROR(reader.ReadTextProto(header_name, header));

  if (header.num_datapoints() <= 0) {
    return absl::InvalidArgumentError("Number of datapoints is <= 0");
  }
  if (header.dimensionality() <= 0) {
    return absl::InvalidArgumentError("Dimensionality is <= 0");
  }
  if (header.serialized_point_size() <= 0) {
    return absl::InvalidArgumentError("Serialized point size is <= 0");
  }
  const size_t point_size = header.serialized_point_size();
  ABSL_RETURN_IF_ERROR(reader.CheckEntrySize(
      data_name, point_size, {static_cast<uint64_t>(header.num_datapoints())}));

  TurboQuantDistanceMetric metric;
  switch (header.metric()) {
    case QuantizedDatasetHeader::SQUARED_L2:
      metric = TurboQuantDistanceMetric::kSquaredL2Distance;
      break;
    case QuantizedDatasetHeader::DEFAULT_DOT_PRODUCT:
      metric = TurboQuantDistanceMetric::kDotProductDistance;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown distance metric: ", header.metric()));
  }

  if (header.turboslice_dimensionality() < 0 ||
      header.turboslice_dimensionality() > header.dimensionality()) {
    return absl::DataLossError(absl::StrCat(
        "turboslice_dimensionality is ", header.turboslice_dimensionality(),
        ", but must be in [0, ", header.dimensionality(), "]."));
  }
  auto quantizer = std::make_shared<const TurboQuantCodec>(
      header.dimensionality(), header.random_seed(), metric,
      header.turboslice_dimensionality());

  std::vector<float> turboslice_scales;
  if (header.turboslice_dimensionality() > 0) {
    turboslice_scales.resize(header.num_datapoints());
    ABSL_RETURN_IF_ERROR(reader.ReadArray<float>(
        TurboSliceScalesName(data_name), absl::MakeSpan(turboslice_scales)));
  }

  int bits_per_dim = 0;
  int quantization_bits_for_create = 4;
  if (header.quantization_bits() > 0) {
    quantization_bits_for_create = header.quantization_bits();
    bits_per_dim = header.quantization_bits() - 4;
  }

  const TurboQuantConfig& tq_config = config.turbo_quant();
  ScalingStorageMode scaling_mode;
  switch (header.scaling_storage_mode()) {
    case TurboQuantConfig::SCALING_SEPARATE_ARRAY:
      scaling_mode = ScalingStorageMode::kSeparateArray;
      break;
    case TurboQuantConfig::SCALING_ROW_TAIL:
      scaling_mode = ScalingStorageMode::kRowTail;
      break;
    case TurboQuantConfig::SCALING_IN_THREE_BIT_PREFIX:
      scaling_mode = ScalingStorageMode::kIn3BitPrefix;
      break;
    case TurboQuantConfig::SCALING_STORAGE_MODE_UNSPECIFIED:
      // Older files do not record the mode and never use the 3-bit prefix.
      scaling_mode = RecommendedScalingStorageMode(
          header.dimensionality(),
          NormLayoutFor(metric, header.normalized_datapoints()),
          tq_config.padding(), /*allow_3bit_prefix=*/false);
      break;
    default:
      return absl::DataLossError(absl::StrCat("Unknown scaling storage mode: ",
                                              header.scaling_storage_mode()));
  }
  const bool has_3bit_prefix =
      scaling_mode == ScalingStorageMode::kIn3BitPrefix;

  // `ParseSerializedTurboQuantPoint` trusts the point size, so check it before
  // parsing.
  const size_t expected_point_size =
      3 * sizeof(float) + (header.dimensionality() + 1) / 2 +
      GetExtraPrecisionBytes(header.dimensionality(), bits_per_dim,
                             has_3bit_prefix);
  if (point_size != expected_point_size) {
    return absl::DataLossError(
        absl::StrCat("Serialized TurboQuant points have ", point_size,
                     " bytes, but dimensionality ", header.dimensionality(),
                     " with ", 4 + bits_per_dim, " bits per dimension needs ",
                     expected_point_size, " bytes."));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto dataset,
      TurboQuantDatasetSoA::Create(
          *quantizer, header.num_datapoints(),
          TurboQuantDatasetOptions{
              .pad_to_cache_line_size = tq_config.padding(),
              .quantization_bits = quantization_bits_for_create,
              .store_query_byte_sums = false,
              .normalized_datapoints = header.normalized_datapoints(),
              .scaling_storage_mode = scaling_mode}));

  ABSL_RETURN_IF_ERROR(reader.ReadRecords(
      data_name, dataset.num_datapoints(), point_size,
      [&](size_t begin, size_t end, absl::string_view bytes) -> absl::Status {
        for (size_t i = begin; i < end; ++i) {
          TurboQuantBasePointView view = ParseSerializedTurboQuantPoint(
              bytes.substr((i - begin) * point_size, point_size), bits_per_dim,
              has_3bit_prefix);
          if (!turboslice_scales.empty()) {
            view.norm_scaling_factor_turboslice = turboslice_scales[i];
          }
          ABSL_RETURN_IF_ERROR(dataset.UpdatePoint(view, i));
        }
        return absl::OkStatus();
      }));
  return LoadedQuantizedDataset{std::move(dataset), std::move(quantizer)};
}

absl::Status WriteJlDataset(array_io::Writer& writer,
                            const JlDataset& dataset) {
  JlDatasetHeader header;
  header.set_num_points(dataset.size());
  header.set_source_dim(dataset.source_dim());
  header.set_jl_dim(dataset.jl_dim());
  header.set_seed(dataset.seed());
  header.set_metric(dataset.metric());

  ABSL_RETURN_IF_ERROR(
      writer.WriteArray<uint64_t>(kJlDatasetData, dataset.data()));
  return writer.WriteTextProto(kJlDatasetHeader, header);
}

absl::StatusOr<std::unique_ptr<JlDataset>> ReadJlDataset(
    const array_io::Reader& reader) {
  JlDatasetHeader header;
  ABSL_RETURN_IF_ERROR(reader.ReadTextProto(kJlDatasetHeader, header));

  if (header.num_points() < 0 || header.source_dim() <= 0) {
    return absl::InvalidArgumentError(
        "num_points must be non-negative and source_dim must be positive");
  }
  if (header.jl_dim() <= 0 || header.jl_dim() % 64 != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "jl_dim must be positive and a multiple of 64, got ", header.jl_dim()));
  }

  const size_t words_per_point = static_cast<size_t>(header.jl_dim()) / 64;
  ABSL_RETURN_IF_ERROR(reader.CheckArraySize<uint64_t>(
      kJlDatasetData,
      {static_cast<uint64_t>(header.num_points()), words_per_point}));
  hwy::AlignedVector<uint64_t> data(header.num_points() * words_per_point);

  ABSL_RETURN_IF_ERROR(
      reader.ReadArray<uint64_t>(kJlDatasetData, absl::MakeSpan(data)));

  return JlDataset::Create(header.source_dim(), header.jl_dim(), header.seed(),
                           header.metric(), header.num_points(),
                           std::move(data));
}

absl::Status WritePcaProjection(array_io::Writer& writer,
                                const PcaProjection& projection) {
  PcaProjectionHeader header;
  header.set_input_dim(projection.input_dim());
  header.set_output_dim(projection.output_dim());
  header.set_matrix_representation(projection.matrix_representation());

  switch (projection.matrix_representation()) {
    case MATRIX_REPRESENTATION_FLOAT: {
      const auto& mat = static_cast<const FloatPcaProjection&>(projection)
                            .projection_matrix();
      ABSL_RETURN_IF_ERROR(writer.WriteArray<float>(
          kPcaMatrix, absl::MakeConstSpan(mat.data(), mat.size())));
      break;
    }
    case MATRIX_REPRESENTATION_TURBOQUANT: {
      const auto& tq = static_cast<const TurboQuantPcaProjection&>(projection);
      ABSL_RETURN_IF_ERROR(par_do_with_status(
          [&] {
            return writer.WriteArray<uint8_t>(kPcaPackedWeights,
                                              tq.packed_weights());
          },
          [&] {
            return writer.WriteArray<float>(kPcaRowInvScales,
                                            tq.row_inv_scales());
          }));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Unsupported PCA projection matrix representation for "
          "serialization.");
  }
  return writer.WriteTextProto(kPcaHeader, header);
}

absl::StatusOr<std::unique_ptr<PcaProjection>> ReadPcaProjection(
    const array_io::Reader& reader) {
  PcaProjectionHeader header;
  ABSL_RETURN_IF_ERROR(reader.ReadTextProto(kPcaHeader, header));

  if (header.input_dim() <= 0 || header.output_dim() <= 0) {
    return absl::InvalidArgumentError(
        "Projection dimensions must be strictly positive.");
  }

  switch (header.matrix_representation()) {
    case MATRIX_REPRESENTATION_FLOAT: {
      ABSL_RETURN_IF_ERROR(reader.CheckArraySize<float>(
          kPcaMatrix, {static_cast<uint64_t>(header.output_dim()),
                       static_cast<uint64_t>(header.input_dim())}));
      Eigen::MatrixXf mat(header.output_dim(), header.input_dim());
      ABSL_RETURN_IF_ERROR(reader.ReadArray<float>(
          kPcaMatrix, absl::MakeSpan(mat.data(), mat.size())));
      return FloatPcaProjection::Create(std::move(mat));
    }
    case MATRIX_REPRESENTATION_TURBOQUANT: {
      const size_t padded_in = hwy::RoundUpTo(header.input_dim(), 128);
      const size_t padded_out = hwy::RoundUpTo(header.output_dim(), 8);
      ABSL_RETURN_IF_ERROR(reader.CheckArraySize<uint8_t>(
          kPcaPackedWeights, {padded_out, padded_in / 2}));
      hwy::AlignedVector<uint8_t> weights(padded_out * (padded_in / 2));
      hwy::AlignedVector<float> scales(padded_out);

      ABSL_RETURN_IF_ERROR(par_do_with_status(
          [&] {
            return reader.ReadArray<uint8_t>(kPcaPackedWeights,
                                             absl::MakeSpan(weights));
          },
          [&] {
            return reader.ReadArray<float>(kPcaRowInvScales,
                                           absl::MakeSpan(scales));
          }));

      return TurboQuantPcaProjection::Create(
          header.input_dim(), header.output_dim(), std::move(weights),
          std::move(scales));
    }
    default:
      return absl::InvalidArgumentError(
          "Unsupported PCA projection matrix representation for "
          "deserialization.");
  }
}

}  // namespace

absl::Status WriteGingerSearcher(array_io::Writer& writer,
                                 const GingerSearcher& searcher,
                                 bool save_datasets) {
  bool has_flat_graph =
      std::holds_alternative<std::unique_ptr<const FlatGraph>>(
          searcher.index()) &&
      std::get<std::unique_ptr<const FlatGraph>>(searcher.index()) != nullptr;
  bool has_compressed_graph =
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          searcher.index()) &&
      std::get<std::unique_ptr<const CompressedCSRGraph<>>>(searcher.index()) !=
          nullptr;

  if (!has_flat_graph && !has_compressed_graph) {
    return absl::UnimplementedError(
        "Currently only FlatGraph and CompressedCSRGraph indices are "
        "supported for serialization.");
  }

  const auto& dataset = GingerSearcherAccess::dataset(searcher);
  const auto& quantized_dataset =
      GingerSearcherAccess::quantized_dataset(searcher);
  const auto& quantizer = GingerSearcherAccess::quantizer(searcher);
  const auto& jl_dataset = GingerSearcherAccess::jl_dataset(searcher);
  const PcaProjection* projection = GingerSearcherAccess::projection(searcher);
  const auto& rerank_dataset =
      GingerSearcherAccess::rerank_quantized_dataset(searcher);
  const auto& rerank_quantizer =
      GingerSearcherAccess::rerank_quantizer(searcher);

  GingerSearcherHeader header;
  *header.mutable_config() = searcher.config();
  header.set_dimensionality(searcher.dimensionality());
  header.set_has_float_dataset(save_datasets && dataset != nullptr);
  header.set_has_quantized_dataset(save_datasets &&
                                   quantized_dataset != nullptr);
  header.set_has_jl_dataset(save_datasets && jl_dataset != nullptr);
  header.set_has_rerank_quantized_dataset(save_datasets &&
                                          rerank_dataset != nullptr);
  header.set_has_pca_projection(projection != nullptr);
  header.set_has_graph_index(true);
  header.set_entry_point(searcher.entry_point());
  header.set_has_inverse_reordering(!searcher.remapping().empty());
  if (const std::optional<uint32_t> fingerprint =
          GingerSearcherAccess::dataset_fingerprint(searcher)) {
    header.set_dataset_fingerprint(*fingerprint);
  }
  if (has_flat_graph) {
    header.set_index_type(GingerSearcherHeader::INDEX_TYPE_FLAT_GRAPH);
  } else {
    header.set_index_type(GingerSearcherHeader::INDEX_TYPE_COMPRESSED_GRAPH);
  }

  ABSL_RETURN_IF_ERROR(par_do_with_status(
      [&]() -> absl::Status {
        if (searcher.remapping().empty()) return absl::OkStatus();
        return writer.WriteArray<GingerNodeId>(kSearcherInverseReordering,
                                               searcher.remapping());
      },
      [&]() -> absl::Status {
        if (!save_datasets || !dataset) return absl::OkStatus();
        return WriteFloatDataset(writer, *dataset);
      },
      [&]() -> absl::Status {
        if (!save_datasets || !quantized_dataset) {
          return absl::OkStatus();
        }
        return WriteQuantizedDataset(writer, *quantized_dataset,
                                     quantizer->seed(), kQuantizedDatasetHeader,
                                     kQuantizedDatasetData);
      },
      [&]() -> absl::Status {
        if (!save_datasets || !rerank_dataset) return absl::OkStatus();
        return WriteQuantizedDataset(writer, *rerank_dataset,
                                     rerank_quantizer->seed(),
                                     kRerankDatasetHeader, kRerankDatasetData);
      },
      [&]() -> absl::Status {
        if (!save_datasets || !jl_dataset) return absl::OkStatus();
        return WriteJlDataset(writer, *jl_dataset);
      },
      [&]() -> absl::Status {
        if (!projection) return absl::OkStatus();
        return WritePcaProjection(writer, *projection);
      },
      [&]() -> absl::Status {
        if (has_flat_graph) {
          return WriteFlatGraph(
              writer,
              *std::get<std::unique_ptr<const FlatGraph>>(searcher.index()));
        }
        return WriteCompressedGraph(
            writer, *std::get<std::unique_ptr<const CompressedCSRGraph<>>>(
                        searcher.index()));
      }));
  return writer.WriteTextProto(kSearcherHeader, header);
}

namespace {

absl::StatusOr<GingerSearcherBuilder> ReadSearcherBuilder(
    const array_io::Reader& reader, bool require_datasets) {
  GingerSearcherHeader header;
  ABSL_RETURN_IF_ERROR(reader.ReadTextProto(kSearcherHeader, header));

  if (require_datasets && header.config().has_jl() &&
      !header.has_jl_dataset()) {
    return absl::FailedPreconditionError(
        "Serialized searcher configured with JL filtering is missing JL "
        "dataset.");
  }
  if (header.config().has_pca() && !header.has_pca_projection()) {
    return absl::FailedPreconditionError(
        "Serialized searcher configured with PCA projection is missing PCA "
        "projection.");
  }

  std::shared_ptr<SpanDataset> float_dataset;
  std::shared_ptr<TurboQuantDatasetSoA> quantized_dataset;
  std::shared_ptr<const TurboQuantCodec> quantizer;
  std::shared_ptr<TurboQuantDatasetSoA> rerank_dataset;
  std::shared_ptr<const TurboQuantCodec> rerank_quantizer;
  GingerSearcher::IndexVariant index;
  std::vector<GingerNodeId> inverse_reordering;
  std::shared_ptr<JlDataset> jl_dataset;
  std::unique_ptr<const PcaProjection> pca_projection;

  auto load_index = [&]() -> absl::Status {
    if (!header.has_graph_index()) {
      return absl::UnimplementedError(
          "Currently only FlatGraph and CompressedCSRGraph indices are "
          "supported for deserialization.");
    }
    GingerNodeId num_nodes = 0;
    if (header.index_type() ==
        GingerSearcherHeader::INDEX_TYPE_COMPRESSED_GRAPH) {
      ABSL_ASSIGN_OR_RETURN(auto graph,
                            ReadCompressedGraph<CompressedGraphConfig>(reader));
      num_nodes = graph.NumNodes();
      index = std::make_unique<const CompressedCSRGraph<>>(std::move(graph));
    } else if (header.index_type() ==
                   GingerSearcherHeader::INDEX_TYPE_FLAT_GRAPH ||
               header.index_type() ==
                   GingerSearcherHeader::INDEX_TYPE_UNSPECIFIED) {
      // Default to FlatGraph for backwards compatibility with older headers
      ABSL_ASSIGN_OR_RETURN(auto graph, ReadFlatGraph(reader));
      num_nodes = graph.NumNodes();
      index = std::make_unique<const FlatGraph>(std::move(graph));
    } else {
      return absl::UnimplementedError(
          "Currently only FlatGraph and CompressedCSRGraph indices are "
          "supported for deserialization.");
    }
    if (header.has_inverse_reordering()) {
      inverse_reordering.resize(num_nodes);
      ABSL_RETURN_IF_ERROR(reader.ReadArray<GingerNodeId>(
          kSearcherInverseReordering, absl::MakeSpan(inverse_reordering)));
    }
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(par_do_with_status(
      [&]() -> absl::Status {
        if (!header.has_float_dataset()) return absl::OkStatus();
        ABSL_ASSIGN_OR_RETURN(float_dataset, ReadFloatDataset(reader));
        return absl::OkStatus();
      },
      [&]() -> absl::Status {
        if (!header.has_quantized_dataset()) return absl::OkStatus();
        ABSL_ASSIGN_OR_RETURN(
            LoadedQuantizedDataset loaded,
            ReadQuantizedDatasetAndCodec(reader, header.config(),
                                         kQuantizedDatasetHeader,
                                         kQuantizedDatasetData));
        quantized_dataset =
            std::make_shared<TurboQuantDatasetSoA>(std::move(loaded.dataset));
        quantizer = std::move(loaded.quantizer);
        return absl::OkStatus();
      },
      [&]() -> absl::Status {
        if (!header.has_rerank_quantized_dataset()) return absl::OkStatus();
        ABSL_ASSIGN_OR_RETURN(LoadedQuantizedDataset loaded,
                              ReadQuantizedDatasetAndCodec(
                                  reader, header.config(), kRerankDatasetHeader,
                                  kRerankDatasetData));
        rerank_dataset =
            std::make_shared<TurboQuantDatasetSoA>(std::move(loaded.dataset));
        rerank_quantizer = std::move(loaded.quantizer);
        return absl::OkStatus();
      },
      load_index,
      [&]() -> absl::Status {
        if (!header.has_jl_dataset()) return absl::OkStatus();
        ABSL_ASSIGN_OR_RETURN(jl_dataset, ReadJlDataset(reader));
        return absl::OkStatus();
      },
      [&]() -> absl::Status {
        if (!header.has_pca_projection()) return absl::OkStatus();
        ABSL_ASSIGN_OR_RETURN(pca_projection, ReadPcaProjection(reader));
        return absl::OkStatus();
      }));

  if (require_datasets && !header.has_float_dataset() &&
      !header.has_quantized_dataset()) {
    return absl::FailedPreconditionError(
        "Searcher was saved without datasets. Use LoadGingerSearcherBuilder "
        "and attach a dataset.");
  }

  GingerSearcherConfig config = header.config();
  // The saved searcher has already been reordered. Clear the build-time
  // reordering algorithm so the builder does not attempt to re-run reordering
  // on the loaded index.
  config.clear_reordering_algorithm();

  GingerSearcherBuilder builder(std::move(config));
  builder.SetEntryPoint(header.entry_point());
  if (!inverse_reordering.empty()) {
    builder.AddInverseReordering(std::move(inverse_reordering));
  }
  if (float_dataset != nullptr) {
    builder.AddMutableDataset(std::move(float_dataset));
  }
  if (quantized_dataset != nullptr) {
    builder.AddQuantizedDataset(std::move(quantized_dataset),
                                std::move(quantizer));
  }
  if (jl_dataset != nullptr) {
    builder.AddJlDataset(std::move(jl_dataset));
  }
  if (rerank_dataset != nullptr) {
    GingerSearcherBuilderAccess::AddRerankQuantizedDataset(
        builder, std::move(rerank_dataset), std::move(rerank_quantizer));
  }
  if (pca_projection != nullptr) {
    builder.AddProjection(std::move(pca_projection));
  }
  builder.AddIndex(std::move(index));
  if (header.has_dataset_fingerprint()) {
    GingerSearcherBuilderAccess::SetDatasetFingerprint(
        builder, header.dataset_fingerprint());
  }
  return builder;
}

}  // namespace

absl::StatusOr<GingerSearcherBuilder> ReadGingerSearcherBuilder(
    const array_io::Reader& reader) {
  ABSL_ASSIGN_OR_RETURN(
      GingerSearcherBuilder builder,
      ReadSearcherBuilder(reader, /*require_datasets=*/false));
  // ReadSearcherBuilder attaches the datasets saved in the file. Those are
  // already in the reordered graph order. We enable this flag afterwards so
  // that ReorderAddedDatasets only shuffles new, unreordered datasets attached
  // by the caller.
  GingerSearcherBuilderAccess::MarkAttachedDatasetsUnordered(builder);
  return builder;
}

absl::StatusOr<std::unique_ptr<GingerSearcher>> ReadGingerSearcher(
    const array_io::Reader& reader) {
  ABSL_ASSIGN_OR_RETURN(GingerSearcherBuilder builder,
                        ReadSearcherBuilder(reader, /*require_datasets=*/true));
  return builder.Build();
}

}  // namespace ginger::io::internal
