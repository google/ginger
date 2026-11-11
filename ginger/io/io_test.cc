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

#include "ginger/io/io.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/candidate_filtering/jl.h"
#include "ginger/dataset.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/compressed_graph_builder.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/internal/array_io.h"
#include "ginger/internal/searcher_access.h"
#include "ginger/io/index_io.h"
#include "ginger/io/io.pb.h"
#include "ginger/projection/config.pb.h"
#include "ginger/projection/pca.h"
#include "ginger/projection/pca_float.h"
#include "ginger/projection/pca_turboquant.h"
#include "ginger/searcher.h"
#include "ginger/searcher.pb.h"
#include "ginger/searcher_builder.h"
#include "ginger/testing/parse_text_proto.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_reference_stub.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace ginger::io {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAreArray;

class FakeIndex final : public array_io::Writer, public array_io::Reader {
 public:
  absl::Status WriteBytes(absl::string_view name,
                          absl::string_view bytes) override {
    absl::MutexLock lock(mu_);
    entries_[name] = std::string(bytes);
    return absl::OkStatus();
  }

  absl::Status WriteRecords(absl::string_view name, size_t num_records,
                            size_t record_size,
                            array_io::SerializeFn serialize) override {
    std::string data(num_records * record_size, '\0');
    if (num_records > 0) {
      ABSL_RETURN_IF_ERROR(serialize(0, num_records, absl::MakeSpan(data)));
    }
    absl::MutexLock lock(mu_);
    entries_[name] = std::move(data);
    return absl::OkStatus();
  }

  absl::Status WriteTextProto(absl::string_view name,
                              const google::protobuf::Message& proto) override {
    std::string text;
    ABSL_CHECK(google::protobuf::TextFormat::PrintToString(proto, &text));
    absl::MutexLock lock(mu_);
    entries_[name] = std::move(text);
    return absl::OkStatus();
  }

  absl::Status Close() override { return absl::OkStatus(); }

  absl::Status ReadBytes(absl::string_view name,
                         absl::Span<char> dest) const override {
    absl::MutexLock lock(mu_);
    auto it = entries_.find(name);
    if (it == entries_.end()) return absl::NotFoundError(name);
    if (it->second.size() != dest.size()) return absl::DataLossError(name);
    if (!dest.empty()) std::memcpy(dest.data(), it->second.data(), dest.size());
    return absl::OkStatus();
  }

  absl::Status ReadRecords(absl::string_view name, size_t num_records,
                           size_t record_size,
                           array_io::ParseFn parse) const override {
    std::string data;
    {
      absl::MutexLock lock(mu_);
      auto it = entries_.find(name);
      if (it == entries_.end()) return absl::NotFoundError(name);
      if (it->second.size() != num_records * record_size) {
        return absl::DataLossError(name);
      }
      data = it->second;
    }
    return num_records == 0 ? absl::OkStatus() : parse(0, num_records, data);
  }

  absl::Status ReadTextProto(absl::string_view name,
                             google::protobuf::Message& proto) const override {
    absl::MutexLock lock(mu_);
    auto it = entries_.find(name);
    if (it == entries_.end()) return absl::NotFoundError(name);
    if (!google::protobuf::TextFormat::ParseFromString(it->second, &proto)) {
      return absl::DataLossError(name);
    }
    return absl::OkStatus();
  }

  absl::Status CheckEntrySize(
      absl::string_view name, size_t record_size,
      std::initializer_list<uint64_t> dims) const override {
    absl::MutexLock lock(mu_);
    auto it = entries_.find(name);
    if (it == entries_.end()) return absl::NotFoundError(name);
    if (!array_io::SizeMatches(it->second.size(), record_size, dims)) {
      return absl::DataLossError(name);
    }
    return absl::OkStatus();
  }

  void Erase(absl::string_view name) {
    absl::MutexLock lock(mu_);
    entries_.erase(name);
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::string> entries_ ABSL_GUARDED_BY(mu_);
};

void CheckThatGraphsAreEqual(const FlatGraph& a, const FlatGraph& b) {
  ASSERT_EQ(a.NumNodes(), b.NumNodes());
  EXPECT_EQ(a.MaxDegree(), b.MaxDegree());
  for (GingerNodeId i = 0; i < a.NumNodes(); ++i) {
    EXPECT_THAT(a.Neighbors(i), ElementsAreArray(b.Neighbors(i)));
  }
}

TEST(FlatGraphIOTest, SaveAndLoadEmptyGraph) {
  FlatGraph graph(0, 0);
  std::string prefix = absl::StrCat(::testing::TempDir(), "/empty_graph");
  ASSERT_OK(SaveFlatGraph(graph, prefix));
  ASSERT_OK_AND_ASSIGN(FlatGraph loaded_graph, LoadFlatGraph(prefix));
  CheckThatGraphsAreEqual(graph, loaded_graph);
}

TEST(FlatGraphIOTest, SaveAndLoadGraphWithNodesButNoEdges) {
  FlatGraph graph(3, 5);
  std::string prefix = absl::StrCat(::testing::TempDir(), "/no_edges_graph");
  ASSERT_OK(SaveFlatGraph(graph, prefix));
  ASSERT_OK_AND_ASSIGN(FlatGraph loaded_graph, LoadFlatGraph(prefix));
  CheckThatGraphsAreEqual(graph, loaded_graph);
}

TEST(FlatGraphIOTest, SaveAndLoadGraphWithEdges) {
  FlatGraph graph(4, 3);
  graph.SetNeighbors(0, {1, 2});
  graph.SetNeighbors(1, {0});
  graph.SetNeighbors(2, {0, 3});
  graph.SetNeighbors(3, {2});
  std::string prefix = absl::StrCat(::testing::TempDir(), "/with_edges_graph");
  ASSERT_OK(SaveFlatGraph(graph, prefix));
  ASSERT_OK_AND_ASSIGN(FlatGraph loaded_graph, LoadFlatGraph(prefix));
  CheckThatGraphsAreEqual(graph, loaded_graph);
}

TEST(FlatGraphIOTest, LoadGraphWithNegativeDimensions) {
  FakeIndex index;
  FlatGraphHeader header;
  header.set_num_nodes(-1);
  header.set_max_degree(0);
  ASSERT_OK(index.WriteTextProto("ginger_flat_graph.header.pbtxt", header));
  EXPECT_THAT(internal::ReadFlatGraph(index).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  header.set_num_nodes(1);
  header.set_max_degree(-1);
  ASSERT_OK(index.WriteTextProto("ginger_flat_graph.header.pbtxt", header));
  EXPECT_THAT(internal::ReadFlatGraph(index).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FlatGraphIOTest, LoadGraphWithOutOfRangeNeighbor) {
  FakeIndex index;
  FlatGraph graph(/*num_nodes=*/2, /*max_degree=*/1);
  graph.SetNeighbors(0, {1});
  // Node 0's only neighbor follows its degree in the flat layout.
  graph.MutableData()[1] = 5;
  ASSERT_OK(internal::WriteFlatGraph(index, graph));
  EXPECT_THAT(internal::ReadFlatGraph(index).status(),
              StatusIs(absl::StatusCode::kDataLoss));
}

TEST(FlatGraphIOTest, LoadGraphWithHeaderThatDoesNotMatchData) {
  FakeIndex index;
  FlatGraph graph(/*num_nodes=*/4, /*max_degree=*/2);
  ASSERT_OK(internal::WriteFlatGraph(index, graph));

  // A corrupt header that claims a huge graph must not trigger a huge
  // allocation.
  FlatGraphHeader header;
  header.set_num_nodes(1 << 30);
  header.set_max_degree(1 << 20);
  ASSERT_OK(index.WriteTextProto("ginger_flat_graph.header.pbtxt", header));
  EXPECT_THAT(internal::ReadFlatGraph(index).status(),
              StatusIs(absl::StatusCode::kDataLoss));
}

TEST(FlatGraphIOTest, LoadNonExistentGraph) {
  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/non_existent_graph");
  EXPECT_THAT(LoadFlatGraph(prefix).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

template <typename Config = CompressedGraphConfig>
void CheckThatCompressedGraphsAreEqual(const CompressedCSRGraph<Config>& a,
                                       const CompressedCSRGraph<Config>& b) {
  ASSERT_EQ(a.NumNodes(), b.NumNodes());
  EXPECT_EQ(a.NumArcs(), b.NumArcs());
  EXPECT_EQ(a.Offsets(), b.Offsets());
  EXPECT_THAT(a.CompressedTargets(), ElementsAreArray(b.CompressedTargets()));
  for (GingerNodeId i = 0; i < a.NumNodes(); ++i) {
    EXPECT_THAT(a.Neighbors(i), ElementsAreArray(b.Neighbors(i)));
  }
}

TEST(CompressedGraphIOTest, SaveAndLoadEmptyGraph) {
  CompressedCSRGraph<> graph(std::vector<int64_t>{0}, std::vector<uint8_t>{},
                             0);
  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/empty_compressed_graph");
  ASSERT_OK(SaveCompressedGraph(graph, prefix));
  ASSERT_OK_AND_ASSIGN(CompressedCSRGraph<> loaded_graph,
                       LoadCompressedGraph(prefix));
  CheckThatCompressedGraphsAreEqual(graph, loaded_graph);
}

TEST(CompressedGraphIOTest, SaveAndLoadGraphWithEdges) {
  FlatGraph flat_graph(4, 3);
  flat_graph.SetNeighbors(0, {1, 2});
  flat_graph.SetNeighbors(1, {0});
  flat_graph.SetNeighbors(2, {0, 3});
  flat_graph.SetNeighbors(3, {2});

  CompressedCSRGraph<> graph = CompressGraph(flat_graph);
  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/compressed_with_edges");
  ASSERT_OK(SaveCompressedGraph(graph, prefix));
  ASSERT_OK_AND_ASSIGN(CompressedCSRGraph<> loaded_graph,
                       LoadCompressedGraph(prefix));
  CheckThatCompressedGraphsAreEqual(graph, loaded_graph);

  AdaptiveCompressedGraph adaptive_graph =
      CompressGraph<FlatGraph, AdaptiveCompressedGraphConfig>(flat_graph);
  std::string adaptive_prefix =
      absl::StrCat(::testing::TempDir(), "/adaptive_compressed_with_edges");
  ASSERT_OK(SaveCompressedGraph(adaptive_graph, adaptive_prefix));
  ASSERT_OK_AND_ASSIGN(
      AdaptiveCompressedGraph loaded_adaptive_graph,
      LoadCompressedGraph<AdaptiveCompressedGraphConfig>(adaptive_prefix));
  CheckThatCompressedGraphsAreEqual(adaptive_graph, loaded_adaptive_graph);
}

TEST(CompressedGraphIOTest, LoadGraphWithNegativeDimensions) {
  FakeIndex index;
  CompressedGraphHeader header;
  header.set_num_nodes(-1);
  header.set_num_arcs(0);
  header.set_num_compressed_bytes(0);
  ASSERT_OK(
      index.WriteTextProto("ginger_compressed_graph.header.pbtxt", header));
  EXPECT_THAT(
      internal::ReadCompressedGraph<CompressedGraphConfig>(index).status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  header.set_num_nodes(1);
  header.set_num_arcs(-1);
  ASSERT_OK(
      index.WriteTextProto("ginger_compressed_graph.header.pbtxt", header));
  EXPECT_THAT(
      internal::ReadCompressedGraph<CompressedGraphConfig>(index).status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  header.set_num_arcs(0);
  header.set_num_compressed_bytes(-1);
  ASSERT_OK(
      index.WriteTextProto("ginger_compressed_graph.header.pbtxt", header));
  EXPECT_THAT(
      internal::ReadCompressedGraph<CompressedGraphConfig>(index).status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompressedGraphIOTest, LoadGraphWithDecreasingOffsets) {
  FakeIndex index;
  CompressedGraphHeader header;
  header.set_num_nodes(2);
  header.set_num_arcs(0);
  header.set_num_compressed_bytes(4);
  ASSERT_OK(
      index.WriteTextProto("ginger_compressed_graph.header.pbtxt", header));
  const std::vector<int64_t> offsets = {0, 3, 1};
  ASSERT_OK(index.WriteArray<int64_t>("ginger_compressed_graph.offsets",
                                      absl::MakeConstSpan(offsets)));
  const std::vector<uint8_t> targets(4, 0);
  ASSERT_OK(index.WriteArray<uint8_t>("ginger_compressed_graph.targets",
                                      absl::MakeConstSpan(targets)));
  EXPECT_THAT(
      internal::ReadCompressedGraph<CompressedGraphConfig>(index).status(),
      StatusIs(absl::StatusCode::kDataLoss));
}

TEST(CompressedGraphIOTest, LoadNonExistentGraph) {
  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/non_existent_compressed_graph");
  EXPECT_THAT(LoadCompressedGraph(prefix).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GingerSearcherIOTest, SaveAndLoadGingerSearcher) {
  int dim = 16;
  int num_points = 20;

  auto float_dataset = std::make_shared<SpanDataset>(num_points, dim);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> pt(dim, static_cast<float>(i));
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  auto tq = std::make_shared<TurboQuantCodec>(dim, 42);
  std::vector<std::vector<float>> points_data(num_points,
                                              std::vector<float>(dim));
  std::vector<absl::Span<const float>> points_spans;
  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < dim; ++j) {
      points_data[i][j] = static_cast<float>(i);
    }
    points_spans.push_back(absl::MakeSpan(points_data[i]));
  }
  ASSERT_OK_AND_ASSIGN(
      auto quantized_base_points,
      TurboQuantDatasetSoA::Create(*tq, absl::MakeSpan(points_spans)));
  auto quantized_dataset =
      std::make_shared<TurboQuantDatasetSoA>(std::move(quantized_base_points));

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);

  auto flat_graph = std::make_unique<FlatGraph>(num_points, 5);
  flat_graph->SetNeighbors(0, {1});
  flat_graph->SetNeighbors(1, {0, 2});
  flat_graph->SetNeighbors(2, {1, 3});
  flat_graph->SetNeighbors(3, {2});
  flat_graph->SetNeighbors(4, {3});

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(float_dataset)
          .AddQuantizedDataset(quantized_dataset, tq)
          .AddIndex(std::move(flat_graph))
          .Build());

  std::string prefix = absl::StrCat(::testing::TempDir(), "/ginger_searcher");
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_searcher, LoadGingerSearcher(prefix));
  std::vector<float> query(dim, 1.0f);
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_TURBO_QUANT
    reranking_strategy: RERANKING_STRATEGY_FLOAT
    beam_size: 10
    num_neighbors: 1
  )pb");
  EXPECT_THAT(loaded_searcher->FindNeighbors(absl::MakeSpan(query), options),
              IsOkAndHolds(SizeIs(1)));
}

TEST(GingerSearcherIOTest, BuildSaveLoadAndSearchReorderedVarintIndex) {
  constexpr int kPoints = 32;
  constexpr int kDimensions = 16;
  auto dataset = std::make_shared<SpanDataset>(kPoints, kDimensions);
  auto flat = std::make_unique<FlatGraph>(kPoints, 4);
  std::vector<GingerNodeId> reordering;
  for (int i = 0; i < kPoints; ++i) {
    std::vector<float> point(kDimensions, 0.0f);
    point[i % kDimensions] = i + 1;
    dataset->Set<float>(i, absl::MakeSpan(point));
    flat->SetNeighbors(i, {(i + 1) % kPoints, (i + 7) % kPoints});
    // Use an asymmetric permutation (differs from its inverse) to verify
    // stored-ID <-> original-ID mapping direction.
    reordering.push_back((i + 5) % kPoints);
  }
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.set_use_graph_compression(true);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(config)
                                          .AddDataset(dataset)
                                          .AddIndex(std::move(flat))
                                          .AddReordering(std::move(reordering))
                                          .Build());
  ASSERT_TRUE(
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          searcher->index()));
  const auto prefix =
      absl::StrCat(::testing::TempDir(), "/reordered_varint_searcher");
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));
  ASSERT_OK_AND_ASSIGN(auto loaded, LoadGingerSearcher(prefix));
  ASSERT_TRUE(
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          loaded->index()));
  EXPECT_EQ(*searcher, *loaded);
  std::vector<float> query(kDimensions, 1.0f);
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_FLOAT
    reranking_strategy: RERANKING_STRATEGY_NONE
    beam_size: 20
    num_neighbors: 5
  )pb");
  ASSERT_OK_AND_ASSIGN(
      auto expected, searcher->FindNeighbors(absl::MakeSpan(query), options));
  ASSERT_OK_AND_ASSIGN(auto actual,
                       loaded->FindNeighbors(absl::MakeSpan(query), options));
  EXPECT_EQ(actual, expected);
}

TEST(GingerSearcherIOTest, BuildSaveLoadAndSearchWithReorderingAlgorithm) {
  constexpr int kPoints = 32;
  constexpr int kDimensions = 16;
  auto dataset = std::make_shared<SpanDataset>(kPoints, kDimensions);
  auto flat = std::make_unique<FlatGraph>(kPoints, 4);
  for (int i = 0; i < kPoints; ++i) {
    std::vector<float> point(kDimensions, 0.0f);
    point[i % kDimensions] = i + 1;
    dataset->Set<float>(i, absl::MakeSpan(point));
    flat->SetNeighbors(i, {(i + 1) % kPoints, (i + 7) % kPoints});
  }
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.set_use_graph_compression(true);
  config.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(config)
                                          .AddDataset(dataset)
                                          .AddIndex(std::move(flat))
                                          .Build());
  ASSERT_TRUE(
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          searcher->index()));
  const auto prefix =
      absl::StrCat(::testing::TempDir(), "/reordering_algorithm_varint_searcher");
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));
  ASSERT_OK_AND_ASSIGN(auto loaded, LoadGingerSearcher(prefix));
  ASSERT_TRUE(
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          loaded->index()));
  EXPECT_EQ(*searcher, *loaded);
  std::vector<float> query(kDimensions, 1.0f);
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_FLOAT
    reranking_strategy: RERANKING_STRATEGY_NONE
    beam_size: 20
    num_neighbors: 5
  )pb");
  ASSERT_OK_AND_ASSIGN(
      auto expected, searcher->FindNeighbors(absl::MakeSpan(query), options));
  ASSERT_OK_AND_ASSIGN(auto actual,
                       loaded->FindNeighbors(absl::MakeSpan(query), options));
  EXPECT_EQ(actual, expected);
}

TEST(GingerSearcherIOTest, LoadGingerSearcherWithLegacyHeaderDefaults) {
  constexpr int kPoints = 16;
  constexpr int kDimensions = 8;
  auto dataset = std::make_shared<SpanDataset>(kPoints, kDimensions);
  auto flat = std::make_unique<FlatGraph>(kPoints, 2);
  for (int i = 0; i < kPoints; ++i) {
    std::vector<float> point(kDimensions, static_cast<float>(i + 1));
    dataset->Set<float>(i, absl::MakeSpan(point));
    flat->SetNeighbors(i, {(i + 1) % kPoints});
  }
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(config)
                                          .AddDataset(dataset)
                                          .AddIndex(std::move(flat))
                                          .Build());

  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/true));

  GingerSearcherHeader header;
  ASSERT_OK(index.ReadTextProto("ginger_searcher.header.pbtxt", header));
  header.clear_entry_point();
  header.clear_has_inverse_reordering();
  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));

  ASSERT_OK_AND_ASSIGN(auto loaded, internal::ReadGingerSearcher(index));
  EXPECT_EQ(*loaded, *searcher);
}

TEST(GingerSearcherIOTest, LoadGingerSearcherFailsWithInvalidEntryPoint) {
  constexpr int kPoints = 16;
  constexpr int kDimensions = 8;
  auto dataset = std::make_shared<SpanDataset>(kPoints, kDimensions);
  auto flat = std::make_unique<FlatGraph>(kPoints, 2);
  for (int i = 0; i < kPoints; ++i) {
    std::vector<float> point(kDimensions, 1.0f);
    dataset->Set<float>(i, absl::MakeSpan(point));
    flat->SetNeighbors(i, {(i + 1) % kPoints});
  }
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(config)
                                          .AddDataset(dataset)
                                          .AddIndex(std::move(flat))
                                          .Build());

  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/true));

  GingerSearcherHeader header;
  ASSERT_OK(index.ReadTextProto("ginger_searcher.header.pbtxt", header));
  header.set_entry_point(kPoints);
  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));
  EXPECT_THAT(internal::ReadGingerSearcher(index),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Entry point is out of bounds")));

  header.set_entry_point(-1);
  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));
  EXPECT_THAT(internal::ReadGingerSearcher(index),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Entry point is out of bounds")));
}

TEST(GingerSearcherIOTest,
     LoadGingerSearcherFailsWithMissingInverseReordering) {
  constexpr int kPoints = 16;
  constexpr int kDimensions = 8;
  auto dataset = std::make_shared<SpanDataset>(kPoints, kDimensions);
  auto flat = std::make_unique<FlatGraph>(kPoints, 2);
  for (int i = 0; i < kPoints; ++i) {
    std::vector<float> point(kDimensions, 1.0f);
    dataset->Set<float>(i, absl::MakeSpan(point));
    flat->SetNeighbors(i, {(i + 1) % kPoints});
  }
  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(config)
                                          .AddDataset(dataset)
                                          .AddIndex(std::move(flat))
                                          .Build());

  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/true));

  GingerSearcherHeader header;
  ASSERT_OK(index.ReadTextProto("ginger_searcher.header.pbtxt", header));
  header.set_has_inverse_reordering(true);
  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));
  EXPECT_THAT(internal::ReadGingerSearcher(index),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GingerSearcherIOTest, SaveAndLoadGingerSearcherWithCompressedIndex) {
  int dim = 16;
  int num_points = 20;

  auto float_dataset = std::make_shared<SpanDataset>(num_points, dim);
  std::vector<float> pt(dim, 0.0f);
  for (int i = 0; i < num_points; ++i) {
    pt[i % dim] = static_cast<float>(i + 1);
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);

  FlatGraph flat_graph(num_points, 5);
  for (GingerNodeId i = 0; i < num_points; ++i) {
    flat_graph.SetNeighbors(i, {(i + 1) % num_points, (i + 2) % num_points});
  }
  auto compressed_index =
      std::make_unique<const CompressedCSRGraph<>>(CompressGraph(flat_graph));

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(float_dataset)
          .AddIndex(std::move(compressed_index))
          .Build());

  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_compressed");
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_searcher, LoadGingerSearcher(prefix));
  ASSERT_NE(loaded_searcher, nullptr);
  EXPECT_TRUE(
      std::holds_alternative<std::unique_ptr<const CompressedCSRGraph<>>>(
          loaded_searcher->index()));

  const auto* loaded_graph =
      std::get<std::unique_ptr<const CompressedCSRGraph<>>>(
          loaded_searcher->index())
          .get();
  ASSERT_NE(loaded_graph, nullptr);
  for (GingerNodeId i = 0; i < num_points; ++i) {
    EXPECT_THAT(loaded_graph->Neighbors(i),
                UnorderedElementsAreArray(flat_graph.Neighbors(i)));
  }

  std::vector<float> query(dim, 1.0f);
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_FLOAT
    reranking_strategy: RERANKING_STRATEGY_NONE
    beam_size: 10
    num_neighbors: 1
  )pb");
  EXPECT_THAT(loaded_searcher->FindNeighbors(absl::MakeSpan(query), options),
              IsOkAndHolds(SizeIs(1)));
}

TEST(GingerSearcherIOTest, SaveGingerSearcherFailsWithCSRGraphIndex) {
  int dim = 16;
  int num_points = 20;

  auto float_dataset = std::make_shared<SpanDataset>(num_points, dim);
  std::vector<float> pt(dim, 0.0f);
  for (int i = 0; i < num_points; ++i) {
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);

  auto csr_index = std::make_unique<const CSRGraph>(
      std::vector<int64_t>(num_points + 1, 0), std::vector<GingerNodeId>());

  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(config)
                                          .AddDataset(float_dataset)
                                          .AddIndex(std::move(csr_index))
                                          .Build());

  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_csr");
  EXPECT_THAT(SaveGingerSearcher(*searcher, prefix),
              StatusIs(absl::StatusCode::kUnimplemented,
                       testing::HasSubstr("Currently only FlatGraph and "
                                          "CompressedCSRGraph indices are "
                                          "supported for serialization.")));
}

TEST(GingerSearcherIOTest, LoadGingerSearcherFailsWithoutVamanaIndex) {
  FakeIndex index;
  GingerSearcherHeader header;
  header.set_has_graph_index(false);

  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));

  EXPECT_THAT(internal::ReadGingerSearcher(index).status(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       testing::HasSubstr("Currently only FlatGraph and "
                                          "CompressedCSRGraph indices are "
                                          "supported for deserialization.")));
}

TEST(GingerSearcherIOTest, LoadGingerSearcherFailsWithCorruptedVamanaIndex) {
  int dim = 16;
  int num_points = 20;

  auto float_dataset = std::make_shared<SpanDataset>(num_points, dim);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> pt(dim, static_cast<float>(i));
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  auto tq = std::make_shared<TurboQuantCodec>(dim, 42);
  std::vector<std::vector<float>> points_data(num_points,
                                              std::vector<float>(dim));
  std::vector<absl::Span<const float>> points_spans;
  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < dim; ++j) {
      points_data[i][j] = static_cast<float>(i);
    }
    points_spans.push_back(absl::MakeSpan(points_data[i]));
  }
  ASSERT_OK_AND_ASSIGN(
      auto quantized_base_points,
      TurboQuantDatasetSoA::Create(*tq, absl::MakeSpan(points_spans)));
  auto quantized_dataset =
      std::make_shared<TurboQuantDatasetSoA>(std::move(quantized_base_points));

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);

  auto flat_graph = std::make_unique<FlatGraph>(num_points, 5);
  flat_graph->SetNeighbors(0, {1});
  flat_graph->SetNeighbors(1, {0, 2});
  flat_graph->SetNeighbors(2, {1, 3});
  flat_graph->SetNeighbors(3, {2});
  flat_graph->SetNeighbors(4, {3});

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(float_dataset)
          .AddQuantizedDataset(quantized_dataset, tq)
          .AddIndex(std::move(flat_graph))
          .Build());

  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/true));

  index.Erase("ginger_flat_graph");

  EXPECT_THAT(internal::ReadGingerSearcher(index).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GingerSearcherIOTest,
     LoadGingerSearcherFailsWithCorruptedQuantizedDataset) {
  int dim = 16;
  int num_points = 20;

  auto float_dataset = std::make_shared<SpanDataset>(num_points, dim);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> pt(dim, static_cast<float>(i));
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  auto tq = std::make_shared<TurboQuantCodec>(dim, 42);
  std::vector<std::vector<float>> points_data(num_points,
                                              std::vector<float>(dim));
  std::vector<absl::Span<const float>> points_spans;
  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < dim; ++j) {
      points_data[i][j] = static_cast<float>(i);
    }
    points_spans.push_back(absl::MakeSpan(points_data[i]));
  }
  ASSERT_OK_AND_ASSIGN(
      auto quantized_base_points,
      TurboQuantDatasetSoA::Create(*tq, absl::MakeSpan(points_spans)));
  auto quantized_dataset =
      std::make_shared<TurboQuantDatasetSoA>(std::move(quantized_base_points));

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);

  auto flat_graph = std::make_unique<FlatGraph>(num_points, 5);
  flat_graph->SetNeighbors(0, {1});
  flat_graph->SetNeighbors(1, {0, 2});
  flat_graph->SetNeighbors(2, {1, 3});
  flat_graph->SetNeighbors(3, {2});
  flat_graph->SetNeighbors(4, {3});

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(float_dataset)
          .AddQuantizedDataset(quantized_dataset, tq)
          .AddIndex(std::move(flat_graph))
          .Build());

  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/true));

  index.Erase("turboquant_dataset");

  EXPECT_THAT(internal::ReadGingerSearcher(index).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GingerSearcherIOTest, SaveAndLoadWithJlCandidateFiltering) {
  int dim = 128;
  int num_points = 20;

  auto float_dataset = std::make_shared<SpanDataset>(num_points, dim);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> pt(dim, static_cast<float>(i));
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  auto tq = std::make_shared<TurboQuantCodec>(dim, 42);
  std::vector<std::vector<float>> points_data(num_points,
                                              std::vector<float>(dim));
  std::vector<absl::Span<const float>> points_spans;
  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < dim; ++j) {
      points_data[i][j] = static_cast<float>(i);
    }
    points_spans.push_back(absl::MakeSpan(points_data[i]));
  }
  ASSERT_OK_AND_ASSIGN(
      auto quantized_base_points,
      TurboQuantDatasetSoA::Create(*tq, absl::MakeSpan(points_spans)));
  auto quantized_dataset =
      std::make_shared<TurboQuantDatasetSoA>(std::move(quantized_base_points));

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_jl()->set_jl_dimensionality(64);
  config.mutable_jl()->set_seed(1234);

  auto flat_graph = std::make_unique<FlatGraph>(num_points, 5);
  flat_graph->SetNeighbors(0, {1});
  flat_graph->SetNeighbors(1, {0, 2});
  flat_graph->SetNeighbors(2, {1, 3});
  flat_graph->SetNeighbors(3, {2});
  flat_graph->SetNeighbors(4, {3});

  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      GingerSearcherBuilder(config)
          .AddDataset(float_dataset)
          .AddQuantizedDataset(quantized_dataset, tq)
          .AddIndex(std::move(flat_graph))
          .Build());

  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_jl");
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_searcher, LoadGingerSearcher(prefix));
  std::vector<float> query(dim, 1.0f);
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_TURBO_QUANT
    reranking_strategy: RERANKING_STRATEGY_FLOAT
    beam_size: 10
    num_neighbors: 1
    candidate_filter { jl {} }
  )pb");
  EXPECT_THAT(loaded_searcher->FindNeighbors(absl::MakeSpan(query), options),
              IsOkAndHolds(SizeIs(1)));
}

TEST(GingerSearcherIOTest, SaveAndLoadWithJlWithoutFloatDataset) {
  int dim = 128;
  int num_points = 20;

  auto float_dataset = std::make_shared<SpanDataset>(num_points, dim);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> pt(dim, static_cast<float>(i));
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  auto tq = std::make_shared<TurboQuantCodec>(dim, 42);
  std::vector<std::vector<float>> points_data(num_points,
                                              std::vector<float>(dim));
  std::vector<absl::Span<const float>> points_spans;
  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < dim; ++j) {
      points_data[i][j] = static_cast<float>(i);
    }
    points_spans.push_back(absl::MakeSpan(points_data[i]));
  }
  ASSERT_OK_AND_ASSIGN(
      auto quantized_base_points,
      TurboQuantDatasetSoA::Create(*tq, absl::MakeSpan(points_spans)));
  auto quantized_dataset =
      std::make_shared<TurboQuantDatasetSoA>(std::move(quantized_base_points));

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_jl()->set_jl_dimensionality(64);
  config.mutable_jl()->set_seed(1234);

  ASSERT_OK_AND_ASSIGN(auto jl_dataset, JlDataset::Create(64, 1234, DOT_PRODUCT,
                                                          *float_dataset));

  auto flat_graph = std::make_unique<FlatGraph>(num_points, 5);
  flat_graph->SetNeighbors(0, {1});
  flat_graph->SetNeighbors(1, {0, 2});
  flat_graph->SetNeighbors(2, {1, 3});
  flat_graph->SetNeighbors(3, {2});
  flat_graph->SetNeighbors(4, {3});

  ASSERT_OK_AND_ASSIGN(auto searcher,
                       GingerSearcherBuilder(config)
                           .AddQuantizedDataset(quantized_dataset, tq)
                           .AddMutableJlDataset(std::move(jl_dataset))
                           .AddIndex(std::move(flat_graph))
                           .Build());

  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_jl_no_float");
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_searcher, LoadGingerSearcher(prefix));
  std::vector<float> query(dim, 1.0f);
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_TURBO_QUANT
    reranking_strategy: RERANKING_STRATEGY_NONE
    beam_size: 10
    num_neighbors: 1
    candidate_filter { jl {} }
  )pb");
  EXPECT_THAT(loaded_searcher->FindNeighbors(absl::MakeSpan(query), options),
              IsOkAndHolds(SizeIs(1)));
}

TEST(GingerSearcherIOTest, LoadGingerSearcherWithMissingJlDatasetFails) {
  int dim = 128;
  int num_points = 20;

  auto float_dataset = std::make_shared<SpanDataset>(num_points, dim);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> pt(dim, static_cast<float>(i));
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  auto tq = std::make_shared<TurboQuantCodec>(dim, 42);
  std::vector<std::vector<float>> points_data(num_points,
                                              std::vector<float>(dim));
  std::vector<absl::Span<const float>> points_spans;
  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < dim; ++j) {
      points_data[i][j] = static_cast<float>(i);
    }
    points_spans.push_back(absl::MakeSpan(points_data[i]));
  }
  ASSERT_OK_AND_ASSIGN(
      auto quantized_base_points,
      TurboQuantDatasetSoA::Create(*tq, absl::MakeSpan(points_spans)));
  auto quantized_dataset =
      std::make_shared<TurboQuantDatasetSoA>(std::move(quantized_base_points));

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_jl()->set_jl_dimensionality(64);
  config.mutable_jl()->set_seed(1234);

  ASSERT_OK_AND_ASSIGN(auto jl_dataset, JlDataset::Create(64, 1234, DOT_PRODUCT,
                                                          *float_dataset));

  auto flat_graph = std::make_unique<FlatGraph>(num_points, 5);
  flat_graph->SetNeighbors(0, {1});
  flat_graph->SetNeighbors(1, {0, 2});
  flat_graph->SetNeighbors(2, {1, 3});
  flat_graph->SetNeighbors(3, {2});
  flat_graph->SetNeighbors(4, {3});

  ASSERT_OK_AND_ASSIGN(auto searcher,
                       GingerSearcherBuilder(config)
                           .AddQuantizedDataset(quantized_dataset, tq)
                           .AddMutableJlDataset(std::move(jl_dataset))
                           .AddIndex(std::move(flat_graph))
                           .Build());

  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/true));

  index.Erase("jl_dataset.header.pbtxt");

  EXPECT_THAT(internal::ReadGingerSearcher(index).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GingerSearcherIOTest, LoadGingerSearcherConfiguredWithJlMissingJlDataset) {
  GingerSearcherHeader header;
  header.mutable_config()->set_distance_metric(DOT_PRODUCT);
  header.mutable_config()->mutable_jl()->set_jl_dimensionality(64);
  header.set_has_jl_dataset(false);

  FakeIndex index;
  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));

  EXPECT_THAT(internal::ReadGingerSearcher(index).status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       testing::HasSubstr("missing JL dataset")));
}

TEST(GingerSearcherIOTest,
     LoadGingerSearcherConfiguredWithPcaMissingPcaProjection) {
  GingerSearcherHeader header;
  header.mutable_config()->set_distance_metric(DOT_PRODUCT);
  header.mutable_config()->mutable_pca()->set_dimension(64);
  header.set_has_pca_projection(false);

  FakeIndex index;
  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));

  EXPECT_THAT(internal::ReadGingerSearcher(index).status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       testing::HasSubstr("missing PCA projection")));
  EXPECT_THAT(internal::ReadGingerSearcherBuilder(index).status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       testing::HasSubstr("missing PCA projection")));
}

TEST(GingerSearcherIOTest,
     LoadGingerSearcherFailsWithInvalidComponentDimensions) {
  FakeIndex index;
  FlatGraph graph(2, 1);
  ASSERT_OK(internal::WriteFlatGraph(index, graph));

  GingerSearcherHeader base_header;
  base_header.mutable_config()->set_distance_metric(DOT_PRODUCT);
  base_header.set_has_graph_index(true);
  base_header.set_index_type(GingerSearcherHeader::INDEX_TYPE_FLAT_GRAPH);

  {
    GingerSearcherHeader h = base_header;
    h.set_has_float_dataset(true);
    ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", h));
    DatasetHeader ds_header;
    ds_header.set_num_points(10);
    ds_header.set_dimensionality(10);

    ds_header.set_num_points(-1);
    ASSERT_OK(index.WriteTextProto("float_dataset.header.pbtxt", ds_header));
    EXPECT_THAT(internal::ReadGingerSearcher(index),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("num_points must be non-negative and "
                                   "dimensionality must be positive")));

    ds_header.set_num_points(10);
    ds_header.set_dimensionality(0);
    ASSERT_OK(index.WriteTextProto("float_dataset.header.pbtxt", ds_header));
    EXPECT_THAT(internal::ReadGingerSearcher(index),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("num_points must be non-negative and "
                                   "dimensionality must be positive")));
  }

  {
    GingerSearcherHeader h = base_header;
    h.set_has_jl_dataset(true);
    ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", h));
    JlDatasetHeader jl_header;
    jl_header.set_num_points(10);
    jl_header.set_source_dim(64);
    jl_header.set_jl_dim(128);
    jl_header.set_metric(DOT_PRODUCT);

    jl_header.set_num_points(-1);
    ASSERT_OK(index.WriteTextProto("jl_dataset.header.pbtxt", jl_header));
    EXPECT_THAT(internal::ReadGingerSearcher(index),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("num_points must be non-negative and "
                                   "source_dim must be positive")));

    jl_header.set_num_points(10);
    jl_header.set_source_dim(0);
    ASSERT_OK(index.WriteTextProto("jl_dataset.header.pbtxt", jl_header));
    EXPECT_THAT(internal::ReadGingerSearcher(index),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("num_points must be non-negative and "
                                   "source_dim must be positive")));

    jl_header.set_source_dim(64);
    jl_header.set_jl_dim(63);
    ASSERT_OK(index.WriteTextProto("jl_dataset.header.pbtxt", jl_header));
    EXPECT_THAT(
        internal::ReadGingerSearcher(index),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("jl_dim must be positive and a multiple of 64")));
  }

  {
    GingerSearcherHeader h = base_header;
    h.set_has_pca_projection(true);
    ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", h));
    PcaProjectionHeader pca_header;
    pca_header.set_matrix_representation(MATRIX_REPRESENTATION_FLOAT);

    pca_header.set_input_dim(0);
    pca_header.set_output_dim(16);
    ASSERT_OK(index.WriteTextProto("pca_matrix.header.pbtxt", pca_header));
    EXPECT_THAT(
        internal::ReadGingerSearcher(index),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("Projection dimensions must be strictly positive.")));

    pca_header.set_input_dim(32);
    pca_header.set_output_dim(0);
    ASSERT_OK(index.WriteTextProto("pca_matrix.header.pbtxt", pca_header));
    EXPECT_THAT(
        internal::ReadGingerSearcher(index),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("Projection dimensions must be strictly positive.")));
  }
}

absl::StatusOr<std::unique_ptr<GingerSearcher>> CreateTestSearcherWithPca(
    MatrixRepresentation rep, std::shared_ptr<TurboQuantCodec> tq,
    bool include_jl = false) {
  constexpr int input_dim = 128;
  constexpr int output_dim = 64;
  constexpr int num_points = 20;

  auto float_dataset = std::make_shared<SpanDataset>(num_points, input_dim);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> pt(input_dim, static_cast<float>(i));
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  Eigen::MatrixXf mat(output_dim, input_dim);
  mat.setIdentity();
  std::unique_ptr<PcaProjection> proj;
  if (rep == MATRIX_REPRESENTATION_FLOAT) {
    ABSL_ASSIGN_OR_RETURN(proj, FloatPcaProjection::Create(mat));
  } else {
    ABSL_ASSIGN_OR_RETURN(proj, TurboQuantPcaProjection::Create(mat));
  }

  std::vector<std::vector<float>> points_data(num_points,
                                              std::vector<float>(output_dim));
  std::vector<absl::Span<const float>> points_spans;
  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < output_dim; ++j) {
      points_data[i][j] = static_cast<float>(i);
    }
    points_spans.push_back(absl::MakeSpan(points_data[i]));
  }
  ABSL_ASSIGN_OR_RETURN(
      auto quantized_base_points,
      TurboQuantDatasetSoA::Create(*tq, absl::MakeSpan(points_spans)));
  auto quantized_dataset =
      std::make_shared<TurboQuantDatasetSoA>(std::move(quantized_base_points));

  auto flat_graph = std::make_unique<FlatGraph>(num_points, 5);
  for (int i = 0; i < num_points; ++i) {
    flat_graph->SetNeighbors(i, {(i + 1) % num_points});
  }

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_pca()->set_dimension(output_dim);
  config.mutable_pca()->set_matrix_representation(rep);
  if (include_jl) {
    config.mutable_jl()->set_jl_dimensionality(64);
    config.mutable_jl()->set_seed(1234);
  }

  GingerSearcherBuilder builder(config);
  builder.AddMutableDataset(float_dataset)
      .AddQuantizedDataset(quantized_dataset, tq)
      .AddProjection(std::move(proj))
      .AddIndex(std::move(flat_graph));

  if (include_jl) {
    ABSL_ASSIGN_OR_RETURN(
        auto jl_dataset,
        JlDataset::Create(64, 1234, DOT_PRODUCT, *float_dataset));
    builder.AddMutableJlDataset(std::move(jl_dataset));
  }

  return builder.Build();
}

class GingerSearcherPcaIOTest
    : public ::testing::TestWithParam<MatrixRepresentation> {};

TEST_P(GingerSearcherPcaIOTest, SaveAndLoadWithPca) {
  auto tq = std::make_shared<TurboQuantCodec>(64, 42);
  ASSERT_OK_AND_ASSIGN(auto searcher,
                       CreateTestSearcherWithPca(GetParam(), tq));

  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_pca_", GetParam());
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_searcher, LoadGingerSearcher(prefix));
  std::vector<float> query(128, 1.0f);
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_TURBO_QUANT
    reranking_strategy: RERANKING_STRATEGY_FLOAT
    beam_size: 10
    num_neighbors: 1
  )pb");
  EXPECT_THAT(loaded_searcher->FindNeighbors(absl::MakeSpan(query), options),
              IsOkAndHolds(SizeIs(1)));
}

TEST_P(GingerSearcherPcaIOTest,
       LoadGingerSearcherWithMissingPcaProjectionFails) {
  auto tq = std::make_shared<TurboQuantCodec>(64, 42);
  ASSERT_OK_AND_ASSIGN(auto searcher,
                       CreateTestSearcherWithPca(GetParam(), tq));

  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/true));

  index.Erase("pca_matrix.header.pbtxt");

  EXPECT_THAT(internal::ReadGingerSearcher(index).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

INSTANTIATE_TEST_SUITE_P(AllRepresentations, GingerSearcherPcaIOTest,
                         ::testing::Values(MATRIX_REPRESENTATION_FLOAT,
                                           MATRIX_REPRESENTATION_TURBOQUANT));

TEST(GingerSearcherIOTest, SaveAndLoadWithPcaAndJl) {
  auto tq = std::make_shared<TurboQuantCodec>(64, 42);
  ASSERT_OK_AND_ASSIGN(
      auto searcher, CreateTestSearcherWithPca(MATRIX_REPRESENTATION_FLOAT, tq,
                                               /*include_jl=*/true));

  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_pca_jl");
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_searcher, LoadGingerSearcher(prefix));
  std::vector<float> query(128, 1.0f);
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_TURBO_QUANT
    reranking_strategy: RERANKING_STRATEGY_FLOAT
    beam_size: 10
    num_neighbors: 1
    candidate_filter { jl {} }
  )pb");
  EXPECT_THAT(loaded_searcher->FindNeighbors(absl::MakeSpan(query), options),
              IsOkAndHolds(SizeIs(1)));
}

TEST(GingerSearcherIOTest, SaveAndLoadWithPcaTruncationFallback) {
  ginger::internal::ParallelSchedulerReference scheduler{4};
  constexpr int num_points = 4;
  constexpr int input_dim = 2;
  auto float_dataset = std::make_shared<SpanDataset>(num_points, input_dim);
  std::vector<std::vector<float>> points = {
      {1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}};
  for (int i = 0; i < num_points; ++i) {
    float_dataset->Set<float>(i, absl::MakeSpan(points[i]));
  }

  auto flat_graph = std::make_unique<FlatGraph>(num_points, 5);
  for (int i = 0; i < num_points; ++i) {
    flat_graph->SetNeighbors(i, {(i + 1) % num_points});
  }

  GingerSearcherConfig config;
  config.set_distance_metric(DOT_PRODUCT);
  config.mutable_turbo_quant()->set_seed(123);

  PcaProjectionConfig pca_config;
  auto* truncation = pca_config.mutable_truncation();
  truncation->set_significance_threshold(0.9f);
  truncation->set_truncation_threshold(0.1f);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GingerSearcher> searcher,
                       GingerSearcherBuilder(config)
                           .AddDataset(float_dataset)
                           .AddIndex(std::move(flat_graph))
                           .SetPcaProjectionConfig(pca_config)
                           .Build());

  ASSERT_NE(searcher, nullptr);

  std::string prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_pca_fallback");
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_searcher, LoadGingerSearcher(prefix));
  ASSERT_NE(loaded_searcher, nullptr);

  std::vector<float> query = {1.0f, 0.0f};
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_TURBO_QUANT
    reranking_strategy: RERANKING_STRATEGY_FLOAT
    beam_size: 2
    num_neighbors: 1
  )pb");
  EXPECT_THAT(searcher->FindNeighbors(absl::MakeSpan(query), options),
              IsOkAndHolds(SizeIs(1)));
  EXPECT_THAT(loaded_searcher->FindNeighbors(absl::MakeSpan(query), options),
              IsOkAndHolds(SizeIs(1)));
}

TEST(GingerSearcherIOTest, LoadGingerSearcherWithoutExplicitCodec) {
  ginger::internal::ParallelSchedulerReference scheduler{4};
  constexpr int kDim = 16;
  constexpr int kNumPoints = 20;

  auto float_dataset = std::make_shared<SpanDataset>(kNumPoints, kDim);
  for (int i = 0; i < kNumPoints; ++i) {
    std::vector<float> pt(kDim, 0.0f);
    pt[i % kDim] = static_cast<float>(i + 1);
    float_dataset->Set<float>(i, absl::MakeSpan(pt));
  }

  auto make_graph = [&]() {
    auto flat_graph = std::make_unique<FlatGraph>(kNumPoints, 4);
    for (int i = 0; i < kNumPoints; ++i) {
      flat_graph->SetNeighbors(i, {(i + 1) % kNumPoints, (i + 3) % kNumPoints});
    }
    return flat_graph;
  };

  // Quantized SQUARED_L2 searcher reconstructed without passing a codec.
  GingerSearcherConfig quantized_config;
  quantized_config.set_distance_metric(SQUARED_L2);
  quantized_config.mutable_turbo_quant()->set_seed(77);
  quantized_config.mutable_turbo_quant()->set_extra_bits(2);

  ASSERT_OK_AND_ASSIGN(auto quantized_searcher,
                       GingerSearcherBuilder(quantized_config)
                           .AddDataset(float_dataset)
                           .AddIndex(make_graph())
                           .Build());

  const std::string quantized_prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_auto_codec");
  ASSERT_OK(SaveGingerSearcher(*quantized_searcher, quantized_prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_quantized,
                       LoadGingerSearcher(quantized_prefix));
  EXPECT_EQ(*quantized_searcher, *loaded_quantized);

  std::vector<float> query(kDim, 1.0f);
  SearchOptions tq_options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_TURBO_QUANT
    reranking_strategy: RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS
    beam_size: 10
    num_neighbors: 3
  )pb");
  ASSERT_OK_AND_ASSIGN(
      auto expected_tq,
      quantized_searcher->FindNeighbors(absl::MakeSpan(query), tq_options));
  ASSERT_OK_AND_ASSIGN(auto actual_tq, loaded_quantized->FindNeighbors(
                                           absl::MakeSpan(query), tq_options));
  EXPECT_EQ(actual_tq, expected_tq);

  // Float-only searcher loaded with the single-argument overload.
  GingerSearcherConfig float_config;
  float_config.set_distance_metric(DOT_PRODUCT);
  ASSERT_OK_AND_ASSIGN(auto float_searcher, GingerSearcherBuilder(float_config)
                                                .AddDataset(float_dataset)
                                                .AddIndex(make_graph())
                                                .Build());

  const std::string float_prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_float_auto_codec");
  ASSERT_OK(SaveGingerSearcher(*float_searcher, float_prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_float, LoadGingerSearcher(float_prefix));
  EXPECT_EQ(*float_searcher, *loaded_float);

  SearchOptions float_options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_FLOAT
    reranking_strategy: RERANKING_STRATEGY_NONE
    beam_size: 10
    num_neighbors: 3
  )pb");
  ASSERT_OK_AND_ASSIGN(
      auto expected_float,
      float_searcher->FindNeighbors(absl::MakeSpan(query), float_options));
  ASSERT_OK_AND_ASSIGN(
      auto actual_float,
      loaded_float->FindNeighbors(absl::MakeSpan(query), float_options));
  EXPECT_EQ(actual_float, expected_float);
}

TEST(GingerSearcherIOTest, LoadGingerSearcherWithPcaWithoutExplicitCodec) {
  auto tq = std::make_shared<TurboQuantCodec>(64, 42);
  ASSERT_OK_AND_ASSIGN(
      auto searcher,
      CreateTestSearcherWithPca(MATRIX_REPRESENTATION_TURBOQUANT, tq));

  const std::string prefix =
      absl::StrCat(::testing::TempDir(), "/ginger_searcher_pca_auto_codec");
  ASSERT_OK(SaveGingerSearcher(*searcher, prefix));

  ASSERT_OK_AND_ASSIGN(auto loaded_searcher, LoadGingerSearcher(prefix));
  EXPECT_EQ(*searcher, *loaded_searcher);

  std::vector<float> query(128, 1.0f);
  SearchOptions options = PARSE_TEXT_PROTO(R"pb(
    search_space: SEARCH_SPACE_TURBO_QUANT
    reranking_strategy: RERANKING_STRATEGY_FLOAT
    beam_size: 10
    num_neighbors: 3
  )pb");
  ASSERT_OK_AND_ASSIGN(auto expected,
                       searcher->FindNeighbors(absl::MakeSpan(query), options));
  ASSERT_OK_AND_ASSIGN(auto actual, loaded_searcher->FindNeighbors(
                                        absl::MakeSpan(query), options));
  EXPECT_EQ(actual, expected);
}

class GingerSearcherBuilderIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (int i = 0; i < kPoints; ++i) {
      std::vector<float> pt(kDim, 0.1f * static_cast<float>(i + 1));
      pt[i % kDim] += static_cast<float>(i + 1);
      dataset_->Set<float>(i, absl::MakeSpan(pt));
      mutable_copy_->Set<float>(i, absl::MakeSpan(pt));
      reordering_.push_back((i + 5) % kPoints);
    }
  }

  std::unique_ptr<FlatGraph> MakeGraph() {
    auto flat = std::make_unique<FlatGraph>(kPoints, 4);
    for (int i = 0; i < kPoints; ++i) {
      flat->SetNeighbors(i, {(i + 1) % kPoints, (i + 7) % kPoints});
    }
    return flat;
  }

  static absl::Span<const float> FloatSpan(const GingerSearcher& s) {
    const auto& d = *GingerSearcherAccess::dataset(s);
    return absl::MakeConstSpan(static_cast<const float*>(d[0]),
                               d.size() * d.dimensionality());
  }

  static void CheckTqJl(const GingerSearcher& a, const GingerSearcher& b) {
    const auto& qa = *GingerSearcherAccess::quantized_dataset(a);
    const auto& qb = *GingerSearcherAccess::quantized_dataset(b);
    EXPECT_EQ(absl::MakeConstSpan(qa.quantized_data(),
                                  qa.TestingOnlyGetQuantizedDataSize()),
              absl::MakeConstSpan(qb.quantized_data(),
                                  qb.TestingOnlyGetQuantizedDataSize()));
    for (size_t i = 0; i < qa.num_datapoints(); ++i) {
      EXPECT_EQ(qa.norm_scaling_factor(i), qb.norm_scaling_factor(i));
    }
    EXPECT_EQ(GingerSearcherAccess::jl_dataset(a)->data(),
              GingerSearcherAccess::jl_dataset(b)->data());
  }

  ginger::internal::ParallelSchedulerReference scheduler_{4};
  static constexpr int kPoints = 32;
  static constexpr int kDim = 128;
  std::shared_ptr<SpanDataset> dataset_ =
      std::make_shared<SpanDataset>(kPoints, kDim);
  std::shared_ptr<SpanDataset> mutable_copy_ =
      std::make_shared<SpanDataset>(kPoints, kDim);
  std::vector<GingerNodeId> reordering_;
};

TEST_F(GingerSearcherBuilderIOTest, AttachFloatDatasetCopy) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.set_use_graph_compression(true);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .SetEntryPoint(3)
                                          .AddIndex(MakeGraph())
                                          .AddReordering(reordering_)
                                          .Build());
  const std::string path =
      absl::StrCat(::testing::TempDir(), "/builder_float_copy");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/false));
  EXPECT_THAT(LoadGingerSearcher(path),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("saved without datasets")));

  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder b,
                       LoadGingerSearcherBuilder(path));
  ASSERT_OK_AND_ASSIGN(auto loaded, b.AddDataset(dataset_).Build());
  EXPECT_EQ(*loaded, *searcher);
  EXPECT_EQ(FloatSpan(*loaded), FloatSpan(*searcher));
}

TEST_F(GingerSearcherBuilderIOTest, AttachFloatDatasetInPlace) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.set_use_graph_compression(true);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .SetEntryPoint(3)
                                          .AddIndex(MakeGraph())
                                          .AddReordering(reordering_)
                                          .Build());
  const std::string path =
      absl::StrCat(::testing::TempDir(), "/builder_float_inplace");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/false));

  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder b,
                       LoadGingerSearcherBuilder(path));
  ASSERT_OK_AND_ASSIGN(
      auto loaded,
      b.AddMutableDataset(mutable_copy_).SetReorderInPlace(true).Build());
  EXPECT_EQ(*loaded, *searcher);
  EXPECT_EQ(FloatSpan(*loaded), FloatSpan(*searcher));
}

TEST_F(GingerSearcherBuilderIOTest, RecomputePcaTqJlWithDropDataset) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.set_use_graph_compression(true);
  cfg.set_reordering_algorithm(RECURSIVE_GRAPH_PARTITIONING);
  cfg.mutable_turbo_quant()->set_seed(77);
  cfg.mutable_turbo_quant()->set_extra_bits(2);
  cfg.mutable_pca()->set_dimension(64);
  cfg.mutable_pca()->set_matrix_representation(
      MATRIX_REPRESENTATION_TURBOQUANT);
  cfg.mutable_jl()->set_jl_dimensionality(64);
  cfg.mutable_jl()->set_seed(1234);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .AddIndex(MakeGraph())
                                          .SetDropDataset(true)
                                          .Build());
  const std::string path =
      absl::StrCat(::testing::TempDir(), "/builder_pca_tq_jl");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/false));
  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder b,
                       LoadGingerSearcherBuilder(path));
  ASSERT_OK_AND_ASSIGN(auto loaded,
                       b.AddDataset(dataset_).SetDropDataset(true).Build());
  EXPECT_EQ(*loaded, *searcher);
  CheckTqJl(*loaded, *searcher);
}

TEST_F(GingerSearcherBuilderIOTest, AttachManualQuantizedAndJlDatasets) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.set_use_graph_compression(true);
  ASSERT_OK_AND_ASSIGN(auto float_searcher, GingerSearcherBuilder(cfg)
                                                .AddDataset(dataset_)
                                                .SetEntryPoint(3)
                                                .AddIndex(MakeGraph())
                                                .AddReordering(reordering_)
                                                .Build());
  const std::string path =
      absl::StrCat(::testing::TempDir(), "/builder_manual");
  ASSERT_OK(SaveGingerSearcher(*float_searcher, path, /*save_datasets=*/false));

  auto tq_codec = std::make_shared<TurboQuantCodec>(kDim, 42);
  std::vector<absl::Span<const float>> spans;
  for (int i = 0; i < kPoints; ++i) spans.push_back(dataset_->get<float>(i));

  ASSERT_OK_AND_ASSIGN(auto q_base,
                       TurboQuantDatasetSoA::Create(*tq_codec, spans));
  ASSERT_OK_AND_ASSIGN(auto jl_base,
                       JlDataset::Create(64, 1234, DOT_PRODUCT, *dataset_));
  ASSERT_OK_AND_ASSIGN(
      auto manual_base,
      GingerSearcherBuilder(cfg)
          .AddQuantizedDataset(
              std::make_shared<TurboQuantDatasetSoA>(std::move(q_base)),
              tq_codec)
          .AddJlDataset(std::move(jl_base))
          .SetEntryPoint(3)
          .AddIndex(MakeGraph())
          .AddReordering(reordering_)
          .Build());

  ASSERT_OK_AND_ASSIGN(auto q_copy,
                       TurboQuantDatasetSoA::Create(*tq_codec, spans));
  ASSERT_OK_AND_ASSIGN(auto jl_copy,
                       JlDataset::Create(64, 1234, DOT_PRODUCT, *dataset_));
  ASSERT_OK_AND_ASSIGN(auto b1, LoadGingerSearcherBuilder(path));
  ASSERT_OK_AND_ASSIGN(
      auto manual_copy,
      b1.AddQuantizedDataset(
            std::make_shared<TurboQuantDatasetSoA>(std::move(q_copy)), tq_codec)
          .AddJlDataset(std::move(jl_copy))
          .Build());
  CheckTqJl(*manual_copy, *manual_base);

  ASSERT_OK_AND_ASSIGN(auto q_mut,
                       TurboQuantDatasetSoA::Create(*tq_codec, spans));
  ASSERT_OK_AND_ASSIGN(auto jl_mut,
                       JlDataset::Create(64, 1234, DOT_PRODUCT, *dataset_));
  ASSERT_OK_AND_ASSIGN(auto b2, LoadGingerSearcherBuilder(path));
  ASSERT_OK_AND_ASSIGN(
      auto manual_mut,
      b2.AddMutableQuantizedDataset(
            std::make_shared<TurboQuantDatasetSoA>(std::move(q_mut)), tq_codec)
          .AddMutableJlDataset(std::move(jl_mut))
          .SetReorderInPlace(true)
          .Build());
  CheckTqJl(*manual_mut, *manual_base);
}

TEST_F(GingerSearcherBuilderIOTest, LoadPcaSearcherWithDropDataset) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.mutable_turbo_quant()->set_seed(77);
  cfg.mutable_pca()->set_dimension(64);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .AddIndex(MakeGraph())
                                          .SetDropDataset(true)
                                          .Build());
  const std::string path = absl::StrCat(::testing::TempDir(), "/pca_drop");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/true));
  ASSERT_OK_AND_ASSIGN(auto loaded, LoadGingerSearcher(path));
  EXPECT_EQ(*loaded, *searcher);
}

TEST_F(GingerSearcherBuilderIOTest, LoadBuilderWithSavedDatasets) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.mutable_turbo_quant()->set_seed(77);
  cfg.mutable_jl()->set_jl_dimensionality(64);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .SetEntryPoint(3)
                                          .AddIndex(MakeGraph())
                                          .AddReordering(reordering_)
                                          .Build());
  const std::string path = absl::StrCat(::testing::TempDir(), "/saved_ds");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/true));
  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder b,
                       LoadGingerSearcherBuilder(path));
  ASSERT_OK_AND_ASSIGN(auto loaded, b.Build());
  EXPECT_EQ(*loaded, *searcher);
  EXPECT_EQ(FloatSpan(*loaded), FloatSpan(*searcher));
  CheckTqJl(*loaded, *searcher);
}

TEST_F(GingerSearcherBuilderIOTest, LoadBuilderMixedSavedAndAttachedDatasets) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.mutable_turbo_quant()->set_seed(77);
  cfg.mutable_jl()->set_jl_dimensionality(64);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .SetEntryPoint(3)
                                          .AddIndex(MakeGraph())
                                          .AddReordering(reordering_)
                                          .SetDropDataset(true)
                                          .Build());
  const std::string path = absl::StrCat(::testing::TempDir(), "/mixed_ds");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/true));

  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder b,
                       LoadGingerSearcherBuilder(path));
  ASSERT_OK_AND_ASSIGN(auto loaded,
                       b.AddDataset(dataset_).SetDropDataset(false).Build());

  ASSERT_OK_AND_ASSIGN(auto ref, GingerSearcherBuilder(cfg)
                                     .AddDataset(dataset_)
                                     .SetEntryPoint(3)
                                     .AddIndex(MakeGraph())
                                     .AddReordering(reordering_)
                                     .Build());
  EXPECT_EQ(FloatSpan(*loaded), FloatSpan(*ref));
  CheckTqJl(*loaded, *searcher);
}

TEST_F(GingerSearcherBuilderIOTest, SaveAndLoadTurboSliceDataset) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.mutable_turbo_quant()->set_seed(77);
  cfg.mutable_turbo_quant()->set_turboslice_dimensionality(64);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .AddIndex(MakeGraph())
                                          .Build());
  const std::string path = absl::StrCat(::testing::TempDir(), "/turboslice");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/true));
  ASSERT_OK_AND_ASSIGN(auto loaded, LoadGingerSearcher(path));

  const auto& orig_tq = *GingerSearcherAccess::quantized_dataset(*searcher);
  const auto& loaded_tq = *GingerSearcherAccess::quantized_dataset(*loaded);
  ASSERT_TRUE(loaded_tq.IsTurboSliceEnabled());
  EXPECT_EQ(
      GingerSearcherAccess::quantizer(*loaded)->turboslice_dimensionality(),
      64);
  EXPECT_EQ(absl::MakeConstSpan(loaded_tq.quantized_data(),
                                loaded_tq.TestingOnlyGetQuantizedDataSize()),
            absl::MakeConstSpan(orig_tq.quantized_data(),
                                orig_tq.TestingOnlyGetQuantizedDataSize()));
  for (size_t i = 0; i < orig_tq.num_datapoints(); ++i) {
    EXPECT_EQ(loaded_tq.norm_scaling_factor_turboslice(i),
              orig_tq.norm_scaling_factor_turboslice(i));
  }
}

TEST_F(GingerSearcherBuilderIOTest, LoadUsesSavedScalingStorageMode) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.mutable_turbo_quant()->set_seed(77);
  cfg.mutable_turbo_quant()->set_padding(true);
  cfg.mutable_turbo_quant()->set_scaling_storage_mode(
      TurboQuantConfig::SCALING_ROW_TAIL);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .AddIndex(MakeGraph())
                                          .Build());
  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/true));

  // The mode in the saved config is ignored.
  GingerSearcherHeader header;
  ASSERT_OK(index.ReadTextProto("ginger_searcher.header.pbtxt", header));
  header.mutable_config()->mutable_turbo_quant()->set_scaling_storage_mode(
      TurboQuantConfig::SCALING_SEPARATE_ARRAY);
  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));
  ASSERT_OK_AND_ASSIGN(auto loaded, internal::ReadGingerSearcher(index));
  const auto& orig_tq = *GingerSearcherAccess::quantized_dataset(*searcher);
  const auto& loaded_tq = *GingerSearcherAccess::quantized_dataset(*loaded);
  EXPECT_EQ(loaded_tq.scaling_storage_mode(), ScalingStorageMode::kRowTail);
  EXPECT_EQ(loaded_tq.stride(), orig_tq.stride());
  EXPECT_EQ(absl::MakeConstSpan(loaded_tq.quantized_data(),
                                loaded_tq.TestingOnlyGetQuantizedDataSize()),
            absl::MakeConstSpan(orig_tq.quantized_data(),
                                orig_tq.TestingOnlyGetQuantizedDataSize()));
}

TEST_F(GingerSearcherBuilderIOTest, SaveAndLoadPcaRerankDataset) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.mutable_turbo_quant()->set_seed(77);
  cfg.mutable_turbo_quant()->set_extra_bits(2);
  cfg.mutable_pca()->set_dimension(64);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .AddIndex(MakeGraph())
                                          .Build());
  const std::string path = absl::StrCat(::testing::TempDir(), "/pca_rerank");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/true));
  ASSERT_OK_AND_ASSIGN(auto loaded, LoadGingerSearcher(path));

  const auto& orig = GingerSearcherAccess::rerank_quantized_dataset(*searcher);
  const auto& got = GingerSearcherAccess::rerank_quantized_dataset(*loaded);
  ASSERT_NE(orig, nullptr);
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(got->dimensionality(), kDim);
  EXPECT_EQ(got->quantization_bits(), 6);
  EXPECT_EQ(absl::MakeConstSpan(got->quantized_data(),
                                got->TestingOnlyGetQuantizedDataSize()),
            absl::MakeConstSpan(orig->quantized_data(),
                                orig->TestingOnlyGetQuantizedDataSize()));
  for (size_t i = 0; i < orig->num_datapoints(); ++i) {
    EXPECT_EQ(got->norm_scaling_factor_4plusk(i),
              orig->norm_scaling_factor_4plusk(i));
  }
  EXPECT_EQ(GingerSearcherAccess::rerank_quantizer(*loaded)->seed(), 77);
}

TEST_F(GingerSearcherBuilderIOTest, SaveAndLoad3BitPrefixDataset) {
  for (const int extra_bits : {0, 2}) {
    for (const int turboslice_dimensionality : {0, 64}) {
      SCOPED_TRACE(absl::StrCat("extra_bits=", extra_bits,
                                " turboslice=", turboslice_dimensionality));
      GingerSearcherConfig cfg;
      cfg.set_distance_metric(DOT_PRODUCT);
      cfg.mutable_turbo_quant()->set_seed(77);
      cfg.mutable_turbo_quant()->set_extra_bits(extra_bits);
      cfg.mutable_turbo_quant()->set_turboslice_dimensionality(
          turboslice_dimensionality);
      cfg.mutable_turbo_quant()->set_scaling_storage_mode(
          TurboQuantConfig::SCALING_IN_THREE_BIT_PREFIX);
      ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                              .AddDataset(dataset_)
                                              .AddIndex(MakeGraph())
                                              .Build());
      const std::string path =
          absl::StrCat(::testing::TempDir(), "/prefix_", extra_bits, "_",
                       turboslice_dimensionality);
      ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/true));
      ASSERT_OK_AND_ASSIGN(auto loaded, LoadGingerSearcher(path));

      const auto& orig_tq = *GingerSearcherAccess::quantized_dataset(*searcher);
      const auto& loaded_tq = *GingerSearcherAccess::quantized_dataset(*loaded);
      ASSERT_TRUE(orig_tq.has_3bit_prefix());
      EXPECT_TRUE(loaded_tq.has_3bit_prefix());
      EXPECT_EQ(
          absl::MakeConstSpan(loaded_tq.quantized_data(),
                              loaded_tq.TestingOnlyGetQuantizedDataSize()),
          absl::MakeConstSpan(orig_tq.quantized_data(),
                              orig_tq.TestingOnlyGetQuantizedDataSize()));
      const size_t extra_precision_size =
          orig_tq.num_datapoints() * orig_tq.extra_precision_stride();
      ASSERT_EQ(loaded_tq.extra_precision_stride(),
                orig_tq.extra_precision_stride());
      EXPECT_EQ(absl::MakeConstSpan(loaded_tq.extra_precision_data(),
                                    extra_precision_size),
                absl::MakeConstSpan(orig_tq.extra_precision_data(),
                                    extra_precision_size));
      for (size_t i = 0; i < orig_tq.num_datapoints(); ++i) {
        EXPECT_EQ(loaded_tq.norm_scaling_factor(i),
                  orig_tq.norm_scaling_factor(i));
        EXPECT_EQ(loaded_tq.norm_scaling_factor_4plusk(i),
                  orig_tq.norm_scaling_factor_4plusk(i));
        EXPECT_EQ(loaded_tq.norm_scaling_factor_turboslice(i),
                  orig_tq.norm_scaling_factor_turboslice(i));
      }

      std::vector<float> query(kDim, 1.0f);
      SearchOptions options = PARSE_TEXT_PROTO(R"pb(
        search_space: SEARCH_SPACE_TURBO_QUANT
        beam_size: 10
        num_neighbors: 5
      )pb");
      options.set_reranking_strategy(
          extra_bits > 0 ? RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS
                         : RERANKING_STRATEGY_NONE);
      ASSERT_OK_AND_ASSIGN(auto expected, searcher->FindNeighbors(
                                              absl::MakeSpan(query), options));
      ASSERT_OK_AND_ASSIGN(
          auto actual, loaded->FindNeighbors(absl::MakeSpan(query), options));
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST_F(GingerSearcherBuilderIOTest, LoadWithoutScalingStorageModeHasNoPrefix) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.mutable_turbo_quant()->set_seed(77);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .AddIndex(MakeGraph())
                                          .Build());
  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/true));

  // Older files do not record the mode. Their points never use the 3-bit
  // prefix, even if the saved config asks for it.
  QuantizedDatasetHeader dataset_header;
  ASSERT_OK(
      index.ReadTextProto("turboquant_dataset.header.pbtxt", dataset_header));
  dataset_header.clear_scaling_storage_mode();
  ASSERT_OK(
      index.WriteTextProto("turboquant_dataset.header.pbtxt", dataset_header));
  GingerSearcherHeader header;
  ASSERT_OK(index.ReadTextProto("ginger_searcher.header.pbtxt", header));
  header.mutable_config()->mutable_turbo_quant()->set_scaling_storage_mode(
      TurboQuantConfig::SCALING_IN_THREE_BIT_PREFIX);
  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));
  ASSERT_OK_AND_ASSIGN(auto loaded, internal::ReadGingerSearcher(index));
  const auto& orig_tq = *GingerSearcherAccess::quantized_dataset(*searcher);
  const auto& loaded_tq = *GingerSearcherAccess::quantized_dataset(*loaded);
  ASSERT_FALSE(orig_tq.has_3bit_prefix());
  EXPECT_FALSE(loaded_tq.has_3bit_prefix());
  for (size_t i = 0; i < orig_tq.num_datapoints(); ++i) {
    EXPECT_EQ(loaded_tq.norm_scaling_factor(i), orig_tq.norm_scaling_factor(i));
  }
}

TEST_F(GingerSearcherBuilderIOTest, AttachDatasetChecksFingerprint) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .AddIndex(MakeGraph())
                                          .AddReordering(reordering_)
                                          .Build());
  const std::string path = absl::StrCat(::testing::TempDir(), "/fingerprint");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/false));

  // The same points in their original order pass the check.
  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder same,
                       LoadGingerSearcherBuilder(path));
  ASSERT_OK(same.AddDataset(dataset_).Build());

  // Swapping two points is caught.
  auto shuffled = std::make_shared<SpanDataset>(kPoints, kDim);
  for (int i = 0; i < kPoints; ++i) {
    const int src = i < 2 ? 1 - i : i;
    shuffled->Set<float>(i, dataset_->get<float>(src));
  }
  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder swapped,
                       LoadGingerSearcherBuilder(path));
  EXPECT_THAT(swapped.AddDataset(shuffled).Build(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("does not match the dataset")));
}

TEST_F(GingerSearcherBuilderIOTest, LoadWithoutFingerprintSkipsCheck) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .AddIndex(MakeGraph())
                                          .Build());
  FakeIndex index;
  ASSERT_OK(
      internal::WriteGingerSearcher(index, *searcher, /*save_datasets=*/false));
  GingerSearcherHeader header;
  ASSERT_OK(index.ReadTextProto("ginger_searcher.header.pbtxt", header));
  ASSERT_TRUE(header.has_dataset_fingerprint());
  header.clear_dataset_fingerprint();
  ASSERT_OK(index.WriteTextProto("ginger_searcher.header.pbtxt", header));

  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder builder,
                       internal::ReadGingerSearcherBuilder(index));
  // Modify the attached copy so passing proves the check was skipped.
  mutable_copy_->get<float>(0)[0] += 1.0f;
  EXPECT_OK(builder.AddDataset(mutable_copy_).Build());
}

TEST_F(GingerSearcherBuilderIOTest,
       AttachDatasetChecksFingerprintForLargeDataset) {
  constexpr int kLargePoints = 2048;
  constexpr int kSmallDim = 4;
  auto large_dataset = std::make_shared<SpanDataset>(kLargePoints, kSmallDim);
  auto flat = std::make_unique<FlatGraph>(kLargePoints, 2);
  for (int i = 0; i < kLargePoints; ++i) {
    std::vector<float> pt(kSmallDim, static_cast<float>(i + 1));
    large_dataset->Set<float>(i, absl::MakeSpan(pt));
    flat->SetNeighbors(i, {(i + 1) % kLargePoints});
  }
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(large_dataset)
                                          .AddIndex(std::move(flat))
                                          .Build());
  const std::string path =
      absl::StrCat(::testing::TempDir(), "/fingerprint_large");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/false));

  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder same,
                       LoadGingerSearcherBuilder(path));
  EXPECT_OK(same.AddDataset(large_dataset).Build());

  // Modify only the second half (>= 1024 points) so a prefix-only fingerprint
  // would miss the mismatch.
  auto tail_modified = std::make_shared<SpanDataset>(kLargePoints, kSmallDim);
  for (int i = 0; i < kLargePoints; ++i) {
    const float value = i < kLargePoints / 2 ? static_cast<float>(i + 1)
                                             : -static_cast<float>(i + 1);
    std::vector<float> pt(kSmallDim, value);
    tail_modified->Set<float>(i, absl::MakeSpan(pt));
  }
  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder mismatched,
                       LoadGingerSearcherBuilder(path));
  EXPECT_THAT(mismatched.AddDataset(tail_modified).Build(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("does not match the dataset")));
}

TEST_F(GingerSearcherBuilderIOTest,
       AttachDatasetRejectsShapeAndCoordinateTypeMismatch) {
  auto make_zero_dataset = [](GingerNodeId size, size_t dim,
                              CoordinateType type) {
    auto ds = std::make_shared<SpanDataset>(size, dim, type);
    std::memset((*ds)[0], 0,
                static_cast<size_t>(size) * ds->point_size_in_bytes());
    return ds;
  };

  constexpr int kBasePoints = 1024;
  constexpr size_t kBaseDim = 16;
  auto base = make_zero_dataset(kBasePoints, kBaseDim, CoordinateType::kInt8);
  auto flat = std::make_unique<FlatGraph>(kBasePoints, 2);
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(base)
                                          .AddIndex(std::move(flat))
                                          .Build());
  const std::string path =
      absl::StrCat(::testing::TempDir(), "/fingerprint_shape");
  ASSERT_OK(SaveGingerSearcher(*searcher, path, /*save_datasets=*/false));

  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder same,
                       LoadGingerSearcherBuilder(path));
  EXPECT_OK(same.AddDataset(base).Build());

  auto expect_mismatch = [&](absl::string_view label,
                             const std::shared_ptr<const SpanDataset>& wrong) {
    SCOPED_TRACE(label);
    ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder builder,
                         LoadGingerSearcherBuilder(path));
    EXPECT_THAT(builder.AddDataset(wrong).Build(),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("does not match the dataset")));
  };

  // All candidates below have zero-initialized buffers so mismatches test the
  // shape header (`size`, `dimensionality`, `coordinate_type`) rather than
  // differing coordinate values, including identical sampled byte streams.
  expect_mismatch("wrong_size_same_sampled_bytes",
                  make_zero_dataset(kBasePoints + 1, kBaseDim,
                                    CoordinateType::kInt8));
  expect_mismatch("wrong_dim",
                  make_zero_dataset(kBasePoints, kBaseDim / 2,
                                    CoordinateType::kInt8));
  expect_mismatch("wrong_type_same_sampled_bytes",
                  make_zero_dataset(kBasePoints, kBaseDim,
                                    CoordinateType::kUint8));
  expect_mismatch("wrong_dim_and_type_same_point_bytes",
                  make_zero_dataset(kBasePoints, kBaseDim / sizeof(float),
                                    CoordinateType::kFloat));
}

TEST_F(GingerSearcherBuilderIOTest,
       PreservesFingerprintAcrossReloadWithoutRawDataset) {
  GingerSearcherConfig cfg;
  cfg.set_distance_metric(DOT_PRODUCT);
  cfg.mutable_turbo_quant()->set_seed(77);
  ASSERT_OK_AND_ASSIGN(auto searcher, GingerSearcherBuilder(cfg)
                                          .AddDataset(dataset_)
                                          .AddIndex(MakeGraph())
                                          .AddReordering(reordering_)
                                          .SetDropDataset(true)
                                          .Build());
  const std::string full_path =
      absl::StrCat(::testing::TempDir(), "/fingerprint_full_dropped");
  ASSERT_OK(SaveGingerSearcher(*searcher, full_path, /*save_datasets=*/true));
  ASSERT_OK_AND_ASSIGN(auto reloaded, LoadGingerSearcher(full_path));

  const std::string builder_path =
      absl::StrCat(::testing::TempDir(), "/fingerprint_resaved_no_datasets");
  ASSERT_OK(
      SaveGingerSearcher(*reloaded, builder_path, /*save_datasets=*/false));

  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder same,
                       LoadGingerSearcherBuilder(builder_path));
  EXPECT_OK(same.AddDataset(dataset_).Build());

  ASSERT_OK_AND_ASSIGN(GingerSearcherBuilder mismatched,
                       LoadGingerSearcherBuilder(builder_path));
  mutable_copy_->get<float>(0)[0] += 1.0f;
  EXPECT_THAT(mismatched.AddDataset(mutable_copy_).Build(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("does not match the dataset")));
}

}  // namespace
}  // namespace ginger::io
