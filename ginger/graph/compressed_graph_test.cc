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

#include "ginger/graph/compressed_graph.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ginger/graph/compressed_graph_builder.h"
#include "ginger/graph/compressed_offsets.h"
#include "ginger/graph/compression/streamvbyte_codec.h"
#include "ginger/graph/compression/varint_codec.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;

template <typename Config>
void ValidateGraphs(const CSRGraph& graph,
                    const CompressedCSRGraph<Config>& compressed_graph) {
  EXPECT_EQ(graph.NumNodes(), compressed_graph.NumNodes());
  EXPECT_EQ(graph.NumArcs(), compressed_graph.NumArcs());
  if (graph.NumArcs() == 0) {
    EXPECT_FLOAT_EQ(compressed_graph.BitsPerEdge(), 0.0f);
  } else {
    EXPECT_GT(compressed_graph.BitsPerEdge(), 0.0f);
  }

  for (GingerNodeId u = 0; u < graph.NumNodes(); ++u) {
    SCOPED_TRACE(absl::StrCat("Node ", u));
    EXPECT_EQ(graph.Degree(u), compressed_graph.Degree(u));
    EXPECT_THAT(compressed_graph.Neighbors(u),
                UnorderedElementsAreArray(graph.Neighbors(u)));
  }
}

CSRGraph CreateCSRGraph(
    const std::vector<std::vector<GingerNodeId>>& adjacency_list) {
  const size_t num_nodes = adjacency_list.size();
  size_t num_arcs = 0;
  for (const auto& neighbors : adjacency_list) {
    num_arcs += neighbors.size();
  }

  std::vector<int64_t> offsets = {0};
  std::vector<GingerNodeId> targets;
  targets.reserve(num_arcs);
  for (const auto& neighbors : adjacency_list) {
    for (const auto neighbor : neighbors) {
      EXPECT_LT(neighbor, num_nodes);
      targets.push_back(neighbor);
    }
    offsets.push_back(targets.size());
  }
  return CSRGraph(std::move(offsets), std::move(targets));
}

template <typename T>
class CompressedGraphTest : public ::testing::Test {};

struct VarintConfig : public CompressedGraphConfig {
  using VariableLengthEncoder = VarintEncoder;
  using VariableLengthDecoder = VarintDecoder;

  static constexpr bool kUseBatchGapDecoding = false;

  static constexpr bool kUseIntervalEncoding = false;
};

struct IntervalVarintConfig : public CompressedGraphConfig {
  using VariableLengthEncoder = VarintEncoder;
  using VariableLengthDecoder = VarintDecoder;

  static constexpr bool kUseBatchGapDecoding = false;

  static constexpr bool kUseIntervalEncoding = true;
};

struct StreamVByteConfig : public CompressedGraphConfig {
  using VariableLengthEncoder = VarintEncoder;
  using VariableLengthDecoder = VarintDecoder;

  static constexpr bool kUseBatchGapDecoding = true;
  using BatchGapEncoder = StreamVByteEncoder;
  using BatchGapDecoder = StreamVByteDecoder;

  static constexpr bool kUseIntervalEncoding = false;
};

struct IntervalStreamVByteConfig : public CompressedGraphConfig {
  using VariableLengthEncoder = VarintEncoder;
  using VariableLengthDecoder = VarintDecoder;

  static constexpr bool kUseBatchGapDecoding = true;
  using BatchGapEncoder = StreamVByteEncoder;
  using BatchGapDecoder = StreamVByteDecoder;

  static constexpr bool kUseIntervalEncoding = true;
};

struct IntervalAdaptiveConfig : public AdaptiveCompressedGraphConfig {
  static constexpr bool kUseIntervalEncoding = true;
};

// A view of compressed offsets must never silently bind to a temporary vector.
using OffsetView =
    decltype(std::declval<const AdaptiveCompressedGraph&>().Offsets());
static_assert(std::is_lvalue_reference_v<OffsetView>);
static_assert(!std::is_convertible_v<OffsetView, absl::Span<const int64_t>>);

using Configs =
    ::testing::Types<VarintConfig, IntervalVarintConfig, StreamVByteConfig,
                     IntervalStreamVByteConfig, AdaptiveCompressedGraphConfig,
                     IntervalAdaptiveConfig>;
TYPED_TEST_SUITE(CompressedGraphTest, Configs);

TYPED_TEST(CompressedGraphTest, ReusesOutputAcrossChangingDegrees) {
  CSRGraph graph = CreateCSRGraph({{1, 2, 3, 4}, {}, {1}, {0, 1, 2}, {3}});
  const CompressedCSRGraph<TypeParam> compressed =
      CompressGraph<CSRGraph, TypeParam>(graph);
  std::vector<GingerNodeId> output(17, 999);
  for (GingerNodeId u : {0, 1, 2, 3, 4, 0, 0, 1}) {
    compressed.FillNeighbors(u, output);
    EXPECT_THAT(output, ElementsAreArray(graph.Neighbors(u)));
  }
}

TYPED_TEST(CompressedGraphTest, EmptyGraph) {
  CSRGraph graph = CreateCSRGraph({});
  ValidateGraphs(graph, CompressGraph<CSRGraph, TypeParam>(graph));
}

TYPED_TEST(CompressedGraphTest, IsolatedNodes) {
  CSRGraph graph = CreateCSRGraph({{}, {}, {}, {}, {}});
  ValidateGraphs(graph, CompressGraph<CSRGraph, TypeParam>(graph));
}

TYPED_TEST(CompressedGraphTest, PublicConstructorAddsMissingPadding) {
  const CSRGraph graph = CreateCSRGraph({{1}, {0, 2}, {0}});
  const CompressedCSRGraph<TypeParam> original =
      CompressGraph<CSRGraph, TypeParam>(graph);
  const std::vector<int64_t> offsets = original.Offsets().ToVector();
  const absl::Span<const uint8_t> targets = original.CompressedTargets();
  constexpr size_t kSlop = CompressedCSRGraph<TypeParam>::kSlopBytes;
  for (size_t padding : {size_t{0}, kSlop - 1, kSlop, kSlop + 5}) {
    // An exact allocation makes decoder overreads detectable by ASan, and
    // non-zero trailing bytes verify canonical padding normalization.
    std::vector<uint8_t> bytes(offsets.back() + padding, 0xFF);
    std::copy_n(targets.begin(), offsets.back(), bytes.begin());
    const CompressedCSRGraph<TypeParam> restored(offsets, std::move(bytes),
                                                 graph.NumArcs());
    EXPECT_EQ(restored.CompressedTargets().size(), offsets.back() + kSlop);
    EXPECT_EQ(restored.Offsets().ToVector(), offsets);
    EXPECT_EQ(restored, original);
    ValidateGraphs(graph, restored);
  }
}

TYPED_TEST(CompressedGraphTest, MultipleOffsetBlocksAndLargeDegrees) {
  std::vector<std::vector<GingerNodeId>> adjacency(260);
  constexpr int kDegrees[] = {0, 1, 15, 16, 17, 63, 64, 65, 129};
  for (size_t u = 0; u < adjacency.size(); ++u) {
    for (int j = 0; j < kDegrees[u % std::size(kDegrees)]; ++j) {
      adjacency[u].push_back((2 * j + u) % adjacency.size());
    }
  }
  const CSRGraph graph = CreateCSRGraph(adjacency);
  const CompressedCSRGraph<TypeParam> compressed =
      CompressGraph<CSRGraph, TypeParam>(graph);
  EXPECT_EQ(compressed.Offsets().mode(), CompressedOffsets::Mode::kBlock16);
  ValidateGraphs(graph, compressed);
}

TYPED_TEST(CompressedGraphTest, PathGraph) {
  CSRGraph graph = CreateCSRGraph({{1}, {2}, {3}, {4}, {}});
  ValidateGraphs(graph, CompressGraph<CSRGraph, TypeParam>(graph));
}

TYPED_TEST(CompressedGraphTest, StarGraph) {
  CSRGraph graph = CreateCSRGraph({{1, 2, 3, 4}, {0}, {0}, {0}, {0}});
  ValidateGraphs(graph, CompressGraph<CSRGraph, TypeParam>(graph));
}

TYPED_TEST(CompressedGraphTest, CompleteGraph) {
  CSRGraph graph = CreateCSRGraph({{0, 1, 2, 3, 4},
                                   {0, 1, 2, 3, 4},
                                   {0, 1, 2, 3, 4},
                                   {0, 1, 2, 3, 4},
                                   {0, 1, 2, 3, 4}});
  ValidateGraphs(graph, CompressGraph<CSRGraph, TypeParam>(graph));
}

TYPED_TEST(CompressedGraphTest, GraphWithIntervals) {
  const GingerNodeId num_nodes = 106;
  std::vector<std::vector<GingerNodeId>> adjacency_list(num_nodes);
  adjacency_list[0] = {1, 2, 3, 4, 10, 15, 16, 17};
  adjacency_list[2] = {100, 101, 102, 103, 104, 105};
  adjacency_list[3] = {5, 6};

  CSRGraph graph = CreateCSRGraph(adjacency_list);
  ValidateGraphs(graph, CompressGraph<CSRGraph, TypeParam>(graph));
}

TYPED_TEST(CompressedGraphTest, GraphWithUnsortedNeighbors) {
  const GingerNodeId num_nodes = 7;
  std::vector<std::vector<GingerNodeId>> adjacency_list(num_nodes);
  adjacency_list[0] = {2, 5, 4, 1, 6, 3};
  adjacency_list[1] = {0, 3, 6, 2};
  adjacency_list[2] = {1, 4, 0, 3};
  adjacency_list[3] = {2, 0, 5, 4};
  adjacency_list[4] = {3, 6, 5, 0};
  adjacency_list[5] = {0, 4, 1, 6};
  adjacency_list[6] = {5, 2, 1, 0};

  CSRGraph graph = CreateCSRGraph(adjacency_list);
  ValidateGraphs(graph, CompressGraph<CSRGraph, TypeParam>(graph));
}

}  // namespace
}  // namespace ginger
