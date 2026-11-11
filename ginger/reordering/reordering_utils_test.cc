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

#include "ginger/reordering/reordering_utils.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/layout/norm_storage.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class ReorderingUtilsTest : public ::testing::Test {
 protected:
  using TurboQuantQuantizer = TurboQuantCodec;
  using TurboQuantDataset = TurboQuantDatasetSoA;
  CSRGraph CreateCSRGraph(
      const std::vector<std::vector<GingerNodeId>>& adjacency_list) {
    int64_t num_arcs = 0;
    for (const auto& neighbors : adjacency_list) {
      num_arcs += neighbors.size();
    }

    std::vector<int64_t> offsets = {0};
    std::vector<GingerNodeId> targets;
    targets.reserve(num_arcs);
    for (const auto& neighbors : adjacency_list) {
      for (const auto neighbor : neighbors) {
        targets.push_back(neighbor);
      }
      offsets.push_back(targets.size());
    }
    return CSRGraph(std::move(offsets), std::move(targets));
  }

  FlatGraph CreateFlatGraph(
      const std::vector<std::vector<GingerNodeId>>& adjacency_list) {
    GingerNodeId num_nodes = adjacency_list.size();
    GingerNodeId max_degree = 0;
    for (const auto& neighbors : adjacency_list) {
      max_degree = std::max<GingerNodeId>(max_degree, neighbors.size());
    }

    FlatGraph graph(num_nodes, max_degree);
    for (GingerNodeId u = 0; u < num_nodes; ++u) {
      graph.SetNeighbors(u, adjacency_list[u]);
    }
    return graph;
  }

  std::shared_ptr<SpanDataset> CreateDataset(
      const std::vector<std::vector<float>>& points) {
    if (points.empty()) {
      return std::make_shared<SpanDataset>(/*num_points=*/0,
                                           /*dimensionality=*/0);
    }

    const int32_t num_points = points.size();
    const int32_t dimensionality = points[0].size();

    SpanDataset dataset(num_points, dimensionality);
    for (int32_t i = 0; i < num_points; ++i) {
      dataset.Set<float>(i, points[i]);
    }

    return std::make_shared<SpanDataset>(std::move(dataset));
  }

  absl::StatusOr<
      std::pair<TurboQuantQuantizer, std::shared_ptr<TurboQuantDataset>>>
  CreateQuantizedDataset(const std::vector<std::vector<float>>& data) {
    const GingerNodeId num_points = data.size();
    const int32_t dimensionality = data.empty() ? 4 : data[0].size();

    TurboQuantQuantizer quantizer(dimensionality, /*seed=*/123,
                                  TurboQuantDistanceMetric::kSquaredL2Distance);

    ABSL_ASSIGN_OR_RETURN(
        auto dataset,
        TurboQuantDataset::Create(
            quantizer, num_points,
            TurboQuantDatasetOptions{.pad_to_cache_line_size = false}));

    for (GingerNodeId i = 0; i < num_points; ++i) {
      ABSL_RETURN_IF_ERROR(dataset.UpdatePoint(quantizer, data[i], i));
    }

    return std::make_pair(
        std::move(quantizer),
        std::make_shared<TurboQuantDataset>(std::move(dataset)));
  }

  std::vector<TurboQuantBasePoint> CopyPoints(
      const TurboQuantDataset& dataset) {
    const GingerNodeId num_points = dataset.num_datapoints();

    std::vector<TurboQuantBasePoint> original_points;
    original_points.reserve(num_points);

    for (GingerNodeId i = 0; i < num_points; ++i) {
      original_points.emplace_back(dataset.DataPointView(i));
    }

    return original_points;
  }

  void ExpectDataPointEqual(const TurboQuantBasePointView& a,
                            const TurboQuantBasePointView& b) {
    EXPECT_THAT(a.quantized_data, ElementsAreArray(b.quantized_data));
    EXPECT_THAT(a.extra_precision_data,
                ElementsAreArray(b.extra_precision_data));
    EXPECT_FLOAT_EQ(a.norm_scaling_factor_4bit, b.norm_scaling_factor_4bit);
    EXPECT_FLOAT_EQ(a.norm_scaling_factor_4plusk, b.norm_scaling_factor_4plusk);
    EXPECT_FLOAT_EQ(a.unquantized_squared_norm, b.unquantized_squared_norm);
    EXPECT_FLOAT_EQ(a.norm_scaling_factor_turboslice,
                    b.norm_scaling_factor_turboslice);
    EXPECT_EQ(a.query_byte_sum, b.query_byte_sum);
    EXPECT_EQ(a.has_3bit_prefix, b.has_3bit_prefix);
    EXPECT_EQ(a.extra_bits_per_dim, b.extra_bits_per_dim);
  }
};

// =============================================================================
// ReorderGraph Tests
// =============================================================================

TEST_F(ReorderingUtilsTest, ReorderGraphEmptyGraph) {
  std::vector<GingerNodeId> reordering = {};
  CSRGraph graph = CreateCSRGraph({});

  CSRGraph reordered = ReorderGraph(graph, reordering);
  EXPECT_EQ(reordered.NumNodes(), 0);
  EXPECT_EQ(reordered.NumArcs(), 0);
}

TEST_F(ReorderingUtilsTest, ReorderGraphSingleNode) {
  // We map node 0 -> 0.
  std::vector<GingerNodeId> reordering = {0};
  CSRGraph graph = CreateCSRGraph({{0}});

  CSRGraph reordered = ReorderGraph(graph, reordering);
  EXPECT_EQ(reordered.NumNodes(), 1);
  EXPECT_EQ(reordered.NumArcs(), 1);

  // Node 0 retains its self-loop neighbor 0.
  EXPECT_THAT(reordered.Neighbors(0), ElementsAre(0));
}

TEST_F(ReorderingUtilsTest, ReorderGraphMultipleNodes) {
  // We map node 0 -> 1, node 1 -> 2, and node 2 -> 0.
  std::vector<GingerNodeId> reordering = {1, 2, 0};
  CSRGraph graph = CreateCSRGraph({{1, 2}, {2}, {0}});

  CSRGraph reordered = ReorderGraph(graph, reordering);
  EXPECT_EQ(reordered.NumNodes(), 3);
  EXPECT_EQ(reordered.NumArcs(), 4);

  // Node 2 maps to node 0. Neighbors map to {0 -> 1}
  EXPECT_THAT(reordered.Neighbors(0), ElementsAre(1));
  // Node 0 maps to node 1. Neighbors map to {1 -> 2, 2 -> 0}
  EXPECT_THAT(reordered.Neighbors(1), ElementsAre(2, 0));
  // Node 1 maps to node 2. Neighbors map to {2 -> 0}
  EXPECT_THAT(reordered.Neighbors(2), ElementsAre(0));
}

TEST_F(ReorderingUtilsTest, ReorderGraphLarge) {
  constexpr GingerNodeId kNumNodes = 1001;

  // We create a graph where each node has two neighbors, the node with the next
  // ID and the node with the second next ID.
  std::vector<std::vector<GingerNodeId>> adjacency_list(kNumNodes);
  for (GingerNodeId i = 0; i < kNumNodes; ++i) {
    adjacency_list[i] = {(i + 1) % kNumNodes, (i + 2) % kNumNodes};
  }
  CSRGraph graph = CreateCSRGraph(adjacency_list);

  // We map node i -> (i + 13) % kNumNodes.
  std::vector<GingerNodeId> reordering(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i) {
    reordering[i] = (i + 13) % kNumNodes;
  }

  CSRGraph reordered = ReorderGraph(graph, reordering);
  EXPECT_EQ(reordered.NumNodes(), kNumNodes);
  EXPECT_EQ(reordered.NumArcs(), kNumNodes * 2);

  for (GingerNodeId i = 0; i < kNumNodes; ++i) {
    EXPECT_THAT(reordered.Neighbors(reordering[i]),
                ElementsAreArray({
                    reordering[(i + 1) % kNumNodes],
                    reordering[(i + 2) % kNumNodes],
                }));
  }
}

TEST_F(ReorderingUtilsTest, ReorderGraphFlatGraphMultipleNodes) {
  // We map node 0 -> 1, node 1 -> 2, and node 2 -> 0.
  std::vector<GingerNodeId> reordering = {1, 2, 0};
  FlatGraph graph = CreateFlatGraph({{1, 2}, {2}, {0}});

  CSRGraph reordered = ReorderGraph(graph, reordering);
  EXPECT_EQ(reordered.NumNodes(), 3);
  EXPECT_EQ(reordered.NumArcs(), 4);

  // Node 2 maps to node 0. Neighbors map to {0 -> 1}
  EXPECT_THAT(reordered.Neighbors(0), ElementsAre(1));
  // Node 0 maps to node 1. Neighbors map to {1 -> 2, 2 -> 0}
  EXPECT_THAT(reordered.Neighbors(1), ElementsAre(2, 0));
  // Node 1 maps to node 2. Neighbors map to {2 -> 0}
  EXPECT_THAT(reordered.Neighbors(2), ElementsAre(0));
}

TEST_F(ReorderingUtilsTest, ReorderFlatGraphMultipleNodes) {
  // We map node 0 -> 1, node 1 -> 2, and node 2 -> 0.
  std::vector<GingerNodeId> reordering = {1, 2, 0};
  FlatGraph graph = CreateFlatGraph({{1, 2}, {2}, {0}});

  FlatGraph reordered = ReorderFlatGraph(graph, reordering);
  EXPECT_EQ(reordered.NumNodes(), 3);
  EXPECT_EQ(reordered.MaxDegree(), 2);

  // Node 2 maps to node 0. Neighbors map to {0 -> 1}
  EXPECT_THAT(reordered.Neighbors(0), ElementsAre(1));
  // Node 0 maps to node 1. Neighbors map to {1 -> 2, 2 -> 0}
  EXPECT_THAT(reordered.Neighbors(1), ElementsAre(2, 0));
  // Node 1 maps to node 2. Neighbors map to {2 -> 0}
  EXPECT_THAT(reordered.Neighbors(2), ElementsAre(0));
}

// =============================================================================
// ReorderDataset Tests
// =============================================================================

TEST_F(ReorderingUtilsTest, ReorderDatasetEmpty) {
  // We reorder an empty dataset.
  std::vector<GingerNodeId> reordering = {};
  auto dataset = CreateDataset({});

  auto reordered = ReorderDataset(*dataset, reordering);
  EXPECT_EQ(reordered->size(), 0);
  EXPECT_EQ(reordered->point_size_in_bytes(), 0);
}

TEST_F(ReorderingUtilsTest, ReorderDatasetSinglePoint) {
  // We map point 0 -> 0.
  std::vector<GingerNodeId> reordering = {0};
  auto dataset = CreateDataset({{1.0f, 2.0f}});

  auto reordered = ReorderDataset(*dataset, reordering);
  EXPECT_EQ(reordered->size(), 1);
  EXPECT_EQ(reordered->point_size_in_bytes(), 2 * sizeof(float));

  // Point 0 maps to index 0. Expected value is {1.0f, 2.0f}.
  EXPECT_THAT(reordered->get<float>(0), ElementsAre(1.0f, 2.0f));
}

TEST_F(ReorderingUtilsTest, ReorderDatasetMultiplePoints) {
  // We map point 0 -> 1, point 1 -> 2, and point 2 -> 0.
  std::vector<GingerNodeId> reordering = {1, 2, 0};
  std::vector<std::vector<float>> data = {
      {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  auto dataset = CreateDataset(data);

  auto reordered = ReorderDataset(*dataset, reordering);
  EXPECT_EQ(reordered->size(), 3);
  EXPECT_EQ(reordered->point_size_in_bytes(), 2 * sizeof(float));

  // Point 2 maps to index 0. Expected value is {5.0f, 6.0f}.
  EXPECT_THAT(reordered->get<float>(0), ElementsAre(5.0f, 6.0f));
  // Point 0 maps to index 1. Expected value is {1.0f, 2.0f}.
  EXPECT_THAT(reordered->get<float>(1), ElementsAre(1.0f, 2.0f));
  // Point 1 maps to index 2. Expected value is {3.0f, 4.0f}.
  EXPECT_THAT(reordered->get<float>(2), ElementsAre(3.0f, 4.0f));
}

TEST_F(ReorderingUtilsTest, ReorderDatasetLarge) {
  constexpr GingerNodeId kNumPoints = 1001;

  // We create a dataset where each point consists of two values, i and i * 2.
  std::vector<std::vector<float>> data(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    data[i] = {static_cast<float>(i), static_cast<float>(i * 2)};
  }
  auto dataset = CreateDataset(data);

  // We map point i -> (i + 13) % kSize.
  std::vector<GingerNodeId> reordering(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    reordering[i] = (i + 13) % kNumPoints;
  }

  auto reordered = ReorderDataset(*dataset, reordering);
  EXPECT_EQ(reordered->size(), kNumPoints);
  EXPECT_EQ(reordered->point_size_in_bytes(), 2 * sizeof(float));

  for (GingerNodeId i = 0; i < kNumPoints; ++i) {
    EXPECT_THAT(reordered->get<float>(reordering[i]),
                ElementsAre(static_cast<float>(i), static_cast<float>(i * 2)));
  }
}

// =============================================================================
// ReorderDatasetInPlace Tests
// =============================================================================

TEST_F(ReorderingUtilsTest, ReorderDatasetInPlaceEmpty) {
  // We reorder an empty dataset.
  std::vector<GingerNodeId> reordering = {};
  auto dataset = CreateDataset({});

  ReorderDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->size(), 0);
  EXPECT_EQ(dataset->point_size_in_bytes(), 0);
}

TEST_F(ReorderingUtilsTest, ReorderDatasetInPlaceSinglePoint) {
  // We map point 0 -> 0.
  std::vector<GingerNodeId> reordering = {0};
  auto dataset = CreateDataset({{1.0f, 2.0f}});

  ReorderDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->size(), 1);
  EXPECT_EQ(dataset->point_size_in_bytes(), 2 * sizeof(float));

  // Point 0 maps to index 0. Expected value is {1.0f, 2.0f}.
  EXPECT_THAT(dataset->get<float>(0), ElementsAre(1.0f, 2.0f));
}

TEST_F(ReorderingUtilsTest, ReorderDatasetInPlaceMultiplePoints) {
  // We map point 0 -> 1, point 1 -> 2, and point 2 -> 0.
  std::vector<GingerNodeId> reordering = {1, 2, 0};
  std::vector<std::vector<float>> data = {
      {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  auto dataset = CreateDataset(data);

  ReorderDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->size(), 3);
  EXPECT_EQ(dataset->point_size_in_bytes(), 2 * sizeof(float));

  // Point 2 maps to index 0. Expected value is {5.0f, 6.0f}.
  EXPECT_THAT(dataset->get<float>(0), ElementsAre(5.0f, 6.0f));
  // Point 0 maps to index 1. Expected value is {1.0f, 2.0f}.
  EXPECT_THAT(dataset->get<float>(1), ElementsAre(1.0f, 2.0f));
  // Point 1 maps to index 2. Expected value is {3.0f, 4.0f}.
  EXPECT_THAT(dataset->get<float>(2), ElementsAre(3.0f, 4.0f));
}

TEST_F(ReorderingUtilsTest, ReorderDatasetInPlaceMultipleCycles) {
  // We have 5 points, permutation contains:
  // - cycle of length 2: (0, 1) -> 0 maps to 1, 1 maps to 0.
  // - cycle of length 3: (2, 3, 4) -> 2 maps to 3, 3 maps to 4, 4 maps to 2.
  std::vector<GingerNodeId> reordering = {1, 0, 3, 4, 2};
  std::vector<std::vector<float>> data = {
      {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}};
  auto dataset = CreateDataset(data);

  ReorderDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->size(), 5);
  EXPECT_EQ(dataset->point_size_in_bytes(), 2 * sizeof(float));

  // Point 1 maps to index 0. Expected value is {3.0f, 4.0f}.
  EXPECT_THAT(dataset->get<float>(0), ElementsAre(3.0f, 4.0f));
  // Point 0 maps to index 1. Expected value is {1.0f, 2.0f}.
  EXPECT_THAT(dataset->get<float>(1), ElementsAre(1.0f, 2.0f));
  // Point 4 maps to index 2. Expected value is {9.0f, 10.0f}.
  EXPECT_THAT(dataset->get<float>(2), ElementsAre(9.0f, 10.0f));
  // Point 2 maps to index 3. Expected value is {5.0f, 6.0f}.
  EXPECT_THAT(dataset->get<float>(3), ElementsAre(5.0f, 6.0f));
  // Point 3 maps to index 4. Expected value is {7.0f, 8.0f}.
  EXPECT_THAT(dataset->get<float>(4), ElementsAre(7.0f, 8.0f));
}

TEST_F(ReorderingUtilsTest, ReorderDatasetInPlaceLarge) {
  constexpr GingerNodeId kNumPoints = 1001;

  // We create a dataset where each point consists of two values, i and i * 2.
  std::vector<std::vector<float>> data(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    data[i] = {static_cast<float>(i), static_cast<float>(i * 2)};
  }
  auto dataset = CreateDataset(data);

  // We map point i -> (i + 13) % kSize.
  std::vector<GingerNodeId> reordering(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    reordering[i] = (i + 13) % kNumPoints;
  }

  ReorderDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->size(), kNumPoints);
  EXPECT_EQ(dataset->point_size_in_bytes(), 2 * sizeof(float));

  for (GingerNodeId i = 0; i < kNumPoints; ++i) {
    EXPECT_THAT(dataset->get<float>(reordering[i]),
                ElementsAre(static_cast<float>(i), static_cast<float>(i * 2)));
  }
}

// =============================================================================
// ReorderQuantizedDataset Tests
// =============================================================================

TEST_F(ReorderingUtilsTest, ReorderQuantizedDatasetEmpty) {
  std::vector<GingerNodeId> reordering = {};
  ASSERT_OK_AND_ASSIGN((auto [quantizer, dataset]), CreateQuantizedDataset({}));

  ASSERT_OK_AND_ASSIGN(auto reordered,
                       ReorderQuantizedDataset(*dataset, reordering));
  EXPECT_EQ(reordered->num_datapoints(), 0);
  EXPECT_EQ(reordered->dimensionality(), 4);
}

TEST_F(ReorderingUtilsTest, ReorderQuantizedDatasetSinglePoint) {
  std::vector<GingerNodeId> reordering = {0};
  std::vector<std::vector<float>> data = {{1.0f, 2.0f, 3.0f, 4.0f}};

  ASSERT_OK_AND_ASSIGN((auto [quantizer, dataset]),
                       CreateQuantizedDataset(data));
  auto original_points = CopyPoints(*dataset);

  ASSERT_OK_AND_ASSIGN(auto reordered,
                       ReorderQuantizedDataset(*dataset, reordering));
  EXPECT_EQ(reordered->num_datapoints(), 1);
  EXPECT_EQ(reordered->dimensionality(), 4);

  ExpectDataPointEqual(reordered->DataPointView(0), original_points[0].view());
}

TEST_F(ReorderingUtilsTest, ReorderQuantizedDatasetMultiplePoints) {
  // We map point 0 -> 1, point 1 -> 2, and point 2 -> 0.
  std::vector<GingerNodeId> reordering = {1, 2, 0};
  std::vector<std::vector<float>> data = {{0.5f, 0.5f, 0.5f, 0.5f},
                                          {1.0f, 0.0f, 0.0f, 0.0f},
                                          {0.0f, 1.0f, 0.0f, 0.0f}};
  ASSERT_OK_AND_ASSIGN((auto [quantizer, dataset]),
                       CreateQuantizedDataset(data));

  ASSERT_OK_AND_ASSIGN(auto reordered,
                       ReorderQuantizedDataset(*dataset, reordering));
  EXPECT_EQ(reordered->num_datapoints(), 3);
  EXPECT_EQ(reordered->dimensionality(), 4);

  // Point 2 maps to index 0. We verify it matches original point 2.
  ExpectDataPointEqual(reordered->DataPointView(0), dataset->DataPointView(2));
  // Point 0 maps to index 1. We verify it matches original point 0.
  ExpectDataPointEqual(reordered->DataPointView(1), dataset->DataPointView(0));
  // Point 1 maps to index 2. We verify it matches original point 1.
  ExpectDataPointEqual(reordered->DataPointView(2), dataset->DataPointView(1));
}

TEST_F(ReorderingUtilsTest, ReorderQuantizedDatasetLarge) {
  const GingerNodeId kNumPoints = 1001;

  // We create a dataset where each point consists of four values, i, i * 2, i *
  // 3, and i * 4.
  std::vector<std::vector<float>> data(kNumPoints);
  for (GingerNodeId i = 0; i < kNumPoints; ++i) {
    data[i] = {static_cast<float>(i) / kNumPoints,
               static_cast<float>(i * 2) / kNumPoints,
               static_cast<float>(i * 3) / kNumPoints,
               static_cast<float>(i * 4) / kNumPoints};
  }
  ASSERT_OK_AND_ASSIGN((auto [quantizer, dataset]),
                       CreateQuantizedDataset(data));
  auto original_points = CopyPoints(*dataset);

  // We map point i -> (i + 13) % kNumPoints.
  std::vector<GingerNodeId> reordering(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    reordering[i] = (i + 13) % kNumPoints;
  }

  ASSERT_OK_AND_ASSIGN(auto reordered,
                       ReorderQuantizedDataset(*dataset, reordering));
  EXPECT_EQ(reordered->num_datapoints(), kNumPoints);
  EXPECT_EQ(reordered->dimensionality(), 4);

  for (int i = 0; i < kNumPoints; ++i) {
    ExpectDataPointEqual(reordered->DataPointView(reordering[i]),
                         original_points[i].view());
  }
}

// =============================================================================
// ReorderQuantizedDatasetInPlace Tests
// =============================================================================

TEST_F(ReorderingUtilsTest, ReorderQuantizedDatasetInPlaceEmpty) {
  std::vector<GingerNodeId> reordering = {};
  ASSERT_OK_AND_ASSIGN((auto [quantizer, dataset]), CreateQuantizedDataset({}));

  ReorderQuantizedDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->num_datapoints(), 0);
  EXPECT_EQ(dataset->dimensionality(), 4);
}

TEST_F(ReorderingUtilsTest, ReorderQuantizedDatasetInPlaceSinglePoint) {
  std::vector<GingerNodeId> reordering = {0};
  std::vector<std::vector<float>> data = {{1.0f, 2.0f, 3.0f, 4.0f}};
  ASSERT_OK_AND_ASSIGN((auto [quantizer, dataset]),
                       CreateQuantizedDataset(data));
  auto original_points = CopyPoints(*dataset);

  ReorderQuantizedDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->num_datapoints(), 1);
  EXPECT_EQ(dataset->dimensionality(), 4);

  // Point 0 maps to index 0. We verify it matches copy of original point 0.
  ExpectDataPointEqual(dataset->DataPointView(0), original_points[0].view());
}

TEST_F(ReorderingUtilsTest, ReorderQuantizedDatasetInPlaceMultiplePoints) {
  // We map point 0 -> 1, point 1 -> 2, and point 2 -> 0.
  std::vector<GingerNodeId> reordering = {1, 2, 0};
  std::vector<std::vector<float>> data = {{0.5f, 0.5f, 0.5f, 0.5f},
                                          {1.0f, 0.0f, 0.0f, 0.0f},
                                          {0.0f, 1.0f, 0.0f, 0.0f}};
  ASSERT_OK_AND_ASSIGN((auto [quantizer, dataset]),
                       CreateQuantizedDataset(data));
  auto original_points = CopyPoints(*dataset);

  ReorderQuantizedDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->num_datapoints(), 3);
  EXPECT_EQ(dataset->dimensionality(), 4);

  // Point 2 maps to index 0. We verify it matches copy of original point 2.
  ExpectDataPointEqual(dataset->DataPointView(0), original_points[2].view());
  // Point 0 maps to index 1. We verify it matches copy of original point 0.
  ExpectDataPointEqual(dataset->DataPointView(1), original_points[0].view());
  // Point 1 maps to index 2. We verify it matches copy of original point 1.
  ExpectDataPointEqual(dataset->DataPointView(2), original_points[1].view());
}

TEST_F(ReorderingUtilsTest, ReorderQuantizedDatasetInPlaceMultipleCycles) {
  // We have 5 points, permutation contains:
  // - cycle of length 2: (0, 1) -> 0 maps to 1, 1 maps to 0.
  // - cycle of length 3: (2, 3, 4) -> 2 maps to 3, 3 maps to 4, 4 maps to 2.
  std::vector<GingerNodeId> reordering = {1, 0, 3, 4, 2};
  std::vector<std::vector<float>> data = {{0.1f, 0.1f, 0.1f, 0.1f},
                                          {0.2f, 0.2f, 0.2f, 0.2f},
                                          {0.3f, 0.3f, 0.3f, 0.3f},
                                          {0.4f, 0.4f, 0.4f, 0.4f},
                                          {0.5f, 0.5f, 0.5f, 0.5f}};
  ASSERT_OK_AND_ASSIGN((auto [quantizer, dataset]),
                       CreateQuantizedDataset(data));
  auto original_points = CopyPoints(*dataset);

  ReorderQuantizedDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->num_datapoints(), 5);
  EXPECT_EQ(dataset->dimensionality(), 4);

  // Point 1 maps to index 0. We verify it matches copy of original point 1.
  ExpectDataPointEqual(dataset->DataPointView(0), original_points[1].view());
  // Point 0 maps to index 1. We verify it matches copy of original point 0.
  ExpectDataPointEqual(dataset->DataPointView(1), original_points[0].view());
  // Point 4 maps to index 2. We verify it matches copy of original point 4.
  ExpectDataPointEqual(dataset->DataPointView(2), original_points[4].view());
  // Point 2 maps to index 3. We verify it matches copy of original point 2.
  ExpectDataPointEqual(dataset->DataPointView(3), original_points[2].view());
  // Point 3 maps to index 4. We verify it matches copy of original point 3.
  ExpectDataPointEqual(dataset->DataPointView(4), original_points[3].view());
}

TEST_F(ReorderingUtilsTest, ReorderQuantizedDatasetInPlaceLarge) {
  const GingerNodeId kNumPoints = 1001;

  // We create a dataset where each point consists of four values, i, i * 2, i *
  // 3, and i * 4.
  std::vector<std::vector<float>> data(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    data[i] = {static_cast<float>(i) / kNumPoints,
               static_cast<float>(i * 2) / kNumPoints,
               static_cast<float>(i * 3) / kNumPoints,
               static_cast<float>(i * 4) / kNumPoints};
  }

  ASSERT_OK_AND_ASSIGN((auto [quantizer, dataset]),
                       CreateQuantizedDataset(data));
  auto original_points = CopyPoints(*dataset);

  // We map point i -> (i + 13) % kNumPoints.
  std::vector<GingerNodeId> reordering(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    reordering[i] = (i + 13) % kNumPoints;
  }

  ReorderQuantizedDatasetInPlace(*dataset, reordering);
  EXPECT_EQ(dataset->num_datapoints(), kNumPoints);
  EXPECT_EQ(dataset->dimensionality(), 4);

  for (GingerNodeId i = 0; i < kNumPoints; ++i) {
    ExpectDataPointEqual(dataset->DataPointView(reordering[i]),
                         original_points[i].view());
  }
}

TEST_F(ReorderingUtilsTest, ReorderThreeBitPrefixQuantizedDataset) {
  constexpr GingerNodeId kNumPoints = 101;
  constexpr int kDim = 128;
  std::mt19937 rng(1);
  std::normal_distribution<float> dist;
  std::vector<std::vector<float>> data(kNumPoints, std::vector<float>(kDim));
  for (auto& point : data) {
    for (float& x : point) x = dist(rng);
  }
  std::vector<GingerNodeId> reordering(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    reordering[i] = (i + 13) % kNumPoints;
  }

  TurboQuantQuantizer quantizer(kDim, /*seed=*/123,
                                TurboQuantDistanceMetric::kDotProductDistance);
  for (const int quantization_bits : {4, 6}) {
    SCOPED_TRACE(quantization_bits);
    ASSERT_OK_AND_ASSIGN(
        TurboQuantDataset dataset,
        TurboQuantDataset::Create(
            quantizer, kNumPoints,
            TurboQuantDatasetOptions{
                .pad_to_cache_line_size = false,
                .quantization_bits = quantization_bits,
                .store_query_byte_sums = false,
                .normalized_datapoints = false,
                .scaling_storage_mode = ScalingStorageMode::kIn3BitPrefix}));
    for (GingerNodeId i = 0; i < kNumPoints; ++i) {
      ASSERT_OK(dataset.UpdatePoint(quantizer, data[i], i));
    }
    ASSERT_TRUE(dataset.has_3bit_prefix());
    auto original_points = CopyPoints(dataset);

    ASSERT_OK_AND_ASSIGN(auto reordered,
                         ReorderQuantizedDataset(dataset, reordering));
    ReorderQuantizedDatasetInPlace(dataset, reordering);
    EXPECT_TRUE(reordered->has_3bit_prefix());
    for (GingerNodeId i = 0; i < kNumPoints; ++i) {
      ExpectDataPointEqual(reordered->DataPointView(reordering[i]),
                           original_points[i].view());
      ExpectDataPointEqual(dataset.DataPointView(reordering[i]),
                           original_points[i].view());
    }
  }
}

}  // namespace
}  // namespace ginger
