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

// A simple example demonstrating PiPNN directed k-NN graph construction.
//
// This program generates a synthetic float dataset, quantizes it into a 4-bit
// TurboQuant SoA dataset, builds a directed k-NN graph with exact float
// reranking via `BuildPipnnKnnGraph`, and prints the neighbors of node 0.
//
// Build and run:
//   bazel run -c opt //ginger/examples:pipnn_knn_example
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "absl/flags/parse.h"
#include "absl/log/absl_check.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/pipnn/config.pb.h"
#include "ginger/pipnn/pipnn_knn.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/types/distance_metric.h"
#include "ginger/types.h"
#include "google/protobuf/text_format.h"

namespace ginger {
namespace {

SpanDataset GenerateNormalizedDataset(int num_points, int dim, int seed = 42) {
  SpanDataset dataset(num_points, dim);
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> point(dim);
  for (int i = 0; i < num_points; ++i) {
    float norm = 0.0f;
    for (int d = 0; d < dim; ++d) {
      point[d] = dist(gen);
      norm += point[d] * point[d];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
      for (int d = 0; d < dim; ++d) point[d] /= norm;
    }
    dataset.Set<float>(i, absl::MakeConstSpan(point));
  }
  return dataset;
}

absl::Status Main() {
  constexpr int kNumPoints = 100'000;
  constexpr int kDimensionality = 128;
  constexpr int kNumNeighbors = 10;
  constexpr int kOverretrieveFactor = 2;

  // 1. Generate a random L2-normalized float dataset.
  std::cout << "Generating dataset with " << kNumPoints
            << " points of dimension " << kDimensionality << std::endl;
  SpanDataset float_dataset =
      GenerateNormalizedDataset(kNumPoints, kDimensionality);

  // 2. Quantize the dataset into 4-bit TurboQuant SoA layout.
  TurboQuantCodec quantizer(kDimensionality, /*seed=*/42,
                            TurboQuantDistanceMetric::kDotProductDistance);
  std::vector<absl::Span<const float>> point_spans;
  point_spans.reserve(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    point_spans.push_back(float_dataset.get<float>(i));
  }
  ABSL_ASSIGN_OR_RETURN(TurboQuantDatasetSoA quantized_points,
                        TurboQuantDatasetSoA::Create(quantizer, point_spans));

  // 3. Configure PiPNN recursive partitioning and build the k-NN graph.
  //
  // Only `fanout_sequence` must be set ([10, 3] is a good default); all other
  // fields keep their defaults from pipnn/config.proto. Knobs that trade
  // running time for recall (higher values boost recall but run slower):
  // - `num_passes` (default 1): number of independent partitioning passes;
  //   running time grows roughly linearly with it.
  // - `fanout_sequence`: number of nearest leaders each point is assigned to
  //   at each level; the product is roughly how many leaves a point lands in.
  // - `max_leaf_size` (default 1000): larger leaves compare more points with
  //   each other. Keep `min_leaf_size <= max_leaf_size / 2`.
  // - `overretrieve_factor`: number of 4-bit candidates per node, as a
  //   multiple of `k`, that are reranked with exact float distances. Also
  //   increases memory usage.
  PipnnConfig pipnn_config;
  ABSL_QCHECK(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        leader_selection { fanout_sequence: [ 10, 3 ] }
      )pb",
      &pipnn_config));

  std::vector<std::vector<DistanceAndId>> knn_graph(kNumPoints);
  std::cout << "Building PiPNN directed k-NN graph (k=" << kNumNeighbors
            << ", overretrieve_factor=" << kOverretrieveFactor << ")..."
            << std::endl;
  ABSL_RETURN_IF_ERROR(BuildPipnnKnnGraph(
      quantized_points, quantizer, pipnn_config, kNumNeighbors,
      kOverretrieveFactor,
      [&knn_graph](GingerNodeId u, std::vector<DistanceAndId> neighbors) {
        knn_graph[u] = std::move(neighbors);
      },
      &float_dataset));

  // 4. Print the top-k nearest neighbors of node 0.
  std::cout << "Top " << knn_graph[0].size()
            << " nearest neighbors of node 0:" << std::endl;
  for (const auto& [distance, id] : knn_graph[0]) {
    std::cout << "  id=" << id << "  distance=" << distance << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace ginger

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  ABSL_QCHECK_OK(ginger::Main());
  return 0;
}
