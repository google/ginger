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

// A simple example demonstrating Ginger's BYOD (Bring Your Own Distance) API.
//
// Instead of using GingerSearcherBuilder (which supports dot product and
// squared L2), this example uses the lower-level BuildVamanaIndex and
// NavigationBeamSearch APIs with a custom cosine distance function.
//
// Build and run:
//   bazel run -c opt //ginger/examples:ginger_byod_example
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include "absl/flags/parse.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "ginger/search/beam_search.h"
#include "ginger/types.h"
#include "ginger/vamana/config.pb.h"
#include "ginger/vamana/index_builder.h"
#include "google/protobuf/text_format.h"

namespace ginger {

namespace {

// Generates a dataset of `num_points` vectors, each of dimension `dim`,
// drawn from a standard normal distribution and normalized to unit length.
std::vector<std::vector<float>> GenerateDataset(int num_points, int dim,
                                                int seed = 42) {
  std::vector<std::vector<float>> dataset(num_points, std::vector<float>(dim));
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < num_points; ++i) {
    float norm = 0.0f;
    for (int d = 0; d < dim; ++d) {
      dataset[i][d] = dist(gen);
      norm += dataset[i][d] * dataset[i][d];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
      for (int d = 0; d < dim; ++d) dataset[i][d] /= norm;
    }
  }
  return dataset;
}

// Computes the cosine distance between two vectors. Since the dataset is
// normalized to unit length, cosine distance = 1 - dot product, which is
// non-negative (in [0, 2]).
float CosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
  float dot = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
  return 1.0f - dot;
}

}  // namespace

absl::Status Main() {
  constexpr int kNumPoints = 10'000;
  constexpr int kDimensionality = 128;
  constexpr int kBeamSize = 64;

  // 1. Generate a random normalized dataset.
  std::cout << "Generating dataset with " << kNumPoints
            << " points of dimension " << kDimensionality << std::endl;
  auto dataset = GenerateDataset(kNumPoints, kDimensionality);

  // 2. Define a custom distance function for index construction.
  //    The callback receives a query node ID and a span of candidates
  //    (with ids already set) and must fill in the .distance field.
  auto compute_distances = [&](GingerNodeId query,
                               absl::Span<DistanceAndId> candidates) {
    for (auto& candidate : candidates) {
      candidate.distance =
          CosineDistance(dataset[query], dataset[candidate.id]);
    }
  };

  // 3. Configure and build the Vamana index.
  //    Since cosine distance is non-negative, we must set
  //    nonpositive_distances = false. We use two passes: the first with
  //    alpha = 1.0 (more edges) and the second with alpha = 1.2 (prunes
  //    redundant edges for better navigability).
  VamanaConfig config;
  ABSL_QCHECK(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        max_degree: 64
        seed: 123
        passes {
          max_beam_width: 128
          prune_config { alpha: 1.0 nonpositive_distances: false }
        }
        passes {
          max_beam_width: 128
          prune_config { alpha: 1.2 nonpositive_distances: false }
        }
      )pb",
      &config));

  std::cout << "Building index with custom cosine distance..." << std::endl;
  ABSL_ASSIGN_OR_RETURN(
      auto index, BuildVamanaIndex(kNumPoints, config, compute_distances));
  std::cout << "Index built with " << index.NumNodes() << " nodes."
            << std::endl;

  // 4. Search using the same custom distance function.
  //    The query callback receives candidate spans and fills in distances.
  std::vector<float> query = dataset[0];  // Use the first point as query.
  auto search_distance = [&](absl::Span<DistanceAndId> candidates) {
    for (auto& candidate : candidates) {
      candidate.distance = CosineDistance(query, dataset[candidate.id]);
    }
    return candidates;
  };

  auto result =
      NavigationBeamSearch(index, search_distance, /*entry_points=*/{0},
                           /*max_beam_size=*/kBeamSize);

  // 5. Print results.
  std::cout << "Top " << result.final_beam.size()
            << " nearest neighbors (cosine distance):" << std::endl;
  for (const auto& [distance, id] : result.final_beam) {
    std::cout << "  id=" << id << "  distance=" << distance << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace ginger

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  ABSL_QCHECK_OK(ginger::Main());
  return 0;
}
