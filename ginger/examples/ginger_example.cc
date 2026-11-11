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

// A simple example demonstrating the Ginger API.
//
// This program builds a small Ginger index from a randomly generated dataset,
// performs a nearest neighbor search, and prints the results.
//
// Build and run:
//   bazel run -c opt //ginger/examples:ginger_example
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "absl/flags/parse.h"
#include "absl/log/absl_check.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "ginger/ginger.h"

namespace ginger {

namespace {

// Generates a dataset of `num_points` vectors, each of dimension `dim`.
// Vectors are drawn from a standard normal distribution and normalized to
// unit length.
std::shared_ptr<SpanDataset> GenerateDataset(int num_points, int dim,
                                             int seed = 42) {
  auto dataset = std::make_shared<SpanDataset>(num_points, dim);
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < num_points; ++i) {
    std::vector<float> point(dim);
    float norm = 0.0f;
    for (int d = 0; d < dim; ++d) {
      point[d] = dist(gen);
      norm += point[d] * point[d];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
      for (int d = 0; d < dim; ++d) point[d] /= norm;
    }
    dataset->Set<float>(i, absl::MakeConstSpan(point));
  }
  return dataset;
}

}  // namespace

absl::Status Main() {
  constexpr int kNumPoints = 10'000;
  constexpr int kDimensionality = 128;

  // 1. Generate a random dataset.
  std::cout << "Generating dataset with " << kNumPoints
            << " points of dimension " << kDimensionality << std::endl;
  auto dataset = GenerateDataset(kNumPoints, kDimensionality);

  // 2. Configure and build the searcher.
  ABSL_ASSIGN_OR_RETURN(GingerSearcherConfig config,
                        RecommendGingerConfig(kDimensionality, DOT_PRODUCT,
                                              /*robust_prune_alpha=*/1.08f));

  // Query vector 0 before Build() drops the full-precision dataset.
  std::vector<float> query(dataset->get<float>(0).begin(),
                           dataset->get<float>(0).end());

  std::cout << "Building index..." << std::endl;
  ABSL_ASSIGN_OR_RETURN(
      auto searcher, GingerSearcherBuilder(config).AddDataset(dataset).Build());
  std::cout << "Index built with " << searcher->size() << " points."
            << std::endl;

  // 3. Search for nearest neighbors of the first dataset point.
  SearchOptions options =
      searcher->DefaultSearchOptions(/*num_neighbors=*/10, /*beam_size=*/64);

  ABSL_ASSIGN_OR_RETURN(auto results, searcher->FindNeighbors(query, options));

  // 4. Print results.
  std::cout << "Top " << results.size() << " nearest neighbors:" << std::endl;
  for (const auto& [distance, id] : results) {
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
