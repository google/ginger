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

#ifndef GINGER_REORDERING_REORDERING_UTILS_H_
#define GINGER_REORDERING_REORDERING_UTILS_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"

namespace ginger {

// Reorders the node IDs of a graph according to a permutation.
//
// This function constructs a new CSR graph that is structurally equivalent to
// the input graph, but with the node IDs arranged as specified by the
// permutation mapping. It maps each node index from its old index to
// `reordering[old_index]` and accordingly updates the neighbor IDs.
//
// Parameters:
//   - graph: The original graph to reorder.
//   - reordering: A span containing the target ID for each node.
//
// Preconditions:
//   - The reordering must be a valid permutation of size matching the number
//     of nodes in the graph.
//
// Returns:
//   - A new reordered CSR graph.
template <typename Graph>
CSRGraph ReorderGraph(const Graph& graph,
                      absl::Span<const GingerNodeId> reordering);

// Reorders the node IDs of a flat graph according to a permutation.
//
// This function constructs a new FlatGraph that is structurally equivalent to
// the input graph, but with the node IDs arranged as specified by the
// permutation mapping. It maps each node index from its old index to
// `reordering[old_index]` and accordingly updates the neighbor IDs.
//
// Parameters:
//   - graph: The original flat graph to reorder.
//   - reordering: A span containing the target ID for each node.
//
// Preconditions:
//   - The reordering must be a valid permutation of size matching the number
//     of nodes in the graph.
//
// Returns:
//   - A new reordered FlatGraph.
FlatGraph ReorderFlatGraph(const FlatGraph& graph,
                           absl::Span<const GingerNodeId> reordering);

// Reorders a dataset according to a permutation.
//
// This function constructs a new dataset containing a copy of the input dataset
// with point IDs arranged as specified by the permutation mapping. It maps
// each point index from its old index to `reordering[old_index]`.
//
// Parameters:
//   - dataset: The original dataset to reorder.
//   - reordering: A span containing the target ID for each point.
//
// Preconditions:
//   - The reordering must be a valid permutation of size matching the dataset
//     size.
//
// Returns:
//   - A shared pointer to the new reordered dataset.
std::shared_ptr<const SpanDataset> ReorderDataset(
    const SpanDataset& dataset, absl::Span<const GingerNodeId> reordering);

// Permutes a dataset in-place according to a permutation.
//
// This function reorders the elements of the input dataset in-place according
// to the permutation mapping. It maps each point index from its old index to
// `reordering[old_index]`. It decomposes the permutation into disjoint cycles
// and moves elements along each cycle to their target indices without
// allocating a duplicate dataset.
//
// Parameters:
//   - dataset: A reference to the dataset to reorder in-place.
//   - reordering: A span containing the target ID for each point.
//
// Preconditions:
//   - The reordering must be a valid permutation of size matching the dataset
//     size.
void ReorderDatasetInPlace(SpanDataset& dataset,
                           absl::Span<const GingerNodeId> reordering);

// Reorders a quantized dataset according to a permutation.
//
// This function constructs a new quantized dataset containing a copy of the
// input dataset with quantized point IDs arranged as specified by the
// permutation mapping. It maps each point index from its old index to
// `reordering[old_index]`.
//
// Parameters:
//   - dataset: The original quantized dataset to reorder.
//   - reordering: A span containing the target ID for each point.
//
// Preconditions:
//   - The reordering must be a valid permutation of size matching the dataset
//     size.
//
// Returns:
//   - A status or a shared pointer to the new reordered quantized dataset.
absl::StatusOr<std::shared_ptr<const TurboQuantDatasetSoA>>
ReorderQuantizedDataset(const TurboQuantDatasetSoA& dataset,
                        absl::Span<const GingerNodeId> reordering);

// Permutes a quantized dataset in-place according to a permutation.
//
// This function reorders the elements of the input quantized dataset in-place
// according to the permutation mapping. It maps each point index from its
// old index to `reordering[old_index]`.  It decomposes the permutation into
// disjoint cycles and moves elements along each cycle to their target indices
// without allocating a duplicate dataset.
//
// Parameters:
//   - dataset: A reference to the quantized dataset to reorder in-place.
//   - reordering: A span containing the target ID for each point.
//
// Preconditions:
//   - The reordering must be a valid permutation of size matching the dataset
//     size.
void ReorderQuantizedDatasetInPlace(TurboQuantDatasetSoA& dataset,
                                    absl::Span<const GingerNodeId> reordering);

}  // namespace ginger

#endif  // GINGER_REORDERING_REORDERING_UTILS_H_
