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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/internal/parallel_for_with_status.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace ginger {
namespace {


template <typename Graph>
int64_t NumArcs(const Graph& graph) {
  if constexpr (requires { graph.NumArcs(); }) {
    return graph.NumArcs();
  } else {
    return parlay::reduce(parlay::delayed_tabulate(
        graph.NumNodes(),
        [&](const GingerNodeId u) -> int64_t { return graph.Degree(u); }));
  }
}

}  // namespace

template <typename Graph>
CSRGraph ReorderGraph(const Graph& graph,
                      absl::Span<const GingerNodeId> reordering) {
  const GingerNodeId num_nodes = graph.NumNodes();
  const int64_t num_arcs = NumArcs(graph);

  std::vector<int64_t> offsets(num_nodes + 1, 0);
  parlay::parallel_for(0, num_nodes, [&](const GingerNodeId u) {
    offsets[reordering[u]] = graph.Degree(u);
  });
  parlay::scan_inplace(offsets);
  ABSL_DCHECK_EQ(offsets[0], 0);
  ABSL_DCHECK_EQ(offsets[num_nodes], num_arcs);

  std::vector<GingerNodeId> targets(num_arcs);
  parlay::parallel_for(0, num_nodes, [&](const GingerNodeId u) {
    int64_t offset = offsets[reordering[u]];
    for (const GingerNodeId v : graph.Neighbors(u)) {
      targets[offset++] = reordering[v];
    }
  });

  return CSRGraph(std::move(offsets), std::move(targets));
}

template CSRGraph ReorderGraph<CSRGraph>(
    const CSRGraph& graph, absl::Span<const GingerNodeId> reordering);
template CSRGraph ReorderGraph<FlatGraph>(
    const FlatGraph& graph, absl::Span<const GingerNodeId> reordering);

FlatGraph ReorderFlatGraph(const FlatGraph& graph,
                           absl::Span<const GingerNodeId> reordering) {
  const GingerNodeId num_nodes = graph.NumNodes();
  FlatGraph reordered_graph(num_nodes, graph.MaxDegree());

  thread_local std::vector<GingerNodeId> buffer;
  parlay::parallel_for(0, num_nodes, [&](const GingerNodeId u) {
    const GingerNodeId new_u = reordering[u];

    const absl::Span<const GingerNodeId> old_neighbors = graph.Neighbors(u);
    buffer.resize(old_neighbors.size());
    for (size_t i = 0; i < old_neighbors.size(); ++i) {
      buffer[i] = reordering[old_neighbors[i]];
    }

    reordered_graph.SetNeighbors(new_u, absl::MakeSpan(buffer));
  });

  return reordered_graph;
}

std::shared_ptr<const SpanDataset> ReorderDataset(
    const SpanDataset& dataset, absl::Span<const GingerNodeId> reordering) {
  SpanDataset reordered_dataset(dataset.size(), dataset.dimensionality(),
                                dataset.coordinate_type());

  parlay::parallel_for(0, dataset.size(), [&](const size_t i) {
    absl::Span<char> new_point = reordered_dataset.get_bytes(reordering[i]);
    absl::Span<const char> old_point = dataset.get_bytes(i);
    absl::c_copy(old_point, new_point.begin());
  });

  return std::make_shared<SpanDataset>(std::move(reordered_dataset));
}

void ReorderDatasetInPlace(SpanDataset& dataset,
                           absl::Span<const GingerNodeId> reordering) {
  const GingerNodeId num_points = dataset.size();
  const size_t point_size = dataset.point_size_in_bytes();

  std::vector<char> buffer1(point_size);
  std::vector<char> buffer2(point_size);
  std::vector<bool> visited(num_points, false);

  for (GingerNodeId i = 0; i < num_points; ++i) {
    if (reordering[i] == i || visited[i]) continue;

    GingerNodeId curr = i;
    GingerNodeId next = reordering[i];
    absl::c_copy(dataset.get_bytes(curr), buffer1.begin());

    // We traverse the permutation cycle starting at index `i` until it loops
    // back. At each step, we back up the original value of the destination
    // elements and place the values from the previous position in the cycle
    // path.
    while (next != i) {
      ABSL_QCHECK(!visited[next]);
      visited[next] = true;

      absl::c_copy(dataset.get_bytes(next), buffer2.begin());
      dataset.SetBytes(next, buffer1);

      curr = next;
      next = reordering[next];
      std::swap(buffer1, buffer2);
    }

    // Write back the original value of the final element to close the
    // permutation cycle.
    ABSL_QCHECK(!visited[i]);
    visited[i] = true;
    dataset.SetBytes(i, buffer1);
  }
}

absl::StatusOr<std::shared_ptr<const TurboQuantDatasetSoA>>
ReorderQuantizedDataset(const TurboQuantDatasetSoA& dataset,
                        absl::Span<const GingerNodeId> reordering) {
  TurboQuantDatasetSoA quantized_points =
      TurboQuantDatasetSoA::UninitializedLike(dataset);

  ABSL_RETURN_IF_ERROR(parallel_for_with_status(
      0, dataset.num_datapoints(), [&](const size_t i) -> absl::Status {
        return quantized_points.UpdatePoint(dataset.DataPointView(i),
                                            reordering[i]);
      }));

  return std::make_shared<TurboQuantDatasetSoA>(std::move(quantized_points));
}

void ReorderQuantizedDatasetInPlace(TurboQuantDatasetSoA& dataset,
                                    absl::Span<const GingerNodeId> reordering) {
  const GingerNodeId num_points = dataset.num_datapoints();
  std::vector<bool> visited(num_points, false);

  for (GingerNodeId i = 0; i < num_points; ++i) {
    if (reordering[i] == i || visited[i]) continue;

    GingerNodeId curr = i;
    GingerNodeId next = reordering[i];
    TurboQuantBasePoint point1(dataset.DataPointView(curr));

    // We traverse the permutation cycle starting at index `i` until it loops
    // back. At each step, we back up the original value of the destination
    // elements and place the values from the previous position in the cycle
    // path.
    while (next != i) {
      ABSL_QCHECK(!visited[next]);
      visited[next] = true;

      TurboQuantBasePoint point2(dataset.DataPointView(next));
      ABSL_CHECK_OK(dataset.UpdatePoint(point1.view(), next));

      curr = next;
      next = reordering[next];
      point1 = std::move(point2);
    }

    // Write back the original value of the final element to close the
    // permutation cycle.
    ABSL_QCHECK(!visited[i]);
    visited[i] = true;
    ABSL_CHECK_OK(dataset.UpdatePoint(point1.view(), i));
  }
}

}  // namespace ginger
