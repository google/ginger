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

#include "ginger/reordering/recursive_graph_bisection_impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_reference_stub.h"
#include "parlay/parallel.h"

namespace ginger {
namespace internal {

void ComputeDegrees(const QueryDataGraph& graph,
                    absl::Span<const GingerNodeId> data_nodes,
                    absl::Span<GingerNodeId> degrees) {
  std::fill(degrees.begin(), degrees.end(), 0);

  for (const GingerNodeId data_node : data_nodes) {
    for (const GingerNodeId query_node : graph.AdjacentQueryNodes(data_node)) {
      degrees[query_node] += 1;
    }
  }
}

void UpdateDegrees(const QueryDataGraph& graph, const GingerNodeId data_node,
                   absl::Span<GingerNodeId> from_degrees,
                   absl::Span<GingerNodeId> to_degrees) {
  for (const GingerNodeId query_node : graph.AdjacentQueryNodes(data_node)) {
    from_degrees[query_node] -= 1;
    to_degrees[query_node] += 1;
  }
}

double ComputeQueryCosts(const QueryDataGraph& graph,
                         const Bisection& bisection,
                         std::array<std::vector<GingerNodeId>, 2>& degrees,
                         absl::Span<QueryCost> query_costs) {
  // Since computing log2 is a performance-critical part of the algorithm, we
  // use a lookup table with precomputed log2 values for small degrees and
  // otherwise fall back to a very fast log2 routine.
  constexpr GingerNodeId kLookupTableSize = 128;
  static const std::array<float, kLookupTableSize> kLookupTable = [] {
    std::array<float, kLookupTableSize> table{};
    table[0] = 0.0f;
    for (size_t i = 1; i < kLookupTableSize; ++i) {
      table[i] = std::log2f(i);
    }
    return table;
  }();
  const auto log2 = [&](const GingerNodeId x) -> float {
    return x < kLookupTableSize ? kLookupTable[x] : std::log2f(x);
  };

  const float log_n1 = log2(bisection.LeftSize());
  const float log_n2 = log2(bisection.RightSize());

  double partition_cost = 0.0f;

  // TODO: The compiler auto-vectorizes the following loop, however
  // we could look into improving it by manually writing SIMD code.
  for (GingerNodeId query_node = 0; query_node < graph.NumQueryNodes();
       ++query_node) {
    const GingerNodeId left_degree = degrees[0][query_node];
    const GingerNodeId right_degree = degrees[1][query_node];

    const float query_cost = left_degree * (log_n1 - log2(left_degree + 1)) +
                             right_degree * (log_n2 - log2(right_degree + 1));
    partition_cost += query_cost;

    query_costs[query_node] = {
        .cost = query_cost,
        .left_to_right_cost =
            (left_degree - 1) * (log_n1 - log2(left_degree)) +
            (right_degree + 1) * (log_n2 - log2(right_degree + 2)),
        .right_to_left_cost =
            (left_degree + 1) * (log_n1 - log2(left_degree + 2)) +
            (right_degree - 1) * (log_n2 - log2(right_degree)),
    };
  }

  return partition_cost;
}

void ComputeMoves(const QueryDataGraph& graph,
                  absl::Span<const GingerNodeId> data_nodes,
                  absl::Span<const QueryCost> query_costs,
                  absl::Span<Move> moves, bool left_to_right) {
  for (size_t i = 0; i < data_nodes.size(); ++i) {
    const GingerNodeId data_node = data_nodes[i];

    float gain = 0.0f;
    for (GingerNodeId query_node : graph.AdjacentQueryNodes(data_node)) {
      const QueryCost& query_cost = query_costs[query_node];
      const float move_cost = left_to_right ? query_cost.left_to_right_cost
                                            : query_cost.right_to_left_cost;
      gain += query_cost.cost - move_cost;
    }

    moves[i] = {data_node, gain};
  }
}

GingerNodeId PerformSwapsAndUpdateDegrees(
    const QueryDataGraph& graph, Bisection& bisection, const Moves& moves,
    std::array<std::vector<GingerNodeId>, 2>& degrees) {
  absl::Span<const Move> left_to_right_moves = moves.LeftToRightMoves();
  absl::Span<const Move> right_to_left_moves = moves.RightToLeftMoves();

  absl::Span<GingerNodeId> left_degrees = absl::MakeSpan(degrees[0]);
  absl::Span<GingerNodeId> right_degrees = absl::MakeSpan(degrees[1]);

  // Since moves are sorted by decreasing gain, we use a binary search to find
  // the first move pair with a non-positive total gain.
  const size_t max_num_swaps =
      std::min(moves.NumLeftMoves(), moves.NumRightMoves());
  const auto range = std::views::iota(size_t{0}, max_num_swaps);
  const auto it = std::ranges::partition_point(range, [&](size_t i) {
    return left_to_right_moves[i].gain + right_to_left_moves[i].gain > 0.0f;
  });
  const size_t num_swaps_to_perform = std::ranges::distance(range.begin(), it);

  for (size_t i = 0; i < num_swaps_to_perform; ++i) {
    const GingerNodeId left_data_node = left_to_right_moves[i].data_node;
    const GingerNodeId right_data_node = right_to_left_moves[i].data_node;
    bisection.Swap(left_data_node, right_data_node);

    UpdateDegrees(graph, left_data_node, left_degrees, right_degrees);
    UpdateDegrees(graph, right_data_node, right_degrees, left_degrees);
  }

  return num_swaps_to_perform;
}

void RefineBisection(const RecursiveGraphBisectionReorderingConfig& config,
                     const QueryDataGraph& graph, Bisection& bisection,
                     Moves& moves) {
  std::array<std::vector<GingerNodeId>, 2> degrees = {
      std::vector<GingerNodeId>(graph.NumQueryNodes()),
      std::vector<GingerNodeId>(graph.NumQueryNodes()),
  };
  std::vector<QueryCost> query_costs(graph.NumQueryNodes());

  ComputeDegrees(graph, bisection.LeftDataNodes(), absl::MakeSpan(degrees[0]));
  ComputeDegrees(graph, bisection.RightDataNodes(), absl::MakeSpan(degrees[1]));

  double prev_partition_cost = 0.0f;
  for (int i = 0; i < config.max_num_refinement_iterations(); ++i) {
    const double partition_cost = ComputeQueryCosts(
        graph, bisection, degrees, absl::MakeSpan(query_costs));

    // Abort the refinement if the relative improvement in the partition cost
    // compared to the previous iteration is less than the provided threshold.
    if (i > 0) {
      const double relative_improvement =
          (prev_partition_cost - partition_cost) / prev_partition_cost;
      if (relative_improvement <= config.min_refinement_improvement_factor()) {
        break;
      }
    }

    ComputeMoves(graph, bisection.LeftDataNodes(), query_costs,
                 moves.LeftToRightMoves(),
                 /*left_to_right=*/true);
    ComputeMoves(graph, bisection.RightDataNodes(), query_costs,
                 moves.RightToLeftMoves(),
                 /*left_to_right=*/false);
    moves.SortByGain();

    const GingerNodeId num_performed_swaps =
        PerformSwapsAndUpdateDegrees(graph, bisection, moves, degrees);

    // Abort the refinement if the relative number of nodes moved in this
    // iteration is less than the provided threshold.
    const double relative_num_nodes_moved =
        static_cast<float>(num_performed_swaps * 2) / graph.NumDataNodes();
    if (relative_num_nodes_moved <= config.min_refinement_swap_factor()) {
      break;
    }

    prev_partition_cost = partition_cost;
  }
}

GingerNodeId ComputeCompactQueryNodeIds(
    const QueryDataGraph& graph, absl::Span<const GingerNodeId> data_nodes,
    absl::Span<GingerNodeId> query_node_ids) {
  // We explicitly compute the compact query node IDs here instead of assigning
  // them on the fly while computing the subgraph data. This allows us to reuse
  // the same code for computing the subgraph data when the parallel subroutines
  // are added. We compute the compact query node IDs by writing a one to the
  // respective slot in the `query_node_ids` array for each query node present
  // in the subgraph, leaving all not used entries as zero. Then, we compute an
  // exclusive prefix sum, which results in `query_node_ids[old_id]` containing
  // the new, compact ID.

  std::fill(query_node_ids.begin(), query_node_ids.end(), 0);

  for (const GingerNodeId data_node : data_nodes) {
    for (const GingerNodeId query_node : graph.AdjacentQueryNodes(data_node)) {
      query_node_ids[query_node] = 1;
    }
  }

  GingerNodeId current_sum = 0;
  for (size_t i = 0; i < query_node_ids.size(); ++i) {
    GingerNodeId original_value = query_node_ids[i];
    query_node_ids[i] = current_sum;
    current_sum += original_value;
  }

  return current_sum;
}

void ComputeSubgraphData(const QueryDataGraph& graph,
                         absl::Span<const GingerNodeId> data_nodes_in_subgraph,
                         absl::Span<const GingerNodeId> new_query_node_ids,
                         absl::Span<int64_t> data_nodes,
                         absl::Span<GingerNodeId> data_edges,
                         absl::Span<GingerNodeId> mapping) {
  GingerNodeId cur_data_node = 0;
  int64_t cur_data_edge = 0;
  for (const GingerNodeId data_node : data_nodes_in_subgraph) {
    mapping[cur_data_node] = graph.OriginalId(data_node);
    data_nodes[cur_data_node] = cur_data_edge;
    cur_data_node++;

    for (const GingerNodeId query_node : graph.AdjacentQueryNodes(data_node)) {
      data_edges[cur_data_edge++] = new_query_node_ids[query_node];
    }
  }
}

SubgraphExtractionResult ExtractSubgraphs(QueryDataGraph& graph,
                                          QueryDataGraphBuffers& graph_buffers,
                                          const Bisection& bisection) {
  // The first step of subgraphs extraction is to compact the query node IDs.
  // This allows us to reduce the size of the allocated buffers used to compute
  // the gains and reduce cache misses.

  std::array<std::vector<GingerNodeId>, 2> query_node_ids = {
      std::vector<GingerNodeId>(graph.NumQueryNodes()),
      std::vector<GingerNodeId>(graph.NumQueryNodes()),
  };

  const GingerNodeId num_left_query_nodes = ComputeCompactQueryNodeIds(
      graph, bisection.LeftDataNodes(), absl::MakeSpan(query_node_ids[0]));
  const GingerNodeId num_right_query_nodes = ComputeCompactQueryNodeIds(
      graph, bisection.RightDataNodes(), absl::MakeSpan(query_node_ids[1]));

  // The next step of subgraph extraction is to write the edges into the CSR
  // format for the left and right subgraph. For this, we use the buffer memory.

  absl::Span<int64_t> offsets = graph_buffers.offsets;
  absl::Span<GingerNodeId> targets = graph_buffers.targets;
  absl::Span<GingerNodeId> mapping = graph_buffers.mapping;

  const GingerNodeId num_left_data_nodes = bisection.LeftSize();
  int64_t num_left_edges = 0;
  for (const GingerNodeId data_node : bisection.LeftDataNodes()) {
    num_left_edges += graph.Degree(data_node);
  }

  auto left_offsets = offsets.first(num_left_data_nodes);
  auto right_offsets = offsets.subspan(num_left_data_nodes);

  auto left_targets = targets.first(num_left_edges);
  auto right_targets = targets.subspan(num_left_edges);

  auto left_mapping = mapping.first(num_left_data_nodes);
  auto right_mapping = mapping.subspan(num_left_data_nodes);

  ComputeSubgraphData(graph, bisection.LeftDataNodes(), query_node_ids[0],
                      left_offsets, left_targets, left_mapping);
  ComputeSubgraphData(graph, bisection.RightDataNodes(), query_node_ids[1],
                      right_offsets, right_targets, right_mapping);

  QueryDataGraph left_subgraph(num_left_query_nodes, left_offsets, left_targets,
                               left_mapping);
  QueryDataGraph right_subgraph(num_right_query_nodes, right_offsets,
                                right_targets, right_mapping);

  // The memory used by the current graph becomes the buffer for the next two
  // subgraph extractions of the subproblems.

  absl::Span<int64_t> offsets_buffer = graph.Offsets();
  auto left_offsets_buffer = offsets_buffer.first(num_left_data_nodes);
  auto right_offsets_buffer = offsets_buffer.subspan(num_left_data_nodes);

  absl::Span<GingerNodeId> targets_buffer = graph.Targets();
  auto left_targets_buffer = targets_buffer.first(num_left_edges);
  auto right_targets_buffer = targets_buffer.subspan(num_left_edges);

  absl::Span<GingerNodeId> mapping_buffer = graph.Mapping();
  auto left_mapping_buffer = mapping_buffer.first(num_left_data_nodes);
  auto right_mapping_buffer = mapping_buffer.subspan(num_left_data_nodes);

  QueryDataGraphBuffers left_subgraph_buffers = {
      .offsets = left_offsets_buffer,
      .targets = left_targets_buffer,
      .mapping = left_mapping_buffer,
  };
  QueryDataGraphBuffers right_subgraph_buffers = {
      .offsets = right_offsets_buffer,
      .targets = right_targets_buffer,
      .mapping = right_mapping_buffer,
  };

  return SubgraphExtractionResult(left_subgraph, left_subgraph_buffers,
                                  right_subgraph, right_subgraph_buffers);
}

void ComputeReorderingRecursive(
    const RecursiveGraphBisectionReorderingConfig& config,
    absl::Span<GingerNodeId> reordering, GingerNodeId base_id,
    QueryDataGraph graph, QueryDataGraphBuffers graph_buffers,
    Bisection bisection, Moves moves) {
  if (graph.NumDataNodes() <= config.base_case_threshold()) {
    for (GingerNodeId u = 0; u < graph.NumDataNodes(); ++u) {
      reordering[graph.OriginalId(u)] = base_id + u;
    }
    return;
  }

  bisection.InitArbitrarily();
  RefineBisection(config, graph, bisection, moves);

  auto extraction_result = ExtractSubgraphs(graph, graph_buffers, bisection);
  auto [left_bisection, right_bisection] = bisection.Split();
  auto [left_moves, right_moves] = moves.Split();

  const GingerNodeId left_base_id = base_id;
  const GingerNodeId right_base_id =
      base_id + extraction_result.left_subgraph.NumDataNodes();

  parlay::parallel_do(
      [&] {
        ComputeReorderingRecursive(config, reordering, left_base_id,
                                   extraction_result.left_subgraph,
                                   extraction_result.left_subgraph_buffers,
                                   left_bisection, left_moves);
      },
      [&] {
        ComputeReorderingRecursive(config, reordering, right_base_id,
                                   extraction_result.right_subgraph,
                                   extraction_result.right_subgraph_buffers,
                                   right_bisection, right_moves);
      });
}

template <typename Graph>
void CopyGraph(const Graph& graph, absl::Span<int64_t> offsets,
               absl::Span<GingerNodeId> targets) {
  ABSL_CHECK_EQ(offsets.size(), graph.NumNodes());

  const GingerNodeId num_nodes = graph.NumNodes();
  for (GingerNodeId u = 0; u < num_nodes; ++u) {
    offsets[u] = graph.Degree(u);
  }
  std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(),
                      static_cast<int64_t>(0), std::plus<int64_t>());

  for (GingerNodeId u = 0; u < num_nodes; ++u) {
    int64_t offset = offsets[u];

    for (const GingerNodeId v : graph.Neighbors(u)) {
      targets[offset++] = v;
    }
  }
}

template void CopyGraph<CSRGraph>(const CSRGraph& graph,
                                  absl::Span<int64_t> offsets,
                                  absl::Span<GingerNodeId> targets);
template void CopyGraph<FlatGraph>(const FlatGraph& graph,
                                   absl::Span<int64_t> offsets,
                                   absl::Span<GingerNodeId> targets);

template <typename Graph>
std::vector<GingerNodeId> ComputeReordering(
    const Graph& graph, RecursiveGraphBisectionReorderingConfig config) {
  using QueryDataGraph = internal::QueryDataGraph;
  using Bisection = internal::Bisection;
  using Move = internal::Move;
  using Moves = internal::Moves;

  ABSL_CHECK_GE(config.base_case_threshold(), 1);

  const GingerNodeId num_nodes = graph.NumNodes();
  int64_t num_arcs = 0;
  for (GingerNodeId u = 0; u < num_nodes; ++u) {
    num_arcs += graph.Degree(u);
  }

  // To avoid repeated allocations during recursive calls, we preallocate two
  // sets of buffers for the data representing the query data graphs. In each
  // recursive step, one set holds the current graph's data, while the other is
  // prepared to hold the data for the next two subproblems, allowing us to
  // switch between them.
  std::vector<int64_t> offsets1(num_nodes);
  std::vector<int64_t> offsets2(num_nodes);

  std::vector<GingerNodeId> targets1(num_arcs);
  std::vector<GingerNodeId> targets2(num_arcs);

  std::vector<GingerNodeId> mapping1(num_nodes);
  std::vector<GingerNodeId> mapping2(num_nodes);

  // The initial query data graph is equivalent to the input graph. We compute
  // the CSR representation of the input graph and store it in the first set
  // of buffers.
  CopyGraph(graph, absl::MakeSpan(offsets1), absl::MakeSpan(targets1));

  // The `mapping` is used to store the original node ID of each data node in
  // the current subgraph. This is necessary to determine the original node ID
  // when writing the final reordering at the base case of the recursion.
  // Initially, each local ID is the same as the input ID, as the initial
  // QueryDataGraph represents the entire input graph.
  std::iota(mapping1.begin(), mapping1.end(), 0);

  QueryDataGraph query_data_graph(
      /*num_query_nodes=*/num_nodes, absl::MakeSpan(offsets1),
      absl::MakeSpan(targets1), absl::MakeSpan(mapping1));

  QueryDataGraphBuffers query_data_graph_buffers = {
      .offsets = absl::MakeSpan(offsets2),
      .targets = absl::MakeSpan(targets2),
      .mapping = absl::MakeSpan(mapping2),
  };

  // Similar to the graph data, we preallocate memory for the bisection data
  // and move data. These are split at each recursion level, and the respective
  // shares are passed to the subproblems.
  const size_t left_size = num_nodes / 2;
  const size_t right_size = num_nodes - left_size;

  std::vector<GingerNodeId> left_data_nodes(left_size);
  std::vector<GingerNodeId> right_data_nodes(right_size);
  std::vector<GingerNodeId> bisection_mapping(num_nodes);

  Bisection bisection(absl::MakeSpan(left_data_nodes),
                      absl::MakeSpan(right_data_nodes),
                      absl::MakeSpan(bisection_mapping));

  std::vector<Move> left_to_right_moves(left_size);
  std::vector<Move> right_to_left_moves(right_size);
  Moves moves(absl::MakeSpan(left_to_right_moves),
              absl::MakeSpan(right_to_left_moves));

  ginger::internal::ParallelSchedulerReference scheduler;
  std::vector<GingerNodeId> reordering(graph.NumNodes());
  ComputeReorderingRecursive(config, absl::MakeSpan(reordering),
                             /*base_id=*/0, query_data_graph,
                             query_data_graph_buffers, bisection, moves);
  return reordering;
}

template std::vector<GingerNodeId> ComputeReordering<CSRGraph>(
    const CSRGraph& graph, RecursiveGraphBisectionReorderingConfig config);
template std::vector<GingerNodeId> ComputeReordering<FlatGraph>(
    const FlatGraph& graph, RecursiveGraphBisectionReorderingConfig config);

}  // namespace internal
}  // namespace ginger
