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

#include "ginger/reordering/recursive_graph_partitioning_impl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"
#include "ginger/internal/scheduler_reference_stub.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"
#include "parlay/thread_specific.h"
#include "kaminpar.h"

namespace ginger {
namespace internal {

void RunBaseCase(const GingerNodeId base_id, const rgp::CSRGraph& graph,
                 absl::Span<GingerNodeId> reordering) {
  for (rgp::NodeId u = 0; u < graph.NumNodes(); ++u) {
    reordering[graph.OriginalId(u)] = base_id + u;
  }
}

std::vector<rgp::PartitionId> Partition(
    const RecursiveGraphPartitioningReorderingConfig& config,
    rgp::CSRGraph& graph, const size_t num_subgraphs, const int parallelism) {
  if (graph.NumNodes() == 0) {
    return {};
  }

  namespace terapart = kaminpar::shm;
  using TeraPart = kaminpar::KaMinPar;

  TeraPart instance(parallelism, terapart::create_fast_context());
  instance.context().node_ordering = terapart::NodeOrdering::NATURAL;
  instance.context().refinement.lp.num_iterations = 1;
  instance.set_output_level(kaminpar::OutputLevel::QUIET);

  instance.borrow_and_mutate_graph(graph.Offsets(), graph.Targets());

  std::vector<rgp::PartitionId> partition(graph.NumNodes());
  instance.compute_partition(num_subgraphs, config.imbalance_parameter(),
                             partition);

  return partition;
}

std::vector<rgp::CSRGraph> ExtractSubgraphs(
    const RecursiveGraphPartitioningReorderingConfig& config,
    const rgp::CSRGraph& graph, absl::Span<const rgp::PartitionId> partition,
    const size_t num_subgraphs) {
  std::vector<rgp::NodeId> new_node_ids(graph.NumNodes());

  std::vector<rgp::NodeId> num_nodes(num_subgraphs, 0);
  std::vector<rgp::EdgeId> num_edges(num_subgraphs, 0);
  for (rgp::NodeId u = 0; u < graph.NumNodes(); ++u) {
    const rgp::PartitionId u_partition = partition[u];
    new_node_ids[u] = num_nodes[u_partition]++;

    for (const rgp::NodeId v : graph.Neighbors(u)) {
      if (u_partition == partition[v]) {
        ++num_edges[u_partition];
      }
    }
  }

  std::vector<rgp::CSRGraph> subgraphs;
  subgraphs.reserve(num_subgraphs);
  for (size_t i = 0; i < num_subgraphs; ++i) {
    subgraphs.emplace_back(num_nodes[i], num_edges[i]);
  }

  std::vector<rgp::EdgeId> local_offsets(num_subgraphs, 0);
  for (rgp::NodeId u = 0; u < graph.NumNodes(); ++u) {
    const rgp::NodeId new_node_id = new_node_ids[u];
    const rgp::PartitionId u_partition = partition[u];

    auto mapping = subgraphs[u_partition].Mapping();
    mapping[new_node_id] = graph.OriginalId(u);

    auto offsets = subgraphs[u_partition].Offsets();
    auto targets = subgraphs[u_partition].Targets();

    rgp::EdgeId& offset = local_offsets[u_partition];
    offsets[new_node_id] = offset;

    for (const rgp::NodeId v : graph.Neighbors(u)) {
      if (u_partition == partition[v]) {
        targets[offset++] = new_node_ids[v];
      }
    }
  }

  for (size_t i = 0; i < num_subgraphs; ++i) {
    subgraphs[i].Offsets().back() = local_offsets[i];
  }

  return subgraphs;
}

std::vector<rgp::CSRGraph> ExtractSubgraphsParallel(
    const RecursiveGraphPartitioningReorderingConfig& config,
    const rgp::CSRGraph& graph, absl::Span<const rgp::PartitionId> partition,
    const size_t num_subgraphs, const int parallelism) {
  // The first step of the parallel extraction is to count the number of arcs in
  // each block-induced subgraph and store the new degree of each node.

  std::vector<rgp::NodeId> new_degrees(graph.NumNodes());
  parlay::ThreadSpecific<std::vector<rgp::EdgeId>> num_arcs_ts(
      [&] { return std::vector<rgp::EdgeId>(num_subgraphs, 0); });

  // Since accessing the thread specific has a non-negligible overhead, we set
  // the `block_size` such that the nodes are split roughly equally.
  const size_t block_size = std::max<size_t>(1, graph.NumNodes() / parallelism);
  parlay::blocked_for(
      0, graph.NumNodes(), block_size,
      [&](const size_t /*block_id*/, const size_t begin, const size_t end) {
        auto& s_m = num_arcs_ts.get();

        for (size_t u = begin; u < end; ++u) {
          const rgp::PartitionId u_partition = partition[u];

          rgp::NodeId new_degree = 0;
          for (const rgp::NodeId v : graph.Neighbors(u)) {
            if (u_partition == partition[v]) {
              ++new_degree;
            }
          }

          new_degrees[u] = new_degree;
          s_m[u_partition] += new_degree;
        }
      });

  std::vector<rgp::EdgeId> num_arcs(num_subgraphs);
  num_arcs_ts.for_each([&](const auto& s_m) {
    for (size_t i = 0; i < num_subgraphs; ++i) {
      num_arcs[i] += s_m[i];
    }
  });

  // The second step is to compute the new node IDs. For that, we group the
  // nodes by partition and store the new IDs based on the group indices inside
  // a vector.

  std::vector<rgp::NodeId> new_node_ids(graph.NumNodes());
  const auto nodes_by_partition = parlay::group_by_index(
      parlay::delayed_seq<std::pair<rgp::NodeId, rgp::NodeId>>(
          graph.NumNodes(),
          [&](const size_t u) { return std::make_pair(partition[u], u); }),
      num_subgraphs);
  parlay::parallel_for(0, num_subgraphs, [&](const size_t partition_id) {
    const auto& nodes_in_subgraph = nodes_by_partition[partition_id];
    parlay::parallel_for(0, nodes_in_subgraph.size(), [&](const size_t i) {
      new_node_ids[nodes_in_subgraph[i]] = i;
    });
  });

  // The third and final step is to allocate the subgraphs and store the
  // offsets into the target array, the target array itself, and a mapping
  // from new node ids to the original ones of the input graph.

  std::vector<rgp::CSRGraph> subgraphs;
  subgraphs.reserve(num_subgraphs);
  for (size_t i = 0; i < num_subgraphs; ++i) {
    subgraphs.emplace_back(nodes_by_partition[i].size(), num_arcs[i]);
  }

  parlay::parallel_for(0, graph.NumNodes(), [&](const size_t u) {
    const rgp::NodeId new_node_id = new_node_ids[u];
    const rgp::PartitionId u_partition = partition[u];

    auto offsets = subgraphs[u_partition].Offsets();
    offsets[new_node_id] = new_degrees[u];
  });
  parlay::parallel_for(0, subgraphs.size(), [&](const size_t i) {
    ABSL_CHECK_EQ(subgraphs[i].Offsets().size(), subgraphs[i].NumNodes() + 1);
    subgraphs[i].Offsets().back() = 0;

    parlay::scan_inplace(parlay::make_slice(subgraphs[i].Offsets()));
    ABSL_CHECK_EQ(subgraphs[i].Offsets().back(), num_arcs[i]);
  });

  parlay::parallel_for(0, graph.NumNodes(), [&](const size_t u) {
    const rgp::NodeId new_node_id = new_node_ids[u];
    const rgp::PartitionId u_partition = partition[u];

    auto mapping = subgraphs[u_partition].Mapping();
    mapping[new_node_id] = graph.OriginalId(u);

    auto offsets = subgraphs[u_partition].Offsets();
    auto targets = subgraphs[u_partition].Targets();

    rgp::EdgeId offset = offsets[new_node_id];
    for (const rgp::NodeId v : graph.Neighbors(u)) {
      if (u_partition == partition[v]) {
        targets[offset++] = new_node_ids[v];
      }
    }
  });

  return subgraphs;
}

void ComputeReorderingRecursive(
    const RecursiveGraphPartitioningReorderingConfig& config,
    absl::Span<GingerNodeId> reordering, rgp::CSRGraph graph,
    const GingerNodeId base_id, const size_t depth) {
  if (graph.NumNodes() <= config.base_case_threshold()) {
    RunBaseCase(base_id, graph, reordering);
    return;
  }

  // We use parallelism for partitioning and subgraph extraction on the first
  // levels of recursion, as long as the number of available threads is greater
  // than the number of subproblems. To compute the parallelism to use per
  // subproblem, we need to know how many subproblems are solved concurrently at
  // this depth.
  const size_t num_subproblems_at_cur_depth =
      depth == 0 ? 1
                 : config.top_level_branching_factor() *
                       std::pow(config.branching_factor(), depth - 1);

  // We compute the parallelism by giving each subproblem the same amount of
  // tasks on this recursion level and multiply it by some factor to account for
  // load imbalances.
  constexpr float load_balancing_factor = 4.0f;
  const size_t parallelism = std::max<size_t>(
      1, (parlay::num_workers() / num_subproblems_at_cur_depth) *
             load_balancing_factor);

  const size_t branching_factor = depth == 0
                                      ? config.top_level_branching_factor()
                                      : config.branching_factor();
  const size_t num_subgraphs =
      std::min<size_t>(graph.NumNodes(), branching_factor);

  auto subgraphs = [&]() {
    const auto partition = Partition(config, graph, num_subgraphs, parallelism);
    if (parallelism > 1) {
      return ExtractSubgraphsParallel(config, graph, partition, num_subgraphs,
                                      parallelism);
    } else {
      return ExtractSubgraphs(config, graph, partition, num_subgraphs);
    }
  }();

  std::vector<rgp::NodeId> base_offsets(subgraphs.size(), 0);
  for (size_t i = 1; i < subgraphs.size(); ++i) {
    base_offsets[i] = base_offsets[i - 1] + subgraphs[i - 1].NumNodes();
  }

  // Free the parent graph before launching subproblems to reduce peak memory.
  graph.Clear();

  parlay::parallel_for(
      0, subgraphs.size(),
      [&](const size_t i) {
        ComputeReorderingRecursive(config, reordering, std::move(subgraphs[i]),
                                   base_id + base_offsets[i], depth + 1);
      },
      /*granularity=*/1);
}

template <typename Graph>
rgp::CSRGraph CopyGraph(const Graph& graph) {
  const size_t num_nodes = graph.NumNodes();
  const size_t num_arcs = [&]() -> size_t {
    if constexpr (requires { graph.NumArcs(); }) {
      return graph.NumArcs();
    } else {
      return parlay::reduce(parlay::delayed_tabulate(
          num_nodes,
          [&](const size_t u) -> size_t { return graph.Degree(u); }));
    }
  }();

  parlay::sequence<rgp::NodeId> in_degrees(num_nodes, 0);
  parlay::parallel_for(0, num_nodes, [&](const size_t u) {
    for (const GingerNodeId v : graph.Neighbors(u)) {
      std::atomic_ref(in_degrees[v]).fetch_add(1, std::memory_order_relaxed);
    }
  });

  rgp::CSRGraph internal_graph(num_nodes, 2 * num_arcs);
  auto internal_offsets = internal_graph.Offsets();
  parlay::parallel_for(0, num_nodes, [&](const size_t u) {
    internal_offsets[u] = graph.Degree(u) + in_degrees[u];
    in_degrees[u] = 0;
  });
  internal_offsets.back() = 0;
  const rgp::EdgeId total_edges = parlay::scan_inplace(internal_offsets);
  ABSL_CHECK_EQ(total_edges, 2 * num_arcs);
  ABSL_CHECK_EQ(internal_offsets[num_nodes], total_edges);

  auto targets = internal_graph.Targets();
  parlay::parallel_for(0, num_nodes, [&](const size_t u) {
    rgp::EdgeId offset = internal_offsets[u];
    for (const GingerNodeId v : graph.Neighbors(u)) {
      targets[offset++] = v;
      const rgp::NodeId in_pos =
          std::atomic_ref(in_degrees[v])
              .fetch_add(1, std::memory_order_relaxed);
      targets[internal_offsets[v] + graph.Degree(v) + in_pos] = u;
    }
  });

  auto mapping = internal_graph.Mapping();
  parlay::parallel_for(0, num_nodes, [&](const size_t u) {
    mapping[u] = u;
    auto reverse_neighbors =
        targets.subspan(internal_offsets[u] + graph.Degree(u), in_degrees[u]);
    // NOMUTANTS -- Required for determinism when parallel_for scatters across
    // threads.
    std::sort(reverse_neighbors.begin(), reverse_neighbors.end());
  });

  return internal_graph;
}

template rgp::CSRGraph CopyGraph<CSRGraph>(const CSRGraph& graph);
template rgp::CSRGraph CopyGraph<FlatGraph>(const FlatGraph& graph);

template <typename Graph>
std::vector<GingerNodeId> ComputeReordering(
    const Graph& graph, RecursiveGraphPartitioningReorderingConfig config) {
  ABSL_CHECK_GE(config.base_case_threshold(), 1);
  ABSL_CHECK_GE(config.top_level_branching_factor(), 2);
  ABSL_CHECK_GE(config.branching_factor(), 2);
  ABSL_CHECK_GE(config.imbalance_parameter(), 0.0);
  ABSL_CHECK_LE(config.imbalance_parameter(), 1.0);
  ginger::internal::ParallelSchedulerReference scheduler;

  std::vector<GingerNodeId> reordering(graph.NumNodes());
  ComputeReorderingRecursive(config, absl::MakeSpan(reordering),
                             CopyGraph(graph));

  return reordering;
}

template std::vector<GingerNodeId> ComputeReordering<CSRGraph>(
    const CSRGraph& graph, RecursiveGraphPartitioningReorderingConfig config);
template std::vector<GingerNodeId> ComputeReordering<FlatGraph>(
    const FlatGraph& graph, RecursiveGraphPartitioningReorderingConfig config);

}  // namespace internal
}  // namespace ginger
