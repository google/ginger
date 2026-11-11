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

#ifndef GINGER_REORDERING_RECURSIVE_GRAPH_PARTITIONING_IMPL_H_
#define GINGER_REORDERING_RECURSIVE_GRAPH_PARTITIONING_IMPL_H_

#include <cstddef>
#include <vector>

#include "absl/types/span.h"
#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"
#include "parlay/sequence.h"
#include "kaminpar.h"

namespace ginger {
namespace internal {

namespace rgp {
using NodeId = kaminpar::shm::NodeID;
using EdgeId = kaminpar::shm::EdgeID;
using PartitionId = kaminpar::shm::BlockID;

// We use our own CSRGraph implementation to ensure that the node and edge ID
// types match those used by KaMinPar, which allows us to avoid copying the
// graph data when interfacing with KaMinPar's API. Additionally, we store a
// mapping from the internal node IDs to the original node IDs. This allows us
// to write the computed reordering directly into a vector indexed by the
// original graph's node IDs.
//
// Note: Constructing a CSRGraph allocates uninitialized memory. Callers must
// initialize these data structures before accessing them.
class CSRGraph {
 public:
  CSRGraph(NodeId num_nodes, EdgeId num_arcs)
      : offsets_(parlay::sequence<EdgeId>::uninitialized(num_nodes + 1)),
        targets_(parlay::sequence<NodeId>::uninitialized(num_arcs)),
        mapping_(parlay::sequence<GingerNodeId>::uninitialized(num_nodes)) {}

  NodeId NumNodes() const { return offsets_.size() - 1; }
  EdgeId NumArcs() const { return targets_.size(); }

  NodeId Degree(NodeId node) const {
    return offsets_[node + 1] - offsets_[node];
  }

  absl::Span<const NodeId> Neighbors(NodeId node) const {
    return absl::MakeConstSpan(targets_).subspan(offsets_[node], Degree(node));
  }

  GingerNodeId OriginalId(NodeId node) const { return mapping_[node]; }

  void Clear() {
    offsets_ = parlay::sequence<EdgeId>();
    targets_ = parlay::sequence<NodeId>();
    mapping_ = parlay::sequence<GingerNodeId>();
  }

  absl::Span<EdgeId> Offsets() { return absl::MakeSpan(offsets_); }
  absl::Span<NodeId> Targets() { return absl::MakeSpan(targets_); }
  absl::Span<GingerNodeId> Mapping() { return absl::MakeSpan(mapping_); }

 private:
  parlay::sequence<EdgeId> offsets_;
  parlay::sequence<NodeId> targets_;
  parlay::sequence<GingerNodeId> mapping_;
};
}  // namespace rgp

// Partitions the given `graph` into `num_subgraphs` pieces using the settings
// provided in `config` and the given `parallelism`. Returns a vector where
// the u-th element is the partition ID assigned to node u.
std::vector<rgp::PartitionId> Partition(
    const RecursiveGraphPartitioningReorderingConfig& config,
    rgp::CSRGraph& graph, size_t num_subgraphs, int parallelism);

// Extracts the partition-induced subgraphs from the given `graph`.
std::vector<rgp::CSRGraph> ExtractSubgraphs(
    const RecursiveGraphPartitioningReorderingConfig& config,
    const rgp::CSRGraph& graph, absl::Span<const rgp::PartitionId> partition,
    size_t num_subgraphs);

// Extracts the partition-induced subgraphs from the given `graph` in parallel.
std::vector<rgp::CSRGraph> ExtractSubgraphsParallel(
    const RecursiveGraphPartitioningReorderingConfig& config,
    const rgp::CSRGraph& graph, absl::Span<const rgp::PartitionId> partition,
    size_t num_subgraphs, int parallelism);

// Recursively computes a reordering for the nodes of the given `graph`. The
// `reordering` span is populated with the new node IDs for the original IDs of
// the data nodes when a base case is reached in the recursion. The `base_id`
// parameter provides the current offset for assigning new IDs within the
// current recursive call's scope.
void ComputeReorderingRecursive(
    const RecursiveGraphPartitioningReorderingConfig& config,
    absl::Span<GingerNodeId> reordering, rgp::CSRGraph graph,
    GingerNodeId base_id = 0, size_t depth = 0);

// Copies the input `Graph` to our internal `CSRGraph` format. This process also
// ensures that the graph is undirected by adding any missing reverse edges, as
// the used graph partitioner requires an undirected graph.
template <typename Graph>
rgp::CSRGraph CopyGraph(const Graph& graph);

// Computes a reordering of the nodes in the provided `graph`, as specified by
// the `config`. It returns a vector `reordering` of length n where
// `reordering[u]` equals to the new node ID of node u.
template <typename Graph>
std::vector<GingerNodeId> ComputeReordering(
    const Graph& graph, RecursiveGraphPartitioningReorderingConfig config = {});

}  // namespace internal
}  // namespace ginger

#endif  // GINGER_REORDERING_RECURSIVE_GRAPH_PARTITIONING_IMPL_H_
