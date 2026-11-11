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

#ifndef GINGER_REORDERING_RECURSIVE_GRAPH_BISECTION_IMPL_H_
#define GINGER_REORDERING_RECURSIVE_GRAPH_BISECTION_IMPL_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ginger/reordering/config.pb.h"
#include "ginger/types.h"

namespace ginger {
namespace internal {

// Represents the bipartite graph of query and data nodes. It allows an
// arbitrary data node to obtain its adjacent query nodes. As it also is used
// to represent a subgraph of the original query data graph, it provides a
// mapping to the original IDs.
class QueryDataGraph {
 public:
  // This class does not hold the underlying memory for the graph data.
  // It only uses views of the provided data. The caller is responsible for
  // ensuring the validity and lifetime of the underlying data for the duration
  // of the QueryDataGraph's usage.
  QueryDataGraph(GingerNodeId num_query_nodes, absl::Span<int64_t> offsets,
                 absl::Span<GingerNodeId> targets,
                 absl::Span<GingerNodeId> mapping)
      : num_query_nodes_(num_query_nodes),
        offsets_(offsets),
        targets_(targets),
        mapping_(mapping) {
    ABSL_CHECK_EQ(offsets_.size(), mapping_.size());
  }

  GingerNodeId NumQueryNodes() const { return num_query_nodes_; }

  GingerNodeId NumDataNodes() const { return offsets_.size(); }

  int64_t NumEdges() const { return targets_.size(); }

  GingerNodeId Degree(GingerNodeId data_node) const {
    // We use a modified CSR representation for the query data graph where the
    // offsets array has only size `n` instead of `n + 1`. This simplifies the
    // subgraph extraction process. However, to determine the degree, we now
    // have to check if a node is the last node. If it is, the next offset is
    // the size of the edges array.
    const bool is_last_data_node = data_node + 1 == NumDataNodes();
    const int64_t next_offset =
        is_last_data_node ? targets_.size() : offsets_[data_node + 1];
    return next_offset - offsets_[data_node];
  }

  absl::Span<const GingerNodeId> AdjacentQueryNodes(
      GingerNodeId data_node) const {
    return targets_.subspan(offsets_[data_node], Degree(data_node));
  }

  GingerNodeId OriginalId(GingerNodeId data_node) const {
    return mapping_[data_node];
  }

  absl::Span<int64_t> Offsets() { return offsets_; }

  absl::Span<GingerNodeId> Targets() { return targets_; }

  absl::Span<GingerNodeId> Mapping() { return mapping_; }

 private:
  GingerNodeId num_query_nodes_;

  absl::Span<int64_t> offsets_;
  absl::Span<GingerNodeId> targets_;
  absl::Span<GingerNodeId> mapping_;
};

struct QueryDataGraphBuffers {
  absl::Span<int64_t> offsets;
  absl::Span<GingerNodeId> targets;
  absl::Span<GingerNodeId> mapping;
};

// Represents the bisection of the data nodes within a subproblem into a left
// and a right half.
class Bisection {
 public:
  // This class does not own the underlying memory for the bisection data.
  // It only uses views of the provided data. The caller is responsible for
  // ensuring the validity and lifetime of the underlying data for the duration
  // of the Bisection's usage.
  Bisection(absl::Span<GingerNodeId> left_data_nodes,
            absl::Span<GingerNodeId> right_data_nodes,
            absl::Span<GingerNodeId> mapping)
      : left_data_nodes_(left_data_nodes),
        right_data_nodes_(right_data_nodes),
        mapping_(mapping) {
    ABSL_CHECK_EQ(left_data_nodes_.size() + right_data_nodes_.size(),
                  mapping_.size());
  }

  GingerNodeId Size() const { return LeftSize() + RightSize(); }

  GingerNodeId LeftSize() const { return left_data_nodes_.size(); }

  GingerNodeId RightSize() const { return right_data_nodes_.size(); }

  absl::Span<const GingerNodeId> LeftDataNodes() const {
    return left_data_nodes_;
  }

  absl::Span<const GingerNodeId> RightDataNodes() const {
    return right_data_nodes_;
  }

  // Initializes the bisection by assigning all nodes with IDs in the range
  // `[0, LeftSize())` to the left bisection half and the remaining nodes,
  // i.e., with IDs `[LeftSize(), Size())`, to the right bisection half.
  void InitArbitrarily() {
    std::iota(left_data_nodes_.begin(), left_data_nodes_.end(), 0);
    std::iota(right_data_nodes_.begin(), right_data_nodes_.end(), LeftSize());

    std::iota(mapping_.begin(), mapping_.begin() + LeftSize(), 0);
    std::iota(mapping_.begin() + LeftSize(), mapping_.end(), 0);
  }

  // Swaps `left_data_node` from the left bisection half to the right half and
  // `right_data_node` from the right bisection half to the left half. The
  // caller is responsible for ensuring that `left_data_node` (resp.
  // `right_data_node`) is currently assigned to the left (resp. right)
  // bisection half.
  void Swap(GingerNodeId left_data_node, GingerNodeId right_data_node) {
    const GingerNodeId left_index = mapping_[left_data_node];
    const GingerNodeId right_index = mapping_[right_data_node];

    std::swap(left_data_nodes_[left_index], right_data_nodes_[right_index]);
    std::swap(mapping_[left_data_node], mapping_[right_data_node]);
  }

  // Constructs Bisection objects for the next two subproblems from this
  // Bisection object.
  std::pair<Bisection, Bisection> Split() {
    Bisection left_subproblem(left_data_nodes_.first(LeftSize() / 2),
                              left_data_nodes_.subspan(LeftSize() / 2),
                              mapping_.first(LeftSize()));

    Bisection right_subproblem(right_data_nodes_.first(RightSize() / 2),
                               right_data_nodes_.subspan(RightSize() / 2),
                               mapping_.subspan(LeftSize()));

    return std::make_pair(left_subproblem, right_subproblem);
  }

 private:
  absl::Span<GingerNodeId> left_data_nodes_;
  absl::Span<GingerNodeId> right_data_nodes_;
  absl::Span<GingerNodeId> mapping_;
};

// Represents the cost in the objective function of a query node.
// - `cost`:               The current contribution of this query node to the
//                         objective function.
// - `left_to_right_cost`: The contribution to the objective function if a
//                         single adjacent data node were moved from the left
//                         side of the bisection to the right side.
// - `right_to_left_cost`: The contribution to the objective function if a
//                         single adjacent data node were moved from the right
//                         side of the bisection to the left side.
struct QueryCost {
  float cost;
  float left_to_right_cost;
  float right_to_left_cost;
};

// Represents a potential move of a data node from one side of the bipartition
// to the other, along with the gain in the objective function achieved by this
// move.
struct Move {
  GingerNodeId data_node;
  float gain;
};

// Represents all potential moves of data nodes from the left to the right side
// of the bipartition and vice versa.
class Moves {
 public:
  // This class does not own the underlying memory for the moves data.
  // It only uses views of the provided data. The caller is responsible for
  // ensuring the validity and lifetime of the underlying data for the duration
  // of the Moves' usage.
  Moves(absl::Span<Move> left_to_right_moves,
        absl::Span<Move> right_to_left_moves)
      : left_to_right_moves_(left_to_right_moves),
        right_to_left_moves_(right_to_left_moves) {}

  size_t NumLeftMoves() const { return left_to_right_moves_.size(); }

  size_t NumRightMoves() const { return right_to_left_moves_.size(); }

  absl::Span<const Move> LeftToRightMoves() const {
    return left_to_right_moves_;
  }

  absl::Span<const Move> RightToLeftMoves() const {
    return right_to_left_moves_;
  }

  absl::Span<Move> LeftToRightMoves() { return left_to_right_moves_; }

  absl::Span<Move> RightToLeftMoves() { return right_to_left_moves_; }

  // Sorts the moves in descending order of gain.
  void SortByGain() {
    const auto compare_moves = [](const Move& a, const Move& b) {
      return (a.gain > b.gain) ||
             (a.gain == b.gain && a.data_node < b.data_node);
    };

    std::sort(left_to_right_moves_.begin(), left_to_right_moves_.end(),
              compare_moves);
    std::sort(right_to_left_moves_.begin(), right_to_left_moves_.end(),
              compare_moves);
  }

  // Constructs Moves objects for the next two subproblems from this Moves
  // object.
  std::pair<Moves, Moves> Split() {
    Moves left_subproblem(left_to_right_moves_.first(NumLeftMoves() / 2),
                          left_to_right_moves_.subspan(NumLeftMoves() / 2));

    Moves right_subproblem(right_to_left_moves_.first(NumRightMoves() / 2),
                           right_to_left_moves_.subspan(NumRightMoves() / 2));

    return std::make_pair(left_subproblem, right_subproblem);
  }

 private:
  absl::Span<Move> left_to_right_moves_;
  absl::Span<Move> right_to_left_moves_;
};

// The following methods are used to compute the gain of moving each data node
// from one side of the bisection to the other. The gain is defined as the
// decrease in the objective function that would result from moving the node.
//
// The objective function we seek to minimize is the sum of costs of all query
// nodes, where the cost of a query node q is given by
// deg_1(q) * log2(n_1 / (deg_1(q) + 1)) + deg_2(q) * log2(n_2 / (deg_2(q) + 1))
// Here, n_1 and n_2 are the sizes of the left and right bisection halves, and
// deg_1(q) and deg_2(q) are the number of q's neighbors in the left and right
// bisection half, respectively. This cost represents the number of bits
// required to encode q's neighborhood using gap encoding, assuming its
// neighbors are uniformly distributed at random within their respective
// bisection halves.
//
// To compute gains efficiently, we use precomputed query costs: `cost` is the
// current cost of a query node, `left_to_right_cost` is the cost if one of its
// neighbors moves from left to right, and `right_to_left_cost` is the cost if
// one of its neighbors moves from right to left.
//
// The gain of moving a data node u from left to right is thus the sum of
// `query_costs[q].cost - query_costs[q].left_to_right_cost` for all query
// nodes q adjacent to u. Similarly, moving u from right to left yields a
// gain of `query_costs[q].cost - query_costs[q].right_to_left_cost` for all
// adjacent query nodes q.

// Populates the `degrees` span such that `degrees[q]` contains the number of
// neighbors of query node `q` that are present in the `data_nodes` span. The
// caller is responsible for ensuring that the `degrees` span is large enough
// to accommodate all query nodes in the provided `graph`.
void ComputeDegrees(const QueryDataGraph& graph,
                    absl::Span<const GingerNodeId> data_nodes,
                    absl::Span<GingerNodeId> degrees);

// Updates the degrees of query nodes adjacent to `data_node`. This method is
// used when moving `data_node` from one side of a bipartition to the other.
// For each query node adjacent to `data_node`, its degree in `from_degrees`
// is decremented by one, and its degree in `to_degrees` is incremented by one.
// The caller is responsible for ensuring that both `from_degrees` and
// `to_degrees` spans are large enough to accommodate all query nodes in the
// provided `graph`.
void UpdateDegrees(const QueryDataGraph& graph, GingerNodeId data_node,
                   absl::Span<GingerNodeId> from_degrees,
                   absl::Span<GingerNodeId> to_degrees);

// Computes the cost for all query nodes in the `graph` based on the provided
// `bisection` and the degrees of each query node in the left and right
// partitions. The `degrees` array contains two vectors: `degrees[0]` stores
// the number of neighbors in the left partition, and `degrees[1]` stores the
// number of neighbors in the right partition for each query node. The computed
// costs are written to the `query_costs` span. The caller is responsible for
// ensuring that `query_costs` is large enough to accommodate all query nodes.
// Returns the total partition cost, which is the sum of all individual query
// costs.
double ComputeQueryCosts(const QueryDataGraph& graph,
                         const Bisection& bisection,
                         std::array<std::vector<GingerNodeId>, 2>& degrees,
                         absl::Span<QueryCost> query_costs);

// Computes potential moves for the provided `data_nodes` within the `graph`.
// Each move represents a data node moving from one side of the bisection to
// the other. The direction of the move (left to right or right to left) is
// determined by the `left_to_right` boolean. The computation uses the provided
// `query_costs`. The resulting `Move` objects, including their gains, are
// written to the `moves` span. The caller is responsible for ensuring that
// the `moves` span is large enough to store all computed moves.
void ComputeMoves(const QueryDataGraph& graph,
                  absl::Span<const GingerNodeId> data_nodes,
                  absl::Span<const QueryCost> query_costs,
                  absl::Span<Move> moves, bool left_to_right);

// Performs bisection swaps for data nodes based on the provided `moves`.
// It updates the `bisection` to reflect the new partition and adjusts the
// `degrees` array to account for the moved data nodes. Returns the total
// number of swaps executed.
GingerNodeId PerformSwapsAndUpdateDegrees(
    const QueryDataGraph& graph, Bisection& bisection, const Moves& moves,
    std::array<std::vector<GingerNodeId>, 2>& degrees);

// Refines the provided `bisection` by performing a local search to improve the
// objective function. The `moves` object is used to store potential move
// candidates and must be large enough to accommodate all data nodes in the
// `graph`.
void RefineBisection(const RecursiveGraphBisectionReorderingConfig& config,
                     const QueryDataGraph& graph, Bisection& bisection,
                     Moves& moves);

struct SubgraphExtractionResult {
  QueryDataGraph left_subgraph;
  QueryDataGraphBuffers left_subgraph_buffers;

  QueryDataGraph right_subgraph;
  QueryDataGraphBuffers right_subgraph_buffers;
};

// Computes new, compact query node IDs for the subgraph induced by the
// provided `data_nodes`. The new IDs are written to the `query_node_ids` span.
// The caller is responsible for ensuring that `query_node_ids` is large enough
// to accommodate all query nodes in the subgraph. Returns the total number of
// unique query nodes present in this subgraph.
GingerNodeId ComputeCompactQueryNodeIds(
    const QueryDataGraph& graph, absl::Span<const GingerNodeId> data_nodes,
    absl::Span<GingerNodeId> query_node_ids);

// Computes the offsets and targets for the subgraph induced by the given
// `data_nodes_in_subgraph`. The `new_query_node_ids` must provide a precomputed
// mapping from old query node IDs to new, compact query node IDs of the
// subgraph. The computed offsets are written to `data_nodes`, and the targets
// are written to `data_edges`. The caller is responsible for ensuring that
// `data_nodes` and `data_edges` are large enough. The `mapping` span is used
// to store a mapping from the new node IDs to the original node IDs of the
// input graph. The caller is responsible for ensuring that `mapping` is large
// enough to accommodate all nodes in the subgraph.
void ComputeSubgraphData(const QueryDataGraph& graph,
                         absl::Span<const GingerNodeId> data_nodes_in_subgraph,
                         absl::Span<const GingerNodeId> new_query_node_ids,
                         absl::Span<int64_t> data_nodes,
                         absl::Span<GingerNodeId> data_edges,
                         absl::Span<GingerNodeId> mapping);

// Extracts two subgraphs from the provided `graph`, where each subgraph is
// induced by one side of the `bisection`. The first element of the returned
// pair represents the subgraph for the left side of the bisection, and the
// second element represents the subgraph for the right side.
SubgraphExtractionResult ExtractSubgraphs(QueryDataGraph& graph,
                                          QueryDataGraphBuffers& graph_buffers,
                                          const Bisection& bisection);

// Recursively computes a reordering for the data nodes within the provided
// `graph`. The `reordering` span is populated with the new node IDs for the
// original IDs of the data nodes when a base case is reached in the recursion.
// The `base_id` parameter provides the current offset for assigning new IDs
// within the current recursive call's scope. The `bisection` and `moves`
// objects are passed, allowing their underlying `absl::Span`s to be reused
// for computations on subgraphs.
void ComputeReorderingRecursive(
    const RecursiveGraphBisectionReorderingConfig& config,
    absl::Span<GingerNodeId> reordering, GingerNodeId base_id,
    QueryDataGraph graph, QueryDataGraphBuffers graph_buffers,
    Bisection bisection, Moves moves);

// Copies a `Graph` to a custom CSR format, writing the offsets to the `offsets`
// span and the targets to the `targets` span. The caller is responsible for
// ensuring that both `offsets` and `targets` spans are large enough to
// accommodate the graph data.
template <typename Graph>
void CopyGraph(const Graph& graph, absl::Span<int64_t> offsets,
               absl::Span<GingerNodeId> targets);

// Computes a reordering of the nodes in the provided `graph`, as specified by
// the `config`. It returns a vector `reordering` of length n where
// `reordering[u]` equals to the new node ID of node u.
template <typename Graph>
std::vector<GingerNodeId> ComputeReordering(
    const Graph& graph, RecursiveGraphBisectionReorderingConfig config = {});

}  // namespace internal
}  // namespace ginger

#endif  // GINGER_REORDERING_RECURSIVE_GRAPH_BISECTION_IMPL_H_
