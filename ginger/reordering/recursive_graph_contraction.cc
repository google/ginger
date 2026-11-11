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

#include "ginger/reordering/recursive_graph_contraction.h"

#include <vector>

#include "ginger/graph/csr_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/reordering/config.pb.h"
#include "ginger/reordering/recursive_graph_contraction_impl.h"
#include "ginger/types.h"

namespace ginger {

template <typename Graph>
std::vector<GingerNodeId> ComputeRecursiveGraphContractionReordering(
    const Graph& graph, RecursiveGraphContractionReorderingConfig config) {
  return internal::ComputeRecursiveGraphContractionReordering(graph, config);
}

template std::vector<GingerNodeId>
ComputeRecursiveGraphContractionReordering<CSRGraph>(
    const CSRGraph& graph, RecursiveGraphContractionReorderingConfig config);
template std::vector<GingerNodeId>
ComputeRecursiveGraphContractionReordering<FlatGraph>(
    const FlatGraph& graph, RecursiveGraphContractionReorderingConfig config);

}  // namespace ginger
