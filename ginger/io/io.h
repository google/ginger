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

#ifndef GINGER_IO_IO_H_
#define GINGER_IO_IO_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/searcher.h"
#include "ginger/searcher_builder.h"

namespace ginger::io {

// Loading checks that a file is a Ginger index in a supported format version
// and that its entries agree with their headers: entry sizes match, flat graph
// neighbor IDs are valid nodes, compressed graph offsets are non-decreasing,
// and the inverse reordering is a permutation. This catches truncated and
// corrupted files. Compressed neighbor lists and dataset values are not
// checked, so only load files from trusted sources.

// Saves and loads a `FlatGraph` in a single file at `path`.
absl::Status SaveFlatGraph(const FlatGraph& graph, absl::string_view path);
absl::StatusOr<FlatGraph> LoadFlatGraph(absl::string_view path);

// Saves and loads a `CompressedCSRGraph` in a single file at `path`.
template <typename Config = CompressedGraphConfig>
absl::Status SaveCompressedGraph(const CompressedCSRGraph<Config>& graph,
                                 absl::string_view path);
template <typename Config = CompressedGraphConfig>
absl::StatusOr<CompressedCSRGraph<Config>> LoadCompressedGraph(
    absl::string_view path);

// Saves and loads a `GingerSearcher` in a single file at `path`. When
// `save_datasets` is false, the raw, TurboQuant, and JL datasets are omitted so
// callers can attach an external dataset via `LoadGingerSearcherBuilder`.
absl::Status SaveGingerSearcher(const GingerSearcher& searcher,
                                absl::string_view path);
absl::Status SaveGingerSearcher(const GingerSearcher& searcher,
                                absl::string_view path, bool save_datasets);

// Restores a searcher's config, entry point, inverse reordering, PCA
// projection, and index into a builder. The reordering algorithm is cleared so
// the loaded index is not reordered again. Any datasets saved in the file are
// attached directly without further permutation. Callers can attach omitted
// datasets before calling `Build()`. Attached datasets must contain the exact
// same points in the original pre-reordering ID order. `Build()` returns
// `absl::FailedPreconditionError` if an attached float dataset does not match
// the one the index was built from.
absl::StatusOr<GingerSearcherBuilder> LoadGingerSearcherBuilder(
    absl::string_view path);

absl::StatusOr<std::unique_ptr<GingerSearcher>> LoadGingerSearcher(
    absl::string_view path);

}  // namespace ginger::io

#endif  // GINGER_IO_IO_H_
