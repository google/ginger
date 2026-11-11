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

#ifndef GINGER_IO_INDEX_IO_H_
#define GINGER_IO_INDEX_IO_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/internal/array_io.h"
#include "ginger/searcher.h"
#include "ginger/searcher_builder.h"

// Entry layout for the `ginger::io` persistence backend.

namespace ginger::io::internal {

absl::Status WriteFlatGraph(array_io::Writer& writer, const FlatGraph& graph);
absl::StatusOr<FlatGraph> ReadFlatGraph(const array_io::Reader& reader);

template <typename Config>
absl::Status WriteCompressedGraph(array_io::Writer& writer,
                                  const CompressedCSRGraph<Config>& graph);
template <typename Config>
absl::StatusOr<CompressedCSRGraph<Config>> ReadCompressedGraph(
    const array_io::Reader& reader);

absl::Status WriteGingerSearcher(array_io::Writer& writer,
                                 const GingerSearcher& searcher,
                                 bool save_datasets);

// Returns a builder with the saved state attached. Datasets added by the caller
// afterwards will be permuted into the stored graph order during `Build()`.
absl::StatusOr<GingerSearcherBuilder> ReadGingerSearcherBuilder(
    const array_io::Reader& reader);

absl::StatusOr<std::unique_ptr<GingerSearcher>> ReadGingerSearcher(
    const array_io::Reader& reader);

}  // namespace ginger::io::internal

#endif  // GINGER_IO_INDEX_IO_H_
