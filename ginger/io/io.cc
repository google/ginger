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

#include "ginger/io/io.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/flat_graph.h"
#include "ginger/internal/array_io.h"
#include "ginger/io/index_io.h"
#include "ginger/searcher.h"
#include "ginger/searcher_builder.h"

namespace ginger::io {

absl::Status SaveFlatGraph(const FlatGraph& graph, absl::string_view path) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<array_io::Writer> writer,
                        array_io::Writer::Create(path));
  ABSL_RETURN_IF_ERROR(internal::WriteFlatGraph(*writer, graph));
  return writer->Close();
}

absl::StatusOr<FlatGraph> LoadFlatGraph(absl::string_view path) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<array_io::Reader> reader,
                        array_io::Reader::Open(path));
  return internal::ReadFlatGraph(*reader);
}

template <typename Config>
absl::Status SaveCompressedGraph(const CompressedCSRGraph<Config>& graph,
                                 absl::string_view path) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<array_io::Writer> writer,
                        array_io::Writer::Create(path));
  ABSL_RETURN_IF_ERROR(internal::WriteCompressedGraph(*writer, graph));
  return writer->Close();
}

template <typename Config>
absl::StatusOr<CompressedCSRGraph<Config>> LoadCompressedGraph(
    absl::string_view path) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<array_io::Reader> reader,
                        array_io::Reader::Open(path));
  return internal::ReadCompressedGraph<Config>(*reader);
}

template absl::Status SaveCompressedGraph<CompressedGraphConfig>(
    const CompressedCSRGraph<CompressedGraphConfig>&, absl::string_view);
template absl::Status SaveCompressedGraph<AdaptiveCompressedGraphConfig>(
    const CompressedCSRGraph<AdaptiveCompressedGraphConfig>&,
    absl::string_view);
template absl::StatusOr<CompressedCSRGraph<CompressedGraphConfig>>
    LoadCompressedGraph<CompressedGraphConfig>(absl::string_view);
template absl::StatusOr<CompressedCSRGraph<AdaptiveCompressedGraphConfig>>
    LoadCompressedGraph<AdaptiveCompressedGraphConfig>(absl::string_view);

absl::Status SaveGingerSearcher(const GingerSearcher& searcher,
                                absl::string_view path) {
  return SaveGingerSearcher(searcher, path, /*save_datasets=*/true);
}

absl::Status SaveGingerSearcher(const GingerSearcher& searcher,
                                absl::string_view path, bool save_datasets) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<array_io::Writer> writer,
                        array_io::Writer::Create(path));
  ABSL_RETURN_IF_ERROR(
      internal::WriteGingerSearcher(*writer, searcher, save_datasets));
  return writer->Close();
}

absl::StatusOr<GingerSearcherBuilder> LoadGingerSearcherBuilder(
    absl::string_view path) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<array_io::Reader> reader,
                        array_io::Reader::Open(path));
  return internal::ReadGingerSearcherBuilder(*reader);
}

absl::StatusOr<std::unique_ptr<GingerSearcher>> LoadGingerSearcher(
    absl::string_view path) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<array_io::Reader> reader,
                        array_io::Reader::Open(path));
  return internal::ReadGingerSearcher(*reader);
}

}  // namespace ginger::io
