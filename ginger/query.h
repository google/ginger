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

#ifndef GINGER_QUERY_H_
#define GINGER_QUERY_H_

#include <cstdint>
#include <functional>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/query/query.h"

namespace ginger {

// Non-owning view of a search query.
//
// It supports various query representations (e.g. float spans/vectors,
// int8/uint8 byte coordinates, and pre-quantized TurboQuant queries). The query
// representation must be compatible with the dataset's coordinate type and the
// searcher's configuration (e.g., int8 queries require an int8 dataset, and
// pre-quantized queries require TurboQuant search).
class SearchQuery {
 public:
  using Variant =
      std::variant<absl::Span<const float>, absl::Span<const int8_t>,
                   absl::Span<const uint8_t>,
                   std::reference_wrapper<const TurboQuantQuery>,
                   TurboQuantBasePointView>;

  // Constructors are intentionally non-explicit to allow transparent conversion
  // at FindNeighbors call sites.
  // NOLINTBEGIN(google-explicit-constructor)
  SearchQuery(absl::Span<float> query) : query_(query) {}
  SearchQuery(absl::Span<int8_t> query) : query_(query) {}
  SearchQuery(absl::Span<uint8_t> query) : query_(query) {}

  SearchQuery(absl::Span<const float> query) : query_(query) {}
  SearchQuery(absl::Span<const int8_t> query) : query_(query) {}
  SearchQuery(absl::Span<const uint8_t> query) : query_(query) {}

  SearchQuery(const std::vector<float>& query) : query_(query) {}
  SearchQuery(const std::vector<int8_t>& query) : query_(query) {}
  SearchQuery(const std::vector<uint8_t>& query) : query_(query) {}

  SearchQuery(const TurboQuantQuery& query) : query_(query) {}
  SearchQuery(TurboQuantBasePointView query) : query_(query) {}
  // NOLINTEND(google-explicit-constructor)

  // Delete rvalue constructors to prevent SearchQuery (a non-owning view) from
  // binding to temporary objects, which would create dangling references.
  SearchQuery(std::vector<float>&&) = delete;
  SearchQuery(std::vector<int8_t>&&) = delete;
  SearchQuery(std::vector<uint8_t>&&) = delete;
  SearchQuery(const std::vector<float>&&) = delete;
  SearchQuery(const std::vector<int8_t>&&) = delete;
  SearchQuery(const std::vector<uint8_t>&&) = delete;
  SearchQuery(TurboQuantQuery&&) = delete;
  SearchQuery(const TurboQuantQuery&&) = delete;

  const Variant& data() const { return query_; }

 private:
  Variant query_;
};

}  // namespace ginger

#endif  // GINGER_QUERY_H_
