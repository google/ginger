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

#ifndef GINGER_INTERNAL_SEARCHER_ACCESS_H_
#define GINGER_INTERNAL_SEARCHER_ACCESS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "ginger/candidate_filtering/jl.h"
#include "ginger/dataset.h"
#include "ginger/projection/pca.h"
#include "ginger/searcher.h"
#include "ginger/searcher_builder.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/layout/dataset_soa.h"

namespace ginger {

// Grants library internals (such as serialization) read access to the private
// components of a `GingerSearcher`.
struct GingerSearcherAccess {
  static const std::shared_ptr<const SpanDataset>& dataset(
      const GingerSearcher& s) {
    return s.dataset_;
  }
  static const std::shared_ptr<const TurboQuantDatasetSoA>& quantized_dataset(
      const GingerSearcher& s) {
    return s.quantized_dataset_;
  }
  static const std::shared_ptr<const TurboQuantCodec>& quantizer(
      const GingerSearcher& s) {
    return s.quantizer_;
  }
  static const std::shared_ptr<const JlDataset>& jl_dataset(
      const GingerSearcher& s) {
    return s.jl_dataset_;
  }
  static const PcaProjection* projection(const GingerSearcher& s) {
    return s.projection_.get();
  }
  static const std::shared_ptr<const TurboQuantDatasetSoA>&
  rerank_quantized_dataset(const GingerSearcher& s) {
    return s.rerank_quantized_dataset_;
  }
  static const std::shared_ptr<const TurboQuantCodec>& rerank_quantizer(
      const GingerSearcher& s) {
    return s.rerank_quantizer_;
  }
  static std::optional<uint32_t> dataset_fingerprint(const GingerSearcher& s) {
    return s.dataset_fingerprint_;
  }
};

struct GingerSearcherBuilderAccess {
  static void MarkAttachedDatasetsUnordered(GingerSearcherBuilder& b) {
    b.MarkAttachedDatasetsUnordered();
  }
  // Attaches a saved PCA rerank dataset, which is already in graph order.
  static void AddRerankQuantizedDataset(
      GingerSearcherBuilder& b,
      std::shared_ptr<const TurboQuantDatasetSoA> dataset,
      std::shared_ptr<const TurboQuantCodec> quantizer) {
    b.rerank_quantized_dataset_ = std::move(dataset);
    b.rerank_quantizer_ = std::move(quantizer);
  }
  static void SetDatasetFingerprint(GingerSearcherBuilder& b,
                                    uint32_t fingerprint) {
    b.expected_dataset_fingerprint_ = fingerprint;
  }
};

}  // namespace ginger

#endif  // GINGER_INTERNAL_SEARCHER_ACCESS_H_
