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

#include "ginger/candidate_filtering/jl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ginger/dataset.h"
#include "ginger/distance/hamming.h"
#include "ginger/internal/portable_random.h"
#include "ginger/searcher.pb.h"
#include "ginger/types.h"
#include "hwy/aligned_allocator.h"
#include "parlay/parallel.h"

namespace ginger {
namespace {

constexpr int32_t kSquaredL2NonZeros = 6;
constexpr int32_t kDotProductNonZeros = 5;

absl::Status ValidateParams(int32_t source_dim, int32_t jl_dim, int32_t seed,
                            DistanceMetric metric) {
  if (source_dim <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("source_dim must be positive, got ", source_dim));
  }
  if (jl_dim <= 0 || (jl_dim % 64 != 0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "jl_dim must be positive and a multiple of 64, got ", jl_dim));
  }
  if (seed < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("seed must be non-negative, got ", seed));
  }
  if (metric != DOT_PRODUCT && metric != SQUARED_L2) {
    return absl::InvalidArgumentError(
        "metric must be DOT_PRODUCT or SQUARED_L2");
  }
  return absl::OkStatus();
}

struct SparseProjection {
  int32_t non_zeros;
  std::vector<int32_t> indices;
  std::vector<int8_t> signs;
};

// Generates a sparse random projection matrix for 1-bit JL quantization.
//
// Using sparse projections rather than dense matrices allows fast vector
// quantization using only a few additions and subtractions per output bit,
// without requiring floating-point multiplications.
//
// - SQUARED_L2: Samples 6 random coordinate indices per dimension: implicitly
//   treated as 3 positive (+1) signs and 3 negative (-1) signs in quantization.
// - DOT_PRODUCT: Samples 5 random coordinate indices per dimension with random
// signs (-1 or +1).
SparseProjection GenerateSparseProjection(int32_t source_dim, int32_t jl_dim,
                                          int32_t seed, DistanceMetric metric) {
  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  SparseProjection proj;

  if (metric == SQUARED_L2) {
    proj.non_zeros = kSquaredL2NonZeros;
    proj.indices.resize(static_cast<size_t>(jl_dim) * proj.non_zeros);
    proj.signs.clear();
    for (int32_t i = 0; i < jl_dim; ++i) {
      const size_t offset = static_cast<size_t>(i) * proj.non_zeros;
      for (int32_t j = 0; j < proj.non_zeros; ++j) {
        proj.indices[offset + j] = static_cast<int32_t>(
            internal::portable_random::UniformInt(rng, source_dim));
      }
    }
  } else {
    // DOT_PRODUCT
    proj.non_zeros = kDotProductNonZeros;
    proj.indices.resize(static_cast<size_t>(jl_dim) * proj.non_zeros);
    proj.signs.resize(static_cast<size_t>(jl_dim) * proj.non_zeros);
    for (int32_t i = 0; i < jl_dim; ++i) {
      const size_t offset = static_cast<size_t>(i) * proj.non_zeros;
      for (int32_t j = 0; j < proj.non_zeros; ++j) {
        proj.indices[offset + j] = static_cast<int32_t>(
            internal::portable_random::UniformInt(rng, source_dim));
        proj.signs[offset + j] =
            internal::portable_random::Bernoulli(rng, 0.5) ? 1 : -1;
      }
    }
  }
  return proj;
}

}  // namespace

JlDataset::JlDataset(int32_t source_dim, int32_t seed, size_t words_per_point,
                     DistanceMetric metric, std::vector<int32_t> indices,
                     std::vector<int8_t> signs, size_t num_points,
                     hwy::AlignedVector<uint64_t> data)
    : source_dim_(source_dim),
      seed_(seed),
      metric_(metric),
      words_per_point_(words_per_point),
      non_zeros_(metric == SQUARED_L2 ? kSquaredL2NonZeros
                                      : kDotProductNonZeros),
      indices_(std::move(indices)),
      signs_(std::move(signs)),
      num_points_(num_points),
      data_(std::move(data)) {}

absl::StatusOr<std::unique_ptr<JlDataset>> JlDataset::Create(
    int32_t source_dim, int32_t jl_dim, int32_t seed, DistanceMetric metric,
    size_t num_points, hwy::AlignedVector<uint64_t> data) {
  ABSL_RETURN_IF_ERROR(ValidateParams(source_dim, jl_dim, seed, metric));

  const size_t words_per_point = static_cast<size_t>(jl_dim) / 64;
  if (data.size() != num_points * words_per_point) {
    return absl::InvalidArgumentError(absl::StrCat("Expected data size ",
                                                   num_points * words_per_point,
                                                   ", got ", data.size()));
  }

  SparseProjection proj =
      GenerateSparseProjection(source_dim, jl_dim, seed, metric);

  return absl::WrapUnique(new JlDataset(
      source_dim, seed, words_per_point, metric, std::move(proj.indices),
      std::move(proj.signs), num_points, std::move(data)));
}

absl::StatusOr<std::unique_ptr<JlDataset>> JlDataset::Create(
    int32_t source_dim, int32_t jl_dim, int32_t seed, DistanceMetric metric,
    size_t num_points) {
  ABSL_RETURN_IF_ERROR(ValidateParams(source_dim, jl_dim, seed, metric));
  const size_t words_per_point = static_cast<size_t>(jl_dim) / 64;
  hwy::AlignedVector<uint64_t> data(num_points * words_per_point);
  return Create(source_dim, jl_dim, seed, metric, num_points, std::move(data));
}

absl::StatusOr<std::unique_ptr<JlDataset>> JlDataset::Create(
    int32_t jl_dim, int32_t seed, DistanceMetric metric,
    const SpanDataset& dataset) {
  if (dataset.coordinate_type() != CoordinateType::kFloat) {
    return absl::InvalidArgumentError(
        "JL dataset only supports float datasets.");
  }
  if (dataset.point_size_in_bytes() % sizeof(float) != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Dataset point size in bytes (", dataset.point_size_in_bytes(),
        ") is not a multiple of sizeof(float)"));
  }
  const int32_t source_dim = static_cast<int32_t>(dataset.dimensionality());

  ABSL_ASSIGN_OR_RETURN(
      auto jl_dataset,
      Create(source_dim, jl_dim, seed, metric, /*num_points=*/dataset.size()));

  const size_t num_points = dataset.size();
  const size_t words_per_point = jl_dataset->words_per_point();
  uint64_t* const data_ptr = jl_dataset->data_.data();

  parlay::parallel_for(0, num_points, [&](size_t i) {
    jl_dataset->QuantizePoint(
        dataset.get<float>(i),
        absl::MakeSpan(data_ptr + i * words_per_point, words_per_point));
  });

  return jl_dataset;
}

JlQuery JlDataset::QuantizeQuery(absl::Span<const float> query) const {
  ABSL_CHECK_EQ(query.size(), source_dim_);
  JlQuery query_obj(words_per_point_);
  QuantizePoint(query, query_obj.mutable_data());
  return query_obj;
}

void JlDataset::QuantizePoint(absl::Span<const float> point,
                              absl::Span<uint64_t> output) const {
  ABSL_CHECK_EQ(point.size(), source_dim_);
  ABSL_CHECK_EQ(output.size(), words_per_point_);

  const float* const pt_ptr = point.data();
  const int32_t* const idx_ptr = indices_.data();

  if (metric_ == SQUARED_L2) {
    for (size_t w = 0; w < words_per_point_; ++w) {
      uint64_t word_val = 0;
      const size_t dim_base = w * 64;
      for (size_t b = 0; b < 64; ++b) {
        // 6 non-zeros. First 3 are positive, last 3 are negative.
        const size_t offset = (dim_base + b) * non_zeros_;
        const float sum =
            (pt_ptr[idx_ptr[offset]] + pt_ptr[idx_ptr[offset + 1]] +
             pt_ptr[idx_ptr[offset + 2]]) -
            (pt_ptr[idx_ptr[offset + 3]] + pt_ptr[idx_ptr[offset + 4]] +
             pt_ptr[idx_ptr[offset + 5]]);
        if (sum > 0.0f) {
          word_val |= (uint64_t{1} << b);
        }
      }
      output[w] = word_val;
    }
  } else {
    // For DOT_PRODUCT, each projection row consists of 5 non-zero entries with
    // independently sampled random signs (+1 or -1).
    const int8_t* const sign_ptr = signs_.data();
    for (size_t w = 0; w < words_per_point_; ++w) {
      uint64_t word_val = 0;
      const size_t dim_base = w * 64;
      for (size_t b = 0; b < 64; ++b) {
        const size_t offset = (dim_base + b) * non_zeros_;
        float sum = 0.0f;
        for (int32_t nz = 0; nz < non_zeros_; ++nz) {
          sum += (sign_ptr[offset + nz] > 0 ? pt_ptr[idx_ptr[offset + nz]]
                                            : -pt_ptr[idx_ptr[offset + nz]]);
        }
        if (sum > 0.0f) {
          word_val |= (uint64_t{1} << b);
        }
      }
      output[w] = word_val;
    }
  }
}

void JlDataset::UpdatePoint(GingerNodeId id, absl::Span<const float> point) {
  ABSL_CHECK_LT(id, num_points_);
  QuantizePoint(point, absl::MakeSpan(data_).subspan(id * words_per_point_,
                                                     words_per_point_));
}

absl::Span<const uint64_t> JlDataset::GetPoint(GingerNodeId id) const {
  ABSL_CHECK_LT(id, num_points_);
  return absl::MakeConstSpan(data_).subspan(id * words_per_point_,
                                            words_per_point_);
}

void JlDataset::UnsafeComputeDistances(
    const JlQuery& query, absl::Span<DistanceAndId> candidates) const {
  HammingDistance(query.data(), candidates, data_.data(), words_per_point_);
}

void JlDataset::ReorderInPlace(absl::Span<const GingerNodeId> permutation) {
  ABSL_CHECK_EQ(permutation.size(), num_points_);
  if (num_points_ == 0) return;
  hwy::AlignedVector<uint64_t> reordered(num_points_ * words_per_point_);

  parlay::parallel_for(0, num_points_, [&](size_t i) {
    const GingerNodeId old_id = permutation[i];
    std::copy_n(data_.data() + old_id * words_per_point_, words_per_point_,
                reordered.data() + i * words_per_point_);
  });

  data_ = std::move(reordered);
}

std::unique_ptr<JlDataset> JlDataset::Reorder(
    absl::Span<const GingerNodeId> permutation) const {
  ABSL_CHECK_EQ(permutation.size(), num_points_);
  if (num_points_ == 0) {
    return absl::WrapUnique(new JlDataset(source_dim_, seed_, words_per_point_,
                                          metric_, indices_, signs_, 0,
                                          hwy::AlignedVector<uint64_t>()));
  }
  hwy::AlignedVector<uint64_t> reordered(num_points_ * words_per_point_);

  parlay::parallel_for(0, num_points_, [&](size_t i) {
    const GingerNodeId old_id = permutation[i];
    std::copy_n(data_.data() + old_id * words_per_point_, words_per_point_,
                reordered.data() + i * words_per_point_);
  });

  return absl::WrapUnique(new JlDataset(source_dim_, seed_, words_per_point_,
                                        metric_, indices_, signs_, num_points_,
                                        std::move(reordered)));
}

}  // namespace ginger
