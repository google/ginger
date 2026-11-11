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

#ifndef GINGER_DATASET_H_
#define GINGER_DATASET_H_

#if __has_include(<sys/mman.h>)
#include <sys/mman.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {

constexpr size_t CoordinateSizeInBytes(CoordinateType coordinate_type) {
  switch (coordinate_type) {
    case CoordinateType::kFloat:
      return sizeof(float);
    case CoordinateType::kInt8:
      return sizeof(int8_t);
    case CoordinateType::kUint8:
      return sizeof(uint8_t);
  }
}

template <typename T>
constexpr CoordinateType CoordinateTypeFor() {
  if constexpr (std::is_same_v<T, float>) {
    return CoordinateType::kFloat;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return CoordinateType::kInt8;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return CoordinateType::kUint8;
  } else {
    static_assert(sizeof(T) == 0, "Unsupported coordinate type.");
  }
}

constexpr absl::string_view CoordinateTypeName(CoordinateType coordinate_type) {
  switch (coordinate_type) {
    case CoordinateType::kFloat:
      return "float";
    case CoordinateType::kInt8:
      return "int8";
    case CoordinateType::kUint8:
      return "uint8";
  }
}

template <typename T>
constexpr absl::string_view CoordinateTypeName() {
  return CoordinateTypeName(CoordinateTypeFor<T>());
}

// A dense dataset of `size()` points, each with `dimensionality()` coordinates
// of type `coordinate_type()`, stored back to back in one contiguous buffer.
// The coordinate type is a runtime value rather than a template parameter to
// avoid unnecessary templatization.
//
// The constructor allocates owned, huge-page-aligned memory that the caller
// fills in with `Set` or `SetBytes`. `Wrap` and `WrapMutable` instead create a
// zero-copy view over memory that the caller owns.
class SpanDataset {
 public:
  // Creates a read-only view over `data` without copying it. `data` holds the
  // points back to back, `dimensionality` coordinates each. The element type
  // of `data` determines the coordinate type.
  //
  // The caller keeps ownership of `data` and must keep it alive as long as the
  // returned dataset is in use. A `GingerSearcher` built from the dataset holds
  // a reference to it, so `data` must also outlive the searcher.
  //
  // Returns InvalidArgumentError if `data` is non-empty but has a null data
  // pointer, if `dimensionality` is 0 or too large for the point size to fit in
  // `size_t`, if `data.size()` is not a multiple of `dimensionality`, or if the
  // number of points does not fit in `GingerNodeId`.
  static absl::StatusOr<std::shared_ptr<const SpanDataset>> Wrap(
      absl::Span<const float> data, size_t dimensionality) {
    return WrapImpl(data.data(), data.size(), dimensionality,
                    CoordinateType::kFloat);
  }
  static absl::StatusOr<std::shared_ptr<const SpanDataset>> Wrap(
      absl::Span<const int8_t> data, size_t dimensionality) {
    return WrapImpl(data.data(), data.size(), dimensionality,
                    CoordinateType::kInt8);
  }
  static absl::StatusOr<std::shared_ptr<const SpanDataset>> Wrap(
      absl::Span<const uint8_t> data, size_t dimensionality) {
    return WrapImpl(data.data(), data.size(), dimensionality,
                    CoordinateType::kUint8);
  }

  // Like `Wrap`, but the view is writable: `Set`, `SetBytes` and the mutable
  // accessors write directly into `data`. Note that
  // `GingerSearcherBuilder::AddMutableDataset` permutes the points in `data` in
  // place if reordering is enabled and `SetReorderInPlace(true)` was called.
  static absl::StatusOr<std::shared_ptr<SpanDataset>> WrapMutable(
      absl::Span<float> data, size_t dimensionality) {
    return WrapImpl(data.data(), data.size(), dimensionality,
                    CoordinateType::kFloat);
  }
  static absl::StatusOr<std::shared_ptr<SpanDataset>> WrapMutable(
      absl::Span<int8_t> data, size_t dimensionality) {
    return WrapImpl(data.data(), data.size(), dimensionality,
                    CoordinateType::kInt8);
  }
  static absl::StatusOr<std::shared_ptr<SpanDataset>> WrapMutable(
      absl::Span<uint8_t> data, size_t dimensionality) {
    return WrapImpl(data.data(), data.size(), dimensionality,
                    CoordinateType::kUint8);
  }

  // Note that after constructing the dataset, the user needs to fill in the
  // coordinates themselves, later.
  SpanDataset(GingerNodeId num_points, size_t dimensionality,
              CoordinateType coordinate_type = CoordinateType::kFloat)
      : data_(nullptr, &FreeDeleter),
        num_points_(num_points),
        point_size_in_bytes_(dimensionality *
                             CoordinateSizeInBytes(coordinate_type)),
        coordinate_type_(coordinate_type) {
    if (num_points == 0 || point_size_in_bytes_ == 0) {
      return;
    }
    size_t total_bytes = num_points * point_size_in_bytes_;
#ifdef MADV_HUGEPAGE
    constexpr size_t kHugePageAlignment = 1l << 21;  // 2MB
    // Round up total_bytes to the next multiple of kHugePageAlignment.
    total_bytes =
        ((total_bytes + kHugePageAlignment - 1) / kHugePageAlignment) *
        kHugePageAlignment;
    void* ptr = std::aligned_alloc(kHugePageAlignment, total_bytes);
    ABSL_CHECK(ptr != nullptr);
    // Advise the kernel to back this memory with huge pages.
    madvise(ptr, total_bytes, MADV_HUGEPAGE);
#else
    // Don't align without huge pages.
    void* ptr = std::malloc(total_bytes);
    ABSL_CHECK(ptr != nullptr);
#endif
    data_.reset(ptr);
  }

  SpanDataset(SpanDataset&& other) noexcept
      : data_(std::move(other.data_)),
        num_points_(other.num_points_),
        point_size_in_bytes_(other.point_size_in_bytes_),
        coordinate_type_(other.coordinate_type_) {
    other.num_points_ = 0;
    other.point_size_in_bytes_ = 0;
  }

  SpanDataset& operator=(SpanDataset&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      num_points_ = other.num_points_;
      point_size_in_bytes_ = other.point_size_in_bytes_;
      coordinate_type_ = other.coordinate_type_;
      other.num_points_ = 0;
      other.point_size_in_bytes_ = 0;
    }
    return *this;
  }

  GingerNodeId size() const { return num_points_; }

  size_t point_size_in_bytes() const { return point_size_in_bytes_; }

  size_t dimensionality() const {
    return point_size_in_bytes_ / CoordinateSizeInBytes(coordinate_type_);
  }

  CoordinateType coordinate_type() const { return coordinate_type_; }

  // Returns a pointer to the first coordinate of point `i`.
  void* operator[](GingerNodeId i) {
    return static_cast<char*>(data_.get()) + i * point_size_in_bytes_;
  }

  const void* operator[](GingerNodeId i) const {
    return static_cast<const char*>(data_.get()) + i * point_size_in_bytes_;
  }

  absl::Span<char> get_bytes(GingerNodeId i) {
    return absl::MakeSpan(static_cast<char*>(operator[](i)),
                          point_size_in_bytes_);
  }

  absl::Span<const char> get_bytes(GingerNodeId i) const {
    return absl::MakeConstSpan(static_cast<const char*>(operator[](i)),
                               point_size_in_bytes_);
  }

  void SetBytes(GingerNodeId i, absl::Span<const char> values) {
    ABSL_CHECK_EQ(values.size(), point_size_in_bytes_);
    absl::c_copy(values, get_bytes(i).begin());
  }

  template <typename T>
  absl::Span<T> get(GingerNodeId i) {
    ABSL_CHECK_EQ(coordinate_type_, CoordinateTypeFor<T>());
    return absl::MakeSpan(static_cast<T*>(operator[](i)),
                          point_size_in_bytes_ / sizeof(T));
  }

  template <typename T>
  void Set(GingerNodeId i, absl::Span<const T> values) {
    ABSL_CHECK_EQ(coordinate_type_, CoordinateTypeFor<T>());
    ABSL_CHECK_EQ(values.size(), point_size_in_bytes_ / sizeof(T));
    absl::c_copy(values, get<T>(i).begin());
  }

  template <typename T>
  absl::Span<const T> get(GingerNodeId i) const {
    ABSL_CHECK_EQ(coordinate_type_, CoordinateTypeFor<T>());
    return absl::MakeConstSpan(static_cast<const T*>(operator[](i)),
                               point_size_in_bytes_ / sizeof(T));
  }

 private:
  static void FreeDeleter(void* ptr) { std::free(ptr); }
  static void NoopDeleter(void*) {}

  static absl::StatusOr<GingerNodeId> ValidateWrapSize(size_t total_elements,
                                                       size_t dimensionality,
                                                       size_t coordinate_size) {
    if (dimensionality == 0) {
      return absl::InvalidArgumentError(
          "SpanDataset dimensionality must be positive, got 0.");
    }
    const size_t max_dimensionality =
        std::numeric_limits<size_t>::max() / coordinate_size;
    if (dimensionality > max_dimensionality) {
      return absl::InvalidArgumentError(
          absl::StrCat("SpanDataset dimensionality is ", dimensionality,
                       ", but must be at most ", max_dimensionality, "."));
    }
    if (total_elements % dimensionality != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Span size (", total_elements,
          ") must be a multiple of dimensionality (", dimensionality, ")."));
    }
    const size_t num_points = total_elements / dimensionality;
    if (num_points >
        static_cast<size_t>(std::numeric_limits<GingerNodeId>::max())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Number of points (", num_points, ") exceeds maximum GingerNodeId (",
          std::numeric_limits<GingerNodeId>::max(), ")."));
    }
    return static_cast<GingerNodeId>(num_points);
  }

  // Shared implementation of `Wrap` and `WrapMutable`. `data` points to
  // `num_elements` coordinates of type `coordinate_type`.
  static absl::StatusOr<std::shared_ptr<SpanDataset>> WrapImpl(
      const void* data, size_t num_elements, size_t dimensionality,
      CoordinateType coordinate_type) {
    if (num_elements > 0 && data == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Span has ", num_elements, " elements but a null data pointer."));
    }
    const size_t coordinate_size = CoordinateSizeInBytes(coordinate_type);
    ABSL_ASSIGN_OR_RETURN(
        GingerNodeId num_points,
        ValidateWrapSize(num_elements, dimensionality, coordinate_size));
    void* ptr = num_points == 0 ? nullptr : const_cast<void*>(data);
    return std::shared_ptr<SpanDataset>(new SpanDataset(
        ptr, num_points, dimensionality * coordinate_size, coordinate_type));
  }

  SpanDataset(void* ptr, GingerNodeId num_points, size_t point_size_in_bytes,
              CoordinateType coordinate_type)
      : data_(ptr, &NoopDeleter),
        num_points_(num_points),
        point_size_in_bytes_(point_size_in_bytes),
        coordinate_type_(coordinate_type) {}

  std::unique_ptr<void, void (*)(void*)> data_;
  GingerNodeId num_points_;         // Number of points
  size_t point_size_in_bytes_;      // Size of each point in bytes
  CoordinateType coordinate_type_;  // Coordinate type of each point
};

}  // namespace ginger

#endif  // GINGER_DATASET_H_
