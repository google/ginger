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

#include "ginger/dataset.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

// =============================================================================
// SpanDataset Tests
// =============================================================================

TEST(SpanDatasetTest, EmptyDataset) {
  SpanDataset dataset(0, 10);
  EXPECT_EQ(dataset.size(), 0);
  EXPECT_EQ(dataset.point_size_in_bytes(), 10 * sizeof(float));
  EXPECT_EQ(dataset.dimensionality(), 10);
  EXPECT_EQ(dataset.coordinate_type(), CoordinateType::kFloat);
}

TEST(SpanDatasetTest, ZeroDimensionality) {
  SpanDataset dataset(10, 0);
  EXPECT_EQ(dataset.size(), 10);
  EXPECT_EQ(dataset.point_size_in_bytes(), 0);
  EXPECT_EQ(dataset.dimensionality(), 0);
  EXPECT_EQ(dataset.coordinate_type(), CoordinateType::kFloat);
}

TEST(SpanDatasetTest, BasicOperations) {
  SpanDataset dataset(2, 3);
  EXPECT_EQ(dataset.size(), 2);
  EXPECT_EQ(dataset.point_size_in_bytes(), 3 * sizeof(float));
  EXPECT_EQ(dataset.dimensionality(), 3);
  EXPECT_EQ(dataset.coordinate_type(), CoordinateType::kFloat);

  // Modify the dataset
  float* point0 = static_cast<float*>(dataset[0]);
  point0[0] = 1.0;
  point0[1] = 2.0;
  point0[2] = 3.0;
  float* point1 = static_cast<float*>(dataset[1]);
  point1[0] = 4.0;
  point1[1] = 5.0;
  point1[2] = 6.0;

  const auto& const_dataset = dataset;
  static_assert(std::is_same_v<decltype(dataset[0]), void*>);
  static_assert(std::is_same_v<decltype(const_dataset[0]), const void*>);
  const float* const_point0 = static_cast<const float*>(const_dataset[0]);
  EXPECT_THAT(absl::MakeConstSpan(const_point0, 3), ElementsAre(1.0, 2.0, 3.0));

  const float* const_point1 = static_cast<const float*>(const_dataset[1]);
  EXPECT_THAT(absl::MakeConstSpan(const_point1, 3), ElementsAre(4.0, 5.0, 6.0));
}

TEST(SpanDatasetTest, Get) {
  SpanDataset dataset(1, 2);
  EXPECT_EQ(dataset.size(), 1);
  EXPECT_EQ(dataset.point_size_in_bytes(), 2 * sizeof(float));
  EXPECT_EQ(dataset.dimensionality(), 2);

  float* point0 = static_cast<float*>(dataset[0]);
  point0[0] = 1.0;
  point0[1] = 2.0;

  const auto& const_dataset = dataset;
  EXPECT_THAT(dataset.get<float>(0), ElementsAre(1.0, 2.0));
  EXPECT_THAT(const_dataset.get<float>(0), ElementsAre(1.0, 2.0));
}

TEST(SpanDatasetTest, GetBytes) {
  SpanDataset dataset(1, 1);
  EXPECT_EQ(dataset.size(), 1);
  EXPECT_EQ(dataset.point_size_in_bytes(), 4);
  EXPECT_EQ(dataset.dimensionality(), 1);

  char* point0 = static_cast<char*>(dataset[0]);
  point0[0] = 'a';
  point0[1] = 'b';
  point0[2] = 'c';
  point0[3] = 'd';

  const auto& const_dataset = dataset;
  EXPECT_THAT(dataset.get_bytes(0), ElementsAre('a', 'b', 'c', 'd'));
  EXPECT_THAT(const_dataset.get_bytes(0), ElementsAre('a', 'b', 'c', 'd'));
}

TEST(SpanDatasetTest, Set) {
  SpanDataset dataset(1, 2);
  EXPECT_EQ(dataset.size(), 1);
  EXPECT_EQ(dataset.point_size_in_bytes(), 2 * sizeof(float));
  EXPECT_EQ(dataset.dimensionality(), 2);

  dataset.Set<float>(0, {1.0, 2.0});
  EXPECT_THAT(dataset.get<float>(0), ElementsAre(1.0, 2.0));

  const auto& const_dataset = dataset;
  EXPECT_THAT(const_dataset.get<float>(0), ElementsAre(1.0, 2.0));
}

TEST(SpanDatasetTest, SetBytes) {
  SpanDataset dataset(1, 1);
  EXPECT_EQ(dataset.size(), 1);
  EXPECT_EQ(dataset.point_size_in_bytes(), 4);
  EXPECT_EQ(dataset.dimensionality(), 1);

  std::vector<char> bytes = {'w', 'x', 'y', 'z'};
  dataset.SetBytes(0, absl::MakeConstSpan(bytes));
  EXPECT_THAT(dataset.get_bytes(0), ElementsAre('w', 'x', 'y', 'z'));

  const auto& const_dataset = dataset;
  EXPECT_THAT(const_dataset.get_bytes(0), ElementsAre('w', 'x', 'y', 'z'));
}

TEST(SpanDatasetTest, Int8Coordinates) {
  SpanDataset dataset(2, 3, CoordinateType::kInt8);
  EXPECT_EQ(dataset.size(), 2);
  EXPECT_EQ(dataset.point_size_in_bytes(), 3 * sizeof(int8_t));
  EXPECT_EQ(dataset.dimensionality(), 3);
  EXPECT_EQ(dataset.coordinate_type(), CoordinateType::kInt8);

  dataset.Set<int8_t>(0, {1, 2, 3});
  dataset.Set<int8_t>(1, {-1, -2, -3});
  EXPECT_THAT(dataset.get<int8_t>(0), ElementsAre(1, 2, 3));
  EXPECT_THAT(dataset.get<int8_t>(1), ElementsAre(-1, -2, -3));
}

TEST(SpanDatasetTest, Uint8Coordinates) {
  SpanDataset dataset(2, 4, CoordinateType::kUint8);
  EXPECT_EQ(dataset.size(), 2);
  EXPECT_EQ(dataset.point_size_in_bytes(), 4 * sizeof(uint8_t));
  EXPECT_EQ(dataset.dimensionality(), 4);
  EXPECT_EQ(dataset.coordinate_type(), CoordinateType::kUint8);

  dataset.Set<uint8_t>(0, {10, 20, 30, 40});
  EXPECT_THAT(dataset.get<uint8_t>(0), ElementsAre(10, 20, 30, 40));
}

TEST(SpanDatasetTest, CoordinateTypeMismatchDeathTest) {
  SpanDataset dataset(1, 4, CoordinateType::kInt8);
  const auto& const_dataset = dataset;

  EXPECT_DEATH(dataset.get<float>(0), "");
  EXPECT_DEATH(const_dataset.get<float>(0), "");
  EXPECT_DEATH(dataset.Set<float>(0, {1.0f}), "");
  EXPECT_DEATH(dataset.get<uint8_t>(0), "");
}

TEST(SpanDatasetTest, WrapZeroCopy) {
  const std::vector<float> raw = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  ASSERT_OK_AND_ASSIGN(auto dataset, SpanDataset::Wrap(raw, 3));
  EXPECT_EQ(dataset->size(), 2);
  EXPECT_EQ(dataset->dimensionality(), 3);
  EXPECT_EQ(dataset->point_size_in_bytes(), 3 * sizeof(float));
  EXPECT_EQ(dataset->coordinate_type(), CoordinateType::kFloat);
  EXPECT_EQ((*dataset)[0], raw.data());
  EXPECT_EQ(dataset->get<float>(0).data(), raw.data());
  EXPECT_EQ(dataset->get<float>(1).data(), raw.data() + 3);
  EXPECT_THAT(dataset->get<float>(0), ElementsAre(1.0f, 2.0f, 3.0f));
  EXPECT_THAT(dataset->get<float>(1), ElementsAre(4.0f, 5.0f, 6.0f));
}

TEST(SpanDatasetTest, WrapEmptySpan) {
  ASSERT_OK_AND_ASSIGN(auto dataset,
                       SpanDataset::Wrap(absl::Span<const float>(), 4));
  EXPECT_EQ(dataset->size(), 0);
  EXPECT_EQ(dataset->dimensionality(), 4);
  EXPECT_EQ(dataset->point_size_in_bytes(), 4 * sizeof(float));
  EXPECT_EQ((*dataset)[0], nullptr);
}

TEST(SpanDatasetTest, WrapByteCoordinates) {
  const std::vector<int8_t> raw_i8 = {1, -2, 3, -4};
  ASSERT_OK_AND_ASSIGN(auto ds_i8, SpanDataset::Wrap(raw_i8, 2));
  EXPECT_EQ(ds_i8->size(), 2);
  EXPECT_EQ(ds_i8->coordinate_type(), CoordinateType::kInt8);
  EXPECT_EQ(ds_i8->get<int8_t>(0).data(), raw_i8.data());
  EXPECT_THAT(ds_i8->get<int8_t>(1), ElementsAre(3, -4));

  const std::vector<uint8_t> raw_u8 = {10, 20, 30};
  ASSERT_OK_AND_ASSIGN(auto ds_u8, SpanDataset::Wrap(raw_u8, 3));
  EXPECT_EQ(ds_u8->size(), 1);
  EXPECT_EQ(ds_u8->coordinate_type(), CoordinateType::kUint8);
  EXPECT_EQ(ds_u8->get<uint8_t>(0).data(), raw_u8.data());
  EXPECT_THAT(ds_u8->get<uint8_t>(0), ElementsAre(10, 20, 30));
}

TEST(SpanDatasetTest, WrapMutableZeroCopy) {
  std::vector<float> raw = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(auto dataset,
                       SpanDataset::WrapMutable(absl::MakeSpan(raw), 2));
  EXPECT_EQ(dataset->size(), 2);
  EXPECT_EQ(dataset->dimensionality(), 2);
  EXPECT_EQ(dataset->get<float>(0).data(), raw.data());

  dataset->get<float>(0)[0] = 9.0f;
  dataset->Set<float>(1, {7.0f, 8.0f});
  EXPECT_THAT(raw, ElementsAre(9.0f, 2.0f, 7.0f, 8.0f));
}

TEST(SpanDatasetTest, WrapValidationErrors) {
  std::vector<float> raw = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  EXPECT_THAT(SpanDataset::Wrap(raw, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SpanDataset::Wrap(absl::Span<const float>(), 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SpanDataset::WrapMutable(absl::MakeSpan(raw), 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SpanDataset::Wrap(raw, 2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SpanDataset::WrapMutable(absl::MakeSpan(raw), 3),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SpanDatasetTest, WrapRejectsTooManyPoints) {
  // Wrap only inspects the span size on this path, so the span can claim more
  // elements than the buffer holds.
  const uint8_t byte = 0;
  const size_t too_many =
      static_cast<size_t>(std::numeric_limits<GingerNodeId>::max()) + 1;
  EXPECT_THAT(SpanDataset::Wrap(absl::Span<const uint8_t>(&byte, too_many), 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SpanDatasetTest, WrapRejectsNullDataWithElements) {
  EXPECT_THAT(SpanDataset::Wrap(absl::Span<const float>(nullptr, 4), 2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SpanDataset::WrapMutable(absl::Span<float>(nullptr, 4), 2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SpanDatasetTest, WrapRejectsDimensionalityOverflowingPointSize) {
  // The span is empty, so only the point size in bytes can overflow.
  const size_t huge_dimensionality = size_t{1} << 62;
  EXPECT_THAT(SpanDataset::Wrap(absl::Span<const float>(), huge_dimensionality),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace ginger
