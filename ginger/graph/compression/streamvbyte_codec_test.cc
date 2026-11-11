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

#include "ginger/graph/compression/streamvbyte_codec.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/types.h"

#undef HWY_TARGET_INCLUDE
// clang-format off
#define HWY_TARGET_INCLUDE \
  "ginger/graph/compression/streamvbyte_codec_test.cc"
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

using ginger::GingerNodeId;
using ginger::StreamVByteDecoder;
using ginger::StreamVByteEncoder;

HWY_NOINLINE void StreamVByteCodecTestExampleUsage() {
  std::vector<GingerNodeId> values = {1, 2, 3, 10, 12};
  std::vector<GingerNodeId> delta_values;
  delta_values.reserve(values.size());

  GingerNodeId prev_value = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    delta_values.push_back(values[i] - prev_value);
    prev_value = values[i];
  }

  // Step 1: Dry run to determine the required buffer size.
  ASSERT_OK_AND_ASSIGN(size_t encoded_size,
                       StreamVByteEncoder::Encode(delta_values, {},
                                                  /*dry_run=*/true));
  encoded_size += StreamVByteDecoder::kSlopBytes;

  // Step 2: Allocate the buffer and encode the gaps.
  std::vector<uint8_t> encoded_buffer(encoded_size);
  EXPECT_OK(
      StreamVByteEncoder::Encode(delta_values, absl::MakeSpan(encoded_buffer)));

  // Step 3: Decode the values.
  std::vector<GingerNodeId> decoded_values(delta_values.size());
  StreamVByteDecoder::DecodeGaps(encoded_buffer,
                                 absl::MakeSpan(decoded_values));

  EXPECT_EQ(decoded_values, values);
}

HWY_NOINLINE void StreamVByteCodecTestSingleNeighborhood() {
  const std::vector<GingerNodeId> neighborhood = {1, 127, 128, 5000, 100000};
  std::vector<GingerNodeId> delta_neighborhood;
  delta_neighborhood.reserve(neighborhood.size());
  GingerNodeId prev_value = 0;
  for (GingerNodeId value : neighborhood) {
    delta_neighborhood.push_back(value - prev_value);
    prev_value = value;
  }

  // Step 1: Dry run to determine the required buffer size.
  ASSERT_OK_AND_ASSIGN(size_t encoded_size,
                       StreamVByteEncoder::Encode(delta_neighborhood, {},
                                                  /*dry_run=*/true));
  encoded_size += StreamVByteDecoder::kSlopBytes;

  // Step 2: Allocate the buffer and encode the gaps.
  std::vector<uint8_t> encoded_buffer(encoded_size);
  EXPECT_OK(StreamVByteEncoder::Encode(delta_neighborhood,
                                       absl::MakeSpan(encoded_buffer)));

  // Step 3: Decode the values.
  std::vector<GingerNodeId> decoded_neighborhood(delta_neighborhood.size());
  StreamVByteDecoder::DecodeGaps(encoded_buffer,
                                 absl::MakeSpan(decoded_neighborhood));

  EXPECT_EQ(decoded_neighborhood, neighborhood);
}

HWY_NOINLINE void StreamVByteCodecTestMultipleNeighborhoods() {
  const std::vector<std::vector<GingerNodeId>> neighborhoods = {
      {1, 2, 3}, {128, 129}, {50000, 60000, 70000, 80000}};
  std::vector<std::vector<GingerNodeId>> delta_neighborhoods(
      neighborhoods.size());
  for (size_t i = 0; i < neighborhoods.size(); ++i) {
    delta_neighborhoods[i].reserve(neighborhoods[i].size());
    GingerNodeId prev_value = 0;
    for (GingerNodeId value : neighborhoods[i]) {
      delta_neighborhoods[i].push_back(value - prev_value);
      prev_value = value;
    }
  }

  // Step 1: Dry run to determine encoded sizes and offsets.
  std::vector<size_t> encoded_sizes;
  size_t total_encoded_size = 0;
  for (const auto& delta_neighborhood : delta_neighborhoods) {
    ASSERT_OK_AND_ASSIGN(
        const size_t size,
        StreamVByteEncoder::Encode(delta_neighborhood, {}, /*dry_run=*/true));
    encoded_sizes.push_back(size);
    total_encoded_size += size;
  }

  std::vector<size_t> offsets;
  offsets.push_back(0);
  for (size_t i = 0; i < encoded_sizes.size(); ++i) {
    offsets.push_back(offsets.back() + encoded_sizes[i]);
  }

  // Step 2: Allocate buffer and encode neighborhoods.
  std::vector<uint8_t> encoded_buffer(total_encoded_size +
                                      StreamVByteDecoder::kSlopBytes);
  for (size_t i = 0; i < delta_neighborhoods.size(); ++i) {
    EXPECT_OK(StreamVByteEncoder::Encode(
        delta_neighborhoods[i],
        absl::MakeSpan(encoded_buffer).subspan(offsets[i], encoded_sizes[i])));
  }

  // Step 3: Decode neighborhoods.
  std::vector<std::vector<GingerNodeId>> decoded_neighborhoods_vec;
  for (size_t i = 0; i < neighborhoods.size(); ++i) {
    std::vector<GingerNodeId> decoded_neighborhood(neighborhoods[i].size());
    StreamVByteDecoder::DecodeGaps(
        absl::MakeConstSpan(encoded_buffer).subspan(offsets[i]),
        absl::MakeSpan(decoded_neighborhood));
    decoded_neighborhoods_vec.push_back(std::move(decoded_neighborhood));
  }

  // Step 4: Validate the results.
  EXPECT_EQ(decoded_neighborhoods_vec, neighborhoods);
}

HWY_NOINLINE void StreamVByteCodecTestSequenceWithNegativeValues() {
  const std::vector<GingerNodeId> delta_neighborhood = {0, 1, -1};
  EXPECT_THAT(
      StreamVByteEncoder::Encode(delta_neighborhood, /*output_buffer=*/{},
                                 /*dry_run=*/true),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {

HWY_BEFORE_TEST(StreamVByteCodecTest);
HWY_EXPORT_AND_TEST_P(StreamVByteCodecTest, StreamVByteCodecTestExampleUsage);
HWY_EXPORT_AND_TEST_P(StreamVByteCodecTest,
                      StreamVByteCodecTestSingleNeighborhood);
HWY_EXPORT_AND_TEST_P(StreamVByteCodecTest,
                      StreamVByteCodecTestMultipleNeighborhoods);
HWY_EXPORT_AND_TEST_P(StreamVByteCodecTest,
                      StreamVByteCodecTestSequenceWithNegativeValues);
HWY_AFTER_TEST();

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
#endif  // HWY_ONCE
