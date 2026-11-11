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

#include "ginger/graph/compression/varint_codec.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {
namespace {

TEST(VarintCodecTest, ExampleUsage) {
  std::vector<GingerNodeId> values_to_encode = {10, 250, 65536, 1};

  // Step 1: Dry run to determine the required buffer size.
  VarintEncoder dry_run_encoder({}, /*dry_run=*/true);
  for (GingerNodeId value : values_to_encode) {
    dry_run_encoder.Encode(value);
  }
  dry_run_encoder.Flush();  // Optional for Varint, required for other encoders.
  size_t encoded_size = dry_run_encoder.CurOffset() + VarintDecoder::kSlopBytes;

  // Step 2: Allocate the buffer and encode the values.
  std::vector<uint8_t> encoded_buffer(encoded_size);
  VarintEncoder encoder(absl::MakeSpan(encoded_buffer));
  for (GingerNodeId value : values_to_encode) {
    encoder.Encode(value);
  }
  encoder.Flush();

  // Step 3: Decode the values.
  VarintDecoder decoder(encoded_buffer);
  std::vector<GingerNodeId> decoded_values;
  for (size_t i = 0; i < values_to_encode.size(); ++i) {
    decoded_values.push_back(decoder.Next());
  }

  EXPECT_EQ(decoded_values, values_to_encode);
}

TEST(VarintCodecTest, SingleNeighborhood) {
  const std::vector<GingerNodeId> neighborhood = {1, 127, 128, 5000, 100000};
  std::vector<GingerNodeId> decoded_neighborhood;

  // Step 1: Dry run to determine the required buffer size and setup the encoder
  // and decoder.
  VarintEncoder sizer({}, /*dry_run=*/true);
  for (const auto value : neighborhood) {
    sizer.Encode(value);
  }
  sizer.Flush();

  const size_t compressed_size = sizer.CurOffset() + VarintDecoder::kSlopBytes;
  std::vector<uint8_t> encoded_data(compressed_size);
  VarintEncoder encoder(absl::MakeSpan(encoded_data));

  // Step 2: Encode and decode the values.
  for (const auto value : neighborhood) {
    encoder.Encode(value);
  }
  encoder.Flush();

  VarintDecoder decoder(encoded_data);
  for (size_t i = 0; i < neighborhood.size(); ++i) {
    decoded_neighborhood.push_back(decoder.Next());
  }

  // Step 3: Validate the results.
  EXPECT_EQ(compressed_size, 9 + VarintDecoder::kSlopBytes);
  EXPECT_EQ(decoded_neighborhood, neighborhood);
}

TEST(VarintCodecTest, AllByteLengthBoundaries) {
  const std::vector<GingerNodeId> values = {
      0,
      (1 << 7) - 1,
      1 << 7,
      (1 << 14) - 1,
      1 << 14,
      (1 << 21) - 1,
      1 << 21,
      (1 << 28) - 1,
      1 << 28,
      std::numeric_limits<GingerNodeId>::max(),
  };

  VarintEncoder sizer({}, /*dry_run=*/true);
  for (const GingerNodeId value : values) {
    sizer.Encode(value);
  }
  sizer.Flush();

  // 2 values each of 1, 2, 3, 4, and 5 bytes = 30 bytes total.
  EXPECT_EQ(sizer.CurOffset(), 30);

  const size_t compressed_size = sizer.CurOffset() + VarintDecoder::kSlopBytes;
  std::vector<uint8_t> encoded_data(compressed_size);
  VarintEncoder encoder(absl::MakeSpan(encoded_data));
  for (const GingerNodeId value : values) {
    encoder.Encode(value);
  }
  encoder.Flush();
  EXPECT_EQ(encoder.CurOffset(), 30);

  VarintDecoder decoder(encoded_data);
  std::vector<GingerNodeId> decoded_values;
  decoded_values.reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    decoded_values.push_back(decoder.Next());
  }
  EXPECT_EQ(decoded_values, values);
}

TEST(VarintCodecTest, MultipleNeighborhoods) {
  const std::vector<std::vector<GingerNodeId>> neighborhoods = {
      {1, 2, 3}, {128, 129}, {50000, 60000, 70000, 80000}};
  std::vector<std::vector<GingerNodeId>> decoded_neighborhoods;

  // Step 1: Dry run to determine the required buffer size and setup the encoder
  // and decoder.
  std::vector<size_t> offsets;
  VarintEncoder sizer({}, /*dry_run=*/true);
  for (const auto& neighborhood : neighborhoods) {
    offsets.push_back(sizer.CurOffset());
    for (const auto value : neighborhood) {
      sizer.Encode(value);
    }
    sizer.Flush();
  }
  offsets.push_back(sizer.CurOffset());

  const size_t compressed_size = offsets.back() + VarintDecoder::kSlopBytes;
  std::vector<uint8_t> encoded_data(compressed_size);
  VarintEncoder encoder(absl::MakeSpan(encoded_data));

  // Step 2: Encode and decode the values.
  for (const auto& neighborhood : neighborhoods) {
    for (const auto value : neighborhood) {
      encoder.Encode(value);
    }
    encoder.Flush();
  }

  VarintDecoder decoder(encoded_data);
  for (size_t i = 0; i < neighborhoods.size(); ++i) {
    const size_t offset = offsets[i];
    const size_t neighborhood_size = offsets[i + 1] - offset;
    VarintDecoder decoder(
        absl::MakeConstSpan(encoded_data).subspan(offset, neighborhood_size));

    std::vector<GingerNodeId> decoded_neighborhood;
    for (size_t j = 0; j < neighborhoods[i].size(); ++j) {
      decoded_neighborhood.push_back(decoder.Next());
    }
    decoded_neighborhoods.push_back(std::move(decoded_neighborhood));
  }

  // Step 3: Validate the results.
  EXPECT_EQ(compressed_size, 19 + VarintDecoder::kSlopBytes);
  EXPECT_EQ(decoded_neighborhoods, neighborhoods);
}

TEST(VarintCodecDeathTest, NegativeValueInDryRun) {
  VarintEncoder dry_run_encoder({}, /*dry_run=*/true);
  EXPECT_DEATH(dry_run_encoder.Encode(-1), "value >= 0");
}

TEST(VarintCodecDeathTest, NegativeValueInActualRun) {
  std::vector<uint8_t> encoded_buffer(10);
  VarintEncoder encoder(absl::MakeSpan(encoded_buffer));
  EXPECT_DEATH(encoder.Encode(-1), "value >= 0");
}

}  // namespace
}  // namespace ginger
