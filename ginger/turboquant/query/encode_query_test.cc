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

#include "ginger/turboquant/query/encode_query.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/rotation/rotation.h"
#include "ginger/turboquant/types/distance_metric.h"

namespace ginger {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr size_t kDim = 64;
constexpr size_t kTurboSliceDim = 4;
constexpr uint32_t kSeed = 123;

class EncodeQueryTest : public ::testing::Test {
 protected:
  // Builds a query whose rotated coordinates are zero on the TurboSlice prefix
  // [0, kTurboSliceDim) and non-zero outside it. Inverse-rotating and then
  // applying the random signs makes the rotated prefix exactly zero, so the
  // quantized prefix norm is zero without relying on rounding.
  void SetUp() override {
    std::vector<float> target_rotated(kDim, 0.0f);
    target_rotated[kTurboSliceDim] = 1.0f;
    ASSERT_OK(rotation_.InverseRotate(absl::MakeSpan(target_rotated),
                                      absl::MakeSpan(query_)));
    const absl::Span<const float> signs = rotation_.signs();
    for (size_t i = 0; i < kDim; ++i) {
      query_[i] *= signs[i];
    }
  }

  const TurboQuantCodec tq_{kDim, kSeed,
                            TurboQuantDistanceMetric::kDotProductDistance,
                            kTurboSliceDim};
  const HadamardAndMatrixRotation rotation_{kDim, kSeed};
  std::vector<float> query_ = std::vector<float>(kDim, 0.0f);
  std::vector<int8_t> q_bytes_ = std::vector<int8_t>(tq_.num_bytes_per_query());
  std::vector<int8_t> ts_bytes_ =
      std::vector<int8_t>(tq_.num_bytes_per_turboslice_query());
};

TEST_F(EncodeQueryTest, ZeroTurboSlicePrefixNormYieldsZeroScale) {
  ASSERT_OK_AND_ASSIGN(
      EncodeQueryResult full,
      internal::EncodeQuery(
          query_, absl::MakeSpan(q_bytes_), {}, kDim, tq_.num_bytes_per_query(),
          tq_.num_bytes_per_datapoint(), rotation_, kTurboSliceDim));
  EXPECT_EQ(full.norm_scaling_factor_turboslice, 0.0f);

  ASSERT_OK_AND_ASSIGN(EncodeQueryResult ts,
                       internal::EncodeTurboSliceQuery(
                           query_, absl::MakeSpan(ts_bytes_), kDim,
                           kTurboSliceDim, tq_.num_bytes_per_turboslice_query(),
                           tq_.turboslice_bytes_per_datapoint(), rotation_));
  EXPECT_EQ(ts.norm_scaling_factor_turboslice, 0.0f);
}

TEST_F(EncodeQueryTest, RejectsTurboSliceDimensionalityAboveDimensionality) {
  EXPECT_THAT(internal::EncodeQuery(query_, absl::MakeSpan(q_bytes_), {}, kDim,
                                    tq_.num_bytes_per_query(),
                                    tq_.num_bytes_per_datapoint(), rotation_,
                                    /*turboslice_dimensionality=*/kDim + 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("TurboSlice dimensionality cannot exceed")));
  EXPECT_THAT(internal::EncodeTurboSliceQuery(
                  query_, absl::MakeSpan(ts_bytes_), kDim,
                  /*turboslice_dimensionality=*/kDim + 1,
                  tq_.num_bytes_per_turboslice_query(),
                  tq_.turboslice_bytes_per_datapoint(), rotation_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("TurboSlice dimensionality cannot exceed")));
}

TEST_F(EncodeQueryTest, RejectsInconsistentByteSizes) {
  EXPECT_THAT(
      internal::EncodeQuery(query_, absl::MakeSpan(q_bytes_), {}, kDim,
                            tq_.num_bytes_per_query(),
                            /*num_bytes_per_datapoint=*/kDim / 4, rotation_),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("bytes per datapoint does not match")));
  std::vector<int8_t> small_q_bytes(kDim / 2);
  EXPECT_THAT(
      internal::EncodeQuery(query_, absl::MakeSpan(small_q_bytes), {}, kDim,
                            /*num_bytes_per_query=*/small_q_bytes.size(),
                            tq_.num_bytes_per_datapoint(), rotation_),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("bytes per query is too small")));

  EXPECT_THAT(
      internal::EncodeTurboSliceQuery(
          query_, absl::MakeSpan(ts_bytes_), kDim, kTurboSliceDim,
          tq_.num_bytes_per_turboslice_query(),
          /*turboslice_bytes_per_datapoint=*/kDim / 2, rotation_),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("TurboSlice bytes per datapoint does not match")));
  constexpr size_t kLargeTurboSliceDim = kDim;
  std::vector<int8_t> small_ts_bytes(kLargeTurboSliceDim / 2);
  EXPECT_THAT(
      internal::EncodeTurboSliceQuery(
          query_, absl::MakeSpan(small_ts_bytes), kDim, kLargeTurboSliceDim,
          /*num_bytes_per_turboslice_query=*/small_ts_bytes.size(),
          /*turboslice_bytes_per_datapoint=*/(kLargeTurboSliceDim + 1) / 2,
          rotation_),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("TurboSlice query is too small")));
}

TEST_F(EncodeQueryTest, AcceptsCodecDerivedSizes) {
  EXPECT_THAT(internal::EncodeQuery(query_, absl::MakeSpan(q_bytes_), {}, kDim,
                                    tq_.num_bytes_per_query(),
                                    tq_.num_bytes_per_datapoint(), rotation_,
                                    kTurboSliceDim),
              IsOk());
  EXPECT_THAT(internal::EncodeTurboSliceQuery(
                  query_, absl::MakeSpan(ts_bytes_), kDim, kTurboSliceDim,
                  tq_.num_bytes_per_turboslice_query(),
                  tq_.turboslice_bytes_per_datapoint(), rotation_),
              IsOk());
}

TEST_F(EncodeQueryTest, EncodeQuantizedDataPointAsQueryAcceptsUnalignedBuffer) {
  for (size_t dim : {size_t{128}, size_t{70}}) {
    TurboQuantCodec codec(dim, kSeed);
    std::vector<uint8_t> point(codec.num_bytes_per_datapoint());
    for (size_t i = 0; i < point.size(); ++i) {
      point[i] =
          static_cast<uint8_t>(((2 * i) & 0xF) | (((2 * i + 1) & 0xF) << 4));
    }

    const size_t q_bytes = codec.num_bytes_per_query();
    std::vector<int8_t> expected(q_bytes, 0);
    std::vector<int8_t> raw_unaligned(q_bytes + 64, 0x5a);
    // Pick an offset guaranteed not to be 16-byte aligned.
    int8_t* unaligned_ptr = raw_unaligned.data();
    if (reinterpret_cast<uintptr_t>(unaligned_ptr) % 16 == 0) {
      ++unaligned_ptr;
    }
    absl::Span<int8_t> unaligned_span(unaligned_ptr, q_bytes);

    ASSERT_THAT(
        codec.EncodeQuantizedDataPointAsQuery(point, absl::MakeSpan(expected),
                                              /*has_3bit_prefix=*/false),
        IsOk());
    ASSERT_THAT(codec.EncodeQuantizedDataPointAsQuery(
                    point, unaligned_span, /*has_3bit_prefix=*/false),
                IsOk());
    EXPECT_EQ(unaligned_span, absl::MakeConstSpan(expected)) << "dim=" << dim;
  }
}

}  // namespace
}  // namespace ginger
