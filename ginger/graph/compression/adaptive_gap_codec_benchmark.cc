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

// Measures decoding plus prefix sums through the public codec APIs. Setup,
// encoding, allocation, and correctness checks are outside the timed loop.
// Each reported iteration decodes one list (KeepRunningBatch amortizes the
// benchmark loop overhead). Use --benchmark_filter to select sizes/patterns.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"  // copybara:preserve
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/graph/compression/adaptive_gap_codec.h"
#include "ginger/graph/compression/streamvbyte_codec.h"
#include "ginger/graph/compression/varint_codec.h"
#include "ginger/types.h"
#include "hwy/detect_targets.h"
#include "hwy/targets.h"

namespace ginger {
namespace {

enum class Codec { kAdaptive, kStreamVByte, kPrefixVarint };
enum class Pattern {
  kTiny,
  kMixed,
  kWideFirst,
  kByteBook,
  kMedium,
  kSparseMulti,
  kMixedBooks
};

constexpr std::array<const char*, 7> kPatternNames = {
    "tiny",   "mixed",        "wide_first", "byte_book",
    "medium", "sparse_multi", "mixed_books"};

// The same generator and seed are used for every codec and Highway target.
std::vector<GingerNodeId> MakeGaps(size_t count, Pattern pattern, size_t row,
                                   std::mt19937& rng) {
  if (pattern == Pattern::kMixedBooks) {
    pattern = static_cast<Pattern>(row % 5);
  }
  std::vector<GingerNodeId> gaps(count);
  for (size_t i = 0; i < count; ++i) {
    auto& gap = gaps[i];
    switch (pattern) {
      case Pattern::kTiny:
        gap = rng() & 3;
        break;
      case Pattern::kMixed: {
        constexpr uint32_t masks[] = {15, 255, 4095, 65535};
        const uint32_t mask = masks[rng() % 4];
        gap = rng() & mask;
        break;
      }
      case Pattern::kWideFirst:
        gap = rng() & 15;
        break;
      case Pattern::kByteBook:
        gap = rng() & 255;
        break;
      case Pattern::kMedium:
        gap = 256 + (rng() & 65535);
        break;
      case Pattern::kSparseMulti: {
        // Several nonzero controls across both low and high halves of the
        // 16-bit sparse mask (e.g. groups 1, 4, 7, 10, 13 for 64 IDs).
        const size_t group = i / 4;
        if ((i % 4) == 0 && (group % 3) == 1) {
          gap = 16 + (rng() & 239);
        } else {
          gap = rng() & 3;
        }
        break;
      }
      case Pattern::kMixedBooks:
        break;  // Replaced above.
    }
  }
  if (pattern == Pattern::kWideFirst) {
    gaps[0] = (1u << 24) + (rng() & ((1u << 23) - 1));
  } else if (pattern == Pattern::kByteBook) {
    gaps[0] = (1u << 29) + (rng() & ((1u << 24) - 1));
  }
  return gaps;
}

template <Codec codec>
absl::StatusOr<size_t> EncodedSize(absl::Span<const GingerNodeId> gaps) {
  if constexpr (codec == Codec::kAdaptive) {
    return AdaptiveGapEncoder::EncodedSize(gaps);
  } else if constexpr (codec == Codec::kStreamVByte) {
    return StreamVByteEncoder::Encode(gaps, {}, /*dry_run=*/true);
  } else {
    VarintEncoder encoder({}, /*dry_run=*/true);
    for (GingerNodeId gap : gaps) encoder.Encode(gap);
    return encoder.CurOffset();
  }
}

template <Codec codec>
absl::StatusOr<size_t> Encode(absl::Span<const GingerNodeId> gaps,
                              absl::Span<uint8_t> output) {
  if constexpr (codec == Codec::kAdaptive) {
    return AdaptiveGapEncoder::Encode(gaps, output);
  } else if constexpr (codec == Codec::kStreamVByte) {
    return StreamVByteEncoder::Encode(gaps, output, /*dry_run=*/false);
  } else {
    VarintEncoder encoder(output, /*dry_run=*/false);
    for (GingerNodeId gap : gaps) encoder.Encode(gap);
    return encoder.CurOffset();
  }
}

template <Codec codec>
void Decode(absl::Span<const uint8_t> input, absl::Span<GingerNodeId> output) {
  if constexpr (codec == Codec::kAdaptive) {
    AdaptiveGapDecoder::DecodeGaps(input, output);
  } else if constexpr (codec == Codec::kStreamVByte) {
    StreamVByteDecoder::DecodeGaps(input, output);
  } else {
    VarintDecoder decoder(input);
    GingerNodeId previous = 0;
    for (auto& id : output) id = previous += decoder.Next();
  }
}

// Reset target selection even if setup detects an invalid input or mismatch.
class ScopedTarget {
 public:
  explicit ScopedTarget(int64_t target) {
    hwy::SetSupportedTargetsForTest(target);
  }
  ~ScopedTarget() { hwy::SetSupportedTargetsForTest(0); }
};

struct EncodedList {
  size_t offset;
  size_t size;
};

template <Codec codec>
void BM_Decode(benchmark::State& state, int64_t target, Pattern pattern,
               size_t degree, size_t num_rows) {
  const ScopedTarget selected_target(target);
  std::mt19937 rng(123456789);
  std::vector<EncodedList> lists;
  std::vector<uint8_t> encoded;
  std::vector<GingerNodeId> output_storage(degree + 2, -1);
  auto output = absl::MakeSpan(output_storage).subspan(1, degree);
  std::array<size_t, 16> books{};
  size_t sparse_rows = 0;
  size_t anchor_rows = 0;
  uint64_t gap_hash = 14695981039346656037ull;
  constexpr size_t kSlopBytes =
      std::max({AdaptiveGapDecoder::kSlopBytes, StreamVByteDecoder::kSlopBytes,
                VarintDecoder::kSlopBytes});

  for (size_t row = 0; row < num_rows; ++row) {
    const auto gaps = MakeGaps(degree, pattern, row, rng);
    std::vector<GingerNodeId> expected;
    int64_t sum = 0;
    for (GingerNodeId gap : gaps) {
      sum += gap;
      if (gap < 0 || sum > std::numeric_limits<GingerNodeId>::max()) {
        state.SkipWithError("Generated gaps overflow GingerNodeId");
        return;
      }
      expected.push_back(static_cast<GingerNodeId>(sum));
      for (int byte = 0; byte < 4; ++byte) {
        gap_hash ^= (static_cast<uint32_t>(gap) >> (8 * byte)) & 255;
        gap_hash *= 1099511628211ull;
      }
    }
    const auto size = EncodedSize<codec>(gaps);
    if (!size.ok()) {
      state.SkipWithError(size.status().ToString());
      return;
    }
    // No per-list padding in the corpus: padding is shared at its final end.
    const size_t offset = encoded.size();
    encoded.resize(offset + *size);
    const auto actual =
        Encode<codec>(gaps, absl::MakeSpan(encoded).subspan(offset, *size));
    if (!actual.ok() || *actual != *size) {
      state.SkipWithError("Encoding failed or disagreed with EncodedSize");
      return;
    }
    lists.push_back({offset, *size});
    if constexpr (codec == Codec::kAdaptive) {
      ++books[AdaptiveGapDecoder::TagBook(encoded[offset])];
      sparse_rows +=
          AdaptiveGapDecoder::TagHasCompressedControls(encoded[offset]);
      anchor_rows += AdaptiveGapDecoder::TagHasAnchor(encoded[offset]);
    }
    // Validate before the next resize, with exact logical spans and nonzero
    // slop. Repeating validation after assembly also checks shared padding.
    encoded.resize(offset + *size + kSlopBytes, 0xa5);
    Decode<codec>(absl::MakeConstSpan(encoded).subspan(offset, *size), output);
    if (!std::equal(expected.begin(), expected.end(), output.begin()) ||
        output_storage.front() != -1 || output_storage.back() != -1) {
      state.SkipWithError("Decoded IDs or output sentinels do not match");
      return;
    }
    encoded.resize(offset + *size);
  }
  const size_t encoded_bytes = encoded.size();
  encoded.resize(encoded_bytes + kSlopBytes, 0xa5);

  // Recreate the deterministic inputs and verify the finished packed corpus.
  rng.seed(123456789);
  for (size_t row = 0; row < num_rows; ++row) {
    const auto gaps = MakeGaps(degree, pattern, row, rng);
    const auto& list = lists[row];
    Decode<codec>(absl::MakeConstSpan(encoded).subspan(list.offset, list.size),
                  output);
    GingerNodeId sum = 0;
    for (size_t i = 0; i < degree; ++i) {
      sum += gaps[i];
      if (output[i] != sum) {
        state.SkipWithError("Packed corpus decoded IDs do not match");
        return;
      }
    }
    if (output_storage.front() != -1 || output_storage.back() != -1) {
      state.SkipWithError("Packed corpus overwrote output sentinels");
      return;
    }
  }

  // Fixed pseudorandom visitation avoids making the 1024-row corpus a simple
  // sequential payload stream. A one-row corpus measures repeated hot input.
  std::vector<size_t> order(num_rows);
  std::iota(order.begin(), order.end(), size_t{0});
  std::shuffle(order.begin(), order.end(), rng);
  size_t next = 0;
  constexpr int64_t kBatchSize = 128;
  benchmark::DoNotOptimize(output.data());
  while (state.KeepRunningBatch(kBatchSize)) {
    for (int64_t i = 0; i < kBatchSize; ++i) {
      const auto& list = lists[order[next]];
      Decode<codec>(
          absl::MakeConstSpan(encoded).subspan(list.offset, list.size), output);
      benchmark::ClobberMemory();
      next = (next + 1) &
             (num_rows - 1);  // Registered row counts are powers of 2.
    }
  }
  state.SetItemsProcessed(state.iterations() * degree);
  state.SetBytesProcessed(state.iterations() * encoded_bytes / num_rows);
  state.counters["bytes_per_id"] =
      static_cast<double>(encoded_bytes) / (num_rows * degree);
  std::ostringstream label;
  label << "gap_hash=" << std::hex << gap_hash << std::dec;
  if constexpr (codec == Codec::kAdaptive) {
    state.counters["sparse_fraction"] =
        static_cast<double>(sparse_rows) / num_rows;
    state.counters["anchor_fraction"] =
        static_cast<double>(anchor_rows) / num_rows;
    label << " books=";
    for (size_t book = 0; book < books.size(); ++book) {
      if (books[book] != 0) label << book << ':' << books[book] << ',';
    }
  }
  state.SetLabel(label.str());
}

// Measures `EncodedSize` (kDryRun) or `Encode` per list over a pregenerated
// corpus. Encoders do not use runtime SIMD dispatch.
template <Codec codec, bool kDryRun>
void BM_Encode(benchmark::State& state, Pattern pattern, size_t degree,
               size_t num_rows) {
  std::mt19937 rng(123456789);
  std::vector<std::vector<GingerNodeId>> rows;
  size_t max_size = 0;
  size_t total_size = 0;
  for (size_t row = 0; row < num_rows; ++row) {
    rows.push_back(MakeGaps(degree, pattern, row, rng));
    const auto size = EncodedSize<codec>(rows.back());
    if (!size.ok()) {
      state.SkipWithError(size.status().ToString());
      return;
    }
    max_size = std::max(max_size, *size);
    total_size += *size;
  }
  std::vector<uint8_t> output(max_size);
  size_t next = 0;
  constexpr int64_t kBatchSize = 128;
  while (state.KeepRunningBatch(kBatchSize)) {
    for (int64_t i = 0; i < kBatchSize; ++i) {
      absl::StatusOr<size_t> size =
          kDryRun ? EncodedSize<codec>(rows[next])
                  : Encode<codec>(rows[next], absl::MakeSpan(output));
      benchmark::DoNotOptimize(size);
      benchmark::ClobberMemory();
      next = (next + 1) & (num_rows - 1);
    }
  }
  state.SetItemsProcessed(state.iterations() * degree);
  state.counters["bytes_per_id"] =
      static_cast<double>(total_size) / (num_rows * degree);
}

template <Codec codec>
void RegisterEncode(const char* name) {
  for (size_t pattern = 0; pattern < kPatternNames.size(); ++pattern) {
    for (size_t degree : {4, 16, 32, 64, 128, 256}) {
      for (const bool dry_run : {true, false}) {
        std::ostringstream name_stream;
        name_stream << (dry_run ? "EncodedSize/" : "Encode/") << name << '/'
                    << kPatternNames[pattern] << "/ids:" << degree;
        const std::string benchmark_name = name_stream.str();
        benchmark::RegisterBenchmark(
            benchmark_name.c_str(),
            dry_run ? &BM_Encode<codec, true> : &BM_Encode<codec, false>,
            static_cast<Pattern>(pattern), degree, size_t{1024})
            ->Unit(benchmark::kNanosecond);
      }
    }
  }
}

template <Codec codec>
void RegisterCodec(const char* name, int64_t target) {
  for (size_t pattern = 0; pattern < kPatternNames.size(); ++pattern) {
    for (size_t degree : {1, 3, 4, 15, 16, 17, 32, 63, 64, 65, 96, 128, 256}) {
      for (size_t rows : {1, 1024}) {
        std::ostringstream name_stream;
        name_stream << "Decode/" << name << '/'
                    << (target == 0 ? "Baseline" : hwy::TargetName(target))
                    << '/' << kPatternNames[pattern] << "/ids:" << degree
                    << "/rows:" << rows;
        const std::string benchmark_name = name_stream.str();
        benchmark::RegisterBenchmark(benchmark_name.c_str(), &BM_Decode<codec>,
                                     target, static_cast<Pattern>(pattern),
                                     degree, rows)
            ->Unit(benchmark::kNanosecond);
      }
    }
  }
}

void RegisterBenchmarks() {
  const auto targets = hwy::SupportedAndGeneratedTargets();
  // Measure the best available SIMD target, plus AVX2 when available. Both
  // codec libraries and this harness must use the same compilation flags.
  for (int64_t target : targets) {
    if (target == targets.front() || target == HWY_AVX2) {
      RegisterCodec<Codec::kAdaptive>("Adaptive", target);
      RegisterCodec<Codec::kStreamVByte>("StreamVByte", target);
    }
  }
  RegisterCodec<Codec::kPrefixVarint>("PrefixVarint", 0);
  RegisterEncode<Codec::kAdaptive>("Adaptive");
  RegisterEncode<Codec::kStreamVByte>("StreamVByte");
}

}  // namespace
}  // namespace ginger

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ginger::RegisterBenchmarks();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
