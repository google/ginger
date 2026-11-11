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

#include "ginger/internal/array_io.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "ginger/internal/gunit_support.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ginger/internal/array_io.pb.h"
#include "ginger/internal/parallel_for_with_status.h"

namespace ginger::array_io {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Not;

std::string TestPath(absl::string_view name) {
  return absl::StrCat(::testing::TempDir(), "/", name);
}

TEST(ArrayIoTest, ConcurrentArraysRecordsAndEmpty) {
  const std::string path = TestPath("round_trip");
  const std::vector<int64_t> ints(1'000'003, 42);
  constexpr size_t kNumRecords = 1'000'003;
  constexpr size_t kRecordSize = 7;
  auto byte_at = [](size_t r, size_t j) {
    return static_cast<char>((r * 31 + j) % 251);
  };
  TableOfContents::Range header;
  header.set_size(99);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Writer> writer, Writer::Create(path));
  ASSERT_OK(parallel_for_with_status(
      0, 4,
      [&](size_t task) -> absl::Status {
        switch (task) {
          case 0:
            return writer->WriteArray<int64_t>("ints", ints);
          case 1:
            return writer->WriteArray<int32_t>("empty", std::vector<int32_t>{});
          case 2:
            return writer->WriteRecords(
                "records", kNumRecords, kRecordSize,
                [&](size_t begin, size_t end, absl::Span<char> out) {
                  for (size_t i = begin; i < end; ++i) {
                    for (size_t j = 0; j < kRecordSize; ++j) {
                      out[(i - begin) * kRecordSize + j] = byte_at(i, j);
                    }
                  }
                  return absl::OkStatus();
                });
          default:
            return writer->WriteTextProto("header", header);
        }
      },
      /*granularity=*/1));
  ASSERT_OK(writer->Close());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Reader> reader, Reader::Open(path));
  std::vector<int64_t> loaded_ints(ints.size());
  std::vector<int32_t> loaded_empty;
  std::string loaded_records(kNumRecords * kRecordSize, '\0');
  TableOfContents::Range loaded_header;
  ASSERT_OK(parallel_for_with_status(
      0, 4,
      [&](size_t task) -> absl::Status {
        switch (task) {
          case 0:
            return reader->ReadArray<int64_t>("ints",
                                              absl::MakeSpan(loaded_ints));
          case 1:
            return reader->ReadArray<int32_t>("empty",
                                              absl::MakeSpan(loaded_empty));
          case 2:
            return reader->ReadRecords(
                "records", kNumRecords, kRecordSize,
                [&](size_t begin, size_t, absl::string_view bytes) {
                  bytes.copy(loaded_records.data() + begin * kRecordSize,
                             bytes.size());
                  return absl::OkStatus();
                });
          default:
            return reader->ReadTextProto("header", loaded_header);
        }
      },
      /*granularity=*/1));
  EXPECT_EQ(loaded_ints, ints);
  EXPECT_TRUE(loaded_empty.empty());
  EXPECT_EQ(loaded_header.size(), 99);
  for (size_t i = 0; i < kNumRecords; i += 9973) {
    for (size_t j = 0; j < kRecordSize; ++j) {
      EXPECT_EQ(loaded_records[i * kRecordSize + j], byte_at(i, j));
    }
  }
}

TEST(ArrayIoTest, FailedWriteMakesCloseFail) {
  const std::string path = TestPath("failed_write");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Writer> writer, Writer::Create(path));
  EXPECT_THAT(writer->WriteRecords("bad", 10, 4,
                                   [](size_t, size_t, absl::Span<char>) {
                                     return absl::AbortedError("serialize");
                                   }),
              StatusIs(absl::StatusCode::kAborted));
  ASSERT_OK(writer->WriteArray<int32_t>("data", std::vector<int32_t>(10, 1)));
  EXPECT_THAT(writer->Close(), StatusIs(absl::StatusCode::kAborted));
  writer.reset();
  EXPECT_THAT(Reader::Open(path), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(std::fopen(absl::StrCat(path, ".tmp").c_str(), "rb"), nullptr);

  EXPECT_THAT(Writer::Create(TestPath("missing_dir/file")), Not(IsOk()));

  const std::string fifo = TestPath("fifo");
  const std::string fifo_tmp = absl::StrCat(fifo, ".tmp");
  ASSERT_EQ(::mkfifo(fifo_tmp.c_str(), 0644), 0);
  const int fifo_rd = ::open(fifo_tmp.c_str(), O_RDONLY | O_NONBLOCK);
  ASSERT_GE(fifo_rd, 0);
  ASSERT_OK_AND_ASSIGN(writer, Writer::Create(fifo));
  EXPECT_THAT(writer->WriteArray<int32_t>("data", std::vector<int32_t>{1}),
              Not(IsOk()));
  EXPECT_THAT(writer->Close(), Not(IsOk()));
  writer.reset();
  ::close(fifo_rd);
}

TEST(ArrayIoTest, ReadErrorsAndTruncatedFile) {
  const std::string path = TestPath("read_errors");
  TableOfContents::Range header;
  header.set_size(42);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Writer> writer, Writer::Create(path));
  ASSERT_OK(writer->WriteArray<int32_t>("data", std::vector<int32_t>(10, 1)));
  ASSERT_OK(writer->WriteTextProto("header", header));
  ASSERT_OK(writer->Close());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Reader> reader, Reader::Open(path));
  std::vector<int32_t> buf(10), long_buf(11), short_buf(9);
  EXPECT_THAT(reader->ReadArray<int32_t>("missing", absl::MakeSpan(buf)),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(reader->ReadArray<int32_t>("data", absl::MakeSpan(long_buf)),
              StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(reader->ReadArray<int32_t>("data", absl::MakeSpan(short_buf)),
              StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(reader->ReadRecords("data", 10, 4,
                                  [](size_t, size_t, absl::string_view) {
                                    return absl::AbortedError("parse");
                                  }),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_OK(reader->CheckArraySize<int32_t>("data", {2, 5}));
  EXPECT_THAT(reader->CheckArraySize<int32_t>("data", {11}),
              StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(reader->CheckArraySize<int32_t>("data", {uint64_t{1} << 62, 4}),
              StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(reader->CheckArraySize<int32_t>("missing", {10}),
              StatusIs(absl::StatusCode::kNotFound));
  TableOfContents wrong_proto;
  EXPECT_THAT(reader->ReadTextProto("missing", wrong_proto),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(reader->ReadTextProto("header", wrong_proto), Not(IsOk()));
  ASSERT_EQ(::truncate(path.c_str(), 0), 0);
  EXPECT_THAT(reader->ReadArray<int32_t>("data", absl::MakeSpan(buf)),
              StatusIs(absl::StatusCode::kDataLoss));
}

TEST(ArrayIoTest, UnclosedWriterAndFailedRenameRemoveTempFile) {
  const std::string unclosed = TestPath("unclosed");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Writer> writer,
                       Writer::Create(unclosed));
  ASSERT_OK(writer->WriteArray<int32_t>("data", std::vector<int32_t>{1, 2}));
  writer.reset();
  EXPECT_THAT(Reader::Open(unclosed), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(std::fopen(absl::StrCat(unclosed, ".tmp").c_str(), "rb"), nullptr);

  const std::string dir_path = TestPath("existing_dir");
  ASSERT_EQ(::mkdir(dir_path.c_str(), 0755), 0);
  ASSERT_EQ(::mkdir(absl::StrCat(dir_path, "/sub").c_str(), 0755), 0);
  EXPECT_THAT(Reader::Open(dir_path), Not(IsOk()));
  ASSERT_OK_AND_ASSIGN(writer, Writer::Create(dir_path));
  ASSERT_OK(writer->WriteArray<int32_t>("data", std::vector<int32_t>{1, 2}));
  EXPECT_THAT(writer->Close(), Not(IsOk()));
  writer.reset();
  EXPECT_EQ(std::fopen(absl::StrCat(dir_path, ".tmp").c_str(), "rb"), nullptr);
}

TEST(ArrayIoTest, RejectsCorruptTrailerAndTableOfContents) {
  const auto write_raw = [&](absl::string_view content) {
    const std::string corrupt_path = TestPath("corrupt");
    FILE* f = std::fopen(corrupt_path.c_str(), "wb");
    ABSL_CHECK(f != nullptr);
    ABSL_CHECK_EQ(std::fwrite(content.data(), 1, content.size(), f),
                  content.size());
    ABSL_CHECK_EQ(std::fclose(f), 0);
    EXPECT_THAT(Reader::Open(corrupt_path),
                StatusIs(absl::StatusCode::kDataLoss));
  };
  const uint64_t bad_toc = 1;
  const std::string trailer = absl::StrCat(
      absl::string_view(reinterpret_cast<const char*>(&bad_toc), 8),
      "GINGERIX");
  write_raw("");
  write_raw("short");
  write_raw(std::string(16, '\0'));
  write_raw(trailer);
  write_raw(absl::StrCat("\xff", trailer));
  TableOfContents out_of_bounds_toc;
  (*out_of_bounds_toc.mutable_entries())["x"].set_size(1);
  const std::string toc_bytes = out_of_bounds_toc.SerializeAsString();
  const uint64_t toc_size = toc_bytes.size();
  write_raw(
      absl::StrCat(toc_bytes,
                   absl::string_view(reinterpret_cast<const char*>(&toc_size),
                                     sizeof(toc_size)),
                   "GINGERIX"));
}

TEST(ArrayIoTest, RejectsNewerFormatVersion) {
  const std::string path = TestPath("newer_version");
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<Writer> writer, Writer::Create(path));
    ASSERT_OK(writer->Close());
  }
  ASSERT_OK(Reader::Open(path).status());

  TableOfContents toc;
  toc.set_format_version(1000);
  const std::string toc_bytes = toc.SerializeAsString();
  const uint64_t toc_size = toc_bytes.size();
  const std::string content =
      absl::StrCat(toc_bytes,
                   absl::string_view(reinterpret_cast<const char*>(&toc_size),
                                     sizeof(toc_size)),
                   "GINGERIX");
  FILE* f = std::fopen(path.c_str(), "wb");
  ASSERT_NE(f, nullptr);
  ASSERT_EQ(std::fwrite(content.data(), 1, content.size(), f), content.size());
  ASSERT_EQ(std::fclose(f), 0);
  EXPECT_THAT(Reader::Open(path),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace ginger::array_io
