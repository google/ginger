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

// All entries of an index are stored in a single file:
//
//   entry | entry | ... | TableOfContents | Trailer
//
// Entries are written in parallel with pwrite to a temporary file, which is
// renamed to the index path on Close(), and read in parallel with pread.

#include "ginger/internal/array_io.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "ginger/internal/array_io.pb.h"
#include "ginger/internal/parallel_for_with_status.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace ginger::array_io {
namespace {

// Entries are written and read in parallel blocks of about this size.
constexpr size_t kBlockBytes = 4 << 20;

constexpr char kMagic[8] = {'G', 'I', 'N', 'G', 'E', 'R', 'I', 'X'};

// Written to `TableOfContents.format_version`. Increment it whenever the
// layout of any entry changes, so older readers reject the file.
constexpr uint32_t kFormatVersion = 1;

// The last bytes of the file. The table of contents ends where it starts.
struct Trailer {
  uint64_t toc_size;
  char magic[8];
};
static_assert(sizeof(Trailer) == 16);

absl::Status ErrnoError(absl::string_view message, absl::string_view path) {
  return absl::ErrnoToStatus(errno, absl::StrCat(message, ": ", path));
}

absl::Status CorruptError(absl::string_view path) {
  return absl::DataLossError(absl::StrCat("Not a Ginger index: ", path));
}

absl::Status WriteAll(int fd, absl::string_view data, uint64_t offset,
                      absl::string_view path) {
  while (!data.empty()) {
    const ssize_t n = pwrite(fd, data.data(), data.size(), offset);
    if (n < 0) {
      if (errno == EINTR) continue;
      return ErrnoError("Failed to write file", path);
    }
    data.remove_prefix(n);
    offset += n;
  }
  return absl::OkStatus();
}

absl::Status ReadAll(int fd, absl::Span<char> dest, uint64_t offset,
                     absl::string_view path) {
  while (!dest.empty()) {
    const ssize_t n = pread(fd, dest.data(), dest.size(), offset);
    if (n < 0) {
      if (errno == EINTR) continue;
      return ErrnoError("Failed to read file", path);
    }
    if (n == 0) {
      return absl::DataLossError(
          absl::StrCat("Unexpected end of file: ", path));
    }
    dest = dest.subspan(n);
    offset += n;
  }
  return absl::OkStatus();
}

// Calls `f(begin, end)` in parallel for blocks of records covering
// [0, num_records).
template <typename F>
absl::Status ForEachBlock(size_t num_records, size_t record_size, F f) {
  ABSL_DCHECK_GT(record_size, 0);
  const size_t block_records = std::max<size_t>(1, kBlockBytes / record_size);
  const size_t num_blocks = (num_records + block_records - 1) / block_records;
  return parallel_for_with_status(
      0, num_blocks,
      [&](size_t block) {
        const size_t begin = block * block_records;
        return f(begin, std::min(begin + block_records, num_records));
      },
      /*granularity=*/1);
}

class FileWriter final : public Writer {
 public:
  FileWriter(absl::string_view path, std::string temp_path, int fd)
      : path_(path), temp_path_(std::move(temp_path)), fd_(fd) {}

  ~FileWriter() override {
    if (fd_ >= 0) close(fd_);
    if (!temp_path_.empty()) unlink(temp_path_.c_str());
  }

  absl::Status WriteBytes(absl::string_view name,
                          absl::string_view bytes) override {
    const uint64_t offset = AddEntry(name, bytes.size());
    return RecordStatus(
        ForEachBlock(bytes.size(), 1, [&](size_t begin, size_t end) {
          return WriteAll(fd_, bytes.substr(begin, end - begin), offset + begin,
                          path_);
        }));
  }

  absl::Status WriteRecords(absl::string_view name, size_t num_records,
                            size_t record_size,
                            SerializeFn serialize) override {
    const uint64_t offset = AddEntry(name, num_records * record_size);
    return RecordStatus(ForEachBlock(
        num_records, record_size,
        [&](size_t begin, size_t end) -> absl::Status {
          const size_t size = (end - begin) * record_size;
          auto buffer = std::make_unique_for_overwrite<char[]>(size);
          ABSL_RETURN_IF_ERROR(
              serialize(begin, end, absl::MakeSpan(buffer.get(), size)));
          return WriteAll(fd_, absl::string_view(buffer.get(), size),
                          offset + begin * record_size, path_);
        }));
  }

  absl::Status WriteTextProto(absl::string_view name,
                              const google::protobuf::Message& proto) override {
    std::string text;
    if (!google::protobuf::TextFormat::PrintToString(proto, &text)) {
      return RecordStatus(absl::InternalError(
          absl::StrCat("Failed to format text proto ", name)));
    }
    return WriteBytes(name, text);
  }

  absl::Status Close() override {
    absl::MutexLock lock(mu_);
    ABSL_RETURN_IF_ERROR(status_);
    toc_.set_format_version(kFormatVersion);
    std::string tail = toc_.SerializeAsString();
    Trailer trailer = {.toc_size = tail.size()};
    std::memcpy(trailer.magic, kMagic, sizeof(kMagic));
    tail.append(reinterpret_cast<const char*>(&trailer), sizeof(trailer));
    ABSL_RETURN_IF_ERROR(WriteAll(fd_, tail, end_, path_));
    if (close(std::exchange(fd_, -1)) != 0) {
      return ErrnoError("Failed to close file", path_);
    }
    if (rename(temp_path_.c_str(), path_.c_str()) != 0) {
      return ErrnoError("Failed to rename file", path_);
    }
    temp_path_.clear();
    return absl::OkStatus();
  }

 private:
  // Reserves `size` bytes for entry `name` and returns their offset.
  uint64_t AddEntry(absl::string_view name, uint64_t size) {
    absl::MutexLock lock(mu_);
    const uint64_t offset = std::exchange(end_, end_ + size);
    auto [it, inserted] = toc_.mutable_entries()->try_emplace(name);
    ABSL_DCHECK(inserted) << "Duplicate entry: " << name;
    it->second.set_offset(offset);
    it->second.set_size(size);
    return offset;
  }

  absl::Status RecordStatus(absl::Status status) {
    if (!status.ok()) {
      absl::MutexLock lock(mu_);
      status_.Update(status);
    }
    return status;
  }

  const std::string path_;
  std::string temp_path_;
  int fd_;
  absl::Mutex mu_;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  TableOfContents toc_ ABSL_GUARDED_BY(mu_);
  uint64_t end_ ABSL_GUARDED_BY(mu_) = 0;
};

class FileReader final : public Reader {
 public:
  FileReader(absl::string_view path, int fd) : path_(path), fd_(fd) {}

  ~FileReader() override { close(fd_); }

  absl::Status ReadTableOfContents(uint64_t file_size) {
    Trailer trailer;
    const uint64_t toc_end = file_size - sizeof(trailer);
    ABSL_RETURN_IF_ERROR(ReadAll(
        fd_, absl::MakeSpan(reinterpret_cast<char*>(&trailer), sizeof(trailer)),
        toc_end, path_));
    if (std::memcmp(trailer.magic, kMagic, sizeof(kMagic)) != 0 ||
        trailer.toc_size > toc_end) {
      return CorruptError(path_);
    }
    const uint64_t toc_offset = toc_end - trailer.toc_size;
    std::string toc_bytes(trailer.toc_size, '\0');
    ABSL_RETURN_IF_ERROR(
        ReadAll(fd_, absl::MakeSpan(toc_bytes), toc_offset, path_));
    if (!toc_.ParseFromString(toc_bytes)) {
      return CorruptError(path_);
    }
    if (toc_.format_version() > kFormatVersion) {
      return absl::FailedPreconditionError(absl::StrCat(
          path_, " has index format version ", toc_.format_version(),
          ", but this version of Ginger only reads versions up to ",
          kFormatVersion, "."));
    }
    for (const auto& [name, range] : toc_.entries()) {
      if (range.offset() > toc_offset ||
          range.size() > toc_offset - range.offset()) {
        return CorruptError(path_);
      }
    }
    return absl::OkStatus();
  }

  absl::Status ReadBytes(absl::string_view name,
                         absl::Span<char> dest) const override {
    ABSL_ASSIGN_OR_RETURN(const uint64_t offset,
                          SizedEntryOffset(name, dest.size()));
    return ForEachBlock(dest.size(), 1, [&](size_t begin, size_t end) {
      return ReadAll(fd_, dest.subspan(begin, end - begin), offset + begin,
                     path_);
    });
  }

  absl::Status ReadRecords(absl::string_view name, size_t num_records,
                           size_t record_size, ParseFn parse) const override {
    ABSL_ASSIGN_OR_RETURN(const uint64_t offset,
                          SizedEntryOffset(name, num_records * record_size));
    return ForEachBlock(
        num_records, record_size,
        [&](size_t begin, size_t end) -> absl::Status {
          const size_t size = (end - begin) * record_size;
          auto buffer = std::make_unique_for_overwrite<char[]>(size);
          ABSL_RETURN_IF_ERROR(ReadAll(fd_, absl::MakeSpan(buffer.get(), size),
                                       offset + begin * record_size, path_));
          return parse(begin, end, absl::string_view(buffer.get(), size));
        });
  }

  absl::Status CheckEntrySize(
      absl::string_view name, size_t record_size,
      std::initializer_list<uint64_t> dims) const override {
    ABSL_ASSIGN_OR_RETURN(const TableOfContents::Range range, Entry(name));
    if (!SizeMatches(range.size(), record_size, dims)) {
      return absl::DataLossError(
          absl::StrCat("Entry ", name, " of ", path_, " has ", range.size(),
                       " bytes, which does not match its header."));
    }
    return absl::OkStatus();
  }

  absl::Status ReadTextProto(absl::string_view name,
                             google::protobuf::Message& proto) const override {
    ABSL_ASSIGN_OR_RETURN(const TableOfContents::Range range, Entry(name));
    std::string text(range.size(), '\0');
    ABSL_RETURN_IF_ERROR(
        ReadAll(fd_, absl::MakeSpan(text), range.offset(), path_));
    if (!google::protobuf::TextFormat::ParseFromString(text, &proto)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to parse text proto entry ", name, " of ", path_));
    }
    return absl::OkStatus();
  }

 private:
  absl::StatusOr<TableOfContents::Range> Entry(absl::string_view name) const {
    const auto it = toc_.entries().find(name);
    if (it == toc_.entries().end()) {
      return absl::NotFoundError(
          absl::StrCat("No entry ", name, " in ", path_));
    }
    return it->second;
  }

  absl::StatusOr<uint64_t> SizedEntryOffset(absl::string_view name,
                                            size_t expected_size) const {
    ABSL_ASSIGN_OR_RETURN(const TableOfContents::Range range, Entry(name));
    if (range.size() != expected_size) {
      return absl::DataLossError(
          absl::StrCat("Expected ", expected_size, " bytes but found ",
                       range.size(), " bytes in entry ", name, " of ", path_));
    }
    return range.offset();
  }

  const std::string path_;
  const int fd_;
  TableOfContents toc_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Writer>> Writer::Create(absl::string_view path) {
  std::string temp_path = absl::StrCat(path, ".tmp");
  const int fd =
      open(temp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) return ErrnoError("Failed to open file for writing", path);
  return std::make_unique<FileWriter>(path, std::move(temp_path), fd);
}

absl::StatusOr<std::unique_ptr<Reader>> Reader::Open(absl::string_view path) {
  const std::string filename(path);
  const int fd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return ErrnoError("Failed to open file", path);
  auto reader = std::make_unique<FileReader>(path, fd);
  struct stat st;
  if (fstat(fd, &st) != 0) return ErrnoError("Failed to stat file", path);
  if (st.st_size < static_cast<off_t>(sizeof(Trailer))) {
    return CorruptError(path);
  }
  ABSL_RETURN_IF_ERROR(reader->ReadTableOfContents(st.st_size));
  return reader;
}

}  // namespace ginger::array_io
