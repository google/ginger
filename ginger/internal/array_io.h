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

#ifndef GINGER_INTERNAL_ARRAY_IO_H_
#define GINGER_INTERNAL_ARRAY_IO_H_

// Storage of an index as a set of named entries: arrays of fixed-size records
// and text protos.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/message.h"

namespace ginger::array_io {

static_assert(std::endian::native == std::endian::little,
              "Ginger's on-disk formats assume a little-endian host.");

// Serializes records [begin, end) into `out`, which holds exactly
// (end - begin) * record_size bytes. Called concurrently on disjoint ranges.
using SerializeFn = absl::FunctionRef<absl::Status(size_t begin, size_t end,
                                                   absl::Span<char> out)>;

// Consumes records [begin, end) from `bytes`, which holds
// (end - begin) * record_size bytes and is only valid during the call.
// Called concurrently on disjoint ranges.
using ParseFn = absl::FunctionRef<absl::Status(size_t begin, size_t end,
                                               absl::string_view bytes)>;

// Writes an index of named entries. Entry names must not contain '/' and each
// name must be written at most once. Different entries may be written
// concurrently. `WriteBytes`/`WriteArray` entries must be read with
// `ReadBytes`/`ReadArray`; `WriteRecords` entries (`record_size > 0`) must be
// read with `ReadRecords` of the same `record_size`, or with
// `ReadBytes`/`ReadArray`. If any write fails, `Close()` also fails.
class Writer {
 public:
  // Opens a single-file POSIX writer that only appears at `path` once `Close()`
  // succeeds.
  static absl::StatusOr<std::unique_ptr<Writer>> Create(absl::string_view path);

  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  virtual ~Writer() = default;

  virtual absl::Status WriteBytes(absl::string_view name,
                                  absl::string_view bytes) = 0;
  virtual absl::Status WriteRecords(absl::string_view name, size_t num_records,
                                    size_t record_size,
                                    SerializeFn serialize) = 0;
  virtual absl::Status WriteTextProto(absl::string_view name,
                                      const google::protobuf::Message& proto) = 0;
  // Must be called once, after the last write.
  virtual absl::Status Close() = 0;

  template <typename T>
  absl::Status WriteArray(absl::string_view name, absl::Span<const T> data) {
    static_assert(std::is_trivially_copyable_v<T>);
    return WriteBytes(
        name, data.empty() ? absl::string_view()
                           : absl::string_view(
                                 reinterpret_cast<const char*>(data.data()),
                                 data.size() * sizeof(T)));
  }

 protected:
  Writer() = default;
};

// Returns whether `size` bytes hold records of `record_size` bytes whose number
// is the product of `dims`. Divides `size` instead of multiplying `dims`, which
// come from untrusted headers and could overflow.
inline bool SizeMatches(uint64_t size, size_t record_size,
                        std::initializer_list<uint64_t> dims) {
  if (size % record_size != 0) return false;
  uint64_t remaining = size / record_size;
  for (const uint64_t dim : dims) {
    if (dim == 0) return remaining == 0;
    if (remaining % dim != 0) return false;
    remaining /= dim;
  }
  return remaining == 1;
}

// Reads an index. `record_size` must be positive. All methods may be called
// concurrently; missing entries return `NotFoundError` and size mismatches or
// corrupt files return `DataLossError`.
// `Open` returns `FailedPreconditionError` for files written in a newer format
// version than this reader supports.
class Reader {
 public:
  // Opens a single-file POSIX index written by `Writer::Create`.
  static absl::StatusOr<std::unique_ptr<Reader>> Open(absl::string_view path);

  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  virtual ~Reader() = default;

  virtual absl::Status ReadBytes(absl::string_view name,
                                 absl::Span<char> dest) const = 0;
  virtual absl::Status ReadRecords(absl::string_view name, size_t num_records,
                                   size_t record_size, ParseFn parse) const = 0;
  virtual absl::Status ReadTextProto(absl::string_view name,
                                     google::protobuf::Message& proto) const = 0;

  // Checks that entry `name` holds records of `record_size` bytes whose number
  // is the product of `dims`. Call this before allocating memory for an entry
  // from header values, so that a corrupt header cannot cause a huge
  // allocation.
  virtual absl::Status CheckEntrySize(
      absl::string_view name, size_t record_size,
      std::initializer_list<uint64_t> dims) const = 0;

  template <typename T>
  absl::Status CheckArraySize(absl::string_view name,
                              std::initializer_list<uint64_t> dims) const {
    static_assert(std::is_trivially_copyable_v<T>);
    return CheckEntrySize(name, sizeof(T), dims);
  }

  template <typename T>
  absl::Status ReadArray(absl::string_view name, absl::Span<T> dest) const {
    static_assert(std::is_trivially_copyable_v<T>);
    return ReadBytes(
        name, dest.empty()
                  ? absl::Span<char>()
                  : absl::Span<char>(reinterpret_cast<char*>(dest.data()),
                                     dest.size() * sizeof(T)));
  }

 protected:
  Reader() = default;
};

}  // namespace ginger::array_io

#endif  // GINGER_INTERNAL_ARRAY_IO_H_
