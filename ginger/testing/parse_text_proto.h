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

#ifndef GINGER_TESTING_PARSE_TEXT_PROTO_H_
#define GINGER_TESTING_PARSE_TEXT_PROTO_H_

#include <string>
#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
namespace google::protobuf::contrib::parse_proto {
class ParseTextProtoOrDieHelper {
 public:
  explicit ParseTextProtoOrDieHelper(absl::string_view text) : text_(text) {}
  template <typename T>
  operator T() const {
    T proto;
    ABSL_CHECK(google::protobuf::TextFormat::ParseFromString(std::string(text_), &proto));
    return proto;
  }
 private:
  absl::string_view text_;
};
inline ParseTextProtoOrDieHelper ParseTextProtoOrDie(absl::string_view text) {
  return ParseTextProtoOrDieHelper(text);
}
}  // namespace google::protobuf::contrib::parse_proto
#define PARSE_TEXT_PROTO(text) \
  ::google::protobuf::contrib::parse_proto::ParseTextProtoOrDie(text)

#endif  // GINGER_TESTING_PARSE_TEXT_PROTO_H_
