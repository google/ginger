# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(
    default_visibility = ["//visibility:public"],  # copybara:preserve
)

cc_library(
    name = "query",
    srcs = ["query.cc"],
    hdrs = ["query.h"],
    deps = [
        ":encode_query",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/datapoint:encode_datapoint",
        "//ginger/turboquant/decoding:dequantize",
        "//ginger/turboquant/types:distance_metric",
        "//ginger/turboquant/types:quantization_bits",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "encode_query",
    srcs = ["encode_query.cc"],
    hdrs = ["encode_query.h"],
    deps = [
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/rotation",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "encode_query_test",
    srcs = ["encode_query_test.cc"],
    deps = [
        ":encode_query",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/rotation",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
