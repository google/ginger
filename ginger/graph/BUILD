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
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "csr_graph",
    hdrs = ["csr_graph.h"],
    deps = [
        "//ginger:types",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "flat_graph",
    hdrs = ["flat_graph.h"],
    deps = [
        "//ginger:types",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
        "@parlaylib//:parallel",
    ],
)

cc_library(
    name = "graph_stats",
    srcs = ["graph_stats.cc"],
    hdrs = ["graph_stats.h"],
    deps = [
        ":flat_graph",
        "//ginger:types",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
        "@parlaylib//:sequence",
    ],
)

cc_library(
    name = "varint",
    hdrs = ["varint.h"],
)

cc_library(
    name = "compressed_offsets",
    hdrs = ["compressed_offsets.h"],
    deps = [
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
    ],
)

cc_library(
    name = "compressed_graph",
    srcs = ["compressed_graph.cc"],
    hdrs = ["compressed_graph.h"],
    deps = [
        ":compressed_offsets",
        "//ginger:types",
        "//ginger/graph/compression:adaptive_gap_codec",
        "//ginger/graph/compression:varint_codec",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "compressed_graph_builder",
    hdrs = ["compressed_graph_builder.h"],
    deps = [
        ":compressed_graph",
        "//ginger:types",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
        "@parlaylib//:slice",
    ],
)

cc_test(
    name = "compressed_graph_test",
    srcs = ["compressed_graph_test.cc"],
    deps = [
        ":compressed_graph",
        ":compressed_graph_builder",
        ":compressed_offsets",
        ":csr_graph",
        "//ginger:types",
        "//ginger/graph/compression:streamvbyte_codec",
        "//ginger/graph/compression:varint_codec",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "flat_graph_test",
    srcs = ["flat_graph_test.cc"],
    deps = [
        ":flat_graph",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "compressed_offsets_test",
    srcs = ["compressed_offsets_test.cc"],
    deps = [
        ":compressed_offsets",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
    ],
)
