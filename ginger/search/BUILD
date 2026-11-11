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
    name = "fast_visited_table",
    hdrs = ["fast_visited_table.h"],
    deps = [
        "//ginger:types",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "fast_visited_table_test",
    srcs = ["fast_visited_table_test.cc"],
    deps = [
        ":fast_visited_table",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "visited_set",
    hdrs = ["visited_set.h"],
    deps = [
        ":fast_visited_table",
        "//ginger:types",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_check",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "beam_search",
    srcs = ["beam_search.cc"],
    hdrs = ["beam_search.h"],
    deps = [
        ":visited_set",
        "//ginger:types",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "beam_search_test",
    srcs = ["beam_search_test.cc"],
    deps = [
        ":beam_search",
        ":visited_set",
        "//ginger:types",
        "//ginger/graph:compressed_graph",
        "//ginger/graph:compressed_graph_builder",
        "//ginger/graph:csr_graph",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "search_functors",
    hdrs = ["search_functors.h"],
    deps = [
        "//ginger:dataset",
        "//ginger:searcher_cc_proto",
        "//ginger:types",
        "//ginger/distance",
        "//ginger/distance:distance_bytes",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:scoring_accessors",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "search_functors_test",
    srcs = ["search_functors_test.cc"],
    deps = [
        ":search_functors",
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/query",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
