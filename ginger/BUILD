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

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_library(
    name = "build_progress",
    hdrs = ["build_progress.h"],
    deps = [
        ":types",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "dataset",
    hdrs = ["dataset.h"],
    deps = [
        ":types",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "dataset_test",
    srcs = ["dataset_test.cc"],
    deps = [
        ":dataset",
        ":types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "query",
    hdrs = ["query.h"],
    deps = [
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/query",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "searcher",
    srcs = [
        "searcher.cc",
        "searcher_builder.cc",
    ],
    hdrs = [
        "searcher.h",
        "searcher_builder.h",
    ],
    deps = [
        ":build_progress",
        ":dataset",
        ":query",
        ":robust_prune_config_cc_proto",
        ":searcher_cc_proto",
        ":types",
        "//ginger/candidate_filtering:candidate_filter",
        "//ginger/candidate_filtering:jl",
        "//ginger/distance",
        "//ginger/graph:compressed_graph",
        "//ginger/graph:compressed_graph_builder",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
        "//ginger/graph:graph_stats",
        "//ginger/internal:parallel_for_with_status",
        "//ginger/pipnn",
        "//ginger/pipnn:config_cc_proto",
        "//ginger/projection:pca",
        "//ginger/reordering:recursive_graph_bisection",
        "//ginger/reordering:recursive_graph_contraction",
        "//ginger/reordering:recursive_graph_partitioning",
        "//ginger/reordering:reordering_utils",
        "//ginger/search:beam_search",
        "//ginger/search:search_functors",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:scoring_accessors",
        "//ginger/turboquant/types:distance_metric",
        "//ginger/turboquant/types:quantization_bits",
        "//ginger/vamana:index_builder",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/crc:crc32c",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
    ],
)

cc_library(
    name = "config_recommender",
    srcs = ["config_recommender.cc"],
    hdrs = ["config_recommender.h"],
    deps = [
        ":robust_prune_config_cc_proto",
        ":searcher_cc_proto",
        "//ginger/pipnn:config_cc_proto",
        "//ginger/projection:config_cc_proto",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

# Umbrella target for Ginger's main API (see ginger.h).
cc_library(
    name = "ginger",
    hdrs = ["ginger.h"],
    deps = [
        ":config_recommender",
        ":dataset",
        ":searcher",
        ":searcher_cc_proto",
        ":types",
        "//ginger/io",
    ],
)

cc_test(
    name = "config_recommender_test",
    srcs = ["config_recommender_test.cc"],
    deps = [
        ":config_recommender",
        ":dataset",
        ":searcher",
        ":searcher_cc_proto",
        ":types",
        "//ginger/internal:gunit_support",
        "//ginger/internal:scheduler_reference_stub",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "searcher_test",
    srcs = ["searcher_test.cc"],
    deps = [
        ":build_progress",
        ":dataset",
        ":searcher",
        ":searcher_cc_proto",
        ":types",
        "//ginger/candidate_filtering:jl",
        "//ginger/internal:gunit_support",
        "//ginger/internal:portable_random",
        "//ginger/internal:scheduler_reference_stub",
        "//ginger/projection:config_cc_proto",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/query",
        "//ginger/turboquant/types:distance_metric",
        "//ginger/vamana:config_cc_proto",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "searcher_builder_test",
    srcs = ["searcher_builder_test.cc"],
    deps = [
        ":build_progress",
        ":dataset",
        ":robust_prune_config_cc_proto",
        ":searcher",
        ":searcher_cc_proto",
        ":types",
        "//ginger/candidate_filtering:jl",
        "//ginger/graph:compressed_graph",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
        "//ginger/internal:gunit_support",
        "//ginger/internal:scheduler_reference_stub",
        "//ginger/pipnn:config_cc_proto",
        "//ginger/projection:config_cc_proto",
        "//ginger/projection:pca",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/types:distance_metric",
        "//ginger/vamana:config_cc_proto",
        "//ginger/vamana:index_builder",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

proto_library(
    name = "robust_prune_config_proto",
    srcs = ["robust_prune_config.proto"],
)

cc_proto_library(
    name = "robust_prune_config_cc_proto",
    deps = [":robust_prune_config_proto"],
)

proto_library(
    name = "searcher_proto",
    srcs = ["searcher.proto"],
    deps = [
        "//ginger/pipnn:config_proto",
        "//ginger/projection:config_proto",
        "//ginger/vamana:config_proto",
    ],
)

cc_proto_library(
    name = "searcher_cc_proto",
    deps = [":searcher_proto"],
)
