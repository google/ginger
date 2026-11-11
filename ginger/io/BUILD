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
    name = "serialization",
    srcs = ["serialization.cc"],
    hdrs = ["serialization.h"],
    deps = [
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/layout:norm_storage",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/numeric:bits",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
    ],
)

proto_library(
    name = "io_proto",
    srcs = ["io.proto"],
    deps = [
        "//ginger:searcher_proto",
        "//ginger/projection:config_proto",
    ],
)

cc_proto_library(
    name = "io_cc_proto",
    deps = [":io_proto"],
)

cc_library(
    name = "index_io",
    srcs = ["index_io.cc"],
    hdrs = ["index_io.h"],
    deps = [
        ":io_cc_proto",
        ":serialization",
        "//ginger:dataset",
        "//ginger:searcher",
        "//ginger:searcher_cc_proto",
        "//ginger:types",
        "//ginger/candidate_filtering:jl",
        "//ginger/graph:compressed_graph",
        "//ginger/graph:compressed_offsets",
        "//ginger/graph:flat_graph",
        "//ginger/internal:array_io",
        "//ginger/internal:parallel_for_with_status",
        "//ginger/internal:searcher_access",
        "//ginger/projection:config_cc_proto",
        "//ginger/projection:pca",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/datapoint:encode_datapoint",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "io",
    srcs = ["io.cc"],
    hdrs = ["io.h"],
    deps = [
        ":index_io",
        "//ginger:searcher",
        "//ginger/graph:compressed_graph",
        "//ginger/graph:flat_graph",
        "//ginger/internal:array_io",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_test(
    name = "io_test",
    srcs = ["io_test.cc"],
    deps = [
        ":index_io",
        ":io",
        ":io_cc_proto",
        "//ginger:dataset",
        "//ginger:searcher",
        "//ginger:searcher_cc_proto",
        "//ginger:types",
        "//ginger/candidate_filtering:jl",
        "//ginger/graph:compressed_graph",
        "//ginger/graph:compressed_graph_builder",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
        "//ginger/internal:array_io",
        "//ginger/internal:gunit_support",
        "//ginger/internal:scheduler_reference_stub",
        "//ginger/internal:searcher_access",
        "//ginger/projection:config_cc_proto",
        "//ginger/projection:pca",
        "//ginger/testing:parse_text_proto",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@googletest//:gtest_main",
        "@protobuf",
    ],
)
