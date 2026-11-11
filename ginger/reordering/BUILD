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

proto_library(
    name = "config_proto",
    srcs = ["config.proto"],
)

cc_proto_library(
    name = "config_cc_proto",
    deps = [":config_proto"],
)

cc_library(
    name = "recursive_graph_bisection_impl",
    srcs = ["recursive_graph_bisection_impl.cc"],
    hdrs = ["recursive_graph_bisection_impl.h"],
    deps = [
        ":config_cc_proto",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
        "//ginger/internal:scheduler_reference_stub",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@parlaylib//:parallel",
    ],
)

cc_library(
    name = "recursive_graph_bisection",
    srcs = ["recursive_graph_bisection.cc"],
    hdrs = ["recursive_graph_bisection.h"],
    deps = [
        ":config_cc_proto",
        ":recursive_graph_bisection_impl",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
    ],
)

cc_test(
    name = "recursive_graph_bisection_impl_test",
    srcs = ["recursive_graph_bisection_impl_test.cc"],
    deps = [
        ":recursive_graph_bisection_impl",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "recursive_graph_contraction",
    srcs = ["recursive_graph_contraction.cc"],
    hdrs = ["recursive_graph_contraction.h"],
    deps = [
        ":config_cc_proto",
        ":recursive_graph_contraction_impl",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
    ],
)

cc_library(
    name = "recursive_graph_contraction_impl",
    srcs = ["recursive_graph_contraction_impl.cc"],
    hdrs = ["recursive_graph_contraction_impl.h"],
    linkopts = ["-latomic"],
    deps = [
        ":config_cc_proto",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
        "//ginger/internal:scheduler_reference_stub",
        "@abseil-cpp//absl/types:span",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
        "@parlaylib//:random",
        "@parlaylib//:sequence",
        "@parlaylib//:utilities",
    ],
)

cc_test(
    name = "recursive_graph_contraction_impl_test",
    srcs = ["recursive_graph_contraction_impl_test.cc"],
    deps = [
        ":recursive_graph_contraction_impl",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@parlaylib//:sequence",
    ],
)

cc_library(
    name = "recursive_graph_partitioning_impl",
    srcs = ["recursive_graph_partitioning_impl.cc"],
    hdrs = ["recursive_graph_partitioning_impl.h"],
    deps = [
        ":config_cc_proto",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
        "//ginger/internal:scheduler_reference_stub",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@kaminpar",
        "@parlaylib//:delayed_sequence",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
        "@parlaylib//:sequence",
        "@parlaylib//:slice",
        "@parlaylib//:thread_specific",
    ],
)

cc_library(
    name = "recursive_graph_partitioning",
    srcs = ["recursive_graph_partitioning.cc"],
    hdrs = ["recursive_graph_partitioning.h"],
    deps = [
        ":config_cc_proto",
        ":recursive_graph_partitioning_impl",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
    ],
)

cc_test(
    name = "recursive_graph_partitioning_impl_test",
    srcs = ["recursive_graph_partitioning_impl_test.cc"],
    deps = [
        ":config_cc_proto",
        ":recursive_graph_partitioning_impl",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "reordering_utils",
    srcs = ["reordering_utils.cc"],
    hdrs = ["reordering_utils.h"],
    deps = [
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
        "//ginger/internal:parallel_for_with_status",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
    ],
)

cc_test(
    name = "reordering_utils_test",
    srcs = ["reordering_utils_test.cc"],
    deps = [
        ":reordering_utils",
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/graph:csr_graph",
        "//ginger/graph:flat_graph",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
