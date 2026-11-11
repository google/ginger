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
    name = "index_builder",
    srcs = ["index_builder.cc"],
    hdrs = ["index_builder.h"],
    deps = [
        ":config_cc_proto",
        ":prune",
        "//ginger:build_progress",
        "//ginger:types",
        "//ginger/graph:flat_graph",
        "//ginger/internal:portable_random",
        "//ginger/internal:scheduler_reference_stub",
        "//ginger/search:beam_search",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
        "@parlaylib//:sequence",
    ],
)

cc_test(
    name = "index_builder_test",
    srcs = ["index_builder_test.cc"],
    deps = [
        ":config_cc_proto",
        ":index_builder",
        "//ginger:build_progress",
        "//ginger:robust_prune_config_cc_proto",
        "//ginger:types",
        "//ginger/graph:graph_stats",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "fixed_size_priority_queue",
    hdrs = ["fixed_size_priority_queue.h"],
    deps = [
        "@abseil-cpp//absl/log:absl_check",
    ],
)

cc_test(
    name = "fixed_size_priority_queue_test",
    srcs = ["fixed_size_priority_queue_test.cc"],
    deps = [
        ":fixed_size_priority_queue",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "prune",
    srcs = ["prune.cc"],
    hdrs = ["prune.h"],
    deps = [
        ":fixed_size_priority_queue",
        "//ginger:robust_prune_config_cc_proto",
        "//ginger:types",
        "//ginger/internal:portable_random",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "prune_test",
    srcs = ["prune_test.cc"],
    deps = [
        ":prune",
        "//ginger:robust_prune_config_cc_proto",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

proto_library(
    name = "config_proto",
    srcs = ["config.proto"],
    deps = ["//ginger:robust_prune_config_proto"],
)

cc_proto_library(
    name = "config_cc_proto",
    deps = [":config_proto"],
)
