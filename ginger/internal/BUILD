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
    name = "parallel_for_with_status",
    hdrs = ["parallel_for_with_status.h"],
    deps = [
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@parlaylib//:parallel",
    ],
)

cc_library(
    name = "searcher_access",
    hdrs = ["searcher_access.h"],
    deps = [
        "//ginger:dataset",
        "//ginger:searcher",
        "//ginger/candidate_filtering:jl",
        "//ginger/projection:pca",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
    ],
)

cc_library(
    name = "portable_random",
    hdrs = ["portable_random.h"],
    deps = [
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "portable_random_test",
    srcs = ["portable_random_test.cc"],
    deps = [
        ":portable_random",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "scheduler_reference_stub",
    hdrs = ["scheduler_reference_stub.h"],
)

cc_library(
    name = "gunit_support",
    testonly = True,
    hdrs = ["gunit_support.h"],
    deps = [
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "scheduler_test_stub",
    testonly = True,
    hdrs = ["scheduler_test_stub.h"],
    deps = ["@googletest//:gtest"],
)

proto_library(
    name = "array_io_proto",
    srcs = ["array_io.proto"],
)

cc_proto_library(
    name = "array_io_cc_proto",
    deps = [":array_io_proto"],
)

cc_library(
    name = "array_io",
    srcs = ["array_io.cc"],
    hdrs = ["array_io.h"],
    deps = [
        ":array_io_cc_proto",
        ":parallel_for_with_status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@protobuf",
    ],
)

cc_test(
    name = "array_io_test",
    srcs = ["array_io_test.cc"],
    deps = [
        ":array_io",
        ":array_io_cc_proto",
        ":parallel_for_with_status",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
