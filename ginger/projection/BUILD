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
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

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
    name = "pca_training",
    srcs = [
        "pca_training.cc",
    ],
    hdrs = [
        "pca_training.h",
    ],
    deps = [
        ":config_cc_proto",
        "//ginger:dataset",
        "//ginger:types",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@eigen",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
    ],
)

cc_library(
    name = "pca",
    srcs = [
        "pca.cc",
        "pca_float.cc",
        "pca_turboquant.cc",
    ],
    hdrs = [
        "pca.h",
        "pca_float.h",
        "pca_turboquant.h",
    ],
    deps = [
        ":config_cc_proto",
        ":pca_training",
        "//ginger:dataset",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/simd:vec_group",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "pca_test",
    srcs = ["pca_test.cc"],
    deps = [
        ":config_cc_proto",
        ":pca",
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "//ginger/internal:scheduler_reference_stub",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "pca_training_test",
    srcs = ["pca_training_test.cc"],
    deps = [
        ":config_cc_proto",
        ":pca_training",
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "//ginger/internal:scheduler_reference_stub",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "pca_benchmark",
    testonly = 1,
    srcs = ["pca_benchmark.cc"],
    deps = [
        ":pca",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@google_benchmark//:benchmark",
    ],
)
