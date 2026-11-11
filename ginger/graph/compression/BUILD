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

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "adaptive_gap_codec",
    srcs = ["adaptive_gap_codec.cc"],
    hdrs = ["adaptive_gap_codec.h"],
    deps = [
        "//ginger:types",
        "//ginger/graph:varint",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "adaptive_gap_codec_test",
    srcs = ["adaptive_gap_codec_test.cc"],
    deps = [
        ":adaptive_gap_codec",
        "//ginger:types",
        "//ginger/graph:varint",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)

cc_library(
    name = "streamvbyte_codec",
    srcs = ["streamvbyte_codec.cc"],
    hdrs = ["streamvbyte_codec.h"],
    deps = [
        "//ginger:types",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "streamvbyte_codec_test",
    srcs = ["streamvbyte_codec_test.cc"],
    deps = [
        ":streamvbyte_codec",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)

cc_library(
    name = "varint_codec",
    hdrs = ["varint_codec.h"],
    deps = [
        "//ginger:types",
        "//ginger/graph:varint",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "varint_codec_test",
    srcs = ["varint_codec_test.cc"],
    deps = [
        ":varint_codec",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "adaptive_gap_codec_benchmark",
    testonly = 1,
    srcs = ["adaptive_gap_codec_benchmark.cc"],
    deps = [
        ":adaptive_gap_codec",
        ":streamvbyte_codec",
        ":varint_codec",
        "//ginger:types",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@google_benchmark//:benchmark",
        "@highway//:hwy",
    ],
)
