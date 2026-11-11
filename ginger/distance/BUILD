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
    name = "distance",
    srcs = ["distance.cc"],
    hdrs = ["distance.h"],
    deps = [
        "//ginger:dataset",
        "//ginger:types",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "distance_test",
    srcs = ["distance_test.cc"],
    deps = [
        ":distance",
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "distance_bytes",
    srcs = ["distance_bytes.cc"],
    hdrs = ["distance_bytes.h"],
    deps = [
        "//ginger:dataset",
        "//ginger:types",
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "distance_bytes_test",
    srcs = ["distance_bytes_test.cc"],
    deps = [
        ":distance_bytes",
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "distance_benchmark",
    testonly = 1,
    srcs = ["distance_benchmark.cc"],
    deps = [
        ":distance",
        "//ginger:dataset",
        "//ginger:types",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "distance_bytes_benchmark",
    testonly = 1,
    srcs = ["distance_bytes_benchmark.cc"],
    deps = [
        ":distance_bytes",
        "//ginger:dataset",
        "//ginger:types",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/types:span",
        "@google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "chamfer",
    srcs = ["chamfer.cc"],
    hdrs = ["chamfer.h"],
    deps = [
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/turboquant/simd:vec_group",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "chamfer_test",
    srcs = ["chamfer_test.cc"],
    deps = [
        ":chamfer",
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "chamfer_benchmark",
    testonly = 1,
    srcs = ["chamfer_benchmark.cc"],
    deps = [
        ":chamfer",
        "//ginger:types",
        "//ginger/turboquant/chamfer",
        "//ginger/turboquant/chamfer:point_cloud_query",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "hamming",
    srcs = ["hamming.cc"],
    hdrs = ["hamming.h"],
    deps = [
        "//ginger:types",
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/numeric:bits",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "hamming_test",
    srcs = ["hamming_test.cc"],
    deps = [
        ":hamming",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/numeric:bits",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)
