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
    default_visibility = ["//visibility:public"],  # copybara:preserve
)

cc_library(
    name = "prefetch",
    hdrs = ["prefetch.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:prefetch",
    ],
)

cc_library(
    name = "scoring_accessors",
    hdrs = ["scoring_accessors.h"],
    deps = [
        "//ginger:types",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "simd_helpers",
    hdrs = ["simd_helpers-inl.h"],
    deps = [
        "//ginger/turboquant/codebooks",
        "@abseil-cpp//absl/log:absl_check",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "score_4bit_soa",
    srcs = ["score_4bit_soa.cc"],
    hdrs = ["score_4bit_soa.h"],
    deps = [
        ":prefetch",
        ":scoring_accessors",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/query",
        "@abseil-cpp//absl/log:absl_check",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "score_extra_precision_soa",
    srcs = ["score_extra_precision_soa.cc"],
    hdrs = ["score_extra_precision_soa.h"],
    deps = [
        ":prefetch",
        ":scoring_accessors",
        ":simd_helpers",
        "//ginger/turboquant/codebooks:centroid_tables",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/query",
        "//ginger/turboquant/simd:vec_group",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "scoring_scalar_reference",
    testonly = True,
    hdrs = ["scoring_scalar_reference.h"],
    deps = [
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/codebooks:centroid_tables",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/query",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "scaling_storage_mode_scoring_test",
    srcs = ["scaling_storage_mode_scoring_test.cc"],
    deps = [
        ":scoring_accessors",
        ":scoring_scalar_reference",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/query",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@highway//:abort_header_only",
    ],
)
