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
    default_visibility = ["//visibility:public"],  # copybara:preserve
)

cc_library(
    name = "topk_result",
    hdrs = ["topk_result.h"],
)

cc_library(
    name = "topk_tracker_hdr",
    textual_hdrs = ["topk_tracker-inl.h"],
    deps = [
        ":topk_result",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "topk_tracker_test",
    srcs = ["topk_tracker_test.cc"],
    deps = [
        ":topk_result",
        ":topk_tracker_hdr",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)

cc_library(
    name = "assign_leaders",
    srcs = ["assign_leaders.cc"],
    hdrs = ["assign_leaders.h"],
    deps = [
        ":topk_result",
        ":topk_tracker_hdr",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/many_to_many:decode_queries",
        "//ginger/turboquant/many_to_many:gemm_loop_inl",
        "//ginger/turboquant/many_to_many:index_subset",
        "//ginger/turboquant/many_to_many:pretranspose",
        "//ginger/turboquant/simd:vec_group",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "process_leaf",
    srcs = ["process_leaf.cc"],
    hdrs = ["process_leaf.h"],
    textual_hdrs = ["scatter_epilogue-inl.h"],
    deps = [
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/many_to_many:accumulate_inl",
        "//ginger/turboquant/many_to_many:decode_db_inl",
        "//ginger/turboquant/many_to_many:decode_queries",
        "//ginger/turboquant/many_to_many:gemm_loop_inl",
        "//ginger/turboquant/many_to_many:index_subset",
        "//ginger/turboquant/query",
        "//ginger/turboquant/simd:vec_group",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "scatter_epilogue_test",
    srcs = ["scatter_epilogue_test.cc"],
    deps = [
        ":process_leaf",  # buildcleaner: keep
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)

cc_test(
    name = "process_leaf_test",
    srcs = ["process_leaf_test.cc"],
    deps = [
        ":process_leaf",  # buildcleaner: keep
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:scoring_accessors",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "process_leaf_benchmark",
    srcs = ["process_leaf_benchmark.cc"],
    deps = [
        ":process_leaf",  # buildcleaner: keep
        "//ginger/internal:scheduler_reference_stub",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@google_benchmark//:benchmark",
        "@highway//:hwy",
        "@parlaylib//:parallel",
    ],
)

cc_test(
    name = "assign_leaders_test",
    srcs = ["assign_leaders_test.cc"],
    deps = [
        ":assign_leaders",  # buildcleaner: keep
        ":topk_result",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/many_to_many:index_subset",
        "//ginger/turboquant/many_to_many:many_to_many_turboquant",
        "//ginger/turboquant/many_to_many:pretranspose",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "assign_leaders_benchmark",
    srcs = ["assign_leaders_benchmark.cc"],
    deps = [
        ":assign_leaders",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/many_to_many:benchmark_util",
        "//ginger/turboquant/many_to_many:index_subset",
        "//ginger/turboquant/many_to_many:many_to_many_topk",
        "//ginger/turboquant/many_to_many:pretranspose",
        "@abseil-cpp//absl/types:span",
        "@google_benchmark//:benchmark",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "robust_prune",
    srcs = ["robust_prune.cc"],
    hdrs = ["robust_prune.h"],
    deps = [
        "//ginger:robust_prune_config_cc_proto",
        "//ginger:types",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/layout:dataset_soa",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/many_to_many:decode_queries",
        "//ginger/turboquant/simd:vec_group",
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
        "@highway//hwy/contrib/sort:vqsort",
    ],
)

cc_test(
    name = "robust_prune_test",
    srcs = ["robust_prune_test.cc"],
    deps = [
        ":robust_prune",  # buildcleaner: keep
        "//ginger:robust_prune_config_cc_proto",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:scoring_accessors",
        "//ginger/turboquant/types:distance_metric",
        "//ginger/vamana:prune",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
