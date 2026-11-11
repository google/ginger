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
    name = "index_subset",
    hdrs = ["index_subset.h"],
    deps = [
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "decode_queries",
    textual_hdrs = [
        "decode_queries-inl.h",
    ],
    deps = [
        ":index_subset",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "vec_group",
    textual_hdrs = [
        "vec_group-inl.h",
        "@highway//:hwy",
    ],
    deps = ["//ginger/turboquant/simd:vec_group"],
)

cc_library(
    name = "decode_db_inl",
    hdrs = ["decode_db-inl.h"],
    deps = [
        ":index_subset",
        ":vec_group",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/base:prefetch",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "accumulate_inl",
    hdrs = ["accumulate-inl.h"],
    deps = [
        ":vec_group",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "gemm_loop_inl",
    hdrs = ["gemm_loop-inl.h"],
    deps = [
        ":accumulate_inl",
        ":decode_queries",
        ":vec_group",
        "@abseil-cpp//absl/log:absl_check",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "gemm_loop_test",
    srcs = ["gemm_loop_test.cc"],
    deps = [
        ":decode_db_inl",
        ":decode_queries",
        ":gemm_loop_inl",
        ":index_subset",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/query",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)

cc_test(
    name = "accumulate_test",
    srcs = ["accumulate_test.cc"],
    deps = [
        ":accumulate_inl",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)

cc_test(
    name = "decode_db_test",
    srcs = ["decode_db_test.cc"],
    deps = [
        ":decode_db_inl",
        ":index_subset",
        ":vec_group",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)

cc_library(
    name = "vec_group_inl",
    hdrs = ["vec_group-inl.h"],
    deps = ["//ginger/turboquant/simd:vec_group"],
)

cc_test(
    name = "decode_queries_test",
    srcs = ["decode_queries_test.cc"],
    deps = [
        ":decode_queries",
        ":index_subset",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@highway//:hwy",
        "@highway//:hwy_test_util",
    ],
)

cc_library(
    name = "benchmark_util",
    hdrs = ["benchmark_util.h"],
    deps = [
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "many_to_many_turboquant_benchmark",
    srcs = ["many_to_many_turboquant_benchmark.cc"],
    deps = [
        ":benchmark_util",
        ":index_subset",
        ":many_to_many_turboquant",
        ":pretranspose",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@google_benchmark//:benchmark",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "many_to_many_turboquant",
    srcs = ["many_to_many_turboquant.cc"],
    hdrs = ["many_to_many_turboquant.h"],
    deps = [
        ":decode_db_inl",
        ":decode_queries",
        ":gemm_loop_inl",
        ":index_subset",
        ":pretranspose",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "many_to_many_topk",
    srcs = ["many_to_many_topk.cc"],
    hdrs = ["many_to_many_topk.h"],
    deps = [
        ":index_subset",
        ":many_to_many_turboquant",
        ":pretranspose",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
        "@highway//hwy/contrib/sort:vqsort_k32v32",
    ],
)

cc_binary(
    name = "many_to_many_topk_benchmark",
    srcs = ["many_to_many_topk_benchmark.cc"],
    deps = [
        ":benchmark_util",
        ":index_subset",
        ":many_to_many_topk",
        ":many_to_many_turboquant",
        ":pretranspose",
        "@abseil-cpp//absl/types:span",
        "@google_benchmark//:benchmark",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "many_to_many_topk_test",
    srcs = ["many_to_many_topk_test.cc"],
    deps = [
        ":index_subset",
        ":many_to_many_topk",
        ":many_to_many_turboquant",
        ":pretranspose",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "many_to_many_turboquant_test",
    srcs = ["many_to_many_turboquant_test.cc"],
    deps = [
        ":index_subset",
        ":many_to_many_turboquant",
        ":pretranspose",
        "//ginger/internal:gunit_support",
        "//ginger/internal:scheduler_reference_stub",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:scoring_scalar_reference",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "pretranspose",
    srcs = ["pretranspose.cc"],
    hdrs = ["pretranspose.h"],
    deps = [
        ":decode_db_inl",
        ":index_subset",
        ":vec_group",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "pretranspose_test",
    srcs = ["pretranspose_test.cc"],
    deps = [
        ":pretranspose",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
