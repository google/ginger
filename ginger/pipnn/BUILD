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
    deps = ["//ginger:robust_prune_config_proto"],
)

cc_proto_library(
    name = "config_cc_proto",
    deps = [":config_proto"],
)

cc_library(
    name = "structured_random_matrix",
    srcs = ["structured_random_matrix.cc"],
    hdrs = ["structured_random_matrix.h"],
    deps = [
        "//ginger/turboquant/rotation:walsh_hadamard",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@eigen",
    ],
)

cc_test(
    name = "structured_random_matrix_test",
    srcs = ["structured_random_matrix_test.cc"],
    deps = [
        ":structured_random_matrix",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "simhash",
    srcs = ["simhash.cc"],
    hdrs = ["simhash.h"],
    deps = [
        ":structured_random_matrix",
        "//ginger:types",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@parlaylib//:parallel",
    ],
)

cc_library(
    name = "hash_prune",
    srcs = ["hash_prune.cc"],
    hdrs = ["hash_prune.h"],
    deps = [
        ":simhash",
        "//ginger:types",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@highway//:hwy",
        "@parlaylib//:parallel",
    ],
)

cc_library(
    name = "pipnn_impl",
    srcs = ["pipnn_impl.cc"],
    hdrs = ["pipnn_impl.h"],
    deps = [
        ":config_cc_proto",
        "//ginger:types",
        "//ginger/internal:parallel_for_with_status",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/many_to_many:index_subset",
        "//ginger/turboquant/many_to_many:many_to_many_topk",
        "//ginger/turboquant/many_to_many:pretranspose",
        "//ginger/turboquant/pipnn:assign_leaders",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
        "@parlaylib//:sequence",
    ],
)

cc_library(
    name = "pipnn_knn",
    srcs = ["pipnn_knn.cc"],
    hdrs = ["pipnn_knn.h"],
    deps = [
        ":config_cc_proto",
        ":pipnn_impl",
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/distance",
        "//ginger/internal:scheduler_reference_stub",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/many_to_many:index_subset",
        "//ginger/turboquant/many_to_many:many_to_many_topk",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:scoring_accessors",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
        "@highway//hwy/contrib/sort:vqsort",
        "@parlaylib//:parallel",
    ],
)

cc_library(
    name = "pipnn",
    srcs = ["pipnn.cc"],
    hdrs = ["pipnn.h"],
    deps = [
        ":config_cc_proto",
        ":hash_prune",
        ":pipnn_impl",
        ":simhash",
        "//ginger:robust_prune_config_cc_proto",
        "//ginger:types",
        "//ginger/graph:flat_graph",
        "//ginger/internal:parallel_for_with_status",
        "//ginger/internal:scheduler_reference_stub",
        "//ginger/search:search_functors",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/pipnn:process_leaf",
        "//ginger/turboquant/pipnn:robust_prune",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@parlaylib//:parallel",
        "@parlaylib//:primitives",
        "@parlaylib//:sequence",
    ],
)

cc_test(
    name = "simhash_test",
    srcs = ["simhash_test.cc"],
    deps = [
        ":simhash",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "hash_prune_test",
    srcs = ["hash_prune_test.cc"],
    deps = [
        ":hash_prune",
        ":simhash",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "pipnn_impl_test",
    srcs = ["pipnn_impl_test.cc"],
    deps = [
        ":config_cc_proto",
        ":pipnn_impl",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "//ginger/internal:scheduler_test_stub",
        "//ginger/testing:parse_text_proto",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@parlaylib//:sequence",
    ],
)

cc_test(
    name = "pipnn_knn_test",
    srcs = ["pipnn_knn_test.cc"],
    deps = [
        ":config_cc_proto",
        ":pipnn_knn",
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/distance",
        "//ginger/internal:gunit_support",
        "//ginger/internal:scheduler_test_stub",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "pipnn_test",
    srcs = ["pipnn_test.cc"],
    deps = [
        ":config_cc_proto",
        ":pipnn",
        "//ginger:dataset",
        "//ginger:robust_prune_config_cc_proto",
        "//ginger:searcher",
        "//ginger:searcher_cc_proto",
        "//ginger:types",
        "//ginger/graph:flat_graph",
        "//ginger/internal:gunit_support",
        "//ginger/internal:scheduler_test_stub",
        "//ginger/search:search_functors",
        "//ginger/testing:parse_text_proto",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
