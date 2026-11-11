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
    name = "point_cloud_query",
    srcs = ["point_cloud_query.cc"],
    hdrs = ["point_cloud_query.h"],
    deps = [
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/many_to_many:decode_queries",
        "//ginger/turboquant/many_to_many:index_subset",
        "//ginger/turboquant/query",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "chamfer",
    srcs = ["chamfer.cc"],
    hdrs = ["chamfer.h"],
    textual_hdrs = ["chamfer.cc"],
    deps = [
        ":point_cloud_query",
        "//ginger:types",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/many_to_many:decode_db_inl",
        "//ginger/turboquant/many_to_many:decode_queries",
        "//ginger/turboquant/many_to_many:gemm_loop_inl",
        "//ginger/turboquant/types:distance_metric",
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
        ":point_cloud_query",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/many_to_many:index_subset",
        "//ginger/turboquant/query",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
