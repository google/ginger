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
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "candidate_filter",
    hdrs = ["candidate_filter.h"],
    deps = [
        "//ginger:types",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "candidate_filter_test",
    srcs = ["candidate_filter_test.cc"],
    deps = [
        ":candidate_filter",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "jl",
    srcs = ["jl.cc"],
    hdrs = ["jl.h"],
    deps = [
        "//ginger:dataset",
        "//ginger:searcher_cc_proto",
        "//ginger:types",
        "//ginger/distance:hamming",
        "//ginger/internal:portable_random",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
        "@parlaylib//:parallel",
    ],
)

cc_test(
    name = "jl_test",
    srcs = ["jl_test.cc"],
    deps = [
        ":candidate_filter",
        ":jl",
        "//ginger:dataset",
        "//ginger:searcher_cc_proto",
        "//ginger:types",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
