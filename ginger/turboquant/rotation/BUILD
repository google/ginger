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
    name = "rotation",
    srcs = ["rotation.cc"],
    hdrs = ["rotation.h"],
    deps = [
        ":walsh_hadamard",
        "//ginger/internal:portable_random",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "rotation_test",
    size = "small",
    srcs = ["rotation_test.cc"],
    deps = [
        ":rotation",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "rotation_benchmark",
    srcs = ["rotation_benchmark.cc"],
    deps = [
        ":rotation",
        ":walsh_hadamard",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/types:span",
        "@google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "walsh_hadamard",
    srcs = ["walsh_hadamard.cc"],
    hdrs = ["walsh_hadamard.h"],
    deps = [
        "//ginger/turboquant/simd:vec_group",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "walsh_hadamard_test",
    size = "small",
    srcs = ["walsh_hadamard_test.cc"],
    deps = [
        ":walsh_hadamard",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
