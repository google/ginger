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
    name = "datapoint",
    hdrs = ["datapoint.h"],
    deps = [
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "encode_datapoint",
    srcs = ["encode_datapoint.cc"],
    hdrs = ["encode_datapoint.h"],
    deps = [
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/rotation",
        "//ginger/turboquant/types:quantization_bits",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen",
        "@highway//:hwy",
    ],
)

cc_test(
    name = "encode_datapoint_test",
    srcs = ["encode_datapoint_test.cc"],
    deps = [
        ":encode_datapoint",
        "//ginger/internal:gunit_support",
        "//ginger/turboquant/codebooks",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/rotation",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@highway//:hwy",
    ],
)
