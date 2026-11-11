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
    name = "distance_metric",
    hdrs = ["distance_metric.h"],
    deps = [
        "//ginger:searcher_cc_proto",
        "@abseil-cpp//absl/log:absl_log",
    ],
)

cc_test(
    name = "distance_metric_test",
    srcs = ["distance_metric_test.cc"],
    deps = [
        ":distance_metric",
        "//ginger:searcher_cc_proto",
        "//ginger/internal:gunit_support",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantization_bits",
    hdrs = ["quantization_bits.h"],
    deps = [
        "@abseil-cpp//absl/status",
    ],
)
