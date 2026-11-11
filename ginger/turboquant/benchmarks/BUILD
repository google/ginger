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

load("@rules_cc//cc:defs.bzl", "cc_binary")

package(
    default_visibility = ["//visibility:public"],  # copybara:preserve
)

cc_binary(
    name = "extra_precision_scoring_benchmark",
    srcs = ["extra_precision_scoring_benchmark.cc"],
    deps = [
        "//ginger/internal:scheduler_reference_stub",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/layout:norm_storage",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:scoring_accessors",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@google_benchmark//:benchmark",
        "@highway//:hwy",
        "@parlaylib//:parallel",
    ],
)
