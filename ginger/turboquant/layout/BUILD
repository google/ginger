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

load("@rules_cc//cc:defs.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],  # copybara:preserve
)

cc_library(
    name = "norm_storage",
    hdrs = ["norm_storage.h"],
    deps = [
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@highway//:hwy",
    ],
)

cc_library(
    name = "dataset",
    hdrs = ["dataset.h"],
    deps = [
        ":norm_storage",
        "//ginger:types",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/datapoint:encode_datapoint",
        "//ginger/turboquant/decoding:dequantize",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:scoring_accessors",
        "//ginger/turboquant/types:distance_metric",
        "//ginger/turboquant/types:quantization_bits",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "no_init_allocator",
    hdrs = ["no_init_allocator.h"],
    deps = [
        "@highway//:hwy",
    ],
)

cc_library(
    name = "dataset_soa_hdr",
    hdrs = ["dataset_soa.h"],
    deps = [
        ":dataset",
        ":no_init_allocator",
        ":norm_storage",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/datapoint:encode_datapoint",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:scoring_accessors",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "dataset_soa",
    srcs = ["dataset_soa.cc"],
    deps = [
        ":dataset",
        ":dataset_soa_hdr",
        ":norm_storage",
        "//ginger/internal:parallel_for_with_status",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/datapoint",
        "//ginger/turboquant/datapoint:encode_datapoint",
        "//ginger/turboquant/query",
        "//ginger/turboquant/scoring:score_4bit_soa",
        "//ginger/turboquant/scoring:score_extra_precision_soa",
        "//ginger/turboquant/scoring:scoring_accessors",
        "//ginger/turboquant/types:distance_metric",
        "//ginger/turboquant/types:quantization_bits",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@highway//:hwy",
    ],
    alwayslink = True,
)
