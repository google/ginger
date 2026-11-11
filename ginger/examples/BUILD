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
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "ginger_example",
    srcs = ["ginger_example.cc"],
    deps = [
        "//ginger",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_binary(
    name = "ginger_byod_example",
    srcs = ["ginger_byod_example.cc"],
    deps = [
        "//ginger:types",
        "//ginger/search:beam_search",
        "//ginger/vamana:config_cc_proto",
        "//ginger/vamana:index_builder",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/types:span",
        "@protobuf",
    ],
)

cc_binary(
    name = "pipnn_knn_example",
    srcs = ["pipnn_knn_example.cc"],
    deps = [
        "//ginger:dataset",
        "//ginger:types",
        "//ginger/pipnn:config_cc_proto",
        "//ginger/pipnn:pipnn_knn",
        "//ginger/turboquant/codec",
        "//ginger/turboquant/layout:dataset_soa",  # buildcleaner: keep
        "//ginger/turboquant/layout:dataset_soa_hdr",
        "//ginger/turboquant/types:distance_metric",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/types:span",
        "@protobuf",
    ],
)
