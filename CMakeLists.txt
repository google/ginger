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

cmake_minimum_required(VERSION 3.24)

project(
  ginger
  VERSION 0.1.0
  LANGUAGES CXX
)

# Ginger is intended to be consumed from source via FetchContent or
# add_subdirectory; disable install rules from subprojects.
set(CMAKE_SKIP_INSTALL_RULES ON)

option(GINGER_BUILD_TESTING "Build Ginger unit tests" ${PROJECT_IS_TOP_LEVEL})
option(GINGER_BUILD_EXAMPLES "Build Ginger example binaries" ${PROJECT_IS_TOP_LEVEL})
option(GINGER_BUILD_BENCHMARKS "Build Ginger benchmarks" OFF)
option(GINGER_ENABLE_MARCH_NATIVE "Compile with -march=native" ON)
option(
  GINGER_USE_SYSTEM_DEPS
  "Prefer system-installed packages via find_package before falling back to FetchContent"
  OFF
)
option(GINGER_USE_JEMALLOC "Link jemalloc into binaries that use Ginger" OFF)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GINGER_DEFAULT_COPTS
  -funsigned-char
  -fno-exceptions
)
if(GINGER_ENABLE_MARCH_NATIVE)
  list(APPEND GINGER_DEFAULT_COPTS -march=native)
endif()

if(GINGER_BUILD_TESTING)
  enable_testing()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ginger_deps)
include(ginger_helpers)

set(GINGER_DEFAULT_LINKOPTS
  Threads::Threads
)
if(UNIX AND NOT APPLE)
  list(APPEND GINGER_DEFAULT_LINKOPTS atomic)
endif()
# The ginger library is static, so its private link dependencies, including
# jemalloc, are linked into every binary that uses it.
if(GINGER_USE_JEMALLOC)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(JEMALLOC REQUIRED jemalloc)
  if(APPLE)
    list(APPEND GINGER_DEFAULT_LINKOPTS ${JEMALLOC_LINK_LIBRARIES})
  else()
    # jemalloc replaces malloc at load time, so a binary need not reference any
    # of its symbols. Keep it even when the linker defaults to --as-needed.
    list(APPEND GINGER_DEFAULT_LINKOPTS
      -Wl,--push-state,--no-as-needed
      ${JEMALLOC_LINK_LIBRARIES}
      -Wl,--pop-state
    )
  endif()
endif()

# DO NOT EDIT BELOW THIS LINE. Generated from the Bazel BUILD files.

# ==============================================================================
# Ginger Library
# ==============================================================================

set(_ginger_proto_srcs
  "ginger/internal/array_io.proto"
  "ginger/io/io.proto"
  "ginger/pipnn/config.proto"
  "ginger/projection/config.proto"
  "ginger/reordering/config.proto"
  "ginger/robust_prune_config.proto"
  "ginger/searcher.proto"
  "ginger/vamana/config.proto"
)
set(_ginger_pb_srcs "")
set(_ginger_pb_hdrs "")
foreach(_src IN LISTS _ginger_proto_srcs)
  get_filename_component(_dir "${_src}" DIRECTORY)
  get_filename_component(_we "${_src}" NAME_WE)
  set(_pb_cc "${PROJECT_BINARY_DIR}/${_dir}/${_we}.pb.cc")
  set(_pb_h "${PROJECT_BINARY_DIR}/${_dir}/${_we}.pb.h")
  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/${_dir}")
  add_custom_command(
    OUTPUT "${_pb_cc}" "${_pb_h}"
    COMMAND protobuf::protoc
    ARGS
      "--cpp_out=${PROJECT_BINARY_DIR}"
      -I "${PROJECT_SOURCE_DIR}"
      "${PROJECT_SOURCE_DIR}/${_src}"
    DEPENDS
      "${PROJECT_SOURCE_DIR}/${_src}"
      protobuf::protoc
    COMMENT "Running C++ protocol buffer compiler on ${_src}"
    VERBATIM
  )
  list(APPEND _ginger_pb_srcs "${_pb_cc}")
  list(APPEND _ginger_pb_hdrs "${_pb_h}")
endforeach()

add_library(ginger STATIC
  ${_ginger_pb_srcs}
  ${_ginger_pb_hdrs}
  "ginger/candidate_filtering/jl.cc"
  "ginger/config_recommender.cc"
  "ginger/distance/chamfer.cc"
  "ginger/distance/distance.cc"
  "ginger/distance/distance_bytes.cc"
  "ginger/distance/hamming.cc"
  "ginger/eval/recall.cc"
  "ginger/graph/compressed_graph.cc"
  "ginger/graph/compression/adaptive_gap_codec.cc"
  "ginger/graph/compression/streamvbyte_codec.cc"
  "ginger/graph/graph_stats.cc"
  "ginger/internal/array_io.cc"
  "ginger/io/index_io.cc"
  "ginger/io/io.cc"
  "ginger/io/serialization.cc"
  "ginger/pipnn/hash_prune.cc"
  "ginger/pipnn/pipnn.cc"
  "ginger/pipnn/pipnn_impl.cc"
  "ginger/pipnn/pipnn_knn.cc"
  "ginger/pipnn/simhash.cc"
  "ginger/pipnn/structured_random_matrix.cc"
  "ginger/projection/pca.cc"
  "ginger/projection/pca_float.cc"
  "ginger/projection/pca_training.cc"
  "ginger/projection/pca_turboquant.cc"
  "ginger/reordering/recursive_graph_bisection.cc"
  "ginger/reordering/recursive_graph_bisection_impl.cc"
  "ginger/reordering/recursive_graph_contraction.cc"
  "ginger/reordering/recursive_graph_contraction_impl.cc"
  "ginger/reordering/recursive_graph_partitioning.cc"
  "ginger/reordering/recursive_graph_partitioning_impl.cc"
  "ginger/reordering/reordering_utils.cc"
  "ginger/search/beam_search.cc"
  "ginger/searcher.cc"
  "ginger/searcher_builder.cc"
  "ginger/turboquant/chamfer/chamfer.cc"
  "ginger/turboquant/chamfer/point_cloud_query.cc"
  "ginger/turboquant/codec/codec.cc"
  "ginger/turboquant/datapoint/encode_datapoint.cc"
  "ginger/turboquant/decoding/dequantize.cc"
  "ginger/turboquant/layout/dataset_soa.cc"
  "ginger/turboquant/many_to_many/many_to_many_topk.cc"
  "ginger/turboquant/many_to_many/many_to_many_turboquant.cc"
  "ginger/turboquant/many_to_many/pretranspose.cc"
  "ginger/turboquant/pipnn/assign_leaders.cc"
  "ginger/turboquant/pipnn/process_leaf.cc"
  "ginger/turboquant/pipnn/robust_prune.cc"
  "ginger/turboquant/query/encode_query.cc"
  "ginger/turboquant/query/query.cc"
  "ginger/turboquant/rotation/rotation.cc"
  "ginger/turboquant/rotation/walsh_hadamard.cc"
  "ginger/turboquant/scoring/score_4bit_soa.cc"
  "ginger/turboquant/scoring/score_extra_precision_soa.cc"
  "ginger/vamana/index_builder.cc"
  "ginger/vamana/prune.cc"
  "ginger/build_progress.h"
  "ginger/candidate_filtering/candidate_filter.h"
  "ginger/candidate_filtering/jl.h"
  "ginger/config_recommender.h"
  "ginger/dataset.h"
  "ginger/distance/chamfer.h"
  "ginger/distance/distance.h"
  "ginger/distance/distance_bytes.h"
  "ginger/distance/hamming.h"
  "ginger/eval/recall.h"
  "ginger/ginger.h"
  "ginger/graph/compressed_graph.h"
  "ginger/graph/compressed_graph_builder.h"
  "ginger/graph/compressed_offsets.h"
  "ginger/graph/compression/adaptive_gap_codec.h"
  "ginger/graph/compression/streamvbyte_codec.h"
  "ginger/graph/compression/varint_codec.h"
  "ginger/graph/csr_graph.h"
  "ginger/graph/flat_graph.h"
  "ginger/graph/graph_stats.h"
  "ginger/graph/varint.h"
  "ginger/internal/array_io.h"
  "ginger/internal/parallel_for_with_status.h"
  "ginger/internal/portable_random.h"
  "ginger/internal/scheduler_reference_stub.h"
  "ginger/internal/searcher_access.h"
  "ginger/io/index_io.h"
  "ginger/io/io.h"
  "ginger/io/serialization.h"
  "ginger/pipnn/hash_prune.h"
  "ginger/pipnn/pipnn.h"
  "ginger/pipnn/pipnn_impl.h"
  "ginger/pipnn/pipnn_knn.h"
  "ginger/pipnn/simhash.h"
  "ginger/pipnn/structured_random_matrix.h"
  "ginger/projection/pca.h"
  "ginger/projection/pca_float.h"
  "ginger/projection/pca_training.h"
  "ginger/projection/pca_turboquant.h"
  "ginger/query.h"
  "ginger/reordering/recursive_graph_bisection.h"
  "ginger/reordering/recursive_graph_bisection_impl.h"
  "ginger/reordering/recursive_graph_contraction.h"
  "ginger/reordering/recursive_graph_contraction_impl.h"
  "ginger/reordering/recursive_graph_partitioning.h"
  "ginger/reordering/recursive_graph_partitioning_impl.h"
  "ginger/reordering/reordering_utils.h"
  "ginger/search/beam_search.h"
  "ginger/search/fast_visited_table.h"
  "ginger/search/search_functors.h"
  "ginger/search/visited_set.h"
  "ginger/searcher.h"
  "ginger/searcher_builder.h"
  "ginger/turboquant/chamfer/chamfer.h"
  "ginger/turboquant/chamfer/point_cloud_query.h"
  "ginger/turboquant/codebooks/centroid_tables.h"
  "ginger/turboquant/codebooks/codebooks.h"
  "ginger/turboquant/codebooks/codebooks_10bit.h"
  "ginger/turboquant/codebooks/codebooks_12bit.h"
  "ginger/turboquant/codebooks/codebooks_8bit.h"
  "ginger/turboquant/codebooks/piecewise_cubic.h"
  "ginger/turboquant/codec/codec.h"
  "ginger/turboquant/datapoint/datapoint.h"
  "ginger/turboquant/datapoint/encode_datapoint.h"
  "ginger/turboquant/decoding/dequantize.h"
  "ginger/turboquant/layout/dataset.h"
  "ginger/turboquant/layout/dataset_soa.h"
  "ginger/turboquant/layout/no_init_allocator.h"
  "ginger/turboquant/layout/norm_storage.h"
  "ginger/turboquant/many_to_many/accumulate-inl.h"
  "ginger/turboquant/many_to_many/decode_db-inl.h"
  "ginger/turboquant/many_to_many/decode_queries-inl.h"
  "ginger/turboquant/many_to_many/gemm_loop-inl.h"
  "ginger/turboquant/many_to_many/index_subset.h"
  "ginger/turboquant/many_to_many/many_to_many_topk.h"
  "ginger/turboquant/many_to_many/many_to_many_turboquant.h"
  "ginger/turboquant/many_to_many/pretranspose.h"
  "ginger/turboquant/many_to_many/vec_group-inl.h"
  "ginger/turboquant/pipnn/assign_leaders.h"
  "ginger/turboquant/pipnn/process_leaf.h"
  "ginger/turboquant/pipnn/robust_prune.h"
  "ginger/turboquant/pipnn/scatter_epilogue-inl.h"
  "ginger/turboquant/pipnn/topk_result.h"
  "ginger/turboquant/pipnn/topk_tracker-inl.h"
  "ginger/turboquant/query/encode_query.h"
  "ginger/turboquant/query/query.h"
  "ginger/turboquant/rotation/rotation.h"
  "ginger/turboquant/rotation/walsh_hadamard.h"
  "ginger/turboquant/scoring/prefetch.h"
  "ginger/turboquant/scoring/score_4bit_soa.h"
  "ginger/turboquant/scoring/score_extra_precision_soa.h"
  "ginger/turboquant/scoring/scoring_accessors.h"
  "ginger/turboquant/scoring/simd_helpers-inl.h"
  "ginger/turboquant/simd/vec_group-inl.h"
  "ginger/turboquant/types/distance_metric.h"
  "ginger/turboquant/types/quantization_bits.h"
  "ginger/types.h"
  "ginger/vamana/fixed_size_priority_queue.h"
  "ginger/vamana/index_builder.h"
  "ginger/vamana/prune.h"
)
add_library(ginger::ginger ALIAS ginger)
target_compile_features(ginger PUBLIC cxx_std_20)
target_include_directories(ginger PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
  "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
)
target_compile_options(ginger PRIVATE ${GINGER_DEFAULT_COPTS})
target_link_libraries(ginger
  PUBLIC
    absl::absl_check
    absl::absl_log
    absl::algorithm_container
    absl::base
    absl::bits
    absl::check
    absl::cleanup
    absl::core_headers
    absl::crc32c
    absl::flat_hash_map
    absl::flat_hash_set
    absl::function_ref
    absl::inlined_vector
    absl::log
    absl::memory
    absl::overload
    absl::prefetch
    absl::random_random
    absl::span
    absl::status
    absl::status_macros
    absl::statusor
    absl::str_format
    absl::string_view
    absl::strings
    absl::synchronization
    absl::time
    Eigen3::Eigen
    hwy::hwy
    hwy::hwy_contrib
    KaMinPar::KaMinPar
    parlay::parlay
    protobuf::libprotobuf
  PRIVATE
    ${GINGER_DEFAULT_LINKOPTS}
)

# ==============================================================================
# Unit Tests
# ==============================================================================

if(GINGER_BUILD_TESTING)
  ginger_cc_test(
    NAME dataset_test
    SRCS "ginger/dataset_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
  )

  ginger_cc_test(
    NAME config_recommender_test
    SRCS "ginger/config_recommender_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
  )

  ginger_cc_test(
    NAME searcher_test
    SRCS "ginger/searcher_test.cc"
    DEPS
      ginger::ginger
      absl::algorithm_container
      absl::span
      absl::status
      absl::statusor
      Eigen3::Eigen
  )

  ginger_cc_test(
    NAME searcher_builder_test
    SRCS "ginger/searcher_builder_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
      absl::status_macros
      absl::statusor
      Eigen3::Eigen
  )

  ginger_cc_test(
    NAME candidate_filtering_candidate_filter_test
    SRCS "ginger/candidate_filtering/candidate_filter_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )

  ginger_cc_test(
    NAME candidate_filtering_jl_test
    SRCS "ginger/candidate_filtering/jl_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
  )

  ginger_cc_test(
    NAME distance_test
    SRCS "ginger/distance/distance_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )

  ginger_cc_test(
    NAME distance_bytes_test
    SRCS "ginger/distance/distance_bytes_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )

  ginger_cc_test(
    NAME distance_chamfer_test
    SRCS "ginger/distance/chamfer_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )

  ginger_cc_test(
    NAME distance_hamming_test
    SRCS "ginger/distance/hamming_test.cc"
    DEPS
      ginger::ginger
      absl::bits
      absl::span
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME eval_recall_test
    SRCS "ginger/eval/recall_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
      parlay::parlay
  )

  ginger_cc_test(
    NAME graph_compressed_graph_test
    SRCS "ginger/graph/compressed_graph_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::strings
  )

  ginger_cc_test(
    NAME graph_flat_graph_test
    SRCS "ginger/graph/flat_graph_test.cc"
    DEPS ginger::ginger
  )

  ginger_cc_test(
    NAME graph_compressed_offsets_test
    SRCS "ginger/graph/compressed_offsets_test.cc"
    DEPS ginger::ginger
  )

  ginger_cc_test(
    NAME graph_compression_adaptive_gap_codec_test
    SRCS "ginger/graph/compression/adaptive_gap_codec_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::statusor
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME graph_compression_streamvbyte_codec_test
    SRCS "ginger/graph/compression/streamvbyte_codec_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME graph_compression_varint_codec_test
    SRCS "ginger/graph/compression/varint_codec_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )

  ginger_cc_test(
    NAME internal_portable_random_test
    SRCS "ginger/internal/portable_random_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )

  ginger_cc_test(
    NAME internal_array_io_test
    SRCS "ginger/internal/array_io_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::span
      absl::status
      absl::string_view
      absl::strings
  )

  ginger_cc_test(
    NAME io_test
    SRCS "ginger/io/io_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::core_headers
      absl::flat_hash_map
      absl::span
      absl::status
      absl::status_macros
      absl::string_view
      absl::strings
      absl::synchronization
      Eigen3::Eigen
      protobuf::libprotobuf
  )

  ginger_cc_test(
    NAME pipnn_structured_random_matrix_test
    SRCS "ginger/pipnn/structured_random_matrix_test.cc"
    DEPS
      ginger::ginger
      Eigen3::Eigen
  )

  ginger_cc_test(
    NAME pipnn_simhash_test
    SRCS "ginger/pipnn/simhash_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )

  ginger_cc_test(
    NAME pipnn_hash_prune_test
    SRCS "ginger/pipnn/hash_prune_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )

  ginger_cc_test(
    NAME pipnn_impl_test
    SRCS "ginger/pipnn/pipnn_impl_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::span
      absl::status
      absl::strings
      absl::synchronization
      GTest::gtest
      parlay::parlay
      protobuf::libprotobuf
  )

  ginger_cc_test(
    NAME pipnn_knn_test
    SRCS "ginger/pipnn/pipnn_knn_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::algorithm_container
      absl::flat_hash_set
      absl::span
      absl::status
      GTest::gtest
  )

  ginger_cc_test(
    NAME pipnn_test
    SRCS "ginger/pipnn/pipnn_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::algorithm_container
      absl::check
      absl::log
      absl::span
      absl::status
      absl::strings
      GTest::gtest
      protobuf::libprotobuf
  )

  ginger_cc_test(
    NAME projection_pca_test
    SRCS "ginger/projection/pca_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
      absl::statusor
      Eigen3::Eigen
  )

  ginger_cc_test(
    NAME projection_pca_training_test
    SRCS "ginger/projection/pca_training_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
      absl::statusor
      Eigen3::Eigen
  )

  ginger_cc_test(
    NAME reordering_recursive_graph_bisection_impl_test
    SRCS "ginger/reordering/recursive_graph_bisection_impl_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )

  ginger_cc_test(
    NAME reordering_recursive_graph_contraction_impl_test
    SRCS "ginger/reordering/recursive_graph_contraction_impl_test.cc"
    DEPS
      ginger::ginger
      absl::span
      parlay::parlay
  )

  ginger_cc_test(
    NAME reordering_recursive_graph_partitioning_impl_test
    SRCS "ginger/reordering/recursive_graph_partitioning_impl_test.cc"
    DEPS
      ginger::ginger
      absl::algorithm_container
      absl::random_random
      absl::span
  )

  ginger_cc_test(
    NAME reordering_utils_test
    SRCS "ginger/reordering/reordering_utils_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status_macros
      absl::statusor
  )

  ginger_cc_test(
    NAME search_fast_visited_table_test
    SRCS "ginger/search/fast_visited_table_test.cc"
    DEPS ginger::ginger
  )

  ginger_cc_test(
    NAME search_beam_search_test
    SRCS "ginger/search/beam_search_test.cc"
    DEPS
      ginger::ginger
      absl::core_headers
      absl::span
  )

  ginger_cc_test(
    NAME search_functors_test
    SRCS "ginger/search/search_functors_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::statusor
  )

  ginger_cc_test(
    NAME turboquant_chamfer_test
    SRCS "ginger/turboquant/chamfer/chamfer_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::span
      absl::strings
  )

  ginger_cc_test(
    NAME turboquant_codebooks_piecewise_cubic_test
    SRCS "ginger/turboquant/codebooks/piecewise_cubic_test.cc"
    DEPS ginger::ginger
  )

  ginger_cc_test(
    NAME turboquant_datapoint_encode_datapoint_test
    SRCS "ginger/turboquant/datapoint/encode_datapoint_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
      absl::strings
      hwy::hwy
  )

  ginger_cc_test(
    NAME turboquant_many_to_many_gemm_loop_test
    SRCS "ginger/turboquant/many_to_many/gemm_loop_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::span
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME turboquant_many_to_many_accumulate_test
    SRCS "ginger/turboquant/many_to_many/accumulate_test.cc"
    DEPS
      ginger::ginger
      absl::algorithm_container
      absl::span
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME turboquant_many_to_many_decode_db_test
    SRCS "ginger/turboquant/many_to_many/decode_db_test.cc"
    DEPS
      ginger::ginger
      absl::span
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME turboquant_many_to_many_decode_queries_test
    SRCS "ginger/turboquant/many_to_many/decode_queries_test.cc"
    DEPS
      ginger::ginger
      absl::span
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME turboquant_many_to_many_topk_test
    SRCS "ginger/turboquant/many_to_many/many_to_many_topk_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::base
      absl::random_random
      absl::span
  )

  ginger_cc_test(
    NAME turboquant_many_to_many_turboquant_test
    SRCS "ginger/turboquant/many_to_many/many_to_many_turboquant_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::absl_log
      absl::flat_hash_map
      absl::no_destructor
      absl::random_random
      absl::span
  )

  ginger_cc_test(
    NAME turboquant_many_to_many_pretranspose_test
    SRCS "ginger/turboquant/many_to_many/pretranspose_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::random_random
      absl::span
  )

  ginger_cc_test(
    NAME turboquant_pipnn_topk_tracker_test
    SRCS "ginger/turboquant/pipnn/topk_tracker_test.cc"
    DEPS
      ginger::ginger
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME turboquant_pipnn_scatter_epilogue_test
    SRCS "ginger/turboquant/pipnn/scatter_epilogue_test.cc"
    DEPS
      ginger::ginger
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME turboquant_pipnn_process_leaf_test
    SRCS "ginger/turboquant/pipnn/process_leaf_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::span
  )

  ginger_cc_test(
    NAME turboquant_pipnn_assign_leaders_test
    SRCS "ginger/turboquant/pipnn/assign_leaders_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::algorithm_container
      absl::random_random
      absl::span
      absl::strings
  )

  ginger_cc_test(
    NAME turboquant_pipnn_robust_prune_test
    SRCS "ginger/turboquant/pipnn/robust_prune_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::algorithm_container
      absl::span
  )

  ginger_cc_test(
    NAME turboquant_query_encode_query_test
    SRCS "ginger/turboquant/query/encode_query_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
  )

  ginger_cc_test(
    NAME turboquant_rotation_test
    SRCS "ginger/turboquant/rotation/rotation_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::span
      absl::status
      Eigen3::Eigen
  )

  ginger_cc_test(
    NAME turboquant_rotation_walsh_hadamard_test
    SRCS "ginger/turboquant/rotation/walsh_hadamard_test.cc"
    DEPS
      ginger::ginger
      absl::random_random
      absl::span
  )

  ginger_cc_test(
    NAME turboquant_scoring_scaling_storage_mode_scoring_test
    SRCS "ginger/turboquant/scoring/scaling_storage_mode_scoring_test.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::absl_log
      absl::core_headers
      absl::random_random
      absl::span
      absl::status
  )

  ginger_cc_test(
    NAME turboquant_simd_vec_group_test
    SRCS "ginger/turboquant/simd/vec_group_test.cc"
    DEPS
      ginger::ginger
      hwy::hwy
      hwy::hwy_test
  )

  ginger_cc_test(
    NAME turboquant_types_distance_metric_test
    SRCS "ginger/turboquant/types/distance_metric_test.cc"
    DEPS ginger::ginger
  )

  ginger_cc_test(
    NAME vamana_index_builder_test
    SRCS "ginger/vamana/index_builder_test.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::status
      absl::time
  )

  ginger_cc_test(
    NAME vamana_fixed_size_priority_queue_test
    SRCS "ginger/vamana/fixed_size_priority_queue_test.cc"
    DEPS ginger::ginger
  )

  ginger_cc_test(
    NAME vamana_prune_test
    SRCS "ginger/vamana/prune_test.cc"
    DEPS
      ginger::ginger
      absl::span
  )
endif()

# ==============================================================================
# Examples
# ==============================================================================

if(GINGER_BUILD_EXAMPLES)
  ginger_cc_binary(
    NAME ginger_example
    SRCS "ginger/examples/ginger_example.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::flags_parse
      absl::log_initialize
      absl::span
      absl::status
      absl::status_macros
  )

  ginger_cc_binary(
    NAME ginger_byod_example
    SRCS "ginger/examples/ginger_byod_example.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::check
      absl::flags_parse
      absl::log_initialize
      absl::span
      absl::status
      absl::status_macros
      protobuf::libprotobuf
  )

  ginger_cc_binary(
    NAME pipnn_knn_example
    SRCS "ginger/examples/pipnn_knn_example.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::flags_parse
      absl::log_initialize
      absl::span
      absl::status
      absl::status_macros
      protobuf::libprotobuf
  )
endif()

# ==============================================================================
# Benchmarks
# ==============================================================================

if(GINGER_BUILD_BENCHMARKS)
  ginger_cc_binary(
    NAME distance_benchmark
    SRCS "ginger/distance/distance_benchmark.cc"
    DEPS
      ginger::ginger
      absl::random_random
      absl::span
      benchmark::benchmark
      Eigen3::Eigen
  )

  ginger_cc_binary(
    NAME distance_bytes_benchmark
    SRCS "ginger/distance/distance_bytes_benchmark.cc"
    DEPS
      ginger::ginger
      absl::random_random
      absl::span
      benchmark::benchmark
  )

  ginger_cc_binary(
    NAME chamfer_benchmark
    SRCS "ginger/distance/chamfer_benchmark.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::random_random
      absl::span
      absl::strings
      benchmark::benchmark
      Eigen3::Eigen
  )

  ginger_cc_binary(
    NAME adaptive_gap_codec_benchmark
    SRCS "ginger/graph/compression/adaptive_gap_codec_benchmark.cc"
    DEPS
      ginger::ginger
      absl::span
      absl::statusor
      benchmark::benchmark
      hwy::hwy
  )

  ginger_cc_binary(
    NAME pca_benchmark
    SRCS "ginger/projection/pca_benchmark.cc"
    DEPS
      ginger::ginger
      absl::span
      benchmark::benchmark
      Eigen3::Eigen
  )

  ginger_cc_binary(
    NAME extra_precision_scoring_benchmark
    SRCS "ginger/turboquant/benchmarks/extra_precision_scoring_benchmark.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::no_destructor
      absl::random_random
      absl::span
      absl::strings
      benchmark::benchmark
      hwy::hwy
      parlay::parlay
  )

  ginger_cc_binary(
    NAME many_to_many_turboquant_benchmark
    SRCS "ginger/turboquant/many_to_many/many_to_many_turboquant_benchmark.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::core_headers
      absl::random_random
      absl::span
      absl::synchronization
      benchmark::benchmark
      Eigen3::Eigen
      hwy::hwy
  )

  ginger_cc_binary(
    NAME many_to_many_topk_benchmark
    SRCS "ginger/turboquant/many_to_many/many_to_many_topk_benchmark.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::core_headers
      absl::random_random
      absl::span
      absl::synchronization
      benchmark::benchmark
      hwy::hwy
  )

  ginger_cc_binary(
    NAME process_leaf_benchmark
    SRCS "ginger/turboquant/pipnn/process_leaf_benchmark.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::no_destructor
      absl::span
      benchmark::benchmark
      hwy::hwy
      parlay::parlay
  )

  ginger_cc_binary(
    NAME assign_leaders_benchmark
    SRCS "ginger/turboquant/pipnn/assign_leaders_benchmark.cc"
    DEPS
      ginger::ginger
      absl::absl_check
      absl::core_headers
      absl::random_random
      absl::span
      absl::synchronization
      benchmark::benchmark
      hwy::hwy
  )

  ginger_cc_binary(
    NAME rotation_benchmark
    SRCS "ginger/turboquant/rotation/rotation_benchmark.cc"
    DEPS
      ginger::ginger
      absl::random_random
      absl::span
      benchmark::benchmark
  )
endif()
