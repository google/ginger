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

include(FetchContent)

find_package(Threads REQUIRED)

# Avoid building third-party unit tests when fetching dependencies.
set(_ginger_saved_build_testing "${BUILD_TESTING}")
set(BUILD_TESTING OFF)

# By default, fetch exact pinned versions to match MODULE.bazel unless the user
# explicitly opts into system packages.
if(NOT DEFINED FETCHCONTENT_TRY_FIND_PACKAGE_MODE)
  if(GINGER_USE_SYSTEM_DEPS)
    set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE OPT_IN)
  else()
    set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
  endif()
endif()

# 1. GoogleTest (must be available before Abseil when building test helpers)
if(GINGER_BUILD_TESTING)
  if(GINGER_USE_SYSTEM_DEPS AND NOT TARGET GTest::gtest_main)
    find_package(GTest QUIET)
  endif()
  if(NOT TARGET GTest::gtest_main)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.17.0
      GIT_SHALLOW TRUE
      FIND_PACKAGE_ARGS NAMES GTest
    )
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

# 2. Abseil
if(GINGER_USE_SYSTEM_DEPS AND NOT TARGET absl::base)
  find_package(absl QUIET)
endif()
if(NOT TARGET absl::base)
  set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
  set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  if(GINGER_BUILD_TESTING)
    set(ABSL_BUILD_TEST_HELPERS ON CACHE BOOL "" FORCE)
    set(ABSL_USE_EXTERNAL_GOOGLETEST ON CACHE BOOL "" FORCE)
    set(ABSL_FIND_GOOGLETEST OFF CACHE BOOL "" FORCE)
  endif()
  FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20260817.0
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS NAMES absl
  )
  FetchContent_MakeAvailable(abseil-cpp)
endif()

# 3. Protobuf
if(GINGER_USE_SYSTEM_DEPS AND NOT TARGET protobuf::libprotobuf)
  find_package(Protobuf CONFIG QUIET)
  if(NOT TARGET protobuf::libprotobuf)
    find_package(Protobuf QUIET)
  endif()
endif()
if(NOT TARGET protobuf::libprotobuf)
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_CONFORMANCE OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "" FORCE)
  set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
  set(protobuf_ABSL_PROVIDER "package" CACHE STRING "" FORCE)
  FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v35.1
    GIT_SHALLOW TRUE
    GIT_SUBMODULES ""
    FIND_PACKAGE_ARGS NAMES Protobuf protobuf
  )
  FetchContent_MakeAvailable(protobuf)
endif()

# 4. Eigen (header-only)
if(GINGER_USE_SYSTEM_DEPS AND NOT TARGET Eigen3::Eigen)
  find_package(Eigen3 QUIET)
endif()
if(NOT TARGET Eigen3::Eigen)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c
    SOURCE_SUBDIR non_existent_dir
    FIND_PACKAGE_ARGS NAMES Eigen3
  )
  FetchContent_MakeAvailable(eigen)
  if(NOT TARGET Eigen3::Eigen)
    add_library(ginger_eigen INTERFACE)
    target_include_directories(ginger_eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
    add_library(Eigen3::Eigen ALIAS ginger_eigen)
  endif()
endif()

# 5. Highway
if(GINGER_USE_SYSTEM_DEPS AND NOT TARGET hwy::hwy AND NOT TARGET hwy)
  find_package(hwy QUIET)
endif()
if(NOT TARGET hwy::hwy AND NOT TARGET hwy)
  set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.4.0
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS NAMES hwy
  )
  FetchContent_MakeAvailable(highway)
endif()

if(GINGER_BUILD_TESTING AND NOT TARGET hwy::hwy_test AND NOT TARGET hwy_test AND DEFINED highway_SOURCE_DIR)
  add_library(hwy_test STATIC "${highway_SOURCE_DIR}/hwy/tests/test_util.cc")
  target_link_libraries(hwy_test PUBLIC hwy GTest::gtest)
  target_include_directories(hwy_test PUBLIC "${highway_SOURCE_DIR}")
endif()

if(TARGET hwy AND NOT TARGET hwy::hwy)
  add_library(hwy::hwy ALIAS hwy)
endif()
if(TARGET hwy_contrib AND NOT TARGET hwy::hwy_contrib)
  add_library(hwy::hwy_contrib ALIAS hwy_contrib)
endif()
if(TARGET hwy_test AND NOT TARGET hwy::hwy_test)
  add_library(hwy::hwy_test ALIAS hwy_test)
endif()

# 6. OneTBB
if(GINGER_USE_SYSTEM_DEPS AND NOT TARGET TBB::tbb)
  find_package(TBB QUIET)
endif()
if(NOT TARGET TBB::tbb)
  set(TBB_TEST OFF CACHE BOOL "" FORCE)
  set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(TBB_STRICT OFF CACHE BOOL "" FORCE)
  set(TBBMALLOC_BUILD OFF CACHE BOOL "" FORCE)
  set(TBB_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    onetbb
    GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
    GIT_TAG v2022.3.0
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS NAMES TBB
  )
  FetchContent_MakeAvailable(onetbb)
endif()

# 7. ParlayLib (header-only)
if(GINGER_USE_SYSTEM_DEPS AND NOT TARGET parlay::parlay)
  find_package(parlaylib QUIET)
endif()
if(NOT TARGET parlay::parlay)
  FetchContent_Declare(
    parlaylib
    GIT_REPOSITORY https://github.com/ParAlg/parlaylib.git
    GIT_TAG 352f5367a6b3bd2f0c6f05882cdd02e978e349ea
    SOURCE_SUBDIR non_existent_dir
    FIND_PACKAGE_ARGS NAMES parlaylib
  )
  FetchContent_MakeAvailable(parlaylib)
  if(NOT TARGET parlay::parlay)
    add_library(ginger_parlaylib INTERFACE)
    target_include_directories(ginger_parlaylib SYSTEM INTERFACE
      "${parlaylib_SOURCE_DIR}/include"
      "${parlaylib_SOURCE_DIR}"
    )
    target_link_libraries(ginger_parlaylib INTERFACE Threads::Threads)
    add_library(parlay::parlay ALIAS ginger_parlaylib)
  endif()
endif()

# 8. KaMinPar
if(GINGER_USE_SYSTEM_DEPS AND NOT TARGET KaMinPar::KaMinPar)
  find_package(KaMinPar QUIET)
endif()
if(NOT TARGET KaMinPar::KaMinPar)
  set(KAMINPAR_BUILD_APPS OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_DISTRIBUTED OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_IO OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_WITH_CCACHE OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_WITH_WHFC OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_WITH_KASSERT OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_BUILD_WITH_SPARSEHASH OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_ENABLE_TBB_MALLOC OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_64BIT_EDGE_IDS ON CACHE BOOL "" FORCE)
  set(KAMINPAR_64BIT_WEIGHTS ON CACHE BOOL "" FORCE)
  set(KAMINPAR_ENABLE_TIMERS OFF CACHE BOOL "" FORCE)
  set(KAMINPAR_DOWNLOAD_TBB OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    kaminpar
    GIT_REPOSITORY https://github.com/KaHIP/KaMinPar.git
    GIT_TAG e212b7d056167d91027e1569cf7e5c109a91d153
    PATCH_COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_LIST_DIR}/patch_kaminpar.cmake"
    FIND_PACKAGE_ARGS NAMES KaMinPar
  )
  FetchContent_MakeAvailable(kaminpar)
  if(TARGET KaMinPar)
    target_include_directories(KaMinPar SYSTEM INTERFACE
      "${kaminpar_SOURCE_DIR}/kaminpar-shm"
      "${kaminpar_SOURCE_DIR}"
    )
    target_compile_definitions(KaMinPar INTERFACE
      KAMINPAR_64BIT_EDGE_IDS
      KAMINPAR_64BIT_WEIGHTS
    )
    if(NOT TARGET KaMinPar::KaMinPar)
      add_library(KaMinPar::KaMinPar ALIAS KaMinPar)
    endif()
  endif()
endif()

# 9. Google Benchmark (optional)
if(GINGER_BUILD_BENCHMARKS)
  if(GINGER_USE_SYSTEM_DEPS AND NOT TARGET benchmark::benchmark)
    find_package(benchmark QUIET)
  endif()
  if(NOT TARGET benchmark::benchmark)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      google_benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG v1.9.5
      GIT_SHALLOW TRUE
      FIND_PACKAGE_ARGS NAMES benchmark
    )
    FetchContent_MakeAvailable(google_benchmark)
  endif()
endif()

set(BUILD_TESTING "${_ginger_saved_build_testing}")
