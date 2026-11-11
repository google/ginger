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

include(CMakeParseArguments)

function(ginger_cc_test)
  if(NOT GINGER_BUILD_TESTING)
    return()
  endif()

  cmake_parse_arguments(
    GINGER_CC_TEST
    ""
    "NAME"
    "SRCS;COPTS;DEFINES;DEPS;LINKOPTS"
    ${ARGN}
  )

  set(_target "ginger_${GINGER_CC_TEST_NAME}")
  add_executable(${_target} ${GINGER_CC_TEST_SRCS})
  target_compile_features(${_target} PRIVATE cxx_std_20)
  target_include_directories(${_target} PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_BINARY_DIR}"
  )
  target_compile_options(${_target} PRIVATE
    ${GINGER_DEFAULT_COPTS}
    ${GINGER_CC_TEST_COPTS}
  )
  target_compile_definitions(${_target} PRIVATE ${GINGER_CC_TEST_DEFINES})
  target_link_libraries(${_target} PRIVATE
    ${GINGER_CC_TEST_DEPS}
    absl::status_matchers
    GTest::gmock
    GTest::gtest_main
    ${GINGER_DEFAULT_LINKOPTS}
    ${GINGER_CC_TEST_LINKOPTS}
  )
  # Like Bazel, give each test run its own empty TEST_TMPDIR.
  add_test(
    NAME ${_target}
    COMMAND ${CMAKE_COMMAND}
      -DTEST_BINARY=$<TARGET_FILE:${_target}>
      -DTEST_TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/${_target}_tmp
      -P ${PROJECT_SOURCE_DIR}/cmake/ginger_run_test.cmake
  )
endfunction()

function(ginger_cc_binary)
  cmake_parse_arguments(
    GINGER_CC_BIN
    ""
    "NAME"
    "SRCS;COPTS;DEFINES;DEPS;LINKOPTS"
    ${ARGN}
  )

  set(_target "${GINGER_CC_BIN_NAME}")
  add_executable(${_target} ${GINGER_CC_BIN_SRCS})
  target_compile_features(${_target} PRIVATE cxx_std_20)
  target_include_directories(${_target} PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_BINARY_DIR}"
  )
  target_compile_options(${_target} PRIVATE
    ${GINGER_DEFAULT_COPTS}
    ${GINGER_CC_BIN_COPTS}
  )
  target_compile_definitions(${_target} PRIVATE ${GINGER_CC_BIN_DEFINES})
  target_link_libraries(${_target} PRIVATE
    ${GINGER_CC_BIN_DEPS}
    ${GINGER_DEFAULT_LINKOPTS}
    ${GINGER_CC_BIN_LINKOPTS}
  )
endfunction()
