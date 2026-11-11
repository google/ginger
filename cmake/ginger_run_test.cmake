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

# Runs TEST_BINARY with TEST_TMPDIR set to a freshly emptied directory.
file(REMOVE_RECURSE "${TEST_TMPDIR}")
file(MAKE_DIRECTORY "${TEST_TMPDIR}")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env "TEST_TMPDIR=${TEST_TMPDIR}" "${TEST_BINARY}"
  RESULT_VARIABLE result
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "${TEST_BINARY} failed: ${result}")
endif()
