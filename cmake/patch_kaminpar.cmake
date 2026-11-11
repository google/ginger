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

file(WRITE "cmake/modules/FindTBB.cmake" [=[
if(NOT TARGET TBB::tbb)
  add_library(TBB::tbb UNKNOWN IMPORTED)
  find_path(TBB_INCLUDE_DIR NAMES tbb/tbb.h oneapi/tbb.h PATH_SUFFIXES include)
  find_library(TBB_LIBRARY NAMES tbb PATH_SUFFIXES lib)
  set_target_properties(TBB::tbb PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}"
      IMPORTED_LOCATION "${TBB_LIBRARY}"
  )
endif()
set(TBB_FOUND TRUE)
set(TBB_VERSION "2022.3.0")
]=])

file(READ "CMakeLists.txt" _kaminpar_cmake)
if(NOT _kaminpar_cmake MATCHES "add_compile_options\\(-fexceptions\\)")
  file(WRITE "CMakeLists.txt" "add_compile_options(-fexceptions)\n${_kaminpar_cmake}")
endif()

file(READ "kaminpar-common/math.h" _kaminpar_math)
if(NOT _kaminpar_math MATCHES "if \\(arg == 0\\) return 0;")
  string(REPLACE
    "template <typename T> T floor_log2(const T arg) {"
    "template <typename T> T floor_log2(const T arg) { if (arg == 0) return 0;"
    _kaminpar_math "${_kaminpar_math}")
  file(WRITE "kaminpar-common/math.h" "${_kaminpar_math}")
endif()

file(READ "kaminpar-shm/partitioning/partition_utils.cc" _kaminpar_part_utils)
if(NOT _kaminpar_part_utils MATCHES "#include <cstdint>")
  string(REPLACE
    "const BlockID reversed_block ="
    "const uint64_t reversed_block = (uint64_t)"
    _kaminpar_part_utils "${_kaminpar_part_utils}")
  file(WRITE "kaminpar-shm/partitioning/partition_utils.cc" "#include <cstdint>\n${_kaminpar_part_utils}")
endif()

file(READ "kaminpar-common/graph_compression/compressed_neighborhoods.h" _kaminpar_comp_neigh)
if(NOT _kaminpar_comp_neigh MATCHES "bool has_intervals = false;")
  string(REPLACE
    "bool has_intervals;"
    "bool has_intervals = false;"
    _kaminpar_comp_neigh "${_kaminpar_comp_neigh}")
  file(WRITE "kaminpar-common/graph_compression/compressed_neighborhoods.h" "${_kaminpar_comp_neigh}")
endif()

file(READ "kaminpar-common/parallel/tbb_malloc.h" _kaminpar_tbb_malloc)
if(NOT _kaminpar_tbb_malloc MATCHES "aligned_alloc\\(alignof\\(T\\), nbytes\\)")
  string(REPLACE
    "ptr = static_cast<T *>(malloc(nbytes));"
    "nbytes = ((nbytes + alignof(T) - 1) / alignof(T)) * alignof(T); ptr = static_cast<T *>(aligned_alloc(alignof(T), nbytes));"
    _kaminpar_tbb_malloc "${_kaminpar_tbb_malloc}")
  file(WRITE "kaminpar-common/parallel/tbb_malloc.h" "${_kaminpar_tbb_malloc}")
endif()
