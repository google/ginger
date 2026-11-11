# Compiling Ginger

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This guide covers compiling Ginger from source with Bazel, running the test
suite and example programs, and adding Ginger as a Bzlmod dependency. It also
lists the CMake options and shows how to add Ginger to a CMake project.

## Requirements

-   **Bazel:** A recent Bazel release with Bzlmod enabled.
-   **C++20 compiler:** GCC 13+ or Clang 17+. When building against libc++,
    libc++ 19+ is required.
-   **CMake and Make:** Must be installed on the host machine; Bazel invokes
    them automatically via `rules_foreign_cc` to compile the KaMinPar
    dependency.

## Target CPU

Both builds compile with `-march=native` by default, so the SIMD kernels use
every instruction set the build machine supports. The resulting binaries may
not run on older CPUs. Turn this off with `-DGINGER_ENABLE_MARCH_NATIVE=OFF` in
CMake, or by removing the flag from `.bazelrc`.

## Compiling and testing

Compile and run the test suite with:

```sh
bazel build -c opt //...
bazel test -c opt //...
```

Always pass `-c opt` when benchmarking or measuring search and index build
performance.

## Running the examples

Run the examples in [`examples/`](../examples/) with `bazel run -c opt`:

```sh
bazel run -c opt //ginger/examples:ginger_example
bazel run -c opt //ginger/examples:ginger_byod_example
bazel run -c opt //ginger/examples:pipnn_knn_example
```

-   `//ginger/examples:ginger_example`: Builds a `GingerSearcher` over synthetic
    vectors and runs nearest-neighbor queries.
-   `//ginger/examples:ginger_byod_example`: Builds a Vamana index and runs beam
    search with a custom distance callback.
-   `//ginger/examples:pipnn_knn_example`: Constructs an approximate k-NN graph
    with PiPNN.

## Using Ginger in your project

### 1. Add Ginger to `MODULE.bazel`

Declare Ginger as a Bzlmod dependency in your project's `MODULE.bazel`:

```starlark
bazel_dep(name = "ginger", version = "0.1.0")
git_override(
    module_name = "ginger",
    remote = "https://github.com/google/ginger.git",
    commit = "<commit_sha>",
)
```

### 2. Depend on Ginger targets in `BUILD.bazel`

Add the required Ginger targets to your `cc_library` or `cc_binary` (and ensure
your project's `.bazelrc` enables C++20 with `build --cxxopt=-std=c++20` and
`build --host_cxxopt=-std=c++20`):

```starlark
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "my_app",
    srcs = ["my_app.cc"],
    deps = [
        "@ginger//ginger",
    ],
)
```

In your C++ source files, include headers under the `ginger/` prefix:

```cpp
#include "ginger/ginger.h"
```

## CMake

The [top-level README](../../README.md) shows how to configure, build, and test
Ginger with CMake.

### CMake options

<!-- mdformat off -->
Option | Default | Description
------ | ------- | -----------
`GINGER_BUILD_TESTING` | `ON` (top-level) / `OFF` (subproject) | Build Ginger unit tests and register them with CTest.
`GINGER_BUILD_EXAMPLES` | `ON` (top-level) / `OFF` (subproject) | Build example binaries (`ginger_example`, `ginger_byod_example`, `pipnn_knn_example`).
`GINGER_BUILD_BENCHMARKS` | `OFF` | Build Google Benchmark binaries.
`GINGER_ENABLE_MARCH_NATIVE` | `ON` | Compile with `-march=native` for host SIMD optimizations.
`GINGER_USE_SYSTEM_DEPS` | `OFF` | Prefer system-installed packages via `find_package` before falling back to `FetchContent`.
`GINGER_USE_JEMALLOC` | `OFF` | Link jemalloc (found via `pkg-config`) into binaries that use Ginger.
<!-- mdformat on -->

### Using Ginger in your CMake project

Ginger is designed to be consumed from source via `FetchContent` or
`add_subdirectory`:

```cmake
include(FetchContent)

FetchContent_Declare(
  ginger
  GIT_REPOSITORY https://github.com/google/ginger.git
  GIT_TAG main
)
FetchContent_MakeAvailable(ginger)

target_link_libraries(my_app PRIVATE ginger::ginger)
```
