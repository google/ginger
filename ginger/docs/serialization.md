# Saving and loading

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This page explains how to save and load Ginger indices using
[`io/io.h`](../io/io.h), which stores each object in a single file at `path`.

You can persist indices in three ways:
1.  **Save the searcher without its datasets (recommended):** Persist the
    graph, configuration, entry point and reordering, and attach your existing
    dataset at load time. This avoids duplicating large raw datasets on disk
    and works with any coordinate type (`float`, `int8_t`, `uint8_t`).
2.  **Save and load the entire searcher:** Persist the full `GingerSearcher`
    bundle (graph, datasets, projections, and remapping metadata) at a single
    path.
3.  **Save and load the graph index only:** Persist only the graph structure
    and attach your dataset with `GingerSearcherBuilder`. This only works for
    graphs built without reordering.

## Saving a searcher without its datasets (recommended)

If you already store your dataset (embeddings) elsewhere, you can keep the
searcher's configuration, entry point and reordering without duplicating the
dataset. Save it with `save_datasets = false`. This omits the raw, TurboQuant,
and JL datasets. `LoadGingerSearcherBuilder` then returns a
`GingerSearcherBuilder` to which you attach your dataset. The attached dataset
must contain the exact same points in the original pre-reordering order, and
`Build()` returns `absl::FailedPreconditionError` if it does not match the
dataset the index was built from. `Build()` recomputes the omitted TurboQuant
and JL datasets from it.

```cpp
ABSL_RETURN_IF_ERROR(
    ginger::io::SaveGingerSearcher(*searcher, path, /*save_datasets=*/false));

ABSL_ASSIGN_OR_RETURN(ginger::GingerSearcherBuilder builder,
                      ginger::io::LoadGingerSearcherBuilder(path));
ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ginger::GingerSearcher> loaded_searcher,
                      builder.AddDataset(dataset).Build());
```

## Saving and loading the graph index

Only use this for graphs built without reordering. Reordering gives graph nodes
new IDs, and a graph file stores neither the remapping to the original IDs nor
the entry point. Attaching your dataset to a reordered graph therefore returns
wrong neighbors without an error. Searchers built with a `reordering_algorithm`,
including the recommended configuration, have a reordered graph, so save them
without their datasets instead.

Saving only the graph index avoids duplicating large raw datasets on disk and
allows reconstructing a `GingerSearcher` over your existing dataset (in-memory
or memory-mapped) with `GingerSearcherBuilder`.

### Saving the graph

Save a `FlatGraph` or `CompressedCSRGraph` using `SaveFlatGraph` or
`SaveCompressedGraph`:

```cpp
#include "ginger/graph/flat_graph.h"
#include "ginger/io/io.h"

// Save a FlatGraph to disk.
// The parent directory ("/path/to/dir") must already exist.
const std::string path = "/path/to/dir/my_graph";
ABSL_RETURN_IF_ERROR(ginger::io::SaveFlatGraph(graph, path));

// Or save a CompressedCSRGraph:
// ABSL_RETURN_IF_ERROR(ginger::io::SaveCompressedGraph(compressed_graph, path));
```

### Loading and reconstructing the searcher

Load the graph and attach your dataset using `GingerSearcherBuilder`:

```cpp
#include "ginger/graph/flat_graph.h"
#include "ginger/io/io.h"
#include "ginger/searcher_builder.h"

// 1. Load the graph index from disk.
ABSL_ASSIGN_OR_RETURN(ginger::FlatGraph graph, ginger::io::LoadFlatGraph(path));

// 2. Attach your dataset and reconstruct the searcher.
ABSL_ASSIGN_OR_RETURN(
    std::unique_ptr<ginger::GingerSearcher> searcher,
    ginger::GingerSearcherBuilder(config)
        .AddDataset(dataset)
        .AddIndex(std::make_unique<ginger::FlatGraph>(std::move(graph)))
        .Build());

// 3. Execute queries.
ABSL_ASSIGN_OR_RETURN(std::vector<ginger::DistanceAndId> results,
                      searcher->FindNeighbors(query, search_options));
```

Key advantages of this approach:
-   **No dataset duplication:** Raw vectors are not re-persisted alongside the
    index graph.
-   **Any coordinate type:** Works with `float`, `int8_t`, and `uint8_t`
    datasets.
-   **Index formats:** Supports both uncompressed (`FlatGraph`) and compressed
    (`CompressedCSRGraph`) indices.

## Saving and loading an entire searcher

For batch pipelines or self-contained deployments where bundling the graph,
datasets, projections, and remapping metadata together is desirable,
`SaveGingerSearcher` and `LoadGingerSearcher` serialize the complete searcher.

```cpp
#include "ginger/io/io.h"
#include "ginger/searcher.h"

// Save an existing searcher to disk.
// The parent directory ("/path/to/dir") must already exist.
const std::string path = "/path/to/dir/my_index";
ABSL_RETURN_IF_ERROR(ginger::io::SaveGingerSearcher(*searcher, path));

// Reload the searcher from disk.
ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ginger::GingerSearcher> loaded_searcher,
                      ginger::io::LoadGingerSearcher(path));

// Execute queries on the reloaded searcher.
ABSL_ASSIGN_OR_RETURN(std::vector<ginger::DistanceAndId> results,
                      loaded_searcher->FindNeighbors(query, search_options));
```

-   **File path:** `path` is a file (not a directory) whose parent directory
    must already exist.
-   **TurboQuant codec:** `LoadGingerSearcher(path)` reconstructs the
    `TurboQuantCodec` from the serialized dimensionality, seed and metric.

## Current limitations of full-searcher serialization

`SaveGingerSearcher` and `LoadGingerSearcher` currently have the following
limitations:

-   **Non-float raw datasets (`int8_t` / `uint8_t`):** `SaveGingerSearcher`
    only persists `float` raw datasets (`CoordinateType::kFloat`) and returns
    `absl::UnimplementedError` for `int8_t` or `uint8_t` datasets. Save with
    `save_datasets = false` instead and attach the non-float dataset after
    `LoadGingerSearcherBuilder`.
-   **`CSRGraph` indices:** `SaveGingerSearcher` only supports `FlatGraph` and
    `CompressedCSRGraph` indices, returning `absl::UnimplementedError` for
    `CSRGraph`.
