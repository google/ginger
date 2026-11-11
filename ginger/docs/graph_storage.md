# Graph storage

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This page describes Ginger's in-memory graph representations, node reordering,
and neighbor-list compression, along with how to configure them in
`searcher.proto` and `GingerSearcherBuilder`.

## Motivation

An uncompressed search graph stores 32-bit neighbor IDs per edge (for example,
260 bytes per node at `max_degree = 64`), which can account for a large share
of the total index memory. To conserve memory, Ginger can use lossless
compression techniques such as delta encoding and variable-width integer
coding. To improve the compression ratio, Ginger can reorder nodes so that gaps
between sorted neighbor IDs are smaller. Decompression runs on the fly during
beam search and is SIMD-accelerated, so it does not adversely affect search
performance.

## Recommendation

For most deployments, enable graph compression and recursive graph partitioning
in `GingerSearcherConfig` (`searcher.proto`):

```textproto
# proto-message: ginger.GingerSearcherConfig

# ... other fields ...
use_graph_compression: true
reordering_algorithm: RECURSIVE_GRAPH_PARTITIONING
```

Then pass this `GingerSearcherConfig` to `GingerSearcherBuilder`:

```cpp
#include "ginger/searcher_builder.h"

ABSL_ASSIGN_OR_RETURN(
    std::unique_ptr<ginger::GingerSearcher> searcher,
    ginger::GingerSearcherBuilder(config)
        .AddDataset(dataset)
        .Build());
```

Alternatively, enable graph compression and reordering directly on
`GingerSearcherBuilder`:

```cpp
ABSL_ASSIGN_OR_RETURN(
    std::unique_ptr<ginger::GingerSearcher> searcher,
    ginger::GingerSearcherBuilder()
        .AddDataset(dataset)
        // ... other builder settings ...
        .SetUseGraphCompression(true)
        .SetReorderingAlgorithm(ginger::RECURSIVE_GRAPH_PARTITIONING)
        .Build());
```

Compressed graphs with node reordering match the search speed of uncompressed
`FlatGraph` at a fraction of the graph memory footprint. Set
`use_graph_compression: false` if you want to skip the build-time graph
compression pass and keep the uncompressed `FlatGraph`.

## Graph formats

Ginger represents the search graph in one of three formats:

-   **`FlatGraph`:** Stores each node's degree and outgoing neighbor IDs in a
    fixed-stride contiguous array of `(max_degree + 1)` 32-bit integers per
    node. Produced directly by PiPNN and Vamana, it is the default index format
    when `use_graph_compression` is `false`.
-   **`CSRGraph`:** A standard
    [compressed sparse row (CSR)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
    representation (`offsets` and `targets` arrays).
-   **`CompressedCSRGraph`** (recommended): Stores the graph in a CSR layout
    where each node's sorted neighbor list is delta-encoded as non-negative ID
    gaps using a SIMD-accelerated adaptive gap codec, paired with compact
    cache-line-blocked or 32/64-bit CSR offsets (`CompressedOffsets`). Enabled
    by setting `use_graph_compression: true`.

## Node reordering

Assigning consecutive IDs to topologically close nodes shrinks the differences
(gaps) between sorted neighbor IDs to maximize graph compression ratios (with
improved search cache locality as a byproduct).

-   **Algorithms (`reordering_algorithm`):**
    -   `RECURSIVE_GRAPH_PARTITIONING` (recommended): Recursively partitions the
        symmetrized graph into balanced blocks using multi-level graph
        partitioning ([KaMinPar](https://github.com/KaHIP/KaMinPar)). Achieves
        the best compression ratio.
    -   `RECURSIVE_GRAPH_BISECTION`: Recursively bisects the graph to minimize
        the log-gap cost of sorted neighbor lists
        ([Dhulipala et al.](https://arxiv.org/abs/1602.08820)). Achieves good
        compression quality, but is slower than `RECURSIVE_GRAPH_PARTITIONING`.
    -   `RECURSIVE_GRAPH_CONTRACTION`: Recursively contracts matched edges to
        order densely connected clusters of nodes contiguously.
    -   `REORDERING_ALGORITHM_NONE`: Disables node reordering.
-   **ID mapping:** When node reordering is applied, `GingerSearcher` remaps the
    graph's entry point and stores the inverse permutation (`remapping()`).
    `FindNeighbors` automatically translates internal node IDs back to the
    caller's original dataset IDs before returning results.

## Compression formats

Setting `use_graph_compression: true` in `GingerSearcherConfig` compresses the
graph into a `CompressedCSRGraph`. For each node, `CompressedCSRGraph` encodes
the node degree and delta-encodes its sorted neighbor IDs as consecutive
non-negative gaps (`{id_0, id_1 - id_0, id_2 - id_1, ...}`).

`GingerSearcherConfig` uses the **AdaptiveGap** codec by default and does not
expose a proto field to switch codecs. Other codecs and options can be selected
in C++ by defining a custom `CompressedGraphConfig` struct for
`CompressedCSRGraph<Config>` and `CompressGraph`:

```cpp
#include "ginger/graph/compressed_graph.h"
#include "ginger/graph/compressed_graph_builder.h"
#include "ginger/graph/compression/adaptive_gap_codec.h"

// Default AdaptiveGap compression:
ginger::CompressedCSRGraph<> compressed = ginger::CompressGraph(flat_graph);

// Custom configuration (shown with the default AdaptiveGap settings):
struct CustomGraphConfig : public ginger::CompressedGraphConfig {
  static constexpr bool kUseBatchGapDecoding = true;
  using BatchGapEncoder = ginger::AdaptiveGapEncoder;
  using BatchGapDecoder = ginger::AdaptiveGapDecoder;
  static constexpr bool kUseIntervalEncoding = false;
};

ginger::CompressedCSRGraph<CustomGraphConfig> custom_compressed =
    ginger::CompressGraph<ginger::FlatGraph, CustomGraphConfig>(flat_graph);
```

-   **AdaptiveGap (`AdaptiveGapEncoder` / `AdaptiveGapDecoder`, default):**
    Generalizes [StreamVByte](https://arxiv.org/abs/1709.08990) to sub-byte bit
    widths by selecting a 4-width codebook per neighbor list, with SIMD
    byte-shuffle decoding.
-   **StreamVByte (`StreamVByteEncoder` / `StreamVByteDecoder`):** Encodes
    groups of four integers into 1–4 byte payloads with a separate
    2-bit-per-value control byte for SIMD decoding
    ([StreamVByte](https://arxiv.org/abs/1709.08990)). To use it, set
    `BatchGapEncoder` and `BatchGapDecoder` to these types.
-   **PrefixVarint (`VarintEncoder` / `VarintDecoder`):** Variable-byte encoding
    that stores the byte length in the first byte of each integer (also used to
    encode node degrees and interval metadata). To gap-encode neighbors with
    PrefixVarint, set `kUseBatchGapDecoding = false`.
-   **Interval encoding (`kUseIntervalEncoding`):** Optionally encodes
    contiguous runs of consecutive neighbor IDs as `(start, length)` intervals
    before gap-encoding the remaining neighbors.
