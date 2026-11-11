# Index construction

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This page describes how Ginger constructs navigable search graphs with PiPNN and
Vamana, how candidate edges are pruned with RobustPrune, and how to configure
and monitor index builds. Index construction is deterministic given a fixed
`seed`.

## Choosing an algorithm

-   **PiPNN:** The recommended algorithm for vector search (dot product and
    squared L2). It builds graphs up to an order of magnitude faster than
    Vamana while achieving competitive QPS-recall search quality.
-   **Vamana:** The widely used graph construction algorithm
    ([DiskANN](https://papers.nips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)).
    In addition to vector search, Vamana supports non-vector search and custom
    distance callbacks via the low-level `BuildVamanaIndex` API.

## Shared configuration

Index builds are configured through `GingerSearcherConfig`
([`searcher.proto`](../searcher.proto)), which pairs dataset settings with
either a `pipnn_config` or `vamana_config`, optional graph compression and
reordering, and optional quantization:

```cpp
#include "google/protobuf/text_format.h"

GingerSearcherConfig config;
if (!google::protobuf::TextFormat::ParseFromString(
        R"pb(
          distance_metric: DOT_PRODUCT
          pipnn_config {
            max_degree: 64
            num_passes: 2
            leader_selection { fanout_sequence: [10, 3] }
            pruning_strategy: PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY
            prune_config { alpha: 1.08 }
          }
          use_graph_compression: true
          reordering_algorithm: RECURSIVE_GRAPH_PARTITIONING
          turbo_quant { extra_bits: 4 }
        )pb",
        &config)) {
  return absl::InvalidArgumentError("Failed to parse GingerSearcherConfig.");
}

auto dataset = std::make_shared<SpanDataset>(num_points, dimensionality);
for (GingerNodeId i = 0; i < num_points; ++i) {
  dataset->Set<float>(i, vectors[i]);
}
ABSL_ASSIGN_OR_RETURN(
    std::unique_ptr<GingerSearcher> searcher,
    GingerSearcherBuilder(config).AddDataset(dataset).Build());
```

Here `vectors[i]` holds the coordinates of point `i`, and the dataset owns a
copy of them. If your points are already stored back to back in one contiguous
buffer, you can avoid the copy by wrapping the buffer instead:

```cpp
ABSL_ASSIGN_OR_RETURN(
    auto dataset,
    SpanDataset::Wrap(buffer, dimensionality));
```

`SpanDataset::Wrap` does not copy, so `buffer` must stay alive as long as the
dataset is used, including by a searcher that keeps the float dataset.

-   **Distance metric (`distance_metric`):** Required. Set to `DOT_PRODUCT`
    (negated inner product, `-dot(x, y)`) or `SQUARED_L2` (`||x - y||^2`).
-   **Unit-normalized datasets (`normalized_datapoints`):** Indicates that
    dataset vectors have unit L2 norm (`||x|| = 1`). When left unset,
    `GingerSearcherBuilder` autodetects unit-normalized `float` datasets and
    sets this to `true` (setting it explicitly disables autodetection).
    Autodetection requires every point to have a norm within 0.1% of `1`, so if
    you know that your dataset is normalized, set it explicitly.
    RobustPrune works best when distances are non-negative and approach `0` for
    nearby points. When `normalized_datapoints` is set under `DOT_PRODUCT`, we
    shift distances from `-dot(x, y) <= 0` to `1 - dot(x, y) >= 0` during
    RobustPrune. Under `SQUARED_L2`, it allows TurboQuant to use less memory by
    not storing the squared norm of each vector (which is `1`). For
    unit-normalized data, we recommend `SQUARED_L2`, whose graph does not
    depend on this flag.

## Monitoring a build

-   **Progress reporting:** Register a callback with
    `GingerSearcherBuilder::SetProgressCallback` to receive `IndexBuildProgress`
    updates reporting the number of completed points, total points, current
    pass, total passes, and elapsed time during Vamana construction.
-   **Stage timings:** Pass a `GingerSearcherBuilder::BuildStats` pointer to
    `Build(&stats)` to record how much time different stages of the build take.
    We track time for the following stages: PCA training, quantization, graph
    construction, reordering, and compression.

## PiPNN

[PiPNN](https://arxiv.org/abs/2602.21247) (configured via
[`pipnn/config.proto`](../pipnn/config.proto)) recursively partitions the
dataset around sampled leaders with overlapping assignments. At the leaves, it
constructs exact bidirected k-NN graphs in small clusters, prunes candidate
edges during leaf processing, and applies a final RobustPrune pass.

```protobuf
distance_metric: DOT_PRODUCT
pipnn_config {
  max_degree: 64
  num_passes: 2
  leader_selection { fanout_sequence: [10, 3] }
  pruning_strategy: PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY
  prune_config { alpha: 1.08 }
}
```

-   **Key parameters:**
    -   `num_passes`: Number of independent recursive clustering passes to
        perform (`2` is recommended; default `1`). Multiple passes increase
        overlap between clusters and improve graph connectivity.
    -   `fanout_sequence`: Required. Specifies how many nearest leaders each
        point is assigned to at each recursion depth. `[10, 3]` is recommended
        for both quality and performance (deeper levels default to a fanout of
        `1`).
    -   `max_degree`: Maximum outgoing edges per node in the final pruned graph
        (default `64`).
    -   `prune_config.alpha`: Controls edge diversity in RobustPrune. `1.08` is
        a good starting point, and should be tuned for your dataset (typically
        `1.05`–`1.2`).
-   **Leaf edge pruning (`pruning_strategy`):**
    -   `PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY` (recommended): Applies
        RobustPrune on the fly whenever a node exceeds `max_degree`, followed by
        a final RobustPrune pass.
    -   `PRUNING_STRATEGY_HASH_PRUNE`: Uses SimHash (locality-sensitive hashing)
        to partition directions around each node into angular buckets and
        retains only the closest candidate per bucket in a bounded reservoir
        during leaf processing, followed by a final RobustPrune pass. Configure
        `hash_prune.reservoir_size` (default `96`) to roughly `1.5 * max_degree`
        (larger reservoirs pass more candidates to final pruning at higher
        memory cost), and leave `hash_prune.num_hash_bits` (default `12`, valid
        range `[1, 16]`, for `2^num_hash_bits` angular buckets) at its default.
-   **Partitioning bounds:** `max_leaf_size` (default `1000`, maximum cluster
    size before exact leaf k-NN construction) and `min_leaf_size` (default
    `100`, minimum cluster size before merging into a neighbor).

## Vamana

Vamana (configured via [`vamana/config.proto`](../vamana/config.proto)) inserts
points in growing batches. Each batch runs greedy beam search over the current
graph to collect candidate neighbors, adds reverse edges, and prunes over-full
adjacency lists with RobustPrune
([DiskANN](https://papers.nips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf);
[ParlayANN](https://arxiv.org/abs/2305.04359)).

```protobuf
distance_metric: DOT_PRODUCT
vamana_config {
  max_degree: 64
  seed: 123
  passes { max_beam_width: 128 prune_config { alpha: 1.0 } }
  passes { max_beam_width: 128 prune_config { alpha: 1.08 } }
}
```

-   **Multi-pass refinement (`passes`):** Typically runs two passes, such as
    `alpha = 1.0` followed by `alpha = 1.08` (tune the second-pass `alpha` in
    `1.05`–`1.2` for your dataset).
-   **Search width (`max_beam_width`) and degree (`max_degree`):**
    `max_beam_width` sets the beam width used when searching for candidate
    neighbors during each insertion pass (larger values improve graph quality at
    higher build cost). `max_degree` caps the number of outgoing edges retained
    per node.

## RobustPrune

Connecting each node `p` only to its `max_degree` closest neighbors often
clusters outgoing edges into one dense direction. This leaves other directions
uncovered and causes greedy search to take many small hops or get trapped in
local minima.

RobustPrune
([DiskANN](https://papers.nips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf);
[`robust_prune_config.proto`](../robust_prune_config.proto)), used by both PiPNN
and Vamana, enforces **angular diversity** across `p`'s outgoing edges so the
graph retains both local connections and long-range routing shortcuts:

1.  Candidate neighbors of `p` are sorted by increasing distance from `p` and
    considered one by one (up to `max_degree`).
2.  A candidate `c` is **pruned** if an already-selected neighbor `s` of `p` is
    sufficiently closer to `c` than `p` is: `d(s, c) < d(p, c) / alpha` (for
    non-negative distances). Intuitively, `(p, c)` is redundant because a search
    at `p` can step to `s` and make progress toward `c` from there.
3.  Discarding redundant edges in covered directions frees up degree slots for
    candidates in new directions and at longer distances.

Configuration options:

-   **Distance threshold (`alpha`):** Controls how aggressively redundant edges
    are pruned (`alpha >= 1.0`, typically `1.05`–`1.2`). At `alpha = 1.0`, any
    candidate `c` closer to an existing neighbor `s` than to `p` is pruned.
    Larger `alpha` values make pruning less strict (`s` must be at least `alpha`
    times closer to `c` than `p` is to prune `c`), retaining more edges and
    producing a denser graph.
-   **Backfilling pruned neighbors (`keep_pruned_neighbors`):** When `true`,
    fills any remaining degree slots (up to `max_degree`) with pruned candidates
    in order of increasing distance from `p`. Supported by both PiPNN and
    Vamana.
-   **Adaptive alpha backfill (`desired_degree`, Vamana only):** If regular
    pruning leaves fewer than `desired_degree` edges on a node, greedily adds
    back pruned candidates in increasing order of the minimum `alpha` needed for
    them to survive, until `desired_degree` is reached.

> **Warning — Distance sign (`nonpositive_distances`, low-level
> `BuildVamanaIndex` only):** `GingerSearcherBuilder` sets
> `nonpositive_distances` automatically. However, if you invoke the low-level
> API `BuildVamanaIndex` directly with a custom distance callback,
> `nonpositive_distances` **must** match the sign of your distances:
>
> -   When `false` (default), RobustPrune expects non-negative distances
>     (`>= 0`), uses the threshold `d(s, c) < d(p, c) / alpha`, and **silently
>     discards any candidate with a negative distance**.
> -   When `true` (such as unnormalized negated dot products
>     `-dot(x, y) <= 0`), RobustPrune inverts the threshold to
>     `d(s, c) < d(p, c) * alpha` and **silently discards any candidate with a
>     positive distance**.
>
> Mismatching this setting will silently drop candidate edges or invert `alpha`
> pruning, producing a degraded or disconnected graph.

## How and when is quantization used during index build

Configuring `turbo_quant` in `GingerSearcherConfig` (see
[`searcher.proto`](../searcher.proto)) quantizes the input vectors to 4 bits per
dimension (plus optional `extra_bits`) before constructing the graph:

-   **PiPNN:** Always uses TurboQuant-encoded vectors for fast leaf k-NN and
    pruning computations. If `turbo_quant` is configured on the searcher, PiPNN
    reuses that quantized dataset. Otherwise, `GingerSearcherBuilder` builds a
    temporary 4-bit TurboQuant dataset for graph construction and releases it
    once the graph is built.
-   **Vamana:** Uses TurboQuant distance scoring during graph construction when
    `turbo_quant` is enabled. Regardless of whether higher-precision
    `extra_bits` are enabled, the build always uses 4-bit quantization when it
    uses quantization. When `turbo_quant` is not set, Vamana scores directly on
    the unquantized dataset (`float`, `int8`, or `uint8`).
-   **PCA projection (`pca`):** We do not use PCA at build time, even if it is
    enabled. PCA is strictly an optimization for queries:
    `GingerSearcherBuilder` constructs the graph using full-dimensional vectors
    first to preserve graph quality, then projects and re-quantizes the dataset
    into the reduced PCA subspace for search.
-   **Discarding full-precision vectors (`SetDropDataset(true)`):** When search
    does not require float reranking, calling `builder.SetDropDataset(true)`
    releases the unquantized dataset after building so the `GingerSearcher` only
    retains the quantized vectors in memory.
