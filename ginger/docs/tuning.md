# Tuning guide

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This guide explains how to configure Ginger for your dataset. We recommend
starting with the automatic config recommender, and then adjusting a few key
parameters if you have specific latency, recall, or memory targets.

## Recommended configuration

The easiest way to configure Ginger for a float dataset is to call
`RecommendGingerConfig` ([`config_recommender.h`](../config_recommender.h)). It
takes your dataset's dimensionality, the distance metric, and a graph pruning
threshold (`alpha`). It returns a ready-to-use `GingerSearcherConfig`. At query
time, call `DefaultSearchOptions` on the built searcher to get matching query
settings.

```cpp
#include "ginger/config_recommender.h"
#include "ginger/searcher.h"
#include "ginger/searcher_builder.h"

// 1. Get a recommended searcher configuration:
ABSL_ASSIGN_OR_RETURN(
    GingerSearcherConfig config,
    RecommendGingerConfig(dataset->dimensionality(), SQUARED_L2,
                          /*robust_prune_alpha=*/1.08f));

// 2. Build the searcher:
ABSL_ASSIGN_OR_RETURN(
    std::unique_ptr<GingerSearcher> searcher,
    GingerSearcherBuilder(config).AddDataset(dataset).Build());

// 3. Run a query using matching default search options:
SearchOptions options =
    searcher->DefaultSearchOptions(/*num_neighbors=*/10, /*beam_size=*/64);
ABSL_ASSIGN_OR_RETURN(std::vector<DistanceAndId> neighbors,
                      searcher->FindNeighbors(query_span, options));
```

A pruning threshold (`alpha`) between 1.05 and 1.2 usually works well. 1.08 is a
good starting point. `DefaultSearchOptions` inspects the built searcher to pick
the right search space and reranking strategy automatically, defaulting to a
rerank factor of 6.

Both functions return standard protos, so you can modify any field before
building or querying if you need custom settings. Note that the recommender only
supports float datasets.

## How the recommender works

Under the hood, the recommender sets up a pipeline optimized for both speed and
memory. It builds the graph using PiPNN with a maximum degree of 64. PiPNN
builds graphs an order of magnitude faster than Vamana while giving similar
search quality. It also reorders and compresses the graph to improve cache
locality and shrink memory usage.

For distance scoring, it compresses the dataset with 4-bit TurboQuant for fast
SIMD beam search, and keeps 8 extra bits per dimension (12 bits total) for
high-precision reranking. Because 12-bit reranking is highly accurate, it drops
the original float dataset after building the index to save memory.

If your dataset has 512 or more dimensions, the recommender applies PCA to speed
up the beam search. It picks a target dimension that preserves 90% of the
variance, rounding it so the quantized rows align perfectly to CPU cache lines.
It skips PCA if it cannot cut the dimensionality at least in half. Reranking
still uses a separate full-dimensional 12-bit dataset so you do not lose recall
to PCA truncation error.

## Manual tuning

Once you have a baseline from the recommender, you can adjust individual
parameters to shift between recall, QPS, build time, and memory footprint.

### 1. Query latency vs. recall

You can trade query latency for recall at search time without rebuilding the
index.

The main knob is `beam_size` in `SearchOptions`. Search work scales roughly
linearly with the beam size, so sweep it to find the right QPS and recall
balance for your workload. It must be at least as large as `num_neighbors`.

You can also adjust `rerank_factor`. By default, `DefaultSearchOptions` sets it
to 6.0, which rescores the top `6 * num_neighbors` candidates from the beam.
Increase it (or clear it to rerank the whole beam) if you need very high recall
(e.g. 99% 10@10). Lower it to 2.0 or 4.0 if you request a large `num_neighbors`
and reranking starts to dominate your query time.

### 2. Graph quality and size

The pruning threshold (`alpha` passed to `RecommendGingerConfig`) controls graph
connectivity. Increasing it favors diverse long-range hops over redundant
short-range edges, which helps high-recall queries converge in fewer steps.
Sweep between 1.05 and 1.2 to find the sweet spot for your data.

The maximum out-degree (`pipnn_config.max_degree`) defaults to 64, which works
well for most workloads. Lowering it to 32 halves the graph memory and speeds up
low-recall queries. Raising it to 96 or 128 can improve high-recall throughput
on difficult datasets, especially if you also add a third PiPNN pass
(`num_passes = 3`).

### 3. Memory footprint vs. reranking precision

By default, the recommender uses 8 extra bits per dimension for reranking and
drops the float dataset.

To save memory at a small cost to recall, lower `turbo_quant.extra_bits` to 6,
4, or 2.

If you have enough memory and want exact full-precision reranking, you can keep
the original float dataset instead. Set `extra_bits` to 0 and `drop_dataset` to
false. `DefaultSearchOptions` will automatically switch to float reranking:

```cpp
ABSL_ASSIGN_OR_RETURN(
    GingerSearcherConfig config,
    RecommendGingerConfig(dataset->dimensionality(), SQUARED_L2,
                          /*robust_prune_alpha=*/1.08f));
config.mutable_turbo_quant()->set_extra_bits(0);
config.set_drop_dataset(false);

ABSL_ASSIGN_OR_RETURN(
    std::unique_ptr<GingerSearcher> searcher,
    GingerSearcherBuilder(config).AddDataset(dataset).Build());

SearchOptions options =
    searcher->DefaultSearchOptions(/*num_neighbors=*/10, /*beam_size=*/64);
```
