# Search

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This page explains how we execute queries in Ginger using
`GingerSearcher::FindNeighbors`, and how you can configure beam search, distance
scoring, reranking, PCA dimensionality reduction, and candidate pre-filtering
with `SearchOptions` ([`searcher.proto`](../searcher.proto)).

## Running a query

To run a query, fill out a `SearchOptions` proto and pass your query vector to
`GingerSearcher::FindNeighbors`:

```cpp
#include "google/protobuf/text_format.h"

SearchOptions options;
if (!google::protobuf::TextFormat::ParseFromString(R"pb(
      num_neighbors: 10
      beam_size: 64
      search_space: SEARCH_SPACE_TURBO_QUANT
      reranking_strategy: RERANKING_STRATEGY_FLOAT
      rerank_factor: 4.0
    )pb",
                                                   &options)) {
  return absl::InvalidArgumentError("Failed to parse SearchOptions.");
}

std::vector<float> query = {1, 2, 3, 4, 5};
ABSL_ASSIGN_OR_RETURN(std::vector<DistanceAndId> neighbors,
                      searcher->FindNeighbors(query, options));
```

-   **Results:** `FindNeighbors` returns up to `num_neighbors` entries sorted
    from smallest to largest distance, breaking ties by smaller node ID. The
    returned IDs always match your original dataset indices, even if the graph
    was reordered during index construction.
-   **Query inputs:** You can pass a `std::vector<T>` or `absl::Span<const T>`
    matching the dataset's coordinate type (`float`, `int8_t`, or `uint8_t`). If
    you are running a purely quantized search (without PCA or float reranking),
    you can also pass a pre-encoded `TurboQuantQuery`:

    ```cpp
    ABSL_ASSIGN_OR_RETURN(TurboQuantQuery tq_query,
                          TurboQuantQuery::Create(quantizer, query));
    ABSL_ASSIGN_OR_RETURN(std::vector<DistanceAndId> neighbors,
                          searcher->FindNeighbors(tq_query, options));
    ```

-   **Thread safety:** `FindNeighbors` is `const` and thread-safe, so multiple
    threads can query the same `GingerSearcher` concurrently.

## Beam search

During a query, Ginger runs a greedy beam search
([DiskANN](https://papers.nips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf);
[ParlayANN](https://arxiv.org/abs/2305.04359)) starting at the graph's entry
point `s`. It maintains a bounded priority queue (the *beam* `B`) of the
`beam_size` closest visited nodes, repeatedly expanding the closest unexplored
node in the beam, scoring its outgoing neighbors, and merging closer candidates
into the beam. The search finishes once all `beam_size` nodes in the beam have
been expanded:

```text
BeamSearch(G, s, q, beam_size):
  Input:  G = (X, E) a graph over X, s in X an entry point,
          q a query point, and beam_size a beam width
  Output: B subseteq X containing the beam_size closest points to q
          encountered during the search

  B <- {s}                                      // beam of closest candidates
  V <- {}                                       // expanded nodes
  while B \ V is not empty:
    p <- argmin_{u in B \ V} d(u, q)            // closest unexpanded node in B
    B <- B union Neighbors(p)
    V <- V union {p}
    if |B| > beam_size:
      B <- the beam_size points in B with smallest distance to q
  return B
```

-   **Beam width (`beam_size`):** `beam_size` is the main knob for trading query
    latency against recall. Because the search stops once every node in the beam
    has been expanded, the number of node expansions and candidate distance
    evaluations scales roughly linearly with `beam_size`. Increasing `beam_size`
    explores more of the graph and improves recall at a roughly proportional
    cost in latency. Since the beam holds at most `beam_size` candidates, make
    sure `beam_size >= num_neighbors`.
-   **Inspecting search work (`BeamSearchResult::Statistics`):** If you want to
    see how much work a query performed, pass an optional
    `BeamSearchResult::Statistics*` as the third argument to `FindNeighbors`. It
    records the number of nodes visited, candidates considered, distance
    computations, candidate filter evaluations, and reranked points.

## Distance scoring during traversal

In `SearchOptions`, `search_space` controls how we score candidate neighbors
while traversing the graph. You must set `search_space` explicitly
(`SEARCH_SPACE_UNSPECIFIED` returns an error; Ginger does not pick one
automatically):

-   **`SEARCH_SPACE_TURBO_QUANT` (recommended):** Scores candidates during beam
    traversal using 4-bit [TurboQuant](https://openreview.net/pdf?id=tO3ASKZlok)
    codes (`turbo_quant` in `GingerSearcherConfig`). Whenever `turbo_quant` is
    configured on the searcher, you should use `SEARCH_SPACE_TURBO_QUANT`. It is
    substantially faster and uses much less memory bandwidth than uncompressed
    scoring, and pairing it with reranking recovers full-precision recall.
-   **`SEARCH_SPACE_FLOAT`:** Scores candidates directly against the
    unquantized dataset (`float`, `int8_t`, or `uint8_t`) during traversal. Only
    use this when `turbo_quant` is not configured on the searcher.

## Reranking

Once beam search converges, we can rescore the top candidates in the beam at
higher precision before returning the `num_neighbors` closest results. Like
`search_space`, you must set `reranking_strategy` explicitly in `SearchOptions`:

-   **`RERANKING_STRATEGY_FLOAT`:** Rescores candidates with exact distances on
    the full-precision `float` dataset. We recommend this strategy whenever the
    `float` dataset fits in memory. (For `int8_t` or `uint8_t` datasets, do not
    use TurboQuant or reranking; use `SEARCH_SPACE_FLOAT` with
    `RERANKING_STRATEGY_NONE`.)
-   **`RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS`:** Rescores candidates using
    the additional refinement bits per dimension configured in
    `turbo_quant.extra_bits` (`1`, `2`, `3`, `4`, `6`, or `8` extra bits, for
    5–12 bits per dimension in total). We recommend this strategy when you drop
    the uncompressed dataset after index construction via
    `GingerSearcherBuilder::SetDropDataset(true)` to save memory while still
    recovering high recall.
-   **`RERANKING_STRATEGY_NONE`:** Returns the top `num_neighbors` from the beam
    directly without a second scoring pass. Use this when traversing with
    `SEARCH_SPACE_FLOAT`, or when you want the lowest possible memory footprint
    with 4-bit-only search.
-   **Rerank window (`rerank_factor`):** When `rerank_factor` is left unset,
    Ginger reranks all `beam_size` candidates in the converged beam. Since
    `beam_size` is often much larger than `num_neighbors`, setting
    `rerank_factor` (for example, `3.0` to `5.0`) limits reranking to only the
    top `rerank_factor * num_neighbors` candidates in the beam, avoiding the
    cost of rescoring the rest of the beam.

## Dimensionality reduction with PCA

On high-dimensional `float` datasets, you can configure `pca`
(`PcaProjectionConfig` in [`projection/config.proto`](../projection/config.proto))
alongside `turbo_quant` in `GingerSearcherConfig` to project vectors into a
lower-dimensional principal subspace before 4-bit quantization. At query time,
Ginger projects each `float` query into the same PCA subspace and quantizes it
to 4 bits for beam traversal, cutting both memory bandwidth and per-candidate
scoring time.

```protobuf
turbo_quant {
  extra_bits: 4
}
pca {
  truncation {
    significance_threshold: 0.9
    truncation_threshold: 0.75
    alignment: 32
  }
  matrix_representation: MATRIX_REPRESENTATION_TURBOQUANT
}
```

-   **Target dimensionality (`dimension` or `truncation`):** You can either set
    a fixed output `dimension` or use variance-based `truncation`, which picks
    the smallest dimension (rounded up to `alignment`) that preserves at least
    `significance_threshold` of the total variance and is at most
    `truncation_threshold` of the input dimensionality.
-   **Projection matrix format (`matrix_representation`):** We strongly
    recommend setting `MATRIX_REPRESENTATION_TURBOQUANT`, which quantizes the
    PCA projection matrix to 4 bits with TurboQuant to reduce memory traffic and
    accelerate query projection. By default, the matrix is stored in
    uncompressed `float32` (`MATRIX_REPRESENTATION_FLOAT`).
-   **Training subsampling (`sampling`):** By default, PCA computes the sample
    mean and covariance matrix over all points in the dataset during index
    construction. On large datasets, you can subsample the training points
    uniformly without replacement (seeded by `sampling.seed`, default `42`) by
    setting either `sampling.sample_size` (a fixed number of training points) or
    `sampling.samples_per_dimension` (which samples
    `samples_per_dimension * input_dimensionality` points). This significantly
    speeds up PCA training at build time without noticeably affecting projection
    quality.
-   **Full-dimensional reranking:** Because truncating dimensions loses accuracy
    for the final ranking, reranking always runs in the original
    full-dimensional space. `RERANKING_STRATEGY_FLOAT` rescores against the
    unprojected `float` dataset, while
    `RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS` rescores against a separate
    full-dimensional `4 + extra_bits` dataset.

## Candidate filters

Candidate pre-filtering (`candidate_filter` in `SearchOptions`) turns distance
evaluation during beam search into a two-stage check: for each unvisited
neighbor, Ginger first computes a cheap low-dimensional distance estimate and
only evaluates the full traversal distance for candidates whose estimated
distance falls below the mean of the `beam_mean_width` (default `5`) worst
entries in the current beam.

For high-dimensional `float` datasets, we generally recommend PCA projection
(`pca`) over candidate pre-filtering, as it typically achieves better speed and
recall by scoring every candidate in a single compact pass. When PCA is not
used, you can configure one of two candidate filters:

-   **`turbo_slice`:** Scores a prefix of the rotated 4-bit TurboQuant
    dimensions (requires `turbo_quant.turboslice_dimensionality` in
    `GingerSearcherConfig` and `SEARCH_SPACE_TURBO_QUANT`).
-   **`jl`:** Scores 1-bit Johnson-Lindenstrauss random projection sketches
    (requires `jl` in `GingerSearcherConfig` and `float` queries).

## Low-level search

If you need to search over custom distance metrics or non-vector data outside
`GingerSearcher`, you can call `NavigationBeamSearch`
([`search/beam_search.h`](../search/beam_search.h)) directly with your own
distance callback; see
[`examples/ginger_byod_example.cc`](../examples/ginger_byod_example.cc) for a
complete example.
