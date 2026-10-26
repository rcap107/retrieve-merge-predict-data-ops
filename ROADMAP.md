# Roadmap

This document tracks the improvements that were considered during the skrub
DataOps rewrite of the evaluation pipeline but are **not implemented yet**.
Each item sketches the design so that it can be picked up later without
re-deriving the analysis.

The engines implemented so far are described in the README (`legacy`,
`data_ops`, `data_ops_native`); `scripts/parity_check.py` documents the
equivalence guarantees between them.

## Tunable wrangling choices with Optuna (D4) — implemented

`search.py` + `src.methods.data_ops_search` implement the single-plan search:
retrieval method (over the offline query results), estimator, aggregation,
`top_k` and predictor are choices of one DataOps plan, explored with
`skb.make_randomized_search(backend="optuna")`. Remaining extensions:

- joint search across **base tables** (the current plan covers one base
  table per run; a campaign-level plan would add the table as another axis);
- richer model hyperparameter spaces (e.g. CatBoost `l2_leaf_reg` and the
  high-cardinality encoder as choices inside the predictor branch);
- consuming `search_results.csv` from the `scripts/evaluation` analysis
  tools (the run-log format is not used by the search).

## Learner serialization and replay (D5) — partially implemented

DataOps plans are serializable. The `save_artifacts` option (see README)
already covers the basic case: per-estimator plan graphs (SVG) and the
fitted learner of the last fold (`learners/<estimator>.joblib`), also
exposed programmatically through `ScenarioLogger.learners`. Remaining
extensions:

- refitting the winning plan on the **full** dataset (rather than saving the
  last CV fold's learner) for deployment-grade artifacts;
- generating `skb.report` HTML artifacts per run for debugging (every node's
  input preview, duration and fitted estimator). Note that report generation
  evaluates the plan without clearing intermediate results (more memory) and
  writes HTML files; it should stay opt-in;
- replaying a saved learner whose candidate tables have moved (the pickled
  plan references parquet paths; a path-remapping step would be needed).

## Caching for the estimator-internal join paths

The skrub disk cache (`run_parameters.cache_dir`) only applies to nodes of
the DataOps plans. In the `data_ops` and `data_ops_native` engines, the
selectors that join inside `fit` (`highest_containment`, `best_single_join`
in estimator form, `stepwise_greedy_join`) re-read and re-aggregate the
candidate tables at every fold without any caching. A custom memoization
(e.g. an in-process LRU keyed by `(candidate hash, aggregation)` for the
aggregated tables) would remove this redundancy without touching skrub, at
the cost of another code path to keep in sync with `src.utils.joining`.

## Rejected alternatives (kept for reference)

- **Pre-split containment ranking for `highest_containment`**: computing the
  containment ranking once on the full base table (before `mark_as_X`) would
  make the selection a cacheable, fold-independent node. It was rejected
  because the ranking would be computed on test data as well; containment is
  unsupervised so there is no target leakage, but the selected candidate may
  differ from the paper's procedure.
- **`skrub.AggJoiner` / `MultiAggJoiner`**: conceptually the merge step of
  the paper, but `AggJoiner` has no `first` operation and different
  mode tie-breaking than `aggregate_mean`; the docs themselves recommend
  plain dataframe aggregation for large tables. The pipeline keeps its
  polars aggregation code.
- **`skrub.AggTarget`**: the paper never aggregates the target; not
  applicable.
- **`skb.cross_validate` as the only evaluation loop**: the "light" mode
  does this, but the paper's metrics require memory profiling and phase
  timings, which skrub does not provide; the instrumented loop stays the
  default for experiments.
