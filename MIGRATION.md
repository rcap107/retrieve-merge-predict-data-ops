# Migration summary: skrub DataOps in the Retrieve-Merge-Predict pipeline

This document records how the migration of the evaluation pipeline to
[skrub DataOps](https://skrub-data.org/) was executed, the decisions made
along the way, and the resulting code layout. It complements `ROADMAP.md`
(planned extensions) and the README (usage).

## Goal and guiding principle

The paper's pipeline augments a base table with tables retrieved from a data
lake (retrieval → merge → predict) and benchmarks join-selection strategies.
The objective was to rewrite the evaluation around skrub DataOps — a single
declarative plan per experiment, with cross-validation handled by skrub —
**while preserving the paper's semantics**: the same CV procedure, the same
selectors, the same aggregation/join code, and byte-compatible run logs so
that the analysis scripts keep working. Every rewrite step was therefore
gated by a parity harness (`scripts/parity_check.py`) before moving on.

## Environment and tooling

- **venv**: everything runs in the local `remepre` uv venv (Python 3.14).
  `requirements.txt` (paper-era pins, e.g. pandas 1.5.3) is **not**
  installable on 3.14 and was kept for reference only; the `pyproject.toml`
  pins were relaxed to minimums and the project installed with
  `uv pip install -e .`.
- **pixi**: the `[tool.pixi]` manifest was completed (envs `default`/`analysis`,
  tasks for every pipeline step, `pixi.lock`) as the official reproduction
  path, while day-to-day work kept using the uv venv (hybrid decision).
- Version fallout fixed to keep the legacy path runnable: skrub
  `tabular_learner` → `tabular_pipeline`, polars 1.x `df[list]` semantics in
  `execute_join_with_aggregation`, `InvalidOperationError` in
  `cast_features`.

## Phases

### Phase 1 — DataOps engine behind a flag (parity-first)

`src/methods/data_ops_plan.py` builds one DataOps plan per (fold, estimator):

```
skrub.var("base_table") -> cleaning node -> mark_as_X(cv=GroupShuffleSplit,
split_kwargs={"groups": <DataOp>}) / mark_as_y -> skb.apply(<selector>)
```

Key decisions:
- the **existing selector classes are reused** via `skb.apply` rather than
  rewritten; they need a `FrozenEstimator` wrapper because they are not
  scikit-learn-clone-safe (`get_params`/`set_params` round-trip is broken by
  construction — a pre-existing issue, never exposed before cloning);
- the **outer CV splits are computed once** (`prepare_splits`) and passed as
  explicit `(train, test)` pairs, guaranteeing the fold loop, the tqdm
  progress and skrub's internal splitting agree; this also gives the parity
  harness a hook (`run_parameters["_splits"]`);
- the **outer instrumentation is unchanged**: `memory_profiler`, `RunLogger`,
  metrics and log schema identical to the legacy loop;
- a config flag `engine = "legacy" | "data_ops"` selects the path; legacy
  stays the default.

### Phase 2 — joins as plan nodes

The `full_join` augmentation (join of all candidates + aggregation) moved
into a plan node (`_augment_all_candidates`), downstream of `mark_as_X` so
the split-before-join order is preserved. `skrub.AggJoiner` was rejected: no
`first` operation, different mode tie-breaking (the docs themselves recommend
dataframe aggregation for large tables). `BestSingleJoin` and
`StepwiseGreedyJoin` keep their joins inside `fit` because their candidate
selection state must survive until `predict` — DataOps can only carry fit
state through fitted estimators.

### Instrumentation cleanup (from the sessionization review)

The sessionization example (custom splitter, inline choices, one plan object)
motivated three changes:
- `RunLogger.timed()` / `BaseJoinSelector.timed()` context managers replace
  ~40 `_start_time`/`_end_time` pairs (kept raw where a timed region crosses
  methods or is abandoned on `break`);
- `RunLogger.profiled_call()` fuses memory sampling + timer + `mark_memory`;
- memory sampling switched from `memory_profiler`'s spawned `MemTimer`
  subprocess to an in-process `psutil` thread (`sampled_memory_usage`) — on
  macOS the subprocess re-imports the whole program per call (~3 s inside
  every timed region); this also makes the timings closer to the paper's
  Linux/fork setup.
- a "light" mode (`run_parameters.light`) runs the evaluation with a single
  `skb.cross_validate` per estimator — no run logs, timings or memory — for
  debugging and iteration.
- the sessionization-style custom splitter was evaluated and deliberately
  **kept unused** (the engine always passes explicit splits); a sessionization
  pattern would only be needed if skrub were to drive splitting directly.

### Deep rewrite (D1–D3)

- **D1**: the model machinery (model builders, inner train/validation split,
  `eval_set`/`val_idxs` handling, preprocessing, timers) was extracted into
  `src/methods/model_core.py` as `EvalModel`; `join_selectors.py` shrank from
  1400 to 920 lines and now contains only selection/join logic (~270 lines of
  duplication/dead code removed overall; ~470 lines moved).
- **D2**: `engine = "data_ops_native"` (`src/methods/data_ops_native.py`):
  `no_join` and `full_join` become pure plan constructs;
  `highest_containment` keeps a thin estimator with train-time containment
  ranking (a pre-split ranking variant was rejected as a semantic deviation);
  `stepwise_greedy_join` stays estimator-internal (adaptive control flow).
- **D3**: native `best_single_join`: one plan branch per candidate combined
  with `skrub.choose_from`, selected per fold by `make_grid_search` over an
  inner validation split (optionally seeded via `inner_cv_seed`). The winner
  is refit on the whole training fold (legacy refits on the inner 80%), so
  results are near- (here: exactly-) equal; reported, not asserted.

### Caching

`run_parameters.cache_dir` enables the skrub disk cache. The aggregation of
the candidate tables (`first`/`mean` only — `dfs` depends on the left table)
is split into a **fold-independent node** (`_aggregate_candidates`,
`_aggregate_one`) whose result is cached across folds *and* runs, followed by
an uncached join node. Cached runs are verified to give identical results;
the cache must stay off for timing measurements.

### Run artifacts

`run_parameters.save_artifacts` writes the plan graph (SVG) and the fitted
learner of the last fold per estimator (`ScenarioLogger.learners` exposes
them programmatically). The saved learner is self-contained and replayable:
`joblib.load(...).predict({"base_table": table})`.

### Tier B — one plan for the whole campaign (HPO)

`search.py` + `src/methods/data_ops_search.py`: a **single DataOps plan**
whose choices span the retrieval method (a choice over the offline query
results — retrieval/indexing itself stays offline), the estimator
(flat `"<method>__<estimator>"` branches), the aggregation, `top_k` and the
predictor, searched with `skb.make_randomized_search(backend="optuna")`.
`search.results_` is the benchmark table (parameters + mean CV score per
trial on identical seeded folds). Verified skrub constraints that shaped the
design: a search cannot be embedded inside a plan (so `best_single_join`'s
inner selection is *flattened* into the search space — deviation, flagged;
the paper-faithful engine remains available), skrub rejects duplicate choice
names (per-branch choice names), and estimator-internal selectors need their
candidates at construction time (hence per-method branches).

## Verification

`scripts/parity_check.py` is the gate for every step: synthetic data lake +
base table, explicit frozen splits shared by all engines, seeded inner splits
and RNG (harness-only; the original pipeline is unseeded), and comparisons:

- legacy vs `data_ops`: exact metric equality (0.00e+00) for all five
  selectors, including the plan-native `full_join`;
- legacy vs `data_ops_native`: exact for the strict selectors;
- native `best_single_join`: report-only comparison;
- `data_ops_native` with and without disk cache: exact;
- duration-key comparison from real run logs (this uncovered that
  `RunLogger` only *prints* in debug mode — the harness now writes CSV logs);
- `--save-artifacts` copies graphs/learners to `results/parity_artifacts/`
  and replays a saved learner; `--hpo-smoke` runs the unified search plan,
  saves the plan (`plan.joblib` + graph) and round-trips it through joblib.

## Notable bugs found and fixed

- `TopKFullJoin` referenced but never imported (evaluation.py);
- `_prepare_inner_model` called with a wrong signature in
  `StepwiseGreedyJoin` (use_rf path);
- `RunLogger.to_logfile` silently no-ops in debug mode;
- `@skrub.deferred` shadows the module attribute, making raw (unfitted)
  plans unpicklable — fixed by binding the undecorated function to its own
  name (`_compute_groups_impl` / `_compute_groups`);
- skrub choice names must be unique within a plan;
- the flattened `best_single_join` branch initially returned the augmented
  table without applying the predictor (NaN scores in the search).

## Current layout

| File | Role |
|---|---|
| `src/methods/model_core.py` | `EvalModel`: model building/fitting core, inner split, timers |
| `src/methods/join_selectors.py` | selection/join strategies (thin, on `EvalModel`) |
| `src/methods/data_ops_plan.py` | `data_ops` engine, plan builders, plan-native joins, artifact saving |
| `src/methods/data_ops_native.py` | `data_ops_native` engine |
| `src/methods/data_ops_search.py` | unified HPO plan (retrieval × estimator × predictor) |
| `search.py` | HPO entrypoint (`search_results.csv`, best learner) |
| `scripts/parity_check.py` | parity harness + `--save-artifacts` + `--hpo-smoke` |
| `ROADMAP.md` | out-of-scope/planned work (D4 leftovers, D5 extensions, rejected alternatives) |

## How to run

See the README ("Installing the requirements", "Reproducibility", "Hyperparameter
search") for the full runbook: `python scripts/parity_check.py` needs no data;
the full pipeline requires a YADL download plus the metadata → index → query
steps, then `python main.py <evaluation config>` or `python search.py <search
config>`.
