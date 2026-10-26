"""Compare the legacy and skrub DataOps evaluation engines on identical inputs.

This script generates a synthetic data lake and base table (deterministically),
then runs the evaluation pipeline with both engines using the *same* explicit
cross-validation splits, and compares the produced metrics.

Deterministic selectors (`no_join`, `highest_containment`, `full_join`) with a
deterministic model (`ridgecv`) are compared strictly: both engines should
produce identical results, since they share the same join/aggregation code and
the same estimator classes.

Selectors whose candidate selection depends on an unseeded inner validation
split (`best_single_join`, `stepwise_greedy_join`) and stochastic models
(`catboost`) are only reported side by side, as the original pipeline is not
reproducible at that granularity.

Usage:
    python scripts/parity_check.py [--n-samples N] [--n-candidates K]
                                   [--n-splits N] [--model ridgecv|catboost]
                                   [--include-stochastic]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_structures.loggers import ScenarioLogger  # noqa: E402
from src.data_structures.metadata import CandidateJoin  # noqa: E402
from src.methods.evaluation import evaluate_joins  # noqa: E402

DETERMINISTIC_SELECTORS = ["no_join", "highest_containment", "full_join"]
STOCHASTIC_SELECTORS = ["best_single_join", "stepwise_greedy_join"]
TOLERANCE = 1e-8


def _freeze_randomness():
    """Make both engines deterministic *within this harness only*.

    The original pipeline draws unseeded inner train/valid splits (and one
    `random.shuffle` in StepwiseGreedyJoin), so two runs of the *same* engine
    differ. To compare the two engines strictly, the harness seeds the inner
    split and the global RNG. This does not modify the repository code.
    """
    import csv
    import random

    import src.data_structures.loggers as dlg
    import src.methods.data_ops_plan as dp
    import src.methods.join_selectors as js
    import src.methods.model_core as mc

    random.seed(0)

    def _patch(module):
        if not hasattr(module, "train_test_split"):
            return
        original_split = module.train_test_split

        def seeded_split(*args, **kwargs):
            kwargs.setdefault("random_state", 0)
            return original_split(*args, **kwargs)

        module.train_test_split = seeded_split

    _patch(js)
    _patch(dp)
    _patch(mc)

    # RunLogger only prints its JSON when `debug` is set; the harness needs
    # the run logs on disk to compare the duration keys between engines.
    def always_write_logfile(self, path_logfile):
        path = Path(path_logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with open(path, "a" if exists else "w") as fp:
            writer = csv.DictWriter(fp, fieldnames=dlg.HEADER_RUN_LOGFILE)
            if not exists:
                writer.writeheader()
            writer.writerow(self.to_dict())

    dlg.RunLogger.to_logfile = always_write_logfile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--model", type=str, default="ridgecv")
    parser.add_argument(
        "--include-stochastic",
        action="store_true",
        help="Also run selectors whose selection depends on an unseeded "
        "inner split (results reported, not asserted).",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Enable `save_artifacts` for every engine and copy the produced "
        "graphs and fitted learners to results/parity_artifacts/.",
    )
    parser.add_argument(
        "--hpo-smoke",
        action="store_true",
        help="Run the hyperparameter search plan (data_ops_search) on the "
        "synthetic data instead of the engine comparisons.",
    )
    return parser.parse_args()


def fabricate_data(n_samples: int, n_candidates: int, tmp_dir: Path):
    """Generate a synthetic base table and candidate tables."""
    rng = np.random.default_rng(0)
    n_groups = max(n_samples // 10, 2)
    group_keys = [f"k{i}" for i in range(n_groups)]
    group_effect = rng.normal(size=n_groups)

    base = pl.DataFrame(
        {
            "col_to_embed": [group_keys[i % n_groups] for i in range(n_samples)],
            "num1": rng.normal(size=n_samples),
            "num2": rng.uniform(0, 10, size=n_samples),
            "cat1": rng.choice(["a", "b", "c"], size=n_samples),
        }
    )
    # The target depends on the group effect (recoverable through joins) and
    # on the base features.
    base = base.with_columns(
        target=pl.lit(group_effect).gather(
            pl.col("col_to_embed").str.slice(1).cast(pl.Int64)
        )
        + pl.col("num1") * 0.5
        + rng.normal(size=n_samples) * 0.1
    )

    candidates = {}
    for c in range(n_candidates):
        cand = pl.DataFrame(
            {
                "col_to_embed": group_keys,
                "f1": group_effect + rng.normal(size=n_groups) * 0.01,
                "f2": rng.normal(size=n_groups),
            }
        )
        path = tmp_dir / f"cand_{c}.parquet"
        cand.write_parquet(path)
        cjoin = CandidateJoin(
            indexing_method="exact_matching",
            source_table_metadata={
                "hash": "src_hash",
                "full_path": "base_table",
                "df_name": "base_table",
            },
            candidate_table_metadata={
                "hash": f"cnd_{c}",
                "full_path": str(path),
                "df_name": f"cand_{c}",
            },
            how="left",
            left_on="col_to_embed",
            right_on="col_to_embed",
            similarity_score=1.0 - c / 10,
        )
        candidates[cjoin.candidate_id] = cjoin

    return base, candidates


def make_run_config(args, engine: str, plan_native: bool, splits: list, extra=None):
    estimators = {
        "no_join": {"active": True},
        "highest_containment": {"active": True},
        "full_join": {"active": True},
    }
    if args.include_stochastic:
        estimators["best_single_join"] = {"active": True}
        estimators["stepwise_greedy_join"] = {
            "active": True,
            "budget_type": "iterations",
            "budget_amount": 2,
            "epsilon": 0.0,
            "ranking_metric": "containment",
        }
    run_parameters = {
        "task": "regression",
        "debug": True,
        "n_splits": args.n_splits,
        "test_size": 0.2,
        "split_kind": "group_shuffle",
        "engine": engine,
        "plan_native_join": plan_native,
        "_splits": splits,
    }
    if getattr(args, "save_artifacts", False):
        run_parameters["save_artifacts"] = True
    if extra:
        run_parameters.update(extra)
    return {
        "run_parameters": run_parameters,
        "join_parameters": {"join_strategy": "left", "aggregation": "first"},
        "evaluation_models": {"chosen_model": args.model},
        "estimators": estimators,
        "query_cases": {
            "data_lake": "parity_test",
            "join_discovery_method": "exact_matching",
            "table_path": "base_table",
            "query_column": "col_to_embed",
            "top_k": args.n_candidates,
        },
    }


def run_engine(name: str, run_config: dict, base_table, candidates) -> pl.DataFrame:
    scenario_logger = ScenarioLogger(
        base_table_name="parity_base",
        git_hash="parity",
        run_config=run_config,
        exp_name=f"parity_{name}",
        debug=True,
    )
    evaluate_joins(
        scenario_logger,
        base_table,
        join_candidates=candidates,
        target_column="target",
        group_column="col_to_embed",
        estim_parameters=run_config["estimators"],
        join_parameters=run_config["join_parameters"],
        model_parameters=run_config["evaluation_models"],
        run_parameters=run_config["run_parameters"],
    )
    return scenario_logger.results


def compare_results(
    reference: pl.DataFrame,
    other: pl.DataFrame,
    ref_name: str,
    other_name: str,
    strict: bool,
):
    """Compare two results frame row by row (same fold/estimator order)."""
    assert reference.shape[0] == other.shape[0], (
        f"Result count mismatch between {ref_name} ({reference.shape[0]}) "
        f"and {other_name} ({other.shape[0]})."
    )
    mismatches = []
    for i in range(reference.shape[0]):
        ref_row, other_row = reference.row(i, named=True), other.row(i, named=True)
        assert ref_row["estimator"] == other_row["estimator"], (
            f"Row {i}: estimator mismatch {ref_row['estimator']} vs "
            f"{other_row['estimator']}"
        )
        for metric in ["r2", "rmse"]:
            ref_v, other_v = ref_row[metric], other_row[metric]
            if ref_v is None or other_v is None:
                continue
            diff = abs(ref_v - other_v)
            if strict and diff > TOLERANCE:
                mismatches.append((i, ref_row["estimator"], metric, ref_v, other_v))
            else:
                print(
                    f"  [{ref_name} vs {other_name}] row {i} "
                    f"{ref_row['estimator']} {metric}: {ref_v:.6f} vs {other_v:.6f} "
                    f"(diff {diff:.2e})"
                )
    return mismatches


def compare_duration_keys(
    reference_logs: list[dict], other_logs: list[dict], ref_name: str, other_name: str
):
    """Compare the set of non-empty `time_*` duration keys between the run
    logs of the two engines. Values are noisy (wall-clock times), so only the
    structure is asserted."""

    def duration_keys(log: dict) -> set:
        return {
            k for k, v in log.items() if k.startswith("time_") and v not in ("", None)
        }

    mismatches = []
    for ref_log, other_log in zip(reference_logs, other_logs):
        ref_key = f"{ref_log['estimator']} fold {ref_log.get('fold_id', '')}"
        other_key = f"{other_log['estimator']} fold {other_log.get('fold_id', '')}"
        ref_keys, other_keys = duration_keys(ref_log), duration_keys(other_log)
        if ref_keys != other_keys:
            mismatches.append(
                f"{ref_key} ({ref_name}) {sorted(ref_keys)} != "
                f"{other_key} ({other_name}) {sorted(other_keys)}"
            )
    return mismatches


def load_run_logs(exp_name: str) -> list[dict]:
    import csv

    log_dir = Path("results/logs", exp_name, "run_logs")
    logs = []
    for path in sorted(log_dir.glob("*.log")):
        with open(path) as fp:
            logs.extend(csv.DictReader(fp))
    return logs


def run_hpo_smoke(args, base_table, candidates):
    """Smoke test for the unified search plan (data_ops_search).

    Builds one plan spanning two synthetic retrieval methods, four estimators
    and two predictors, then runs a small randomized search over it.
    """
    import skrub

    from src.methods.data_ops_search import run_search

    # Two "retrieval methods" with the same candidates in different order.
    query_results = {
        "exact_matching": candidates,
        "minhash": dict(reversed(list(candidates.items()))),
    }
    config = {
        "run_parameters": {
            "task": "regression",
            "debug": True,
            "n_splits": 2,
            "test_size": 0.2,
            "split_kind": "group_shuffle",
        },
        "join_parameters": {"join_strategy": "left", "aggregation": "first"},
        "evaluation_models": {
            "chosen_model": ["ridgecv", "catboost"],
            "catboost": {
                "iterations": 10,
                "od_type": "Iter",
                "od_wait": 5,
                "l2_leaf_reg": 0.01,
                "thread_count": 2,
            },
        },
        "estimators": {
            "no_join": {"active": True},
            "full_join": {"active": True},
            "best_single_join": {"active": True},
            "highest_containment": {"active": True},
        },
        "query_cases": {"query_column": "col_to_embed"},
        "search": {
            "n_iter": 8,
            "backend": "sklearn",
            "seed": 0,
            "aggregation_choices": ["first"],
        },
    }
    search = run_search(config, base_table, query_results)
    results = search.results_
    print("\n== hpo smoke: search results ==")
    print(results.to_string())
    assert len(results) >= 8, f"expected at least 8 trials, got {len(results)}"
    assert "config" in results.columns, results.columns.tolist()
    predictions = search.best_learner_.predict({"base_table": base_table})
    assert len(predictions) == len(base_table)

    # Save the plan (unfitted DataOp) and its graph for inspection; the
    # reloaded plan can be evaluated directly with new data.
    import joblib

    plans_dir = Path("results/hpo_smoke")
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan = search.data_op
    joblib.dump(plan, plans_dir / "plan.joblib")
    (plans_dir / "plan_graph.svg").write_bytes(plan.skb.draw_graph().svg)
    reloaded = joblib.load(plans_dir / "plan.joblib")
    reloaded_predictions = reloaded.skb.eval({"base_table": base_table})
    assert len(reloaded_predictions) == len(base_table)
    print(
        f"Plan saved to {plans_dir / 'plan.joblib'} "
        f"(graph: {plans_dir / 'plan_graph.svg'}); "
        f"reloaded plan produced {len(reloaded_predictions)} predictions."
    )
    print("HPO SMOKE PASSED")


def main():
    args = parse_args()
    _freeze_randomness()
    # Clean the run logs of previous harness invocations (they are appended
    # to by the patched RunLogger).
    import shutil
    import tempfile

    for stale in Path("results/logs").glob("parity_*"):
        shutil.rmtree(stale)

    tmp_dir = Path(tempfile.mkdtemp(prefix="parity_check_"))
    base_table, candidates = fabricate_data(args.n_samples, args.n_candidates, tmp_dir)

    if args.hpo_smoke:
        run_hpo_smoke(args, base_table, candidates)
        return

    # Explicit splits shared by both engines. Computed the same way the legacy
    # engine does it (unseeded GroupShuffleSplit on the query column), but
    # frozen so that both engines see exactly the same folds.
    from src.methods.evaluation import prepare_splits

    probe_config = make_run_config(args, "legacy", False, None)
    splits = prepare_splits(probe_config["run_parameters"], base_table, "col_to_embed")

    print(f"Running legacy engine ({args.n_splits} folds)...")
    legacy_results = run_engine(
        "legacy", make_run_config(args, "legacy", False, splits), base_table, candidates
    )

    print("Running data_ops engine...")
    dops_results = run_engine(
        "dops", make_run_config(args, "data_ops", False, splits), base_table, candidates
    )

    print("Running data_ops engine with plan-native joins...")
    native_plan_results = run_engine(
        "dops_native_plan",
        make_run_config(args, "data_ops", True, splits),
        base_table,
        candidates,
    )

    print("Running data_ops_native engine...")
    native_results = run_engine(
        "dopsn",
        make_run_config(
            args, "data_ops_native", False, splits, extra={"inner_cv_seed": 0}
        ),
        base_table,
        candidates,
    )

    print("Running data_ops_native engine with skrub disk cache...")
    cached_results = run_engine(
        "dopsn_cache",
        make_run_config(
            args,
            "data_ops_native",
            False,
            splits,
            extra={
                "cache_dir": str(tmp_dir / "skrub_cache"),
                "inner_cv_seed": 0,
            },
        ),
        base_table,
        candidates,
    )

    if args.save_artifacts:
        import shutil

        dest_root = Path("results/parity_artifacts")
        for engine_name in [
            "legacy",
            "dops",
            "dops_native_plan",
            "dopsn",
            "dopsn_cache",
        ]:
            src = Path("results/logs", f"parity_{engine_name}")
            for subdir in ["graphs", "learners"]:
                if (src / subdir).exists():
                    shutil.copytree(
                        src / subdir,
                        dest_root / engine_name / subdir,
                        dirs_exist_ok=True,
                    )
        print(
            f"\nRun artifacts copied to {dest_root} "
            "(the legacy engine has no DataOps learners, so it produces none)"
        )
        try:
            import joblib

            learner = joblib.load(dest_root / "dops" / "learners" / "nojoin.joblib")
            predictions = learner.predict({"base_table": base_table})
            print(
                f"Replay check: the saved nojoin learner produced "
                f"{len(predictions)} predictions on the base table."
            )
        except Exception as exception:
            print(f"Replay check failed: {exception}")

    all_mismatches = []

    print("\n== legacy vs data_ops (strict) ==")
    all_mismatches += compare_results(
        legacy_results, dops_results, "legacy", "data_ops", strict=True
    )

    print("\n== legacy full_join vs data_ops plan-native full_join (strict) ==")
    legacy_full = legacy_results.filter(pl.col("estimator").str.contains("full_join"))
    native_plan_full = native_plan_results.filter(
        pl.col("estimator").str.contains("full_join")
    )
    all_mismatches += compare_results(
        legacy_full, native_plan_full, "legacy", "data_ops(plan-native)", strict=True
    )

    print("\n== legacy vs data_ops_native (strict, except best_single_join) ==")
    native_strict = native_results.filter(
        ~pl.col("estimator").str.contains("best_single_join")
    )
    legacy_strict = legacy_results.filter(
        pl.col("estimator").is_in(native_strict["estimator"].unique().to_list())
    )
    all_mismatches += compare_results(
        legacy_strict, native_strict, "legacy", "data_ops_native", strict=True
    )

    print("\n== legacy vs data_ops_native best_single_join (report only) ==")
    legacy_bsj = legacy_results.filter(
        pl.col("estimator").str.contains("best_single_join")
    )
    native_bsj = native_results.filter(
        pl.col("estimator").str.contains("best_single_join")
    )
    compare_results(legacy_bsj, native_bsj, "legacy", "data_ops_native", strict=False)

    print("\n== data_ops_native with and without skrub cache (strict) ==")
    # best_single_join is excluded: with an unseeded inner split its selection
    # is not run-to-run reproducible in either engine.
    native_cached_strict = native_results.filter(
        ~pl.col("estimator").str.contains("best_single_join")
    )
    cached_strict = cached_results.filter(
        ~pl.col("estimator").str.contains("best_single_join")
    )
    all_mismatches += compare_results(
        native_cached_strict,
        cached_strict,
        "data_ops_native",
        "data_ops_native(cache)",
        strict=True,
    )

    print("\n== duration keys (legacy vs data_ops / data_ops_native) ==")
    all_mismatches += compare_duration_keys(
        load_run_logs("parity_legacy"),
        load_run_logs("parity_dops"),
        "legacy",
        "data_ops",
    )
    all_mismatches += compare_duration_keys(
        load_run_logs("parity_legacy"),
        load_run_logs("parity_dopsn"),
        "legacy",
        "data_ops_native",
    )

    if all_mismatches:
        print("\nPARITY CHECK FAILED:")
        for item in all_mismatches:
            if isinstance(item, str):
                print(f"  {item}")
            else:
                row_id, estimator, metric, ref_v, other_v = item
                print(f"  row {row_id} ({estimator}) {metric}: {ref_v} vs {other_v}")
        sys.exit(1)
    print("\nPARITY CHECK PASSED")


if __name__ == "__main__":
    main()
