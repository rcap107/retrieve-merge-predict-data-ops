"""Native DataOps evaluation engine (`engine = "data_ops_native"`).

This engine expresses as much of the evaluation as possible with skrub
DataOps primitives, while producing logs identical to the other engines:

- `no_join`: pure plan (base table -> `EvalModel`);
- `full_join` (and `top_k_full_join`): the join/aggregation of all candidate
  tables is a node of the plan;
- `best_single_join`: one plan branch per candidate, combined with
  `skrub.choose_from`; the best candidate is selected per fold with a grid
  search over an inner validation split (mirroring the selection principle
  of the legacy estimator);
- `highest_containment` and `stepwise_greedy_join`: reuse the (thin) selector
  estimators, applied to the plan like in the `data_ops` engine.

Differences with respect to the legacy `best_single_join`: the search refits
the winning candidate on the whole training fold (the legacy estimator
retrains on the inner training split only), and the inner split can be
seeded with `run_parameters["inner_cv_seed"]` for reproducibility. Results
are therefore expected to be very close but not identical to the legacy
engine for this selector.
"""

import time

import polars as pl
import skrub
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

import src.utils.joining as ju
from src.data_structures.loggers import RunLogger
from src.methods.data_ops_plan import (
    FrozenEstimator,
    _compute_groups,
    _get_fitted_estimator,
    _join_timings,
    _last_join_time,
    _make_cv,
    _prepare_X,
    _save_artifacts,
    build_plan,
)
from src.methods.evaluation import prepare_estimator, prepare_splits
from src.methods.model_core import EvalModel


def _join_one(left_part: pl.DataFrame, cjoin, aggregation: str) -> pl.DataFrame:
    """Join a single candidate table onto a part of the base table.

    Used as the body of the per-candidate branches of the native
    `best_single_join` plan. Its duration is recorded in `_join_timings`.
    """
    _, cnd_md, left_on, right_on = cjoin.get_join_information()
    cnd_table = pl.read_parquet(cnd_md["full_path"])
    start = time.perf_counter()
    merged = ju.execute_join_with_aggregation(
        left_part,
        cnd_table,
        left_on=left_on,
        right_on=right_on,
        aggregation=aggregation,
    )
    _join_timings.append(time.perf_counter() - start)
    return merged


def _aggregate_one(cjoin, aggregation: str):
    """Aggregate a single candidate table on its join key.

    Fold-independent operation; its result can be cached across folds and
    runs when the skrub cache is enabled. Only usable with the `first` and
    `mean` aggregations.
    """
    _, cnd_md, left_on, right_on = cjoin.get_join_information()
    cand_table = pl.read_parquet(cnd_md["full_path"])
    aggr_table = ju.aggregate_table(cand_table, right_on, aggregation)
    return aggr_table, left_on, right_on


def _join_one_aggregated(left_part: pl.DataFrame, aggregated) -> pl.DataFrame:
    """Join a pre-aggregated candidate table onto a part of the base table."""
    aggr_table, left_on, right_on = aggregated
    start = time.perf_counter()
    merged = ju.execute_join(
        left_part,
        aggr_table,
        left_on=left_on,
        right_on=right_on,
        how="left",
        suffix="_right",
    )
    _join_timings.append(time.perf_counter() - start)
    return merged


def _build_best_single_join_plan(
    run_parameters: dict,
    model_parameters: dict,
    group_column: str,
    base_table: pl.DataFrame,
    estimator,
    splits: list,
):
    """Build the native `best_single_join` plan.

    One branch per candidate join, selected with `skrub.choose_from`; the
    branch augments the base table with the candidate's columns and fits the
    model. The selection is performed per fold by a grid search over an inner
    validation split.

    Returns:
        tuple: The DataOps plan and the environment expected by its methods.
    """
    target_column = run_parameters.get("target_column", "target")
    use_cache = run_parameters.get("cache_dir") is not None
    aggregation = estimator.join_parameters["aggregation"]

    base = skrub.var("base_table")
    X = base.skb.apply_func(_prepare_X, target_column=target_column, no_cache=True)
    y = base[target_column]

    groups = _compute_groups(base, group_column)
    X = X.skb.mark_as_X(
        cv=_make_cv(run_parameters, splits), split_kwargs={"groups": groups}
    )
    y = y.skb.mark_as_y()

    branches = {}
    for candidate_id, cjoin in estimator.candidate_joins.items():
        if use_cache and aggregation in ("first", "mean"):
            aggregated = skrub.deferred(_aggregate_one)(cjoin, aggregation)
            branch = X.skb.apply_func(_join_one_aggregated, aggregated, no_cache=True)
        else:
            branch = X.skb.apply_func(_join_one, cjoin, aggregation, no_cache=True)
        branches[candidate_id] = branch

    chosen = skrub.choose_from(branches, name="candidate").as_data_op()
    model = EvalModel(
        target_column=target_column,
        model_parameters=model_parameters,
        task=run_parameters["task"],
    )
    model.name = estimator.name
    pred = chosen.skb.apply(FrozenEstimator(model), y=y, no_cache=True)
    return pred, {"base_table": base_table}


def _inner_cv(run_parameters: dict):
    """Inner splitter used by the search to score the candidate branches.

    Mirrors the inner train/validation split of the legacy selectors
    (test size 0.2). Seeding is optional through `inner_cv_seed` (the legacy
    pipeline is unseeded).
    """
    seed = run_parameters.get("inner_cv_seed")
    return ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)


def _inner_scoring(task: str):
    return "r2" if task == "regression" else "f1"


def evaluate_joins_native(
    scenario_logger,
    base_table: pl.DataFrame,
    join_candidates,
    target_column: str = "target",
    group_column: str = "col_to_embed",
    estim_parameters: dict | None = None,
    join_parameters: dict | None = None,
    model_parameters: dict | None = None,
    run_parameters: dict | None = None,
):
    """Evaluate the join estimators with the native DataOps engine.

    Produces the same logs as the other engines; see the module docstring for
    the mapping between selectors and plan constructs.

    Args:
        scenario_logger (ScenarioLogger): ScenarioLogger object used to track the results of the runs.
        base_table (pl.DataFrame): Base table to evaluate.
        join_candidates: List of candidate joins that will be used for the estimators.
        target_column (str): Target column that is used for training the model. Defaults to "target".
        group_column (str): Column that will be used for joining. Defaults to "col_to_embed".
        estim_parameters (dict | None): Additional parameters for the estimator. Defaults to None.
        join_parameters (dict | None): Additional parameters to be used in the join operation. Defaults to None.
        model_parameters (dict | None): Additional parameters to be passed to the ML model. Defaults to None.
        run_parameters (dict | None): Additional parameters relative to the run. Defaults to None.
    """
    splits = prepare_splits(run_parameters, base_table, group_column)

    estim_common_parameters = {
        "scenario_logger": None,
        "target_column": target_column,
        "model_parameters": model_parameters,
        "task": run_parameters["task"],
    }

    res_list = []
    add_info_dict = {}

    estimator_names = [
        estim for estim in estim_parameters if estim_parameters[estim]["active"]
    ]
    save_artifacts = run_parameters.get("save_artifacts", False)

    for fold_id, split in tqdm(
        enumerate(splits),
        total=len(splits),
        desc="CV progress: ",
        position=1,
        leave=False,
    ):
        for estim_name in estimator_names:
            params = dict(estim_common_parameters)
            params.update(estim_parameters.get(estim_name, {}))

            estimator = prepare_estimator(
                estim_name,
                params,
                join_candidates,
                join_parameters,
            )
            if estimator is None:
                run_logger = RunLogger(
                    scenario_logger, additional_parameters={}, fold_id=fold_id
                )
                run_logger.set_run_status("FAILURE")
                run_logger.to_run_log_file()
                continue

            run_logger = RunLogger(
                scenario_logger,
                additional_parameters=estimator.get_estimator_parameters(),
                fold_id=fold_id,
            )
            run_logger.start_time("run")

            try:
                use_search = (
                    estim_name == "best_single_join"
                    and join_parameters["aggregation"] != "dfs"
                )
                if use_search:
                    pred, env = _build_best_single_join_plan(
                        run_parameters=run_parameters,
                        model_parameters=model_parameters,
                        group_column=group_column,
                        base_table=base_table,
                        estimator=estimator,
                        splits=[split],
                    )
                    search = pred.skb.make_grid_search(
                        cv=_inner_cv(run_parameters),
                        scoring=_inner_scoring(run_parameters["task"]),
                    )
                else:
                    native_full = (
                        estim_name == "full_join"
                        and join_parameters["aggregation"] != "dfs"
                    )
                    pred, env = build_plan(
                        run_parameters=run_parameters,
                        model_parameters=model_parameters,
                        group_column=group_column,
                        base_table=base_table,
                        estimator=estimator,
                        plan_native_join=native_full,
                        splits=[split],
                    )
                    learner = pred.skb.make_learner()

                split_data = next(pred.skb.iter_cv_splits(env))

                # Execute the fit operation
                _join_timings.clear()
                if use_search:
                    run_logger.profiled_call(
                        "fit", search.fit, split_data["train"], time_label="fit"
                    )
                    join_train_time = sum(_join_timings)
                else:
                    run_logger.profiled_call(
                        "fit", learner.fit, split_data["train"], time_label="fit"
                    )
                    join_train_time = _last_join_time()

                y_test = split_data["y_test"]

                # Execute the predict operation
                _join_timings.clear()
                if use_search:
                    y_pred = run_logger.profiled_call(
                        "predict",
                        search.predict,
                        split_data["test"],
                        time_label="predict",
                        retval=True,
                    )
                else:
                    y_pred = run_logger.profiled_call(
                        "predict",
                        learner.predict,
                        split_data["test"],
                        time_label="predict",
                        retval=True,
                    )
                join_predict_time = _last_join_time()

                # Evaluate the results
                results = run_logger.profiled_call(
                    "test", run_logger.measure_results, y_test, y_pred, retval=True
                )
                curr_res = {"estimator": estimator.name, **results}

                # Additional info includes best candidate join and relative info
                if use_search:
                    # The selected branch is reported as an index in the
                    # named choices of the refitted learner.
                    best_idx = search.best_learner_.get_named_params()["candidate"]
                    best_hash = list(estimator.candidate_joins.keys())[best_idx]
                    best_cjoin = estimator.candidate_joins[best_hash]
                    _, _, left_on, right_on = best_cjoin.get_join_information()
                    fitted_estimator = _get_fitted_estimator(search.best_learner_)
                    additional_info = {
                        "best_candidate_hash": best_hash,
                        "left_on": left_on,
                        "right_on": right_on,
                        "n_joined_columns": fitted_estimator.n_joined_columns,
                    }
                else:
                    fitted_estimator = _get_fitted_estimator(learner)
                    additional_info = fitted_estimator.get_additional_info()

                durations = dict(fitted_estimator.get_durations())
                if use_search or native_full:
                    # The join is a node of the plan; its duration was captured
                    # around the fit/predict calls. For the estimator-internal
                    # methods the durations already include the join phases.
                    durations["time_join_train"] = join_train_time
                    durations["time_join_predict"] = join_predict_time
                run_logger.set_additional_info(additional_info)
                run_logger.update_durations(durations)
            except Exception:
                run_logger.end_time("run")
                run_logger.set_run_status("FAILURE")
                run_logger.to_run_log_file()
                raise

            run_logger.end_time("run")
            run_logger.set_run_status("SUCCESS")
            run_logger.to_run_log_file()
            res_list.append(curr_res)
            add_info_dict[f"{fold_id}_{estimator.name}"] = run_logger.additional_info

            if save_artifacts:
                if use_search:
                    # The graph of the winning candidate branch depends on the
                    # fold, so it is saved per fold; the learner is the refit
                    # winner of the search.
                    _save_artifacts(
                        scenario_logger,
                        estimator.name,
                        fold_id,
                        search.best_learner_,
                        resolved=True,
                    )
                else:
                    _save_artifacts(scenario_logger, estimator.name, fold_id, learner)

    scenario_logger.additional_info = add_info_dict
    scenario_logger.results = pl.from_dicts(res_list)
    scenario_logger.print_results()
