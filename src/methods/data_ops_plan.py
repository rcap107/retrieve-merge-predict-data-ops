"""Evaluation of join selector methods using skrub DataOps plans.

This module provides an alternative evaluation engine for the pipeline that
expresses the whole per-run procedure (data preparation, train/test splitting
and model training) as a skrub DataOps plan.

The "data_ops" engine builds a plan for each (fold, estimator) pair and
executes it with a :class:`skrub.SkrubLearner`, keeping the outer evaluation
loop (memory profiling, timings, logging) identical to the legacy engine so
that the produced logs are compatible.

Joins and aggregations of candidate tables are kept in `src.utils.joining`
so that both engines share the exact same semantics.
"""

import copy
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import skrub
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from tqdm import tqdm

import src.utils.joining as ju
from src.data_structures.loggers import RunLogger
from src.methods.evaluation import prepare_estimator, prepare_splits
from src.methods.model_core import EvalModel

logger = logging.getLogger("data_ops_plan")

# Durations of the calls to `_augment_all_candidates`. Functions are preserved
# by `deepcopy` (used when skrub clones the plans), so this module-level list
# is shared by all evaluations. The evaluation loop consumes it to report the
# join durations as done by the legacy engine.
_join_timings: list[float] = []


def _save_artifacts(
    scenario_logger,
    estimator_name: str,
    fold_id: int,
    learner,
    resolved: bool = False,
):
    """Save the computation graph and the fitted learner of a run.

    Enabled by `run_parameters["save_artifacts"]`. The graph is written once
    per estimator (the structure of the plan does not change across folds),
    except when `resolved` is set (native `best_single_join`): the graph of
    the winning candidate branch depends on the fold, so it is written per
    fold. The fitted learner is overwritten at every fold, so the saved one
    corresponds to the last successful fold; it can be re-applied to new data
    with `learner.predict({"base_table": table})`.

    Failures to save artifacts are logged as warnings and never interrupt the
    evaluation.
    """
    try:
        root = Path(f"results/logs/{scenario_logger.exp_name}")
        graphs_dir = root / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        if resolved:
            graph_path = graphs_dir / f"{estimator_name}_fold{fold_id}.svg"
        else:
            graph_path = graphs_dir / f"{estimator_name}.svg"
        if not graph_path.exists():
            drawing = learner.data_op.skb.draw_graph()
            graph_path.write_bytes(drawing.svg)

        learners_dir = root / "learners"
        learners_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(learner, learners_dir / f"{estimator_name}.joblib")
        scenario_logger.learners[estimator_name] = learner
    except Exception as exception:
        logger.warning("Could not save artifacts for %s: %s", estimator_name, exception)


class FrozenEstimator:
    """Wrapper that makes a pre-built estimator safe for DataOps plans.

    skrub clones (or deep-copies) the estimators added to a plan through
    `.skb.apply`. The join selectors are not compatible with scikit-learn's
    `get_params`/`set_params` round-trip, so they are wrapped and duplicated
    with `deepcopy` instead, which preserves them exactly.
    """

    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def __sklearn_clone__(self):
        return copy.deepcopy(self)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def __getattr__(self, name):
        # Delegate attribute access (e.g., `name`, `get_durations`,
        # `get_additional_info`) to the wrapped estimator. Accessing
        # `estimator` itself must fail cleanly when the wrapper is not yet
        # initialized (e.g., during deepcopy reconstruction).
        try:
            estimator = self.__dict__["estimator"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(estimator, name)

    def fit(self, X, y):
        return self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict(X)

    @property
    def name(self):
        return self.estimator.name


def _prepare_X(df: pl.DataFrame, target_column: str) -> pl.DataFrame:
    """Prepare the design matrix from a (part of a) base table.

    This mirrors `src.methods.evaluation.prepare_X_y`: fill nulls and nans,
    then drop the target column. The target column is left untouched, and `y`
    is extracted directly from the base table so that it matches the legacy
    behavior.
    """
    df = df.clone().fill_null(value="null").fill_nan(value=np.nan)
    return df.drop(target_column)


def _compute_groups_impl(table: pl.DataFrame, group_column: str) -> np.ndarray:
    """Compute the groups used by the GroupShuffleSplit, mirroring
    `src.methods.evaluation.prepare_splits`.

    Note: the undecorated function is bound to its own module attribute
    (instead of being shadowed by the deferred wrapper) so that plans
    containing it can be pickled by reference.
    """
    return table.select(
        pl.col(group_column).cast(pl.Categorical).to_physical().alias("group")
    ).to_numpy()


_compute_groups = skrub.deferred(_compute_groups_impl)


def _augment_all_candidates(
    X_part: pl.DataFrame,
    candidate_joins: dict,
    aggregation: str,
) -> pl.DataFrame:
    """Join and aggregate all candidate tables onto a part of the base table.

    This is a stateless operation, hence it can be expressed as a node of the
    DataOps plan. Its duration is recorded in `_join_timings` and reported by
    the evaluation loop as `time_join_train` (fit) or `time_join_predict`
    (predict), as done by the legacy engine.
    """
    start = time.perf_counter()
    merged = ju.execute_join_all_candidates(
        X_part.clone().lazy(), candidate_joins, aggregation
    )
    _join_timings.append(time.perf_counter() - start)
    return merged


def _aggregate_candidates(candidate_joins: dict, aggregation: str) -> dict:
    """Aggregate every candidate table on its join key.

    This operation depends only on the candidate tables and the aggregation
    method, not on the base table: the resulting node is fold-independent and
    its result can be cached (on disk with the skrub cache, or in memory)
    across folds and runs. Only usable with the `first` and `mean`
    aggregations, whose result per join key does not depend on the rows of
    the left table (the `dfs` aggregation depends on the left table and must
    stay fused with the join).

    Returns:
        dict: For each candidate hash, a tuple with the candidate metadata,
        the aggregated table and the join keys.
    """
    aggregated = {}
    for hash_, mdata in candidate_joins.items():
        _, cnd_md, left_on, right_on = mdata.get_join_information()
        cand_table = pl.read_parquet(cnd_md["full_path"])
        aggr_table = ju.aggregate_table(cand_table, right_on, aggregation)
        aggregated[hash_] = (mdata, aggr_table, left_on, right_on)
    return aggregated


def _join_aggregated(left_part: pl.DataFrame, aggregated: dict) -> pl.DataFrame:
    """Join an (already aggregated) set of candidate tables onto a part of the
    base table. Counterpart of `_aggregate_candidates`; mirrors the join order
    and suffixes of `execute_join_all_candidates`."""
    start = time.perf_counter()
    merged = left_part.clone().lazy()
    for hash_, (mdata, aggr_table, left_on, right_on) in aggregated.items():
        merged = ju.execute_join(
            merged,
            aggr_table,
            left_on=left_on,
            right_on=right_on,
            how="left",
            suffix="_" + hash_[:10],
        )
    _join_timings.append(time.perf_counter() - start)
    return merged


class ModelOnlySelector(EvalModel):
    """Selector that relies on the DataOps plan for the join step.

    The base table provided at `fit` time is expected to be already augmented
    by the DataOps plan. The model logic (inner validation split, model
    building and fitting) is inherited from `EvalModel`.
    """

    def __init__(
        self,
        scenario_logger=None,
        target_column: str | None = None,
        model_parameters: dict | None = None,
        task: str = "regression",
        name: str = "full_join",
        join_parameters: dict | None = None,
    ) -> None:
        super().__init__(scenario_logger, target_column, model_parameters, task)
        self.name = name
        self.join_parameters = join_parameters if join_parameters is not None else {}

    def get_estimator_parameters(self):
        d = super().get_estimator_parameters()
        d.update(self.join_parameters)
        return d


def _get_fitted_estimator(learner):
    """Retrieve the estimator fitted by a learner, walking its DataOp graph."""

    def _walk(node):
        impl = getattr(node, "_skrub_impl", None)
        if impl is None:
            return None
        estimator = getattr(impl, "estimator_", None)
        if estimator is not None and hasattr(estimator, "get_durations"):
            return estimator
        for field in getattr(impl, "_fields", []):
            value = getattr(impl, field, None)
            children = value if isinstance(value, list) else [value]
            for child in children:
                if hasattr(child, "_skrub_impl"):
                    found = _walk(child)
                    if found is not None:
                        return found
        return None

    return _walk(learner.data_op)


def _last_join_time() -> float:
    """Duration of the last call to `_augment_all_candidates` (0.0 if none)."""
    return _join_timings[-1] if _join_timings else 0.0


def _make_cv(run_parameters: dict, splits: list | None):
    """Prepare the cross-validation strategy for the plan.

    If explicit `splits` are provided (e.g., for parity testing), they are used
    directly. Otherwise, a splitter equivalent to the legacy `prepare_splits`
    is built. Note that, as in the legacy engine, the splitters are not seeded,
    so the procedure is reproducible at the procedure level rather than
    bit-for-bit.
    """
    if splits is not None:
        return splits
    split_kind = run_parameters["split_kind"]
    n_splits = run_parameters["n_splits"]
    test_size = run_parameters["test_size"]
    if split_kind == "group_shuffle":
        return GroupShuffleSplit(
            n_splits=n_splits, test_size=test_size, train_size=None
        )
    if split_kind == "shuffle":
        return ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=None)
    raise ValueError(f"Inappropriate value {split_kind} for `split_kind`.")


def build_plan(
    run_parameters: dict,
    model_parameters: dict,
    group_column: str,
    base_table: pl.DataFrame | None = None,
    estimator=None,
    plan_native_join: bool = False,
    splits: list | None = None,
):
    """Build the DataOps plan for a single (fold, estimator) evaluation.

    Args:
        run_parameters: The `run_parameters` section of the run configuration.
        model_parameters: The `evaluation_models` section of the run config.
        group_column: Column used for joining, also used as grouping variable
            for the train/test split.
        base_table: The base table value to bind to the plan's variable.
        estimator: The join selector estimator to apply on the base table.
        plan_native_join: If True, the augmentation (join of all candidates
            with aggregation) is performed by a node of the plan instead of
            inside the estimator's `fit`. Only valid for `full_join` without
            the `dfs` aggregation.
        splits: Optional explicit list of (train, test) index pairs, used
            instead of a splitter object (for parity testing).

    Returns:
        tuple: The DataOps plan and the environment expected by its methods.
    """
    target_column = run_parameters.get("target_column", "target")
    use_cache = run_parameters.get("cache_dir") is not None

    base = skrub.var("base_table")
    X = base.skb.apply_func(_prepare_X, target_column=target_column, no_cache=True)
    y = base[target_column]

    groups = _compute_groups(base, group_column)
    X = X.skb.mark_as_X(
        cv=_make_cv(run_parameters, splits), split_kwargs={"groups": groups}
    )
    y = y.skb.mark_as_y()

    if plan_native_join:
        if estimator.join_parameters["aggregation"] == "dfs":
            raise ValueError("Plan-native joins do not support the `dfs` aggregation.")
        if use_cache and estimator.join_parameters["aggregation"] in ("first", "mean"):
            # Split aggregation and join: the aggregated candidates do not
            # depend on the fold, so their computation can be cached across
            # folds and runs.
            aggregated = skrub.deferred(_aggregate_candidates)(
                estimator.candidate_joins,
                estimator.join_parameters["aggregation"],
            )
            augmented = X.skb.apply_func(_join_aggregated, aggregated, no_cache=True)
        else:
            augmented = X.skb.apply_func(
                _augment_all_candidates,
                candidate_joins=estimator.candidate_joins,
                aggregation=estimator.join_parameters["aggregation"],
                no_cache=True,
            )
        model_selector = ModelOnlySelector(
            target_column=target_column,
            model_parameters=model_parameters,
            task=run_parameters["task"],
            name=estimator.name,
            join_parameters=estimator.join_parameters,
        )
        pred = augmented.skb.apply(FrozenEstimator(model_selector), y=y, no_cache=True)
        environment = {"base_table": base_table}
    else:
        pred = X.skb.apply(FrozenEstimator(estimator), y=y, no_cache=True)
        environment = {"base_table": base_table}

    return pred, environment


def evaluate_joins_data_ops(
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
    """Evaluate the join estimators on the given base table and join candidates
    using skrub DataOps plans.

    This function mirrors `src.methods.evaluation.evaluate_joins` and produces
    identical logs, replacing the manual fold loop with DataOps.

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
    plan_native = run_parameters.get("plan_native_join", False)
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

            # The plan-native path is only available for the `full_join`
            # estimator, whose augmentation is stateless. All other selectors
            # perform their joins inside `fit` because their candidate
            # selection depends on the training data.
            native = plan_native and estim_name == "full_join"
            if native and join_parameters["aggregation"] == "dfs":
                native = False

            run_logger = RunLogger(
                scenario_logger,
                additional_parameters=estimator.get_estimator_parameters(),
                fold_id=fold_id,
            )
            run_logger.start_time("run")

            try:
                pred, env = build_plan(
                    run_parameters=run_parameters,
                    model_parameters=model_parameters,
                    group_column=group_column,
                    base_table=base_table,
                    estimator=estimator,
                    plan_native_join=native,
                    splits=[split],
                )
                learner = pred.skb.make_learner()

                split_data = next(pred.skb.iter_cv_splits(env))

                # Execute the fit operation
                _join_timings.clear()
                run_logger.profiled_call(
                    "fit", learner.fit, split_data["train"], time_label="fit"
                )
                join_train_time = _last_join_time()

                y_test = split_data["y_test"]

                # Execute the predict operation
                _join_timings.clear()
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
                fitted_estimator = _get_fitted_estimator(learner)
                durations = dict(fitted_estimator.get_durations())
                if native:
                    # The join is a node of the plan; its duration was captured
                    # around the fit/predict calls.
                    durations["time_join_train"] = join_train_time
                    durations["time_join_predict"] = join_predict_time
                run_logger.set_additional_info(fitted_estimator.get_additional_info())
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
                _save_artifacts(scenario_logger, estimator.name, fold_id, learner)

    scenario_logger.additional_info = add_info_dict
    scenario_logger.results = pl.from_dicts(res_list)
    scenario_logger.print_results()


REGRESSION_SCORING = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
CLASSIFICATION_SCORING = {"f1": "f1", "auc": "roc_auc"}


def _cv_results_frame(cv_results, estimator_name: str, task: str) -> pl.DataFrame:
    """Convert the pandas DataFrame returned by `skb.cross_validate` into the
    results frame used by the ScenarioLogger (one row per fold)."""
    if task == "regression":
        return pl.DataFrame(
            {
                "estimator": [estimator_name] * len(cv_results),
                "r2": cv_results["test_r2"].to_numpy(),
                "rmse": -cv_results["test_rmse"].to_numpy(),
            }
        )
    if task == "classification":
        return pl.DataFrame(
            {
                "estimator": [estimator_name] * len(cv_results),
                "f1": cv_results["test_f1"].to_numpy(),
                "auc": cv_results["test_auc"].to_numpy(),
            }
        )
    raise ValueError(f"Task {task} not supported.")


def evaluate_joins_light(
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
    """Lightweight evaluation of the join estimators using skrub DataOps.

    Intended for debugging and fast iteration: one plan is built per active
    estimator and cross-validated with a single call to `skb.cross_validate`.
    The splitter in `mark_as_X` drives the folds, so no manual loop is needed.

    Differently from `evaluate_joins_data_ops`, no memory profiling is
    performed, no run log files are produced and the phase timings
    (`time_join_train`, etc.) are not reported. The results are stored in the
    scenario logger so that the standard analysis tools can be used.

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
    task = run_parameters["task"]
    scoring = REGRESSION_SCORING if task == "regression" else CLASSIFICATION_SCORING

    estim_common_parameters = {
        "scenario_logger": None,
        "target_column": target_column,
        "model_parameters": model_parameters,
        "task": task,
    }

    estimator_names = [
        estim for estim in estim_parameters if estim_parameters[estim]["active"]
    ]
    if len(estimator_names) == 0:
        raise ValueError("No active estimators were found.")

    res_frames = []
    save_artifacts = run_parameters.get("save_artifacts", False)
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
            continue

        # One plan per estimator; the folds are driven by the splitter
        # configured on `mark_as_X`.
        pred, env = build_plan(
            run_parameters=run_parameters,
            model_parameters=model_parameters,
            group_column=group_column,
            base_table=base_table,
            estimator=estimator,
            splits=None,
        )
        cv_results = pred.skb.cross_validate(
            environment=env, scoring=scoring, return_learner=save_artifacts
        )
        res_frames.append(_cv_results_frame(cv_results, estimator.name, task))

        if save_artifacts:
            learners = cv_results["learner"]
            for fold_id, learner in enumerate(learners):
                _save_artifacts(scenario_logger, estimator.name, fold_id, learner)

    scenario_logger.results = pl.concat(res_frames, how="vertical")
    scenario_logger.print_results()
