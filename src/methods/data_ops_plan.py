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
import time

import numpy as np
import polars as pl
import skrub
from memory_profiler import memory_usage
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, train_test_split
from tqdm import tqdm

import src.utils.joining as ju
from src.data_structures.loggers import RunLogger
from src.methods.evaluation import prepare_estimator, prepare_splits
from src.methods.join_selectors import BaseJoinSelector

# Durations of the calls to `_augment_all_candidates`. Functions are preserved
# by `deepcopy` (used when skrub clones the plans), so this module-level list
# is shared by all evaluations. The evaluation loop consumes it to report the
# join durations as done by the legacy engine.
_join_timings: list[float] = []


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


@skrub.deferred
def _compute_groups(table: pl.DataFrame, group_column: str) -> np.ndarray:
    """Compute the groups used by the GroupShuffleSplit, mirroring
    `src.methods.evaluation.prepare_splits`."""
    return table.select(
        pl.col(group_column).cast(pl.Categorical).to_physical().alias("group")
    ).to_numpy()


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


class ModelOnlySelector(BaseJoinSelector):
    """Selector that relies on the DataOps plan for the join step.

    The base table provided at `fit` time is expected to be already augmented
    by the DataOps plan. This class only takes care of splitting a validation
    set and fitting/predicting with the model, mirroring the model-related
    behavior of the legacy `FullJoin` estimator.
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

    def fit(self, X, y):
        self._start_time("prepare")
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        self.n_joined_columns = len(X_train.columns)
        self._end_time("prepare")

        self._start_time("model_train")
        self.build_model(X_train)
        self.fit_model(X_train, y_train, validation_set=(X_valid, y_valid))
        self._end_time("model_train")

    def predict(self, X):
        self._start_time("model_predict")
        pred = self.predict_model(X)
        self._end_time("model_predict")
        return pred

    def get_estimator_parameters(self):
        d = super().get_estimator_parameters()
        d.update(self.join_parameters)
        return d

    def get_additional_info(self):
        return {
            "best_candidate_hash": self.name,
            "left_on": "",
            "right_on": "",
            "n_joined_columns": self.n_joined_columns,
        }


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

    base = skrub.var("base_table")
    X = base.skb.apply_func(_prepare_X, target_column=target_column)
    y = base[target_column]

    groups = _compute_groups(base, group_column)
    X = X.skb.mark_as_X(
        cv=_make_cv(run_parameters, splits), split_kwargs={"groups": groups}
    )
    y = y.skb.mark_as_y()

    if plan_native_join:
        if estimator.join_parameters["aggregation"] == "dfs":
            raise ValueError("Plan-native joins do not support the `dfs` aggregation.")
        augmented = X.skb.apply_func(
            _augment_all_candidates,
            candidate_joins=estimator.candidate_joins,
            aggregation=estimator.join_parameters["aggregation"],
        )
        model_selector = ModelOnlySelector(
            target_column=target_column,
            model_parameters=model_parameters,
            task=run_parameters["task"],
            name=estimator.name,
            join_parameters=estimator.join_parameters,
        )
        pred = augmented.skb.apply(FrozenEstimator(model_selector), y=y)
        environment = {"base_table": base_table}
    else:
        pred = X.skb.apply(FrozenEstimator(estimator), y=y)
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
                run_logger.start_time("fit")
                mem_usage = memory_usage(
                    (learner.fit, (split_data["train"],)),
                    timestamps=True,
                    max_iterations=1,
                )
                run_logger.end_time("fit")
                run_logger.mark_memory(mem_usage, "fit")
                join_train_time = _last_join_time()

                y_test = split_data["y_test"]

                # Execute the predict operation
                _join_timings.clear()
                run_logger.start_time("predict")
                mem_usage, y_pred = memory_usage(
                    (learner.predict, (split_data["test"],)),
                    timestamps=True,
                    max_iterations=1,
                    retval=True,
                )
                run_logger.end_time("predict")
                run_logger.mark_memory(mem_usage, "predict")
                join_predict_time = _last_join_time()

                # Evaluate the results
                mem_usage, results = memory_usage(
                    (
                        run_logger.measure_results,
                        (
                            y_test,
                            y_pred,
                        ),
                    ),
                    timestamps=True,
                    max_iterations=1,
                    retval=True,
                )
                run_logger.mark_memory(mem_usage, "test")
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

    scenario_logger.additional_info = add_info_dict
    scenario_logger.results = pl.from_dicts(res_list)
    scenario_logger.print_results()
