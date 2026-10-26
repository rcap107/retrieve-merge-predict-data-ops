"""One DataOps plan for the whole experimental campaign (Tier B).

This module builds a single skrub DataOps plan whose hyperparameters span the
retrieval method, the join selection strategy and the predictor, and runs a
randomized search (Optuna backend) over it. It replaces the manual
configuration combinatorics of `main.py` with hyperparameter search: every
trial of the search corresponds to one configuration of the paper's grid, and
`search.results_` is the benchmark table (parameters + mean CV score per
trial, evaluated on the same folds for all trials).

What stays outside the plan:
- the retrieval/indexing steps (offline, see `query_indices.py`); the plan
  consumes the query results (candidate joins) of every retrieval method;
- the experiment instrumentation (memory profiling, phase timings, run
  logs). The search results are the only artifact.

Semantics differences with respect to the paper (documented in the README):
- the outer CV folds are seeded, so that every trial is scored on the same
  folds;
- `best_single_join`'s candidate selection is *flattened*: the candidate is a
  choice of the search space, selected on the outer folds, instead of the
  legacy inner train/validation selection;
- `highest_containment` and `stepwise_greedy_join` keep their joins inside
  the selector (not plan nodes, not cached), and their aggregation is the
  configuration value rather than a choice;
- the `top_k` choice applies to the `full_join` branch only.
"""

import logging

import polars as pl
import skrub
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from src.methods.data_ops_native import (
    _aggregate_one,
    _join_one,
    _join_one_aggregated,
)
from src.methods.data_ops_plan import (
    FrozenEstimator,
    _aggregate_candidates,
    _augment_all_candidates,
    _compute_groups,
    _join_aggregated,
    _prepare_X,
)
from src.methods.join_selectors import HighestContainmentJoin, StepwiseGreedyJoin
from src.methods.model_core import EvalModel
from src.utils.constants import SUPPORTED_MODELS, SUPPORTED_RETRIEVAL_METHODS

logger = logging.getLogger("data_ops_search")

SELECTOR_ESTIMATORS = ["highest_containment", "stepwise_greedy_join"]
NATIVE_ESTIMATORS = ["no_join", "full_join", "best_single_join"]


def _filter_top_k(candidates: dict, top_k: int) -> dict:
    """Keep the first `top_k` candidates (the dict is in ranking order)."""
    if not top_k or top_k >= len(candidates):
        return candidates
    return dict(list(candidates.items())[:top_k])


def _make_model_parameters(model_parameters: dict, predictor: str) -> dict:
    """Model parameters for one predictor, mirroring the structure expected
    by `EvalModel` (`chosen_model` plus the predictor's own section)."""
    parameters = dict(model_parameters)
    parameters["chosen_model"] = predictor
    return parameters


def _make_predictor_choice(predictors, model_parameters, task, name="chosen_model"):
    """A choice between the predictors, applied directly as the estimator of
    a plan branch."""
    outcomes = {}
    for predictor in predictors:
        model = EvalModel(
            model_parameters=_make_model_parameters(model_parameters, predictor),
            task=task,
        )
        model.name = predictor
        outcomes[predictor] = FrozenEstimator(model)
    return skrub.choose_from(outcomes, name=name)


def _make_native_branch(
    X,
    y,
    candidates,
    method: str,
    estimator_name: str,
    aggregation,
    top_k,
    predictor,
    use_cache: bool,
):
    """Build a plan branch for the estimators whose joins are DataOps nodes.

    `candidates` is the plain dict of candidate joins for the branch's
    retrieval method.
    """
    if estimator_name == "no_join":
        return X.skb.apply(predictor, y=y, no_cache=True)

    if estimator_name == "full_join":
        if use_cache and aggregation not in ("first", "mean"):
            raise ValueError(
                "With the skrub cache enabled, the aggregation must be "
                "`first` or `mean`."
            )
        if use_cache:
            aggregated = skrub.deferred(_aggregate_candidates)(
                skrub.deferred(_filter_top_k)(candidates, top_k), aggregation
            )
            augmented = X.skb.apply_func(_join_aggregated, aggregated, no_cache=True)
        else:
            augmented = X.skb.apply_func(
                _augment_all_candidates,
                candidate_joins=skrub.deferred(_filter_top_k)(candidates, top_k),
                aggregation=aggregation,
                no_cache=True,
            )
        return augmented.skb.apply(predictor, y=y, no_cache=True)

    if estimator_name == "best_single_join":
        # Flattened candidate selection: the candidate itself is a choice of
        # the search space (see the module docstring). The choice is named
        # per method because skrub choice names must be unique.
        candidate = skrub.choose_from(
            list(candidates.values()), name=f"candidate__{method}"
        )
        if use_cache:
            aggregated = skrub.deferred(_aggregate_one)(candidate, aggregation)
            augmented = X.skb.apply_func(
                _join_one_aggregated, aggregated, no_cache=True
            )
        else:
            augmented = X.skb.apply_func(
                _join_one, candidate, aggregation, no_cache=True
            )
        return augmented.skb.apply(predictor, y=y, no_cache=True)

    raise ValueError(f"Unknown native estimator {estimator_name}.")


def _make_selector_branch(
    X,
    y,
    candidates,
    estimator_name: str,
    estim_parameters: dict,
    model_parameters: dict,
    task: str,
    join_parameters: dict,
    predictors,
    choice_name: str,
):
    """Build a plan branch for the selectors that join inside `fit`.

    The selector needs the candidate joins at construction time, so the
    predictor becomes a nested choice over selector instances. The choice
    name must be unique per branch (skrub rejects duplicate names).
    """
    outcomes = {}
    for predictor in predictors:
        parameters = _make_model_parameters(model_parameters, predictor)
        if estimator_name == "highest_containment":
            selector = HighestContainmentJoin(
                candidate_joins=candidates,
                model_parameters=parameters,
                join_parameters=join_parameters,
                task=task,
            )
        elif estimator_name == "stepwise_greedy_join":
            selector = StepwiseGreedyJoin(
                candidate_joins=candidates,
                model_parameters=parameters,
                join_parameters=join_parameters,
                task=task,
                budget_type=estim_parameters.get("budget_type", "iterations"),
                budget_amount=estim_parameters.get("budget_amount", 30),
                epsilon=estim_parameters.get("epsilon", 0.0),
                ranking_metric=estim_parameters.get("ranking_metric", "containment"),
            )
        else:
            raise ValueError(f"Unknown selector estimator {estimator_name}.")
        outcomes[predictor] = FrozenEstimator(selector)
    choice = skrub.choose_from(outcomes, name=choice_name)
    return X.skb.apply(choice, y=y, no_cache=True)


def build_search_plan(
    run_config: dict,
    base_table: pl.DataFrame,
    query_results: dict,
):
    """Build the DataOps plan spanning retrieval methods, estimators and
    predictors.

    Args:
        run_config: The run configuration (with the additional `[search]`
            section, see `search.py`).
        base_table: The cleaned base table.
        query_results: Mapping retrieval method name -> candidate joins dict
            (in ranking order, not clamped by top_k).

    Returns:
        tuple: The DataOps plan and the environment expected by its methods.
    """
    run_parameters = run_config["run_parameters"]
    search_config = run_config.get("search", {})
    task = run_parameters["task"]
    target_column = run_parameters.get("target_column", "target")
    group_column = run_config["query_cases"]["query_column"]
    join_parameters = run_config["join_parameters"]
    model_parameters = run_config["evaluation_models"]

    methods = search_config.get("join_discovery_methods") or list(query_results)
    unknown = [m for m in methods if m not in query_results]
    if unknown:
        raise ValueError(f"No query results for retrieval methods: {unknown}")
    unsupported = [m for m in methods if m not in SUPPORTED_RETRIEVAL_METHODS]
    if unsupported:
        raise ValueError(f"Unsupported retrieval methods: {unsupported}")

    estimators = [
        name
        for name, parameters in run_config["estimators"].items()
        if parameters.get("active", True)
    ]
    unknown_estimators = [
        e for e in estimators if e not in NATIVE_ESTIMATORS + SELECTOR_ESTIMATORS
    ]
    if unknown_estimators:
        raise ValueError(f"Unknown estimators: {unknown_estimators}")

    predictors = model_parameters.get("chosen_model")
    if isinstance(predictors, str):
        predictors = [predictors]
    bad_predictors = [p for p in predictors if p not in SUPPORTED_MODELS]
    if bad_predictors:
        raise ValueError(
            f"Unsupported predictors {bad_predictors}; supported: {SUPPORTED_MODELS}"
        )

    aggregation_choices = search_config.get("aggregation_choices", ["first", "mean"])
    bad_aggregations = [a for a in aggregation_choices if a not in ("first", "mean")]
    if bad_aggregations:
        raise ValueError(
            f"Aggregation choices must be in ('first', 'mean'), got {bad_aggregations}"
        )
    aggregation = skrub.choose_from(aggregation_choices, name="aggregation")

    top_k_choices = search_config.get("top_k_choices", [])
    top_k = skrub.choose_from(top_k_choices, name="top_k") if top_k_choices else None

    use_cache = search_config.get("cache_dir") is not None

    base = skrub.var("base_table")
    X = base.skb.apply_func(_prepare_X, target_column=target_column, no_cache=True)
    y = base[target_column]

    groups = _compute_groups(base, group_column)
    # The folds are seeded so that every trial of the search is scored on the
    # same splits.
    X = X.skb.mark_as_X(
        cv=GroupShuffleSplit(
            n_splits=run_parameters["n_splits"],
            test_size=run_parameters["test_size"],
            random_state=search_config.get("seed"),
        ),
        split_kwargs={"groups": groups},
    )
    y = y.skb.mark_as_y()

    predictor = _make_predictor_choice(predictors, model_parameters, task)

    branches = {}
    for method in methods:
        candidates = query_results[method]
        for estimator_name in estimators:
            if estimator_name == "no_join":
                # Independent of the retrieval method: a single branch.
                if "no_join" not in branches:
                    branches["no_join"] = _make_native_branch(
                        X,
                        y,
                        candidates,
                        method,
                        "no_join",
                        aggregation,
                        top_k,
                        predictor,
                        use_cache,
                    )
                continue
            key = f"{method}__{estimator_name}"
            if estimator_name in NATIVE_ESTIMATORS:
                branches[key] = _make_native_branch(
                    X,
                    y,
                    candidates,
                    method,
                    estimator_name,
                    aggregation,
                    top_k,
                    predictor,
                    use_cache,
                )
            else:
                branches[key] = _make_selector_branch(
                    X,
                    y,
                    candidates,
                    estimator_name,
                    run_config["estimators"].get(estimator_name, {}),
                    model_parameters,
                    task,
                    join_parameters,
                    predictors,
                    choice_name=f"chosen_model__{key}",
                )

    if len(branches) == 1:
        (only_branch,) = branches.values()
        return only_branch, {"base_table": base_table}
    plan = skrub.choose_from(branches, name="config").as_data_op()
    return plan, {"base_table": base_table}


def run_search(run_config: dict, base_table: pl.DataFrame, query_results: dict):
    """Build the search plan and run the randomized search over it.

    Returns:
        The fitted `ParamSearch`; `search.results_` maps every trial
        (parameter combination) to its mean cross-validated score.
    """
    search_config = run_config.get("search", {})
    task = run_config["run_parameters"]["task"]

    pred, env = build_search_plan(run_config, base_table, query_results)

    cache_dir = search_config.get("cache_dir")
    if cache_dir is not None:
        skrub.set_config(cache=cache_dir)
    try:
        search = pred.skb.make_randomized_search(
            backend=search_config.get("backend", "optuna"),
            n_iter=search_config.get("n_iter", 20),
            scoring="r2" if task == "regression" else "f1",
            random_state=search_config.get("seed"),
            storage=search_config.get("storage"),
            study_name=search_config.get("study_name"),
            n_jobs=search_config.get("n_jobs"),
        )
        search.fit(env)
    finally:
        if cache_dir is not None:
            skrub.set_config(cache=None)
    return search
