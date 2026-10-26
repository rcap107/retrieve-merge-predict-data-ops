import random
import warnings
from collections import deque
from itertools import islice
from multiprocessing import cpu_count
from typing import Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from tqdm import tqdm

import src.utils.joining as ju
from src.data_structures.loggers import ScenarioLogger
from src.data_structures.metadata import CandidateJoin
from src.methods.model_core import EvalModel

warnings.filterwarnings("ignore")

MAX_THREADS = min([cpu_count(), 32])

SUPPORTED_BUDGET_TYPES = ["iterations"]
SUPPORTED_RANKING_METHODS = ["containment"]


def measure_containment(
    source_table: pl.DataFrame, cand_table: pl.DataFrame, left_on: list, right_on: list
):
    """This function measures the exact Jaccard containment of the query column
    `left_on` in `source_table` in column `right_on` in table `cand_table`.

    Containment is defined as JC(Q,X) = |Q intersection X| / |Q|

    Args:
        source_table (pl.DataFrame): Source table.
        cand_table (pl.DataFrame): Candidate table to query on.
        left_on (list): Query column.
        right_on (list): Candidate column

    Returns:
        float: Containment value.

    Raises:
        NotImplmentedError if more than one column is provided.
    """
    if len(left_on) > 1 or len(right_on) > 1:
        raise NotImplementedError("Only single query columns are supported")

    unique_source = source_table[left_on].unique()
    unique_cand = cand_table[right_on].unique()

    s1 = set(unique_source[left_on].to_series().to_list())
    s2 = set(unique_cand[right_on].to_series().to_list())
    return len(s1.intersection(s2)) / len(s1)


def build_containment_ranking(X: pd.DataFrame, candidate_joins: dict):
    """Given a table X and a list of candidates in any ranking, re-rank everything
    based on Jaccard containment.

    Args:
        X (pd.DataFrame): The base table to evaluate the containment on.
        candidate_joins (dict): All the candidate join objects to be ranked.

    Returns:
        pl.DataFrame: The candidate ranking.
    """
    containment_list = []
    for hash_, mdata in tqdm(
        candidate_joins.items(),
        total=len(candidate_joins),
        leave=False,
        desc="Building Containment Ranking: ",
        position=2,
    ):
        _, cnd_md, left_on, right_on = mdata.get_join_information()
        cnd_table = pl.read_parquet(cnd_md["full_path"])
        containment = measure_containment(X, cnd_table, left_on, right_on)
        containment_list.append(
            {
                "candidate": hash_,
                "table_hash": mdata.candidate_table,
                "containment": containment,
            }
        )
    ranking_df = pl.from_dicts(containment_list).sort(by="containment", descending=True)

    # If a table is present multiple times, drop the candidates with lower containment
    # ranking_df = ranking_df.unique("table_hash", keep="first", maintain_order=True)

    return ranking_df


def _split_X_y(X, y, test_size=0.2):
    # I want the indices for the
    s = ShuffleSplit(test_size=test_size)
    train_idx, valid_idx = next(s.split(X, y))

    return train_idx, valid_idx


class BaseJoinSelector(EvalModel):
    """BaseJoinSelector class. This class is extended by the other estimators.
    Note that, despite it being called `BaseJoinSelector`, this class is used as base for the NoJoin estimator as well.

    The model-related logic (building, fitting and predicting with the ML
    models, the inner train/validation split and the phase timers) lives in
    `EvalModel` (src.methods.model_core); this class defines the interface of
    the join selection strategies.
    """

    def fit(self, X, y):
        """Abstract method, to be extended by other estimators. It should take a
        prepared table `X` and a suitable target array `y`. It will execute all
        the operations required to fit prepare and fit the model to be used later
        in the `predict` step.

        Args:
            X (pd.DataFrame): Input dataframe.
            y (pd.Series): Target column.
        """
        raise NotImplementedError

    def predict(self, X):
        """Abstract method, to be extended by other estimators.

        It should take a prepared table `X`, which is fed to the prepared model.
        If additional operations are required, they are executed (e.g., joining
        X on a candidate table) before calling self.model.predict.

        Args:
            X (pd.DataFrame): Input dataframe.
        """
        raise NotImplementedError

    def get_additional_info(self):
        raise NotImplementedError


class BaseJoinWithCandidatesMethod(BaseJoinSelector):
    """Base class that extends `BaseJoinMethod` to account for cases where multiple candidate joins are provided.

    Args:
        BaseJoinMethod (_type_): _description_
    """

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        candidate_joins=None,
        target_column: str = None,
        model_parameters: dict = None,
        join_parameters: dict = None,
        task: str = "regression",
    ) -> None:
        super().__init__(scenario_logger, target_column, model_parameters, task)
        self.candidate_joins = candidate_joins
        self.n_candidates = len(candidate_joins)

        _join_params = {"aggregation": "first"}
        _join_params.update(join_parameters)
        self.join_parameters = _join_params

    def _execute_joins(
        self, X_train, X_valid=None, cnd_table=None, left_on=None, right_on=None
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        # Execute the join on the main table
        merged_train = ju.execute_join_with_aggregation(
            X_train,
            cnd_table,
            left_on=left_on,
            right_on=right_on,
            aggregation=self.join_parameters["aggregation"],
        )

        # If a validation split is provided, join that separately
        # Join separately to avoid aggr. leakage
        if X_valid is not None:
            merged_valid = ju.execute_join_with_aggregation(
                X_valid,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
            )
            return merged_train, merged_valid

        return merged_train

    def _evaluate_candidate(self, y_true, y_pred):
        if self.task == "regression":
            return r2_score(y_true, y_pred)
        return f1_score(y_true, y_pred)


class NoJoin(BaseJoinSelector):
    """Reference selector that just trains a model on the base table and predicts on it, with no joins."""

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        target_column: str = None,
        model_parameters: dict = None,
        task: str = "regression",
    ) -> None:
        super().__init__(scenario_logger, target_column, model_parameters, task)
        self.name = "nojoin"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_set: tuple | None = None,
    ):
        if X.shape[0] != y.shape[0]:
            raise ValueError

        with self.timed("prepare"):
            if validation_set is not None:
                X_valid, y_valid = validation_set
                X_train, y_train = X, y
            else:
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X, y, test_size=0.2
                )

            self.build_model(X_train)

            self.n_joined_columns = len(X_train.columns)

        with self.timed("model_train"):
            self.fit_model(X_train, y_train, validation_set=(X_valid, y_valid))
            # else:
            #     with self.timed("prepare"):
            #         self.build_model(X)
            #         self.n_joined_columns = len(X.columns)

            #     with self.timed("model_train"):
            #         self.fit_model(X, y)

    def predict(self, X):
        with self.timed("model_predict"):
            pred = self.predict_model(X)
        return pred

    def get_additional_info(self):
        return {
            "best_candidate_hash": "nojoin",
            "left_on": "",
            "right_on": "",
            "n_joined_columns": self.n_joined_columns,
        }


class HighestContainmentJoin(BaseJoinWithCandidatesMethod):
    def __init__(
        self,
        scenario_logger: ScenarioLogger | None = None,
        candidate_joins: dict | None = None,
        target_column: str | None = None,
        model_parameters: dict | None = None,
        join_parameters: dict | None = None,
        task: str = "regression",
    ) -> None:
        super().__init__(
            scenario_logger,
            candidate_joins,
            target_column,
            model_parameters,
            join_parameters,
            task,
        )
        self.name = "highest_containment"
        self.candidate_ranking = None
        self.cat_features = []
        self.best_cnd_hash = None
        self.best_cnd_table = None
        self.left_on = self.right_on = None

    def fit(self, X, y):
        with self.timed("prepare"):
            # Find the exact containment ranking
            self.candidate_ranking = build_containment_ranking(X, self.candidate_joins)
            # Select the top-1 candidate
            self.best_cnd_hash = self.candidate_ranking.top_k(1, by="containment")[
                "candidate"
            ].item()

            mdata = self.candidate_joins[self.best_cnd_hash]  # type: ignore
            _, best_cnd_md, self.left_on, self.right_on = mdata.get_join_information()

            self.best_cnd_table = pl.read_parquet(best_cnd_md["full_path"])

            # Split in train and validation
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        with self.timed("join_train"):
            # Execute join separately on each split (to avoid contamination with aggr)
            merged_train, merged_valid = self._execute_joins(
                X_train=X_train,
                X_valid=X_valid,
                cnd_table=self.best_cnd_table,
                left_on=self.left_on,
                right_on=self.right_on,
            )
            self.n_joined_columns = len(merged_train.columns)
            # else:
            #     merged_train = self._execute_joins(
            #         X_train=X,
            #         cnd_table=self.best_cnd_table,
            #         left_on=self.left_on,
            #         right_on=self.right_on,
            #     )
            #     self.n_joined_columns = len(merged_train.columns)

        with self.timed("model_train"):
            self.build_model(merged_train)

            # Fit the given model
            self.fit_model(
                merged_train, y_train, validation_set=(merged_valid, y_valid)
            )

    def predict(self, X):
        with self.timed("join_predict"):
            merged_test = ju.execute_join_with_aggregation(
                X,
                self.best_cnd_table,
                left_on=self.left_on,
                right_on=self.right_on,
                aggregation=self.join_parameters["aggregation"],
                suffix="_right",
            )
        with self.timed("model_predict"):
            pred = self.predict_model(merged_test)
        return pred

    def get_estimator_parameters(self):
        d = super().get_estimator_parameters()
        d.update(self.join_parameters)
        return d

    def get_additional_info(self):
        return {
            "best_candidate_hash": self.best_cnd_hash,
            "left_on": self.left_on,
            "right_on": self.right_on,
            "n_joined_columns": self.n_joined_columns,
        }


class BestSingleJoin(BaseJoinWithCandidatesMethod):
    # pylint: disable=too-many-instance-attributes
    """
    The `BestSingleJoin` estimator takes at init time a set of candidates that will be trained at `fit` time to
    find the best single candidate among them.

    The selection of the "best candidate" is done at fit time.
    """

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        candidate_joins: dict = None,
        target_column: str = None,
        model_parameters: dict = None,
        join_parameters: dict = None,
        task: str = "regression",
        valid_size: float = 0.2,
        use_rf: bool = False,
    ) -> None:
        """
        Args:
            scenario_logger (ScenarioLogger, optional): ScenarioLogger object that contains the parameters relative to this run
            candidate_joins (dict, optional): Dictionary that includes all generated candidate joins
            target_column (str, optional): Target column.
            model_parameters (dict, optional): Additional parameters to be passed to the evaluation model (catboost or linear).
            join_parameters (dict, optional): Additional parameters to be considered when executing the join.
            task (str, optional): The task to be executed. Either "classification" or "regression". Defaults to "regression".
            valid_size (float, optional): The size of the validation set to be held out for evaluation of the best join. Defaults to 0.2.
        """
        super().__init__(
            scenario_logger,
            candidate_joins,
            target_column,
            model_parameters,
            join_parameters,
            task,
        )
        self.use_rf = use_rf
        self.name = "best_single_join"
        if use_rf:
            self.name += "_rf"
        self.candidate_ranking = None

        self.best_cnd_hash = None
        self.best_cjoin = None
        self.best_cnd_metric = -np.inf
        self.best_cnd_table = None
        self.left_on = self.right_on = None
        self.valid_size = valid_size

    def get_best_candidates(self, top_k=None):
        if top_k is None:
            return self.candidate_ranking
        return self.candidate_ranking.limit(top_k)

    def fit(self, X, y):
        with self.timed("prepare"):
            # Prepare splits
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        ranking = []

        for hash_, cjoin in tqdm(
            self.candidate_joins.items(),
            total=self.n_candidates,
            leave=False,
            desc="BestSingleJoin",
            position=2,
        ):
            # Join on the candidate
            self._start_time("join_train")
            _, cnd_md, left_on, right_on = cjoin.get_join_information()
            cnd_table = pl.read_parquet(cnd_md["full_path"])

            merged_train, merged_valid = self._execute_joins(
                X_train, X_valid, cnd_table, left_on, right_on
            )

            self._end_time("join_train")

            # NOTE: the `model_train` region ends in different places depending
            # on the branch (inside `_fit_predict_inner_model` for `use_rf`),
            # so explicit timer calls are kept here.
            self._start_time("model_train")
            # Build the model
            self.build_model(merged_train)

            method = "rf"
            if self.use_rf:
                inner_model = self._prepare_inner_model()
                y_pred = self._fit_predict_inner_model(
                    merged_train, merged_valid, y_train, inner_model
                )
            else:
                self.fit_model(
                    merged_train, y_train, validation_set=(merged_valid, y_valid)
                )
                self._end_time("model_train")
                self._start_time("prepare")
                y_pred = self.predict_model(merged_valid)
            metric = self._evaluate_candidate(y_valid, y_pred)

            ranking.append({"candidate": hash_, "metric": metric})

            if metric > self.best_cnd_metric:
                self.best_cnd_metric = metric
                self.best_cnd_hash = hash_
                self.best_cjoin = cjoin.candidate_id
            self._end_time("prepare")

        # RETRAINING THE MODEL
        with self.timed("prepare"):
            best_join_mdata = self.candidate_joins[self.best_cnd_hash]
            (
                _,
                best_cnd_md,
                self.left_on,
                self.right_on,
            ) = best_join_mdata.get_join_information()

            self.best_cnd_table = pl.read_parquet(best_cnd_md["full_path"])
        with self.timed("join_train"):
            best_train, best_valid = self._execute_joins(
                X_train, X_valid, self.best_cnd_table, self.left_on, self.right_on
            )
            self.n_joined_columns = len(best_train.columns)

        with self.timed("model_train"):
            self.build_model(best_train)

            self.fit_model(best_train, y_train, validation_set=(best_valid, y_valid))
            self.candidate_ranking = pl.from_dicts(ranking).sort(
                "metric", descending=True
            )

    def predict(self, X):
        with self.timed("join_predict"):
            merged_test = ju.execute_join_with_aggregation(
                X,
                self.best_cnd_table,
                left_on=self.left_on,
                right_on=self.right_on,
                aggregation=self.join_parameters["aggregation"],
                suffix="_right",
            )

        with self.timed("model_predict"):
            pred = self.predict_model(merged_test)

        return pred

    def get_estimator_parameters(self):
        d = super().get_estimator_parameters()
        d.update(self.join_parameters)
        return d

    def get_additional_info(self):
        return {
            "best_candidate_hash": self.best_cjoin,
            "left_on": self.left_on,
            "right_on": self.right_on,
            "n_joined_columns": self.n_joined_columns,
        }

    def _execute_joins(
        self, X_train, X_valid=None, cnd_table=None, left_on=None, right_on=None
    ):
        merged_train = ju.execute_join_with_aggregation(
            X_train,
            cnd_table,
            left_on=left_on,
            right_on=right_on,
            aggregation=self.join_parameters["aggregation"],
        )

        if X_valid is not None:
            merged_valid = ju.execute_join_with_aggregation(
                X_valid,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
            )
            return merged_train, merged_valid
        else:
            return merged_train


class FullJoin(BaseJoinWithCandidatesMethod):
    """The `FullJoin` estimator executes the full left join (with aggregation) between the
    input table and all provided candidates, then trains a model on the resulting table.
    """

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        candidate_joins: dict = None,
        target_column: str = None,
        model_parameters: dict = None,
        join_parameters: dict = None,
        task: str = "regression",
        top_k: None | int = None,
    ) -> None:
        super().__init__(
            scenario_logger,
            candidate_joins,
            target_column,
            model_parameters,
            join_parameters,
            task,
        )
        if top_k is None:
            self.name = "full_join"
        else:
            if top_k < 1:
                raise ValueError(f"k must be >= 1, found {top_k}.")
            if not isinstance(top_k, int):
                raise TypeError(f"top_k must be integer, found {type(top_k)}")
            self.top_k = top_k
            self.name = f"top_{top_k}_full_join"
            self.candidate_joins = {
                k: v
                for k, v in self.candidate_joins.items()
                if k in islice(self.candidate_joins, top_k)
            }

        if self.join_parameters["aggregation"] == "dfs":
            print("Full join not available with DFS.")
            return None

    def fit(self, X=None, y=None):
        with self.timed("prepare"):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
            merged_train = X_train.clone().lazy()
        with self.timed("join_train"):
            merged_train = ju.execute_join_all_candidates(
                merged_train, self.candidate_joins, self.join_parameters["aggregation"]
            )
            merged_valid = X_valid.clone().lazy()
            merged_valid = ju.execute_join_all_candidates(
                merged_valid, self.candidate_joins, self.join_parameters["aggregation"]
            )
            # else:
            #     merged_train = X.clone().lazy()
            #     merged_train = ju.execute_join_all_candidates(
            #         merged_train, self.candidate_joins, self.join_parameters["aggregation"]
            #     )
        with self.timed("model_train"):
            self.build_model(merged_train)
            self.fit_model(
                merged_train, y_train, validation_set=(merged_valid, y_valid)
            )
        self.n_joined_columns = len(merged_train.columns)

    def predict(self, X: pd.DataFrame):
        with self.timed("join_predict"):
            merged_test = X.clone().lazy()
            merged_test = ju.execute_join_all_candidates(
                merged_test, self.candidate_joins, self.join_parameters["aggregation"]
            )

        with self.timed("model_predict"):
            pred = self.predict_model(merged_test)

        return pred

    def get_estimator_parameters(self):
        d = super().get_estimator_parameters()
        d.update(self.join_parameters)
        return d

    def get_additional_info(self):
        # TODO: add the list of all left_on, right_on
        return {
            "best_candidate_hash": "full_join",
            "left_on": "",
            "right_on": "",
            "n_joined_columns": self.n_joined_columns,
        }


class StepwiseGreedyJoin(BaseJoinWithCandidatesMethod):
    """StepwiseGreedyJoin produces a full join path employing a greedy strategy. It evaluates"""

    def __init__(
        self,
        scenario_logger: ScenarioLogger = None,
        candidate_joins: dict = None,
        target_column: str = None,
        model_parameters: dict = None,
        join_parameters: dict = None,
        budget_type: str = None,
        budget_amount: str = None,
        epsilon: float = None,
        ranking_metric: str = "containment",
        valid_size: float = 0.2,
        max_candidates: int = 50,
        task: str = "regression",
        use_rf: bool = False,
    ) -> None:
        super().__init__(
            scenario_logger,
            candidate_joins,
            target_column,
            model_parameters,
            join_parameters,
            task,
        )

        self.name = "stepwise_greedy_join"
        self.use_rf = use_rf
        if use_rf:
            self.name += "_rf"
        self.budget_type, self.budget_amount = self._check_budget(
            budget_type, budget_amount
        )
        self.ranking_metric = self._check_ranking_method(ranking_metric)
        self.valid_size = valid_size

        # Calling this "current_metric" to be more generic
        self.current_metric = -np.inf
        self.current_X_train = self.current_X_valid = None
        self.selected_candidates = {}
        self.valid_size = valid_size
        self.blacklist = deque([], maxlen=max_candidates)
        self.candidate_ranking = None
        self.max_candidates = max_candidates
        self.with_validation = True
        self.already_joined_tables = []
        # TODO: account for multiple candidate joins on the same table
        self.already_evaluated = {cjoin: 0 for cjoin in self.candidate_joins.keys()}
        self.base_epsilon = epsilon
        self.epsilon = epsilon

        self.inner_model = None

        self.wrap_up_joiner = None
        self.wrap_up_joiner_params = model_parameters

    def fit(self, X: pd.DataFrame, y):
        with self.timed("prepare"):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

            self.candidate_ranking = self._build_ranking(X)
            ########## BASE TABLE
            with self.timed("model_train"):
                self.build_model(X_train)

                # Fit the given model
                self.fit_model(X_train, y_train, validation_set=(X_valid, y_valid))
            with self.timed("model_predict"):
                y_pred = self.predict_model(X_valid)
            metric = self._evaluate_candidate(y_valid, y_pred)
        self.current_metric = metric

        self.current_X_train = X_train
        self.current_X_valid = X_valid

        for _ in tqdm(
            range(self.budget_amount),
            total=self.budget_amount,
            desc="StepwiseGreedyJoin - Iterating: ",
            leave=False,
            position=2,
        ):
            # NOTE: when no candidates are left, the timer segment started
            # here is abandoned (as in the original implementation), so
            # explicit timer calls are kept for this block.
            self._start_time("prepare")
            cjoin = self._get_candidate()
            if cjoin is None:
                # No more candidates:
                break
            cjoin_id = cjoin.candidate_id
            _, cnd_md, left_on, right_on = cjoin.get_join_information()
            cnd_table = pl.read_parquet(cnd_md["full_path"])
            cnd_hash = cnd_md["hash"]
            self._end_time("prepare")
            if cjoin_id not in self.already_joined_tables:
                with self.timed("join_train"):
                    temp_X_train, temp_X_valid = self._execute_joins(
                        self.current_X_train,
                        self.current_X_valid,
                        cnd_table,
                        left_on,
                        right_on,
                        suffix=cjoin.candidate_id[:10],
                    )

                # NOTE: the `model_train` region ends in different places
                # depending on the branch, so explicit timer calls are kept.
                self._start_time("model_train")
                self.build_model(temp_X_train)

                # Fit the given model
                if self.use_rf:
                    y_pred = self._prepare_inner_model(
                        merged_train, y_train, merged_valid, method
                    )
                else:
                    self.fit_model(
                        temp_X_train,
                        y_train,
                        validation_set=(temp_X_valid, y_valid),
                    )
                    self._end_time("model_train")
                    self._start_time("prepare")
                    y_pred = self.predict_model(temp_X_valid)

                metric = self._evaluate_candidate(y_valid, y_pred)
                self._end_time("prepare")
            else:
                with self.timed("prepare"):
                    metric = -np.inf
                    temp_X_train = temp_X_valid = None
            with self.timed("prepare"):
                self._update_ranking(cjoin, metric, temp_X_train, temp_X_valid)

        if len(self.selected_candidates) > 0:
            # Full join all the selected candidates
            with self.timed("prepare"):
                self.wrap_up_joiner = FullJoin(
                    scenario_logger=self.scenario_logger,
                    candidate_joins=self.selected_candidates,
                    target_column=self.target_column,
                    model_parameters=self.wrap_up_joiner_params,
                    join_parameters=self.join_parameters,
                    task=self.task,
                )
            self.wrap_up_joiner.fit(X=X, y=y)

        else:
            # No good candidates found, train on the base table
            with self.timed("prepare"):
                self.wrap_up_joiner = NoJoin(
                    scenario_logger=self.scenario_logger,
                    target_column=self.target_column,
                    model_parameters=self.wrap_up_joiner_params,
                    task=self.task,
                )
            self.wrap_up_joiner.fit(X=X, y=y)

        self.n_joined_columns = self.wrap_up_joiner.n_joined_columns
        self.update_durations(self.wrap_up_joiner.get_durations())

    def _build_ranking(self, X):
        if self.ranking_metric == "containment":
            _ranking = build_containment_ranking(X, self.candidate_joins)
            candidate_ranking = deque(
                _ranking["candidate"].to_list(), maxlen=self.max_candidates
            )

        return candidate_ranking

    def _get_candidate(self) -> CandidateJoin | None:
        if len(self.candidate_ranking) > 0:
            _cnd = self.candidate_ranking.popleft()
            return self.candidate_joins[_cnd]
        elif 0 < len(self.blacklist) < self.n_candidates:
            # If the length of the blacklist is == n_candidates, then all candidates have already been discarded, stopping
            # All candidates have been evaluated, selecting one of the discarded candidates
            random.shuffle(self.blacklist)
            _cnd = self.blacklist.popleft()
            return self.candidate_joins[_cnd]
        else:
            # No more candidates are left, stop iterations
            return None

    def _get_curr_eps(self, n):
        # TODO: expand as needed
        return 1 / (np.log(n) + 1) * self.base_epsilon

    def _update_ranking(self, cjoin, metric, temp_X_train, temp_X_valid):
        cjoin_hash = cjoin.candidate_id
        self.already_evaluated[cjoin_hash] += 1
        if metric > self.current_metric + self.epsilon:
            self.selected_candidates.update({cjoin_hash: cjoin})
            self.current_metric = metric
            self.current_X_train = temp_X_train
            self.current_X_valid = temp_X_valid
            n_cnd = len(self.selected_candidates)
            self.epsilon = self._get_curr_eps(n_cnd)
            self.already_joined_tables.append(cjoin_hash)
        else:
            # self.candidate_ranking.append(cnd_hash)
            if self.already_evaluated[cjoin_hash] < 2:
                self.blacklist.append(cjoin_hash)

    def predict(self, X):
        pred = self.wrap_up_joiner.predict(X)
        self.durations["time_join_predict"] = self.wrap_up_joiner.durations.get(
            "time_join_predict", 0
        )
        self.durations["time_model_predict"] = self.wrap_up_joiner.durations.get(
            "time_model_predict", 0
        )
        return pred

    def _check_ranking_method(self, ranking_method):
        if ranking_method not in SUPPORTED_RANKING_METHODS:
            raise ValueError(
                f"`ranking_method` {ranking_method} not in SUPPORTED_RANKING_METHODS."
            )
        return ranking_method

    def _check_budget(self, budget_type, budget_amount):
        if budget_type == "iterations":
            try:
                budget_amount = int(budget_amount)
            except:
                raise ValueError("A non-numeric number of iterations was provided.")
            if not isinstance(budget_amount, int):
                raise ValueError("`budget_amount` must be integer")
            if budget_amount <= 0:
                raise ValueError("The number of iterations must be strictly positive.")
            return budget_type, budget_amount

        raise ValueError(
            f"Provided budget_type {budget_type} not in SUPPORTED_BUDGET_TYPES."
        )

    def _execute_joins(
        self,
        X_train,
        X_valid=None,
        cnd_table=None,
        left_on=None,
        right_on=None,
        suffix="_right",
    ):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train
        merged_train = ju.execute_join_with_aggregation(
            X_train,
            cnd_table,
            left_on=left_on,
            right_on=right_on,
            aggregation=self.join_parameters["aggregation"],
            suffix=suffix,
        )

        if X_valid is not None:
            if isinstance(X_valid, pd.DataFrame):
                X_valid = X_valid
            merged_valid = ju.execute_join_with_aggregation(
                X_valid,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                aggregation=self.join_parameters["aggregation"],
                suffix=suffix,
            )
            return merged_train, merged_valid
        else:
            return merged_train

    def get_additional_info(self):
        # TODO: add the list of all left_on, right_on
        return {
            "n_joined_columns": self.n_joined_columns,
            "budget_type": self.budget_type,
            "budget_amount": self.budget_amount,
            "epsilon": self.base_epsilon,
            "chosen_candidates": [
                cjoin.candidate_id for cjoin in self.selected_candidates.values()
            ],
        }
