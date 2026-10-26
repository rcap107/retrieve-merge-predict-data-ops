"""Model-fitting core shared by all join selectors.

This module contains `EvalModel`, the class responsible for building,
fitting and predicting with the ML models used by the evaluation pipeline
(CatBoost, RidgeCV/LogisticRegression, RealMLP and ResNet). It also handles
the inner train/validation split used to provide a validation set to the
models that support one, and the phase timers (`prepare`, `model_train`,
`model_predict`) recorded in the run logs.

The join selection strategies (`src.methods.join_selectors`) extend this
class and only add the candidate selection and join orchestration logic.
"""

import datetime as dt
from contextlib import contextmanager

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from catboost import CatBoostClassifier, CatBoostRegressor
from pytabkit.models.sklearn.sklearn_interfaces import (
    RealMLP_TD_Classifier,
    RealMLP_TD_Regressor,
    Resnet_RTDL_D_Classifier,
    Resnet_RTDL_D_Regressor,
)
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from skrub import MinHashEncoder, TableVectorizer, tabular_pipeline

from src.utils.constants import SUPPORTED_MODELS

CACHE_PATH = "results/cache"

# For reproducibility
CATBOOST_RANDOM_SEED = 42


def _build_preprocessor():
    num_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    # Transformer for high cardinality categorical features
    high_card_transformer = Pipeline(
        steps=[
            (
                "onehot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
            ("pca", PCA(n_components=30)),
        ],
    )

    low_card_transformer = Pipeline(
        steps=[
            (
                "onehot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ],
    )

    tv = TableVectorizer(
        low_cardinality=low_card_transformer,
        high_cardinality=high_card_transformer,
        numeric=num_transformer,
        n_jobs=1,
    )

    ct = make_column_transformer(
        (
            SimpleImputer(strategy="mean", keep_empty_features=True),
            make_column_selector(dtype_include=np.number),
        ),
        (
            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            make_column_selector(dtype_include=object),
        ),
    )

    preprocessor = Pipeline(
        steps=[
            ("table_vectorizer", tv),
            ("column_transformer", ct),
        ]
    )

    return preprocessor


def _reorder_indices_for_validation(X_train, X_valid, y_train, y_valid):
    # Concatenate the two splits for fitting
    X_concat = np.concatenate([X_train, X_valid])
    y_concat = np.concatenate([y_train.to_numpy(), y_valid.to_numpy()])

    # Prepare the indices for fitting
    n_train = len(X_train)
    n_val = len(X_valid)

    train_idx = np.arange(0, n_train)
    valid_idx = np.arange(n_train, n_train + n_val)
    return (X_concat, y_concat, train_idx, valid_idx)


class EvalModel(BaseEstimator):
    """Model-fitting core of the evaluation pipeline.

    This class takes a prepared table `X` and a target `y`, splits an inner
    validation set, builds the model requested in `model_parameters` and fits
    it (with an evaluation set when the model supports one). It records the
    phase durations (`prepare`, `model_train`, `model_predict`) in the same
    format as the rest of the pipeline.

    It is used directly as the model step of the DataOps plans, and as the
    base class of the join selectors.
    """

    def __init__(
        self,
        scenario_logger=None,
        target_column: str | None = None,
        model_parameters: dict | None = None,
        task: str = "regression",
    ) -> None:
        """Base Selector class. This class is supposed to be extended by the
        other selectors. It uses the `fit`/`predict` paradigm to implement the
        evaluation of a join suggestion method.

        Args:
            scenario_logger (ScenarioLogger, optional): ScenarioLogger object to be used for tracking
            the parameters and the current experiment.
            target_column (str, optional): Column that contains the target to be used for
            supervised learning. Defaults to None.
            model_parameters (dict, optional): Parameters to be passed to the model. Defaults to None.
            task (str, optional): Task to be executed, either `regression` or `classification`. Defaults to "regression".

        Raises:
            ValueError: Raise ValueError if the value of `task` is not `regression` or `classification`.
            ValueError: Raise ValueError if the value of `chosen_model` is not `catboost` or `linear`.
        """
        super().__init__()

        if task == "regression":
            self.target_type = "continuous"
        elif task == "classification":
            self.target_type = "binary"
        else:
            raise ValueError(f"Task {task} not supported.")

        self.chosen_model = model_parameters["chosen_model"]

        if self.chosen_model not in SUPPORTED_MODELS:
            raise ValueError(f"Model {self.chosen_model} not supported.")

        self.name = "base_estimator"
        self.scenario_logger = scenario_logger
        self.model_parameters = model_parameters.get(
            model_parameters["chosen_model"], {}
        )

        self.target_column = target_column
        self.task = task
        self.n_joined_columns = None
        self.with_validation = True

        self.timestamps = {}
        self.durations = {}

        self.model = None
        self.cat_features = None

        self.preprocessor = _build_preprocessor()

        if self.chosen_model == "linear":
            self.with_validation = False

    def build_model(self, X: pd.DataFrame | None = None):
        """Build the model using the parameters supplied during __init__.
        Note that multiple models may be created by a JoinEstimator during training.

        Args:
            X (pd.DataFrame, optional): Input dataframe used to build the linear model. Defaults to None.
            cat_features (list, optional): List of features to be considered categorical by CatBoost. Defaults to None.

        Raises:
            ValueError: Raise ValueError if the model name provided is not `linear` or `catboost`.

        Returns:
            The required model.
        """
        if self.chosen_model == "catboost":
            cat_features = self.get_cat_features(X)
            self.build_catboost(cat_features)
        elif self.chosen_model == "ridgecv":
            self.build_linear()
        elif self.chosen_model == "resnet":
            self.build_resnet()
        elif self.chosen_model == "realmlp":
            self.build_realmlp()
        else:
            raise ValueError(f"Chosen model {self.model} is not recognized.")

    def build_catboost(self, cat_features):
        """Build a catboost model with the model parameters defined in the __init__
        and with the categorical features provided in `cat_features`.

        Args:
            cat_features (list): List of features to be considered categorical.

        Returns:
            CatBoostRegressor or CatBoostClassifier: The initialized model.
        """
        defaults = {
            "l2_leaf_reg": 0.01,
            "od_type": "Iter",
            "od_wait": 10,
            "iterations": 300,
            "verbose": 0,
        }

        parameters = dict(defaults)
        parameters.update(self.model_parameters)
        parameters["random_seed"] = CATBOOST_RANDOM_SEED
        if self.task == "regression":
            self.model = CatBoostRegressor(cat_features=cat_features, **parameters)
        elif self.task == "classification":
            self.model = CatBoostClassifier(cat_features=cat_features, **parameters)
        else:
            raise ValueError

    def build_linear(self):
        """Build either a Ridge regressor, or a LogisticRegression classifier
        with default parameters.

        Raises:
            ValueError: Raise ValueError if the value in `self.task` is not a
            valid task (`regression` or `classification`).
        """
        if self.task == "regression":
            self.model = RidgeCV()
        elif self.task == "classification":
            self.model = LogisticRegression()
        else:
            raise ValueError

    def build_realmlp(self):
        """Build an estimator based on RealMLP."""
        defaults = {
            "n_threads": 32,
        }

        parameters = dict(defaults)
        parameters.update(self.model_parameters)

        if self.task == "regression":
            self.model = RealMLP_TD_Regressor(**parameters)
        elif self.task == "classification":
            self.model = RealMLP_TD_Classifier(**parameters)
        else:
            raise ValueError

    def build_resnet(self):
        """Build an estimator based on ResNet."""
        if self.task == "regression":
            self.model = Resnet_RTDL_D_Regressor(n_threads=32)
        elif self.task == "classification":
            self.model = Resnet_RTDL_D_Classifier(n_threads=32)
        else:
            raise ValueError

    def fit(self, X, y):
        """Split an inner validation set, then build and fit the model.

        Args:
            X (pd.DataFrame): Input dataframe.
            y (pd.Series): Target column.
        """
        self._start_time("prepare")
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        self.n_joined_columns = len(X_train.columns)
        self._end_time("prepare")

        self._start_time("model_train")
        self.build_model(X_train)
        self.fit_model(X_train, y_train, validation_set=(X_valid, y_valid))
        self._end_time("model_train")

    def predict(self, X):
        """Predict with the fitted model.

        Args:
            X (pd.DataFrame): Input dataframe.
        """
        self._start_time("model_predict")
        pred = self.predict_model(X)
        self._end_time("model_predict")
        return pred

    def get_cat_features(self, X: pl.DataFrame | pd.DataFrame) -> list:
        """Given an input table X, return the list of features that are considered
        to be categorical by the appropriate type selection function.

        Args:
            X (pl.DataFrame | pd.DataFrame): The input table to be evaluated.

        Raises:
            TypeError: Raise TypeError if X is not pl.DataFrame or pd.DataFrame.

        Returns:
            list: The list of categorical features.
        """
        if type(X) == pl.DataFrame:
            cat_features = X.select(cs.string()).columns
        elif type(X) == pd.DataFrame:
            cat_features = X.select_dtypes("object").columns.to_list()
        else:
            raise TypeError(f"Type {type(X)} unsupported.")

        return cat_features

    def fit_model(
        self,
        X,
        y,
        validation_set: tuple | None = None,
    ):
        X_train, y_train = X, y
        X_valid, y_valid = validation_set
        if self.chosen_model == "catboost":
            X_train = self.prepare_table(X_train)
            X_valid = self.prepare_table(X_valid)
            y_train, y_valid = y_train.to_pandas(), y_valid.to_pandas()
            self.model.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))
        elif self.chosen_model in ["resnet", "realmlp"]:
            X_train, _ = self.prepare_table_preprocessing(X_train)
            X_train_enc = self.preprocessor.fit_transform(X_train, y_train)
            X_valid, _ = self.prepare_table_preprocessing(X_valid)
            X_valid_enc = self.preprocessor.transform(X_valid)

            (
                X_concat,
                y_concat,
                _train_idx,
                _valid_idx,
            ) = _reorder_indices_for_validation(
                X_train_enc, X_valid_enc, y_train, y_valid
            )

            self.model.fit(
                X=X_concat,
                y=y_concat,
                val_idxs=_valid_idx,
            )
        else:
            # No validation set, keep all passed samples as train
            X_train, y_train = X, y
            # Linear
            X_train, _ = self.prepare_table_preprocessing(X_train)
            X_train_enc = self.preprocessor.fit_transform(X_train, y_train)
            self.model.fit(X=X_train_enc, y=y_train)

    def predict_model(self, X):
        if self.chosen_model == "catboost":
            X = self.prepare_table(X)
            y_pred = self.model.predict(X)
        else:
            X, _ = self.prepare_table_preprocessing(X)
            X_enc = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_enc)
        return y_pred

    def get_estimator_parameters(self):
        return {"estimator": self.name, "chosen_model": self.chosen_model}

    def get_additional_info(self):
        return {
            "best_candidate_hash": self.name,
            "left_on": "",
            "right_on": "",
            "n_joined_columns": self.n_joined_columns,
        }

    def _prepare_inner_model(self):
        if self.task == "regression":
            inner_model = make_pipeline(
                TableVectorizer(
                    high_cardinality=MinHashEncoder(),
                    low_cardinality=OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    ),
                    n_jobs=1,
                ),
                RandomForestRegressor(n_jobs=1),
                memory=CACHE_PATH,
            )
        else:
            inner_model = make_pipeline(
                TableVectorizer(
                    high_cardinality=MinHashEncoder(),
                    low_cardinality=OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    ),
                    n_jobs=32,
                ),
                RandomForestClassifier(n_jobs=32),
                memory=CACHE_PATH,
            )
        return inner_model

    def _fit_predict_inner_model(self, X_train, X_valid, y, inner_model):
        # if isinstance(X_train, pl.DataFrame):
        #     X_train = X_train.to_pandas()

        # if isinstance(X_valid, pl.DataFrame):
        #     X_valid = X_valid.to_pandas()

        inner_model.fit(X_train, y)
        self._end_time("model_train")
        self._start_time("prepare")

        return inner_model.predict(X_valid)

    def _fit_inner_model(self, X, y, method="rf"):
        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()
        # prep = _build_preprocessor()
        # prep.fit(X, y)
        # X_enc = prep.transform(X)
        if self.task == "regression":
            if method == "rf":
                self.inner_model = tabular_pipeline(RandomForestRegressor())
        elif self.task == "classification":
            self.inner_model = RandomForestClassifier()
        else:
            raise ValueError
        self.inner_model.fit(X, y)
        # self.inner_model.fit(X_enc, y)
        return None
        # return prep

    def _predict_inner_model(self, X, preprocessor):
        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()
        # X_enc = preprocessor.transform(X)
        X_enc = X
        return self.inner_model.predict(X_enc)

    def _start_time(self, label: str):
        self.timestamps[label] = [dt.datetime.now(), None]
        if "time_" + label not in self.durations:
            self.durations["time_" + label] = 0

    def _end_time(self, label: str):
        self.timestamps[label][1] = dt.datetime.now()
        this_segment = self.timestamps[label]
        self.durations["time_" + label] += (
            this_segment[1] - this_segment[0]
        ).total_seconds()

    @contextmanager
    def timed(self, label: str):
        """Context manager that times the execution of its body.

        Equivalent to calling `_start_time` before and `_end_time` after the
        wrapped block. Durations for the same label are cumulative.

        Example:
            >>> with self.timed("join_train"):
            ...     merged = self._execute_joins(...)
        """
        self._start_time(label)
        try:
            yield
        finally:
            self._end_time(label)

    def clean_timers(self):
        self.timestamps = {}
        self.durations = {}

    def get_durations(self):
        return self.durations

    def update_durations(self, new_durations):
        for k, v in new_durations.items():
            self.durations[k] += v

    def prepare_table(self, table):
        if isinstance(table, pd.DataFrame):
            table = pl.from_pandas(table)

        table = table.fill_null(value="null").fill_nan(value=np.nan)
        return table.to_pandas()

    def prepare_table_preprocessing(self, table):
        """Prepare the given table applying a pre-processing step that fills
        missing values and encodes categorical variables.

        Args:
            table (pl.DataFrame, pd.DataFrame): Table to prepare

        Returns:
            pd.DataFrame: Prepared dataframe
            list: List of categorical columns
        """
        cat_features = table.select(cs.string()).columns
        # Storing the original column order
        _columns = table.columns

        # We should try new and more clever imputation methods
        table = table.select(
            cs.string().fill_null("null"),
            cs.boolean().fill_null("null"),
            cs.numeric().fill_null("mean"),
            cs.datetime().dt.to_string("%Y-%m-%d").fill_null("null"),
        ).select(_columns)
        return table.to_pandas(), cat_features
