"""Hyperparameter search over retrieval methods, estimators and predictors.

Builds a single skrub DataOps plan that spans the retrieval method (as a
choice over the offline query results), the join selection strategy and the
predictor, then runs a randomized search (Optuna backend) over it.

The retrieval/indexing steps are offline (see `query_indices.py`): the query
results of every retrieval method must exist before running the search. This
entrypoint does not produce the run logs of `main.py`; its artifacts are the
search results table and the best fitted learner.

Run the script as follows:
```
python search.py path/to/search_config.toml
```

See `config/evaluation/general/search-binary.toml` for the configuration
format.
"""

import argparse
import os
import pickle
from pathlib import Path

import toml

# Fixing the number of polars threads, as in main.py.
os.environ["POLARS_MAX_THREADS"] = "32"

from src.utils.indexing import load_query_result  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", action="store", type=argparse.FileType("r"))
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Load only a small sample of the base table for debugging.",
    )
    return parser.parse_args()


def load_base_table(query_info: dict, debug: bool):
    import polars as pl
    import polars.selectors as cs

    path = Path(query_info["table_path"])
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix == ".csv":
        df = pl.read_csv(path)
    else:
        raise ValueError(f"Unsupported base table format {path.suffix}.")
    df = df.select(~cs.by_dtype(pl.Null)).unique()
    if debug:
        df = df.head(200)
    return df


def load_query_results(config: dict, methods: list) -> dict:
    """Load the candidate joins of every retrieval method (not clamped by
    top_k, so that the search can tune it)."""
    query_info = config["query_cases"]
    query_results = {}
    for method in methods:
        query_result = load_query_result(
            query_info["data_lake"],
            method,
            Path(query_info["table_path"]).stem,
            query_info["query_column"],
            top_k=0,
        )
        query_results[method] = query_result.candidates
    return query_results


def save_search_artifacts(search, config: dict, destination: Path):
    import joblib

    destination.mkdir(parents=True, exist_ok=True)
    search.results_.to_csv(destination / "search_results.csv", index=False)
    joblib.dump(search.best_learner_, destination / "best_learner.joblib")
    with open(destination / "best_named_params.pickle", "wb") as fp:
        pickle.dump(search.best_learner_.get_named_params(), fp)
    print(f"Search artifacts saved under {destination}")


if __name__ == "__main__":
    args = parse_args()
    config = toml.load(args.config_file)
    search_config = config.get("search", {})

    methods = search_config.get("join_discovery_methods") or list(
        config.get("query_cases", {}).get("join_discovery_method", [])
    )
    if isinstance(methods, str):
        methods = [methods]

    query_results = load_query_results(config, methods)
    base_table = load_base_table(config["query_cases"], args.debug)

    from src.methods.data_ops_search import run_search

    search = run_search(config, base_table, query_results)

    print("\n=== best configuration ===")
    print("named params:", search.best_learner_.get_named_params())
    print(f"best score: {search.best_score_}")
    print(search.results_.to_string())

    experiment_name = search_config.get(
        "experiment_name", f"search_{Path(args.config_file.name).stem}"
    )
    save_search_artifacts(search, config, Path("results/logs", experiment_name))
