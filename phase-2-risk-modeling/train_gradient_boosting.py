"""Train a gradient boosting model (LightGBM) for 30-day readmission.

Loads processed features and target from Phase 1 (`data/processed/features.csv` and
`data/processed/target.csv`), runs a randomized hyperparameter search with
stratified CV, evaluates on a holdout test set, and saves model + metrics.

Usage (from project root):
    python phase-2-risk-modeling/train_gradient_boosting.py

The script accepts a few CLI args for data and output directories.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")


# Default LightGBM parameters you can edit directly in this file.
# Edit these values to control training when running with --skip-search or without --params.
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
}


class Trainer:
    """Simple in-file Trainer abstraction for sklearn/LightGBM estimators.

    Responsibilities:
    - hold model and train/val data
    - fit with optional early stopping
    - evaluate and return metrics
    - save model artifact
    """

    def __init__(self, model, X_train, y_train, X_val=None, y_val=None, output_dir="models"):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.output_dir = Path(output_dir)

    def fit(self, early_stopping_rounds: int | None = None, **fit_kwargs):
        """Fit the underlying model. For LightGBM sklearn API we pass eval_set/early_stopping_rounds when val provided."""
        fit_args = fit_kwargs.copy()
        if self.X_val is not None and early_stopping_rounds:
            fit_args.setdefault("eval_set", [(self.X_val, self.y_val)])
            fit_args.setdefault("early_stopping_rounds", early_stopping_rounds)
            # prefer AUC for evaluation
            fit_args.setdefault("eval_metric", "auc")

        # Some sklearn-style estimators accept verbose; allow user to pass via fit_kwargs
        self.model.fit(self.X_train, self.y_train, **fit_args)

    def evaluate(self, X, y, threshold: float = 0.5):
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y, proba)),
            "precision": float(precision_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)



def load_data(data_dir: str = "data/processed"):
    data_dir = Path(data_dir)
    X_path = data_dir / "features.csv"
    y_path = data_dir / "target.csv"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Processed data not found in {data_dir}. Run phase-1 preprocessing first."
        )

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    # support both columnar and single-column target files
    if "target" in y.columns:
        y = y["target"]
    else:
        y = y.iloc[:, 0]

    return X, y


def build_default_param_dist(random_state: int = 42):
    # Parameter distribution for RandomizedSearchCV
    return {
        "n_estimators": [100, 200, 400, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 3, 5, 8, 12],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
    }


def main(args: argparse.Namespace):
    X, y = load_data(args.data_dir)

    # small sanity check
    print(f"Loaded features: {X.shape}, target: {y.shape}")

    # simple stratified holdout for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # also create a small validation split from the training set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, random_state=args.random_state, stratify=y_train
    )

    # prefer LightGBM if available
    try:
        import lightgbm as lgb

        LGB_CLASS = lgb.LGBMClassifier
    except Exception:  # pragma: no cover - instructive fallback
        raise ImportError(
            "LightGBM is required for this training script. Please install it with `pip install lightgbm`."
        )

    # If user provided explicit params (JSON string or file) or asked to skip search,
    # use those params to build the model and fit directly. Otherwise run RandomizedSearchCV.
    user_params = None
    if args.params:
        try:
            user_params = json.loads(args.params)
        except Exception as exc:
            raise ValueError(f"Could not parse JSON params string: {exc}")
    if args.params_file:
        params_path = Path(args.params_file)
        if not params_path.exists():
            raise FileNotFoundError(f"Params file not found: {params_path}")
        with open(params_path, "r") as f:
            user_params = json.load(f)

    if user_params is not None or args.skip_search:
        # Train with user-specified params (or DEFAULT_PARAMS from this file) without hyperparameter search
        model_kwargs = user_params or DEFAULT_PARAMS.copy()
        # ensure random_state and n_jobs are set/overridden
        model_kwargs.setdefault("random_state", args.random_state)
        model_kwargs.setdefault("n_jobs", args.n_jobs)

        print("Training LightGBM with user-specified parameters (no hyperparameter search)")
        print(f"Model kwargs: {model_kwargs}")

        model = LGB_CLASS(**model_kwargs)

        # use Trainer to fit with optional early stopping
        trainer = Trainer(model, X_tr, y_tr, X_val, y_val, output_dir=args.output_dir)
        es_rounds = args.early_stopping_rounds if args.early_stopping_rounds > 0 else None
        fit_kwargs = {}
        if args.verbose:
            fit_kwargs["verbose"] = 10
        trainer.fit(early_stopping_rounds=es_rounds, **fit_kwargs)

        best_model = trainer.model
        search = None
        search_summary = {"best_params": best_model.get_params(), "best_cv_score": None}
    else:
        model = LGB_CLASS(random_state=args.random_state, n_jobs=args.n_jobs)

        param_dist = build_default_param_dist(args.random_state)

        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="roc_auc",
            cv=cv,
            verbose=2 if args.verbose else 1,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

    print("Starting hyperparameter search (this may take a while)...")
    search.fit(X_tr, y_tr)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    best_model = search.best_estimator_
    search_summary = {"best_params": search.best_params_, "best_cv_score": float(search.best_score_)}

    # Final evaluation on holdout test set
    # Use Trainer.evaluate if we have a trainer, otherwise evaluate directly
    if "trainer" in locals():
        metrics = trainer.evaluate(X_test, y_test)
    else:
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
        }

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model and artifacts
    model_path = out_dir / "gradient_boosting_model.joblib"
    # If trainer exists, use its save helper to ensure dirs exist
    if "trainer" in locals():
        trainer.save(model_path)
    else:
        joblib.dump(best_model, model_path)

    metrics_path = out_dir / "gradient_boosting_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    search_summary_path = out_dir / "gradient_boosting_search_summary.json"
    # Save a compact search summary (best params + cv score) when available
    with open(search_summary_path, "w") as f:
        json.dump(search_summary, f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gradient boosting model for readmission risk.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with features.csv and target.csv")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model and metrics")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size fraction")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds for hyperparameter search")
    parser.add_argument("--n-iter", type=int, default=20, help="Number of parameter settings sampled")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output from search")
    parser.add_argument("--params", type=str, default=None, help="JSON string of model params to use (overrides DEFAULT_PARAMS)")
    parser.add_argument("--params-file", type=str, default=None, help="Path to JSON file with model params")
    parser.add_argument("--skip-search", action="store_true", help="Skip randomized search and train with DEFAULT_PARAMS or --params")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction of train set to use as validation for early stopping")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Early stopping rounds (0 to disable)")

    args = parser.parse_args()
    main(args)
