"""Train a LightGBM model for 30-day readmission.

Loads processed features and target from Phase 1 (`data/processed/features.csv` and
`data/processed/target.csv`), runs a randomized hyperparameter search with
stratified CV, evaluates on a holdout test set, and saves model + metrics.

Features:
- Auto-detects GPU and uses it if available
- Progress tracking with tqdm
- Saves visualizations (ROC curve, confusion matrix)
- Kaggle-optimized defaults (faster CV)

Usage (from project root):
    python phase-2-risk-modeling/lightgbm_model.py
    
Kaggle usage:
    !python phase-2-risk-modeling/lightgbm_model.py --skip-search --verbose

Requirements:
    pip install lightgbm scikit-learn pandas joblib matplotlib seaborn tqdm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import warnings

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Install with: pip install tqdm")

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


def detect_gpu():
    """Detect if GPU is available for LightGBM."""
    try:
        import lightgbm as lgb
        # Try to create a simple dataset and train with GPU
        import numpy as np
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 2, 10)
        lgb_train = lgb.Dataset(X_test, y_test)
        params = {'device': 'gpu', 'verbose': -1}
        lgb.train(params, lgb_train, num_boost_round=1, verbose_eval=False)
        return True
    except Exception:
        return False


def is_kaggle_environment():
    """Detect if running in Kaggle environment."""
    return os.path.exists('/kaggle/working')


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}\n")


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
            # prefer AUC for evaluation
            fit_args.setdefault("eval_metric", "auc")
            
            # Handle both old and new LightGBM API for early stopping
            try:
                # Try new API first (LightGBM >= 4.0)
                import lightgbm as lgb
                if hasattr(lgb, 'early_stopping'):
                    fit_args.setdefault("callbacks", [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)])
                else:
                    # Fall back to old API (LightGBM < 4.0)
                    fit_args.setdefault("early_stopping_rounds", early_stopping_rounds)
            except:
                # If all else fails, use old API
                fit_args.setdefault("early_stopping_rounds", early_stopping_rounds)

        # Some sklearn-style estimators accept verbose; allow user to pass via fit_kwargs
        self.model.fit(self.X_train, self.y_train, **fit_args)

    def evaluate(self, X, y, threshold: float = 0.5):
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y, proba)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
        }, proba, pred

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)



def load_data(data_dir: str = "data/processed"):
    """Load features and target data."""
    print("📂 Loading data...")
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

    print(f"✅ Loaded features: {X.shape}, target: {y.shape}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    return X, y


def build_default_param_dist():
    """Parameter distribution for RandomizedSearchCV."""
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


def save_visualizations(y_true, y_proba, y_pred, output_dir: Path):
    """Save ROC curve and confusion matrix visualizations."""
    print("📊 Generating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Hospital Readmission Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ ROC curve saved: {roc_path}")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Readmission', 'Readmission'],
                yticklabels=['No Readmission', 'Readmission'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Hospital Readmission Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Confusion matrix saved: {cm_path}")


def print_metrics_table(metrics: dict, title: str = "Model Performance Metrics"):
    """Print metrics in a formatted table."""
    print_section(title, "=")
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title():<20} {v:>10.4f}")
    print("-" * 32)


def main(args: argparse.Namespace):
    start_time = time.time()
    
    # Print configuration
    print_section("🚀 LightGBM Training - Hospital Readmission Risk", "=")
    print(f"⚙️  Configuration:")
    print(f"   - Data directory: {args.data_dir}")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - Test size: {args.test_size}")
    print(f"   - CV folds: {args.cv}")
    print(f"   - Random seed: {args.random_state}")
    print(f"   - Skip search: {args.skip_search}")
    print(f"   - Fast mode: {args.fast_mode}")
    
    # Detect environment
    on_kaggle = is_kaggle_environment()
    if on_kaggle:
        print(f"   - Environment: 🏆 Kaggle")
    else:
        print(f"   - Environment: 💻 Local")
    
    # Load data
    X, y = load_data(args.data_dir)

    # Split data
    print("\n🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, random_state=args.random_state, stratify=y_train
    )
    print(f"   Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Import LightGBM
    try:
        import lightgbm as lgb
        LGB_CLASS = lgb.LGBMClassifier
    except Exception:
        raise ImportError(
            "LightGBM is required for this training script. Please install it with `pip install lightgbm`."
        )
    
    # GPU Detection
    print("\n🖥️  Checking GPU availability...")
    gpu_available = detect_gpu()
    device_type = "gpu" if gpu_available and args.use_gpu else "cpu"
    if gpu_available and args.use_gpu:
        print("   ✅ GPU detected and will be used for training!")
    elif gpu_available and not args.use_gpu:
        print("   ⚠️  GPU available but --use-gpu not set. Using CPU.")
    else:
        print("   ℹ️  No GPU detected. Using CPU for training.")

    # Parse user parameters
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

    # Decide training mode
    if user_params is not None or args.skip_search or args.fast_mode:
        # Train with fixed params (no hyperparameter search)
        print_section("⚡ Fast Training Mode (No Hyperparameter Search)", "-")
        
        model_kwargs = user_params or DEFAULT_PARAMS.copy()
        model_kwargs.setdefault("random_state", args.random_state)
        model_kwargs.setdefault("n_jobs", args.n_jobs)
        model_kwargs.setdefault("device_type", device_type)

        print(f"📋 Model parameters:")
        for k, v in model_kwargs.items():
            print(f"   {k}: {v}")

        model = LGB_CLASS(**model_kwargs)

        print("\n🏋️  Training model with early stopping...")
        trainer = Trainer(model, X_tr, y_tr, X_val, y_val, output_dir=args.output_dir)
        es_rounds = args.early_stopping_rounds if args.early_stopping_rounds > 0 else None
        fit_kwargs = {}
        if args.verbose:
            fit_kwargs["verbose"] = 10
        
        train_start = time.time()
        trainer.fit(early_stopping_rounds=es_rounds, **fit_kwargs)
        train_time = time.time() - train_start
        print(f"✅ Training completed in {train_time:.2f} seconds")

        best_model = trainer.model
        search = None
        search_summary = {
            "best_params": best_model.get_params(),
            "best_cv_score": None,
            "training_time_seconds": train_time
        }
    else:
        # Run hyperparameter search
        print_section("🔍 Hyperparameter Search Mode", "-")
        
        model = LGB_CLASS(
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            device_type=device_type
        )

        param_dist = build_default_param_dist()

        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)

        print(f"⚙️  Search configuration:")
        print(f"   - Iterations: {args.n_iter}")
        print(f"   - CV folds: {args.cv}")
        print(f"   - Scoring: roc_auc")
        print(f"   - Parallel jobs: {args.n_jobs}")

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

        print("\n🔎 Starting hyperparameter search (this may take a while)...")
        search_start = time.time()
        search.fit(X_tr, y_tr)
        search_time = time.time() - search_start

        print(f"\n✅ Search completed in {search_time:.2f} seconds")
        print(f"🏆 Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"📋 Best parameters:")
        for k, v in search.best_params_.items():
            print(f"   {k}: {v}")

        best_model = search.best_estimator_
        search_summary = {
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "search_time_seconds": search_time
        }

    # Final evaluation on holdout test set
    print_section("📊 Final Evaluation on Test Set", "-")
    print("🧪 Evaluating model performance...")
    
    if "trainer" in locals():
        metrics, y_proba, y_pred = trainer.evaluate(X_test, y_test)
    else:
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
        }

    print_metrics_table(metrics, "🎯 FINAL TEST SET RESULTS")

    # Create output directory
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save visualizations
    save_visualizations(y_test, y_proba, y_pred, out_dir)

    # Save model and artifacts
    print_section("💾 Saving Results", "-")
    
    model_path = out_dir / "gradient_boosting_model.joblib"
    if "trainer" in locals():
        trainer.save(model_path)
    else:
        joblib.dump(best_model, model_path)
    print(f"✅ Model saved: {model_path}")

    metrics_path = out_dir / "gradient_boosting_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")

    # Create comprehensive summary
    total_time = time.time() - start_time
    summary = {
        "model": "LightGBM Classifier",
        "task": "Hospital 30-Day Readmission Risk Prediction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": "kaggle" if on_kaggle else "local",
        "device": device_type,
        "data": {
            "train_size": len(X_tr),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_features": X.shape[1]
        },
        "best_params": search_summary["best_params"],
        "best_cv_score": search_summary.get("best_cv_score"),
        "test_metrics": metrics,
        "training_time_seconds": search_summary.get("training_time_seconds") or search_summary.get("search_time_seconds"),
        "total_time_seconds": total_time
    }
    
    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {summary_path}")

    # Final summary
    print_section("✨ Training Complete!", "=")
    print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📁 All outputs saved to: {out_dir}")
    print(f"🎯 Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\n🎉 Ready for deployment!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LightGBM model for readmission risk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast training with default params (good for quick testing)
  python lightgbm_model.py --fast-mode --verbose
  
  # Full hyperparameter search
  python lightgbm_model.py --n-iter 50 --cv 5
  
  # Kaggle optimized (fast CV, use all cores)
  python lightgbm_model.py --skip-search --n-jobs -1 --use-gpu
  
  # Custom params from file
  python lightgbm_model.py --params-file params.json --skip-search
        """
    )
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with features.csv and target.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save model and metrics (auto-detects Kaggle)")
    
    # Training arguments
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Holdout test size fraction (default: 0.2)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation size for early stopping (default: 0.1)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Hyperparameter search arguments
    parser.add_argument("--cv", type=int, default=None,
                        help="Number of CV folds (default: 3 for Kaggle, 5 for local)")
    parser.add_argument("--n-iter", type=int, default=None,
                        help="Number of parameter settings sampled (default: 5 for Kaggle, 20 for local)")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip randomized search and train with DEFAULT_PARAMS or --params")
    
    # Performance arguments
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of parallel jobs (default: -1 for Kaggle, 1 for local)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU if available (auto-detected)")
    parser.add_argument("--early-stopping-rounds", type=int, default=50,
                        help="Early stopping rounds (0 to disable, default: 50)")
    
    # Model configuration arguments
    parser.add_argument("--params", type=str, default=None,
                        help="JSON string of model params to use (overrides DEFAULT_PARAMS)")
    parser.add_argument("--params-file", type=str, default=None,
                        help="Path to JSON file with model params")
    
    # Mode arguments
    parser.add_argument("--fast-mode", action="store_true",
                        help="Quick training with default params (same as --skip-search)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output from training/search")

    args = parser.parse_args()
    
    # Auto-detect Kaggle and set sensible defaults
    on_kaggle = is_kaggle_environment()
    
    if args.output_dir is None:
        args.output_dir = "/kaggle/working/models" if on_kaggle else "models"
    
    if args.cv is None:
        args.cv = 3 if on_kaggle else 5
    
    if args.n_iter is None:
        args.n_iter = 5 if on_kaggle else 20
    
    if args.n_jobs is None:
        args.n_jobs = -1 if on_kaggle else 1
    
    # Auto-enable GPU on Kaggle if available
    if on_kaggle and not args.use_gpu:
        args.use_gpu = detect_gpu()
    
    main(args)
