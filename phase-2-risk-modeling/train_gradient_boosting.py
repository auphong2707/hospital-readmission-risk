"""Complete pipeline to preprocess data and train LightGBM model for 30-day readmission.

This script combines preprocessing orchestration and model training with a robust
evaluation pipeline:

Evaluation Pipeline:
1. Final Holdout Split: Split entire dataset into development_set and final_test_set
   - final_test_set remains untouched until final evaluation
2. Hyperparameter Search: GridSearchCV on development_set to find best parameters
3. K-Fold Cross-Validation with Nested Early Stopping:
   - Perform K-fold CV only on development_set
   - Inside each fold: split training data into inner_train and inner_val
   - Train on inner_train with early stopping monitored on inner_val
   - Evaluate on fold's holdout set
4. Calculate CV statistics (mean, std, confidence intervals)
5. Train final model on full development_set
6. Final evaluation on untouched final_test_set

Features:
- Robust nested cross-validation with proper data separation
- Early stopping within each fold to prevent overfitting
- Auto-detects and uses optimal performance settings (GPU, CPU cores)
- Progress tracking with tqdm
- Saves visualizations (ROC curve, confusion matrix)
- Exhaustive grid search with K-fold cross-validation
- Comprehensive metrics and fold statistics

Usage (from project root):
    python phase-2-risk-modeling/train_gradient_boosting.py
    
Kaggle usage:
    !python phase-2-risk-modeling/train_gradient_boosting.py --verbose

Examples:
    # Full hyperparameter search with 5-fold CV and early stopping
    # Performance (GPU, CPU cores) auto-detected
    python train_gradient_boosting.py
    
    # Custom K-fold splits and validation size
    python train_gradient_boosting.py --n-splits 10 --val-size 0.15
    
    # With verbose output
    python train_gradient_boosting.py --verbose

Requirements:
    pip install lightgbm scikit-learn pandas joblib matplotlib seaborn tqdm
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Install with: pip install tqdm")

warnings.filterwarnings("ignore")


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


def run_preprocessing(preprocess_script: Path) -> None:
    """Run preprocessing script to generate features and target files."""
    print_section("🔄 Running Preprocessing", "-")
    print(f"📂 Running: {preprocess_script}")
    subprocess.run([sys.executable, str(preprocess_script)], check=True)
    print("✅ Preprocessing completed")


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
    """Parameter grid for GridSearchCV.
    
    Balanced grid for thorough but practical search:
    - 4 × 4 × 3 × 3 × 3 × 3 × 2 × 2 = 2,592 combinations
    - With 5-fold CV = 12,960 model fits
    - Estimated time: 2-4 hours on CPU (depends on data size)
    """
    return {
        "n_estimators": [100, 200, 300, 400],      # Added 300
        "learning_rate": [0.03, 0.05, 0.08, 0.1],  # Added 0.08
        "num_leaves": [31, 63, 127],               # Keep 3 values (good coverage)
        "max_depth": [-1, 6, 10],                  # Added 6 and 10 (more tree depth options)
        "subsample": [0.7, 0.8, 0.9],              # Added 0.7 (more data sampling options)
        "colsample_bytree": [0.7, 0.8, 1.0],       # Added 0.7 (more feature sampling)
        # Regularization: Keep simple
        "reg_alpha": [0.0, 0.1],                   # L1 regularization
        "reg_lambda": [0.0, 0.1],                  # L2 regularization
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


def train_model(args: argparse.Namespace):
    """Main training function with robust evaluation pipeline.
    
    Pipeline structure:
    1. Final holdout split: entire dataset -> development_set + final_test_set
    2. K-fold CV on development_set only
    3. Inside each fold: training folds -> inner_train + inner_val (for early stopping)
    4. Evaluate on fold's holdout
    5. Final evaluation on untouched final_test_set
    """
    start_time = time.time()
    
    # Print configuration
    print_section("🚀 LightGBM Training - Hospital Readmission Risk", "=")
    print(f"⚙️  Configuration:")
    print(f"   - Data directory: {args.data_dir}")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - Test size: {args.test_size}")
    print(f"   - Validation size (for early stopping): {args.val_size}")
    print(f"   - K-fold splits: {args.n_splits}")
    print(f"   - Early stopping rounds: {args.early_stopping_rounds}")
    print(f"   - Random seed: {args.random_state}")
    print(f"   - Parallel jobs: {args.n_jobs}")
    print(f"   - Device: {'GPU' if args.use_gpu else 'CPU'}")
    print(f"   - Environment: {'🏆 Kaggle' if args.environment == 'kaggle' else '💻 Local'}")
    
    # Load data
    X, y = load_data(args.data_dir)

    # STEP 1: Final holdout split - create untouched test set
    print_section("🔀 Step 1: Final Holdout Split", "-")
    print(f"Splitting entire dataset into development_set ({1-args.test_size:.0%}) and final_test_set ({args.test_size:.0%})...")
    X_development, X_final_test, y_development, y_final_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"   ✅ Development set: {X_development.shape}")
    print(f"   ✅ Final test set (untouched until end): {X_final_test.shape}")
    print(f"   📊 Development class distribution: {y_development.value_counts().to_dict()}")
    print(f"   📊 Final test class distribution: {y_final_test.value_counts().to_dict()}")

    # Import LightGBM
    try:
        import lightgbm as lgb
        LGB_CLASS = lgb.LGBMClassifier
    except Exception:
        raise ImportError(
            "LightGBM is required for this training script. Please install it with `pip install lightgbm`."
        )
    
    # Set device type based on auto-detected GPU availability
    device_type = "gpu" if args.use_gpu else "cpu"

    # STEP 2: K-Fold Cross-Validation on Development Set
    print_section("� Step 2: K-Fold Cross-Validation with Nested Early Stopping", "-")
    print(f"Running {args.n_splits}-fold cross-validation on development set...")
    print(f"Each fold will:")
    print(f"  1. Split training folds into inner_train ({1-args.val_size:.0%}) and inner_val ({args.val_size:.0%})")
    print(f"  2. Train on inner_train with early stopping monitored on inner_val")
    print(f"  3. Evaluate on the fold's holdout set")
    
    # Get best parameters from hyperparameter search (if desired)
    # For simplicity, we'll use a single set of parameters and do K-fold evaluation
    # If you want hyperparameter search, it should be done on a smaller subset or with nested CV
    
    param_dist = build_default_param_dist()
    
    # For this implementation, we'll first do hyperparameter search on development set
    # using traditional GridSearchCV, then retrain with K-fold and early stopping
    print("\n🔍 Phase 2a: Hyperparameter Search (finding best parameters)...")
    print(f"   Using GridSearchCV with {args.n_splits}-fold CV on development set")
    
    base_model = LGB_CLASS(
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        device_type=device_type
    )
    
    cv_search = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_dist,
        scoring="roc_auc",
        cv=cv_search,
        verbose=2 if args.verbose else 1,
        n_jobs=args.n_jobs,
    )
    
    search_start = time.time()
    search.fit(X_development, y_development)
    search_time = time.time() - search_start
    
    print(f"\n✅ Hyperparameter search completed in {search_time:.2f} seconds")
    print(f"🏆 Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"📋 Best parameters:")
    best_params = search.best_params_
    for k, v in best_params.items():
        print(f"   {k}: {v}")
    
    # STEP 3: K-Fold Training with Nested Early Stopping
    print("\n🔍 Phase 2b: K-Fold Evaluation with Early Stopping...")
    print(f"Training {args.n_splits} models with best parameters and early stopping\n")
    
    cv_kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    fold_scores = []
    fold_details = []
    trained_models = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_kfold.split(X_development, y_development), 1):
        print(f"\n{'='*60}")
        print(f"📁 Fold {fold_idx}/{args.n_splits}")
        print(f"{'='*60}")
        
        # Get fold data
        X_fold_train = X_development.iloc[train_idx]
        y_fold_train = y_development.iloc[train_idx]
        X_fold_holdout = X_development.iloc[test_idx]
        y_fold_holdout = y_development.iloc[test_idx]
        
        print(f"   Fold train size: {len(X_fold_train)}")
        print(f"   Fold holdout size: {len(X_fold_holdout)}")
        
        # STEP 3.1: Nested split for early stopping
        print(f"\n   🔀 Nested split for early stopping (val_size={args.val_size})...")
        X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
            X_fold_train, y_fold_train, 
            test_size=args.val_size, 
            random_state=args.random_state,
            stratify=y_fold_train
        )
        print(f"      Inner train size: {len(X_inner_train)}")
        print(f"      Inner val size: {len(X_inner_val)}")
        
        # STEP 3.2: Train model with early stopping
        print(f"\n   🏋️  Training model with early stopping...")
        fold_model = LGB_CLASS(
            **best_params,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            device_type=device_type
        )
        
        # Use Trainer class for consistent early stopping handling
        trainer = Trainer(
            model=fold_model,
            X_train=X_inner_train,
            y_train=y_inner_train,
            X_val=X_inner_val,
            y_val=y_inner_val
        )
        
        # Train with early stopping
        if args.early_stopping_rounds > 0:
            trainer.fit(
                early_stopping_rounds=args.early_stopping_rounds,
                verbose=False
            )
            print(f"      ✅ Training complete (with early stopping)")
        else:
            trainer.fit()
            print(f"      ✅ Training complete (no early stopping)")
        
        # STEP 3.3: Evaluate on fold holdout
        print(f"\n   📊 Evaluating on fold holdout...")
        fold_metrics, fold_proba, fold_pred = trainer.evaluate(X_fold_holdout, y_fold_holdout)
        
        print(f"      ROC-AUC: {fold_metrics['roc_auc']:.4f}")
        print(f"      Precision: {fold_metrics['precision']:.4f}")
        print(f"      Recall: {fold_metrics['recall']:.4f}")
        print(f"      F1: {fold_metrics['f1']:.4f}")
        
        fold_scores.append(fold_metrics['roc_auc'])
        fold_details.append({
            'fold': fold_idx,
            'metrics': fold_metrics,
            'train_size': len(X_inner_train),
            'val_size': len(X_inner_val),
            'holdout_size': len(X_fold_holdout)
        })
        trained_models.append(fold_model)
    
    # STEP 4: Calculate cross-validation statistics
    print_section("📊 Step 3: K-Fold Cross-Validation Results", "=")
    fold_scores_array = pd.Series(fold_scores)
    mean_score = fold_scores_array.mean()
    std_score = fold_scores_array.std()
    
    print(f"🎯 Cross-Validation ROC-AUC Scores:")
    for i, score in enumerate(fold_scores, 1):
        print(f"   Fold {i}: {score:.4f}")
    print(f"\n   {'─'*40}")
    print(f"   Mean ROC-AUC:   {mean_score:.4f}")
    print(f"   Std Dev:        {std_score:.4f}")
    print(f"   95% CI:         [{mean_score - 1.96*std_score:.4f}, {mean_score + 1.96*std_score:.4f}]")
    print(f"   {'─'*40}")
    
    # STEP 5: Train final model on entire development set
    print_section("🏗️  Step 4: Training Final Model on Entire Development Set", "-")
    print("Training final model on full development set for deployment...")
    
    # Split development set for early stopping in final model
    X_dev_train, X_dev_val, y_dev_train, y_dev_val = train_test_split(
        X_development, y_development,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_development
    )
    
    final_model = LGB_CLASS(
        **best_params,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        device_type=device_type
    )
    
    final_trainer = Trainer(
        model=final_model,
        X_train=X_dev_train,
        y_train=y_dev_train,
        X_val=X_dev_val,
        y_val=y_dev_val
    )
    
    if args.early_stopping_rounds > 0:
        final_trainer.fit(early_stopping_rounds=args.early_stopping_rounds, verbose=False)
    else:
        final_trainer.fit()
    
    print(f"✅ Final model trained on {len(X_dev_train)} samples")
    
    # STEP 6: Final evaluation on untouched test set
    print_section("🎯 Step 5: Final Evaluation on Untouched Test Set", "=")
    print("Evaluating final model on the untouched final test set...")
    
    final_metrics, y_final_proba, y_final_pred = final_trainer.evaluate(
        X_final_test, y_final_test
    )
    
    print_metrics_table(final_metrics, "🎯 FINAL TEST SET RESULTS")
    
    print(f"\n📈 Model Performance Summary:")
    print(f"   Cross-Validation (Development Set):")
    print(f"      Mean ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
    print(f"   Final Test Set (Untouched Holdout):")
    print(f"      ROC-AUC: {final_metrics['roc_auc']:.4f}")

    # Create output directory
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save visualizations
    print_section("📊 Generating Visualizations", "-")
    save_visualizations(y_final_test, y_final_proba, y_final_pred, out_dir)

    # Save model and artifacts
    print_section("💾 Saving Results", "-")
    
    model_path = out_dir / "gradient_boosting_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"✅ Model saved: {model_path}")

    metrics_path = out_dir / "gradient_boosting_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")
    
    # Save fold details
    fold_details_path = out_dir / "cv_fold_details.json"
    with open(fold_details_path, "w") as f:
        json.dump(fold_details, f, indent=2)
    print(f"✅ Fold details saved: {fold_details_path}")

    # Create comprehensive summary
    total_time = time.time() - start_time
    summary = {
        "model": "LightGBM Classifier",
        "task": "Hospital 30-Day Readmission Risk Prediction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": args.environment,
        "device": device_type,
        "evaluation_pipeline": {
            "description": "Robust nested CV with final holdout",
            "final_holdout_size": args.test_size,
            "k_folds": args.n_splits,
            "inner_val_size": args.val_size,
            "early_stopping_rounds": args.early_stopping_rounds
        },
        "data": {
            "total_samples": len(X),
            "development_size": len(X_development),
            "final_test_size": len(X_final_test),
            "n_features": X.shape[1]
        },
        "best_params": best_params,
        "cross_validation": {
            "mean_roc_auc": float(mean_score),
            "std_roc_auc": float(std_score),
            "fold_scores": [float(s) for s in fold_scores],
            "n_folds": args.n_splits
        },
        "final_test_metrics": final_metrics,
        "hyperparameter_search_time_seconds": search_time,
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
    print(f"\n📊 Performance Summary:")
    print(f"   🔄 {args.n_splits}-Fold CV ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
    print(f"   🎯 Final Test ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print("\n🎉 Ready for deployment!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: preprocess data and train LightGBM with robust nested CV evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Default: Full hyperparameter search with 5-fold CV, early stopping, and final holdout
            # Performance settings (CPU cores, GPU) are auto-detected
            python train_gradient_boosting.py --verbose
            
            # Custom K-fold splits and larger validation set for early stopping
            python train_gradient_boosting.py --n-splits 10 --val-size 0.15
            
            # Larger final test set (30% holdout)
            python train_gradient_boosting.py --test-size 0.3
            
            # Disable early stopping
            python train_gradient_boosting.py --early-stopping-rounds 0
            
            # Kaggle usage (auto-detects GPU and uses all cores)
            !python phase-2-risk-modeling/train_gradient_boosting.py --verbose
        """
    )
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with features.csv and target.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save model and metrics (auto-detects Kaggle)")
    
    # Training arguments
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Final holdout test size fraction (default: 0.2)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Inner validation size for early stopping within each fold (default: 0.1)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of K-fold cross-validation splits (default: 5)")
    parser.add_argument("--early-stopping-rounds", type=int, default=50,
                        help="Early stopping rounds (0 to disable, default: 50)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Output arguments
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output from training/search")

    args = parser.parse_args()
    
    # Get repository root
    repo_root = Path(__file__).resolve().parents[1]
    
    # Auto-detect Kaggle environment
    on_kaggle = is_kaggle_environment()
    args.environment = "kaggle" if on_kaggle else "local"
    
    # Auto-detect and use best performance options
    print("🔍 Auto-detecting optimal performance settings...")
    
    # Set n_jobs: use all cores on Kaggle, all-but-one on local (to keep system responsive)
    if on_kaggle:
        args.n_jobs = -1  # Use all available cores
        print(f"   ✅ Kaggle detected: using all CPU cores (n_jobs=-1)")
    else:
        # On local machines, use all cores but leave one free
        import os
        cpu_count = os.cpu_count() or 1
        args.n_jobs = max(1, cpu_count - 1) if cpu_count > 2 else 1
        print(f"   ✅ Local environment: using {args.n_jobs} CPU cores (leaving 1 free)")
    
    # Auto-detect and enable GPU if available
    gpu_available = detect_gpu()
    args.use_gpu = gpu_available
    if gpu_available:
        print(f"   ✅ GPU detected and will be used for training")
    else:
        print(f"   ℹ️  No GPU detected, using CPU")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = "/kaggle/working/models" if on_kaggle else str(repo_root / "models")
    
    # Resolve paths
    data_dir = repo_root / args.data_dir
    preprocess_script = repo_root / "phase-1-data-explore-preprocessing" / "simple_preprocessing.py"
    
    features_file = data_dir / "features.csv"
    target_file = data_dir / "target.csv"
    
    # Check for processed data and run preprocessing if needed
    if not features_file.exists() or not target_file.exists():
        if not preprocess_script.exists():
            raise FileNotFoundError(f"Preprocessing script not found: {preprocess_script}")
        run_preprocessing(preprocess_script)
    else:
        print("✅ Processed data found, skipping preprocessing")
    
    # Update data_dir to absolute path for training
    args.data_dir = str(data_dir)
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()
