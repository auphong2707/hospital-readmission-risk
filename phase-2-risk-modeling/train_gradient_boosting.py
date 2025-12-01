"""Complete pipeline to train LightGBM model for 30-day readmission prediction.

This script loads preprocessed data from HuggingFace Hub and trains a LightGBM model
with a robust evaluation pipeline:

Data Source:
- HuggingFace repository: auphong2707/hospital-readmission-risk-data
- Automatically downloads preprocessed features and target
- No local preprocessing needed

Evaluation Pipeline:
1. Final Holdout Split: Split entire dataset into development_set and final_test_set
   - final_test_set remains untouched until final evaluation
2. Hyperparameter Search: Manual grid search with K-fold CV to find best parameters
   - Each parameter combination evaluated with K-fold CV
   - Inside each fold: split training data into inner_train and inner_val
   - Train on inner_train with early stopping monitored on inner_val
   - Evaluate on fold's holdout set
3. K-Fold Cross-Validation with best parameters:
   - Perform K-fold CV only on development_set with best parameters
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
- Comprehensive visualizations:
  * ROC curves with AUC
  * Precision-Recall curves
  * Confusion matrices
  * Calibration curves
  * Feature importance plots
  * Learning curves
  * Validation curves for hyperparameters
  * Cross-fold metrics comparison
- Exhaustive grid search with K-fold cross-validation
- Comprehensive metrics:
  * Primary: ROC-AUC, PR-AUC, Accuracy, Balanced Accuracy, F1
  * Classification: Precision, Recall, Specificity, FPR, FNR
  * Clinical: Sensitivity, Specificity, PPV, NPV
  * Calibration: Brier score

Usage (from project root):
    python ./phase-2-risk-modeling/train_gradient_boosting.py
    
Kaggle usage:
    !python ./phase-2-risk-modeling/train_gradient_boosting.py

Examples:
    # Full hyperparameter search with 5-fold CV and early stopping
    # Performance (GPU, CPU cores) auto-detected
    python train_gradient_boosting.py
    
    # Custom K-fold splits and validation size
    python train_gradient_boosting.py --n-splits 10 --val-size 0.15

Requirements:
    pip install lightgbm scikit-learn pandas joblib matplotlib seaborn tqdm datasets
    
Note: Data is automatically loaded from HuggingFace Hub (auphong2707/hospital-readmission-risk-data)
    
HuggingFace Upload:
    Results are automatically uploaded to HuggingFace Hub after training.
    Set HF_TOKEN and HF_USERNAME in .env file to enable:
        HF_TOKEN=your_token_here
        HF_USERNAME=your_username
    Get token from: https://huggingface.co/settings/tokens
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
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from itertools import product

from utilities import (
    calculate_comprehensive_metrics,
    print_metrics_table,
    save_visualizations,
    save_learning_curves,
    save_validation_curves,
    save_metrics_comparison,
    detect_gpu,
    is_kaggle_environment,
    print_section,
    load_phase1_splits,
    get_lgbm_param_grid,
    upload_results_to_hf
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Install with: pip install tqdm")

warnings.filterwarnings("ignore")


class Trainer:
    """Simple trainer abstraction for sklearn/LightGBM estimators.

    Responsibilities:
    - Hold model and train/val data
    - Fit with optional early stopping
    - Evaluate and return comprehensive metrics
    - Save model artifact
    
    Attributes:
        model: The sklearn-compatible model to train
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features for early stopping
        y_val: Optional validation labels for early stopping
        output_dir: Directory for saving model artifacts
    """

    def __init__(self, model, X_train, y_train, X_val=None, y_val=None, output_dir="models"):
        """Initialize the Trainer.
        
        Args:
            model: sklearn-compatible model instance
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            output_dir: Directory to save models (default: "models")
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.output_dir = Path(output_dir)

    def fit(self, early_stopping_rounds: int | None = None, **fit_kwargs):
        """Fit the underlying model with optional early stopping.
        
        For LightGBM sklearn API, passes eval_set and early_stopping_rounds 
        when validation data is provided.
        
        Args:
            early_stopping_rounds: Number of rounds for early stopping (None to disable)
            **fit_kwargs: Additional arguments to pass to model.fit()
        """
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
                    fit_args.setdefault("callbacks", [lgb.early_stopping(stopping_rounds=early_stopping_rounds)])
                else:
                    # Fall back to old API (LightGBM < 4.0)
                    fit_args.setdefault("early_stopping_rounds", early_stopping_rounds)
            except:
                # If all else fails, use old API
                fit_args.setdefault("early_stopping_rounds", early_stopping_rounds)

        # Fit the model
        self.model.fit(self.X_train, self.y_train, **fit_args)

    def evaluate(self, X, y, threshold: float = 0.5):
        """Evaluate model with comprehensive metrics.
        
        Args:
            X: Features to evaluate on
            y: True labels
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            tuple: (metrics_dict, probabilities, predictions)
        """
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        
        # Use the comprehensive metrics calculation function
        metrics = calculate_comprehensive_metrics(y, proba, threshold)
        
        return metrics, proba, pred

    def save(self, path: str | Path):
        """Save the trained model to disk.
        
        Args:
            path: File path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

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
    print(f"   - Data source: HuggingFace (auphong2707/hospital-readmission-risk-data)")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - Test size: {args.test_size}")
    print(f"   - Validation size (for early stopping): {args.val_size}")
    print(f"   - K-fold splits: {args.n_splits}")
    print(f"   - Early stopping rounds: {args.early_stopping_rounds}")
    print(f"   - Random seed: {args.random_state}")
    print(f"   - Parallel jobs: {args.n_jobs}")
    print(f"   - Device: {'GPU' if args.use_gpu else 'CPU'}")
    print(f"   - Environment: {'🏆 Kaggle' if args.environment == 'kaggle' else '💻 Local'}")
    
    # Load Phase 1 splits from HuggingFace
    X_train, X_val, X_test, y_train, y_val, y_test = load_phase1_splits()

    # STEP 1: Combine train + validation for development set (K-fold CV)
    print_section("🔀 Step 1: Prepare Development Set from Phase 1 Splits", "-")
    print(f"Using Phase 1's preprocessed splits (single source of truth)")
    print(f"Combining train + validation for development set (K-fold CV)...")
    
    X_development = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_development = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    X_final_test = X_test
    y_final_test = y_test
    
    print(f"   ✅ Development set (train + val): {X_development.shape}")
    print(f"   ✅ Final test set (Phase 1): {X_final_test.shape}")
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
    
    param_dist = get_lgbm_param_grid()
    
    # Generate all parameter combinations for manual grid search
    param_combinations = [dict(zip(param_dist.keys(), v)) for v in product(*param_dist.values())]
    total_combinations = len(param_combinations)
    
    print(f"\n📊 Hyperparameter Search Space:")
    print(f"   Total parameter combinations: {total_combinations}")
    print(f"   K-fold splits: {args.n_splits}")
    print(f"   Total model fits: {total_combinations * args.n_splits}")
    print(f"   Scoring metric: ROC-AUC\n")
    
    # Track best parameters and scores
    best_score = -np.inf
    best_params = None
    all_search_results = []
    
    cv_search = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    search_start = time.time()
    
    # Iterate through all parameter combinations
    for combo_idx, params in enumerate(param_combinations, 1):
        if combo_idx % 10 == 0 or combo_idx == 1 or combo_idx == total_combinations:
            print(f"🔍 Evaluating combination {combo_idx}/{total_combinations}")
            print(f"   Parameters: {params}")
        
        # Evaluate this parameter combination with K-fold CV
        combo_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_search.split(X_development, y_development), 1):
            X_combo_train = X_development.iloc[train_idx]
            y_combo_train = y_development.iloc[train_idx]
            X_combo_val = X_development.iloc[val_idx]
            y_combo_val = y_development.iloc[val_idx]
            
            # Split training data for early stopping
            X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                X_combo_train, y_combo_train,
                test_size=args.val_size,
                random_state=args.random_state,
                stratify=y_combo_train
            )
            
            # Train model with current parameters
            combo_model = LGB_CLASS(
                **params,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                device_type=device_type
            )
            
            combo_trainer = Trainer(
                model=combo_model,
                X_train=X_inner_train,
                y_train=y_inner_train,
                X_val=X_inner_val,
                y_val=y_inner_val
            )
            
            # Train with early stopping
            if args.early_stopping_rounds > 0:
                combo_trainer.fit(early_stopping_rounds=args.early_stopping_rounds)
            else:
                combo_trainer.fit()
            
            # Evaluate on fold's validation set
            fold_metrics, _, _ = combo_trainer.evaluate(X_combo_val, y_combo_val)
            combo_scores.append(fold_metrics['roc_auc'])
        
        # Calculate mean score for this parameter combination
        mean_score_combo = np.mean(combo_scores)
        std_score_combo = np.std(combo_scores)
        
        all_search_results.append({
            'params': params,
            'mean_score': mean_score_combo,
            'std_score': std_score_combo,
            'fold_scores': combo_scores
        })
        
        if combo_idx % 10 == 0 or combo_idx == 1 or combo_idx == total_combinations:
            print(f"   Mean ROC-AUC: {mean_score_combo:.4f} ± {std_score_combo:.4f}")
        
        # Update best parameters
        if mean_score_combo > best_score:
            best_score = mean_score_combo
            best_params = params
            print(f"   >>> 🏆 New best score: {best_score:.4f}")
    
    search_time = time.time() - search_start
    
    print(f"\n{'='*60}")
    print("✅ Hyperparameter search completed")
    print(f"{'='*60}")
    print(f"⏱️  Search time: {search_time:.2f} seconds")
    print(f"🏆 Best CV ROC-AUC: {best_score:.4f}")
    print(f"📋 Best parameters:")
    best_params = best_params
    for k, v in best_params.items():
        print(f"   {k}: {v}")
    
    # STEP 3: K-Fold Training with Nested Early Stopping using best parameters
    print("\n" + "="*60)
    print("� Step 3: Final K-Fold CV with Best Parameters")
    print("="*60)
    print(f"Re-training with best parameters to collect detailed metrics across {args.n_splits} folds\n")
    
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
                early_stopping_rounds=args.early_stopping_rounds
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
        final_trainer.fit(early_stopping_rounds=args.early_stopping_rounds)
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

    # Save comprehensive visualizations
    print_section("📊 Generating Comprehensive Visualizations", "-")
    
    # Main visualizations (ROC, PR, Confusion Matrix, Calibration, Feature Importance)
    save_visualizations(
        y_final_test, y_final_proba, y_final_pred, out_dir,
        model=final_model, X=X_final_test, feature_names=X.columns.tolist()
    )
    
    # Learning curves
    save_learning_curves(
        final_model, X_development, y_development, out_dir, cv=args.n_splits
    )
    
    # Validation curves from hyperparameter search
    save_validation_curves(all_search_results, out_dir)
    
    # Metrics comparison across folds
    save_metrics_comparison(fold_details, out_dir)

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

    # Upload to HuggingFace Hub (automatically)
    print_section("📤 Uploading to HuggingFace Hub", "-")
    upload_success = upload_results_to_hf(
        summary=summary,
        output_dir=out_dir,
        model_name="hospital-readmission-lgbm"
    )
    if not upload_success:
        print("⚠️  Upload to HuggingFace Hub was skipped (set HF_TOKEN in .env to enable)")

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
        description="Train LightGBM with robust nested CV evaluation using preprocessed data from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Default: Full hyperparameter search with 5-fold CV, early stopping, and final holdout
            # Data loaded from HuggingFace: auphong2707/hospital-readmission-risk-data
            # Performance settings (CPU cores, GPU) are auto-detected
            python train_gradient_boosting.py
            
            # Custom K-fold splits and larger validation set for early stopping
            python train_gradient_boosting.py --n-splits 10 --val-size 0.15
            
            # Larger final test set (30% holdout)
            python train_gradient_boosting.py --test-size 0.3
            
            # Disable early stopping
            python train_gradient_boosting.py --early-stopping-rounds 0
            
            # Kaggle usage (auto-detects GPU and uses all cores)
            !python phase-2-risk-modeling/train_gradient_boosting.py
        """
    )
    
    # Output arguments
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
    gpu_available = detect_gpu(verbose=True)
    args.use_gpu = gpu_available
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = "/kaggle/working/models" if on_kaggle else str(repo_root / "models")
    
    print("\n📥 Data will be loaded directly from HuggingFace: auphong2707/hospital-readmission-risk-data")
    print("   (No local preprocessing needed)")
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()
