"""Complete pipeline to train Random Forest model for 30-day readmission prediction.

This script loads preprocessed data from HuggingFace Hub and trains a Random Forest model
with a robust evaluation pipeline:

Data Source:
- HuggingFace repository: auphong2707/hospital-readmission-risk-data
- Automatically downloads preprocessed features and target
- No local preprocessing needed

Evaluation Pipeline:
1. Combine Phase 1 Splits: Combine train+validation from Phase 1 as development_set
   - Phase 1 test set becomes final_test_set (untouched until final evaluation)
2. Hyperparameter Search: Manual grid search with K-fold CV to find best parameters
   - Each parameter combination evaluated with K-fold CV
   - Evaluate on fold's holdout set
3. K-Fold Cross-Validation with best parameters:
   - Perform K-fold CV only on development_set with best parameters
   - Evaluate on fold's holdout set
4. Calculate CV statistics (mean, std, confidence intervals)
5. Train final model on full development_set
6. Final evaluation on untouched final_test_set

Features:
- Robust nested cross-validation with proper data separation
- Auto-detects and uses optimal performance settings (CPU cores)
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
    python ./phase-2-risk-modeling/train_random_forest.py
    
Kaggle usage:
    !python ./phase-2-risk-modeling/train_random_forest.py

Examples:
    # Full hyperparameter search with 5-fold CV
    # Performance (CPU cores) auto-detected
    python train_random_forest.py
    
    # Custom K-fold splits
    python train_random_forest.py --n-splits 10

Requirements:
    pip install scikit-learn==1.2.2 pandas joblib matplotlib seaborn tqdm datasets
    
    Note: This script is compatible with scikit-learn 1.2.2
    Data is automatically loaded from HuggingFace Hub (auphong2707/hospital-readmission-risk-data)
    
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
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from itertools import product

# Version check
print(f"🔍 scikit-learn version: {sklearn.__version__}")
if sklearn.__version__ < "1.2.0":
    warnings.warn(f"Warning: scikit-learn {sklearn.__version__} detected. Recommended version: 1.2.2+")

from utilities import (
    calculate_comprehensive_metrics,
    print_metrics_table,
    save_visualizations,
    save_learning_curves,
    save_validation_curves,
    save_metrics_comparison,
    is_kaggle_environment,
    print_section,
    load_phase1_splits,
    upload_results_to_hf
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Install with: pip install tqdm")

warnings.filterwarnings("ignore")


def get_rf_param_grid():
    """Get parameter grid for Random Forest hyperparameter search.
    
    Returns:
        dict: Parameter grid with ranges for:
            - n_estimators: Number of trees in the forest
            - max_depth: Maximum depth of trees
            - min_samples_split: Minimum samples required to split
            - min_samples_leaf: Minimum samples required at leaf
            - max_features: Number of features to consider at each split
            - class_weight: Class balancing strategy
    """
    return {
        'n_estimators': [100, 250, 500],
        'max_depth': [10, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt'],
        'class_weight': ['balanced', {0: 1, 1: 8}],
        'bootstrap': [True],
        'oob_score': [True]
    }


class Trainer:
    """Simple trainer abstraction for sklearn estimators.

    Responsibilities:
    - Hold model and train data
    - Fit the model
    - Evaluate and return comprehensive metrics
    - Save model artifact
    
    Attributes:
        model: The sklearn-compatible model to train
        X_train: Training features
        y_train: Training labels
        output_dir: Directory for saving model artifacts
    """

    def __init__(self, model, X_train, y_train, output_dir="models"):
        """Initialize the Trainer.
        
        Args:
            model: sklearn-compatible model instance
            X_train: Training features
            y_train: Training labels
            output_dir: Directory to save models (default: "models")
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.output_dir = Path(output_dir)

    def fit(self, **fit_kwargs):
        """Fit the underlying model.
        
        Args:
            **fit_kwargs: Additional arguments to pass to model.fit()
        """
        self.model.fit(self.X_train, self.y_train, **fit_kwargs)

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
    1. Combine Phase 1 Splits: Combine train+validation from Phase 1 as development_set
    2. Hyperparameter search with K-fold CV on development_set
    3. K-fold CV with best parameters on development_set
    4. Evaluate on fold's holdout
    5. Train final model on full development_set
    6. Final evaluation on untouched final_test_set
    """
    start_time = time.time()
    
    # Print configuration
    print_section("🚀 Random Forest Training - Hospital Readmission Risk", "=")
    print(f"⚙️  Configuration:")
    print(f"   - Data source: HuggingFace (auphong2707/hospital-readmission-risk-data)")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - K-fold splits: {args.n_splits}")
    print(f"   - Random seed: {args.random_state}")
    print(f"   - Parallel jobs: {args.n_jobs}")
    print(f"   - Environment: {'🏆 Kaggle' if args.environment == 'kaggle' else '💻 Local'}")
    print(f"   - sklearn version: {sklearn.__version__}")
    
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

    # STEP 2: Hyperparameter Search with K-Fold Cross-Validation
    print_section("🔍 Step 2: Hyperparameter Search with K-Fold Cross-Validation", "-")
    print(f"Running {args.n_splits}-fold cross-validation on development set...")
    print(f"Note: Random Forest uses bootstrap samples and does not require separate validation split")
    
    param_dist = get_rf_param_grid()
    
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
            
            # Train model with current parameters
            combo_model = RandomForestClassifier(
                **params,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                verbose=0
            )
            
            combo_trainer = Trainer(
                model=combo_model,
                X_train=X_combo_train,
                y_train=y_combo_train
            )
            
            # Train
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
    for k, v in best_params.items():
        print(f"   {k}: {v}")
    
    # STEP 3: K-Fold Training with best parameters
    print("\n" + "="*60)
    print("📊 Step 3: Final K-Fold CV with Best Parameters")
    print("="*60)
    print(f"Re-training with best parameters to collect detailed metrics across {args.n_splits} folds")
    print(f"Using bootstrap=True and oob_score=True for out-of-bag evaluation\n")
    
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
        
        # Train model
        print(f"\n   🏋️  Training model...")
        fold_model = RandomForestClassifier(
            **best_params,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=0
        )
        
        # Use Trainer class
        trainer = Trainer(
            model=fold_model,
            X_train=X_fold_train,
            y_train=y_fold_train
        )
        
        # Train
        trainer.fit()
        print(f"      ✅ Training complete")
        
        # Evaluate on fold holdout
        print(f"\n   📊 Evaluating on fold holdout...")
        fold_metrics, fold_proba, fold_pred = trainer.evaluate(X_fold_holdout, y_fold_holdout)
        
        print(f"      ROC-AUC: {fold_metrics['roc_auc']:.4f}")
        print(f"      Precision: {fold_metrics['precision']:.4f}")
        print(f"      Recall: {fold_metrics['recall']:.4f}")
        print(f"      F1: {fold_metrics['f1']:.4f}")
        
        # Get OOB score if available
        oob_score = float(fold_model.oob_score_) if hasattr(fold_model, 'oob_score_') else None
        if oob_score:
            print(f"      OOB Score: {oob_score:.4f}")
        
        fold_scores.append(fold_metrics['roc_auc'])
        fold_details.append({
            'fold': fold_idx,
            'metrics': fold_metrics,
            'train_size': len(X_fold_train),
            'holdout_size': len(X_fold_holdout),
            'oob_score': oob_score
        })
        trained_models.append(fold_model)
    
    # STEP 4: Calculate cross-validation statistics
    print_section("📊 Step 4: K-Fold Cross-Validation Results", "=")
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
    print_section("🏗️  Step 5: Training Final Model on Entire Development Set", "-")
    print("Training final model on full development set for deployment...")
    
    final_model = RandomForestClassifier(
        **best_params,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=0
    )
    
    final_trainer = Trainer(
        model=final_model,
        X_train=X_development,
        y_train=y_development
    )
    
    final_trainer.fit()
    
    print(f"✅ Final model trained on {len(X_development)} samples")
    if hasattr(final_model, 'oob_score_'):
        print(f"   OOB Score: {final_model.oob_score_:.4f}")
    
    # STEP 6: Final evaluation on untouched test set
    print_section("🎯 Step 6: Final Evaluation on Untouched Test Set", "=")
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
        model=final_model, X=X_final_test, feature_names=X_development.columns.tolist()
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
    
    model_path = out_dir / "random_forest_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"✅ Model saved: {model_path}")

    metrics_path = out_dir / "random_forest_metrics.json"
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
        "model": "Random Forest Classifier",
        "task": "Hospital 30-Day Readmission Risk Prediction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": args.environment,
        "sklearn_version": sklearn.__version__,
        "device": "cpu",
        "evaluation_pipeline": {
            "description": "Robust nested CV with final holdout",
            "k_folds": args.n_splits,
            "uses_bootstrap": True,
            "uses_oob_score": True
        },
        "data": {
            "total_samples": len(X_train) + len(X_val) + len(X_test),
            "development_size": len(X_development),
            "final_test_size": len(X_final_test),
            "n_features": X_development.shape[1]
        },
        "best_params": best_params,
        "oob_score": float(final_model.oob_score_) if hasattr(final_model, 'oob_score_') else None,
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
        model_name="hospital-readmission-rf"
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
    if hasattr(final_model, 'oob_score_'):
        print(f"   🌳 OOB Score: {final_model.oob_score_:.4f}")
    print("\n🎉 Ready for deployment!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest with robust nested CV evaluation using preprocessed data from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Default: Full hyperparameter search with 5-fold CV
            # Data loaded from HuggingFace: auphong2707/hospital-readmission-risk-data
            # Performance settings (CPU cores) are auto-detected
            python train_random_forest.py
            
            # Custom K-fold splits
            python train_random_forest.py --n-splits 10
            
            # Kaggle usage (auto-detects and uses all cores)
            !python phase-2-risk-modeling/train_random_forest.py
        """
    )
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save model and metrics (auto-detects Kaggle)")
    
    # Training arguments
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of K-fold cross-validation splits (default: 5)")
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
        cpu_count = os.cpu_count() or 1
        args.n_jobs = max(1, cpu_count - 1) if cpu_count > 2 else 1
        print(f"   ✅ Local environment: using {args.n_jobs} CPU cores (leaving 1 free)")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = "/kaggle/working/models" if on_kaggle else str(repo_root / "models")
    
    print("\n📥 Data will be loaded directly from HuggingFace: auphong2707/hospital-readmission-risk-data")
    print("   (No local preprocessing needed)")
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()

