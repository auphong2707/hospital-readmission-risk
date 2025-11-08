"""
Logistic Regression Training for Hospital Readmission Risk Prediction
=====================================================================

Baseline model with comprehensive configuration and robust evaluation pipeline:

Evaluation Pipeline:
1. Final Holdout Split: Split entire dataset into development_set and final_test_set
   - final_test_set remains untouched until final evaluation
2. Hyperparameter Search: Grid search with K-fold CV to find best parameters
   - Each parameter combination evaluated with K-fold CV on development set
   - Evaluate on fold's holdout set
3. K-Fold Cross-Validation with best parameters:
   - Perform K-fold CV only on development_set with best parameters
   - Calculate CV statistics (mean, std, confidence intervals)
4. Train final model on full development_set
5. Final evaluation on untouched final_test_set

Features:
- L1/L2 regularization (Elastic Net)
- Class weight balancing for imbalanced data
- Stratified K-fold cross-validation (k=5)
- Grid search for hyperparameter optimization
- Comprehensive evaluation metrics

Author: Hospital Readmission Risk Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import os
import time
import json
import joblib
from datetime import datetime
from pathlib import Path
import pickle
import os
import warnings
import time
import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import utilities
from utilities import (
    calculate_comprehensive_metrics,
    print_metrics_table,
    save_visualizations,
    print_section,
    upload_results_to_hf,
    load_data
)

warnings.filterwarnings('ignore')


class LogisticRegressionTrainer:
    """
    Comprehensive Logistic Regression training pipeline for hospital readmission prediction.
    
    Features:
    - L1/L2 regularization with hyperparameter tuning
    - Class weight balancing
    - Stratified K-fold cross-validation
    - Grid search optimization
    - Feature importance analysis
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.grid_search = None
        self.feature_names = None
        self.scaler = StandardScaler()
    
    def create_hyperparameter_grid(self):
        """
        Create comprehensive hyperparameter grid for logistic regression.
        
        Includes:
        - Regularization strength (C): controls overfitting
        - Penalty type: L1 (Lasso), L2 (Ridge), Elastic Net
        - L1 ratio: mixing parameter for Elastic Net (only used when penalty='elasticnet')
        - Class weights: balanced vs custom ratios
        """
        param_grid = {
            # Regularization strength (inverse of regularization: smaller = more regularization)
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            
            # Penalty type: L1, L2, or Elastic Net
            'penalty': ['l1', 'l2', 'elasticnet'],
            
            # Solver compatibility: 'saga' supports all penalty types including elasticnet
            'solver': ['saga'],
            
            # L1 ratio for Elastic Net (0 = L2, 1 = L1, 0.5 = equal mix)
            # Only used when penalty='elasticnet'
            'l1_ratio': [0.25, 0.5, 0.75],
            
            # Class weight balancing
            'class_weight': ['balanced', {0: 1, 1: 8}],
            
            # Maximum iterations
            'max_iter': [2000]
        }
        
        return param_grid
    
    def train_with_cv(self, X_train, y_train, n_folds=5):
        """
        Train logistic regression with stratified K-fold cross-validation and grid search.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_folds: Number of folds for cross-validation
            
        Returns:
            Best model from grid search
        """
        print(f"\n{'='*70}")
        print("Hyperparameter Tuning with Stratified K-Fold Cross-Validation")
        print(f"{'='*70}")
        
        # Create parameter grid
        param_grid = self.create_hyperparameter_grid()
        
        print(f"Grid search configuration:")
        print(f"  Number of CV folds: {n_folds}")
        print(f"  Stratification: Yes (balanced across readmission classes)")
        print(f"  Scoring metric: AUC-ROC")
        
        # Calculate total combinations (considering l1_ratio is only used with elasticnet)
        # For L1 and L2: C * class_weight combinations each
        # For elasticnet: C * l1_ratio * class_weight combinations
        base_combos = len(param_grid['C']) * len(param_grid['class_weight'])
        elasticnet_combos = base_combos * len(param_grid['l1_ratio'])
        total_combos = 2 * base_combos + elasticnet_combos  # 2 for L1 and L2
        print(f"  Total combinations: ~{total_combos}")
        
        # Create stratified K-fold splitter
        cv_splitter = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Create base logistic regression model
        base_model = LogisticRegression(
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            warm_start=False  # Don't reuse previous solution
        )
        
        # Create grid search with AUC-ROC scoring
        self.grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=3,  # Increased verbosity for better progress tracking
            refit=True,
            return_train_score=True
        )
        
        # Calculate total number of fits
        # L1 and L2 each: C * class_weight combinations
        # Elasticnet: C * l1_ratio * class_weight combinations
        base_combos = len(param_grid['C']) * len(param_grid['class_weight'])
        elasticnet_combos = base_combos * len(param_grid['l1_ratio'])
        total_combinations = 2 * base_combos + elasticnet_combos
        total_fits = total_combinations * n_folds
        
        print(f"\n🔄 Starting grid search...")
        print(f"📊 Training {total_combinations} hyperparameter combinations with {n_folds}-fold CV")
        print(f"📈 Total model fits: {total_fits}")
        print(f"⏱️  Progress will be displayed below with timing for each fit...\n")
        print(f"{'='*70}")
        
        # Fit grid search with timing
        start_time = time.time()
        self.grid_search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ Grid search completed in {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
        print(f"⚡ Average time per fit: {elapsed_time/total_fits:.1f} seconds")
        print(f"{'='*70}")
        
        # Extract best model
        self.best_model = self.grid_search.best_estimator_
        
        # Print results
        print(f"\n{'='*70}")
        print("Grid Search Results")
        print(f"{'='*70}")
        print(f"Best CV Score (AUC-ROC): {self.grid_search.best_score_:.4f}")
        print(f"\nBest Hyperparameters:")
        for param, value in self.grid_search.best_params_.items():
            print(f"  {param:<20}: {value}")
        print(f"{'='*70}\n")
        
        return self.best_model
    
    def evaluate_model(self, X, y, set_name="", output_dir=None):
        """
        Evaluate model on a dataset.
        
        Args:
            X: Features
            y: Target
            set_name: Name of the dataset (e.g., "Training", "Validation", "Test")
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary of metrics
        """
        print_section(f"Evaluating on {set_name} Set", "-")
        
        # Make predictions
        y_pred = self.best_model.predict(X)
        y_prob = self.best_model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(y, y_prob, threshold=0.5)
        
        # Print metrics
        print_metrics_table(metrics, f"Logistic Regression - {set_name} Set")
        
        # Generate plots if output directory provided
        if output_dir:
            set_output_dir = Path(output_dir) / set_name.lower().replace(" ", "_")
            set_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate visualizations
            save_visualizations(
                y, y_prob, y_pred, set_output_dir,
                model=self.best_model, X=X, feature_names=self.feature_names
            )
        
        return metrics
    
    def save_model(self, output_dir="./models", include_metadata=True):
        """
        Save trained model with metadata.
        
        Args:
            output_dir: Directory to save the model
            include_metadata: Whether to save hyperparameters and feature names
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"logistic_regression.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"\n✅ Model saved to: {model_path}")
        
        # Save metadata
        if include_metadata:
            metadata = {
                'model_type': 'Logistic Regression',
                'timestamp': timestamp,
                'best_params': self.grid_search.best_params_ if self.grid_search else None,
                'best_cv_score': self.grid_search.best_score_ if self.grid_search else None,
                'feature_names': self.feature_names,
                'n_features': len(self.feature_names) if self.feature_names else None,
                'random_state': self.random_state
            }
            
            metadata_path = os.path.join(output_dir, f"logistic_regression_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"✅ Metadata saved to: {metadata_path}")
        
        return model_path
    
    def plot_cv_results(self, output_dir=None):
        """
        Plot cross-validation results to analyze hyperparameter impact.
        """
        if self.grid_search is None:
            print("⚠️ No grid search results available. Train the model first.")
            return
        
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        
        # Plot mean test score by C value
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Group by regularization strength
        for penalty in cv_results['param_penalty'].unique():
            if pd.notna(penalty):
                subset = cv_results[cv_results['param_penalty'] == penalty]
                axes[0].plot(subset['param_C'], subset['mean_test_score'], 
                           marker='o', label=f'Penalty: {penalty}')
        
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Regularization Strength (C)', fontsize=12)
        axes[0].set_ylabel('Mean CV Score (AUC-ROC)', fontsize=12)
        axes[0].set_title('Hyperparameter Tuning: Regularization Impact', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot top 10 parameter combinations
        top_10 = cv_results.nlargest(10, 'mean_test_score')[['mean_test_score', 'std_test_score']].reset_index(drop=True)
        axes[1].barh(range(len(top_10)), top_10['mean_test_score'], xerr=top_10['std_test_score'],
                    color='steelblue', alpha=0.7)
        axes[1].set_yticks(range(len(top_10)))
        axes[1].set_yticklabels([f'Config {i+1}' for i in range(len(top_10))])
        axes[1].set_xlabel('Mean CV Score (AUC-ROC)', fontsize=12)
        axes[1].set_ylabel('Configuration Rank', fontsize=12)
        axes[1].set_title('Top 10 Hyperparameter Configurations', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / 'cv_results_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ CV results plot saved to: {save_path}")
        
        plt.close()


def main(hf_repo_id: Optional[str] = None,
         hf_token: Optional[str] = None,
         hf_private: bool = False):
    """
    Main training pipeline for Logistic Regression baseline model.
    
    Automatically uploads to HuggingFace if HF_TOKEN is available in environment.
    
    Args:
        hf_repo_id: HuggingFace repository ID (e.g., "username/hospital-readmission-lr")
                   If None and HF_TOKEN exists, will use "username/hospital-readmission-lr"
        hf_token: HuggingFace API token (if None, uses HF_TOKEN env variable)
        hf_private: Whether to make the HuggingFace repository private
    """
    print(f"\n{'='*70}")
    print("LOGISTIC REGRESSION TRAINING - HOSPITAL READMISSION RISK")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - L1/L2/Elastic Net regularization (with l1_ratio tuning)")
    print(f"  - Class weight balancing")
    print(f"  - Robust evaluation: 85% development, 15% final holdout")
    print(f"  - Stratified 5-fold cross-validation")
    print(f"  - Grid search hyperparameter optimization")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Initialize trainer
    trainer = LogisticRegressionTrainer(random_state=42)
    
    # STEP 1: Load preprocessed data and create final holdout split
    print_section("📊 Step 1: Load Data & Create Final Holdout Split", "=")
    X, y = load_data()
    trainer.feature_names = X.columns.tolist()
    
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {len(trainer.feature_names)}")
    print(f"Target distribution: {dict(y.value_counts())}")
    
    # Create final holdout split (85% development, 15% final test)
    X_development, X_final_test, y_development, y_final_test = train_test_split(
        X, y, 
        test_size=0.15, 
        random_state=42,
        stratify=y
    )
    
    print(f"\n✅ Final holdout split created:")
    print(f"   Development set: {len(X_development)} samples ({len(X_development)/len(X)*100:.1f}%)")
    print(f"   Final test set: {len(X_final_test)} samples ({len(X_final_test)/len(X)*100:.1f}%)")
    
    # STEP 2: Hyperparameter search with K-fold CV on development set
    print_section("🔍 Step 2: Hyperparameter Search with Cross-Validation", "=")
    print("Performing grid search with 5-fold stratified CV on development set...")
    
    best_model = trainer.train_with_cv(X_development, y_development, n_folds=5)
    best_params = trainer.grid_search.best_params_
    best_cv_score = trainer.grid_search.best_score_
    
    print(f"\n✅ Hyperparameter search completed")
    print(f"🏆 Best CV ROC-AUC: {best_cv_score:.4f}")
    print(f"📋 Best parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")
    
    # STEP 3: K-Fold CV with best parameters to collect statistics
    print_section("📊 Step 3: K-Fold Cross-Validation with Best Parameters", "=")
    print(f"Re-training with best parameters to collect detailed metrics across 5 folds\n")
    
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    fold_details = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_kfold.split(X_development, y_development), 1):
        print(f"\n{'='*60}")
        print(f"📁 Fold {fold_idx}/5")
        print(f"{'='*60}")
        
        # Get fold data
        X_fold_train = X_development.iloc[train_idx]
        y_fold_train = y_development.iloc[train_idx]
        X_fold_test = X_development.iloc[test_idx]
        y_fold_test = y_development.iloc[test_idx]
        
        print(f"   Fold train size: {len(X_fold_train)}")
        print(f"   Fold test size: {len(X_fold_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_test_scaled = scaler.transform(X_fold_test)
        
        # Train model with best parameters
        fold_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
        fold_model.fit(X_fold_train_scaled, y_fold_train)
        
        # Evaluate on fold test set
        y_fold_pred = fold_model.predict(X_fold_test_scaled)
        y_fold_proba = fold_model.predict_proba(X_fold_test_scaled)[:, 1]
        
        fold_metrics = calculate_comprehensive_metrics(y_fold_test, y_fold_pred, y_fold_proba)
        
        print(f"\n   📊 Fold {fold_idx} Results:")
        print(f"      ROC-AUC: {fold_metrics['roc_auc']:.4f}")
        print(f"      Precision: {fold_metrics['precision']:.4f}")
        print(f"      Recall: {fold_metrics['recall']:.4f}")
        print(f"      F1: {fold_metrics['f1']:.4f}")
        
        fold_scores.append(fold_metrics['roc_auc'])
        fold_details.append({
            'fold': fold_idx,
            'metrics': fold_metrics,
            'train_size': len(X_fold_train),
            'test_size': len(X_fold_test)
        })
    
    # Calculate CV statistics
    print_section("📊 Cross-Validation Results", "=")
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
    
    # STEP 4: Train final model on entire development set
    print_section("🏗️  Step 4: Training Final Model on Development Set with Validation", "-")
    print("Training final model with validation split for monitoring...")
    
    # Split development set for validation monitoring
    X_dev_train, X_dev_val, y_dev_train, y_dev_val = train_test_split(
        X_development, y_development,
        test_size=0.15,
        random_state=42,
        stratify=y_development
    )
    
    print(f"   Inner train size: {len(X_dev_train)}")
    print(f"   Inner val size: {len(X_dev_val)}")
    
    # Scale features
    X_dev_train_scaled = trainer.scaler.fit_transform(X_dev_train)
    X_dev_val_scaled = trainer.scaler.transform(X_dev_val)
    
    # Train final model
    final_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
    final_model.fit(X_dev_train_scaled, y_dev_train)
    trainer.best_model = final_model
    
    # Monitor validation performance
    y_dev_val_proba = final_model.predict_proba(X_dev_val_scaled)[:, 1]
    dev_val_auc = roc_auc_score(y_dev_val, y_dev_val_proba)
    
    print(f"✅ Final model trained on {len(X_dev_train)} samples")
    print(f"   Validation AUC-ROC: {dev_val_auc:.4f}")
    
    # STEP 5: Final evaluation on untouched test set
    print_section("🎯 Step 5: Final Evaluation on Untouched Test Set", "=")
    print("Evaluating final model on the untouched final test set...")
    
    X_final_test_scaled = trainer.scaler.transform(X_final_test)
    y_final_pred = final_model.predict(X_final_test_scaled)
    y_final_proba = final_model.predict_proba(X_final_test_scaled)[:, 1]
    
    final_metrics = calculate_comprehensive_metrics(y_final_test, y_final_pred, y_final_proba)
    print_metrics_table(final_metrics, "🎯 FINAL TEST SET RESULTS")
    
    print(f"\n📈 Model Performance Summary:")
    print(f"   Cross-Validation (Development Set):")
    print(f"      Mean ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
    print(f"   Final Test Set (Untouched Holdout):")
    print(f"      ROC-AUC: {final_metrics['roc_auc']:.4f}")
    
    # Create output directories
    output_dir = Path("../models")
    reports_dir = Path("../reports/logistic_regression")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save comprehensive visualizations
    print_section("📊 Generating Comprehensive Visualizations", "-")
    save_visualizations(
        y_final_test, y_final_proba, y_final_pred, reports_dir,
        model=final_model, X=X_final_test_scaled, feature_names=trainer.feature_names
    )
    
    # Plot CV results
    trainer.plot_cv_results(str(reports_dir))
    
    # Save model
    print_section("💾 Saving Results", "-")
    model_path = trainer.save_model(output_dir=str(output_dir))
    
    # Save metrics as JSON
    metrics_json_path = output_dir / "logistic_regression_metrics.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"✅ Metrics saved: {metrics_json_path}")
    
    # Save fold details
    fold_details_path = output_dir / "logistic_regression_cv_fold_details.json"
    with open(fold_details_path, 'w') as f:
        json.dump(fold_details, f, indent=2)
    print(f"✅ Fold details saved: {fold_details_path}")
    
    # Create comprehensive training summary
    total_time = time.time() - start_time
    training_summary = {
        'model': 'Logistic Regression',
        'task': 'Hospital 30-Day Readmission Risk Prediction',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'evaluation_pipeline': {
            'description': 'Robust nested CV with final holdout and validation monitoring',
            'final_holdout_size': 0.15,
            'inner_val_size': 0.15,
            'k_folds': 5,
            'cv_strategy': 'StratifiedKFold'
        },
        'data': {
            'total_samples': len(X),
            'development_size': len(X_development),
            'dev_train_size': len(X_dev_train),
            'dev_val_size': len(X_dev_val),
            'final_test_size': len(X_final_test),
            'n_features': len(trainer.feature_names)
        },
        'best_hyperparameters': best_params,
        'cross_validation': {
            'mean_roc_auc': float(mean_score),
            'std_roc_auc': float(std_score),
            'fold_scores': [float(s) for s in fold_scores],
            'n_folds': 5
        },
        'validation_monitoring': {
            'dev_val_auc': float(dev_val_auc)
        },
        'final_test_metrics': final_metrics,
        'total_time_seconds': total_time,
        'random_state': 42
    }
    
    summary_json_path = output_dir / "logistic_regression_training_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"✅ Training summary saved: {summary_json_path}")
    
    # Auto-upload to HuggingFace if token is available
    hf_url = None
    hf_token_from_env = os.getenv('HF_TOKEN')
    hf_username = os.getenv('HF_USERNAME', 'auphong2707')
    
    if hf_token_from_env or hf_token:
        print_section("🚀 HuggingFace Upload", "-")
        
        # Prepare summary for upload
        summary = {
            'model_name': 'Logistic Regression',
            'model_type': 'logistic-regression',
            'best_params': best_params,
            'best_cv_score': float(best_cv_score),
            'cv_mean_score': float(mean_score),
            'cv_std_score': float(std_score),
            'final_test_metrics': final_metrics,
            'n_features': len(trainer.feature_names),
            'feature_names': trainer.feature_names,
            'data_split': {
                'development_size': len(X_development),
                'final_test_size': len(X_final_test)
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Use provided repo_id or construct from username
        if hf_repo_id is None:
            hf_repo_id = f"{hf_username}/logistic-regression"
            print(f"⚠️  No repo_id provided. Using default: {hf_repo_id}")
        
        # Upload to HuggingFace
        upload_success = upload_results_to_hf(
            summary=summary,
            output_dir=str(output_dir),
            model_name="Logistic Regression",
            hf_repo_name=hf_repo_id,
            hf_token=hf_token or hf_token_from_env,
            metrics_json_path=str(metrics_json_path),
            summary_json_path=str(summary_json_path)
        )
        
        if upload_success:
            hf_url = f"https://huggingface.co/{hf_repo_id}"
    else:
        print_section("ℹ️  No HuggingFace Token Found - Skipping Upload", "-")
        print("To enable auto-upload, set HF_TOKEN environment variable.")
        print("Get your token from: https://huggingface.co/settings/tokens")
    
    # Final summary
    print_section("✅ TRAINING COMPLETE!", "=")
    print(f"\n📊 Final Performance Summary:")
    print(f"  Cross-Validation ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
    print(f"  Final Test ROC-AUC:       {final_metrics['roc_auc']:.4f}")
    print(f"\n⏱️  Total training time: {total_time:.2f} seconds")
    print(f"\n📁 Outputs saved to:")
    print(f"  Model: {model_path}")
    print(f"  Metrics JSON: {metrics_json_path}")
    print(f"  Training Summary JSON: {summary_json_path}")
    print(f"  Fold Details JSON: {fold_details_path}")
    print(f"  Reports: {reports_dir}")
    if hf_url:
        print(f"  HuggingFace: {hf_url}")
    print(f"{'='*70}\n")
    
    return trainer, final_metrics


if __name__ == "__main__":
    trainer, final_metrics = main()
