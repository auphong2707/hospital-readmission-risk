from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
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
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve


# ============================================================================
# ENVIRONMENT AND HARDWARE DETECTION
# ============================================================================

def detect_gpu(verbose: bool = False):
    """Detect if GPU is available for LightGBM.
    
    Args:
        verbose: If True, print detailed error information when GPU is not available
    
    Returns:
        bool: True if GPU is available and functional, False otherwise
    """
    try:
        import lightgbm as lgb
        import warnings
        
        # Suppress warnings during GPU detection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Try to create a simple dataset and train with GPU
            X_test = np.random.rand(10, 5)
            y_test = np.random.randint(0, 2, 10)
            lgb_train = lgb.Dataset(X_test, y_test)
            params = {'device': 'gpu', 'verbose': -1}
            
            # Attempt to train - this will fail if GPU is not properly configured
            lgb.train(params, lgb_train, num_boost_round=1)
            
        if verbose:
            print("   ✅ GPU detected and verified working with LightGBM")
        return True
        
    except Exception as e:
        if verbose:
            error_msg = str(e).lower()
            print(f"   ❌ GPU not available: {type(e).__name__}")
            
            # Provide helpful diagnostics
            if 'cuda' in error_msg or 'opencl' in error_msg:
                print(f"      Reason: {e}")
                print("      💡 Possible fixes:")
                print("         - Install GPU-enabled LightGBM: pip install lightgbm --config-settings=cmake.define.USE_GPU=ON")
                print("         - Check CUDA/OpenCL installation")
                print("         - Verify GPU drivers are up to date")
            elif 'boost' in error_msg.lower():
                print("      Reason: LightGBM may not be compiled with GPU support")
                print("      💡 Install GPU version: pip install lightgbm --config-settings=cmake.define.USE_GPU=ON")
            else:
                print(f"      Reason: {e}")
        
        return False


def is_kaggle_environment():
    """Detect if running in Kaggle environment.
    
    Returns:
        bool: True if in Kaggle environment, False otherwise
    """
    return os.path.exists('/kaggle/working')


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_section(title: str, char: str = "="):
    """Print a formatted section header.
    
    Args:
        title: Section title text
        char: Character to use for the border (default: "=")
    """
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}\n")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_from_huggingface(repo_id: str = "auphong2707/hospital-readmission-risk-data", 
                               split: str = "full"):
    """Load preprocessed data directly from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID (default: auphong2707/hospital-readmission-risk-data)
        split: Which split to load - 'full', 'train', 'validation', or 'test'
               Default 'full' loads the complete preprocessed dataset
    
    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
    
    Raises:
        ImportError: If datasets library is not installed
        ValueError: If split is invalid
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required for HuggingFace loading. "
            "Install with: pip install datasets"
        )
    
    print(f"📥 Loading data from HuggingFace: {repo_id}")
    print(f"   Split: {split}")
    
    # Map split names to file paths
    file_mapping = {
        "full": "hospital_readmission_full.csv",
        "train": "splits/train.csv",
        "validation": "splits/validation.csv",
        "test": "splits/test.csv"
    }
    
    if split not in file_mapping:
        raise ValueError(f"Invalid split: {split}. Choose from: {list(file_mapping.keys())}")
    
    # Load dataset from HuggingFace
    dataset = load_dataset(
        repo_id,
        data_files=file_mapping[split],
        split="train"  # HF datasets always uses 'train' for custom CSV files
    )
    
    # Convert to pandas DataFrame
    df = dataset.to_pandas()
    
    # Separate features and target
    if "target" not in df.columns:
        raise ValueError(f"No 'target' column found in dataset")
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    print(f"✅ Loaded from HuggingFace: features {X.shape}, target {y.shape}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def load_data(data_dir: str = "data/processed", 
              from_huggingface: bool = True,
              hf_repo_id: str = "auphong2707/hospital-readmission-risk-data"):
    """Load features and target data from HuggingFace or local directory.
    
    Args:
        data_dir: Directory containing features.csv and target.csv (used if from_huggingface=False)
        from_huggingface: If True, load from HuggingFace Hub; if False, load from local directory
        hf_repo_id: HuggingFace repository ID (default: auphong2707/hospital-readmission-risk-data)
        
    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
        
    Raises:
        FileNotFoundError: If local processed data files are not found (when from_huggingface=False)
        ImportError: If datasets library is not installed (when from_huggingface=True)
    """
    if from_huggingface:
        return load_data_from_huggingface(repo_id=hf_repo_id, split="full")
    
    # Load from local directory
    print("📂 Loading data from local directory...")
    data_dir = Path(data_dir)
    X_path = data_dir / "features.csv"
    y_path = data_dir / "target.csv"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Processed data not found in {data_dir}. "
            f"Either run phase-1 preprocessing or use from_huggingface=True to load from HuggingFace Hub."
        )

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    # support both columnar and single-column target files
    if "target" in y.columns:
        y = y["target"]
    else:
        y = y.iloc[:, 0]

    print(f"✅ Loaded from local: features {X.shape}, target {y.shape}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    return X, y


def run_preprocessing(preprocess_script: Path) -> None:
    """Run preprocessing script to generate features and target files.
    
    Args:
        preprocess_script: Path to the preprocessing script to execute
    """
    print_section("🔄 Running Preprocessing", "-")
    print(f"📂 Running: {preprocess_script}")
    subprocess.run([sys.executable, str(preprocess_script)], check=True)
    print("✅ Preprocessing completed")


# ============================================================================
# HYPERPARAMETER GRIDS
# ============================================================================

def get_lgbm_param_grid():
    """Get default LightGBM parameter grid for hyperparameter search.
    
    Returns a balanced grid for efficient but thorough search:
    - 3 × 3 × 3 × 2 × 2 × 2 × 2 × 2 = 864 combinations
    - With 5-fold CV = 4,320 model fits
    - Estimated time: 1-2 hours on CPU (depends on data size)
    
    Returns:
        dict: Parameter grid with parameter names as keys and lists of values
    """
    return {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 10],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 1.0],
        "reg_alpha": [0.0, 0.1],      # L1 regularization
        "reg_lambda": [0.0, 0.1],     # L2 regularization
    }


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_comprehensive_metrics(y_true, y_proba, threshold=0.5):
    """
    Calculate comprehensive evaluation metrics for binary classification.
    
    Args:
        y_true: True labels (array-like)
        y_proba: Predicted probabilities for positive class (array-like)
        threshold: Decision threshold (default: 0.5)
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Primary metrics
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    
    # Clinical metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Same as precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    metrics.update({
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    })
    
    # Calibration metric
    try:
        brier = brier_score_loss(y_true, y_proba)
        metrics["brier_score"] = float(brier)
    except:
        metrics["brier_score"] = None
    
    return metrics


def print_metrics_table(metrics: dict, title: str = "Model Performance Metrics"):
    """
    Print comprehensive metrics in organized sections.
    
    Args:
        metrics: Dictionary containing calculated metrics
        title: Title for the metrics table
    """
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Primary Metrics
    print("\n📊 PRIMARY METRICS")
    print(f"{'Metric':<25} {'Value':>12}")
    print("-" * 40)
    primary_metrics = ['roc_auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'f1']
    for metric in primary_metrics:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').upper():<25} {metrics[metric]:>12.4f}")
    
    # Classification Metrics
    print(f"\n📈 CLASSIFICATION METRICS")
    print(f"{'Metric':<25} {'Value':>12}")
    print("-" * 40)
    class_metrics = ['precision', 'recall', 'specificity', 'fpr', 'fnr']
    for metric in class_metrics:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').upper():<25} {metrics[metric]:>12.4f}")
    
    # Clinical Metrics
    print(f"\n🏥 CLINICAL METRICS")
    print(f"{'Metric':<25} {'Value':>12}")
    print("-" * 40)
    clinical_metrics = ['sensitivity', 'specificity', 'ppv', 'npv']
    for metric in clinical_metrics:
        if metric in metrics:
            display_name = {
                'sensitivity': 'Sensitivity (TPR)',
                'specificity': 'Specificity (TNR)',
                'ppv': 'PPV (Precision)',
                'npv': 'NPV'
            }.get(metric, metric.upper())
            print(f"{display_name:<25} {metrics[metric]:>12.4f}")
    
    # Confusion Matrix Components
    print(f"\n🔢 CONFUSION MATRIX")
    print(f"{'Component':<25} {'Count':>12}")
    print("-" * 40)
    cm_metrics = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
    for metric in cm_metrics:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').title():<25} {metrics[metric]:>12}")
    
    # Calibration
    if 'brier_score' in metrics and metrics['brier_score'] is not None:
        print(f"\n📐 CALIBRATION")
        print(f"{'Metric':<25} {'Value':>12}")
        print("-" * 40)
        print(f"{'Brier Score':<25} {metrics['brier_score']:>12.4f}")
    
    print("=" * 80 + "\n")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def save_visualizations(
    y_true, 
    y_proba, 
    y_pred, 
    output_dir: Path, 
    model=None, 
    X=None, 
    feature_names=None
):
    """
    Save comprehensive visualizations including ROC, PR, confusion matrix, 
    calibration, and feature importance.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        y_pred: Predicted class labels
        output_dir: Directory to save plots
        model: Optional trained model (for feature importance)
        X: Optional feature matrix (for feature names)
        feature_names: Optional list of feature names
    """
    print("📊 Generating comprehensive visualizations...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2, color='#2E86AB')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - Hospital Readmission Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ ROC curve saved: {roc_path}")
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {pr_auc:.4f})', linewidth=2, color='#A23B72')
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (PPV)', fontsize=12)
    plt.title('Precision-Recall Curve - Hospital Readmission Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = output_dir / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Precision-Recall curve saved: {pr_path}")
    
    # 3. Confusion Matrix
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
    
    # 4. Calibration Curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10, strategy='quantile'
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label='Model', linewidth=2, markersize=8, color='#F18F01')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1)
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curve - Probability Calibration Assessment', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        cal_path = output_dir / "calibration_curve.png"
        plt.savefig(cal_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Calibration curve saved: {cal_path}")
    except Exception as e:
        print(f"   ⚠️  Could not generate calibration curve: {e}")
    
    # 5. Feature Importance (if model and features provided)
    if model is not None and hasattr(model, 'feature_importances_'):
        try:
            importances = model.feature_importances_
            if feature_names is None and X is not None:
                feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'Feature {i}' for i in range(len(importances))]
            elif feature_names is None:
                feature_names = [f'Feature {i}' for i in range(len(importances))]
            
            # Get top 20 features
            indices = np.argsort(importances)[::-1][:20]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            plt.figure(figsize=(10, 8))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
            plt.barh(range(len(top_features)), top_importances, color=colors)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            fi_path = output_dir / "feature_importance.png"
            plt.savefig(fi_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Feature importance saved: {fi_path}")
            
            # Save feature importance as CSV
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            fi_csv_path = output_dir / "feature_importance.csv"
            fi_df.to_csv(fi_csv_path, index=False)
            print(f"   ✅ Feature importance CSV saved: {fi_csv_path}")
        except Exception as e:
            print(f"   ⚠️  Could not generate feature importance: {e}")


def save_learning_curves(model, X_train, y_train, output_dir: Path, cv=5, scoring='roc_auc'):
    """
    Generate and save learning curves showing model performance vs training size.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        output_dir: Directory to save the plot
        cv: Number of cross-validation folds
        scoring: Scoring metric (default: 'roc_auc')
    """
    print("📈 Generating learning curves...")
    output_dir = Path(output_dir)
    
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, 
            cv=cv, 
            scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
            random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='#2E86AB', label='Training Score', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='#2E86AB')
        plt.plot(train_sizes, val_mean, 'o-', color='#A23B72', label='Cross-Validation Score', linewidth=2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.2, color='#A23B72')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        plt.title('Learning Curves - Model Performance vs Training Size', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        lc_path = output_dir / "learning_curves.png"
        plt.savefig(lc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Learning curves saved: {lc_path}")
    except Exception as e:
        print(f"   ⚠️  Could not generate learning curves: {e}")


def save_validation_curves(search_results, output_dir: Path):
    """
    Generate validation curves from manual hyperparameter search results.
    
    Args:
        search_results: List of dicts containing 'params', 'mean_score', 'std_score', 'fold_scores'
        output_dir: Directory to save the plots
    """
    print("📊 Generating validation curves for hyperparameters...")
    output_dir = Path(output_dir)
    
    try:
        # Extract parameter names from first result
        if not search_results:
            print("   ⚠️  No search results to plot")
            return
            
        param_names = list(search_results[0]['params'].keys())
        
        # Create subplots for each parameter
        n_params = len(param_names)
        n_cols = 2
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.ravel() if n_params > 1 else [axes]
        
        for idx, param_name in enumerate(param_names):
            if idx >= len(axes):
                break
                
            # Collect all unique values for this parameter and their scores
            param_scores = {}
            
            for result in search_results:
                param_val = result['params'][param_name]
                score = result['mean_score']
                std = result['std_score']
                
                if param_val not in param_scores:
                    param_scores[param_val] = {'scores': [], 'stds': []}
                param_scores[param_val]['scores'].append(score)
                param_scores[param_val]['stds'].append(std)
            
            # Sort by parameter value and calculate means
            unique_values = sorted(param_scores.keys())
            grouped_means = [np.mean(param_scores[val]['scores']) for val in unique_values]
            grouped_stds = [np.mean(param_scores[val]['stds']) for val in unique_values]
            
            # Plot
            x_positions = range(len(unique_values))
            axes[idx].plot(x_positions, grouped_means, 'o-', linewidth=2, markersize=8, color='#2E86AB')
            axes[idx].fill_between(
                x_positions,
                [m - s for m, s in zip(grouped_means, grouped_stds)],
                [m + s for m, s in zip(grouped_means, grouped_stds)],
                alpha=0.2,
                color='#2E86AB'
            )
            axes[idx].set_xlabel(param_name.replace('_', ' ').title(), fontsize=11)
            axes[idx].set_ylabel('Mean ROC-AUC', fontsize=11)
            axes[idx].set_title(f'Validation Curve: {param_name}', fontsize=12, fontweight='bold')
            axes[idx].grid(alpha=0.3, linestyle='--')
            axes[idx].set_xticks(x_positions)
            axes[idx].set_xticklabels(unique_values, rotation=45)
        
        # Remove extra subplots
        for idx in range(len(param_names), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        vc_path = output_dir / "validation_curves.png"
        plt.savefig(vc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Validation curves saved: {vc_path}")
        
    except Exception as e:
        print(f"   ⚠️  Could not generate validation curves: {e}")


def save_metrics_comparison(fold_details, output_dir: Path):
    """
    Create visual comparison of metrics across folds.
    
    Args:
        fold_details: List of dicts containing 'fold', 'metrics', etc.
        output_dir: Directory to save the plot
    """
    print("📊 Generating metrics comparison across folds...")
    output_dir = Path(output_dir)
    
    try:
        # Extract metrics from each fold
        metrics_list = []
        for fold in fold_details:
            fold_metrics = fold['metrics'].copy()
            fold_metrics['fold'] = fold['fold']
            metrics_list.append(fold_metrics)
        
        df = pd.DataFrame(metrics_list)
        
        # Select key metrics to plot
        key_metrics = ['roc_auc', 'pr_auc', 'f1', 'balanced_accuracy', 'sensitivity', 'specificity']
        key_metrics = [m for m in key_metrics if m in df.columns]
        
        if key_metrics:
            # Create box plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for idx, metric in enumerate(key_metrics[:6]):
                if idx >= len(axes):
                    break
                    
                values = df[metric].values
                axes[idx].boxplot([values], labels=[''])
                axes[idx].scatter([1] * len(values), values, alpha=0.6, s=100)
                axes[idx].set_ylabel('Score', fontsize=11)
                axes[idx].set_title(f'{metric.replace("_", " ").title()}\nMean: {values.mean():.4f} ± {values.std():.4f}', 
                                   fontsize=12, fontweight='bold')
                axes[idx].grid(alpha=0.3, axis='y')
                axes[idx].set_ylim([0, 1])
            
            plt.tight_layout()
            comp_path = output_dir / "metrics_comparison_across_folds.png"
            plt.savefig(comp_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Metrics comparison saved: {comp_path}")
            
    except Exception as e:
        print(f"   ⚠️  Could not generate metrics comparison: {e}")


# ============================================================================
# HUGGINGFACE HUB INTEGRATION
# ============================================================================

def generate_model_card(summary: dict, model_name: str = "LightGBM") -> str:
    """
    Generate a comprehensive HuggingFace model card with training results.
    
    Args:
        summary: Training summary dictionary with metrics, params, etc.
        model_name: Name of the model (default: "LightGBM")
        
    Returns:
        str: Markdown-formatted model card content
    """
    # Extract key information
    task = summary.get('task', 'Hospital 30-Day Readmission Risk Prediction')
    timestamp = summary.get('timestamp', 'N/A')
    environment = summary.get('environment', 'unknown')
    device = summary.get('device', 'cpu')
    
    # Data info
    data_info = summary.get('data', {})
    total_samples = data_info.get('total_samples', 'N/A')
    n_features = data_info.get('n_features', 'N/A')
    dev_size = data_info.get('development_size', 'N/A')
    test_size = data_info.get('final_test_size', 'N/A')
    
    # Training info
    eval_pipeline = summary.get('evaluation_pipeline', {})
    k_folds = eval_pipeline.get('k_folds', 'N/A')
    
    # Cross-validation results
    cv_results = summary.get('cross_validation', {})
    mean_cv_auc = cv_results.get('mean_roc_auc', 0)
    std_cv_auc = cv_results.get('std_roc_auc', 0)
    
    # Final test metrics
    test_metrics = summary.get('final_test_metrics', {})
    test_auc = test_metrics.get('roc_auc', 0)
    test_pr_auc = test_metrics.get('pr_auc', 0)
    test_f1 = test_metrics.get('f1', 0)
    test_precision = test_metrics.get('precision', 0)
    test_recall = test_metrics.get('recall', 0)
    test_sensitivity = test_metrics.get('sensitivity', 0)
    test_specificity = test_metrics.get('specificity', 0)
    
    # Best parameters
    best_params = summary.get('best_params', {})
    
    # Training time
    total_time = summary.get('total_time_seconds', 0)
    search_time = summary.get('hyperparameter_search_time_seconds', 0)
    
    # Generate model card
    card = f"""---
tags:
- healthcare
- clinical-ml
- diabetes
- readmission-prediction
- lightgbm
- gradient-boosting
library_name: lightgbm
pipeline_tag: tabular-classification
---

# {model_name} - Hospital Readmission Risk Prediction

## Model Description

This {model_name} model predicts the risk of 30-day hospital readmission for diabetic patients. The model was trained on the UCI Diabetes 130-US Hospitals dataset with robust cross-validation and comprehensive evaluation.

**Task:** {task}  
**Model Type:** Gradient Boosting Machine (LightGBM)  
**Training Date:** {timestamp}  
**Environment:** {environment} ({device.upper()})

## Performance Metrics

### Cross-Validation Results ({k_folds}-Fold CV)

| Metric | Value |
|--------|-------|
| Mean ROC-AUC | {mean_cv_auc:.4f} ± {std_cv_auc:.4f} |

### Final Test Set Results

#### Primary Metrics
| Metric | Value |
|--------|-------|
| ROC-AUC | {test_auc:.4f} |
| PR-AUC | {test_pr_auc:.4f} |
| F1 Score | {test_f1:.4f} |

#### Classification Metrics
| Metric | Value |
|--------|-------|
| Precision | {test_precision:.4f} |
| Recall | {test_recall:.4f} |

#### Clinical Metrics
| Metric | Value |
|--------|-------|
| Sensitivity (TPR) | {test_sensitivity:.4f} |
| Specificity (TNR) | {test_specificity:.4f} |

## Model Visualizations

### ROC Curve
![ROC Curve](./roc_curve.png)

### Precision-Recall Curve
![Precision-Recall Curve](./precision_recall_curve.png)

### Confusion Matrix
![Confusion Matrix](./confusion_matrix.png)

### Calibration Curve
![Calibration Curve](./calibration_curve.png)

### Feature Importance
![Feature Importance](./feature_importance.png)

### Learning Curves
![Learning Curves](./learning_curves.png)

### Validation Curves
![Validation Curves](./validation_curves.png)

### Cross-Fold Metrics Comparison
![Metrics Comparison](./metrics_comparison_across_folds.png)

## Dataset Information

| Property | Value |
|----------|-------|
| Total Samples | {total_samples:,} |
| Features | {n_features} |
| Development Set | {dev_size:,} |
| Final Test Set | {test_size:,} |

## Training Configuration

### Evaluation Pipeline
- **Final Holdout Split:** Stratified split into development and test sets
- **Hyperparameter Search:** Grid search with {k_folds}-fold cross-validation
- **Nested Early Stopping:** Inner validation split within each fold
- **Final Evaluation:** Untouched holdout test set

### Best Hyperparameters

```python
{json.dumps(best_params, indent=2)}
```

## Training Details

- **Total Training Time:** {total_time/60:.2f} minutes
- **Hyperparameter Search Time:** {search_time/60:.2f} minutes
- **Cross-Validation Folds:** {k_folds}
- **Early Stopping:** Yes
- **Device:** {device.upper()}

## Usage

### Loading the Model

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('gradient_boosting_model.joblib')

# Load your preprocessed features
X_new = pd.read_csv('your_features.csv')

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

### Feature Requirements

The model expects preprocessed features from the UCI Diabetes 130-US Hospitals dataset. Features include:
- Patient demographics (age, gender, race)
- Admission details (admission type, source, length of stay)
- Medical history (number of diagnoses, procedures)
- Medication information
- Lab results (A1c test results, glucose serum test)
- Previous utilization (outpatient, inpatient, emergency visits)

See `feature_importance.csv` for complete feature list and importance scores.

## Limitations and Biases

- **Domain-Specific:** Model is trained specifically for diabetic patient readmissions
- **Dataset Bias:** Training data from 130 US hospitals (1999-2008) may not generalize to all healthcare settings
- **Class Imbalance:** Dataset may have imbalanced readmission rates
- **Temporal Drift:** Healthcare practices have evolved since data collection
- **Geographic Limitation:** US-based dataset may not apply to other healthcare systems

## Ethical Considerations

This model is intended to assist healthcare providers in identifying patients at risk of readmission. It should:
- **NOT** be used as the sole basis for treatment decisions
- Be validated on your specific patient population before deployment
- Be monitored for fairness across different demographic groups
- Be regularly retrained with recent data to account for changing patterns

## Citation

```bibtex
@misc{{hospital-readmission-lgbm,
  author = {{Your Name}},
  title = {{LightGBM Model for Hospital Readmission Prediction}},
  year = {{2025}},
  url = {{https://huggingface.co/your-repo}}
}}
```

## Dataset Citation

```bibtex
@misc{{strack2014impact,
  title={{Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records}},
  author={{Strack, Beata and DeShazo, Jonathan P and Gennings, Chris and Olmo, Juan L and Ventura, Sebastian and Cios, Krzysztof J and Clore, John N}},
  journal={{BioMed Research International}},
  volume={{2014}},
  year={{2014}},
  publisher={{Hindawi}}
}}
```

## License

This model is released under the MIT License. The underlying dataset has its own license terms.

## Contact

For questions or issues, please open an issue in the repository.

---

**Disclaimer:** This model is for research and educational purposes. Always consult healthcare professionals for medical decisions.
"""
    
    return card


def upload_results_to_hf(
    summary: dict,
    output_dir: str | Path,
    model_name: str = "hospital-readmission-lgbm",
    hf_repo_name: str | None = None,
    hf_token: str | None = None
) -> bool:
    """
    Upload training results, visualizations, and model artifacts to HuggingFace Hub.
    
    Automatically loads credentials from .env file and generates repo name.
    
    Args:
        summary: Training summary dictionary (from training_summary.json)
        output_dir: Directory containing model files and visualizations
        model_name: Name/identifier for the model (default: "hospital-readmission-lgbm")
        hf_repo_name: Optional HuggingFace repo name (auto-generated if None)
        hf_token: Optional HuggingFace API token (loads from .env if None)
    
    Returns:
        bool: True if upload was successful, False otherwise
        
    Example:
        >>> import json
        >>> with open('models/training_summary.json') as f:
        ...     summary = json.load(f)
        >>> upload_results_to_hf(summary=summary, output_dir='models')
    """
    try:
        from huggingface_hub import HfApi, create_repo
        
        # Try to load from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # python-dotenv not installed, will use environment variables directly
        
        # Get token from .env or environment if not provided
        if hf_token is None:
            hf_token = os.getenv('HF_TOKEN')
        
        if not hf_token:
            print("⚠️  HF_TOKEN not found in .env file or environment variables.")
            print("   Skipping upload to HuggingFace Hub.")
            print("   To enable automatic upload:")
            print("   1. Create a .env file in your project root")
            print("   2. Add: HF_TOKEN=your_token_here")
            print("   3. Add: HF_USERNAME=your_username")
            print("   4. Get token from: https://huggingface.co/settings/tokens")
            return False
        
        # Auto-generate repo name if not provided
        if hf_repo_name is None:
            username = os.getenv('HF_USERNAME')
            if not username:
                print("⚠️  HF_USERNAME not found in .env file.")
                print("   Add HF_USERNAME=your_username to .env file")
                return False
            
            # Generate repo name: username/hospital-readmission-lgbm
            hf_repo_name = f"{username}/{model_name}"
        
        print(f"\n{'='*70}")
        print("📤 Uploading Results to HuggingFace Hub")
        print(f"{'='*70}")
        print(f"Repository: {hf_repo_name}")
        print(f"Model: {model_name}")
        
        # Initialize HF API
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            repo_url = create_repo(
                repo_id=hf_repo_name,
                token=hf_token,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            print(f"✅ Repository ready: {hf_repo_name}")
        except Exception as e:
            print(f"⚠️  Could not create/access repository: {e}")
            return False
        
        output_dir = Path(output_dir)
        
        # Generate and save model card
        print("📝 Generating model card...")
        readme_content = generate_model_card(summary, model_name)
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"   ✅ Model card saved: {readme_path}")
        
        # List files to upload
        print("\n📂 Files to upload:")
        files_to_upload = []
        for file_path in output_dir.glob('*'):
            if file_path.is_file():
                files_to_upload.append(file_path.name)
                print(f"   - {file_path.name}")
        
        if not files_to_upload:
            print("⚠️  No files found in output directory!")
            return False
        
        # Upload all files from output directory
        print("\n⏳ Uploading files to HuggingFace Hub...")
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=hf_repo_name,
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload {model_name} training results and visualizations"
        )
        
        print(f"\n{'='*70}")
        print("✅ Results uploaded successfully!")
        print(f"{'='*70}")
        print(f"🌐 View at: https://huggingface.co/{hf_repo_name}")
        print(f"{'='*70}\n")
        
        return True
        
    except ImportError:
        print("⚠️  huggingface_hub not installed.")
        print("   Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"⚠️  Error uploading to HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return False

