from __future__ import annotations

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

def detect_gpu():
    """Detect if GPU is available for LightGBM.
    
    Returns:
        bool: True if GPU is available and functional, False otherwise
    """
    try:
        import lightgbm as lgb
        # Try to create a simple dataset and train with GPU
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 2, 10)
        lgb_train = lgb.Dataset(X_test, y_test)
        params = {'device': 'gpu', 'verbose': -1}
        lgb.train(params, lgb_train, num_boost_round=1, verbose_eval=False)
        return True
    except Exception:
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

def load_data(data_dir: str = "data/processed"):
    """Load features and target data from processed directory.
    
    Args:
        data_dir: Directory containing features.csv and target.csv
        
    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
        
    Raises:
        FileNotFoundError: If processed data files are not found
    """
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
    
    Returns a balanced grid for thorough but practical search:
    - 4 × 4 × 3 × 3 × 3 × 3 × 2 × 2 = 2,592 combinations
    - With 5-fold CV = 12,960 model fits
    - Estimated time: 2-4 hours on CPU (depends on data size)
    
    Returns:
        dict: Parameter grid with parameter names as keys and lists of values
    """
    return {
        "n_estimators": [100, 200, 300, 400],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 6, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1],      # L1 regularization
        "reg_lambda": [0.0, 0.1],     # L2 regularization
    }


# ============================================================================
# MODEL TRAINING UTILITIES
# ============================================================================

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

