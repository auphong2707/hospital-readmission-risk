"""
Retrain Phase 2 models and extract actual ROC and PR curve data.
Saves curve points to JSON for plotting.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# Add phase-2 to path to import utilities
project_root = Path(__file__).parent.parent.parent
phase2_path = project_root / "phase-2-risk-modeling"
sys.path.insert(0, str(phase2_path))

from utilities import load_phase1_splits
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_and_extract_curves():
    """Train models and extract ROC/PR curve data."""
    
    print("Loading preprocessed data from HuggingFace...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_phase1_splits()
    
    # Combine train and validation for final training
    X_dev = np.vstack([X_train, X_val])
    y_dev = np.concatenate([y_train, y_val])
    
    print(f"Development set: {X_dev.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Positive class in test: {y_test.sum()} / {len(y_test)} ({y_test.mean():.2%})")
    
    curve_data = {}
    
    # === 1. Gradient Boosting (LightGBM) ===
    print("\n" + "="*60)
    print("Training Gradient Boosting Model...")
    print("="*60)
    
    gb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 200
    }
    
    gb_model = lgb.LGBMClassifier(**gb_params)
    gb_model.fit(X_dev, y_dev)
    
    # Get predictions
    y_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    
    # Calculate curves
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
    precision_gb, recall_gb, _ = precision_recall_curve(y_test, y_proba_gb)
    roc_auc_gb = roc_auc_score(y_test, y_proba_gb)
    pr_auc_gb = average_precision_score(y_test, y_proba_gb)
    
    # Downsample to ~200 points for plotting
    n_points = 200
    if len(fpr_gb) > n_points:
        indices_roc = np.linspace(0, len(fpr_gb) - 1, n_points, dtype=int)
        fpr_gb = fpr_gb[indices_roc]
        tpr_gb = tpr_gb[indices_roc]
    if len(recall_gb) > n_points:
        indices_pr = np.linspace(0, len(recall_gb) - 1, n_points, dtype=int)
        recall_gb = recall_gb[indices_pr]
        precision_gb = precision_gb[indices_pr]
    
    print(f"✅ Gradient Boosting - ROC AUC: {roc_auc_gb:.4f}, PR AUC: {pr_auc_gb:.4f}")
    print(f"   ROC curve points: {len(fpr_gb)}, PR curve points: {len(precision_gb)}")
    
    curve_data['gradient_boosting'] = {
        'roc': {
            'fpr': fpr_gb.tolist(),
            'tpr': tpr_gb.tolist(),
            'auc': float(roc_auc_gb)
        },
        'pr': {
            'recall': recall_gb.tolist(),
            'precision': precision_gb.tolist(),
            'auc': float(pr_auc_gb)
        }
    }
    
    # === 2. Random Forest ===
    print("\n" + "="*60)
    print("Training Random Forest Model...")
    print("="*60)
    
    rf_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_dev, y_dev)
    
    # Get predictions
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate curves
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_proba_rf)
    roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
    pr_auc_rf = average_precision_score(y_test, y_proba_rf)
    
    # Downsample to ~200 points for plotting
    n_points = 200
    if len(fpr_rf) > n_points:
        indices_roc = np.linspace(0, len(fpr_rf) - 1, n_points, dtype=int)
        fpr_rf = fpr_rf[indices_roc]
        tpr_rf = tpr_rf[indices_roc]
    if len(recall_rf) > n_points:
        indices_pr = np.linspace(0, len(recall_rf) - 1, n_points, dtype=int)
        recall_rf = recall_rf[indices_pr]
        precision_rf = precision_rf[indices_pr]
    
    print(f"✅ Random Forest - ROC AUC: {roc_auc_rf:.4f}, PR AUC: {pr_auc_rf:.4f}")
    print(f"   ROC curve points: {len(fpr_rf)}, PR curve points: {len(precision_rf)}")
    
    curve_data['random_forest'] = {
        'roc': {
            'fpr': fpr_rf.tolist(),
            'tpr': tpr_rf.tolist(),
            'auc': float(roc_auc_rf)
        },
        'pr': {
            'recall': recall_rf.tolist(),
            'precision': precision_rf.tolist(),
            'auc': float(pr_auc_rf)
        }
    }
    
    # === 3. Logistic Regression ===
    print("\n" + "="*60)
    print("Training Logistic Regression Model...")
    print("="*60)
    
    lr_params = {
        'C': 0.1,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42,
        'n_jobs': -1
    }
    
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_dev, y_dev)
    
    # Get predictions
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    
    # Calculate curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_proba_lr)
    roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
    pr_auc_lr = average_precision_score(y_test, y_proba_lr)
    
    # Downsample to ~200 points for plotting
    n_points = 200
    if len(fpr_lr) > n_points:
        indices_roc = np.linspace(0, len(fpr_lr) - 1, n_points, dtype=int)
        fpr_lr = fpr_lr[indices_roc]
        tpr_lr = tpr_lr[indices_roc]
    if len(recall_lr) > n_points:
        indices_pr = np.linspace(0, len(recall_lr) - 1, n_points, dtype=int)
        recall_lr = recall_lr[indices_pr]
        precision_lr = precision_lr[indices_pr]
    
    print(f"✅ Logistic Regression - ROC AUC: {roc_auc_lr:.4f}, PR AUC: {pr_auc_lr:.4f}")
    print(f"   ROC curve points: {len(fpr_lr)}, PR curve points: {len(precision_lr)}")
    
    curve_data['logistic_regression'] = {
        'roc': {
            'fpr': fpr_lr.tolist(),
            'tpr': tpr_lr.tolist(),
            'auc': float(roc_auc_lr)
        },
        'pr': {
            'recall': recall_lr.tolist(),
            'precision': precision_lr.tolist(),
            'auc': float(pr_auc_lr)
        }
    }
    
    # Save to JSON
    output_path = Path(__file__).parent / "actual_curve_data.json"
    with open(output_path, 'w') as f:
        json.dump(curve_data, f, indent=2)
    
    print(f"\n✅ Saved curve data to {output_path}")
    print("\nSummary:")
    print(f"  Gradient Boosting: ROC AUC={roc_auc_gb:.4f}, PR AUC={pr_auc_gb:.4f}")
    print(f"  Random Forest:     ROC AUC={roc_auc_rf:.4f}, PR AUC={pr_auc_rf:.4f}")
    print(f"  Logistic Regression: ROC AUC={roc_auc_lr:.4f}, PR AUC={pr_auc_lr:.4f}")
    
    return curve_data

if __name__ == "__main__":
    train_and_extract_curves()
