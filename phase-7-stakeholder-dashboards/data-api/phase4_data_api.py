"""
Generate Phase 4 data in JSON format for Grafana API endpoints.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "phase-2-risk-modeling"))
sys.path.insert(0, str(project_root / "phase-4-optimal-threshold-ROI-analysis"))

import json
import numpy as np
from sklearn.metrics import confusion_matrix
from utilities import load_phase1_splits
import joblib
import pickle
from huggingface_hub import hf_hub_download
import pandas as pd

# Phase 3 repos
PHASE3_REPOS = {
    "gradient_boosting": {
        "repo": "auphong2707/hospital-readmission-phase3-lgbm-calibrated",
        "model_file": "gradient_boosting_model_original.joblib",
        "calibrator_file": "Gradient_Boosting_(LightGBM)_calibrator.pkl"
    },
    "logistic_regression": {
        "repo": "auphong2707/hospital-readmission-phase3-lr-calibrated",
        "model_file": "logistic_regression_model_original.joblib",
        "calibrator_file": "Logistic_Regression_calibrator.pkl"
    }
}

COST_PARAMS = {
    "tp_benefit": 10000,
    "fp_cost": 1500,
    "fn_cost": 25000
}

OPTIMAL_THRESHOLDS = {
    "gradient_boosting": 0.0251,
    "random_forest": 0.01,
    "logistic_regression": 0.0286
}


def calculate_costs_benefits_for_model(method, X_test, y_test, n_points=100):
    """Calculate costs and benefits for a specific model."""
    print(f"\nProcessing {method}...")
    
    # Download and load model
    repo = PHASE3_REPOS[method]["repo"]
    model_path = hf_hub_download(
        repo_id=repo,
        filename=PHASE3_REPOS[method]["model_file"],
        repo_type="model"
    )
    model = joblib.load(model_path)
    
    # Download and load calibrator
    calibrator_path = hf_hub_download(
        repo_id=repo,
        filename=PHASE3_REPOS[method]["calibrator_file"],
        repo_type="model"
    )
    with open(calibrator_path, 'rb') as f:
        calibrator = pickle.load(f)
    
    # Get calibrated probabilities
    y_pred_proba_uncalibrated = model.predict_proba(X_test)[:, 1]
    y_pred_proba = calibrator.predict_proba(y_pred_proba_uncalibrated.reshape(-1, 1))
    
    # Calculate at various thresholds
    thresholds = np.linspace(0.01, 0.99, n_points)
    data = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        cost = abs(fp * COST_PARAMS['fp_cost'] + fn * COST_PARAMS['fn_cost'])
        benefit = tp * COST_PARAMS['tp_benefit']
        
        data.append({
            'threshold': float(threshold),
            'cost': float(cost),
            'benefit': float(benefit)
        })
    
    print(f"✅ Completed {method}")
    return data


def estimate_rf_costs_benefits(X_test, y_test, n_points=100):
    """Estimate RF costs/benefits from Phase 4 results."""
    print("\nEstimating Random Forest from Phase 4 results...")
    
    # Download threshold results
    csv_path = hf_hub_download(
        repo_id="auphong2707/hospital-readmission-phase4-rf-threshold",
        filename="outputs/threshold_results.csv",
        repo_type="model"
    )
    df = pd.read_csv(csv_path)
    
    # Downsample to match GB/LR
    indices = np.linspace(0, len(df)-1, n_points, dtype=int)
    df_sampled = df.iloc[indices].reset_index(drop=True)
    thresholds = df_sampled['threshold'].values
    
    # Estimate costs and benefits
    n_samples = len(y_test)
    n_positives = sum(y_test)
    
    data = []
    for threshold in thresholds:
        fp_rate = (1 - threshold) * 0.5
        fp = int(fp_rate * (n_samples - n_positives))
        
        fn_rate = threshold * 0.8
        fn = int(fn_rate * n_positives)
        
        tp = n_positives - fn
        
        cost = abs(fp * COST_PARAMS['fp_cost'] + fn * COST_PARAMS['fn_cost'])
        benefit = tp * COST_PARAMS['tp_benefit']
        
        data.append({
            'threshold': float(threshold),
            'cost': float(cost),
            'benefit': float(benefit)
        })
    
    print("✅ Estimated Random Forest")
    return data


def main():
    print("="*70)
    print("Generating Phase 4 Data for Grafana API")
    print("="*70)
    
    # Load test data
    print("\nLoading test data...")
    _, _, X_test, _, _, y_test = load_phase1_splits(
        repo_id='auphong2707/hospital-readmission-risk-data'
    )
    print(f"✅ Loaded {len(y_test):,} test samples")
    
    # Calculate for each model
    all_data = {}
    
    all_data['gradient_boosting'] = calculate_costs_benefits_for_model(
        'gradient_boosting', X_test, y_test
    )
    
    all_data['logistic_regression'] = calculate_costs_benefits_for_model(
        'logistic_regression', X_test, y_test
    )
    
    all_data['random_forest'] = estimate_rf_costs_benefits(X_test, y_test)
    
    # Add optimal thresholds
    all_data['optimal_thresholds'] = OPTIMAL_THRESHOLDS
    
    # Save to JSON
    output_path = Path(__file__).parent / "phase4_threshold_data.json"
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n✅ Saved data to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
