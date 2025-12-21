"""
Recalculate costs and benefits data from Phase 4 for all models.
Uses calibrated models from Phase 3 and calculates costs/benefits at each threshold.
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

# Phase 3 repos (calibrated models)
PHASE3_REPOS = {
    "gradient_boosting": {
        "repo": "auphong2707/hospital-readmission-phase3-lgbm-calibrated",
        "model_file": "gradient_boosting_model_original.joblib",
        "calibrator_file": "Gradient_Boosting_(LightGBM)_calibrator.pkl"
    },
    "random_forest": {
        "repo": "auphong2707/hospital-readmission-phase3-rf-calibrated",
        "model_file": "random_forest_model_original.joblib",
        "calibrator_file": "Random_Forest_calibrator.pkl"
    },
    "logistic_regression": {
        "repo": "auphong2707/hospital-readmission-phase3-lr-calibrated",
        "model_file": "logistic_regression_model_original.joblib",
        "calibrator_file": "Logistic_Regression_calibrator.pkl"
    }
}

# Cost parameters (same as Phase 4)
COST_PARAMS = {
    'tp_benefit': 10000,  # Benefit of correctly identifying high-risk patient
    'fp_cost': 1500,      # Cost of unnecessary intervention
    'fn_cost': 25000      # Cost of missed readmission
}

def calculate_costs_benefits(method, n_thresholds=100):
    """Calculate costs and benefits across thresholds for a model."""
    print(f"\nProcessing {method}...")
    
    # Download model and calibrator
    repo = PHASE3_REPOS[method]["repo"]
    try:
        model_path = hf_hub_download(repo_id=repo, filename=PHASE3_REPOS[method]["model_file"], repo_type="model")
        calibrator_path = hf_hub_download(repo_id=repo, filename=PHASE3_REPOS[method]["calibrator_file"], repo_type="model")
    except Exception as e:
        print(f"  ❌ Error downloading {method}: {e}")
        return None
    
    # Load model and calibrator
    model = joblib.load(model_path)
    with open(calibrator_path, 'rb') as f:
        calibrator = pickle.load(f)
    
    # Load test data
    print("  Loading test data...")
    _, _, X_test, _, _, y_test = load_phase1_splits(
        repo_id='auphong2707/hospital-readmission-risk-data'
    )
    
    # Get calibrated probabilities
    print(f"  Calculating predictions...")
    y_pred_proba_uncalibrated = model.predict_proba(X_test)[:, 1]
    y_pred_proba = calibrator.predict_proba(y_pred_proba_uncalibrated.reshape(-1, 1))
    
    # Calculate costs/benefits at various thresholds
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    costs_data = []
    benefits_data = []
    
    print(f"  Calculating costs/benefits for {n_thresholds} thresholds...")
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate costs and benefits
        cost = abs(fp * COST_PARAMS['fp_cost'] + fn * COST_PARAMS['fn_cost'])
        benefit = tp * COST_PARAMS['tp_benefit']
        
        costs_data.append(float(cost))
        benefits_data.append(float(benefit))
    
    print(f"  ✅ Calculated {len(thresholds)} data points")
    
    return {
        'thresholds': thresholds.tolist(),
        'costs': costs_data,
        'benefits': benefits_data
    }

def main():
    """Extract costs/benefits data for all models."""
    print("="*70)
    print("Calculating Phase 4 Costs & Benefits Data")
    print("="*70)
    
    model_labels = {
        "gradient_boosting": "Gradient Boosting",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression"
    }
    
    all_data = {
        'costs': [],
        'benefits': []
    }
    
    for method, label in model_labels.items():
        result = calculate_costs_benefits(method, n_thresholds=100)
        
        if result:
            # Add to costs data (for line plot)
            for threshold, cost in zip(result['thresholds'], result['costs']):
                all_data['costs'].append({
                    'threshold': threshold,
                    'model': label,
                    'value': cost
                })
            
            # Add to benefits data (for line plot)
            for threshold, benefit in zip(result['thresholds'], result['benefits']):
                all_data['benefits'].append({
                    'threshold': threshold,
                    'model': label,
                    'value': benefit
                })
    
    # Save to JSON
    output_path = Path(__file__).parent / "phase4_threshold_data.json"
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Saved threshold data to {output_path}")
    print(f"   Total costs data points: {len(all_data['costs'])}")
    print(f"   Total benefits data points: {len(all_data['benefits'])}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
