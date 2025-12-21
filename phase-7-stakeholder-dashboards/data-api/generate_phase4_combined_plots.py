"""
Generate combined Phase 4 cost/benefit plots for all 3 models.
Downloads calibrated models from Phase 3 and creates overlay plots.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "phase-2-risk-modeling"))
sys.path.insert(0, str(project_root / "phase-4-optimal-threshold-ROI-analysis"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utilities import load_phase1_splits
import joblib
import pickle
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')

# Phase 3 repos (calibrated models)
PHASE3_REPOS = {
    "gradient_boosting": {
        "repo": "auphong2707/hospital-readmission-phase3-lgbm-calibrated",
        "model_file": "gradient_boosting_model_original.joblib",
        "calibrator_file": "Gradient_Boosting_(LightGBM)_calibrator.pkl",
        "color": "#1f77b4",  # Blue
        "label": "Gradient Boosting"
    },
    "random_forest": {
        "repo": "auphong2707/hospital-readmission-phase3-rf-calibrated",
        "model_file": "random_forest_model_original.joblib",
        "calibrator_file": "Random_Forest_calibrator.pkl",
        "color": "#ff7f0e",  # Orange
        "label": "Random Forest"
    },
    "logistic_regression": {
        "repo": "auphong2707/hospital-readmission-phase3-lr-calibrated",
        "model_file": "logistic_regression_model_original.joblib",
        "calibrator_file": "Logistic_Regression_calibrator.pkl",
        "color": "#2ca02c",  # Green
        "label": "Logistic Regression"
    }
}

# Cost parameters from Phase 4
COST_PARAMS = {
    "tp_benefit": 10000,   # Benefit per true positive (prevented readmission)
    "fp_cost": 1500,       # Cost per false positive (unnecessary intervention)
    "fn_cost": 25000       # Cost per false negative (missed readmission)
}


def download_model_and_calibrator(method: str):
    """Download model and calibrator from HuggingFace."""
    print(f"\nDownloading {method}...")
    repo = PHASE3_REPOS[method]["repo"]
    model_file = PHASE3_REPOS[method]["model_file"]
    calibrator_file = PHASE3_REPOS[method]["calibrator_file"]
    
    # Download model
    print(f"  - Model: {model_file}")
    model_path = hf_hub_download(
        repo_id=repo, 
        filename=model_file,
        repo_type="model"
    )
    model = joblib.load(model_path)
    
    # Download calibrator
    print(f"  - Calibrator: {calibrator_file}")
    calibrator_path = hf_hub_download(
        repo_id=repo,
        filename=calibrator_file,
        repo_type="model"
    )
    with open(calibrator_path, 'rb') as f:
        calibrator = pickle.load(f)
    
    return model, calibrator


def calculate_costs_benefits(model, calibrator, X_test, y_test, n_thresholds=100):
    """Calculate costs and benefits at various thresholds."""
    # Get calibrated probabilities
    print(f"  Calculating predictions...")
    y_pred_proba_uncalibrated = model.predict_proba(X_test)[:, 1]
    y_pred_proba = calibrator.predict_proba(y_pred_proba_uncalibrated.reshape(-1, 1))
    
    # Calculate costs/benefits at various thresholds
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    costs = []
    benefits = []
    
    print(f"  Calculating costs/benefits for {n_thresholds} thresholds...")
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Cost = false positives * fp_cost + false negatives * fn_cost
        cost = abs(fp * COST_PARAMS['fp_cost'] + fn * COST_PARAMS['fn_cost'])
        
        # Benefit = true positives * tp_benefit
        benefit = tp * COST_PARAMS['tp_benefit']
        
        costs.append(cost)
        benefits.append(benefit)
    
    return thresholds, costs, benefits


def create_combined_plots(all_results):
    """Create two combined plots: costs vs threshold and benefits vs threshold."""
    print("\n" + "="*70)
    print("Creating Combined Plots")
    print("="*70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Costs vs Threshold (all 3 models)
    for method, data in all_results.items():
        ax1.plot(
            data['thresholds'], 
            data['costs'],
            linewidth=2.5,
            color=PHASE3_REPOS[method]['color'],
            label=PHASE3_REPOS[method]['label'],
            marker='o',
            markersize=2,
            alpha=0.8
        )
    
    ax1.set_xlabel('Decision Threshold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Costs ($)', fontsize=14, fontweight='bold')
    ax1.set_title('Total Costs vs Threshold\n(All Models)', fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
    
    # Plot 2: Benefits vs Threshold (all 3 models)
    for method, data in all_results.items():
        ax2.plot(
            data['thresholds'], 
            data['benefits'],
            linewidth=2.5,
            color=PHASE3_REPOS[method]['color'],
            label=PHASE3_REPOS[method]['label'],
            marker='s',
            markersize=2,
            alpha=0.8
        )
    
    ax2.set_xlabel('Decision Threshold', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Benefits ($)', fontsize=14, fontweight='bold')
    ax2.set_title('Total Benefits vs Threshold\n(All Models)', fontsize=16, fontweight='bold')
    ax2.legend(loc='best', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
    
    plt.tight_layout()
    
    # Save combined plot
    output_path = Path(__file__).parent / "phase4_combined_cost_benefit.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved combined plot: {output_path}")
    
    plt.close()
    
    # Also create individual plots for costs and benefits
    create_individual_plots(all_results)


def create_individual_plots(all_results):
    """Create separate high-res plots for costs and benefits."""
    # Costs plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for method, data in all_results.items():
        ax.plot(
            data['thresholds'], 
            data['costs'],
            linewidth=3,
            color=PHASE3_REPOS[method]['color'],
            label=PHASE3_REPOS[method]['label'],
            marker='o',
            markersize=3,
            alpha=0.8
        )
    
    ax.set_xlabel('Decision Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Costs ($)', fontsize=14, fontweight='bold')
    ax.set_title('Total Costs vs Threshold (All Models)', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=13, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
    
    plt.tight_layout()
    costs_path = Path(__file__).parent / "phase4_costs_vs_threshold.png"
    plt.savefig(costs_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved costs plot: {costs_path}")
    plt.close()
    
    # Benefits plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for method, data in all_results.items():
        ax.plot(
            data['thresholds'], 
            data['benefits'],
            linewidth=3,
            color=PHASE3_REPOS[method]['color'],
            label=PHASE3_REPOS[method]['label'],
            marker='s',
            markersize=3,
            alpha=0.8
        )
    
    ax.set_xlabel('Decision Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Benefits ($)', fontsize=14, fontweight='bold')
    ax.set_title('Total Benefits vs Threshold (All Models)', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=13, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
    
    plt.tight_layout()
    benefits_path = Path(__file__).parent / "phase4_benefits_vs_threshold.png"
    plt.savefig(benefits_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved benefits plot: {benefits_path}")
    plt.close()


def main():
    print("="*70)
    print("Phase 4: Combined Cost-Benefit Analysis")
    print("="*70)
    print("\nThis will download calibrated models and generate combined plots.")
    print("Note: Random Forest model is ~304MB, this may take a few minutes.\n")
    
    # Load test data once
    print("Loading Phase 1 test data...")
    _, _, X_test, _, _, y_test = load_phase1_splits(
        repo_id='auphong2707/hospital-readmission-risk-data'
    )
    print(f"✅ Test data loaded: {X_test.shape[0]:,} samples\n")
    
    # Process each model
    all_results = {}
    
    for method in ["gradient_boosting", "random_forest", "logistic_regression"]:
        print(f"\nProcessing {method}...")
        print("-" * 70)
        
        try:
            # Download model and calibrator
            model, calibrator = download_model_and_calibrator(method)
            
            # Calculate costs and benefits
            thresholds, costs, benefits = calculate_costs_benefits(
                model, calibrator, X_test, y_test, n_thresholds=100
            )
            
            all_results[method] = {
                'thresholds': thresholds,
                'costs': costs,
                'benefits': benefits
            }
            
            print(f"✅ Completed {method}")
            print(f"   Cost range: ${min(costs):,.0f} - ${max(costs):,.0f}")
            print(f"   Benefit range: ${min(benefits):,.0f} - ${max(benefits):,.0f}")
        except Exception as e:
            print(f"⚠️  Error processing {method}: {str(e)}")
            print(f"   Skipping {method} and continuing with other models...")
            continue
    
    # Create combined plots
    create_combined_plots(all_results)
    
    print("\n" + "="*70)
    print("✅ All plots generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. phase4_combined_cost_benefit.png (side-by-side)")
    print("  2. phase4_costs_vs_threshold.png (costs only)")
    print("  3. phase4_benefits_vs_threshold.png (benefits only)")
    print()


if __name__ == "__main__":
    main()
