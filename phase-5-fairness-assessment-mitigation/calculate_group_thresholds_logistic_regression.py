"""
Phase 6: Calculate Group-Specific Thresholds for Fairness Mitigation - Logistic Regression

This script implements post-hoc fairness mitigation using the equalized odds
strategy by calculating optimal thresholds for each demographic group to
equalize both True Positive Rate (TPR) and False Positive Rate (FPR).

Strategy: Equalized Odds
    - Aims to equalize TPR and FPR across all demographic groups
    - Finds group-specific thresholds that minimize gaps from target metrics
    - Balances both error rates to ensure fair treatment

Usage:
    # Basic usage
    python calculate_group_thresholds_logistic_regression.py
    
    # With custom threshold search configuration
    python calculate_group_thresholds_logistic_regression.py \
        --threshold-min 0.01 --threshold-max 0.99 --num-thresholds 50000

Inputs:
    - Phase 5 fairness report (fairness_report.json)
    - Test data and demographics
    - Calibrated Logistic Regression model, scaler, and calibrator
    
Outputs:
    - group_thresholds.json (group-specific thresholds)
    - mitigation_impact.json (before/after comparison)
    - Visualizations (before/after charts)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from dotenv import load_dotenv
import json
import pickle

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities import (
    load_test_data_and_demographics,
    ThresholdOptimizer,
    MitigationEvaluator,
    TradeoffAnalyzer,
    MitigationVisualizer,
    save_results,
    upload_results_to_hf,
    ModelCalibrator
)

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate group-specific thresholds for fairness mitigation - Logistic Regression"
    )
    
    # Phase 5 inputs
    parser.add_argument(
        '--phase5-report',
        type=str,
        default='./phase-5-fairness-evaluation/outputs/logistic_regression/fairness_report.json',
        help='Path to Phase 5 fairness report file'
    )
    
    # Fairness configuration
    parser.add_argument(
        '--fairness-tolerance',
        type=float,
        default=0.05,
        help='Target fairness gap tolerance (default: 0.05 = 5%%)'
    )
    
    # Threshold search configuration
    parser.add_argument(
        '--threshold-min',
        type=float,
        default=0.01,
        help='Minimum threshold to test (default: 0.01)'
    )
    parser.add_argument(
        '--threshold-max',
        type=float,
        default=0.99,
        help='Maximum threshold to test (default: 0.99)'
    )
    parser.add_argument(
        '--num-thresholds',
        type=int,
        default=50000,
        help='Number of thresholds to test (default: 50000)'
    )
    
    # Data sources
    parser.add_argument(
        '--data-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-risk-data',
        help='HuggingFace data repository ID'
    )
    
    parser.add_argument(
        '--model-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-logistic-regression-calibrated',
        help='HuggingFace model repository ID'
    )
    
    parser.add_argument(
        '--use-local',
        action='store_true',
        help='Use local files instead of HuggingFace Hub'
    )
    
    parser.add_argument(
        '--local-test',
        type=str,
        default='./data/processed/splits/test.csv',
        help='Path to local test.csv'
    )
    
    parser.add_argument(
        '--local-demographics',
        type=str,
        default='./data/processed/splits/test_demographics.csv',
        help='Path to local test_demographics.csv'
    )
    
    parser.add_argument(
        '--local-model',
        type=str,
        default=None,
        help='Path to local model file'
    )
    
    parser.add_argument(
        '--local-scaler',
        type=str,
        default=None,
        help='Path to local scaler file'
    )
    
    parser.add_argument(
        '--local-calibrator',
        type=str,
        default=None,
        help='Path to local calibrator file'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./phase-6-fairness-mitigation-bias-correction/outputs/logistic_regression',
        help='Directory to save mitigation outputs'
    )
    
    # Cache options
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./data/downloaded',
        help='Directory to cache downloaded files'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download from HuggingFace Hub'
    )
    
    return parser.parse_args()


def load_phase5_fairness_report(report_path):
    """
    Load Phase 5 fairness report.
    
    Args:
        report_path: Path to fairness_report.json
        
    Returns:
        dict: Phase 5 results
    """
    print(f"📂 Loading Phase 5 fairness report from: {report_path}")
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print(f"✅ Loaded Phase 5 report")
    print(f"   Optimal threshold: {report['optimal_threshold']:.4f}")
    if 'deployment_readiness' in report:
        print(f"   Fairness assessment: {report['deployment_readiness']['fairness_assessment']}")
    
    return report


def load_lr_model_for_mitigation(args):
    """
    Load Logistic Regression model, scaler, and calibrator.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (model, scaler, calibrator)
    """
    print("📥 Loading Logistic Regression model...")
    
    if args.use_local:
        # Load from local files
        if args.local_model:
            model_path = Path(args.local_model)
        else:
            model_path = Path("./calibration_outputs/logistic_regression/logistic_regression_model_original.pkl")
        
        if args.local_scaler:
            scaler_path = Path(args.local_scaler)
        else:
            scaler_path = Path("./calibration_outputs/logistic_regression/logistic_regression_scaler.pkl")
            
        if args.local_calibrator:
            calibrator_path = Path(args.local_calibrator)
        else:
            calibrator_path = Path("./calibration_outputs/logistic_regression/Logistic_Regression_calibrator.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        calibrator = ModelCalibrator.load(str(calibrator_path))
        
        print(f"✅ Loaded from local files")
        
    else:
        # Download from HuggingFace Hub
        from huggingface_hub import hf_hub_download
        
        model_path = hf_hub_download(
            repo_id=args.model_repo_id,
            filename="logistic_regression_model_original.pkl",
            cache_dir=args.cache_dir,
            force_download=args.force_download
        )
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        scaler_path = hf_hub_download(
            repo_id=args.model_repo_id,
            filename="logistic_regression_scaler.pkl",
            cache_dir=args.cache_dir,
            force_download=args.force_download
        )
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        calibrator_path = hf_hub_download(
            repo_id=args.model_repo_id,
            filename="Logistic_Regression_calibrator.pkl",
            cache_dir=args.cache_dir,
            force_download=args.force_download
        )
        calibrator = ModelCalibrator.load(calibrator_path)
        
        print(f"✅ Downloaded from HuggingFace Hub")
    
    return model, scaler, calibrator


def generate_lr_predictions(model, scaler, calibrator, X_test):
    """
    Generate calibrated predictions for Logistic Regression.
    
    Args:
        model: Trained Logistic Regression model
        scaler: StandardScaler
        calibrator: Calibrator
        X_test: Test features
        
    Returns:
        np.ndarray: Calibrated probabilities
    """
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Get uncalibrated probabilities
    y_pred_proba_uncalibrated = model.predict_proba(X_test_scaled)[:, 1]
    
    # Apply calibration
    try:
        pred = calibrator.predict_proba(y_pred_proba_uncalibrated.reshape(-1, 1))
    except Exception:
        pred = calibrator.predict_proba(y_pred_proba_uncalibrated)
    
    # Normalize to 1D
    if isinstance(pred, np.ndarray):
        if pred.ndim == 2 and pred.shape[1] >= 2:
            y_pred_proba = pred[:, 1]
        else:
            y_pred_proba = pred.ravel()
    else:
        y_pred_proba = pred
    
    return y_pred_proba


def main():
    """Main fairness mitigation workflow."""
    
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("Phase 6: Fairness Mitigation - Logistic Regression".center(80))
    print("="*80 + "\n")
    
    print(f"📋 Configuration:")
    print(f"   Phase 5 report: {args.phase5_report}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Fairness tolerance: {args.fairness_tolerance:.1%}")
    print(f"   Threshold search: {args.num_thresholds} candidates [{args.threshold_min}, {args.threshold_max}]")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # STEP 1: Load Phase 5 results
        print("\n" + "="*80)
        print("Step 1: Load Phase 5 Fairness Report")
        print("="*80)
        phase5_report = load_phase5_fairness_report(args.phase5_report)
        optimal_threshold = phase5_report['optimal_threshold']
        global_threshold = optimal_threshold  # Use as global threshold for mitigation
        
        # STEP 2: Load test data and demographics
        print("\n" + "="*80)
        print("Step 2: Load Test Data and Demographics")
        print("="*80)
        X_test, y_test, demographics = load_test_data_and_demographics(
            use_local=args.use_local,
            local_test_path=args.local_test,
            local_demographics_path=args.local_demographics,
            data_repo_id=args.data_repo_id,
            cache_dir=args.cache_dir
        )
        
        # STEP 3: Load model
        print("\n" + "="*80)
        print("Step 3: Load Calibrated Model")
        print("="*80)
        model, scaler, calibrator = load_lr_model_for_mitigation(args)
        
        # STEP 4: Generate predictions
        print("\n" + "="*80)
        print("Step 4: Generate Calibrated Predictions")
        print("="*80)
        y_pred_proba = generate_lr_predictions(model, scaler, calibrator, X_test)
        print(f"✅ Generated {len(y_pred_proba)} calibrated predictions")
        
        # STEP 4b: Evaluate Baseline (Global Threshold)
        print("\n" + "="*80)
        print("Step 4b: Evaluate Baseline with Global Threshold")
        print("="*80)
        
        phase4_results = phase5_report.get('phase4_results', None)
        evaluator = MitigationEvaluator(phase4_results=phase4_results)
        
        baseline = evaluator.evaluate_baseline(
            y_true=y_test.values,
            y_pred_proba=y_pred_proba,
            demographics=demographics,
            global_threshold=global_threshold
        )
        
        # STEP 5: Calculate group-specific thresholds
        print("\n" + "="*80)
        print("Step 5: Calculate Group-Specific Thresholds (Equalized Odds)")
        print("="*80)
        
        # Calculate step size from num_thresholds
        threshold_step = (args.threshold_max - args.threshold_min) / (args.num_thresholds - 1)
        
        optimizer = ThresholdOptimizer(
            fairness_tolerance=args.fairness_tolerance,
            threshold_range=(args.threshold_min, args.threshold_max),
            threshold_step=threshold_step
        )
        
        print(f"\n🔍 Threshold Search Configuration:")
        print(f"   Range: [{args.threshold_min:.2f}, {args.threshold_max:.2f}]")
        print(f"   Number of thresholds: {args.num_thresholds:,}")
        print(f"   Step size: {threshold_step:.6f}")
        
        # Calculate overall metrics for target
        y_pred_global = (y_pred_proba >= global_threshold).astype(int)
        tn, fp, fn, tp = np.bincount(y_test.values * 2 + y_pred_global, minlength=4)
        
        overall_metrics = {
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'intervention_rate': np.mean(y_pred_global)
        }
        
        group_thresholds = {}
        for attribute in ['race', 'gender', 'age_group']:
            if attribute not in demographics.columns:
                print(f"\n⚠️  Skipping {attribute}: column not found in demographics")
                continue
            
            thresholds = optimizer.calculate_group_thresholds(
                y_true=y_test.values,
                y_pred_proba=y_pred_proba,
                demographics=demographics,
                attribute=attribute,
                overall_metrics=overall_metrics,
                global_threshold=global_threshold
            )
            
            group_thresholds[attribute] = thresholds
        
        # Save group thresholds
        thresholds_output = {
            'mitigation_strategy': 'equalized_odds',
            'fairness_tolerance': args.fairness_tolerance,
            'global_threshold': global_threshold,
            'group_specific_thresholds': group_thresholds,
            'threshold_search_config': {
                'min': args.threshold_min,
                'max': args.threshold_max,
                'num_thresholds': args.num_thresholds,
                'step': threshold_step
            },
            'target_metrics': {
                'tpr': overall_metrics['tpr'],
                'fpr': overall_metrics['fpr'],
                'intervention_rate': overall_metrics['intervention_rate']
            }
        }
        
        thresholds_path = output_dir / "group_thresholds.json"
        save_results(thresholds_output, str(thresholds_path))
        
        # STEP 6: Evaluate Mitigated (Group-Specific Thresholds)
        print("\n" + "="*80)
        print("Step 6: Evaluate with Group-Specific Thresholds")
        print("="*80)
        
        mitigated = evaluator.evaluate_mitigated(
            y_true=y_test.values,
            y_pred_proba=y_pred_proba,
            demographics=demographics,
            group_thresholds=group_thresholds
        )
        
        # STEP 7: Analyze Trade-offs
        print("\n" + "="*80)
        print("Step 7: Analyze Performance/Fairness/ROI Trade-offs")
        print("="*80)
        
        improvements = TradeoffAnalyzer.calculate_improvements(baseline, mitigated)
        
        # Save mitigation impact
        mitigation_impact = {
            'phase': 6,
            'mitigation_strategy': 'equalized_odds',
            'baseline_metrics': baseline,
            'mitigated_metrics': mitigated,
            'improvements': improvements,
            'summary': {
                'fairness_targets_met': improvements['summary']['fairness_targets_met'],
                'performance_drop_acceptable': improvements['summary']['performance_drop_acceptable'],
                'roi_reduction_acceptable': improvements['summary']['roi_reduction_acceptable'],
                'recommended_for_deployment': (
                    improvements['summary']['fairness_targets_met'] and
                    improvements['summary']['performance_drop_acceptable'] and
                    improvements['summary']['roi_reduction_acceptable']
                )
            }
        }
        
        impact_path = output_dir / "mitigation_impact.json"
        save_results(mitigation_impact, str(impact_path))
        
        # STEP 8: Generate Visualizations
        print("\n" + "="*80)
        print("Step 8: Generate Visualizations")
        print("="*80)
        
        MitigationVisualizer.generate_all_visualizations(
            baseline=baseline,
            mitigated=mitigated,
            improvements=improvements,
            output_dir=str(output_dir)
        )
        
        # STEP 9: Upload to HuggingFace Hub
        print("\n" + "="*80)
        print("Step 9: Upload to HuggingFace Hub")
        print("="*80)
        
        upload_results_to_hf(
            output_dir=str(output_dir),
            repo_id=args.model_repo_id,
            commit_message="Upload Phase 6 Logistic Regression fairness mitigation results"
        )
        
        # FINAL SUMMARY
        print("\n" + "="*80)
        print("✨ Fairness Mitigation Complete!")
        print("="*80 + "\n")
        
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"\n📊 Key Results:")
        print(f"   Mitigation strategy: Equalized Odds")
        print(f"   Attributes mitigated: {', '.join(group_thresholds.keys())}")
        print(f"   Fairness targets met: {mitigation_impact['summary']['fairness_targets_met']}")
        print(f"   Performance drop acceptable: {mitigation_impact['summary']['performance_drop_acceptable']}")
        print(f"   ROI reduction acceptable: {mitigation_impact['summary']['roi_reduction_acceptable']}")
        print(f"   Recommended for deployment: {mitigation_impact['summary']['recommended_for_deployment']}")
        
        print(f"\n📄 Generated Files:")
        print(f"   - group_thresholds.json")
        print(f"   - mitigation_impact.json")
        print(f"   - visualizations/ (before/after charts)")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review mitigation impact: {impact_path}")
        print(f"   2. Review visualizations in: {output_dir}/visualizations/")
        print(f"   3. Proceed to Phase 7: Results Collection & Publication")
        
        print(f"\n{'='*80}")
        print("🎉 Fairness mitigation complete!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error during fairness mitigation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
