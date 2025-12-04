"""
Phase 6: Calculate Group-Specific Thresholds for Fairness Mitigation

This script implements post-hoc fairness mitigation using the equalized odds
strategy by calculating optimal thresholds for each demographic group to
equalize both True Positive Rate (TPR) and False Positive Rate (FPR).

Strategy: Equalized Odds
    - Aims to equalize TPR and FPR across all demographic groups
    - Finds group-specific thresholds that minimize gaps from target metrics
    - Balances both error rates to ensure fair treatment

Usage:
    # Basic usage
    python calculate_group_thresholds_gradient_boosting.py
    
    # With custom threshold search configuration
    python calculate_group_thresholds_gradient_boosting.py \
        --threshold-min 0.01 --threshold-max 0.99 --num-thresholds 10000

Inputs:
    - Phase 5 summary (phase5_summary_for_phase6.json)
    - Test data and demographics
    - Calibrated model and calibrator
    
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

import numpy as np
import pandas as pd
import joblib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities import (
    load_phase5_summary,
    load_test_data_and_demographics,
    load_model_and_calibrator,
    ThresholdOptimizer,
    MitigationEvaluator,
    TradeoffAnalyzer,
    MitigationVisualizer,
    save_results,
    upload_results_to_hf
)

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate group-specific thresholds for fairness mitigation"
    )
    
    # Phase 5 inputs
    parser.add_argument(
        '--phase5-summary',
        type=str,
        default='./phase-5-fairness-evaluation/outputs/phase5_summary_for_phase6.json',
        help='Path to Phase 5 summary file'
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
        default=10000,
        help='Number of thresholds to test (default: 10000)'
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
        default='auphong2707/hospital-readmission-lgbm-calibrated',
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
        '--local-calibrator',
        type=str,
        default=None,
        help='Path to local calibrator file'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./phase-6-fairness-mitigation/outputs',
        help='Output directory'
    )
    
    return parser.parse_args()


def print_section(title: str, char: str = "="):
    """Print formatted section header."""
    print("\n" + char*80)
    print(f"{title:^80}")
    print(char*80)


def generate_calibrated_predictions(
    model,
    calibrator,
    X_test: pd.DataFrame
) -> np.ndarray:
    """Generate calibrated probability predictions."""
    print("🔮 Generating calibrated predictions...")
    
    # Get uncalibrated probabilities
    y_pred_proba_uncalibrated = model.predict_proba(X_test)[:, 1]
    
    # Apply calibration (handle different calibrator types)
    try:
        pred = calibrator.predict_proba(y_pred_proba_uncalibrated.reshape(-1, 1))
    except Exception:
        pred = calibrator.predict_proba(y_pred_proba_uncalibrated)
    
    # Normalize to 1D
    if isinstance(pred, np.ndarray):
        if pred.ndim == 2 and pred.shape[1] >= 2:
            y_pred_proba_calibrated = pred[:, 1]
        else:
            y_pred_proba_calibrated = pred.ravel()
    else:
        y_pred_proba_calibrated = np.asarray(pred).ravel()
    
    print(f"✅ Generated {len(y_pred_proba_calibrated)} calibrated predictions")
    print(f"   Mean probability: {y_pred_proba_calibrated.mean():.3f}")
    print(f"   Probability range: [{y_pred_proba_calibrated.min():.3f}, {y_pred_proba_calibrated.max():.3f}]")
    
    return y_pred_proba_calibrated


def main():
    """Main execution pipeline."""
    
    print("\n" + "="*80)
    print("PHASE 6: FAIRNESS MITIGATION & BIAS CORRECTION")
    print("Hospital Readmission Risk Prediction - Gradient Boosting Model")
    print("="*80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load Phase 5 Decision Inputs
    # ========================================================================
    
    print_section("Step 1: Load Phase 5 Decision Inputs", "-")
    
    phase5_summary = load_phase5_summary(args.phase5_summary)
    
    # Check if mitigation is required
    if not phase5_summary['requires_mitigation']:
        print("\n" + "="*80)
        print("✅ NO MITIGATION REQUIRED")
        print("="*80)
        print("\nPhase 5 detected no significant fairness violations.")
        print("You can skip Phase 6 and proceed directly to Phase 7 with the global threshold.")
        print(f"\nGlobal threshold: {phase5_summary['optimal_threshold']:.4f}")
        print("\nRecommendations:")
        print("   1. Skip Phase 6 (no bias correction needed)")
        print("   2. Proceed to Phase 7 (Deployment Preparation)")
        print("   3. Use global optimal threshold from Phase 4")
        print("   4. Document 'no fairness violations' in model card")
        print("\n" + "="*80)
        return
    
    # Show mitigation priority
    priority = phase5_summary['mitigation_priority']
    print(f"\n⚠️  MITIGATION REQUIRED")
    print(f"   Priority: {priority.upper()}")
    print(f"   Strategy: equalized_odds")
    
    if priority == 'high':
        print("   ⚠️  HIGH PRIORITY: Significant fairness violations detected")
    elif priority == 'medium':
        print("   ⚠️  MEDIUM PRIORITY: Moderate fairness violations detected")
    else:
        print("   ℹ️  LOW PRIORITY: Minor fairness violations detected")
    
    # ========================================================================
    # STEP 2: Load Data and Model
    # ========================================================================
    
    print_section("Step 2: Load Test Data, Demographics, and Model", "-")
    
    X_test, y_test, demographics = load_test_data_and_demographics(
        data_repo_id=args.data_repo_id,
        use_local=args.use_local,
        local_test_path=args.local_test,
        local_demographics_path=args.local_demographics
    )
    
    model, calibrator = load_model_and_calibrator(
        model_repo_id=args.model_repo_id,
        use_local=args.use_local,
        local_model_path=args.local_model,
        local_calibrator_path=args.local_calibrator
    )
    
    # ========================================================================
    # STEP 3: Generate Calibrated Predictions
    # ========================================================================
    
    print_section("Step 3: Generate Calibrated Predictions", "-")
    
    y_pred_proba = generate_calibrated_predictions(model, calibrator, X_test)
    
    # ========================================================================
    # STEP 4: Evaluate Baseline (Global Threshold)
    # ========================================================================
    
    print_section("Step 4: Evaluate Baseline with Global Threshold", "-")
    
    global_threshold = phase5_summary['optimal_threshold']
    phase4_results = phase5_summary.get('phase4_results', None)
    
    evaluator = MitigationEvaluator(phase4_results=phase4_results)
    
    baseline = evaluator.evaluate_baseline(
        y_true=y_test.values,
        y_pred_proba=y_pred_proba,
        demographics=demographics,
        global_threshold=global_threshold
    )
    
    # ========================================================================
    # STEP 5: Calculate Group-Specific Thresholds
    # ========================================================================
    
    print_section("Step 5: Calculate Group-Specific Thresholds (Equalized Odds)", "=")
    
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
    
    group_thresholds = {}
    
    # Calculate overall metrics for target
    y_pred_global = (y_pred_proba >= global_threshold).astype(int)
    tn, fp, fn, tp = np.bincount(y_test.values * 2 + y_pred_global, minlength=4)
    
    overall_metrics = {
        'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'intervention_rate': np.mean(y_pred_global)
    }
    
    # Optimize for each demographic attribute
    for attribute in ['race', 'gender', 'age']:
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
    
    # ========================================================================
    # STEP 6: Evaluate Mitigated (Group-Specific Thresholds)
    # ========================================================================
    
    print_section("Step 6: Evaluate with Group-Specific Thresholds", "-")
    
    mitigated = evaluator.evaluate_mitigated(
        y_true=y_test.values,
        y_pred_proba=y_pred_proba,
        demographics=demographics,
        group_thresholds=group_thresholds
    )
    
    # ========================================================================
    # STEP 7: Analyze Trade-offs
    # ========================================================================
    
    print_section("Step 7: Analyze Performance/Fairness/ROI Trade-offs", "-")
    
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
    
    # ========================================================================
    # STEP 8: Generate Visualizations
    # ========================================================================
    
    print_section("Step 8: Generate Visualizations", "-")
    
    MitigationVisualizer.generate_all_visualizations(
        baseline=baseline,
        mitigated=mitigated,
        improvements=improvements,
        output_dir=str(output_dir)
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print_section("✅ Phase 6 Complete: Fairness Mitigation Summary", "=")
    
    print("\n📊 Key Findings:")
    print(f"   Mitigation strategy: equalized_odds")
    print(f"   Fairness targets met: {'YES ✅' if improvements['summary']['fairness_targets_met'] else 'NO ❌'}")
    print(f"   Performance drop acceptable: {'YES ✅' if improvements['summary']['performance_drop_acceptable'] else 'NO ❌'}")
    print(f"   ROI reduction acceptable: {'YES ✅' if improvements['summary']['roi_reduction_acceptable'] else 'NO ❌'}")
    
    print(f"\n📁 Outputs:")
    print(f"   Results: {args.output_dir}")
    print(f"   Group thresholds: {thresholds_path}")
    print(f"   Mitigation impact: {impact_path}")
    print(f"   Visualizations: {output_dir / 'visualizations'}/*.png")
    
    print(f"\n💡 Recommendations:")
    if mitigation_impact['summary']['recommended_for_deployment']:
        print(f"   ✅ Group-specific thresholds RECOMMENDED for deployment")
        print(f"   1. Review mitigation impact report and visualizations")
        print(f"   2. Present to clinical and ethics teams for approval")
        print(f"   3. Proceed to Phase 7 with group-specific thresholds")
        print(f"   4. Document fairness mitigation in model card")
    else:
        reasons = []
        if not improvements['summary']['fairness_targets_met']:
            reasons.append("Fairness targets not met")
        if not improvements['summary']['performance_drop_acceptable']:
            reasons.append("Performance drop too large")
        if not improvements['summary']['roi_reduction_acceptable']:
            reasons.append("ROI reduction too large")
        
        print(f"   ⚠️  Group-specific thresholds NOT RECOMMENDED")
        print(f"   Reasons: {', '.join(reasons)}")
        print(f"   1. Review mitigation impact and identify issues")
        print(f"   2. Consider adjusting threshold search parameters:")
        print(f"      - Widen search range (--threshold-min, --threshold-max)")
        print(f"      - Increase fairness tolerance (--fairness-tolerance)")
        print(f"   3. If no configuration works, consider Phase 1-3 retraining with:")
        print(f"      - Fairness-aware sampling/reweighting")
        print(f"      - Additional fairness regularization")
        print(f"   4. Document limitations and escalate to clinical team")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review all outputs in {args.output_dir}")
    print(f"   2. Generate clinical approval package (run generate_approval_package.py)")
    print(f"   3. Present to clinical and ethics stakeholders")
    print(f"   4. Document approval decision")
    if mitigation_impact['summary']['recommended_for_deployment']:
        print(f"   5. Proceed to Phase 7 (Deployment Preparation)")
    else:
        print(f"   5. Revisit mitigation strategy or retrain model")
    
    # ========================================================================
    # Step 9: Upload to HuggingFace Hub
    # ========================================================================
    
    print("\n" + "="*80)
    print("Step 9: Upload Results to HuggingFace Hub")
    print("="*80)
    
    try:
        repo_url = upload_results_to_hf(
            output_dir=args.output_dir,
            repo_id='auphong2707/hospital-readmission-gradient-boosting-mitigation-results',
            commit_message="Upload Phase 6 fairness mitigation results"
        )
        print(f"✅ Results uploaded successfully!")
        print(f"🌐 View at: {repo_url}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"💡 You can upload manually later or set HF_TOKEN environment variable")
    
    print("\n" + "="*80)
    print("✅ Phase 6 Completed Successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
