"""
Phase 4: Optimal Threshold & ROI Analysis for Logistic Regression Model

This script performs cost-sensitive threshold optimization and ROI analysis for
the calibrated hospital readmission prediction model from Phase 3.

Pipeline:
1. Load calibrated model and predictions from Phase 3
2. Load Phase 1 test data (consistent with all phases)
3. Define business cost parameters (readmission cost, intervention cost)
4. Find optimal decision threshold maximizing expected value
5. Define risk categories (Low, Medium, High) based on optimal threshold
6. Calculate comprehensive ROI metrics
7. Perform sensitivity analysis with different cost assumptions
8. Generate 8 visualizations for threshold optimization and ROI analysis
9. Save results for Phase 5 (fairness evaluation)

Cost Matrix:
- TP: +$14,500 (prevented readmission)
- FP: -$500 (unnecessary intervention)
- TN: $0 (correct prediction, no action)
- FN: -$15,000 (missed readmission)

Business Goal:
- Maximize expected value
- Achieve positive ROI (>3.3% readmission reduction)
- Ensure intervention volume is operationally feasible

Usage (from project root):
    # Basic usage with default parameters
    python ./phase-4-optimal-threshold-ROI-analysis/optimize_threshold_logistic_regression.py
    
    # With custom cost parameters
    python ./phase-4-optimal-threshold-ROI-analysis/optimize_threshold_logistic_regression.py \
        --readmission-cost 20000 --intervention-cost 1000
    
    # With operational constraints
    python ./phase-4-optimal-threshold-ROI-analysis/optimize_threshold_logistic_regression.py \
        --max-intervention-rate 0.25

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib

Phase 3 Output Required:
    - Calibrated model from ./calibration_outputs/logistic_regression/
    - Phase 1 data splits (automatically downloaded from HuggingFace Hub)
    - Files: logistic_regression_model_original.joblib, logistic_regression_scaler.joblib, Logistic_Regression_calibrator.pkl

Output:
    - Optimal threshold and risk category thresholds
    - ROI analysis report
    - 8 visualizations (threshold optimization + ROI analysis)
    - Results saved for Phase 5
"""

import argparse
import sys
import warnings
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utilities import (
    load_phase1_splits,
    ThresholdOptimizer,
    RiskCategoryMapper,
    ROIAnalyzer,
    ROIVisualizer,
    save_threshold_results,
    upload_results_to_hf,
    print_section,
    _convert_to_serializable
)

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Phase 4: Optimal Threshold & ROI Analysis for Logistic Regression'
    )
    
    # Cost parameters
    parser.add_argument(
        '--readmission-cost',
        type=float,
        default=15000,
        help='Cost of a readmission event (default: $15,000)'
    )
    parser.add_argument(
        '--intervention-cost',
        type=float,
        default=500,
        help='Cost of intervention per patient (default: $500)'
    )
    
    # Threshold optimization
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
    
    # Constraints
    parser.add_argument(
        '--max-intervention-rate',
        type=float,
        default=None,
        help='Maximum intervention rate constraint (0-1, optional)'
    )
    
    # Data and model paths
    parser.add_argument(
        '--data-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-risk-data',
        help='HuggingFace repository ID for Phase 1 data (default: auphong2707/hospital-readmission-risk-data)'
    )
    parser.add_argument(
        '--calibration-dir',
        type=str,
        default='./calibration_outputs/logistic_regression',
        help='Directory containing calibrated model files (default: ./calibration_outputs/logistic_regression)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./models/downloaded',
        help='Directory to cache downloaded data (default: ./models/downloaded)'
    )
    
    # Output paths
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./phase-4-optimal-threshold-ROI-analysis/outputs_logistic_regression',
        help='Output directory for results'
    )
    parser.add_argument(
        '--viz-dir',
        type=str,
        default='./phase-4-optimal-threshold-ROI-analysis/visualizations_logistic_regression',
        help='Output directory for visualizations'
    )
    
    return parser.parse_args()


def load_calibrated_model_local(calibration_dir: str):
    """
    Load calibrated Logistic Regression model, scaler, and calibrator from local directory.
    
    Args:
        calibration_dir: Directory containing calibration outputs from Phase 3
        
    Returns:
        tuple: (model, scaler, calibrator)
    """
    print("📥 Loading calibrated model from local directory...")
    
    calibration_path = Path(calibration_dir)
    
    # Load model
    model_path = calibration_path / "logistic_regression_model_original.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please run Phase 3 calibration first:\n"
            f"  python ./phase-3-model-calibration/calibrate_logistic_regression.py"
        )
    model = joblib.load(model_path)
    print(f"   ✅ Model loaded: {model_path}")
    
    # Load scaler
    scaler_path = calibration_path / "logistic_regression_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"   ✅ Scaler loaded: {scaler_path}")
    
    # Load calibrator
    calibrator_path = calibration_path / "Logistic_Regression_calibrator.pkl"
    if not calibrator_path.exists():
        raise FileNotFoundError(f"Calibrator not found: {calibrator_path}")
    
    # Load calibrator using utilities
    from utilities import ModelCalibrator
    calibrator = ModelCalibrator.load(str(calibrator_path))
    print(f"   ✅ Calibrator loaded: {calibrator_path}")
    
    return model, scaler, calibrator


def get_calibrated_predictions_lr(model, scaler, calibrator, X_test):
    """
    Generate calibrated probability predictions for Logistic Regression.
    
    Args:
        model: Trained Logistic Regression model
        scaler: Fitted StandardScaler
        calibrator: Fitted calibrator
        X_test: Test features
        
    Returns:
        np.ndarray: Calibrated probabilities
    """
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Get uncalibrated probabilities
    y_pred_proba_uncalibrated = model.predict_proba(X_test_scaled)[:, 1]
    
    # Apply calibration
    y_pred_proba_calibrated = calibrator.predict_proba(y_pred_proba_uncalibrated)
    
    return y_pred_proba_calibrated


def main():
    """Main execution pipeline."""
    
    print("\n" + "="*80)
    print("PHASE 4: OPTIMAL THRESHOLD & ROI ANALYSIS")
    print("Logistic Regression - Hospital Readmission Risk Prediction")
    print("="*80)
    
    # Parse arguments
    args = parse_arguments()
    
    # ========================================================================
    # STEP 1: Load Calibrated Model and Data
    # ========================================================================
    
    print_section("Step 1: Load Calibrated Model and Data", "-")
    
    # Load Phase 1 test data (consistent with all phases)
    print("📥 Loading Phase 1 test data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_phase1_splits(
        repo_id=args.data_repo_id
    )
    
    print(f"✅ Test data loaded: {X_test.shape}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Readmission rate: {y_test.mean():.1%}")
    
    # Load calibrated model from local directory
    print("\n📥 Loading calibrated model from local directory...")
    try:
        model, scaler, calibrator = load_calibrated_model_local(args.calibration_dir)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    
    # Generate calibrated predictions
    print("🔮 Generating calibrated predictions...")
    y_pred_proba_calibrated = get_calibrated_predictions_lr(model, scaler, calibrator, X_test)
    
    print(f"✅ Calibrated predictions generated")
    print(f"   Mean predicted probability: {y_pred_proba_calibrated.mean():.3f}")
    print(f"   Probability range: [{y_pred_proba_calibrated.min():.3f}, {y_pred_proba_calibrated.max():.3f}]")
    
    # ========================================================================
    # STEP 2: Define Cost Parameters
    # ========================================================================
    
    print_section("Step 2: Define Business Cost Parameters", "-")
    
    cost_params = {
        'readmission_cost': args.readmission_cost,
        'intervention_cost': args.intervention_cost,
        'tp_benefit': args.readmission_cost - args.intervention_cost,
        'fp_cost': -args.intervention_cost,
        'fn_cost': -args.readmission_cost,
        'tn_cost': 0
    }
    
    print("💰 Cost Matrix:")
    print(f"   Readmission Cost: ${cost_params['readmission_cost']:,.2f}")
    print(f"   Intervention Cost: ${cost_params['intervention_cost']:,.2f}")
    print(f"\n   TP (Prevented Readmission): +${cost_params['tp_benefit']:,.2f}")
    print(f"   FP (Unnecessary Intervention): ${cost_params['fp_cost']:,.2f}")
    print(f"   FN (Missed Readmission): ${cost_params['fn_cost']:,.2f}")
    print(f"   TN (Correct Negative): ${cost_params['tn_cost']:,.2f}")
    
    # Calculate break-even rate
    break_even_rate = cost_params['intervention_cost'] / cost_params['readmission_cost']
    print(f"\n📊 Break-Even Analysis:")
    print(f"   Break-even threshold: {break_even_rate:.1%}")
    print(f"   Need to prevent >1 readmission per {1/break_even_rate:.0f} interventions")
    
    # ========================================================================
    # STEP 3: Find Optimal Threshold
    # ========================================================================
    
    print_section("Step 3: Optimal Threshold Search", "-")
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(
        y_true=y_test,
        y_pred_proba=y_pred_proba_calibrated,
        cost_params=cost_params
    )
    
    # Find optimal threshold
    if args.max_intervention_rate is not None:
        print(f"⚙️  Using intervention rate constraint: {args.max_intervention_rate:.1%}")
        optimal_threshold = optimizer.find_optimal_threshold_with_constraint(
            max_intervention_rate=args.max_intervention_rate,
            threshold_range=(args.threshold_min, args.threshold_max),
            num_points=args.num_thresholds
        )
    else:
        optimal_threshold = optimizer.find_optimal_threshold(
            threshold_range=(args.threshold_min, args.threshold_max),
            num_points=args.num_thresholds
        )
    
    # Get detailed metrics at optimal threshold
    optimal_metrics = optimizer.get_metrics_at_threshold(optimal_threshold)
    
    print("\n📊 Detailed Metrics at Optimal Threshold:")
    print(f"   Precision: {optimal_metrics['precision']:.3f}")
    print(f"   Recall (Sensitivity): {optimal_metrics['recall']:.3f}")
    print(f"   F1-Score: {optimal_metrics['f1_score']:.3f}")
    print(f"   Specificity: {optimal_metrics['tnr']:.3f}")
    print(f"   Accuracy: {optimal_metrics['accuracy']:.3f}")
    print(f"   TPR: {optimal_metrics['tpr']:.3f}")
    print(f"   FPR: {optimal_metrics['fpr']:.3f}")
    
    # ========================================================================
    # STEP 4: Define Risk Categories
    # ========================================================================
    
    print_section("Step 4: Define Risk Categories", "-")
    
    # Create risk mapper based on optimal threshold
    risk_mapper = RiskCategoryMapper(optimal_threshold=optimal_threshold)
    
    print(f"🎯 Risk Category Thresholds:")
    print(f"   Low Risk: < {risk_mapper.low_threshold:.4f}")
    print(f"   Medium Risk: {risk_mapper.low_threshold:.4f} - {risk_mapper.high_threshold:.4f}")
    print(f"   High Risk: ≥ {risk_mapper.high_threshold:.4f}")
    
    # Get category statistics
    risk_mapper.print_category_summary(y_test, y_pred_proba_calibrated)
    
    # Validate risk categories
    stats_df = risk_mapper.get_category_statistics(y_test, y_pred_proba_calibrated)
    
    print("✅ Risk Category Validation:")
    for _, row in stats_df.iterrows():
        category = row['risk_category']
        actual_rate = row['actual_readmission_rate']
        
        if category == 'Low' and actual_rate < 0.05:
            print(f"   ✅ {category} risk: {actual_rate:.1%} < 5% (Good)")
        elif category == 'Medium' and 0.05 <= actual_rate <= 0.15:
            print(f"   ✅ {category} risk: {actual_rate:.1%} in [5%, 15%] range (Good)")
        elif category == 'High' and actual_rate > 0.15:
            print(f"   ✅ {category} risk: {actual_rate:.1%} > 15% (Good)")
        else:
            print(f"   ⚠️  {category} risk: {actual_rate:.1%} (Check alignment)")
    
    # ========================================================================
    # STEP 5: Calculate ROI Metrics
    # ========================================================================
    
    print_section("Step 5: ROI Analysis", "-")
    
    # Create ROI analyzer
    roi_analyzer = ROIAnalyzer(
        y_true=y_test,
        y_pred_proba=y_pred_proba_calibrated,
        optimal_threshold=optimal_threshold,
        cost_params=cost_params
    )
    
    # Print ROI report
    roi_analyzer.print_roi_report()
    
    # Get ROI metrics
    roi_metrics = roi_analyzer.calculate_roi_metrics()
    
    # Check success criteria
    print("✅ Success Criteria Validation:")
    
    # 1. Positive expected value
    if roi_metrics['net_benefit'] > 0:
        print(f"   ✅ Positive Expected Value: ${roi_metrics['net_benefit']:,.2f}")
    else:
        print(f"   ❌ Negative Expected Value: ${roi_metrics['net_benefit']:,.2f}")
    
    # 2. ROI above break-even
    if roi_metrics['readmission_reduction_absolute'] > break_even_rate:
        margin = (roi_metrics['readmission_reduction_absolute'] - break_even_rate) / break_even_rate * 100
        print(f"   ✅ Above Break-Even: {roi_metrics['readmission_reduction_absolute']:.1%} > {break_even_rate:.1%} ({margin:.1f}% margin)")
    else:
        print(f"   ❌ Below Break-Even: {roi_metrics['readmission_reduction_absolute']:.1%} < {break_even_rate:.1%}")
    
    # 3. Intervention volume feasibility
    if roi_metrics['intervention_rate'] < 0.30:
        print(f"   ✅ Feasible Intervention Volume: {roi_metrics['intervention_rate']:.1%} < 30%")
    else:
        print(f"   ⚠️  High Intervention Volume: {roi_metrics['intervention_rate']:.1%} (may need capacity review)")
    
    # 4. ROI percentage
    if roi_metrics['roi_percentage'] > 0:
        print(f"   ✅ Positive ROI: {roi_metrics['roi_percentage']:.1f}%")
    else:
        print(f"   ❌ Negative ROI: {roi_metrics['roi_percentage']:.1f}%")
    
    # ========================================================================
    # STEP 6: Sensitivity Analysis
    # ========================================================================
    
    print_section("Step 6: Sensitivity Analysis", "-")
    
    # Define scenarios
    scenarios = {
        'Conservative': {
            'readmission_cost': cost_params['readmission_cost'] * 0.8,
            'intervention_cost': cost_params['intervention_cost'] * 1.2
        },
        'Base Case': cost_params,
        'Aggressive': {
            'readmission_cost': cost_params['readmission_cost'] * 1.2,
            'intervention_cost': cost_params['intervention_cost'] * 0.8
        }
    }
    
    print("🔬 Testing ROI under different cost scenarios:")
    sensitivity_df = roi_analyzer.sensitivity_analysis(scenarios)
    
    print(f"\n{sensitivity_df.to_string(index=False)}")
    
    # Check robustness
    all_positive = (sensitivity_df['net_benefit'] > 0).all()
    if all_positive:
        print("\n✅ Robust: Positive ROI across all scenarios")
    else:
        print("\n⚠️  Not robust: Some scenarios have negative ROI")
    
    # ========================================================================
    # STEP 7: Generate Visualizations
    # ========================================================================
    
    print_section("Step 7: Generate Visualizations", "-")
    
    # Generate all visualizations
    viz_paths = ROIVisualizer.generate_all_visualizations(
        threshold_optimizer=optimizer,
        risk_mapper=risk_mapper,
        roi_analyzer=roi_analyzer,
        y_true=y_test,
        y_pred_proba=y_pred_proba_calibrated,
        output_dir=args.viz_dir,
        scenarios=scenarios
    )
    
    # ========================================================================
    # STEP 8: Save Results
    # ========================================================================
    
    print_section("Step 8: Save Results", "-")
    
    # Save threshold optimization results
    save_threshold_results(
        threshold_optimizer=optimizer,
        risk_mapper=risk_mapper,
        roi_analyzer=roi_analyzer,
        output_dir=args.output_dir
    )
    
    # Save summary for Phase 5
    phase5_summary = {
        'optimal_threshold': optimal_threshold,
        'low_risk_threshold': risk_mapper.low_threshold,
        'high_risk_threshold': risk_mapper.high_threshold,
        'expected_value': optimizer.optimal_ev,
        'roi_percentage': roi_metrics['roi_percentage'],
        'net_benefit': roi_metrics['net_benefit'],
        'intervention_rate': roi_metrics['intervention_rate'],
        'readmission_reduction_absolute': roi_metrics['readmission_reduction_absolute'],
        'readmission_reduction_relative': roi_metrics['readmission_reduction_relative'],
        'cost_parameters': cost_params,
        'optimal_metrics': optimal_metrics,
        'success_criteria': {
            'positive_ev': roi_metrics['net_benefit'] > 0,
            'above_break_even': roi_metrics['readmission_reduction_absolute'] > break_even_rate,
            'feasible_volume': roi_metrics['intervention_rate'] < 0.30,
            'positive_roi': roi_metrics['roi_percentage'] > 0
        }
    }
    
    # Convert to JSON-serializable format
    phase5_summary = _convert_to_serializable(phase5_summary)
    
    summary_path = Path(args.output_dir) / "phase4_summary_for_phase5.json"
    with open(summary_path, 'w') as f:
        json.dump(phase5_summary, f, indent=2)
    print(f"   ✅ Phase 5 summary: {summary_path}")
    
    # ========================================================================
    # STEP 9: Upload to HuggingFace Hub (Optional)
    # ========================================================================
    
    print_section("Step 9: Upload Results to HuggingFace Hub (Optional)", "-")
    
    # Hardcoded repository ID for Logistic Regression
    results_repo_id = 'auphong2707/hospital-readmission-logistic-regression-threshold-results'
    
    try:
        repo_url = upload_results_to_hf(
            output_dir=args.output_dir,
            viz_dir=args.viz_dir,
            repo_id=results_repo_id,
            commit_message=f"Phase 4 results: Optimal threshold={optimal_threshold:.4f}, ROI={roi_metrics['roi_percentage']:.1f}%"
        )
        print(f"✅ Results successfully uploaded!")
        print(f"🌐 View results at: {repo_url}")
    except ImportError as e:
        print(f"⚠️  Upload skipped: {e}")
        print(f"💡 Install huggingface_hub: pip install huggingface_hub")
    except ValueError as e:
        print(f"⚠️  Upload skipped: {e}")
        print(f"💡 Set HF_TOKEN environment variable or pass --hf-token")
    except Exception as e:
        print(f"⚠️  Upload skipped: {e}")
        print(f"💡 Results are still saved locally in {args.output_dir}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print_section("✅ Phase 4 Complete: Optimal Threshold & ROI Analysis", "=")
    
    print("📊 Key Results:")
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   Expected Value: ${optimizer.optimal_ev:,.2f}")
    print(f"   ROI: {roi_metrics['roi_percentage']:.1f}%")
    print(f"   Net Benefit: ${roi_metrics['net_benefit']:,.2f}")
    print(f"   Total Savings: ${roi_metrics['total_savings']:,.2f} ({roi_metrics['savings_percentage']:.1f}%)")
    print(f"   Readmission Reduction: {roi_metrics['readmission_reduction_absolute']:.1%} ({roi_metrics['readmission_reduction_relative']:.1f}%)")
    print(f"   Intervention Volume: {roi_metrics['intervention_volume']:,} ({roi_metrics['intervention_rate']:.1%})")
    
    print(f"\n📁 Outputs:")
    print(f"   Results: {args.output_dir}")
    print(f"   Visualizations: {args.viz_dir}")
    print(f"   Total visualizations: {len(viz_paths)}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review visualizations in: {args.viz_dir}")
    print(f"   2. Validate business assumptions with stakeholders")
    print(f"   3. Proceed to Phase 5: Fairness Evaluation")
    print(f"      python ./phase-5-fairness-evaluation/evaluate_fairness_logistic_regression.py")
    
    print("\n" + "="*80)
    print("✅ Phase 4 Completed Successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
