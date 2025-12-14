"""
Phase 5: Fairness Evaluation & Deployment Readiness for Gradient Boosting Model

This script performs comprehensive fairness evaluation of the calibrated hospital 
readmission prediction model using optimal thresholds from Phase 4.

Pipeline:
1. Load test data, demographics, calibrated model, and Phase 4 results
2. Generate calibrated predictions
3. Apply optimal threshold from Phase 4
4. Compute overall performance metrics
5. Compute group-specific metrics (race, gender, age)
6. Calculate fairness metrics (demographic parity, equalized odds, equal opportunity)
7. Perform statistical significance testing
8. Analyze risk category distribution by group
9. Generate fairness visualizations
10. Create fairness report and deployment recommendations

Fairness Metrics:
- Demographic Parity: Intervention rate similar across groups
- Equalized Odds: TPR and FPR similar across groups
- Equal Opportunity: TPR similar across groups

Statistical Tests:
- Chi-square test for intervention rate independence
- Two-proportion z-tests for TPR/FPR differences

Usage (from project root):
    # Using HuggingFace Hub (recommended)
    python ./phase-5-fairness-evaluation/evaluate_fairness_gradient_boosting.py
    
    # Using local files
    python ./phase-5-fairness-evaluation/evaluate_fairness_gradient_boosting.py \
        --use-local \
        --local-test ./data/processed/splits/test.csv \
        --local-demographics ./data/processed/splits/test_demographics.csv
    
    # Custom output directory
    python ./phase-5-fairness-evaluation/evaluate_fairness_gradient_boosting.py \
        --output-dir ./phase-5-fairness-evaluation/outputs
    
    Note: Results are automatically uploaded to HuggingFace Hub
          Repository: auphong2707/hospital-readmission-gradient-boosting-fairness-results
          Requires: HF_TOKEN environment variable

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib huggingface_hub

Phase Dependencies:
    - Phase 1: Demographics files (test_demographics.csv)
    - Phase 3: Calibrated model (model + calibrator)
    - Phase 4: Optimal thresholds and ROI results

Output:
    - fairness_report.json (comprehensive fairness metrics)
    - group_metrics_*.csv (performance by group for each attribute)
    - statistical_tests.json (significance tests)
    - risk_categories_*.csv (risk distribution by group)
    - visualizations/ (fairness assessment plots)
        - group_comparison_*.png (TPR, FPR, precision, intervention rate)
        - fairness_heatmap.png (pass/fail status for fairness metrics)
        - confusion_matrices_*.png (by race, gender, age)
        - calibration_by_*.png (calibration curves by demographic group)
        - risk_distribution_*.png (risk categories by demographic group)
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict
import json
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utilities import (
    load_test_data_and_demographics,
    load_model_and_calibrator,
    load_phase4_results,
    generate_calibrated_predictions,
    GroupPerformanceAnalyzer,
    FairnessMetrics,
    StatisticalTester,
    FairnessVisualizer,
    print_section,
    save_results,
    upload_results_to_hf
)

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Phase 5: Fairness Evaluation & Deployment Readiness'
    )
    
    # Data source options
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
    
    # HuggingFace options
    parser.add_argument(
        '--data-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-risk-data',
        help='HuggingFace dataset repository ID'
    )
    parser.add_argument(
        '--model-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-lgbm-calibrated',
        help='HuggingFace model repository ID'
    )
    
    # Phase 4 results
    parser.add_argument(
        '--phase4-summary',
        type=str,
        default='./phase-4-optimal-threshold-ROI-analysis/outputs/phase4_summary_for_phase5.json',
        help='Path to Phase 4 summary JSON'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./phase-5-fairness-evaluation/outputs',
        help='Output directory for results'
    )
    
    # Fairness thresholds
    parser.add_argument(
        '--fairness-tolerance',
        type=float,
        default=0.05,
        help='Tolerance for fairness metric violations (default: 0.05 = ±5%%)'
    )
    
    return parser.parse_args()


def apply_threshold_and_categorize(
    y_pred_proba: np.ndarray,
    optimal_threshold: float,
    low_threshold: float,
    high_threshold: float
) -> tuple:
    """
    Apply optimal threshold and assign risk categories.
    
    Args:
        y_pred_proba: Calibrated probabilities
        optimal_threshold: Optimal decision threshold
        low_threshold: Low risk boundary
        high_threshold: High risk boundary
        
    Returns:
        tuple: (y_pred, risk_categories)
    """
    # Binary predictions at optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Risk categories
    risk_categories = np.full(len(y_pred_proba), 'Medium')
    risk_categories[y_pred_proba < low_threshold] = 'Low'
    risk_categories[y_pred_proba >= high_threshold] = 'High'
    
    return y_pred, risk_categories


def analyze_risk_categories_by_group(
    risk_categories: np.ndarray,
    y_true: np.ndarray,
    demographics: pd.DataFrame,
    attribute: str
) -> pd.DataFrame:
    """
    Analyze risk category distribution by demographic group.
    
    Args:
        risk_categories: Assigned risk categories
        y_true: True labels
        demographics: Demographics DataFrame
        attribute: Demographic attribute
        
    Returns:
        DataFrame with risk category analysis per group
    """
    if attribute not in demographics.columns:
        return pd.DataFrame()
    
    groups = demographics[attribute].unique()
    results = []
    
    for group in groups:
        mask = (demographics[attribute] == group).values
        
        if mask.sum() == 0:
            continue
        
        group_risk = risk_categories[mask]
        group_true = y_true[mask]
        
        # Risk category distribution
        risk_dist = pd.Series(group_risk).value_counts(normalize=True)
        
        # Actual readmission rate per risk category
        readmit_by_risk = {}
        for risk_cat in ['Low', 'Medium', 'High']:
            risk_mask = group_risk == risk_cat
            if risk_mask.sum() > 0:
                readmit_by_risk[risk_cat] = float(group_true[risk_mask].mean())
            else:
                readmit_by_risk[risk_cat] = 0.0
        
        group_result = {
            'attribute': attribute,
            'group': group,
            'n_samples': int(mask.sum()),
            'pct_low_risk': float(risk_dist.get('Low', 0)),
            'pct_medium_risk': float(risk_dist.get('Medium', 0)),
            'pct_high_risk': float(risk_dist.get('High', 0)),
            'readmit_rate_low': readmit_by_risk.get('Low', 0.0),
            'readmit_rate_medium': readmit_by_risk.get('Medium', 0.0),
            'readmit_rate_high': readmit_by_risk.get('High', 0.0)
        }
        
        results.append(group_result)
    
    return pd.DataFrame(results)


def generate_fairness_summary(
    overall_metrics: Dict,
    all_group_metrics: Dict,
    fairness_results: Dict,
    statistical_tests: Dict,
    risk_category_analysis: Dict,
    phase4_results: Dict
) -> Dict:
    """
    Generate comprehensive fairness summary report.
    
    Args:
        overall_metrics: Overall performance metrics
        all_group_metrics: Group-specific metrics
        fairness_results: Fairness metric results
        statistical_tests: Statistical test results
        risk_category_analysis: Risk category fairness analysis
        phase4_results: Phase 4 results
        
    Returns:
        dict: Comprehensive fairness report with Phase 6 inputs
    """
    summary = {
        'phase': 5,
        'evaluation_name': 'Fairness Evaluation & Deployment Readiness',
        'model': 'Gradient Boosting (LightGBM) with Platt Calibration',
        'optimal_threshold': phase4_results['optimal_threshold'],
        'overall_performance': overall_metrics,
        'fairness_evaluation': {},
        'bias_detected': False,
        'recommendations': [],
        # Phase 6 decision inputs
        'phase6_inputs': {
            'requires_mitigation': False,
            'group_metrics_summary': {},
            'worst_violations': [],
            'mitigation_priority': 'none'  # none, low, medium, high
        }
    }
    
    # Track worst violations for Phase 6 prioritization
    worst_violations = []
    
    # Evaluate fairness for each attribute
    for attribute, fairness_metrics in fairness_results.items():
        attr_summary = {
            'demographic_parity': fairness_metrics['demographic_parity'],
            'equalized_odds': fairness_metrics['equalized_odds'],
            'equal_opportunity': fairness_metrics['equal_opportunity'],
            'all_passed': (
                fairness_metrics['demographic_parity']['passed'] and
                fairness_metrics['equalized_odds']['passed'] and
                fairness_metrics['equal_opportunity']['passed']
            )
        }
        
        summary['fairness_evaluation'][attribute] = attr_summary
        
        # Check for bias
        if not attr_summary['all_passed']:
            summary['bias_detected'] = True
            
            # Record violations for Phase 6
            violations = []
            if not fairness_metrics['demographic_parity']['passed']:
                violations.append({
                    'metric': 'demographic_parity',
                    'gap': fairness_metrics['demographic_parity']['gap'],
                    'threshold': 0.05
                })
            if not fairness_metrics['equalized_odds']['passed']:
                violations.append({
                    'metric': 'equalized_odds',
                    'tpr_gap': fairness_metrics['equalized_odds']['tpr_gap'],
                    'fpr_gap': fairness_metrics['equalized_odds']['fpr_gap'],
                    'threshold': 0.05
                })
            if not fairness_metrics['equal_opportunity']['passed']:
                violations.append({
                    'metric': 'equal_opportunity',
                    'gap': fairness_metrics['equal_opportunity']['gap'],
                    'threshold': 0.05
                })
            
            worst_violations.append({
                'attribute': attribute,
                'violations': violations
            })
        
        # Collect group-level data for Phase 6
        if attribute in all_group_metrics:
            group_data = all_group_metrics[attribute].to_dict('records')
            summary['phase6_inputs']['group_metrics_summary'][attribute] = group_data
    
    # Determine mitigation priority
    summary['phase6_inputs']['requires_mitigation'] = summary['bias_detected']
    summary['phase6_inputs']['worst_violations'] = worst_violations
    
    if summary['bias_detected']:
        # Calculate severity based on number of violations and gap sizes
        max_gap = 0.0
        violation_count = sum(len(v['violations']) for v in worst_violations)
        
        for v in worst_violations:
            for violation in v['violations']:
                if 'gap' in violation:
                    max_gap = max(max_gap, abs(violation['gap']))
                if 'tpr_gap' in violation:
                    max_gap = max(max_gap, abs(violation['tpr_gap']))
                if 'fpr_gap' in violation:
                    max_gap = max(max_gap, abs(violation['fpr_gap']))
        
        # Priority logic: high if many violations or large gaps
        if max_gap > 0.10 or violation_count >= 6:
            summary['phase6_inputs']['mitigation_priority'] = 'high'
        elif max_gap > 0.07 or violation_count >= 3:
            summary['phase6_inputs']['mitigation_priority'] = 'medium'
        else:
            summary['phase6_inputs']['mitigation_priority'] = 'low'
    
    # Add recommendations
    if summary['bias_detected']:
        summary['recommendations'].append(
            "⚠️ Fairness violations detected. Consider group-specific thresholds or recalibration."
        )
        summary['recommendations'].append(
            "Review group-specific metrics and statistical tests for detailed analysis."
        )
        summary['recommendations'].append(
            "Consult with clinical and ethics teams before deployment."
        )
        summary['recommendations'].append(
            f"Phase 6 mitigation priority: {summary['phase6_inputs']['mitigation_priority'].upper()}"
        )
    else:
        summary['recommendations'].append(
            "✅ No significant fairness violations detected at ±5% tolerance."
        )
        summary['recommendations'].append(
            "Proceed with deployment using global optimal threshold."
        )
        summary['recommendations'].append(
            "Implement monitoring plan to track fairness metrics in production."
        )
    
    return summary


def main():
    """Main execution pipeline."""
    
    print("\n" + "="*80)
    print("PHASE 5: FAIRNESS EVALUATION & DEPLOYMENT READINESS")
    print("Hospital Readmission Risk Prediction - Gradient Boosting Model")
    print("="*80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load All Inputs
    # ========================================================================
    
    print_section("Step 1: Load Test Data and Demographics", "-")
    
    X_test, y_test, demographics = load_test_data_and_demographics(
        data_repo_id=args.data_repo_id,
        use_local=args.use_local,
        local_test_path=args.local_test,
        local_demographics_path=args.local_demographics
    )
    
    print_section("Step 2: Load Calibrated Model", "-")
    
    model, calibrator = load_model_and_calibrator(
        model_repo_id=args.model_repo_id,
        use_local=args.use_local,
        local_model_path=args.local_model,
        local_calibrator_path=args.local_calibrator,
        method='gradient_boosting'
    )
    
    print_section("Step 3: Load Phase 4 Results", "-")
    
    phase4_results = load_phase4_results(args.phase4_summary)
    
    optimal_threshold = phase4_results['optimal_threshold']
    low_threshold = phase4_results['low_risk_threshold']
    high_threshold = phase4_results['high_risk_threshold']
    
    # ========================================================================
    # STEP 2: Generate Predictions
    # ========================================================================
    
    print_section("Step 4: Generate Calibrated Predictions", "-")
    
    y_pred_proba = generate_calibrated_predictions(model, calibrator, X_test)
    
    print_section("Step 5: Apply Optimal Threshold and Categorize Risk", "-")
    
    y_pred, risk_categories = apply_threshold_and_categorize(
        y_pred_proba,
        optimal_threshold,
        low_threshold,
        high_threshold
    )
    
    print(f"✅ Applied optimal threshold: {optimal_threshold:.4f}")
    print(f"   Predictions: {dict(pd.Series(y_pred).value_counts())}")
    print(f"   Risk categories: {dict(pd.Series(risk_categories).value_counts())}")
    
    # ========================================================================
    # STEP 3: Compute Performance Metrics
    # ========================================================================
    
    print_section("Step 6: Compute Overall Performance Metrics", "-")
    
    analyzer = GroupPerformanceAnalyzer(
        y_true=y_test.values,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        demographics=demographics
    )
    
    overall_metrics = analyzer.compute_overall_metrics()
    
    print("📊 Overall Performance:")
    print(f"   Accuracy: {overall_metrics['accuracy']:.3f}")
    print(f"   Precision: {overall_metrics['precision']:.3f}")
    print(f"   Recall (TPR): {overall_metrics['recall']:.3f}")
    print(f"   F1-Score: {overall_metrics['f1_score']:.3f}")
    print(f"   FPR: {overall_metrics['fpr']:.3f}")
    print(f"   ROC-AUC: {overall_metrics['roc_auc']:.3f}")
    print(f"   Intervention Rate: {overall_metrics['intervention_rate']:.1%}")
    
    print_section("Step 7: Compute Group-Specific Metrics", "-")
    
    all_group_metrics = analyzer.compute_all_group_metrics()
    
    for attribute, group_metrics_df in all_group_metrics.items():
        print(f"\n📊 {attribute.upper()} Group Metrics:")
        print(group_metrics_df[['group', 'n_samples', 'tpr', 'fpr', 'precision', 'intervention_rate']].to_string(index=False))
        
        # Save to CSV
        output_file = output_dir / f"group_metrics_{attribute}.csv"
        group_metrics_df.to_csv(output_file, index=False)
        print(f"   ✅ Saved: {output_file}")
    
    # ========================================================================
    # STEP 4: Calculate Fairness Metrics
    # ========================================================================
    
    print_section("Step 8: Calculate Fairness Metrics", "-")
    
    fairness_results = FairnessMetrics.compute_all_fairness_metrics(all_group_metrics)
    
    for attribute, metrics in fairness_results.items():
        print(f"\n🎯 {attribute.upper()} Fairness:")
        
        # Demographic Parity
        dp = metrics['demographic_parity']
        print(f"   Demographic Parity: {'✅ PASSED' if dp['passed'] else '❌ FAILED'}")
        print(f"      Intervention rate gap: {dp['gap']:.3f} (max-min)")
        
        # Equalized Odds
        eo = metrics['equalized_odds']
        print(f"   Equalized Odds: {'✅ PASSED' if eo['passed'] else '❌ FAILED'}")
        print(f"      TPR gap: {eo['tpr_gap']:.3f}, FPR gap: {eo['fpr_gap']:.3f}")
        
        # Equal Opportunity
        eop = metrics['equal_opportunity']
        print(f"   Equal Opportunity: {'✅ PASSED' if eop['passed'] else '❌ FAILED'}")
        print(f"      TPR gap: {eop['gap']:.3f}")
    
    # ========================================================================
    # STEP 5: Statistical Testing
    # ========================================================================
    
    print_section("Step 9: Perform Statistical Significance Testing", "-")
    
    statistical_tests = {}
    
    for attribute in all_group_metrics.keys():
        print(f"\n🔬 {attribute.upper()} Statistical Tests:")
        
        # Chi-square test for intervention rate
        chi2_result = StatisticalTester.chi_square_test_intervention_rate(
            y_pred, demographics, attribute
        )
        statistical_tests[f'{attribute}_chi2'] = chi2_result
        
        print(f"   Chi-square test (intervention rate):")
        print(f"      χ² = {chi2_result['chi2_statistic']:.2f}, p-value = {chi2_result['p_value']:.4f}")
        print(f"      Result: {chi2_result['interpretation']}")
    
    # ========================================================================
    # STEP 6: Risk Category Fairness
    # ========================================================================
    
    print_section("Step 10: Analyze Risk Category Distribution by Group", "-")
    
    risk_category_analysis = {}
    
    for attribute in all_group_metrics.keys():
        risk_df = analyze_risk_categories_by_group(
            risk_categories, y_test.values, demographics, attribute
        )
        
        if not risk_df.empty:
            risk_category_analysis[attribute] = risk_df
            
            print(f"\n📊 {attribute.upper()} Risk Categories:")
            print(risk_df[['group', 'pct_low_risk', 'pct_medium_risk', 'pct_high_risk']].to_string(index=False))
            
            # Save to CSV
            output_file = output_dir / f"risk_categories_{attribute}.csv"
            risk_df.to_csv(output_file, index=False)
            print(f"   ✅ Saved: {output_file}")
    
    # ========================================================================
    # STEP 7: Generate Fairness Summary Report
    # ========================================================================
    
    print_section("Step 11: Generate Fairness Summary Report", "-")
    
    fairness_summary = generate_fairness_summary(
        overall_metrics,
        all_group_metrics,
        fairness_results,
        statistical_tests,
        risk_category_analysis,
        phase4_results
    )
    
    # Save fairness report
    report_path = output_dir / "fairness_report.json"
    save_results(fairness_summary, str(report_path))
    
    # Save statistical tests
    tests_path = output_dir / "statistical_tests.json"
    save_results(statistical_tests, str(tests_path))
    
    # Save Phase 6 decision inputs (compact summary for mitigation decisions)
    phase6_summary_path = output_dir / "phase5_summary_for_phase6.json"
    phase6_summary = {
        'requires_mitigation': fairness_summary['phase6_inputs']['requires_mitigation'],
        'mitigation_priority': fairness_summary['phase6_inputs']['mitigation_priority'],
        'bias_detected': fairness_summary['bias_detected'],
        'optimal_threshold': fairness_summary['optimal_threshold'],
        'overall_performance': fairness_summary['overall_performance'],
        'worst_violations': fairness_summary['phase6_inputs']['worst_violations'],
        'group_metrics_summary': fairness_summary['phase6_inputs']['group_metrics_summary'],
        'phase4_results': phase4_results,
        'input_files': {
            'fairness_report': str(report_path),
            'statistical_tests': str(tests_path),
            'group_metrics_csvs': [str(output_dir / f"group_metrics_{attr}.csv") for attr in all_group_metrics.keys()],
            'risk_categories_csvs': [str(output_dir / f"risk_categories_{attr}.csv") for attr in risk_category_analysis.keys()]
        }
    }
    save_results(phase6_summary, str(phase6_summary_path))
    print(f"✅ Saved Phase 6 summary: {phase6_summary_path}")
    
    # ========================================================================
    # STEP 8: Generate Visualizations
    # ========================================================================
    
    print_section("Step 12: Generate Fairness Visualizations", "-")
    
    FairnessVisualizer.generate_all_visualizations(
        y_true=y_test.values,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        risk_categories=risk_categories,
        demographics=demographics,
        all_group_metrics=all_group_metrics,
        fairness_results=fairness_results,
        output_dir=str(output_dir)
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print_section("✅ Phase 5 Complete: Fairness Evaluation Summary", "=")
    
    print("📊 Key Findings:")
    print(f"   Bias Detected: {'YES ⚠️' if fairness_summary['bias_detected'] else 'NO ✅'}")
    print(f"   Overall ROC-AUC: {overall_metrics['roc_auc']:.3f}")
    print(f"   Overall TPR: {overall_metrics['tpr']:.3f}")
    print(f"   Overall FPR: {overall_metrics['fpr']:.3f}")
    
    print(f"\n📁 Outputs:")
    print(f"   Results: {args.output_dir}")
    print(f"   Fairness report: {report_path}")
    print(f"   Statistical tests: {tests_path}")
    print(f"   Phase 6 summary: {phase6_summary_path}")
    print(f"   Group metrics: group_metrics_*.csv")
    print(f"   Risk categories: risk_categories_*.csv")
    print(f"   Visualizations: {output_dir / 'visualizations'}/*.png")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(fairness_summary['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n🎯 Next Steps:")
    if fairness_summary['bias_detected']:
        priority = fairness_summary['phase6_inputs']['mitigation_priority'].upper()
        print(f"   ⚠️  MITIGATION REQUIRED - Priority: {priority}")
        print(f"   1. Proceed to Phase 6 (Fairness Mitigation)")
        print(f"   2. Use phase5_summary_for_phase6.json as input")
        print(f"   3. Calculate group-specific thresholds")
        print(f"   4. Consult with clinical and ethics teams")
        print(f"   5. Document fairness-ROI trade-offs")
    else:
        print(f"   ✅ NO MITIGATION NEEDED")
        print(f"   1. Skip Phase 6 (no fairness violations)")
        print(f"   2. Proceed to Phase 7 (Deployment Preparation)")
        print(f"   3. Use global optimal threshold from Phase 4")
        print(f"   4. Implement production monitoring for fairness metrics")
    
    # ========================================================================
    # Note: HuggingFace Upload
    # ========================================================================
    # Upload is now handled by the orchestrator script after combining
    # evaluation and mitigation results. This avoids duplicate uploads.
    # See: run_fairness_assessment_and_mitigation.sh
    
    print("\n" + "="*80)
    print("✅ Phase 5 Part A (Evaluation) Completed Successfully!")
    print("="*80)
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"� Upload will be handled by orchestrator after mitigation check")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
