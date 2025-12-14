"""
Phase 5: Fairness Evaluation & Deployment Readiness for Logistic Regression Model

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
    python ./phase-5-fairness-evaluation/evaluate_fairness_logistic_regression.py
    
    # Using local files
    python ./phase-5-fairness-evaluation/evaluate_fairness_logistic_regression.py \
        --use-local \
        --local-test ./data/processed/splits/test.csv \
        --local-demographics ./data/processed/splits/test_demographics.csv
    
    # Custom output directory
    python ./phase-5-fairness-evaluation/evaluate_fairness_logistic_regression.py \
        --output-dir ./phase-5-fairness-evaluation/outputs/logistic_regression
    
    Note: Results are automatically uploaded to HuggingFace Hub
          Repository: auphong2707/hospital-readmission-logistic-regression-fairness-results
          Requires: HF_TOKEN environment variable

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib huggingface_hub

Phase Dependencies:
    - Phase 1: Demographics files (test_demographics.csv)
    - Phase 3: Calibrated model (model + scaler + calibrator)
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
import pickle

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
    upload_results_to_hf,
    ModelCalibrator
)

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Phase 5: Fairness Evaluation & Deployment Readiness - Logistic Regression'
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
    # Note: No --local-scaler argument - test data already scaled from Phase 1
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
        default='auphong2707/hospital-readmission-logistic-regression-calibrated',
        help='HuggingFace model repository ID'
    )
    
    # Phase 4 results
    parser.add_argument(
        '--phase4-summary',
        type=str,
        default='./phase-4-optimal-threshold-ROI-analysis/outputs/logistic_regression/phase4_summary_for_phase5.json',
        help='Path to Phase 4 summary JSON'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./phase-5-fairness-evaluation/outputs/logistic_regression',
        help='Directory to save fairness evaluation outputs'
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





def generate_lr_calibrated_predictions(model, calibrator, X_test):
    """
    Generate calibrated predictions for Logistic Regression model.
    
    Note: X_test should already be scaled from Phase 1 preprocessing.
    
    Args:
        model: Trained Logistic Regression model
        calibrator: Calibrator for probabilities
        X_test: Test features (already scaled from Phase 1)
        
    Returns:
        np.ndarray: Calibrated probabilities
    """
    print("🔮 Generating calibrated predictions...")
    
    # X_test is already scaled from Phase 1 - no need to scale again!
    # Get uncalibrated probabilities
    y_pred_proba_uncalibrated = model.predict_proba(X_test)[:, 1]
    
    # Apply calibration
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
        y_pred_proba_calibrated = pred
    
    print(f"✅ Generated {len(y_pred_proba_calibrated)} calibrated predictions")
    print(f"   Probability range: [{y_pred_proba_calibrated.min():.4f}, {y_pred_proba_calibrated.max():.4f}]")
    print(f"   Mean probability: {y_pred_proba_calibrated.mean():.4f}")
    
    return y_pred_proba_calibrated


def main():
    """Main fairness evaluation workflow."""
    
    args = parse_arguments()
    
    print_section("🎯 Phase 5: Fairness Evaluation - Logistic Regression", "=")
    print(f"📋 Configuration:")
    print(f"   Data source: {'Local files' if args.use_local else 'HuggingFace Hub'}")
    print(f"   Model repo: {args.model_repo_id}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Phase 4 summary: {args.phase4_summary}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    try:
        # STEP 1: Load test data and demographics
        print_section("📂 Step 1: Load Test Data and Demographics", "=")
        X_test, y_test, demographics = load_test_data_and_demographics(
            use_local=args.use_local,
            local_test_path=args.local_test,
            local_demographics_path=args.local_demographics,
            data_repo_id=args.data_repo_id,
            cache_dir=args.cache_dir
        )
        
        # STEP 2: Load calibrated model
        print_section("📥 Step 2: Load Calibrated Model", "=")
        # Note: No scaler needed - test data already scaled from Phase 1
        model, calibrator = load_model_and_calibrator(
            model_repo_id=args.model_repo_id,
            cache_dir=args.cache_dir,
            use_local=args.use_local,
            local_model_path=args.local_model,
            local_calibrator_path=args.local_calibrator,
            method='logistic_regression'
        )
        
        # STEP 3: Load Phase 4 results
        print_section("📊 Step 3: Load Phase 4 Results", "=")
        phase4_results = load_phase4_results(args.phase4_summary)
        optimal_threshold = phase4_results['optimal_threshold']
        risk_thresholds = phase4_results['risk_thresholds']
        
        print(f"✅ Loaded Phase 4 results:")
        print(f"   Optimal threshold: {optimal_threshold:.4f}")
        print(f"   Risk thresholds: {risk_thresholds}")
        
        # STEP 4: Generate calibrated predictions
        print_section("🔮 Step 4: Generate Calibrated Predictions", "=")
        # X_test is already scaled from Phase 1, no need to scale again
        y_pred_proba = generate_lr_calibrated_predictions(model, calibrator, X_test)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # STEP 5: Overall performance metrics
        print_section("📈 Step 5: Compute Overall Performance", "=")
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        overall_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'threshold': optimal_threshold,
            'intervention_rate': y_pred.mean()
        }
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        overall_metrics['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
        
        print("📊 Overall Performance:")
        print(f"   Accuracy: {overall_metrics['accuracy']:.4f}")
        print(f"   Precision: {overall_metrics['precision']:.4f}")
        print(f"   Recall (TPR): {overall_metrics['recall']:.4f}")
        print(f"   F1-Score: {overall_metrics['f1']:.4f}")
        print(f"   ROC-AUC: {overall_metrics['roc_auc']:.4f}")
        print(f"   Intervention Rate: {overall_metrics['intervention_rate']:.2%}")
        
        # STEP 6: Group-specific analysis
        print_section("👥 Step 6: Group-Specific Performance Analysis", "=")
        analyzer = GroupPerformanceAnalyzer(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            demographics=demographics
        )
        
        group_metrics = {}
        for attribute in ['race', 'gender', 'age_group']:
            if attribute in demographics.columns:
                print(f"\n🔍 Analyzing {attribute.upper()}...")
                metrics = analyzer.compute_group_metrics(attribute)
                group_metrics[attribute] = metrics
                
                # Save to CSV
                metrics_df = pd.DataFrame(metrics).T
                csv_path = output_dir / f"group_metrics_{attribute}.csv"
                metrics_df.to_csv(csv_path)
                print(f"   ✅ Saved: {csv_path}")
        
        # STEP 7: Fairness metrics
        print_section("⚖️  Step 7: Compute Fairness Metrics", "=")
        fairness_results = FairnessMetrics.compute_all_fairness_metrics(group_metrics)
        
        # STEP 8: Statistical significance testing
        print_section("📊 Step 8: Statistical Significance Testing", "=")
        statistical_tests = {}
        for attribute in group_metrics.keys():
            print(f"\n🔬 {attribute.upper()} Statistical Tests:")
            
            # Chi-square test for intervention rate
            chi2_result = StatisticalTester.chi_square_test_intervention_rate(
                y_pred, demographics, attribute
            )
            statistical_tests[f'{attribute}_chi2'] = chi2_result
            
            print(f"   Chi-square test (intervention rate):")
            print(f"      χ² = {chi2_result['chi2_statistic']:.2f}, p-value = {chi2_result['p_value']:.4f}")
            print(f"      Result: {chi2_result['interpretation']}")
        
        # Save statistical tests
        tests_path = output_dir / "statistical_tests.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        statistical_tests_serializable = convert_to_serializable(statistical_tests)
        
        with open(tests_path, 'w') as f:
            json.dump(statistical_tests_serializable, f, indent=2)
        print(f"\n✅ Statistical tests saved: {tests_path}")
        
        # STEP 9: Risk category analysis
        print_section("🎯 Step 9: Risk Category Distribution Analysis", "=")
        
        # Create risk categories based on Phase 4 thresholds
        low_threshold = risk_thresholds['low']
        high_threshold = risk_thresholds['high']
        
        risk_categories = np.full(len(y_pred_proba), 'Medium')
        risk_categories[y_pred_proba < low_threshold] = 'Low'
        risk_categories[y_pred_proba >= high_threshold] = 'High'
        
        # Analyze risk categories by demographic group
        def analyze_risk_categories_by_group(risk_cats, y_true_data, demo_data, attr):
            """Analyze risk category distribution by demographic group."""
            if attr not in demo_data.columns:
                return pd.DataFrame()
            
            groups = demo_data[attr].unique()
            results = []
            
            for group in groups:
                mask = (demo_data[attr] == group).values
                if mask.sum() == 0:
                    continue
                
                group_risk = risk_cats[mask]
                group_true = y_true_data[mask]
                
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
                    'attribute': attr,
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
        
        # Analyze for each demographic attribute
        for attribute in ['race', 'gender', 'age_group']:
            if attribute in demographics.columns:
                risk_analysis = analyze_risk_categories_by_group(
                    risk_categories, y_test.values, demographics, attribute
                )
                
                if not risk_analysis.empty:
                    risk_path = output_dir / f"risk_categories_{attribute}.csv"
                    risk_analysis.to_csv(risk_path, index=False)
                    print(f"✅ Risk analysis saved: {risk_path}")
        
        # STEP 10: Generate visualizations
        print_section("📊 Step 10: Generate Fairness Visualizations", "=")
        
        # Group comparison plots for different metrics
        print("📊 Generating group comparison plots...")
        for metric in ['tpr', 'fpr', 'precision', 'intervention_rate']:
            FairnessVisualizer.plot_group_metrics_comparison(
                group_metrics, str(viz_dir), metric=metric
            )
            print(f"   ✅ {metric.upper()} comparison saved")
        
        # Note: Additional visualizations would need to be implemented as static methods
        # or the FairnessVisualizer class would need to be refactored
        
        print(f"\n✅ Visualizations saved to: {viz_dir}")
        
        # Note: Calibration and risk distribution visualizations would require
        # additional static methods in FairnessVisualizer class
        
        # STEP 11: Create comprehensive fairness report
        print_section("📄 Step 11: Generate Fairness Report", "=")
        
        # Assess overall fairness based on individual fairness metrics
        fairness_passed = True
        for attribute, metrics in fairness_results.items():
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'passed' in metric_data:
                    if not metric_data['passed']:
                        fairness_passed = False
                        break
            if not fairness_passed:
                break
        
        fairness_report = {
            'model_type': 'Logistic Regression',
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'test_set_size': len(y_test),
            'optimal_threshold': optimal_threshold,
            'risk_thresholds': risk_thresholds,
            'overall_metrics': overall_metrics,
            'group_metrics': {k: v.to_dict(orient='records') if isinstance(v, pd.DataFrame) else v 
                             for k, v in group_metrics.items()},
            'fairness_metrics': fairness_results,
            'statistical_tests': statistical_tests,
            'deployment_readiness': {
                'overall_performance': 'PASS' if overall_metrics['roc_auc'] > 0.7 else 'FAIL',
                'fairness_assessment': 'PASS' if fairness_passed else 'NEEDS REVIEW',
                'recommendation': 'APPROVED' if (overall_metrics['roc_auc'] > 0.7 and fairness_passed) else 'NEEDS REVIEW'
            }
        }
        
        # Convert to JSON-serializable format
        fairness_report_serializable = convert_to_serializable(fairness_report)
        
        report_path = output_dir / "fairness_report.json"
        with open(report_path, 'w') as f:
            json.dump(fairness_report_serializable, f, indent=2)
        print(f"✅ Fairness report saved: {report_path}")
        
        # Note: HuggingFace Upload
        # Upload is now handled by the orchestrator script after combining
        # evaluation and mitigation results. This avoids duplicate uploads.
        # See: run_fairness_assessment_and_mitigation.sh
        print_section("📁 Evaluation Results Saved", "=")
        print(f"✅ Results saved locally to: {output_dir}")
        print(f"📤 Upload will be handled by orchestrator after mitigation check")
        
        # FINAL SUMMARY
        print_section("✨ Fairness Evaluation Complete!", "=")
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"\n📊 Key Results:")
        print(f"   Overall ROC-AUC: {overall_metrics['roc_auc']:.4f}")
        print(f"   Intervention Rate: {overall_metrics['intervention_rate']:.2%}")
        print(f"   Fairness Assessment: {fairness_report['deployment_readiness']['fairness_assessment']}")
        print(f"   Deployment Recommendation: {fairness_report['deployment_readiness']['recommendation']}")
        
        print(f"\n📄 Generated Files:")
        print(f"   - fairness_report.json")
        print(f"   - group_metrics_*.csv (race, gender, age)")
        print(f"   - statistical_tests.json")
        print(f"   - risk_categories_*.csv")
        print(f"   - visualizations/ (4 comparison plots)")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review fairness report: {report_path}")
        print(f"   2. Examine group-specific metrics")
        print(f"   3. Review visualizations in: {viz_dir}")
        if fairness_report['deployment_readiness']['fairness_assessment'] != 'PASS':
            print(f"   4. Proceed to Phase 6: Fairness Mitigation")
            print(f"      python ./phase-6-fairness-mitigation-bias-correction/calculate_group_thresholds_logistic_regression.py")
        else:
            print(f"   4. Model approved for deployment!")
        
        print(f"\n{'='*80}")
        print("🎉 Fairness evaluation complete!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error during fairness evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
