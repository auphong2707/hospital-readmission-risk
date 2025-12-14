"""
Phase 6: Final System Evaluation for Gradient Boosting Model

This script performs comprehensive final evaluation of the deployed hospital readmission
prediction system using the threshold configuration determined in Phase 5 (either global
from Phase 4 or group-specific from Phase 5 mitigation).

Pipeline:
1. Load deployment configuration from Phase 5 (deployment_config.json)
2. Load test data and model predictions
3. Apply deployed thresholds (global or group-specific)
4. Calculate comprehensive metrics:
   - Performance: accuracy, precision, recall, F1, ROC-AUC, etc.
   - Calibration: Brier score, ECE, reliability diagrams
   - Group performance: metrics by race, gender, age
   - Fairness: TPR/FPR disparities, demographic parity
   - ROI: cost savings, financial impact
   - Risk stratification: performance by risk category
5. Generate 12 comprehensive visualizations
6. Create final_system_metrics.json (single source of truth for Phase 7)
7. Create deployment_report.json (for clinical stakeholders)

This is the FINAL evaluation - the metrics produced here are what will be published
and used for deployment decisions.

Usage (from project root):
    # Basic usage with auphong2707 repositories
    python ./phase-6-final-system-evaluation/final_evaluation_gradient_boosting.py
    
    # With custom repositories (if you forked the project)
    python ./phase-6-final-system-evaluation/final_evaluation_gradient_boosting.py \
        --data-repo-id your-username/hospital-readmission-risk-data \
        --model-repo-id your-username/hospital-readmission-lgbm-calibrated \
        --fairness-repo-id your-username/hospital-readmission-gradient-boosting-fairness-assessment-mitigation

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn huggingface_hub joblib

Phase 5 Output Required:
    - deployment_config.json from Phase 5 fairness assessment & mitigation
"""

import os
import sys
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Add parent directory to path for utilities import
sys.path.append(str(Path(__file__).parent))
from utilities import (
    DeploymentConfigLoader,
    ThresholdApplicator,
    FinalMetricsCalculator,
    ROICalculator,
    RiskCategoryAnalyzer,
    FinalEvaluationVisualizer,
    DeploymentReportGenerator
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Phase 6: Final System Evaluation for Gradient Boosting Model'
    )
    
    # Repository IDs
    parser.add_argument(
        '--data-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-risk-data',
        help='HuggingFace repository ID for data'
    )
    parser.add_argument(
        '--model-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-lgbm-calibrated',
        help='HuggingFace repository ID for calibrated model'
    )
    parser.add_argument(
        '--fairness-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-gradient-boosting-fairness-assessment-mitigation',
        help='HuggingFace repository ID for fairness assessment & mitigation results'
    )
    
    # Output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/gradient_boosting/final_evaluation',
        help='Directory to save evaluation results'
    )
    
    # Cost parameters (same as Phase 4)
    parser.add_argument(
        '--readmission-cost',
        type=float,
        default=15000,
        help='Cost of a readmission ($)'
    )
    parser.add_argument(
        '--intervention-cost',
        type=float,
        default=500,
        help='Cost of preventive intervention ($)'
    )
    
    return parser.parse_args()


def load_data_and_predictions(data_repo_id: str, model_repo_id: str):
    """
    Load test data and model predictions from HuggingFace.
    
    Args:
        data_repo_id: Repository ID for data
        model_repo_id: Repository ID for calibrated model
        
    Returns:
        Tuple of (y_true, y_proba, demographics)
    """
    print("\n" + "="*80)
    print("LOADING DATA AND PREDICTIONS")
    print("="*80)
    
    # Load test data demographics from Phase 1
    print("\n📥 Downloading test data from HuggingFace...")
    test_data_path = hf_hub_download(
        repo_id=data_repo_id,
        filename="splits/test.csv",
        repo_type="dataset"
    )
    test_data = pd.read_csv(test_data_path)
    print(f"✓ Loaded test data: {test_data.shape}")
    
    # Extract true labels and demographics
    y_true = test_data['readmitted'].values
    demographics = test_data[['race', 'gender', 'age']].copy()
    
    print(f"✓ True labels: {len(y_true)} samples")
    print(f"  - Readmission rate: {np.mean(y_true):.2%}")
    
    # Load calibrated predictions from Phase 3
    print("\n📥 Downloading calibrated predictions...")
    pred_path = hf_hub_download(
        repo_id=model_repo_id,
        filename="test_predictions_calibrated.csv",
        repo_type="model"
    )
    predictions = pd.read_csv(pred_path)
    y_proba = predictions['probability_calibrated'].values
    
    print(f"✓ Loaded calibrated predictions: {len(y_proba)} samples")
    print(f"  - Mean probability: {np.mean(y_proba):.4f}")
    print(f"  - Probability range: [{np.min(y_proba):.4f}, {np.max(y_proba):.4f}]")
    
    # Verify alignment
    assert len(y_true) == len(y_proba), "Mismatch between labels and predictions"
    assert len(y_true) == len(demographics), "Mismatch between labels and demographics"
    
    return y_true, y_proba, demographics


def load_deployment_config(fairness_repo_id: str, output_base: str) -> dict:
    """
    Load deployment configuration from Phase 5.
    
    Args:
        fairness_repo_id: Repository ID for fairness assessment outputs
        output_base: Base output directory (will download to parent)
        
    Returns:
        Deployment configuration dictionary
    """
    print("\n" + "="*80)
    print("LOADING DEPLOYMENT CONFIGURATION")
    print("="*80)
    
    # Download deployment config from Phase 5
    print("\n📥 Downloading deployment configuration from Phase 5...")
    config_path = hf_hub_download(
        repo_id=fairness_repo_id,
        filename="deployment_config.json",
        repo_type="model"
    )
    
    # Load config
    loader = DeploymentConfigLoader(config_path)
    config = loader.load_config()
    
    return config


def apply_deployed_thresholds(y_proba, demographics, config):
    """
    Apply deployed threshold configuration to predictions.
    
    Args:
        y_proba: Probability predictions
        demographics: Demographics dataframe
        config: Deployment configuration
        
    Returns:
        Tuple of (predictions, threshold_summary)
    """
    print("\n" + "="*80)
    print("APPLYING DEPLOYED THRESHOLDS")
    print("="*80)
    
    threshold_config = config['threshold_configuration']
    use_group_thresholds = config['use_group_thresholds']
    
    applicator = ThresholdApplicator(threshold_config, use_group_thresholds)
    y_pred = applicator.apply_thresholds(y_proba, demographics)
    threshold_summary = applicator.get_threshold_summary()
    
    print(f"\n✓ Generated predictions: {len(y_pred)} samples")
    print(f"  - Positive predictions: {np.sum(y_pred)} ({np.mean(y_pred):.2%})")
    
    return y_pred, threshold_summary


def calculate_all_metrics(y_true, y_pred, y_proba, demographics, cost_matrix):
    """
    Calculate comprehensive metrics for final evaluation.
    
    Args:
        y_true: True labels
        y_pred: Binary predictions
        y_proba: Probability predictions
        demographics: Demographics dataframe
        cost_matrix: Cost parameters for ROI calculation
        
    Returns:
        Dictionary containing all metrics
    """
    print("\n" + "="*80)
    print("CALCULATING COMPREHENSIVE METRICS")
    print("="*80)
    
    # Performance metrics
    print("\n📊 Calculating performance metrics...")
    metrics_calc = FinalMetricsCalculator()
    performance_metrics = metrics_calc.calculate_performance_metrics(y_true, y_pred, y_proba)
    print(f"✓ Performance metrics calculated")
    print(f"  - Accuracy: {performance_metrics['accuracy']:.4f}")
    print(f"  - ROC-AUC: {performance_metrics['roc_auc']:.4f}")
    print(f"  - Precision: {performance_metrics['precision']:.4f}")
    print(f"  - Recall: {performance_metrics['recall']:.4f}")
    
    # Calibration metrics
    print("\n📊 Calculating calibration metrics...")
    calibration_metrics = metrics_calc.calculate_calibration_metrics(y_true, y_proba)
    print(f"✓ Calibration metrics calculated")
    print(f"  - Brier score: {calibration_metrics['brier_score']:.4f}")
    print(f"  - ECE: {calibration_metrics['expected_calibration_error']:.4f}")
    
    # Group-specific metrics
    print("\n📊 Calculating group-specific metrics...")
    group_metrics = metrics_calc.calculate_group_metrics(y_true, y_pred, y_proba, demographics)
    print(f"✓ Group metrics calculated for {len(group_metrics)} features")
    for feature, groups in group_metrics.items():
        print(f"  - {feature.capitalize()}: {len(groups)} groups")
    
    # Fairness metrics
    print("\n📊 Calculating fairness metrics...")
    fairness_metrics = metrics_calc.calculate_fairness_metrics(group_metrics)
    print(f"✓ Fairness metrics calculated")
    for feature, metrics in fairness_metrics.items():
        print(f"  - {feature.capitalize()} TPR disparity: {metrics['tpr_disparity']:.4f}")
    
    # ROI metrics
    print("\n📊 Calculating ROI metrics...")
    roi_calc = ROICalculator(cost_matrix)
    roi_metrics = roi_calc.calculate_roi_metrics(y_true, y_pred)
    print(f"✓ ROI metrics calculated")
    print(f"  - Total cost: ${roi_metrics['total_cost']:,.0f}")
    print(f"  - Cost savings: ${roi_metrics['cost_savings']:,.0f}")
    print(f"  - ROI: {roi_metrics['roi_percentage']:.2f}%")
    
    # Risk stratification analysis
    print("\n📊 Analyzing risk stratification...")
    risk_boundaries = {
        'low': (0.0, 0.3),
        'medium': (0.3, 0.7),
        'high': (0.7, 1.0)
    }
    risk_analyzer = RiskCategoryAnalyzer(risk_boundaries)
    risk_analysis = risk_analyzer.analyze_risk_distribution(y_true, y_proba)
    print(f"✓ Risk stratification analyzed")
    for level, stats in risk_analysis.items():
        print(f"  - {level.capitalize()} risk: {stats['count']} patients ({stats['percentage']:.1f}%)")
    
    return {
        'performance': performance_metrics,
        'calibration': calibration_metrics,
        'group_metrics': group_metrics,
        'fairness': fairness_metrics,
        'roi': roi_metrics,
        'risk_stratification': risk_analysis
    }


def generate_visualizations(y_true, y_pred, y_proba, demographics, metrics, 
                           threshold_summary, output_dir):
    """
    Generate comprehensive visualizations for final evaluation.
    
    Args:
        y_true: True labels
        y_pred: Binary predictions
        y_proba: Probability predictions
        demographics: Demographics dataframe
        metrics: Dictionary of all calculated metrics
        threshold_summary: Summary of deployed thresholds
        output_dir: Directory to save visualizations
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    visualizer = FinalEvaluationVisualizer(viz_dir)
    
    viz_count = 0
    
    # 1. Confusion Matrix
    print("\n📊 Creating confusion matrix...")
    visualizer.plot_confusion_matrix(y_true, y_pred, "Final System Confusion Matrix")
    viz_count += 1
    
    # 2. Calibration Curve
    print("📊 Creating calibration curve...")
    visualizer.plot_calibration_curve(y_true, y_proba)
    viz_count += 1
    
    # 3-5. Group Performance Comparisons (TPR, FPR, Precision)
    print("📊 Creating group performance comparisons...")
    for metric in ['tpr', 'fpr', 'precision']:
        visualizer.plot_group_performance_comparison(metrics['group_metrics'], metric)
        viz_count += 1
    
    # 6. Fairness Disparities
    print("📊 Creating fairness disparity visualization...")
    visualizer.plot_fairness_disparities(metrics['fairness'])
    viz_count += 1
    
    # 7. ROI Breakdown
    print("📊 Creating ROI breakdown...")
    visualizer.plot_roi_breakdown(metrics['roi'])
    viz_count += 1
    
    # 8. Risk Distribution
    print("📊 Creating risk distribution visualization...")
    visualizer.plot_risk_distribution(metrics['risk_stratification'])
    viz_count += 1
    
    # 9. Threshold Configuration
    print("📊 Creating threshold configuration visualization...")
    visualizer.plot_threshold_comparison(threshold_summary, demographics)
    viz_count += 1
    
    print(f"\n✓ Generated {viz_count} visualizations in {viz_dir}/")


def generate_reports(model_name, config, metrics, threshold_summary, output_dir):
    """
    Generate final evaluation reports.
    
    Args:
        model_name: Name of the model
        config: Deployment configuration
        metrics: Dictionary of all calculated metrics
        threshold_summary: Summary of deployed thresholds
        output_dir: Directory to save reports
    """
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    
    report_gen = DeploymentReportGenerator(output_dir)
    
    # Generate final_system_metrics.json (single source of truth)
    print("\n📝 Creating final_system_metrics.json...")
    report_gen.generate_final_metrics_json(
        model_name=model_name,
        deployment_config=config,
        performance_metrics=metrics['performance'],
        calibration_metrics=metrics['calibration'],
        group_metrics=metrics['group_metrics'],
        fairness_metrics=metrics['fairness'],
        roi_metrics=metrics['roi'],
        risk_analysis=metrics['risk_stratification'],
        threshold_summary=threshold_summary
    )
    
    # Generate deployment_report.json (for stakeholders)
    print("📝 Creating deployment_report.json...")
    report_gen.generate_deployment_report_json(
        model_name=model_name,
        deployment_config=config,
        performance_metrics=metrics['performance'],
        fairness_metrics=metrics['fairness'],
        roi_metrics=metrics['roi'],
        risk_analysis=metrics['risk_stratification']
    )
    
    print("\n✓ Reports generated successfully")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("PHASE 6: FINAL SYSTEM EVALUATION - GRADIENT BOOSTING")
    print("="*80)
    print(f"Data Repository: {args.data_repo_id}")
    print(f"Model Repository: {args.model_repo_id}")
    print(f"Fairness Repository: {args.fairness_repo_id}")
    print(f"Output Directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define cost matrix
    cost_matrix = {
        'TP': args.intervention_cost - args.readmission_cost,  # +$14,500 (saved readmission)
        'TN': 0,  # No cost
        'FP': -args.intervention_cost,  # -$500 (unnecessary intervention)
        'FN': -args.readmission_cost  # -$15,000 (missed readmission)
    }
    
    print("\n💰 Cost Matrix:")
    for outcome, cost in cost_matrix.items():
        print(f"  {outcome}: ${cost:,.0f}")
    
    try:
        # Step 1: Load data and predictions
        y_true, y_proba, demographics = load_data_and_predictions(
            args.data_repo_id, args.model_repo_id
        )
        
        # Step 2: Load deployment configuration
        config = load_deployment_config(args.fairness_repo_id, args.output_dir)
        
        # Step 3: Apply deployed thresholds
        y_pred, threshold_summary = apply_deployed_thresholds(
            y_proba, demographics, config
        )
        
        # Step 4: Calculate all metrics
        metrics = calculate_all_metrics(
            y_true, y_pred, y_proba, demographics, cost_matrix
        )
        
        # Step 5: Generate visualizations
        generate_visualizations(
            y_true, y_pred, y_proba, demographics, metrics,
            threshold_summary, args.output_dir
        )
        
        # Step 6: Generate reports
        generate_reports(
            "Gradient Boosting (LightGBM)",
            config,
            metrics,
            threshold_summary,
            args.output_dir
        )
        
        print("\n" + "="*80)
        print("✅ FINAL SYSTEM EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\n📁 All outputs saved to: {args.output_dir}/")
        print(f"  - final_system_metrics.json (single source of truth)")
        print(f"  - deployment_report.json (stakeholder summary)")
        print(f"  - visualizations/ ({9} plots)")
        
        # Print key results
        print("\n" + "="*80)
        print("KEY RESULTS SUMMARY")
        print("="*80)
        print(f"\n🎯 Performance:")
        print(f"  - Accuracy: {metrics['performance']['accuracy']:.2%}")
        print(f"  - ROC-AUC: {metrics['performance']['roc_auc']:.4f}")
        print(f"  - Sensitivity: {metrics['performance']['sensitivity']:.2%}")
        print(f"  - Specificity: {metrics['performance']['specificity']:.2%}")
        
        print(f"\n⚖️ Fairness:")
        for feature, fm in metrics['fairness'].items():
            print(f"  - {feature.capitalize()} TPR Disparity: {fm['tpr_disparity']:.4f}")
            print(f"  - {feature.capitalize()} FPR Disparity: {fm['fpr_disparity']:.4f}")
        
        print(f"\n💰 Financial Impact:")
        print(f"  - ROI: {metrics['roi']['roi_percentage']:.2f}%")
        print(f"  - Cost Savings: ${metrics['roi']['cost_savings']:,.0f}")
        print(f"  - Avg Cost per Patient: ${metrics['roi']['avg_cost_per_patient']:,.2f}")
        
        print(f"\n📊 Risk Stratification:")
        for level, stats in metrics['risk_stratification'].items():
            print(f"  - {level.capitalize()}: {stats['count']} patients "
                  f"({stats['percentage']:.1f}%), "
                  f"readmission rate: {stats['actual_readmission_rate']:.2%}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
