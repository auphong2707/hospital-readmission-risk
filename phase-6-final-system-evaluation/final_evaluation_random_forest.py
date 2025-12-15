"""
Phase 6: Final System Evaluation for Random Forest Model

This script performs comprehensive final evaluation of the deployed hospital readmission
prediction system using the threshold configuration determined in Phase 5 (either global
from Phase 4 or group-specific from Phase 5 mitigation).

Usage (from project root):
    python ./phase-6-final-system-evaluation/final_evaluation_random_forest.py
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Add parent directory to path for utilities import
sys.path.append(str(Path(__file__).parent))
from utilities import (
    load_data_and_predictions,
    load_deployment_config,
    apply_deployed_thresholds,
    calculate_all_metrics,
    generate_visualizations,
    generate_reports,
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
        description='Phase 6: Final System Evaluation for Random Forest Model'
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
        default='auphong2707/hospital-readmission-phase3-rf-calibrated',
        help='HuggingFace repository ID for calibrated model'
    )
    parser.add_argument(
        '--fairness-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-phase5-rf-fairness',
        help='HuggingFace repository ID for fairness assessment & mitigation results'
    )
    
    # Output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/random_forest/final_evaluation',
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


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("PHASE 6: FINAL SYSTEM EVALUATION - RANDOM FOREST")
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
            "Random Forest",
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
