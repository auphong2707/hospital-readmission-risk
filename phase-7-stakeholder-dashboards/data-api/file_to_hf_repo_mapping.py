"""
File to HuggingFace Repository Mapping
======================================
This dictionary maps files from each phase to their corresponding HuggingFace repositories.
The structure is organized by phase and method (gradient_boosting, random_forest, logistic_regression).
"""

# Repository mappings by method
REPO_MAPPING = {
    "gradient_boosting": {
        "phase1": {
            "repo": "auphong2707/hospital-readmission-risk-data",
            "repo_type": "dataset",
            "files": {
                "split_info.txt": "splits/split_info.txt",
            }
        },
        "phase2": {
            "repo": "auphong2707/hospital-readmission-phase2-lgbm",
            "repo_type": "model",
            "files": {
                "gradient_boosting_model.joblib": "gradient_boosting_model.joblib",
                "gradient_boosting_metrics.json": "gradient_boosting_metrics.json",
                "cv_fold_details.json": "cv_fold_details.json",
                "training_summary.json": "training_summary.json",
                # Visualizations
                "roc_curve.png": "roc_curve.png",
                "precision_recall_curve.png": "precision_recall_curve.png",
                "confusion_matrix.png": "confusion_matrix.png",
                "calibration_curve.png": "calibration_curve.png",
                "feature_importance.png": "feature_importance.png",
                "feature_importance.csv": "feature_importance.csv",
                "learning_curves.png": "learning_curves.png",
                "metrics_comparison_across_folds.png": "metrics_comparison_across_folds.png",
                "validation_curves.png": "validation_curves.png",
            }
        },
        "phase3": {
            "repo": "auphong2707/hospital-readmission-phase3-lgbm-calibrated",
            "repo_type": "model",
            "files": {
                "gradient_boosting_model_original.joblib": "gradient_boosting_model_original.joblib",
                "Gradient_Boosting_(LightGBM)_calibrator.pkl": "Gradient_Boosting_(LightGBM)_calibrator.pkl",
                "Gradient_Boosting_(LightGBM)_metrics.json": "Gradient_Boosting_(LightGBM)_metrics.json",
                "Gradient_Boosting_(LightGBM)_report.txt": "Gradient_Boosting_(LightGBM)_report.txt",
                "calibration_comparison_metrics.json": "calibration_comparison_metrics.json",
                # Visualizations
                "Gradient_Boosting_(LightGBM)_reliability_diagram.png": "Gradient_Boosting_(LightGBM)_reliability_diagram.png",
                "Gradient_Boosting_(LightGBM)_risk_distribution.png": "Gradient_Boosting_(LightGBM)_risk_distribution.png",
                "Gradient_Boosting_(LightGBM)_risk_validation.csv": "Gradient_Boosting_(LightGBM)_risk_validation.csv",
                "01_reliability_diagram_before_after.png": "01_reliability_diagram_before_after.png",
                "02_calibration_improvement_metrics.png": "02_calibration_improvement_metrics.png",
                "03_probability_distribution_changes.png": "03_probability_distribution_changes.png",
                "reliability_diagram_comparison.png": "reliability_diagram_comparison.png",
                "risk_distribution_detailed.png": "risk_distribution_detailed.png",
                "risk_validation_detailed.csv": "risk_validation_detailed.csv",
            }
        },
        "phase4": {
            "repo": "auphong2707/hospital-readmission-phase4-lgbm-threshold",
            "repo_type": "model",
            "files": {
                "threshold_results.csv": "outputs/threshold_results.csv",
                "optimal_thresholds.json": "outputs/optimal_thresholds.json",
                "roi_metrics.json": "outputs/roi_metrics.json",
                "roi_report.txt": "outputs/roi_report.txt",
                "phase4_summary_for_phase5.json": "outputs/phase4_summary_for_phase5.json",
                # Visualizations
                "1_expected_value_curve.png": "visualizations/1_expected_value_curve.png",
                "2_cost_benefit_analysis.png": "visualizations/2_cost_benefit_analysis.png",
                "3_metrics_vs_threshold.png": "visualizations/3_metrics_vs_threshold.png",
                "4_confusion_matrix.png": "visualizations/4_confusion_matrix.png",
                "5_risk_category_distribution.png": "visualizations/5_risk_category_distribution.png",
                "6_roi_sensitivity_analysis.png": "visualizations/6_roi_sensitivity_analysis.png",
                "7_intervention_volume_forecast.png": "visualizations/7_intervention_volume_forecast.png",
                "8_cost_savings_projection.png": "visualizations/8_cost_savings_projection.png",
            }
        },
        "phase5": {
            "repo": "auphong2707/hospital-readmission-gradient-boosting-final",
            "repo_type": "model",
            "files": {
                "deployment_config.json": "phase5_fairness_assessment/deployment_config.json",
                # Evaluation subfolder
                "fairness_report.json": "phase5_fairness_assessment/evaluation/fairness_report.json",
                "group_metrics_age.csv": "phase5_fairness_assessment/evaluation/group_metrics_age.csv",
                "group_metrics_gender.csv": "phase5_fairness_assessment/evaluation/group_metrics_gender.csv",
                "group_metrics_race.csv": "phase5_fairness_assessment/evaluation/group_metrics_race.csv",
                "statistical_tests.json": "phase5_fairness_assessment/evaluation/statistical_tests.json",
                "risk_categories_age.csv": "phase5_fairness_assessment/evaluation/risk_categories_age.csv",
                "risk_categories_gender.csv": "phase5_fairness_assessment/evaluation/risk_categories_gender.csv",
                "risk_categories_race.csv": "phase5_fairness_assessment/evaluation/risk_categories_race.csv",
                "phase5_summary_for_phase6.json": "phase5_fairness_assessment/evaluation/phase5_summary_for_phase6.json",
                # Mitigation subfolder
                "group_thresholds.json": "phase5_fairness_assessment/mitigation/group_thresholds.json",
                "mitigation_impact.json": "phase5_fairness_assessment/mitigation/mitigation_impact.json",
            }
        },
        "phase6": {
            "repo": "auphong2707/hospital-readmission-gradient-boosting-final",
            "repo_type": "model",
            "files": {
                "final_system_metrics.json": "phase6_final_evaluation/final_system_metrics.json",
                "deployment_report.json": "phase6_final_evaluation/deployment_report.json",
                # Visualizations
                "calibration_curve.png": "phase6_final_evaluation/visualizations/calibration_curve.png",
                "confusion_matrix.png": "phase6_final_evaluation/visualizations/confusion_matrix.png",
                "fairness_disparities.png": "phase6_final_evaluation/visualizations/fairness_disparities.png",
                "group_fpr_comparison.png": "phase6_final_evaluation/visualizations/group_fpr_comparison.png",
                "group_precision_comparison.png": "phase6_final_evaluation/visualizations/group_precision_comparison.png",
                "group_tpr_comparison.png": "phase6_final_evaluation/visualizations/group_tpr_comparison.png",
                "risk_distribution.png": "phase6_final_evaluation/visualizations/risk_distribution.png",
                "roi_breakdown.png": "phase6_final_evaluation/visualizations/roi_breakdown.png",
                "threshold_configuration.png": "phase6_final_evaluation/visualizations/threshold_configuration.png",
            }
        }
    },
    "random_forest": {
        "phase1": {
            "repo": "auphong2707/hospital-readmission-risk-data",
            "repo_type": "dataset",
            "files": {
                "split_info.txt": "splits/split_info.txt",
            }
        },
        "phase2": {
            "repo": "auphong2707/hospital-readmission-phase2-rf",
            "repo_type": "model",
            "files": {
                "random_forest_model.joblib": "random_forest_model.joblib",
                "random_forest_metrics.json": "random_forest_metrics.json",
                "cv_fold_details.json": "cv_fold_details.json",
                "training_summary.json": "training_summary.json",
                # Visualizations
                "roc_curve.png": "roc_curve.png",
                "precision_recall_curve.png": "precision_recall_curve.png",
                "confusion_matrix.png": "confusion_matrix.png",
                "calibration_curve.png": "calibration_curve.png",
                "feature_importance.png": "feature_importance.png",
                "feature_importance.csv": "feature_importance.csv",
                "learning_curves.png": "learning_curves.png",
                "metrics_comparison_across_folds.png": "metrics_comparison_across_folds.png",
                "validation_curves.png": "validation_curves.png",
            }
        },
        "phase3": {
            "repo": "auphong2707/hospital-readmission-phase3-rf-calibrated",
            "repo_type": "model",
            "files": {
                "random_forest_model_original.joblib": "random_forest_model_original.joblib",
                "Random_Forest_calibrator.pkl": "Random_Forest_calibrator.pkl",
                "Random_Forest_metrics.json": "Random_Forest_metrics.json",
                "Random_Forest_report.txt": "Random_Forest_report.txt",
                "calibration_comparison_metrics.json": "calibration_comparison_metrics.json",
                # Visualizations
                "Random_Forest_reliability_diagram.png": "Random_Forest_reliability_diagram.png",
                "Random_Forest_risk_distribution.png": "Random_Forest_risk_distribution.png",
                "Random_Forest_risk_validation.csv": "Random_Forest_risk_validation.csv",
                "01_reliability_diagram_before_after.png": "01_reliability_diagram_before_after.png",
                "02_calibration_improvement_metrics.png": "02_calibration_improvement_metrics.png",
                "03_probability_distribution_changes.png": "03_probability_distribution_changes.png",
                "reliability_diagram_comparison.png": "reliability_diagram_comparison.png",
                "risk_distribution_detailed.png": "risk_distribution_detailed.png",
                "risk_validation_detailed.csv": "risk_validation_detailed.csv",
            }
        },
        "phase4": {
            "repo": "auphong2707/hospital-readmission-phase4-rf-threshold",
            "repo_type": "model",
            "files": {
                "threshold_results.csv": "outputs/threshold_results.csv",
                "optimal_thresholds.json": "outputs/optimal_thresholds.json",
                "roi_metrics.json": "outputs/roi_metrics.json",
                "roi_report.txt": "outputs/roi_report.txt",
                "phase4_summary_for_phase5.json": "outputs/phase4_summary_for_phase5.json",
                # Visualizations
                "1_expected_value_curve.png": "visualizations/1_expected_value_curve.png",
                "2_cost_benefit_analysis.png": "visualizations/2_cost_benefit_analysis.png",
                "3_metrics_vs_threshold.png": "visualizations/3_metrics_vs_threshold.png",
                "4_confusion_matrix.png": "visualizations/4_confusion_matrix.png",
                "5_risk_category_distribution.png": "visualizations/5_risk_category_distribution.png",
                "6_roi_sensitivity_analysis.png": "visualizations/6_roi_sensitivity_analysis.png",
                "7_intervention_volume_forecast.png": "visualizations/7_intervention_volume_forecast.png",
                "8_cost_savings_projection.png": "visualizations/8_cost_savings_projection.png",
            }
        },
        "phase5": {
            "repo": "auphong2707/hospital-readmission-rf-final",
            "repo_type": "model",
            "files": {
                "deployment_config.json": "phase5_fairness_assessment/deployment_config.json",
                # Evaluation files
                "fairness_report.json": "phase5_fairness_assessment/evaluation/fairness_report.json",
                "group_metrics_age.csv": "phase5_fairness_assessment/evaluation/group_metrics_age.csv",
                "group_metrics_gender.csv": "phase5_fairness_assessment/evaluation/group_metrics_gender.csv",
                "group_metrics_race.csv": "phase5_fairness_assessment/evaluation/group_metrics_race.csv",
                "statistical_tests.json": "phase5_fairness_assessment/evaluation/statistical_tests.json",
                "risk_categories_age.csv": "phase5_fairness_assessment/evaluation/risk_categories_age.csv",
                "risk_categories_gender.csv": "phase5_fairness_assessment/evaluation/risk_categories_gender.csv",
                "risk_categories_race.csv": "phase5_fairness_assessment/evaluation/risk_categories_race.csv",
                "phase5_summary_for_phase6.json": "phase5_fairness_assessment/evaluation/phase5_summary_for_phase6.json",
                # Mitigation files
                "group_thresholds.json": "phase5_fairness_assessment/mitigation/group_thresholds.json",
                "mitigation_impact.json": "phase5_fairness_assessment/mitigation/mitigation_impact.json",
            }
        },
        "phase6": {
            "repo": "auphong2707/hospital-readmission-rf-final",
            "repo_type": "model",
            "files": {
                "final_system_metrics.json": "phase6_final_evaluation/final_system_metrics.json",
                "deployment_report.json": "phase6_final_evaluation/deployment_report.json",
                # Visualizations
                "calibration_curve.png": "phase6_final_evaluation/visualizations/calibration_curve.png",
                "confusion_matrix.png": "phase6_final_evaluation/visualizations/confusion_matrix.png",
                "fairness_disparities.png": "phase6_final_evaluation/visualizations/fairness_disparities.png",
                "group_fpr_comparison.png": "phase6_final_evaluation/visualizations/group_fpr_comparison.png",
                "group_precision_comparison.png": "phase6_final_evaluation/visualizations/group_precision_comparison.png",
                "group_tpr_comparison.png": "phase6_final_evaluation/visualizations/group_tpr_comparison.png",
                "risk_distribution.png": "phase6_final_evaluation/visualizations/risk_distribution.png",
                "roi_breakdown.png": "phase6_final_evaluation/visualizations/roi_breakdown.png",
                "threshold_configuration.png": "phase6_final_evaluation/visualizations/threshold_configuration.png",
            }
        }
    },
    "logistic_regression": {
        "phase1": {
            "repo": "auphong2707/hospital-readmission-risk-data",
            "repo_type": "dataset",
            "files": {
                "split_info.txt": "splits/split_info.txt",
            }
        },
        "phase2": {
            "repo": "auphong2707/hospital-readmission-phase2-lr",
            "repo_type": "model",
            "files": {
                "logistic_regression_model.joblib": "logistic_regression_model.joblib",
                "logistic_regression_metrics.json": "logistic_regression_metrics.json",
                "logistic_regression_cv_fold_details.json": "logistic_regression_cv_fold_details.json",
                "logistic_regression_training_summary.json": "logistic_regression_training_summary.json",
                # Visualizations
                "roc_curve.png": "roc_curve.png",
                "precision_recall_curve.png": "precision_recall_curve.png",
                "confusion_matrix.png": "confusion_matrix.png",
                "calibration_curve.png": "calibration_curve.png",
                "feature_importance.png": "feature_importance.png",
                "feature_importance.csv": "feature_importance.csv",
                "learning_curves.png": "learning_curves.png",
                "metrics_comparison_across_folds.png": "metrics_comparison_across_folds.png",
                "validation_curves.png": "validation_curves.png",
            },
            "notes": "Logistic Regression uses scaler from Phase 1 (data/processed/splits/scaler.pkl)"
        },
        "phase3": {
            "repo": "auphong2707/hospital-readmission-phase3-lr-calibrated",
            "repo_type": "model",
            "files": {
                "logistic_regression_model_original.joblib": "logistic_regression_model_original.joblib",
                "Logistic_Regression_calibrator.pkl": "Logistic_Regression_calibrator.pkl",
                "Logistic_Regression_metrics.json": "Logistic_Regression_metrics.json",
                "Logistic_Regression_report.txt": "Logistic_Regression_report.txt",
                "calibration_comparison_metrics.json": "calibration_comparison_metrics.json",
                # Visualizations
                "Logistic_Regression_reliability_diagram.png": "Logistic_Regression_reliability_diagram.png",
                "Logistic_Regression_risk_distribution.png": "Logistic_Regression_risk_distribution.png",
                "Logistic_Regression_risk_validation.csv": "Logistic_Regression_risk_validation.csv",
                "01_reliability_diagram_before_after.png": "01_reliability_diagram_before_after.png",
                "02_calibration_improvement_metrics.png": "02_calibration_improvement_metrics.png",
                "03_probability_distribution_changes.png": "03_probability_distribution_changes.png",
                "reliability_diagram_comparison.png": "reliability_diagram_comparison.png",
                "risk_distribution_detailed.png": "risk_distribution_detailed.png",
                "risk_validation_detailed.csv": "risk_validation_detailed.csv",
            }
        },
        "phase4": {
            "repo": "auphong2707/hospital-readmission-phase4-lr-threshold",
            "repo_type": "model",
            "files": {
                "threshold_results.csv": "outputs/threshold_results.csv",
                "optimal_thresholds.json": "outputs/optimal_thresholds.json",
                "roi_metrics.json": "outputs/roi_metrics.json",
                "roi_report.txt": "outputs/roi_report.txt",
                "phase4_summary_for_phase5.json": "outputs/phase4_summary_for_phase5.json",
                # Visualizations
                "1_expected_value_curve.png": "visualizations/1_expected_value_curve.png",
                "2_cost_benefit_analysis.png": "visualizations/2_cost_benefit_analysis.png",
                "3_metrics_vs_threshold.png": "visualizations/3_metrics_vs_threshold.png",
                "4_confusion_matrix.png": "visualizations/4_confusion_matrix.png",
                "5_risk_category_distribution.png": "visualizations/5_risk_category_distribution.png",
                "6_roi_sensitivity_analysis.png": "visualizations/6_roi_sensitivity_analysis.png",
                "7_intervention_volume_forecast.png": "visualizations/7_intervention_volume_forecast.png",
                "8_cost_savings_projection.png": "visualizations/8_cost_savings_projection.png",
            }
        },
        "phase5": {
            "repo": "auphong2707/hospital-readmission-lr-final",
            "repo_type": "model",
            "files": {
                "deployment_config.json": "phase5_fairness_assessment/deployment_config.json",
                # Evaluation files
                "fairness_report.json": "phase5_fairness_assessment/evaluation/fairness_report.json",
                "group_metrics_age.csv": "phase5_fairness_assessment/evaluation/group_metrics_age.csv",
                "group_metrics_gender.csv": "phase5_fairness_assessment/evaluation/group_metrics_gender.csv",
                "group_metrics_race.csv": "phase5_fairness_assessment/evaluation/group_metrics_race.csv",
                "statistical_tests.json": "phase5_fairness_assessment/evaluation/statistical_tests.json",
                "risk_categories_age.csv": "phase5_fairness_assessment/evaluation/risk_categories_age.csv",
                "risk_categories_gender.csv": "phase5_fairness_assessment/evaluation/risk_categories_gender.csv",
                "risk_categories_race.csv": "phase5_fairness_assessment/evaluation/risk_categories_race.csv",
                "phase5_summary_for_phase6.json": "phase5_fairness_assessment/evaluation/phase5_summary_for_phase6.json",
                # Mitigation files
                "group_thresholds.json": "phase5_fairness_assessment/mitigation/group_thresholds.json",
                "mitigation_impact.json": "phase5_fairness_assessment/mitigation/mitigation_impact.json",
            }
        },
        "phase6": {
            "repo": "auphong2707/hospital-readmission-lr-final",
            "repo_type": "model",
            "files": {
                "final_system_metrics.json": "phase6_final_evaluation/final_system_metrics.json",
                "deployment_report.json": "phase6_final_evaluation/deployment_report.json",
                # Visualizations
                "calibration_curve.png": "phase6_final_evaluation/visualizations/calibration_curve.png",
                "confusion_matrix.png": "phase6_final_evaluation/visualizations/confusion_matrix.png",
                "fairness_disparities.png": "phase6_final_evaluation/visualizations/fairness_disparities.png",
                "group_fpr_comparison.png": "phase6_final_evaluation/visualizations/group_fpr_comparison.png",
                "group_precision_comparison.png": "phase6_final_evaluation/visualizations/group_precision_comparison.png",
                "group_tpr_comparison.png": "phase6_final_evaluation/visualizations/group_tpr_comparison.png",
                "risk_distribution.png": "phase6_final_evaluation/visualizations/risk_distribution.png",
                "roi_breakdown.png": "phase6_final_evaluation/visualizations/roi_breakdown.png",
                "threshold_configuration.png": "phase6_final_evaluation/visualizations/threshold_configuration.png",
            }
        }
    }
}

# Helper function to get full HuggingFace URL
def get_hf_url(method, phase, filename):
    """
    Get the full HuggingFace URL for a specific file.
    
    Args:
        method: 'gradient_boosting', 'random_forest', or 'logistic_regression'
        phase: 'phase1', 'phase2', 'phase3', or 'phase4'
        filename: The name of the file (as stored locally)
    
    Returns:
        Full HuggingFace URL or None if not found
    """
    if method not in REPO_MAPPING:
        return None
    
    if phase not in REPO_MAPPING[method]:
        return None
    
    phase_info = REPO_MAPPING[method][phase]
    if filename not in phase_info["files"]:
        return None
    
    repo = phase_info["repo"]
    repo_type = phase_info["repo_type"]
    file_path = phase_info["files"][filename]
    
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo}/blob/main/{file_path}"
    else:  # model
        return f"https://huggingface.co/{repo}/blob/main/{file_path}"


# Helper function to download from HuggingFace
def get_download_info(method, phase, filename):
    """
    Get download information for a file from HuggingFace.
    
    Args:
        method: 'gradient_boosting', 'random_forest', or 'logistic_regression'
        phase: 'phase1', 'phase2', 'phase3', or 'phase4'
        filename: The name of the file (as stored locally)
    
    Returns:
        Dictionary with repo_id, file_path, and repo_type, or None if not found
    """
    if method not in REPO_MAPPING:
        return None
    
    if phase not in REPO_MAPPING[method]:
        return None
    
    phase_info = REPO_MAPPING[method][phase]
    if filename not in phase_info["files"]:
        return None
    
    return {
        "repo_id": phase_info["repo"],
        "file_path": phase_info["files"][filename],
        "repo_type": phase_info["repo_type"]
    }


# Example usage
if __name__ == "__main__":
    # Example 1: Get URL for a specific file
    url = get_hf_url("gradient_boosting", "phase2", "gradient_boosting_model.joblib")
    print(f"Model URL: {url}")
    
    # Example 2: Get download info
    download_info = get_download_info("random_forest", "phase3", "Random_Forest_calibrator.pkl")
    print(f"\nDownload info: {download_info}")
    
    # Example 3: List all files for a specific method and phase
    method = "logistic_regression"
    phase = "phase4"
    print(f"\nAll files for {method} - {phase}:")
    if method in REPO_MAPPING and phase in REPO_MAPPING[method]:
        for filename in REPO_MAPPING[method][phase]["files"].keys():
            print(f"  - {filename}")
    
    # Example 4: Generate markdown table for all phase 2 models
    print("\n\n## Phase 2 Models - HuggingFace Links\n")
    for method in ["gradient_boosting", "random_forest", "logistic_regression"]:
        print(f"\n### {method.replace('_', ' ').title()}")
        repo = REPO_MAPPING[method]["phase2"]["repo"]
        print(f"Repository: https://huggingface.co/{repo}\n")
