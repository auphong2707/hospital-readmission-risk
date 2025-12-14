# Phase Output Mapping - For Phase 7 Collection

This document maps the actual outputs from each phase based on the code implementation.

## Phase 1: Data Preprocessing

**Output Directory**: `data/processed/` (contains `huggingface/` and `splits/` subdirectories)
**Files**:
- `huggingface/preprocessing_metadata.txt` - Preprocessing metadata
- `splits/split_info.txt` - Split statistics (uploaded to HF as `splits/split_info.txt`)
- `splits/train.csv` - Training data (uploaded to HF)
- `splits/validation.csv` - Validation data (uploaded to HF)
- `splits/test.csv` - Test data (uploaded to HF)
- `splits/scaler.pkl` - Feature scaler (fitted on train only)

**Phase 7 Collection**: Copy from `data/processed/huggingface/` and `data/processed/splits/`, skip CSV files (too large)

## Phase 2: Risk Modeling

### Gradient Boosting
**Output Directory**: `outputs/gradient_boosting/` + HuggingFace model repo
**Files**:
- `gradient_boosting_model.joblib` - Trained model
- `gradient_boosting_metrics.json` - Performance metrics
- `cv_fold_details.json` - Cross-validation fold metrics (**NOT** gradient_boosting_fold_details.json)
- `training_summary.json` - Training log (**NOT** gradient_boosting_training_summary.json)
**Visualizations** (in `visualizations/` subdirectory):
- `gradient_boosting_roc_curve.png`
- `gradient_boosting_pr_curve.png`
- `gradient_boosting_confusion_matrix.png`
- `gradient_boosting_calibration_curve.png`
- `gradient_boosting_feature_importance.png`
- `gradient_boosting_feature_importance.csv`
- `gradient_boosting_learning_curves.png`
- `gradient_boosting_validation_curves.png`

### Random Forest
**Output Directory**: `outputs/random_forest/` + HuggingFace model repo
**Files**:
- `random_forest_model.joblib` - Trained model
- `random_forest_metrics.json` - Performance metrics
- `cv_fold_details.json` - Cross-validation fold metrics (**NOT** random_forest_fold_details.json)
- `training_summary.json` - Training log (**NOT** random_forest_training_summary.json)
**Visualizations** (in `visualizations/` subdirectory):
- `random_forest_roc_curve.png`
- `random_forest_pr_curve.png`
- `random_forest_confusion_matrix.png`
- `random_forest_calibration_curve.png`
- `random_forest_feature_importance.png`
- `random_forest_feature_importance.csv`
- `random_forest_learning_curves.png`
- `random_forest_validation_curves.png`

### Logistic Regression
**Output Directory**: `outputs/logistic_regression/` + HuggingFace model repo
**Files**:
- `logistic_regression.pkl` - Trained model
- **NO SCALER** - Uses Phase 1 scaler from `data/processed/splits/scaler.pkl`
- `logistic_regression_metrics.json` - Performance metrics
- `logistic_regression_cv_fold_details.json` - Cross-validation fold metrics (**NOTE**: includes "cv" prefix)
- `logistic_regression_training_summary.json` - Training log
**Visualizations** (in `visualizations/` subdirectory):
- `logistic_regression_roc_curve.png`
- `logistic_regression_pr_curve.png`
- `logistic_regression_confusion_matrix.png`
- `logistic_regression_calibration_curve.png`
- `logistic_regression_coefficients.png`
- `logistic_regression_learning_curves.png`
- `logistic_regression_validation_curves.png`

## Phase 3: Model Calibration

### Gradient Boosting
**Output Directory**: `outputs/gradient_boosting/calibration/` + HuggingFace model repo
**Files**:
- `gradient_boosting_model_original.joblib` - Original uncalibrated model (**NOT** calibrated_model.joblib)
- `Gradient_Boosting_(LightGBM)_calibrator.pkl` - Platt scaling calibrator
- `Gradient_Boosting_(LightGBM)_metrics.json` - Calibration metrics (**NOT** calibration_metrics.json)
- `Gradient_Boosting_(LightGBM)_report.txt` - Text report
- `calibration_comparison_metrics.json` - Before/after comparison
**Visualizations** (saved directly in calibration/, **NOT** in visualizations/ subdirectory):
- `reliability_diagram.png`
- `reliability_diagram_comparison.png`
- Various demographic-specific calibration plots

### Random Forest
**Output Directory**: `outputs/random_forest/calibration/` + HuggingFace model repo
**Files**:
- `random_forest_model_original.joblib` - Original uncalibrated model
- `Random_Forest_calibrator.pkl` - Platt scaling calibrator
- `Random_Forest_metrics.json` - Calibration metrics
- `Random_Forest_report.txt` - Text report
- `calibration_comparison_metrics.json` - Before/after comparison
**Visualizations** (saved directly in calibration/):
- Similar structure to Gradient Boosting

### Logistic Regression
**Output Directory**: `outputs/logistic_regression/calibration/` + HuggingFace model repo
**Files**:
- `logistic_regression_model_original.joblib` - Original uncalibrated model
- `Logistic_Regression_calibrator.pkl` - Platt scaling calibrator
- `Logistic_Regression_metrics.json` - Calibration metrics
- `Logistic_Regression_report.txt` - Text report
- `calibration_comparison_metrics.json` - Before/after comparison
**Visualizations** (saved directly in calibration/):
- Similar structure to Gradient Boosting

## Phase 4: Optimal Threshold & ROI Analysis

### Gradient Boosting
**Output Directory**: `outputs/gradient_boosting/threshold_optimization/` + HuggingFace model repo
**Files**:
- `threshold_results.csv` - Full threshold search results (**NOT** threshold_search_results.csv)
- `optimal_thresholds.json` - Selected optimal thresholds
- `roi_metrics.json` - ROI calculations
- `roi_report.txt` - Detailed ROI report
- `phase4_summary_for_phase5.json` - Summary for Phase 5 (**NOT** phase5_input_summary.json)
**Visualizations** (may be in `visualizations/` subdirectory or root):
- `expected_value_curve.png`
- `confusion_matrix_at_optimal.png`
- `roi_breakdown.png`
- `intervention_volume.png`
- `risk_distribution.png`
- `sensitivity_analysis.png`
- `cost_components.png`
- `performance_vs_cost.png`

### Random Forest
**Output Directory**: `outputs/random_forest/threshold_optimization/` + HuggingFace model repo
**Files**: Same as Gradient Boosting
**Visualizations**: Same as Gradient Boosting (8 plots)

### Logistic Regression
**Output Directory**: `outputs/logistic_regression/threshold_optimization/` + HuggingFace model repo
**Files**: Same as Gradient Boosting
**Visualizations**: Same as Gradient Boosting (8 plots)

## Phase 5: Fairness Assessment & Mitigation

### Part A: Evaluation (always runs)
**Output Directory**: `phase-5-fairness-assessment-mitigation/outputs/{method}/evaluation/`

**Files**:
- `group_metrics_{attribute}.csv` - Performance metrics by demographic (one file per attribute: race, gender, age)
- `statistical_tests.json` - Statistical significance tests
- `risk_categories_{attribute}.csv` - Risk analysis by demographics (one file per attribute)
- `fairness_report.json` - Comprehensive fairness report
- `phase5_summary_for_phase6.json` - Summary for Phase 6

**Visualizations** (saved directly in evaluation/, **NOT** in visualizations/ subdirectory):
- `tpr_by_race.png`
- `fpr_by_race.png`
- `ppv_by_race.png`
- `tpr_by_gender.png`
- `fpr_by_gender.png`
- `ppv_by_gender.png`
- `tpr_by_age.png`
- `fpr_by_age.png`
- `ppv_by_age.png`
- `disparities_heatmap.png`
- `performance_overview.png`

### Part B: Mitigation (conditional - only if violations detected)
**Output Directory**: `phase-5-fairness-assessment-mitigation/outputs/{method}/mitigation/`

**Files**:
- `group_thresholds.json` - Optimized group-specific thresholds
- `mitigation_impact.json` - Before/after comparison
- `tradeoff_analysis.json` - Performance-fairness tradeoffs
- `phase6_deployment_config.json` - Deployment config with mitigation

**Visualizations** (saved directly in mitigation/):
- `threshold_comparison.png`
- `disparity_reduction.png`
- `group_performance_change.png`
- `overall_performance_change.png`
- `tradeoff_curve.png`

**Note**: If no violations, mitigation folder contains `no_mitigation_needed.json` placeholder

### Combined Output (always created)
**File**: `deployment_config.json` (at `phase-5-fairness-assessment-mitigation/outputs/{method}/`)
- Contains: is_mitigated flag, use_group_thresholds flag, threshold_configuration, phase6_instructions
- Created by the run script after evaluation (and optionally mitigation)

## Phase 6: Final System Evaluation

### All Methods
**Output Directory**: `outputs/{method}/final_evaluation/`

**Files**:
- `final_system_metrics.json` - **SINGLE SOURCE OF TRUTH** (comprehensive metrics)
- `deployment_report.json` - Stakeholder-friendly summary

**Visualizations** (in `visualizations/` subdirectory):
- `confusion_matrix.png` - Final system confusion matrix
- `calibration_curve.png` - Final calibration assessment
- `group_tpr_comparison.png` - TPR across demographic groups
- `group_fpr_comparison.png` - FPR across demographic groups
- `group_precision_comparison.png` - Precision across demographic groups
- `fairness_disparities.png` - TPR/FPR/DP disparities
- `roi_breakdown.png` - Financial impact visualization
- `risk_distribution.png` - Risk stratification analysis
- `threshold_configuration.png` - Deployed threshold visualization

**Key Change**: Phase 7 should use `final_system_metrics.json` from Phase 6, NOT `*_metrics.json` from Phase 2!

## Summary for Phase 7 Collection

### Phase 1
- **Files**: 1 (split_info.txt)
- **Total**: 1 file

### Phase 2 (per method)
- **Files**: 4 (model, metrics, fold_details, training_summary) + 1 scaler (LR only)
- **Visualizations**: 7-8 plots
- **Total**: ~12 files per method

### Phase 3 (per method)
- **Files**: 3 (calibrated_model, calibration_metrics, predictions)
- **Visualizations**: 5 plots
- **Total**: 8 files per method

### Phase 4 (per method)
- **Files**: 4 (threshold_search, optimal_thresholds, roi_metrics, phase5_summary)
- **Visualizations**: 8 plots
- **Total**: 12 files per method

### Phase 5 (per method)
- **Files**: 4-7 (evaluation always: 4 files, mitigation if needed: +3 files, deployment_config: 1)
- **Visualizations**: 11-16 plots (evaluation: 11, mitigation if needed: +5)
- **Total**: 15-23 files per method

### Phase 6 (per method) - NEW
- **Files**: 2 (final_system_metrics.json, deployment_report.json)
- **Visualizations**: 9 plots
- **Total**: 11 files per method

### Grand Total (for full pipeline with one method)
- **Phase 1**: 1 file
- **Phase 2**: 12 files
- **Phase 3**: 8 files
- **Phase 4**: 12 files
- **Phase 5**: 15-23 files
- **Phase 6**: 11 files
- **TOTAL**: **59-67 files** (varies based on whether mitigation was needed)

## Phase 7 Collection Strategy

1. **Download from HuggingFace** (if not local):
   - Phase 1: Split info from data repo
   - Phase 2: Model + metrics from model repo
   - Phase 3: Calibrated model + metrics from calibrated repo
   - Phase 4: Thresholds + ROI from threshold repo
   - Phase 5: Fairness reports from fairness repo
   - Phase 6: Final metrics from fairness repo (same location as Phase 5)

2. **Aggregate for Publication**:
   - Use `final_system_metrics.json` from Phase 6 as the authoritative source
   - Include Phase 2-5 outputs for research transparency
   - Create `aggregated_results.json` combining all phases
   - Generate `model_card.md` with deployment-ready documentation

3. **Upload to HuggingFace**:
   - Create new publication repository
   - Upload all collected files
   - Include comprehensive model card
   - Tag with version and deployment status
