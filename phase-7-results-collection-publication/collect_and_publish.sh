#!/bin/bash

################################################################################
# Phase 7: Results Collection & Publication (REFACTORED)
#
# Collects outputs from Phases 1-6 and publishes to HuggingFace for reproducibility
#
# Key Changes:
# - Phase 5: Now collects from fairness-assessment-mitigation (evaluation + mitigation)
# - Phase 6: Now collects from final-system-evaluation (NEW - comprehensive final metrics)
# - Uses Phase 6 final_system_metrics.json as the SINGLE SOURCE OF TRUTH
#
# Usage:
#   bash collect_and_publish.sh --method <method_name> [--repo-id <username/repo>] [OPTIONS]
#
# Methods:
#   - gradient_boosting: Full pipeline (Phases 1-6)
#   - random_forest: Full pipeline (Phases 1-6)
#   - logistic_regression: Full pipeline (Phases 1-6)
################################################################################

set -e  # Exit on error

# Default configuration
REPO_ID=""
REPO_TYPE="model"
PRIVATE=false
DRY_RUN=false
METHOD=""
PROJECT_ROOT="."
OUTPUT_DIR="./outputs"
COLLECTION_DIR="${OUTPUT_DIR}/collection_${METHOD}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        --private)
            PRIVATE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash collect_and_publish.sh --method <name> [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --method <name>             Method: gradient_boosting, random_forest, or logistic_regression"
            echo ""
            echo "Optional:"
            echo "  --repo-id <username/repo>   HuggingFace repository ID (auto-generated if not provided)"
            echo "  --private                   Create private repository"
            echo "  --dry-run                   Collect files but don't upload"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash collect_and_publish.sh --method gradient_boosting"
            echo "  bash collect_and_publish.sh --method gradient_boosting --repo-id user/hospital-gb-final"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Validation
################################################################################

if [ -z "${METHOD}" ]; then
    echo -e "${RED}Error: --method is required${NC}"
    echo "Valid options: gradient_boosting, random_forest, logistic_regression"
    exit 1
fi

if [[ ! "${METHOD}" =~ ^(gradient_boosting|random_forest|logistic_regression)$ ]]; then
    echo -e "${RED}Error: Invalid method '${METHOD}'${NC}"
    echo "Valid options: gradient_boosting, random_forest, logistic_regression"
    exit 1
fi

# Auto-set repo-id if not provided
if [ -z "${REPO_ID}" ]; then
    # Convert underscores to hyphens for consistent repo naming
    METHOD_HYPHENATED="${METHOD//_/-}"
    REPO_ID="auphong2707/hospital-readmission-${METHOD_HYPHENATED}-final"
    echo -e "${YELLOW}Note: Using auto-generated repo-id: ${REPO_ID}${NC}"
fi

# Update collection dir with method name
COLLECTION_DIR="${OUTPUT_DIR}/collection_${METHOD}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         PHASE 7: RESULTS COLLECTION & PUBLICATION                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Method: ${METHOD}"
echo "  Repository: ${REPO_ID}"
echo "  Private: ${PRIVATE}"
echo "  Dry Run: ${DRY_RUN}"
echo "  Collection Dir: ${COLLECTION_DIR}"
echo ""

################################################################################
# Step 1: Create Collection Structure
################################################################################

echo -e "${YELLOW}[Step 1/4] Creating collection structure...${NC}"

rm -rf "${COLLECTION_DIR}"  # Clean previous collection
mkdir -p "${COLLECTION_DIR}"
mkdir -p "${COLLECTION_DIR}/phase1_preprocessing"
mkdir -p "${COLLECTION_DIR}/phase2_modeling"
mkdir -p "${COLLECTION_DIR}/phase2_modeling/visualizations"
mkdir -p "${COLLECTION_DIR}/phase3_calibration"
mkdir -p "${COLLECTION_DIR}/phase3_calibration/visualizations"
mkdir -p "${COLLECTION_DIR}/phase4_threshold_optimization"
mkdir -p "${COLLECTION_DIR}/phase4_threshold_optimization/visualizations"
mkdir -p "${COLLECTION_DIR}/phase5_fairness_assessment"
mkdir -p "${COLLECTION_DIR}/phase5_fairness_assessment/evaluation"
mkdir -p "${COLLECTION_DIR}/phase5_fairness_assessment/mitigation"
mkdir -p "${COLLECTION_DIR}/phase5_fairness_assessment/visualizations"
mkdir -p "${COLLECTION_DIR}/phase6_final_evaluation"
mkdir -p "${COLLECTION_DIR}/phase6_final_evaluation/visualizations"

echo -e "${GREEN}✓ Collection structure created${NC}"

################################################################################
# Step 2: Collect Files from Each Phase
################################################################################

echo ""
echo -e "${YELLOW}[Step 2/4] Collecting outputs from all phases...${NC}"

FILE_COUNT=0
SUMMARY_FILE="${COLLECTION_DIR}/collection_summary.txt"
echo "Phase 7: Results Collection Summary" > "${SUMMARY_FILE}"
echo "Method: ${METHOD}" >> "${SUMMARY_FILE}"
echo "Collection Date: $(date)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Helper function to copy file if exists
copy_file() {
    local src="$1"
    local dst="$2"
    local desc="$3"
    
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}"
        echo -e "  ${GREEN}✓${NC} ${desc}"
        echo "  [✓] ${desc}" >> "${SUMMARY_FILE}"
        ((FILE_COUNT++))
        return 0
    else
        echo -e "  ${RED}✗${NC} ${desc} (NOT FOUND: ${src})"
        echo "  [✗] ${desc} (NOT FOUND)" >> "${SUMMARY_FILE}"
        return 1
    fi
}

# Helper function to copy directory if exists
copy_dir() {
    local src="$1"
    local dst="$2"
    local desc="$3"
    
    if [ -d "${src}" ]; then
        cp -r "${src}"/* "${dst}/" 2>/dev/null || true
        local count=$(ls -1 "${dst}" 2>/dev/null | wc -l)
        if [ ${count} -gt 0 ]; then
            echo -e "  ${GREEN}✓${NC} ${desc} (${count} files)"
            echo "  [✓] ${desc} (${count} files)" >> "${SUMMARY_FILE}"
            FILE_COUNT=$((FILE_COUNT + count))
            return 0
        fi
    fi
    echo -e "  ${YELLOW}⚠${NC} ${desc} (empty or not found)"
    echo "  [⚠] ${desc} (empty or not found)" >> "${SUMMARY_FILE}"
    return 1
}

################################################################################
# Phase 1: Data Preprocessing
################################################################################

echo "" | tee -a "${SUMMARY_FILE}"
echo -e "${BLUE}Phase 1 - Data Preprocessing:${NC}" | tee -a "${SUMMARY_FILE}"

# Note: CSV files are too large, we only collect metadata
copy_file \
    "./data/processed/preprocessing_metadata.txt" \
    "${COLLECTION_DIR}/phase1_preprocessing/preprocessing_metadata.txt" \
    "Preprocessing metadata"

echo -e "  ${YELLOW}ℹ${NC}  Note: CSV data files (train/val/test) not collected (too large, available on HF dataset)"

################################################################################
# Phase 2: Risk Modeling
################################################################################

echo "" | tee -a "${SUMMARY_FILE}"
echo -e "${BLUE}Phase 2 - Risk Modeling (${METHOD}):${NC}" | tee -a "${SUMMARY_FILE}"

# Map method to filenames
case "${METHOD}" in
    gradient_boosting)
        MODEL_PREFIX="gradient_boosting"
        DISPLAY_NAME="Gradient Boosting"
        ;;
    random_forest)
        MODEL_PREFIX="random_forest"
        DISPLAY_NAME="Random Forest"
        ;;
    logistic_regression)
        MODEL_PREFIX="logistic_regression"
        DISPLAY_NAME="Logistic Regression"
        ;;
esac

# Copy model files
copy_file \
    "./outputs/${METHOD}/${MODEL_PREFIX}_model.joblib" \
    "${COLLECTION_DIR}/phase2_modeling/${MODEL_PREFIX}_model.joblib" \
    "${DISPLAY_NAME} model" || \
copy_file \
    "./outputs/${METHOD}/${MODEL_PREFIX}_model_original.joblib" \
    "${COLLECTION_DIR}/phase2_modeling/${MODEL_PREFIX}_model.joblib" \
    "${DISPLAY_NAME} model (original)"

# Copy metrics
copy_file \
    "./outputs/${METHOD}/${MODEL_PREFIX}_metrics.json" \
    "${COLLECTION_DIR}/phase2_modeling/${MODEL_PREFIX}_metrics.json" \
    "${DISPLAY_NAME} metrics"

copy_file \
    "./outputs/${METHOD}/${MODEL_PREFIX}_fold_details.json" \
    "${COLLECTION_DIR}/phase2_modeling/${MODEL_PREFIX}_fold_details.json" \
    "${DISPLAY_NAME} fold details"

copy_file \
    "./outputs/${METHOD}/${MODEL_PREFIX}_training_summary.json" \
    "${COLLECTION_DIR}/phase2_modeling/${MODEL_PREFIX}_training_summary.json" \
    "${DISPLAY_NAME} training summary"

# For logistic regression, also copy scaler
if [ "${METHOD}" = "logistic_regression" ]; then
    copy_file \
        "./outputs/${METHOD}/logistic_regression_scaler.pkl" \
        "${COLLECTION_DIR}/phase2_modeling/logistic_regression_scaler.pkl" \
        "Feature scaler"
fi

# Copy visualizations
copy_dir \
    "./outputs/${METHOD}/visualizations" \
    "${COLLECTION_DIR}/phase2_modeling/visualizations" \
    "Phase 2 visualizations"

################################################################################
# Phase 3: Model Calibration
################################################################################

echo "" | tee -a "${SUMMARY_FILE}"
echo -e "${BLUE}Phase 3 - Model Calibration:${NC}" | tee -a "${SUMMARY_FILE}"

# Copy calibrated model
copy_file \
    "./outputs/${METHOD}/calibration/calibrated_model.joblib" \
    "${COLLECTION_DIR}/phase3_calibration/calibrated_model.joblib" \
    "Calibrated model"

# Copy calibration metrics
copy_file \
    "./outputs/${METHOD}/calibration/calibration_metrics.json" \
    "${COLLECTION_DIR}/phase3_calibration/calibration_metrics.json" \
    "Calibration metrics"

# Copy calibrated predictions
copy_file \
    "./outputs/${METHOD}/calibration/test_predictions_calibrated.csv" \
    "${COLLECTION_DIR}/phase3_calibration/test_predictions_calibrated.csv" \
    "Calibrated predictions"

# Copy visualizations
copy_dir \
    "./outputs/${METHOD}/calibration/visualizations" \
    "${COLLECTION_DIR}/phase3_calibration/visualizations" \
    "Phase 3 visualizations"

################################################################################
# Phase 4: Optimal Threshold & ROI Analysis
################################################################################

echo "" | tee -a "${SUMMARY_FILE}"
echo -e "${BLUE}Phase 4 - Threshold Optimization & ROI:${NC}" | tee -a "${SUMMARY_FILE}"

# Copy threshold results
copy_file \
    "./outputs/${METHOD}/threshold_optimization/threshold_search_results.csv" \
    "${COLLECTION_DIR}/phase4_threshold_optimization/threshold_search_results.csv" \
    "Threshold search results"

copy_file \
    "./outputs/${METHOD}/threshold_optimization/optimal_thresholds.json" \
    "${COLLECTION_DIR}/phase4_threshold_optimization/optimal_thresholds.json" \
    "Optimal thresholds"

# Copy ROI metrics
copy_file \
    "./outputs/${METHOD}/threshold_optimization/roi_metrics.json" \
    "${COLLECTION_DIR}/phase4_threshold_optimization/roi_metrics.json" \
    "ROI metrics"

# Copy Phase 5 input summary
copy_file \
    "./outputs/${METHOD}/threshold_optimization/phase5_input_summary.json" \
    "${COLLECTION_DIR}/phase4_threshold_optimization/phase5_input_summary.json" \
    "Phase 5 input summary"

# Copy visualizations
copy_dir \
    "./outputs/${METHOD}/threshold_optimization/visualizations" \
    "${COLLECTION_DIR}/phase4_threshold_optimization/visualizations" \
    "Phase 4 visualizations"

################################################################################
# Phase 5: Fairness Assessment & Mitigation
################################################################################

echo "" | tee -a "${SUMMARY_FILE}"
echo -e "${BLUE}Phase 5 - Fairness Assessment & Mitigation:${NC}" | tee -a "${SUMMARY_FILE}"

# Part A: Evaluation (always present)
echo -e "  ${BLUE}Part A: Evaluation${NC}"

copy_file \
    "./outputs/${METHOD}/fairness/evaluation/group_metrics.csv" \
    "${COLLECTION_DIR}/phase5_fairness_assessment/evaluation/group_metrics.csv" \
    "Group performance metrics"

copy_file \
    "./outputs/${METHOD}/fairness/evaluation/statistical_tests.json" \
    "${COLLECTION_DIR}/phase5_fairness_assessment/evaluation/statistical_tests.json" \
    "Statistical significance tests"

copy_file \
    "./outputs/${METHOD}/fairness/evaluation/risk_stratification.csv" \
    "${COLLECTION_DIR}/phase5_fairness_assessment/evaluation/risk_stratification.csv" \
    "Risk stratification analysis"

copy_file \
    "./outputs/${METHOD}/fairness/evaluation/fairness_report.json" \
    "${COLLECTION_DIR}/phase5_fairness_assessment/evaluation/fairness_report.json" \
    "Fairness evaluation report"

# Part B: Mitigation (conditional - only if violations detected)
echo -e "  ${BLUE}Part B: Mitigation${NC}"

if [ -f "./outputs/${METHOD}/fairness/mitigation/group_thresholds.json" ]; then
    copy_file \
        "./outputs/${METHOD}/fairness/mitigation/group_thresholds.json" \
        "${COLLECTION_DIR}/phase5_fairness_assessment/mitigation/group_thresholds.json" \
        "Group-specific thresholds"
    
    copy_file \
        "./outputs/${METHOD}/fairness/mitigation/mitigation_impact.json" \
        "${COLLECTION_DIR}/phase5_fairness_assessment/mitigation/mitigation_impact.json" \
        "Mitigation impact analysis"
    
    copy_file \
        "./outputs/${METHOD}/fairness/mitigation/tradeoff_analysis.json" \
        "${COLLECTION_DIR}/phase5_fairness_assessment/mitigation/tradeoff_analysis.json" \
        "Performance-fairness tradeoffs"
    
    echo -e "  ${GREEN}✓${NC} Mitigation was applied"
else
    copy_file \
        "./outputs/${METHOD}/fairness/mitigation/no_mitigation_needed.json" \
        "${COLLECTION_DIR}/phase5_fairness_assessment/mitigation/no_mitigation_needed.json" \
        "Mitigation not needed (placeholder)"
    
    echo -e "  ${YELLOW}ℹ${NC}  Mitigation was not needed (no fairness violations detected)"
fi

# Deployment configuration (always present)
copy_file \
    "./outputs/${METHOD}/fairness/deployment_config.json" \
    "${COLLECTION_DIR}/phase5_fairness_assessment/deployment_config.json" \
    "Deployment configuration"

# Copy visualizations
copy_dir \
    "./outputs/${METHOD}/fairness/visualizations" \
    "${COLLECTION_DIR}/phase5_fairness_assessment/visualizations" \
    "Phase 5 visualizations"

################################################################################
# Phase 6: Final System Evaluation (NEW - SINGLE SOURCE OF TRUTH)
################################################################################

echo "" | tee -a "${SUMMARY_FILE}"
echo -e "${BLUE}Phase 6 - Final System Evaluation (COMPREHENSIVE):${NC}" | tee -a "${SUMMARY_FILE}"

# Copy final system metrics (SINGLE SOURCE OF TRUTH for publication)
copy_file \
    "./outputs/${METHOD}/final_evaluation/final_system_metrics.json" \
    "${COLLECTION_DIR}/phase6_final_evaluation/final_system_metrics.json" \
    "Final system metrics (SINGLE SOURCE OF TRUTH)" 

# Copy deployment report (stakeholder-friendly)
copy_file \
    "./outputs/${METHOD}/final_evaluation/deployment_report.json" \
    "${COLLECTION_DIR}/phase6_final_evaluation/deployment_report.json" \
    "Deployment report"

# Copy comprehensive visualizations
copy_dir \
    "./outputs/${METHOD}/final_evaluation/visualizations" \
    "${COLLECTION_DIR}/phase6_final_evaluation/visualizations" \
    "Phase 6 visualizations (comprehensive)"

################################################################################
# Collection Summary
################################################################################

echo "" | tee -a "${SUMMARY_FILE}"
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                     COLLECTION COMPLETED                                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Total files collected: ${FILE_COUNT}${NC}" | tee -a "${SUMMARY_FILE}"
echo -e "${YELLOW}Collection directory: ${COLLECTION_DIR}/${NC}"
echo ""

################################################################################
# Step 3: Create Aggregated Results & Model Card
################################################################################

echo -e "${YELLOW}[Step 3/4] Creating aggregated results and model card...${NC}"

# Create aggregated_results.json from Phase 6 final metrics
python3 << PYTHON_SCRIPT
import json
import os
from datetime import datetime
from pathlib import Path

method = "${METHOD}"
collection_dir = "${COLLECTION_DIR}"

# Initialize aggregated results
aggregated = {
    "project_info": {
        "name": "Hospital Readmission Risk Prediction",
        "method": method,
        "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repository": "https://github.com/auphong2707/hospital-readmission-risk",
        "branch": "topic/final-refinement"
    },
    "pipeline_summary": {
        "phases_completed": 6,
        "phase_1": "Data Preprocessing",
        "phase_2": "Risk Modeling",
        "phase_3": "Model Calibration",
        "phase_4": "Threshold Optimization & ROI",
        "phase_5": "Fairness Assessment & Mitigation",
        "phase_6": "Final System Evaluation (AUTHORITATIVE)"
    }
}

# Load Phase 6 final system metrics (SINGLE SOURCE OF TRUTH)
final_metrics_path = f"{collection_dir}/phase6_final_evaluation/final_system_metrics.json"
if os.path.exists(final_metrics_path):
    with open(final_metrics_path, 'r') as f:
        final_metrics = json.load(f)
    
    aggregated["final_system_metrics"] = final_metrics
    print("✓ Loaded Phase 6 final system metrics (single source of truth)")
else:
    print("⚠ Warning: Phase 6 final_system_metrics.json not found")
    aggregated["final_system_metrics"] = {"error": "not_found"}

# Load deployment configuration from Phase 5
deployment_config_path = f"{collection_dir}/phase5_fairness_assessment/deployment_config.json"
if os.path.exists(deployment_config_path):
    with open(deployment_config_path, 'r') as f:
        deployment_config = json.load(f)
    
    aggregated["deployment_configuration"] = deployment_config
    print("✓ Loaded Phase 5 deployment configuration")
else:
    print("⚠ Warning: deployment_config.json not found")

# Add metadata about collection
aggregated["collection_metadata"] = {
    "collected_files_count": ${FILE_COUNT},
    "collection_timestamp": datetime.now().isoformat(),
    "key_outputs": {
        "phase_1": ["preprocessing_metadata.txt"],
        "phase_2": ["model", "metrics", "visualizations"],
        "phase_3": ["calibrated_model", "calibration_metrics", "visualizations"],
        "phase_4": ["optimal_thresholds", "roi_metrics", "visualizations"],
        "phase_5": ["fairness_report", "deployment_config", "group_thresholds (if mitigated)", "visualizations"],
        "phase_6": ["final_system_metrics.json (AUTHORITATIVE)", "deployment_report.json", "visualizations"]
    },
    "notes": [
        "Phase 6 final_system_metrics.json is the SINGLE SOURCE OF TRUTH for all metrics",
        "All previous phase metrics are included for research transparency",
        "Deployment configuration from Phase 5 specifies thresholds used in Phase 6"
    ]
}

# Save aggregated results
output_path = f"{collection_dir}/aggregated_results.json"
with open(output_path, 'w') as f:
    json.dump(aggregated, f, indent=2)

print(f"✓ Created aggregated results: {output_path}")
PYTHON_SCRIPT

echo -e "${GREEN}✓ Aggregated results created${NC}"

# Create comprehensive model card
cat > "${COLLECTION_DIR}/MODEL_CARD.md" << 'EOF'
# Hospital Readmission Risk Prediction - Model Card

## Model Details

**Model Name**: Hospital Readmission Risk Prediction System  
**Method**: ${METHOD}  
**Version**: 1.0 (Final Evaluation)  
**Date**: $(date +%Y-%m-%d)  
**Repository**: [hospital-readmission-risk](https://github.com/auphong2707/hospital-readmission-risk)

## Model Description

This is a comprehensive machine learning system for predicting 30-day hospital readmission risk for diabetes patients. The system has undergone a complete 6-phase development and evaluation pipeline:

1. **Phase 1**: Data preprocessing and train/validation/test splits
2. **Phase 2**: Model training and initial evaluation
3. **Phase 3**: Probability calibration for reliable risk scores
4. **Phase 4**: Threshold optimization and ROI analysis
5. **Phase 5**: Fairness assessment and bias mitigation
6. **Phase 6**: Final comprehensive system evaluation (**AUTHORITATIVE METRICS**)

## Key Metrics (from Phase 6 Final Evaluation)

**Phase 6 produces the authoritative metrics** that represent the deployed system performance with the selected threshold configuration (global or group-specific).

> **Note**: Refer to `phase6_final_evaluation/final_system_metrics.json` for complete authoritative metrics.

### Performance
- See Phase 6 final_system_metrics.json for accuracy, ROC-AUC, precision, recall, F1-score

### Calibration
- See Phase 6 final_system_metrics.json for Brier score and Expected Calibration Error

### Fairness
- See Phase 6 final_system_metrics.json for TPR/FPR disparities across demographic groups

### Financial Impact
- See Phase 6 final_system_metrics.json for ROI percentage and cost savings

### Risk Stratification
- See Phase 6 final_system_metrics.json for patient distribution across risk categories

## Deployment Configuration

The system uses the deployment configuration from Phase 5 (`deployment_config.json`):
- **Threshold Type**: Global or group-specific (see deployment_config.json)
- **Fairness Mitigation**: Applied if violations were detected in Phase 5
- **Deployment Readiness**: See Phase 6 deployment_report.json

## Intended Use

**Primary Use Cases**:
- Identifying high-risk patients for preventive interventions
- Resource allocation for post-discharge care management
- Clinical decision support for discharge planning

**Users**:
- Healthcare providers and clinical decision-makers
- Hospital administrators and care coordinators
- Healthcare data scientists and ML practitioners

## Limitations and Biases

- Model trained on diabetes patients from US hospitals (1999-2008)
- May not generalize to non-diabetes patients or different healthcare systems
- Fairness evaluation conducted across race, gender, and age groups
- See Phase 5 fairness reports for detailed bias analysis

## Ethical Considerations

- **Fairness**: Phase 5 evaluated and (if needed) mitigated disparities across demographic groups
- **Transparency**: Complete pipeline documented across 6 phases
- **Accountability**: All metrics, thresholds, and decisions are traceable
- **Privacy**: Model trained on de-identified historical data

## Training Data

**Dataset**: Diabetes 130-US Hospitals (1999-2008)
**Size**: ~100,000 patient encounters  
**Features**: Demographics, diagnoses, medications, lab values, prior encounters
**Source**: UCI Machine Learning Repository

**Splits**:
- Training: ~56,000 encounters (56%)
- Validation: ~22,000 encounters (22%)
- Test: ~22,000 encounters (22%) - **used for all evaluation phases**

## Evaluation Data

**All evaluation phases (2-6)** use the same held-out test set to ensure consistency.

## Model Architecture

See `phase2_modeling/` for model details:
- Gradient Boosting: LightGBM with 500 estimators
- Random Forest: scikit-learn with 500 estimators  
- Logistic Regression: Elastic Net with L1/L2 regularization

## Training Procedure

1. **Phase 2**: 5-fold stratified cross-validation with hyperparameter tuning
2. **Phase 3**: Platt scaling calibration on validation set
3. **Phase 4**: Threshold optimization to maximize expected value
4. **Phase 5**: Fairness evaluation and (if needed) group-specific threshold mitigation
5. **Phase 6**: Final evaluation with deployed configuration

## Computational Requirements

- **Training**: ~15-30 minutes per method (on standard CPU)
- **Inference**: < 1ms per prediction
- **Memory**: < 500 MB for model

## Citation

```bibtex
@software{hospital_readmission_risk_2024,
  title = {Hospital Readmission Risk Prediction System},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/auphong2707/hospital-readmission-risk}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.

## Version History

- **v1.0** (2024): Initial release with complete 6-phase pipeline
  - Phase 6 final evaluation produces authoritative metrics
  - Fairness assessment and mitigation integrated
  - Comprehensive documentation and reproducibility

## Files in This Repository

### Phase 1: Data Preprocessing
- `phase1_preprocessing/preprocessing_metadata.txt` - Split statistics

### Phase 2: Risk Modeling
- `phase2_modeling/{method}_model.joblib` - Trained model
- `phase2_modeling/{method}_metrics.json` - Initial metrics
- `phase2_modeling/visualizations/` - 7-8 plots

### Phase 3: Model Calibration
- `phase3_calibration/calibrated_model.joblib` - Calibrated model
- `phase3_calibration/calibration_metrics.json` - Calibration quality
- `phase3_calibration/visualizations/` - 5 plots

### Phase 4: Threshold Optimization & ROI
- `phase4_threshold_optimization/optimal_thresholds.json` - Selected thresholds
- `phase4_threshold_optimization/roi_metrics.json` - Financial analysis
- `phase4_threshold_optimization/visualizations/` - 8 plots

### Phase 5: Fairness Assessment & Mitigation
- `phase5_fairness_assessment/evaluation/` - Fairness evaluation results
- `phase5_fairness_assessment/mitigation/` - Bias mitigation (if applied)
- `phase5_fairness_assessment/deployment_config.json` - Deployment configuration
- `phase5_fairness_assessment/visualizations/` - 11-16 plots

### Phase 6: Final System Evaluation ⭐
- `phase6_final_evaluation/final_system_metrics.json` - **SINGLE SOURCE OF TRUTH**
- `phase6_final_evaluation/deployment_report.json` - Stakeholder summary
- `phase6_final_evaluation/visualizations/` - 9 comprehensive plots

### Aggregated Results
- `aggregated_results.json` - Combined results from all phases
- `MODEL_CARD.md` - This file
- `collection_summary.txt` - File collection log

## Important Notes

🎯 **Phase 6 Metrics are Authoritative**: The `final_system_metrics.json` file contains the definitive evaluation of the deployed system. Previous phase metrics are included for transparency and research purposes, but Phase 6 metrics reflect the actual deployed configuration.

📊 **Threshold Configuration**: The system uses either:
- Global threshold (from Phase 4), OR
- Group-specific thresholds (from Phase 5 mitigation)

The configuration used is specified in `phase5_fairness_assessment/deployment_config.json`.

✅ **Deployment Ready**: See `phase6_final_evaluation/deployment_report.json` for deployment readiness assessment.
EOF

# Replace ${METHOD} in model card
sed -i "s/\${METHOD}/${METHOD}/g" "${COLLECTION_DIR}/MODEL_CARD.md"

echo -e "${GREEN}✓ Model card created${NC}"

################################################################################
# Step 4: Upload to HuggingFace (if not dry-run)
################################################################################

if [ "${DRY_RUN}" = true ]; then
    echo ""
    echo -e "${YELLOW}[Step 4/4] Skipping upload (dry-run mode)${NC}"
    echo -e "${YELLOW}Files collected in: ${COLLECTION_DIR}${NC}"
else
    echo ""
    echo -e "${YELLOW}[Step 4/4] Uploading to HuggingFace...${NC}"
    
    # Check if repo exists, create if not
    if huggingface-cli repo info "${REPO_ID}" > /dev/null 2>&1; then
        echo -e "${YELLOW}Repository exists, will upload to existing repo${NC}"
    else
        echo -e "${YELLOW}Creating new repository: ${REPO_ID}${NC}"
        PRIVATE_FLAG=""
        if [ "${PRIVATE}" = true ]; then
            PRIVATE_FLAG="--private"
        fi
        huggingface-cli repo create "${REPO_ID}" --type model ${PRIVATE_FLAG}
    fi
    
    # Upload all files
    echo -e "${YELLOW}Uploading files...${NC}"
    huggingface-cli upload "${REPO_ID}" "${COLLECTION_DIR}" / --repo-type model
    
    echo -e "${GREEN}✓ Upload complete!${NC}"
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                   PUBLICATION COMPLETED SUCCESSFULLY                       ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}📦 Repository: https://huggingface.co/${REPO_ID}${NC}"
    echo -e "${YELLOW}📊 Files uploaded: ${FILE_COUNT}${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. Review the uploaded files on HuggingFace"
    echo "  2. Check Phase 6 final_system_metrics.json for authoritative results"
    echo "  3. Share deployment_report.json with stakeholders"
    echo "  4. Document the model card on HuggingFace"
fi

echo ""
echo -e "${GREEN}✅ Phase 7 complete!${NC}"
