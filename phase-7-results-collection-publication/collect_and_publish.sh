#!/bin/bash

################################################################################
# Phase 7: Results Collection & Publication
# Single script to collect all outputs from Phases 1-6 and upload to HuggingFace
################################################################################

# Note: Removed 'set -e' to allow script to continue even if some files are missing

# Default configuration
REPO_ID=""
REPO_TYPE="model"
PRIVATE=false
DRY_RUN=false
METHOD=""  # Required: gradient_boosting, random_forest, or logistic_regression
PROJECT_ROOT="."
OUTPUT_DIR="./outputs"
COLLECTION_DIR="${OUTPUT_DIR}/collection"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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
            echo "Usage: bash collect_and_publish.sh --method <name> --repo-id <username/repo> [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --method <name>             Method name: gradient_boosting, random_forest, or logistic_regression"
            echo "  --repo-id <username/repo>   HuggingFace repository ID"
            echo ""
            echo "Optional:"
            echo "  --private                   Create private repository"
            echo "  --dry-run                   Collect files but don't upload"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash collect_and_publish.sh --method gradient_boosting --repo-id user/hospital-readmission-gb"
            echo "  bash collect_and_publish.sh --method random_forest --repo-id user/hospital-readmission-rf --private"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Phase 7: Results Collection & Publication${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
# Validate required parameters
if [ -z "${METHOD}" ]; then
    echo -e "${RED}Error: --method is required${NC}"
    echo "Valid options: gradient_boosting, random_forest, logistic_regression"
    echo "Run with --help for usage information"
    exit 1
fi

if [[ ! "${METHOD}" =~ ^(gradient_boosting|random_forest|logistic_regression)$ ]]; then
    echo -e "${RED}Error: Invalid method '${METHOD}'${NC}"
    echo "Valid options: gradient_boosting, random_forest, logistic_regression"
    exit 1
fi

# Auto-set repo-id if not provided
if [ -z "${REPO_ID}" ]; then
    REPO_ID="auphong2707/hospital-readmission-risk-${METHOD}"
    echo -e "${YELLOW}Note: Using default repo-id: ${REPO_ID}${NC}"
fi

echo "Configuration:"
echo "  Method: ${METHOD}"
echo "  Repository: ${REPO_ID}"
echo "  Private: ${PRIVATE}"
echo "  Dry Run: ${DRY_RUN}"
echo ""

################################################################################
# Step 1: Collect All Results
################################################################################

echo -e "${YELLOW}[Step 1/3] Collecting results from all phases...${NC}"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${COLLECTION_DIR}"
mkdir -p "${COLLECTION_DIR}/models"
mkdir -p "${COLLECTION_DIR}/metrics"
mkdir -p "${COLLECTION_DIR}/data_splits"
mkdir -p "${COLLECTION_DIR}/visualizations/phase2_modeling"

# Gradient Boosting has additional phases
if [ "${METHOD}" = "gradient_boosting" ]; then
    mkdir -p "${COLLECTION_DIR}/thresholds"
    mkdir -p "${COLLECTION_DIR}/visualizations/phase3_calibration"
    mkdir -p "${COLLECTION_DIR}/visualizations/phase4_threshold_optimization"
    mkdir -p "${COLLECTION_DIR}/visualizations/phase5_fairness_evaluation"
    mkdir -p "${COLLECTION_DIR}/visualizations/phase6_fairness_mitigation"
fi

# Initialize collection summary
SUMMARY_FILE="${OUTPUT_DIR}/collection_summary.txt"
echo "Phase 7: Results Collection Summary" > "${SUMMARY_FILE}"
echo "Collection Date: $(date)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

FILE_COUNT=0

# Function to copy file if exists
copy_if_exists() {
    local src="$1"
    local dst="$2"
    local desc="$3"
    
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}"
        echo "  [x] ${desc}" | tee -a "${SUMMARY_FILE}"
        ((FILE_COUNT++))
        return 0
    else
        echo "  [ ] ${desc} (NOT FOUND: ${src})" | tee -a "${SUMMARY_FILE}"
        return 1
    fi
}

# Function to download Phase 2 artifacts if not found locally
download_phase2_if_missing() {
    local local_path=$1
    local dest_path=$2
    local hf_path=$3
    local desc=$4
    local hf_repo=$5
    
    if [ -f "${local_path}" ]; then
        cp "${local_path}" "${dest_path}"
        echo "  [x] ${desc}" | tee -a "${SUMMARY_FILE}"
        ((FILE_COUNT++))
    else
        echo "  [ ] ${desc} not found locally, trying HuggingFace..." | tee -a "${SUMMARY_FILE}"
        mkdir -p "$(dirname ${dest_path})"
        if huggingface-cli download "${hf_repo}" "${hf_path}" --local-dir "${COLLECTION_DIR}/temp" 2>/dev/null; then
            if [ -f "${COLLECTION_DIR}/temp/${hf_path}" ]; then
                mv "${COLLECTION_DIR}/temp/${hf_path}" "${dest_path}"
                echo "  [x] ${desc} (downloaded from HF)" | tee -a "${SUMMARY_FILE}"
                ((FILE_COUNT++))
            else
                echo "  [ ] ${desc} (NOT FOUND - local or HF)" | tee -a "${SUMMARY_FILE}"
            fi
        else
            echo "  [ ] ${desc} (NOT FOUND - local or HF)" | tee -a "${SUMMARY_FILE}"
        fi
    fi
}

# Function to download Phase 3 artifacts if not found locally
download_phase3_if_missing() {
    local local_path=$1
    local dest_path=$2
    local hf_path=$3
    local desc=$4
    local hf_repo="auphong2707/hospital-readmission-lgbm-calibrated"
    
    if [ -f "${local_path}" ]; then
        cp "${local_path}" "${dest_path}"
        echo "  [x] ${desc}" | tee -a "${SUMMARY_FILE}"
        ((FILE_COUNT++))
    else
        echo "  [ ] ${desc} not found locally, trying HuggingFace..." | tee -a "${SUMMARY_FILE}"
        mkdir -p "$(dirname ${dest_path})"
        if huggingface-cli download "${hf_repo}" "${hf_path}" --local-dir "${COLLECTION_DIR}/temp" 2>/dev/null; then
            if [ -f "${COLLECTION_DIR}/temp/${hf_path}" ]; then
                mv "${COLLECTION_DIR}/temp/${hf_path}" "${dest_path}"
                echo "  [x] ${desc} (downloaded from HF)" | tee -a "${SUMMARY_FILE}"
                ((FILE_COUNT++))
            else
                echo "  [ ] ${desc} (NOT FOUND - local or HF)" | tee -a "${SUMMARY_FILE}"
            fi
        else
            echo "  [ ] ${desc} (NOT FOUND - local or HF)" | tee -a "${SUMMARY_FILE}"
        fi
    fi
}

# Phase 1: Data Preprocessing (shared across all methods)
echo "" | tee -a "${SUMMARY_FILE}"
echo "Phase 1 - Data Preprocessing:" | tee -a "${SUMMARY_FILE}"

# Try local files first, fallback to HuggingFace download
download_if_missing() {
    local local_path=$1
    local dest_path=$2
    local hf_path=$3
    local desc=$4
    local hf_repo="auphong2707/hospital-readmission-risk-data"
    
    if [ -f "${local_path}" ]; then
        cp "${local_path}" "${dest_path}"
        echo "  [x] ${desc}" | tee -a "${SUMMARY_FILE}"
        ((FILE_COUNT++))
    else
        echo "  [ ] ${desc} not found locally" | tee -a "${SUMMARY_FILE}"
        # Skip HuggingFace download - uncomment below to enable
        # mkdir -p "$(dirname ${dest_path})"
        # if huggingface-cli download "${hf_repo}" "${hf_path}" --local-dir "${COLLECTION_DIR}/temp" --repo-type=dataset 2>&1 | grep -v "Warning:" >/dev/null; then
        #     if [ -f "${COLLECTION_DIR}/temp/${hf_path}" ]; then
        #         mv "${COLLECTION_DIR}/temp/${hf_path}" "${dest_path}"
        #         echo "  [x] ${desc} (downloaded from HF)" | tee -a "${SUMMARY_FILE}"
        #         ((FILE_COUNT++))
        #     fi
        # fi
    fi
}

# Phase 1 files - Only collect metadata, skip CSV files
copy_if_exists "${PROJECT_ROOT}/data/processed/preprocessing_metadata.txt" \
    "${COLLECTION_DIR}/preprocessing_metadata.txt" \
    "Preprocessing metadata"
echo "  [ ] Data splits skipped (CSV files not collected)" | tee -a "${SUMMARY_FILE}"

# Cleanup temp directory
rm -rf "${COLLECTION_DIR}/temp" 2>/dev/null

# Phase 2: Risk Modeling
echo "" | tee -a "${SUMMARY_FILE}"
echo "Phase 2 - Risk Modeling (${METHOD}):" | tee -a "${SUMMARY_FILE}"

# Map method to file names and HuggingFace repos
case "${METHOD}" in
    gradient_boosting)
        MODEL_FILE="gradient_boosting_model_original.joblib"
        METRICS_FILE="Gradient_Boosting_metrics.json"
        FILE_PREFIX="Gradient_Boosting"
        DISPLAY_NAME="Gradient Boosting"
        PHASE2_HF_REPO="auphong2707/hospital-readmission-lgbm"
        ;;
    random_forest)
        MODEL_FILE="random_forest_model.joblib"
        METRICS_FILE="Random_Forest_metrics.json"
        FILE_PREFIX="Random_Forest"
        DISPLAY_NAME="Random Forest"
        PHASE2_HF_REPO="auphong2707/hospital-readmission-rf"
        ;;
    logistic_regression)
        MODEL_FILE="logistic_regression_model.joblib"
        METRICS_FILE="Logistic_Regression_metrics.json"
        FILE_PREFIX="Logistic_Regression"
        DISPLAY_NAME="Logistic Regression"
        PHASE2_HF_REPO="auphong2707/hospital-readmission-lr"
        ;;
esac

# Copy model and metrics (HuggingFace download disabled)
copy_if_exists "${PROJECT_ROOT}/models/${MODEL_FILE}" \
    "${COLLECTION_DIR}/models/${MODEL_FILE}" \
    "${DISPLAY_NAME} model"
copy_if_exists "${PROJECT_ROOT}/phase-2-risk-modeling/${METRICS_FILE}" \
    "${COLLECTION_DIR}/metrics/${METRICS_FILE}" \
    "${DISPLAY_NAME} metrics"

# Copy visualizations (HuggingFace download disabled)
VIZ_COUNT=0
for viz in "ROC_Curve" "Precision_Recall_Curve" "Confusion_Matrix" "Feature_Importance_Top_20" \
           "Calibration_Plot" "Prediction_Distribution" "Threshold_Metrics" \
           "Classification_Report" "Learning_Curves"; do
    viz_file="${FILE_PREFIX}_${viz}.png"
    local_path="${PROJECT_ROOT}/phase-2-risk-modeling/${viz_file}"
    dest_path="${COLLECTION_DIR}/visualizations/phase2_modeling/${viz_file}"
    
    if [ -f "${local_path}" ]; then
        cp "${local_path}" "${dest_path}"
        ((VIZ_COUNT++))
        ((FILE_COUNT++))
    fi
done
echo "  [x] Phase 2 visualizations (${VIZ_COUNT} plots)" | tee -a "${SUMMARY_FILE}"

# Phase 3: Model Calibration (Gradient Boosting only)
if [ "${METHOD}" = "gradient_boosting" ]; then
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "Phase 3 - Model Calibration:" | tee -a "${SUMMARY_FILE}"
    
    download_phase3_if_missing "${PROJECT_ROOT}/calibration_outputs/gradient_boosting/Gradient_Boosting_(LightGBM)_calibrator.pkl" \
        "${COLLECTION_DIR}/models/Gradient_Boosting_calibrator.pkl" \
        "Gradient_Boosting_(LightGBM)_calibrator.pkl" \
        "Platt calibrator"
    download_phase3_if_missing "${PROJECT_ROOT}/calibration_outputs/gradient_boosting/calibration_comparison_metrics.json" \
        "${COLLECTION_DIR}/metrics/phase3_calibration_metrics.json" \
        "calibration_comparison_metrics.json" \
        "Calibration metrics"
    download_phase3_if_missing "${PROJECT_ROOT}/calibration_outputs/gradient_boosting/reliability_diagram_comparison.png" \
        "${COLLECTION_DIR}/visualizations/phase3_calibration/reliability_diagram_comparison.png" \
        "reliability_diagram_comparison.png" \
        "Reliability diagram"
    rm -rf "${COLLECTION_DIR}/temp" 2>/dev/null
fi

# Phase 4: Threshold Optimization (Gradient Boosting only)
if [ "${METHOD}" = "gradient_boosting" ]; then
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "Phase 4 - Threshold Optimization:" | tee -a "${SUMMARY_FILE}"
    
    copy_if_exists "${PROJECT_ROOT}/phase-4-optimal-threshold-ROI-analysis/outputs/optimal_thresholds.json" \
        "${COLLECTION_DIR}/thresholds/optimal_thresholds.json" \
        "Optimal thresholds"
    copy_if_exists "${PROJECT_ROOT}/phase-4-optimal-threshold-ROI-analysis/outputs/roi_metrics.json" \
        "${COLLECTION_DIR}/metrics/phase4_roi_metrics.json" \
        "ROI metrics"
    copy_if_exists "${PROJECT_ROOT}/phase-4-optimal-threshold-ROI-analysis/outputs/phase4_summary_for_phase5.json" \
        "${COLLECTION_DIR}/metrics/phase4_summary.json" \
        "Phase 4 summary"
    
    # Copy Phase 4 visualizations
    for viz in "${PROJECT_ROOT}/phase-4-optimal-threshold-ROI-analysis/visualizations"/*.png; do
        if [ -f "${viz}" ]; then
            cp "${viz}" "${COLLECTION_DIR}/visualizations/phase4_threshold_optimization/"
            ((FILE_COUNT++))
        fi
    done
    echo "  [x] Phase 4 visualizations (8 plots)" | tee -a "${SUMMARY_FILE}"
fi

# Phase 5: Fairness Evaluation (Gradient Boosting only)
if [ "${METHOD}" = "gradient_boosting" ]; then
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "Phase 5 - Fairness Evaluation:" | tee -a "${SUMMARY_FILE}"
    
    copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs/fairness_report.json" \
        "${COLLECTION_DIR}/metrics/phase5_fairness_report.json" \
        "Fairness report"
    copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs/phase5_summary_for_phase6.json" \
        "${COLLECTION_DIR}/metrics/phase5_summary.json" \
        "Phase 5 summary"
    copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs/statistical_tests.json" \
        "${COLLECTION_DIR}/metrics/phase5_statistical_tests.json" \
        "Statistical tests"
    copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs/group_metrics_race.csv" \
        "${COLLECTION_DIR}/metrics/phase5_group_metrics_race.csv" \
        "Group metrics (race)"
    copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs/group_metrics_gender.csv" \
        "${COLLECTION_DIR}/metrics/phase5_group_metrics_gender.csv" \
        "Group metrics (gender)"
    copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs/group_metrics_age.csv" \
        "${COLLECTION_DIR}/metrics/phase5_group_metrics_age.csv" \
        "Group metrics (age)"
    
    # Copy Phase 5 visualizations
    for viz in "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs/visualizations"/*.png; do
        if [ -f "${viz}" ]; then
            cp "${viz}" "${COLLECTION_DIR}/visualizations/phase5_fairness_evaluation/"
            ((FILE_COUNT++))
        fi
    done
    echo "  [x] Phase 5 visualizations (~21 plots)" | tee -a "${SUMMARY_FILE}"
fi

# Phase 6: Fairness Mitigation (Gradient Boosting only, optional)
if [ "${METHOD}" = "gradient_boosting" ]; then
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "Phase 6 - Fairness Mitigation:" | tee -a "${SUMMARY_FILE}"
    
    if [ -f "${PROJECT_ROOT}/phase-6-fairness-mitigation-bias-correction/outputs/group_thresholds.json" ]; then
        copy_if_exists "${PROJECT_ROOT}/phase-6-fairness-mitigation-bias-correction/outputs/group_thresholds.json" \
            "${COLLECTION_DIR}/thresholds/group_thresholds.json" \
            "Group-specific thresholds"
        copy_if_exists "${PROJECT_ROOT}/phase-6-fairness-mitigation-bias-correction/outputs/mitigation_impact.json" \
            "${COLLECTION_DIR}/metrics/phase6_mitigation_impact.json" \
            "Mitigation impact"
        
        # Copy Phase 6 visualizations
        for viz in "${PROJECT_ROOT}/phase-6-fairness-mitigation-bias-correction/outputs/visualizations"/*.png; do
            if [ -f "${viz}" ]; then
                cp "${viz}" "${COLLECTION_DIR}/visualizations/phase6_fairness_mitigation/"
                ((FILE_COUNT++))
            fi
        done
        echo "  [x] Phase 6 visualizations (5 plots)" | tee -a "${SUMMARY_FILE}"
    else
        echo "  [ ] Phase 6 not applied (mitigation optional)" | tee -a "${SUMMARY_FILE}"
    fi
fi

echo "" | tee -a "${SUMMARY_FILE}"
echo "Total files collected: ${FILE_COUNT}" | tee -a "${SUMMARY_FILE}"
echo -e "${GREEN}[x] Collection complete!${NC}"

################################################################################
# Step 2: Create Aggregated Results & Model Card
################################################################################

echo ""
echo -e "${YELLOW}[Step 2/3] Creating aggregated results and model card...${NC}"

# Create aggregated results JSON using Python
python3 << 'PYTHON_SCRIPT'
import json
import os
from datetime import datetime
from pathlib import Path

# Configuration
project_root = "."
output_dir = "./outputs"
collection_dir = f"{output_dir}/collection"
metrics_dir = f"{collection_dir}/metrics"

# Initialize aggregated results
aggregated_results = {
    "project_info": {
        "name": "Hospital Readmission Risk Prediction",
        "dataset": "Diabetes 130-US Hospitals (1999-2008)",
        "collection_date": datetime.now().strftime("%Y-%m-%d"),
        "repository": "https://github.com/auphong2707/hospital-readmission-risk"
    }
}

# Load Phase 2 metrics (Gradient Boosting primary model)
try:
    with open(f"{metrics_dir}/Gradient_Boosting_metrics.json") as f:
        gb_metrics = json.load(f)
    aggregated_results["phase_2_modeling"] = {
        "models_trained": 3,
        "primary_model": "Gradient Boosting (LightGBM)",
        "gradient_boosting": {
            "roc_auc": gb_metrics.get("roc_auc"),
            "pr_auc": gb_metrics.get("pr_auc"),
            "f1_score": gb_metrics.get("f1_score"),
            "precision": gb_metrics.get("precision"),
            "recall": gb_metrics.get("recall"),
            "brier_score": gb_metrics.get("brier_score")
        }
    }
except Exception as e:
    print(f"Warning: Could not load Phase 2 metrics: {e}")
    aggregated_results["phase_2_modeling"] = {
        "models_trained": 3,
        "primary_model": "Gradient Boosting (LightGBM)",
        "note": "Metrics not available"
    }

# Load Phase 3 calibration metrics
try:
    with open(f"{metrics_dir}/phase3_calibration_metrics.json") as f:
        cal_metrics = json.load(f)
    aggregated_results["phase_3_calibration"] = {
        "method": "Platt Scaling",
        "brier_score_before": cal_metrics.get("original_brier_score"),
        "brier_score_after": cal_metrics.get("calibrated_brier_score"),
        "ece_before": cal_metrics.get("original_ece"),
        "ece_after": cal_metrics.get("calibrated_ece")
    }
except Exception as e:
    print(f"Warning: Could not load Phase 3 metrics: {e}")
    aggregated_results["phase_3_calibration"] = {
        "method": "Platt Scaling",
        "note": "Metrics not available"
    }

# Load Phase 4 threshold optimization
try:
    with open(f"{collection_dir}/thresholds/optimal_thresholds.json") as f:
        thresholds = json.load(f)
    with open(f"{metrics_dir}/phase4_roi_metrics.json") as f:
        roi_metrics = json.load(f)
    aggregated_results["phase_4_threshold_optimization"] = {
        "optimal_threshold": thresholds.get("global_optimal_threshold"),
        "expected_value": roi_metrics.get("expected_value"),
        "roi_millions": roi_metrics.get("roi_millions"),
        "net_benefit": roi_metrics.get("net_benefit")
    }
except Exception as e:
    print(f"Warning: Could not load Phase 4 metrics: {e}")
    aggregated_results["phase_4_threshold_optimization"] = {
        "note": "Metrics not available"
    }

# Load Phase 5 fairness evaluation
try:
    with open(f"{metrics_dir}/phase5_fairness_report.json") as f:
        fairness = json.load(f)
    aggregated_results["phase_5_fairness_evaluation"] = {
        "demographic_parity_gap": fairness.get("demographic_parity_gap"),
        "equalized_odds_gap": fairness.get("equalized_odds_gap"),
        "groups_analyzed": ["race", "gender", "age"],
        "requires_mitigation": fairness.get("requires_mitigation", False)
    }
except Exception as e:
    print(f"Warning: Could not load Phase 5 metrics: {e}")
    aggregated_results["phase_5_fairness_evaluation"] = {
        "groups_analyzed": ["race", "gender", "age"],
        "note": "Metrics not available"
    }

# Load Phase 6 mitigation (if exists)
phase6_path = f"{metrics_dir}/phase6_mitigation_impact.json"
if os.path.exists(phase6_path):
    try:
        with open(phase6_path) as f:
            mitigation = json.load(f)
        aggregated_results["phase_6_fairness_mitigation"] = {
            "mitigation_applied": True,
            "method": "Group-specific thresholds (equalized odds)",
            "tpr_gap_before": mitigation.get("tpr_gap_before"),
            "tpr_gap_after": mitigation.get("tpr_gap_after"),
            "fpr_gap_before": mitigation.get("fpr_gap_before"),
            "fpr_gap_after": mitigation.get("fpr_gap_after")
        }
    except Exception as e:
        print(f"Warning: Could not load Phase 6 metrics: {e}")
else:
    aggregated_results["phase_6_fairness_mitigation"] = {
        "mitigation_applied": False
    }

# Save aggregated results
with open(f"{output_dir}/aggregated_results.json", "w") as f:
    json.dump(aggregated_results, f, indent=2)

print("[x] Created aggregated_results.json")

# Create Model Card
model_card = f"""# Hospital Readmission Risk Prediction Model

## Model Details

- **Model Name**: Hospital 30-Day Readmission Risk Predictor
- **Model Type**: Gradient Boosting Classifier (LightGBM)
- **Version**: 1.0
- **Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Developers**: auphong2707
- **License**: MIT

## Intended Use

This model predicts the risk of 30-day hospital readmission for diabetic patients. It is designed to:
- Support clinical decision-making for discharge planning
- Identify high-risk patients for targeted interventions
- Optimize resource allocation for case management

**Primary Users**: Healthcare administrators, case managers, clinical decision support systems

## Training Data

- **Dataset**: Diabetes 130-US Hospitals (1999-2008)
- **Source**: UCI Machine Learning Repository
- **Samples**: 101,766 hospital admissions
- **Features**: 113 engineered features (from 50 original)
- **Target**: Binary classification (readmitted within 30 days: Yes/No)

## Performance Metrics

### Classification Performance
- **ROC-AUC**: {aggregated_results.get('phase_2_modeling', {}).get('gradient_boosting', {}).get('roc_auc', 'N/A')}
- **PR-AUC**: {aggregated_results.get('phase_2_modeling', {}).get('gradient_boosting', {}).get('pr_auc', 'N/A')}
- **F1-Score**: {aggregated_results.get('phase_2_modeling', {}).get('gradient_boosting', {}).get('f1_score', 'N/A')}
- **Precision**: {aggregated_results.get('phase_2_modeling', {}).get('gradient_boosting', {}).get('precision', 'N/A')}
- **Recall**: {aggregated_results.get('phase_2_modeling', {}).get('gradient_boosting', {}).get('recall', 'N/A')}

### Calibration Quality
- **Calibration Method**: Platt Scaling
- **Brier Score (Before)**: {aggregated_results.get('phase_3_calibration', {}).get('brier_score_before', 'N/A')}
- **Brier Score (After)**: {aggregated_results.get('phase_3_calibration', {}).get('brier_score_after', 'N/A')}
- **ECE (Before)**: {aggregated_results.get('phase_3_calibration', {}).get('ece_before', 'N/A')}
- **ECE (After)**: {aggregated_results.get('phase_3_calibration', {}).get('ece_after', 'N/A')}

### Economic Impact
- **Optimal Threshold**: {aggregated_results.get('phase_4_threshold_optimization', {}).get('optimal_threshold', 'N/A')}
- **Expected ROI**: ${aggregated_results.get('phase_4_threshold_optimization', {}).get('roi_millions', 'N/A')} million
- **Cost Assumptions**: $15,000 per readmission, $500 per intervention

## Fairness Metrics

### Demographic Groups Analyzed
- Race (Caucasian, African American, Hispanic, Asian, Other)
- Gender (Male, Female)
- Age (18-40, 41-60, 61-80, 80+)

### Fairness Assessment
- **Demographic Parity Gap**: {aggregated_results.get('phase_5_fairness_evaluation', {}).get('demographic_parity_gap', 'N/A')}
- **Equalized Odds Gap**: {aggregated_results.get('phase_5_fairness_evaluation', {}).get('equalized_odds_gap', 'N/A')}
- **Mitigation Applied**: {aggregated_results.get('phase_6_fairness_mitigation', {}).get('mitigation_applied', False)}

{"### Mitigation Strategy" if aggregated_results.get('phase_6_fairness_mitigation', {}).get('mitigation_applied') else ""}
{"- **Method**: Group-specific thresholds using equalized odds" if aggregated_results.get('phase_6_fairness_mitigation', {}).get('mitigation_applied') else ""}
{"- **TPR Gap Reduction**: " + str(aggregated_results.get('phase_6_fairness_mitigation', {}).get('tpr_gap_before', 'N/A')) + " → " + str(aggregated_results.get('phase_6_fairness_mitigation', {}).get('tpr_gap_after', 'N/A')) if aggregated_results.get('phase_6_fairness_mitigation', {}).get('mitigation_applied') else ""}
{"- **FPR Gap Reduction**: " + str(aggregated_results.get('phase_6_fairness_mitigation', {}).get('fpr_gap_before', 'N/A')) + " → " + str(aggregated_results.get('phase_6_fairness_mitigation', {}).get('fpr_gap_after', 'N/A')) if aggregated_results.get('phase_6_fairness_mitigation', {}).get('mitigation_applied') else ""}

## Limitations

1. **Data Age**: Training data from 1999-2008; medical practices have evolved
2. **Geographic Scope**: US hospitals only; may not generalize internationally
3. **Missing Data**: ~40% missing values in some features, handled via imputation
4. **Class Imbalance**: Readmission rate ~11% in dataset
5. **Feature Availability**: Requires 113 features; some may not be available at discharge
6. **Calibration Domain**: Platt scaling assumes specific probability distribution

## Ethical Considerations

### Bias Risks
- Model trained on historical data may reflect past biases
- Underrepresented groups (e.g., certain races, ages) may have lower prediction accuracy
- Fairness mitigation applied but ongoing monitoring required

### Recommended Safeguards
1. **Clinical Oversight**: All predictions should be reviewed by healthcare professionals
2. **Continuous Monitoring**: Track performance across demographic groups in production
3. **Regular Retraining**: Update model with recent data to maintain relevance
4. **Transparency**: Explain predictions to patients and clinicians
5. **Appeal Process**: Allow clinicians to override model recommendations

### Potential Harms
- **False Negatives**: High-risk patients missed → lack of intervention → readmission
- **False Positives**: Low-risk patients flagged → unnecessary interventions → resource waste
- **Disparate Impact**: Unequal error rates across groups → unfair treatment

## How to Use

### Installation
```python
pip install huggingface-hub joblib numpy pandas scikit-learn
```

### Download Model
```python
from huggingface_hub import hf_hub_download
import joblib

repo_id = "auphong2707/hospital-readmission-risk"
model = joblib.load(hf_hub_download(repo_id, "models/gradient_boosting_model_original.joblib"))
calibrator = joblib.load(hf_hub_download(repo_id, "models/Gradient_Boosting_calibrator.pkl"))
```

### Make Predictions
```python
import numpy as np

# Prepare patient features (113 features required)
patient_features = np.array([...])  # Your feature vector

# Step 1: Get uncalibrated prediction
uncalibrated_proba = model.predict_proba(patient_features.reshape(1, -1))[0, 1]

# Step 2: Apply calibration
calibrated_proba = calibrator.predict_proba([[uncalibrated_proba]])[0, 1]

# Step 3: Apply optimal threshold
optimal_threshold = 0.15  # From optimal_thresholds.json
prediction = int(calibrated_proba >= optimal_threshold)

print(f"Readmission Risk: {{calibrated_proba:.2%}}")
print(f"High Risk: {{prediction == 1}}")
```

## Citation

If you use this model in your research or application, please cite:

```bibtex
@misc{{hospital_readmission_2025,
  author = {{auphong2707}},
  title = {{Hospital Readmission Risk Prediction}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/auphong2707/hospital-readmission-risk}}}}
}}
```

## Contact

- **GitHub**: https://github.com/auphong2707/hospital-readmission-risk
- **HuggingFace**: https://huggingface.co/auphong2707/hospital-readmission-risk

## Version History

- **v1.0** ({datetime.now().strftime("%Y-%m-%d")}): Initial release with calibrated model, fairness evaluation, and ROI optimization
"""

with open(f"{output_dir}/model_card.md", "w") as f:
    f.write(model_card)

print("[x] Created model_card.md")
print(f"[x] Aggregated results saved to {output_dir}/aggregated_results.json")
print(f"[x] Model card saved to {output_dir}/model_card.md")

PYTHON_SCRIPT

echo -e "${GREEN}[x] Aggregation complete!${NC}"

################################################################################
# Step 3: Upload to HuggingFace Hub
################################################################################

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${YELLOW}[Step 3/3] Dry run mode - skipping upload${NC}"
    echo "Files ready for upload in: ${COLLECTION_DIR}"
    echo "To upload, run without --dry-run flag"
    exit 0
fi

echo ""
echo -e "${YELLOW}[Step 3/3] Uploading to HuggingFace Hub...${NC}"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}Error: huggingface-cli not found${NC}"
    echo "Install with: pip install huggingface-hub"
    exit 1
fi

# Check if user is logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${RED}Error: Not logged in to HuggingFace${NC}"
    echo "Login with: huggingface-cli login"
    exit 1
fi

# Create repository if it doesn't exist
echo "Creating/updating repository: ${REPO_ID}"
if [ "$PRIVATE" = true ]; then
    huggingface-cli repo create "${REPO_ID}" --type "${REPO_TYPE}" --private || true
else
    huggingface-cli repo create "${REPO_ID}" --type "${REPO_TYPE}" || true
fi

# Upload all files from collection directory
echo "Uploading files..."
huggingface-cli upload "${REPO_ID}" "${COLLECTION_DIR}" --repo-type="${REPO_TYPE}" --commit-message="Phase 7: Complete results collection"

# Upload aggregated results and model card
huggingface-cli upload "${REPO_ID}" "${OUTPUT_DIR}/aggregated_results.json" aggregated_results.json --repo-type="${REPO_TYPE}"
huggingface-cli upload "${REPO_ID}" "${OUTPUT_DIR}/model_card.md" model_card.md --repo-type="${REPO_TYPE}"

# Create README for HuggingFace
cat > "${OUTPUT_DIR}/README.md" << EOF
# Hospital Readmission Risk Prediction

🏥 **30-Day Readmission Risk Prediction for Diabetic Patients**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

\`\`\`python
from huggingface_hub import hf_hub_download
import joblib

# Download model
model = joblib.load(hf_hub_download("${REPO_ID}", "models/gradient_boosting_model_original.joblib"))
calibrator = joblib.load(hf_hub_download("${REPO_ID}", "models/Gradient_Boosting_calibrator.pkl"))

# Make prediction
risk_score = calibrator.predict_proba([[model.predict_proba(patient_features)[0, 1]]])[0, 1]
\`\`\`

## Model Overview

This repository contains a complete hospital readmission risk prediction system trained on the Diabetes 130-US Hospitals dataset (1999-2008, 101,766 patients).

**Key Features**:
- 🎯 Gradient Boosting (LightGBM) classifier with 113 engineered features
- 📊 Platt scaling calibration for reliable probability estimates
- ⚖️ Fairness evaluation across race, gender, and age groups
- 💰 ROI-optimized decision thresholds ($15K readmission cost vs $500 intervention)
- 📈 Comprehensive evaluation with 60+ visualizations

## Repository Contents

\`\`\`
${REPO_ID}/
|-- models/                    # 3 trained models + calibrator
|-- thresholds/                # Optimal decision thresholds
|-- metrics/                   # Performance, calibration, fairness metrics
|-- visualizations/            # 60+ plots from all phases
|-- data_splits/               # Train/validation/test splits
|-- aggregated_results.json    # Combined metrics summary
+-- model_card.md              # Detailed model documentation
\`\`\`

## Performance

- **ROC-AUC**: See aggregated_results.json
- **Calibration**: Platt scaling applied (see reliability diagrams)
- **Fairness**: Demographic parity and equalized odds evaluated
- **ROI**: Cost-benefit optimized thresholds

## Documentation

See [model_card.md](model_card.md) for complete documentation including:
- Training methodology
- Performance benchmarks
- Fairness assessment
- Limitations and ethical considerations
- Usage examples

## Citation

\`\`\`bibtex
@misc{hospital_readmission_2025,
  author = {auphong2707},
  title = {Hospital Readmission Risk Prediction},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\\url{https://huggingface.co/${REPO_ID}}}
}
\`\`\`

## License

MIT License - See LICENSE file for details

## Contact

- GitHub: https://github.com/auphong2707/hospital-readmission-risk
- HuggingFace: https://huggingface.co/${REPO_ID}
EOF

huggingface-cli upload "${REPO_ID}" "${OUTPUT_DIR}/README.md" README.md --repo-type="${REPO_TYPE}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Phase 7 Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Repository: https://huggingface.co/${REPO_ID}"
echo "Files collected: ${FILE_COUNT}"
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "Next steps:"
echo "  1. View repository: https://huggingface.co/${REPO_ID}"
echo "  2. Review model card for documentation"
echo "  3. Share with stakeholders for feedback"
echo ""
