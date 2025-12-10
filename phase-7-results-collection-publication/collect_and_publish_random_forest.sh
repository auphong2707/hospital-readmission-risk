#!/bin/bash

################################################################################
# Phase 7: Results Collection & Publication - Random Forest Model
# Single script to collect all outputs from Phases 1-6 and upload to HuggingFace
################################################################################

set -e  # Exit on error

# Default configuration
REPO_ID="auphong2707/hospital-readmission-risk-random-forest"
REPO_TYPE="model"
PRIVATE=false
DRY_RUN=false
PROJECT_ROOT=".."
OUTPUT_DIR="./outputs_random_forest"
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
        --help)
            echo "Usage: bash collect_and_publish_random_forest.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --repo-id <username/repo>   HuggingFace repository ID (default: auphong2707/hospital-readmission-risk-random-forest)"
            echo "  --private                   Create private repository"
            echo "  --dry-run                   Collect files but don't upload"
            echo "  --help                      Show this help message"
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
echo -e "${GREEN}Random Forest Model${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
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
mkdir -p "${COLLECTION_DIR}/thresholds"
mkdir -p "${COLLECTION_DIR}/metrics"
mkdir -p "${COLLECTION_DIR}/visualizations/phase2_modeling"
mkdir -p "${COLLECTION_DIR}/visualizations/phase3_calibration"
mkdir -p "${COLLECTION_DIR}/visualizations/phase4_threshold_optimization"
mkdir -p "${COLLECTION_DIR}/visualizations/phase5_fairness_evaluation"
mkdir -p "${COLLECTION_DIR}/visualizations/phase6_fairness_mitigation"

# Initialize collection summary
SUMMARY_FILE="${OUTPUT_DIR}/collection_summary.txt"
echo "Phase 7: Results Collection Summary - Random Forest Model" > "${SUMMARY_FILE}"
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

# Phase 1: Data Preprocessing (shared across models)
echo "" | tee -a "${SUMMARY_FILE}"
echo "Phase 1 - Data Preprocessing:" | tee -a "${SUMMARY_FILE}"
copy_if_exists "${PROJECT_ROOT}/data/processed/preprocessing_metadata.txt" \
    "${COLLECTION_DIR}/metrics/phase1_preprocessing_metadata.txt" \
    "Preprocessing metadata"
copy_if_exists "${PROJECT_ROOT}/data/processed/splits/train.csv" \
    "${COLLECTION_DIR}/data_splits/train.csv" \
    "Training split" && mkdir -p "${COLLECTION_DIR}/data_splits"
copy_if_exists "${PROJECT_ROOT}/data/processed/splits/validation.csv" \
    "${COLLECTION_DIR}/data_splits/validation.csv" \
    "Validation split"
copy_if_exists "${PROJECT_ROOT}/data/processed/splits/test.csv" \
    "${COLLECTION_DIR}/data_splits/test.csv" \
    "Test split"
copy_if_exists "${PROJECT_ROOT}/data/processed/splits/test_demographics.csv" \
    "${COLLECTION_DIR}/data_splits/test_demographics.csv" \
    "Test demographics"

# Phase 2: Risk Modeling - Random Forest
echo "" | tee -a "${SUMMARY_FILE}"
echo "Phase 2 - Risk Modeling (Random Forest):" | tee -a "${SUMMARY_FILE}"
copy_if_exists "${PROJECT_ROOT}/models/random_forest_model.joblib" \
    "${COLLECTION_DIR}/models/random_forest_model.joblib" \
    "Random Forest model"
copy_if_exists "${PROJECT_ROOT}/phase-2-risk-modeling/Random_Forest_metrics.json" \
    "${COLLECTION_DIR}/metrics/phase2_random_forest_metrics.json" \
    "Random Forest metrics"

# Copy Phase 2 visualizations - Random Forest only
for viz in "ROC_Curve" "Precision_Recall_Curve" "Confusion_Matrix" "Feature_Importance_Top_20" \
           "Calibration_Plot" "Prediction_Distribution" "Threshold_Metrics" \
           "Classification_Report" "Learning_Curves"; do
    src="${PROJECT_ROOT}/phase-2-risk-modeling/Random_Forest_${viz}.png"
    if [ -f "${src}" ]; then
        cp "${src}" "${COLLECTION_DIR}/visualizations/phase2_modeling/"
        ((FILE_COUNT++))
    fi
done
echo "  [x] Phase 2 visualizations (9 plots)" | tee -a "${SUMMARY_FILE}"

# Phase 3: Model Calibration - Random Forest
echo "" | tee -a "${SUMMARY_FILE}"
echo "Phase 3 - Model Calibration (Random Forest):" | tee -a "${SUMMARY_FILE}"
copy_if_exists "${PROJECT_ROOT}/calibration_outputs/random_forest/Random_Forest_calibrator.pkl" \
    "${COLLECTION_DIR}/models/Random_Forest_calibrator.pkl" \
    "Platt calibrator"
copy_if_exists "${PROJECT_ROOT}/calibration_outputs/random_forest/calibration_comparison_metrics.json" \
    "${COLLECTION_DIR}/metrics/phase3_calibration_metrics.json" \
    "Calibration metrics"
copy_if_exists "${PROJECT_ROOT}/calibration_outputs/random_forest/reliability_diagram_comparison.png" \
    "${COLLECTION_DIR}/visualizations/phase3_calibration/reliability_diagram_comparison.png" \
    "Reliability diagram"

# Phase 4: Threshold Optimization - Random Forest
echo "" | tee -a "${SUMMARY_FILE}"
echo "Phase 4 - Threshold Optimization (Random Forest):" | tee -a "${SUMMARY_FILE}"
copy_if_exists "${PROJECT_ROOT}/phase-4-optimal-threshold-ROI-analysis/outputs_random_forest/optimal_thresholds.json" \
    "${COLLECTION_DIR}/thresholds/optimal_thresholds.json" \
    "Optimal thresholds"
copy_if_exists "${PROJECT_ROOT}/phase-4-optimal-threshold-ROI-analysis/outputs_random_forest/roi_metrics.json" \
    "${COLLECTION_DIR}/metrics/phase4_roi_metrics.json" \
    "ROI metrics"
copy_if_exists "${PROJECT_ROOT}/phase-4-optimal-threshold-ROI-analysis/outputs_random_forest/phase4_summary_for_phase5.json" \
    "${COLLECTION_DIR}/metrics/phase4_summary.json" \
    "Phase 4 summary"

# Copy Phase 4 visualizations
for viz in "${PROJECT_ROOT}/phase-4-optimal-threshold-ROI-analysis/visualizations_random_forest"/*.png; do
    if [ -f "${viz}" ]; then
        cp "${viz}" "${COLLECTION_DIR}/visualizations/phase4_threshold_optimization/"
        ((FILE_COUNT++))
    fi
done
echo "  [x] Phase 4 visualizations (8 plots)" | tee -a "${SUMMARY_FILE}"

# Phase 5: Fairness Evaluation - Random Forest
echo "" | tee -a "${SUMMARY_FILE}"
echo "Phase 5 - Fairness Evaluation (Random Forest):" | tee -a "${SUMMARY_FILE}"
copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs_random_forest/fairness_report.json" \
    "${COLLECTION_DIR}/metrics/phase5_fairness_report.json" \
    "Fairness report"
copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs_random_forest/phase5_summary_for_phase6.json" \
    "${COLLECTION_DIR}/metrics/phase5_summary.json" \
    "Phase 5 summary"
copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs_random_forest/statistical_tests.json" \
    "${COLLECTION_DIR}/metrics/phase5_statistical_tests.json" \
    "Statistical tests"
copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs_random_forest/group_metrics_race.csv" \
    "${COLLECTION_DIR}/metrics/phase5_group_metrics_race.csv" \
    "Group metrics (race)"
copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs_random_forest/group_metrics_gender.csv" \
    "${COLLECTION_DIR}/metrics/phase5_group_metrics_gender.csv" \
    "Group metrics (gender)"
copy_if_exists "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs_random_forest/group_metrics_age.csv" \
    "${COLLECTION_DIR}/metrics/phase5_group_metrics_age.csv" \
    "Group metrics (age)"

# Copy Phase 5 visualizations
for viz in "${PROJECT_ROOT}/phase-5-fairness-evaluation/outputs_random_forest/visualizations"/*.png; do
    if [ -f "${viz}" ]; then
        cp "${viz}" "${COLLECTION_DIR}/visualizations/phase5_fairness_evaluation/"
        ((FILE_COUNT++))
    fi
done
echo "  [x] Phase 5 visualizations (~21 plots)" | tee -a "${SUMMARY_FILE}"

# Phase 6: Fairness Mitigation (optional) - Random Forest
echo "" | tee -a "${SUMMARY_FILE}"
echo "Phase 6 - Fairness Mitigation (Random Forest):" | tee -a "${SUMMARY_FILE}"
if [ -f "${PROJECT_ROOT}/phase-6-fairness-mitigation-bias-correction/outputs_random_forest/group_thresholds.json" ]; then
    copy_if_exists "${PROJECT_ROOT}/phase-6-fairness-mitigation-bias-correction/outputs_random_forest/group_thresholds.json" \
        "${COLLECTION_DIR}/thresholds/group_thresholds.json" \
        "Group-specific thresholds"
    copy_if_exists "${PROJECT_ROOT}/phase-6-fairness-mitigation-bias-correction/outputs_random_forest/mitigation_impact.json" \
        "${COLLECTION_DIR}/metrics/phase6_mitigation_impact.json" \
        "Mitigation impact"
    
    # Copy Phase 6 visualizations
    for viz in "${PROJECT_ROOT}/phase-6-fairness-mitigation-bias-correction/outputs_random_forest/visualizations"/*.png; do
        if [ -f "${viz}" ]; then
            cp "${viz}" "${COLLECTION_DIR}/visualizations/phase6_fairness_mitigation/"
            ((FILE_COUNT++))
        fi
    done
    echo "  [x] Phase 6 visualizations (5 plots)" | tee -a "${SUMMARY_FILE}"
else
    echo "  [ ] Phase 6 not applied (mitigation optional)" | tee -a "${SUMMARY_FILE}"
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
project_root = ".."
output_dir = "./outputs_random_forest"
collection_dir = f"{output_dir}/collection"
metrics_dir = f"{collection_dir}/metrics"

# Initialize aggregated results
aggregated_results = {
    "project_info": {
        "name": "Hospital Readmission Risk Prediction - Random Forest",
        "model_type": "Random Forest",
        "dataset": "Diabetes 130-US Hospitals (1999-2008)",
        "collection_date": datetime.now().strftime("%Y-%m-%d"),
        "repository": "https://github.com/auphong2707/hospital-readmission-risk"
    }
}

# Load Phase 2 metrics (Random Forest)
try:
    with open(f"{metrics_dir}/phase2_random_forest_metrics.json") as f:
        rf_metrics = json.load(f)
    aggregated_results["phase_2_modeling"] = {
        "model": "Random Forest",
        "roc_auc": rf_metrics.get("roc_auc"),
        "pr_auc": rf_metrics.get("pr_auc"),
        "f1_score": rf_metrics.get("f1_score"),
        "precision": rf_metrics.get("precision"),
        "recall": rf_metrics.get("recall"),
        "brier_score": rf_metrics.get("brier_score")
    }
except Exception as e:
    print(f"Warning: Could not load Phase 2 metrics: {e}")

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
model_card = f"""# Hospital Readmission Risk Prediction Model - Random Forest

## Model Details

- **Model Name**: Hospital 30-Day Readmission Risk Predictor (Random Forest)
- **Model Type**: Random Forest Classifier with Platt Calibration
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
- **ROC-AUC**: {aggregated_results.get('phase_2_modeling', {}).get('roc_auc', 'N/A')}
- **PR-AUC**: {aggregated_results.get('phase_2_modeling', {}).get('pr_auc', 'N/A')}
- **F1-Score**: {aggregated_results.get('phase_2_modeling', {}).get('f1_score', 'N/A')}
- **Precision**: {aggregated_results.get('phase_2_modeling', {}).get('precision', 'N/A')}
- **Recall**: {aggregated_results.get('phase_2_modeling', {}).get('recall', 'N/A')}

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

## Model Architecture

- **Algorithm**: Random Forest (ensemble of decision trees)
- **Framework**: scikit-learn 1.2.2
- **Trees**: Configured via nested cross-validation
- **Bootstrap**: Yes (with out-of-bag scoring)
- **Max Features**: Sqrt (default for classification)
- **Calibration**: Post-hoc Platt Scaling for probability calibration

## Limitations

1. **Data Age**: Training data from 1999-2008; medical practices have evolved
2. **Geographic Scope**: US hospitals only; may not generalize internationally
3. **Missing Data**: ~40% missing values in some features, handled via imputation
4. **Class Imbalance**: Readmission rate ~11% in dataset
5. **Feature Availability**: Requires 113 features; some may not be available at discharge
6. **Calibration Quality**: Random Forest may have higher Expected Calibration Error (ECE) compared to gradient boosting methods due to tree voting mechanism
7. **Interpretability**: Ensemble model less interpretable than single decision tree

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

repo_id = "auphong2707/hospital-readmission-risk-random-forest"
model = joblib.load(hf_hub_download(repo_id, "models/random_forest_model.joblib"))
calibrator = joblib.load(hf_hub_download(repo_id, "models/Random_Forest_calibrator.pkl"))
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

## Comparison with Gradient Boosting

This Random Forest model is part of a multi-model comparison study. For the Gradient Boosting (LightGBM) model, see:
- Repository: https://huggingface.co/auphong2707/hospital-readmission-risk
- GitHub: https://github.com/auphong2707/hospital-readmission-risk

Key differences:
- **Random Forest**: Better for handling overfitting, robust to outliers, fully parallel training
- **Gradient Boosting**: Often better calibration, potentially higher accuracy, sequential training

## Citation

If you use this model in your research or application, please cite:

```bibtex
@misc{{hospital_readmission_rf_2025,
  author = {{auphong2707}},
  title = {{Hospital Readmission Risk Prediction - Random Forest}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/auphong2707/hospital-readmission-risk-random-forest}}}}
}}
```

## Contact

- **GitHub**: https://github.com/auphong2707/hospital-readmission-risk
- **HuggingFace**: https://huggingface.co/auphong2707/hospital-readmission-risk-random-forest

## Version History

- **v1.0** ({datetime.now().strftime("%Y-%m-%d")}): Initial release with calibrated Random Forest model, fairness evaluation, and ROI optimization
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
huggingface-cli upload "${REPO_ID}" "${COLLECTION_DIR}" --repo-type="${REPO_TYPE}" --commit-message="Phase 7: Complete results collection for Random Forest model"

# Upload aggregated results and model card
huggingface-cli upload "${REPO_ID}" "${OUTPUT_DIR}/aggregated_results.json" aggregated_results.json --repo-type="${REPO_TYPE}"
huggingface-cli upload "${REPO_ID}" "${OUTPUT_DIR}/model_card.md" model_card.md --repo-type="${REPO_TYPE}"

# Create README for HuggingFace
cat > "${OUTPUT_DIR}/README.md" << EOF
# Hospital Readmission Risk Prediction - Random Forest

🏥 **30-Day Readmission Risk Prediction for Diabetic Patients (Random Forest Model)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

\`\`\`python
from huggingface_hub import hf_hub_download
import joblib

# Download model
model = joblib.load(hf_hub_download("${REPO_ID}", "models/random_forest_model.joblib"))
calibrator = joblib.load(hf_hub_download("${REPO_ID}", "models/Random_Forest_calibrator.pkl"))

# Make prediction
risk_score = calibrator.predict_proba([[model.predict_proba(patient_features)[0, 1]]])[0, 1]
\`\`\`

## Model Overview

This repository contains a Random Forest-based hospital readmission risk prediction system trained on the Diabetes 130-US Hospitals dataset (1999-2008, 101,766 patients).

**Key Features**:
- 🌲 Random Forest classifier (scikit-learn 1.2.2) with 113 engineered features
- 📊 Platt scaling calibration for reliable probability estimates
- ⚖️ Fairness evaluation across race, gender, and age groups
- 💰 ROI-optimized decision thresholds ($15K readmission cost vs $500 intervention)
- 📈 Comprehensive evaluation with 40+ visualizations

## Repository Contents

\`\`\`
${REPO_ID}/
|-- models/                    # Random Forest model + calibrator
|-- thresholds/                # Optimal decision thresholds
|-- metrics/                   # Performance, calibration, fairness metrics
|-- visualizations/            # 40+ plots from all phases
|-- data_splits/               # Train/validation/test splits
|-- aggregated_results.json    # Combined metrics summary
+-- model_card.md              # Detailed model documentation
\`\`\`

## Performance

- **ROC-AUC**: See aggregated_results.json
- **Calibration**: Platt scaling applied (see reliability diagrams)
- **Fairness**: Demographic parity and equalized odds evaluated
- **ROI**: Cost-benefit optimized thresholds

## Model Comparison

This is the **Random Forest** variant. For comparison with **Gradient Boosting (LightGBM)**:
- Repository: https://huggingface.co/auphong2707/hospital-readmission-risk
- GitHub: https://github.com/auphong2707/hospital-readmission-risk

## Documentation

See [model_card.md](model_card.md) for complete documentation including:
- Training methodology
- Performance benchmarks
- Fairness assessment
- Limitations and ethical considerations
- Usage examples

## Citation

\`\`\`bibtex
@misc{hospital_readmission_rf_2025,
  author = {auphong2707},
  title = {Hospital Readmission Risk Prediction - Random Forest},
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
echo "  3. Compare with Gradient Boosting model: https://huggingface.co/auphong2707/hospital-readmission-risk"
echo "  4. Share with stakeholders for feedback"
echo ""
