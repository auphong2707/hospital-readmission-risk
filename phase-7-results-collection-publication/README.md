# Phase 7: Results Collection & Publication

## Overview

Phase 7 is the final step that collects all outputs from Phases 1-6, creates a comprehensive summary, and publishes everything to HuggingFace Hub for reproducibility, transparency, and public access.

This phase focuses on:
1. **Collecting** all critical files and visualizations from previous phases
2. **Aggregating** metrics into a unified results package
3. **Creating** a comprehensive model card documenting the entire project
4. **Publishing** everything to HuggingFace Hub as the final deliverable

## Objectives

- \u2705 Collect all 89-91 critical outputs from Phases 1-6
- \u2705 Create aggregated summary with combined metrics
- \u2705 Generate comprehensive model card
- \u2705 Upload complete results package to HuggingFace Hub
- \u2705 Ensure reproducibility and public accessibility

## Prerequisites

### Required Inputs from Previous Phases

**Phase 1 - Data Preprocessing** (5 files):
- `../data/processed/preprocessing_metadata.txt`
- `../data/processed/splits/train.csv`, `validation.csv`, `test.csv`, `test_demographics.csv`

**Phase 2 - Risk Modeling** (6 files + 27 plots):
- `../models/gradient_boosting_model_original.joblib` (primary)
- `../models/logistic_regression_model.joblib`, `../models/random_forest_model.joblib`
- Metrics: `*_metrics.json` files
- Visualizations: 9 plots per model (ROC, PR, confusion matrix, etc.)

**Phase 3 - Model Calibration** (3 files + 1 plot):
- `../calibration_outputs/gradient_boosting/Gradient_Boosting_(LightGBM)_calibrator.pkl`
- `../calibration_outputs/gradient_boosting/calibration_comparison_metrics.json`
- `../calibration_outputs/gradient_boosting/reliability_diagram_comparison.png`

**Phase 4 - Threshold Optimization** (4 files + 8 plots):
- `../phase-4-optimal-threshold-ROI-analysis/outputs/phase4_summary_for_phase5.json`
- `../phase-4-optimal-threshold-ROI-analysis/outputs/optimal_thresholds.json`
- `../phase-4-optimal-threshold-ROI-analysis/outputs/roi_metrics.json`
- `../phase-4-optimal-threshold-ROI-analysis/visualizations/*.png` (8 plots)

**Phase 5 - Fairness Evaluation** (7 files + ~21 plots):
- `../phase-5-fairness-evaluation/outputs/phase5_summary_for_phase6.json`
- `../phase-5-fairness-evaluation/outputs/fairness_report.json`
- `../phase-5-fairness-evaluation/outputs/group_metrics_*.csv` (3 files)
- `../phase-5-fairness-evaluation/outputs/statistical_tests.json`
- `../phase-5-fairness-evaluation/outputs/visualizations/*.png` (~21 plots)

**Phase 6 - Fairness Mitigation** (2-3 files + 5 plots, if applied):
- `../phase-6-fairness-mitigation-bias-correction/outputs/group_thresholds.json`
- `../phase-6-fairness-mitigation-bias-correction/outputs/mitigation_impact.json`
- `../phase-6-fairness-mitigation-bias-correction/outputs/visualizations/*.png` (5 plots)

### Dependencies

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Or use conda
conda install -c conda-forge huggingface_hub
```

**HuggingFace Setup**:
- Create account at https://huggingface.co/join
- Generate access token at https://huggingface.co/settings/tokens (with "Write" permission)
- Login via CLI: `huggingface-cli login` (enter your token when prompted)
- Or set environment variable: `export HF_TOKEN=your_token_here` (Linux/Mac) or `$env:HF_TOKEN="your_token_here"` (PowerShell)

## Project Structure

```
phase-7-results-collection-publication/
\u251c\u2500\u2500 README.md                          # This file
\u251c\u2500\u2500 collect_results.py                 # Aggregate all phase outputs
\u251c\u2500\u2500 create_model_card.py               # Generate comprehensive model card
\u251c\u2500\u2500 upload_to_huggingface.py           # Upload to HuggingFace Hub
\u251c\u2500\u2500 utilities.py                       # Helper functions
\u2514\u2500\u2500 outputs/                           # Generated outputs
    \u251c\u2500\u2500 aggregated_results.json        # All metrics combined
    \u251c\u2500\u2500 model_card.md                  # Complete model documentation
    \u251c\u2500\u2500 collection_summary.txt         # File collection report
    \u2514\u2500\u2500 visualizations/                # Summary plots (optional)
        \u251c\u2500\u2500 performance_summary.png
        \u251c\u2500\u2500 fairness_summary.png
        \u251c\u2500\u2500 roi_summary.png
        \u2514\u2500\u2500 calibration_summary.png
```

## Workflow

### Single Command Execution

```bash
# Linux/Mac
bash collect_and_publish.sh

# Windows (Git Bash)
bash collect_and_publish.sh

# Windows (PowerShell - if you have bash installed via Git for Windows)
bash collect_and_publish.sh
```

**What the script does**:

**Step 1: Collect All Results**
- Scans all phase directories for critical output files
- Verifies all 89-91 expected items are present
- Creates directory structure for organized upload
- Extracts key metrics from JSON files
- Generates `aggregated_results.json` with combined metrics
- Creates `collection_summary.txt` with inventory report

**Output Structure** (`aggregated_results.json`):
```json
{
  "project_info": {
    "name": "Hospital Readmission Risk Prediction",
    "dataset": "Diabetes 130-US Hospitals (1999-2008)",
    "total_samples": 101766,
    "collection_date": "2025-12-05"
  },
  "phase_1_preprocessing": {
    "train_samples": 73526,
    "val_samples": 12975,
    "test_samples": 15265,
    "features_engineered": 113,
    "original_features": 50
  },
  "phase_2_modeling": {
    "models_trained": 3,
    "primary_model": "Gradient Boosting (LightGBM)",
    "gradient_boosting": {
      "roc_auc": "...",
      "pr_auc": "...",
      "f1_score": "...",
      "brier_score_uncalibrated": "..."
    },
    "logistic_regression": { "..." },
    "random_forest": { "..." }
  },
  "phase_3_calibration": {
    "method": "Platt Scaling",
    "brier_score_before": "...",
    "brier_score_after": "...",
    "ece_before": "...",
    "ece_after": "...",
    "hosmer_lemeshow_pvalue": "..."
  },
  "phase_4_threshold_optimization": {
    "optimal_threshold": "...",
    "expected_value": "...",
    "roi_millions": "...",
    "risk_categories": {
      "low": "0.0 - X",
      "medium": "X - Y",
      "high": "Y - 1.0"
    }
  },
  "phase_5_fairness_evaluation": {
    "demographic_parity_gap": "...",
    "equalized_odds_gap": "...",
    "requires_mitigation": true/false,
    "groups_analyzed": ["race", "gender", "age"]
  },
  "phase_6_fairness_mitigation": {
    "mitigation_applied": true/false,
    "method": "Group-specific thresholds (equalized odds)",
    "tpr_gap_before": "...",
    "tpr_gap_after": "...",
    "fpr_gap_before": "...",
    "fpr_gap_after": "..."
**Step 2: Create Model Card**
- Generates comprehensive model card following ML best practices
- Includes all sections: model details, performance, fairness, limitations, ethics
- Outputs to `outputs/model_card.md`

**Model Card Sections**:
1. **Model Details**: Architecture, training date, version, authors
2. **Intended Use**: 30-day readmission prediction for diabetic patients
3. **Training Data**: Dataset description, sample size, features, preprocessing
4. **Performance Metrics**: ROC-AUC, precision, recall, calibration quality
5. **Fairness Metrics**: Demographic parity, equalized odds, group performance
6. **Limitations**: Data age (1999-2008), missing values, generalizability
7. **Ethical Considerations**: Bias risks, monitoring requirements, clinical oversight
8. **How to Use**: Code examples for making predictions
9. **Citation**: How to cite the project

**Step 3: Upload to HuggingFace Hub**
- Creates HuggingFace repository (or updates existing)
- Uploads all model artifacts
- Uploads aggregated results and model card
- Uploads all visualizations from Phases 2-6
- Generates README.md for the repository
- Sets repository visibility (public by default)

### Script Options

You can customize the script behavior by editing variables at the top of `collect_and_publish.sh`:

```bash
# Repository configuration
REPO_ID="auphong2707/hospital-readmission-risk"  # Change to your HF username
REPO_TYPE="model"                                 # Or "dataset" or "space"
PRIVATE=false                                     # Set to true for private repo

# Paths
PROJECT_ROOT=".."                                 # Path to project root
OUTPUT_DIR="./outputs"                            # Where to create outputs
```

Or pass arguments when running:

```bash
# Upload to custom repository
bash collect_and_publish.sh --repo-id your-username/custom-repo-name

# Create private repository
bash collect_and_publish.sh --private

# Dry run (collect files but don't upload)
bash collect_and_publish.sh --dry-run

# Show help
bash collect_and_publish.sh --help
```
```

**What it does**:
- Creates HuggingFace repository (or updates existing)
- Uploads all model artifacts
- Uploads aggregated results and model card
- Uploads all visualizations from Phases 2-6
- Generates README.md for the repository
- Sets repository visibility (public by default)

**Repository Structure on HuggingFace**:
```
502   \u251c\u2500\u2500 phase4_roi_metrics.json
\u2502   \u251c\u2500\u2500 phase5_fairness_report.json
\u2502   \u2514\u2500\u2500 phase6_mitigation_impact.json
\u2514\u2500\u2500 visualizations/
    \u251c\u2500\u2500 phase2_modeling/                       # 27 plots (9 per model)
    \u251c\u2500\u2500 phase3_calibration/                    # 1 plot
    \u251c\u2500\u2500 phase4_threshold_optimization/         # 8 plots
    \u251c\u2500\u2500 phase5_fairness_evaluation/            # ~21 plots
    \u2514\u2500\u2500 phase6_fairness_mitigation/            # 5 plots (if applied)
```

**Command Options**:
```bash
# Default upload (public repository)
python upload_to_huggingface.py

# Custom repository name
python upload_to_huggingface.py --repo-id your-username/custom-repo-name

# Private repository
python upload_to_huggingface.py --private

# Dry run (validate files without uploading)
python upload_to_huggingface.py --dry-run

# Specify HF token directly (instead of .env)
python upload_to_huggingface.py --token your_hf_token_here
```

## Key Deliverables

### 1. Aggregated Results (`outputs/aggregated_results.json`)
- \u2705 Combined metrics from all phases (1-6)
- \u2705 Complete project timeline and statistics
- \u2705 File inventory (89-91 items catalogued)
- \u2705 Machine-readable format for downstream analysis

### 2. Model Card (`outputs/model_card.md`)
- \u2705 Comprehensive model documentation (all 9 sections)
- \u2705 Performance benchmarks and calibration quality
- \u2705 Fairness assessment and mitigation strategies
- \u2705 Limitations, ethical considerations, and usage guidelines
- \u2705 Ready for publication and stakeholder review

### 3. HuggingFace Repository
- \u2705 Public repository with all artifacts
- \u2705 3 trained models + calibrator + thresholds
- \u2705 All metrics files from 6 phases
- \u2705 62-63 visualizations organized by phase
- \u2705 Versioned for reproducibility and audit trail
- \u2705 Accessible at `https://huggingface.co/username/hospital-readmission-risk`

## Success Criteria

- \u2705 All 89-91 critical files successfully collected
- \u2705 Aggregated results JSON validates against schema
- \u2705 Model card includes all 9 required sections
- \u2705 HuggingFace upload completes without errors
- \u2705 Repository is publicly accessible (or private if specified)
- \u2705 All visualizations are properly organized by phase
- \u2705 README.md auto-generated for HF repository

## Usage Examples

### Download Complete Results from HuggingFace

```python
from huggingface_hub import hf_hub_download
import joblib
import json

repo_id = "auphong2707/hospital-readmission-risk"

# Download model and calibrator
model = joblib.load(hf_hub_download(repo_id, "models/gradient_boosting_model_original.joblib"))
calibrator = joblib.load(hf_hub_download(repo_id, "models/Gradient_Boosting_calibrator.pkl"))

# Download thresholds
thresholds_path = hf_hub_download(repo_id, "thresholds/optimal_thresholds.json")
with open(thresholds_path) as f:
    thresholds = json.load(f)

# Download aggregated results
results_path = hf_hub_download(repo_id, "aggregated_results.json")
with open(results_path) as f:
    all_results = json.load(f)

print(f"Model: {type(model).__name__}")
print(f"Optimal Threshold: {thresholds['global_optimal_threshold']}")
print(f"ROC-AUC: {all_results['phase_2_modeling']['gradient_boosting']['roc_auc']}")
```

### Make Predictions with Downloaded Model

```python
import numpy as np

def predict_readmission_risk(patient_features, model, calibrator, thresholds):
    """
    Predict 30-day readmission risk with calibrated probabilities.
    
    Args:
        patient_features: Array of 113 engineered features
        model: Trained Gradient Boosting model
        calibrator: Platt scaling calibrator
        thresholds: Dict with optimal thresholds
    
    Returns:
        risk_score: Calibrated probability (0-1)
        risk_category: 'low', 'medium', or 'high'
        recommended_action: Clinical intervention
    """
    # Step 1: Model prediction
    uncalibrated_proba = model.predict_proba(patient_features.reshape(1, -1))[0, 1]
    
    # Step 2: Calibration
    calibrated_proba = calibrator.predict_proba([[uncalibrated_proba]])[0, 1]
    
    # Step 3: Risk categorization
    if calibrated_proba >= thresholds['high_risk_threshold']:
        category = 'high'
        action = 'Intensive case management + home visit'
    elif calibrated_proba >= thresholds['medium_risk_threshold']:
        category = 'medium'
        action = 'Enhanced follow-up call within 48 hours'
    else:
        category = 'low'
        action = 'Standard discharge planning'
    
    return calibrated_proba, category, action

### Issue: Missing Phase Outputs
**Symptom**: Script fails with "File not found" errors.

**Solution**: Ensure all phases (1-6) completed successfully. Check:
print(f"Risk Score: {risk_score:.3f}")
print(f"Risk Category: {category}")
print(f"Recommended Action: {action}")
```

## Troubleshooting

### Issue: HuggingFace Upload Fails
**Symptom**: Authentication or permission errors during upload.

**Solution**:
- Login first: `huggingface-cli login`
- Or set token: `export HF_TOKEN=your_token_here` (Linux/Mac) or `$env:HF_TOKEN="your_token_here"` (PowerShell)
- Check token has "Write" permission: https://huggingface.co/settings/tokens
- Test token: `huggingface-cli whoami`
- Ensure repository name is valid (lowercase, hyphens only, no underscores)
- For private repos: Check your HF plan supports private repositories

### Issue: Bash Not Found (Windows)
**Symptom**: `bash: command not found` or similar error on Windows.

**Solution**:
- Install Git for Windows (includes Git Bash): https://git-scm.com/download/win
- Use Git Bash terminal instead of PowerShell/CMD
- Or install WSL (Windows Subsystem for Linux)
- Or convert script to PowerShell (create `collect_and_publish.ps1`)

### Issue: Permission Denied
**Symptom**: `Permission denied` when running script.

**Solution**:
```bash
# Make script executable (Linux/Mac)
chmod +x collect_and_publish.sh

# Then run
./collect_and_publish.sh
```

### Issue: Repository Already Exists
**Symptom**: Upload fails with "repository already exists" error.

**Solution**:
- The script should update existing repository by default
- Or manually delete repository on HuggingFace and re-run
- Or use different repository name: Edit `REPO_ID` in script or use `--repo-id username/different-name`
- Check for missing Phase 6 outputs if mitigation wasn't applied (should be handled gracefully)
- Re-run `collect_results.py` to ensure complete aggregation

### Issue: Repository Already Exists
**Symptom**: Upload fails with "repository already exists" error.

**Solution**:
- The script should update existing repository by default
- Use `--force` flag to overwrite: `python upload_to_huggingface.py --force`
- Or manually delete repository on HuggingFace and re-upload
- Or use different repository name: `--repo-id username/different-name`

## Summary: What Gets Published

| Category | Count | Description |
|----------|-------|-------------|
| **Model Files** | 4 | 3 models + 1 calibrator |
| **Configuration** | 2-3 | Optimal thresholds + group thresholds (if applied) |
| **Metrics** | 6-7 | JSON files from each phase |
| **Metadata** | 2 | Preprocessing metadata + aggregated results |
| **Documentation** | 1 | Comprehensive model card |
| **Visualizations** | 62-63 | All plots from Phases 2-6 |
| **TOTAL** | **77-80 items** | Complete results package on HuggingFace |

## Next Steps After Phase 7

### For Researchers
- Download complete results from HuggingFace for reproducibility
- Cite the model card in publications
- Use as baseline for future studies
- Extend methodology to other healthcare prediction tasks

### For Healthcare Organizations
- Download model artifacts for integration
- Review model card for deployment considerations
- Assess fairness metrics for your patient population
- Adapt thresholds based on local cost parameters

### For Model Improvement
- Collect recent data (post-2008) for retraining
- Re-run entire pipeline (Phases 1-7) with updated data
- Upload new version to HuggingFace with version tags
- Compare performance between versions

## References

- **Phase 1**: Data preprocessing, feature engineering, splits
- **Phase 2**: 3 models trained (Logistic, RF, GBM)
- **Phase 3**: Platt scaling calibration
- **Phase 4**: Cost-sensitive threshold optimization
- **Phase 5**: Fairness evaluation (demographic parity, equalized odds)
- **Phase 6**: Bias mitigation via group-specific thresholds

## Contact

For questions about Phase 7 results collection and publication:
- Open an issue on GitHub
- Visit the HuggingFace repository for documentation
- Contact project maintainers

---

**Status**: \ud83d\udd32 Ready to implement (Phases 1-6 completed)

**HuggingFace Repository**: Will be created at `https://huggingface.co/username/hospital-readmission-risk`
