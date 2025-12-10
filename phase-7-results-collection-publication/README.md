# Phase 7: Results Collection & Publication

## Overview

Phase 7 collects all outputs from Phases 1-6, creates a comprehensive summary, and publishes everything to HuggingFace Hub for reproducibility and transparency.

**Single Command Execution**:
```bash
bash collect_and_publish.sh
```

This script will:
1. Collect all 89-91 files from previous phases
2. Generate `aggregated_results.json` and `model_card.md`
3. Upload everything to HuggingFace Hub

## Prerequisites

### Required Inputs from Previous Phases

| Phase | Files | Description |
|-------|-------|-------------|
| **Phase 1** | 5 files | Preprocessing metadata, train/val/test splits, demographics |
| **Phase 2** | 6 files + 27 plots | 3 models (LR, RF, GB), metrics, visualizations |
| **Phase 3** | 3 files + 1 plot | Calibrator, calibration metrics, reliability diagram |
| **Phase 4** | 4 files + 8 plots | Optimal thresholds, ROI metrics, optimization plots |
| **Phase 5** | 7 files + ~21 plots | Fairness report, group metrics, statistical tests |
| **Phase 6** | 2-3 files + 5 plots | Group thresholds, mitigation impact (optional) |

**Total**: 27-28 files + 62-63 visualizations = **89-91 items**

### Setup Requirements

1. **Install HuggingFace CLI**:
   ```bash
   pip install huggingface-hub
   ```

2. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```
   - Get token from: https://huggingface.co/settings/tokens
   - Ensure token has "Write" permission

3. **Verify setup**:
   ```bash
   huggingface-cli whoami
   ```

## Usage

### Basic Usage

```bash
# Default: Upload to public repository
bash collect_and_publish.sh
```

### Advanced Options

```bash
# Custom repository name
bash collect_and_publish.sh --repo-id your-username/custom-repo-name

# Create private repository
bash collect_and_publish.sh --private

# Preview without uploading (recommended first run)
bash collect_and_publish.sh --dry-run

# Show all options
bash collect_and_publish.sh --help
```

### What the Script Does

**Step 1: Collection**
- Scans all phase directories for outputs
- Copies files to `outputs/collection/` with organized structure:
  - `models/` - 3 trained models + calibrator
  - `thresholds/` - Optimal and group-specific thresholds
  - `metrics/` - All JSON files from 6 phases
  - `visualizations/` - All plots organized by phase
- Creates `collection_summary.txt` with file inventory

**Step 2: Aggregation**
- Extracts metrics from all JSON files
- Creates `aggregated_results.json` with combined data:
  ```json
  {
    "project_info": {...},
    "phase_2_modeling": {"gradient_boosting": {...}},
    "phase_3_calibration": {...},
    "phase_4_threshold_optimization": {...},
    "phase_5_fairness_evaluation": {...},
    "phase_6_fairness_mitigation": {...}
  }
  ```
- Generates comprehensive `model_card.md` with 9 sections:
  - Model details, intended use, training data
  - Performance, calibration, fairness metrics
  - Limitations, ethics, usage examples, citation

**Step 3: Upload**
- Creates/updates HuggingFace repository
- Uploads all files with proper structure
- Generates repository README.md
- Provides repository URL

## Output Structure

### Local Outputs
```
outputs/
|-- collection/              # All collected files
|   |-- models/
|   |-- thresholds/
|   |-- metrics/
|   |-- visualizations/
|   +-- data_splits/
|-- aggregated_results.json  # Combined metrics
|-- model_card.md            # Full documentation
|-- collection_summary.txt   # File inventory
+-- README.md                # HuggingFace README
```

### HuggingFace Repository
```
username/hospital-readmission-risk/
|-- README.md                              # Repository overview
|-- model_card.md                          # Comprehensive documentation
|-- aggregated_results.json                # All metrics combined
|-- models/
|   |-- gradient_boosting_model_original.joblib
|   |-- logistic_regression_model.joblib
|   |-- random_forest_model.joblib
|   +-- Gradient_Boosting_calibrator.pkl
|-- thresholds/
|   |-- optimal_thresholds.json
|   +-- group_thresholds.json (if Phase 6 applied)
|-- metrics/
|   |-- phase1_preprocessing_metadata.txt
|   |-- phase2_*_metrics.json (3 files)
|   |-- phase3_calibration_metrics.json
|   |-- phase4_roi_metrics.json
|   |-- phase5_fairness_report.json
|   +-- phase6_mitigation_impact.json (if applied)
+-- visualizations/
    |-- phase2_modeling/         (27 plots)
    |-- phase3_calibration/      (1 plot)
    |-- phase4_threshold_optimization/  (8 plots)
    |-- phase5_fairness_evaluation/     (~21 plots)
    +-- phase6_fairness_mitigation/     (5 plots, if applied)
```

## Usage Examples

### Download and Use Model

```python
from huggingface_hub import hf_hub_download
import joblib
import json

repo_id = "auphong2707/hospital-readmission-risk"

# Download model artifacts
model = joblib.load(hf_hub_download(repo_id, "models/gradient_boosting_model_original.joblib"))
calibrator = joblib.load(hf_hub_download(repo_id, "models/Gradient_Boosting_calibrator.pkl"))

# Load thresholds
with open(hf_hub_download(repo_id, "thresholds/optimal_thresholds.json")) as f:
    thresholds = json.load(f)

# Make prediction
def predict_risk(patient_features):
    # Step 1: Model prediction
    uncalibrated = model.predict_proba(patient_features.reshape(1, -1))[0, 1]
    
    # Step 2: Calibration
    calibrated = calibrator.predict_proba([[uncalibrated]])[0, 1]
    
    # Step 3: Risk category
    threshold = thresholds['global_optimal_threshold']
    high_risk = calibrated >= threshold
    
    return calibrated, high_risk

# Example usage
risk_score, is_high_risk = predict_risk(patient_features)
print(f"Risk: {risk_score:.2%}, High Risk: {is_high_risk}")
```

## Troubleshooting

### Authentication Issues
**Problem**: Upload fails with authentication error

**Solution**:
```bash
# Re-login
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=your_token_here  # Linux/Mac
$env:HF_TOKEN="your_token_here"  # PowerShell

# Verify
huggingface-cli whoami
```

### Bash Not Found (Windows)
**Problem**: `bash: command not found`

**Solution**:
- Install Git for Windows: https://git-scm.com/download/win
- Use Git Bash terminal (comes with Git for Windows)
- Or use WSL (Windows Subsystem for Linux)

### Permission Denied
**Problem**: Script won't execute

**Solution**:
```bash
# Linux/Mac: Make executable
chmod +x collect_and_publish.sh

# Then run
./collect_and_publish.sh
```

### Missing Files
**Problem**: Script reports missing phase outputs

**Solution**:
- Verify all phases 1-6 completed successfully
- Check phase directories for expected output files
- Phase 6 is optional - script handles missing Phase 6 gracefully
- Re-run failed phases before running Phase 7

### Repository Already Exists
**Problem**: Upload fails - repository exists

**Solution**:
- Script updates existing repository by default (no action needed)
- Or delete repository on HuggingFace and re-run
- Or use different name: `--repo-id username/different-name`

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Models** | 4 | 3 classifiers + 1 calibrator |
| **Thresholds** | 2-3 | Optimal + group-specific (optional) |
| **Metrics** | 6-7 | JSON files from each phase |
| **Metadata** | 2 | Preprocessing + aggregated results |
| **Documentation** | 1 | Comprehensive model card |
| **Visualizations** | 62-63 | All plots from Phases 2-6 |
| **TOTAL** | **77-80** | Complete results package |

## Next Steps

### For Researchers
- Download results for reproducibility studies
- Use as baseline for comparisons
- Cite model card in publications

### For Healthcare Organizations
- Download models for integration
- Review model card for deployment considerations
- Adapt thresholds to local cost parameters

### For Model Updates
- Collect recent data (post-2008)
- Re-run pipeline with new data
- Upload new version with version tags
- Compare performance across versions

## Reference

- **Repository**: https://github.com/auphong2707/hospital-readmission-risk
- **HuggingFace**: https://huggingface.co/auphong2707/hospital-readmission-risk
- **Dataset**: Diabetes 130-US Hospitals (1999-2008), 101,766 patients
- **Model**: Gradient Boosting (LightGBM), Platt calibration, fairness-aware thresholds

---

**Status**: Ready to execute (requires Phases 1-6 completed)
