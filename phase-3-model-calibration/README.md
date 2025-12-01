# Phase 3: Model Calibration

## Purpose
Ensure predicted probabilities accurately reflect actual readmission risk for reliable clinical decision-making.

## Methodology

### Calibration Technique
**Platt Scaling**: Logistic regression transformation of predicted probabilities
- Simple, interpretable, and widely adopted in healthcare ML
- Fits a logistic regression model: calibrated_prob = sigmoid(a * uncalibrated_prob + b)
- Parameters (a, b) learned on validation set to map probabilities to true event rates

### Validation Methods
- **Reliability Diagrams**: Plot predicted vs. observed probabilities
- **Brier Score**: Measure calibration quality (target: < 0.15)
- **Hosmer-Lemeshow Test**: Statistical calibration assessment (p-value > 0.05)

### Note on Risk Categories
**⚠️ Important**: Risk category thresholds shown in Phase 3 visualizations are **temporary placeholders** for visualization purposes only. The actual risk thresholds will be **determined in Phase 4** using cost-sensitive threshold optimization.

**Temporary thresholds used in Phase 3 visualizations:**
| Risk Category | Probability | Purpose |
|--------------|-------------|---------|
| Low | 0-5% | Visualization baseline only |
| Medium | 5-15% | Visualization reference only |
| High | 15%+ | Visualization reference only |

**Actual thresholds (Phase 4)** will be derived from:
- Optimal decision threshold that maximizes expected value
- Cost matrix ($15K readmission cost vs $500 intervention cost)
- ROI analysis and resource allocation constraints

## Input
- Best-performing model from Phase 2
- Raw probability predictions on validation set
- True readmission outcomes
- Patient demographic data

## Output
1. **Calibrated Probabilities**: Reliable probabilities ready for threshold optimization in Phase 4
2. **Calibration Report**: Reliability diagrams, Brier scores, H-L test results, ECE metrics
3. **Calibration Method Selection**: Best calibration technique (Platt vs Isotonic vs Group-specific)
4. **Calibration Fairness Assessment**: ECE and Brier scores by demographic group
5. **Comprehensive Visualizations**: See visualization section below

**Note**: Risk category thresholds and clinical decision rules are **not finalized in Phase 3**. They will be determined in Phase 4 through cost-sensitive threshold optimization.

## 📊 Visualizations

### Calibration Quality Assessment
**Primary Plots:**
1. **Before/After Calibration Comparison**
   - Side-by-side reliability diagrams: uncalibrated vs Platt-calibrated
   - Shows improvement in probability-observation alignment
   - Includes Brier score and ECE for both versions

2. **Calibration Improvement Metrics**
   - Bar chart comparing uncalibrated vs calibrated: Brier score, ECE, log loss
   - Demonstrates quantitative improvement from Platt Scaling
   - All metrics should improve after calibration

### Fairness & Group Analysis
5. **Group-Specific Calibration Curves**
   - Separate calibration plots by demographic groups (race, gender, age)
   - Check if calibration quality is consistent across groups
   - Identify if certain groups need group-specific calibration

6. **Calibration Fairness Metrics**
   - ECE and Brier score by demographic group
   - Statistical tests for calibration differences across groups
   - Table showing metrics by protected attributes

### Probability Analysis
7. **Probability Distribution Changes**
   - Histograms showing before/after calibration
   - Visualize how extreme probabilities were adjusted
   - Overlay **temporary** risk category thresholds (for reference only)
   - **Note**: Thresholds shown are placeholders; actual thresholds determined in Phase 4

8. **Risk Score Distribution (Preliminary)**
   - Patients by **temporary** risk category (Low/Medium/High)
   - Bar chart with counts and percentages using placeholder thresholds (5%, 15%)
   - **Purpose**: Visualize probability distribution shape, not final clinical stratification
   - **Final risk categories**: Will be defined in Phase 4 based on ROI optimization

### Reused from Phase 2
- ROC/PR curves for reference (model discrimination preserved)
- Confusion matrix at optimal threshold

## Success Criteria
- Brier score < 0.15 (probability accuracy)
- ECE < 0.05 (Expected Calibration Error - within ±5% of perfect calibration)
- Predictions within ±5% of diagonal on reliability plot
- Hosmer-Lemeshow p-value > 0.05 (statistical calibration goodness-of-fit)
- Calibration fairness: ECE difference < 0.03 across demographic groups
- ROC-AUC preserved (calibration shouldn't reduce discrimination ability)
- Platt Scaling improves Brier score and ECE vs uncalibrated baseline

**Note**: Clinical risk categories and decision thresholds are **NOT** success criteria for Phase 3. These will be validated in Phase 4 after threshold optimization.

## 🚀 Usage

### Quick Start: Generate All Visualizations

```python
from phase_3_model_calibration.utilities import (
    CalibrationVisualizer, 
    RiskScoreMapper,
    download_model_from_hf,
    load_data
)

# Load model and data
model, summary = download_model_from_hf()
X, y = load_data(from_huggingface=True)

# Get predictions
y_pred_proba_uncalibrated = model.predict_proba(X_test)[:, 1]

# Calibrate (using your chosen method)
# ... calibration code ...
y_pred_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

# Create risk mapper
risk_mapper = RiskScoreMapper(
    low_threshold=0.05,
    high_threshold=0.15
)

# Generate ALL visualizations at once
visualization_paths = CalibrationVisualizer.generate_all_calibration_visualizations(
    y_true=y_test,
    y_pred_proba_uncalibrated=y_pred_proba_uncalibrated,
    y_pred_proba_calibrated=platt_calibrated_proba,
    probabilities_dict={
        'Uncalibrated': y_pred_proba_uncalibrated,
        'Platt Scaling': platt_calibrated_proba
    },
    risk_mapper=risk_mapper,
    groups=demographics['race'].values,
    group_name='Race',
    output_dir='./calibration_outputs/gradient_boosting',
    n_bins=10
)

print(f"✅ Generated {len(visualization_paths)} visualizations!")
```

### Individual Visualization Examples

```python
# 1. Before/After Reliability Diagram
CalibrationVisualizer.plot_reliability_diagram(
    y_true=y_test,
    y_pred_proba_uncalibrated=uncalibrated_proba,
    y_pred_proba_calibrated=calibrated_proba,
    save_path='./outputs/reliability_diagram.png'
)

# 2. Calibration Improvement Metrics
CalibrationVisualizer.plot_brier_score_comparison(
    y_true=y_test,
    probabilities_dict={
        'Uncalibrated': uncalibrated_proba,
        'Platt Scaling': platt_proba
    },
    save_path='./outputs/calibration_improvement.png'
)

# 5. Probability Distribution Changes
CalibrationVisualizer.plot_probability_distribution_changes(
    y_pred_proba_before=uncalibrated_proba,
    y_pred_proba_after=calibrated_proba,
    risk_thresholds=(0.05, 0.15),
    save_path='./outputs/prob_distribution.png'
)

# 6. Group-Specific Calibration
CalibrationVisualizer.plot_group_calibration(
    y_true=y_test,
    y_pred_proba=calibrated_proba,
    groups=demographics['race'].values,
    group_name='Race',
    save_path='./outputs/group_calibration_race.png'
)

# 7. Calibration Fairness Metrics
CalibrationVisualizer.plot_calibration_fairness_metrics(
    y_true=y_test,
    y_pred_proba=calibrated_proba,
    groups=demographics['gender'].values,
    group_name='Gender',
    save_path='./outputs/fairness_metrics_gender.png'
)
```

## 📦 Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```