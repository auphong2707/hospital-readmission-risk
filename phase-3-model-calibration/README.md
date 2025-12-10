# Phase 3: Model Calibration

## Purpose
Calibrate probability predictions using Platt Scaling to ensure reliable risk estimates for threshold optimization.

## Methodology

### Calibration Technique
**Platt Scaling**: Logistic regression transformation of predicted probabilities
- Simple, interpretable, and widely adopted in healthcare ML
- Fits a logistic regression model: calibrated_prob = sigmoid(a * uncalibrated_prob + b)
- Parameters (a, b) learned on validation set to map probabilities to true event rates

### Validation Methods
- **Reliability Diagrams**: Plot predicted vs actual probabilities (before/after calibration)
- **Brier Score**: Measure calibration quality (target: < 0.15)
- **Expected Calibration Error (ECE)**: Target < 0.05
- **Hosmer-Lemeshow Test**: Statistical calibration assessment (p-value > 0.05)

### Note on Risk Thresholds
**⚠️ Temporary thresholds (5%, 15%)** are used in Phase 3 visualizations as placeholders only. Actual thresholds will be determined in Phase 4 using cost-sensitive optimization.

## Input
- Best-performing model from Phase 2
- Raw probability predictions on validation set
- True readmission outcomes
- Patient demographic data

## Output
1. **Calibrated Probabilities**: Platt-calibrated probabilities ready for Phase 4 threshold optimization
2. **Calibration Metrics**: Brier score, ECE, H-L test, reliability diagrams
3. **Calibration Fairness**: ECE and Brier scores by demographic group
4. **Visualizations**: 8 calibration quality and fairness plots

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
   - Visualize how probabilities were adjusted by Platt Scaling

8. **Risk Score Distribution**
   - Patient distribution by temporary thresholds (5%, 15%)
   - For visualization reference only

## Success Criteria
- Brier score < 0.15
- ECE < 0.05
- Hosmer-Lemeshow p-value > 0.05
- Calibration fairness: ECE difference < 0.03 across demographic groups
- ROC-AUC preserved after calibration
- Platt Scaling improves Brier/ECE vs uncalibrated

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