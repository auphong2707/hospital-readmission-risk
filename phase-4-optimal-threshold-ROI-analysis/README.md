# Phase 4: Optimal Threshold & ROI Analysis

## Purpose
Determine the optimal decision threshold using cost-sensitive optimization to maximize expected value and return on investment (ROI) from readmission prevention interventions.

## Methodology

### Cost-Sensitive Threshold Optimization

**Business Context:**
- **Readmission Cost**: ~$15,000 per readmission event
- **Intervention Cost**: ~$500 per patient (discharge planning, medication reconciliation, follow-up calls)
- **Objective**: Maximize net benefit by identifying high-risk patients for targeted interventions

**Cost Matrix:**
- **True Positive (TP)**: +$14,500 (prevented readmission: $15K saved - $500 intervention)
- **False Positive (FP)**: -$500 (unnecessary intervention cost)
- **True Negative (TN)**: $0 (correct prediction, no action needed)
- **False Negative (FN)**: -$15,000 (missed readmission, no intervention)

**Optimization Process:**
1. Load calibrated probabilities from Phase 3
2. Test thresholds from 0.05 to 0.95 in small increments
3. Calculate expected value (EV) for each threshold:
   ```
   EV = (TP × $14,500) + (FP × -$500) + (FN × -$15,000)
   ```
4. Select threshold that maximizes expected value
5. Perform sensitivity analysis with different cost assumptions

### Risk Category Definition

Risk categories are derived from the optimal threshold to guide clinical intervention strategies:

- **Low Risk**: 0 to ~0.67 × optimal_threshold
  - **Action**: Standard discharge process
  - **Volume**: Majority of patients
  
- **Medium Risk**: ~0.67 × optimal_threshold to ~1.5 × optimal_threshold
  - **Action**: Enhanced follow-up call within 48 hours
  - **Volume**: Moderate intervention group
  
- **High Risk**: > 1.5 × optimal_threshold
  - **Action**: Intensive case management with home visit
  - **Volume**: Highest-priority intervention group

**Category Validation:**
- Verify actual readmission rates align with predicted risk levels
- Calculate intervention volume per risk category
- Ensure operational capacity matches intervention requirements

### Break-Even Analysis

**Break-Even Threshold:**
- Minimum readmission reduction rate for positive ROI
- Target: >3.3% absolute reduction in readmission rate
- Validates intervention cost-effectiveness

**Sensitivity Analysis:**
- Conservative scenario: Lower intervention effectiveness
- Base case: Expected intervention effectiveness
- Aggressive scenario: Higher intervention effectiveness

## Input
- Calibrated probabilities from Phase 3 (Platt Scaling output)
- True readmission outcomes from test set
- Business cost parameters (readmission cost, intervention cost)
- Patient demographic data for fairness pre-check

## Output
1. **Optimal Threshold**: Single decision threshold maximizing expected value
2. **Risk Category Thresholds**: Low/medium/high risk boundaries
3. **ROI Analysis**: Expected net benefit, cost savings, intervention volume
4. **Performance Metrics**: Confusion matrix, TPR, FPR, precision, recall at optimal threshold
5. **Resource Allocation Plan**: Intervention volume and capacity requirements
6. **Visualizations**: 8 threshold optimization and ROI analysis plots

## 📊 Visualizations

### Threshold Optimization
**Primary Plots:**
1. **Expected Value vs Threshold Curve**
   - Plot expected value across all tested thresholds
   - Identify optimal threshold maximizing EV
   - Show break-even threshold for reference

2. **Cost-Benefit Analysis**
   - Compare costs (interventions) vs benefits (prevented readmissions)
   - Visualize net benefit at different threshold levels
   - Include total cost of FPs and FNs

3. **Threshold Impact on Classification Metrics**
   - Plot precision, recall, F1-score vs threshold
   - Show TPR and FPR curves
   - Highlight optimal threshold position

### ROI & Business Impact
4. **Confusion Matrix at Optimal Threshold**
   - Visualize TP, FP, TN, FN at selected threshold
   - Annotate with financial impact per quadrant
   - Show classification accuracy metrics

5. **Risk Category Distribution**
   - Histogram of patients by risk category (low/medium/high)
   - Overlay actual readmission rates per category
   - Show intervention volume estimates

6. **ROI Sensitivity Analysis**
   - Test ROI under different cost assumptions
   - Conservative: Lower effectiveness, higher costs
   - Aggressive: Higher effectiveness, lower costs
   - Visualize ROI range and confidence intervals

### Resource Planning
7. **Intervention Volume Forecast**
   - Bar chart showing number of patients per risk category
   - Compare current state (no intervention) vs proposed (targeted intervention)
   - Include capacity requirements for each intervention type

8. **Cost Savings Projection**
   - Expected annual cost savings from intervention program
   - Break down by prevented readmissions, intervention costs, net savings
   - Project 1-year, 3-year, 5-year cumulative savings

## Success Criteria
- Optimal threshold achieves positive expected value (EV > 0)
- ROI exceeds break-even threshold (>3.3% readmission reduction)
- Risk categories show clear separation in actual readmission rates:
  - Low risk: <5% actual readmission rate
  - Medium risk: 5-15% actual readmission rate
  - High risk: >15% actual readmission rate
- Intervention volume is operationally feasible (<30% of total patients)
- Sensitivity analysis shows robust positive ROI across scenarios
- Clear resource allocation plan aligns with hospital capacity

## 🚀 Usage

### Quick Start: Complete Threshold Optimization Pipeline

```python
from phase_4_optimal_threshold.utilities import (
    ThresholdOptimizer,
    RiskCategoryMapper,
    ROIAnalyzer,
    load_calibrated_predictions
)

# Load calibrated probabilities from Phase 3
y_true, y_pred_proba_calibrated, demographics = load_calibrated_predictions(
    model_name='gradient_boosting'
)

# Define cost parameters
cost_params = {
    'readmission_cost': 15000,
    'intervention_cost': 500,
    'tp_benefit': 14500,  # Saved readmission cost minus intervention
    'fp_cost': -500,      # Wasted intervention
    'fn_cost': -15000,    # Missed readmission
    'tn_cost': 0          # Correct prediction, no cost
}

# Initialize threshold optimizer
optimizer = ThresholdOptimizer(
    y_true=y_true,
    y_pred_proba=y_pred_proba_calibrated,
    cost_params=cost_params
)

# Find optimal threshold
optimal_threshold = optimizer.find_optimal_threshold(
    threshold_range=(0.05, 0.95),
    num_points=200
)

print(f"Optimal Threshold: {optimal_threshold:.4f}")

# Calculate expected value at optimal threshold
optimal_ev = optimizer.calculate_expected_value(optimal_threshold)
print(f"Expected Value: ${optimal_ev:,.2f}")

# Define risk categories
risk_mapper = RiskCategoryMapper(
    low_threshold=optimal_threshold * 0.67,
    high_threshold=optimal_threshold * 1.5
)

# Generate risk category assignments
risk_categories = risk_mapper.assign_risk_categories(y_pred_proba_calibrated)

# Analyze ROI
roi_analyzer = ROIAnalyzer(
    y_true=y_true,
    y_pred_proba=y_pred_proba_calibrated,
    optimal_threshold=optimal_threshold,
    cost_params=cost_params
)

# Generate ROI report
roi_report = roi_analyzer.generate_roi_report()
print(roi_report)

# Generate ALL visualizations at once
visualization_paths = optimizer.generate_all_threshold_visualizations(
    risk_mapper=risk_mapper,
    output_dir='./threshold_outputs/gradient_boosting'
)

print(f"✅ Generated {len(visualization_paths)} visualizations!")
```

### Individual Analysis Examples

```python
# 1. Expected Value vs Threshold Curve
optimizer.plot_expected_value_curve(
    threshold_range=(0.05, 0.95),
    save_path='./outputs/ev_curve.png'
)

# 2. Confusion Matrix at Optimal Threshold
optimizer.plot_confusion_matrix(
    threshold=optimal_threshold,
    cost_params=cost_params,
    save_path='./outputs/confusion_matrix.png'
)

# 3. Risk Category Distribution
risk_mapper.plot_risk_distribution(
    y_pred_proba=y_pred_proba_calibrated,
    y_true=y_true,
    save_path='./outputs/risk_distribution.png'
)

# 4. ROI Sensitivity Analysis
roi_analyzer.plot_sensitivity_analysis(
    scenarios={
        'Conservative': {'tp_benefit': 12000, 'intervention_cost': 700},
        'Base Case': cost_params,
        'Aggressive': {'tp_benefit': 16000, 'intervention_cost': 400}
    },
    save_path='./outputs/roi_sensitivity.png'
)

# 5. Intervention Volume Forecast
roi_analyzer.plot_intervention_volume(
    risk_categories=risk_categories,
    save_path='./outputs/intervention_volume.png'
)

# 6. Cost Savings Projection
roi_analyzer.plot_cost_savings_projection(
    years=[1, 3, 5],
    save_path='./outputs/cost_savings.png'
)
```

### Threshold Optimization with Constraints

```python
# Optimize with operational constraints
constrained_optimizer = ThresholdOptimizer(
    y_true=y_true,
    y_pred_proba=y_pred_proba_calibrated,
    cost_params=cost_params
)

# Find threshold with maximum intervention volume constraint
optimal_threshold_constrained = constrained_optimizer.find_optimal_threshold_with_constraint(
    max_intervention_rate=0.30,  # Maximum 30% of patients
    threshold_range=(0.05, 0.95)
)

print(f"Constrained Optimal Threshold: {optimal_threshold_constrained:.4f}")
print(f"Intervention Volume: {(y_pred_proba_calibrated >= optimal_threshold_constrained).mean():.2%}")
```

## 📦 Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## 🔗 Integration with Other Phases

### Input from Phase 3 (Calibration)
- Calibrated probabilities from Platt Scaling
- Calibration quality metrics (Brier score, ECE)
- Reliability diagrams confirming probability calibration

### Output to Phase 5 (Fairness Evaluation)
- Optimal threshold for global fairness assessment
- Risk category definitions for group-specific analysis
- Expected value baseline for fairness-ROI trade-off analysis
- Resource allocation targets for fairness in intervention distribution

### Data Consistency
All phases (2-5) use Phase 1's preprocessed splits:
- **Train**: 73,526 samples (72.25%)
- **Validation**: 12,975 samples (12.75%)
- **Test**: 15,265 samples (15%) ← **Phase 4 uses this split**

```python
# Load Phase 1 splits (single source of truth)
from utilities import load_phase1_splits

X_train, X_val, X_test, y_train, y_val, y_test = load_phase1_splits()

# Phase 4 uses test set for threshold optimization
print(f"Phase 4 optimization set: {X_test.shape}")  # (15265, 113)
```

## 📋 Next Steps

After completing Phase 4:

1. **Validate Optimal Threshold**
   - Confirm threshold aligns with clinical intuition
   - Review risk categories with healthcare stakeholders
   - Verify intervention volume is operationally feasible

2. **Proceed to Phase 5 (Fairness Evaluation)**
   - Evaluate threshold fairness across demographic groups
   - Check for disparate impact in intervention allocation
   - Implement bias mitigation if needed

3. **Prepare Deployment Package**
   - Document optimal threshold and risk category definitions
   - Create clinical decision support guidelines
   - Develop resource allocation recommendations

## 📝 Notes

### Threshold Selection Considerations

**Clinical Perspective:**
- Higher threshold = fewer interventions, lower cost, higher missed readmissions
- Lower threshold = more interventions, higher cost, fewer missed readmissions
- Optimal threshold balances cost-effectiveness with clinical coverage

**Operational Perspective:**
- Intervention capacity constraints may limit feasible thresholds
- Staff availability and resource constraints must be considered
- Gradual rollout may require initial conservative thresholds

**Fairness Perspective:**
- Global optimal threshold may not be fair across demographic groups
- Phase 5 will evaluate need for group-specific thresholds
- Trade-off between overall ROI and fairness must be documented

### Limitations

- Assumes intervention effectiveness is uniform across all patients
- Cost parameters are estimates and should be validated with finance team
- Does not account for indirect costs (e.g., patient satisfaction, staff time)
- Optimal threshold is data-dependent and should be re-evaluated periodically

## 🔍 Validation Checklist

Before proceeding to Phase 5:

- [ ] Optimal threshold produces positive expected value
- [ ] ROI exceeds break-even threshold (>3.3% reduction)
- [ ] Risk categories show clear separation in readmission rates
- [ ] Intervention volume is feasible (<30% of patients)
- [ ] Sensitivity analysis confirms robust ROI
- [ ] Confusion matrix reviewed by clinical stakeholders
- [ ] Resource allocation plan approved by operations team
- [ ] Cost parameters validated by finance team
- [ ] All visualizations generated and documented
- [ ] Threshold rationale documented for stakeholders
