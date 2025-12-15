# Phase 6: Final System Evaluation

## Overview

**Phase 6** performs comprehensive final evaluation of the deployed hospital readmission prediction system using the threshold configuration determined in **Phase 5** (either global thresholds from Phase 4 or group-specific thresholds from Phase 5 mitigation).

This phase produces the **single source of truth** metrics (`final_system_metrics.json`) that will be used in Phase 7 for publication and deployment decisions.

## Purpose

While previous phases evaluated different aspects of the model:
- **Phase 2**: Initial model performance
- **Phase 3**: Calibration quality
- **Phase 4**: Threshold optimization and ROI
- **Phase 5**: Fairness assessment and mitigation

**Phase 6** evaluates the **complete deployed system** with:
- The **actual thresholds** that will be used in production
- **All metrics** in one comprehensive evaluation
- **Stakeholder-friendly** deployment reports
- **Complete visualizations** for all aspects of system performance

## Key Differences from Previous Phases

| Aspect | Previous Phases | Phase 6 |
|--------|----------------|---------|
| **Threshold** | Varies by phase (0.5, optimal, group-specific) | **Deployed configuration** from Phase 5 |
| **Metrics** | Specialized (calibration, ROI, fairness) | **All metrics** in one place |
| **Purpose** | Development and tuning | **Final deployment evaluation** |
| **Output** | Phase-specific files | **Single source of truth** |
| **Audience** | Data scientists | **Clinical stakeholders + scientists** |

## Inputs

### Required Files (from Phase 5)
- `deployment_config.json` - Contains deployed threshold configuration
  - Located in Phase 5 fairness assessment outputs
  - Available in HuggingFace repository for each method

### Data Sources (HuggingFace)
- Test data from Phase 1: `auphong2707/hospital-readmission-risk-data`
- Calibrated predictions from Phase 3:
  - Gradient Boosting: `auphong2707/hospital-readmission-phase3-lgbm-calibrated`
  - Random Forest: `auphong2707/hospital-readmission-phase3-rf-calibrated`
  - Logistic Regression: `auphong2707/hospital-readmission-phase3-lr-calibrated`
- Deployment configs from Phase 5:
  - Gradient Boosting: `auphong2707/hospital-readmission-phase5-lgbm-fairness`
  - Random Forest: `auphong2707/hospital-readmission-phase5-rf-fairness`
  - Logistic Regression: `auphong2707/hospital-readmission-phase5-lr-fairness`

## Outputs

### Per Method Output Structure
```
outputs/{method}/final_evaluation/
├── final_system_metrics.json       # Single source of truth for Phase 7
├── deployment_report.json          # Stakeholder-friendly summary
└── visualizations/                 # 9 comprehensive plots
    ├── confusion_matrix.png
    ├── calibration_curve.png
    ├── group_tpr_comparison.png
    ├── group_fpr_comparison.png
    ├── group_precision_comparison.png
    ├── fairness_disparities.png
    ├── roi_breakdown.png
    ├── risk_distribution.png
    └── threshold_configuration.png
```

### Key Output Files

#### 1. `final_system_metrics.json`
**Purpose**: Single source of truth for Phase 7 publication

**Contents**:
- Model name and deployment configuration
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC, etc.)
- Calibration metrics (Brier score, ECE)
- Group-specific performance (by race, gender, age)
- Fairness metrics (TPR/FPR disparities, demographic parity)
- ROI metrics (cost savings, financial impact)
- Risk stratification analysis
- Executive summary

**Used by**: Phase 7 for aggregated results and publication

#### 2. `deployment_report.json`
**Purpose**: Simplified report for clinical stakeholders

**Contents**:
- Deployment readiness assessment (YES/NO)
- Key performance indicators
- Fairness assessment summary
- Financial impact summary
- Risk stratification summary
- Actionable recommendations

**Used by**: Clinical leadership for deployment decisions

#### 3. Visualizations
Nine comprehensive plots covering all aspects of system performance:
1. **Confusion Matrix**: Overall classification performance with percentages
2. **Calibration Curve**: Reliability diagram showing probability calibration
3. **Group TPR Comparison**: True Positive Rate across demographic groups
4. **Group FPR Comparison**: False Positive Rate across demographic groups
5. **Group Precision Comparison**: Precision across demographic groups
6. **Fairness Disparities**: TPR, FPR, and demographic parity disparities
7. **ROI Breakdown**: Cost comparison and savings visualization
8. **Risk Distribution**: Patient distribution and outcomes by risk category
9. **Threshold Configuration**: Visualization of deployed thresholds

## Evaluation Metrics

### Performance Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **Specificity**: True negative rate
- **PPV/NPV**: Positive/Negative predictive values

### Calibration Metrics
- **Brier Score**: Mean squared error of probability predictions
- **Expected Calibration Error (ECE)**: Average calibration deviation
- **Reliability Diagram**: Visual assessment of calibration

### Fairness Metrics
- **TPR Disparity**: Max difference in True Positive Rate across groups
- **FPR Disparity**: Max difference in False Positive Rate across groups
- **Demographic Parity Disparity**: Max difference in positive prediction rate
- **TPR Ratio**: Ratio of min to max TPR (closer to 1.0 is better)
- **FPR Ratio**: Ratio of min to max FPR (closer to 1.0 is better)

### ROI Metrics
- **Total Cost**: Cost of system with model
- **Baseline Cost**: Cost without model (predict all negative)
- **Cost Savings**: Baseline cost - Total cost
- **ROI Percentage**: (Savings / Baseline) × 100%
- **Avg Cost per Patient**: Total cost / Number of patients
- **Intervention Costs**: Cost of preventive interventions
- **Missed Intervention Costs**: Cost of missed readmissions

### Risk Stratification
- **Patient Distribution**: Count and percentage in each risk category
- **Readmission Rates**: Actual readmission rate per risk category
- **Mean Predicted Probability**: Average predicted probability per category
- **Calibration by Risk**: How well predictions align with outcomes in each category

## Usage

### Method 1: Run All Methods (Recommended)
```bash
# From project root
cd phase-6-final-system-evaluation
./run_final_evaluation.sh
```

### Method 2: Run Individual Methods
```bash
# Gradient Boosting only
./run_final_evaluation.sh --skip-random-forest --skip-logistic-regression

# Random Forest only
./run_final_evaluation.sh --skip-gradient-boosting --skip-logistic-regression

# Logistic Regression only
./run_final_evaluation.sh --skip-gradient-boosting --skip-random-forest
```

### Method 3: Custom Cost Parameters
```bash
# Run with different cost assumptions
./run_final_evaluation.sh \
    --readmission-cost 20000 \
    --intervention-cost 1000
```

### Method 4: Direct Python Execution
```bash
# Run individual evaluation script directly
python final_evaluation_gradient_boosting.py

python final_evaluation_random_forest.py

python final_evaluation_logistic_regression.py
```

## Cost Parameters

The evaluation uses the same cost matrix as Phase 4:

| Outcome | Description | Cost |
|---------|-------------|------|
| **TP** | Prevented readmission (successful intervention) | +$14,500 |
| **TN** | Correct negative prediction (no action needed) | $0 |
| **FP** | Unnecessary intervention | -$500 |
| **FN** | Missed readmission (no intervention) | -$15,000 |

**Assumptions**:
- Average readmission cost: $15,000
- Preventive intervention cost: $500
- Net benefit of preventing readmission: $14,500

You can customize these with command-line arguments:
```bash
./run_final_evaluation.sh --readmission-cost 20000 --intervention-cost 1000
```

## Deployment Readiness Criteria

The system assesses deployment readiness based on:

### Performance Criteria
- ✅ Accuracy ≥ 60%
- ✅ ROC-AUC ≥ 0.65

### Fairness Criteria
- ✅ TPR Disparity < 0.15 (across all demographic features)
- ✅ FPR Disparity < 0.15 (across all demographic features)

### Financial Criteria
- ✅ ROI > 0% (positive return on investment)

**Deployment Recommendation**:
- **READY**: All criteria met
- **NEEDS ATTENTION**: One or more criteria not met

## Interpretation Guide

### Understanding Performance Metrics

**High Accuracy + High ROC-AUC** (> 0.70):
- ✅ Model performs well overall
- ✅ Good discrimination between readmission and non-readmission

**High Sensitivity (> 0.70)**:
- ✅ Successfully identifies most patients at risk
- ✅ Fewer missed readmissions (FN)

**High Specificity (> 0.70)**:
- ✅ Successfully identifies low-risk patients
- ✅ Fewer unnecessary interventions (FP)

**Trade-off**: Typically cannot maximize both sensitivity and specificity simultaneously

### Understanding Calibration

**Low Brier Score** (< 0.15):
- ✅ Probabilities are well-calibrated
- ✅ Predicted probabilities match actual outcomes

**Low ECE** (< 0.10):
- ✅ Minimal calibration error
- ✅ Can trust predicted probabilities for risk assessment

**Reliability Diagram close to diagonal**:
- ✅ Perfect calibration would lie on diagonal
- ✅ Points above diagonal: Under-prediction
- ✅ Points below diagonal: Over-prediction

### Understanding Fairness Metrics

**Low TPR Disparity** (< 0.10):
- ✅ Model identifies high-risk patients equally well across groups
- ✅ No group systematically denied preventive care

**Low FPR Disparity** (< 0.10):
- ✅ Model misclassifies low-risk patients at similar rates
- ✅ No group systematically over-intervened

**Demographic Parity Disparity**:
- Not always desirable to achieve (groups may have different base rates)
- More important: Equal TPR and FPR across groups

### Understanding ROI

**Positive ROI**:
- ✅ Model saves money compared to no model
- ✅ Business case for deployment

**ROI > 10%**:
- ✅ Strong financial benefit
- ✅ Clear value proposition

**Negative ROI**:
- ❌ Model costs more than baseline
- ❌ May need threshold adjustment or model improvement

### Understanding Risk Stratification

**Good Risk Stratification**:
- Low risk: < 20% readmission rate
- Medium risk: 20-50% readmission rate
- High risk: > 50% readmission rate

**Operational Considerations**:
- High risk group size: Should be manageable (< 30% of patients)
- Medium risk group: May need triage protocols
- Low risk group: Can safely receive minimal intervention

## Dependencies

All dependencies are included in the project's `requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
huggingface_hub
```

Install with:
```bash
pip install -r requirements.txt
```

## HuggingFace Authentication

This phase downloads data from HuggingFace. Ensure you're authenticated:

```bash
# One-time setup
huggingface-cli login

# Or set token as environment variable
export HUGGINGFACE_TOKEN=your_token_here
```

## Troubleshooting

### Issue: "Deployment config not found"
**Solution**: Ensure Phase 5 has been completed for the method you're evaluating. The `deployment_config.json` file should exist in the Phase 5 fairness repository.

### Issue: "Mismatch between labels and predictions"
**Solution**: This indicates data alignment issues. Verify:
- Phase 1 test data hasn't changed
- Phase 3 predictions are from the same test set
- No data preprocessing differences

### Issue: "All thresholds are 0.5"
**Solution**: This means Phase 5 didn't find fairness violations and kept global thresholds. This is expected if fairness metrics were acceptable. Check `is_mitigated: false` in `deployment_config.json`.

### Issue: "ROI is negative"
**Solution**: 
- Review threshold configuration
- Consider adjusting cost parameters if business assumptions changed
- May need model retraining if performance degraded

### Issue: "Large fairness disparities"
**Solution**:
- Check if Phase 5 mitigation was applied (`use_group_thresholds: true`)
- Review group-specific thresholds in `threshold_configuration`
- May need to re-run Phase 5 with different mitigation parameters

## Next Steps

After completing Phase 6:

1. **Review Outputs**:
   - Examine `final_system_metrics.json` for comprehensive results
   - Check `deployment_report.json` for deployment readiness
   - Review all visualizations

2. **Stakeholder Review**:
   - Share `deployment_report.json` with clinical leadership
   - Present visualizations in deployment meetings
   - Discuss any fairness or performance concerns

3. **Deployment Decision**:
   - If deployment-ready: Proceed to production planning
   - If not ready: Iterate on model or fairness mitigation

4. **Phase 7 - Results Collection & Publication**:
   - Phase 7 will use `final_system_metrics.json` as source of truth
   - Aggregated results will be published for research dissemination
   - Run Phase 7 only after completing Phase 6 for all methods

## Architecture

### Utility Classes (`utilities.py`)

1. **DeploymentConfigLoader**: Load and validate deployment configuration from Phase 5
2. **ThresholdApplicator**: Apply global or group-specific thresholds to predictions
3. **FinalMetricsCalculator**: Calculate comprehensive performance, calibration, and fairness metrics
4. **ROICalculator**: Calculate financial impact and ROI metrics
5. **RiskCategoryAnalyzer**: Analyze risk stratification performance
6. **FinalEvaluationVisualizer**: Generate all 9 visualizations
7. **DeploymentReportGenerator**: Create JSON reports (metrics + deployment summary)

### Evaluation Scripts

- `final_evaluation_gradient_boosting.py`: Gradient Boosting (LightGBM) evaluation
- `final_evaluation_random_forest.py`: Random Forest evaluation
- `final_evaluation_logistic_regression.py`: Logistic Regression evaluation

Each script follows the same pipeline:
1. Load deployment config from Phase 5
2. Load test data and predictions
3. Apply deployed thresholds
4. Calculate all metrics
5. Generate visualizations
6. Create reports

### Orchestrator

`run_final_evaluation.sh`: Master script that runs all three evaluation scripts with consistent parameters and provides unified progress reporting.

## Files in This Phase

```
phase-6-final-system-evaluation/
├── README.md                                  # This file
├── utilities.py                               # Utility classes (1,000+ lines)
├── final_evaluation_gradient_boosting.py      # GB evaluation script
├── final_evaluation_random_forest.py          # RF evaluation script
├── final_evaluation_logistic_regression.py    # LR evaluation script
└── run_final_evaluation.sh                    # Orchestrator (executable)
```

## References

- **Phase 5**: Fairness Assessment & Mitigation (provides deployment_config.json)
- **Phase 4**: Optimal Threshold & ROI Analysis (provides cost matrix)
- **Phase 3**: Model Calibration (provides calibrated predictions)
- **Phase 2**: Risk Modeling (provides trained models)
- **Phase 1**: Data Preprocessing (provides test data)

## Contact & Support

For questions or issues:
1. Check this README troubleshooting section
2. Review Phase 5 outputs to ensure deployment configs exist
3. Verify HuggingFace authentication and repository access
4. Check error messages for specific guidance

---

**Phase 6** is the culmination of all previous modeling work, providing the definitive evaluation of the system that will be deployed to production. The outputs from this phase are what stakeholders will use to make go/no-go deployment decisions.
