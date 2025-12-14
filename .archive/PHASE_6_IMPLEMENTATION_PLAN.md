# Phase 6: Final System Evaluation - Implementation Plan

## Executive Summary

Phase 6 is the **final comprehensive evaluation** of the deployed system using the **actual threshold configuration** (global from Phase 4 OR group-specific from Phase 5) that will be used in production. This phase aggregates all previous metrics and provides the definitive performance, fairness, and ROI assessment for clinical approval and deployment.

### Critical Difference from Previous Phases

**Why Phase 6 is needed:**
- **Phase 2**: Model selection with default 0.5 threshold
- **Phase 3**: Calibration quality assessment
- **Phase 4**: Optimal threshold identification (but no fairness consideration)
- **Phase 5**: Fairness evaluation + optional mitigation (but with separate evaluation and mitigation paths)
- **Phase 6**: **Final evaluation with DEPLOYED configuration** (combines everything)

**What's unique:**
- Uses **deployment_config.json** from Phase 5 to determine which thresholds to apply
- Evaluates system **exactly as it will run in production**
- Provides **single source of truth** for all final metrics
- Generates **deployment report** for clinical stakeholders
- This is what **Phase 7 publishes** as the final system performance

---

## Purpose & Goals

### Primary Purpose
Perform comprehensive final evaluation of the hospital readmission risk prediction system using the **deployed threshold configuration** determined in previous phases.

### Goals
1. **Load deployment configuration** from Phase 5 (global or group-specific thresholds)
2. **Apply thresholds** exactly as they will be used in production
3. **Calculate final metrics** across all dimensions:
   - Overall performance (accuracy, precision, recall, F1, ROC-AUC, Brier)
   - Group-specific fairness (TPR, FPR, intervention rate by demographics)
   - ROI and business impact (expected value, cost savings, intervention volume)
   - Risk category distribution (low/medium/high risk assignments)
4. **Generate comprehensive visualizations** showing final system behavior
5. **Create deployment report** for clinical approval
6. **Provide Phase 7 input** - single file with all final metrics

---

## Input Requirements

### From Phase 5 (Critical)
- **`deployment_config.json`** - Standardized configuration file:
  ```json
  {
    "method": "gradient_boosting",
    "is_mitigated": true/false,
    "use_group_thresholds": true/false,
    "threshold_configuration": {
      "type": "global" | "group_specific",
      "source": "phase4_roi_optimization" | "phase5_mitigation",
      "group_thresholds": {...} | null
    },
    "fairness_status": {...},
    "phase6_instructions": {...}
  }
  ```

### From Phase 4
- **Global threshold** (if `use_group_thresholds: false`)
- **Risk category boundaries** (low/medium/high thresholds)
- **Cost parameters** (intervention cost, readmission cost)

### From Phase 3
- **Calibrated model** and calibrator

### From Phase 1
- **Test data** (X_test, y_test)
- **Demographics** (test_demographics.csv)

---

## Output Specification

### Directory Structure
```
phase-6-final-system-evaluation/outputs/{method}/
├── final_system_metrics.json          # PRIMARY OUTPUT - All metrics
├── deployment_report.json             # Clinical stakeholder summary
├── deployment_report.pdf              # Human-readable report
├── performance_breakdown.csv          # Detailed metrics table
└── visualizations/
    ├── 01_confusion_matrix_deployed.png
    ├── 02_roc_curve_with_threshold.png
    ├── 03_precision_recall_curve.png
    ├── 04_calibration_curve_deployed.png
    ├── 05_group_fairness_comparison.png
    ├── 06_intervention_rate_by_group.png
    ├── 07_tpr_fpr_by_group.png
    ├── 08_risk_distribution_overall.png
    ├── 09_risk_distribution_by_group.png
    ├── 10_roi_summary.png
    ├── 11_cost_benefit_breakdown.png
    └── 12_threshold_visualization.png
```

### Primary Output: `final_system_metrics.json`

**This is the SINGLE SOURCE OF TRUTH for Phase 7**

```json
{
  "metadata": {
    "phase": "Phase 6: Final System Evaluation",
    "method": "gradient_boosting",
    "evaluation_date": "2025-12-14T10:30:00Z",
    "model_version": "v1.0.0",
    "deployment_configuration": {
      "threshold_type": "group_specific" | "global",
      "is_mitigated": true/false,
      "source": "phase5_mitigation" | "phase4_roi_optimization"
    }
  },
  
  "deployment_thresholds": {
    "global_threshold": 0.485 | null,
    "risk_category_thresholds": {
      "low_to_medium": 0.324,
      "medium_to_high": 0.728
    },
    "group_specific_thresholds": {
      "race": {...},
      "gender": {...},
      "age": {...}
    } | null
  },
  
  "overall_performance": {
    "test_set_size": 15265,
    "confusion_matrix": {
      "TP": 2450,
      "FP": 1820,
      "TN": 9975,
      "FN": 1020
    },
    "classification_metrics": {
      "accuracy": 0.8140,
      "precision": 0.5738,
      "recall": 0.7061,
      "f1_score": 0.6330,
      "specificity": 0.8457
    },
    "probability_metrics": {
      "roc_auc": 0.8462,
      "pr_auc": 0.6891,
      "brier_score": 0.1245,
      "log_loss": 0.3892
    },
    "rates": {
      "tpr": 0.7061,
      "fpr": 0.1543,
      "intervention_rate": 0.2797,
      "positive_rate": 0.2272
    }
  },
  
  "group_fairness_metrics": {
    "race": {
      "groups": {
        "AfricanAmerican": {
          "n_samples": 2450,
          "tpr": 0.7250,
          "fpr": 0.1650,
          "precision": 0.5820,
          "recall": 0.7250,
          "intervention_rate": 0.2950,
          "actual_positive_rate": 0.2405
        },
        "Caucasian": {...},
        "Hispanic": {...},
        "Asian": {...},
        "Other": {...}
      },
      "fairness_gaps": {
        "tpr_gap": 0.0450,
        "tpr_max": 0.7250,
        "tpr_min": 0.6800,
        "fpr_gap": 0.0380,
        "fpr_max": 0.1720,
        "fpr_min": 0.1340,
        "intervention_rate_gap": 0.0520,
        "max_intervention_rate": 0.3050,
        "min_intervention_rate": 0.2530
      },
      "statistical_tests": {
        "chi_square_intervention": {
          "statistic": 12.45,
          "p_value": 0.0145,
          "significant": true
        },
        "tpr_homogeneity": {
          "test": "proportion_test",
          "p_value": 0.0823,
          "significant": false
        }
      }
    },
    "gender": {...},
    "age": {...}
  },
  
  "risk_category_analysis": {
    "overall_distribution": {
      "low_risk": {
        "count": 10180,
        "percentage": 66.68,
        "actual_readmission_rate": 0.0834
      },
      "medium_risk": {
        "count": 3565,
        "percentage": 23.35,
        "actual_readmission_rate": 0.2875
      },
      "high_risk": {
        "count": 1520,
        "percentage": 9.96,
        "actual_readmission_rate": 0.5789
      }
    },
    "by_demographic": {
      "race": {
        "AfricanAmerican": {
          "low_risk_pct": 64.50,
          "medium_risk_pct": 24.80,
          "high_risk_pct": 10.70
        },
        "Caucasian": {...}
      },
      "gender": {...},
      "age": {...}
    },
    "calibration_by_category": {
      "low_risk": {
        "predicted_avg": 0.15,
        "actual_rate": 0.0834,
        "calibrated": true
      },
      "medium_risk": {...},
      "high_risk": {...}
    }
  },
  
  "roi_analysis": {
    "cost_parameters": {
      "readmission_cost": 15000,
      "intervention_cost": 500,
      "tp_benefit": 14500,
      "fp_cost": -500,
      "fn_cost": -15000,
      "tn_cost": 0
    },
    "financial_impact": {
      "total_interventions": 4270,
      "intervention_cost_total": -2135000,
      "readmissions_prevented": 2450,
      "readmission_savings": 36750000,
      "missed_readmissions": 1020,
      "missed_readmission_cost": -15300000,
      "net_expected_value": 19315000,
      "expected_value_per_patient": 1265.15
    },
    "roi_metrics": {
      "roi_percentage": 804.91,
      "break_even_threshold": 0.0333,
      "cost_benefit_ratio": 9.0491
    },
    "intervention_efficiency": {
      "precision": 0.5738,
      "nnt": 1.74,
      "interventions_per_prevented_readmission": 1.74
    }
  },
  
  "fairness_roi_tradeoff": {
    "baseline_global": {
      "expected_value": 18500000,
      "tpr_gap": 0.0780,
      "fpr_gap": 0.0650
    },
    "deployed_system": {
      "expected_value": 19315000,
      "tpr_gap": 0.0450,
      "fpr_gap": 0.0380
    },
    "improvement": {
      "fairness_improved": true,
      "ev_change": 815000,
      "ev_change_pct": 4.41,
      "tpr_gap_reduction": 0.0330,
      "fpr_gap_reduction": 0.0270
    }
  },
  
  "deployment_readiness": {
    "performance_acceptable": true,
    "fairness_acceptable": true,
    "roi_acceptable": true,
    "recommended_for_deployment": true,
    "clinical_approval_required": true,
    "monitoring_required": true,
    "notes": [
      "Model meets all performance criteria",
      "Fairness gaps reduced below 5% threshold",
      "Positive ROI with group-specific thresholds",
      "Requires ongoing monitoring of group metrics"
    ]
  }
}
```

### Secondary Output: `deployment_report.json`

**Simplified summary for clinical stakeholders**

```json
{
  "executive_summary": {
    "model_name": "Hospital Readmission Risk Prediction",
    "method": "Gradient Boosting (LightGBM)",
    "evaluation_date": "2025-12-14",
    "recommendation": "APPROVED FOR DEPLOYMENT",
    "key_findings": [
      "Model achieves 81.4% accuracy with ROC-AUC of 0.846",
      "Identifies 70.6% of patients at risk for readmission",
      "Fairness criteria met: TPR gap 4.5%, FPR gap 3.8%",
      "Expected net benefit: $19.3M annually",
      "Intervention cost: $2.1M with $36.8M in prevented readmissions"
    ]
  },
  
  "performance_summary": {
    "accuracy": "81.4%",
    "sensitivity_recall": "70.6%",
    "precision": "57.4%",
    "roc_auc": "84.6%",
    "intervention_rate": "28.0%"
  },
  
  "fairness_summary": {
    "fairness_strategy": "Group-specific thresholds applied",
    "tpr_gap": "4.5% (below 5% target)",
    "fpr_gap": "3.8% (below 5% target)",
    "groups_evaluated": ["race", "gender", "age"],
    "bias_mitigation_applied": true
  },
  
  "business_impact": {
    "patients_screened": "15,265 (test set)",
    "high_risk_interventions": "4,270 patients",
    "readmissions_prevented": "2,450 cases",
    "cost_savings": "$19.3M net benefit",
    "roi": "805%"
  },
  
  "risk_stratification": {
    "low_risk": "66.7% of patients, 8.3% readmission rate",
    "medium_risk": "23.4% of patients, 28.8% readmission rate",
    "high_risk": "10.0% of patients, 57.9% readmission rate"
  },
  
  "deployment_requirements": {
    "infrastructure": "Real-time scoring API required",
    "monitoring": "Monthly fairness audits across demographics",
    "clinical_workflow": "Integrate with discharge planning system",
    "intervention_capacity": "4,270 patients per 15,265 encounters (28%)",
    "staff_training": "Risk score interpretation and intervention protocols"
  },
  
  "limitations_and_risks": {
    "model_limitations": [
      "Performance may vary with population drift",
      "Small sample sizes in some demographic subgroups",
      "Relies on complete EHR data at discharge"
    ],
    "operational_risks": [
      "Requires clinical decision support integration",
      "Staff training and change management needed",
      "Monitoring system must be in place before deployment"
    ],
    "ethical_considerations": [
      "Ongoing fairness monitoring required",
      "Group-specific thresholds must be reviewed quarterly",
      "Patient consent and transparency protocols needed"
    ]
  },
  
  "approval_checklist": [
    {"item": "Model performance meets minimum criteria", "status": "✓"},
    {"item": "Fairness evaluation completed", "status": "✓"},
    {"item": "ROI analysis positive", "status": "✓"},
    {"item": "Clinical workflow integration planned", "status": "○"},
    {"item": "Monitoring dashboard deployed", "status": "○"},
    {"item": "Staff training completed", "status": "○"},
    {"item": "Patient notification process established", "status": "○"},
    {"item": "Ethics committee approval", "status": "○"}
  ]
}
```

---

## Visualization Requirements

### 1. Confusion Matrix with Deployed Configuration
- **Purpose**: Show final TP/FP/TN/FN with deployed thresholds
- **Annotations**: Financial impact per quadrant
- **Labels**: Clearly state threshold configuration used

### 2. ROC Curve with Operating Point
- **Purpose**: Show ROC-AUC with final threshold marked
- **Elements**: 
  - Full ROC curve
  - Operating point (actual threshold used)
  - If group-specific: show multiple operating points
  - Comparison to random baseline

### 3. Precision-Recall Curve
- **Purpose**: Demonstrate tradeoff at deployed threshold
- **Elements**:
  - PR curve
  - Operating point marked
  - F1 score contours
  - Baseline (prevalence line)

### 4. Calibration Curve (Deployed System)
- **Purpose**: Show probability reliability in production
- **Elements**:
  - Predicted vs observed probabilities
  - Perfect calibration line
  - Brier score annotation
  - ECE (Expected Calibration Error)

### 5. Group Fairness Comparison
- **Purpose**: Show TPR, FPR, Precision across demographics
- **Format**: Grouped bar chart
- **Groups**: Race, Gender, Age categories
- **Metrics**: TPR, FPR, Precision, Intervention Rate
- **Annotations**: Fairness gaps, target thresholds

### 6. Intervention Rate by Group
- **Purpose**: Show fairness in intervention distribution
- **Format**: Stacked or grouped bar chart
- **Dimensions**: Intervention vs no-intervention by demographic
- **Statistical**: Chi-square test results

### 7. TPR/FPR by Group Detailed
- **Purpose**: Detailed view of equalized odds metrics
- **Format**: Side-by-side comparison
- **Elements**: 
  - TPR bars with confidence intervals
  - FPR bars with confidence intervals
  - Target fairness bands (±5%)

### 8. Risk Distribution (Overall)
- **Purpose**: Show low/medium/high risk assignments
- **Format**: Pie or bar chart
- **Annotations**: Actual readmission rates per category

### 9. Risk Distribution by Demographics
- **Purpose**: Show risk category fairness across groups
- **Format**: Stacked bar charts per demographic
- **Validation**: Check for disparate impact in risk assignment

### 10. ROI Summary
- **Purpose**: Financial business case visualization
- **Format**: Waterfall or bar chart
- **Elements**:
  - Intervention costs (negative)
  - Readmission savings (positive)
  - Missed readmissions (negative)
  - Net benefit (total)

### 11. Cost-Benefit Breakdown
- **Purpose**: Detailed financial analysis
- **Format**: Breakdown chart
- **Elements**:
  - Cost per TP, FP, FN, TN
  - Total costs/benefits
  - Break-even analysis

### 12. Threshold Visualization
- **Purpose**: Show how thresholds differ (if group-specific)
- **Format**: Bar chart or heatmap
- **Elements**:
  - Global threshold baseline
  - Group-specific adjustments
  - +/- difference from global

---

## Implementation Structure

### Files to Create

```
phase-6-final-system-evaluation/
├── final_evaluation_gradient_boosting.py     # Main script
├── final_evaluation_random_forest.py         # Main script
├── final_evaluation_logistic_regression.py   # Main script
├── utilities.py                              # Helper classes
├── README.md                                 # Documentation
└── outputs/                                  # Generated outputs
    ├── gradient_boosting/
    │   ├── final_system_metrics.json
    │   ├── deployment_report.json
    │   ├── deployment_report.pdf
    │   └── visualizations/
    ├── random_forest/
    └── logistic_regression/
```

### Utility Classes Needed

```python
# utilities.py

class DeploymentConfigLoader:
    """Load and validate deployment configuration from Phase 5"""
    def load_config(path: str) -> Dict
    def validate_config(config: Dict) -> bool
    def get_thresholds(config: Dict) -> Dict

class ThresholdApplicator:
    """Apply thresholds (global or group-specific)"""
    def apply_global_threshold(y_proba, threshold) -> np.ndarray
    def apply_group_thresholds(y_proba, demographics, thresholds) -> np.ndarray

class FinalMetricsCalculator:
    """Calculate all final system metrics"""
    def calculate_confusion_matrix(y_true, y_pred) -> Dict
    def calculate_classification_metrics(y_true, y_pred) -> Dict
    def calculate_probability_metrics(y_true, y_proba) -> Dict
    def calculate_group_metrics(y_true, y_pred, demographics) -> Dict
    def calculate_fairness_gaps(group_metrics) -> Dict

class ROICalculator:
    """Calculate financial impact"""
    def calculate_financial_impact(confusion_matrix, costs) -> Dict
    def calculate_roi_metrics(financial_impact) -> Dict
    def calculate_intervention_efficiency(confusion_matrix) -> Dict

class RiskCategoryAnalyzer:
    """Analyze risk category distributions"""
    def assign_risk_categories(y_proba, thresholds) -> np.ndarray
    def analyze_overall_distribution(y_true, risk_categories) -> Dict
    def analyze_demographic_distribution(risk_categories, demographics) -> Dict
    def validate_calibration_by_category(y_true, y_proba, risk_categories) -> Dict

class FinalEvaluationVisualizer:
    """Generate all Phase 6 visualizations"""
    def plot_confusion_matrix(...)
    def plot_roc_curve_with_threshold(...)
    def plot_precision_recall_curve(...)
    def plot_calibration_curve(...)
    def plot_group_fairness_comparison(...)
    def plot_intervention_rate_by_group(...)
    def plot_tpr_fpr_by_group(...)
    def plot_risk_distribution_overall(...)
    def plot_risk_distribution_by_group(...)
    def plot_roi_summary(...)
    def plot_cost_benefit_breakdown(...)
    def plot_threshold_visualization(...)
    def generate_all_visualizations(...)

class DeploymentReportGenerator:
    """Generate deployment report (JSON and PDF)"""
    def create_executive_summary(...) -> Dict
    def create_deployment_report_json(...) -> Dict
    def create_deployment_report_pdf(...) -> None  # Uses reportlab
```

---

## Success Criteria

### Performance
- ✅ ROC-AUC ≥ 0.80
- ✅ Recall (TPR) ≥ 0.65
- ✅ Precision ≥ 0.50
- ✅ Brier score ≤ 0.15

### Fairness
- ✅ TPR gap ≤ 5% across all demographics
- ✅ FPR gap ≤ 5% across all demographics
- ✅ Intervention rate gap ≤ 8% across demographics
- ✅ No statistically significant disparities (p < 0.05)

### Business Impact
- ✅ Positive ROI (net benefit > 0)
- ✅ Expected value per patient > $1,000
- ✅ Intervention efficiency (NNT < 3)

### Deployment Readiness
- ✅ All metrics calculated and documented
- ✅ Deployment report approved by clinical team
- ✅ Monitoring plan in place
- ✅ Integration with clinical workflow defined

---

## Execution Plan

### Step 1: Create utilities.py (Helper Classes)
**Priority**: HIGH  
**Dependencies**: None  
**Estimated Time**: 4-6 hours

### Step 2: Create final_evaluation_gradient_boosting.py
**Priority**: HIGH  
**Dependencies**: utilities.py  
**Estimated Time**: 3-4 hours

### Step 3: Test with Gradient Boosting
**Priority**: HIGH  
**Dependencies**: Steps 1-2  
**Estimated Time**: 1-2 hours

### Step 4: Create final_evaluation_random_forest.py
**Priority**: MEDIUM  
**Dependencies**: utilities.py  
**Estimated Time**: 1 hour (copy and modify)

### Step 5: Create final_evaluation_logistic_regression.py
**Priority**: MEDIUM  
**Dependencies**: utilities.py  
**Estimated Time**: 1 hour (copy and modify)

### Step 6: Create README.md
**Priority**: MEDIUM  
**Dependencies**: Steps 1-5  
**Estimated Time**: 1 hour

### Step 7: Update Phase 7 to use Phase 6 outputs
**Priority**: HIGH  
**Dependencies**: Phase 6 complete  
**Estimated Time**: 2 hours

---

## Key Design Decisions

### 1. Single vs Multiple Evaluations
**Decision**: Single evaluation with deployment config  
**Rationale**: Phase 6 evaluates the FINAL system, not alternatives

### 2. Threshold Application Logic
**Decision**: Use Phase 5 deployment_config.json as single source of truth  
**Rationale**: Ensures Phase 6 matches production exactly

### 3. Metrics Aggregation
**Decision**: Comprehensive final_system_metrics.json with ALL metrics  
**Rationale**: Phase 7 needs single file, stakeholders need complete view

### 4. Visualization Strategy
**Decision**: 12 comprehensive visualizations covering all aspects  
**Rationale**: Clinical approval requires visual evidence

### 5. Report Generation
**Decision**: Both JSON (machine) and PDF (human) formats  
**Rationale**: Serves both technical and clinical audiences

---

## Integration with Phase 7

Phase 7 will:
1. Read `final_system_metrics.json` from Phase 6
2. Extract metrics for aggregated_results.json
3. Copy Phase 6 visualizations to publication
4. Include deployment_report.pdf in final package
5. Reference Phase 6 as THE authoritative evaluation

**Phase 7 should NOT**:
- Use Phase 2 model metrics (outdated, no threshold)
- Use Phase 4 metrics (no fairness)
- Use Phase 5 metrics (evaluation vs mitigation split)

**Phase 7 should ONLY use Phase 6** as the single source of truth.

---

## Timeline Estimate

**Total**: ~15-20 hours for complete implementation

- **Day 1** (6-8 hours): utilities.py + gradient_boosting script
- **Day 2** (4-6 hours): Testing, debugging, RF/LR scripts
- **Day 3** (3-4 hours): README, Phase 7 updates, validation
- **Day 4** (2 hours): Final testing and documentation

---

## Next Steps

1. ✅ Review and approve this plan
2. 🔨 Implement utilities.py (core functionality)
3. 🔨 Create gradient_boosting evaluation script
4. 🧪 Test with actual data
5. 🔁 Create RF and LR scripts
6. 📝 Write README
7. 🔗 Update Phase 7 collection script
8. ✅ End-to-end testing

