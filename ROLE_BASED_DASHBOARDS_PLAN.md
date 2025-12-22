# Role-Based Dashboards Plan
## Hospital Readmission Risk Prediction System

**Created**: December 18, 2025  
**Purpose**: Provide tailored views and insights for three distinct stakeholder roles

---

## Executive Summary

This plan creates three specialized dashboards that leverage existing pipeline outputs without modifying phase code. Each dashboard aggregates relevant data from Phases 1-6 to address specific stakeholder needs.

### Three Stakeholder Roles

| Role | Primary Focus | Key Questions |
|------|---------------|---------------|
| **Data Analyst** | Data quality, model performance, technical metrics | "Is the model working correctly? What patterns exist?" |
| **Doctor/Clinician** | Clinical applicability, patient impact, actionable insights | "How does this affect my patients? Can I trust it?" |
| **Manager/Executive** | ROI, costs, resource allocation, business impact | "What's the financial return? How many resources needed?" |

---

## Architecture Overview

```
phase-7-stakeholder-dashboards/
├── README.md                           # Overview and usage instructions
├── docker-compose.yml                  # Grafana + data source containers
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── datasource.yml         # Configure JSON API data source
│   │   └── dashboards/
│   │       ├── dashboard.yml          # Dashboard provisioning config
│   │       ├── data_analyst.json      # Data Analyst dashboard definition
│   │       ├── clinician.json         # Clinician dashboard definition
│   │       └── executive.json         # Executive dashboard definition
│   └── plugins/
│       └── custom-plugins/            # Custom Grafana plugins if needed
├── data-api/
│   ├── requirements.txt               # FastAPI, pandas, huggingface-hub
│   ├── main.py                        # FastAPI app serving JSON endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Data models for API responses
│   │   └── aggregations.py           # Aggregation logic
│   └── routers/
│       ├── __init__.py
│       ├── phase1.py                  # Phase 1 data endpoints
│       ├── phase2.py                  # Phase 2 model metrics endpoints
│       ├── phase3.py                  # Phase 3 calibration endpoints
│       ├── phase4.py                  # Phase 4 ROI endpoints
│       ├── phase5.py                  # Phase 5 fairness endpoints
│       └── phase6.py                  # Phase 6 final evaluation endpoints
├── utilities/
│   ├── __init__.py
│   ├── data_aggregator.py             # Download from HuggingFace
│   ├── metrics_calculator.py          # Calculate dashboard-specific metrics
│   └── cache_manager.py               # Redis/file-based caching
└── scenarios/
    ├── feature_examples.json          # Example patient profiles
    ├── budget_scenarios.json          # Budget constraint scenarios
    └── capacity_scenarios.json        # Intervention capacity scenarios
```

---

## Dashboard 1: Data Analyst Dashboard

### Target Audience
Data scientists, ML engineers, statisticians, technical reviewers

### Primary Goals
- **Understand the complete ML pipeline** from data to deployment
- **Compare model performance** across 3 models (RF, GB, LR)
- **Validate each phase** of the system (data, training, calibration, optimization, fairness, deployment)
- Keep it technical but organized by workflow stages

### Key Features

**Analysis Flow**: Pipeline-Based (Phase 1 → Phase 6)

#### 1. **Quick Insights Panel** (Best Model Summary Card)
**Purpose:** Immediate answer to "Which model should we deploy?"

**Summary Card:**
```
🏆 RECOMMENDED MODEL: Gradient Boosting

Performance:
  • ROC-AUC: 0.842 (highest)
  • Brier Score: 0.18 (best calibration)
  • F1 Score: 0.72

Business Impact:
  • ROI: 325%
  • Annual Net Savings: $2,450,000
  • Readmissions Prevented: 850

Fairness:
  • Status: ✅ PASS
  • Max Disparity: 4.2% (within threshold)

Deployment Readiness: ✅ READY
```

**Data Sources:**
- Aggregated from Phase 2 (performance), Phase 3 (calibration), Phase 6 (ROI, final metrics)
- Best model determined by composite scoring across all phases

**Code to Reuse:**
- Load Phase 2 metrics for all 3 models
- Load Phase 6 final evaluation for all 3 models
- Calculate composite score to determine best model
- Format as summary card

---

#### 2. **Phase 1: Data Quality & Preprocessing**
**Purpose:** Validate input data quality

**Data Sources:**
- `data/processed/splits/train.csv`, `val.csv`, `test.csv`
- `data/processed/split_info.txt`
- Phase 1 EDA notebooks

**Key Metrics:**
```
Dataset Statistics:
  • Total Patients: 100,000
  • Train: 70,000 (70%)
  • Validation: 15,000 (15%)
  • Test: 15,000 (15%)

Class Distribution:
  • Readmitted: 47,000 (47%)
  • Not Readmitted: 53,000 (53%)
  • Balance Status: ✅ Acceptable

Data Quality:
  • Missing Values: 0 (after preprocessing)
  • Feature Count: 42
  • Outliers Handled: ✅ Yes
  • Encoding Applied: ✅ Complete
```

**Visualizations:**
- Dataset split pie chart
- Class balance chart
- Feature type breakdown (numerical vs categorical)

---

#### 3. **Phase 2: Model Training & Performance**
**Purpose:** Compare base model performance before calibration

**Data Sources:**
- `outputs/{model}/[model]_metrics.json` (all 3 models)
- `outputs/{model}/visualizations/[model]_roc_curve.png`
- `outputs/{model}/visualizations/confusion_matrix.png`

**Metrics Table:**
```
Model              | ROC-AUC | Precision | Recall | F1    | Specificity
-------------------|---------|-----------|--------|-------|------------
Gradient Boosting  |  0.842  |   0.63    |  0.84  | 0.72  |    0.78
Random Forest      |  0.820  |   0.61    |  0.81  | 0.70  |    0.76
Logistic Regression|  0.790  |   0.58    |  0.78  | 0.67  |    0.72

Winner: Gradient Boosting (highest ROC-AUC and F1)
```

**Visualizations:**
- ROC curves overlay (3 models)
- Confusion matrices (3x1 grid)
- Precision-Recall curves

---

#### 4. **Phase 3: Model Calibration**
**Purpose:** Validate probability calibration quality

**Data Sources:**
- `outputs/{model}/calibration/[Model]_metrics.json` (all 3 models)
- `outputs/{model}/calibration/reliability_diagram.png`

**Metrics Table:**
```
Model              | Brier (before) | Brier (after) | Improvement | ECE (after) | Status
-------------------|----------------|---------------|-------------|-------------|--------
Gradient Boosting  |     0.22       |     0.18      |    18.2%    |   0.042     | ✅ Good
Random Forest      |     0.24       |     0.19      |    20.8%    |   0.051     | ✅ Good
Logistic Regression|     0.25       |     0.21      |    16.0%    |   0.063     | ⚠️ Fair

All models: Well-calibrated after Platt scaling
```

**Visualizations:**
- Reliability diagrams (before/after, 3 models)
- Calibration improvement bar chart

---

#### 5. **Phase 4: Threshold Optimization & ROI**
**Purpose:** Validate business value and optimal decision threshold

**Data Sources:**
- `outputs/{model}/threshold_optimization/roi_metrics.json` (all 3 models)
- `outputs/{model}/threshold_optimization/optimal_thresholds.json`

**Metrics Table:**
```
Model              | Optimal Threshold | ROI (%) | Net Savings | Intervention Rate
-------------------|-------------------|---------|-------------|------------------
Gradient Boosting  |       0.35        |  325%   | $2,450,000  |      40%
Random Forest      |       0.38        |  285%   | $2,200,000  |      42%
Logistic Regression|       0.42        |  245%   | $1,950,000  |      45%

Winner: Gradient Boosting (highest ROI and net savings)
```

**Visualizations:**
- ROI comparison bar chart
- Threshold vs Metrics curve (precision, recall, F1 at different thresholds)
- Cost-benefit breakdown

---

#### 6. **Phase 5: Fairness Assessment & Mitigation**
**Purpose:** Ensure equitable performance across demographic groups

**Data Sources:**
- `phase-5-fairness-assessment-mitigation/outputs/{model}/evaluation/group_metrics_*.csv`
- `phase-5-fairness-assessment-mitigation/outputs/{model}/evaluation/statistical_tests.json`
- `phase-5-fairness-assessment-mitigation/outputs/{model}/deployment_config.json`

**Metrics Table:**
```
Fairness Metric              | Gradient Boosting | Random Forest | Logistic Reg
-----------------------------|-------------------|---------------|-------------
Max TPR Disparity (Race)     |       2.3%        |     3.1%      |    4.2%
Max FPR Disparity (Race)     |       1.8%        |     2.5%      |    3.1%
Max TPR Disparity (Gender)   |       0.5%        |     0.8%      |    1.2%
Max TPR Disparity (Age)      |       3.5%        |     4.2%      |    5.1%

Threshold: < 5% disparity = PASS ✅
Status: All models PASS fairness assessment
```

**Visualizations:**
- Disparity heatmap (TPR, FPR by demographic group)
- Max disparity bar chart
- Group-specific thresholds (if mitigation applied)

---

#### 7. **Phase 6: Final System Evaluation**
**Purpose:** Validate complete deployed system performance

**Data Sources:**
- `outputs/{model}/final_evaluation/final_system_metrics.json` (all 3 models)
- `outputs/{model}/final_evaluation/deployment_report.json`

**⚠️ Critical:** Phase 6 contains the **authoritative metrics** for the final deployed system (includes fairness mitigation adjustments)

**Metrics Table:**
```
Model              | Final ROC-AUC | Final Brier | Final ROI | Readm. Prevented | Status
-------------------|---------------|-------------|-----------|------------------|--------
Gradient Boosting  |     0.842     |    0.18     |   325%    |       850        | ✅ Ready
Random Forest      |     0.820     |    0.19     |   285%    |       820        | ✅ Ready
Logistic Regression|     0.790     |    0.21     |   245%    |       780        | ✅ Ready

Recommended: Gradient Boosting (best overall performance)
```

**Visualizations:**
- Final performance comparison (all metrics)
- Deployment readiness checklist
- System integration diagram

---

**Dashboard Flow Summary:**
```
1. Quick Insights          → "Which model to deploy?" (Best Model Card)
2. Phase 1: Data Quality   → "Is the data good?"
3. Phase 2: Model Training → "Which model performs best?"
4. Phase 3: Calibration    → "Can we trust the probabilities?"
5. Phase 4: ROI Analysis   → "What's the business value?"
6. Phase 5: Fairness       → "Does it work fairly for everyone?"
7. Phase 6: Final System   → "Is it ready for deployment?"
```

**Design Philosophy:** 
- Pipeline-based navigation (follows ML workflow chronologically)
- Each panel = one phase of the system
- Easy to trace issues back to specific pipeline stage
- Technical depth appropriate for data scientists
- Emphasizes both technical metrics AND business impact

---

## Dashboard 2: Doctor/Clinician Dashboard

### Target Audience
Physicians, nurses, clinical staff, care coordinators

### Primary Goals
- **Understand which clinical factors drive readmission risk**
- **Review recently discharged patients and their readmission risk**
- Keep it simple and actionable

### Key Features

#### 1. **Clinical Risk Factors Panel** (Phase 1 + Phase 2)
**Data Sources:**
- `data/processed/splits/train.csv` (EDA insights from Phase 1)
- `outputs/{model}/visualizations/[model]_feature_importance.csv` (Phase 2)
- `outputs/{model}/visualizations/[model]_feature_importance.png` (Phase 2)

**Visualizations:**
- **Top 20 Risk Factors** (feature importance bar chart)
- **Feature distributions by outcome** (readmitted vs not readmitted)
  - Number of prior inpatient visits
  - Number of emergency visits
  - Number of diagnoses
  - Time in hospital
  - Number of procedures
  - Number of medications
  - A1C test results
  - Insulin/medication changes
- **Demographic distributions** by readmission status (age, gender, race)

**Clinical Insights Table:**
```
Top Risk Factors for Readmission:

Rank | Feature                      | Importance | Clinical Meaning
-----|------------------------------|------------|----------------------------------
1    | number_inpatient            | 0.145      | Prior hospitalizations (last year)
2    | number_emergency            | 0.098      | ER visits (last year)
3    | number_diagnoses            | 0.087      | Comorbidity burden
4    | time_in_hospital            | 0.076      | Length of stay (days)
5    | num_medications             | 0.065      | Medication complexity
6    | num_procedures              | 0.054      | Procedure count
7    | discharge_disposition_id    | 0.048      | Where patient discharged to
8    | admission_type_id           | 0.042      | How patient admitted
9    | age                         | 0.038      | Patient age group
10   | insulin                     | 0.035      | Insulin medication changes

Key Patterns:
- Patients with 2+ prior admissions have 3x higher readmission risk
- ER visits in past year are strong predictor
- More diagnoses = more complex patients = higher risk
- Longer hospital stays correlate with readmission
```

**Interactive Filters:**
- Select feature to see distribution (readmitted vs not)
- Filter by demographic group (race, gender, age)

**Code to Reuse:**
- Load feature_importance.csv from Phase 2
- Load train.csv and compute distributions by readmission status
- Create comparative histograms/box plots
- Use EDA insights from Phase 1 notebooks

#### 2. **Latest Patients Panel** (Ensemble Predictions)
**Purpose:** Show recently discharged patients with their readmission risk scores to help prioritize follow-up care.

**Data Sources:**
- `data/diabetic_data.csv` (original unprocessed data for clinical values)
- Model predictions from all 3 models (Gradient Boosting, Random Forest, Logistic Regression)
- Phase 2 metrics for performance-weighted ensemble

**Ensemble Prediction Logic:**
```python
# Weight predictions by ROC-AUC performance from Phase 2
weight_GB = auc_GB / (auc_GB + auc_RF + auc_LR)
weight_RF = auc_RF / (auc_GB + auc_RF + auc_LR)
weight_LR = auc_LR / (auc_GB + auc_RF + auc_LR)

# Calculate ensemble risk score
ensemble_risk = (weight_GB * prob_GB) + (weight_RF * prob_RF) + (weight_LR * prob_LR)
```

**Patient List Table:**
```
Latest Patients (Last 7 Days)

Patient ID | Discharge Date/Time | Risk Score | Age  | Prior Admits | ER Visits | Diagnoses | LOS | Medications | Intervention
-----------|---------------------|------------|------|--------------|-----------|-----------|-----|-------------|-------------
25478      | Dec 21, 2025 14:30 | 87%       | 65-70 |      4       |     6     |    15     |  9  |     18      | 🔴 Critical
18392      | Dec 21, 2025 09:15 | 78%       | 55-60 |      3       |     4     |    12     |  7  |     14      | 🟠 High
31205      | Dec 20, 2025 16:45 | 72%       | 70-75 |      2       |     5     |    11     |  8  |     16      | 🟠 High
42891      | Dec 20, 2025 11:20 | 58%       | 45-50 |      2       |     2     |     9     |  5  |     10      | 🟡 Moderate
19847      | Dec 19, 2025 13:00 | 52%       | 60-65 |      1       |     3     |     8     |  6  |     12      | 🟡 Moderate
36524      | Dec 19, 2025 08:30 | 45%       | 50-55 |      1       |     1     |     7     |  4  |      9      | 🟡 Moderate
27168      | Dec 18, 2025 15:10 | 38%       | 40-45 |      1       |     2     |     6     |  3  |      8      | 🟢 Low
33901      | Dec 18, 2025 10:45 | 34%       | 35-40 |      0       |     1     |     5     |  3  |      7      | 🟢 Low
45672      | Dec 17, 2025 14:20 | 28%       | 30-35 |      0       |     0     |     4     |  2  |      6      | 🟢 Low
29384      | Dec 17, 2025 09:00 | 22%       | 25-30 |      0       |     1     |     3     |  2  |      5      | 🟢 Low

Showing 1-10 of 87 patients | Page size: [10 ▼] | Pages: « 1 2 3 4 ... 9 »
```

**Column Definitions:**
- **Patient ID**: encounter_id from original CSV
- **Discharge Date/Time**: Simulated timestamp within last 7 days (Dec 15-22, 2025)
- **Risk Score**: Ensemble readmission risk percentage (performance-weighted average)
- **Age**: Age group from original data
- **Prior Admits**: number_inpatient (prior hospitalizations in last year)
- **ER Visits**: number_emergency (ER visits in last year)
- **Diagnoses**: number_diagnoses (total diagnosis count)
- **LOS**: time_in_hospital (length of stay in days)
- **Medications**: num_medications (medication count)
- **Intervention**: Recommended action based on risk level

**Intervention Levels:**
- 🔴 **Critical (80-100%)**: "Immediate Case Management" - Schedule within 24 hours, assign care coordinator
- 🟠 **High (60-80%)**: "Intensive Follow-up" - Schedule within 3 days, home visit recommended
- 🟡 **Moderate (40-60%)**: "Standard Follow-up" - Schedule within 1 week, telehealth check-in
- 🟢 **Low (<40%)**: "Routine Care" - Standard discharge instructions

**Interactive Features:**
- **Sortable columns**: Click column headers to sort (default: discharge date descending)
- **Page size selector**: Choose 10, 25, 50, or 100 patients per page (default: 10)
- **Pagination controls**: Previous/Next buttons, page numbers
- **Row highlighting**: Hover over row to highlight, click for patient detail view (future enhancement)

**Time Window:**
- Fixed 7-day window showing most recent discharges
- Keeps list focused and actionable for daily clinical rounds
- Simulated timestamps distributed across last 7 days

**Code to Implement:**
```python
# Load original unprocessed data
import pandas as pd
from huggingface_hub import hf_hub_download
import random
from datetime import datetime, timedelta

# Load original diabetic data for clinical values
original_data_path = hf_hub_download(
    repo_id="auphong2707/hospital-readmission-risk-data",
    filename="diabetic_data.csv",
    repo_type="dataset"
)
original_data = pd.read_csv(original_data_path)

# Load model predictions from all 3 models
# (Assume predictions are generated and stored)

# Load Phase 2 ROC-AUC scores for weighting
auc_scores = {
    'gradient_boosting': 0.842,
    'random_forest': 0.820,
    'logistic_regression': 0.790
}

# Calculate ensemble weights
total_auc = sum(auc_scores.values())
weights = {model: auc / total_auc for model, auc in auc_scores.items()}

# Calculate ensemble risk scores
ensemble_risk = (
    weights['gradient_boosting'] * predictions_gb +
    weights['random_forest'] * predictions_rf +
    weights['logistic_regression'] * predictions_lr
)

# Generate simulated discharge timestamps (last 7 days)
base_date = datetime(2025, 12, 22)  # Current date
for i in range(len(original_data)):
    days_ago = random.randint(0, 6)
    hours = random.randint(8, 18)
    minutes = random.choice([0, 15, 30, 45])
    discharge_datetime = base_date - timedelta(days=days_ago, hours=24-hours, minutes=60-minutes)
    original_data.loc[i, 'discharge_datetime'] = discharge_datetime

# Select relevant clinical columns
patient_list = original_data[[
    'encounter_id',  # Patient ID
    'discharge_datetime',
    'age',
    'number_inpatient',
    'number_emergency',
    'number_diagnoses',
    'time_in_hospital',
    'num_medications'
]].copy()

patient_list['risk_score'] = ensemble_risk

# Add intervention level
def get_intervention(risk):
    if risk >= 0.80:
        return {'level': 'Critical', 'icon': '🔴', 'action': 'Immediate Case Management'}
    elif risk >= 0.60:
        return {'level': 'High', 'icon': '🟠', 'action': 'Intensive Follow-up'}
    elif risk >= 0.40:
        return {'level': 'Moderate', 'icon': '🟡', 'action': 'Standard Follow-up'}
    else:
        return {'level': 'Low', 'icon': '🟢', 'action': 'Routine Care'}

patient_list['intervention'] = patient_list['risk_score'].apply(get_intervention)

# Filter to last 7 days
cutoff_date = base_date - timedelta(days=7)
patient_list = patient_list[patient_list['discharge_datetime'] >= cutoff_date]

# Sort by discharge date descending (most recent first)
patient_list = patient_list.sort_values('discharge_datetime', ascending=False)

# Return paginated results
def get_patient_list(page=1, page_size=10, sort_by='discharge_datetime', sort_order='desc'):
    sorted_list = patient_list.sort_values(sort_by, ascending=(sort_order=='asc'))
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    return {
        'patients': sorted_list.iloc[start_idx:end_idx].to_dict('records'),
        'total': len(sorted_list),
        'page': page,
        'page_size': page_size,
        'total_pages': (len(sorted_list) + page_size - 1) // page_size
    }
```

---

**Summary: Doctor Dashboard Design Philosophy**
- ✅ **Simple**: Only 2 panels - Clinical Risk Factors (population-level) + Latest Patients (individual-level)
- ✅ **Actionable**: Focus on what doctors can understand and use for daily rounds
- ✅ **Clinical Language**: Use real clinical terms (prior admissions, ER visits) not technical features
- ✅ **Patient-Centered**: Show actual patient list with specific intervention recommendations
- ✅ **Ensemble-Based**: Combine all 3 models for more robust predictions (doctors don't care about individual models)
- ❌ **No Technical Jargon**: Avoid ROC-AUC, calibration, Brier scores, fairness metrics
- ❌ **No Financial Metrics**: That's for managers
- ❌ **No Model Comparison**: Doctors need one answer, not three options

---

## Dashboard 3: Manager/Executive Dashboard

### Target Audience
Hospital administrators, finance directors, operations managers, C-suite

### Primary Goals
- **Understand savings ratio and cost avoidance** (primary focus)
- **Plan resource allocation** and staffing
- **Compare cost-effectiveness** of models
- **Make budget decisions**
- Ensure fairness (basic awareness level, not deep technical analysis)

### Key Features

#### 1. **Savings Summary Panel** (Phase 6 - Authoritative)
**Data Sources:**
- `outputs/{model}/final_evaluation/final_system_metrics.json` (Phase 6 - **SINGLE SOURCE OF TRUTH**)
- `outputs/{model}/final_evaluation/deployment_report.json` (Phase 6 - stakeholder-friendly)
- `outputs/{model}/final_evaluation/visualizations/roi_breakdown.png`

**⚠️ Important:** Use Phase 6 data, NOT Phase 4!
- Phase 6 reflects the **final deployed configuration** (includes fairness mitigation if applied)
- Phase 4 is based on single global threshold (may be outdated)
- Phase 6 = actual production system performance

**Visualizations:**
- Savings summary card (large numbers)
- Cost-benefit breakdown (simplified waterfall chart)
- Net savings chart

**Executive Summary Card:**
```
┌─────────────────────────────────────────────────┐
│        FINANCIAL IMPACT SUMMARY                 │
│   Hospital Readmission Prevention System        │
├─────────────────────────────────────────────────┤
│                                                 │
│  Net Program Value:        -$16,544,500        │
│  Baseline Cost:            -$25,560,000        │
│  Net Savings:               $9,015,500         │
│                                                 │
│  Savings per $1 Spent:           $56.66        │
│                                                 │
│  ────────────────────────────────────────────  │
│                                                 │
│  Confusion Matrix (Test Set):                  │
│    True Positives (TP):          311           │
│    False Positives (FP):         318           │
│    True Negatives (TN):       13,243           │
│    False Negatives (FN):       1,393           │
│    Total Test Patients:       15,265           │
│                                                 │
│  ────────────────────────────────────────────  │
│                                                 │
│  Cost Matrix (per patient):                    │
│    TP: +$14,500 (readmission prevented)        │
│    FP:    -$500 (unnecessary intervention)     │
│    TN:       $0 (no action needed)             │
│    FN: -$15,000 (missed readmission)           │
│                                                 │
│  ────────────────────────────────────────────  │
│                                                 │
│  Financial Breakdown:                          │
│    TP Value:  311 × $14.5K =  $4,509,500       │
│    FP Cost:   318 × $500   =   -$159,000       │
│    TN Value:  13,243 × $0  =         $0        │
│    FN Cost:   1,393 × $15K = -$20,895,000      │
│                          ──────────────────     │
│    Net Program Value:        -$16,544,500      │
│                                                 │
│  Baseline (do nothing):                        │
│    All readmissions: 1,704 × $15K              │
│    Baseline Cost:            -$25,560,000      │
│                                                 │
│  Net Savings:                                  │
│    Baseline - Program = $9,015,500 saved       │
│                                                 │
│  ────────────────────────────────────────────  │
│                                                 │
│  Intervention Costs: $159,000                  │
│  Savings per $1: $9,015,500 ÷ $159,000 = $56.66│
│                                                 │
│  ────────────────────────────────────────────  │
│                                                 │
│  Recommended Model: Gradient Boosting          │
│  Status: READY FOR DEPLOYMENT                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Cost Matrix Methodology:**

All calculations use this standard healthcare cost matrix:
- **TP (True Positive)**: +$14,500
  - Prevented readmission saves $15,000
  - Intervention costs $500
  - Net value: $15,000 - $500 = $14,500

- **FP (False Positive)**: -$500
  - Unnecessary intervention costs $500
  - No readmission to prevent
  - Net cost: -$500

- **TN (True Negative)**: $0
  - No intervention needed
  - No readmission occurs
  - Net value: $0

- **FN (False Negative)**: -$15,000
  - Missed readmission costs $15,000
  - No intervention performed
  - Net cost: -$15,000

**Key Metrics Explained:**
- **Net Program Value**: Total financial outcome of running the program
  - Formula: (TP × $14.5K) + (FP × -$500) + (TN × $0) + (FN × -$15K)
  - Example: (311 × $14.5K) - (318 × $500) - (1,393 × $15K) = -$16,544,500

- **Baseline Cost**: Total cost if we do nothing (all potential readmissions occur)
  - Formula: (TP + FN) × -$15K
  - Example: 1,704 × -$15K = -$25,560,000

- **Net Savings**: How much we save compared to doing nothing
  - Formula: Baseline Cost - Net Program Value
  - Example: -$25,560,000 - (-$16,544,500) = $9,015,500

- **Savings per $1 Spent**: Return on intervention investment
  - Formula: Net Savings ÷ Total Intervention Costs
  - Total Intervention Costs = (TP + FP) × $500 = 629 × $500 = $314,500
  - Actually, FP already IS the intervention cost, so:
  - Total Intervention Costs = FP × $500 = 318 × $500 = $159,000
  - Example: $9,015,500 ÷ $159,000 = $56.66 saved per $1 spent

**Metrics:**
- Net savings (annual)
- Savings ratio (cost avoidance efficiency)
**Code to Reuse:**
- Load `final_system_metrics.json` from Phase 6 (contains roi section)
- Load `deployment_report.json` for manager-friendly summary
- Display roi_breakdown.png from Phase 6 visualizations folder

**Code to Reuse:**
- Load roi_metrics.json
- Parse ROI calculations from Phase 4
- Display existing roi_breakdown.png

#### 2. **Resource Planning Panel** (Phase 6)
**Data Sources:**
- `outputs/{model}/final_evaluation/final_system_metrics.json` (Phase 6 - includes intervention counts)
- `outputs/{model}/final_evaluation/visualizations/risk_distribution.png`
- `phase-5-fairness-assessment-mitigation/outputs/{model}/deployment_config.json` (threshold configuration)

**Visualizations:**
- Intervention volume chart (patients per month)
- Staffing requirements calculator
- Capacity utilization chart
- Threshold vs Volume curve

**Resource Planning Table:**
```
Current Hospital Volume: 10,000 patients/year

At Recommended Threshold (0.35):
  Patients Flagged:        4,000 (40%)
  Monthly Volume:            333 patients
  Weekly Volume:              77 patients
  Daily Volume:               11 patients

Staffing Requirements:
  Care Coordinators Needed:  6 FTE
  Nurse Case Managers:       4 FTE
  Social Workers:            2 FTE
  
Annual Personnel Cost:    $1,200,000
Cost per Patient Served:        $300
```

**Interactive Controls:**
- Hospital volume slider (5K - 50K patients/year)
- Staff cost inputs (salary ranges)
- Threshold adjuster (see impact on volume)

**Code to Reuse:**
- Load final_system_metrics.json from Phase 6 (roi section has intervention counts)
- Load deployment_config.json from Phase 5 (threshold settings)
- Calculate staffing based on final intervention volume

#### 3. **Cost Breakdown Panel** (Phase 6)
**Data Sources:**
- `outputs/{model}/final_evaluation/final_system_metrics.json` (Phase 6 - roi section)
- `outputs/{model}/final_evaluation/visualizations/roi_breakdown.png`

**Visualizations:**
- Waterfall chart: Baseline Cost → TP Benefits → FP Costs → FN Costs → Net Savings
- Cost matrix breakdown table
- Confusion matrix with financial values

**Cost Breakdown (Using Cost Matrix):**
```
COST MATRIX (per patient):
  TP (Prevented Readmission):     +$14,500
  FP (Unnecessary Intervention):     -$500
  TN (Correct Non-Intervention):        $0
  FN (Missed Readmission):        -$15,000

──────────────────────────────────────────────

CONFUSION MATRIX (Test Set - 15,265 patients):
  TP:    311 patients
  FP:    318 patients
  TN: 13,243 patients
  FN:  1,393 patients

──────────────────────────────────────────────

FINANCIAL BREAKDOWN:

1. True Positives (Prevented Readmissions):
   311 patients × $14,500 = +$4,509,500
   
2. False Positives (Unnecessary Interventions):
   318 patients × $500 = -$159,000
   
3. True Negatives (No Action Needed):
   13,243 patients × $0 = $0
   
4. False Negatives (Missed Readmissions):
   1,393 patients × $15,000 = -$20,895,000

──────────────────────────────────────────────

NET PROGRAM VALUE:
  $4,509,500 - $159,000 - $20,895,000 = -$16,544,500
  
──────────────────────────────────────────────

BASELINE (No Intervention):
  All potential readmissions occur:
  (TP + FN) = 1,704 patients × $15,000 = -$25,560,000

──────────────────────────────────────────────
         
NET SAVINGS:
  Baseline Cost - Net Program Value:
  -$25,560,000 - (-$16,544,500) = +$9,015,500
  (We save $9M by running the program vs doing nothing)
  
SAVINGS PER DOLLAR SPENT:
  Total Intervention Costs = (TP + FP) × $500 = 629 × $500 = $314,500
  Wait - only FP is pure cost, TP is net benefit after intervention
  
  Actually:
  Total Spent on Interventions = (TP + FP) × $500 = $314,500
  Net Savings = $9,015,500
  Savings per $1 = $9,015,500 ÷ $314,500 = $28.67 per $1 spent
  
  OR if we only count FP cost:
  Intervention Waste = FP × $500 = $159,000
  Savings per $1 = $9,015,500 ÷ $159,000 = $56.66 per $1 spent
```

**Key Insights:**
- The program saves $9M annually compared to doing nothing
- For every $1 spent on interventions, we save $28.67 (or $56.66 depending on cost accounting method)
- The challenge: We miss 1,393 readmissions (FN) which represents $20.9M in losses
- Opportunity: Lowering the threshold could catch more readmissions but increase FP costs

**Code to Implement:**
```python
def calculate_financial_metrics(tp, fp, tn, fn):
    """
    Calculate all financial metrics using the cost matrix.
    
    Cost Matrix:
    - TP: +$14,500 (prevented readmission value)
    - FP: -$500 (unnecessary intervention cost)
    - TN: $0 (no action needed)
    - FN: -$15,000 (missed readmission cost)
    """
    # Program outcomes
    tp_value = tp * 14500
    fp_cost = fp * 500
    tn_value = tn * 0
    fn_cost = fn * 15000
    
    net_program_value = tp_value - fp_cost - fn_cost
    
    # Baseline (do nothing)
    baseline_cost = (tp + fn) * 15000
    
    # Net savings
    net_savings = baseline_cost - abs(net_program_value)
    
    # Intervention costs
    intervention_costs = (tp + fp) * 500
    
    # Savings ratio
    savings_per_dollar = net_savings / intervention_costs if intervention_costs > 0 else 0
    
    return {
        'net_savings': net_savings,
        'savings_per_dollar': savings_per_dollar,
        'baseline_cost': baseline_cost,
        'net_program_value': net_program_value,
        'intervention_costs': intervention_costs,
        'tp_value': tp_value,
        'fp_cost': fp_cost,
        'fn_cost': fn_cost
    }
```

**Code to Reuse:**
- Load final_system_metrics.json from Phase 6 (performance_metrics has TP, FP, TN, FN)
- Apply cost matrix to calculate all financial metrics
- No need to use roi_metrics from Phase 6 - calculate directly from confusion matrix

#### 4. **Model Comparison Panel** (Phase 6)
**Data Sources:**
- All three models: `outputs/[random_forest|gradient_boosting|logistic_regression]/final_evaluation/final_system_metrics.json`

**Note:** All metrics calculated using the cost matrix from confusion matrix values in Phase 6

**Comparison Table (Using Cost Matrix):**
```
Metric                      | Random Forest | Gradient Boost | Logistic Reg
----------------------------|---------------|----------------|---------------
True Positives (TP)         |     TBD       |     311        |     TBD
False Positives (FP)        |     TBD       |     318        |     TBD
False Negatives (FN)        |     TBD       |   1,393        |     TBD
True Negatives (TN)         |     TBD       |  13,243        |     TBD
                            |               |                |
TP Value ($14.5K each)      |     TBD       | $4,509,500     |     TBD
FP Cost ($500 each)         |     TBD       |  -$159,000     |     TBD
FN Cost ($15K each)         |     TBD       |-$20,895,000    |     TBD
TN Value ($0 each)          |     TBD       |        $0      |     TBD
                            |               |                |
Net Program Value           |     TBD       |-$16,544,500    |     TBD
Baseline Cost               |     TBD       |-$25,560,000    |     TBD
Net Savings                 |     TBD       | $9,015,500     |     TBD
                            |               |                |
Intervention Costs          |     TBD       |   $314,500     |     TBD
Savings per $1 Spent        |     TBD       |    $28.67      |     TBD
                            |               |                |
ROC-AUC (Model Performance) |     0.82      |     0.85       |     0.79
Intervention Rate           |     42%       |     41.2%      |     45%
Training Time               |   45 min      |   60 min       |   15 min
Inference Speed             |   50 ms       |   35 ms        |    5 ms
Model Complexity            |   MEDIUM      |   HIGH         |   LOW
Interpretability            |   MEDIUM      |   LOW          |   HIGH
```

**Note**: Intervention Rate = (TP + FP) / Total = 629 / 15,265 = 41.2%

**Recommendation Algorithm:**
```python
# Calculate net savings using cost matrix for each model
def calculate_net_savings(tp, fp, tn, fn):
    net_program_value = (tp * 14500) + (fp * -500) + (tn * 0) + (fn * -15000)
    baseline_cost = (tp + fn) * -15000
    net_savings = baseline_cost - net_program_value
    return net_savings

def calculate_savings_per_dollar(tp, fp, tn, fn, net_savings):
    intervention_costs = (tp + fp) * 500
    return net_savings / intervention_costs if intervention_costs > 0 else 0

# Prioritize by net savings (primary business goal)
if max_net_savings and acceptable_intervention_rate:
    recommend = "Gradient Boosting"
elif need_interpretability_for_clinical_buy_in:
    recommend = "Logistic Regression"
elif balance_savings_and_operational_simplicity:
    recommend = "Random Forest"
```

**Visualizations:**
- Net Savings comparison bar chart
- Savings per $1 Spent comparison
- Confusion matrix heatmap for each model
- Cost-effectiveness scatter plot (Net Savings vs Intervention Rate)

**Fairness Status (Simple for Managers):**
```
Fairness Check:
  Gradient Boosting: ✅ PASS (max disparity 4.2%)
  Random Forest:     ✅ PASS (max disparity 5.8%)
  Logistic Reg:      ⚠️ WARNING (max disparity 6.1%)

→ All models meet fairness requirements
```

**Why Managers Care About Fairness:**
- **Legal/Compliance Risk**: Biased models = discrimination lawsuits
- **Reputation Risk**: Public backlash if bias discovered
- **Regulatory Requirements**: Healthcare AI must be fair
- **Bottom Line**: Simple pass/fail status, not technical details

**Code to Reuse:**
- Load final_system_metrics.json for all 3 models (contains everything)
- Extract performance, ROI, and fairness status
- Compare side-by-side with fairness indicatorl 3 models
- Compare metrics side-by-side

#### 5. **Budget Scenarios**
**Predefined Scenarios:**

**Scenario 1: Limited Budget ($1M/year)**
```
Constraint: Maximum $1M intervention budget
Question: What threshold to use?

Recommended Threshold: 0.45 (stricter)
  - Intervention Volume: 2,500 patients
  - Intervention Cost: $750,000
  - Readmissions Prevented: 650
  - Net Savings: $7,000,000
  - ROI: 280%
  
Strategy: Focus on highest-risk patients only
```

**Scenario 2: Maximum Impact (Unlimited Budget)**
```
Constraint: Prevent as many readmissions as possible
Question: What threshold to use?

Recommended Threshold: 0.25 (permissive)
  - Intervention Volume: 6,000 patients
  - Intervention Cost: $3,000,000
  - Readmissions Prevented: 950
  - Net Savings: $9,000,000
  - ROI: 200%
  
Strategy: Cast wider net, accept more false positives
```

**Scenario 3: Phased Rollout**
```
Phase 1 (Year 1): Pilot - 2,000 patients, $600K investment
  → Net savings: $1.5M, ROI: 150%
  → Build confidence, refine process
  
Phase 2 (Year 2): Expand - 5,000 patients, $1.5M investment
  → Net savings: $4.2M, ROI: 180%
  → Scale operations, hire staff
  
Phase 3 (Year 3): Full deployment - 10,000 patients, $3M investment
  → Net savings: $9.5M, ROI: 217%
  → Mature system, optimize continuously
```

**Code to Reuse:**
- Load threshold_results.csv from Phase 4
- Filter by different threshold values
- Recalculate ROI for different scenarios
- Use Phase 4 ROIAnalyzer class

---

## Implementation Plan

### Phase 7: Stakeholder Dashboards (Extension of Results Collection)

This phase **extends** (not replaces) the existing Phase 7 by creating role-based dashboards that consume data from HuggingFace.

**Relationship to Existing Phase 7:**
- **Keep**: `collect_and_publish.sh` and upload scripts (continue uploading results to HuggingFace)
- **Add**: Dashboard applications that pull data from the HuggingFace repos
- **Benefit**: Dashboards automatically get latest data whenever models are retrained and uploaded

**Directory Structure:**
```
phase-7-results-collection-publication/    # KEEP (existing)
├── collect_and_publish.sh                 # KEEP (uploads to HF)
├── PHASE_OUTPUT_MAPPING.md                # KEEP (reference)
└── README.md                              # KEEP

phase-7-stakeholder-dashboards/            # NEW (add this)
├── docker-compose.yml                     # Grafana + FastAPI stack
├── grafana/                               # Grafana dashboards
├── data-api/                              # FastAPI backend
└── utilities/                             # Data aggregation from HF
```

#### Step 1: Setup (Week 1)
- [ ] **Keep existing Phase 7 collection scripts** (for HuggingFace repo IDs and references)
- [ ] Use `file_to_hf_repo_mapping.py` for all HuggingFace repository links
- [ ] Create `phase-7-stakeholder-dashboards/` directory alongside existing Phase 7
- [ ] Install dependencies: `grafana`, `fastapi`, `uvicorn`, `huggingface-hub`, `redis` (optional)
- [ ] Create `docker-compose.yml` for Grafana + FastAPI services
- [ ] Configure Grafana provisioning for data sources and dashboards
#### Step 2: FastAPI Backend (Week 1-2)
**File**: `data-api/main.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import json
import pandas as pd
from typing import Dict, List
import sys
from pathlib import Path
from functools import lru_cache

# Import the mapping file from project root
sys.path.append(str(Path(__file__).parent.parent.parent))
from file_to_hf_repo_mapping import REPO_MAPPING, get_download_info

app = FastAPI(
    title="Hospital Readmission Risk - Dashboard API",
    description="REST API for Grafana dashboards",
    version="1.0.0"
)

# Enable CORS for Grafana
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DashboardDataAggregator:
class DashboardDataAggregator:
    """
    Aggregate data from Phases 1-6 via HuggingFace for dashboard consumption.
    
    Uses file_to_hf_repo_mapping.py as single source of truth for all repository links.
    """
    
    def __init__(self, method: str):
        """
        Initialize aggregator using the mapping file.
        
        Args:
            method: 'random_forest', 'gradient_boosting', 'logistic_regression'
        """
        if method not in REPO_MAPPING:
            raise ValueError(f"Method {method} not found in REPO_MAPPING")
        
        self.method = method
        self.mapping = REPO_MAPPING[method]
    
    def download_file(self, phase: str, filename: str) -> str:
        """
        Download a file from HuggingFace using the mapping file.
        
        Args:
            phase: 'phase1', 'phase2', 'phase3', or 'phase4'
            filename: The filename as defined in the mapping
        
        Returns:
            Local path to downloaded file (cached by HuggingFace Hub)
        """
        download_info = get_download_info(self.method, phase, filename)
        if not download_info:
            raise ValueError(f"File {filename} not found in mapping for {self.method}/{phase}")
        
        return hf_hub_download(
            repo_id=download_info["repo_id"],
            filename=download_info["file_path"],
            repo_type=download_info["repo_type"]
        )
    
    def load_phase1_data(self) -> Dict:
        """Load Phase 1 preprocessing data using mapping file."""
        split_info_path = self.download_file("phase1", "split_info.txt")
        with open(split_info_path, 'r') as f:
            split_info = f.read()
        return {'split_info': split_info}
    
    def load_phase2_metrics(self) -> Dict:
        """Load Phase 2 model metrics and visualizations using mapping file."""
        # Determine correct filename based on method
        if self.method == 'logistic_regression':
            metrics_file = "logistic_regression_metrics.json"
        else:
            metrics_file = f"{self.method}_metrics.json"
        
        metrics_path = self.download_file("phase2", metrics_file)
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load feature importance (same naming across all models)
        feat_imp_path = self.download_file("phase2", "feature_importance.csv")
        feature_importance = pd.read_csv(feat_imp_path)
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    def load_phase3_calibration(self) -> Dict:
        """Load Phase 3 calibration metrics using mapping file."""
        # Determine calibrator prefix based on method (defined in mapping file)
        calibrator_prefix_map = {
            "gradient_boosting": "Gradient_Boosting_(LightGBM)",
            "random_forest": "Random_Forest",
            "logistic_regression": "Logistic_Regression"
        }
        calibrator_prefix = calibrator_prefix_map[self.method]
        metrics_file = f"{calibrator_prefix}_metrics.json"
        
        metrics_path = self.download_file("phase3", metrics_file)
        with open(metrics_path, 'r') as f:
            calibration_metrics = json.load(f)
        
        return {'calibration_metrics': calibration_metrics}
    
    def load_phase4_roi(self) -> Dict:
        """Load Phase 4 ROI and threshold optimization using mapping file."""
        roi_path = self.download_file("phase4", "roi_metrics.json")
        thresholds_path = self.download_file("phase4", "optimal_thresholds.json")
        
        with open(roi_path, 'r') as f:
            roi_metrics = json.load(f)
        
        with open(thresholds_path, 'r') as f:
**FastAPI Endpoints:**
```python
# Phase 2: Model Performance
@app.get("/api/v1/models/{method}/metrics")
@lru_cache()
def get_model_metrics(method: str):
    """Get Phase 2 model performance metrics"""
    aggregator = DashboardDataAggregator(method)
    return aggregator.load_phase2_metrics()

# Phase 4: ROI Analysis
@app.get("/api/v1/models/{method}/roi")
@lru_cache()
def get_roi_metrics(method: str):
#### Step 3: Grafana Dashboards (Week 2-4)

**Docker Compose Setup:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  grafana:
    image: grafana/grafana:latest
    container_name: hospital-dashboards-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-json-datasource
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - grafana-storage:/var/lib/grafana
    depends_on:
      - data-api
```
┌──────────────────────────────────────────────────────────┐
│              Web Browser (Grafana UI)                    │
│  http://localhost:3000                                   │
│  - Data Analyst Dashboard                                │
│  - Clinician Dashboard                                   │
│  - Executive Dashboard                                   │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP (port 3000)
┌──────────────────────▼───────────────────────────────────┐
│              Grafana Server (Container)                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Dashboard Renderer (pre-configured JSON)         │ │
│  │  - Panels, charts, tables                         │ │
│  │  - Variables for model selection                  │ │
│  │  - Templating and filters                         │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────────────────┘
                       │ REST API calls
┌──────────────────────▼───────────────────────────────────┐
│            FastAPI Backend (Container)                   │
│  ┌────────────────────────────────────────────────────┐ │
│  │  REST Endpoints:                                  │ │
│  │  GET /api/v1/models/{method}/metrics              │ │
│  │  GET /api/v1/models/{method}/roi                  │ │
│  │  GET /api/v1/models/{method}/fairness             │ │
│  │  GET /api/v1/models/compare                       │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Data Aggregator (HuggingFace downloads)         │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Cache Layer (Redis + @lru_cache)                │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────────────────┘
                       │ HuggingFace Hub API
┌──────────────────────▼───────────────────────────────────┐
│              HuggingFace Hub (Cloud)                     │
│                                                           │
│  📦 hospital-readmission-risk-data (dataset)            │
│  📦 hospital-readmission-phase2-{model} (models)        │
│  📦 hospital-readmission-phase3-{model}-calibrated      │
│  📦 hospital-readmission-phase4-{model}-threshold       │
│  📦 hospital-readmission-phase5-{model}-fairness        │
│                                                           │
│  ✅ Already uploaded by existing Phase 7 scripts        │
└───────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- ✅ Enterprise-grade dashboarding with Grafana
- ✅ No local file dependencies (all via API)
- ✅ Automatic caching at multiple layers (Redis + FastAPI + Grafana)
- ✅ Real-time updates when data changes
- ✅ Role-based access control (Grafana built-in)
- ✅ Alert capabilities for monitoring model performance
- ✅ Easy to containerize and deploy (Docker Compose)
- ✅ Leverages existing Phase 7 upload infrastructure
  - name: Hospital-API
    type: grafana-json-datasource
    access: proxy
    url: http://data-api:8000
    isDefault: true
    jsonData:
      httpMethod: GET
    editable: false
```

#### Step 3a: Data Analyst Dashboard (Week 2-3)
**File:** `grafana/provisioning/dashboards/data_analyst.json`

# Phase 5: Fairness
@app.get("/api/v1/models/{method}/fairness")
@lru_cache()
def get_fairness_metrics(method: str):
    """Get Phase 5 fairness assessment"""
    aggregator = DashboardDataAggregator(method)
    return aggregator.load_phase5_fairness()

# Phase 6: Final Evaluation
@app.get("/api/v1/models/{method}/final-evaluation")
@lru_cache()
def get_final_evaluation(method: str):
    """Get Phase 6 final system metrics"""
    aggregator = DashboardDataAggregator(method)
    return aggregator.load_phase6_final()

# Comparison endpoint
@app.get("/api/v1/models/compare")
def compare_models():
**Grafana Dashboard Panels:**
- [ ] **ROI Summary**: Stat panels showing net savings, ROI %, payback period
- [ ] **Cost Breakdown**: Pie chart showing cost categories
- [ ] **Model Comparison**: Table panel comparing all 3 models
- [ ] **Resource Planning**: Bar chart showing intervention volumes
- [ ] **Savings Waterfall**: Bar chart showing costs → benefits → net savings
- [ ] **Variables**: Dropdown for model selection (RF, GB, LR)
- [ ] **Annotations**: Mark important milestones or changes

**Technology**: Grafana (with export to PDF via Grafana Image Renderer)
    return comparison
#### Step 6: Testing & Documentation (Week 5-6)
- [ ] Test all Grafana dashboards with different models (RF, GB, LR)
- [ ] Test FastAPI endpoints with Swagger UI
- [ ] Test with missing data scenarios and error handling
- [ ] Configure Grafana alerts for data freshness
- [ ] Create user guides for each role
- [ ] Record demo videos
- [ ] Write README.md for Phase 7
- [ ] Document API endpoints with OpenAPI/Swagger
- [ ] Create deployment guide for production (Kubernetes, cloud)

**Tasks:**
- [ ] Implement DashboardDataAggregator class with HuggingFace Hub API
- [ ] Use `file_to_hf_repo_mapping.py` for all repository and file path references
- [ ] Import REPO_MAPPING and get_download_info() helper functions
- [ ] Add error handling for HuggingFace download failures
- [ ] Add caching for performance (HF already caches locally)
- [ ] Write unit tests

**Key Implementation Details:**
- Use `get_download_info(method, phase, filename)` to get repo_id, file_path, and repo_type
- All file paths are maintained in the mapping file (single source of truth)
- No hardcoded HuggingFace repository URLs in dashboard code
- Automatically handles different naming conventions (e.g., LR vs GB/RF file names)

#### Step 3: Data Analyst Dashboard (Week 2-3)
**Files:**
#### Step 3: Data Analyst Dashboard (Week 2-3)
```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                          │
│  (Analyst Dashboard | Doctor Dashboard | Manager Dash)  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (port 8501, 8502, 8503)
┌────────────────────────▼────────────────────────────────┐
│               Streamlit Application Server              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Data Aggregator (downloads from HuggingFace)   │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Metrics Calculator (computes dashboard metrics)│  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Visualization Helpers (plotly/matplotlib)      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ HuggingFace Hub API
┌────────────────────────▼────────────────────────────────┐
│              HuggingFace Hub (Cloud)                    │
│                                                          │
│  📦 hospital-readmission-risk-data (dataset)           │
│  📦 hospital-readmission-phase2-{model} (models)       │
│  📦 hospital-readmission-phase3-{model}-calibrated     │
│  📦 hospital-readmission-phase4-{model}-threshold      │
│  📦 hospital-readmission-phase5-{model}-fairness       │
│                                                          │
│  ✅ Already uploaded by existing Phase 7 scripts       │
└──────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- ✅ No local file dependencies
- ✅ Automatic caching by HuggingFace Hub
- ✅ Dashboards work anywhere (cloud deployment, local dev)
- ✅ Leverages existing Phase 7 upload infrastructuremanager_dashboard/executive_dashboard.py` (main Streamlit app)
- `manager_dashboard/components/*.py` (individual panels)

**Tasks:**
- [ ] Create ROI summary card (big numbers!)
- [ ] Implement resource planning panel
- [ ] Implement cost breakdown
- [ ] Implement model comparison
- [ ] Add budget scenario calculator
- [ ] Create executive summary report export
- [ ] Focus on $ and ROI metrics

**Technology**: Streamlit (can export to PDF)

#### Step 6: Testing & Documentation (Week 5-6)
- [ ] Test all dashboards with different models (RF, GB, LR)
- [ ] Test with missing data scenarios
- [ ] Create user guides for each role
- [ ] Record demo videos
- [ ] Write README.md for Phase 8

@st.cache_data
def load_dashboard_data(model_name):
    """Load data from HuggingFace (cached for performance)."""
    # Use mapping file to ensure correct repository references
    from file_to_hf_repo_mapping import REPO_MAPPING
    
## Technology Stack

### Framework Choice: Grafana + FastAPI
**Why Grafana:**
- ✅ Industry-standard dashboarding platform
- ✅ Enterprise-grade features (RBAC, alerts, annotations)
- ✅ Highly customizable and extensible
- ✅ Excellent performance with large datasets
- ✅ Built-in authentication and user management
- ✅ Easy containerization with Docker
- ✅ Real-time data refresh capabilities
- ✅ Export to PDF/PNG for reports

**Why FastAPI (Backend):**
- ✅ High performance (async support)
- ✅ Automatic OpenAPI documentation
- ✅ Type hints and validation (Pydantic)
- ✅ Easy to test and maintain
- ✅ Python-native (reuse existing utilities)

### Key Libraries

```python
# Backend API
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.4.0

# Data Processing
pandas>=1.5.0
numpy>=1.23.0

# HuggingFace Integration
huggingface-hub>=0.19.0

# Caching
### Dashboard Architecture (Grafana-based)

```
┌──────────────────────────────────────────────────────────┐
│         Web Browser (http://localhost:3000)              │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │   Analyst   │  │  Clinician   │  │   Executive    │ │
│  │  Dashboard  │  │  Dashboard   │  │   Dashboard    │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP/JSON
┌──────────────────────▼───────────────────────────────────┐
│            Grafana (Docker Container)                    │
│  - Rendering engine                                      │
│  - User authentication                                   │
│  - Dashboard provisioning                                │
│  - Data source management                                │
└──────────────────────┬───────────────────────────────────┘
                       │ REST API (JSON)
┌──────────────────────▼───────────────────────────────────┐
│        FastAPI Backend (Docker Container)                │
│  - Endpoint routing                                      │
│  - Data aggregation from HuggingFace                     │
│  - Response caching (Redis)                              │
│  - Metrics calculation                                   │
└──────────────────────┬───────────────────────────────────┘
                       │ HuggingFace Hub API
┌──────────────────────▼───────────────────────────────────┐
│              HuggingFace Hub (Cloud)                     │
│  - Phase 1-6 outputs (JSON, CSV, images)                │
│  - Model artifacts                                       │
└───────────────────────────────────────────────────────────┘
```
```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                          │
│  (Analyst Dashboard | Doctor Dashboard | Manager Dash)  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (port 8501, 8502, 8503)
┌────────────────────────▼────────────────────────────────┐
│               Streamlit Application Server              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Data Aggregator (loads from Phases 1-6)        │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Metrics Calculator (computes dashboard metrics)│  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Visualization Helpers (plotly/matplotlib)      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────┘
---

## Sample Code Structure

### Example: FastAPI Backend

```python
# data-api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import sys
from pathlib import Path

# Import utilities
sys.path.append(str(Path(__file__).parent.parent))
from utilities.data_aggregator import DashboardDataAggregator
from utilities.cache_manager import CacheManager

app = FastAPI(
    title="Hospital Readmission Dashboard API",
    description="REST API serving Grafana dashboards",
    version="1.0.0"
)

# Enable CORS for Grafana
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize cache
cache = CacheManager(redis_url="redis://redis:6379")

@app.get("/")
def root():
    return {"message": "Hospital Readmission Dashboard API", "version": "1.0.0"}

@app.get("/api/v1/models")
def list_models():
    """List available models"""
    return {
        "models": [
            {"id": "gradient_boosting", "name": "Gradient Boosting", "recommended": True},
            {"id": "random_forest", "name": "Random Forest", "recommended": False},
            {"id": "logistic_regression", "name": "Logistic Regression", "recommended": False}
        ]
    }

@app.get("/api/v1/models/{method}/roi")
def get_roi_metrics(method: str):
    """Get ROI metrics for Executive Dashboard"""
    try:
        # Check cache first
        cached = cache.get(f"roi_{method}")
        if cached:
            return cached
        
        # Load from HuggingFace
        aggregator = DashboardDataAggregator(method)
        data = aggregator.load_phase6_final()  # Use Phase 6 for authoritative ROI
        
        roi_data = {
            "method": method,
            "net_savings": data["roi"]["net_savings"],
            "roi_percentage": data["roi"]["roi_percentage"],
            "readmissions_prevented": data["roi"]["readmissions_prevented"],
            "intervention_cost": data["roi"]["total_intervention_cost"],
            "cost_avoidance": data["roi"]["cost_avoidance"],
            "payback_months": data["roi"]["payback_months"]
        }
        
        # Cache for 1 hour
        cache.set(f"roi_{method}", roi_data, ttl=3600)
        
        return roi_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/{method}/metrics")
def get_model_metrics(method: str):
    """Get Phase 2 model performance metrics"""
    try:
        cached = cache.get(f"metrics_{method}")
        if cached:
            return cached
        
        aggregator = DashboardDataAggregator(method)
        data = aggregator.load_phase2_metrics()
        
        metrics = {
            "method": method,
            "roc_auc": data["metrics"]["roc_auc"],
            "pr_auc": data["metrics"]["pr_auc"],
            "precision": data["metrics"]["precision"],
            "recall": data["metrics"]["recall"],
            "f1": data["metrics"]["f1_score"],
            "feature_importance": data["feature_importance"].to_dict(orient="records")
        }
        
        cache.set(f"metrics_{method}", metrics, ttl=3600)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/compare")
def compare_models():
    """Compare all three models for Data Analyst Dashboard"""
    methods = ["gradient_boosting", "random_forest", "logistic_regression"]
    comparison = []
    
    for method in methods:
        try:
            aggregator = DashboardDataAggregator(method)
            phase2 = aggregator.load_phase2_metrics()
            phase6 = aggregator.load_phase6_final()
            
            comparison.append({
                "method": method,
                "roc_auc": phase2["metrics"]["roc_auc"],
                "precision": phase2["metrics"]["precision"],
                "recall": phase2["metrics"]["recall"],
                "roi_percentage": phase6["roi"]["roi_percentage"],
                "net_savings": phase6["roi"]["net_savings"]
            })
        except Exception as e:
            print(f"Error loading {method}: {e}")
            continue
    
    return {"comparison": comparison}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Example: Grafana Dashboard JSON (Executive Dashboard)

```json
{
  "dashboard": {
    "title": "Executive Dashboard - Hospital Readmission Risk",
    "tags": ["executive", "roi", "financial"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Annual Net Savings",
        "type": "stat",
        "targets": [
          {
            "refId": "A",
            "target": "api/v1/models/${model}/roi",
            "jsonPath": "$.net_savings"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "color": {"mode": "thresholds"},
            "thresholds": {
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 1000000, "color": "yellow"},
                {"value": 2000000, "color": "green"}
              ]
            }
          }
        },
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "ROI Percentage",
        "type": "stat",
        "targets": [
          {
            "refId": "A",
            "target": "api/v1/models/${model}/roi",
            "jsonPath": "$.roi_percentage"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "color": {"mode": "thresholds"}
          }
        },
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0}
      },
      {
        "id": 3,
        "title": "Model Comparison",
        "type": "table",
        "targets": [
          {
            "refId": "A",
            "target": "api/v1/models/compare"
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 4}
      }
    ],
    "templating": {
      "list": [
        {
          "name": "model",
          "type": "custom",
          "options": [
            {"text": "Gradient Boosting", "value": "gradient_boosting"},
            {"text": "Random Forest", "value": "random_forest"},
            {"text": "Logistic Regression", "value": "logistic_regression"}
          ],
          "current": {"value": "gradient_boosting"}
        }
      ]
    }
  }
}
```         height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Executive summary text
        st.success(f"""
        **Recommendation**: Deploy this model for a **{self.roi_metrics['roi_percentage']:.0f}% ROI**.
        
        Expected to prevent **{self.roi_metrics['readmissions_prevented']:,} readmissions** 
        annually, saving **${self.roi_metrics['net_savings']:,.0f}** after intervention costs.
        """)
```

---

## Data Requirements

### Minimum Data Needed (Per Model)

| Phase | Critical Files | Optional Files |
|-------|----------------|----------------|
| Phase 1 | `split_info.txt` | CSV samples |
| Phase 2 | `*_metrics.json`, `*_feature_importance.csv` | Visualizations, training logs |
| Phase 3 | `*_metrics.json`, calibration comparison | Reliability diagrams |
| Phase 4 | `roi_metrics.json`, `optimal_thresholds.json` | All visualizations |
| Phase 5 | `group_metrics_*.csv`, `deployment_config.json` | Statistical tests, heatmaps |
| Phase 6 | `final_system_metrics.json`, `deployment_report.json` | Visualizations |

**Total**: ~15-20 JSON/CSV files per model (core data)

### Configuration File Structure

```python
# dashboard_config.json
{
  "data_source": "huggingface",
  "mapping_file": "../file_to_hf_repo_mapping.py",
  "cache_dir": "./cache",
  "refresh_interval_hours": 24,
  "available_methods": ["gradient_boosting", "random_forest", "logistic_regression"],
  "default_method": "gradient_boosting"
}
```

**Key Benefits of Using Mapping File:**
- ✅ Single source of truth for all HuggingFace repository links
- ✅ Easy to update if repositories change (update one file)
- ✅ Automatic handling of different naming conventions per method
- ✅ Type-safe access with helper functions
- ✅ Consistent across all three dashboards
## Usage Instructions

### Starting the Dashboard System

#### Method 1: Docker Compose (Recommended)
```bash
# Navigate to project root
cd hospital-readmission-risk/phase-7-stakeholder-dashboards

# Start all services (Grafana + FastAPI + Redis)
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access Dashboards:**
- Grafana UI: `http://localhost:3000` (default login: admin/admin)
- FastAPI Docs: `http://localhost:8000/docs`
- Redis: `localhost:6379`

#### Method 2: Local Development
```bash
# Terminal 1: Start FastAPI backend
cd phase-7-stakeholder-dashboards/data-api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2: Start Grafana (local installation)
grafana-server --config grafana.ini

# Terminal 3: Start Redis (optional, for caching)
redis-server
```

### Accessing Dashboards in Grafana

1. **Login to Grafana**: http://localhost:3000 (admin/admin)
2. **Navigate to Dashboards** → Browse
3. **Select Dashboard:**
   - **Data Analyst Dashboard**: Technical metrics and model comparison
   - **Clinician Dashboard**: Clinical risk factors and fairness
   - **Executive Dashboard**: ROI, costs, and resource planning
4. **Use Variables**: Select model from dropdown (Gradient Boosting, Random Forest, Logistic Regression)
5. **Refresh Data**: Click refresh icon or set auto-refresh interval """Safely download file with error handling."""
    try:
        download_info = get_download_info(method, phase, filename)
### ⚠️ Considerations & Solutions

1. ✅ **Data Availability**: Phases 1-6 already run, results uploaded to HuggingFace
2. ✅ **File Path Management**: Use HuggingFace Hub API to download data (no local path issues)
3. ✅ **Performance**: Multi-layer caching (Redis + FastAPI @lru_cache + Grafana)
4. ✅ **Authentication**: Grafana has built-in user management and RBAC
5. ✅ **Data Privacy**: Patient data already anonymized in Phase 1
6. ✅ **Keep Phase 7 Collection Scripts**: Reuse repository IDs and download logic from existing scripts
7. ✅ **Scalability**: Docker Compose for local, can migrate to Kubernetes for production
8. ✅ **Real-time Updates**: Grafana auto-refresh + FastAPI async support
9. ✅ **Monitoring**: Grafana alerts can notify when data is stale or API errors occur
        )
        return file_path
    except Exception as e:
        st.error(f"Failed to download {filename}: {str(e)}")
        return None

def safe_load_json(filepath: str, default: dict = None) -> dict:
    """Load JSON with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"File not found: {filepath}")
        return default or {}
    except json.JSONDecodeError:
        st.error(f"Invalid JSON: {filepath}")
        return default or {}
```

---

## Usage Instructions

#### Data Analyst
### Quick Win: Minimal Viable Dashboard (MVP)

**Goal**: Get something working in 2-3 days

**MVP Approach with Grafana:**

#### Phase 1: Backend API (Day 1 - 4 hours)
- Set up FastAPI with basic structure
- Implement 2-3 core endpoints:
  - `GET /api/v1/models/{method}/roi` (Phase 6 data)
  - `GET /api/v1/models/{method}/metrics` (Phase 2 data)
  - `GET /api/v1/models/compare` (all models)
- Test with Swagger UI
- Add basic caching

#### Phase 2: Docker Setup (Day 1 - 2 hours)
- Create docker-compose.yml with Grafana + FastAPI
- Configure JSON data source in Grafana
- Test connectivity between containers

#### Phase 3: Executive Dashboard (Day 2 - 6 hours)
- Create Grafana dashboard JSON
- Add 4-5 key panels:
  - **Stat Panel**: Net Savings (big number)
  - **Stat Panel**: ROI % (big number)
  - **Table Panel**: Model comparison
  - **Bar Chart**: Cost breakdown
  - **Gauge**: Readmissions prevented
- Configure variables for model selection
- Test with real data

#### Phase 4: Polish & Documentation (Day 3 - 4 hours)
- Add error handling to API
- Create README with usage instructions
- Record 5-minute demo video
- Export dashboard to PDF

**Total Estimated Time**: 16 hours over 3 days

**Deliverable**: Working Grafana dashboard showing ROI metrics for all 3 models, accessible at http://localhost:3000

---

## Why Grafana Over Streamlit?

### Advantages of Grafana:
1. ✅ **Enterprise-Ready**: Built for production environments
2. ✅ **Better Performance**: Optimized for real-time data visualization
3. ✅ **User Management**: Built-in RBAC, teams, and permissions
4. ✅ **Alerting**: Native support for alerts and notifications
5. ✅ **Extensibility**: Plugin ecosystem for custom visualizations
6. ✅ **Export Options**: PDF reports, snapshots, embeds
7. ✅ **Dashboard Versioning**: Track changes over time
8. ✅ **API-First**: Separate concerns (frontend/backend)
9. ✅ **Industry Standard**: Widely adopted, well-documented
10. ✅ **Scalability**: Can handle large datasets and concurrent users

### When to Use Streamlit Instead:
- ❌ Rapid prototyping (need something in 1 hour)
- ❌ Heavy Python interactivity (widgets, forms)
- ❌ Data exploration/analysis tools (not dashboards)
- ❌ Single-user applications

**Recommendation**: Use Grafana for this project. The separation of concerns (FastAPI backend + Grafana frontend) makes it more maintainable, scalable, and production-ready.

---

## Summary

This updated plan:
- ✅ **Uses Grafana for dashboards** instead of Streamlit/Plotly Dash
- ✅ **FastAPI backend** serves data from HuggingFace
- ✅ **Docker Compose** for easy deployment
- ✅ **Multi-layer caching** (Redis + FastAPI + Grafana)
- ✅ **Replaces static reports** with interactive, enterprise-grade dashboards
- ✅ **Maximizes code reuse**: Leverages existing utilities and outputs from Phases 1-6
- ✅ **Serves all stakeholders**: Data analyst, doctor, manager (3 separate dashboards)
- ✅ **Enables decision-making**: Interactive model comparison and what-if scenarios
- ✅ **Production-ready**: Authentication, alerts, monitoring built-in

**Key Insight**: The hard work is already done (Phases 1-6). Phase 7 is about **presenting** the results in role-appropriate ways using industry-standard tools.

**Migration from Old Phase 7**: Keep the HuggingFace publication scripts. They continue to upload data that the dashboards consume via API.

Would you like me to start implementing the MVP (Executive Dashboard with Grafana + FastAPI)?

1. **Replaces Static Reports**: Interactive dashboards instead of static JSON/text files
2. **Reuses Existing Code**: Leverages utilities from Phases 4-6 (ROI, fairness, visualization)
3. **Role-Specific Views**: Each stakeholder sees only relevant information
4. **Interactive**: Streamlit allows dynamic exploration without code changes
5. **Extensible**: Easy to add new panels or scenarios
6. **Reproducible**: Dashboards always reflect actual pipeline outputs
7. **Deployable**: Can containerize and deploy to cloud or local server
8. **Educational**: Helps stakeholders understand ML system end-to-end
9. **More Useful**: Actual decision-support tools vs static result collection
### ⚠️ Considerations & Solutions

1. ✅ **Data Availability**: Phases 1-6 already run, results uploaded to HuggingFace
2. ✅ **File Path Management**: Use HuggingFace Hub API to download data (no local path issues)
3. ✅ **Performance**: HuggingFace caching + Streamlit caching = fast loading
4. ⚠️ **Authentication**: May need login system for production deployment
5. ✅ **Data Privacy**: Patient data already anonymized in Phase 1
6. ✅ **Keep Phase 7 Collection Scripts**: Reuse repository IDs and download logic from existing scripts

---

## Next Steps

### Immediate Actions (for You)

1. **Review this plan**: Ensure it meets your requirements
2. **Choose framework**: Streamlit (recommended) vs Dash vs other
3. **Prioritize dashboards**: Which role to build first?
   - Recommendation: Start with **Manager Dashboard** (highest impact, clearest metrics)
4. **Verify data availability**: Ensure Phases 1-6 have been run for at least one model

### Implementation Order (Suggested)

1. **Week 1**: Setup + Data Aggregation Layer
2. **Week 2**: Manager Dashboard (easiest, clearest requirements)
3. **Week 3**: Doctor Dashboard (most impactful for users)
4. **Week 4**: Data Analyst Dashboard (most complex)
5. **Week 5**: Testing + Documentation
6. **Week 6**: Deployment + Demo

### Quick Win: Minimal Viable Dashboard (MVP)

**Goal**: Get something working in 2-3 days

**Scope**: Manager Dashboard with 2 panels only
### Quick Win: Minimal Viable Dashboard (MVP)

**Goal**: Get something working in 2-3 days

**Option 1 - Manager Dashboard** (Highest Business Impact):
- ROI Summary Panel (Phase 4 data)
- Model Comparison Panel (Phase 6 data)
- **Estimated Time**: 6-8 hours

**Option 2 - Doctor Dashboard** (Simplest & Most Focused):
- Risk Factors Panel (Phase 1 EDA + Phase 2 feature importance)
- Fairness Panel (Phase 5 group metrics)
**Option 2 - Doctor Dashboard** (Simplest & Most Focused):
- Risk Factors Panel (Phase 1 EDA + Phase 2 feature importance)
- Fairness Panel (Phase 5 group metrics)
- **Estimated Time**: 4-6 hours (simplest dashboard)

**Option 3 - Data Analyst Dashboard** (Most Technical):
- Model Comparison Panel (Phase 2 + 6)
- Calibration Panel (Phase 3)
- Fairness Panel (Phase 5)
- Quick Insights (auto-generated)
- **Estimated Time**: 6-8 hours (technical but streamlined)
- `[dashboard]/[dashboard].py` (main app)
- `[dashboard]/components/[panel].py` (1-2 panels)n static result collection
- **Maximizes code reuse**: Leverages existing utilities and outputs from Phases 1-6
- **Serves all stakeholders**: Data analyst, doctor, manager
- **Enables decision-making**: Interactive scenarios and what-if analysis
- **Is implementable**: Clear structure, technology choices, and timeline

**Key Insight**: The hard work is already done (Phases 1-6). New Phase 7 is about **presenting** the results in role-appropriate ways for actual decision-making.

**Migration from Old Phase 7**: You can optionally keep the HuggingFace publication scripts for model artifacts, but the main Phase 7 deliverable shifts from "collected JSON files" to "interactive dashboards."

Would you like me to start implementing the MVP (Manager Dashboard) or proceed with the full implementation plan?role-appropriate ways.

Would you like me to start implementing the MVP (Manager Dashboard) or proceed with the full implementation plan?
