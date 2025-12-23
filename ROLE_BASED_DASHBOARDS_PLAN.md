# Role-Based Dashboards - Implementation Documentation
## Hospital Readmission Risk Prediction System

**Last Updated**: December 24, 2025  
**Status**: ✅ Implemented  
**Purpose**: Interactive web dashboards for three distinct stakeholder roles

---

## Executive Summary

Phase 7 delivers three specialized dashboards that present insights from the ML pipeline (Phases 1-6) in role-appropriate formats. Built with FastAPI backend + HTML/CSS/Plotly.js frontend, the dashboards serve data scientists, clinicians, and managers with tailored visualizations and metrics.

### Three Dashboards

| Dashboard | Target Audience | Primary Focus | Access URL |
|-----------|----------------|---------------|------------|
| **Data Analyst** | Data scientists, ML engineers | Pipeline validation, model comparison, technical metrics | `/dashboards/data-analyst` |
| **Doctor** | Clinicians, nurses | Risk factors, clinical insights, patient list | `/dashboards/doctor` |
| **Manager** | Executives, administrators | Financial impact, ROI, cost breakdown | `/dashboards/manager` |

---

## Technology Stack

- **Backend**: FastAPI (REST API + Jinja2 templating)
- **Frontend**: HTML/CSS + Plotly.js (interactive visualizations)
- **Data Source**: HuggingFace Hub (pipeline outputs from Phases 1-6)
- **Deployment**: Standalone Python application (no Docker required)

### Quick Start

```bash
# From project root
start.bat

# Or manually:
cd phase-7-stakeholder-dashboards\data-api
python main.py
```

Access dashboards at:
- Data Analyst: http://localhost:8000/dashboards/data-analyst
- Doctor: http://localhost:8000/dashboards/doctor
- Manager: http://localhost:8000/dashboards/manager
- API Docs: http://localhost:8000/docs

---

## Dashboard 1: Data Analyst Dashboard

### Target Audience
Data scientists, ML engineers, technical reviewers

### Purpose
End-to-end pipeline visibility with technical metrics and model comparison across all 6 phases.

### Implemented Sections

#### 1. Quick Start: Recommended Model
- Best performing model card with key metrics
- Performance summary (ROC-AUC, Brier Score, Recall, F1)
- AI-powered recommendation badge

**API**: `/api/v1/quick-insights`

#### 2. Phase 1: Data Preprocessing
- Class distribution pie chart (Readmitted vs Not Readmitted)
- Dataset split pie chart (Train/Val/Test)
- Missing data overview bar chart
- Feature importance comparison (all 3 models)

**APIs**: `/api/v1/phase1/missing-data`, `/api/v1/phase1/feature-importance-comparison`

#### 3. Phase 2: Risk Modeling
- ROC curves overlay (all 3 models with AUC values)
- Precision-Recall curves comparison
- Confusion matrices grid (side-by-side)
- Model performance table (ROC-AUC, Precision, Recall, F1, Specificity)

**APIs**: `/api/v1/models/compare`, `/api/v1/visualizations/roc-curves-data`, `/api/v1/visualizations/pr-curves-data`

#### 4. Phase 3: Model Calibration
- Reliability diagrams (3 panels, one per model)
- Calibration curves (predicted vs actual probabilities)
- Brier Score and ECE metrics (before/after)

**APIs**: `/api/v1/models/{method}/calibration-metrics`, `/api/v1/visualizations/calibration-curve-data/{method}`

#### 5. Phase 4: Optimal Threshold & ROI
- Costs vs Threshold curves
- Benefits vs Threshold curves
- Confusion matrices at optimal threshold (3-model comparison)

**APIs**: `/api/v1/models/phase4/threshold-curves`, `/api/v1/phase4/costs-data`, `/api/v1/phase4/benefits-data`

#### 6. Phase 5: Fairness Assessment
- Demographic selector dropdown (Race/Gender/Age)
- TPR Gap charts (Before vs After mitigation)
- FPR Gap charts (Before vs After mitigation)
- Pass/fail status vs fairness thresholds

**APIs**: `/phase5/fairness-gaps/{method}` (dynamic filtering by demographic)

#### 7. Phase 6: Final System Evaluation
- Deployment readiness table with pass/fail status
- Comprehensive evaluation checklist

**API**: `/api/v1/models/phase6/final-evaluation`

---

## Dashboard 2: Doctor Dashboard

### Target Audience
Clinicians, nurses, clinical staff

### Purpose
Actionable clinical insights and patient-level risk information for care planning.

### Implemented Sections

#### 1. Clinical Risk Factors
- **Top 10 Risk Factors**: Interactive bar chart with feature importance
- **Feature Detail Card**: Click any bar to view:
  - Clinical interpretation
  - Risk association
  - Actionable recommendations
  - Distribution statistics

**Example Top Factors**:
1. Number of inpatient visits (prior hospitalizations)
2. Number of emergency visits
3. Number of diagnoses (comorbidity burden)
4. Time in hospital (length of stay)
5. Number of medications
6. Number of procedures
7. Age group
8. Discharge disposition
9. Admission type
10. Insulin medication changes

**API**: `/api/v1/models/{method}/risk-factors`

#### 2. Clinical Insights
- **Age Stratification Analysis**: Readmission rates across age groups with 95% confidence intervals
- **Diagnosis Category Prevalence**: Patient distribution by primary diagnosis

Shows high-risk age groups and most common diagnosis categories.

**API**: `/api/v1/models/{method}/clinical-patterns`

#### 3. Latest Patients
- **Risk Score Distribution**: Histogram of risk scores across all patients
- **Patient List Table**: Recently discharged patients with:
  - Patient ID
  - Discharge date/time
  - Risk score percentage
  - Age group
  - Prior hospitalizations
  - ER visits
  - Number of diagnoses
  - Length of stay
  - Number of medications
  - Recommended intervention level

**Interactive Features**:
- Sortable columns (click headers)
- Page size selector (10/25/50/100)
- Pagination controls
- Color-coded risk levels (🔴 Critical, 🟠 High, 🟡 Moderate, 🟢 Low)

**API**: `/api/v1/clinician/latest-patients` (paginated)

---

## Dashboard 3: Manager Dashboard

### Target Audience
Hospital administrators, finance directors, operations managers, C-suite

### Purpose
Financial impact analysis and business case for readmission prevention program.

### Implemented Sections

#### 1. Savings Summary
- **Key Savings Metrics**: Net savings, ROI, cost avoidance
- **Clinical Impact Metrics**: Readmissions prevented, intervention volume
- **Executive Summary Card**: Comprehensive financial overview with all calculations

**Key Metrics Displayed**:
- Net Program Value (cost of running program)
- Baseline Cost (cost of doing nothing)
- Net Savings (difference)
- Savings per $1 spent (ROI ratio)

**APIs**: `/api/v1/manager/savings-metrics`, `/api/v1/manager/impact-metrics`, `/api/v1/manager/executive-summary`

#### 2. Cost Breakdown
- **Financial Waterfall Chart**: Visual flow from costs to net savings
- **Cost Components Table**: Breakdown of all cost categories
- **Benefit Components Table**: Breakdown of all benefit categories

Shows detailed accounting of:
- True Positive value (prevented readmissions)
- False Positive cost (unnecessary interventions)
- False Negative cost (missed readmissions)
- True Negative value (no action needed)

**APIs**: `/api/v1/manager/waterfall-data`, `/api/v1/manager/cost-components`, `/api/v1/manager/benefit-components`

#### 3. Model Comparison
- **Model Performance & Financial Comparison Table**: Side-by-side comparison of all 3 models

Compares models on:
- Performance metrics (ROC-AUC, Recall, Precision)
- Financial metrics (Net Savings, ROI)
- Intervention volume
- Readmissions prevented
- Deployment recommendation

**API**: `/api/v1/manager/model-comparison`

---

## Cost Matrix Methodology

All financial calculations use this standard healthcare cost matrix:

| Outcome | Description | Cost/Benefit | Formula |
|---------|-------------|--------------|---------|
| **TP** | Prevented readmission | +$14,500 | $15K saved - $500 intervention |
| **FP** | Unnecessary intervention | -$500 | Intervention cost only |
| **TN** | No action needed | $0 | No cost, no benefit |
| **FN** | Missed readmission | -$15,000 | Full readmission cost |

### Key Financial Metrics

**Net Program Value** = (TP × $14.5K) + (FP × -$500) + (TN × $0) + (FN × -$15K)
- Total financial outcome of running the program

**Baseline Cost** = (TP + FN) × -$15K
- Total cost if we do nothing (all readmissions occur)

**Net Savings** = Baseline Cost - Net Program Value
- How much we save compared to doing nothing

**Savings per $1 Spent** = Net Savings ÷ Total Intervention Costs
- Return on intervention investment

---

## API Endpoints Summary

### Core Endpoints
- `GET /` - API information with dashboard links
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)

### Dashboard Routes
- `GET /dashboards/data-analyst` - Data Analyst Dashboard (HTML)
- `GET /dashboards/doctor` - Doctor Dashboard (HTML)
- `GET /dashboards/manager` - Manager Dashboard (HTML)

### Data Endpoints
- **Models**: `/api/v1/models`, `/api/v1/models/compare`
- **Phase 1**: `/api/v1/phase1/*` (data quality, features)
- **Phase 2**: `/api/v1/models/{method}/*` (performance metrics)
- **Phase 3**: `/api/v1/models/{method}/calibration-metrics`
- **Phase 4**: `/api/v1/models/phase4/threshold-curves`
- **Phase 5**: `/phase5/fairness-gaps/{method}`
- **Phase 6**: `/api/v1/models/phase6/final-evaluation`
- **Clinician**: `/api/v1/clinician/*` (patient list, risk factors)
- **Manager**: `/api/v1/manager/*` (financial metrics, ROI)

### Visualization Endpoints
- **ROC/PR Curves**: `/api/v1/visualizations/roc-curves-data`, `/api/v1/visualizations/pr-curves-data`
- **Confusion Matrix**: `/api/v1/visualizations/confusion-matrix-data`
- **Calibration**: `/api/v1/visualizations/calibration-curve-data/{method}`

---

## File Structure

```
phase-7-stakeholder-dashboards/
├── README.md
├── data-api/
│   ├── main.py                        # Main FastAPI app (1500+ lines)
│   ├── phase5_fairness_api.py         # Fairness endpoints
│   ├── visualization_generator.py     # Plot generation
│   ├── file_to_hf_repo_mapping.py    # HuggingFace mappings
│   ├── curve_data_for_plotly.json    # Cached curve data
│   ├── phase4_threshold_data.json    # Cached threshold data
│   ├── templates/
│   │   ├── base.html                 # Base template with navigation
│   │   ├── data_analyst_dashboard.html  (354 lines)
│   │   ├── doctor_dashboard.html        (163 lines)
│   │   └── manager_dashboard.html       (137 lines)
│   ├── static/
│   │   ├── css/                      # Stylesheets
│   │   └── js/                       # Dashboard JavaScript
│   │       ├── data_analyst.js
│   │       ├── doctor.js
│   │       └── manager.js
│   ├── routers/
│   │   ├── clinician.py              # Doctor endpoints
│   │   └── manager.py                # Manager endpoints
│   └── utilities/
│       ├── data_aggregator.py        # HuggingFace loader
│       └── cache_manager.py          # Caching
└── models/
    └── random_forest_rebuilt_metadata.json
```

---

## Design Philosophy

### Data Analyst Dashboard
- **Pipeline-based navigation**: Follows ML workflow (Phase 1 → Phase 6)
- **Technical depth**: Full metrics, curves, and comparisons
- **Model comparison**: Side-by-side analysis of all 3 models
- **Validation focus**: Verify each pipeline stage

### Doctor Dashboard
- **Clinical language**: Uses terms familiar to clinicians
- **Actionable insights**: Risk factors with recommendations
- **Patient-centered**: Individual patient risk scores and intervention levels
- **Simplicity**: Avoids technical jargon (ROC-AUC, Brier scores, etc.)

### Manager Dashboard
- **Financial focus**: ROI, savings, cost breakdown
- **Business impact**: Intervention volume, staffing needs
- **Decision support**: Model comparison for deployment choice
- **Executive summary**: High-level overview with detailed drilldown

---

## Key Features

### Interactive Visualizations
- Plotly.js charts with zoom, pan, hover tooltips
- Dynamic filtering and selection
- Color-coded metrics (green/yellow/red thresholds)
- Responsive design for different screen sizes

### Data Loading
- Automatic download from HuggingFace Hub
- File-based caching (curve_data_for_plotly.json, phase4_threshold_data.json)
- Lazy loading for large datasets
- Error handling with user-friendly messages

### Navigation
- Consistent header with dashboard links
- Section navigation within each dashboard
- Smooth scrolling to sections
- Active page highlighting

### Auto-Refresh
- Can be configured per dashboard
- Currently disabled (manual refresh)
- Future: Real-time updates when new data available

---

## Future Enhancements

### Planned Features (Not Yet Implemented)
- [ ] User authentication and role-based access control
- [ ] Real-time data refresh from production system
- [ ] Export to PDF functionality
- [ ] Alerts and notifications
- [ ] Custom date range selection
- [ ] Interactive what-if scenarios (threshold adjustment)
- [ ] Model retraining triggers
- [ ] Integration with EHR systems

### Technical Debt
- [ ] Add comprehensive unit tests
- [ ] Implement proper logging
- [ ] Add performance monitoring
- [ ] Optimize large data transfers
- [ ] Add database backend (currently file-based)
- [ ] Containerize with Docker
- [ ] Add CI/CD pipeline

---

## Summary

This implementation successfully delivers three role-specific dashboards that:
- ✅ Present ML pipeline insights (Phases 1-6) in accessible formats
- ✅ Serve distinct stakeholder needs with tailored content
- ✅ Provide interactive visualizations with Plotly.js
- ✅ Enable model comparison and deployment decisions
- ✅ Use production-ready web technologies (FastAPI, Jinja2)
- ✅ Load data automatically from HuggingFace Hub
- ✅ Maintain separation of concerns (frontend/backend)

**Key Insight**: Phase 7 transforms static pipeline outputs into actionable intelligence for different organizational roles, enabling data-driven decision-making for ML deployment.
