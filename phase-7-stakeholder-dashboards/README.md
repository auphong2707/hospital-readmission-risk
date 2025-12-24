# Phase 7: Stakeholder Dashboards

Interactive web dashboards for Hospital Readmission Risk Prediction System.

## Architecture

- **Frontend**: HTML/CSS + Plotly.js (interactive visualizations)
- **Backend**: FastAPI (REST API + template rendering)
- **Deployment**: Standalone Python application

## Quick Start

### Prerequisites

- Python 3.8+ installed
- Phase 1-6 outputs uploaded to HuggingFace
- `file_to_hf_repo_mapping.py` in project root

### Install Dependencies

```bash
# From project root
pip install -r requirements.txt
```

### Start Dashboard

```bash
# From project root
start.bat

# Or manually:
cd phase-7-stakeholder-dashboards\data-api
python main.py
```

### Access Dashboards

- **Data Analyst Dashboard**: http://localhost:8000/dashboards/data-analyst
- **Doctor Dashboard**: http://localhost:8000/dashboards/doctor
- **FastAPI Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Available Dashboards

### 1. Data Analyst Dashboard
**Target Audience**: Data scientists, ML engineers, technical analysts

**Key Features**:
- **Phase 1**: Class distribution and dataset split visualizations
- **Phase 2**: ROC/PR curves comparison across all 3 models
- **Phase 2**: Model performance metrics (ROC-AUC, Precision, Recall, F1)
- **Phase 3**: Calibration diagrams (before/after) for each model
- **Phase 4**: Cost-benefit analysis and threshold optimization
- **Phase 5**: Fairness assessment across demographics
- **Phase 6**: Final system evaluation and deployment readiness
- **Recommended Model**: Quick insights with best model selection

**Interactive Features**:
- Plotly charts with zoom, pan, and hover tooltips
- Color-coded metrics (green/yellow/red thresholds)
- Auto-refresh every 5 minutes

### 2. Doctor Dashboard (Clinician-Focused)
**Target Audience**: Doctors, nurses, clinical staff

**Key Features**:
- **Model Selector**: Choose between Gradient Boosting, Random Forest, or Logistic Regression
- **Top 20 Risk Factors**: Feature importance with clinical interpretations
- **Performance Metrics**: Clinical terms (Sensitivity, Specificity, PPV, NPV)
- **Fairness by Demographics**: Performance across race, gender, and age groups
- **Overall Fairness Assessment**: Demographic parity, equal opportunity, equalized odds

**Interactive Features**:
- Model switching without page reload
- Sortable tables
- Color-coded fairness metrics
- Auto-refresh every 5 minutes

## API Endpoints

### Dashboard Routes
- `GET /dashboards/data-analyst` - Data Analyst Dashboard (HTML)
- `GET /dashboards/doctor` - Doctor Dashboard (HTML)

### Health & Status
- `GET /` - API information with dashboard links
- `GET /health` - API health check
- `GET /api/v1/models` - List available models

### Clinician Dashboard Endpoints
- `GET /api/v1/models/{method}/risk-factors` - Top risk factors with importance
- `GET /api/v1/models/{method}/clinical-patterns` - Clinical patterns by demographics
- `GET /api/v1/models/{method}/fairness-summary` - Fairness assessment summary
- `GET /api/v1/models/{method}/performance-clinical` - Performance in clinical terms

### Parameters
- `method`: One of `gradient_boosting`, `random_forest`, `logistic_regression`

## ROI Calculation Formulas (Phase 6)

All dashboards use the **Cost Comparison Framework** for ROI calculations:

### Core Metrics

#### 1. **Baseline Cost** (Do Nothing Scenario)
```
Baseline Cost = (TP + FN) × $15,000
```
- **Meaning**: Total cost if we never intervene and all actual positives become readmissions
- **Components**: All patients who would be readmitted (both caught and missed)
- **Unit Cost**: $15,000 per readmission

#### 2. **Model Cost** (AI-Driven Intervention Scenario)
```
Model Cost = (TP + FP) × $500 + (FN) × $15,000
```
- **Meaning**: Total cost when using the AI model to guide interventions
- **Components**:
  - **(TP + FP) × $500**: Cost of interventions on all flagged patients (both correct and incorrect flags)
  - **(FN) × $15,000**: Cost of readmissions we missed
- **Unit Costs**: $500 per intervention, $15,000 per readmission

#### 3. **Cost Savings**
```
Cost Savings = Baseline Cost - Model Cost
```
**Simplified Formula:**
```
Cost Savings = (TP × $14,500) - (FP × $500)
```
- **Meaning**: Net financial benefit of using the AI model vs doing nothing
- **Interpretation**: 
  - Each **True Positive (TP)** saves $14,500 (avoided $15K readmission - paid $500 intervention)
  - Each **False Positive (FP)** costs $500 (unnecessary intervention)

#### 4. **Intervention Costs**
```
Intervention Costs = (TP + FP) × $500
```
- **Meaning**: Total amount spent on preventive interventions
- **Components**: All patients flagged by the model (both successful and unnecessary)
- **Use**: Budget planning for intervention programs

#### 5. **Savings per $1 Spent** (Savings Ratio / ROI)
```
Savings Ratio = Cost Savings ÷ Intervention Costs
```
- **Meaning**: Return on investment for every dollar spent on interventions
- **Example**: Savings Ratio of $24 means every $1 spent on interventions saves $24 in avoided readmissions
- **Interpretation**: Higher is better; measures intervention efficiency

#### 6. **Intervention Rate**
```
Intervention Rate = (TP + FP) ÷ Total Patients × 100%
```
- **Meaning**: Percentage of patients flagged for intervention
- **Use**: Resource planning (staffing, capacity, workload estimation)
- **Example**: 12.9% means ~13 out of every 100 patients need intervention

#### 7. **Prevented Readmissions Value**
```
TP Value = TP × $14,500
```
- **Meaning**: Gross value generated from successfully preventing readmissions
- **Note**: Net value per prevented readmission (saved $15K - paid $500)

#### 8. **Unnecessary Interventions Cost**
```
FP Cost = FP × $500
```
- **Meaning**: Money spent on patients who wouldn't have been readmitted anyway
- **Interpretation**: Model precision affects this cost

#### 9. **Missed Readmissions Cost**
```
FN Cost = FN × $15,000
```
- **Meaning**: Cost of readmissions the model failed to prevent
- **Interpretation**: Model recall affects this cost

### Cost Matrix (Per Patient)

| Outcome | Clinical Meaning | Financial Impact |
|---------|-----------------|------------------|
| **TP** (True Positive) | Correctly flagged, prevented readmission | **+$14,500** (saved $15K - paid $500) |
| **FP** (False Positive) | Incorrectly flagged, unnecessary intervention | **-$500** (wasted intervention) |
| **TN** (True Negative) | Correctly not flagged, no action needed | **$0** (no cost, no benefit) |
| **FN** (False Negative) | Missed readmission, patient readmitted | **-$15,000** (failed to prevent) |

### Example Calculation

Given a model with:
- TP = 1,000
- FP = 200
- TN = 8,000
- FN = 100
- Total Patients = 9,300

**Calculations:**
```
Baseline Cost = (1,000 + 100) × $15,000 = $16,500,000
Model Cost = (1,000 + 200) × $500 + (100 × $15,000) = $600,000 + $1,500,000 = $2,100,000
Cost Savings = $16,500,000 - $2,100,000 = $14,400,000
Intervention Costs = (1,000 + 200) × $500 = $600,000
Savings Ratio = $14,400,000 ÷ $600,000 = $24.00
Intervention Rate = (1,000 + 200) ÷ 9,300 × 100 = 12.9%
```

**Interpretation:** 
- This model saves **$14.4 million** compared to doing nothing
- For every **$1 spent** on interventions, it saves **$24** in avoided readmissions
- It requires intervening on **12.9%** of all patients
- It successfully prevents **1,000** readmissions but misses **100**

## Development

### Local Development (without Docker)

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start FastAPI
cd data-api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 3: Start Grafana (local installation)
grafana-server --config grafana.ini
```

### Running Tests

```bash
cd data-api
pytest tests/
```

## Configuration

### Environment Variables

Create `.env` file in `data-api/`:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Cache TTL (seconds)
CACHE_TTL=3600

# HuggingFace
HF_HOME=/cache/huggingface
```

### Grafana Data Source

Configured automatically via provisioning at:
`grafana/provisioning/datasources/datasource.yml`

### Dashboards

Dashboard definitions at:
`grafana/provisioning/dashboards/*.json`

## Troubleshooting

### Grafana not connecting to API
```bash
# Check if API is running
curl http://localhost:8000/health

# Check network connectivity
docker-compose exec grafana ping data-api
```

### Data not loading from HuggingFace
```bash
# Check API logs
docker-compose logs data-api

# Verify HuggingFace cache
ls ~/.cache/huggingface
```

### Clear cache
```bash
# Clear Redis cache
docker-compose exec redis redis-cli FLUSHALL

# Restart services
docker-compose restart
```

## License

See project root LICENSE file.
