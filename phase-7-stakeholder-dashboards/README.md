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
