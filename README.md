# Hospital Readmission Risk - Operational Analytics

## Project Overview

This project focuses on predicting 30-day hospital readmissions for diabetic patients and designing care pathway interventions to reduce readmission rates. By leveraging machine learning and operational analytics, we aim to identify high-risk patients and provide actionable insights for healthcare providers.

## Business Goal

**Primary Objective**: Predict 30-day readmission risk and design care pathway interventions to improve patient outcomes and reduce healthcare costs.

**Key Outcomes**:
- Identify patients at high risk of readmission within 30 days
- Enable proactive interventions through early risk identification
- Optimize resource allocation and care planning
- Reduce hospital readmission rates and associated costs
- Improve overall patient care quality

## Dataset

### Source
- **Dataset**: Diabetes 130-US Hospitals for Years 1999-2008
- **Provider**: UCI Machine Learning Repository
- **URL**: [https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008)

### Dataset Description
The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes:
- Over 100,000 hospital admissions for diabetic patients
- 50+ features including demographics, diagnoses, medications, and procedures
- Patient readmission information (readmitted within 30 days, after 30 days, or not readmitted)

### Key Features
- **Patient Demographics**: Age, gender, race
- **Clinical Information**: Admission type, discharge disposition, time in hospital
- **Diagnoses**: Primary, secondary, and additional diagnoses codes
- **Medications**: Diabetes medications and dosage changes
- **Lab Results**: HbA1c test results
- **Previous Visits**: Number of prior emergency, inpatient, and outpatient visits

## Methodology

### Planned Approach

#### 1. Data Exploration & Preprocessing
- Exploratory data analysis (EDA) to understand data distributions
- Handle missing values and outliers
- Feature engineering and selection
- Data encoding and normalization

#### 2. Risk Modeling
- Develop predictive models for 30-day readmission risk
- Model candidates:
  - Logistic Regression (baseline)
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM)
  - Neural Networks
- Cross-validation and hyperparameter tuning

#### 3. Model Calibration
- Calibrate probability predictions for reliable risk scores
- Techniques: Platt scaling, isotonic regression
- Ensure predicted probabilities align with actual readmission rates

#### 4. Fairness Checks
- Evaluate model fairness across demographic groups
- Metrics: Demographic parity, equalized odds, equal opportunity
- Identify and mitigate potential biases in predictions
- Ensure equitable treatment across patient populations

#### 5. Intervention ROI Estimation
- Estimate cost-benefit of readmission prevention interventions
- Calculate potential cost savings from reduced readmissions
- Prioritize high-risk patients for intervention programs
- Develop resource allocation strategies

## Deliverables

### 1. Risk Prediction Model
- Trained and calibrated machine learning model
- Model documentation and performance metrics
- Feature importance analysis
- Fairness assessment report

### 2. Care Pathway Dashboard
- Interactive visualization dashboard for healthcare providers
- Real-time risk scoring for admitted patients
- Patient risk stratification (low, medium, high)
- Recommended intervention pathways
- Historical trends and performance metrics

### 3. Intervention ROI Analysis
- Cost-benefit analysis of intervention strategies
- Resource optimization recommendations
- Expected reduction in readmission rates
- Projected cost savings

### 4. Technical Documentation
- Data preprocessing pipeline
- Model training and evaluation procedures
- Deployment guidelines
- API documentation (if applicable)

## Project Structure

```
hospital-readmission-risk/
├── data/                   # Data storage (not tracked in git)
│   ├── raw/               # Original dataset
│   ├── processed/         # Cleaned and processed data
│   └── features/          # Engineered features
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/                   # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   ├── fairness/         # Fairness evaluation utilities
│   └── dashboard/        # Dashboard application
├── tests/                # Unit and integration tests
├── models/               # Saved model artifacts
├── reports/              # Generated reports and figures
├── dashboard/            # Dashboard deployment files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda for package management

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/auphong2707/hospital-readmission-risk.git
cd hospital-readmission-risk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Instructions to be added
```

## Usage

*To be implemented*

## Model Performance

*To be updated after model training*

## Fairness Metrics

*To be updated after fairness evaluation*

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Healthcare professionals who contributed to data collection
- Research community for established methodologies in healthcare analytics