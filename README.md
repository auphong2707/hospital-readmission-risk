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
* ✅ **Exploratory data analysis (EDA) to understand data distributions** *(COMPLETED)*
* ✅ **Handle missing values and ensure data quality**
  * Replace `'?'` placeholders with `NaN`
  * Assess missingness patterns (MCAR/MAR/MNAR)
  * Apply median/mode or group-wise imputation based on feature type
  * Add binary `is_missing` indicators for clinically relevant variables
  * Validate value ranges, data types, and domain constraints
* ✅ **Feature engineering and selection**
  * Create binary 30-day readmission target
  * Bucket age into clinically meaningful ordered categories
  * Aggregate diagnosis codes into higher-level groups
  * Build utilization features through counts and group-by statistics
  * Perform feature relevance checks using statistical and model-based methods
* ✅ **Data encoding and normalization**
  * One-hot encode low-cardinality categorical variables
  * Target encode high-cardinality categorical variables using CV-safe encoding
  * Scale numerical features using StandardScaler/RobustScaler
  * Implement preprocessing inside sklearn Pipelines to avoid data leakage

#### 2. Risk Modeling
- ✅ **Develop predictive models for 30-day readmission risk**
- **Model candidates with specific configurations:**
  - **Logistic Regression (baseline)**: L1/L2 regularization, class weights for imbalance
  - **Random Forest**: 100-500 trees, max_depth tuning, feature importance extraction
  - **Gradient Boosting**: XGBoost/LightGBM with early stopping, DART/GOSS sampling
- ✅ **Cross-validation and hyperparameter tuning**
  - **Stratified K-fold** (k=5) for readmission classes
  - **Grid/Random search** for hyperparameters
  - **Early stopping** to prevent overfitting

#### 3. Model Calibration
- 🔲 **Calibrate probability predictions for reliable risk scores**
- **Calibration techniques:**
  - **Platt scaling**: Sigmoid function fitting for probability calibration
  - **Isotonic regression**: Non-parametric calibration for non-linear relationships
  - **Group-specific calibration**: Separate calibration by demographic groups
- 🔲 **Validation methods:**
  - **Reliability diagrams**: Plot predicted vs actual probabilities
  - **Brier score**: Measure calibration quality
  - **Hosmer-Lemeshow test**: Statistical calibration assessment
- 🔲 **Clinical risk score mapping:**
  - Map probabilities to risk categories (Low: 0-5%, Medium: 5-15%, High: 15%+)
  - Validate risk score interpretability with clinical experts

#### 4. Fairness Checks
- 🔲 **Evaluate model fairness across demographic groups**
  - **Protected attributes**: Race, gender, age groups (based on EDA findings)
  - **Subgroup analysis**: Separate performance metrics by demographic groups
- **Fairness metrics implementation:**
  - **Demographic parity**: Equal positive prediction rates across groups
  - **Equalized odds**: Equal TPR and FPR across groups  
  - **Equal opportunity**: Equal TPR for positive class across groups
  - **Calibration fairness**: Equal calibration quality across groups
- 🔲 **Bias identification and mitigation:**
  - **Statistical significance testing**: Chi-square tests for group differences
  - **Fairness constraints**: Add fairness penalties to loss functions
  - **Post-processing**: Threshold optimization by group
  - **Re-sampling**: Balanced training data across demographics
- 🔲 **Healthcare compliance:**
  - **Documentation**: Bias testing reports for regulatory compliance
  - **Monitoring**: Ongoing fairness tracking in production

#### 5. Intervention ROI Estimation
- 🔲 **Cost-benefit analysis framework:**
  - **Baseline costs**: $15K average per 30-day readmission (from EDA)
  - **Intervention costs**: $500 per patient program cost estimate
  - **Break-even analysis**: Need >3.3% readmission reduction for positive ROI
- 🔲 **High-risk patient prioritization:**
  - **Risk stratification**: Top 20-30% risk patients for maximum ROI
  - **Resource allocation**: Budget optimization across risk categories
  - **Intervention targeting**: Focus on modifiable risk factors
- 🔲 **ROI scenarios and projections:**
  - **Conservative (5-10% reduction)**: Basic care coordination
  - **Moderate (10-15% reduction)**: Enhanced discharge planning
  - **Aggressive (15-25% reduction)**: Comprehensive care pathways
- 🔲 **Business case development:**
  - **Implementation timeline**: Phased rollout plan (pilot → full deployment)
  - **Success metrics**: Clinical, operational, and financial KPIs
  - **Stakeholder presentation**: Executive summary with ROI projections

### Immediate Next Steps (Priority Order)

#### **Phase 1: Data Preprocessing (Week 1-2)**
1. **Create preprocessing pipeline** (`notebooks/02_preprocessing.ipynb`)
   - Missing value imputation strategy
   - Outlier treatment implementation
   - Feature engineering pipeline
2. **Generate clean dataset** with engineered features
3. **Create train/validation/test splits** (temporal: 70%/15%/15%)

#### **Phase 2: Baseline Modeling (Week 3)**
1. **Implement Logistic Regression baseline** (`notebooks/03_modeling.ipynb`)
2. **Establish evaluation framework** (metrics, cross-validation)
3. **Initial fairness assessment** across demographic groups

#### **Phase 3: Advanced Modeling (Week 4-5)**
1. **Train ensemble models** (Random Forest, XGBoost, LightGBM)
2. **Hyperparameter optimization** with cross-validation
3. **Model comparison and selection**

#### **Phase 4: Calibration & Fairness (Week 6)**
1. **Implement calibration techniques** (Platt scaling, isotonic regression)
2. **Comprehensive fairness evaluation** with bias mitigation
3. **Final model validation** and documentation

#### **Phase 5: Business Impact (Week 7-8)**
1. **ROI analysis implementation** with scenario modeling
2. **Dashboard prototype** for risk scoring and intervention guidance
3. **Final presentation** and deployment strategy

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
├── data/                   # Data storage
│   ├── diabetic_data.csv  # Original dataset
│   ├── IDS_mapping.csv    # ID mappings
│   └── processed/         # ✅ Cleaned and processed data (generated by preprocessing script)
│       ├── preprocessed_hospital_data.csv  # Complete processed dataset
│       ├── features.csv   # Features only
│       ├── target.csv     # Target variable
│       └── preprocessing_metadata.txt      # Processing details
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb       # ✅ Completed
│   ├── 02_preprocessing.ipynb  # 🔲 Next
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
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation by running the preprocessing pipeline
python phase-1-data-explore-preprocessing/simple_preprocessing.py
```

**Note**: The dataset files (`diabetic_data.csv`, `IDS_mapping.csv`) should be placed in the `data/` folder before running the preprocessing script.

## EDA Key Findings (Completed)

### **Dataset Overview**
- **101,766 patients** with 50 clinical features
- **11.2%** 30-day readmission rate (primary target)
- **46.1%** patients have some form of readmission

### **Data Quality Insights**
- **Missing data challenges**: A1C (83% missing), weight (97% missing)
- **'?' indicators**: Race (2.23%), medical specialty (49%), payer code (40%)
- **No duplicate records** detected
- **Class imbalance**: 30-day readmission is minority class

### **Clinical Risk Factors**
- **Length of stay**: Risk increases significantly after 4+ days
- **Medication complexity**: Higher medication counts correlate with readmission
- **Prior utilization**: Emergency visits and inpatient stays predict readmission
- **Demographics**: Significant differences across race, age, and gender groups

### **Business Impact Potential**
- **ROI scenarios**: 10-20% reduction rates show strong positive ROI
- **Cost framework**: ~$15K per readmission vs ~$500 intervention cost
- **High-risk targeting**: Top 20-30% risk patients offer maximum ROI
- **Break-even**: Need >3.3% readmission reduction for positive returns

## Usage

### Option 1: Load Data from HuggingFace Hub (Recommended)

For Phase 2 modeling, preprocessed data is available on HuggingFace Hub:

```python
from phase-2-risk-modeling.utilities import load_data

# Automatically downloads from auphong2707/hospital-readmission-risk-data
X, y = load_data(from_huggingface=True)

# Data specs: 101,766 samples, 113 features
# Class distribution: ~11.2% readmission rate
```

Training scripts automatically use HuggingFace data:
```bash
# No preprocessing needed - data loaded automatically
python phase-2-risk-modeling/train_gradient_boosting.py
python phase-2-risk-modeling/train_logistic_regression.py
python phase-2-risk-modeling/train_random_forest.py
```

### Option 2: Run Local Data Preprocessing Pipeline

To preprocess data locally and upload to HuggingFace:

```bash
# Using Python virtual environment (recommended)
python phase-1-data-explore-preprocessing/simple_preprocessing.py

# Or with full path to virtual environment Python
.venv/Scripts/python.exe phase-1-data-explore-preprocessing/simple_preprocessing.py
```

This preprocessing script will:
- Load data from `./data/diabetic_data.csv`
- Apply comprehensive preprocessing covering 100% of README requirements
- Create train/validation/test splits (70%/15%/15%)
- Upload to HuggingFace Hub
- Output: 101,766 samples with 113 engineered features

**Prerequisites**: Ensure pandas, numpy, scikit-learn, and python-dotenv are installed.

**Data Structure**: The script expects the following data files in the `data/` folder:
- `data/diabetic_data.csv` - Main dataset (automatically loaded by the script)
- `data/IDS_mapping.csv` - ID mappings (reference file)

**Expected Output**: When run successfully, the script will:
1. Load 101,766 patients with 50 features from `data/diabetic_data.csv`
2. Apply comprehensive preprocessing (missing value handling, outlier treatment, feature engineering)
3. Generate 101,766 samples with 113 engineered features
4. **Save processed data to `data/processed/` folder**:
   - `preprocessed_hospital_data.csv` - Complete processed dataset
   - `features.csv` - Features only (for ML workflows)
   - `target.csv` - Target variable only
   - `preprocessing_metadata.txt` - Detailed preprocessing information
   - Train/validation/test splits in `splits/` folder
5. **Upload to HuggingFace Hub** (if `.env` configured with `HF_TOKEN`)
6. Display detailed preprocessing steps and final dataset statistics

### Using Locally Processed Data

After running the preprocessing script, you can load the processed data:

```python
import pandas as pd

# Load the complete processed dataset
data = pd.read_csv('./data/processed/preprocessed_hospital_data.csv')

# Or load features and target separately
X = pd.read_csv('./data/processed/features.csv')
y = pd.read_csv('./data/processed/target.csv')['target']

# View preprocessing details
with open('./data/processed/preprocessing_metadata.txt', 'r') as f:
    print(f.read())
```

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