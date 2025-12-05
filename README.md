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

#### 0. Problem Framing & Business Understanding

* 🔲 **Define Decision Framework (DOC Template)**
  * **Context**: High baseline readmission costs ($15k/event)
  * **Decision**: Which specific patients should be enrolled in the intervention program?
  * **Options**: Treat top risk %, treat random, or treat specific diagnosis groups
  * **Criteria**: Maximize ROI while maintaining operational capacity

* 🔲 **Design KPI Tree**
  * **North Star Metric**: Total Healthcare Cost Savings
  * **Drivers**: Readmission Rate, Intervention Success Rate, Cost per Patient
  * **Levers**: Targeted discharge planning, medication reconciliation, follow-up calls

* 🔲 **Establish Success Metrics & Guardrails**
  * **Primary Metric**: Reduction in 30-day readmission rate (Target: >3.3%)
  * **Guardrails**: Do not increase Length of Stay (LoS); maintain fairness across demographic groups
  * **Stakeholders**: Define RACI (Data Scientist, Clinical Staff, Hospital Admin)

#### 1. Data Exploration & Preprocessing
* [x] **Exploratory data analysis (EDA) to understand data distributions** *(COMPLETED)*
* [x] **Handle missing values and ensure data quality**
  * Replace `'?'` placeholders with `NaN`
  * Assess missingness patterns (MCAR/MAR/MNAR)
  * Apply median/mode or group-wise imputation based on feature type
  * Add binary `is_missing` indicators for clinically relevant variables
  * Validate value ranges, data types, and domain constraints
* [x] **Feature engineering and selection**
  * Create binary 30-day readmission target
  * Bucket age into clinically meaningful ordered categories
  * Aggregate diagnosis codes into higher-level groups
  * Build utilization features through counts and group-by statistics
  * Perform feature relevance checks using statistical and model-based methods
* [x] **Data encoding and normalization**
  * One-hot encode low-cardinality categorical variables
  * Target encode high-cardinality categorical variables using CV-safe encoding
  * Scale numerical features using StandardScaler/RobustScaler
  * Implement preprocessing inside sklearn Pipelines to avoid data leakage
* [x] **Demographics preservation (for Phase 5 fairness evaluation)**
  * Save original `race`, `gender`, `age` values BEFORE encoding
  * Export `*_demographics.csv` files aligned with train/val/test splits
  * Upload demographics to HuggingFace Hub for Phase 5 access

#### 2. Risk Modeling
- [x] **Develop predictive models for 30-day readmission risk**
- **Model candidates with specific configurations:**
  - **Logistic Regression (baseline)**: L1/L2 regularization, class weights for imbalance
  - **Random Forest**: 100-500 trees, max_depth tuning, feature importance extraction
  - **Gradient Boosting**: XGBoost/LightGBM with early stopping, DART/GOSS sampling
- [x] **Cross-validation and hyperparameter tuning**
  - **Stratified K-fold** (k=5) for readmission classes
  - **Grid/Random search** for hyperparameters
  - **Early stopping** to prevent overfitting

#### 3. Model Calibration
- [x] **Calibrate probability predictions using Platt Scaling**
- **Calibration technique:**
  - **Platt scaling**: Logistic regression transformation of predicted probabilities
  - Simple, interpretable, and widely used in healthcare applications
- [x] **Validation methods:**
  - **Reliability diagrams**: Plot predicted vs actual probabilities (before and after calibration)
  - **Brier score**: Measure calibration quality (target: < 0.15)
  - **Expected Calibration Error (ECE)**: Target < 0.05
  - **Hosmer-Lemeshow test**: Statistical calibration assessment (p-value > 0.05)
- [x] **Calibration quality assessment:**
  - **Calibration improvement**: Verify Brier score and ECE improved vs uncalibrated
  - **Calibration fairness**: Evaluate ECE and Brier score across demographic groups
  - Ensure ROC-AUC preserved (calibration shouldn't hurt discrimination)
  - **Output**: Platt-calibrated probabilities ready for threshold optimization

#### 4. Optimal Threshold & ROI Analysis
- [x] **Cost-sensitive threshold optimization:**
  - **Input**: Calibrated probabilities from Phase 3
  - **Cost matrix**: TP = +$14.5K, FP = -$500, TN = $0, FN = -$15K
  - **Optimal threshold**: Find threshold that maximizes expected value
  - **Expected value calculation**: EV = (TP × $14.5K) + (FP × -$500) + (FN × -$15K)
  - **Threshold search**: Test thresholds from 0.05 to 0.95 to find optimal point
  - **Break-even analysis**: Validate that intervention cost ($500) < expected benefit
  - **Sensitivity analysis**: Test ROI under different cost assumptions (conservative/aggressive)
- [x] **Risk category definition (derived from optimal threshold):**
  - **Low risk**: 0 to ~0.67 × optimal_threshold → Standard discharge
  - **Medium risk**: ~0.67×1.5 × optimal_threshold → Enhanced follow-up call
  - **High risk**: >1.5 × optimal_threshold → Intensive case management
  - **Validation**: Confirm actual readmission rates align with risk categories
  - **Resource allocation**: Calculate intervention volume per risk category
  - **Output**: Optimal threshold + risk category thresholds ready for fairness evaluation
  - **📝 Note**: Phase 4 saves results locally (`phase4_summary_for_phase5.json`) - Phase 5 requires this file to be present

#### 5. Fairness Evaluation
- 🔲 **Threshold fairness evaluation:**
  - **Input**: Optimal threshold and risk categories from Phase 4 + demographics from Phase 1
  - **Protected attributes**: Race, gender, age groups
  - **Performance by group**: Calculate TPR, FPR, precision, recall per demographic group at optimal threshold
  - **Threshold fairness metrics**: Demographic parity, equalized odds, equal opportunity
  - **Risk category distribution**: Check if interventions allocated fairly across groups
  - **Statistical testing**: Chi-square tests for significant group differences
  - **Note**: Demographics files (`*_demographics.csv`) now exported by Phase 1 preprocessing
  - **📝 Note**: Phase 5 requires Phase 4 results (`phase4_summary_for_phase5.json`) to be present locally
- 🔲 **Bias detection and analysis:**
  - **Group-specific decision thresholds**: Analyze if adjustments needed per group to equalize TPR/FPR
  - **Fairness-ROI trade-off**: Document impact of potential fairness adjustments on overall ROI
  - **Recommendations**: Determine if global threshold is acceptable or if bias mitigation required
  - **Output**: Fairness assessment report for deployment decision-making

#### 6. Fairness Mitigation & Bias Correction
- 🔲 **Calculate group-specific thresholds using equalized odds:**
  - **Input**: Phase 5 fairness violations and group metrics (`phase5_summary_for_phase6.json`)
  - **Strategy**: Equalized odds (minimize both TPR and FPR gaps across demographic groups)
  - **Threshold search**: Wide range [0.01, 0.99] with 0.01 step size (99 thresholds tested per group)
  - **Optimization**: Find threshold per group that minimizes combined TPR+FPR gap from target
  - **Trade-off analysis**: Measure impact on overall performance, ROI, and intervention volume
  - **Validation**: Test thresholds on test data, verify fairness improvements vs baseline
- 🔲 **Clinical and ethics approval:**
  - **Presentation**: Show fairness violations, proposed group-specific thresholds, performance/fairness trade-offs
  - **Decision**: Clinical team approves threshold strategy (global vs group-specific)
  - **Documentation**: Record approval decision, rationale, affected groups, and audit trail
  - **Output**: Fairness mitigation config with group-specific thresholds (`.json`) + clinical approval document (`.md`)

#### 7. Results Collection & Publication
- [ ] **Collect all phase outputs:**
  - **Phase 1**: Preprocessing metadata, data splits, demographics files
  - **Phase 2**: 3 trained models (Logistic Regression, Random Forest, Gradient Boosting), metrics, visualizations
  - **Phase 3**: Calibrated model, calibrator, calibration metrics, reliability diagrams
  - **Phase 4**: Optimal thresholds, ROI analysis, 8 threshold optimization plots
  - **Phase 5**: Fairness report, group metrics, ~21 fairness visualizations, statistical tests
  - **Phase 6**: Group-specific thresholds (if applied), mitigation impact, 5 mitigation plots
- [ ] **Create aggregated summary:**
  - Combine all metrics into single JSON file
  - Create comprehensive model card with all findings
  - Generate summary visualizations (performance, calibration, fairness, ROI)
  - Document complete project timeline and results
- [ ] **Upload to HuggingFace Hub:**
  - Upload all model artifacts (models, calibrator, thresholds)
  - Upload complete results package (metrics, visualizations, model card)
  - Create public repository for reproducibility and transparency
  - Version all artifacts for audit trail and future reference

## Deliverables

### 1. Risk Prediction Model
- Trained and calibrated Gradient Boosting model (primary)
- Alternative models: Logistic Regression, Random Forest
- Platt scaling calibrator for probability calibration
- Optimal decision thresholds (global + group-specific if needed)

### 2. Comprehensive Analysis Reports
- Model performance metrics (ROC-AUC, precision, recall, calibration quality)
- Cost-benefit analysis and ROI projections
- Fairness evaluation across demographic groups
- Bias mitigation strategies (if fairness violations detected)

### 3. Complete Model Card
- Model architecture and training details
- Performance benchmarks and calibration metrics
- Fairness assessment and mitigation (if applied)
- Limitations, ethical considerations, and monitoring requirements
- Intended use cases and out-of-scope scenarios

### 4. HuggingFace Repository
- Public repository with all model artifacts
- Complete results package (89-91 items)
- 62-63 visualizations across all phases
- Versioned for reproducibility and audit trail

## Project Structure

```
hospital-readmission-risk/
|-- data/                                          # Data storage
|   |-- diabetic_data.csv                         # Original dataset (101,766 patients)
|   |-- IDS_mapping.csv                           # ID mappings
|   +-- processed/                                # [x] Processed data (generated by Phase 1)
|       |-- preprocessed_hospital_data.csv        # Complete processed dataset
|       |-- features.csv                          # Features only (113 features)
|       |-- target.csv                            # Target variable
|       |-- preprocessing_metadata.txt            # Processing details
|       +-- splits/                               # Train/val/test splits
|           |-- train.csv                         # Training features + target (73,526 samples)
|           |-- train_demographics.csv            # Training demographics (race, gender, age)
|           |-- validation.csv                    # Validation features + target (12,975 samples)
|           |-- validation_demographics.csv       # Validation demographics
|           |-- test.csv                          # Test features + target (15,265 samples)
|           |-- test_demographics.csv             # Test demographics (for Phase 5)
|           +-- split_info.txt                    # Split details
|-- phase-1-data-explore-preprocessing/           # [x] Phase 1: EDA & Preprocessing
|   |-- eda.ipynb                                 # Exploratory data analysis
|-- phase-5-fairness-evaluation/                  # [x] Phase 5: Fairness Evaluation
|   |-- evaluate_fairness_gradient_boosting.py    # Fairness evaluation script
|   |-- utilities.py                              # Fairness metrics utilities
|   |-- outputs/                                  # Phase 5 results
|   |   |-- fairness_report.json                  # Comprehensive fairness metrics
|   |   |-- group_metrics_*.csv                   # Performance by demographic group
|   |   |-- statistical_tests.json                # Significance tests
|   |   |-- risk_categories_*.csv                 # Risk distribution by group
|   |   +-- visualizations/                       # Fairness assessment plots
|   +-- README.md                                 # Phase 5 documentation
|-- phase-6-fairness-mitigation/                  # 🔲 Phase 6: Bias Correction
|   |-- calculate_group_thresholds.py             # Compute group-specific thresholds (TBD)
|   |-- evaluate_mitigation_impact.py             # Test fairness improvements (TBD)
|   |-- utilities.py                              # Mitigation utilities (TBD)
|   |-- outputs/                                  # Phase 6 results (TBD)
|   |   |-- group_thresholds.json                 # Group-specific thresholds
|   |   |-- mitigation_impact.json                # Performance/fairness trade-offs
|   |   |-- clinical_approval.md                  # Clinical review documentation
|   |   +-- visualizations/                       # Before/after fairness plots
|   +-- README.md                                 # Phase 6 documentation (TBD)
|- phase-7-results-collection-publication/       # [ ] Phase 7: Results Collection & Publication
|   |-- collect_and_publish.sh                    # Single bash script to collect & upload (TBD)
|   |-- outputs/                                  # Collected results (TBD)
|   |   |-- aggregated_results.json               # All metrics combined
|   |   |-- model_card.md                         # Complete model documentation
|   |   +-- collection_summary.txt                # File inventory
|   +-- README.md                                 # Phase 7 documentation
|-- requirements.txt                              # Python dependencies
+-- README.md                                     # This file (project overview)
|   |-- utilities.py                              # Training utilities
|   +-- README.md                                 # Phase 2 documentation
|-- phase-3-model-calibration/                    # [x] Phase 3: Probability Calibration
|   |-- calibrate_gradient_boosting.py            # Platt scaling calibration
|   |-- utilities.py                              # Calibration utilities
|   +-- README.md                                 # Phase 3 documentation
|-- phase-4-optimal-threshold-ROI-analysis/       # [x] Phase 4: Threshold Optimization
|   |-- optimize_threshold_gradient_boosting.py   # Cost-sensitive threshold search
|   |-- utilities.py                              # ROI analysis utilities
|   |-- outputs/                                  # Phase 4 results
|   |   |-- phase4_summary_for_phase5.json        # Summary for Phase 5
|   |   |-- optimal_thresholds.json               # Optimal thresholds
|   |   |-- roi_metrics.json                      # ROI analysis
|   |   +-- threshold_results.csv                 # Threshold sweep results
|   +-- README.md                                 # Phase 4 documentation
|-- phase-5-fairness-evaluation-deployment-readiness/  # 🔲 Phase 5: Fairness & Deployment
|   |-- evaluate_fairness_gradient_boosting.py    # Fairness evaluation script (TBD)
|   |-- utilities.py                              # Fairness metrics utilities (TBD)
|   |-- PHASE1_UPDATES.md                         # Phase 1 demographics changes
|   +-- README.md                                 # Phase 5 documentation
|-- requirements.txt                              # Python dependencies
+-- README.md                                     # This file (project overview)
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

For Phase 2 modeling, Phase 1's preprocessed splits are available on HuggingFace Hub:

```python
from phase-2-risk-modeling.utilities import load_phase1_splits

# Automatically downloads Phase 1 splits from auphong2707/hospital-readmission-risk-data
X_train, X_val, X_test, y_train, y_val, y_test = load_phase1_splits()

# Data specs: Train: 73,526 | Val: 12,975 | Test: 15,265 (113 features)
# Single source of truth for all phases (2-5)
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
   - Train/validation/test splits in `splits/` folder:
     - `train.csv`, `validation.csv`, `test.csv` (features + target)
     - `train_demographics.csv`, `validation_demographics.csv`, `test_demographics.csv` (race, gender, age) ← **NEW for Phase 5**
     - `split_info.txt` (split details)
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