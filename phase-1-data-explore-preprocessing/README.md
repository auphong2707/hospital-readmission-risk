# Hospital Readmission Risk - Preprocessing Pipeline Documentation

## Overview

This document provides a comprehensive explanation of the preprocessing pipeline implemented in `simple_preprocessing.py`. The pipeline transforms raw hospital readmission data into a machine learning-ready dataset while addressing data quality issues, engineering meaningful features, and preventing data leakage.

---

## Table of Contents

1. [Pipeline Architecture](#pipeline-architecture)
2. [Data Loading](#1-data-loading)
3. [Missing Value Handling](#2-missing-value-handling)
4. [Data Quality Validation](#3-data-quality-validation)
5. [Outlier Treatment](#4-outlier-treatment)
6. [Target Variable Creation](#5-target-variable-creation)
7. [Feature Engineering](#6-feature-engineering)
8. [Demographics Storage](#7-demographics-storage)
9. [Feature Encoding](#8-feature-encoding)
10. [Column Name Sanitization](#9-column-name-sanitization)
11. [Train/Validation/Test Splitting](#10-trainvalidationtest-splitting)
12. [Feature Scaling](#11-feature-scaling)
13. [Data Export](#12-data-export)
14. [Usage Examples](#usage-examples)

---

## Pipeline Architecture

The `CompletePreprocessor` class implements a sequential preprocessing pipeline that covers 100% of the project requirements. The pipeline is designed to:

- **Prevent data leakage**: Scaling fitted only on training data
- **Maintain reproducibility**: Fixed random seed (42)
- **Support fairness analysis**: Demographics preserved for Phase 5
- **Enable deployment**: Scaler saved for production use

**Key Design Principles:**
- Separate preprocessing logic into modular, testable methods
- Preserve data lineage through metadata files
- Support both StandardScaler and RobustScaler
- Create reproducible train/validation/test splits

---

## 1. Data Loading

### Method: `load_data()`

**Motivation:**
- Load the UCI Diabetes 130-US Hospitals dataset
- Establish baseline dataset characteristics
- Enable initial data exploration

**Process:**
```python
data = pd.read_csv(data_path)
```

**Results:**
- Original dataset: **101,766 patients** with **50+ features**
- Includes demographic, clinical, and administrative data
- Raw format requires extensive preprocessing before modeling

---

## 2. Missing Value Handling

### Method: `handle_missing_values()`

**Motivation:**
Missing data can bias models and reduce predictive power. Different missingness patterns require different strategies:
- **MCAR (Missing Completely At Random)**: Administrative gaps (e.g., A1C tests, weight)
- **MAR (Missing At Random)**: Depends on other variables (e.g., medical specialty depends on admission type)

### Techniques Applied:

#### 2.1 Missing Indicators
**Purpose:** Capture whether missingness itself is predictive

```python
clinically_relevant = ['A1Cresult', 'weight', 'race', 'medical_specialty', 'payer_code']
data[f"{col}_is_missing"] = data[col].isnull().astype(int)
```

**Motivation:** 
- Missing A1C test may indicate less severe diabetes
- Missing weight may correlate with emergency admissions
- Preserves information before imputation

**Results:**
- Creates binary features (0/1) indicating missing status
- Enables model to learn from missingness patterns
- Example: `A1Cresult_is_missing` helps identify undertested patients

#### 2.2 Group-Wise Imputation (MAR Strategy)
**Purpose:** Fill missing values using context from related variables

```python
# Fill medical_specialty by admission_type_id mode
for admission_type in data['admission_type_id'].unique():
    mask = (data['admission_type_id'] == admission_type) & data['medical_specialty'].isnull()
    mode_val = data[data['admission_type_id'] == admission_type]['medical_specialty'].mode()
    data.loc[mask, 'medical_specialty'] = fill_val
```

**Motivation:**
- Emergency admissions → likely general medicine specialty
- Elective admissions → likely specific specialist
- Preserves data distribution patterns

**Results:**
- More accurate imputation than global mode
- Reduces bias from naive filling
- Maintains relationship between admission type and specialty

#### 2.3 Simple Imputation
**Purpose:** Fill remaining missing values with central tendency

- **Categorical features**: Mode (most frequent value)
- **Numerical features**: Median (robust to outliers)

**Motivation:**
- Mode preserves categorical distribution
- Median less sensitive to outliers than mean
- Simple baseline for MCAR features

**Results:**
- Zero remaining missing values
- Dataset ready for algorithm training
- Conservative approach minimizes introduced bias

---

## 3. Data Quality Validation

### Method: `validate_data_quality()`

**Motivation:**
Data entry errors, sensor malfunctions, and system glitches can introduce invalid values that compromise model reliability.

### Validation Checks:

#### 3.1 Numerical Range Validation
**Purpose:** Ensure values fall within clinically plausible ranges

```python
numerical_validations = {
    'time_in_hospital': (1, 14),      # Hospital stays typically 1-14 days
    'num_lab_procedures': (0, 150),   # Reasonable lab test range
    'num_procedures': (0, 10),        # Surgical procedures cap
    'num_medications': (0, 100),      # Medication count limit
    'number_diagnoses': (1, 16)       # Diagnosis code range
}
```

**Motivation:**
- `time_in_hospital = 0`: Data entry error
- `num_medications = 500`: System glitch
- Extreme values can dominate model training

**Action:** Values clipped to valid range boundaries

**Results:**
- Removes physically impossible values
- Preserves data integrity
- Example: 500 medications clipped to 100

#### 3.2 Data Type Validation
**Purpose:** Verify expected data types match actual types

```python
expected_types = {
    'time_in_hospital': 'numeric',
    'race': 'categorical',
    'gender': 'categorical'
}
```

**Motivation:**
- Type mismatches indicate parsing errors
- Categorical-as-numeric breaks encoding
- Ensures correct downstream processing

**Results:**
- Validates schema consistency
- Catches import/export errors
- Logs discrepancies for investigation

#### 3.3 Domain Constraint Validation
**Purpose:** Enforce business logic and domain rules

```python
valid_genders = ['Male', 'Female', 'Unknown/Invalid']
invalid_genders = ~data['gender'].isin(valid_genders)
data.loc[invalid_genders, 'gender'] = 'Unknown/Invalid'
```

**Motivation:**
- Gender must be one of predefined categories
- Invalid codes replaced with "Unknown/Invalid"
- Maintains referential integrity

**Results:**
- All values conform to domain constraints
- Invalid entries safely handled
- Dataset passes integrity checks

---

## 4. Outlier Treatment

### Method: `treat_outliers()`

**Motivation:**
Outliers can:
- Skew statistical measures (mean, variance)
- Dominate distance-based algorithms
- Reduce model generalization
- Represent data errors vs. true anomalies

### Technique: IQR-Based Winsorization

**Process:**
1. Calculate Interquartile Range (IQR) = Q3 - Q1
2. Define bounds: 
   - Lower bound = Q1 - 1.5 × IQR
   - Upper bound = Q3 + 1.5 × IQR
3. Clip values to bounds (winsorization)

```python
Q1 = data[col].quantile(0.25)
Q3 = data[col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
```

**Motivation:**
- **IQR method**: Robust to extreme values (unlike mean ± 3σ)
- **Winsorization vs. removal**: Preserves sample size
- **1.5 × IQR**: Standard threshold balancing sensitivity and specificity

**Applied to:**
- `time_in_hospital`: Extremely long stays
- `num_lab_procedures`: Unusual test volumes
- `num_medications`: Polypharmacy extremes
- `number_emergency`: Frequent flyers
- All numerical utilization features

**Results:**
- **Example**: 40-day hospital stay → clipped to 14 days
- Reduces influence of extreme values
- Maintains data distribution shape
- Typical treatment: 2-5% of values per feature

**Why Not Remove Outliers?**
- Preserves valuable samples in imbalanced dataset
- Outliers may represent valid high-risk patients
- Winsorization retains information while reducing extremity

---

## 5. Target Variable Creation

### Method: `create_target()`

**Motivation:**
Convert multi-class readmission variable into binary classification target aligned with clinical intervention window.

**Process:**
```python
data['target'] = (data['readmitted'] == '<30').astype(int)
# 1 = Readmitted within 30 days
# 0 = Not readmitted or readmitted after 30 days
```

**Why 30 Days?**
- **Clinical relevance**: Medicare penalties apply to 30-day readmissions
- **Intervention window**: Time frame for preventive care
- **Policy alignment**: Industry standard metric
- **Cost optimization**: Early readmissions are most costly

**Original Classes:**
- `<30`: Readmitted in less than 30 days → **Target = 1**
- `>30`: Readmitted after 30 days → **Target = 0**
- `NO`: Not readmitted → **Target = 0**

**Results:**
- Binary classification problem
- **Typical distribution**: ~11% readmitted (class 1), ~89% not readmitted (class 0)
- **Imbalance ratio**: ~1:8
- **Handling**: Class weights in modeling phase (not preprocessing)

---

## 6. Feature Engineering

### Method: `engineer_features()`

**Motivation:**
Raw features alone may not capture complex clinical patterns. Engineered features encode domain knowledge and reveal hidden relationships.

### 6.1 Diagnosis Code Aggregation

**Purpose:** Convert granular ICD-9 codes into clinically meaningful categories

```python
def categorize_diagnosis(code):
    if 250.0 <= code_num < 251:
        return 'Diabetes'
    elif 390 <= code_num < 460:
        return 'Circulatory'
    elif 460 <= code_num < 520:
        return 'Respiratory'
    # ... 10 total categories
```

**Motivation:**
- **Reduce cardinality**: 1000+ codes → 10 categories
- **Clinical grouping**: Diseases with similar prognosis grouped
- **Generalization**: New codes map to existing categories
- **Interpretability**: "Circulatory" more meaningful than "428.0"

**Categories Created:**
1. Diabetes (250-251)
2. Circulatory (390-460, 785)
3. Respiratory (460-520, 786)
4. Digestive (520-580, 787)
5. Injury (800-1000)
6. Neoplasms (140-240)
7. Musculoskeletal (710-740)
8. Genitourinary (580-630)
9. Nervous (320-390)
10. Other / Unknown

**Features Generated:**
- `diag_1_cat`, `diag_2_cat`, `diag_3_cat`: Categories for 3 diagnosis positions
- `unique_diagnosis_categories`: Count of distinct categories (1-3)

**Results:**
- Captures comorbidity complexity
- Example: Patient with Diabetes + Circulatory + Respiratory = 3 unique categories (high complexity)
- Reduces model parameters while preserving clinical signal

### 6.2 Utilization Features

**Purpose:** Quantify healthcare system usage patterns

#### Total Visits Count
```python
data['total_visits'] = (data['number_outpatient'] + 
                        data['number_emergency'] + 
                        data['number_inpatient'])
```

**Motivation:**
- High utilization → chronic/complex conditions
- Proxy for disease severity
- Predicts future healthcare needs

#### Per-Patient Statistics (if patient_nbr available)
```python
patient_stats = data.groupby('patient_nbr')[utilization_cols].agg(['mean', 'max', 'sum'])
```

**Features:**
- `number_outpatient_mean`: Average outpatient visits per patient
- `number_emergency_max`: Peak emergency visit count
- `number_inpatient_sum`: Total inpatient admissions

**Motivation:**
- Captures longitudinal patterns
- Identifies frequent utilizers
- Baseline utilization predicts readmission risk

**Results:**
- Example: Patient with `total_visits = 15` (high utilizer) vs. `total_visits = 2` (low utilizer)
- Enables risk stratification

### 6.3 Medication Complexity Score

**Purpose:** Quantify medication regimen complexity

```python
medication_cols = ['metformin', 'insulin', 'glipizide', ...]  # 23 medications
med_changes = sum(data[col].isin(['Up', 'Down']) for col in medication_cols)
data['medication_complexity_score'] = med_changes
```

**Motivation:**
- Medication changes indicate:
  - **Titration**: Disease management
  - **Polypharmacy**: Multiple conditions
  - **Instability**: Uncontrolled disease
- More changes = higher complexity = higher risk

**Possible Values:** 0 (no changes) to 23 (all medications changed)

**Results:**
- Example: Score of 8 indicates 8 diabetes medications adjusted
- Strong predictor of readmission (unstable regimen)
- Captures therapeutic intensity

### 6.4 Care Utilization Risk Score

**Purpose:** Weighted sum of healthcare encounters

```python
data['care_utilization_risk_score'] = (
    data['number_outpatient'] * 1 +
    data['number_emergency'] * 3 +  # Weighted higher
    data['number_inpatient'] * 2
)
```

**Motivation:**
- **Emergency visits** (×3 weight): Indicate acute decompensation
- **Inpatient visits** (×2 weight): Serious events requiring admission
- **Outpatient visits** (×1 weight): Routine care
- Differential weighting reflects severity

**Results:**
- Example: 
  - Patient A: 10 outpatient, 0 emergency, 0 inpatient → Score = 10
  - Patient B: 0 outpatient, 2 emergency, 1 inpatient → Score = 8 (higher risk despite fewer total visits)
- Captures urgency and severity of care needs

### 6.5 Age Categories

**Purpose:** Create clinically meaningful age groups

```python
age_mapping = {
    '[0-10)': 5, '[10-20)': 15, ..., '[90-100)': 95
}
data['age_numeric'] = data['age'].map(age_mapping)

def bucket_age(age_val):
    if age_val < 40: return 'Young'
    elif age_val < 65: return 'Adult'
    elif age_val < 80: return 'Senior'
    else: return 'Elderly'
```

**Motivation:**
- Original: 10-year bins (`[0-10)`, `[10-20)`, etc.)
- **Age-readmission relationship**:
  - Young (<40): Lower baseline risk
  - Adult (40-64): Moderate risk, comorbidities emerging
  - Senior (65-79): Higher risk, Medicare age
  - Elderly (80+): Highest risk, frailty
- Ordered categorical preserves ordinality

**Features:**
- `age_numeric`: Midpoint of original bin (for continuous operations)
- `age_bucket`: 4-level ordered categorical

**Results:**
- Captures non-linear age effects
- Example: 85-year-old in "Elderly" group (high risk)
- Enables age-specific interventions

### 6.6 BMI Categories

**Purpose:** Estimate body mass index from weight ranges

```python
weight_to_bmi = {
    '[0-25)': 18, '[25-50)': 20, ..., '>200': 35
}
data['estimated_bmi'] = data['weight'].map(weight_to_bmi)

data['bmi_underweight'] = (data['estimated_bmi'] < 18.5).astype(int)
data['bmi_normal'] = ((data['estimated_bmi'] >= 18.5) & (data['estimated_bmi'] < 25)).astype(int)
data['bmi_overweight'] = ((data['estimated_bmi'] >= 25) & (data['estimated_bmi'] < 30)).astype(int)
data['bmi_obese'] = (data['estimated_bmi'] >= 30).astype(int)
```

**Motivation:**
- Weight alone insufficient (doesn't account for height)
- **BMI-health relationship**:
  - Underweight (<18.5): Malnutrition risk
  - Normal (18.5-25): Lowest risk
  - Overweight (25-30): Moderate risk
  - Obese (30+): High cardiovascular risk
- Critical for diabetic patients

**Limitation:** Rough estimation (assumes average height)

**Results:**
- Binary features for each BMI category
- Example: Obese patient (BMI 35) → `bmi_obese = 1`, others = 0
- Enables model to learn category-specific effects

### 6.7 Interaction Features

**Purpose:** Capture synergistic effects between features

```python
# Length of stay × Medication complexity
data['los_medication_interaction'] = (
    data['time_in_hospital'] * data['medication_complexity_score']
)

# Medications × Diagnoses
data['med_diagnosis_interaction'] = (
    data['num_medications'] * data['number_diagnoses']
)
```

**Motivation:**
- **LOS × Medication complexity**: Long stay + many med changes = unstable patient
- **Medications × Diagnoses**: 20 meds + 10 diagnoses = polypharmacy + multimorbidity
- Linear models can't learn multiplicative effects without explicit features

**Results:**
- Example: 
  - Patient A: 7-day stay, 5 med changes → Interaction = 35
  - Patient B: 3-day stay, 2 med changes → Interaction = 6 (much lower complexity)
- Captures compound risk factors

---

## 7. Demographics Storage

### Method: `_store_demographics()`

**Motivation:**
Phase 5 requires original demographic values for fairness evaluation. Must preserve **before** encoding destroys original values.

**Critical Demographics:**
- `race`: African American, Caucasian, Asian, Hispanic, Other
- `gender`: Male, Female, Unknown/Invalid
- `age`: Original age brackets

**Process:**
```python
demographic_cols = ['race', 'gender', 'age']
self.original_demographics = data[demographic_cols].copy()
```

**Why Before Encoding?**
- One-hot encoding: `race` → `race_Caucasian`, `race_Asian`, etc. (original lost)
- Target encoding: `race` → numeric values (semantics lost)
- Fairness analysis needs original categories

**Results:**
- Stored separately from main pipeline
- Aligned with train/val/test indices
- Exported as `train_demographics.csv`, `validation_demographics.csv`, `test_demographics.csv`
- Enables Phase 5 fairness metrics (disparate impact, equalized odds)

---

## 8. Feature Encoding

### Method: `encode_features()`

**Motivation:**
Machine learning algorithms require numerical inputs. Different categorical types need different encoding strategies.

### 8.1 One-Hot Encoding (Low Cardinality)

**Applied to:** Features with <10 unique categories

```python
if unique_count < 10:
    one_hot_data = pd.get_dummies(data[col], prefix=col, drop_first=True)
```

**Examples:**
- `gender`: 3 categories → `gender_Male`, `gender_Female` (drop `Unknown` as reference)
- `admission_type_id`: 8 categories → 7 binary features

**Motivation:**
- **No ordinality assumption**: Male ≠ 2 × Female
- **Interpretability**: Each category gets explicit feature
- **drop_first=True**: Prevents multicollinearity (reference category)

**Results:**
- Example: `gender='Male'` → `gender_Male=1, gender_Female=0`
- Adds interpretable features
- Increases feature count but maintains sparsity

### 8.2 Target Encoding (High Cardinality)

**Applied to:** Features with ≥10 unique categories

```python
# Calculate mean target by category
category_means = data.groupby(col)[y.name].mean()
global_mean = y.mean()

# Bayesian smoothing
smoothing = 100
smoothed_means = (category_counts * category_means + smoothing * global_mean) / 
                 (category_counts + smoothing)
```

**Examples:**
- `medical_specialty`: 73 categories → 1 numerical feature
- `admission_source_id`: 25 categories → 1 numerical feature

**Motivation:**
- **One-hot explosion**: 73 categories → 72 binary features (too sparse)
- **Target encoding**: 73 categories → 1 feature (compact)
- **Captures relationship**: High readmission specialties get higher values

**Bayesian Smoothing:**
- **Problem**: Rare categories (e.g., 1 sample) have unreliable means
- **Solution**: Blend category mean with global mean
- **Formula**: Weighted average based on sample count
- **Effect**: Rare categories → closer to global mean; Common categories → their actual mean

**CV-Safe Implementation:**
- **Training**: Calculate encoding map, store for later use
- **Validation/Test**: Use stored map (prevents data leakage)
- **Unseen categories**: Fall back to global mean

**Results:**
- Example: `medical_specialty='Cardiology'` (15% readmit rate) → 0.15
- Compact representation
- Preserves predictive signal
- No overfitting on rare categories

### 8.3 Ordered Categorical Encoding

**Applied to:** `age_bucket` (ordered: Young < Adult < Senior < Elderly)

```python
data['age_bucket'] = pd.Categorical(
    data['age_bucket'],
    categories=['Young', 'Adult', 'Senior', 'Elderly'],
    ordered=True
)
data['age_bucket'] = data['age_bucket'].cat.codes  # 0, 1, 2, 3
```

**Motivation:**
- Preserves natural ordering
- Treats as numerical with inherent sequence
- More efficient than one-hot for ordinal data

**Results:**
- Young → 0, Adult → 1, Senior → 2, Elderly → 3
- Model learns monotonic age effect

---

## 9. Column Name Sanitization

### Method: `sanitize_column_names()`

**Motivation:**
LightGBM (used in later phases) fails with special JSON characters in column names.

**Problematic Characters:**
- Colons (`:`), Quotes (`"`, `'`), Brackets (`[`, `]`, `{`, `}`)
- Backslashes (`\`), Forward slashes (`/`), Commas (`,`)

**Process:**
```python
def clean_name(name):
    for char in [':', '"', "'", '[', ']', '{', '}', '\\', '/', ',', '<', '>', '|']:
        name = name.replace(char, '_')
    name = name.replace(' ', '_').replace('-', '_')
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_')
```

**Examples:**
- `diag_1_cat` → `diag_1_cat` (no change)
- `race_is_missing` → `race_is_missing` (no change)
- Hypothetical `feature[0-10]` → `feature_0_10_`

**Results:**
- All column names compatible with LightGBM JSON serialization
- Prevents model saving/loading errors in Phase 2-6
- Maintains readability

---

## 10. Train/Validation/Test Splitting

### Method: `fit_transform()` → `create_train_test_split()`

**Motivation:**
- **Generalization**: Test set estimates real-world performance
- **Hyperparameter tuning**: Validation set prevents test set contamination
- **Stratification**: Maintains class balance across splits

**Process:**
```python
# First split: Separate test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Second split: Separate validation from training (15% of remaining)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)
```

**Split Proportions:**
- **Training**: 70% (~71,000 samples)
- **Validation**: 15% (~15,000 samples)
- **Test**: 15% (~15,000 samples)

**Stratification:**
- Maintains 11% readmission rate in all splits
- Prevents sampling bias
- Ensures representative evaluation

**Random Seed (42):**
- Reproducible splits across runs
- Enables consistent comparison
- Documented for transparency

**Results:**
- Three independent datasets
- No overlap between splits
- Balanced class distributions

---

## 11. Feature Scaling

### Method: `scale_features_train_test()`

**Motivation:**
Many algorithms (SVM, neural networks, KNN) are sensitive to feature scales. Large-scale features dominate distance calculations.

### Critical: No Data Leakage

**WRONG Approach (Leakage):**
```python
scaler.fit(entire_dataset)  # ❌ Uses test set statistics!
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**CORRECT Approach (No Leakage):**
```python
scaler.fit(X_train)  # ✅ Only training data!
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**Why This Matters:**
- Test set statistics (mean, std) leak into scaler
- Model indirectly "sees" test set
- Inflates performance estimates
- Violates deployment reality (production data unseen)

### Scaler Options

#### StandardScaler (Default)
```python
X_scaled = (X - mean) / std
```

**Use when:**
- Features approximately normally distributed
- No extreme outliers (after outlier treatment)
- Most algorithms (Logistic Regression, SVM, Neural Networks)

**Results:**
- Mean = 0, Standard Deviation = 1
- Example: `num_medications` (range 1-80) → scaled to (-2 to +3)

#### RobustScaler (Alternative)
```python
X_scaled = (X - median) / IQR
```

**Use when:**
- Features have outliers (even after treatment)
- Skewed distributions
- Robust to extreme values

**Results:**
- Median = 0, IQR = 1
- Less sensitive to outliers than StandardScaler

### Scaler Persistence

**Saved to:** `./data/processed/splits/scaler.pkl`

**Purpose:**
- **Deployment**: Transform new patient data identically
- **Consistency**: All phases (2-6) use same scaler
- **Reproducibility**: Exact transformation preserved

**Usage in Production:**
```python
import joblib
scaler = joblib.load('scaler.pkl')
new_patient_scaled = scaler.transform(new_patient_data)
```

**Results:**
- All features on comparable scales
- Algorithm convergence improved
- No data leakage (fitted on train only)

---

## 12. Data Export

### 12.1 Local File System

**Outputs:**
- `./data/processed/splits/train.csv`: Training set (70%)
- `./data/processed/splits/validation.csv`: Validation set (15%)
- `./data/processed/splits/test.csv`: Test set (15%)
- `./data/processed/splits/train_demographics.csv`: Demographics for fairness (Phase 5)
- `./data/processed/splits/validation_demographics.csv`
- `./data/processed/splits/test_demographics.csv`
- `./data/processed/splits/scaler.pkl`: Fitted scaler (no leakage)
- `./data/processed/splits/split_info.txt`: Metadata

**Results:**
- All datasets saved as CSV (readable, portable)
- Demographics aligned by index with main splits
- Scaler ready for Phase 2-6 deployment

### 12.2 Hugging Face Hub Export

**Method:** `export_for_huggingface()`

**Outputs:**
- `hospital_readmission_full.csv`: Complete dataset
- `README.md`: Dataset card (documentation)
- `dataset_info.json`: Metadata (features, splits, distribution)
- `splits/`: Train/val/test splits

**Upload Process:**
1. Read `HF_TOKEN` from `.env` file
2. Create/verify repository on Hugging Face
3. Upload all files to `datasets/{username}/{repo_name}`

**Environment Variables:**
```bash
# .env file
HF_TOKEN=hf_xxxxxxxxxxxxx
HF_REPO_ID=username/hospital-readmission-risk-data
```

**Benefits:**
- **Versioning**: Track dataset changes over time
- **Collaboration**: Share with team/public
- **Accessibility**: Load with `datasets.load_dataset()`
- **Documentation**: Built-in dataset card

**Results:**
- Public dataset at `https://huggingface.co/datasets/{username}/{repo_name}`
- Citeable DOI
- Community visibility

---

## Usage Examples

### Basic Usage

```python
from simple_preprocessing import CompletePreprocessor

# Initialize preprocessor
preprocessor = CompletePreprocessor(random_state=42, scaler_type='standard')

# Run complete pipeline
splits_dict = preprocessor.fit_transform(data_path="./data/diabetic_data.csv")

# Access splits
X_train = splits_dict['X_train']  # Shape: (71000, 120)
y_train = splits_dict['y_train']  # Shape: (71000,)
X_val = splits_dict['X_val']      # Shape: (15000, 120)
y_val = splits_dict['y_val']
X_test = splits_dict['X_test']    # Shape: (15000, 120)
y_test = splits_dict['y_test']

# Save splits to files
preprocessor.create_train_test_split(splits_dict)

# Export to Hugging Face
preprocessor.export_for_huggingface(splits_dict)
```

### Load Preprocessed Data (Phase 2-6)

```python
import pandas as pd
import joblib

# Load splits
train = pd.read_csv('./data/processed/splits/train.csv')
X_train = train.drop('target', axis=1)
y_train = train['target']

# Load demographics for fairness evaluation
demographics = pd.read_csv('./data/processed/splits/test_demographics.csv')

# Load scaler for new data
scaler = joblib.load('./data/processed/splits/scaler.pkl')
new_data_scaled = scaler.transform(new_patient_data)
```

### Production Deployment

```python
import joblib
import pandas as pd

# Load saved scaler
scaler = joblib.load('scaler.pkl')

# New patient data (after same preprocessing steps)
new_patient = pd.DataFrame({
    'time_in_hospital': [5],
    'num_medications': [15],
    # ... all other features
})

# Apply preprocessing
new_patient_processed = apply_same_preprocessing(new_patient)  # Your function
new_patient_scaled = scaler.transform(new_patient_processed)

# Load model and predict
model = joblib.load('trained_model.pkl')
prediction = model.predict_proba(new_patient_scaled)
```

---

## Feature-by-Feature Preprocessing Summary

This section maps each preprocessing technique to the specific features it affects.

### Missing Value Handling

#### Missing Indicators Created
- `A1Cresult_is_missing` ← `A1Cresult`
- `weight_is_missing` ← `weight`
- `race_is_missing` ← `race`
- `medical_specialty_is_missing` ← `medical_specialty`
- `payer_code_is_missing` ← `payer_code`

#### Group-Wise Imputation (MAR)
- `medical_specialty` ← Imputed by `admission_type_id` mode
- `payer_code` ← Imputed by `admission_source_id` mode

#### Mode Imputation (Categorical)
- `race` (if still missing after group-wise)
- `gender`
- `age`
- `admission_type_id`
- `discharge_disposition_id`
- `admission_source_id`
- All medication columns (metformin, insulin, etc.)
- `max_glu_serum`
- `A1Cresult`
- `change`
- `diabetesMed`
- All diagnosis columns (`diag_1`, `diag_2`, `diag_3`)

#### Median Imputation (Numerical)
- `time_in_hospital`
- `num_lab_procedures`
- `num_procedures`
- `num_medications`
- `number_outpatient`
- `number_emergency`
- `number_inpatient`
- `number_diagnoses`

### Data Quality Validation

#### Range Validation & Clipping
- `time_in_hospital` → [1, 14] days
- `num_lab_procedures` → [0, 150]
- `num_procedures` → [0, 10]
- `num_medications` → [0, 100]
- `number_outpatient` → [0, 50]
- `number_emergency` → [0, 50]
- `number_inpatient` → [0, 20]
- `number_diagnoses` → [1, 16]

#### Domain Constraint Validation
- `gender` → Must be in {Male, Female, Unknown/Invalid}

### Outlier Treatment (IQR Winsorization)

Applied to all numerical features:
- `time_in_hospital`
- `num_lab_procedures`
- `num_procedures`
- `num_medications`
- `number_outpatient`
- `number_emergency`
- `number_inpatient`
- `number_diagnoses`

### Feature Engineering

#### Diagnosis Code Aggregation
**Input Features:**
- `diag_1`, `diag_2`, `diag_3`

**Output Features:**
- `diag_1_cat` ← Clinical category of primary diagnosis
- `diag_2_cat` ← Clinical category of secondary diagnosis
- `diag_3_cat` ← Clinical category of tertiary diagnosis
- `unique_diagnosis_categories` ← Count of distinct categories

**Categories:** Diabetes, Circulatory, Respiratory, Digestive, Injury, Neoplasms, Musculoskeletal, Genitourinary, Nervous, Other, External/Supplemental

#### Utilization Features
**Input Features:**
- `number_outpatient`, `number_emergency`, `number_inpatient`
- `patient_nbr` (for grouping)

**Output Features:**
- `total_visits` ← Sum of all visit types
- `number_outpatient_mean` ← Per-patient average (if patient_nbr available)
- `number_outpatient_max` ← Per-patient maximum
- `number_outpatient_sum` ← Per-patient total
- `number_emergency_mean` ← Per-patient average
- `number_emergency_max` ← Per-patient maximum
- `number_emergency_sum` ← Per-patient total
- `number_inpatient_mean` ← Per-patient average
- `number_inpatient_max` ← Per-patient maximum
- `number_inpatient_sum` ← Per-patient total

#### Medication Complexity Score
**Input Features (23 medications):**
- `metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`
- `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`
- `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`
- `tolazamide`, `insulin`, `glyburide-metformin`, `glipizide-metformin`
- `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`

**Output Feature:**
- `medication_complexity_score` ← Count of medications with changes (Up/Down)

#### Care Utilization Risk Score
**Input Features:**
- `number_outpatient`, `number_emergency`, `number_inpatient`

**Output Feature:**
- `care_utilization_risk_score` ← Weighted sum (outpatient×1 + emergency×3 + inpatient×2)

#### Age Features
**Input Feature:**
- `age` (categorical bins: [0-10), [10-20), ..., [90-100))

**Output Features:**
- `age_numeric` ← Midpoint of age bracket (5, 15, 25, ..., 95)
- `age_bucket` ← Ordered categorical {Young, Adult, Senior, Elderly}

#### BMI Features
**Input Feature:**
- `weight` (categorical ranges: [0-25), [25-50), ..., >200)

**Output Features:**
- `estimated_bmi` ← Estimated BMI value (18-35)
- `bmi_underweight` ← Binary indicator (BMI < 18.5)
- `bmi_normal` ← Binary indicator (18.5 ≤ BMI < 25)
- `bmi_overweight` ← Binary indicator (25 ≤ BMI < 30)
- `bmi_obese` ← Binary indicator (BMI ≥ 30)

#### Interaction Features
**Input Features & Outputs:**
- `los_medication_interaction` ← `time_in_hospital` × `medication_complexity_score`
- `med_diagnosis_interaction` ← `num_medications` × `number_diagnoses`

### Feature Encoding

#### One-Hot Encoding (Low Cardinality < 10 categories)
**Applied to:**
- `gender` → `gender_Male`, `gender_Female`
- `admission_type_id` → 7 binary features (drop first)
- `discharge_disposition_id` → Multiple binary features
- `max_glu_serum` → Binary features for each value
- `A1Cresult` → Binary features for each result type
- `change` → `change_Yes` or similar
- `diabetesMed` → `diabetesMed_Yes` or similar
- `diag_1_cat`, `diag_2_cat`, `diag_3_cat` → Binary for each category
- `age_bucket` → Encoded as ordered numeric (0-3)
- BMI categories → Already binary

#### Target Encoding (High Cardinality ≥ 10 categories)
**Applied to:**
- `medical_specialty` (73 categories) → `medical_specialty_target_encoded`
- `admission_source_id` (25 categories) → `admission_source_id_target_encoded`
- Any other high-cardinality categoricals

**Method:** Bayesian smoothed mean target per category

#### Preserved as Numeric (No encoding needed)
- `time_in_hospital`
- `num_lab_procedures`
- `num_procedures`
- `num_medications`
- `number_outpatient`, `number_emergency`, `number_inpatient`
- `number_diagnoses`
- All engineered numerical features
- All missing indicators (already binary)

### Feature Scaling (StandardScaler or RobustScaler)

**Applied to ALL features** after encoding:
- All original numerical features
- All engineered numerical features
- All one-hot encoded features (0/1 → scaled)
- All target-encoded features
- All missing indicators

**Fitted ONLY on training data, transformed on train/val/test**

### Features Removed/Dropped

**Dropped before modeling:**
- `encounter_id` ← Unique identifier (no predictive value)
- `patient_nbr` ← Patient identifier (would cause data leakage)
- `readmitted` ← Original target (replaced by binary `target`)
- Original `age` ← Replaced by `age_numeric` and `age_bucket`
- Original `weight` ← Replaced by `estimated_bmi` and BMI categories
- Original categorical features ← Replaced by encoded versions
- Diagnosis codes (`diag_1`, `diag_2`, `diag_3`) ← Replaced by category versions

### Demographics Preserved for Fairness Analysis

**Stored separately (before encoding):**
- `race` ← Original categories
- `gender` ← Original categories
- `age` ← Original age brackets

**Files:**
- `train_demographics.csv`
- `validation_demographics.csv`
- `test_demographics.csv`

---

## Summary of Preprocessing Results

### Input
- **101,766 patients** with **50+ raw features**
- Missing values (~40% for some features)
- Categorical variables (race, gender, diagnosis codes)
- Imbalanced classes (11% readmission)

### Output
- **Training**: 71,000 samples × 120 features
- **Validation**: 15,000 samples × 120 features
- **Test**: 15,000 samples × 120 features
- Zero missing values
- All numerical features (ready for ML)
- Scaled and normalized (mean=0, std=1)
- Demographics preserved for fairness analysis
- Scaler saved (no data leakage)

### Key Achievements
1. ✅ **100% data quality**: No missing values, valid ranges, correct types
2. ✅ **Domain knowledge encoded**: Diagnosis categories, utilization scores, interactions
3. ✅ **No data leakage**: Scaler fitted only on training data
4. ✅ **Fairness-ready**: Demographics preserved for Phase 5
5. ✅ **Reproducible**: Fixed random seed, saved artifacts
6. ✅ **Deployment-ready**: Scaler and splits saved for production use
7. ✅ **Well-documented**: Metadata files explain all transformations

---

## References

1. **Original Dataset**: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008)
2. **ICD-9 Code Ranges**: Clinical modification groupings
3. **30-Day Readmission**: [CMS Hospital Readmissions Reduction Program](https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp)
4. **BMI Categories**: [WHO BMI Classification](https://www.who.int/health-topics/obesity)

---

## Next Steps

After preprocessing:
1. **Phase 2**: Train models (Logistic Regression, Random Forest, Gradient Boosting)
2. **Phase 3**: Calibrate probability outputs (Platt scaling, Isotonic regression)
3. **Phase 4**: Optimize decision threshold for ROI (cost-benefit analysis)
4. **Phase 5**: Evaluate fairness across demographic groups
5. **Phase 6**: Final system evaluation and deployment readiness
6. **Phase 7**: Build stakeholder dashboards

All phases use the preprocessed data and saved scaler from this pipeline.
