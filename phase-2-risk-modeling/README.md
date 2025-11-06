# Phase 2: Risk Modeling

This phase focuses on developing and evaluating predictive models for 30-day hospital readmission risk using the preprocessed diabetic patient data.

## 📋 Overview

The risk modeling phase implements multiple machine learning algorithms to predict patient readmission probability, with comprehensive model evaluation and comparison to identify the best-performing approach for clinical deployment.

## 🎯 Objectives

- **Primary Goal**: Develop robust predictive models for 30-day readmission risk
- **Model Comparison**: Evaluate multiple algorithms with rigorous validation
- **Clinical Applicability**: Ensure models are interpretable and actionable for healthcare providers
- **Performance Optimization**: Maximize predictive accuracy while maintaining model explainability

## 🤖 Model Architecture

### Candidate Models

#### 1. **Logistic Regression (Baseline)**
- **Configuration**: L1/L2 regularization with class weight balancing
- **Purpose**: Interpretable baseline model with feature importance
- **Hyperparameters**: Regularization strength (C), penalty type, class weights

#### 2. **Random Forest**
- **Configuration**: 100-500 trees with depth tuning and feature importance
- **Purpose**: Ensemble method handling non-linear relationships
- **Hyperparameters**: n_estimators, max_depth, min_samples_split, feature selection

#### 3. **Gradient Boosting**
- **Configuration**: XGBoost/LightGBM with early stopping and advanced sampling
- **Purpose**: High-performance boosting with DART/GOSS sampling techniques
- **Hyperparameters**: learning_rate, max_depth, subsample, early_stopping_rounds

#### 4. **Neural Networks**
- **Configuration**: Dense layers with dropout and batch normalization
- **Purpose**: Deep learning approach for complex pattern recognition
- **Hyperparameters**: hidden_layers, dropout_rate, learning_rate, batch_size

## 🔄 Validation Strategy
### Cross-Validation
- **Method**: Stratified K-fold (k=5)
- **Stratification**: Balanced across readmission classes
- **Purpose**: Robust performance estimation and hyperparameter tuning

### Hyperparameter Optimization
- **Grid Search**: Systematic parameter space exploration
- **Random Search**: Efficient sampling for large parameter spaces
- **Early Stopping**: Prevents overfitting during training

## 📊 Expected Deliverables

- [ ] **Model Training Scripts**: Individual scripts for each algorithm
- [ ] **Hyperparameter Tuning**: Automated optimization pipelines
- [ ] **Model Evaluation**: Comprehensive performance metrics and comparisons
- [ ] **Feature Importance**: Analysis of key predictive factors
- [ ] **Model Serialization**: Saved trained models for deployment
- [ ] **Performance Reports**: Detailed evaluation with visualizations

## 🚀 Getting Started

### Prerequisites

Ensure you have completed **Phase 1: Data Preprocessing** before proceeding with risk modeling.

#### Required Dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow imbalanced-learn matplotlib seaborn
```

### Step 1: Run Data Preprocessing

**⚠️ Important**: You must run the preprocessing pipeline first to generate the required training data.

From the project root directory:

```powershell
# Using Python virtual environment (recommended)
python phase-1-data-explore-preprocessing/simple_preprocessing.py

# Or with full path to virtual environment Python
.venv/Scripts/python.exe phase-1-data-explore-preprocessing/simple_preprocessing.py
```

#### Preprocessing Output
The preprocessing script generates:
- **180,818 samples** with **96 engineered features**
- Balanced dataset ready for machine learning
- Comprehensive feature engineering and data cleaning

#### Generated Files (in `data/processed/`)
- `preprocessed_hospital_data.csv` - Complete processed dataset
- `features.csv` - Feature matrix (X)
- `target.csv` - Target variable (y)
- `preprocessing_metadata.txt` - Processing details and statistics

### Step 2: Load Processed Data

```python
import pandas as pd
import numpy as np

# Load the complete processed dataset
data = pd.read_csv('./data/processed/preprocessed_hospital_data.csv')

# Or load features and target separately for ML workflows
X = pd.read_csv('./data/processed/features.csv')
y = pd.read_csv('./data/processed/target.csv')['target']

# View preprocessing metadata
with open('./data/processed/preprocessing_metadata.txt', 'r') as f:
    print(f.read())
```

### Step 3: Run Risk Modeling

```powershell
# Run individual model training scripts (to be implemented)
python phase-2-risk-modeling/train_logistic_regression.py
python phase-2-risk-modeling/train_random_forest.py
python phase-2-risk-modeling/train_gradient_boosting.py
python phase-2-risk-modeling/train_neural_network.py

# Or run complete model comparison pipeline
python phase-2-risk-modeling/model_comparison.py
```

## 📈 Expected Plots
- Learning curves for each model
- Validation curves for hyperparameter tuning
- ROC / PR curves for readability calibration
- Feature importance / SHAP value plots

## 📈 Expected Performance Metrics

- **Primary Metrics**: AUC-ROC, Precision, Recall, F1-Score, Accuracy, Specificity (TNR), FPR, F1-score, and Balanced Accuracy
- **Clinical Metrics**: Sensitivity, Specificity, PPV, NPV
- **Fairness Metrics**: Performance across demographic groups
- **Calibration**: Probability calibration assessment

## 🔧 Model Configuration

### Input Data Specifications
- **Features**: 96 engineered features from preprocessing phase
- **Samples**: 180,818 balanced patient records
- **Target**: Binary readmission indicator (0: No readmission, 1: Readmission within 30 days)
- **Data Split**: Temporal validation ensuring no data leakage

### Output Specifications
- **Predictions**: Readmission probability scores (0-1)
- **Classifications**: Binary readmission predictions with optimal thresholds
- **Feature Importance**: Ranked feature contributions for model interpretability
- **Model Artifacts**: Serialized models ready for clinical deployment
