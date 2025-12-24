# Phase 2: Risk Modeling - Training Strategy

## Overview

Phase 2 implements a rigorous model development pipeline comparing three complementary algorithms:
- **Logistic Regression**: Linear baseline with L1/L2/ElasticNet regularization
- **Random Forest**: Ensemble variance reduction with bootstrap aggregation
- **Gradient Boosting (LightGBM)**: Sequential bias reduction with gradient boosting

All models use Phase 1's preprocessed 113-feature dataset from HuggingFace Hub (`auphong2707/hospital-readmission-risk-data`), ensuring consistency across all modeling phases.

---

## Data Partitioning Strategy

### Phase 1 Splits Integration
Phase 2 leverages Phase 1's preprocessed splits as the single source of truth:

```
Phase 1 Output (70-15-15 split):
├── Train split:      73,526 samples (~70%)
├── Validation split: 12,975 samples (~15%)
└── Test split:       15,265 samples (15%)

Phase 2 Recombination:
├── Development Set = Train + Validation: 86,501 samples (~85%)
│   └── Used for: Hyperparameter search, K-fold CV, final model training
└── Final Test Set = Test: 15,265 samples (15%)
    └── Untouched until final evaluation (strict holdout)
```

**Key Principle**: The test set remains completely untouched during hyperparameter tuning and cross-validation to ensure unbiased performance estimation.

---

## Evaluation Architecture

### Three-Level Evaluation Structure

#### Level 1: Data Partitioning
- **Development Set** ($D_{dev}$): 86,501 samples (train + validation from Phase 1)
- **Final Test Set** ($D_{test}$): 15,265 samples (Phase 1 test set)
- **Mathematical Guarantee**: $D = D_{dev} \cup D_{test}$, where $D_{dev} \cap D_{test} = \emptyset$

#### Level 2: Stratified K-Fold Cross-Validation
For each hyperparameter configuration $\theta \in \Theta$:
- Development set partitioned into **k=5 stratified folds**
- Maintains **11.2% positive class ratio** across all folds
- Each fold $i$ provides:
  - Training set: $D_{train}^{(i)}$ (80% of development)
  - Holdout validation: $D_{val}^{(i)}$ (20% of development)

#### Level 3: Nested Early Stopping (Gradient Boosting Only)
Within each fold's training phase for Gradient Boosting:
- $D_{train}^{(i)}$ undergoes **90/10 split** into:
  - Inner training: $D_{inner}^{(i)}$ (90% of fold training data)
  - Inner validation: $D_{stop}^{(i)}$ (10% of fold training data)
- Training on $D_{inner}^{(i)}$ monitored on $D_{stop}^{(i)}$
- **Early stopping**: Halts if AUC doesn't improve for **50 rounds**
- Final evaluation: On original fold holdout $D_{val}^{(i)}$

**Note**: Random Forest and Logistic Regression train directly on $D_{train}^{(i)}$ without nested splits, leveraging their inherent regularization mechanisms (bootstrap sampling for RF, L1/L2 penalties for LR).

---

## Hyperparameter Optimization

### Strategy
Comprehensive grid search with stratified K-fold cross-validation to find optimal configuration $\theta^*$:

$$\theta^* = \operatorname*{argmax}_{\theta \in \Theta} \frac{1}{k} \sum_{i=1}^{k} \text{AUC}(f(D_{val}^{(i)}; \theta))$$

### Search Spaces

#### Logistic Regression (60 combinations, 300 total fits)
```python
# L1/L2 with liblinear solver (24 combinations)
# 6 C values × 2 penalties × 2 class_weights = 24
{
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Regularization strength
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced', {0: 1, 1: 8}]
}

# ElasticNet with saga solver (36 combinations)
# 6 C values × 1 penalty × 3 l1_ratios × 2 class_weights = 36
{
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['elasticnet'],
    'solver': ['saga'],
    'l1_ratio': [0.25, 0.5, 0.75],  # ElasticNet mixing parameter
    'class_weight': ['balanced', {0: 1, 1: 8}]
}
```

#### Random Forest (162 combinations, 810 total fits)
```python
{
    'n_estimators': [100, 250, 500],           # Number of trees
    'max_depth': [10, 25, None],               # Tree depth
    'min_samples_split': [2, 5, 10],           # Min samples to split
    'min_samples_leaf': [1, 2, 4],             # Min samples at leaf
    'max_features': ['sqrt'],                   # Features per split
    'class_weight': ['balanced', {0: 1, 1: 8}],
    'bootstrap': [True],
    'oob_score': [True]                         # Out-of-bag evaluation
}
```

#### LightGBM Gradient Boosting (864 combinations, 4,320 total fits)
```python
{
    'n_estimators': [50, 100, 150],            # Boosting rounds
    'learning_rate': [0.01, 0.05, 0.1],        # Step size shrinkage
    'num_leaves': [31, 63, 127],               # Max leaves per tree
    'max_depth': [-1, 10],                     # Tree depth (-1 = no limit)
    'subsample': [0.7, 0.9],                   # Row sampling ratio
    'colsample_bytree': [0.7, 1.0],            # Column sampling ratio
    'reg_alpha': [0, 0.1],                     # L1 regularization
    'reg_lambda': [0, 0.1]                     # L2 regularization
}
```

---

## Class Imbalance Handling

All models employ **`class_weight='balanced'`** to address the **11.2% positive class prevalence** without synthetic data generation (avoiding SMOTE artifacts).

### Inverse-Frequency Weighting
For class $j$ with sample count $N_j$ in dataset of size $N$ with $C$ classes:

$$w_j = \frac{N}{C \cdot N_j}$$

**Effect**: Equal total influence from both classes in loss function, eliminating majority class bias.

**Example** (with 11.2% readmission rate):
- Class 0 (no readmission): weight ≈ 0.563
- Class 1 (readmission): weight ≈ 4.464

---

## Final Model Training

After identifying optimal $\theta^*$ via grid search, final model training follows **method-specific strategies**:

### Gradient Boosting
```python
# Split development set for early stopping monitoring
X_dev_train, X_dev_val = train_test_split(
    X_development, 
    test_size=0.1,  # 90/10 split
    stratify=y_development
)

# Train with early stopping on inner validation
model.fit(
    X_dev_train, y_dev_train,
    eval_set=[(X_dev_val, y_dev_val)],
    early_stopping_rounds=50
)
```

### Random Forest & Logistic Regression
```python
# Train directly on full development set
# Leverages inherent regularization:
#   - RF: bootstrap sampling + feature randomness
#   - LR: L1/L2 penalty from hyperparameter search
model.fit(X_development, y_development)
```

---

## Final Evaluation

Each deployment candidate evaluated **once** on untouched holdout $D_{test}$ for unbiased performance estimation.

### Comprehensive Metrics Suite

#### Primary Metrics
- **ROC-AUC**: Area under receiver operating characteristic curve
- **PR-AUC**: Area under precision-recall curve
- **Brier Score**: Calibration quality (mean squared error of probabilities)

#### Classification Metrics
- Accuracy, Balanced Accuracy
- Precision, Recall, F1-Score
- Specificity, False Positive Rate, False Negative Rate

#### Clinical Metrics
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- PPV (Positive Predictive Value)
- NPV (Negative Predictive Value)

### Output Artifacts

All results uploaded to HuggingFace Hub for downstream phases:

1. **Serialized Models**: `.joblib` format with metadata
2. **Metrics**: JSON files with comprehensive evaluation results
3. **Visualizations**:
   - ROC curves with confidence intervals
   - Precision-Recall curves
   - Confusion matrices
   - Calibration plots (reliability diagrams)
   - Feature importance rankings
   - Learning curves (performance vs. training size)
   - Validation curves (performance vs. hyperparameters)
   - Cross-fold metrics comparison

---

## Usage

### Training Individual Models

```bash
# Logistic Regression (60 combinations × 5 folds = 300 fits)
python train_logistic_regression.py --n-splits 5

# Random Forest (162 combinations × 5 folds = 810 fits)
python train_random_forest.py --n-splits 5

# Gradient Boosting (864 combinations × 5 folds = 4,320 fits)
python train_gradient_boosting.py --n-splits 5 --early-stopping-rounds 50
```

### Custom Configuration

```bash
# Increase cross-validation folds for more robust estimates
python train_gradient_boosting.py --n-splits 10

# Adjust early stopping patience (Gradient Boosting only)
python train_gradient_boosting.py --early-stopping-rounds 100

# Custom output directory
python train_random_forest.py --output-dir ./custom_output
```

### Automatic Environment Detection
- **Kaggle**: Auto-detects GPU (for LightGBM) and uses all CPU cores
- **Local**: Uses all-but-one cores to keep system responsive

---

## Performance Expectations

### Computational Requirements

| Model | Combinations | Fits (k=5) | Approx. Time* |
|-------|-------------|-----------|---------------|
| Logistic Regression | 60 | 300 | ~8-15 min |
| Random Forest | 162 | 810 | ~30-60 min |
| Gradient Boosting | 864 | 4,320 | ~60-120 min |

*Times vary based on hardware (CPU/GPU) and dataset size

### Expected Performance (on holdout test set)
Based on 11.2% class imbalance and 113 features:
- **ROC-AUC**: 0.60-0.68 (all models)
- **PR-AUC**: 0.20-0.35 (challenging due to class imbalance)
- **Balanced Accuracy**: 0.58-0.64

---

## Key Implementation Details

### Data Preprocessing
- **No additional scaling**: Phase 1 already applied StandardScaler fitted on training data
- **Feature engineering**: 113 features from Phase 1 (demographics, diagnoses, procedures, medications, lab values)
- **Missing value handling**: Completed in Phase 1

### Random Seeds
- **Consistent across all scripts**: `random_state=42`
- Ensures reproducibility of splits and model initialization

### Stratification
- **All splits stratified**: Maintains 11.2% positive class ratio
- Applied at both CV fold level and inner splits

### Parallel Processing
- **Logistic Regression**: Grid search parallelized across all CPU cores
- **Random Forest**: Tree building parallelized (`n_jobs=-1`)
- **Gradient Boosting**: GPU acceleration when available, otherwise CPU parallelization

---

## Model Selection Philosophy

### Complementary Strengths
1. **Logistic Regression**
   - Fast, interpretable linear baseline
   - Well-calibrated probabilities by design
   - Ideal for understanding feature importance directions

2. **Random Forest**
   - Non-linear interactions without explicit feature engineering
   - Robust to outliers via bootstrap aggregation
   - Natural feature importance via impurity reduction

3. **Gradient Boosting**
   - State-of-the-art predictive performance
   - Sequential error correction
   - Handles complex interactions and non-linearities

### Ensemble Consideration
Phase 2 outputs enable Phase 6 to evaluate:
- Individual model performance
- Ensemble combinations (voting, stacking)
- Model-specific use cases based on calibration quality

---

## Validation Strategy Summary

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 2 Evaluation Pipeline                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Development Set (86,501 samples)                         │
│    ├── Grid Search: Find θ*                                │
│    │   └── For each θ ∈ Θ:                                 │
│    │       └── K-Fold CV (k=5):                            │
│    │           ├── Fold 1: Train on 4 folds → Eval on 1    │
│    │           ├── Fold 2: Train on 4 folds → Eval on 1    │
│    │           ├── ...                                      │
│    │           └── Fold 5: Train on 4 folds → Eval on 1    │
│    │           └── Nested (GB only): 90/10 early stopping  │
│    │                                                        │
│    └── Select θ* = argmax(mean CV AUC)                     │
│                                                             │
│ 2. Final Training with θ*                                   │
│    ├── Gradient Boosting: 90/10 split for early stopping   │
│    └── Random Forest / LR: Full development set            │
│                                                             │
│ 3. Final Test Set (15,265 samples) - UNTOUCHED             │
│    └── Single evaluation for unbiased performance          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## HuggingFace Integration

### Automatic Upload
Results automatically uploaded to HuggingFace Hub if `HF_TOKEN` environment variable is set:

```bash
# Set in .env file
HF_TOKEN=your_token_here
HF_USERNAME=your_username
```

### Repository Structure
```
hospital-readmission-phase2-{model}/
├── {model}_model.joblib
├── {model}_metrics.json
├── training_summary.json
├── cv_fold_details.json
└── visualizations/
    ├── roc_curve.png
    ├── pr_curve.png
    ├── confusion_matrix.png
    ├── calibration_curve.png
    ├── feature_importance.png
    ├── learning_curve.png
    └── validation_curves.png
```

---

## Next Steps

After Phase 2 completion:
1. **Phase 3**: Model calibration (Platt scaling, isotonic regression)
2. **Phase 4**: Optimal threshold selection with ROI analysis
3. **Phase 5**: Fairness assessment and mitigation
4. **Phase 6**: Final system evaluation and model selection

---

## References

- **Data Source**: HuggingFace Hub - `auphong2707/hospital-readmission-risk-data`
- **Scikit-learn Version**: 1.2.2+
- **LightGBM**: Latest stable with GPU support
- **Evaluation Framework**: Nested cross-validation with holdout test set

---

*Last Updated: December 2024*
