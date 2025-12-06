# Phase 5 & 6 Implementation Guide for Logistic Regression

## Quick Reference

Since Phase 5 and Phase 6 scripts are 700+ and 500+ lines respectively, they follow the **exact same structure** as the Gradient Boosting versions. The ONLY changes needed are:

1. Model loading (add scaler)
2. File paths (use `outputs_logistic_regression/`)
3. Repository IDs (for HuggingFace upload)

---

## Phase 5: evaluate_fairness_logistic_regression.py

### Step-by-Step Instructions

1. **Copy the GBM version**:
   ```bash
   cp phase-5-fairness-evaluation/evaluate_fairness_gradient_boosting.py \
      phase-5-fairness-evaluation/evaluate_fairness_logistic_regression.py
   ```

2. **Update the docstring** (lines 1-60):
   - Replace "Gradient Boosting" with "Logistic Regression"
   - Update example usage:
     ```bash
     python ./phase-5-fairness-evaluation/evaluate_fairness_logistic_regression.py
     ```

3. **Update model loading function** (around line 150):
   
   **Find**:
   ```python
   def load_calibrated_model_and_calibrator(...):
       model = joblib.load(model_path)
       calibrator = ModelCalibrator.load(calibrator_path)
       return model, calibrator
   ```
   
   **Replace with**:
   ```python
   def load_calibrated_model_and_calibrator(...):
       # Load model
       model = joblib.load(model_path)
       # Load scaler
       scaler_path = model_path.parent / "logistic_regression_scaler.joblib"
       scaler = joblib.load(scaler_path)
       # Load calibrator
       calibrator = ModelCalibrator.load(calibrator_path)
       return model, scaler, calibrator
   ```

4. **Update prediction generation** (around line 250):
   
   **Find**:
   ```python
   def generate_calibrated_predictions(model, calibrator, X_test):
       y_pred_proba_uncal = model.predict_proba(X_test)[:, 1]
       y_pred_proba_cal = calibrator.predict_proba(y_pred_proba_uncal)
       return y_pred_proba_cal
   ```
   
   **Replace with**:
   ```python
   def generate_calibrated_predictions(model, scaler, calibrator, X_test):
       # Scale features first
       X_test_scaled = scaler.transform(X_test)
       # Get uncalibrated predictions
       y_pred_proba_uncal = model.predict_proba(X_test_scaled)[:, 1]
       # Apply calibration
       y_pred_proba_cal = calibrator.predict_proba(y_pred_proba_uncal)
       return y_pred_proba_cal
   ```

5. **Update argument parser** (around line 100):
   
   **Find**:
   ```python
   parser.add_argument(
       '--model-repo-id',
       default='auphong2707/hospital-readmission-lgbm-calibrated',
       ...
   )
   parser.add_argument(
       '--output-dir',
       default='./phase-5-fairness-evaluation/outputs',
       ...
   )
   parser.add_argument(
       '--phase4-summary',
       default='./phase-4-optimal-threshold-ROI-analysis/outputs/phase4_summary_for_phase5.json',
       ...
   )
   ```
   
   **Replace with**:
   ```python
   parser.add_argument(
       '--model-repo-id',
       default='auphong2707/hospital-readmission-lr-calibrated',
       ...
   )
   parser.add_argument(
       '--output-dir',
       default='./phase-5-fairness-evaluation/outputs_logistic_regression',
       ...
   )
   parser.add_argument(
       '--phase4-summary',
       default='./phase-4-optimal-threshold-ROI-analysis/outputs_logistic_regression/phase4_summary_for_phase5.json',
       ...
   )
   ```

6. **Update main() function** (around line 600):
   
   **Find**:
   ```python
   model, calibrator = load_calibrated_model_and_calibrator(...)
   ...
   y_pred_proba = generate_calibrated_predictions(model, calibrator, X_test)
   ```
   
   **Replace with**:
   ```python
   model, scaler, calibrator = load_calibrated_model_and_calibrator(...)
   ...
   y_pred_proba = generate_calibrated_predictions(model, scaler, calibrator, X_test)
   ```

7. **Update HuggingFace upload** (around line 680):
   
   **Find**:
   ```python
   repo_url = upload_results_to_hf(
       output_dir=args.output_dir,
       repo_id='auphong2707/hospital-readmission-gradient-boosting-fairness-results',
       ...
   )
   ```
   
   **Replace with**:
   ```python
   repo_url = upload_results_to_hf(
       output_dir=args.output_dir,
       repo_id='auphong2707/hospital-readmission-logistic-regression-fairness-results',
       ...
   )
   ```

8. **Update print statements** (throughout):
   - Find all: "Gradient Boosting"
   - Replace with: "Logistic Regression"

---

## Phase 6: calculate_group_thresholds_logistic_regression.py

### Step-by-Step Instructions

1. **Copy the GBM version**:
   ```bash
   cp phase-6-fairness-mitigation-bias-correction/calculate_group_thresholds_gradient_boosting.py \
      phase-6-fairness-mitigation-bias-correction/calculate_group_thresholds_logistic_regression.py
   ```

2. **Update the docstring** (lines 1-40):
   - Replace "Gradient Boosting" with "Logistic Regression"
   - Update example usage:
     ```bash
     python calculate_group_thresholds_logistic_regression.py
     ```

3. **Update argument parser** (around line 90):
   
   **Find**:
   ```python
   parser.add_argument(
       '--phase5-summary',
       default='./phase-5-fairness-evaluation/outputs/phase5_summary_for_phase6.json',
       ...
   )
   parser.add_argument(
       '--model-repo-id',
       default='auphong2707/hospital-readmission-lgbm-calibrated',
       ...
   )
   parser.add_argument(
       '--output-dir',
       default='./phase-6-fairness-mitigation/outputs',
       ...
   )
   ```
   
   **Replace with**:
   ```python
   parser.add_argument(
       '--phase5-summary',
       default='./phase-5-fairness-evaluation/outputs_logistic_regression/phase5_summary_for_phase6.json',
       ...
   )
   parser.add_argument(
       '--model-repo-id',
       default='auphong2707/hospital-readmission-lr-calibrated',
       ...
   )
   parser.add_argument(
       '--output-dir',
       default='./phase-6-fairness-mitigation/outputs_logistic_regression',
       ...
   )
   ```

4. **Update model loading** (around line 180):
   
   **Find**:
   ```python
   def load_model_and_calibrator(...):
       model = joblib.load(model_path)
       calibrator = ModelCalibrator.load(calibrator_path)
       return model, calibrator
   ```
   
   **Replace with**:
   ```python
   def load_model_and_calibrator(...):
       # Load model
       model = joblib.load(model_path)
       # Load scaler
       scaler_path = model_path.parent / "logistic_regression_scaler.joblib"
       scaler = joblib.load(scaler_path)
       # Load calibrator
       calibrator = ModelCalibrator.load(calibrator_path)
       return model, scaler, calibrator
   ```

5. **Update prediction generation** (around line 220):
   
   **Find**:
   ```python
   def generate_calibrated_predictions(model, calibrator, X_test):
       y_pred_proba_uncalibrated = model.predict_proba(X_test)[:, 1]
       ...
   ```
   
   **Replace with**:
   ```python
   def generate_calibrated_predictions(model, scaler, calibrator, X_test):
       # Scale features
       X_test_scaled = scaler.transform(X_test)
       # Get predictions
       y_pred_proba_uncalibrated = model.predict_proba(X_test_scaled)[:, 1]
       ...
   ```

6. **Update main() function** (around line 400):
   
   **Find**:
   ```python
   model, calibrator = load_model_and_calibrator(...)
   ...
   y_pred_proba = generate_calibrated_predictions(model, calibrator, X_test)
   ```
   
   **Replace with**:
   ```python
   model, scaler, calibrator = load_model_and_calibrator(...)
   ...
   y_pred_proba = generate_calibrated_predictions(model, scaler, calibrator, X_test)
   ```

7. **Update HuggingFace upload** (around line 480):
   
   **Find**:
   ```python
   repo_url = upload_results_to_hf(
       output_dir=args.output_dir,
       repo_id='auphong2707/hospital-readmission-gradient-boosting-mitigation-results',
       ...
   )
   ```
   
   **Replace with**:
   ```python
   repo_url = upload_results_to_hf(
       output_dir=args.output_dir,
       repo_id='auphong2707/hospital-readmission-logistic-regression-mitigation-results',
       ...
   )
   ```

8. **Update print statements** (throughout):
   - Find all: "Gradient Boosting"
   - Replace with: "Logistic Regression"

---

## Quick Search & Replace Summary

For both Phase 5 and Phase 6:

| Find | Replace |
|------|---------|
| `Gradient Boosting` | `Logistic Regression` |
| `gradient_boosting` | `logistic_regression` |
| `gradient-boosting` | `logistic-regression` |
| `/outputs/` | `/outputs_logistic_regression/` |
| `lgbm` | `lr` |
| `hospital-readmission-lgbm` | `hospital-readmission-lr` |

**Model Loading Pattern** (add after loading model):
```python
# Add this line right after loading model:
scaler_path = model_path.parent / "logistic_regression_scaler.joblib"
scaler = joblib.load(scaler_path)

# Update all function signatures to include scaler:
# Before: def func(model, calibrator, X):
# After:  def func(model, scaler, calibrator, X):

# Add scaling before predictions:
# Before: y_proba = model.predict_proba(X)[:, 1]
# After:  X_scaled = scaler.transform(X)
#         y_proba = model.predict_proba(X_scaled)[:, 1]
```

---

## Testing

After creating both files:

1. **Verify imports**:
   ```bash
   python -m py_compile phase-5-fairness-evaluation/evaluate_fairness_logistic_regression.py
   python -m py_compile phase-6-fairness-mitigation-bias-correction/calculate_group_thresholds_logistic_regression.py
   ```

2. **Run Phase 5**:
   ```bash
   python ./phase-5-fairness-evaluation/evaluate_fairness_logistic_regression.py
   ```
   - Check for outputs in `outputs_logistic_regression/`

3. **Run Phase 6**:
   ```bash
   python phase-6-fairness-mitigation-bias-correction/calculate_group_thresholds_logistic_regression.py
   ```
   - Check for outputs in `outputs_logistic_regression/`

---

## Why This Approach Works

The **only** difference between GBM and LR in these phases is:
1. **Feature Scaling**: LR needs `scaler.transform(X)` before predictions
2. **File Paths**: Different output directories
3. **Model Type Labels**: Display names for reports

Everything else (fairness metrics, statistical tests, visualizations, threshold optimization) is **identical** because they operate on predicted probabilities, not the model internals.

---

## Estimated Time

- Phase 5: 15-20 minutes (mainly search & replace + testing)
- Phase 6: 10-15 minutes (smaller file, same pattern)
- **Total: ~30 minutes**

---

## Alternative: Use Scripts

If you prefer, I can generate complete Phase 5 and Phase 6 files in the next interaction. Just confirm if you want:
1. Full implementation of Phase 5 (~700 lines)
2. Full implementation of Phase 6 (~500 lines)

Or you can follow this guide to do it manually (which is faster and gives you more control).
