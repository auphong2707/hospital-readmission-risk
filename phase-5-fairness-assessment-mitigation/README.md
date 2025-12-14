# Phase 5: Fairness Assessment & Mitigation

## Purpose

Phase 5 performs comprehensive fairness assessment AND conditional mitigation for the hospital readmission risk model. This unified phase evaluates whether the model produces equitable outcomes across protected demographic groups (race, gender, age) and automatically applies fairness mitigation when violations are detected.

**Who this is for:** Data scientists, ML engineers, clinical leads, ethics committees, and stakeholders responsible for safe and fair deployment of predictive models in healthcare.

**What this contains:** 
- Part A: Fairness evaluation (always runs)
- Part B: Fairness mitigation (conditional - runs if violations detected)
- Unified orchestration script for both workflows
- Combined outputs and deployment recommendations

---

**Meaning & Goals**

- **Meaning**: Verify the model does not produce systematically unfair outcomes for protected groups and that deployment decisions (thresholds, risk categories) align with ethical, clinical, and operational constraints.
- **Primary Goals**:
  - Evaluate model performance by protected groups (race, gender, age)
  - Measure fairness with standard metrics (demographic parity, equalized odds, equal opportunity)
  - Determine and apply bias mitigation if disparities are meaningful/ actionable
  - Produce deployment-ready artifacts: calibrated model, thresholds, model card, fairness report, and monitoring scripts

---

**Definitions (concise)**

- **Demographic Parity (Statistical Parity)**: Intervention rate should be similar across groups. Formally: P(ŷ=1 | A=a) ≈ P(ŷ=1) for all groups a.
- **Equalized Odds**: Both TPR and FPR should be similar across groups: P(ŷ=1 | y=1, A=a) and P(ŷ=1 | y=0, A=a) should be equal across a.
- **Equal Opportunity**: TPR equal across groups: P(ŷ=1 | y=1, A=a) ≈ constant.
- **TPR (Recall)**: TP / (TP + FN). Important for ensuring high-risk patients are detected equally.
- **FPR**: FP / (FP + TN). Important for measuring unnecessary interventions.

---

**Inputs (what Phase 5 requires)**

- **Test data splits** (`X_test`, `y_test`) — from Phase 1 splits (HuggingFace dataset or local `data/processed/splits/test.csv`).
- **Demographics** (`race`, `gender`, `age`) — ORIGINAL categorical values aligned to `X_test` indices. (See Critical Blocker below.)
- **Calibrated model** — model + calibrator files produced in Phase 3, e.g. `gradient_boosting_model_original.joblib` and `Gradient_Boosting_(LightGBM)_calibrator.pkl` (HuggingFace model repo).
- **Phase 4 outputs** — `phase4_summary_for_phase5.json`, `optimal_thresholds.json` (or similar), `threshold_results.csv`, `roi_metrics.json`, risk category thresholds (low/medium/high).
- **Business/cost params** — intervention cost, readmission cost (from Phase 4) for fairness-ROI trade-off analysis.

Note: All inputs should be versioned and stored in `phase-4-optimal-threshold-ROI-analysis/outputs` or in the appropriate HuggingFace repos.

---

**Outputs (what Phase 5 produces)**

- `fairness_report.json` (summary metrics and significance tests)
- `fairness_visualizations/` (PNG files): group TPR/FPR/precision bars, calibration curves by group, risk distribution plots, confusion matrices per group, ROI trade-off plots
- `phase5_results_for_deployment.json` (recommended final thresholds, any group-specific thresholds, mitigation applied, net ROI change)
- `model_card.md` (finalized with fairness section and limitations)
- `deployment_package/` (model files, calibrator, preprocessing artifacts, thresholds, readme)
- `monitoring_scripts/` (scripts to compute metrics by group in production)

---

**Step-by-step Procedure**

1. **Load inputs**
   - Load `X_test`, `y_test` via `load_phase1_splits()` or from local `data/processed/splits/test.csv`.
   - Load `demographics_test.csv` (see Critical Blocker) with columns `encounter_id`, `race`, `gender`, `age` mapped to `X_test`.
   - Load calibrated model and calibrator from Phase 3.
   - Load Phase 4 summary file (`phase4_summary_for_phase5.json`) for optimal thresholds.

2. **Produce calibrated probabilities and baseline predictions**
   - Compute calibrated probabilities on `X_test`: `y_pred_proba_calibrated`.
   - Apply optimal global threshold to get `y_pred_global`.
   - Map `y_pred_proba_calibrated` into risk categories using Phase 4 thresholds.

3. **Compute overall metrics**
   - Confusion matrix, Precision, Recall (TPR), FPR, F1, ROC-AUC, Brier score.
   - Record baseline ROI metrics using Phase 4 cost parameters.

4. **Compute group-specific metrics**
   - For each protected attribute (race categories, gender values, age buckets): compute TP, FP, TN, FN and derived metrics (TPR, FPR, Precision, Recall, F1).
   - For each metric, compute 95% confidence intervals (bootstrap or binomial proportion CI).

5. **Statistical testing**
   - Chi-square test for independence for intervention rate across groups.
   - Proportion tests (e.g. two-proportion z-test) to compare TPRs between groups.
   - Record p-values and indicate significance (p < 0.05).

6. **Fairness metrics and thresholds**
   - Compute demographic parity gap: max_group_rate - min_group_rate.
   - Compute TPR/FPR range across groups.
   - Flag metrics exceeding tolerance thresholds (suggest default ±5% tolerance).

7. **Risk-category fairness**
   - Check distribution of Low/Medium/High risk across groups.
   - Verify actual readmission rates within each risk category by group (calibration fairness).

8. **Mitigation experiments (if violation found)**
   - Option A — **Group-specific thresholds**: find per-group thresholds that equalize TPR or FPR (optimize for fairness constraint with minimal ROI loss).
   - Option B — **Group-specific calibration**: calibrate probabilities per group using validation set, then re-evaluate.
   - Option C — **Post-processing methods**: methods such as reject-option classification or cost-sensitive reweighing at decision time.
   - For each mitigation, re-calculate overall ROI and group metrics; produce trade-off table.

9. **Decision & documentation**
   - Decide on final approach: keep global threshold, adopt group-specific thresholds, or require further model retraining.
   - Write `fairness_report.json`, update `model_card.md` with fairness and limitations, include mitigation rationale and ROI trade-off.

10. **Deployment packaging**
   - Bundle `model.joblib`, `calibrator.pkl`, `preprocessing artifacts` (encoders, scaler), `thresholds.json`, `model_card.md`, and `monitoring scripts` into `deployment_package/`.

11. **Clinical validation & sign-off**
   - Present findings and recommended thresholds to clinical team for review and sign-off (attach case examples).

12. **Monitoring plan**
   - Provide `monitoring_scripts/` and metrics to track in production: per-group TPR, FPR, intervention rate, and distribution drift alerts.

---

**Success Criteria (recommended)**

- No group has TPR or FPR differing from the overall metric by more than ±5% (adjustable by stakeholder policy).
- No statistically significant differences in intervention rates (Chi-square p-value > 0.05).
- Net ROI after mitigation remains positive and within acceptable loss tolerances (documented).
- Clinical sign-off obtained for selected deployment policy.

---

**Critical Problem (Blocker) — Demographic Preservation** ✅ **RESOLVED**

- **Previous state**: The Phase 1 preprocessing pipeline encoded `race`, `gender`, and `age` into engineered features without preserving the original categorical columns.
- **Why it was critical**: Fairness evaluation requires the ORIGINAL categorical demographic labels for grouping. Encoded features are insufficient because group membership may be distributed across multiple encoded columns and some encoding (target encoding) may leak target information or be difficult to invert reliably.

**Resolution Applied** ✅:

Phase 1 preprocessing (`simple_preprocessing.py`) has been updated to:
1. **Store demographics before encoding**: Added `_store_demographics()` method that captures original `race`, `gender`, `age` values before any encoding transforms
2. **Save demographics files**: Modified `create_train_test_split()` to save three demographics files aligned with splits:
   - `data/processed/splits/train_demographics.csv`
   - `data/processed/splits/validation_demographics.csv`
   - `data/processed/splits/test_demographics.csv`
3. **Upload to HuggingFace**: Updated `_upload_to_huggingface()` to include demographics files in the Phase 1 dataset repo

**To regenerate demographics files**:

```powershell
# Rerun Phase 1 preprocessing
.venv\Scripts\activate
python phase-1-data-explore-preprocessing/simple_preprocessing.py
```

This will create the demographics files in `data/processed/splits/` and upload them to HuggingFace Hub (if `HF_TOKEN` is configured).

**Note**: Existing Phase 1 outputs remain unchanged. Demographics files are **additional** files that supplement the feature splits.

---

## Unified Workflow (Recommended)

Phase 5 now contains a unified workflow that runs both fairness evaluation AND conditional mitigation:

### Master Orchestration Script

Use the master script to run the complete workflow automatically:

```bash
# Run complete fairness assessment & mitigation for each method
cd phase-5-fairness-assessment-mitigation

./run_fairness_assessment_and_mitigation.sh gradient_boosting
./run_fairness_assessment_and_mitigation.sh random_forest  
./run_fairness_assessment_and_mitigation.sh logistic_regression
```

**What the script does:**
1. **Runs fairness evaluation** (always)
2. **Checks if mitigation needed** (reads phase5_summary_for_phase6.json)
3. **Runs mitigation** (conditionally - only if violations detected)
4. **Generates combined summary** (evaluation + mitigation results)

### Output Structure

```
phase-5-fairness-assessment-mitigation/outputs/{method}/
├── evaluation/                           # Part A: Fairness evaluation
│   ├── fairness_report.json
│   ├── phase5_summary_for_phase6.json
│   ├── statistical_tests.json
│   ├── group_metrics_*.csv
│   └── visualizations/
├── mitigation/                           # Part B: Mitigation (if applied)
│   ├── group_thresholds.json
│   ├── mitigation_impact.json
│   └── visualizations/
└── phase5_complete_summary.json          # Combined final summary
```

---

## Manual Usage (Run Individual Scripts)

If you need to run scripts individually:

### Step 1: Fairness Evaluation

```bash
# Gradient Boosting
python evaluate_fairness_gradient_boosting.py \
  --output-dir ./outputs/gradient_boosting/evaluation

# Random Forest
python evaluate_fairness_random_forest.py \
  --output-dir ./outputs/random_forest/evaluation

# Logistic Regression
python evaluate_fairness_logistic_regression.py \
  --output-dir ./outputs/logistic_regression/evaluation
```

### Step 2: Fairness Mitigation (conditional)

Check `phase5_summary_for_phase6.json`. If `requires_mitigation: true`:

```bash
# Gradient Boosting
python calculate_group_thresholds_gradient_boosting.py \
  --phase5-summary ./outputs/gradient_boosting/evaluation/phase5_summary_for_phase6.json \
  --output-dir ./outputs/gradient_boosting/mitigation

# Random Forest
python calculate_group_thresholds_random_forest.py \
  --phase5-summary ./outputs/random_forest/evaluation/phase5_summary_for_phase6.json \
  --output-dir ./outputs/random_forest/mitigation

# Logistic Regression
python calculate_group_thresholds_logistic_regression.py \
  --phase5-summary ./outputs/logistic_regression/evaluation/phase5_summary_for_phase6.json \
  --output-dir ./outputs/logistic_regression/mitigation
```

### Required Files

- If using local files instead of HuggingFace, ensure the following exist:
  - `data/processed/splits/test.csv`
  - `data/processed/splits/test_demographics.csv` (CRITICAL)
  - `phase-3-model-calibration/outputs/*` (or the HF model repo)

---

**Visualizations (recommended)**

- `group_metrics.png` — bar charts of TPR/FPR/Precision per group
- `calibration_by_group.png` — reliability diagrams per group
- `risk_distribution_by_group.png` — histogram of calibrated probabilities by group
- `confusion_matrices_by_group.png` — side-by-side confusion matrices
- `roi_tradeoff.png` — ROI before/after mitigation per-group
- `risk_category_distribution.png` — stacked bar chart of Low/Medium/High by group

---

**Mitigation Options (concise)**

- **Group-specific thresholds** — Set thresholds per group to equalize TPR or FPR.
- **Group-specific calibration** — Fit Platt scaling per group on validation data.
- **Post-processing techniques** — Reject-option classification or other fairness-aware post-processing to balance errors.
- **Retraining with fairness-aware objective** — Reweight training samples or add fairness constraints to training loss (longer-term option).

Each option requires re-evaluating ROI and documenting trade-offs.

---

**Model Card (must include)**

- **Model description**: architecture, training data, features, preprocessing.
- **Intended use & limitations**: who should use it and who should not.
- **Performance metrics**: overall and by protected groups.
- **Fairness evaluation**: tests run, results, mitigation steps taken.
- **Data provenance**: dataset versions, splits, and date ranges.
- **Monitoring plan**: metrics to track and alert thresholds.

---

**Monitoring & Post-deployment**

- Track per-group TPR/FPR monthly and alert if gap grows beyond policy tolerance.
- Monitor calibration drift by group and retrain or recalibrate if needed.
- Log `encounter_id`, `y_true`, `y_pred_proba`, `y_pred`, and `demographics` (with appropriate privacy controls) for auditing.

---

**Next steps for the repo (recommended)**

- Implement the Phase 5 evaluation script `evaluate_fairness_gradient_boosting.py` in this folder using the procedure above.
- Update Phase 1 preprocessing to save `*_demographics.csv` files and upload them to Phase 1 HF dataset.
- Add unit tests for the fairness metrics calculations and CI checks to ensure no regression.

---

**Contacts & Sign-off**

- Include clinical lead and data privacy officer contacts when preparing for sign-off.
- Require clinical validation meeting to approve risk categories and deployment thresholds.

---

Thank you — after you review this README, I can:
- Implement a skeleton `evaluate_fairness_gradient_boosting.py` script in this folder
- Add visualization helpers and CI tests
- Update Phase 1 preprocessing to export `demographics` files (if you want me to make the change now)

