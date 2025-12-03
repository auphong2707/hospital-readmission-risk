# Phase 5: Fairness Evaluation & Deployment Readiness

## Purpose

Phase 5 is the final ethical and operational validation stage before deploying the readmission risk model. It evaluates whether the model's predictions and resulting interventions are equitable across protected demographic groups (race, gender, age), quantifies any disparities, proposes mitigation strategies (if needed), and packages the model, documentation, and monitoring for production.

**Who this is for:** Data scientists, ML engineers, clinical leads, and stakeholders responsible for safe and fair deployment of predictive models in healthcare.


**What this contains:** Meaning, goals, definitions, step-by-step evaluation plan, required inputs/outputs, critical blocker, mitigation options, run instructions, and monitoring recommendations.

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

**Critical Problem (Blocker) — Demographic Preservation**

- **Current state**: The Phase 1 preprocessing pipeline encodes `race`, `gender`, and `age` into engineered features (one-hot / target encoding / numeric codes) and by default does NOT preserve the original categorical columns in the exported splits used by Phases 2–4.
- **Why this is critical**: Fairness evaluation requires the ORIGINAL categorical demographic labels for grouping. Encoded features are insufficient because group membership may be distributed across multiple encoded columns and some encoding (target encoding) may leak target information or be difficult to invert reliably.

**Required Fix** (Actionable):

1. Update Phase 1 preprocessing to save a demographics file aligned with the split rows, e.g. `data/processed/splits/test_demographics.csv` containing at minimum: `encounter_id`, `patient_nbr` (if available), `race`, `gender`, `age`.
2. Upload `test_demographics.csv` (and `train_demographics.csv`, `val_demographics.csv` if desired) to the Phase 1 HuggingFace dataset repo so later phases can download the file using `hf_hub_download()`.

**Example snippet to add to `simple_preprocessing.py` (Phase 1)**

```python
# After creating splits and before saving CSVs
splits['X_test_demographics'] = raw_test[['encounter_id','patient_nbr','race','gender','age']]
splits['X_test_demographics'].to_csv(os.path.join(output_dir,'splits','test_demographics.csv'), index=False)
```

**Without this fix, Phase 5 cannot compute group-specific metrics and is blocked.**

---

**Run / Usage (example commands)**

- Run the fairness evaluation script (example, to be implemented in `phase-5`):

```powershell
# From project root
.venv\Scripts\activate
python ./phase-5-fairness-evaluation-deployment-readiness/evaluate_fairness_gradient_boosting.py \
  --data-repo-id auphong2707/hospital-readmission-risk-data \
  --model-repo-id auphong2707/hospital-readmission-lgbm-calibrated \
  --phase4-summary ./phase-4-optimal-threshold-ROI-analysis/outputs/phase4_summary_for_phase5.json \
  --output-dir ./phase-5-fairness-evaluation-deployment-readiness/outputs
```

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

