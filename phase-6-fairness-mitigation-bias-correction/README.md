# Phase 6: Fairness Mitigation & Bias Correction

## Purpose

Phase 6 addresses fairness violations detected in Phase 5 by implementing **post-hoc bias correction** through group-specific decision thresholds. This phase ensures the hospital readmission prediction model performs equitably across demographic groups (race, gender, age) while maintaining acceptable performance and ROI trade-offs.

**Key Objectives:**
- Calculate optimal thresholds per demographic group to equalize fairness metrics (TPR/FPR)
- Quantify performance and ROI impact of fairness mitigation strategies
- Obtain clinical and ethics team approval for deployment strategy
- Package fairness-aware decision rules for production deployment

**When to Run Phase 6:**
- **Required**: If Phase 5 detects fairness violations (`bias_detected: true` in `phase5_summary_for_phase6.json`)
- **Priority-driven**: Use `mitigation_priority` field (high/medium/low) to determine urgency
- **Optional**: Skip if Phase 5 shows no significant bias (`bias_detected: false`)

---

## What is Fairness Mitigation?

### The Problem
Even well-calibrated models (Phase 3) with optimal thresholds (Phase 4) can exhibit **disparate performance** across demographic groups:
- Higher false negative rates for certain racial groups (missing at-risk patients)
- Higher false positive rates for certain age groups (over-intervention)
- Unequal intervention rates across gender groups (unfair resource allocation)

### The Solution: Group-Specific Thresholds
Instead of using one global threshold for all patients, assign different thresholds per demographic group:

```
Example:
├── Global threshold (Phase 4): 0.52
├── White patients: 0.52 (no adjustment)
├── African American patients: 0.48 (lower to catch more positives)
├── Hispanic patients: 0.50 (slight adjustment)
└── Asian patients: 0.54 (slight adjustment)
```

**Why this works:**
- Each group gets a threshold optimized for fairness metrics (equalized TPR/FPR)
- Preserves model calibration from Phase 3
- No retraining required (post-hoc adjustment)
- Clinically interpretable and auditable

---

## Input Requirements

### Required Files from Phase 5:
1. **`phase5_summary_for_phase6.json`** (Primary decision file)
   - `requires_mitigation`: Boolean flag
   - `mitigation_priority`: 'high', 'medium', or 'low'
   - `worst_violations`: List of fairness violations by attribute
   - `group_metrics_summary`: Performance by demographic group
   - `optimal_threshold`: Global threshold from Phase 4

2. **`fairness_report.json`** (Detailed metrics)
   - Demographic parity, equalized odds, equal opportunity results
   - Statistical significance tests

3. **`group_metrics_*.csv`** (Group-level performance)
   - TPR, FPR, precision, recall by race/gender/age
   - Sample sizes and intervention rates

4. **Phase 4 Results** (via Phase 5 summary)
   - ROI metrics (cost matrix, expected value)
   - Risk category definitions

### Required Data:
- **Test data**: `test.csv` from Phase 1 (via HuggingFace Hub or local)
- **Demographics**: `test_demographics.csv` from Phase 1
- **Model artifacts**: Calibrated model and calibrator from Phase 3

---

## Methodology

### Step 1: Load Phase 5 Decision Inputs

**Script**: `calculate_group_thresholds.py`

```bash
python phase-6-fairness-mitigation/calculate_group_thresholds.py \
    --phase5-summary ./phase-5-fairness-evaluation/outputs/phase5_summary_for_phase6.json \
    --output-dir ./phase-6-fairness-mitigation/outputs
```

**What it does:**
- Loads Phase 5 fairness violations and group metrics
- Checks `requires_mitigation` flag
- If `false`, exits with message: "No mitigation needed, proceed to Phase 7"
- If `true`, proceeds to threshold optimization

**Key checks:**
```python
# Example decision logic
if not phase5_summary['requires_mitigation']:
    print("✅ No fairness violations detected in Phase 5")
    print("Skip Phase 6, proceed to Phase 7 with global threshold")
    exit(0)

priority = phase5_summary['mitigation_priority']
print(f"⚠️ Mitigation required - Priority: {priority.upper()}")
```

---

### Step 2: Calculate Group-Specific Thresholds

**Strategy: Equalized Odds**

This phase uses the **equalized odds** strategy exclusively to minimize both TPR (True Positive Rate) and FPR (False Positive Rate) disparities across demographic groups.

**Optimization Goal:**
```
Objective: min(TPR_gap + FPR_gap)
Constraint: |group_TPR - target_TPR| + |group_FPR - target_FPR| minimized
```

**Algorithm:**
```python
for each demographic_group:
    # Grid search over wide threshold range [0.01, 0.99]
    # Same range as Phase 4 for consistency
    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred_group = (y_pred_proba[group_mask] >= threshold).astype(int)
        
        # Calculate group-specific TPR/FPR
        group_tpr = recall_score(y_true[group_mask], y_pred_group)
        group_fpr = false_positive_rate(y_true[group_mask], y_pred_group)
        
        # Calculate combined gap from target metrics
        tpr_gap = abs(group_tpr - target_tpr)
        fpr_gap = abs(group_fpr - target_fpr)
        score = tpr_gap + fpr_gap
        
    # Select threshold with minimum combined gap
    optimal_threshold_per_group[group] = best_threshold
```

**Threshold Search Configuration:**
- **Range**: [0.01, 0.99] (99 thresholds tested)
- **Step size**: 0.01 (configurable via `--threshold-step`)
- **Why wide range?** Allows finding optimal thresholds for underperforming groups (e.g., lower thresholds to boost TPR for Asian patients)

**Command:**
```bash
python calculate_group_thresholds_gradient_boosting.py \
    --threshold-min 0.01 \
    --threshold-max 0.99 \
    --threshold-step 0.01 \
    --fairness-tolerance 0.05
```

---

### Step 3: Evaluate Mitigation Impact

**Script**: `evaluate_mitigation_impact.py`

```bash
python phase-6-fairness-mitigation/evaluate_mitigation_impact.py \
    --group-thresholds ./phase-6-fairness-mitigation/outputs/group_thresholds.json \
    --output-dir ./phase-6-fairness-mitigation/outputs
```

**Metrics to compute:**

#### Before vs After Comparison:
| Metric | Global Threshold | Group-Specific Thresholds | Change |
|--------|------------------|---------------------------|--------|
| Overall ROC-AUC | 0.72 | 0.72 | ✅ No change |
| Overall TPR | 0.68 | 0.65 | ⚠️ -3% |
| Overall FPR | 0.35 | 0.37 | ⚠️ +2% |
| TPR Gap (max-min) | 0.12 | 0.04 | ✅ -8% |
| FPR Gap (max-min) | 0.08 | 0.03 | ✅ -5% |
| Demographic Parity | ❌ FAILED | ✅ PASSED | ✅ Fixed |
| Equalized Odds | ❌ FAILED | ✅ PASSED | ✅ Fixed |
| Expected Value (ROI) | $2.5M | $2.3M | ⚠️ -8% |

#### Trade-off Analysis:
1. **Fairness improvement**: Measure reduction in TPR/FPR gaps
2. **Performance cost**: Quantify change in overall accuracy, precision, recall
3. **ROI impact**: Calculate change in expected value and net benefit
4. **Intervention volume**: Track change in total patients receiving interventions

**Acceptable trade-offs:**
- Fairness gap reduced by ≥50%
- Overall performance drop ≤5%
- ROI reduction ≤10%
- Clinical team approval

---

### Step 4: Generate Visualizations

**Visualization outputs** (`./outputs/visualizations/`):

#### 1. Before/After Fairness Comparison
```
├── tpr_by_race_before_after.png       # TPR comparison by race
├── fpr_by_race_before_after.png       # FPR comparison by race
├── intervention_rate_before_after.png # Intervention rate by group
└── fairness_metrics_radar.png         # Radar plot of all fairness metrics
```

#### 2. Threshold Distribution
```
├── group_thresholds_barplot.png       # Threshold per group
└── threshold_adjustments_heatmap.png  # Heatmap of adjustments
```

#### 3. Performance Trade-off Curves
```
├── fairness_performance_tradeoff.png  # TPR gap vs overall recall
└── fairness_roi_tradeoff.png          # Fairness improvement vs ROI
```

**Key insights to visualize:**
- Which groups got threshold adjustments? (higher/lower)
- Did fairness gaps close significantly?
- What's the performance/ROI cost?

---

## Clinical and Ethics Approval

### Presentation Materials

**Package for clinical review** (`./outputs/clinical_approval/`):

1. **`fairness_violations_summary.md`**
   - Which demographic groups were affected
   - Magnitude of fairness violations (TPR/FPR gaps)
   - Statistical significance of disparities

2. **`proposed_mitigation_strategy.md`**
   - Group-specific thresholds proposed
   - Expected fairness improvements
   - Performance/ROI trade-offs

3. **`patient_impact_analysis.md`**
   - Number of patients affected by threshold changes
   - Risk of over-intervention vs under-intervention per group
   - Clinical workflow implications

4. **Visualizations** (all charts from Step 4)

### Approval Process

**Stakeholders:**
- **Clinical Lead**: Validates medical appropriateness
- **Ethics Committee**: Reviews fairness and equity implications
- **Hospital Administration**: Approves ROI trade-offs
- **Legal/Compliance**: Ensures regulatory compliance

**Decision matrix:**
```
├── Approve Global Threshold (no mitigation)
│   └── If: fairness violations acceptable, avoid complexity
│
├── Approve Group-Specific Thresholds
│   └── If: fairness violations significant, trade-offs acceptable
│
├── Request Alternative Mitigation
│   └── If: proposed strategy unacceptable, need different approach
│
└── Reject Deployment
    └── If: fairness cannot be resolved, model not ready
```

### Documentation Requirements

**Record in `clinical_approval.md`:**
- Meeting date and attendees
- Fairness violations presented
- Mitigation strategy proposed
- Trade-offs discussed (performance, ROI, patient impact)
- Decision reached (approved/modified/rejected)
- Rationale for decision
- Approval signatures
- Audit trail for regulatory compliance

---

## Output Files

### 1. `group_thresholds.json`
```json
{
  "mitigation_strategy": "equalized_odds",
  "global_threshold": 0.52,
  "group_specific_thresholds": {
    "race": {
      "Caucasian": 0.52,
      "AfricanAmerican": 0.48,
      "Hispanic": 0.50,
      "Asian": 0.54,
      "Other": 0.51
    },
    "gender": {
      "Male": 0.51,
      "Female": 0.53
    },
    "age": {
      "[0-30)": 0.55,
      "[30-50)": 0.52,
      "[50-70)": 0.51,
      "[70-100)": 0.49
    }
  },
  "target_metrics": {
    "tpr_gap_threshold": 0.05,
    "fpr_gap_threshold": 0.05
  }
}
```

### 2. `mitigation_impact.json`
```json
{
  "phase": 6,
  "mitigation_strategy": "group_specific_thresholds",
  "baseline_metrics": {
    "overall_tpr": 0.68,
    "overall_fpr": 0.35,
    "tpr_gap": 0.12,
    "fpr_gap": 0.08,
    "expected_value": 2500000
  },
  "mitigated_metrics": {
    "overall_tpr": 0.65,
    "overall_fpr": 0.37,
    "tpr_gap": 0.04,
    "fpr_gap": 0.03,
    "expected_value": 2300000
  },
  "improvements": {
    "tpr_gap_reduction": -0.08,
    "fpr_gap_reduction": -0.05,
    "demographic_parity_passed": true,
    "equalized_odds_passed": true
  },
  "trade_offs": {
    "overall_tpr_change": -0.03,
    "overall_fpr_change": 0.02,
    "expected_value_change": -200000,
    "roi_reduction_percent": -8.0
  },
  "clinical_approval": {
    "status": "approved",
    "approved_by": "Dr. Jane Smith (Clinical Lead)",
    "approval_date": "2024-12-04",
    "notes": "Fairness improvements justify minor ROI reduction"
  }
}
```

### 3. `clinical_approval.md`
Comprehensive approval document with:
- Executive summary
- Fairness violations identified
- Mitigation strategy proposed
- Trade-off analysis
- Clinical team decision
- Implementation plan
- Monitoring requirements

---

## Usage Examples

### Example 1: Full Phase 6 Pipeline

```bash
# Step 1: Calculate group-specific thresholds using equalized odds
python phase-6-fairness-mitigation-bias-correction/calculate_group_thresholds_gradient_boosting.py \
    --phase5-summary ./phase-5-fairness-evaluation/outputs/phase5_summary_for_phase6.json \
    --threshold-min 0.01 \
    --threshold-max 0.99 \
    --threshold-step 0.01 \
    --fairness-tolerance 0.05 \
    --output-dir ./phase-6-fairness-mitigation-bias-correction/outputs

# Step 2: Evaluate mitigation impact
python phase-6-fairness-mitigation-bias-correction/evaluate_mitigation_impact.py \
    --group-thresholds ./phase-6-fairness-mitigation-bias-correction/outputs/group_thresholds.json \
    --phase5-summary ./phase-5-fairness-evaluation/outputs/phase5_summary_for_phase6.json \
    --output-dir ./phase-6-fairness-mitigation-bias-correction/outputs

# Step 3: Generate clinical approval package
python phase-6-fairness-mitigation-bias-correction/generate_approval_package.py \
    --mitigation-impact ./phase-6-fairness-mitigation-bias-correction/outputs/mitigation_impact.json \
    --output-dir ./phase-6-fairness-mitigation-bias-correction/outputs/clinical_approval
```

### Example 2: Skip Phase 6 (No Mitigation Needed)

```bash
# Check Phase 5 summary
python -c "
import json
with open('./phase-5-fairness-evaluation/outputs/phase5_summary_for_phase6.json') as f:
    summary = json.load(f)
    
if not summary['requires_mitigation']:
    print('✅ No mitigation needed')
    print('Proceed directly to Phase 7 with global threshold')
else:
    print(f'⚠️ Mitigation required - Priority: {summary[\"mitigation_priority\"]}')
"
```

### Example 3: Custom Threshold Search Configuration

```bash
# Use wider search range with finer granularity
python phase-6-fairness-mitigation-bias-correction/calculate_group_thresholds_gradient_boosting.py \
    --threshold-min 0.05 \
    --threshold-max 0.95 \
    --threshold-step 0.005 \
    --fairness-tolerance 0.03 \
    --output-dir ./phase-6-fairness-mitigation-bias-correction/outputs

# This will search 181 thresholds per group (vs default 99)
# Finer granularity may find better group-specific thresholds
# Lower tolerance (3% vs 5%) aims for stricter fairness
```

---

## Key Decisions and Trade-offs

### Decision 1: Mitigation Strategy

**This phase uses Equalized Odds exclusively** - the most comprehensive fairness strategy that balances both True Positive Rate (TPR) and False Positive Rate (FPR) across demographic groups.

**Why Equalized Odds?**
- ✅ Addresses both types of errors (false negatives and false positives)
- ✅ Ensures fair treatment for patients who need intervention (TPR)
- ✅ Ensures fair treatment for patients who don't need intervention (FPR)
- ✅ Clinically appropriate for hospital readmission prediction
- ✅ Aligns with healthcare equity standards

**Alternative strategies** (equal opportunity, demographic parity) were removed to simplify the implementation and focus on the most robust fairness criterion.

### Decision 2: Acceptable Trade-offs

**Performance trade-offs:**
- ✅ Accept: Overall TPR drop ≤5%, FPR increase ≤5%
- ⚠️ Review: Overall TPR drop 5-10%, FPR increase 5-10%
- ❌ Reject: Overall TPR drop >10%, FPR increase >10%

**ROI trade-offs:**
- ✅ Accept: ROI reduction ≤10%
- ⚠️ Review: ROI reduction 10-20%
- ❌ Reject: ROI reduction >20% (consider retraining instead)

**Fairness improvements:**
- ✅ Target: TPR/FPR gaps reduced by ≥50%
- ⚠️ Minimum: TPR/FPR gaps reduced by ≥30%
- ❌ Insufficient: TPR/FPR gaps reduced by <30%

### Decision 3: Global vs Group-Specific Thresholds

**Use global threshold if:**
- Phase 5 shows no significant bias
- Clinical team prefers simplicity
- Fairness gaps are minimal (<3%)

**Use group-specific thresholds if:**
- Phase 5 shows significant bias (gaps >5%)
- Clinical team approves fairness priority
- Trade-offs are acceptable

---

## Integration with Phase 7

### Outputs for Phase 7 Deployment:

1. **`group_thresholds.json`** → Deploy in prediction API
2. **`mitigation_impact.json`** → Include in model card
3. **`clinical_approval.md`** → Regulatory documentation
4. **Visualizations** → Final report appendix

### Prediction Logic for Deployment:

```python
def predict_with_fairness_mitigation(patient_features, demographics, 
                                      model, calibrator, group_thresholds):
    """
    Apply fairness-aware prediction with group-specific thresholds.
    
    Args:
        patient_features: Clinical features (from Phase 1 preprocessing)
        demographics: {race, gender, age} for threshold selection
        model: Trained model (Phase 2)
        calibrator: Platt calibrator (Phase 3)
        group_thresholds: Group-specific thresholds (Phase 6)
    
    Returns:
        prediction: 0 (no intervention) or 1 (intervention)
        probability: Calibrated risk probability
        risk_category: 'low', 'medium', 'high'
        threshold_used: Which threshold was applied
    """
    # Step 1: Get calibrated probability
    proba = model.predict_proba(patient_features)[0, 1]
    calibrated_proba = calibrator.predict_proba([proba])[0]
    
    # Step 2: Select threshold based on demographics
    race = demographics['race']
    threshold = group_thresholds['race'].get(race, group_thresholds['global'])
    
    # Step 3: Apply threshold
    prediction = 1 if calibrated_proba >= threshold else 0
    
    # Step 4: Determine risk category (from Phase 4)
    risk_category = categorize_risk(calibrated_proba, threshold)
    
    return {
        'prediction': prediction,
        'probability': calibrated_proba,
        'risk_category': risk_category,
        'threshold_used': threshold,
        'group': race
    }
```

---

## Success Criteria

Phase 6 is successful if:

✅ **Fairness improvements:**
- TPR gap reduced by ≥50%
- FPR gap reduced by ≥50%
- Demographic parity passed (gap < 5%)
- Equalized odds passed (gaps < 5%)

✅ **Acceptable trade-offs:**
- Overall performance drop ≤5%
- ROI reduction ≤10%
- Clinical team approval obtained

✅ **Documentation complete:**
- Group thresholds calculated and validated
- Mitigation impact quantified
- Clinical approval documented
- Deployment package ready

✅ **Phase 7 readiness:**
- Fairness config packaged for deployment
- Model card updated with mitigation strategy
- Monitoring plan includes fairness metrics

---

## Troubleshooting

### Issue 1: Cannot achieve fairness targets

**Symptoms:**
- TPR/FPR gaps remain >5% even with group-specific thresholds
- Trade-offs exceed acceptable limits (ROI drop >20%)

**Solutions:**
1. Try different mitigation strategies (equal opportunity vs equalized odds)
2. Consider more granular groups (e.g., race × age combinations)
3. Flag for Phase 1-3 retraining (data balancing, fairness-aware loss functions)

### Issue 2: Clinical team rejects proposed thresholds

**Symptoms:**
- Concerns about different treatment for different groups
- Regulatory/legal concerns
- Patient safety concerns

**Solutions:**
1. Present evidence of existing bias (Phase 5 results)
2. Show fairness improvements and limited performance cost
3. Propose alternative: Flag high-disparity cases for manual review
4. Document rejection and escalate to Phase 1-3 retraining

### Issue 3: Phase 5 summary not found

**Error:**
```
FileNotFoundError: phase5_summary_for_phase6.json not found
```

**Solution:**
```bash
# Re-run Phase 5 to generate summary
python phase-5-fairness-evaluation/evaluate_fairness_gradient_boosting.py \
    --output-dir ./phase-5-fairness-evaluation/outputs

# Verify file exists
ls ./phase-5-fairness-evaluation/outputs/phase5_summary_for_phase6.json
```

---

## References

### Fairness Definitions:
- **Demographic Parity**: P(Ŷ=1|A=a) = P(Ŷ=1|A=b) for groups a, b
- **Equalized Odds**: P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b) for all y
- **Equal Opportunity**: P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b)

### Key Papers:
- Hardt et al. (2016): "Equality of Opportunity in Supervised Learning"
- Chouldechova (2017): "Fair Prediction with Disparate Impact"
- Corbett-Davies & Goel (2018): "The Measure and Mismeasure of Fairness"

### Healthcare Fairness Guidelines:
- FDA (2021): "Artificial Intelligence/Machine Learning Software as a Medical Device"
- AMA Code of Medical Ethics: "Use of AI in Health Care"

---

## Next Steps

After completing Phase 6:

1. ✅ **Review outputs**: `group_thresholds.json`, `mitigation_impact.json`, clinical approval
2. ✅ **Verify fairness improvements**: Check before/after metrics
3. ✅ **Obtain stakeholder sign-off**: Clinical lead, ethics committee
4. ➡️ **Proceed to Phase 7**: Deployment preparation with fairness-aware thresholds

**If Phase 6 shows insufficient improvement:**
- Document limitations in clinical approval
- Recommend Phase 1-3 retraining with fairness-aware methods
- Consider alternative mitigation strategies (e.g., fairness constraints during training)
