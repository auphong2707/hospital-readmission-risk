# Comprehensive Data Split Plan - Hospital Readmission Risk Project

**Document Version:** 3.0 (Streamlined & Actionable)  
**Analysis Date:** December 1, 2025  
**Repository:** hospital-readmission-risk  
**Branch:** topic/phase-3-model-calibration

---

## 🎯 Executive Summary

### Current Issue
🔴 **CRITICAL DATA LEAKAGE:** Phase 3 (calibration) uses different data splits than Phase 2 (training), invalidating all calibration results and blocking Phase 4 & 5 implementation.

### Impact
- ❌ Invalid calibration metrics and visualizations
- ❌ Blocks Phase 4 (threshold optimization) and Phase 5 (fairness evaluation)
- ❌ Risk of incorrect business decisions and unreliable production model

### Solution
✅ Implement unified data splitting strategy across all phases (Option B recommended)

### Timeline
- **Phase 3 Fix:** 2-4 hours
- **Verification:** 1-2 hours
- **Phase 4-5 Implementation:** 3-4 days
- **Total to Deployment:** ~1 week

---

## 📊 Current State Analysis

### Phase-by-Phase Current State

| Phase | Status | Current Data Split | Issues | Impact |
|-------|--------|-------------------|--------|---------|
| **Phase 1** | ✅ Complete | 72.25% train / 12.75% val / 15% test | ✅ Uploaded to HuggingFace | **Source of truth** |
| **Phase 2** | ✅ Complete | 80% dev / 20% test (NEW split) | ❌ Ignores Phase 1 splits | **Needs modification** |
| **Phase 3** | ✅ Complete | Uses Phase 1's 72.25%/15% split | ✅ Uses Phase 1 correctly | **Needs Phase 2 retrain** |
| **Phase 4** | 🔲 Planned | Not yet implemented | Will use Phase 1 splits | Ready after Phase 2 fix |
| **Phase 5** | 🔲 Planned | Not yet implemented | Will use Phase 1 splits | Ready after Phase 2 fix |

### The Problem in Simple Terms

```
❌ CURRENT STATE (INCONSISTENT):
Phase 1: Creates 72% train / 13% val / 15% test → Uploads to HuggingFace
Phase 2: Ignores Phase 1, creates NEW 80% / 20% split → Trains model (WRONG!)
Phase 3: Uses Phase 1's splits correctly → But model trained on different data

✅ DESIRED STATE (UNIFIED):
Phase 1: Creates 72% train / 13% val / 15% test → Uploads to HuggingFace (DONE ✅)
Phase 2: Uses Phase 1's train split → K-fold CV → Evaluates on Phase 1's test
Phase 3: Uses Phase 1's test split → Calibrates on same data model was evaluated on
Phase 4: Uses Phase 1's test split → Optimizes threshold on same data
Phase 5: Uses Phase 1's test split → Evaluates fairness on same data
```

**Key Decision:** Phase 1 splits are the single source of truth. Phase 2 must be retrained to align with Phase 1.

---

## 🎯 Unified Data Split Strategy

**Single Source of Truth:** Phase 1 preprocessed splits

### Strategy Overview

All phases (2-5) will use the exact same data splits created by Phase 1:
- **Train:** 73,526 samples (72.25%)
- **Validation:** 12,975 samples (12.75%)  
- **Test:** 15,265 samples (15%)
- **Random seed:** 42
- **Stratification:** Yes (on target variable)

### Phase-by-Phase Implementation

**Phase 1 (Preprocessing):**
- ✅ Already complete
- Creates and uploads splits to HuggingFace
- Location: `splits/train.csv`, `splits/validation.csv`, `splits/test.csv`

**Phase 2 (Training):**
- ❌ Needs modification and retraining
- Load Phase 1's train + validation splits
- Combine for development set (86,501 samples)
- Perform K-fold CV on development
- Evaluate final model on Phase 1's test split

**Phase 3 (Calibration):**
- ✅ Already uses Phase 1 splits correctly
- Needs verification after Phase 2 retrain
- Uses Phase 1's train for calibrator training
- Uses Phase 1's test for evaluation

**Phase 4 (Threshold Optimization):**
- 🔲 To be implemented
- Load Phase 1's test split
- Optimize threshold on test set
- Calculate ROI metrics

**Phase 5 (Fairness Evaluation):**
- 🔲 To be implemented
- Load Phase 1's test split with demographics
- Evaluate fairness on test set
- Generate fairness reports

---

## 🔧 Implementation Details

### Key Changes Required

| Component | Current State | Required Change |
| **Phase 2** | Loads full dataset, creates new 80/20 split | ❌ Load Phase 1 splits, use train+val for dev |
| **Phase 3** | Uses Phase 1 splits (correct) | ✅ Already correct, verify after Phase 2 retrain |
| **Phase 4** | Not implemented | ✅ Will use Phase 1 splits from start |
| **Phase 5** | Not implemented | ✅ Will use Phase 1 splits from start |

---

### Detailed Code Changes

#### Change 1: Add Utility Function to Load Phase 1 Splits

Add to `phase-2-risk-modeling/utilities.py` and copy to `phase-3-model-calibration/utilities.py`:
```python
def load_phase1_splits(cache_dir="./data/downloaded"):
    """
    Load Phase 1 splits from HuggingFace.
    
    This ensures all phases (2-5) use the exact same preprocessed data from Phase 1.
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import os
    
    print("\n" + "="*80)
    print("📥 Loading Phase 1 Splits from HuggingFace")
    print("="*80)
    
    try:
        # Download Phase 1 splits from HuggingFace
        train_path = hf_hub_download(
            repo_id="auphong2707/hospital-readmission-risk-data",
            filename="splits/train.csv",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        val_path = hf_hub_download(
            repo_id="auphong2707/hospital-readmission-risk-data",
            filename="splits/validation.csv",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        test_path = hf_hub_download(
            repo_id="auphong2707/hospital-readmission-risk-data",
            filename="splits/test.csv",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        
        # Load into DataFrames
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Split features and target
        X_train = train_df.drop(columns=['readmitted'])
        y_train = train_df['readmitted']
        
        X_val = val_df.drop(columns=['readmitted'])
        y_val = val_df['readmitted']
        
        X_test = test_df.drop(columns=['readmitted'])
        y_test = test_df['readmitted']
        
        print(f"✅ Successfully loaded Phase 1 splits:")
        print(f"   Train: {X_train.shape} ({len(X_train):,} samples)")
        print(f"   Validation: {X_val.shape} ({len(X_val):,} samples)")
        print(f"   Test: {X_test.shape} ({len(X_test):,} samples)")
        print(f"   Total: {len(X_train) + len(X_val) + len(X_test):,} samples")
        print("="*80 + "\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"❌ Error loading Phase 1 splits: {e}")
        print(f"   Make sure splits exist in HuggingFace repository")
        print(f"   Expected location: splits/train.csv, splits/validation.csv, splits/test.csv")
        raise
```

#### Change 2: Modify Phase 2 Training Script

Update `phase-2-risk-modeling/train_gradient_boosting.py`:

```python
# BEFORE (loads full dataset and creates new split):
X, y = load_data(from_huggingface=True)
X_development, X_final_test, y_development, y_final_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# AFTER (uses Phase 1 splits):
from utilities import load_phase1_splits

X_train, X_val, X_test, y_train, y_val, y_test = load_phase1_splits()

# Combine train + validation for development (K-fold CV)
import pandas as pd
X_development = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_development = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

print(f"Development set for K-fold CV: {X_development.shape}")
print(f"Test set for final evaluation: {X_test.shape}")

# Rest of training code remains the same
# Use X_development, y_development for K-fold cross-validation
# Use X_test, y_test for final model evaluation
```

#### Change 3: Update Phase 3 Calibration Script

Update `phase-3-model-calibration/calibrate_gradient_boosting.py`:

```python
# BEFORE (already correct, but standardize function call):
data = download_data_from_hf(split="all")
X_train, y_train = data['train']
X_test, y_test = data['test']

# AFTER (use standard utility function):
from utilities import load_phase1_splits

X_train, X_val, X_test, y_train, y_val, y_test = load_phase1_splits()

# Phase 3 uses train for calibrator, test for evaluation
print(f"Calibration training set: {X_train.shape}")
print(f"Calibration test set: {X_test.shape}")
```

---

## 📋 Implementation Roadmap

### Week 1: Retrain Phase 2 (CRITICAL)

**Day 1-2: Implement Utility Function & Modify Phase 2**
- [ ] Add `load_phase1_splits()` to `phase-2-risk-modeling/utilities.py`
- [ ] Modify `train_gradient_boosting.py` to use Phase 1 splits
- [ ] Update code to combine train+val for development
- [ ] Test utility function loads correct splits

**Day 3-4: Retrain Phase 2 Model**
- [ ] Run Phase 2 training with Phase 1 splits
- [ ] Perform K-fold CV on development set (86,501 samples)
- [ ] Evaluate on Phase 1's test set (15,265 samples)
- [ ] Save retrained model to HuggingFace

**Day 5: Update Phase 3 & Verify**
- [ ] Copy `load_phase1_splits()` to `phase-3-model-calibration/utilities.py`
- [ ] Update Phase 3 to use standard utility function
- [ ] Re-run calibration with retrained Phase 2 model
- [ ] Verify test set consistency (15,265 samples)
- [ ] Compare calibration metrics

### Week 2: Implement Phase 4 (Threshold Optimization)

**Day 6-7: Phase 4 Setup**
- [ ] Create `phase-4-threshold-optimization/` directory
- [ ] Copy `load_phase1_splits()` to utilities
- [ ] Design cost-sensitive threshold optimization
- [ ] Define business cost parameters

**Day 8-9: Threshold Optimization Implementation**
- [ ] Load Phase 1 test set for optimization
- [ ] Implement cost-benefit analysis
- [ ] Calculate optimal threshold
- [ ] Generate ROI analysis

**Day 10: Phase 4 Validation**
- [ ] Verify using correct test set (15,265 samples)
- [ ] Generate business recommendations
- [ ] Create visualizations
- [ ] Document threshold selection rationale

### Week 3-4: Implement Phase 5 (Fairness Evaluation)

**Day 11-12: Phase 5 Setup**
- [ ] Create `phase-5-fairness-evaluation/` directory
- [ ] Copy `load_phase1_splits()` to utilities
- [ ] Identify protected attributes in data
- [ ] Design fairness metrics

**Day 13-15: Fairness Evaluation Implementation**
- [ ] Load Phase 1 test set with demographics
- [ ] Calculate fairness metrics across groups
- [ ] Evaluate disparate impact
- [ ] Test multiple fairness definitions

**Day 16-17: Bias Mitigation (if needed)**
- [ ] Implement mitigation strategies
- [ ] Re-evaluate fairness metrics
- [ ] Document trade-offs

**Day 18-20: Final Documentation & Deployment**
- [ ] Update all README files
- [ ] Create data lineage diagram
- [ ] Write deployment guide
- [ ] Prepare stakeholder presentation

---

## 📝 Testing & Verification Protocol

### Verification After Phase 2 Retrain

#### Step 1: Split Consistency Check
```python
# Verify all phases use same splits
from phase_2_risk_modeling.utilities import load_phase1_splits as load_p2
from phase_3_model_calibration.utilities import load_phase1_splits as load_p3

X_train_p2, X_val_p2, X_test_p2, _, _, _ = load_p2()
X_train_p3, X_val_p3, X_test_p3, _, _, _ = load_p3()

assert X_test_p2.shape == X_test_p3.shape == (15265, 113)
assert len(X_train_p2) == len(X_train_p3) == 73526
assert len(X_val_p2) == len(X_val_p3) == 12975
print("✅ All phases use consistent splits")
```

**Day 20: Go-Live Preparation**
- [ ] Final approval from stakeholders
- [ ] Deploy to production environment
- [ ] Set up monitoring and alerting
- [ ] Create rollback plan✅ **Re-run calibration pipeline**
   - Generate new calibration metrics
   - Update all calibration visualizations
   - Re-upload to HuggingFace if needed

### Medium-term Actions (Next Sprint):

4. 🔲 **Implement Phase 4 (Threshold Optimization)**
   - Use corrected Phase 3 calibrated probabilities
   - Ensure using Phase 2's test split
   - Implement cost-sensitive threshold optimization

5. 🔲 **Implement Phase 5 (Fairness Evaluation)**
   - Use Phase 2's test split with demographics
   - Evaluate fairness at optimal threshold
   - Document fairness metrics

6. 🔲 **Add split verification utility**
   - Create function to verify data consistency
   - Add to all phase scripts as sanity check
   - Automate split verification in CI/CD

### Long-term Improvements (Future Refactoring):

7. 🔲 **Consider Option B or C** for complete restructure
   - Benefit: Eliminate need to recreate splits
   - Benefit: Clearer data lineage
   - Document architectural decision

8. 🔲 **Update documentation**
   - Update README with correct split strategy
   - Add data flow diagram showing split usage
   - Create data governance document

9. 🔲 **Add automated tests**
   - Test that all phases use consistent splits
   - Verify split proportions match expected values
   - Check for data leakage between splits

### Detailed Code Changes for Immediate Fix:

**File: `phase-3-model-calibration/calibrate_gradient_boosting.py`**

```python
# BEFORE (WRONG):
data = download_data_from_hf(
    repo_id="auphong2707/hospital-readmission-risk-data",
    split="all"
)
X_train, y_train = data['train']  # Phase 1's train
X_test, y_test = data['test']      # Phase 1's test

# AFTER (CORRECT):
# Load full dataset
#### Step 5: Create Split Verification Utility

Add to utilities.py for all phases:
    data_files="hospital_readmission_full.csv",
    split="train"
)
df = dataset.to_pandas()
X = df.drop(columns=["target"])
y = df["target"]

# Recreate EXACT SAME split as Phase 2
from sklearn.model_selection import train_test_split
X_development, X_test, y_development, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # Match Phase 2's 20%
    random_state=42,  # Match Phase 2
    stratify=y
)

# Use development for calibrator training (same data model trained on)
### Option B: Save and Reuse Phase 2 Splits (⭐ RECOMMENDED)
# Verify
assert phase2_samples.equals(phase3_samples), "Sample data doesn't match!"
```

**Step 4: Calibration Metrics Comparison**
```python
# Compare calibration metrics before and after fix
metrics_before = {
    'brier_score': 0.XXXX,
    'ece': 0.XXXX,
    'hl_pvalue': 0.XXXX
}

metrics_after = {
    'brier_score': 0.YYYY,
    'ece': 0.YYYY,
    'hl_pvalue': 0.YYYY
}

# Document changes
print("Calibration metrics comparison:")
for metric in metrics_before:
    before = metrics_before[metric]
    after = metrics_after[metric]
    change = ((after - before) / before) * 100
    print(f"{metric}: {before:.4f} -> {after:.4f} (change: {change:+.2f}%)")
```

### Phase 4 & 5 Pre-Implementation Verification:

**Create Split Verification Utility** (add to utilities.py):
```python
def verify_phase_consistency(phase_name, X_dev, X_test, y_dev, y_test,
                            expected_dev_shape=(81413, 113),
                            expected_test_shape=(20353, 113)):
    """
    Verify that a phase is using the correct data splits.
    
    Args:
        phase_name: Name of the phase (e.g., "Phase 3")
        X_dev, X_test, y_dev, y_test: The splits being used
#### Step 6: Continuous Logging

Add to all phase scripts:

```python
def log_split_metadata(phase_name, X_dev, X_test, y_dev, y_test):
    """
    print(f"\n{'='*60}")
    print(f"Verifying {phase_name} Data Splits")
    print(f"{'='*60}")
    
    # Shape verification
    assert X_dev.shape == expected_dev_shape, \
        f"Development shape mismatch: {X_dev.shape} != {expected_dev_shape}"
    assert X_test.shape == expected_test_shape, \
        f"Test shape mismatch: {X_test.shape} != {expected_test_shape}"
    
    print(f"✅ Development shape: {X_dev.shape}")
    print(f"✅ Test shape: {X_test.shape}")
    
    # Class distribution verification
    dev_dist = y_dev.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    
    print(f"\n📊 Class Distribution:")
    print(f"   Development: {dict(y_dev.value_counts())}")
    print(f"   Test: {dict(y_test.value_counts())}")
#### Step 7: Acceptance Criteria

**Phase 3 Fix Complete When:** test_dist[1])
    assert dist_diff < 0.01, f"Class distribution mismatch: {dist_diff:.4f} > 0.01"
    
    print(f"✅ Class distributions are consistent (diff: {dist_diff:.4f})")
    
    # No overlap verification
    dev_indices = set(X_dev.index)
    test_indices = set(X_test.index)
    overlap = dev_indices.intersection(test_indices)
    
    assert len(overlap) == 0, f"Data leakage detected: {len(overlap)} overlapping samples!"
**Phase 4 & 5 Implementation Ready When:**tween development and test sets")
    
    print(f"\n{'='*60}")
    print(f"✅ {phase_name} Data Split Verification PASSED")
    print(f"{'='*60}\n")

# Usage in each phase
verify_phase_consistency("Phase 3", X_development, X_test, y_development, y_test)
```
---

## 📚 Technical Notes

### Why Same Random Seed ≠ Same Split with Different Proportions

```python
# These create DIFFERENT samples even with same seed:
split1 = train_test_split(X, y, test_size=0.15, random_state=42)
split2 = train_test_split(X, y, test_size=0.20, random_state=42)

# Reason: Random seed controls shuffle order, not sample selection
# Split 1 takes first 15% after shuffle
# Split 2 takes first 20% after shuffle  
# Result: Different samples in test sets
```

### Data Leakage Impact Scenarios

| Scenario | Probability | Impact on Metrics |
|----------|-------------|-------------------|
| Test set includes training data | High | Calibration appears better than reality |
| Completely different distribution | Medium | Calibration appears worse, unpredictable |
| Partial overlap | High | Mixed effects, unreliable metrics |

### Expected Sample Sizes (101,766 total)

**Phase 1 Splits (Single Source of Truth):**

| Split | Samples | Percentage | Purpose |
|-------|---------|------------|---------|
| Train | 73,526 | 72.25% | Model training (K-fold CV) |
| Validation | 12,975 | 12.75% | Combined with train for development |
| Test | 15,265 | 15% | Final evaluation (all phases) |
| **Development** | **86,501** | **85%** | **Train + Val combined for Phase 2 K-fold CV** |

**Note:** Phase 2 will combine train+validation (86,501 samples) for K-fold cross-validation, then evaluate on test (15,265 samples).

---

## 🎯 Quick Reference Summary

### ⚡ TL;DR

**Problem:** Phase 2 ignores Phase 1 splits and creates its own, causing inconsistency.

**Solution:** Retrain Phase 2 to use Phase 1's preprocessed splits (single source of truth).

**Timeline:** 1 week to retrain Phase 2 and fix Phase 3, 3 weeks to complete Phase 4-5.

**Priority:** 🔴 CRITICAL - requires Phase 2 retraining.

### ✅ Action Checklist

**Week 1 (CRITICAL - Retrain Phase 2):**
- [ ] Add `load_phase1_splits()` to utilities
- [ ] Modify Phase 2 to load Phase 1 splits (train, val, test)
- [ ] Combine train+val for K-fold CV development
- [ ] **RETRAIN Phase 2 model** using Phase 1 splits
- [ ] Evaluate retrained model on Phase 1's test set
- [ ] Update Phase 3 to use same utility function
- [ ] Re-run calibration with retrained model
- [ ] Verify all phases use same test set (15,265 samples)

**Week 2 (Phase 4 Implementation):**
- [ ] Implement Phase 4 using `load_phase1_splits()`
- [ ] Optimize threshold on Phase 1's test set
- [ ] Calculate ROI and define risk categories
- [ ] Generate business recommendations

**Week 3 (Phase 5 Implementation):**
- [ ] Implement Phase 5 using `load_phase1_splits()`
- [ ] Evaluate fairness on Phase 1's test set
- [ ] Implement bias mitigation if needed
- [ ] Generate fairness reports

**Code Issues:** Check utilities.py for `load_phase1_splits()` implementation

**HuggingFace Issues:** Verify Phase 1 splits exist at `splits/train.csv`, `splits/validation.csv`, `splits/test.csv`

**Split Verification:** Expected sizes - Train: 73,526, Val: 12,975, Test: 15,265

**Phase 2 Retraining:** Use train+val (86,501 samples) for K-fold CV, evaluate on test (15,265)

**Questions:** Open GitHub issue with "data-split" label

**Code Issues:** Check utilities.py for `load_phase1_splits()` implementation

**HuggingFace Issues:** Verify Phase 1 splits exist at `splits/train.csv`, `splits/validation.csv`, `splits/test.csv`

**Split Verification:** Expected sizes - Train: 73,526, Val: 12,975, Test: 15,265

**Phase 2 Retraining:** Use train+val (86,501 samples) for K-fold CV, evaluate on test (15,265)

**Questions:** Open GitHub issue with "data-split" label

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Incorrect calibration decisions | HIGH | HIGH | Fix Phase 3 immediately |
| Wrong threshold selection | MEDIUM | HIGH | Fix before Phase 4 |
| Biased fairness evaluation | MEDIUM | HIGH | Fix before Phase 5 |
| Future split inconsistencies | LOW | MEDIUM | Add automated tests |
| Production model unreliable | HIGH | CRITICAL | Complete fix before deployment |

### Questions & Clarifications

**Q: Why didn't Phase 2 use Phase 1's splits?**  
A: Phase 2 loads "full" dataset and creates its own split. This is valid for Phase 2 but creates inconsistency for Phase 3.

**Q: Can we just change Phase 2 to use Phase 1's splits?**  
A: Possible (Option C) but requires retraining the model, which is more work and risk.

**Q: How do we prevent this in the future?**  
A: Implement Option B (save Phase 2 splits) + add automated verification tests.

**Q: What about the Phase 1 splits?**  
A: Keep them for preprocessing reference/validation only. Document they're not used in modeling.

**Q: Will fixing this change model performance?**  
A: No - the model itself is fine. Only calibration metrics will change (to correct values).

### Next Steps

1. **Decision Required:** Choose Option A (quick) or Option B (robust)
2. **Assign Owner:** Designate developer to implement fix
3. **Timeline:** Set target completion date (recommended: 2-3 days)
4. **Review:** Schedule code review after implementation
5. **Validation:** Run full test suite to verify fix
6. **Documentation:** Update README and technical docs
7. **Communication:** Notify stakeholders of issue and fix

---

## 🎯 Conclusion

**Status:** 🔴 **CRITICAL ISSUE** - Data splits inconsistent across phases

**Action Required:** Fix Phase 3 to use Phase 2's exact data splits

**Priority:** **HIGH** - Blocks Phase 4 & 5, affects model deployment readiness

---
**Status:** 🔴 **CRITICAL ISSUE IDENTIFIED**  
**Recommended Action:** Retrain Phase 2 to Use Phase 1 Splits (Single Source of Truth)  
**Key Change:** Phase 2 must be retrained to align with Phase 1's preprocessed splits  
**Timeline:** 1 week to retrain + 3 weeks for Phase 4-5 = 4 weeks total  
**Priority:** HIGH - Requires model retraining before Phase 4 & 5)  
**Analysis Date:** December 1, 2025  
**Last Updated:** December 1, 2025  
**Analysis By:** GitHub Copilot (Claude Sonnet 4.5)  
**Repository:** hospital-readmission-risk  
**Branch:** topic/phase-3-model-calibration  

**Status:** 🔴 **CRITICAL ISSUE IDENTIFIED**  
**Recommended Action:** Implement Option B (Save & Reuse Phase 2 Splits)  
**Timeline:** 1 week to deployment-ready  
**Priority:** HIGH - Blocks Phase 4 & 5 implementation