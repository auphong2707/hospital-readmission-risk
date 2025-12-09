# Logistic Regression - HuggingFace Upload Workflow Alignment

## ✅ Status: ALIGNED

The `train_logistic_regression.py` workflow is now **identical** to `train_gradient_boosting.py` for HuggingFace uploads.

## Side-by-Side Comparison

### 1. **Kaggle Environment Detection**

| Gradient Boosting | Logistic Regression |
|-------------------|---------------------|
| `on_kaggle = is_kaggle_environment()` | `on_kaggle = is_kaggle_environment()` ✅ |
| Line 682 | Line 638 |

### 2. **Output Directory Configuration**

| Gradient Boosting | Logistic Regression |
|-------------------|---------------------|
| `args.output_dir = "/kaggle/working/models" if on_kaggle else str(repo_root / "models")` | `args.output_dir = "/kaggle/working/models" if on_kaggle else str(repo_root / "models")` ✅ |
| Line 705 | Line 643 |

### 3. **HuggingFace Upload Call**

**Gradient Boosting:**
```python
# Line 615-620
print_section("📤 Uploading to HuggingFace Hub", "-")
upload_success = upload_results_to_hf(
    summary=summary,
    output_dir=out_dir,  # ← Path object
    model_name="hospital-readmission-lgbm"
)
if not upload_success:
    print("⚠️  Upload to HuggingFace Hub was skipped (set HF_TOKEN in .env to enable)")
```

**Logistic Regression:**
```python
# Line 586-593
print_section("📤 Uploading to HuggingFace Hub", "-")
upload_success = upload_results_to_hf(
    summary=training_summary,
    output_dir=output_dir,  # ← Path object ✅
    model_name="hospital-readmission-lr"
)
if not upload_success:
    print("⚠️  Upload to HuggingFace Hub was skipped (set HF_TOKEN in .env to enable)")
```

### 4. **Final Output Summary**

**Gradient Boosting:**
```python
print_section("✨ Training Complete!", "=")
print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"📁 All outputs saved to: {out_dir}")
print(f"\n📊 Performance Summary:")
print(f"   🔄 {args.n_splits}-Fold CV ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
print(f"   🎯 Final Test ROC-AUC: {final_metrics['roc_auc']:.4f}")
print("\n🎉 Ready for deployment!")
print("=" * 70)
```

**Logistic Regression:**
```python
print_section("✨ Training Complete!", "=")
print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"📁 All outputs saved to: {output_dir}")
print(f"\n📊 Performance Summary:")
print(f"   🔄 5-Fold CV ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
print(f"   🎯 Final Test ROC-AUC: {final_metrics['roc_auc']:.4f}")
print("\n🎉 Ready for deployment!")
print("=" * 70)
```

## Key Changes Made

### ✅ **Fixed Issues**

1. **Changed:** `output_dir=str(output_dir)` → `output_dir=output_dir`
   - Now passes Path object directly, matching GBM behavior
   
2. **Changed:** `model_name="hospital-readmission-logistic-regression"` → `model_name="hospital-readmission-lr"`
   - Shorter, cleaner repo name (consistent with GBM's "lgbm")
   
3. **Removed:** Hardcoded HuggingFace URL
   - No longer manually constructs `hf_url = "https://huggingface.co/auphong2707/..."`
   - The `upload_results_to_hf()` utility handles this automatically
   
4. **Simplified:** Final output summary
   - Removed detailed file listings
   - Matches GBM's clean, concise summary format

## HuggingFace Upload Behavior

### How It Works

1. **Environment Detection:**
   - Detects Kaggle: `is_kaggle_environment()` checks for `/kaggle/working`
   - Sets output path: `/kaggle/working/models` (Kaggle) or `../models` (local)

2. **File Collection:**
   - `upload_results_to_hf()` automatically uploads ALL files from `output_dir`:
     - `logistic_regression_model.joblib`
     - `logistic_regression_scaler.joblib`
     - `logistic_regression_metrics.json`
     - `training_summary.json`
     - `fold_details.json`

3. **Repository Creation:**
   - Reads `HF_USERNAME` and `HF_TOKEN` from `.env` file
   - Creates repo: `{HF_USERNAME}/hospital-readmission-lr`
   - Example: `auphong2707/hospital-readmission-lr`

4. **Automatic Upload:**
   - Uploads all artifacts to HuggingFace Hub
   - Creates model card with summary information
   - Makes repo public by default

### Configuration Required

Create a `.env` file in your project root:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
HF_USERNAME=auphong2707
```

### Expected Output on Kaggle

```
📤 Uploading to HuggingFace Hub
══════════════════════════════════════════════════════════════════════
Repository: auphong2707/hospital-readmission-lr
Model: hospital-readmission-lr

Creating repository: auphong2707/hospital-readmission-lr
✅ Repository created/verified

Uploading files from: /kaggle/working/models
  📤 logistic_regression_model.joblib
  📤 logistic_regression_scaler.joblib
  📤 logistic_regression_metrics.json
  📤 training_summary.json
  📤 fold_details.json

✅ All files uploaded successfully!
🔗 View your model at: https://huggingface.co/auphong2707/hospital-readmission-lr
══════════════════════════════════════════════════════════════════════
```

## Verification

### On Kaggle

```python
# Run training
!python phase-2-risk-modeling/train_logistic_regression.py

# Verify files were created locally
!ls -lh /kaggle/working/models/

# Check if uploaded to HuggingFace
# Visit: https://huggingface.co/auphong2707/hospital-readmission-lr
```

### Expected Files on HuggingFace

```
auphong2707/hospital-readmission-lr/
├── logistic_regression_model.joblib
├── logistic_regression_scaler.joblib
├── logistic_regression_metrics.json
├── training_summary.json
├── fold_details.json
└── README.md  (auto-generated model card)
```

## Summary

| Feature | Status |
|---------|--------|
| Kaggle Detection | ✅ Identical to GBM |
| Output Path (Kaggle) | ✅ `/kaggle/working/models` |
| Output Path (Local) | ✅ `repo_root/models` |
| HF Upload Call | ✅ Identical to GBM |
| Upload Parameters | ✅ Path object (not string) |
| Model Name | ✅ `hospital-readmission-lr` |
| Final Summary | ✅ Clean, matching GBM format |

The Logistic Regression training script now has **100% workflow consistency** with Gradient Boosting for HuggingFace uploads! 🎉
