"""
Phase 3: Calibrate Logistic Regression Model for Hospital Readmission Prediction

This script performs comprehensive calibration of the trained Logistic Regression model from Phase 2.
It downloads both the pre-trained model AND preprocessed data from HuggingFace Hub,
applies calibration techniques, and generates detailed calibration reports with visualizations.

Calibration Pipeline:
1. Download pre-trained model from HuggingFace Hub
2. Download preprocessed data from HuggingFace Hub (same splits as training)
3. Generate uncalibrated predictions
4. Apply Platt Scaling calibration
5. Evaluate calibration quality (Brier score, ECE, Hosmer-Lemeshow test)
6. Create calibration reports and visualizations
7. Save calibrated model for Phase 4 threshold optimization

Calibration Technique:
- Platt Scaling: Logistic regression transformation of predicted probabilities

Validation Methods:
- Reliability Diagrams (Predicted vs Observed probabilities)
- Brier Score (target < 0.15)
- Expected Calibration Error/ECE (target < 0.05 for ±5% diagonal)
- Hosmer-Lemeshow Test (target p-value > 0.05)

Note:
- Risk thresholds and categories will be determined in Phase 4
- This phase focuses purely on calibration quality validation

Usage (from project root):
    python ./phase-3-model-calibration/calibrate_logistic_regression.py

Requirements:
    pip install scikit-learn pandas numpy matplotlib seaborn huggingface_hub joblib

HuggingFace Model:
    Repository: https://huggingface.co/auphong2707/hospital-readmission-phase2-lr
    Contains: Pre-trained Logistic Regression model, training summary, metrics, visualizations
"""

import argparse
import sys
import warnings
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import joblib

# Add parent directory to path to import preprocessing utilities
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utilities import (
    download_model_from_hf,
    load_phase1_splits,
    CalibrationMetrics,
    CalibrationVisualizer,
    calibrate_model_pipeline,
    upload_calibrated_model_to_hf,
    convert_to_serializable
)

warnings.filterwarnings('ignore')


def print_section(title: str, char: str = "=", width: int = 80):
    """Print formatted section header.
    
    Args:
        title: Section title
        char: Border character (default: "=")
        width: Total width of border (default: 80)
    """
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def load_and_preprocess_data(repo_id: str = "auphong2707/hospital-readmission-risk-data",
                             cache_dir: str = "./data/downloaded",
                             force_download: bool = False):
    """
    Load preprocessed data from HuggingFace Hub using Phase 1 splits.
    
    This function uses the standardized load_phase1_splits() to ensure
    Phase 3 uses the exact same data splits as Phase 2 training.
    
    Phase 1 Split Strategy:
    - Train: 73,526 samples (72.25%) - Used for calibrator training
    - Validation: 12,975 samples (12.75%) - Not used in Phase 3
    - Test: 15,265 samples (15%) - Used for calibration evaluation
    
    Args:
        repo_id: HuggingFace dataset repository ID
        cache_dir: Directory to cache downloaded files
        force_download: Not used (kept for backward compatibility)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print_section("📂 Loading Preprocessed Data from HuggingFace", "-")
    
    try:
        # Load Phase 1 splits using standardized function
        print("⏳ Loading Phase 1 splits from HuggingFace Hub...")
        print(f"   Repository: {repo_id}")
        print(f"   ✅ Using standardized load_phase1_splits() function\n")
        
        X_train, X_val, X_test, y_train, y_val, y_test = load_phase1_splits(
            cache_dir=cache_dir,
            repo_id=repo_id
        )
        
        print("✅ Phase 1 splits loaded successfully")
        print(f"\n📊 Calibration Data Usage:")
        print(f"   Train set (for calibrator): {X_train.shape}")
        print(f"   Test set (for evaluation): {X_test.shape}")
        print(f"   Validation set: {X_val.shape} (not used in Phase 3)")
        print(f"\n   Train class distribution: {dict(y_train.value_counts())}")
        print(f"   Test class distribution: {dict(y_test.value_counts())}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"\n❌ Error loading Phase 1 splits: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Ensure preprocessing script has uploaded data to HuggingFace")
        print(f"   2. Check repository exists: https://huggingface.co/datasets/{repo_id}")
        print(f"   3. Verify splits exist: splits/train.csv, splits/validation.csv, splits/test.csv")
        print(f"   4. Try running preprocessing script first:")
        print(f"      python ./phase-1-data-explore-preprocessing/simple_preprocessing.py")
        raise


def generate_uncalibrated_predictions(model, X_train, X_test, y_train, y_test):
    """
    Generate uncalibrated predictions from the trained model.
    
    This creates the baseline predictions that will be calibrated.
    We generate predictions for both train and test sets:
    - Train set: Used to fit the calibrator
    - Test set: Used to evaluate calibration quality
    
    Note: X_train and X_test are already scaled from Phase 1.
    
    Args:
        model: Trained Logistic Regression model
        X_train: Training features (already scaled from Phase 1)
        X_test: Test features (already scaled from Phase 1)
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing train/test predictions and labels
    """
    print_section("🔮 Generating Uncalibrated Predictions", "-")
    
    # X_train and X_test are already scaled from Phase 1 - use directly
    print("ℹ️  Using pre-scaled features from Phase 1...")
    
    # Generate predictions on train set (for fitting calibrator)
    print("\n⏳ Generating train set predictions...")
    y_train_proba = model.predict_proba(X_train)[:, 1]
    print(f"   Train probabilities shape: {y_train_proba.shape}")
    print(f"   Train probability range: [{y_train_proba.min():.4f}, {y_train_proba.max():.4f}]")
    print(f"   Train mean probability: {y_train_proba.mean():.4f}")
    
    # Generate predictions on test set (for evaluation)
    print("\n⏳ Generating test set predictions...")
    y_test_proba = model.predict_proba(X_test)[:, 1]
    print(f"   Test probabilities shape: {y_test_proba.shape}")
    print(f"   Test probability range: [{y_test_proba.min():.4f}, {y_test_proba.max():.4f}]")
    print(f"   Test mean probability: {y_test_proba.mean():.4f}")
    
    # Calculate initial metrics (before calibration)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    train_brier = brier_score_loss(y_train, y_train_proba)
    test_brier = brier_score_loss(y_test, y_test_proba)
    
    print(f"\n📊 Uncalibrated Performance:")
    print(f"   Train ROC-AUC: {train_auc:.4f}")
    print(f"   Test ROC-AUC: {test_auc:.4f}")
    print(f"   Train Brier Score: {train_brier:.4f}")
    print(f"   Test Brier Score: {test_brier:.4f}")
    
    # Package predictions for calibration pipeline
    predictions = {
        'train': {
            'y_true': y_train.values if hasattr(y_train, 'values') else y_train,
            'y_pred_proba': y_train_proba
        },
        'test': {
            'y_pred_proba': y_test_proba
        }
    }
    
    return predictions


def calibrate_model(predictions, y_test, method='platt', output_dir='./calibration_outputs'):
    """
    Apply Platt Scaling calibration to model predictions.
    
    Transforms uncalibrated probabilities into calibrated probabilities
    using logistic regression on the validation set.
    
    Args:
        predictions: Dictionary with train/test predictions
        y_test: Test labels
        method: Calibration method (default: 'platt')
        output_dir: Directory to save calibration outputs
        
    Returns:
        tuple: (calibrated_proba, report)
    """
    print_section(f"🎯 Applying Calibration: {method.upper()}", "=")
    
    print(f"📋 Calibration Configuration:")
    print(f"   Method: {method}")
    print(f"   Train samples: {len(predictions['train']['y_true'])}")
    print(f"   Test samples: {len(predictions['test']['y_pred_proba'])}")
    print(f"   Output directory: {output_dir}")
    
    # Use the comprehensive calibration pipeline
    calibrated_proba, report = calibrate_model_pipeline(
        model_predictions=predictions,
        y_true=y_test,
        model_name='Logistic Regression',
        calibration_method=method,
        output_dir=output_dir
    )
    
    return calibrated_proba, report


def generate_comparison_report(y_test, uncalibrated_proba, calibrated_proba, output_dir):
    """
    Generate detailed before/after calibration comparison.
    
    This creates comprehensive visualizations and metrics comparing
    uncalibrated vs. calibrated model performance.
    
    Args:
        y_test: True labels
        uncalibrated_proba: Uncalibrated probabilities
        calibrated_proba: Calibrated probabilities
        output_dir: Directory to save outputs
    """
    print_section("📊 Generating Comparison Report", "-")
    
    output_path = Path(output_dir)
    
    # Calculate comprehensive metrics
    print("⏳ Computing comprehensive metrics...")
    metrics = CalibrationMetrics.compute_all_metrics(
        y_test, uncalibrated_proba, calibrated_proba
    )
    
    # Display metrics comparison
    print(f"\n{'='*80}")
    print("📈 CALIBRATION METRICS COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<25} {'Uncalibrated':>15} {'Calibrated':>15} {'Improvement':>15}")
    print(f"{'-'*80}")
    
    metrics_to_show = [
        ('Brier Score', 'brier_score', True),
        ('Log Loss', 'log_loss', True),
        ('ROC-AUC', 'roc_auc', False),
        ('ECE (±5% target)', 'ece', True),
    ]
    
    for label, key, lower_better in metrics_to_show:
        uncal_val = metrics['uncalibrated'][key]
        cal_val = metrics['calibrated'][key]
        improvement = metrics['improvement'].get(f'{key}_delta', uncal_val - cal_val)
        
        # Format improvement with appropriate sign
        if lower_better:
            imp_str = f"{improvement:+.4f} {'✓' if improvement > 0 else '✗'}"
        else:
            imp_str = f"{-improvement:+.4f} {'✗' if improvement < 0 else '✓'}"
        
        print(f"{label:<25} {uncal_val:>15.4f} {cal_val:>15.4f} {imp_str:>15s}")
    
    print(f"{'-'*80}\n")
    
    # Hosmer-Lemeshow Test
    print("📐 Hosmer-Lemeshow Goodness-of-Fit Test:")
    hl_uncal = metrics['uncalibrated']['hosmer_lemeshow']
    hl_cal = metrics['calibrated']['hosmer_lemeshow']
    
    print(f"   Uncalibrated: χ²={hl_uncal['chi2_statistic']:.2f}, "
          f"p={hl_uncal['p_value']:.4f} - {hl_uncal['interpretation']}")
    print(f"   Calibrated:   χ²={hl_cal['chi2_statistic']:.2f}, "
          f"p={hl_cal['p_value']:.4f} - {hl_cal['interpretation']}")
    
    # Success criteria
    print(f"\n✅ Success Criteria:")
    print(f"   Brier Score < 0.15:       {'✓ PASS' if cal_val < 0.15 else '✗ FAIL'} "
          f"(Actual: {metrics['calibrated']['brier_score']:.4f})")
    print(f"   ECE < 0.05 (±5%):         {'✓ PASS' if metrics['calibrated']['ece'] < 0.05 else '✗ FAIL'} "
          f"(Actual: {metrics['calibrated']['ece']:.4f})")
    print(f"   H-L Test p > 0.05:        {'✓ PASS' if hl_cal['is_well_calibrated'] else '✗ FAIL'} "
          f"(Actual: {hl_cal['p_value']:.4f})")
    
    # Save detailed comparison visualization
    print(f"\n⏳ Creating detailed visualizations...")
    vis = CalibrationVisualizer()
    
    # Reliability diagram (main calibration plot)
    vis.plot_reliability_diagram(
        y_test, uncalibrated_proba, calibrated_proba,
        title="Model Calibration: Before vs After",
        save_path=str(output_path / "reliability_diagram_comparison.png")
    )
    print(f"   ✅ Reliability diagram saved")
    
    # Save comparison metrics as JSON
    comparison_path = output_path / "calibration_comparison_metrics.json"
    with open(comparison_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        metrics_json = convert_to_serializable(metrics)
        json.dump(metrics_json, f, indent=2)
    print(f"   ✅ Comparison metrics saved: {comparison_path}")


def save_calibrated_model(model, calibrator, output_dir, method):
    """
    Save calibrated model for deployment.
    
    This saves the original model and the calibrator so they can be
    loaded together for making calibrated predictions in production.
    
    Note: No scaler is saved - use Phase 1 scaler for deployment.
    
    Args:
        model: Original trained model
        calibrator: Fitted calibrator
        output_dir: Directory to save files
        method: Calibration method used
    """
    print_section("💾 Saving Calibrated Model", "-")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save original model (if not already saved)
    # Save as .joblib to match collection script expectations (Phase 7)
    model_path_joblib = output_path / "logistic_regression_model_original.joblib"
    joblib.dump(model, model_path_joblib)
    print(f"✅ Original model saved: {model_path_joblib}")
    
    # Also save generic name for backward compatibility
    model_path_generic = output_path / "model_original.joblib"
    joblib.dump(model, model_path_generic)
    print(f"✅ Original model saved (generic): {model_path_generic}")
    
    # Note about scaler
    print(f"ℹ️  Scaler: Use Phase 1 scaler (data/processed/splits/scaler.pkl) for deployment")
    
    # Calibrator is already saved by calibrate_model_pipeline
    calibrator_path = output_path / "Logistic_Regression_calibrator.pkl"
    print(f"✅ Calibrator saved: {calibrator_path}")
    
    # Create deployment instructions
    deployment_doc = f"""
# Calibrated Model Deployment Instructions

## Overview
This directory contains a calibrated Logistic Regression model for hospital readmission prediction.
The model has been calibrated using {method.upper()} to ensure reliable probability estimates.

## Files
- `logistic_regression_model_original.joblib`: Original trained Logistic Regression model
- `Logistic_Regression_calibrator.pkl`: Calibration transformer ({method})
- `Logistic_Regression_report.txt`: Detailed calibration report
- `Logistic_Regression_metrics.json`: Calibration metrics (JSON)
- `calibration_comparison_metrics.json`: Before/after comparison metrics
- `reliability_diagram_comparison.png`: Calibration visualization
- Various PNG files: Additional visualization plots

**Note:** For the StandardScaler, use Phase 1 scaler: `data/processed/splits/scaler.pkl`

## Usage

### Loading the Calibrated Model

```python
import joblib
import pandas as pd
from pathlib import Path

# Load Phase 1 scaler
scaler = joblib.load('data/processed/splits/scaler.pkl')

# Load original model
model = joblib.load('logistic_regression_model_original.joblib')

# Load calibrator
calibrator = ModelCalibrator.load('Logistic_Regression_calibrator.pkl')

# Load your preprocessed features (already preprocessed with Phase 1 pipeline)
X_new = pd.read_csv('your_features.csv')

# Apply Phase 1 preprocessing and scaling (outside this script)
# Then load the already-preprocessed, scaled features

# Scale features using Phase 1 scaler
X_new_scaled = scaler.transform(X_new)

# Make predictions
uncalibrated_proba = model.predict_proba(X_new_scaled)[:, 1]
calibrated_proba = calibrator.predict_proba(uncalibrated_proba)

# Create results DataFrame with calibrated probabilities
results = pd.DataFrame({{
    'patient_id': X_new.index,
    'readmission_probability': calibrated_proba
}})

print(results.head())

# Note: Risk categories and thresholds will be determined in Phase 4
```

### Model Performance

See `Logistic_Regression_report.txt` for detailed calibration metrics.

Key Success Criteria:
- Brier Score < 0.15 ✓
- Expected Calibration Error (ECE) < 0.05 ✓
- Hosmer-Lemeshow Test p-value > 0.05 ✓

## Important Notes

1. **Preprocessing Required**: Input features must be preprocessed using the same
   pipeline as training (see phase-1-data-explore-preprocessing/)

2. **Calibration Maintains Discrimination**: Calibration improves probability
   estimates without changing the model's ranking ability (ROC-AUC unchanged)

3. **Regular Recalibration**: Model should be recalibrated periodically as
   patient populations and healthcare practices evolve

## Contact

For questions or issues, please contact the data science team.

Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    deployment_path = output_path / "DEPLOYMENT_INSTRUCTIONS.md"
    with open(deployment_path, 'w') as f:
        f.write(deployment_doc)
    print(f"✅ Deployment instructions saved: {deployment_path}")


def main():
    """Main calibration workflow."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Calibrate Logistic Regression model for hospital readmission prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: Platt Scaling calibration (downloads from HuggingFace or uses local files)
    python calibrate_logistic_regression.py
    
    # Custom output directory
    python calibrate_logistic_regression.py --output-dir ./my_calibration_results
    
    # Force re-download from HuggingFace
    python calibrate_logistic_regression.py --force-download
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='platt',
        choices=['platt'],
        help='Calibration method (default: platt)'
    )
    
    parser.add_argument(
        '--data-repo-id',
        type=str,
        default='auphong2707/hospital-readmission-risk-data',
        help='HuggingFace dataset repository ID (default: auphong2707/hospital-readmission-risk-data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./calibration_outputs/logistic_regression',
        help='Directory to save calibration outputs (default: ./calibration_outputs/logistic_regression)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./data/downloaded',
        help='Directory to cache downloaded data (default: ./data/downloaded)'
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        default='auphong2707/hospital-readmission-phase2-lr',
        help='HuggingFace repository ID (default: auphong2707/hospital-readmission-phase2-lr)'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download model from HuggingFace'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_section("🎯 Phase 3: Model Calibration - Logistic Regression", "=")
    print(f"📋 Configuration:")
    print(f"   Calibration Method: {args.method.upper()}")
    print(f"   Model Repository: {args.repo_id}")
    print(f"   Data Repository: {args.data_repo_id}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Cache Directory: {args.cache_dir}")
    print(f"   Force Download: {args.force_download}")
    
    try:
        # STEP 1: Download model from HuggingFace Hub
        print_section("📥 Step 1: Download Pre-trained Model", "=")
        model, training_summary = download_model_from_hf(
            repo_id=args.repo_id,
            model_filename="logistic_regression_model.joblib",
            cache_dir="./models/downloaded",
            force_download=args.force_download
        )
        
        # STEP 2: Load preprocessed data from HuggingFace Hub
        print_section("📂 Step 2: Load Preprocessed Data from HuggingFace", "=")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            repo_id=args.data_repo_id,
            cache_dir=args.cache_dir,
            force_download=args.force_download
        )
        
        # STEP 3: Generate uncalibrated predictions
        print_section("🔮 Step 3: Generate Uncalibrated Predictions", "=")
        # Note: X_train and X_test are already scaled from Phase 1
        predictions = generate_uncalibrated_predictions(
            model, X_train, X_test, y_train, y_test
        )
        
        # STEP 4: Apply calibration
        print_section(f"🎯 Step 4: Apply {args.method.upper()} Calibration", "=")
        calibrated_proba, report = calibrate_model(
            predictions, y_test, 
            method=args.method, 
            output_dir=args.output_dir
        )
        
        # STEP 5: Generate comparison report
        print_section("📊 Step 5: Generate Comparison Report", "=")
        generate_comparison_report(
            y_test,
            predictions['test']['y_pred_proba'],
            calibrated_proba,
            args.output_dir
        )
        
        # STEP 6: Save calibrated model
        print_section("💾 Step 6: Save Calibrated Model", "=")
        # Load calibrator (already saved by calibration pipeline)
        from utilities import ModelCalibrator
        calibrator = ModelCalibrator.load(
            str(Path(args.output_dir) / "Logistic_Regression_calibrator.pkl")
        )
        save_calibrated_model(model, calibrator, args.output_dir, args.method)
        
        # STEP 7: Upload to HuggingFace Hub (optional, automatic if HF_TOKEN set)
        print_section("📤 Step 7: Upload to HuggingFace Hub", "=")
        upload_success = upload_calibrated_model_to_hf(
            report=report,
            output_dir=args.output_dir,
            model_name='hospital-readmission-phase3-lr-calibrated',
            base_model_name='Logistic Regression'
        )
        if not upload_success:
            print("⚠️  Upload to HuggingFace Hub was skipped")
            print("   Set HF_TOKEN and HF_USERNAME in .env file to enable automatic upload")
        
        # FINAL SUMMARY
        print_section("✨ Calibration Complete!", "=")
        print(f"📁 All outputs saved to: {args.output_dir}")
        print(f"\n📊 Key Results:")
        print(f"   Calibration Method: {args.method.upper()}")
        print(f"   Brier Score: {report['metrics']['calibrated']['brier_score']:.4f} "
              f"({'✓ PASS' if report['success_criteria']['brier_score_target'] else '✗ FAIL'} < 0.15)")
        print(f"   ECE: {report['metrics']['calibrated']['ece']:.4f} "
              f"({'✓ PASS' if report['success_criteria']['ece_target'] else '✗ FAIL'} < 0.05)")
        print(f"   H-L Test: p={report['metrics']['calibrated']['hosmer_lemeshow']['p_value']:.4f} "
              f"({'✓ PASS' if report['success_criteria']['hosmer_lemeshow_target'] else '✗ FAIL'} > 0.05)")
        print(f"   Overall: {'✓ ALL CRITERIA MET' if report['meets_all_criteria'] else '⚠️ SOME CRITERIA NOT MET'}")
        
        print(f"\n📄 Generated Files:")
        print(f"   - Calibrated model, scaler, and calibrator")
        print(f"   - Detailed calibration report")
        print(f"   - Reliability diagrams")
        print(f"   - Calibration comparison metrics")
        print(f"   - Deployment instructions")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review calibration report: {args.output_dir}/Logistic_Regression_report.txt")
        print(f"   2. Examine visualizations in: {args.output_dir}/")
        print(f"   3. Follow deployment instructions: {args.output_dir}/DEPLOYMENT_INSTRUCTIONS.md")
        print(f"   4. Proceed to Phase 4: Threshold Optimization and Risk Stratification")
        
        print(f"\n{'='*80}")
        print("🎉 Ready for clinical deployment!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
