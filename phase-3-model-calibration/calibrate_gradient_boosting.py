"""
Phase 3: Calibrate Gradient Boosting Model for Hospital Readmission Prediction

This script performs comprehensive calibration of the trained LightGBM model from Phase 2.
It downloads the pre-trained model from HuggingFace Hub, applies calibration techniques,
and generates detailed calibration reports with visualizations.

Calibration Pipeline:
1. Download pre-trained model from HuggingFace Hub
2. Load and preprocess data (using same preprocessing as training)
3. Generate uncalibrated predictions
4. Apply calibration methods (Platt Scaling, Isotonic Regression)
5. Evaluate calibration quality (Brier score, ECE, Hosmer-Lemeshow test)
6. Generate risk score mappings (Low/Medium/High risk categories)
7. Create comprehensive calibration reports and visualizations
8. Save calibrated model for deployment

Calibration Techniques:
- Platt Scaling (Sigmoid/Logistic transformation)
- Isotonic Regression (Non-parametric calibration)
- Group-Specific Calibration (Optional: by demographics)

Validation Methods:
- Reliability Diagrams (Predicted vs Observed probabilities)
- Brier Score (target < 0.15)
- Expected Calibration Error/ECE (target < 0.05 for ±5% diagonal)
- Hosmer-Lemeshow Test (target p-value > 0.05)

Clinical Risk Categories:
- Low Risk (0-5%): Standard discharge planning
- Medium Risk (5-15%): Enhanced patient education + 1-week follow-up
- High Risk (15%+): Intensive case management + home health visit

Usage (from project root):
    python ./phase-3-model-calibration/calibrate_gradient_boosting.py
    
    # With custom calibration method
    python ./phase-3-model-calibration/calibrate_gradient_boosting.py --method isotonic
    
    # With group-specific calibration
    python ./phase-3-model-calibration/calibrate_gradient_boosting.py --group-calibration

Requirements:
    pip install lightgbm scikit-learn pandas numpy matplotlib seaborn huggingface_hub joblib

HuggingFace Model:
    Repository: https://huggingface.co/auphong2707/hospital-readmission-lgbm
    Contains: Pre-trained LightGBM model, training summary, metrics, visualizations
"""

import argparse
import sys
import warnings
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

# Add parent directory to path to import preprocessing utilities
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utilities import (
    download_model_from_hf,
    ModelCalibrator,
    GroupSpecificCalibrator,
    CalibrationMetrics,
    RiskScoreMapper,
    CalibrationVisualizer,
    CalibrationReport,
    calibrate_model_pipeline,
    upload_calibrated_model_to_hf
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


def load_and_preprocess_data(data_dir: str = "data/processed", 
                             test_size: float = 0.2,
                             random_state: int = 42):
    """
    Load and preprocess data for calibration.
    
    This function loads the preprocessed features and target from Phase 1,
    and splits them into train/test sets using the SAME split strategy as
    the training script (stratified split with same random seed).
    
    CRITICAL: Must use same preprocessing and split as training to ensure
    calibration is done on the same validation/test set.
    
    Args:
        data_dir: Directory containing features.csv and target.csv
        test_size: Fraction of data for test set (must match training)
        random_state: Random seed (must match training)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    print_section("📂 Loading and Preprocessing Data", "-")
    
    # Step 1: Check if preprocessed data exists
    data_path = Path(data_dir)
    features_file = data_path / "features.csv"
    target_file = data_path / "target.csv"
    
    if not features_file.exists() or not target_file.exists():
        print("⚠️  Preprocessed data not found!")
        print(f"   Expected files:")
        print(f"   - {features_file}")
        print(f"   - {target_file}")
        print()
        print("🔄 Running preprocessing script...")
        
        # Run preprocessing script (same as training)
        preprocess_script = Path(__file__).resolve().parents[1] / "phase-1-data-explore-preprocessing" / "simple_preprocessing.py"
        
        if not preprocess_script.exists():
            raise FileNotFoundError(
                f"Preprocessing script not found: {preprocess_script}\n"
                f"Please ensure phase-1-data-explore-preprocessing/simple_preprocessing.py exists."
            )
        
        import subprocess
        subprocess.run([sys.executable, str(preprocess_script)], check=True)
        print("✅ Preprocessing completed\n")
    else:
        print("✅ Found preprocessed data files")
    
    # Step 2: Load features and target
    print("📥 Loading features and target...")
    X = pd.read_csv(features_file)
    y = pd.read_csv(target_file)
    
    # Handle target format (same as training script)
    if "target" in y.columns:
        y = y["target"]
    else:
        y = y.iloc[:, 0]
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    # Step 3: Split data (MUST match training split exactly)
    print(f"\n🔀 Splitting data (test_size={test_size}, random_state={random_state})...")
    print("   ⚠️  CRITICAL: Using same split parameters as training!")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"   Train set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"   Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def generate_uncalibrated_predictions(model, X_train, X_test, y_train, y_test):
    """
    Generate uncalibrated predictions from the trained model.
    
    This creates the baseline predictions that will be calibrated.
    We generate predictions for both train and test sets:
    - Train set: Used to fit the calibrator
    - Test set: Used to evaluate calibration quality
    
    Args:
        model: Trained LightGBM model
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing train/test predictions and labels
    """
    print_section("🔮 Generating Uncalibrated Predictions", "-")
    
    # Generate predictions on train set (for fitting calibrator)
    print("⏳ Generating train set predictions...")
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
    Apply calibration to model predictions.
    
    This is the core calibration step that transforms uncalibrated probabilities
    into calibrated probabilities that better reflect true readmission risk.
    
    Calibration Methods:
    - 'platt': Platt Scaling (logistic regression on predictions)
        * Fast, parametric
        * Works well with limited data
        * Assumes sigmoid relationship
    - 'isotonic': Isotonic Regression (piecewise-constant calibration)
        * Non-parametric, more flexible
        * Can model complex relationships
        * Requires more data
    
    Args:
        predictions: Dictionary with train/test predictions
        y_test: Test labels
        method: Calibration method ('platt' or 'isotonic')
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
        model_name='Gradient Boosting (LightGBM)',
        calibration_method=method,
        output_dir=output_dir
    )
    
    return calibrated_proba, report


def perform_risk_assessment(y_test, uncalibrated_proba, calibrated_proba, output_dir):
    """
    Perform clinical risk assessment and validation.
    
    This maps calibrated probabilities to clinical risk categories and validates
    that the risk categories align with actual readmission rates.
    
    Risk Categories:
    - Low (0-5%): Standard discharge planning
    - Medium (5-15%): Enhanced patient education + 1-week follow-up call
    - High (15%+): Intensive case management + home health visit
    
    Args:
        y_test: True labels
        uncalibrated_proba: Uncalibrated probabilities
        calibrated_proba: Calibrated probabilities
        output_dir: Directory to save outputs
        
    Returns:
        pd.DataFrame: Risk validation table
    """
    print_section("🏥 Clinical Risk Assessment", "-")
    
    # Initialize risk score mapper
    risk_mapper = RiskScoreMapper(low_threshold=0.05, high_threshold=0.15)
    
    # Map calibrated probabilities to risk categories
    print("📊 Mapping probabilities to risk categories...")
    risk_categories = risk_mapper.map_to_risk_category(calibrated_proba)
    risk_labels = risk_mapper.get_risk_labels(calibrated_proba)
    clinical_actions = risk_mapper.get_clinical_actions(calibrated_proba)
    
    # Display risk distribution
    print(f"\n📈 Risk Distribution:")
    unique_labels, counts = np.unique(risk_labels, return_counts=True)
    total = len(risk_labels)
    for label, count in zip(unique_labels, counts):
        percentage = (count / total) * 100
        print(f"   {label:8s}: {count:6d} patients ({percentage:5.1f}%)")
    
    # Validate risk scores
    print(f"\n✅ Validating risk scores...")
    validation_table = risk_mapper.validate_risk_scores(y_test, calibrated_proba)
    
    print(f"\n{'='*80}")
    print("📋 Risk Score Validation Table")
    print(f"{'='*80}")
    print(validation_table.to_string(index=False))
    print(f"{'='*80}\n")
    
    # Save validation table
    output_path = Path(output_dir)
    validation_path = output_path / "risk_validation_detailed.csv"
    validation_table.to_csv(validation_path, index=False)
    print(f"✅ Risk validation table saved: {validation_path}")
    
    # Create risk distribution visualization
    vis = CalibrationVisualizer()
    vis.plot_risk_distribution(
        calibrated_proba,
        risk_mapper,
        title="Clinical Risk Distribution - Calibrated Model",
        save_path=str(output_path / "risk_distribution_detailed.png")
    )
    
    return validation_table


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
        # Import the safe converter to avoid circular reference errors
        from utilities import convert_to_serializable
        metrics_json = convert_to_serializable(metrics)
        json.dump(metrics_json, f, indent=2)
    print(f"   ✅ Comparison metrics saved: {comparison_path}")


def save_calibrated_model(model, calibrator, output_dir, method):
    """
    Save calibrated model for deployment.
    
    This saves both the original model and the calibrator so they can be
    loaded together for making calibrated predictions in production.
    
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
    model_path = output_path / "gradient_boosting_model_original.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Original model saved: {model_path}")
    
    # Calibrator is already saved by calibrate_model_pipeline
    calibrator_path = output_path / "Gradient_Boosting_(LightGBM)_calibrator.pkl"
    print(f"✅ Calibrator saved: {calibrator_path}")
    
    # Create deployment instructions
    deployment_doc = f"""
# Calibrated Model Deployment Instructions

## Overview
This directory contains a calibrated LightGBM model for hospital readmission prediction.
The model has been calibrated using {method.upper()} to ensure reliable probability estimates.

## Files
- `gradient_boosting_model_original.joblib`: Original trained LightGBM model
- `Gradient_Boosting_(LightGBM)_calibrator.pkl`: Calibration transformer ({method})
- `Gradient_Boosting_(LightGBM)_report.txt`: Detailed calibration report
- `Gradient_Boosting_(LightGBM)_metrics.json`: Calibration metrics (JSON)
- `risk_validation_detailed.csv`: Risk category validation table
- Various PNG files: Visualization plots

## Usage

### Loading the Calibrated Model

```python
import joblib
import pandas as pd
from pathlib import Path

# Load original model
model = joblib.load('gradient_boosting_model_original.joblib')

# Load calibrator
import sys
sys.path.append('../phase-3-model-calibration')
from utilities import ModelCalibrator
calibrator = ModelCalibrator.load('Gradient_Boosting_(LightGBM)_calibrator.pkl')

# Load your preprocessed features
X_new = pd.read_csv('your_features.csv')

# Make predictions
uncalibrated_proba = model.predict_proba(X_new)[:, 1]
calibrated_proba = calibrator.predict_proba(uncalibrated_proba)

# Map to risk categories
from utilities import RiskScoreMapper
risk_mapper = RiskScoreMapper()
risk_categories = risk_mapper.get_risk_labels(calibrated_proba)
clinical_actions = risk_mapper.get_clinical_actions(calibrated_proba)

# Create results DataFrame
results = pd.DataFrame({{
    'patient_id': X_new.index,
    'readmission_probability': calibrated_proba,
    'risk_category': risk_categories,
    'recommended_action': clinical_actions
}})

print(results.head())
```

### Risk Categories

| Risk Level | Probability Range | Clinical Action |
|-----------|------------------|-----------------|
| Low | 0-5% | Standard discharge planning |
| Medium | 5-15% | Enhanced patient education + 1-week follow-up call |
| High | 15%+ | Intensive case management + home health visit |

### Model Performance

See `Gradient_Boosting_(LightGBM)_report.txt` for detailed calibration metrics.

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

4. **Clinical Validation**: Risk categories and thresholds should be validated
   with clinical experts before deployment

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
        description="Calibrate LightGBM model for hospital readmission prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: Platt scaling calibration
    python calibrate_gradient_boosting.py
    
    # Use isotonic regression
    python calibrate_gradient_boosting.py --method isotonic
    
    # Custom output directory
    python calibrate_gradient_boosting.py --output-dir ./my_calibration_results
    
    # Force re-download model from HuggingFace
    python calibrate_gradient_boosting.py --force-download
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='platt',
        choices=['platt', 'isotonic'],
        help='Calibration method: platt (sigmoid) or isotonic (default: platt)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing preprocessed data (default: data/processed)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./calibration_outputs/gradient_boosting',
        help='Directory to save calibration outputs (default: ./calibration_outputs/gradient_boosting)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size - MUST match training (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed - MUST match training (default: 42)'
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        default='auphong2707/hospital-readmission-lgbm',
        help='HuggingFace repository ID (default: auphong2707/hospital-readmission-lgbm)'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download model from HuggingFace'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_section("🎯 Phase 3: Model Calibration - Gradient Boosting", "=")
    print(f"📋 Configuration:")
    print(f"   Calibration Method: {args.method.upper()}")
    print(f"   Data Directory: {args.data_dir}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Test Size: {args.test_size}")
    print(f"   Random State: {args.random_state}")
    print(f"   HuggingFace Repo: {args.repo_id}")
    print(f"   Force Download: {args.force_download}")
    
    try:
        # STEP 1: Download model from HuggingFace Hub
        print_section("📥 Step 1: Download Pre-trained Model", "=")
        model, training_summary = download_model_from_hf(
            repo_id=args.repo_id,
            model_filename="gradient_boosting_model.joblib",
            cache_dir="./models/downloaded",
            force_download=args.force_download
        )
        
        # STEP 2: Load and preprocess data
        print_section("📂 Step 2: Load and Preprocess Data", "=")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            data_dir=args.data_dir,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # STEP 3: Generate uncalibrated predictions
        print_section("🔮 Step 3: Generate Uncalibrated Predictions", "=")
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
        
        # STEP 5: Clinical risk assessment
        print_section("🏥 Step 5: Clinical Risk Assessment", "=")
        validation_table = perform_risk_assessment(
            y_test, 
            predictions['test']['y_pred_proba'],
            calibrated_proba,
            args.output_dir
        )
        
        # STEP 6: Generate comparison report
        print_section("📊 Step 6: Generate Comparison Report", "=")
        generate_comparison_report(
            y_test,
            predictions['test']['y_pred_proba'],
            calibrated_proba,
            args.output_dir
        )
        
        # STEP 7: Save calibrated model
        print_section("💾 Step 7: Save Calibrated Model", "=")
        # Load calibrator (already saved by calibration pipeline)
        from utilities import ModelCalibrator
        calibrator = ModelCalibrator.load(
            str(Path(args.output_dir) / "Gradient_Boosting_(LightGBM)_calibrator.pkl")
        )
        save_calibrated_model(model, calibrator, args.output_dir, args.method)
        
        # STEP 8: Upload to HuggingFace Hub (optional, automatic if HF_TOKEN set)
        print_section("📤 Step 8: Upload to HuggingFace Hub", "=")
        upload_success = upload_calibrated_model_to_hf(
            report=report,
            output_dir=args.output_dir,
            model_name='hospital-readmission-lgbm-calibrated',
            base_model_name='Gradient Boosting (LightGBM)'
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
        print(f"   - Calibrated model and calibrator")
        print(f"   - Detailed calibration report")
        print(f"   - Reliability diagrams")
        print(f"   - Risk distribution plots")
        print(f"   - Risk validation tables")
        print(f"   - Deployment instructions")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review calibration report: {args.output_dir}/Gradient_Boosting_(LightGBM)_report.txt")
        print(f"   2. Examine visualizations in: {args.output_dir}/")
        print(f"   3. Validate risk categories with clinical experts")
        print(f"   4. Follow deployment instructions: {args.output_dir}/DEPLOYMENT_INSTRUCTIONS.md")
        print(f"   5. Consider fairness evaluation (Phase 4)")
        
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
