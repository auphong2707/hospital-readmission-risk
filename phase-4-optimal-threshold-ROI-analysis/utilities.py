"""
Phase 4: Optimal Threshold & ROI Analysis Utilities

Provides cost-sensitive threshold optimization and ROI analysis for hospital 
readmission prediction models.

Core Components:
- ThresholdOptimizer: Find optimal decision threshold maximizing expected value
- RiskCategoryMapper: Define and assign risk categories based on thresholds
- ROIAnalyzer: Calculate return on investment and business impact metrics
- ThresholdVisualizer: Visualizations for threshold optimization
- ROIVisualizer: Visualizations for ROI and resource planning

Cost Matrix:
- TP: +$14,500 (prevented readmission)
- FP: -$500 (unnecessary intervention)
- TN: $0 (correct prediction, no action)
- FN: -$15,000 (missed readmission)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import json
import pickle
import warnings
import os
import joblib
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()


# ============================================================================
# MODEL CALIBRATOR CLASS (for pickle compatibility)
# ============================================================================

class ModelCalibrator:
    """
    Platt Scaling calibration for hospital readmission models.
    
    Uses logistic regression to transform uncalibrated probabilities.
    
    Note: This class must be defined here for pickle compatibility when loading
    calibrators saved from Phase 3.
    """
    
    def __init__(self, method: str = 'platt', cv: int = 5):
        """
        Initialize the calibrator.
        
        Parameters:
        -----------
        method : str
            Calibration method - only 'platt' is supported
        cv : int
            Number of cross-validation folds (unused, kept for compatibility)
        """
        if method not in ['platt', 'sigmoid']:
            raise ValueError("method must be 'platt' or 'sigmoid'")
        
        self.method = 'platt'
        self.cv = cv
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> 'ModelCalibrator':
        """
        Fit Platt Scaling calibration on validation data.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba : np.ndarray
            Uncalibrated predicted probabilities
            
        Returns:
        --------
        self : ModelCalibrator
            Fitted calibrator instance
        """
        y_true = np.array(y_true).ravel()
        y_pred_proba = np.array(y_pred_proba).ravel()
        
        # Platt scaling: fit logistic regression on predicted probabilities
        self.calibrator = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        X = y_pred_proba.reshape(-1, 1)
        self.calibrator.fit(X, y_true)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Apply Platt Scaling calibration to uncalibrated probabilities.
        
        Parameters:
        -----------
        y_pred_proba : np.ndarray
            Uncalibrated predicted probabilities
            
        Returns:
        --------
        calibrated_proba : np.ndarray
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        y_pred_proba = np.array(y_pred_proba).ravel()
        X = y_pred_proba.reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(X)[:, 1]
        
        return calibrated
    
    def save(self, filepath: str):
        """Save the fitted calibrator to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'ModelCalibrator':
        """Load a fitted calibrator from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_phase1_splits(
    repo_id: str,
    cache_dir: str = "./data/downloaded"
):
    """Load Phase 1 splits from HuggingFace.
    
    This ensures all phases (2-5) use the exact same preprocessed data from Phase 1.
    Phase 1 created these splits with:
    - Train: 73,526 samples (72.25%)
    - Validation: 12,975 samples (12.75%)
    - Test: 15,265 samples (15%)
    - Random seed: 42
    - Stratification: Yes (on target variable)
    
    Args:
        repo_id: HuggingFace repository ID for Phase 1 data (e.g., 'username/hospital-readmission-risk-data')
        cache_dir: Directory to cache downloaded files
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        
    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If splits cannot be loaded from HuggingFace
    
    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = load_phase1_splits(
        ...     repo_id='your-username/hospital-readmission-risk-data'
        ... )
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub library required. "
            "Install with: pip install huggingface_hub"
        )
    
    print("\n" + "="*80)
    print("📥 Loading Phase 1 Splits from HuggingFace")
    print("="*80)
    print(f"Repository: {repo_id}")
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Download Phase 1 splits from HuggingFace
        train_path = hf_hub_download(
            repo_id=repo_id,
            filename="splits/train.csv",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        val_path = hf_hub_download(
            repo_id=repo_id,
            filename="splits/validation.csv",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        test_path = hf_hub_download(
            repo_id=repo_id,
            filename="splits/test.csv",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        
        # Load into DataFrames
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Split features and target
        target_col = 'target' if 'target' in train_df.columns else 'readmitted'
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        print(f"✅ Successfully loaded Phase 1 splits:")
        print(f"   Train: {X_train.shape} ({len(X_train):,} samples)")
        print(f"   Validation: {X_val.shape} ({len(X_val):,} samples)")
        print(f"   Test: {X_test.shape} ({len(X_test):,} samples)")
        print(f"   Total: {len(X_train) + len(X_val) + len(X_test):,} samples")
        print(f"\n📊 Class distributions:")
        print(f"   Train: {dict(y_train.value_counts())}")
        print(f"   Validation: {dict(y_val.value_counts())}")
        print(f"   Test: {dict(y_test.value_counts())}")
        print("="*80 + "\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"❌ Error loading Phase 1 splits: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"1. Ensure preprocessing script has uploaded data to HuggingFace")
        print(f"2. Check repository: https://huggingface.co/datasets/{repo_id}")
        print(f"3. Check internet connection")
        raise


def load_calibrated_model(
    repo_id: str,
    cache_dir: str = "./models/downloaded",
    force_download: bool = False
) -> Tuple[Any, Any]:
    """Load calibrated model and calibrator from HuggingFace Hub (Phase 3 output).
    
    This function downloads the calibrated model and Platt calibrator that were
    uploaded to HuggingFace Hub in Phase 3. This ensures consistent model artifacts
    across all phases.
    
    Phase 3 uploads:
    - gradient_boosting_model_original.joblib: The base LightGBM model
    - Gradient_Boosting_(LightGBM)_calibrator.pkl: The Platt calibrator
    - Calibration metrics and visualizations
    
    Args:
        repo_id: HuggingFace repository ID for calibrated model (e.g., 'username/hospital-readmission-lgbm-calibrated')
        cache_dir: Directory to cache downloaded files
        force_download: If True, re-download even if files exist locally
        
    Returns:
        tuple: (model, calibrator)
        
    Raises:
        ImportError: If huggingface_hub is not installed
        FileNotFoundError: If model files not found in repository
        
    Example:
        >>> model, calibrator = load_calibrated_model(
        ...     repo_id='your-username/hospital-readmission-lgbm-calibrated'
        ... )
        >>> # Use for predictions
        >>> y_pred = model.predict_proba(X_test)[:, 1]
        >>> y_calibrated = calibrator.predict_proba(y_pred)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub library required. "
            "Install with: pip install huggingface_hub"
        )
    
    print("\n" + "="*80)
    print("📥 Loading Calibrated Model from HuggingFace Hub")
    print("="*80)
    print(f"Repository: {repo_id}")
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Download model file
        print(f"\n⏳ Downloading original model...")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="gradient_boosting_model_original.joblib",
            cache_dir=cache_dir,
            force_download=force_download
        )
        print(f"✅ Model downloaded: {model_path}")
        
        # Load model
        print(f"⏳ Loading model...")
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        
        # Download calibrator file
        print(f"\n⏳ Downloading Platt calibrator...")
        calibrator_path = hf_hub_download(
            repo_id=repo_id,
            filename="Gradient_Boosting_(LightGBM)_calibrator.pkl",
            cache_dir=cache_dir,
            force_download=force_download
        )
        print(f"✅ Calibrator downloaded: {calibrator_path}")
        
        # Load calibrator
        print(f"⏳ Loading calibrator...")
        with open(calibrator_path, 'rb') as f:
            calibrator = pickle.load(f)
        print(f"✅ Calibrator loaded successfully")
        
        print("\n" + "="*80)
        print("✅ Calibrated Model & Calibrator Loaded from HuggingFace")
        print("="*80)
        print(f"🌐 Repository: https://huggingface.co/{repo_id}")
        print(f"💾 Local cache: {cache_dir}")
        print("="*80 + "\n")
        
        return model, calibrator
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model files not found in HuggingFace repository")
        print(f"   {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"1. Ensure Phase 3 calibration has been run and uploaded:")
        print(f"   python ./phase-3-model-calibration/calibrate_gradient_boosting.py")
        print(f"2. Check repository exists: https://huggingface.co/{repo_id}")
        print(f"3. Verify HF_TOKEN and HF_USERNAME are set in .env for Phase 3 upload")
        print(f"4. Check internet connection")
        raise
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(f"\nIf you have local Phase 3 outputs, you can use them directly:")
        print(f"   model = joblib.load('./phase-3-model-calibration/models/gradient_boosting_calibrated.pkl')")
        print(f"   calibrator = pickle.load(open('./phase-3-model-calibration/models/platt_calibrator.pkl', 'rb'))")
        raise


def load_calibrated_random_forest_model(
    repo_id: str,
    cache_dir: str = "./models/downloaded",
    force_download: bool = False
) -> Tuple[Any, Any]:
    """Load calibrated Random Forest model and calibrator from HuggingFace Hub (Phase 3 output).
    
    This function downloads the calibrated Random Forest model and Platt calibrator that were
    uploaded to HuggingFace Hub in Phase 3. This ensures consistent model artifacts
    across all phases.
    
    Phase 3 uploads:
    - random_forest_model.joblib: The base Random Forest model
    - Random_Forest_calibrator.pkl: The Platt calibrator
    - Calibration metrics and visualizations
    
    Args:
        repo_id: HuggingFace repository ID for calibrated model (e.g., 'username/hospital-readmission-rf-calibrated')
        cache_dir: Directory to cache downloaded files
        force_download: If True, re-download even if files exist locally
        
    Returns:
        tuple: (model, calibrator)
        
    Raises:
        ImportError: If huggingface_hub is not installed
        FileNotFoundError: If model files not found in repository
        
    Example:
        >>> model, calibrator = load_calibrated_random_forest_model(
        ...     repo_id='your-username/hospital-readmission-rf-calibrated'
        ... )
        >>> # Use for predictions
        >>> y_pred = model.predict_proba(X_test)[:, 1]
        >>> y_calibrated = calibrator.predict_proba(y_pred)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub library required. "
            "Install with: pip install huggingface_hub"
        )
    
    print("\n" + "="*80)
    print("📥 Loading Calibrated Random Forest Model from HuggingFace Hub")
    print("="*80)
    print(f"Repository: {repo_id}")
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Download model file
        print(f"\n⏳ Downloading Random Forest model...")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="random_forest_model.joblib",
            cache_dir=cache_dir,
            force_download=force_download
        )
        print(f"✅ Model downloaded: {model_path}")
        
        # Load model
        print(f"⏳ Loading model...")
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        
        # Download calibrator file
        print(f"\n⏳ Downloading Platt calibrator...")
        calibrator_path = hf_hub_download(
            repo_id=repo_id,
            filename="Random_Forest_calibrator.pkl",
            cache_dir=cache_dir,
            force_download=force_download
        )
        print(f"✅ Calibrator downloaded: {calibrator_path}")
        
        # Load calibrator
        print(f"⏳ Loading calibrator...")
        with open(calibrator_path, 'rb') as f:
            calibrator = pickle.load(f)
        print(f"✅ Calibrator loaded successfully")
        
        print("\n" + "="*80)
        print("✅ Calibrated Random Forest Model & Calibrator Loaded from HuggingFace")
        print("="*80)
        print(f"🌐 Repository: https://huggingface.co/{repo_id}")
        print(f"💾 Local cache: {cache_dir}")
        print("="*80 + "\n")
        
        return model, calibrator
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model files not found in HuggingFace repository")
        print(f"   {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"1. Ensure Phase 3 calibration has been run and uploaded:")
        print(f"   python ./phase-3-model-calibration/calibrate_random_forest.py")
        print(f"2. Check repository exists: https://huggingface.co/{repo_id}")
        print(f"3. Verify HF_TOKEN and HF_USERNAME are set in .env for Phase 3 upload")
        print(f"4. Check internet connection")
        raise
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(f"\nIf you have local Phase 3 outputs, you can use them directly:")
        print(f"   model = joblib.load('./phase-3-model-calibration/models/random_forest_model.joblib')")
        print(f"   calibrator = pickle.load(open('./phase-3-model-calibration/models/Random_Forest_calibrator.pkl', 'rb'))")
        raise


def get_calibrated_predictions(
    model: Any,
    calibrator: Any,
    X: pd.DataFrame
) -> np.ndarray:
    """Generate calibrated predictions from model and calibrator.
    
    Args:
        model: Trained model
        calibrator: Fitted calibrator from Phase 3
        X: Feature matrix
        
    Returns:
        np.ndarray: Calibrated probability predictions
    """
    # Get uncalibrated predictions
    y_pred_proba_uncalibrated = model.predict_proba(X)[:, 1]
    
    # Apply calibration
    y_pred_proba_calibrated = calibrator.predict_proba(y_pred_proba_uncalibrated)
    
    return y_pred_proba_calibrated


# ============================================================================
# THRESHOLD OPTIMIZER
# ============================================================================

class ThresholdOptimizer:
    """
    Cost-sensitive threshold optimization for readmission prediction.
    
    Finds the optimal decision threshold that maximizes expected value based on
    a business cost matrix.
    
    Attributes:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities (calibrated)
        cost_params: Dictionary with cost parameters
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        cost_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ThresholdOptimizer.
        
        Args:
            y_true: True binary labels (0 = no readmission, 1 = readmission)
            y_pred_proba: Predicted probabilities
            cost_params: Cost parameters dictionary with keys:
                - readmission_cost: Cost of a readmission event (default: 15000)
                - intervention_cost: Cost of intervention (default: 500)
                - tp_benefit: Benefit from true positive (default: 14500)
                - fp_cost: Cost from false positive (default: -500)
                - fn_cost: Cost from false negative (default: -15000)
                - tn_cost: Cost from true negative (default: 0)
        """
        self.y_true = np.array(y_true).ravel()
        self.y_pred_proba = np.array(y_pred_proba).ravel()
        
        # Default cost parameters
        if cost_params is None:
            cost_params = {
                'readmission_cost': 15000,
                'intervention_cost': 500,
                'tp_benefit': 14500,  # readmission_cost - intervention_cost
                'fp_cost': -500,      # intervention cost wasted
                'fn_cost': -15000,    # missed readmission
                'tn_cost': 0          # correct prediction, no action
            }
        
        self.cost_params = cost_params
        self.optimal_threshold = None
        self.optimal_ev = None
        self.threshold_results = None
        
    def calculate_expected_value(self, threshold: float) -> float:
        """
        Calculate expected value at a specific threshold.
        
        Expected Value = (TP × tp_benefit) + (FP × fp_cost) + (FN × fn_cost) + (TN × tn_cost)
        
        Args:
            threshold: Decision threshold (0-1)
            
        Returns:
            float: Expected value in dollars
        """
        # Make predictions at threshold
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        
        # Calculate expected value
        ev = (
            tp * self.cost_params['tp_benefit'] +
            fp * self.cost_params['fp_cost'] +
            fn * self.cost_params['fn_cost'] +
            tn * self.cost_params['tn_cost']
        )
        
        return ev
    
    def find_optimal_threshold(
        self,
        threshold_range: Tuple[float, float] = (0.05, 0.95),
        num_points: int = 200
    ) -> float:
        """
        Find optimal threshold by testing multiple thresholds.
        
        Args:
            threshold_range: (min_threshold, max_threshold) to test
            num_points: Number of thresholds to test
            
        Returns:
            float: Optimal threshold maximizing expected value
        """
        print("\n" + "="*80)
        print("🔍 Finding Optimal Threshold")
        print("="*80)
        print(f"Testing {num_points} thresholds from {threshold_range[0]:.2f} to {threshold_range[1]:.2f}")
        
        # Test multiple thresholds
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        expected_values = []
        
        for threshold in thresholds:
            ev = self.calculate_expected_value(threshold)
            expected_values.append(ev)
        
        # Find optimal threshold
        optimal_idx = np.argmax(expected_values)
        self.optimal_threshold = thresholds[optimal_idx]
        self.optimal_ev = expected_values[optimal_idx]
        
        # Store results
        self.threshold_results = pd.DataFrame({
            'threshold': thresholds,
            'expected_value': expected_values
        })
        
        print(f"\n✅ Optimal Threshold Found:")
        print(f"   Threshold: {self.optimal_threshold:.4f}")
        print(f"   Expected Value: ${self.optimal_ev:,.2f}")
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (self.y_pred_proba >= self.optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred_optimal).ravel()
        
        print(f"\n📊 Confusion Matrix at Optimal Threshold:")
        print(f"   True Positives (TP): {tp:,} → ${tp * self.cost_params['tp_benefit']:,.2f}")
        print(f"   False Positives (FP): {fp:,} → ${fp * self.cost_params['fp_cost']:,.2f}")
        print(f"   False Negatives (FN): {fn:,} → ${fn * self.cost_params['fn_cost']:,.2f}")
        print(f"   True Negatives (TN): {tn:,} → ${tn * self.cost_params['tn_cost']:,.2f}")
        
        # Calculate intervention rate
        intervention_rate = (tp + fp) / len(self.y_true)
        print(f"\n💉 Intervention Volume:")
        print(f"   Patients receiving intervention: {tp + fp:,} ({intervention_rate:.1%})")
        print(f"   Patients not receiving intervention: {tn + fn:,} ({1-intervention_rate:.1%})")
        
        print("="*80 + "\n")
        
        return self.optimal_threshold
    
    def find_optimal_threshold_with_constraint(
        self,
        max_intervention_rate: float = 0.30,
        threshold_range: Tuple[float, float] = (0.05, 0.95),
        num_points: int = 200
    ) -> float:
        """
        Find optimal threshold with intervention volume constraint.
        
        Useful when operational capacity limits the number of patients
        that can receive interventions.
        
        Args:
            max_intervention_rate: Maximum fraction of patients to intervene (0-1)
            threshold_range: (min_threshold, max_threshold) to test
            num_points: Number of thresholds to test
            
        Returns:
            float: Optimal threshold under constraint
        """
        print("\n" + "="*80)
        print("🔍 Finding Optimal Threshold with Constraint")
        print("="*80)
        print(f"Maximum intervention rate: {max_intervention_rate:.1%}")
        
        # Test multiple thresholds
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        valid_thresholds = []
        valid_evs = []
        
        for threshold in thresholds:
            # Check intervention rate
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            intervention_rate = np.mean(y_pred)
            
            # Only consider thresholds meeting constraint
            if intervention_rate <= max_intervention_rate:
                ev = self.calculate_expected_value(threshold)
                valid_thresholds.append(threshold)
                valid_evs.append(ev)
        
        if len(valid_thresholds) == 0:
            raise ValueError(
                f"No thresholds meet constraint (max_intervention_rate={max_intervention_rate}). "
                f"Try increasing max_intervention_rate."
            )
        
        # Find optimal among valid thresholds
        optimal_idx = np.argmax(valid_evs)
        optimal_threshold = valid_thresholds[optimal_idx]
        optimal_ev = valid_evs[optimal_idx]
        
        # Calculate actual intervention rate
        y_pred_optimal = (self.y_pred_proba >= optimal_threshold).astype(int)
        actual_intervention_rate = np.mean(y_pred_optimal)
        
        print(f"\n✅ Constrained Optimal Threshold Found:")
        print(f"   Threshold: {optimal_threshold:.4f}")
        print(f"   Expected Value: ${optimal_ev:,.2f}")
        print(f"   Intervention Rate: {actual_intervention_rate:.1%} (max: {max_intervention_rate:.1%})")
        print("="*80 + "\n")
        
        return optimal_threshold
    
    def get_metrics_at_threshold(self, threshold: float) -> Dict[str, Any]:
        """
        Get comprehensive metrics at a specific threshold.
        
        Args:
            threshold: Decision threshold
            
        Returns:
            dict: Metrics including confusion matrix, rates, and costs
        """
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        
        total = tn + fp + fn + tp
        
        # Use sklearn functions for standard metrics
        precision = precision_score(self.y_true, y_pred, zero_division=0)
        recall = recall_score(self.y_true, y_pred, zero_division=0)
        f1 = f1_score(self.y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(self.y_true, y_pred)
        
        metrics = {
            'threshold': threshold,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'tpr': recall,  # Sensitivity/Recall (same as recall)
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # No sklearn equivalent
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity (no sklearn equivalent)
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # No sklearn equivalent
            'precision': precision,  # PPV
            'recall': recall,    # Sensitivity
            'f1_score': f1,
            'accuracy': accuracy,
            'intervention_rate': (tp + fp) / total,  # Domain-specific metric
            'expected_value': self.calculate_expected_value(threshold),
            'tp_benefit': tp * self.cost_params['tp_benefit'],
            'fp_cost': fp * self.cost_params['fp_cost'],
            'fn_cost': fn * self.cost_params['fn_cost'],
            'tn_cost': tn * self.cost_params['tn_cost']
        }
        
        return metrics
    
    def calculate_break_even_threshold(self) -> float:
        """
        Calculate the break-even threshold where expected value = 0.
        
        Returns:
            float: Break-even threshold
        """
        if self.threshold_results is None:
            raise ValueError("Run find_optimal_threshold() first")
        
        # Find threshold closest to EV = 0
        positive_evs = self.threshold_results[self.threshold_results['expected_value'] >= 0]
        
        if len(positive_evs) == 0:
            return None  # No positive EV thresholds
        
        # Break-even is the lowest threshold with positive EV
        break_even_threshold = positive_evs['threshold'].min()
        
        return break_even_threshold


# ============================================================================
# RISK CATEGORY MAPPER
# ============================================================================

class RiskCategoryMapper:
    """
    Map probability predictions to risk categories for clinical decision support.
    
    Risk Categories:
    - Low: Standard discharge process
    - Medium: Enhanced follow-up call
    - High: Intensive case management
    """
    
    def __init__(
        self,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        optimal_threshold: Optional[float] = None
    ):
        """
        Initialize RiskCategoryMapper.
        
        Args:
            low_threshold: Boundary between low and medium risk
            high_threshold: Boundary between medium and high risk
            optimal_threshold: Optimal threshold from ThresholdOptimizer (used to derive boundaries)
        """
        if optimal_threshold is not None:
            # Derive thresholds from optimal threshold
            self.low_threshold = optimal_threshold * 0.67
            self.high_threshold = optimal_threshold * 1.5
            self.optimal_threshold = optimal_threshold
        elif low_threshold is not None and high_threshold is not None:
            self.low_threshold = low_threshold
            self.high_threshold = high_threshold
            self.optimal_threshold = None
        else:
            raise ValueError("Provide either optimal_threshold or both low_threshold and high_threshold")
    
    def assign_risk_categories(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Assign risk categories based on predicted probabilities.
        
        Args:
            y_pred_proba: Predicted probabilities
            
        Returns:
            np.ndarray: Risk categories ('Low', 'Medium', 'High')
        """
        y_pred_proba = np.array(y_pred_proba).ravel()
        
        categories = np.where(
            y_pred_proba < self.low_threshold, 'Low',
            np.where(y_pred_proba < self.high_threshold, 'Medium', 'High')
        )
        
        return categories
    
    def get_category_statistics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> pd.DataFrame:
        """
        Get statistics for each risk category.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            pd.DataFrame: Statistics by risk category
        """
        categories = self.assign_risk_categories(y_pred_proba)
        
        stats = []
        for category in ['Low', 'Medium', 'High']:
            mask = categories == category
            n_patients = np.sum(mask)
            
            if n_patients > 0:
                actual_readmission_rate = np.mean(y_true[mask])
                avg_predicted_prob = np.mean(y_pred_proba[mask])
                
                stats.append({
                    'risk_category': category,
                    'n_patients': n_patients,
                    'percentage': n_patients / len(y_true) * 100,
                    'actual_readmission_rate': actual_readmission_rate,
                    'avg_predicted_probability': avg_predicted_prob
                })
        
        return pd.DataFrame(stats)
    
    def print_category_summary(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ):
        """
        Print summary of risk categories.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
        """
        print("\n" + "="*80)
        print("📊 Risk Category Summary")
        print("="*80)
        print(f"Low Risk Threshold: {self.low_threshold:.4f}")
        print(f"High Risk Threshold: {self.high_threshold:.4f}")
        if self.optimal_threshold is not None:
            print(f"Optimal Threshold (reference): {self.optimal_threshold:.4f}")
        
        stats_df = self.get_category_statistics(y_true, y_pred_proba)
        
        print(f"\n{stats_df.to_string(index=False)}")
        print("="*80 + "\n")


# ============================================================================
# ROI ANALYZER
# ============================================================================

class ROIAnalyzer:
    """
    Calculate return on investment and business impact metrics.
    
    Analyzes the financial impact of the intervention program based on
    optimal threshold and predicted readmission rates.
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        optimal_threshold: float,
        cost_params: Dict[str, float]
    ):
        """
        Initialize ROIAnalyzer.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            optimal_threshold: Optimal decision threshold
            cost_params: Cost parameters dictionary
        """
        self.y_true = np.array(y_true).ravel()
        self.y_pred_proba = np.array(y_pred_proba).ravel()
        self.optimal_threshold = optimal_threshold
        self.cost_params = cost_params
        
        # Make predictions at optimal threshold
        self.y_pred = (self.y_pred_proba >= self.optimal_threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        self.tp = int(tp)
        self.fp = int(fp)
        self.tn = int(tn)
        self.fn = int(fn)
        self.total = self.tp + self.fp + self.tn + self.fn
    
    def calculate_roi_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive ROI metrics.
        
        Returns:
            dict: ROI metrics including costs, benefits, and net savings
        """
        # Costs and benefits
        intervention_cost_total = (self.tp + self.fp) * self.cost_params['intervention_cost']
        prevented_readmissions = self.tp
        prevented_readmission_savings = prevented_readmissions * self.cost_params['readmission_cost']
        missed_readmissions = self.fn
        missed_readmission_costs = missed_readmissions * self.cost_params['readmission_cost']
        
        # Net benefit
        net_benefit = (
            self.tp * self.cost_params['tp_benefit'] +
            self.fp * self.cost_params['fp_cost'] +
            self.fn * self.cost_params['fn_cost'] +
            self.tn * self.cost_params['tn_cost']
        )
        
        # ROI calculation
        roi = (net_benefit / intervention_cost_total * 100) if intervention_cost_total > 0 else 0
        
        # Baseline cost (no intervention - all readmissions occur)
        baseline_readmissions = np.sum(self.y_true)
        baseline_cost = baseline_readmissions * self.cost_params['readmission_cost']
        
        # Cost with intervention
        cost_with_intervention = (
            intervention_cost_total +  # All interventions
            missed_readmission_costs    # Missed readmissions still occur
        )
        
        # Total savings
        total_savings = baseline_cost - cost_with_intervention
        savings_percentage = (total_savings / baseline_cost * 100) if baseline_cost > 0 else 0
        
        # Readmission rate reduction
        baseline_rate = baseline_readmissions / self.total
        actual_readmissions_with_intervention = self.fn
        intervention_rate = actual_readmissions_with_intervention / self.total
        readmission_reduction_absolute = baseline_rate - intervention_rate
        readmission_reduction_relative = (readmission_reduction_absolute / baseline_rate * 100) if baseline_rate > 0 else 0
        
        metrics = {
            'optimal_threshold': self.optimal_threshold,
            'total_patients': self.total,
            'intervention_volume': self.tp + self.fp,
            'intervention_rate': (self.tp + self.fp) / self.total,
            'intervention_cost_total': intervention_cost_total,
            'prevented_readmissions': prevented_readmissions,
            'prevented_readmission_savings': prevented_readmission_savings,
            'missed_readmissions': missed_readmissions,
            'missed_readmission_costs': missed_readmission_costs,
            'net_benefit': net_benefit,
            'roi_percentage': roi,
            'baseline_readmissions': baseline_readmissions,
            'baseline_cost': baseline_cost,
            'cost_with_intervention': cost_with_intervention,
            'total_savings': total_savings,
            'savings_percentage': savings_percentage,
            'baseline_readmission_rate': baseline_rate,
            'intervention_readmission_rate': intervention_rate,
            'readmission_reduction_absolute': readmission_reduction_absolute,
            'readmission_reduction_relative': readmission_reduction_relative,
            'tp': self.tp,
            'fp': self.fp,
            'tn': self.tn,
            'fn': self.fn
        }
        
        return metrics
    
    def generate_roi_report(self) -> str:
        """
        Generate human-readable ROI report.
        
        Returns:
            str: Formatted ROI report
        """
        metrics = self.calculate_roi_metrics()
        
        report = []
        report.append("\n" + "="*80)
        report.append("💰 ROI ANALYSIS REPORT")
        report.append("="*80)
        
        report.append(f"\n📊 BASELINE (No Intervention):")
        report.append(f"   Total Patients: {metrics['total_patients']:,}")
        report.append(f"   Readmissions: {metrics['baseline_readmissions']:,} ({metrics['baseline_readmission_rate']:.1%})")
        report.append(f"   Total Cost: ${metrics['baseline_cost']:,.2f}")
        
        report.append(f"\n💉 WITH INTERVENTION (Optimal Threshold = {metrics['optimal_threshold']:.4f}):")
        report.append(f"   Intervention Volume: {metrics['intervention_volume']:,} ({metrics['intervention_rate']:.1%} of patients)")
        report.append(f"   Intervention Cost: ${metrics['intervention_cost_total']:,.2f}")
        report.append(f"   Prevented Readmissions: {metrics['prevented_readmissions']:,}")
        report.append(f"   Missed Readmissions: {metrics['missed_readmissions']:,} ({metrics['intervention_readmission_rate']:.1%})")
        report.append(f"   Cost with Intervention: ${metrics['cost_with_intervention']:,.2f}")
        
        report.append(f"\n💵 FINANCIAL IMPACT:")
        report.append(f"   Net Benefit: ${metrics['net_benefit']:,.2f}")
        report.append(f"   Total Savings: ${metrics['total_savings']:,.2f} ({metrics['savings_percentage']:.1f}% reduction)")
        report.append(f"   ROI: {metrics['roi_percentage']:.1f}%")
        
        report.append(f"\n📉 READMISSION RATE REDUCTION:")
        report.append(f"   Baseline Rate: {metrics['baseline_readmission_rate']:.1%}")
        report.append(f"   With Intervention: {metrics['intervention_readmission_rate']:.1%}")
        report.append(f"   Absolute Reduction: {metrics['readmission_reduction_absolute']:.1%}")
        report.append(f"   Relative Reduction: {metrics['readmission_reduction_relative']:.1f}%")
        
        # Break-even analysis
        break_even_rate = self.cost_params['intervention_cost'] / self.cost_params['readmission_cost']
        report.append(f"\n✅ BREAK-EVEN ANALYSIS:")
        report.append(f"   Break-even Threshold: {break_even_rate:.1%} (need >{break_even_rate:.1%} reduction)")
        report.append(f"   Actual Reduction: {metrics['readmission_reduction_absolute']:.1%}")
        if metrics['readmission_reduction_absolute'] > break_even_rate:
            margin = (metrics['readmission_reduction_absolute'] - break_even_rate) / break_even_rate * 100
            report.append(f"   ✅ PROFITABLE: {margin:.1f}% above break-even")
        else:
            report.append(f"   ❌ BELOW BREAK-EVEN")
        
        report.append("="*80 + "\n")
        
        return "\n".join(report)
    
    def print_roi_report(self):
        """Print ROI report to console."""
        print(self.generate_roi_report())
    
    def sensitivity_analysis(
        self,
        scenarios: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis with different cost assumptions.
        
        Args:
            scenarios: Dictionary of scenario names to cost parameter dictionaries
                       Example: {
                           'Conservative': {'readmission_cost': 12000, 'intervention_cost': 700},
                           'Base Case': {'readmission_cost': 15000, 'intervention_cost': 500},
                           'Aggressive': {'readmission_cost': 18000, 'intervention_cost': 400}
                       }
        
        Returns:
            pd.DataFrame: ROI metrics for each scenario
        """
        results = []
        
        for scenario_name, params in scenarios.items():
            # Update cost parameters for this scenario
            temp_cost_params = self.cost_params.copy()
            temp_cost_params.update(params)
            
            # Recalculate tp_benefit based on updated costs
            temp_cost_params['tp_benefit'] = (
                temp_cost_params.get('readmission_cost', self.cost_params['readmission_cost']) -
                temp_cost_params.get('intervention_cost', self.cost_params['intervention_cost'])
            )
            temp_cost_params['fp_cost'] = -temp_cost_params.get('intervention_cost', self.cost_params['intervention_cost'])
            temp_cost_params['fn_cost'] = -temp_cost_params.get('readmission_cost', self.cost_params['readmission_cost'])
            
            # Create temporary analyzer
            temp_analyzer = ROIAnalyzer(
                y_true=self.y_true,
                y_pred_proba=self.y_pred_proba,
                optimal_threshold=self.optimal_threshold,
                cost_params=temp_cost_params
            )
            
            metrics = temp_analyzer.calculate_roi_metrics()
            
            results.append({
                'scenario': scenario_name,
                'readmission_cost': temp_cost_params['readmission_cost'],
                'intervention_cost': temp_cost_params['intervention_cost'],
                'net_benefit': metrics['net_benefit'],
                'roi_percentage': metrics['roi_percentage'],
                'total_savings': metrics['total_savings']
            })
        
        return pd.DataFrame(results)


# ============================================================================
# SAVE/LOAD UTILITIES
# ============================================================================

def _convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    return obj


def save_threshold_results(
    threshold_optimizer: ThresholdOptimizer,
    risk_mapper: RiskCategoryMapper,
    roi_analyzer: ROIAnalyzer,
    output_dir: str = "./phase-4-optimal-threshold-ROI-analysis/outputs"
):
    """
    Save threshold optimization results.
    
    Args:
        threshold_optimizer: Fitted ThresholdOptimizer
        risk_mapper: RiskCategoryMapper
        roi_analyzer: ROIAnalyzer
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving threshold optimization results to: {output_dir}")
    
    # Save threshold results
    if threshold_optimizer.threshold_results is not None:
        results_path = output_path / "threshold_results.csv"
        threshold_optimizer.threshold_results.to_csv(results_path, index=False)
        print(f"   ✅ Threshold results: {results_path}")
    
    # Save optimal threshold and risk thresholds
    thresholds = {
        'optimal_threshold': float(threshold_optimizer.optimal_threshold),
        'optimal_expected_value': float(threshold_optimizer.optimal_ev),
        'low_risk_threshold': float(risk_mapper.low_threshold),
        'high_risk_threshold': float(risk_mapper.high_threshold),
        'cost_parameters': _convert_to_serializable(threshold_optimizer.cost_params)
    }
    
    thresholds_path = output_path / "optimal_thresholds.json"
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"   ✅ Optimal thresholds: {thresholds_path}")
    
    # Save ROI metrics
    roi_metrics = roi_analyzer.calculate_roi_metrics()
    roi_metrics_serializable = _convert_to_serializable(roi_metrics)
    roi_path = output_path / "roi_metrics.json"
    with open(roi_path, 'w') as f:
        json.dump(roi_metrics_serializable, f, indent=2)
    print(f"   ✅ ROI metrics: {roi_path}")
    
    # Save ROI report
    report_path = output_path / "roi_report.txt"
    with open(report_path, 'w') as f:
        f.write(roi_analyzer.generate_roi_report())
    print(f"   ✅ ROI report: {report_path}")
    
    print(f"✅ All results saved successfully!\n")


def upload_results_to_hf(
    output_dir: str,
    viz_dir: str,
    repo_id: str,
    commit_message: str = "Upload Phase 4 threshold optimization results",
    token: Optional[str] = None
):
    """
    Upload Phase 4 results to HuggingFace Hub.
    
    Args:
        output_dir: Directory containing output files (JSON, CSV, TXT)
        viz_dir: Directory containing visualization files
        repo_id: HuggingFace repository ID (e.g., 'username/hospital-readmission-gradient-boosting-threshold-results')
        commit_message: Commit message for the upload
        token: HuggingFace API token (if None, will use HF_TOKEN environment variable)
        
    Returns:
        str: URL to the uploaded repository
        
    Raises:
        ImportError: If huggingface_hub is not installed
        ValueError: If required files are missing
        
    Example:
        >>> upload_results_to_hf(
        ...     output_dir='./phase-4-optimal-threshold-ROI-analysis/outputs',
        ...     viz_dir='./phase-4-optimal-threshold-ROI-analysis/visualizations',
        ...     repo_id='username/hospital-readmission-gradient-boosting-threshold-results'
        ... )
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub library required for uploading. "
            "Install with: pip install huggingface_hub"
        )
    
    # Get token from environment if not provided
    if token is None:
        token = os.getenv('HF_TOKEN')
        if token is None:
            raise ValueError(
                "HuggingFace token not provided. Set HF_TOKEN environment variable or pass token parameter."
            )
    
    print("\n" + "="*80)
    print("📤 Uploading Phase 4 Results to HuggingFace Hub")
    print("="*80)
    print(f"Repository: {repo_id}")
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"✅ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")
    
    # Collect files to upload
    files_to_upload = []
    
    # Output files
    output_path = Path(output_dir)
    if output_path.exists():
        for file_path in output_path.glob('*'):
            if file_path.is_file():
                files_to_upload.append((str(file_path), f"outputs/{file_path.name}"))
    
    # Visualization files
    viz_path = Path(viz_dir)
    if viz_path.exists():
        for file_path in viz_path.glob('*.png'):
            if file_path.is_file():
                files_to_upload.append((str(file_path), f"visualizations/{file_path.name}"))
    
    if len(files_to_upload) == 0:
        raise ValueError(f"No files found to upload in {output_dir} or {viz_dir}")
    
    print(f"\n📦 Uploading {len(files_to_upload)} files...")
    
    # Upload files
    uploaded_count = 0
    for local_path, remote_path in files_to_upload:
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"{commit_message}: {Path(local_path).name}",
                token=token
            )
            uploaded_count += 1
            print(f"   ✅ Uploaded: {remote_path}")
        except Exception as e:
            print(f"   ❌ Failed to upload {remote_path}: {e}")
    
    # Create README if it doesn't exist
    try:
        readme_content = f"""---
license: apache-2.0
tags:
- healthcare
- hospital-readmission
- threshold-optimization
- roi-analysis
---

# Hospital Readmission Risk - Phase 4: Threshold Optimization Results

This repository contains the results from Phase 4: Optimal Threshold & ROI Analysis.

## Contents

### Outputs
- `outputs/optimal_thresholds.json`: Optimal decision threshold and risk category thresholds
- `outputs/roi_metrics.json`: Comprehensive ROI metrics
- `outputs/roi_report.txt`: Human-readable ROI analysis report
- `outputs/threshold_results.csv`: Expected value across all tested thresholds

### Visualizations
1. Expected Value vs Threshold Curve
2. Cost-Benefit Analysis
3. Classification Metrics vs Threshold
4. Confusion Matrix at Optimal Threshold
5. Risk Category Distribution
6. ROI Sensitivity Analysis
7. Intervention Volume Forecast
8. Cost Savings Projection

## Usage

These results can be used for:
- Implementing the optimal decision threshold in production
- Defining risk categories for clinical decision support
- Justifying intervention programs to stakeholders
- Planning resource allocation
- Proceeding to Phase 5: Fairness Evaluation

## Citation

If you use these results, please cite the hospital readmission risk prediction project.
"""
        
        readme_path = Path(output_dir) / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add README",
            token=token
        )
        print(f"   ✅ Uploaded: README.md")
        uploaded_count += 1
    except Exception as e:
        print(f"   ⚠️  Could not create/upload README: {e}")
    
    print("\n" + "="*80)
    print(f"✅ Upload Complete: {uploaded_count} files uploaded")
    print("="*80)
    print(f"🌐 Repository URL: https://huggingface.co/{repo_id}")
    print("="*80 + "\n")
    
    return f"https://huggingface.co/{repo_id}"


def print_section(title: str, char: str = "=", width: int = 80):
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


# ============================================================================
# THRESHOLD VISUALIZER
# ============================================================================

class ThresholdVisualizer:
    """
    Visualizations for threshold optimization analysis.
    
    Generates plots 1-4 from the Phase 4 README:
    1. Expected Value vs Threshold Curve
    2. Cost-Benefit Analysis
    3. Threshold Impact on Classification Metrics
    4. Confusion Matrix at Optimal Threshold
    """
    
    @staticmethod
    def plot_expected_value_curve(
        threshold_optimizer: ThresholdOptimizer,
        break_even_threshold: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot expected value across all tested thresholds.
        
        Args:
            threshold_optimizer: Fitted ThresholdOptimizer
            break_even_threshold: Break-even threshold (optional)
            save_path: Path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        if threshold_optimizer.threshold_results is None:
            raise ValueError("Run find_optimal_threshold() first")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        df = threshold_optimizer.threshold_results
        
        # Plot expected value curve
        ax.plot(df['threshold'], df['expected_value'], 
                linewidth=3, color='#1f77b4', label='Expected Value')
        
        # Mark optimal threshold
        ax.axvline(threshold_optimizer.optimal_threshold, 
                   color='green', linestyle='--', linewidth=2.5,
                   label=f'Optimal Threshold ({threshold_optimizer.optimal_threshold:.4f})')
        ax.scatter([threshold_optimizer.optimal_threshold], 
                  [threshold_optimizer.optimal_ev],
                  color='green', s=200, zorder=5, marker='*',
                  edgecolors='black', linewidths=2)
        
        # Add text annotation for optimal point
        ax.annotate(f'Optimal\n${threshold_optimizer.optimal_ev:,.0f}',
                   xy=(threshold_optimizer.optimal_threshold, threshold_optimizer.optimal_ev),
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))
        
        # Mark break-even threshold
        if break_even_threshold is not None:
            ax.axvline(break_even_threshold,
                      color='orange', linestyle=':', linewidth=2,
                      label=f'Break-Even Threshold ({break_even_threshold:.4f})')
        
        # Zero line
        ax.axhline(0, color='red', linestyle='-', linewidth=1.5, alpha=0.5, label='Break-Even (EV = 0)')
        
        # Fill positive/negative regions
        ax.fill_between(df['threshold'], df['expected_value'], 0,
                       where=(df['expected_value'] >= 0),
                       color='green', alpha=0.1, label='Positive EV')
        ax.fill_between(df['threshold'], df['expected_value'], 0,
                       where=(df['expected_value'] < 0),
                       color='red', alpha=0.1, label='Negative EV')
        
        ax.set_xlabel('Decision Threshold', fontsize=13, fontweight='bold')
        ax.set_ylabel('Expected Value ($)', fontsize=13, fontweight='bold')
        ax.set_title('Expected Value vs Threshold', fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Expected value curve saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_cost_benefit_analysis(
        threshold_optimizer: ThresholdOptimizer,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot costs vs benefits across thresholds.
        
        Args:
            threshold_optimizer: Fitted ThresholdOptimizer
            save_path: Path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        if threshold_optimizer.threshold_results is None:
            raise ValueError("Run find_optimal_threshold() first")
        
        df = threshold_optimizer.threshold_results
        thresholds = df['threshold'].values
        
        # Calculate costs and benefits at each threshold
        total_costs = []
        total_benefits = []
        net_benefits = []
        
        for threshold in thresholds:
            y_pred = (threshold_optimizer.y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(threshold_optimizer.y_true, y_pred).ravel()
            
            # Costs (negative values)
            cost = abs(fp * threshold_optimizer.cost_params['fp_cost'] + 
                      fn * threshold_optimizer.cost_params['fn_cost'])
            
            # Benefits (positive values)
            benefit = tp * threshold_optimizer.cost_params['tp_benefit']
            
            total_costs.append(cost)
            total_benefits.append(benefit)
            net_benefits.append(benefit - cost)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Costs vs Benefits
        ax1.plot(thresholds, total_benefits, linewidth=3, 
                color='green', label='Total Benefits', marker='o', markersize=3)
        ax1.plot(thresholds, total_costs, linewidth=3, 
                color='red', label='Total Costs', marker='s', markersize=3)
        
        # Mark optimal threshold
        optimal_idx = np.argmin(np.abs(thresholds - threshold_optimizer.optimal_threshold))
        ax1.axvline(threshold_optimizer.optimal_threshold,
                   color='blue', linestyle='--', linewidth=2,
                   label=f'Optimal ({threshold_optimizer.optimal_threshold:.4f})')
        
        ax1.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amount ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Costs vs Benefits by Threshold', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Right plot: Net Benefit
        colors = ['green' if nb >= 0 else 'red' for nb in net_benefits]
        ax2.bar(thresholds, net_benefits, width=0.003, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.axvline(threshold_optimizer.optimal_threshold,
                   color='blue', linestyle='--', linewidth=2,
                   label=f'Optimal ({threshold_optimizer.optimal_threshold:.4f})')
        
        ax2.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Net Benefit ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Net Benefit by Threshold', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.suptitle('Cost-Benefit Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Cost-benefit analysis saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_metrics_vs_threshold(
        threshold_optimizer: ThresholdOptimizer,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot precision, recall, F1, TPR, FPR vs threshold.
        
        Args:
            threshold_optimizer: Fitted ThresholdOptimizer
            save_path: Path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        if threshold_optimizer.threshold_results is None:
            raise ValueError("Run find_optimal_threshold() first")
        
        thresholds = threshold_optimizer.threshold_results['threshold'].values
        
        # Calculate metrics at each threshold
        precisions = []
        recalls = []
        f1_scores = []
        tprs = []
        fprs = []
        
        for threshold in thresholds:
            metrics = threshold_optimizer.get_metrics_at_threshold(threshold)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
            tprs.append(metrics['tpr'])
            fprs.append(metrics['fpr'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Precision, Recall, F1
        ax1.plot(thresholds, precisions, linewidth=2.5, label='Precision', marker='o', markersize=4)
        ax1.plot(thresholds, recalls, linewidth=2.5, label='Recall (TPR)', marker='s', markersize=4)
        ax1.plot(thresholds, f1_scores, linewidth=2.5, label='F1-Score', marker='^', markersize=4)
        
        # Mark optimal threshold
        ax1.axvline(threshold_optimizer.optimal_threshold,
                   color='red', linestyle='--', linewidth=2,
                   label=f'Optimal ({threshold_optimizer.optimal_threshold:.4f})', alpha=0.7)
        
        ax1.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Classification Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Bottom plot: TPR and FPR
        ax2.plot(thresholds, tprs, linewidth=2.5, label='True Positive Rate (Sensitivity)', 
                color='green', marker='o', markersize=4)
        ax2.plot(thresholds, fprs, linewidth=2.5, label='False Positive Rate', 
                color='red', marker='s', markersize=4)
        
        # Mark optimal threshold
        ax2.axvline(threshold_optimizer.optimal_threshold,
                   color='red', linestyle='--', linewidth=2,
                   label=f'Optimal ({threshold_optimizer.optimal_threshold:.4f})', alpha=0.7)
        
        ax2.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Rate', fontsize=12, fontweight='bold')
        ax2.set_title('TPR and FPR vs Threshold', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Metrics vs threshold saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(
        threshold_optimizer: ThresholdOptimizer,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix with financial annotations at optimal threshold.
        
        Args:
            threshold_optimizer: Fitted ThresholdOptimizer
            save_path: Path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        if threshold_optimizer.optimal_threshold is None:
            raise ValueError("Run find_optimal_threshold() first")
        
        # Get predictions at optimal threshold
        y_pred = (threshold_optimizer.y_pred_proba >= threshold_optimizer.optimal_threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(threshold_optimizer.y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate financial impact
        tp_value = tp * threshold_optimizer.cost_params['tp_benefit']
        fp_value = fp * threshold_optimizer.cost_params['fp_cost']
        fn_value = fn * threshold_optimizer.cost_params['fn_cost']
        tn_value = tn * threshold_optimizer.cost_params['tn_cost']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'}, ax=ax,
                   xticklabels=['Predicted: No Readmission', 'Predicted: Readmission'],
                   yticklabels=['Actual: No Readmission', 'Actual: Readmission'])
        
        # Add custom annotations with counts and financial impact
        annotations = [
            [f'TN\n{tn:,}\n${tn_value:,.0f}', f'FP\n{fp:,}\n${fp_value:,.0f}'],
            [f'FN\n{fn:,}\n${fn_value:,.0f}', f'TP\n{tp:,}\n${tp_value:,.0f}']
        ]
        
        for i in range(2):
            for j in range(2):
                text = annotations[i][j]
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center',
                       fontsize=14, fontweight='bold', color=color)
        
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        
        title_text = (f'Confusion Matrix at Optimal Threshold ({threshold_optimizer.optimal_threshold:.4f})\n'
                     f'Total Expected Value: ${threshold_optimizer.optimal_ev:,.2f}')
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Confusion matrix saved: {save_path}")
        
        return fig


# ============================================================================
# ROI VISUALIZER
# ============================================================================

class ROIVisualizer:
    """
    Visualizations for ROI and resource planning analysis.
    
    Generates plots 5-8 from the Phase 4 README:
    5. Risk Category Distribution
    6. ROI Sensitivity Analysis
    7. Intervention Volume Forecast
    8. Cost Savings Projection
    """
    
    @staticmethod
    def plot_risk_category_distribution(
        risk_mapper: RiskCategoryMapper,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot patient distribution by risk category with readmission rates.
        
        Args:
            risk_mapper: RiskCategoryMapper
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        categories = risk_mapper.assign_risk_categories(y_pred_proba)
        stats_df = risk_mapper.get_category_statistics(y_true, y_pred_proba)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Patient count by risk category
        colors = ['#06A77D', '#F18F01', '#E63946']  # Green, Orange, Red
        bars1 = ax1.bar(stats_df['risk_category'], stats_df['n_patients'],
                       color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, n, pct in zip(bars1, stats_df['n_patients'], stats_df['percentage']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{n:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('Risk Category', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Number of Patients', fontsize=13, fontweight='bold')
        ax1.set_title('Patient Distribution by Risk Category', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Right plot: Actual readmission rate by category
        bars2 = ax2.bar(stats_df['risk_category'], 
                       stats_df['actual_readmission_rate'] * 100,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, rate in zip(bars2, stats_df['actual_readmission_rate']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate*100:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_xlabel('Risk Category', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Actual Readmission Rate (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Actual Readmission Rate by Risk Category', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add threshold annotations
        threshold_text = (f'Thresholds: Low < {risk_mapper.low_threshold:.3f} | '
                         f'Medium: {risk_mapper.low_threshold:.3f}-{risk_mapper.high_threshold:.3f} | '
                         f'High ≥ {risk_mapper.high_threshold:.3f}')
        fig.text(0.5, 0.02, threshold_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Risk Category Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Risk category distribution saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_sensitivity_analysis(
        roi_analyzer: ROIAnalyzer,
        scenarios: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROI sensitivity analysis across different cost scenarios.
        
        Args:
            roi_analyzer: ROIAnalyzer
            scenarios: Dictionary of scenarios to test
            save_path: Path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        sensitivity_df = roi_analyzer.sensitivity_analysis(scenarios)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Net benefit by scenario
        colors = ['#E63946' if nb < 0 else '#F18F01' if nb < sensitivity_df['net_benefit'].mean() 
                 else '#06A77D' for nb in sensitivity_df['net_benefit']]
        
        bars1 = ax1.bar(sensitivity_df['scenario'], sensitivity_df['net_benefit'],
                       color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, nb in zip(bars1, sensitivity_df['net_benefit']):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${nb:,.0f}',
                    ha='center', va=va, fontsize=11, fontweight='bold')
        
        ax1.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Scenario', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Net Benefit ($)', fontsize=13, fontweight='bold')
        ax1.set_title('Net Benefit by Scenario', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Right plot: ROI percentage by scenario
        colors_roi = ['#E63946' if roi < 0 else '#F18F01' if roi < 100 
                     else '#06A77D' for roi in sensitivity_df['roi_percentage']]
        
        bars2 = ax2.bar(sensitivity_df['scenario'], sensitivity_df['roi_percentage'],
                       color=colors_roi, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, roi in zip(bars2, sensitivity_df['roi_percentage']):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{roi:.1f}%',
                    ha='center', va=va, fontsize=11, fontweight='bold')
        
        ax2.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Break-even')
        ax2.axhline(100, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='100% ROI')
        ax2.set_xlabel('Scenario', fontsize=13, fontweight='bold')
        ax2.set_ylabel('ROI (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Return on Investment by Scenario', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('ROI Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Sensitivity analysis saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_intervention_volume_forecast(
        risk_mapper: RiskCategoryMapper,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot intervention volume by risk category.
        
        Args:
            risk_mapper: RiskCategoryMapper
            y_pred_proba: Predicted probabilities
            save_path: Path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        categories = risk_mapper.assign_risk_categories(y_pred_proba)
        category_counts = pd.Series(categories).value_counts()
        
        # Define intervention types
        interventions = {
            'Low': 'Standard Discharge',
            'Medium': 'Enhanced Follow-up Call',
            'High': 'Intensive Case Management'
        }
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {'Low': '#06A77D', 'Medium': '#F18F01', 'High': '#E63946'}
        categories_ordered = ['Low', 'Medium', 'High']
        
        # Create bars
        bars = []
        for cat in categories_ordered:
            count = category_counts.get(cat, 0)
            bars.append(count)
        
        bar_plot = ax.bar(categories_ordered, bars,
                         color=[colors[c] for c in categories_ordered],
                         alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels and intervention types
        total_patients = len(y_pred_proba)
        for i, (bar, cat) in enumerate(zip(bar_plot, categories_ordered)):
            height = bar.get_height()
            pct = height / total_patients * 100
            
            # Count label
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Intervention type label
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                   interventions[cat],
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        
        ax.set_xlabel('Risk Category', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Patients', fontsize=13, fontweight='bold')
        ax.set_title('Intervention Volume Forecast by Risk Category', fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add total annotation
        total_text = f'Total Patients: {total_patients:,}'
        ax.text(0.98, 0.98, total_text, transform=ax.transAxes,
               fontsize=11, fontweight='bold', ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Intervention volume forecast saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_cost_savings_projection(
        roi_analyzer: ROIAnalyzer,
        years: List[int] = [1, 3, 5],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot projected cost savings over multiple years.
        
        Args:
            roi_analyzer: ROIAnalyzer
            years: List of years to project
            save_path: Path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        metrics = roi_analyzer.calculate_roi_metrics()
        
        # Calculate annual metrics (assuming test set is representative of annual volume)
        annual_net_benefit = metrics['net_benefit']
        annual_savings = metrics['total_savings']
        annual_intervention_cost = metrics['intervention_cost_total']
        annual_prevented_savings = metrics['prevented_readmission_savings']
        
        # Project over years
        projections = []
        for year in years:
            projections.append({
                'year': year,
                'cumulative_net_benefit': annual_net_benefit * year,
                'cumulative_savings': annual_savings * year,
                'cumulative_intervention_cost': annual_intervention_cost * year,
                'cumulative_prevented_savings': annual_prevented_savings * year
            })
        
        proj_df = pd.DataFrame(projections)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Cumulative costs and benefits
        x = np.arange(len(years))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, proj_df['cumulative_prevented_savings'],
                       width, label='Prevented Readmission Savings',
                       color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, proj_df['cumulative_intervention_cost'],
                       width, label='Intervention Costs',
                       color='#E63946', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.0f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Year', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Amount ($)', fontsize=13, fontweight='bold')
        ax1.set_title('Cumulative Costs vs Benefits', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Year {y}' for y in years])
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Right plot: Cumulative net benefit
        ax2.plot(years, proj_df['cumulative_net_benefit'],
                marker='o', markersize=12, linewidth=3,
                color='#1f77b4', label='Cumulative Net Benefit')
        ax2.fill_between(years, 0, proj_df['cumulative_net_benefit'],
                        alpha=0.3, color='#1f77b4')
        
        # Add value labels
        for year, net_benefit in zip(years, proj_df['cumulative_net_benefit']):
            ax2.text(year, net_benefit, f'${net_benefit:,.0f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Year', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Cumulative Net Benefit ($)', fontsize=13, fontweight='bold')
        ax2.set_title('Projected Cumulative Net Benefit', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.suptitle('Cost Savings Projection', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Cost savings projection saved: {save_path}")
        
        return fig
    
    @staticmethod
    def generate_all_visualizations(
        threshold_optimizer: ThresholdOptimizer,
        risk_mapper: RiskCategoryMapper,
        roi_analyzer: ROIAnalyzer,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        output_dir: str = "./phase-4-optimal-threshold-ROI-analysis/visualizations",
        scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> List[str]:
        """
        Generate all 8 visualizations at once.
        
        Args:
            threshold_optimizer: Fitted ThresholdOptimizer
            risk_mapper: RiskCategoryMapper
            roi_analyzer: ROIAnalyzer
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            output_dir: Output directory for visualizations
            scenarios: Optional scenarios for sensitivity analysis
            
        Returns:
            list: Paths to saved visualizations
        """
        print("\n" + "="*80)
        print("📊 Generating All Phase 4 Visualizations")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        # Calculate break-even threshold
        break_even = threshold_optimizer.calculate_break_even_threshold()
        
        # 1. Expected Value Curve
        print("\n1️⃣  Generating Expected Value vs Threshold Curve...")
        path1 = output_path / "1_expected_value_curve.png"
        ThresholdVisualizer.plot_expected_value_curve(
            threshold_optimizer, break_even, str(path1)
        )
        saved_paths.append(str(path1))
        
        # 2. Cost-Benefit Analysis
        print("2️⃣  Generating Cost-Benefit Analysis...")
        path2 = output_path / "2_cost_benefit_analysis.png"
        ThresholdVisualizer.plot_cost_benefit_analysis(
            threshold_optimizer, str(path2)
        )
        saved_paths.append(str(path2))
        
        # 3. Metrics vs Threshold
        print("3️⃣  Generating Classification Metrics vs Threshold...")
        path3 = output_path / "3_metrics_vs_threshold.png"
        ThresholdVisualizer.plot_metrics_vs_threshold(
            threshold_optimizer, str(path3)
        )
        saved_paths.append(str(path3))
        
        # 4. Confusion Matrix
        print("4️⃣  Generating Confusion Matrix at Optimal Threshold...")
        path4 = output_path / "4_confusion_matrix.png"
        ThresholdVisualizer.plot_confusion_matrix(
            threshold_optimizer, str(path4)
        )
        saved_paths.append(str(path4))
        
        # 5. Risk Category Distribution
        print("5️⃣  Generating Risk Category Distribution...")
        path5 = output_path / "5_risk_category_distribution.png"
        ROIVisualizer.plot_risk_category_distribution(
            risk_mapper, y_true, y_pred_proba, str(path5)
        )
        saved_paths.append(str(path5))
        
        # 6. Sensitivity Analysis
        if scenarios is None:
            scenarios = {
                'Conservative': {'readmission_cost': 12000, 'intervention_cost': 700},
                'Base Case': threshold_optimizer.cost_params,
                'Aggressive': {'readmission_cost': 18000, 'intervention_cost': 400}
            }
        print("6️⃣  Generating ROI Sensitivity Analysis...")
        path6 = output_path / "6_roi_sensitivity_analysis.png"
        ROIVisualizer.plot_sensitivity_analysis(
            roi_analyzer, scenarios, str(path6)
        )
        saved_paths.append(str(path6))
        
        # 7. Intervention Volume Forecast
        print("7️⃣  Generating Intervention Volume Forecast...")
        path7 = output_path / "7_intervention_volume_forecast.png"
        ROIVisualizer.plot_intervention_volume_forecast(
            risk_mapper, y_pred_proba, str(path7)
        )
        saved_paths.append(str(path7))
        
        # 8. Cost Savings Projection
        print("8️⃣  Generating Cost Savings Projection...")
        path8 = output_path / "8_cost_savings_projection.png"
        ROIVisualizer.plot_cost_savings_projection(
            roi_analyzer, years=[1, 3, 5], save_path=str(path8)
        )
        saved_paths.append(str(path8))
        
        print("\n" + "="*80)
        print(f"✅ All {len(saved_paths)} visualizations generated successfully!")
        print(f"📁 Output directory: {output_dir}")
        print("="*80 + "\n")
        
        return saved_paths
