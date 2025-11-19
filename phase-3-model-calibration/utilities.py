"""
Phase 3: Model Calibration Utilities

This module provides comprehensive utilities for calibrating machine learning models
for hospital readmission prediction. Includes calibration techniques, validation methods,
risk score mapping, and fairness-aware calibration.

Supported Models:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost/LightGBM)

Calibration Techniques:
1. Platt Scaling (Logistic regression transformation)
2. Isotonic Regression (Non-parametric calibration)
3. Group-Specific Calibration (Demographics-based)

Validation Methods:
- Reliability Diagrams
- Brier Score
- Hosmer-Lemeshow Test
- Risk Score Mapping
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy import stats
import warnings
import pickle
import json
from pathlib import Path
import os
import joblib


# ============================================================================
# HUGGINGFACE MODEL DOWNLOAD
# ============================================================================

def download_model_from_hf(
    repo_id: str = "auphong2707/hospital-readmission-lgbm",
    model_filename: str = "gradient_boosting_model.joblib",
    cache_dir: str = "./models/downloaded",
    force_download: bool = False
) -> tuple:
    """
    Download trained model and associated artifacts from HuggingFace Hub.
    
    This function downloads a pre-trained model from HuggingFace Hub repository,
    along with its training summary and metrics. Useful for loading models for
    calibration without retraining.
    
    Parameters:
    -----------
    repo_id : str
        HuggingFace repository ID (format: "username/repo-name")
        Default: "auphong2707/hospital-readmission-lgbm"
    model_filename : str
        Name of the model file to download
        Default: "gradient_boosting_model.joblib"
    cache_dir : str
        Local directory to cache downloaded files
        Default: "./models/downloaded"
    force_download : bool
        If True, re-download even if file exists locally
        Default: False
    
    Returns:
    --------
    tuple: (model, summary_dict)
        model: Loaded sklearn/LightGBM model object
        summary_dict: Dictionary containing training summary and metrics
    
    Raises:
    -------
    ImportError: If huggingface_hub is not installed
    FileNotFoundError: If model file not found in repository
    
    Example:
    --------
    >>> # Download pre-trained LightGBM model
    >>> model, summary = download_model_from_hf()
    >>> print(f"Model ROC-AUC: {summary['final_test_metrics']['roc_auc']:.4f}")
    >>> 
    >>> # Use model for predictions
    >>> predictions = model.predict_proba(X_test)[:, 1]
    
    Notes:
    ------
    - Requires huggingface_hub: pip install huggingface_hub
    - Downloaded files are cached locally to avoid re-downloading
    - Automatically loads model and training summary
    - No authentication token required for public repositories
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models. "
            "Install with: pip install huggingface_hub"
        )
    
    print(f"\n{'='*80}")
    print(f"📥 Downloading Model from HuggingFace Hub")
    print(f"{'='*80}")
    print(f"Repository: {repo_id}")
    print(f"Model: {model_filename}")
    print(f"Cache directory: {cache_dir}")
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download model file
        print(f"\n⏳ Downloading model file...")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            cache_dir=cache_dir,
            force_download=force_download
        )
        print(f"✅ Model downloaded: {model_path}")
        
        # Load model
        print(f"⏳ Loading model...")
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        
        # Download training summary (optional, but recommended)
        summary = {}
        try:
            print(f"\n⏳ Downloading training summary...")
            summary_path = hf_hub_download(
                repo_id=repo_id,
                filename="training_summary.json",
                cache_dir=cache_dir,
                force_download=force_download
            )
            print(f"✅ Summary downloaded: {summary_path}")
            
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # Print key information
            print(f"\n{'='*80}")
            print(f"📊 Model Information")
            print(f"{'='*80}")
            print(f"Model Type: {summary.get('model', 'N/A')}")
            print(f"Training Date: {summary.get('timestamp', 'N/A')}")
            print(f"Environment: {summary.get('environment', 'N/A')}")
            print(f"Device: {summary.get('device', 'N/A')}")
            
            # Cross-validation results
            cv_results = summary.get('cross_validation', {})
            if cv_results:
                print(f"\n📈 Cross-Validation Results:")
                print(f"   Mean ROC-AUC: {cv_results.get('mean_roc_auc', 'N/A'):.4f}")
                print(f"   Std ROC-AUC: {cv_results.get('std_roc_auc', 'N/A'):.4f}")
                print(f"   K-Folds: {cv_results.get('n_folds', 'N/A')}")
            
            # Final test metrics
            test_metrics = summary.get('final_test_metrics', {})
            if test_metrics:
                print(f"\n🎯 Final Test Set Metrics:")
                print(f"   ROC-AUC: {test_metrics.get('roc_auc', 'N/A'):.4f}")
                print(f"   PR-AUC: {test_metrics.get('pr_auc', 'N/A'):.4f}")
                print(f"   F1 Score: {test_metrics.get('f1', 'N/A'):.4f}")
                print(f"   Precision: {test_metrics.get('precision', 'N/A'):.4f}")
                print(f"   Recall: {test_metrics.get('recall', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"⚠️  Could not download training summary: {e}")
            print(f"   Continuing without summary information...")
        
        # Download metrics file (optional)
        try:
            print(f"\n⏳ Downloading metrics file...")
            metrics_path = hf_hub_download(
                repo_id=repo_id,
                filename="gradient_boosting_metrics.json",
                cache_dir=cache_dir,
                force_download=force_download
            )
            print(f"✅ Metrics downloaded: {metrics_path}")
        except Exception as e:
            print(f"⚠️  Could not download metrics file: {e}")
        
        print(f"\n{'='*80}")
        print(f"✅ Download Complete!")
        print(f"{'='*80}")
        print(f"🌐 Repository: https://huggingface.co/{repo_id}")
        print(f"💾 Local cache: {cache_dir}")
        print(f"{'='*80}\n")
        
        return model, summary
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check repository exists: https://huggingface.co/{repo_id}")
        print(f"2. Verify model filename: {model_filename}")
        print(f"3. Check internet connection")
        print(f"4. Try with force_download=True")
        raise


# ============================================================================
# CALIBRATION CLASSES
# ============================================================================

class ModelCalibrator:
    """
    Main calibration class for hospital readmission models.
    
    Provides Platt scaling and Isotonic regression calibration methods.
    """
    
    def __init__(self, method: str = 'platt', cv: int = 5):
        """
        Initialize the calibrator.
        
        Parameters:
        -----------
        method : str
            Calibration method - 'platt' (sigmoid) or 'isotonic'
        cv : int
            Number of cross-validation folds for calibration
        """
        if method not in ['platt', 'isotonic', 'sigmoid']:
            raise ValueError("method must be 'platt', 'sigmoid', or 'isotonic'")
        
        # Normalize method name
        self.method = 'sigmoid' if method == 'platt' else method
        self.cv = cv
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> 'ModelCalibrator':
        """
        Fit the calibration model on validation data.
        
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
        
        if self.method == 'sigmoid':
            # Platt scaling: fit logistic regression on predicted probabilities
            self.calibrator = LogisticRegression(penalty='none', solver='lbfgs')
            # Reshape for sklearn
            X = y_pred_proba.reshape(-1, 1)
            self.calibrator.fit(X, y_true)
            
        elif self.method == 'isotonic':
            # Isotonic regression: monotonic calibration
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_true)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to uncalibrated probabilities.
        
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
        
        if self.method == 'sigmoid':
            X = y_pred_proba.reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(X)[:, 1]
        else:  # isotonic
            calibrated = self.calibrator.predict(y_pred_proba)
        
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


class GroupSpecificCalibrator:
    """
    Calibrate models separately for different demographic groups.
    
    Ensures fair calibration across protected attributes like age, race, gender.
    """
    
    def __init__(self, method: str = 'platt', min_group_size: int = 100):
        """
        Initialize group-specific calibrator.
        
        Parameters:
        -----------
        method : str
            Calibration method - 'platt' or 'isotonic'
        min_group_size : int
            Minimum samples required for group-specific calibration
        """
        self.method = method
        self.min_group_size = min_group_size
        self.calibrators = {}
        self.global_calibrator = None
        self.groups_calibrated = []
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
            groups: np.ndarray) -> 'GroupSpecificCalibrator':
        """
        Fit separate calibrators for each demographic group.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba : np.ndarray
            Uncalibrated predicted probabilities
        groups : np.ndarray
            Group labels (e.g., age groups, race categories)
            
        Returns:
        --------
        self : GroupSpecificCalibrator
            Fitted calibrator instance
        """
        y_true = np.array(y_true).ravel()
        y_pred_proba = np.array(y_pred_proba).ravel()
        groups = np.array(groups).ravel()
        
        # Fit global calibrator as fallback
        self.global_calibrator = ModelCalibrator(method=self.method)
        self.global_calibrator.fit(y_true, y_pred_proba)
        
        # Fit group-specific calibrators
        unique_groups = np.unique(groups)
        for group in unique_groups:
            group_mask = groups == group
            group_size = np.sum(group_mask)
            
            if group_size >= self.min_group_size:
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred_proba[group_mask]
                
                # Check if group has both classes
                if len(np.unique(y_true_group)) > 1:
                    calibrator = ModelCalibrator(method=self.method)
                    calibrator.fit(y_true_group, y_pred_group)
                    self.calibrators[group] = calibrator
                    self.groups_calibrated.append(group)
                else:
                    warnings.warn(f"Group {group} has only one class. Using global calibrator.")
            else:
                warnings.warn(f"Group {group} size ({group_size}) < min_group_size. Using global calibrator.")
        
        return self
    
    def predict_proba(self, y_pred_proba: np.ndarray, 
                      groups: np.ndarray) -> np.ndarray:
        """
        Apply group-specific calibration.
        
        Parameters:
        -----------
        y_pred_proba : np.ndarray
            Uncalibrated predicted probabilities
        groups : np.ndarray
            Group labels
            
        Returns:
        --------
        calibrated_proba : np.ndarray
            Calibrated probabilities
        """
        y_pred_proba = np.array(y_pred_proba).ravel()
        groups = np.array(groups).ravel()
        
        calibrated = np.zeros_like(y_pred_proba)
        
        for group in np.unique(groups):
            group_mask = groups == group
            
            if group in self.calibrators:
                # Use group-specific calibrator
                calibrated[group_mask] = self.calibrators[group].predict_proba(
                    y_pred_proba[group_mask]
                )
            else:
                # Use global calibrator
                calibrated[group_mask] = self.global_calibrator.predict_proba(
                    y_pred_proba[group_mask]
                )
        
        return calibrated


class CalibrationMetrics:
    """
    Comprehensive calibration evaluation metrics.
    
    Includes Brier score, log loss, Hosmer-Lemeshow test, and reliability metrics.
    """
    
    @staticmethod
    def brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate Brier score (lower is better, target < 0.15).
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba : np.ndarray
            Predicted probabilities
            
        Returns:
        --------
        score : float
            Brier score
        """
        return brier_score_loss(y_true, y_pred_proba)
    
    @staticmethod
    def hosmer_lemeshow_test(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                            n_bins: int = 10) -> Dict[str, float]:
        """
        Hosmer-Lemeshow goodness-of-fit test for calibration.
        
        H0: Model is well-calibrated
        Target: p-value > 0.05 (fail to reject H0)
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        n_bins : int
            Number of bins for grouping probabilities
            
        Returns:
        --------
        results : dict
            Contains chi2_statistic, p_value, df, and interpretation
        """
        y_true = np.array(y_true).ravel()
        y_pred_proba = np.array(y_pred_proba).ravel()
        
        # Create bins based on predicted probabilities
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = np.digitize(y_pred_proba, bin_edges[:-1]) - 1
        bins = np.clip(bins, 0, n_bins - 1)
        
        # Calculate observed and expected events per bin
        chi2_stat = 0.0
        df = 0
        
        for i in range(n_bins):
            mask = bins == i
            n_samples = np.sum(mask)
            
            if n_samples > 0:
                observed = np.sum(y_true[mask])
                expected = np.sum(y_pred_proba[mask])
                
                # Avoid division by zero
                if expected > 0 and expected < n_samples:
                    # Chi-square component for positive class
                    chi2_stat += (observed - expected) ** 2 / expected
                    # Chi-square component for negative class
                    chi2_stat += ((n_samples - observed) - (n_samples - expected)) ** 2 / (n_samples - expected)
                    df += 1
        
        # Degrees of freedom: n_bins - 2 (estimating intercept and slope)
        df = max(df - 2, 1)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'is_well_calibrated': p_value > 0.05,
            'interpretation': 'Well-calibrated' if p_value > 0.05 else 'Poorly calibrated'
        }
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Weighted average of absolute calibration error across bins.
        Target: ECE < 0.05 (within ±5% of diagonal)
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        n_bins : int
            Number of bins
            
        Returns:
        --------
        ece : float
            Expected calibration error
        """
        y_true = np.array(y_true).ravel()
        y_pred_proba = np.array(y_pred_proba).ravel()
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = np.digitize(y_pred_proba, bin_edges[:-1]) - 1
        bins = np.clip(bins, 0, n_bins - 1)
        
        ece = 0.0
        n_total = len(y_true)
        
        for i in range(n_bins):
            mask = bins == i
            n_samples = np.sum(mask)
            
            if n_samples > 0:
                avg_predicted = np.mean(y_pred_proba[mask])
                avg_observed = np.mean(y_true[mask])
                ece += (n_samples / n_total) * np.abs(avg_predicted - avg_observed)
        
        return ece
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, 
                           y_pred_proba_uncalibrated: np.ndarray,
                           y_pred_proba_calibrated: np.ndarray) -> Dict[str, Any]:
        """
        Compute all calibration metrics for before/after comparison.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba_uncalibrated : np.ndarray
            Uncalibrated predicted probabilities
        y_pred_proba_calibrated : np.ndarray
            Calibrated predicted probabilities
            
        Returns:
        --------
        metrics : dict
            Comprehensive calibration metrics
        """
        metrics = {
            'uncalibrated': {
                'brier_score': CalibrationMetrics.brier_score(y_true, y_pred_proba_uncalibrated),
                'log_loss': log_loss(y_true, y_pred_proba_uncalibrated),
                'roc_auc': roc_auc_score(y_true, y_pred_proba_uncalibrated),
                'ece': CalibrationMetrics.expected_calibration_error(y_true, y_pred_proba_uncalibrated),
                'hosmer_lemeshow': CalibrationMetrics.hosmer_lemeshow_test(y_true, y_pred_proba_uncalibrated)
            },
            'calibrated': {
                'brier_score': CalibrationMetrics.brier_score(y_true, y_pred_proba_calibrated),
                'log_loss': log_loss(y_true, y_pred_proba_calibrated),
                'roc_auc': roc_auc_score(y_true, y_pred_proba_calibrated),
                'ece': CalibrationMetrics.expected_calibration_error(y_true, y_pred_proba_calibrated),
                'hosmer_lemeshow': CalibrationMetrics.hosmer_lemeshow_test(y_true, y_pred_proba_calibrated)
            }
        }
        
        # Calculate improvements
        metrics['improvement'] = {
            'brier_score_delta': metrics['uncalibrated']['brier_score'] - metrics['calibrated']['brier_score'],
            'ece_delta': metrics['uncalibrated']['ece'] - metrics['calibrated']['ece'],
            'log_loss_delta': metrics['uncalibrated']['log_loss'] - metrics['calibrated']['log_loss']
        }
        
        return metrics


class RiskScoreMapper:
    """
    Map calibrated probabilities to clinical risk categories.
    
    Risk Categories:
    - Low (0-5%): Standard discharge
    - Medium (5-15%): Enhanced education, 1-week follow-up
    - High (15%+): Intensive case management, home visit
    """
    
    def __init__(self, low_threshold: float = 0.05, high_threshold: float = 0.15):
        """
        Initialize risk score mapper.
        
        Parameters:
        -----------
        low_threshold : float
            Threshold between low and medium risk (default: 5%)
        high_threshold : float
            Threshold between medium and high risk (default: 15%)
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        self.clinical_actions = {
            'Low': 'Standard discharge',
            'Medium': 'Enhanced education, 1-week follow-up',
            'High': 'Intensive case management, home visit'
        }
    
    def map_to_risk_category(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Map probabilities to risk categories (0=Low, 1=Medium, 2=High).
        
        Parameters:
        -----------
        probabilities : np.ndarray
            Calibrated probabilities
            
        Returns:
        --------
        risk_categories : np.ndarray
            Risk category labels (0, 1, 2)
        """
        probabilities = np.array(probabilities).ravel()
        
        risk_categories = np.zeros(len(probabilities), dtype=int)
        risk_categories[probabilities >= self.low_threshold] = 1
        risk_categories[probabilities >= self.high_threshold] = 2
        
        return risk_categories
    
    def get_risk_labels(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Get risk labels as strings.
        
        Parameters:
        -----------
        probabilities : np.ndarray
            Calibrated probabilities
            
        Returns:
        --------
        risk_labels : np.ndarray
            Risk labels ('Low', 'Medium', 'High')
        """
        categories = self.map_to_risk_category(probabilities)
        return np.array([self.risk_labels[cat] for cat in categories])
    
    def get_clinical_actions(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Get recommended clinical actions based on risk.
        
        Parameters:
        -----------
        probabilities : np.ndarray
            Calibrated probabilities
            
        Returns:
        --------
        actions : np.ndarray
            Recommended clinical actions
        """
        risk_labels = self.get_risk_labels(probabilities)
        return np.array([self.clinical_actions[label] for label in risk_labels])
    
    def validate_risk_scores(self, y_true: np.ndarray, 
                            probabilities: np.ndarray) -> pd.DataFrame:
        """
        Validate risk scores by comparing predicted vs. actual readmission rates.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        probabilities : np.ndarray
            Calibrated probabilities
            
        Returns:
        --------
        validation_table : pd.DataFrame
            Risk category validation statistics
        """
        risk_categories = self.map_to_risk_category(probabilities)
        risk_labels = self.get_risk_labels(probabilities)
        
        results = []
        for cat in [0, 1, 2]:
            mask = risk_categories == cat
            n_patients = np.sum(mask)
            
            if n_patients > 0:
                actual_rate = np.mean(y_true[mask])
                predicted_rate = np.mean(probabilities[mask])
                n_readmissions = np.sum(y_true[mask])
                
                results.append({
                    'Risk Category': self.risk_labels[cat],
                    'Probability Range': f"{self.low_threshold if cat > 0 else 0:.0%}-{self.high_threshold if cat == 1 else (1.0 if cat == 2 else self.low_threshold):.0%}",
                    'N Patients': int(n_patients),
                    'Actual Readmissions': int(n_readmissions),
                    'Actual Rate': f"{actual_rate:.2%}",
                    'Predicted Rate': f"{predicted_rate:.2%}",
                    'Rate Difference': f"{abs(actual_rate - predicted_rate):.2%}",
                    'Clinical Action': self.clinical_actions[self.risk_labels[cat]]
                })
        
        return pd.DataFrame(results)


class CalibrationVisualizer:
    """
    Create comprehensive calibration visualizations.
    
    Includes reliability diagrams, calibration curves, and risk distribution plots.
    """
    
    @staticmethod
    def plot_reliability_diagram(y_true: np.ndarray,
                                 y_pred_proba_uncalibrated: np.ndarray,
                                 y_pred_proba_calibrated: np.ndarray,
                                 n_bins: int = 10,
                                 title: str = "Reliability Diagram",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot reliability diagram comparing uncalibrated vs. calibrated predictions.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba_uncalibrated : np.ndarray
            Uncalibrated probabilities
        y_pred_proba_calibrated : np.ndarray
            Calibrated probabilities
        n_bins : int
            Number of bins for calibration curve
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate calibration curves
        prob_true_uncal, prob_pred_uncal = calibration_curve(
            y_true, y_pred_proba_uncalibrated, n_bins=n_bins, strategy='uniform'
        )
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_true, y_pred_proba_calibrated, n_bins=n_bins, strategy='uniform'
        )
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Plot calibration curves
        ax.plot(prob_pred_uncal, prob_true_uncal, 's-', 
                label='Uncalibrated', linewidth=2, markersize=8, alpha=0.7)
        ax.plot(prob_pred_cal, prob_true_cal, 'o-', 
                label='Calibrated', linewidth=2, markersize=8, alpha=0.7)
        
        # Calculate metrics
        brier_uncal = brier_score_loss(y_true, y_pred_proba_uncalibrated)
        brier_cal = brier_score_loss(y_true, y_pred_proba_calibrated)
        ece_uncal = CalibrationMetrics.expected_calibration_error(y_true, y_pred_proba_uncalibrated, n_bins)
        ece_cal = CalibrationMetrics.expected_calibration_error(y_true, y_pred_proba_calibrated, n_bins)
        
        # Add metrics to plot
        metrics_text = (
            f"Uncalibrated: Brier={brier_uncal:.4f}, ECE={ece_uncal:.4f}\n"
            f"Calibrated: Brier={brier_cal:.4f}, ECE={ece_cal:.4f}"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives (Observed)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_risk_distribution(probabilities: np.ndarray,
                              risk_mapper: RiskScoreMapper,
                              title: str = "Risk Score Distribution",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of patients across risk categories.
        
        Parameters:
        -----------
        probabilities : np.ndarray
            Calibrated probabilities
        risk_mapper : RiskScoreMapper
            Risk score mapper instance
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        risk_categories = risk_mapper.map_to_risk_category(probabilities)
        risk_labels = risk_mapper.get_risk_labels(probabilities)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot of risk categories
        unique_labels, counts = np.unique(risk_labels, return_counts=True)
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        bar_colors = [colors[label] for label in unique_labels]
        
        ax1.bar(unique_labels, counts, color=bar_colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Risk Category', fontsize=12)
        ax1.set_ylabel('Number of Patients', fontsize=12)
        ax1.set_title('Patient Distribution by Risk Category', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total = len(probabilities)
        for label, count in zip(unique_labels, counts):
            idx = list(unique_labels).index(label)
            percentage = (count / total) * 100
            ax1.text(idx, count, f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Histogram of probabilities with risk thresholds
        ax2.hist(probabilities, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.axvline(risk_mapper.low_threshold, color='orange', linestyle='--', 
                   linewidth=2, label=f'Low/Medium ({risk_mapper.low_threshold:.0%})')
        ax2.axvline(risk_mapper.high_threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Medium/High ({risk_mapper.high_threshold:.0%})')
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Number of Patients', fontsize=12)
        ax2.set_title('Probability Distribution with Risk Thresholds', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_group_calibration(y_true: np.ndarray,
                               y_pred_proba: np.ndarray,
                               groups: np.ndarray,
                               group_name: str = "Demographic Group",
                               n_bins: int = 10,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curves for different demographic groups.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        groups : np.ndarray
            Group labels
        group_name : str
            Name of the grouping variable
        n_bins : int
            Number of bins
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Color map
        colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
        
        # Plot calibration for each group
        for i, group in enumerate(unique_groups):
            mask = groups == group
            if np.sum(mask) > n_bins:  # Only plot if enough samples
                prob_true, prob_pred = calibration_curve(
                    y_true[mask], y_pred_proba[mask], 
                    n_bins=n_bins, strategy='uniform'
                )
                
                brier = brier_score_loss(y_true[mask], y_pred_proba[mask])
                n_samples = np.sum(mask)
                
                ax.plot(prob_pred, prob_true, 'o-', 
                       color=colors[i], label=f'{group} (n={n_samples}, Brier={brier:.3f})',
                       linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives (Observed)', fontsize=12)
        ax.set_title(f'Calibration by {group_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class CalibrationReport:
    """
    Generate comprehensive calibration reports.
    
    Includes all metrics, visualizations, and validation tables.
    """
    
    @staticmethod
    def generate_report(y_true: np.ndarray,
                       y_pred_proba_uncalibrated: np.ndarray,
                       y_pred_proba_calibrated: np.ndarray,
                       model_name: str,
                       calibration_method: str,
                       output_dir: str,
                       groups: Optional[np.ndarray] = None,
                       group_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate complete calibration report with metrics and visualizations.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba_uncalibrated : np.ndarray
            Uncalibrated probabilities
        y_pred_proba_calibrated : np.ndarray
            Calibrated probabilities
        model_name : str
            Name of the model (e.g., 'Logistic Regression', 'Random Forest')
        calibration_method : str
            Calibration method used ('platt', 'isotonic', 'group-specific')
        output_dir : str
            Directory to save report outputs
        groups : np.ndarray, optional
            Group labels for fairness analysis
        group_name : str, optional
            Name of grouping variable
            
        Returns:
        --------
        report : dict
            Complete calibration report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize report
        report = {
            'model_name': model_name,
            'calibration_method': calibration_method,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Compute metrics
        metrics = CalibrationMetrics.compute_all_metrics(
            y_true, y_pred_proba_uncalibrated, y_pred_proba_calibrated
        )
        report['metrics'] = metrics
        
        # Risk score validation
        risk_mapper = RiskScoreMapper()
        validation_table = risk_mapper.validate_risk_scores(y_true, y_pred_proba_calibrated)
        report['risk_validation'] = validation_table.to_dict('records')
        
        # Success criteria checks
        success_criteria = {
            'brier_score_target': metrics['calibrated']['brier_score'] < 0.15,
            'ece_target': metrics['calibrated']['ece'] < 0.05,
            'hosmer_lemeshow_target': metrics['calibrated']['hosmer_lemeshow']['is_well_calibrated']
        }
        report['success_criteria'] = success_criteria
        report['meets_all_criteria'] = all(success_criteria.values())
        
        # Generate visualizations
        vis = CalibrationVisualizer()
        
        # Reliability diagram
        reliability_fig = vis.plot_reliability_diagram(
            y_true, y_pred_proba_uncalibrated, y_pred_proba_calibrated,
            title=f"{model_name} - Reliability Diagram",
            save_path=str(output_path / f"{model_name.replace(' ', '_')}_reliability_diagram.png")
        )
        plt.close(reliability_fig)
        
        # Risk distribution
        risk_fig = vis.plot_risk_distribution(
            y_pred_proba_calibrated, risk_mapper,
            title=f"{model_name} - Risk Distribution",
            save_path=str(output_path / f"{model_name.replace(' ', '_')}_risk_distribution.png")
        )
        plt.close(risk_fig)
        
        # Group-specific calibration (if provided)
        if groups is not None and group_name is not None:
            group_fig = vis.plot_group_calibration(
                y_true, y_pred_proba_calibrated, groups, group_name,
                save_path=str(output_path / f"{model_name.replace(' ', '_')}_group_calibration.png")
            )
            plt.close(group_fig)
        
        # Save validation table
        validation_table.to_csv(
            output_path / f"{model_name.replace(' ', '_')}_risk_validation.csv",
            index=False
        )
        
        # Save metrics as JSON
        with open(output_path / f"{model_name.replace(' ', '_')}_metrics.json", 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_json = json.loads(
                json.dumps(metrics, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            )
            json.dump(metrics_json, f, indent=2)
        
        # Generate text report
        report_text = CalibrationReport._generate_text_report(
            model_name, calibration_method, metrics, validation_table, success_criteria
        )
        report['text_report'] = report_text
        
        with open(output_path / f"{model_name.replace(' ', '_')}_report.txt", 'w') as f:
            f.write(report_text)
        
        return report
    
    @staticmethod
    def _generate_text_report(model_name: str, calibration_method: str,
                             metrics: Dict, validation_table: pd.DataFrame,
                             success_criteria: Dict) -> str:
        """Generate formatted text report."""
        
        report_lines = [
            "="*80,
            f"CALIBRATION REPORT: {model_name}",
            "="*80,
            "",
            f"Calibration Method: {calibration_method.upper()}",
            f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "="*80,
            "CALIBRATION METRICS",
            "="*80,
            "",
            "UNCALIBRATED MODEL:",
            f"  Brier Score:       {metrics['uncalibrated']['brier_score']:.4f}",
            f"  Log Loss:          {metrics['uncalibrated']['log_loss']:.4f}",
            f"  ROC AUC:           {metrics['uncalibrated']['roc_auc']:.4f}",
            f"  ECE:               {metrics['uncalibrated']['ece']:.4f}",
            f"  Hosmer-Lemeshow:   χ²={metrics['uncalibrated']['hosmer_lemeshow']['chi2_statistic']:.2f}, "
            f"p={metrics['uncalibrated']['hosmer_lemeshow']['p_value']:.4f} "
            f"({metrics['uncalibrated']['hosmer_lemeshow']['interpretation']})",
            "",
            "CALIBRATED MODEL:",
            f"  Brier Score:       {metrics['calibrated']['brier_score']:.4f}",
            f"  Log Loss:          {metrics['calibrated']['log_loss']:.4f}",
            f"  ROC AUC:           {metrics['calibrated']['roc_auc']:.4f}",
            f"  ECE:               {metrics['calibrated']['ece']:.4f}",
            f"  Hosmer-Lemeshow:   χ²={metrics['calibrated']['hosmer_lemeshow']['chi2_statistic']:.2f}, "
            f"p={metrics['calibrated']['hosmer_lemeshow']['p_value']:.4f} "
            f"({metrics['calibrated']['hosmer_lemeshow']['interpretation']})",
            "",
            "IMPROVEMENT:",
            f"  Brier Score:       {metrics['improvement']['brier_score_delta']:+.4f}",
            f"  ECE:               {metrics['improvement']['ece_delta']:+.4f}",
            f"  Log Loss:          {metrics['improvement']['log_loss_delta']:+.4f}",
            "",
            "="*80,
            "SUCCESS CRITERIA",
            "="*80,
            "",
            f"  Brier Score < 0.15:        {'✓ PASS' if success_criteria['brier_score_target'] else '✗ FAIL'} "
            f"(Actual: {metrics['calibrated']['brier_score']:.4f})",
            f"  ECE < 0.05 (±5%):          {'✓ PASS' if success_criteria['ece_target'] else '✗ FAIL'} "
            f"(Actual: {metrics['calibrated']['ece']:.4f})",
            f"  Hosmer-Lemeshow p > 0.05:  {'✓ PASS' if success_criteria['hosmer_lemeshow_target'] else '✗ FAIL'} "
            f"(Actual: {metrics['calibrated']['hosmer_lemeshow']['p_value']:.4f})",
            "",
            f"  OVERALL: {'✓ ALL CRITERIA MET' if all(success_criteria.values()) else '✗ SOME CRITERIA NOT MET'}",
            "",
            "="*80,
            "RISK SCORE VALIDATION",
            "="*80,
            "",
            validation_table.to_string(index=False),
            "",
            "="*80,
            "CLINICAL DECISION SUPPORT",
            "="*80,
            "",
            "Risk Categories:",
            "  LOW (0-5%):       Standard discharge planning",
            "  MEDIUM (5-15%):   Enhanced patient education + 1-week follow-up call",
            "  HIGH (15%+):      Intensive case management + home health visit",
            "",
            "="*80,
        ]
        
        return "\n".join(report_lines)


def calibrate_model_pipeline(model_predictions: Dict[str, np.ndarray],
                             y_true: np.ndarray,
                             model_name: str,
                             calibration_method: str = 'platt',
                             output_dir: str = './calibration_outputs',
                             groups: Optional[np.ndarray] = None,
                             group_name: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
    """
    Complete end-to-end calibration pipeline for a single model.
    
    This is the main function to use for calibrating models.
    
    Parameters:
    -----------
    model_predictions : dict
        Dictionary with keys 'train' and 'test' containing uncalibrated probabilities
    y_true : np.ndarray
        True labels for test set
    model_name : str
        Name of the model
    calibration_method : str
        Calibration method ('platt', 'isotonic', 'group-specific')
    output_dir : str
        Directory for outputs
    groups : np.ndarray, optional
        Group labels for fairness analysis
    group_name : str, optional
        Name of grouping variable
        
    Returns:
    --------
    calibrated_proba : np.ndarray
        Calibrated probabilities for test set
    report : dict
        Complete calibration report
    """
    print(f"\n{'='*80}")
    print(f"CALIBRATING MODEL: {model_name}")
    print(f"{'='*80}\n")
    
    # Initialize calibrator
    if calibration_method == 'group-specific' and groups is not None:
        calibrator = GroupSpecificCalibrator(method='platt')
        # Assuming groups are provided for train set
        calibrator.fit(
            model_predictions['train']['y_true'],
            model_predictions['train']['y_pred_proba'],
            model_predictions['train']['groups']
        )
        calibrated_proba = calibrator.predict_proba(
            model_predictions['test']['y_pred_proba'],
            groups
        )
    else:
        calibrator = ModelCalibrator(method=calibration_method)
        calibrator.fit(
            model_predictions['train']['y_true'],
            model_predictions['train']['y_pred_proba']
        )
        calibrated_proba = calibrator.predict_proba(
            model_predictions['test']['y_pred_proba']
        )
    
    # Save calibrator
    calibrator_path = Path(output_dir) / f"{model_name.replace(' ', '_')}_calibrator.pkl"
    calibrator.save(str(calibrator_path))
    print(f"✓ Calibrator saved to: {calibrator_path}")
    
    # Generate report
    report = CalibrationReport.generate_report(
        y_true=y_true,
        y_pred_proba_uncalibrated=model_predictions['test']['y_pred_proba'],
        y_pred_proba_calibrated=calibrated_proba,
        model_name=model_name,
        calibration_method=calibration_method,
        output_dir=output_dir,
        groups=groups,
        group_name=group_name
    )
    
    print(f"\n✓ Calibration complete!")
    print(f"✓ Report saved to: {output_dir}")
    print(f"\nKey Metrics:")
    print(f"  Brier Score: {report['metrics']['calibrated']['brier_score']:.4f} "
          f"({'✓ PASS' if report['success_criteria']['brier_score_target'] else '✗ FAIL'} < 0.15)")
    print(f"  ECE:         {report['metrics']['calibrated']['ece']:.4f} "
          f"({'✓ PASS' if report['success_criteria']['ece_target'] else '✗ FAIL'} < 0.05)")
    print(f"  H-L Test:    p={report['metrics']['calibrated']['hosmer_lemeshow']['p_value']:.4f} "
          f"({'✓ PASS' if report['success_criteria']['hosmer_lemeshow_target'] else '✗ FAIL'} > 0.05)")
    
    return calibrated_proba, report


# ============================================================================
# HUGGINGFACE HUB UPLOAD
# ============================================================================

def generate_calibration_model_card(report: Dict[str, Any], model_name: str = "Calibrated Model") -> str:
    """
    Generate HuggingFace model card for calibrated model.
    
    Parameters:
    -----------
    report : dict
        Calibration report dictionary from CalibrationReport.generate_report()
    model_name : str
        Name of the model (e.g., 'Gradient Boosting', 'Random Forest')
        
    Returns:
    --------
    str : Markdown-formatted model card
    """
    # Extract key information
    calibration_method = report.get('calibration_method', 'Unknown')
    timestamp = report.get('timestamp', 'N/A')
    
    # Metrics
    metrics = report.get('metrics', {})
    uncal_metrics = metrics.get('uncalibrated', {})
    cal_metrics = metrics.get('calibrated', {})
    improvements = metrics.get('improvement', {})
    
    # Success criteria
    success = report.get('success_criteria', {})
    meets_all = report.get('meets_all_criteria', False)
    
    # Risk validation
    risk_validation = report.get('risk_validation', [])
    
    # Generate model card
    card = f"""---
tags:
- healthcare
- clinical-ml
- diabetes
- readmission-prediction
- model-calibration
- {calibration_method.lower()}
library_name: scikit-learn
pipeline_tag: tabular-classification
---

# {model_name} - Calibrated for Hospital Readmission Prediction

## Model Description

This is a **calibrated** {model_name} model for predicting 30-day hospital readmission risk in diabetic patients. The model has been calibrated using **{calibration_method.upper()}** to ensure that predicted probabilities accurately reflect true readmission risk.

**Calibration Method:** {calibration_method.upper()}  
**Calibration Date:** {timestamp}  
**Base Model:** {model_name}

## Why Calibration Matters

Model calibration ensures that predicted probabilities are reliable for clinical decision-making:
- A predicted 15% risk should mean ~15 out of 100 similar patients are readmitted
- Enables accurate risk stratification and resource allocation
- Critical for patient safety and clinical trust
- Required for regulatory compliance in healthcare AI

## Calibration Performance

### Success Criteria

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| **Brier Score** | < 0.15 | {cal_metrics.get('brier_score', 0):.4f} | {'✅ PASS' if success.get('brier_score_target') else '❌ FAIL'} |
| **ECE (±5% accuracy)** | < 0.05 | {cal_metrics.get('ece', 0):.4f} | {'✅ PASS' if success.get('ece_target') else '❌ FAIL'} |
| **Hosmer-Lemeshow Test** | p > 0.05 | {cal_metrics.get('hosmer_lemeshow', {}).get('p_value', 0):.4f} | {'✅ PASS' if success.get('hosmer_lemeshow_target') else '❌ FAIL'} |

**Overall Result:** {'✅ ALL CRITERIA MET' if meets_all else '⚠️ SOME CRITERIA NOT MET'}

### Before vs After Calibration

| Metric | Uncalibrated | Calibrated | Improvement |
|--------|--------------|------------|-------------|
| **Brier Score** | {uncal_metrics.get('brier_score', 0):.4f} | {cal_metrics.get('brier_score', 0):.4f} | {improvements.get('brier_score_delta', 0):+.4f} |
| **Log Loss** | {uncal_metrics.get('log_loss', 0):.4f} | {cal_metrics.get('log_loss', 0):.4f} | {improvements.get('log_loss_delta', 0):+.4f} |
| **ECE** | {uncal_metrics.get('ece', 0):.4f} | {cal_metrics.get('ece', 0):.4f} | {improvements.get('ece_delta', 0):+.4f} |
| **ROC-AUC** | {uncal_metrics.get('roc_auc', 0):.4f} | {cal_metrics.get('roc_auc', 0):.4f} | Unchanged* |

*Note: Calibration improves probability estimates without changing discrimination (ranking) ability.

### Hosmer-Lemeshow Goodness-of-Fit Test

- **Uncalibrated:** χ²={uncal_metrics.get('hosmer_lemeshow', {}).get('chi2_statistic', 0):.2f}, p={uncal_metrics.get('hosmer_lemeshow', {}).get('p_value', 0):.4f} ({uncal_metrics.get('hosmer_lemeshow', {}).get('interpretation', 'N/A')})
- **Calibrated:** χ²={cal_metrics.get('hosmer_lemeshow', {}).get('chi2_statistic', 0):.2f}, p={cal_metrics.get('hosmer_lemeshow', {}).get('p_value', 0):.4f} ({cal_metrics.get('hosmer_lemeshow', {}).get('interpretation', 'N/A')})

## Clinical Risk Categories

Calibrated probabilities are mapped to actionable risk categories:

| Risk Level | Probability Range | Recommended Action |
|-----------|------------------|-------------------|
| **Low** | 0-5% | Standard discharge planning |
| **Medium** | 5-15% | Enhanced patient education + 1-week follow-up call |
| **High** | 15%+ | Intensive case management + home health visit |

### Risk Category Validation

The table below shows how well predicted risk categories align with actual readmission rates:

"""
    
    # Add risk validation table if available
    if risk_validation:
        card += "\n| Risk Category | Probability Range | N Patients | Actual Rate | Predicted Rate | Clinical Action |\n"
        card += "|--------------|------------------|-----------|-------------|----------------|----------------|\n"
        for entry in risk_validation:
            card += f"| {entry.get('Risk Category', 'N/A')} | "
            card += f"{entry.get('Probability Range', 'N/A')} | "
            card += f"{entry.get('N Patients', 'N/A')} | "
            card += f"{entry.get('Actual Rate', 'N/A')} | "
            card += f"{entry.get('Predicted Rate', 'N/A')} | "
            card += f"{entry.get('Clinical Action', 'N/A')} |\n"
    
    card += f"""

## Visualizations

### Reliability Diagram (Calibration Curve)
![Reliability Diagram](./reliability_diagram.png)

Shows how well predicted probabilities match observed frequencies. The closer to the diagonal, the better calibrated.

### Reliability Comparison (Before vs After)
![Reliability Comparison](./reliability_diagram_comparison.png)

Compares uncalibrated vs calibrated predictions.

### Risk Distribution
![Risk Distribution](./risk_distribution.png)

Distribution of patients across risk categories.

### Detailed Risk Distribution
![Detailed Risk Distribution](./risk_distribution_detailed.png)

Enhanced visualization with probability thresholds.

## Usage

### Loading the Calibrated Model

```python
import joblib
import pandas as pd
import sys

# Add utilities to path
sys.path.append('./phase-3-model-calibration')
from utilities import ModelCalibrator, RiskScoreMapper

# Load original model
model = joblib.load('gradient_boosting_model_original.joblib')

# Load calibrator
calibrator = ModelCalibrator.load('Gradient_Boosting_(LightGBM)_calibrator.pkl')

# Load your preprocessed features (MUST use same preprocessing as training!)
X_new = pd.read_csv('your_preprocessed_features.csv')

# Step 1: Get uncalibrated predictions from original model
uncalibrated_proba = model.predict_proba(X_new)[:, 1]

# Step 2: Apply calibration
calibrated_proba = calibrator.predict_proba(uncalibrated_proba)

# Step 3: Map to clinical risk categories
risk_mapper = RiskScoreMapper()
risk_categories = risk_mapper.get_risk_labels(calibrated_proba)
clinical_actions = risk_mapper.get_clinical_actions(calibrated_proba)

# Create results DataFrame
results = pd.DataFrame({{
    'patient_id': X_new.index,
    'uncalibrated_probability': uncalibrated_proba,
    'calibrated_probability': calibrated_proba,
    'risk_category': risk_categories,
    'recommended_action': clinical_actions
}})

# Display results
print(results.head(10))

# Example output:
#    patient_id  uncalibrated_probability  calibrated_probability risk_category  recommended_action
# 0           0                    0.0834                  0.0234           Low  Standard discharge
# 1           1                    0.2341                  0.1876          High  Intensive case management, home visit
# 2           2                    0.1123                  0.0891        Medium  Enhanced education, 1-week follow-up
```

### Quick Prediction Pipeline

```python
def predict_readmission_risk(patient_features):
    \"\"\"
    Complete pipeline for readmission risk prediction.
    
    Args:
        patient_features: DataFrame with preprocessed patient features
        
    Returns:
        DataFrame with predictions and recommended actions
    \"\"\"
    # Load models
    model = joblib.load('gradient_boosting_model_original.joblib')
    calibrator = ModelCalibrator.load('Gradient_Boosting_(LightGBM)_calibrator.pkl')
    risk_mapper = RiskScoreMapper()
    
    # Generate calibrated predictions
    uncalibrated = model.predict_proba(patient_features)[:, 1]
    calibrated = calibrator.predict_proba(uncalibrated)
    
    # Map to risk categories
    results = pd.DataFrame({{
        'readmission_probability': calibrated,
        'risk_category': risk_mapper.get_risk_labels(calibrated),
        'clinical_action': risk_mapper.get_clinical_actions(calibrated)
    }})
    
    return results

# Use the pipeline
predictions = predict_readmission_risk(X_new)
print(predictions)
```

## Important Notes

### 1. Preprocessing Requirements
Input features **MUST** be preprocessed using the exact same pipeline as training:
- Same missing value imputation
- Same feature engineering
- Same scaling/encoding
- Same feature set

See `phase-1-data-explore-preprocessing/simple_preprocessing.py` for the preprocessing pipeline.

### 2. Calibration Preserves Discrimination
- Calibration improves probability estimates
- Does NOT change model's ranking ability (ROC-AUC stays the same)
- Patients ranked as higher risk remain higher risk

### 3. When to Recalibrate
Recalibrate the model when:
- Patient population characteristics change
- Healthcare practices evolve
- Model performance degrades
- Annually as a best practice

### 4. Clinical Validation Required
Before deployment:
- Validate risk thresholds with clinical experts
- Test on local patient population
- Ensure alignment with clinical workflows
- Obtain necessary regulatory approvals

## Files Included

- `gradient_boosting_model_original.joblib` - Original trained model
- `Gradient_Boosting_(LightGBM)_calibrator.pkl` - Calibration transformer ({calibration_method})
- `Gradient_Boosting_(LightGBM)_report.txt` - Detailed calibration report
- `Gradient_Boosting_(LightGBM)_metrics.json` - Metrics in JSON format
- `calibration_comparison_metrics.json` - Before/after comparison
- `risk_validation_detailed.csv` - Risk category validation table
- `reliability_diagram*.png` - Calibration visualizations
- `risk_distribution*.png` - Risk distribution plots
- `DEPLOYMENT_INSTRUCTIONS.md` - Deployment guide

## Limitations and Ethical Considerations

### Limitations
1. **Domain-Specific:** Trained for diabetic patient readmissions only
2. **Temporal Drift:** Data from 1999-2008 may not reflect current practices
3. **Geographic Bias:** US hospital data may not generalize internationally
4. **Population Shift:** Recalibration needed if patient demographics change

### Ethical Considerations
This model should:
- ✅ **Assist** clinical decision-making, not replace it
- ✅ Be **validated** on your local patient population
- ✅ Be **monitored** for fairness across demographic groups
- ✅ Be **recalibrated** regularly with recent data
- ❌ **NOT** be the sole basis for treatment decisions
- ❌ **NOT** be deployed without clinical expert validation

### Fairness
- Evaluate calibration quality separately for different demographic groups
- Monitor for disparate impact across protected attributes
- Consider group-specific calibration if needed
- Document fairness metrics for regulatory compliance

## Citation

```bibtex
@misc{{hospital-readmission-calibrated,
  title={{Calibrated Model for Hospital Readmission Prediction}},
  author={{Your Name}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/your-username/your-repo}}}}
}}
```

## Dataset Citation

```bibtex
@article{{strack2014impact,
  title={{Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records}},
  author={{Strack, Beata and DeShazo, Jonathan P and Gennings, Chris and Olmo, Juan L and Ventura, Sebastian and Cios, Krzysztof J and Clore, John N}},
  journal={{BioMed Research International}},
  volume={{2014}},
  year={{2014}},
  publisher={{Hindawi}}
}}
```

## License

This calibrated model is released under the MIT License. The underlying dataset and original model have their own license terms.

## Contact

For questions or issues, please open an issue in the repository.

---

**Disclaimer:** This model is for research and educational purposes. Always consult healthcare professionals for medical decisions. Regular monitoring and recalibration are essential for safe deployment.

**Last Updated:** {timestamp}
"""
    
    return card


def upload_calibrated_model_to_hf(
    report: Dict[str, Any],
    output_dir: str,
    model_name: str = "hospital-readmission-calibrated",
    base_model_name: str = "Gradient Boosting",
    hf_repo_name: Optional[str] = None,
    hf_token: Optional[str] = None
) -> bool:
    """
    Upload calibrated model and calibration artifacts to HuggingFace Hub.
    
    This function uploads all calibration outputs including:
    - Calibrated model and calibrator
    - Calibration reports and metrics
    - Visualizations (reliability diagrams, risk distributions)
    - Risk validation tables
    - Deployment instructions
    - Auto-generated model card
    
    Parameters:
    -----------
    report : dict
        Calibration report from CalibrationReport.generate_report()
    output_dir : str
        Directory containing calibration outputs
    model_name : str
        Repository name suffix (default: "hospital-readmission-calibrated")
    base_model_name : str
        Name of base model (e.g., "Gradient Boosting", "Random Forest")
    hf_repo_name : str, optional
        Full HuggingFace repo ID (auto-generated if None)
    hf_token : str, optional
        HuggingFace API token (loads from .env if None)
        
    Returns:
    --------
    bool : True if upload successful, False otherwise
    
    Example:
    --------
    >>> # After calibration
    >>> calibrated_proba, report = calibrate_model_pipeline(...)
    >>> upload_calibrated_model_to_hf(
    ...     report=report,
    ...     output_dir='./calibration_outputs/gradient_boosting',
    ...     model_name='hospital-readmission-lgbm-calibrated',
    ...     base_model_name='Gradient Boosting (LightGBM)'
    ... )
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("⚠️  huggingface_hub is required to upload to HuggingFace Hub.")
        print("   Install with: pip install huggingface_hub")
        return False
    
    # Try to load from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, will use environment variables
    
    # Get token from .env or environment
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        print("⚠️  HF_TOKEN not found in .env file or environment variables.")
        print("   Skipping upload to HuggingFace Hub.")
        print("   To enable automatic upload:")
        print("   1. Create a .env file in your project root")
        print("   2. Add: HF_TOKEN=your_token_here")
        print("   3. Add: HF_USERNAME=your_username")
        print("   4. Get token from: https://huggingface.co/settings/tokens")
        return False
    
    # Auto-generate repo name if not provided
    if hf_repo_name is None:
        username = os.getenv('HF_USERNAME')
        if not username:
            print("⚠️  HF_USERNAME not found in .env file.")
            print("   Add HF_USERNAME=your_username to .env file")
            return False
        
        # Generate repo name: username/model-name
        hf_repo_name = f"{username}/{model_name}"
    
    print(f"\n{'='*80}")
    print("📤 Uploading Calibrated Model to HuggingFace Hub")
    print(f"{'='*80}")
    print(f"Repository: {hf_repo_name}")
    print(f"Model: {base_model_name} (Calibrated)")
    print(f"Method: {report.get('calibration_method', 'Unknown').upper()}")
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create repo if it doesn't exist
        print(f"\n⏳ Creating/accessing repository...")
        repo_url = create_repo(
            repo_id=hf_repo_name,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ Repository ready: {hf_repo_name}")
        
        output_path = Path(output_dir)
        
        # Generate and save model card
        print(f"\n📝 Generating model card...")
        readme_content = generate_calibration_model_card(report, base_model_name)
        readme_path = output_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"   ✅ Model card saved: {readme_path}")
        
        # List files to upload
        print(f"\n📂 Files to upload:")
        files_to_upload = []
        for file_path in output_path.glob('*'):
            if file_path.is_file():
                files_to_upload.append(file_path.name)
                print(f"   - {file_path.name}")
        
        if not files_to_upload:
            print("⚠️  No files found in output directory!")
            return False
        
        # Upload all files from output directory
        print(f"\n⏳ Uploading {len(files_to_upload)} files to HuggingFace Hub...")
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=hf_repo_name,
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload calibrated {base_model_name} with {report.get('calibration_method', 'unknown')} calibration"
        )
        
        print(f"\n{'='*80}")
        print("✅ Calibrated Model Uploaded Successfully!")
        print(f"{'='*80}")
        print(f"🌐 View at: https://huggingface.co/{hf_repo_name}")
        print(f"{'='*80}")
        print(f"\n📊 Summary:")
        print(f"   Model: {base_model_name}")
        print(f"   Calibration: {report.get('calibration_method', 'Unknown').upper()}")
        print(f"   Brier Score: {report['metrics']['calibrated']['brier_score']:.4f}")
        print(f"   ECE: {report['metrics']['calibrated']['ece']:.4f}")
        print(f"   Success: {'✅ ALL CRITERIA MET' if report.get('meets_all_criteria') else '⚠️ SOME CRITERIA NOT MET'}")
        print(f"   Files: {len(files_to_upload)} uploaded")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error uploading to HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return False


# Example usage template (commented out)
"""
# Example: Calibrate a Logistic Regression model

# Assume you have model predictions
model_predictions = {
    'train': {
        'y_true': y_train,
        'y_pred_proba': lr_model.predict_proba(X_train)[:, 1]
    },
    'test': {
        'y_pred_proba': lr_model.predict_proba(X_test)[:, 1]
    }
}

# Run calibration pipeline
calibrated_proba, report = calibrate_model_pipeline(
    model_predictions=model_predictions,
    y_true=y_test,
    model_name='Logistic Regression',
    calibration_method='platt',
    output_dir='./outputs/calibration/logistic_regression'
)

# Upload to HuggingFace Hub
upload_calibrated_model_to_hf(
    report=report,
    output_dir='./outputs/calibration/logistic_regression',
    model_name='hospital-readmission-lr-calibrated',
    base_model_name='Logistic Regression'
)

# Use calibrated probabilities for deployment
risk_mapper = RiskScoreMapper()
risk_categories = risk_mapper.map_to_risk_category(calibrated_proba)
clinical_actions = risk_mapper.get_clinical_actions(calibrated_proba)
"""
