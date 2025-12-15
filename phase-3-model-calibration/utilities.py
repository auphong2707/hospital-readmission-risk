"""
Phase 3: Model Calibration Utilities

Provides Platt Scaling calibration for hospital readmission prediction models.

Calibration Method:
- Platt Scaling: Logistic regression transformation of predicted probabilities

Validation Methods:
- Reliability Diagrams
- Brier Score
- Expected Calibration Error (ECE)
- Hosmer-Lemeshow Test
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
    repo_id: str = "auphong2707/hospital-readmission-phase2-lgbm",
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
        Default: "auphong2707/hospital-readmission-phase2-lgbm"
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


def load_phase1_splits(cache_dir: str = "./data/downloaded",
                       repo_id: str = "auphong2707/hospital-readmission-risk-data"):
    """Load Phase 1 splits from HuggingFace.
    
    This ensures all phases (2-5) use the exact same preprocessed data from Phase 1.
    Phase 1 created these splits with:
    - Train: 73,526 samples (72.25%)
    - Validation: 12,975 samples (12.75%)
    - Test: 15,265 samples (15%)
    - Random seed: 42
    - Stratification: Yes (on target variable)
    
    Args:
        cache_dir: Directory to cache downloaded files
        repo_id: HuggingFace repository ID
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        
    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If splits cannot be loaded from HuggingFace
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
        # Phase 1 uses 'target' as the column name
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


# ============================================================================
# CALIBRATION CLASSES
# ============================================================================

class ModelCalibrator:
    """
    Platt Scaling calibration for hospital readmission models.
    
    Uses logistic regression to transform uncalibrated probabilities.
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
    
    @staticmethod
    def plot_brier_score_comparison(
        y_true: np.ndarray,
        probabilities_dict: Dict[str, np.ndarray],
        title: str = "Brier Score Comparison Across Methods",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Bar chart comparing Brier scores across different calibration methods.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        probabilities_dict : Dict[str, np.ndarray]
            Dictionary mapping method names to predicted probabilities
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        # Calculate Brier scores
        method_names = []
        brier_scores = []
        
        for method_name, y_pred_proba in probabilities_dict.items():
            brier = brier_score_loss(y_true, y_pred_proba)
            method_names.append(method_name)
            brier_scores.append(brier)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color bars based on performance
        colors = ['#E63946' if score > 0.15 else '#06A77D' if score < 0.10 else '#F18F01' 
                 for score in brier_scores]
        
        bars = ax.bar(method_names, brier_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, score in zip(bars, brier_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add reference line for target threshold
        ax.axhline(y=0.15, color='red', linestyle='--', linewidth=2, 
                  label='Target Threshold (< 0.15)', alpha=0.7)
        ax.axhline(y=0.10, color='green', linestyle='--', linewidth=2, 
                  label='Excellent (< 0.10)', alpha=0.7)
        
        ax.set_ylabel('Brier Score (Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Calibration Method', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, max(brier_scores) * 1.2])
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Brier score comparison saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_probability_distribution_changes(
        y_pred_proba_before: np.ndarray,
        y_pred_proba_after: np.ndarray,
        risk_thresholds: Tuple[float, float] = (0.05, 0.15),
        title: str = "Probability Distribution: Before vs After Calibration",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Histogram showing how predicted probabilities changed after calibration.
        
        Parameters:
        -----------
        y_pred_proba_before : np.ndarray
            Uncalibrated probabilities
        y_pred_proba_after : np.ndarray
            Calibrated probabilities
        risk_thresholds : Tuple[float, float]
            (low_threshold, high_threshold) for risk categories
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        bins = np.linspace(0, 1, 51)
        
        # Before calibration
        ax1.hist(y_pred_proba_before, bins=bins, color='#E63946', alpha=0.7, 
                edgecolor='black', linewidth=0.5, label='Uncalibrated')
        ax1.axvline(risk_thresholds[0], color='orange', linestyle='--', linewidth=2, 
                   label=f'Low/Medium ({risk_thresholds[0]:.0%})')
        ax1.axvline(risk_thresholds[1], color='red', linestyle='--', linewidth=2, 
                   label=f'Medium/High ({risk_thresholds[1]:.0%})')
        ax1.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Before Calibration', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # After calibration
        ax2.hist(y_pred_proba_after, bins=bins, color='#06A77D', alpha=0.7, 
                edgecolor='black', linewidth=0.5, label='Calibrated')
        ax2.axvline(risk_thresholds[0], color='orange', linestyle='--', linewidth=2, 
                   label=f'Low/Medium ({risk_thresholds[0]:.0%})')
        ax2.axvline(risk_thresholds[1], color='red', linestyle='--', linewidth=2, 
                   label=f'Medium/High ({risk_thresholds[1]:.0%})')
        ax2.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('After Calibration', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # Overlay comparison
        ax3.hist(y_pred_proba_before, bins=bins, color='#E63946', alpha=0.5, 
                edgecolor='black', linewidth=0.5, label='Before (Uncalibrated)')
        ax3.hist(y_pred_proba_after, bins=bins, color='#06A77D', alpha=0.5, 
                edgecolor='black', linewidth=0.5, label='After (Calibrated)')
        ax3.axvline(risk_thresholds[0], color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax3.axvline(risk_thresholds[1], color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Probability distribution changes saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_calibration_fairness_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        groups: np.ndarray,
        group_name: str = "Demographic Group",
        n_bins: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Bar charts showing calibration metrics (ECE, Brier) by demographic group.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba : np.ndarray
            Calibrated probabilities
        groups : np.ndarray
            Group labels
        group_name : str
            Name of the grouping variable
        n_bins : int
            Number of bins for ECE calculation
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        unique_groups = np.unique(groups)
        
        # Calculate metrics per group
        group_labels = []
        brier_scores = []
        ece_scores = []
        sample_sizes = []
        
        for group in unique_groups:
            mask = groups == group
            if np.sum(mask) > n_bins:  # Only process if enough samples
                group_labels.append(str(group))
                brier_scores.append(brier_score_loss(y_true[mask], y_pred_proba[mask]))
                ece_scores.append(CalibrationMetrics.expected_calibration_error(
                    y_true[mask], y_pred_proba[mask], n_bins))
                sample_sizes.append(np.sum(mask))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Brier score by group
        colors_brier = ['#06A77D' if score < 0.15 else '#F18F01' if score < 0.20 else '#E63946' 
                       for score in brier_scores]
        bars1 = ax1.bar(range(len(group_labels)), brier_scores, color=colors_brier, 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels and sample sizes
        for i, (bar, score, n) in enumerate(zip(bars1, brier_scores, sample_sizes)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}\n(n={n})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.axhline(y=0.15, color='red', linestyle='--', linewidth=2, 
                   label='Target < 0.15', alpha=0.7)
        ax1.set_ylabel('Brier Score', fontsize=11, fontweight='bold')
        ax1.set_xlabel(group_name, fontsize=11, fontweight='bold')
        ax1.set_title(f'Brier Score by {group_name}', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(group_labels)))
        ax1.set_xticklabels(group_labels, rotation=45, ha='right')
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # ECE by group
        colors_ece = ['#06A77D' if score < 0.05 else '#F18F01' if score < 0.10 else '#E63946' 
                     for score in ece_scores]
        bars2 = ax2.bar(range(len(group_labels)), ece_scores, color=colors_ece, 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, score, n) in enumerate(zip(bars2, ece_scores, sample_sizes)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}\n(n={n})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                   label='Target < 0.05', alpha=0.7)
        ax2.set_ylabel('Expected Calibration Error (ECE)', fontsize=11, fontweight='bold')
        ax2.set_xlabel(group_name, fontsize=11, fontweight='bold')
        ax2.set_title(f'ECE by {group_name}', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(group_labels)))
        ax2.set_xticklabels(group_labels, rotation=45, ha='right')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Calibration Fairness: {group_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Calibration fairness metrics saved: {save_path}")
        
        return fig
    
    @staticmethod
    def generate_all_calibration_visualizations(
        y_true: np.ndarray,
        y_pred_proba_uncalibrated: np.ndarray,
        y_pred_proba_calibrated: np.ndarray,
        probabilities_dict: Optional[Dict[str, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
        group_name: str = "Demographic Group",
        output_dir: str = "./calibration_outputs",
        n_bins: int = 10
    ) -> Dict[str, str]:
        """
        Generate all calibration visualizations and save to output directory.
        
        This is a convenience function that generates all recommended Phase 3 visualizations
        in a single call, saving them to the specified output directory.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba_uncalibrated : np.ndarray
            Uncalibrated predicted probabilities
        y_pred_proba_calibrated : np.ndarray
            Calibrated predicted probabilities
        probabilities_dict : Dict[str, np.ndarray], optional
            Dictionary with multiple calibration methods for comparison
            If None, creates one with uncalibrated and calibrated
        groups : np.ndarray, optional
            Group labels for fairness analysis
        group_name : str
            Name of the grouping variable for fairness plots
        output_dir : str
            Directory to save all visualizations
        n_bins : int
            Number of bins for calibration curves
            
        Returns:
        --------
        Dict[str, str] : Dictionary mapping visualization names to file paths
        
        Example:
        --------
        >>> # After calibration
        >>> vis_paths = CalibrationVisualizer.generate_all_calibration_visualizations(
        ...     y_true=y_test,
        ...     y_pred_proba_uncalibrated=uncalibrated_proba,
        ...     y_pred_proba_calibrated=calibrated_proba,
        ...     probabilities_dict={
        ...         'Uncalibrated': uncalibrated_proba,
        ...         'Platt Scaling': platt_proba
        ...     },
        ...     groups=demographics['race'].values,
        ...     group_name='Race',
        ...     output_dir='./calibration_outputs/gradient_boosting'
        ... )
        >>> print(f"Generated {len(vis_paths)} visualizations")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualization_paths = {}
        
        print("\n" + "="*80)
        print("📊 GENERATING COMPREHENSIVE CALIBRATION VISUALIZATIONS")
        print("="*80)
        
        # Create default probabilities_dict if not provided
        if probabilities_dict is None:
            probabilities_dict = {
                'Uncalibrated': y_pred_proba_uncalibrated,
                'Calibrated': y_pred_proba_calibrated
            }
        
        # 1. Before/After Reliability Diagram
        print("\n1️⃣  Before/After Reliability Diagram...")
        save_path = str(output_path / "01_reliability_diagram_before_after.png")
        CalibrationVisualizer.plot_reliability_diagram(
            y_true=y_true,
            y_pred_proba_uncalibrated=y_pred_proba_uncalibrated,
            y_pred_proba_calibrated=y_pred_proba_calibrated,
            n_bins=n_bins,
            title="Reliability Diagram: Before vs After Calibration",
            save_path=save_path
        )
        visualization_paths['reliability_diagram'] = save_path
        
        # 2. Calibration Improvement Metrics (Brier, ECE, Log Loss)
        print("2️⃣  Calibration Improvement Metrics...")
        save_path = str(output_path / "02_calibration_improvement_metrics.png")
        CalibrationVisualizer.plot_brier_score_comparison(
            y_true=y_true,
            probabilities_dict=probabilities_dict,
            title="Calibration Improvement: Brier Score & ECE Comparison",
            save_path=save_path
        )
        visualization_paths['calibration_improvement'] = save_path
        
        # 3. Probability Distribution Changes
        print("3️⃣  Probability Distribution Changes...")
        save_path = str(output_path / "03_probability_distribution_changes.png")
        risk_thresholds = (0.05, 0.15)  # Default thresholds
        CalibrationVisualizer.plot_probability_distribution_changes(
            y_pred_proba_before=y_pred_proba_uncalibrated,
            y_pred_proba_after=y_pred_proba_calibrated,
            risk_thresholds=risk_thresholds,
            title="Probability Distribution: Before vs After Calibration",
            save_path=save_path
        )
        visualization_paths['probability_distribution'] = save_path
        
        # 4. Group-Specific Calibration (if groups provided)
        if groups is not None:
            print(f"4️⃣  Group-Specific Calibration by {group_name}...")
            save_path = str(output_path / f"05_group_calibration_{group_name.lower().replace(' ', '_')}.png")
            CalibrationVisualizer.plot_group_calibration(
                y_true=y_true,
                y_pred_proba=y_pred_proba_calibrated,
                groups=groups,
                group_name=group_name,
                n_bins=n_bins,
                save_path=save_path
            )
            visualization_paths[f'group_calibration_{group_name.lower()}'] = save_path
            
            # 5. Calibration Fairness Metrics
            print(f"5️⃣  Calibration Fairness Metrics by {group_name}...")
            save_path = str(output_path / f"05_calibration_fairness_{group_name.lower().replace(' ', '_')}.png")
            CalibrationVisualizer.plot_calibration_fairness_metrics(
                y_true=y_true,
                y_pred_proba=y_pred_proba_calibrated,
                groups=groups,
                group_name=group_name,
                n_bins=n_bins,
                save_path=save_path
            )
            visualization_paths[f'fairness_metrics_{group_name.lower()}'] = save_path
        
        print("\n" + "="*80)
        print(f"✅ GENERATED {len(visualization_paths)} VISUALIZATIONS")
        print("="*80)
        print(f"📁 Output directory: {output_dir}")
        print("\nGenerated files:")
        for name, path in visualization_paths.items():
            print(f"   • {name}: {Path(path).name}")
        print("="*80 + "\n")
        
        return visualization_paths


def convert_to_serializable(obj):
    """
    Recursively convert numpy types and other non-serializable objects to Python native types.
    
    Handles nested dictionaries, lists, numpy arrays, and numpy scalars.
    Prevents circular reference errors by creating new objects instead of modifying in place.
    
    Parameters:
    -----------
    obj : any
        Object to convert
        
    Returns:
    --------
    Serializable Python object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # For any other type, try to convert to string as fallback
        try:
            return str(obj)
        except:
            return None


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
            Calibration method used ('platt')
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
        
        # Success criteria checks
        success_criteria = {
            'brier_score_target': metrics['calibrated']['brier_score'] < 0.15,
            'ece_target': metrics['calibrated']['ece'] < 0.05,
            'hosmer_lemeshow_target': metrics['calibrated']['hosmer_lemeshow']['is_well_calibrated']
        }
        report['success_criteria'] = success_criteria
        report['meets_all_criteria'] = all(success_criteria.values())
        
        # Generate visualizations using the static method
        visualization_paths = CalibrationVisualizer.generate_all_calibration_visualizations(
            y_true=y_true,
            y_pred_proba_uncalibrated=y_pred_proba_uncalibrated,
            y_pred_proba_calibrated=y_pred_proba_calibrated,
            groups=groups,
            group_name=group_name if group_name else "Demographic Group",
            output_dir=str(output_path),
            n_bins=10
        )
        report['visualization_paths'] = visualization_paths
        
        # Save metrics as JSON
        with open(output_path / f"{model_name.replace(' ', '_')}_metrics.json", 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_json = convert_to_serializable(metrics)
            json.dump(metrics_json, f, indent=2)
        
        # Generate text report
        report_text = CalibrationReport._generate_text_report(
            model_name, calibration_method, metrics, success_criteria
        )
        report['text_report'] = report_text
        
        with open(output_path / f"{model_name.replace(' ', '_')}_report.txt", 'w') as f:
            f.write(report_text)
        
        return report
    
    @staticmethod
    def _generate_text_report(model_name: str, calibration_method: str,
                             metrics: Dict, success_criteria: Dict) -> str:
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
            "NEXT STEPS",
            "="*80,
            "",
            "Phase 4: Threshold Optimization",
            "  - Determine optimal risk thresholds based on cost matrix",
            "  - Map calibrated probabilities to clinical actions",
            "  - Validate thresholds with domain experts",
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
    Complete end-to-end Platt Scaling calibration pipeline.
    
    Main function for calibrating models using Platt Scaling.
    
    Parameters:
    -----------
    model_predictions : dict
        Dictionary with keys 'train' and 'test' containing uncalibrated probabilities
    y_true : np.ndarray
        True labels for test set
    model_name : str
        Name of the model
    calibration_method : str
        Calibration method (default: 'platt')
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
    
    # Initialize Platt Scaling calibrator
    calibrator = ModelCalibrator(method='platt')
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
from utilities import ModelCalibrator

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

# Create results DataFrame
results = pd.DataFrame({{
    'patient_id': X_new.index,
    'uncalibrated_probability': uncalibrated_proba,
    'calibrated_probability': calibrated_proba
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
        DataFrame with calibrated probabilities
    \"\"\"
    # Load models
    model = joblib.load('gradient_boosting_model_original.joblib')
    calibrator = ModelCalibrator.load('Gradient_Boosting_(LightGBM)_calibrator.pkl')
    
    # Generate calibrated predictions
    uncalibrated = model.predict_proba(patient_features)[:, 1]
    calibrated = calibrator.predict_proba(uncalibrated)
    
    # Return calibrated probabilities
    results = pd.DataFrame({{
        'readmission_probability': calibrated
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
    ...     model_name='hospital-readmission-phase3-lgbm-calibrated',
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
    model_name='hospital-readmission-phase3-lr-calibrated',
    base_model_name='Logistic Regression'
)

# Use calibrated probabilities for deployment
# Risk thresholds will be determined in Phase 4 (Threshold Optimization)
"""
