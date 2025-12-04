"""
Phase 5: Fairness Evaluation & Deployment Readiness Utilities

Provides comprehensive fairness evaluation tools for hospital readmission prediction models:
- Data loading (test data, demographics, calibrated models, Phase 4 results)
- Fairness metrics (demographic parity, equalized odds, equal opportunity)
- Group-specific performance analysis
- Statistical significance testing
- Bias mitigation strategies
- Visualization tools
- Deployment packaging utilities

Key Components:
- FairnessMetrics: Calculate fairness metrics across demographic groups
- GroupPerformanceAnalyzer: Compute TPR, FPR, precision by group
- StatisticalTester: Chi-square, proportion tests for bias detection
- FairnessVisualizer: Generate fairness assessment plots
- BiasMitigator: Group-specific thresholds and calibration
- DeploymentPackager: Bundle model, thresholds, documentation
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
import joblib
from sklearn.linear_model import LogisticRegression
import pickle

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

warnings.filterwarnings('ignore')


# ============================================================================
# MODEL CALIBRATOR CLASS (for unpickling Phase 3 calibrator)
# ============================================================================

class ModelCalibrator:
    """
    Platt Scaling calibration for hospital readmission models.
    
    This class is required to unpickle the calibrator trained in Phase 3.
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


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_test_data_and_demographics(
    data_repo_id: str = "auphong2707/hospital-readmission-risk-data",
    cache_dir: str = "./data/downloaded",
    use_local: bool = False,
    local_test_path: str = "./data/processed/splits/test.csv",
    local_demographics_path: str = "./data/processed/splits/test_demographics.csv"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load test data and demographics from HuggingFace or local files.
    
    Args:
        data_repo_id: HuggingFace dataset repository ID
        cache_dir: Cache directory for downloads
        use_local: If True, load from local files instead of HuggingFace
        local_test_path: Path to local test.csv
        local_demographics_path: Path to local test_demographics.csv
        
    Returns:
        tuple: (X_test, y_test, demographics)
            - X_test: Test features (DataFrame)
            - y_test: Test labels (Series)
            - demographics: Demographics with race, gender, age (DataFrame)
            
    Raises:
        FileNotFoundError: If demographics file not found
        ImportError: If huggingface_hub not installed (when use_local=False)
    """
    print("\n" + "="*80)
    print("📥 Loading Test Data and Demographics")
    print("="*80)
    
    if use_local:
        print(f"Loading from local files...")
        
        # Load test data
        if not os.path.exists(local_test_path):
            raise FileNotFoundError(f"Test file not found: {local_test_path}")
        
        test_df = pd.read_csv(local_test_path)
        print(f"✅ Loaded test data: {local_test_path}")
        
        # Load demographics
        if not os.path.exists(local_demographics_path):
            raise FileNotFoundError(
                f"Demographics file not found: {local_demographics_path}\n"
                f"Please rerun Phase 1 preprocessing to generate demographics files."
            )
        
        demographics = pd.read_csv(local_demographics_path)
        print(f"✅ Loaded demographics: {local_demographics_path}")
        
    else:
        # Load from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub library required. "
                "Install with: pip install huggingface_hub"
            )
        
        print(f"Loading from HuggingFace Hub: {data_repo_id}")
        
        # Download test data
        test_path = hf_hub_download(
            repo_id=data_repo_id,
            filename="splits/test.csv",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        test_df = pd.read_csv(test_path)
        print(f"✅ Downloaded test data")
        
        # Download demographics
        try:
            demographics_path = hf_hub_download(
                repo_id=data_repo_id,
                filename="splits/test_demographics.csv",
                repo_type="dataset",
                cache_dir=cache_dir
            )
            demographics = pd.read_csv(demographics_path)
            print(f"✅ Downloaded demographics")
        except Exception as e:
            raise FileNotFoundError(
                f"Demographics file not found in HuggingFace repo.\n"
                f"Error: {e}\n"
                f"Please rerun Phase 1 preprocessing and upload to HuggingFace."
            )
    
    # Split features and target
    target_col = 'target' if 'target' in test_df.columns else 'readmitted'
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"\n📊 Data Summary:")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Features: {X_test.shape[1]}")
    print(f"   Target distribution: {dict(y_test.value_counts())}")
    print(f"   Demographics columns: {list(demographics.columns)}")
    print(f"   Demographics shape: {demographics.shape}")
    
    # Verify alignment
    if len(X_test) != len(demographics):
        raise ValueError(
            f"Misalignment: X_test has {len(X_test)} rows but demographics has {len(demographics)} rows"
        )
    
    print("="*80 + "\n")
    
    return X_test, y_test, demographics


def load_calibrated_model_and_calibrator(
    model_repo_id: str = "auphong2707/hospital-readmission-lgbm-calibrated",
    cache_dir: str = "./models/downloaded",
    use_local: bool = False,
    local_model_path: str = None,
    local_calibrator_path: str = None
) -> Tuple[Any, Any]:
    """
    Load calibrated model and calibrator from HuggingFace or local files.
    
    Args:
        model_repo_id: HuggingFace model repository ID
        cache_dir: Cache directory for downloads
        use_local: If True, load from local files
        local_model_path: Path to local model file
        local_calibrator_path: Path to local calibrator file
        
    Returns:
        tuple: (model, calibrator)
    """
    print("\n" + "="*80)
    print("📥 Loading Calibrated Model and Calibrator")
    print("="*80)
    
    if use_local:
        print(f"Loading from local files...")
        
        if not local_model_path or not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Model file not found: {local_model_path}")
        if not local_calibrator_path or not os.path.exists(local_calibrator_path):
            raise FileNotFoundError(f"Calibrator file not found: {local_calibrator_path}")
        
        model = joblib.load(local_model_path)
        calibrator = joblib.load(local_calibrator_path)
        
        print(f"✅ Loaded model: {local_model_path}")
        print(f"✅ Loaded calibrator: {local_calibrator_path}")
        
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub library required. "
                "Install with: pip install huggingface_hub"
            )
        
        print(f"Loading from HuggingFace Hub: {model_repo_id}")
        
        # Download model
        model_path = hf_hub_download(
            repo_id=model_repo_id,
            filename="gradient_boosting_model_original.joblib",
            repo_type="model",
            cache_dir=cache_dir
        )
        model = joblib.load(model_path)
        print(f"✅ Downloaded model")
        
        # Download calibrator
        calibrator_path = hf_hub_download(
            repo_id=model_repo_id,
            filename="Gradient_Boosting_(LightGBM)_calibrator.pkl",
            repo_type="model",
            cache_dir=cache_dir
        )
        calibrator = joblib.load(calibrator_path)
        print(f"✅ Downloaded calibrator")
    
    print("="*80 + "\n")
    
    return model, calibrator


def load_phase4_results(
    phase4_summary_path: str = "./phase-4-optimal-threshold-ROI-analysis/outputs/phase4_summary_for_phase5.json"
) -> Dict:
    """
    Load Phase 4 threshold optimization results.
    
    Args:
        phase4_summary_path: Path to Phase 4 summary JSON
        
    Returns:
        dict: Phase 4 results with thresholds, metrics, ROI
    """
    print("\n" + "="*80)
    print("📥 Loading Phase 4 Results")
    print("="*80)
    
    if not os.path.exists(phase4_summary_path):
        raise FileNotFoundError(
            f"Phase 4 summary not found: {phase4_summary_path}\n"
            f"Please run Phase 4 threshold optimization first."
        )
    
    with open(phase4_summary_path, 'r') as f:
        phase4_results = json.load(f)
    
    print(f"✅ Loaded Phase 4 summary: {phase4_summary_path}")
    print(f"\n📊 Phase 4 Key Results:")
    print(f"   Optimal threshold: {phase4_results['optimal_threshold']:.4f}")
    print(f"   Low risk threshold: {phase4_results['low_risk_threshold']:.4f}")
    print(f"   High risk threshold: {phase4_results['high_risk_threshold']:.4f}")
    print(f"   ROI: {phase4_results['roi_percentage']:.1f}%")
    print(f"   Net benefit: ${phase4_results['net_benefit']:,.2f}")
    print("="*80 + "\n")
    
    return phase4_results


def generate_calibrated_predictions(
    model: Any,
    calibrator: Any,
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Generate calibrated probability predictions.
    
    Args:
        model: Trained model
        calibrator: Fitted calibrator
        X_test: Test features
        
    Returns:
        np.ndarray: Calibrated probabilities
    """
    print("🔮 Generating calibrated predictions...")
    
    # Get uncalibrated probabilities (shape: (n_samples,))
    y_pred_proba_uncalibrated = model.predict_proba(X_test)[:, 1]

    # Apply calibration. Different calibrator implementations expect
    # different input shapes and return different output shapes:
    # - Our `ModelCalibrator.predict_proba` expects a 1D array and
    #   returns a 1D array of calibrated probabilities.
    # - sklearn-style calibrators expect a 2D array (n_samples, n_features)
    #   and return a 2D array of shape (n_samples, 2) with class probabilities.
    # Try common call patterns and normalize output to a 1D array.
    try:
        # First try passing a 2D column (sklearn compatible)
        pred = calibrator.predict_proba(y_pred_proba_uncalibrated.reshape(-1, 1))
    except Exception:
        # Fallback: try passing 1D array (our ModelCalibrator)
        pred = calibrator.predict_proba(y_pred_proba_uncalibrated)

    # Normalize output to 1D calibrated probabilities
    if isinstance(pred, np.ndarray):
        if pred.ndim == 1:
            y_pred_proba_calibrated = pred.ravel()
        elif pred.ndim == 2:
            # If returned probabilities for two classes, take probability of class 1
            if pred.shape[1] >= 2:
                y_pred_proba_calibrated = pred[:, 1]
            else:
                y_pred_proba_calibrated = pred.ravel()
        else:
            # Unexpected shape, attempt to flatten
            y_pred_proba_calibrated = pred.ravel()
    else:
        # Non-numpy return types (e.g., list), try converting
        pred_arr = np.asarray(pred)
        if pred_arr.ndim == 1:
            y_pred_proba_calibrated = pred_arr.ravel()
        elif pred_arr.ndim == 2 and pred_arr.shape[1] >= 2:
            y_pred_proba_calibrated = pred_arr[:, 1]
        else:
            y_pred_proba_calibrated = pred_arr.ravel()
    
    print(f"✅ Generated calibrated predictions")
    print(f"   Mean probability: {y_pred_proba_calibrated.mean():.3f}")
    print(f"   Probability range: [{y_pred_proba_calibrated.min():.3f}, {y_pred_proba_calibrated.max():.3f}]")
    
    return y_pred_proba_calibrated


# ============================================================================
# FAIRNESS METRICS AND GROUP ANALYSIS
# ============================================================================

class GroupPerformanceAnalyzer:
    """Analyze model performance by demographic groups."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_pred_proba: np.ndarray, demographics: pd.DataFrame):
        """
        Initialize analyzer.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            demographics: Demographics DataFrame with race, gender, age
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.demographics = demographics
        
    def compute_overall_metrics(self) -> Dict:
        """Compute overall performance metrics."""
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        
        metrics = {
            'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)},
            'accuracy': float(accuracy_score(self.y_true, self.y_pred)),
            'precision': float(precision_score(self.y_true, self.y_pred, zero_division=0)),
            'recall': float(recall_score(self.y_true, self.y_pred, zero_division=0)),
            'f1_score': float(f1_score(self.y_true, self.y_pred, zero_division=0)),
            'tpr': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
            'fpr': float(fp / (fp + tn) if (fp + tn) > 0 else 0),
            'tnr': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
            'fnr': float(fn / (fn + tp) if (fn + tp) > 0 else 0),
            'roc_auc': float(roc_auc_score(self.y_true, self.y_pred_proba)),
            'brier_score': float(brier_score_loss(self.y_true, self.y_pred_proba)),
            'intervention_rate': float(self.y_pred.mean())
        }
        
        return metrics
    
    def compute_group_metrics(self, attribute: str) -> pd.DataFrame:
        """
        Compute metrics for each group in a demographic attribute.
        
        Args:
            attribute: Demographic attribute ('race', 'gender', 'age')
            
        Returns:
            DataFrame with metrics per group
        """
        if attribute not in self.demographics.columns:
            raise ValueError(f"Attribute '{attribute}' not found in demographics")
        
        groups = self.demographics[attribute].unique()
        results = []
        
        for group in groups:
            mask = (self.demographics[attribute] == group).values
            
            if mask.sum() == 0:
                continue
            
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]
            y_pred_proba_group = self.y_pred_proba[mask]
            
            # Skip if no positive or negative samples
            if len(np.unique(y_true_group)) < 2:
                continue
            
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            
            group_metrics = {
                'attribute': attribute,
                'group': group,
                'n_samples': int(mask.sum()),
                'n_positive': int(y_true_group.sum()),
                'positive_rate': float(y_true_group.mean()),
                'TP': int(tp),
                'FP': int(fp),
                'TN': int(tn),
                'FN': int(fn),
                'accuracy': float(accuracy_score(y_true_group, y_pred_group)),
                'precision': float(precision_score(y_true_group, y_pred_group, zero_division=0)),
                'recall': float(recall_score(y_true_group, y_pred_group, zero_division=0)),
                'f1_score': float(f1_score(y_true_group, y_pred_group, zero_division=0)),
                'tpr': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
                'fpr': float(fp / (fp + tn) if (fp + tn) > 0 else 0),
                'tnr': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
                'fnr': float(fn / (fn + tp) if (fn + tp) > 0 else 0),
                'intervention_rate': float(y_pred_group.mean()),
                'roc_auc': float(roc_auc_score(y_true_group, y_pred_proba_group)) if len(np.unique(y_true_group)) > 1 else 0.0,
                'brier_score': float(brier_score_loss(y_true_group, y_pred_proba_group))
            }
            
            results.append(group_metrics)
        
        return pd.DataFrame(results)
    
    def compute_all_group_metrics(self) -> Dict[str, pd.DataFrame]:
        """Compute metrics for all demographic attributes."""
        results = {}
        
        for attribute in ['race', 'gender', 'age']:
            if attribute in self.demographics.columns:
                results[attribute] = self.compute_group_metrics(attribute)
        
        return results


class FairnessMetrics:
    """Calculate fairness metrics across demographic groups."""
    
    @staticmethod
    def demographic_parity(group_metrics: pd.DataFrame) -> Dict:
        """
        Calculate demographic parity (statistical parity).
        
        Measures if intervention rate is similar across groups.
        
        Args:
            group_metrics: DataFrame with group-specific metrics
            
        Returns:
            dict: Demographic parity results
        """
        intervention_rates = group_metrics['intervention_rate'].values
        
        return {
            'metric': 'demographic_parity',
            'definition': 'Intervention rate should be similar across groups',
            'intervention_rates': dict(zip(group_metrics['group'], intervention_rates)),
            'max_rate': float(intervention_rates.max()),
            'min_rate': float(intervention_rates.min()),
            'gap': float(intervention_rates.max() - intervention_rates.min()),
            'std': float(intervention_rates.std()),
            'passed': float(intervention_rates.max() - intervention_rates.min()) <= 0.05  # ±5% tolerance
        }
    
    @staticmethod
    def equalized_odds(group_metrics: pd.DataFrame) -> Dict:
        """
        Calculate equalized odds.
        
        Measures if TPR and FPR are similar across groups.
        
        Args:
            group_metrics: DataFrame with group-specific metrics
            
        Returns:
            dict: Equalized odds results
        """
        tpr_values = group_metrics['tpr'].values
        fpr_values = group_metrics['fpr'].values
        
        return {
            'metric': 'equalized_odds',
            'definition': 'TPR and FPR should be similar across groups',
            'tpr': dict(zip(group_metrics['group'], tpr_values)),
            'fpr': dict(zip(group_metrics['group'], fpr_values)),
            'tpr_gap': float(tpr_values.max() - tpr_values.min()),
            'fpr_gap': float(fpr_values.max() - fpr_values.min()),
            'tpr_std': float(tpr_values.std()),
            'fpr_std': float(fpr_values.std()),
            'passed': (float(tpr_values.max() - tpr_values.min()) <= 0.05 and 
                      float(fpr_values.max() - fpr_values.min()) <= 0.05)
        }
    
    @staticmethod
    def equal_opportunity(group_metrics: pd.DataFrame) -> Dict:
        """
        Calculate equal opportunity.
        
        Measures if TPR is similar across groups.
        
        Args:
            group_metrics: DataFrame with group-specific metrics
            
        Returns:
            dict: Equal opportunity results
        """
        tpr_values = group_metrics['tpr'].values
        
        return {
            'metric': 'equal_opportunity',
            'definition': 'TPR should be similar across groups',
            'tpr': dict(zip(group_metrics['group'], tpr_values)),
            'max_tpr': float(tpr_values.max()),
            'min_tpr': float(tpr_values.min()),
            'gap': float(tpr_values.max() - tpr_values.min()),
            'std': float(tpr_values.std()),
            'passed': float(tpr_values.max() - tpr_values.min()) <= 0.05
        }
    
    @staticmethod
    def compute_all_fairness_metrics(all_group_metrics: Dict[str, pd.DataFrame]) -> Dict:
        """Compute all fairness metrics for all attributes."""
        fairness_results = {}
        
        for attribute, group_metrics in all_group_metrics.items():
            if len(group_metrics) < 2:
                continue
            
            fairness_results[attribute] = {
                'demographic_parity': FairnessMetrics.demographic_parity(group_metrics),
                'equalized_odds': FairnessMetrics.equalized_odds(group_metrics),
                'equal_opportunity': FairnessMetrics.equal_opportunity(group_metrics)
            }
        
        return fairness_results


# ============================================================================
# STATISTICAL TESTING
# ============================================================================

class StatisticalTester:
    """Perform statistical significance tests for fairness evaluation."""
    
    @staticmethod
    def chi_square_test_intervention_rate(y_pred: np.ndarray, 
                                          demographics: pd.DataFrame,
                                          attribute: str) -> Dict:
        """
        Chi-square test for independence of intervention rate and demographic group.
        
        Args:
            y_pred: Predicted labels
            demographics: Demographics DataFrame
            attribute: Demographic attribute to test
            
        Returns:
            dict: Test results with chi2 statistic and p-value
        """
        if attribute not in demographics.columns:
            raise ValueError(f"Attribute '{attribute}' not found")
        
        # Create contingency table
        contingency = pd.crosstab(demographics[attribute], y_pred)
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        return {
            'test': 'chi_square_intervention_rate',
            'attribute': attribute,
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < 0.05,
            'interpretation': 'Significant bias detected' if p_value < 0.05 else 'No significant bias'
        }
    
    @staticmethod
    def proportion_test_tpr(group1_metrics: Dict, group2_metrics: Dict) -> Dict:
        """
        Two-proportion z-test for TPR difference between groups.
        
        Args:
            group1_metrics: Metrics for first group
            group2_metrics: Metrics for second group
            
        Returns:
            dict: Test results
        """
        # Calculate TPR for each group
        tp1, fn1 = group1_metrics['TP'], group1_metrics['FN']
        tp2, fn2 = group2_metrics['TP'], group2_metrics['FN']
        
        n1, n2 = tp1 + fn1, tp2 + fn2
        p1, p2 = tp1 / n1 if n1 > 0 else 0, tp2 / n2 if n2 > 0 else 0
        
        # Pooled proportion
        p_pool = (tp1 + tp2) / (n1 + n2) if (n1 + n2) > 0 else 0
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if (n1 > 0 and n2 > 0) else 0
        
        # Z-statistic
        z_stat = (p1 - p2) / se if se > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'test': 'two_proportion_z_test_tpr',
            'group1': group1_metrics['group'],
            'group2': group2_metrics['group'],
            'group1_tpr': float(p1),
            'group2_tpr': float(p2),
            'tpr_difference': float(p1 - p2),
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }


# ============================================================================
# FAIRNESS VISUALIZATIONS
# ============================================================================

class FairnessVisualizer:
    """Generate visualizations for fairness evaluation."""
    
    @staticmethod
    def plot_group_metrics_comparison(
        all_group_metrics: Dict[str, pd.DataFrame],
        output_dir: str,
        metric: str = 'tpr'
    ):
        """
        Plot comparison of a specific metric across demographic groups.
        
        Args:
            all_group_metrics: Dictionary of group metrics DataFrames
            output_dir: Directory to save plots
            metric: Metric to plot ('tpr', 'fpr', 'precision', 'intervention_rate')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metric_names = {
            'tpr': 'True Positive Rate (Recall)',
            'fpr': 'False Positive Rate',
            'precision': 'Precision',
            'intervention_rate': 'Intervention Rate',
            'f1_score': 'F1 Score'
        }
        
        fig, axes = plt.subplots(1, len(all_group_metrics), figsize=(6*len(all_group_metrics), 6))
        if len(all_group_metrics) == 1:
            axes = [axes]
        
        for idx, (attribute, group_df) in enumerate(all_group_metrics.items()):
            ax = axes[idx]
            
            # Sort by group name for consistent ordering
            group_df_sorted = group_df.sort_values('group')
            
            # Create bar plot
            bars = ax.bar(range(len(group_df_sorted)), group_df_sorted[metric], 
                          color=sns.color_palette("husl", len(group_df_sorted)))
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, group_df_sorted[metric])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            # Formatting
            ax.set_xlabel(attribute.capitalize(), fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_names.get(metric, metric), fontsize=12)
            ax.set_title(f'{metric_names.get(metric, metric)} by {attribute.capitalize()}', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(group_df_sorted)))
            ax.set_xticklabels(group_df_sorted['group'], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, min(1.0, group_df_sorted[metric].max() * 1.2))
            
            # Add horizontal line for overall mean
            mean_val = group_df_sorted[metric].mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.3f}', linewidth=2)
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / f'group_comparison_{metric}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {plot_path}")
    
    @staticmethod
    def plot_fairness_metrics_heatmap(
        fairness_results: Dict[str, Dict],
        output_dir: str
    ):
        """
        Create heatmap showing fairness metric violations.
        
        Args:
            fairness_results: Fairness metrics results
            output_dir: Directory to save plot
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for heatmap
        data = []
        attributes = []
        
        for attribute, metrics in fairness_results.items():
            attributes.append(attribute.capitalize())
            row = [
                1 if metrics['demographic_parity']['passed'] else 0,
                1 if metrics['equalized_odds']['passed'] else 0,
                1 if metrics['equal_opportunity']['passed'] else 0
            ]
            data.append(row)
        
        df = pd.DataFrame(data, 
                         index=attributes,
                         columns=['Demographic\nParity', 'Equalized\nOdds', 'Equal\nOpportunity'])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df, annot=True, cmap=['#e74c3c', '#2ecc71'], 
                   cbar_kws={'label': 'Pass (1) / Fail (0)'},
                   linewidths=2, linecolor='white',
                   fmt='g', ax=ax, vmin=0, vmax=1)
        
        ax.set_title('Fairness Metrics Assessment', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Fairness Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Demographic Attribute', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = output_path / 'fairness_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {plot_path}")
    
    @staticmethod
    def plot_confusion_matrices_by_group(
        all_group_metrics: Dict[str, pd.DataFrame],
        output_dir: str,
        attribute: str = 'race'
    ):
        """
        Plot confusion matrices for each demographic group.
        
        Args:
            all_group_metrics: Dictionary of group metrics DataFrames
            output_dir: Directory to save plots
            attribute: Demographic attribute to plot
        """
        if attribute not in all_group_metrics:
            print(f"   ⚠️  Attribute '{attribute}' not found")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        group_df = all_group_metrics[attribute]
        n_groups = len(group_df)
        
        # Calculate grid dimensions
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_groups == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(group_df.iterrows()):
            ax = axes[idx]
            
            # Create confusion matrix
            cm = np.array([[row['TN'], row['FP']], 
                          [row['FN'], row['TP']]])
            
            # Normalize for percentages
            cm_norm = cm.astype('float') / cm.sum()
            
            # Plot
            sns.heatmap(cm, annot=np.array([[f"{cm[0,0]}\n({cm_norm[0,0]:.1%})", 
                                            f"{cm[0,1]}\n({cm_norm[0,1]:.1%})"],
                                           [f"{cm[1,0]}\n({cm_norm[1,0]:.1%})", 
                                            f"{cm[1,1]}\n({cm_norm[1,1]:.1%})"]]),
                       fmt='', cmap='Blues', ax=ax, cbar=False,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            
            ax.set_title(f"{row['group']} (n={row['n_samples']:,})", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Confusion Matrices by {attribute.capitalize()}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plot_path = output_path / f'confusion_matrices_{attribute}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {plot_path}")
    
    @staticmethod
    def plot_calibration_by_group(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        demographics: pd.DataFrame,
        attribute: str,
        output_dir: str,
        n_bins: int = 10
    ):
        """
        Plot calibration curves for each demographic group.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            demographics: Demographics DataFrame
            attribute: Demographic attribute
            output_dir: Directory to save plots
            n_bins: Number of bins for calibration
        """
        if attribute not in demographics.columns:
            print(f"   ⚠️  Attribute '{attribute}' not found")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        groups = demographics[attribute].unique()
        colors = sns.color_palette("husl", len(groups))
        
        for group, color in zip(groups, colors):
            mask = (demographics[attribute] == group).values
            
            if mask.sum() == 0:
                continue
            
            y_true_group = y_true[mask]
            y_pred_group = y_pred_proba[mask]
            
            # Calculate calibration curve
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            observed_freq = []
            predicted_freq = []
            
            for i in range(n_bins):
                bin_mask = (y_pred_group >= bin_edges[i]) & (y_pred_group < bin_edges[i+1])
                if i == n_bins - 1:  # Include upper edge in last bin
                    bin_mask = (y_pred_group >= bin_edges[i]) & (y_pred_group <= bin_edges[i+1])
                
                if bin_mask.sum() > 0:
                    observed_freq.append(y_true_group[bin_mask].mean())
                    predicted_freq.append(y_pred_group[bin_mask].mean())
                else:
                    observed_freq.append(np.nan)
                    predicted_freq.append(np.nan)
            
            # Plot calibration curve
            ax.plot(predicted_freq, observed_freq, 'o-', 
                   label=f"{group} (n={mask.sum():,})", 
                   color=color, linewidth=2, markersize=8)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Calibration Curves by {attribute.capitalize()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        plot_path = output_path / f'calibration_by_{attribute}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {plot_path}")
    
    @staticmethod
    def plot_risk_distribution_by_group(
        risk_categories: np.ndarray,
        demographics: pd.DataFrame,
        attribute: str,
        output_dir: str
    ):
        """
        Plot risk category distribution for each demographic group.
        
        Args:
            risk_categories: Assigned risk categories
            demographics: Demographics DataFrame
            attribute: Demographic attribute
            output_dir: Directory to save plots
        """
        if attribute not in demographics.columns:
            print(f"   ⚠️  Attribute '{attribute}' not found")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        groups = demographics[attribute].unique()
        risk_levels = ['Low', 'Medium', 'High']
        
        data = []
        for group in groups:
            mask = (demographics[attribute] == group).values
            group_risk = risk_categories[mask]
            
            counts = pd.Series(group_risk).value_counts()
            percentages = counts / len(group_risk) * 100
            
            for risk in risk_levels:
                data.append({
                    'group': group,
                    'risk': risk,
                    'percentage': percentages.get(risk, 0)
                })
        
        df = pd.DataFrame(data)
        
        # Create stacked bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Pivot for stacked bar
        pivot_df = df.pivot(index='group', columns='risk', values='percentage')
        pivot_df = pivot_df[risk_levels]  # Ensure correct order
        
        pivot_df.plot(kind='bar', stacked=True, ax=ax,
                     color=['#2ecc71', '#f39c12', '#e74c3c'],
                     width=0.7)
        
        ax.set_xlabel(attribute.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Risk Category Distribution by {attribute.capitalize()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Risk Level', loc='upper right', fontsize=10)
        ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        
        plot_path = output_path / f'risk_distribution_{attribute}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {plot_path}")
    
    @staticmethod
    def generate_all_visualizations(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        risk_categories: np.ndarray,
        demographics: pd.DataFrame,
        all_group_metrics: Dict[str, pd.DataFrame],
        fairness_results: Dict[str, Dict],
        output_dir: str
    ):
        """
        Generate all fairness visualizations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            risk_categories: Risk categories
            demographics: Demographics DataFrame
            all_group_metrics: Group-specific metrics
            fairness_results: Fairness metrics results
            output_dir: Directory to save visualizations
        """
        print(f"\n📊 Generating fairness visualizations...")
        
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Group metrics comparison plots
        for metric in ['tpr', 'fpr', 'precision', 'intervention_rate']:
            FairnessVisualizer.plot_group_metrics_comparison(
                all_group_metrics, str(viz_dir), metric
            )
        
        # 2. Fairness heatmap
        FairnessVisualizer.plot_fairness_metrics_heatmap(
            fairness_results, str(viz_dir)
        )
        
        # 3. Confusion matrices
        for attribute in all_group_metrics.keys():
            FairnessVisualizer.plot_confusion_matrices_by_group(
                all_group_metrics, str(viz_dir), attribute
            )
        
        # 4. Calibration curves
        for attribute in demographics.columns:
            if attribute not in ['encounter_id', 'patient_nbr']:
                FairnessVisualizer.plot_calibration_by_group(
                    y_true, y_pred_proba, demographics, attribute, str(viz_dir)
                )
        
        # 5. Risk distribution
        for attribute in all_group_metrics.keys():
            FairnessVisualizer.plot_risk_distribution_by_group(
                risk_categories, demographics, attribute, str(viz_dir)
            )
        
        print(f"✅ All visualizations saved to: {viz_dir}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str, char: str = "=", width: int = 80):
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"✅ Saved results: {output_path}")


def upload_results_to_hf(
    output_dir: str,
    repo_id: str,
    commit_message: str = "Upload Phase 5 fairness evaluation results",
    token: Optional[str] = None,
    include_visualizations: bool = True
):
    """
    Upload Phase 5 fairness evaluation results to HuggingFace Hub.
    
    Args:
        output_dir: Directory containing output files (JSON, CSV, TXT)
        repo_id: HuggingFace repository ID (e.g., 'username/hospital-readmission-gradient-boosting-fairness-results')
        commit_message: Commit message for the upload
        token: HuggingFace API token (if None, will use HF_TOKEN environment variable)
        include_visualizations: If True, upload visualization files
        
    Returns:
        str: URL to the uploaded repository
        
    Raises:
        ImportError: If huggingface_hub is not installed
        ValueError: If required files are missing or token not provided
        
    Example:
        >>> upload_results_to_hf(
        ...     output_dir='./phase-5-fairness-evaluation-deployment-readiness/outputs',
        ...     repo_id='username/hospital-readmission-gradient-boosting-fairness-results'
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
    print("📤 Uploading Phase 5 Fairness Results to HuggingFace Hub")
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
            if file_path.is_file() and not file_path.name.startswith('.'):
                files_to_upload.append((str(file_path), f"outputs/{file_path.name}"))
    
    # Visualization files
    if include_visualizations:
        viz_path = output_path / "visualizations"
        if viz_path.exists():
            for file_path in viz_path.glob('*.png'):
                if file_path.is_file():
                    files_to_upload.append((str(file_path), f"visualizations/{file_path.name}"))
    
    if len(files_to_upload) == 0:
        raise ValueError(f"No files found to upload in {output_dir}")
    
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
- fairness-evaluation
- bias-detection
- ai-ethics
---

# Hospital Readmission Risk - Phase 5: Fairness Evaluation Results

This repository contains the results from Phase 5: Fairness Evaluation & Deployment Readiness.

## Contents

### Outputs
- `outputs/fairness_report.json`: Comprehensive fairness evaluation report
- `outputs/group_metrics_*.csv`: Performance metrics by demographic group (race, gender, age)
- `outputs/statistical_tests.json`: Statistical significance tests for bias detection
- `outputs/risk_categories_*.csv`: Risk category distribution by demographic group

## Fairness Metrics Evaluated

### Demographic Parity
Measures if intervention rate is similar across demographic groups (±5% tolerance).

### Equalized Odds
Measures if True Positive Rate (TPR) and False Positive Rate (FPR) are similar across groups (±5% tolerance).

### Equal Opportunity
Measures if True Positive Rate (TPR) is similar across groups (±5% tolerance).

## Statistical Tests

- **Chi-square test**: Tests independence of intervention rate and demographic group
- **Two-proportion z-test**: Tests TPR/FPR differences between groups

## Model Information

- **Model**: Gradient Boosting (LightGBM) with Platt Calibration
- **Optimal Threshold**: From Phase 4 ROI analysis
- **Test Set**: 15,265 patients
- **Demographics**: Race (6 categories), Gender (3 categories), Age (10 ranges)

## Usage

These results can be used for:
- Assessing model fairness before deployment
- Identifying potential bias in predictions
- Determining if bias mitigation is needed
- Creating model cards with fairness documentation
- Meeting regulatory requirements for AI fairness

## Deployment Readiness

Review the fairness report to determine if the model is ready for deployment or if bias mitigation strategies are needed.

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


if __name__ == "__main__":
    print("Phase 5 Fairness Evaluation Utilities")
    print("Import this module to use fairness evaluation functions")
