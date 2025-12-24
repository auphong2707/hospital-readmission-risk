"""
Phase 6 - Final System Evaluation Utilities

This module provides utilities for conducting final system evaluation with deployed
threshold configurations (global or group-specific). It aggregates metrics from
previous phases and generates comprehensive evaluation reports.

Classes:
    - DeploymentConfigLoader: Load and validate deployment configuration
    - ThresholdApplicator: Apply global or group-specific thresholds
    - FinalMetricsCalculator: Calculate comprehensive performance metrics
    - ROICalculator: Calculate financial impact and ROI metrics
    - RiskCategoryAnalyzer: Analyze risk stratification performance
    - FinalEvaluationVisualizer: Generate comprehensive visualizations
    - DeploymentReportGenerator: Create JSON and PDF deployment reports
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from scipy import stats
from huggingface_hub import hf_hub_download
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# MODEL CALIBRATOR CLASS (REQUIRED FOR LOADING PICKLED CALIBRATORS)
# ============================================================================

class ModelCalibrator:
    """
    Platt Scaling calibration for hospital readmission models.
    
    This class MUST be defined here to allow unpickling calibrator objects
    that were saved in Phase 3. When pickle loads an object, it needs access
    to the class definition.
    
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
# DATA LOADING FUNCTION (SHARED ACROSS ALL THREE MODELS)
# ============================================================================

def load_data_and_predictions(data_repo_id: str, model_repo_id: str):
    """
    Load test data and generate predictions from HuggingFace.
    
    This function is shared across all three models (GB, RF, LR) and automatically
    detects the model type from the repository ID to load the correct files.
    
    Args:
        data_repo_id: Repository ID for data (e.g., 'auphong2707/hospital-readmission-risk-data')
        model_repo_id: Repository ID for calibrated model (e.g., 'auphong2707/hospital-readmission-phase3-lgbm-calibrated')
        
    Returns:
        Tuple of (y_true, y_proba, demographics)
        
    Raises:
        ValueError: If model type cannot be determined from repo_id
    """
    from huggingface_hub import hf_hub_download
    import joblib
    import pickle
    
    print("\n" + "="*80)
    print("LOADING DATA AND PREDICTIONS")
    print("="*80)
    
    # Detect model type from repo_id to use correct filenames
    if 'lgbm' in model_repo_id.lower() or 'gradient' in model_repo_id.lower():
        model_filename = "gradient_boosting_model_original.joblib"
        calibrator_filename = "Gradient_Boosting_(LightGBM)_calibrator.pkl"
        model_type = "Gradient Boosting (LightGBM)"
    elif 'rf' in model_repo_id.lower() or 'random' in model_repo_id.lower():
        model_filename = "random_forest_model_original.joblib"
        calibrator_filename = "Random_Forest_calibrator.pkl"
        model_type = "Random Forest"
    elif 'lr' in model_repo_id.lower() or 'logistic' in model_repo_id.lower():
        model_filename = "logistic_regression_model_original.joblib"
        calibrator_filename = "Logistic_Regression_calibrator.pkl"
        model_type = "Logistic Regression"
    else:
        raise ValueError(f"Could not determine model type from repo_id: {model_repo_id}")
    
    print(f"📌 Detected model type: {model_type}")
    print(f"   Model file: {model_filename}")
    print(f"   Calibrator file: {calibrator_filename}")
    
    # Load test data from Phase 1
    print("\n📥 Downloading test data from HuggingFace...")
    test_data_path = hf_hub_download(
        repo_id=data_repo_id,
        filename="splits/test.csv",
        repo_type="dataset"
    )
    test_data = pd.read_csv(test_data_path)
    print(f"✓ Loaded test data: {test_data.shape}")
    
    # Load demographics from separate file (Phase 1 saves this separately)
    print("\n📥 Downloading demographics from HuggingFace...")
    demographics_path = hf_hub_download(
        repo_id=data_repo_id,
        filename="splits/test_demographics.csv",
        repo_type="dataset"
    )
    demographics = pd.read_csv(demographics_path)
    print(f"✓ Loaded demographics: {demographics.shape}")
    
    # Extract true labels (Phase 1 uses 'target' as column name)
    target_col = 'target' if 'target' in test_data.columns else 'readmitted'
    y_true = test_data[target_col].values
    
    print(f"✓ True labels: {len(y_true)} samples")
    print(f"  - Readmission rate: {np.mean(y_true):.2%}")
    
    # Load model and calibrator from Phase 3
    print("\n📥 Downloading model and calibrator from Phase 3...")
    model_path = hf_hub_download(
        repo_id=model_repo_id,
        filename=model_filename,
        repo_type="model"
    )
    
    # Load model - handle both .pkl and .joblib formats
    if model_filename.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = joblib.load(model_path)
    print(f"✓ Loaded model: {model_filename}")
    
    calibrator_path = hf_hub_download(
        repo_id=model_repo_id,
        filename=calibrator_filename,
        repo_type="model"
    )
    calibrator = joblib.load(calibrator_path)
    print(f"✓ Loaded calibrator: {calibrator_filename}")
    
    # Generate calibrated predictions
    print("\n🔮 Generating calibrated predictions...")
    # Extract features (drop target column)
    X_test = test_data.drop(columns=[target_col])
    
    # Get uncalibrated predictions
    y_proba_uncalibrated = model.predict_proba(X_test)[:, 1]
    
    # Apply calibration
    y_proba = calibrator.predict_proba(y_proba_uncalibrated)
    
    print(f"✓ Generated calibrated predictions: {len(y_proba)} samples")
    print(f"  - Mean probability: {np.mean(y_proba):.4f}")
    print(f"  - Probability range: [{np.min(y_proba):.4f}, {np.max(y_proba):.4f}]")
    
    # Verify alignment
    assert len(y_true) == len(y_proba), "Mismatch between labels and predictions"
    assert len(y_true) == len(demographics), "Mismatch between labels and demographics"
    
    return y_true, y_proba, demographics


# ============================================================================
# HELPER FUNCTIONS FOR FINAL EVALUATION (SHARED ACROSS ALL THREE MODELS)
# ============================================================================

def load_deployment_config(fairness_repo_id: str, output_base: str) -> dict:
    """
    Load deployment configuration from Phase 5.
    
    Args:
        fairness_repo_id: Repository ID for fairness assessment outputs
        output_base: Base output directory (will download to parent)
        
    Returns:
        Deployment configuration dictionary
    """
    print("\n" + "="*80)
    print("LOADING DEPLOYMENT CONFIGURATION")
    print("="*80)
    
    # Download deployment config from Phase 5
    # Note: Phase 5 uploads files to outputs/ subdirectory
    print("\n📥 Downloading deployment configuration from Phase 5...")
    config_path = hf_hub_download(
        repo_id=fairness_repo_id,
        filename="outputs/deployment_config.json",  # Phase 5 uploads to outputs/ directory
        repo_type="model"
    )
    
    # Load config
    loader = DeploymentConfigLoader(config_path)
    config = loader.load_config()
    print("✓ Successfully loaded deployment_config.json from Phase 5")
    
    return config


def apply_deployed_thresholds(y_proba, demographics, config):
    """
    Apply deployed threshold configuration to predictions.
    
    Args:
        y_proba: Probability predictions
        demographics: Demographics dataframe
        config: Deployment configuration
        
    Returns:
        Tuple of (predictions, threshold_summary)
    """
    print("\n" + "="*80)
    print("APPLYING DEPLOYED THRESHOLDS")
    print("="*80)
    
    threshold_config = config['threshold_configuration']
    use_group_thresholds = config['use_group_thresholds']
    
    applicator = ThresholdApplicator(threshold_config, use_group_thresholds)
    y_pred = applicator.apply_thresholds(y_proba, demographics)
    threshold_summary = applicator.get_threshold_summary()
    
    print(f"\n✓ Generated predictions: {len(y_pred)} samples")
    print(f"  - Positive predictions: {np.sum(y_pred)} ({np.mean(y_pred):.2%})")
    
    return y_pred, threshold_summary


def calculate_all_metrics(y_true, y_pred, y_proba, demographics, cost_matrix):
    """
    Calculate comprehensive metrics for final evaluation.
    
    Args:
        y_true: True labels
        y_pred: Binary predictions
        y_proba: Probability predictions
        demographics: Demographics dataframe
        cost_matrix: Cost parameters for ROI calculation
        
    Returns:
        Dictionary containing all metrics
    """
    print("\n" + "="*80)
    print("CALCULATING COMPREHENSIVE METRICS")
    print("="*80)
    
    # Performance metrics
    print("\n📊 Calculating performance metrics...")
    metrics_calc = FinalMetricsCalculator()
    performance_metrics = metrics_calc.calculate_performance_metrics(y_true, y_pred, y_proba)
    print(f"✓ Performance metrics calculated")
    print(f"  - Accuracy: {performance_metrics['accuracy']:.4f}")
    print(f"  - ROC-AUC: {performance_metrics['roc_auc']:.4f}")
    print(f"  - Precision: {performance_metrics['precision']:.4f}")
    print(f"  - Recall: {performance_metrics['recall']:.4f}")
    
    # Calibration metrics
    print("\n📊 Calculating calibration metrics...")
    calibration_metrics = metrics_calc.calculate_calibration_metrics(y_true, y_proba)
    print(f"✓ Calibration metrics calculated")
    print(f"  - Brier score: {calibration_metrics['brier_score']:.4f}")
    print(f"  - ECE: {calibration_metrics['expected_calibration_error']:.4f}")
    
    # Group-specific metrics
    print("\n📊 Calculating group-specific metrics...")
    group_metrics = metrics_calc.calculate_group_metrics(y_true, y_pred, y_proba, demographics)
    print(f"✓ Group metrics calculated for {len(group_metrics)} features")
    for feature, groups in group_metrics.items():
        print(f"  - {feature.capitalize()}: {len(groups)} groups")
    
    # Fairness metrics
    print("\n📊 Calculating fairness metrics...")
    fairness_metrics = metrics_calc.calculate_fairness_metrics(group_metrics)
    print(f"✓ Fairness metrics calculated")
    for feature, metrics in fairness_metrics.items():
        print(f"  - {feature.capitalize()} TPR disparity: {metrics['tpr_disparity']:.4f}")
    
    # ROI metrics
    print("\n📊 Calculating ROI metrics...")
    roi_calc = ROICalculator(cost_matrix)
    roi_metrics = roi_calc.calculate_roi_metrics(y_true, y_pred)
    print(f"✓ ROI metrics calculated")
    print(f"  - Expected value: ${roi_metrics['expected_value']:,.0f}")
    print(f"  - Cost savings: ${roi_metrics['cost_savings']:,.0f}")
    print(f"  - ROI: {roi_metrics['roi_percentage']:.2f}%")
    
    # Risk stratification analysis
    print("\n📊 Analyzing risk stratification...")
    risk_boundaries = {
        'low': (0.0, 0.3),
        'medium': (0.3, 0.7),
        'high': (0.7, 1.0)
    }
    risk_analyzer = RiskCategoryAnalyzer(risk_boundaries)
    risk_analysis = risk_analyzer.analyze_risk_distribution(y_true, y_proba)
    print(f"✓ Risk stratification analyzed")
    for level, stats in risk_analysis.items():
        print(f"  - {level.capitalize()} risk: {stats['count']} patients ({stats['percentage']:.1f}%)")
    
    return {
        'performance': performance_metrics,
        'calibration': calibration_metrics,
        'group_metrics': group_metrics,
        'fairness': fairness_metrics,
        'roi': roi_metrics,
        'risk_stratification': risk_analysis
    }


def generate_visualizations(y_true, y_pred, y_proba, demographics, metrics, 
                           threshold_summary, output_dir):
    """
    Generate comprehensive visualizations for final evaluation.
    
    Args:
        y_true: True labels
        y_pred: Binary predictions
        y_proba: Probability predictions
        demographics: Demographics dataframe
        metrics: Dictionary of all calculated metrics
        threshold_summary: Summary of deployed thresholds
        output_dir: Directory to save visualizations
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    visualizer = FinalEvaluationVisualizer(viz_dir)
    
    viz_count = 0
    
    # 1. Confusion Matrix
    print("\n📊 Creating confusion matrix...")
    visualizer.plot_confusion_matrix(y_true, y_pred, "Final System Confusion Matrix")
    viz_count += 1
    
    # 2. Calibration Curve
    print("📊 Creating calibration curve...")
    visualizer.plot_calibration_curve(y_true, y_proba)
    viz_count += 1
    
    # 3-5. Group Performance Comparisons (TPR, FPR, Precision)
    print("📊 Creating group performance comparisons...")
    for metric in ['tpr', 'fpr', 'precision']:
        visualizer.plot_group_performance_comparison(metrics['group_metrics'], metric)
        viz_count += 1
    
    # 6. Fairness Disparities
    print("📊 Creating fairness disparity visualization...")
    visualizer.plot_fairness_disparities(metrics['fairness'])
    viz_count += 1
    
    # 7. ROI Breakdown
    print("📊 Creating ROI breakdown...")
    visualizer.plot_roi_breakdown(metrics['roi'])
    viz_count += 1
    
    # 8. Risk Distribution
    print("📊 Creating risk distribution visualization...")
    visualizer.plot_risk_distribution(metrics['risk_stratification'])
    viz_count += 1
    
    # 9. Threshold Configuration
    print("📊 Creating threshold configuration visualization...")
    visualizer.plot_threshold_comparison(threshold_summary, demographics)
    viz_count += 1
    
    print(f"\n✓ Generated {viz_count} visualizations in {viz_dir}/")


def generate_reports(model_name, config, metrics, threshold_summary, output_dir):
    """
    Generate final evaluation reports.
    
    Args:
        model_name: Name of the model
        config: Deployment configuration
        metrics: Dictionary of all calculated metrics
        threshold_summary: Summary of deployed thresholds
        output_dir: Directory to save reports
    """
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    
    report_gen = DeploymentReportGenerator(output_dir)
    
    # Generate final_system_metrics.json (single source of truth)
    print("\n📝 Creating final_system_metrics.json...")
    report_gen.generate_final_metrics_json(
        model_name=model_name,
        deployment_config=config,
        performance_metrics=metrics['performance'],
        calibration_metrics=metrics['calibration'],
        group_metrics=metrics['group_metrics'],
        fairness_metrics=metrics['fairness'],
        roi_metrics=metrics['roi'],
        risk_analysis=metrics['risk_stratification'],
        threshold_summary=threshold_summary
    )
    
    # Generate deployment_report.json (for stakeholders)
    print("📝 Creating deployment_report.json...")
    report_gen.generate_deployment_report_json(
        model_name=model_name,
        deployment_config=config,
        performance_metrics=metrics['performance'],
        fairness_metrics=metrics['fairness'],
        roi_metrics=metrics['roi'],
        risk_analysis=metrics['risk_stratification']
    )
    
    print("\n✓ Reports generated successfully")


# ============================================================================
# CONFIGURATION AND THRESHOLD CLASSES
# ============================================================================

class DeploymentConfigLoader:
    """Load and validate deployment configuration from Phase 5."""
    
    def __init__(self, config_path: str):
        """
        Initialize the deployment config loader.
        
        Args:
            config_path: Path to deployment_config.json from Phase 5
        """
        self.config_path = config_path
        self.config = None
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load and validate deployment configuration.
        
        Returns:
            Dictionary containing deployment configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Deployment config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Validate required fields
        required_fields = ['is_mitigated', 'use_group_thresholds', 'threshold_configuration']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in config: {field}")
        
        print(f"✓ Loaded deployment config: {self.config_path}")
        print(f"  - Mitigated: {self.config['is_mitigated']}")
        print(f"  - Use group thresholds: {self.config['use_group_thresholds']}")
        
        return self.config
    
    def get_threshold_config(self) -> Dict[str, Any]:
        """Get threshold configuration from loaded config."""
        if self.config is None:
            self.load_config()
        return self.config['threshold_configuration']
    
    def is_group_threshold_deployment(self) -> bool:
        """Check if deployment uses group-specific thresholds."""
        if self.config is None:
            self.load_config()
        return self.config.get('use_group_thresholds', False)


class ThresholdApplicator:
    """Apply global or group-specific thresholds to predictions."""
    
    def __init__(self, threshold_config: Dict[str, Any], use_group_thresholds: bool):
        """
        Initialize threshold applicator.
        
        Args:
            threshold_config: Threshold configuration from deployment config
            use_group_thresholds: Whether to use group-specific thresholds
        """
        self.threshold_config = threshold_config
        self.use_group_thresholds = use_group_thresholds
        
    def apply_thresholds(self, 
                        y_proba: np.ndarray,
                        demographics: pd.DataFrame) -> np.ndarray:
        """
        Apply appropriate thresholds to probability predictions.
        
        Args:
            y_proba: Probability predictions (n_samples,)
            demographics: DataFrame with demographic features (race, gender, age)
            
        Returns:
            Binary predictions (n_samples,)
        """
        predictions = np.zeros(len(y_proba), dtype=int)
        
        if self.use_group_thresholds:
            # Apply group-specific thresholds
            group_thresholds = self.threshold_config.get('group_thresholds', {})
            
            for group_key, threshold in group_thresholds.items():
                # Parse group key (e.g., "race_Caucasian", "gender_Female", "age_[70-80)")
                if '_' not in group_key:
                    continue
                    
                feature, value = group_key.split('_', 1)
                
                # Create mask for this group
                mask = demographics[feature] == value
                
                # Apply threshold for this group
                predictions[mask] = (y_proba[mask] >= threshold).astype(int)
            
            print(f"✓ Applied group-specific thresholds to {len(group_thresholds)} groups")
        else:
            # Apply global threshold
            global_threshold = self.threshold_config.get('global_threshold', None)
            
            # If not in config, this is an error - should always be provided
            if global_threshold is None:
                raise ValueError(
                    "Global threshold not found in deployment_config.json. "
                    "Expected 'threshold_configuration.global_threshold' to be set. "
                    "This should come from Phase 4's optimal threshold."
                )
            
            predictions = (y_proba >= global_threshold).astype(int)
            print(f"✓ Applied global threshold: {global_threshold:.4f}")
        
        return predictions
    
    def get_threshold_summary(self) -> Dict[str, Any]:
        """Get summary of applied thresholds."""
        if self.use_group_thresholds:
            group_thresholds = self.threshold_config.get('group_thresholds', {})
            return {
                'type': 'group_specific',
                'num_groups': len(group_thresholds),
                'thresholds': group_thresholds
            }
        else:
            return {
                'type': 'global',
                'threshold': self.threshold_config.get('global_threshold', 0.5)
            }


class FinalMetricsCalculator:
    """Calculate comprehensive performance, calibration, and fairness metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics = {}
        
    def calculate_performance_metrics(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Binary predictions
            y_proba: Probability predictions
            
        Returns:
            Dictionary of performance metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_proba)),
            'pr_auc': float(average_precision_score(y_true, y_proba)),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'sensitivity': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
            'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
            'positive_predictive_value': float(tp / (tp + fp) if (tp + fp) > 0 else 0),
            'negative_predictive_value': float(tn / (tn + fn) if (tn + fn) > 0 else 0)
        }
        
        self.metrics['performance'] = metrics
        return metrics
    
    def calculate_calibration_metrics(self,
                                     y_true: np.ndarray,
                                     y_proba: np.ndarray,
                                     n_bins: int = 10) -> Dict[str, Any]:
        """
        Calculate calibration metrics.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary of calibration metrics
        """
        # Brier score
        brier = float(brier_score_loss(y_true, y_proba))
        
        # Expected Calibration Error (ECE)
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
        
        # Calculate bin sizes
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0
        bin_accuracies = []
        bin_confidences = []
        bin_sizes = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(y_true[mask])
                bin_confidence = np.mean(y_proba[mask])
                bin_size = np.sum(mask)
                
                ece += (bin_size / len(y_true)) * abs(bin_accuracy - bin_confidence)
                
                bin_accuracies.append(float(bin_accuracy))
                bin_confidences.append(float(bin_confidence))
                bin_sizes.append(int(bin_size))
            else:
                bin_accuracies.append(None)
                bin_confidences.append(None)
                bin_sizes.append(0)
        
        metrics = {
            'brier_score': brier,
            'expected_calibration_error': float(ece),
            'calibration_curve': {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            },
            'bin_statistics': {
                'accuracies': bin_accuracies,
                'confidences': bin_confidences,
                'sizes': bin_sizes
            }
        }
        
        self.metrics['calibration'] = metrics
        return metrics
    
    def calculate_group_metrics(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_proba: np.ndarray,
                                demographics: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate metrics for each demographic group.
        
        Args:
            y_true: True labels
            y_pred: Binary predictions
            y_proba: Probability predictions
            demographics: DataFrame with demographic features
            
        Returns:
            Dictionary of group-specific metrics
        """
        group_metrics = {}
        
        for feature in ['race', 'gender', 'age']:
            if feature not in demographics.columns:
                continue
            
            feature_metrics = {}
            for group in demographics[feature].unique():
                mask = demographics[feature] == group
                if np.sum(mask) < 10:  # Skip very small groups
                    continue
                
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                y_proba_group = y_proba[mask]
                
                # Calculate confusion matrix
                if len(np.unique(y_true_group)) > 1:
                    tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
                else:
                    # Handle case where only one class present
                    if y_true_group[0] == 1:
                        tp = np.sum(y_pred_group == 1)
                        fn = np.sum(y_pred_group == 0)
                        tn = fp = 0
                    else:
                        tn = np.sum(y_pred_group == 0)
                        fp = np.sum(y_pred_group == 1)
                        tp = fn = 0
                
                feature_metrics[str(group)] = {
                    'size': int(np.sum(mask)),
                    'base_rate': float(np.mean(y_true_group)),
                    'positive_rate': float(np.mean(y_pred_group)),
                    'tpr': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
                    'fpr': float(fp / (fp + tn) if (fp + tn) > 0 else 0),
                    'precision': float(tp / (tp + fp) if (tp + fp) > 0 else 0),
                    'accuracy': float(accuracy_score(y_true_group, y_pred_group))
                }
            
            group_metrics[feature] = feature_metrics
        
        self.metrics['group_metrics'] = group_metrics
        return group_metrics
    
    def calculate_fairness_metrics(self,
                                   group_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate fairness disparity metrics across groups.
        
        Args:
            group_metrics: Group-specific metrics from calculate_group_metrics
            
        Returns:
            Dictionary of fairness metrics
        """
        fairness_metrics = {}
        
        for feature, groups in group_metrics.items():
            if len(groups) < 2:
                continue
            
            # Extract TPR and FPR for all groups
            tprs = [metrics['tpr'] for metrics in groups.values()]
            fprs = [metrics['fpr'] for metrics in groups.values()]
            positive_rates = [metrics['positive_rate'] for metrics in groups.values()]
            
            # Calculate disparities
            fairness_metrics[feature] = {
                'tpr_disparity': float(max(tprs) - min(tprs)),
                'fpr_disparity': float(max(fprs) - min(fprs)),
                'demographic_parity_disparity': float(max(positive_rates) - min(positive_rates)),
                'tpr_ratio': float(min(tprs) / max(tprs) if max(tprs) > 0 else 0),
                'fpr_ratio': float(min(fprs) / max(fprs) if max(fprs) > 0 else 0)
            }
        
        self.metrics['fairness'] = fairness_metrics
        return fairness_metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all calculated metrics."""
        return self.metrics


class ROICalculator:
    """Calculate financial impact and ROI metrics."""
    
    def __init__(self, cost_matrix: Dict[str, float]):
        """
        Initialize ROI calculator.
        
        Args:
            cost_matrix: Dictionary with costs for TP, TN, FP, FN
        """
        self.cost_matrix = cost_matrix
        
    def calculate_roi_metrics(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive ROI metrics.
        
        Args:
            y_true: True labels
            y_pred: Binary predictions
            
        Returns:
            Dictionary of ROI metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate expected value (net benefit)
        tp_benefit = tp * self.cost_matrix['TP']
        tn_benefit = tn * self.cost_matrix['TN']
        fp_cost = fp * self.cost_matrix['FP']
        fn_cost = fn * self.cost_matrix['FN']
        
        expected_value = tp_benefit + tn_benefit + fp_cost + fn_cost
        
        # Calculate baseline scenario (predict all negative - no intervention)
        baseline_readmissions = int(np.sum(y_true))
        
        # Extract intervention cost per patient from cost matrix
        # TP benefit = readmission_cost - intervention_cost, so:
        # intervention_cost = readmission_cost - TP_benefit
        # FN cost = -readmission_cost (negative because it's a cost)
        # Therefore: intervention_cost = -FN - TP
        intervention_cost_per_patient = abs(self.cost_matrix['FP'])  # FP is the cost of unnecessary intervention
        
        # Calculate intervention costs
        intervention_volume = int(tp + fp)
        intervention_cost_total = intervention_volume * intervention_cost_per_patient
        
        # Calculate prevented readmissions and savings
        prevented_readmissions = int(tp)
        readmission_cost_per_patient = abs(self.cost_matrix['FN'])  # FN is the cost of missed readmission
        prevented_readmission_savings = prevented_readmissions * readmission_cost_per_patient
        
        # Calculate missed readmissions and costs
        missed_readmissions = int(fn)
        missed_readmission_costs = missed_readmissions * readmission_cost_per_patient
        
        # Calculate baseline cost (no intervention - all readmissions occur)
        baseline_cost = baseline_readmissions * readmission_cost_per_patient
        
        # Calculate baseline expected value (predict all negative - no intervention)
        # In expected value framework: TN=0, FN is negative (cost), so baseline is negative
        baseline_expected_value = (len(y_true) - baseline_readmissions) * self.cost_matrix['TN'] + baseline_readmissions * self.cost_matrix['FN']
        
        # Cost with intervention (pure cost accounting view)
        cost_with_intervention = intervention_cost_total + missed_readmission_costs
        
        # Total savings (cost accounting: baseline cost - actual cost)
        total_savings = baseline_cost - cost_with_intervention
        
        # Cost savings (expected value framework: difference in expected values)
        # This should equal total_savings but calculated via expected value difference
        cost_savings = expected_value - baseline_expected_value
        
        # ROI calculation (return on intervention investment)
        # Use net benefit relative to intervention cost
        roi_percentage = (expected_value / intervention_cost_total * 100) if intervention_cost_total > 0 else 0
        
        # Savings percentage
        savings_percentage = (total_savings / baseline_cost * 100) if baseline_cost > 0 else 0
        
        # Readmission rate metrics
        total_patients = int(len(y_true))
        baseline_readmission_rate = baseline_readmissions / total_patients
        intervention_readmission_rate = missed_readmissions / total_patients
        readmission_reduction_absolute = baseline_readmission_rate - intervention_readmission_rate
        readmission_reduction_relative = (readmission_reduction_absolute / baseline_readmission_rate * 100) if baseline_readmission_rate > 0 else 0
        
        metrics = {
            # Confusion matrix components
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'total_patients': total_patients,
            
            # Expected value metrics
            'expected_value': float(expected_value),
            'baseline_expected_value': float(baseline_expected_value),
            'cost_savings': float(cost_savings),
            
            # ROI metrics
            'roi_percentage': float(roi_percentage),
            'savings_percentage': float(savings_percentage),
            
            # Intervention metrics
            'intervention_volume': intervention_volume,
            'intervention_rate': float(intervention_volume / total_patients),
            'intervention_cost_total': float(intervention_cost_total),
            
            # Readmission prevention metrics
            'prevented_readmissions': prevented_readmissions,
            'prevented_readmission_savings': float(prevented_readmission_savings),
            'missed_readmissions': missed_readmissions,
            'missed_readmission_costs': float(missed_readmission_costs),
            
            # Baseline vs intervention comparison
            'baseline_readmissions': baseline_readmissions,
            'baseline_cost': float(baseline_cost),
            'cost_with_intervention': float(cost_with_intervention),
            'total_savings': float(total_savings),
            
            # Readmission rate metrics
            'baseline_readmission_rate': float(baseline_readmission_rate),
            'intervention_readmission_rate': float(intervention_readmission_rate),
            'readmission_reduction_absolute': float(readmission_reduction_absolute),
            'readmission_reduction_relative': float(readmission_reduction_relative),
            
            # Per-unit metrics
            'benefit_per_tp': float(tp_benefit / tp if tp > 0 else 0),
            'benefit_per_tn': float(tn_benefit / tn if tn > 0 else 0),
            'cost_per_fp': float(fp_cost / fp if fp > 0 else 0),
            'cost_per_fn': float(fn_cost / fn if fn > 0 else 0),
            'avg_cost_per_patient': float(expected_value / total_patients),
            'total_missed_readmission_cost': float(abs(fn_cost))
        }
        
        return metrics


class RiskCategoryAnalyzer:
    """Analyze risk stratification performance."""
    
    def __init__(self, risk_boundaries: Dict[str, Tuple[float, float]]):
        """
        Initialize risk category analyzer.
        
        Args:
            risk_boundaries: Dictionary mapping risk levels to (min, max) probability ranges
                            e.g., {'low': (0.0, 0.3), 'medium': (0.3, 0.7), 'high': (0.7, 1.0)}
        """
        self.risk_boundaries = risk_boundaries
        
    def categorize_risk(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Categorize probabilities into risk levels.
        
        Args:
            y_proba: Probability predictions
            
        Returns:
            Array of risk categories
        """
        categories = np.empty(len(y_proba), dtype=object)
        
        for risk_level, (min_prob, max_prob) in self.risk_boundaries.items():
            mask = (y_proba >= min_prob) & (y_proba < max_prob)
            categories[mask] = risk_level
        
        # Handle edge case for maximum boundary
        if len(self.risk_boundaries) > 0:
            max_level = max(self.risk_boundaries.items(), key=lambda x: x[1][1])[0]
            categories[y_proba >= self.risk_boundaries[max_level][1]] = max_level
        
        return categories
    
    def analyze_risk_distribution(self,
                                  y_true: np.ndarray,
                                  y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Analyze performance within each risk category.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            
        Returns:
            Dictionary of risk category statistics
        """
        risk_categories = self.categorize_risk(y_proba)
        
        analysis = {}
        for risk_level in self.risk_boundaries.keys():
            mask = risk_categories == risk_level
            if np.sum(mask) == 0:
                continue
            
            y_true_risk = y_true[mask]
            y_proba_risk = y_proba[mask]
            
            analysis[risk_level] = {
                'count': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(y_true) * 100),
                'actual_readmission_rate': float(np.mean(y_true_risk)),
                'mean_predicted_probability': float(np.mean(y_proba_risk)),
                'std_predicted_probability': float(np.std(y_proba_risk)),
                'min_probability': float(np.min(y_proba_risk)),
                'max_probability': float(np.max(y_proba_risk))
            }
        
        return analysis


class FinalEvaluationVisualizer:
    """Generate comprehensive visualizations for final evaluation."""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             title: str = "Confusion Matrix") -> str:
        """Plot confusion matrix with percentages."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        
        # Add percentages
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]:.1%})',
                       ha='center', va='center', fontsize=10, color='gray')
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        ax.set_xticklabels(['No Readmission', 'Readmission'])
        ax.set_yticklabels(['No Readmission', 'Readmission'])
        
        output_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_calibration_curve(self,
                               y_true: np.ndarray,
                               y_proba: np.ndarray,
                               n_bins: int = 10) -> str:
        """Plot calibration curve."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax.plot(prob_pred, prob_true, 's-', label='Model', linewidth=2, markersize=8)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve (Reliability Diagram)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, 'calibration_curve.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_group_performance_comparison(self,
                                         group_metrics: Dict[str, Dict],
                                         metric_name: str = 'tpr') -> str:
        """Plot comparison of performance metric across groups."""
        fig, axes = plt.subplots(1, len(group_metrics), figsize=(6 * len(group_metrics), 5))
        
        if len(group_metrics) == 1:
            axes = [axes]
        
        for idx, (feature, groups) in enumerate(group_metrics.items()):
            group_names = list(groups.keys())
            metric_values = [groups[g][metric_name] for g in group_names]
            
            axes[idx].bar(range(len(group_names)), metric_values, color='steelblue', alpha=0.7)
            axes[idx].set_xlabel(feature.capitalize())
            axes[idx].set_ylabel(metric_name.upper())
            axes[idx].set_title(f'{metric_name.upper()} by {feature.capitalize()}')
            axes[idx].set_xticks(range(len(group_names)))
            axes[idx].set_xticklabels(group_names, rotation=45, ha='right')
            axes[idx].set_ylim([0, 1])
            axes[idx].axhline(y=np.mean(metric_values), color='r', linestyle='--', 
                             label='Mean', alpha=0.5)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'group_{metric_name}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_fairness_disparities(self, fairness_metrics: Dict[str, Dict]) -> str:
        """Plot fairness disparity metrics."""
        metrics_to_plot = ['tpr_disparity', 'fpr_disparity', 'demographic_parity_disparity']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics_to_plot):
            features = list(fairness_metrics.keys())
            values = [fairness_metrics[f][metric] for f in features]
            
            axes[idx].bar(range(len(features)), values, color='coral', alpha=0.7)
            axes[idx].set_xlabel('Demographic Feature')
            axes[idx].set_ylabel('Disparity')
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].set_xticks(range(len(features)))
            axes[idx].set_xticklabels(features, rotation=0)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'fairness_disparities.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_roi_breakdown(self, roi_metrics: Dict[str, float]) -> str:
        """Plot ROI expected value breakdown."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Expected value comparison
        values = [roi_metrics['expected_value'], roi_metrics['baseline_expected_value']]
        labels = ['With Model', 'Baseline\n(No Model)']
        colors = ['steelblue', 'gray']
        
        bars = ax1.bar(range(len(values)), values, color=colors, alpha=0.7)
        ax1.set_ylabel('Expected Value ($)')
        ax1.set_title('Expected Value Comparison')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        # Cost savings visualization
        cost_savings = roi_metrics['cost_savings']
        roi_pct = roi_metrics['roi_percentage']
        
        ax2.barh(['Cost Savings'], [cost_savings], color='green' if cost_savings > 0 else 'red', alpha=0.7)
        ax2.set_xlabel('Cost Savings ($)')
        ax2.set_title(f'Return on Investment: {roi_pct:.1f}%')
        ax2.text(cost_savings, 0, f'  ${cost_savings:,.0f}', va='center', 
                fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'roi_breakdown.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_risk_distribution(self, risk_analysis: Dict[str, Any]) -> str:
        """Plot risk category distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        risk_levels = list(risk_analysis.keys())
        counts = [risk_analysis[r]['count'] for r in risk_levels]
        readmission_rates = [risk_analysis[r]['actual_readmission_rate'] for r in risk_levels]
        
        # Distribution bar chart
        colors = ['green', 'yellow', 'red'][:len(risk_levels)]
        ax1.bar(range(len(risk_levels)), counts, color=colors, alpha=0.7)
        ax1.set_xlabel('Risk Category')
        ax1.set_ylabel('Number of Patients')
        ax1.set_title('Patient Distribution by Risk Category')
        ax1.set_xticks(range(len(risk_levels)))
        ax1.set_xticklabels([r.capitalize() for r in risk_levels])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add percentages
        total = sum(counts)
        for i, (count, risk) in enumerate(zip(counts, risk_levels)):
            pct = count / total * 100
            ax1.text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
        
        # Readmission rates by risk
        ax2.bar(range(len(risk_levels)), readmission_rates, color=colors, alpha=0.7)
        ax2.set_xlabel('Risk Category')
        ax2.set_ylabel('Actual Readmission Rate')
        ax2.set_title('Readmission Rate by Risk Category')
        ax2.set_xticks(range(len(risk_levels)))
        ax2.set_xticklabels([r.capitalize() for r in risk_levels])
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, rate in enumerate(readmission_rates):
            ax2.text(i, rate, f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'risk_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_threshold_comparison(self,
                                  threshold_summary: Dict[str, Any],
                                  demographics: pd.DataFrame = None) -> str:
        """Plot threshold configuration visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if threshold_summary['type'] == 'global':
            # Simple visualization for global threshold
            threshold = threshold_summary['threshold']
            ax.barh(['Global Threshold'], [threshold], color='steelblue', alpha=0.7)
            ax.set_xlabel('Threshold Value')
            ax.set_title('Deployed Threshold Configuration')
            ax.set_xlim([0, 1])
            ax.text(threshold, 0, f'  {threshold:.3f}', va='center', fontsize=12)
            
        else:
            # Visualization for group-specific thresholds
            thresholds = threshold_summary['thresholds']
            
            # Sort and prepare data
            groups = list(thresholds.keys())
            values = [thresholds[g] for g in groups]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(groups))
            ax.barh(y_pos, values, color='steelblue', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(groups)
            ax.set_xlabel('Threshold Value')
            ax.set_title(f'Group-Specific Thresholds ({len(groups)} groups)')
            ax.set_xlim([0, 1])
            
            # Add value labels
            for i, (group, value) in enumerate(zip(groups, values)):
                ax.text(value, i, f'  {value:.3f}', va='center', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'threshold_configuration.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path


class DeploymentReportGenerator:
    """Generate comprehensive deployment reports in JSON and summary format."""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_final_metrics_json(self,
                                    model_name: str,
                                    deployment_config: Dict[str, Any],
                                    performance_metrics: Dict[str, float],
                                    calibration_metrics: Dict[str, Any],
                                    group_metrics: Dict[str, Dict],
                                    fairness_metrics: Dict[str, Any],
                                    roi_metrics: Dict[str, float],
                                    risk_analysis: Dict[str, Any],
                                    threshold_summary: Dict[str, Any]) -> str:
        """
        Generate comprehensive final_system_metrics.json file.
        
        This is the single source of truth for Phase 7 publication.
        """
        final_metrics = {
            'model_name': model_name,
            'deployment_configuration': {
                'is_mitigated': deployment_config['is_mitigated'],
                'use_group_thresholds': deployment_config['use_group_thresholds'],
                'threshold_summary': threshold_summary
            },
            'performance_metrics': performance_metrics,
            'calibration_metrics': {
                'brier_score': calibration_metrics['brier_score'],
                'expected_calibration_error': calibration_metrics['expected_calibration_error']
            },
            'group_performance': group_metrics,
            'fairness_metrics': fairness_metrics,
            'roi_metrics': roi_metrics,
            'risk_stratification': risk_analysis,
            'summary': self._generate_summary(
                performance_metrics, fairness_metrics, roi_metrics
            )
        }
        
        output_path = os.path.join(self.output_dir, 'final_system_metrics.json')
        with open(output_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"✓ Saved final system metrics: {output_path}")
        return output_path
    
    def generate_deployment_report_json(self,
                                       model_name: str,
                                       deployment_config: Dict[str, Any],
                                       performance_metrics: Dict[str, float],
                                       fairness_metrics: Dict[str, Any],
                                       roi_metrics: Dict[str, float],
                                       risk_analysis: Dict[str, Any]) -> str:
        """
        Generate deployment_report.json for clinical stakeholders.
        
        This is a simplified, stakeholder-friendly version.
        """
        report = {
            'model_name': model_name,
            'deployment_ready': self._assess_deployment_readiness(
                performance_metrics, fairness_metrics, roi_metrics
            ),
            'key_performance_indicators': {
                'accuracy': performance_metrics['accuracy'],
                'sensitivity': performance_metrics['sensitivity'],
                'specificity': performance_metrics['specificity'],
                'roc_auc': performance_metrics['roc_auc']
            },
            'fairness_assessment': {
                'is_mitigated': deployment_config['is_mitigated'],
                'max_tpr_disparity': max([fm['tpr_disparity'] for fm in fairness_metrics.values()]) if fairness_metrics else 0,
                'max_fpr_disparity': max([fm['fpr_disparity'] for fm in fairness_metrics.values()]) if fairness_metrics else 0,
                'fairness_status': 'Acceptable' if all(
                    fm['tpr_disparity'] < 0.1 and fm['fpr_disparity'] < 0.1 
                    for fm in fairness_metrics.values()
                ) else 'Needs Attention'
            },
            'financial_impact': {
                'roi_percentage': roi_metrics['roi_percentage'],
                'cost_savings': roi_metrics['cost_savings'],
                'avg_cost_per_patient': roi_metrics['avg_cost_per_patient']
            },
            'risk_stratification': {
                level: {
                    'patient_count': analysis['count'],
                    'percentage': analysis['percentage'],
                    'readmission_rate': analysis['actual_readmission_rate']
                }
                for level, analysis in risk_analysis.items()
            },
            'recommendations': self._generate_recommendations(
                performance_metrics, fairness_metrics, roi_metrics
            )
        }
        
        output_path = os.path.join(self.output_dir, 'deployment_report.json')
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Saved deployment report: {output_path}")
        return output_path
    
    def _generate_summary(self,
                         performance_metrics: Dict[str, float],
                         fairness_metrics: Dict[str, Any],
                         roi_metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate human-readable summary."""
        return {
            'performance': f"Accuracy: {performance_metrics['accuracy']:.1%}, "
                          f"ROC-AUC: {performance_metrics['roc_auc']:.3f}",
            'fairness': f"Max TPR Disparity: {max([fm['tpr_disparity'] for fm in fairness_metrics.values()]):.3f}" if fairness_metrics else "N/A",
            'roi': f"ROI: {roi_metrics['roi_percentage']:.1f}%, "
                  f"Savings: ${roi_metrics['cost_savings']:,.0f}"
        }
    
    def _assess_deployment_readiness(self,
                                     performance_metrics: Dict[str, float],
                                     fairness_metrics: Dict[str, Any],
                                     roi_metrics: Dict[str, float]) -> bool:
        """Assess if system is ready for deployment."""
        # Performance check
        performance_ok = (
            performance_metrics['accuracy'] >= 0.60 and
            performance_metrics['roc_auc'] >= 0.65
        )
        
        # Fairness check
        fairness_ok = all(
            fm['tpr_disparity'] < 0.15 and fm['fpr_disparity'] < 0.15
            for fm in fairness_metrics.values()
        ) if fairness_metrics else True
        
        # ROI check
        roi_ok = roi_metrics['roi_percentage'] > 0
        
        return performance_ok and fairness_ok and roi_ok
    
    def _generate_recommendations(self,
                                 performance_metrics: Dict[str, float],
                                 fairness_metrics: Dict[str, Any],
                                 roi_metrics: Dict[str, float]) -> List[str]:
        """Generate deployment recommendations."""
        recommendations = []
        
        if performance_metrics['accuracy'] < 0.65:
            recommendations.append(
                "Consider model retraining or feature engineering to improve accuracy"
            )
        
        if fairness_metrics and any(fm['tpr_disparity'] > 0.1 for fm in fairness_metrics.values()):
            recommendations.append(
                "Implement group-specific thresholds to reduce TPR disparities"
            )
        
        if roi_metrics['roi_percentage'] < 10:
            recommendations.append(
                "Review cost-benefit parameters and consider intervention optimization"
            )
        
        if not recommendations:
            recommendations.append(
                "System meets all deployment criteria and is ready for production"
            )
        
        return recommendations
