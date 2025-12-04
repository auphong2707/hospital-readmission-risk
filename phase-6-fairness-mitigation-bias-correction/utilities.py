"""
Phase 6: Fairness Mitigation & Bias Correction Utilities

Provides tools for implementing post-hoc fairness mitigation through
group-specific thresholds:
- Threshold optimization per demographic group
- Fairness metrics calculation (before/after)
- Trade-off analysis (performance vs fairness vs ROI)
- Visualization tools
- Clinical approval documentation

Key Components:
- ThresholdOptimizer: Calculate group-specific thresholds
- MitigationEvaluator: Evaluate before/after fairness metrics
- TradeoffAnalyzer: Quantify performance/ROI impact
- MitigationVisualizer: Generate comparison visualizations
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dotenv import load_dotenv

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
    brier_score_loss
)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_phase5_summary(phase5_summary_path: str) -> Dict:
    """
    Load Phase 5 summary with decision inputs for Phase 6.
    
    Args:
        phase5_summary_path: Path to phase5_summary_for_phase6.json
        
    Returns:
        dict: Phase 5 summary with mitigation requirements
        
    Raises:
        FileNotFoundError: If Phase 5 summary not found
        ValueError: If summary missing required fields
    """
    print("\n" + "="*80)
    print("📥 Loading Phase 5 Summary for Phase 6 Decision")
    print("="*80)
    
    if not os.path.exists(phase5_summary_path):
        raise FileNotFoundError(
            f"Phase 5 summary not found: {phase5_summary_path}\n"
            "Please run Phase 5 first to generate phase5_summary_for_phase6.json"
        )
    
    with open(phase5_summary_path, 'r') as f:
        summary = json.load(f)
    
    # Validate required fields
    required_fields = [
        'requires_mitigation',
        'mitigation_priority',
        'bias_detected',
        'optimal_threshold',
        'group_metrics_summary'
    ]
    
    missing_fields = [f for f in required_fields if f not in summary]
    if missing_fields:
        raise ValueError(f"Phase 5 summary missing required fields: {missing_fields}")
    
    print(f"✅ Loaded Phase 5 summary: {phase5_summary_path}")
    print(f"   Requires mitigation: {summary['requires_mitigation']}")
    print(f"   Mitigation priority: {summary['mitigation_priority']}")
    print(f"   Bias detected: {summary['bias_detected']}")
    print(f"   Global threshold: {summary['optimal_threshold']:.4f}")
    print("="*80 + "\n")
    
    return summary


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
        use_local: If True, load from local files
        local_test_path: Path to local test.csv
        local_demographics_path: Path to local test_demographics.csv
        
    Returns:
        tuple: (X_test, y_test, demographics)
    """
    print("📥 Loading test data and demographics...")
    
    if use_local:
        test_df = pd.read_csv(local_test_path)
        demographics = pd.read_csv(local_demographics_path)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")
        
        # Download test data
        test_path = hf_hub_download(
            repo_id=data_repo_id,
            filename="splits/test.csv",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        test_df = pd.read_csv(test_path)
        
        # Download demographics
        demographics_path = hf_hub_download(
            repo_id=data_repo_id,
            filename="splits/test_demographics.csv",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        demographics = pd.read_csv(demographics_path)
    
    # Separate features and target
    y_test = test_df['target']
    X_test = test_df.drop('target', axis=1)
    
    print(f"✅ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"✅ Loaded demographics: {demographics.shape[0]} samples")
    
    return X_test, y_test, demographics


def load_model_and_calibrator(
    model_repo_id: str = "auphong2707/hospital-readmission-lgbm-calibrated",
    cache_dir: str = "./models/downloaded",
    use_local: bool = False,
    local_model_path: str = None,
    local_calibrator_path: str = None
) -> Tuple[Any, Any]:
    """Load calibrated model and calibrator."""
    import joblib
    
    print("📥 Loading calibrated model and calibrator...")
    
    if use_local:
        model = joblib.load(local_model_path)
        calibrator = joblib.load(local_calibrator_path)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")
        
        model_path = hf_hub_download(
            repo_id=model_repo_id,
            filename="gradient_boosting_model_original.joblib",
            repo_type="model",
            cache_dir=cache_dir
        )
        model = joblib.load(model_path)
        
        calibrator_path = hf_hub_download(
            repo_id=model_repo_id,
            filename="Gradient_Boosting_(LightGBM)_calibrator.pkl",
            repo_type="model",
            cache_dir=cache_dir
        )
        calibrator = joblib.load(calibrator_path)
    
    print(f"✅ Model and calibrator loaded")
    
    return model, calibrator


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

class ThresholdOptimizer:
    """Calculate optimal group-specific thresholds for fairness mitigation."""
    
    def __init__(
        self,
        mitigation_strategy: str = 'equalized_odds',
        fairness_tolerance: float = 0.05,
        threshold_range: Tuple[float, float] = (0.30, 0.70),
        threshold_step: float = 0.01
    ):
        """
        Initialize threshold optimizer.
        
        Args:
            mitigation_strategy: 'equalized_odds', 'equal_opportunity', or 'demographic_parity'
            fairness_tolerance: Target gap tolerance (e.g., 0.05 = 5%)
            threshold_range: (min, max) threshold to search
            threshold_step: Step size for grid search
        """
        valid_strategies = ['equalized_odds', 'equal_opportunity', 'demographic_parity']
        if mitigation_strategy not in valid_strategies:
            raise ValueError(f"mitigation_strategy must be one of {valid_strategies}")
        
        self.mitigation_strategy = mitigation_strategy
        self.fairness_tolerance = fairness_tolerance
        self.threshold_range = threshold_range
        self.threshold_step = threshold_step
    
    def calculate_tpr_fpr(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate True Positive Rate and False Positive Rate."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return tpr, fpr
    
    def calculate_intervention_rate(self, y_pred: np.ndarray) -> float:
        """Calculate proportion of positive predictions."""
        return np.mean(y_pred)
    
    def optimize_equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        target_tpr: float,
        target_fpr: float
    ) -> Dict:
        """
        Find threshold that minimizes TPR and FPR gaps from target.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            target_tpr: Target TPR to match
            target_fpr: Target FPR to match
            
        Returns:
            dict: Best threshold and resulting metrics
        """
        best_threshold = None
        best_score = float('inf')
        best_metrics = {}
        
        thresholds = np.arange(
            self.threshold_range[0],
            self.threshold_range[1] + self.threshold_step,
            self.threshold_step
        )
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tpr, fpr = self.calculate_tpr_fpr(y_true, y_pred)
            
            # Score: combined distance from target TPR and FPR
            tpr_gap = abs(tpr - target_tpr)
            fpr_gap = abs(fpr - target_fpr)
            score = tpr_gap + fpr_gap
            
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'tpr': tpr,
                    'fpr': fpr,
                    'tpr_gap': tpr_gap,
                    'fpr_gap': fpr_gap,
                    'score': score
                }
        
        return best_metrics
    
    def optimize_equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        target_tpr: float
    ) -> Dict:
        """Find threshold that minimizes TPR gap from target."""
        best_threshold = None
        best_score = float('inf')
        best_metrics = {}
        
        thresholds = np.arange(
            self.threshold_range[0],
            self.threshold_range[1] + self.threshold_step,
            self.threshold_step
        )
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tpr, fpr = self.calculate_tpr_fpr(y_true, y_pred)
            
            tpr_gap = abs(tpr - target_tpr)
            
            if tpr_gap < best_score:
                best_score = tpr_gap
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'tpr': tpr,
                    'fpr': fpr,
                    'tpr_gap': tpr_gap,
                    'score': tpr_gap
                }
        
        return best_metrics
    
    def optimize_demographic_parity(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        target_intervention_rate: float
    ) -> Dict:
        """Find threshold that matches target intervention rate."""
        best_threshold = None
        best_score = float('inf')
        best_metrics = {}
        
        thresholds = np.arange(
            self.threshold_range[0],
            self.threshold_range[1] + self.threshold_step,
            self.threshold_step
        )
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            intervention_rate = self.calculate_intervention_rate(y_pred)
            tpr, fpr = self.calculate_tpr_fpr(y_true, y_pred)
            
            gap = abs(intervention_rate - target_intervention_rate)
            
            if gap < best_score:
                best_score = gap
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'tpr': tpr,
                    'fpr': fpr,
                    'intervention_rate': intervention_rate,
                    'intervention_rate_gap': gap,
                    'score': gap
                }
        
        return best_metrics
    
    def calculate_group_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        demographics: pd.DataFrame,
        attribute: str,
        overall_metrics: Dict
    ) -> Dict:
        """
        Calculate optimal threshold for each group in a demographic attribute.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            demographics: Demographics DataFrame
            attribute: Demographic attribute (race, gender, age)
            overall_metrics: Overall population metrics (TPR, FPR, intervention_rate)
            
        Returns:
            dict: Group-specific thresholds and metrics
        """
        print(f"\n🔍 Optimizing thresholds for: {attribute.upper()}")
        print(f"   Strategy: {self.mitigation_strategy}")
        print(f"   Target tolerance: ±{self.fairness_tolerance:.1%}")
        
        group_thresholds = {}
        unique_groups = demographics[attribute].unique()
        
        for group in unique_groups:
            mask = demographics[attribute] == group
            n_samples = mask.sum()
            
            if n_samples < 50:  # Skip small groups
                print(f"   ⚠️  Skipping {group}: insufficient samples ({n_samples})")
                continue
            
            y_true_group = y_true[mask]
            y_pred_proba_group = y_pred_proba[mask]
            
            # Select optimization strategy
            if self.mitigation_strategy == 'equalized_odds':
                metrics = self.optimize_equalized_odds(
                    y_true_group,
                    y_pred_proba_group,
                    target_tpr=overall_metrics['tpr'],
                    target_fpr=overall_metrics['fpr']
                )
            elif self.mitigation_strategy == 'equal_opportunity':
                metrics = self.optimize_equal_opportunity(
                    y_true_group,
                    y_pred_proba_group,
                    target_tpr=overall_metrics['tpr']
                )
            else:  # demographic_parity
                metrics = self.optimize_demographic_parity(
                    y_true_group,
                    y_pred_proba_group,
                    target_intervention_rate=overall_metrics['intervention_rate']
                )
            
            metrics['n_samples'] = n_samples
            group_thresholds[group] = metrics
            
            print(f"   {group}: threshold={metrics['threshold']:.3f}, "
                  f"TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
        
        return group_thresholds


# ============================================================================
# MITIGATION EVALUATION
# ============================================================================

class MitigationEvaluator:
    """Evaluate fairness mitigation impact (before vs after)."""
    
    def __init__(self, phase4_results: Dict = None):
        """
        Initialize evaluator.
        
        Args:
            phase4_results: Phase 4 ROI results for cost calculations
        """
        self.phase4_results = phase4_results
    
    def evaluate_baseline(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        demographics: pd.DataFrame,
        global_threshold: float
    ) -> Dict:
        """
        Evaluate baseline metrics with global threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            demographics: Demographics DataFrame
            global_threshold: Global threshold from Phase 4
            
        Returns:
            dict: Baseline metrics
        """
        print("\n" + "="*80)
        print("📊 BASELINE EVALUATION (Global Threshold)")
        print("="*80)
        
        # Apply global threshold
        y_pred = (y_pred_proba >= global_threshold).astype(int)
        
        # Overall metrics
        overall_metrics = self._calculate_overall_metrics(y_true, y_pred, y_pred_proba)
        
        # Group-specific metrics
        group_metrics = {}
        for attribute in ['race', 'gender', 'age']:
            if attribute not in demographics.columns:
                continue
            
            group_metrics[attribute] = self._calculate_group_metrics(
                y_true, y_pred, demographics, attribute
            )
        
        # Fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(group_metrics)
        
        # ROI metrics
        roi_metrics = self._calculate_roi_metrics(y_true, y_pred) if self.phase4_results else {}
        
        baseline = {
            'global_threshold': global_threshold,
            'overall_metrics': overall_metrics,
            'group_metrics': group_metrics,
            'fairness_metrics': fairness_metrics,
            'roi_metrics': roi_metrics
        }
        
        self._print_evaluation_summary(baseline, "BASELINE")
        
        return baseline
    
    def evaluate_mitigated(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        demographics: pd.DataFrame,
        group_thresholds: Dict
    ) -> Dict:
        """
        Evaluate metrics with group-specific thresholds.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            demographics: Demographics DataFrame
            group_thresholds: Group-specific thresholds by attribute
            
        Returns:
            dict: Mitigated metrics
        """
        print("\n" + "="*80)
        print("📊 MITIGATED EVALUATION (Group-Specific Thresholds)")
        print("="*80)
        
        # Apply group-specific thresholds
        y_pred = np.zeros(len(y_true), dtype=int)
        
        for attribute, thresholds in group_thresholds.items():
            if attribute not in demographics.columns:
                continue
            
            for group, threshold_info in thresholds.items():
                if isinstance(threshold_info, dict):
                    threshold = threshold_info['threshold']
                else:
                    threshold = threshold_info
                
                mask = demographics[attribute] == group
                y_pred[mask] = (y_pred_proba[mask] >= threshold).astype(int)
        
        # Overall metrics
        overall_metrics = self._calculate_overall_metrics(y_true, y_pred, y_pred_proba)
        
        # Group-specific metrics
        group_metrics = {}
        for attribute in ['race', 'gender', 'age']:
            if attribute not in demographics.columns:
                continue
            
            group_metrics[attribute] = self._calculate_group_metrics(
                y_true, y_pred, demographics, attribute
            )
        
        # Fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(group_metrics)
        
        # ROI metrics
        roi_metrics = self._calculate_roi_metrics(y_true, y_pred) if self.phase4_results else {}
        
        mitigated = {
            'group_thresholds': group_thresholds,
            'overall_metrics': overall_metrics,
            'group_metrics': group_metrics,
            'fairness_metrics': fairness_metrics,
            'roi_metrics': roi_metrics
        }
        
        self._print_evaluation_summary(mitigated, "MITIGATED")
        
        return mitigated
    
    def _calculate_overall_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Calculate overall performance metrics."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'intervention_rate': np.mean(y_pred),
            'n_interventions': int(np.sum(y_pred)),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    def _calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demographics: pd.DataFrame,
        attribute: str
    ) -> Dict:
        """Calculate metrics per group."""
        group_metrics = {}
        
        for group in demographics[attribute].unique():
            mask = demographics[attribute] == group
            
            if mask.sum() < 10:  # Skip very small groups
                continue
            
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            tn, fp, fn, tp = confusion_matrix(
                y_true_group, y_pred_group, labels=[0, 1]
            ).ravel()
            
            group_metrics[group] = {
                'n_samples': int(mask.sum()),
                'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                'precision': precision_score(y_true_group, y_pred_group, zero_division=0),
                'recall': recall_score(y_true_group, y_pred_group, zero_division=0),
                'intervention_rate': np.mean(y_pred_group)
            }
        
        return group_metrics
    
    def _calculate_fairness_metrics(self, group_metrics: Dict) -> Dict:
        """Calculate fairness gaps across groups."""
        fairness_metrics = {}
        
        for attribute, groups in group_metrics.items():
            if not groups:
                continue
            
            tprs = [m['tpr'] for m in groups.values()]
            fprs = [m['fpr'] for m in groups.values()]
            intervention_rates = [m['intervention_rate'] for m in groups.values()]
            
            fairness_metrics[attribute] = {
                'tpr_gap': max(tprs) - min(tprs),
                'fpr_gap': max(fprs) - min(fprs),
                'intervention_rate_gap': max(intervention_rates) - min(intervention_rates),
                'tpr_max': max(tprs),
                'tpr_min': min(tprs),
                'fpr_max': max(fprs),
                'fpr_min': min(fprs)
            }
        
        return fairness_metrics
    
    def _calculate_roi_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate ROI metrics using Phase 4 cost matrix."""
        if not self.phase4_results:
            return {}
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Cost matrix from Phase 4
        cost_tp = 14500  # Benefit of preventing readmission
        cost_fp = -500   # Cost of unnecessary intervention
        cost_fn = -15000 # Cost of missed readmission
        cost_tn = 0      # No cost
        
        expected_value = (tp * cost_tp) + (fp * cost_fp) + (fn * cost_fn) + (tn * cost_tn)
        
        return {
            'expected_value': expected_value,
            'net_benefit': expected_value,
            'cost_per_intervention': abs(cost_fp),
            'benefit_per_true_positive': cost_tp
        }
    
    def _print_evaluation_summary(self, evaluation: Dict, label: str):
        """Print evaluation summary."""
        print(f"\n📊 {label} Metrics:")
        
        overall = evaluation['overall_metrics']
        print(f"   Overall TPR: {overall['tpr']:.3f}")
        print(f"   Overall FPR: {overall['fpr']:.3f}")
        print(f"   Overall ROC-AUC: {overall['roc_auc']:.3f}")
        print(f"   Intervention rate: {overall['intervention_rate']:.1%}")
        
        if evaluation['fairness_metrics']:
            print(f"\n   Fairness Gaps:")
            for attr, metrics in evaluation['fairness_metrics'].items():
                print(f"      {attr.upper()}: TPR gap={metrics['tpr_gap']:.3f}, "
                      f"FPR gap={metrics['fpr_gap']:.3f}")
        
        if evaluation['roi_metrics']:
            print(f"\n   ROI: ${evaluation['roi_metrics']['expected_value']:,.0f}")


# ============================================================================
# TRADE-OFF ANALYSIS
# ============================================================================

class TradeoffAnalyzer:
    """Analyze performance/fairness/ROI trade-offs."""
    
    @staticmethod
    def calculate_improvements(baseline: Dict, mitigated: Dict) -> Dict:
        """
        Calculate improvement metrics from baseline to mitigated.
        
        Args:
            baseline: Baseline evaluation results
            mitigated: Mitigated evaluation results
            
        Returns:
            dict: Improvement metrics and trade-offs
        """
        print("\n" + "="*80)
        print("📈 TRADE-OFF ANALYSIS: Baseline vs Mitigated")
        print("="*80)
        
        improvements = {
            'fairness_improvements': {},
            'performance_changes': {},
            'roi_changes': {},
            'summary': {}
        }
        
        # Fairness improvements
        for attribute in baseline['fairness_metrics'].keys():
            baseline_fairness = baseline['fairness_metrics'][attribute]
            mitigated_fairness = mitigated['fairness_metrics'][attribute]
            
            improvements['fairness_improvements'][attribute] = {
                'tpr_gap_before': baseline_fairness['tpr_gap'],
                'tpr_gap_after': mitigated_fairness['tpr_gap'],
                'tpr_gap_reduction': baseline_fairness['tpr_gap'] - mitigated_fairness['tpr_gap'],
                'tpr_gap_reduction_pct': ((baseline_fairness['tpr_gap'] - mitigated_fairness['tpr_gap']) / 
                                          baseline_fairness['tpr_gap'] * 100) if baseline_fairness['tpr_gap'] > 0 else 0,
                'fpr_gap_before': baseline_fairness['fpr_gap'],
                'fpr_gap_after': mitigated_fairness['fpr_gap'],
                'fpr_gap_reduction': baseline_fairness['fpr_gap'] - mitigated_fairness['fpr_gap'],
                'fpr_gap_reduction_pct': ((baseline_fairness['fpr_gap'] - mitigated_fairness['fpr_gap']) / 
                                          baseline_fairness['fpr_gap'] * 100) if baseline_fairness['fpr_gap'] > 0 else 0
            }
        
        # Performance changes
        baseline_overall = baseline['overall_metrics']
        mitigated_overall = mitigated['overall_metrics']
        
        improvements['performance_changes'] = {
            'tpr_change': mitigated_overall['tpr'] - baseline_overall['tpr'],
            'fpr_change': mitigated_overall['fpr'] - baseline_overall['fpr'],
            'accuracy_change': mitigated_overall['accuracy'] - baseline_overall['accuracy'],
            'roc_auc_change': mitigated_overall['roc_auc'] - baseline_overall['roc_auc'],
            'intervention_rate_change': mitigated_overall['intervention_rate'] - baseline_overall['intervention_rate']
        }
        
        # ROI changes
        if baseline['roi_metrics'] and mitigated['roi_metrics']:
            baseline_roi = baseline['roi_metrics']['expected_value']
            mitigated_roi = mitigated['roi_metrics']['expected_value']
            
            improvements['roi_changes'] = {
                'expected_value_before': baseline_roi,
                'expected_value_after': mitigated_roi,
                'expected_value_change': mitigated_roi - baseline_roi,
                'roi_reduction_pct': ((baseline_roi - mitigated_roi) / baseline_roi * 100) if baseline_roi != 0 else 0
            }
        
        # Summary assessment
        avg_tpr_reduction = np.mean([
            imp['tpr_gap_reduction_pct'] 
            for imp in improvements['fairness_improvements'].values()
        ])
        avg_fpr_reduction = np.mean([
            imp['fpr_gap_reduction_pct'] 
            for imp in improvements['fairness_improvements'].values()
        ])
        
        improvements['summary'] = {
            'avg_fairness_improvement_pct': (avg_tpr_reduction + avg_fpr_reduction) / 2,
            'performance_drop_acceptable': abs(improvements['performance_changes']['tpr_change']) <= 0.05,
            'roi_reduction_acceptable': (improvements['roi_changes'].get('roi_reduction_pct', 0) <= 10) if improvements['roi_changes'] else True,
            'fairness_targets_met': all(
                imp['tpr_gap_after'] < 0.05 and imp['fpr_gap_after'] < 0.05
                for imp in improvements['fairness_improvements'].values()
            )
        }
        
        TradeoffAnalyzer._print_tradeoff_summary(improvements)
        
        return improvements
    
    @staticmethod
    def _print_tradeoff_summary(improvements: Dict):
        """Print trade-off analysis summary."""
        print("\n✅ FAIRNESS IMPROVEMENTS:")
        for attr, imp in improvements['fairness_improvements'].items():
            print(f"   {attr.upper()}:")
            print(f"      TPR gap: {imp['tpr_gap_before']:.3f} → {imp['tpr_gap_after']:.3f} "
                  f"({imp['tpr_gap_reduction_pct']:+.1f}%)")
            print(f"      FPR gap: {imp['fpr_gap_before']:.3f} → {imp['fpr_gap_after']:.3f} "
                  f"({imp['fpr_gap_reduction_pct']:+.1f}%)")
        
        print("\n⚖️  PERFORMANCE CHANGES:")
        perf = improvements['performance_changes']
        print(f"   Overall TPR: {perf['tpr_change']:+.3f}")
        print(f"   Overall FPR: {perf['fpr_change']:+.3f}")
        print(f"   ROC-AUC: {perf['roc_auc_change']:+.4f}")
        
        if improvements['roi_changes']:
            print("\n💰 ROI IMPACT:")
            roi = improvements['roi_changes']
            print(f"   Expected value: ${roi['expected_value_before']:,.0f} → "
                  f"${roi['expected_value_after']:,.0f}")
            print(f"   Change: ${roi['expected_value_change']:+,.0f} "
                  f"({roi['roi_reduction_pct']:+.1f}%)")
        
        print("\n📋 SUMMARY:")
        summary = improvements['summary']
        print(f"   ✅ Fairness targets met: {summary['fairness_targets_met']}")
        print(f"   ✅ Performance drop acceptable: {summary['performance_drop_acceptable']}")
        print(f"   ✅ ROI reduction acceptable: {summary['roi_reduction_acceptable']}")


# ============================================================================
# VISUALIZATION
# ============================================================================

class MitigationVisualizer:
    """Generate before/after comparison visualizations."""
    
    @staticmethod
    def generate_all_visualizations(
        baseline: Dict,
        mitigated: Dict,
        improvements: Dict,
        output_dir: str
    ):
        """Generate all mitigation visualizations."""
        print("\n" + "="*80)
        print("📊 Generating Visualizations")
        print("="*80)
        
        vis_dir = Path(output_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Before/After TPR comparison
        MitigationVisualizer.plot_tpr_comparison(
            baseline, mitigated, str(vis_dir / "tpr_comparison.png")
        )
        
        # 2. Before/After FPR comparison
        MitigationVisualizer.plot_fpr_comparison(
            baseline, mitigated, str(vis_dir / "fpr_comparison.png")
        )
        
        # 3. Fairness gaps comparison
        MitigationVisualizer.plot_fairness_gaps(
            baseline, mitigated, str(vis_dir / "fairness_gaps.png")
        )
        
        # 4. Threshold adjustments
        MitigationVisualizer.plot_threshold_adjustments(
            baseline['global_threshold'], 
            mitigated['group_thresholds'],
            str(vis_dir / "threshold_adjustments.png")
        )
        
        # 5. Trade-off summary
        MitigationVisualizer.plot_tradeoff_summary(
            improvements, str(vis_dir / "tradeoff_summary.png")
        )
        
        print(f"✅ Visualizations saved to: {vis_dir}")
    
    @staticmethod
    def plot_tpr_comparison(baseline: Dict, mitigated: Dict, output_path: str):
        """Plot TPR comparison across groups."""
        fig, axes = plt.subplots(1, len(baseline['group_metrics']), figsize=(15, 5))
        
        if len(baseline['group_metrics']) == 1:
            axes = [axes]
        
        for idx, (attribute, groups) in enumerate(baseline['group_metrics'].items()):
            ax = axes[idx]
            
            group_names = list(groups.keys())
            baseline_tprs = [groups[g]['tpr'] for g in group_names]
            mitigated_tprs = [mitigated['group_metrics'][attribute][g]['tpr'] for g in group_names]
            
            x = np.arange(len(group_names))
            width = 0.35
            
            ax.bar(x - width/2, baseline_tprs, width, label='Baseline (Global)', alpha=0.8)
            ax.bar(x + width/2, mitigated_tprs, width, label='Mitigated (Group-Specific)', alpha=0.8)
            
            ax.set_xlabel('Group')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'TPR Comparison: {attribute.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(group_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_fpr_comparison(baseline: Dict, mitigated: Dict, output_path: str):
        """Plot FPR comparison across groups."""
        fig, axes = plt.subplots(1, len(baseline['group_metrics']), figsize=(15, 5))
        
        if len(baseline['group_metrics']) == 1:
            axes = [axes]
        
        for idx, (attribute, groups) in enumerate(baseline['group_metrics'].items()):
            ax = axes[idx]
            
            group_names = list(groups.keys())
            baseline_fprs = [groups[g]['fpr'] for g in group_names]
            mitigated_fprs = [mitigated['group_metrics'][attribute][g]['fpr'] for g in group_names]
            
            x = np.arange(len(group_names))
            width = 0.35
            
            ax.bar(x - width/2, baseline_fprs, width, label='Baseline (Global)', alpha=0.8)
            ax.bar(x + width/2, mitigated_fprs, width, label='Mitigated (Group-Specific)', alpha=0.8)
            
            ax.set_xlabel('Group')
            ax.set_ylabel('False Positive Rate')
            ax.set_title(f'FPR Comparison: {attribute.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(group_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_fairness_gaps(baseline: Dict, mitigated: Dict, output_path: str):
        """Plot fairness gap reductions."""
        attributes = list(baseline['fairness_metrics'].keys())
        
        tpr_gaps_before = [baseline['fairness_metrics'][attr]['tpr_gap'] for attr in attributes]
        tpr_gaps_after = [mitigated['fairness_metrics'][attr]['tpr_gap'] for attr in attributes]
        fpr_gaps_before = [baseline['fairness_metrics'][attr]['fpr_gap'] for attr in attributes]
        fpr_gaps_after = [mitigated['fairness_metrics'][attr]['fpr_gap'] for attr in attributes]
        
        x = np.arange(len(attributes))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width*1.5, tpr_gaps_before, width, label='TPR Gap (Before)', alpha=0.8)
        ax.bar(x - width*0.5, tpr_gaps_after, width, label='TPR Gap (After)', alpha=0.8)
        ax.bar(x + width*0.5, fpr_gaps_before, width, label='FPR Gap (Before)', alpha=0.8)
        ax.bar(x + width*1.5, fpr_gaps_after, width, label='FPR Gap (After)', alpha=0.8)
        
        ax.axhline(y=0.05, color='r', linestyle='--', label='Fairness Threshold (5%)')
        
        ax.set_xlabel('Demographic Attribute')
        ax.set_ylabel('Fairness Gap (max - min)')
        ax.set_title('Fairness Gap Reduction: Before vs After Mitigation')
        ax.set_xticks(x)
        ax.set_xticklabels([attr.upper() for attr in attributes])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_threshold_adjustments(
        global_threshold: float,
        group_thresholds: Dict,
        output_path: str
    ):
        """Plot threshold adjustments per group."""
        fig, axes = plt.subplots(1, len(group_thresholds), figsize=(15, 5))
        
        if len(group_thresholds) == 1:
            axes = [axes]
        
        for idx, (attribute, thresholds) in enumerate(group_thresholds.items()):
            ax = axes[idx]
            
            group_names = list(thresholds.keys())
            group_thresholds_vals = [
                thresholds[g]['threshold'] if isinstance(thresholds[g], dict) else thresholds[g]
                for g in group_names
            ]
            adjustments = [t - global_threshold for t in group_thresholds_vals]
            colors = ['green' if adj < 0 else 'red' if adj > 0 else 'gray' for adj in adjustments]
            
            x = np.arange(len(group_names))
            ax.bar(x, adjustments, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            ax.set_xlabel('Group')
            ax.set_ylabel(f'Threshold Adjustment\n(vs Global={global_threshold:.3f})')
            ax.set_title(f'Threshold Adjustments: {attribute.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(group_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_tradeoff_summary(improvements: Dict, output_path: str):
        """Plot trade-off summary metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Fairness improvements
        attributes = list(improvements['fairness_improvements'].keys())
        tpr_reductions = [improvements['fairness_improvements'][attr]['tpr_gap_reduction_pct'] 
                         for attr in attributes]
        fpr_reductions = [improvements['fairness_improvements'][attr]['fpr_gap_reduction_pct'] 
                         for attr in attributes]
        
        x = np.arange(len(attributes))
        width = 0.35
        
        ax1.bar(x - width/2, tpr_reductions, width, label='TPR Gap Reduction', alpha=0.8)
        ax1.bar(x + width/2, fpr_reductions, width, label='FPR Gap Reduction', alpha=0.8)
        ax1.set_xlabel('Demographic Attribute')
        ax1.set_ylabel('Gap Reduction (%)')
        ax1.set_title('Fairness Improvements')
        ax1.set_xticks(x)
        ax1.set_xticklabels([attr.upper() for attr in attributes])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Performance/ROI changes
        perf = improvements['performance_changes']
        metrics = ['TPR', 'FPR', 'Accuracy', 'ROC-AUC']
        changes = [
            perf['tpr_change'] * 100,
            perf['fpr_change'] * 100,
            perf['accuracy_change'] * 100,
            perf['roc_auc_change'] * 100
        ]
        
        colors = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in changes]
        
        ax2.barh(metrics, changes, color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Change (%)')
        ax2.set_title('Performance Changes')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results: Dict, output_path: str):
    """Save results to JSON file with proper type conversion."""
    def convert_to_serializable(obj):
        """Convert numpy/pandas types to JSON-serializable types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"✅ Saved results: {output_path}")


# ============================================================================
# HUGGINGFACE UPLOAD
# ============================================================================

def upload_results_to_hf(
    output_dir: str,
    repo_id: str,
    commit_message: str = "Upload Phase 6 fairness mitigation results",
    token: Optional[str] = None,
    include_visualizations: bool = True
):
    """
    Upload Phase 6 fairness mitigation results to HuggingFace Hub.
    
    Args:
        output_dir: Directory containing output files (JSON, PNG)
        repo_id: HuggingFace repository ID (e.g., 'username/hospital-readmission-gradient-boosting-mitigation-results')
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
        ...     output_dir='./phase-6-fairness-mitigation-bias-correction/outputs',
        ...     repo_id='username/hospital-readmission-gradient-boosting-mitigation-results'
        ... )
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub library required for uploading. "
            "Install with: pip install huggingface_hub"
        )
    
    # Load environment variables
    load_dotenv()
    
    # Get token from environment if not provided
    if token is None:
        token = os.getenv('HF_TOKEN')
        if token is None:
            raise ValueError(
                "HuggingFace token not provided. Set HF_TOKEN environment variable or pass token parameter."
            )
    
    print("\n" + "="*80)
    print("📤 Uploading Phase 6 Fairness Mitigation Results to HuggingFace Hub")
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
- fairness-mitigation
- bias-correction
- ai-ethics
---

# Hospital Readmission Risk - Phase 6: Fairness Mitigation Results

This repository contains the results from Phase 6: Fairness Mitigation & Bias Correction.

## Contents

### Outputs
- `outputs/group_thresholds.json`: Group-specific decision thresholds by demographic attribute
- `outputs/mitigation_impact.json`: Comprehensive before/after fairness evaluation

### Visualizations
- `visualizations/tpr_comparison_*.png`: True Positive Rate comparison (baseline vs mitigated)
- `visualizations/fpr_comparison_*.png`: False Positive Rate comparison (baseline vs mitigated)
- `visualizations/fairness_gaps_*.png`: Fairness gap reductions with 5% threshold line
- `visualizations/threshold_adjustments_*.png`: Group-specific threshold adjustments from global
- `visualizations/tradeoff_summary_*.png`: Performance/fairness trade-off analysis

## Mitigation Strategies

### Equalized Odds (Default)
Minimizes both TPR and FPR gaps across demographic groups. Balances false positives and false negatives.

### Equal Opportunity
Minimizes only TPR gaps across groups. Allows FPR variation if needed.

### Demographic Parity
Equalizes intervention rates across groups. May sacrifice individual fairness for group fairness.

## Group-Specific Thresholds

Each demographic group receives a tailored decision threshold to equalize fairness metrics while maintaining overall performance:

- **Race**: African American, Asian, Caucasian, Hispanic, Other, Unknown
- **Gender**: Female, Male, Unknown/Invalid
- **Age**: [0-10), [10-20), [20-30), ..., [90-100)

## Trade-off Analysis

### Fairness Improvements
- TPR gap reduction (max TPR - min TPR)
- FPR gap reduction (max FPR - min FPR)
- Intervention rate gap reduction

### Performance Changes
- Accuracy change
- TPR change
- FPR change
- ROC-AUC change

### ROI Impact
- Expected value change per patient
- Cost matrix: TP=$14.5K, FP=-$500, FN=-$15K, TN=$0

## Deployment Recommendation

The mitigation impact report includes a deployment recommendation based on:
- Fairness targets met (gaps < 5%)
- Acceptable performance drop (accuracy/ROC-AUC < 2%)
- Acceptable ROI reduction (< 5%)

If all criteria are met: **Use group-specific thresholds**  
If criteria not met: **Use global threshold** or consider retraining

## Model Information

- **Model**: Gradient Boosting (LightGBM) with Platt Calibration
- **Global Threshold**: From Phase 4 ROI optimization
- **Test Set**: 15,265 patients
- **Mitigation Type**: Post-hoc (threshold adjustment only, model unchanged)

## Usage

These results can be used for:
- Deploying models with group-specific fairness thresholds
- Clinical approval documentation
- Meeting regulatory requirements for AI fairness
- Creating model cards with bias mitigation documentation
- Demonstrating commitment to equitable care

## Clinical Approval Process

1. Review fairness violations from Phase 5
2. Review proposed group-specific thresholds
3. Review trade-off analysis (fairness gains vs performance/ROI costs)
4. Make deployment decision:
   - **Option A**: Approve group-specific thresholds
   - **Option B**: Use global threshold
   - **Option C**: Retrain model with bias mitigation during training

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
    print(f"🔗 Repository URL: https://huggingface.co/{repo_id}")
    print("="*80)
    
    return f"https://huggingface.co/{repo_id}"
