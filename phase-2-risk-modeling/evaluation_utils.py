"""
Model Evaluation Utilities for Hospital Readmission Risk Prediction
===================================================================

Comprehensive evaluation framework including:
- Primary metrics: AUC-ROC, Precision, Recall, F1-Score
- Clinical metrics: Sensitivity, Specificity, PPV, NPV
- Calibration assessment: Reliability plots, Brier score
- Feature importance visualization
- Performance comparison utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation for hospital readmission prediction.
    
    Provides clinical and ML metrics with visualization capabilities.
    """
    
    def __init__(self, model_name="Model"):
        self.model_name = model_name
        self.metrics = {}
        
    def calculate_clinical_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate comprehensive clinical and ML metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUC-ROC)
            
        Returns:
            Dictionary of metrics
        """
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Primary ML metrics
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # AUC-ROC (requires probabilities)
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        
        # Clinical metrics
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (same as precision)
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Confusion matrix components
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        # Accuracy
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        
        self.metrics = metrics
        return metrics
    
    def print_metrics(self, metrics=None):
        """Print metrics in a formatted way."""
        if metrics is None:
            metrics = self.metrics
        
        print(f"\n{'='*70}")
        print(f"{self.model_name} - Performance Metrics")
        print(f"{'='*70}")
        
        print(f"\n{'Primary ML Metrics:':<30}")
        print(f"  {'AUC-ROC:':<28} {metrics.get('auc_roc', 'N/A'):.4f}" if 'auc_roc' in metrics else "  AUC-ROC: N/A")
        print(f"  {'Precision:':<28} {metrics['precision']:.4f}")
        print(f"  {'Recall:':<28} {metrics['recall']:.4f}")
        print(f"  {'F1-Score:':<28} {metrics['f1_score']:.4f}")
        print(f"  {'Accuracy:':<28} {metrics['accuracy']:.4f}")
        
        print(f"\n{'Clinical Metrics:':<30}")
        print(f"  {'Sensitivity (TPR):':<28} {metrics['sensitivity']:.4f}")
        print(f"  {'Specificity (TNR):':<28} {metrics['specificity']:.4f}")
        print(f"  {'PPV (Precision):':<28} {metrics['ppv']:.4f}")
        print(f"  {'NPV:':<28} {metrics['npv']:.4f}")
        
        if 'brier_score' in metrics:
            print(f"\n{'Calibration:':<30}")
            print(f"  {'Brier Score:':<28} {metrics['brier_score']:.4f}")
        
        print(f"\n{'Confusion Matrix:':<30}")
        print(f"  {'True Positives:':<28} {metrics['true_positives']}")
        print(f"  {'True Negatives:':<28} {metrics['true_negatives']}")
        print(f"  {'False Positives:':<28} {metrics['false_positives']}")
        print(f"  {'False Negatives:':<28} {metrics['false_negatives']}")
        
        print(f"{'='*70}\n")
    
    def plot_roc_curve(self, y_true, y_prob, save_path=None):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'{self.model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        plt.close()
        
        return fpr, tpr, thresholds
    
    def plot_precision_recall_curve(self, y_true, y_prob, save_path=None):
        """Plot Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'{self.model_name}')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to: {save_path}")
        plt.close()
        
        return precision, recall, thresholds
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['No Readmission', 'Readmission'],
                    yticklabels=['No Readmission', 'Readmission'])
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_calibration_curve(self, y_true, y_prob, n_bins=10, save_path=None):
        """
        Plot calibration curve (reliability diagram).
        
        Shows how well predicted probabilities match actual outcomes.
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        plt.figure(figsize=(10, 8))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
                 linewidth=2, label=f'{self.model_name}')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title(f'Calibration Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, feature_names, feature_importance, 
                                top_n=20, save_path=None):
        """
        Plot feature importance for interpretability.
        
        Args:
            feature_names: List of feature names
            feature_importance: Array of feature importance values
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(feature_importance)  # Use absolute values
        }).sort_values('Importance', ascending=False)
        
        # Select top N features
        top_features = importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance (Absolute Coefficient Value)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}', 
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        plt.close()
        
        return importance_df
    
    def generate_full_report(self, y_true, y_pred, y_prob, feature_names=None, 
                            feature_importance=None, output_dir=None):
        """
        Generate a comprehensive evaluation report with all plots and metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            feature_names: List of feature names (optional)
            feature_importance: Feature importance values (optional)
            output_dir: Directory to save plots (optional)
        """
        print(f"\n{'='*70}")
        print(f"Generating Comprehensive Evaluation Report for {self.model_name}")
        print(f"{'='*70}\n")
        
        # Calculate and print metrics
        metrics = self.calculate_clinical_metrics(y_true, y_pred, y_prob)
        self.print_metrics(metrics)
        
        # Generate plots
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # ROC curve
            self.plot_roc_curve(y_true, y_prob, 
                               save_path=f"{output_dir}/{self.model_name}_roc_curve.png")
            
            # Precision-Recall curve
            self.plot_precision_recall_curve(y_true, y_prob,
                                            save_path=f"{output_dir}/{self.model_name}_pr_curve.png")
            
            # Confusion matrix
            self.plot_confusion_matrix(y_true, y_pred,
                                      save_path=f"{output_dir}/{self.model_name}_confusion_matrix.png")
            
            # Calibration curve
            self.plot_calibration_curve(y_true, y_prob,
                                       save_path=f"{output_dir}/{self.model_name}_calibration_curve.png")
            
            # Feature importance
            if feature_names is not None and feature_importance is not None:
                importance_df = self.plot_feature_importance(
                    feature_names, feature_importance,
                    save_path=f"{output_dir}/{self.model_name}_feature_importance.png"
                )
                
                # Save feature importance to CSV
                importance_df.to_csv(f"{output_dir}/{self.model_name}_feature_importance.csv", 
                                    index=False)
                print(f"Feature importance data saved to: {output_dir}/{self.model_name}_feature_importance.csv")
        
        print(f"\n{'='*70}")
        print(f"Evaluation Report Complete!")
        print(f"{'='*70}\n")
        
        return metrics


def compare_models(model_results_dict, metric='auc_roc', save_path=None):
    """
    Compare multiple models based on a specific metric.
    
    Args:
        model_results_dict: Dictionary with model names as keys and metrics as values
        metric: Metric to compare (default: 'auc_roc')
        save_path: Path to save the comparison plot
    """
    model_names = list(model_results_dict.keys())
    metric_values = [model_results_dict[name].get(metric, 0) for name in model_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, metric_values, color='steelblue', alpha=0.7)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")
    plt.close()
