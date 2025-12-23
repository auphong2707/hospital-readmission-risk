"""
Visualization Generator for Dashboard API

Generates merged visualizations (ROC curves, PR curves, confusion matrices) for all models.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from pathlib import Path
from typing import Dict, List, Optional
import json

from utilities.data_aggregator import DashboardDataAggregator


class VisualizationGenerator:
    """Generate visualizations comparing all three models."""
    
    def __init__(self):
        """Initialize the visualization generator."""
        self.methods = ['random_forest', 'gradient_boosting', 'logistic_regression']
        self.method_labels = {
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'logistic_regression': 'Logistic Regression'
        }
        self.colors = {
            'random_forest': '#2E7D32',
            'gradient_boosting': '#1976D2',
            'logistic_regression': '#C62828'
        }
        
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['font.size'] = 10
    
    def generate_merged_roc_curves(self) -> io.BytesIO:
        """
        Generate merged ROC curves for all models.
        
        Returns:
            BytesIO buffer containing the PNG image
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
        
        for method in self.methods:
            try:
                aggregator = DashboardDataAggregator(method)
                metrics = aggregator.load_phase2_metrics()
                
                if 'roc_curve' in metrics:
                    roc = metrics['roc_curve']
                    fpr = roc.get('fpr', [])
                    tpr = roc.get('tpr', [])
                    auc = roc.get('auc', 0.0)
                    
                    if fpr and tpr:
                        ax.plot(
                            fpr, 
                            tpr, 
                            color=self.colors[method],
                            lw=2.5,
                            label=f'{self.method_labels[method]} (AUC = {auc:.3f})'
                        )
            except Exception as e:
                print(f"Error loading ROC data for {method}: {e}")
                continue
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)
        
        return buffer
    
    def generate_merged_pr_curves(self) -> io.BytesIO:
        """
        Generate merged Precision-Recall curves for all models.
        
        Returns:
            BytesIO buffer containing the PNG image
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for method in self.methods:
            try:
                aggregator = DashboardDataAggregator(method)
                metrics = aggregator.load_phase2_metrics()
                
                if 'pr_curve' in metrics:
                    pr = metrics['pr_curve']
                    precision = pr.get('precision', [])
                    recall = pr.get('recall', [])
                    avg_precision = pr.get('average_precision', 0.0)
                    
                    if precision and recall:
                        ax.plot(
                            recall, 
                            precision, 
                            color=self.colors[method],
                            lw=2.5,
                            label=f'{self.method_labels[method]} (AP = {avg_precision:.3f})'
                        )
            except Exception as e:
                print(f"Error loading PR data for {method}: {e}")
                continue
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)
        
        return buffer
    
    def generate_confusion_matrices_grid(self) -> io.BytesIO:
        """
        Generate confusion matrices for all models in a 3x1 grid.
        
        Returns:
            BytesIO buffer containing the PNG image
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, method in enumerate(self.methods):
            ax = axes[idx]
            
            try:
                aggregator = DashboardDataAggregator(method)
                metrics = aggregator.load_phase2_metrics()
                
                if 'confusion_matrix' in metrics:
                    cm = np.array(metrics['confusion_matrix'])
                    
                    # Create heatmap
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        cbar=True,
                        ax=ax,
                        square=True,
                        linewidths=0.5,
                        linecolor='gray'
                    )
                    
                    ax.set_title(self.method_labels[method], fontsize=12, fontweight='bold')
                    ax.set_xlabel('Predicted Label', fontsize=10)
                    ax.set_ylabel('True Label', fontsize=10)
                    ax.set_xticklabels(['No Readmit', 'Readmit'])
                    ax.set_yticklabels(['No Readmit', 'Readmit'])
                else:
                    ax.text(0.5, 0.5, 'No Data Available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(self.method_labels[method], fontsize=12, fontweight='bold')
                    
            except Exception as e:
                print(f"Error loading confusion matrix for {method}: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(self.method_labels[method], fontsize=12, fontweight='bold')
        
        plt.suptitle('Confusion Matrices - Model Comparison', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)
        
        return buffer


# Singleton instance
_generator_instance = None


def get_generator() -> VisualizationGenerator:
    """
    Get or create the singleton visualization generator instance.
    
    Returns:
        VisualizationGenerator instance
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = VisualizationGenerator()
    return _generator_instance
