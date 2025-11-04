"""
Utility Functions for Hospital Readmission Risk Prediction
==========================================================

Combined utilities for model evaluation and HuggingFace Hub integration.

Modules:
- ModelEvaluator: Comprehensive model evaluation with clinical metrics
- HuggingFaceUploader: Upload trained models to HuggingFace Hub

Author: Hospital Readmission Risk Team
Date: November 2025
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from huggingface_hub import HfApi, create_repo
import warnings

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file automatically
except ImportError:
    pass  # dotenv not installed, will rely on system environment variables

warnings.filterwarnings('ignore')


# ============================================================================
# MODEL EVALUATION UTILITIES
# ============================================================================

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


# ============================================================================
# HUGGINGFACE HUB INTEGRATION
# ============================================================================

class HuggingFaceUploader:
    """
    Upload trained models to HuggingFace Hub with comprehensive documentation.
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize HuggingFace uploader.
        
        Args:
            hf_token: HuggingFace API token. If None, will use token from environment
                     variable HF_TOKEN or from cached credentials.
        """
        # If no token provided, try to get from environment variable
        if hf_token is None:
            hf_token = os.environ.get('HF_TOKEN')
            if hf_token:
                print(f"✅ Using HF_TOKEN from environment variable")
        
        self.api = HfApi(token=hf_token)
        self.token = hf_token
    
    @staticmethod
    def is_token_available() -> bool:
        """Check if HuggingFace token is available in environment."""
        return os.environ.get('HF_TOKEN') is not None
        
    def create_model_card(self, 
                         model_name: str,
                         model_type: str,
                         metrics: Dict[str, Dict[str, float]],
                         hyperparameters: Dict[str, Any],
                         feature_names: list,
                         description: str = "") -> str:
        """
        Create a comprehensive model card in Markdown format.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., "Logistic Regression", "Random Forest")
            metrics: Dictionary with train/val/test metrics
            hyperparameters: Model hyperparameters
            feature_names: List of feature names
            description: Additional description text
            
        Returns:
            Model card content as Markdown string
        """
        card = f"""---
language: en
license: mit
tags:
- healthcare
- hospital-readmission
- risk-prediction
- scikit-learn
- {model_type.lower().replace(' ', '-')}
datasets:
- diabetic-data
metrics:
- accuracy
- precision
- recall
- f1
- auc
widget:
- text: "Patient risk assessment for hospital readmission"
---

# {model_name}: Hospital Readmission Risk Prediction

## Model Description

This model predicts the risk of hospital readmission for diabetic patients. It is trained on the UCI Diabetes 130-US Hospitals dataset (1999-2008).

**Model Type:** {model_type}

**Task:** Binary Classification (Readmitted within 30 days: Yes/No)

**Framework:** scikit-learn

{description}

## Intended Use

### Primary Use Cases
- **Clinical Decision Support**: Assist healthcare providers in identifying high-risk patients
- **Resource Allocation**: Help hospitals allocate resources for post-discharge care
- **Care Planning**: Support development of personalized care plans for at-risk patients

### Out-of-Scope Use Cases
- This model should NOT be used as the sole basis for clinical decisions
- Not intended for use outside the diabetic patient population
- Not validated for non-US healthcare systems

## Training Data

**Dataset:** UCI Diabetes 130-US Hospitals dataset
- **Time Period:** 1999-2008
- **Patients:** 100,000+ hospital encounters
- **Features:** Demographics, diagnoses, medications, lab procedures

**Data Split:**
- Training: 70% (simulating 1999-2005)
- Validation: 15% (simulating 2006-2007)
- Test: 15% (simulating 2008)

## Model Performance

### Test Set Metrics
"""
        
        # Add test metrics
        if 'test' in metrics:
            test_metrics = metrics['test']
            card += f"""
| Metric | Value |
|--------|-------|
| AUC-ROC | {test_metrics.get('auc_roc', 'N/A'):.4f} |
| Accuracy | {test_metrics.get('accuracy', 'N/A'):.4f} |
| Precision | {test_metrics.get('precision', 'N/A'):.4f} |
| Recall (Sensitivity) | {test_metrics.get('recall', 'N/A'):.4f} |
| Specificity | {test_metrics.get('specificity', 'N/A'):.4f} |
| F1 Score | {test_metrics.get('f1_score', 'N/A'):.4f} |
| NPV | {test_metrics.get('npv', 'N/A'):.4f} |
"""
        
        # Add training and validation metrics
        card += "\n### Training and Validation Performance\n\n"
        card += "| Set | AUC-ROC | Accuracy | F1 Score |\n"
        card += "|-----|---------|----------|----------|\n"
        
        for set_name in ['train', 'val', 'test']:
            if set_name in metrics:
                m = metrics[set_name]
                card += f"| {set_name.capitalize()} | {m.get('auc_roc', 'N/A'):.4f} | {m.get('accuracy', 'N/A'):.4f} | {m.get('f1_score', 'N/A'):.4f} |\n"
        
        # Add hyperparameters
        card += "\n## Hyperparameters\n\n"
        card += "```python\n"
        for param, value in hyperparameters.items():
            card += f"{param}: {value}\n"
        card += "```\n"
        
        # Add feature information
        card += f"\n## Model Features\n\n"
        card += f"**Number of Features:** {len(feature_names)}\n\n"
        card += "**Feature Categories:**\n"
        card += "- Demographics (age, gender, race)\n"
        card += "- Admission information (type, source)\n"
        card += "- Clinical measurements (HbA1c, glucose)\n"
        card += "- Diagnoses (primary, secondary, additional)\n"
        card += "- Medications (count, changes)\n"
        card += "- Procedures (lab procedures, medications prescribed)\n"
        card += "- Prior utilization (emergency visits, inpatient visits, outpatient visits)\n"
        
        # Add usage example
        card += """
## How to Use

```python
import pickle
import pandas as pd
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/hospital-readmission-risk",
    filename="model.pkl"
)

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Make predictions
# X should be a DataFrame with the same features used during training
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

## Limitations and Biases

### Known Limitations
1. **Temporal Drift**: Model trained on 1999-2008 data may not reflect current healthcare practices
2. **Dataset Biases**: Performance may vary across demographic groups
3. **Missing Data**: Model performance depends on data quality and completeness
4. **Generalization**: Validated only on diabetic patients from US hospitals

### Ethical Considerations
- Model predictions should be used as decision support, not as sole decision-maker
- Regular monitoring for performance degradation recommended
- Consider fairness metrics across protected groups
- Ensure transparency in clinical deployment

## Training Procedure

### Preprocessing
1. Missing value imputation
2. Categorical encoding (one-hot, label encoding)
3. Feature scaling (standardization)
4. Class imbalance handling (class weights)

### Training
- **Cross-Validation:** 5-fold stratified
- **Optimization:** Grid search with AUC-ROC metric
- **Validation Strategy:** Temporal split to simulate real-world deployment

## Citation

If you use this model, please cite:

```bibtex
@misc{hospital_readmission_risk_2025,
  title={Hospital Readmission Risk Prediction Model},
  author={Hospital Readmission Risk Team},
  year={2025},
  publisher={HuggingFace},
  howpublished={\\url{https://huggingface.co/YOUR_USERNAME/hospital-readmission-risk}}
}
```

## Contact

For questions or issues, please open an issue in the model repository.

## License

MIT License

**Disclaimer:** This model is for research and educational purposes only. Always consult with qualified healthcare professionals for medical decisions.
"""
        
        return card
    
    def upload_model(self,
                    model_path: str,
                    metadata_path: str,
                    repo_id: str,
                    metrics: Dict[str, Dict[str, float]],
                    model_type: str,
                    description: str = "",
                    private: bool = False,
                    commit_message: str = "Upload trained model") -> str:
        """
        Upload model to HuggingFace Hub.
        
        Args:
            model_path: Path to the pickled model file
            metadata_path: Path to the metadata pickle file
            repo_id: Repository ID (username/repo-name)
            metrics: Dictionary with train/val/test metrics
            model_type: Type of model (e.g., "Logistic Regression", "Random Forest")
            description: Additional description for the model card
            private: Whether to make the repository private
            commit_message: Commit message for the upload
            
        Returns:
            URL to the uploaded model repository
        """
        print(f"\n{'='*70}")
        print(f"Uploading Model to HuggingFace Hub")
        print(f"{'='*70}")
        print(f"Repository: {repo_id}")
        print(f"Model Type: {model_type}")
        print(f"Private: {private}")
        
        try:
            # Create repository if it doesn't exist
            print(f"\n🔄 Creating repository (if not exists)...")
            create_repo(
                repo_id=repo_id,
                token=self.token,
                private=private,
                exist_ok=True,
                repo_type="model"
            )
            print(f"✅ Repository ready: https://huggingface.co/{repo_id}")
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Create model card
            print(f"\n📝 Creating model card...")
            model_card = self.create_model_card(
                model_name=repo_id.split('/')[-1],
                model_type=model_type,
                metrics=metrics,
                hyperparameters=metadata.get('best_params', {}),
                feature_names=metadata.get('feature_names', []),
                description=description
            )
            
            # Save model card temporarily
            card_path = "README.md"
            with open(card_path, 'w', encoding='utf-8') as f:
                f.write(model_card)
            
            # Upload model file
            print(f"\n⬆️  Uploading model file...")
            self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="model.pkl",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message
            )
            print(f"✅ Model file uploaded")
            
            # Upload metadata
            print(f"\n⬆️  Uploading metadata...")
            self.api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo="metadata.pkl",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload model metadata"
            )
            print(f"✅ Metadata uploaded")
            
            # Upload README
            print(f"\n⬆️  Uploading model card...")
            self.api.upload_file(
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload model card"
            )
            print(f"✅ Model card uploaded")
            
            # Clean up temporary README
            if os.path.exists(card_path):
                os.remove(card_path)
            
            repo_url = f"https://huggingface.co/{repo_id}"
            print(f"\n{'='*70}")
            print(f"✅ Upload Complete!")
            print(f"{'='*70}")
            print(f"🔗 Model URL: {repo_url}")
            print(f"{'='*70}\n")
            
            return repo_url
            
        except Exception as e:
            print(f"\n❌ Error uploading model: {str(e)}")
            print(f"{'='*70}\n")
            raise
    
    def upload_with_config(self,
                          model_path: str,
                          metadata_path: str,
                          repo_id: str,
                          metrics: Dict[str, Dict[str, float]],
                          model_type: str,
                          config: Dict[str, Any],
                          private: bool = False) -> str:
        """
        Upload model with additional configuration file.
        
        Args:
            model_path: Path to the pickled model file
            metadata_path: Path to the metadata pickle file
            repo_id: Repository ID (username/repo-name)
            metrics: Dictionary with train/val/test metrics
            model_type: Type of model
            config: Additional configuration dictionary
            private: Whether to make the repository private
            
        Returns:
            URL to the uploaded model repository
        """
        # Upload model first
        repo_url = self.upload_model(
            model_path=model_path,
            metadata_path=metadata_path,
            repo_id=repo_id,
            metrics=metrics,
            model_type=model_type,
            private=private
        )
        
        # Upload config
        print(f"\n⬆️  Uploading configuration...")
        config_path = "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload model configuration"
        )
        
        # Clean up
        if os.path.exists(config_path):
            os.remove(config_path)
        
        print(f"✅ Configuration uploaded")
        
        return repo_url
