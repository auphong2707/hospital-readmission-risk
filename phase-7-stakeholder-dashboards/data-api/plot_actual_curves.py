"""
Plot combined ROC and PR curves using actual model data.
Uses real curve data from retrained models instead of synthetic curves.
"""

import matplotlib.pyplot as plt
import json
from pathlib import Path

# Model information
MODELS = {
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "color": "#1f77b4"  # Blue
    },
    "random_forest": {
        "label": "Random Forest",
        "color": "#ff7f0e"  # Orange
    },
    "logistic_regression": {
        "label": "Logistic Regression",
        "color": "#2ca02c"  # Green
    }
}

def load_curve_data():
    """Load actual curve data from JSON file."""
    data_path = Path(__file__).parent / "actual_curve_data.json"
    with open(data_path, 'r') as f:
        return json.load(f)

def plot_combined_roc_curves():
    """Generate combined ROC curve plot using actual data."""
    curve_data = load_curve_data()
    
    plt.figure(figsize=(10, 8))
    
    for model_key, model_info in MODELS.items():
        print(f"Plotting {model_info['label']}...")
        
        # Get actual curve data
        fpr = curve_data[model_key]['roc']['fpr']
        tpr = curve_data[model_key]['roc']['tpr']
        
        # Plot
        plt.plot(fpr, tpr, color=model_info['color'], linewidth=2.5, 
                label=model_info['label'])
    
    # Formatting
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / "roc_curves_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved ROC curves to {output_path}")
    plt.close()

def plot_combined_pr_curves():
    """Generate combined PR curve plot using actual data."""
    curve_data = load_curve_data()
    
    plt.figure(figsize=(10, 8))
    
    for model_key, model_info in MODELS.items():
        print(f"Plotting {model_info['label']}...")
        
        # Get actual curve data
        recall = curve_data[model_key]['pr']['recall']
        precision = curve_data[model_key]['pr']['precision']
        
        # Plot
        plt.plot(recall, precision, color=model_info['color'], linewidth=2.5,
                label=model_info['label'])
    
    # Formatting
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / "pr_curves_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved PR curves to {output_path}")
    plt.close()

if __name__ == "__main__":
    print("Generating combined ROC curves from actual data...")
    plot_combined_roc_curves()
    
    print("\nGenerating combined PR curves from actual data...")
    plot_combined_pr_curves()
    
    print("\nDone! Generated combined curve plots from actual model data.")
