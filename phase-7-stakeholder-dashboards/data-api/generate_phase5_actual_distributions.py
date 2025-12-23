"""
Generate Predicted Probability Distributions by Demographic
============================================================
Uses local models to generate actual predicted probabilities and plots distributions.
"""

import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import hf_hub_download
from pathlib import Path

# Model paths
MODEL_PATHS = {
    "gradient_boosting": "models/gradient_boosting_model.joblib",
    "random_forest": "models/random_forest_model_rebuilt.joblib",
    "logistic_regression": "models/logistic_regression_model.joblib"
}

MODEL_NAMES = {
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression"
}

COLORS = {
    "gradient_boosting": "#1f77b4",
    "random_forest": "#ff7f0e",
    "logistic_regression": "#2ca02c"
}

# Demographic group colors
DEMO_COLORS = {
    # Race
    "AfricanAmerican": "#e74c3c",
    "Caucasian": "#3498db",
    "Hispanic": "#2ecc71",
    "Other": "#f39c12",
    "Asian": "#9b59b6",
    # Gender
    "Male": "#3498db",
    "Female": "#e74c3c",
    # Age
    "[0-10)": "#1abc9c",
    "[10-20)": "#16a085",
    "[20-30)": "#2ecc71",
    "[30-40)": "#27ae60",
    "[40-50)": "#3498db",
    "[50-60)": "#2980b9",
    "[60-70)": "#9b59b6",
    "[70-80)": "#8e44ad",
    "[80-90)": "#e74c3c",
    "[90-100)": "#c0392b"
}


def load_test_data_with_demographics():
    """Load test data with demographic attributes."""
    # Download processed test data
    test_file = hf_hub_download(
        repo_id="auphong2707/hospital-readmission-risk-data",
        filename="splits/test.csv",
        repo_type="dataset"
    )
    
    # Download test demographics
    demo_file = hf_hub_download(
        repo_id="auphong2707/hospital-readmission-risk-data",
        filename="splits/test_demographics.csv",
        repo_type="dataset"
    )
    
    test_df = pd.read_csv(test_file)
    demo_df = pd.read_csv(demo_file)
    
    # Split features and target
    target_col = 'target' if 'target' in test_df.columns else 'readmitted'
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Merge features, target, and demographics
    merged = pd.concat([X_test, y_test.rename('target'), demo_df], axis=1)
    
    return merged


def load_model_and_predict(model_key, X_test):
    """Load model and generate predictions."""
    model_path = Path(MODEL_PATHS[model_key])
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    print(f"Loading {MODEL_NAMES[model_key]} model...")
    model = joblib.load(model_path)
    
    # Get predicted probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (readmission)
    else:
        # For calibrated models or pipelines
        probs = model.predict(X_test)
    
    return probs


def generate_risk_distribution_by_demographic(demographic='race'):
    """
    Generate actual predicted probability distributions by demographic group.
    
    Args:
        demographic: 'race', 'gender', or 'age'
    """
    print(f"\n{'='*60}")
    print(f"Generating Risk Distribution for: {demographic.upper()}")
    print(f"{'='*60}")
    
    # Load test data with demographics
    print("\n1. Loading test data with demographics...")
    test_df = load_test_data_with_demographics()
    
    # Separate features and target
    feature_cols = [col for col in test_df.columns 
                   if col not in ['target', 'readmitted', 'race', 'gender', 'age', 'encounter_id', 'patient_nbr']]
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Demographic groups in {demographic}: {test_df[demographic].value_counts().to_dict()}")
    
    # Create subplots for 3 models
    models = ["gradient_boosting", "random_forest", "logistic_regression"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[MODEL_NAMES[m] for m in models],
        horizontal_spacing=0.1
    )
    
    for idx, model_key in enumerate(models, start=1):
        print(f"\n2. Processing {MODEL_NAMES[model_key]}...")
        
        # Get predictions
        probs = load_model_and_predict(model_key, X_test)
        
        if probs is None:
            print(f"   Skipping {MODEL_NAMES[model_key]} (model not found)")
            continue
        
        # Add predictions to dataframe
        test_df[f'{model_key}_prob'] = probs
        
        # Get unique groups for this demographic
        groups = sorted(test_df[demographic].unique())
        
        # Plot distribution for each group
        for group in groups:
            group_data = test_df[test_df[demographic] == group]
            group_probs = group_data[f'{model_key}_prob']
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=group_probs,
                    name=str(group),
                    nbinsx=50,
                    opacity=0.6,
                    marker_color=DEMO_COLORS.get(str(group), None),
                    showlegend=(idx == 1),  # Only show legend for first subplot
                    legendgroup=str(group),
                    hovertemplate=f'<b>{group}</b><br>Risk: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>'
                ),
                row=1, col=idx
            )
            
            print(f"   {group}: {len(group_probs)} samples, mean risk={group_probs.mean():.3f}")
    
    # Update layout
    fig.update_xaxes(title_text="Predicted Risk Score", range=[0, 1])
    fig.update_yaxes(title_text="Count")
    
    fig.update_layout(
        title_text=f"<b>Predicted Risk Score Distribution by {demographic.title()}</b><br><sub>Actual model predictions on test set</sub>",
        title_x=0.5,
        height=600,
        barmode='overlay',
        font=dict(size=12),
        margin=dict(t=100, b=80),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05,
            title=dict(text=demographic.title())
        )
    )
    
    # Save
    output_file = f"phase5_risk_distribution_{demographic}_actual.html"
    fig.write_html(output_file)
    print(f"\n✅ Saved: {output_file}")
    
    return fig


if __name__ == "__main__":
    print("Generating actual predicted risk distributions from local models...")
    print("This will use the downloaded models to generate predictions on test data.\n")
    
    # Generate for each demographic
    for demographic in ["race", "gender", "age"]:
        try:
            generate_risk_distribution_by_demographic(demographic)
        except Exception as e:
            print(f"\n❌ Error generating {demographic} distribution: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ All risk distributions generated!")
    print("="*60)
    print("\nFiles created:")
    print("  - phase5_risk_distribution_race_actual.html")
    print("  - phase5_risk_distribution_gender_actual.html")
    print("  - phase5_risk_distribution_age_actual.html")
