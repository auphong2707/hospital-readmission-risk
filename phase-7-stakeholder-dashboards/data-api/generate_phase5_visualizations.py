"""
Phase 5 Plotly Visualization Generator
=======================================
Generates interactive Plotly visualizations for fairness assessment.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from huggingface_hub import hf_hub_download
import json
from typing import Dict, Any

# Repository mapping
REPOS = {
    "gradient_boosting": "auphong2707/hospital-readmission-gradient-boosting-final",
    "random_forest": "auphong2707/hospital-readmission-rf-final",
    "logistic_regression": "auphong2707/hospital-readmission-lr-final"
}

MODEL_DISPLAY_NAMES = {
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression"
}

COLORS = {
    "gradient_boosting": "#1f77b4",  # Blue
    "random_forest": "#ff7f0e",      # Orange
    "logistic_regression": "#2ca02c" # Green
}


def load_group_metrics(model: str, demographic: str) -> pd.DataFrame:
    """Load group-level metrics CSV."""
    file_path = hf_hub_download(
        repo_id=REPOS[model],
        filename=f"phase5_fairness_assessment/evaluation/group_metrics_{demographic}.csv",
        repo_type="model"
    )
    return pd.read_csv(file_path)


def load_mitigation_impact(model: str) -> Dict[str, Any]:
    """Load mitigation impact JSON."""
    file_path = hf_hub_download(
        repo_id=REPOS[model],
        filename="phase5_fairness_assessment/mitigation/mitigation_impact.json",
        repo_type="model"
    )
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_risk_distribution_plot(demographic: str = "race") -> go.Figure:
    """
    Generate risk distribution plot for all 3 models.
    Shows intervention rates by demographic group (as proxy for risk distribution).
    
    3 subplots (1 row, 3 columns) - one per model.
    """
    models = ["gradient_boosting", "random_forest", "logistic_regression"]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[MODEL_DISPLAY_NAMES[m] for m in models],
        horizontal_spacing=0.1
    )
    
    for idx, model in enumerate(models, start=1):
        try:
            df = load_group_metrics(model, demographic)
            
            # Use intervention_rate as proxy for risk (higher rate = higher predicted risk)
            groups = df['group'].tolist()
            intervention_rates = (df['intervention_rate'] * 100).tolist()
            n_samples = df['n_samples'].tolist()
            
            # Create bar chart
            fig.add_trace(
                go.Bar(
                    x=groups,
                    y=intervention_rates,
                    name=MODEL_DISPLAY_NAMES[model],
                    marker_color=COLORS[model],
                    text=[f"{rate:.1f}%<br>N={n}" for rate, n in zip(intervention_rates, n_samples)],
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>Intervention Rate: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=idx
            )
            
            # Add horizontal line at average
            avg_rate = sum(intervention_rates) / len(intervention_rates)
            fig.add_hline(
                y=avg_rate,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                row=1, col=idx,
                annotation_text=f"Avg: {avg_rate:.1f}%",
                annotation_position="top right"
            )
            
        except Exception as e:
            print(f"Error loading {model}: {e}")
    
    # Update layout
    fig.update_xaxes(title_text="Group", tickangle=-45)
    fig.update_yaxes(title_text="Intervention Rate (%)", range=[0, 100])
    
    fig.update_layout(
        title_text=f"<b>Risk Distribution by {demographic.title()}</b><br><sub>Higher intervention rate indicates higher predicted risk</sub>",
        title_x=0.5,
        height=500,
        showlegend=False,
        font=dict(size=12),
        margin=dict(t=100, b=100)
    )
    
    return fig


def generate_fairness_gaps_plot() -> go.Figure:
    """
    Generate fairness gaps (before/after mitigation) plot for all 3 models.
    Shows TPR and FPR gaps for race, gender, age.
    
    3 subplots (1 row, 3 columns) - one per model.
    Each subplot shows grouped bar chart: TPR gap and FPR gap, before vs after.
    """
    models = ["gradient_boosting", "random_forest", "logistic_regression"]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[MODEL_DISPLAY_NAMES[m] for m in models],
        horizontal_spacing=0.12
    )
    
    for idx, model in enumerate(models, start=1):
        try:
            impact = load_mitigation_impact(model)
            fairness_improvements = impact["improvements"]["fairness_improvements"]
            
            # Prepare data for grouped bar chart
            attributes = ['Race', 'Gender', 'Age']
            
            # TPR gaps (before and after)
            tpr_before = [
                fairness_improvements["race"]["tpr_gap_before"] * 100,
                fairness_improvements["gender"]["tpr_gap_before"] * 100,
                fairness_improvements["age"]["tpr_gap_before"] * 100
            ]
            tpr_after = [
                fairness_improvements["race"]["tpr_gap_after"] * 100,
                fairness_improvements["gender"]["tpr_gap_after"] * 100,
                fairness_improvements["age"]["tpr_gap_after"] * 100
            ]
            
            # FPR gaps (before and after)
            fpr_before = [
                fairness_improvements["race"]["fpr_gap_before"] * 100,
                fairness_improvements["gender"]["fpr_gap_before"] * 100,
                fairness_improvements["age"]["fpr_gap_before"] * 100
            ]
            fpr_after = [
                fairness_improvements["race"]["fpr_gap_after"] * 100,
                fairness_improvements["gender"]["fpr_gap_after"] * 100,
                fairness_improvements["age"]["fpr_gap_after"] * 100
            ]
            
            # Add TPR gap bars
            fig.add_trace(
                go.Bar(
                    name='TPR Gap (Before)',
                    x=attributes,
                    y=tpr_before,
                    marker_color='lightcoral',
                    text=[f"{v:.1f}%" for v in tpr_before],
                    textposition='outside',
                    showlegend=(idx == 1),  # Show legend only for first subplot
                    legendgroup='tpr_before',
                    hovertemplate='<b>%{x}</b><br>TPR Gap Before: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=idx
            )
            
            fig.add_trace(
                go.Bar(
                    name='TPR Gap (After)',
                    x=attributes,
                    y=tpr_after,
                    marker_color='darkred',
                    text=[f"{v:.1f}%" for v in tpr_after],
                    textposition='outside',
                    showlegend=(idx == 1),
                    legendgroup='tpr_after',
                    hovertemplate='<b>%{x}</b><br>TPR Gap After: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=idx
            )
            
            # Add threshold line at 5%
            fig.add_hline(
                y=5,
                line_dash="dash",
                line_color="red",
                opacity=0.6,
                row=1, col=idx,
                annotation_text="5% Threshold" if idx == 3 else "",
                annotation_position="top right"
            )
            
        except Exception as e:
            print(f"Error loading {model}: {e}")
    
    # Update layout
    fig.update_xaxes(title_text="Demographic Attribute")
    fig.update_yaxes(title_text="TPR Gap (%)", range=[0, None])
    
    fig.update_layout(
        title_text="<b>Fairness Gaps: Before vs After Mitigation</b><br><sub>TPR Gap (lower is better, <5% target)</sub>",
        title_x=0.5,
        height=500,
        barmode='group',
        font=dict(size=12),
        margin=dict(t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def generate_fairness_gaps_fpr_plot() -> go.Figure:
    """
    Generate FPR gaps (before/after mitigation) plot for all 3 models.
    Similar to TPR gaps but for False Positive Rate.
    """
    models = ["gradient_boosting", "random_forest", "logistic_regression"]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[MODEL_DISPLAY_NAMES[m] for m in models],
        horizontal_spacing=0.12
    )
    
    for idx, model in enumerate(models, start=1):
        try:
            impact = load_mitigation_impact(model)
            fairness_improvements = impact["improvements"]["fairness_improvements"]
            
            attributes = ['Race', 'Gender', 'Age']
            
            # FPR gaps (before and after)
            fpr_before = [
                fairness_improvements["race"]["fpr_gap_before"] * 100,
                fairness_improvements["gender"]["fpr_gap_before"] * 100,
                fairness_improvements["age"]["fpr_gap_before"] * 100
            ]
            fpr_after = [
                fairness_improvements["race"]["fpr_gap_after"] * 100,
                fairness_improvements["gender"]["fpr_gap_after"] * 100,
                fairness_improvements["age"]["fpr_gap_after"] * 100
            ]
            
            # Add FPR gap bars
            fig.add_trace(
                go.Bar(
                    name='FPR Gap (Before)',
                    x=attributes,
                    y=fpr_before,
                    marker_color='lightblue',
                    text=[f"{v:.1f}%" for v in fpr_before],
                    textposition='outside',
                    showlegend=(idx == 1),
                    legendgroup='fpr_before',
                    hovertemplate='<b>%{x}</b><br>FPR Gap Before: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=idx
            )
            
            fig.add_trace(
                go.Bar(
                    name='FPR Gap (After)',
                    x=attributes,
                    y=fpr_after,
                    marker_color='darkblue',
                    text=[f"{v:.1f}%" for v in fpr_after],
                    textposition='outside',
                    showlegend=(idx == 1),
                    legendgroup='fpr_after',
                    hovertemplate='<b>%{x}</b><br>FPR Gap After: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=idx
            )
            
            # Add threshold line at 5%
            fig.add_hline(
                y=5,
                line_dash="dash",
                line_color="red",
                opacity=0.6,
                row=1, col=idx,
                annotation_text="5% Threshold" if idx == 3 else "",
                annotation_position="top right"
            )
            
        except Exception as e:
            print(f"Error loading {model}: {e}")
    
    # Update layout
    fig.update_xaxes(title_text="Demographic Attribute")
    fig.update_yaxes(title_text="FPR Gap (%)", range=[0, None])
    
    fig.update_layout(
        title_text="<b>Fairness Gaps: Before vs After Mitigation</b><br><sub>FPR Gap (lower is better, <5% target)</sub>",
        title_x=0.5,
        height=500,
        barmode='group',
        font=dict(size=12),
        margin=dict(t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


if __name__ == "__main__":
    print("Generating Phase 5 visualizations...")
    
    # Generate risk distribution plots for each demographic
    for demographic in ["race", "gender", "age"]:
        print(f"\nGenerating risk distribution plot for {demographic}...")
        fig = generate_risk_distribution_plot(demographic)
        output_file = f"phase5_risk_distribution_{demographic}.html"
        fig.write_html(output_file)
        print(f"Saved: {output_file}")
    
    # Generate fairness gaps plots
    print("\nGenerating TPR fairness gaps plot...")
    fig_tpr = generate_fairness_gaps_plot()
    fig_tpr.write_html("phase5_fairness_gaps_tpr.html")
    print("Saved: phase5_fairness_gaps_tpr.html")
    
    print("\nGenerating FPR fairness gaps plot...")
    fig_fpr = generate_fairness_gaps_fpr_plot()
    fig_fpr.write_html("phase5_fairness_gaps_fpr.html")
    print("Saved: phase5_fairness_gaps_fpr.html")
    
    print("\n✅ All Phase 5 visualizations generated!")
