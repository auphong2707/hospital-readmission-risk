"""
Phase 5: Fairness Assessment & Mitigation API
==============================================
Serves fairness evaluation data and mitigation results for all 3 models.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
import joblib

router = APIRouter(prefix="/api/phase5", tags=["Phase 5 - Fairness"])

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


def load_fairness_report(model: str) -> Dict[str, Any]:
    """Load fairness evaluation report for a model."""
    if model not in REPOS:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")
    
    file_path = hf_hub_download(
        repo_id=REPOS[model],
        filename="phase5_fairness_assessment/evaluation/fairness_report.json",
        repo_type="model"
    )
    
    with open(file_path, 'r') as f:
        return json.load(f)


def load_mitigation_impact(model: str) -> Dict[str, Any]:
    """Load mitigation impact analysis for a model."""
    if model not in REPOS:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")
    
    file_path = hf_hub_download(
        repo_id=REPOS[model],
        filename="phase5_fairness_assessment/mitigation/mitigation_impact.json",
        repo_type="model"
    )
    
    with open(file_path, 'r') as f:
        return json.load(f)


def load_group_metrics(model: str, demographic: str) -> pd.DataFrame:
    """Load group-level metrics for a specific demographic attribute."""
    if model not in REPOS:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")
    
    if demographic not in ["race", "gender", "age"]:
        raise HTTPException(status_code=400, detail="Demographic must be one of: race, gender, age")
    
    file_path = hf_hub_download(
        repo_id=REPOS[model],
        filename=f"phase5_fairness_assessment/evaluation/group_metrics_{demographic}.csv",
        repo_type="model"
    )
    
    return pd.read_csv(file_path)


@router.get("/fairness-summary")
async def get_fairness_summary():
    """
    Get fairness evaluation summary for all models.
    Returns disparity metrics and pass/fail status.
    """
    summary = {}
    
    for model_key, repo in REPOS.items():
        try:
            report = load_fairness_report(model_key)
            
            summary[model_key] = {
                "model_name": MODEL_DISPLAY_NAMES[model_key],
                "overall_status": "PASS" if not report.get("bias_detected", True) else "FAIL",
                "bias_detected": report.get("bias_detected", True),
                "fairness_by_attribute": {
                    "race": {
                        "tpr_gap": report["fairness_evaluation"]["race"]["equal_opportunity"]["gap"],
                        "fpr_gap": report["fairness_evaluation"]["race"]["equalized_odds"]["fpr_gap"],
                        "passed": report["fairness_evaluation"]["race"]["all_passed"]
                    },
                    "gender": {
                        "tpr_gap": report["fairness_evaluation"]["gender"]["equal_opportunity"]["gap"],
                        "fpr_gap": report["fairness_evaluation"]["gender"]["equalized_odds"]["fpr_gap"],
                        "passed": report["fairness_evaluation"]["gender"]["all_passed"]
                    },
                    "age": {
                        "tpr_gap": report["fairness_evaluation"]["age"]["equal_opportunity"]["gap"],
                        "fpr_gap": report["fairness_evaluation"]["age"]["equalized_odds"]["fpr_gap"],
                        "passed": report["fairness_evaluation"]["age"]["all_passed"]
                    }
                }
            }
        except Exception as e:
            summary[model_key] = {"error": str(e)}
    
    return summary


@router.get("/mitigation-summary")
async def get_mitigation_summary():
    """
    Get mitigation impact summary for all models.
    Shows before/after fairness metrics and performance trade-offs.
    """
    summary = {}
    
    for model_key, repo in REPOS.items():
        try:
            impact = load_mitigation_impact(model_key)
            
            summary[model_key] = {
                "model_name": MODEL_DISPLAY_NAMES[model_key],
                "mitigation_strategy": impact.get("mitigation_strategy", "equalized_odds"),
                "fairness_targets_met": impact["summary"]["fairness_targets_met"],
                "performance_drop_acceptable": impact["summary"]["performance_drop_acceptable"],
                "roi_reduction_acceptable": impact["summary"]["roi_reduction_acceptable"],
                "recommended_for_deployment": impact["summary"]["recommended_for_deployment"],
                "fairness_improvements": impact["improvements"]["fairness_improvements"],
                "performance_changes": impact["improvements"]["performance_changes"],
                "roi_changes": impact["improvements"]["roi_changes"]
            }
        except Exception as e:
            summary[model_key] = {"error": str(e)}
    
    return summary


@router.get("/risk-distribution-data")
async def get_risk_distribution_data(demographic: str = "race"):
    """
    Get risk score distribution data for all models by demographic.
    Used to generate risk distribution visualizations.
    
    Args:
        demographic: One of 'race', 'gender', 'age'
    """
    if demographic not in ["race", "gender", "age"]:
        raise HTTPException(status_code=400, detail="Demographic must be one of: race, gender, age")
    
    data = {}
    
    for model_key in REPOS.keys():
        try:
            # Load group metrics which contains sample sizes and performance by group
            group_metrics = load_group_metrics(model_key, demographic)
            
            data[model_key] = {
                "model_name": MODEL_DISPLAY_NAMES[model_key],
                "demographic": demographic,
                "groups": group_metrics.to_dict(orient="records")
            }
        except Exception as e:
            data[model_key] = {"error": str(e)}
    
    return data


@router.get("/fairness-gaps-data")
async def get_fairness_gaps_data():
    """
    Get fairness gap metrics (before/after mitigation) for all models.
    Shows TPR and FPR gaps across all demographic attributes.
    """
    data = {}
    
    for model_key in REPOS.keys():
        try:
            # Load baseline (before) and mitigated (after) metrics
            impact = load_mitigation_impact(model_key)
            
            fairness_improvements = impact["improvements"]["fairness_improvements"]
            
            data[model_key] = {
                "model_name": MODEL_DISPLAY_NAMES[model_key],
                "gaps": {
                    "race": {
                        "tpr_gap_before": fairness_improvements["race"]["tpr_gap_before"],
                        "tpr_gap_after": fairness_improvements["race"]["tpr_gap_after"],
                        "tpr_gap_reduction": fairness_improvements["race"]["tpr_gap_reduction"],
                        "tpr_gap_reduction_pct": fairness_improvements["race"]["tpr_gap_reduction_pct"],
                        "fpr_gap_before": fairness_improvements["race"]["fpr_gap_before"],
                        "fpr_gap_after": fairness_improvements["race"]["fpr_gap_after"],
                        "fpr_gap_reduction": fairness_improvements["race"]["fpr_gap_reduction"],
                        "fpr_gap_reduction_pct": fairness_improvements["race"]["fpr_gap_reduction_pct"]
                    },
                    "gender": {
                        "tpr_gap_before": fairness_improvements["gender"]["tpr_gap_before"],
                        "tpr_gap_after": fairness_improvements["gender"]["tpr_gap_after"],
                        "tpr_gap_reduction": fairness_improvements["gender"]["tpr_gap_reduction"],
                        "tpr_gap_reduction_pct": fairness_improvements["gender"]["tpr_gap_reduction_pct"],
                        "fpr_gap_before": fairness_improvements["gender"]["fpr_gap_before"],
                        "fpr_gap_after": fairness_improvements["gender"]["fpr_gap_after"],
                        "fpr_gap_reduction": fairness_improvements["gender"]["fpr_gap_reduction"],
                        "fpr_gap_reduction_pct": fairness_improvements["gender"]["fpr_gap_reduction_pct"]
                    },
                    "age": {
                        "tpr_gap_before": fairness_improvements["age"]["tpr_gap_before"],
                        "tpr_gap_after": fairness_improvements["age"]["tpr_gap_after"],
                        "tpr_gap_reduction": fairness_improvements["age"]["tpr_gap_reduction"],
                        "tpr_gap_reduction_pct": fairness_improvements["age"]["tpr_gap_reduction_pct"],
                        "fpr_gap_before": fairness_improvements["age"]["fpr_gap_before"],
                        "fpr_gap_after": fairness_improvements["age"]["fpr_gap_after"],
                        "fpr_gap_reduction": fairness_improvements["age"]["fpr_gap_reduction"],
                        "fpr_gap_reduction_pct": fairness_improvements["age"]["fpr_gap_reduction_pct"]
                    }
                }
            }
        except Exception as e:
            data[model_key] = {"error": str(e)}
    
    return data


@router.get("/group-metrics/{model}/{demographic}")
async def get_group_metrics(model: str, demographic: str):
    """
    Get detailed group-level metrics for a specific model and demographic.
    
    Args:
        model: One of 'gradient_boosting', 'random_forest', 'logistic_regression'
        demographic: One of 'race', 'gender', 'age'
    """
    try:
        df = load_group_metrics(model, demographic)
        return {
            "model": MODEL_DISPLAY_NAMES.get(model, model),
            "demographic": demographic,
            "metrics": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/actual-risk-distributions")
async def get_actual_risk_distributions(demographic: str = "race"):
    """
    Get actual predicted probability distributions from local models.
    Returns histogram data for all 3 models by demographic group.
    
    Args:
        demographic: One of 'race', 'gender', 'age'
    """
    if demographic not in ["race", "gender", "age"]:
        raise HTTPException(status_code=400, detail="Demographic must be one of: race, gender, age")
    
    try:
        # Load test data with demographics
        test_file = hf_hub_download(
            repo_id="auphong2707/hospital-readmission-risk-data",
            filename="splits/test.csv",
            repo_type="dataset"
        )
        demo_file = hf_hub_download(
            repo_id="auphong2707/hospital-readmission-risk-data",
            filename="splits/test_demographics.csv",
            repo_type="dataset"
        )
        
        test_df = pd.read_csv(test_file)
        demo_df = pd.read_csv(demo_file)
        
        # Map 10 age groups to 5 categories
        def map_age_group(age_str):
            """Map 10 age groups to 5 categories: young, adult, middle_age, senior, elderly"""
            age_mapping = {
                '[0-10)': 'young',
                '[10-20)': 'young',
                '[20-30)': 'young',
                '[30-40)': 'adult',
                '[40-50)': 'adult',
                '[50-60)': 'middle_age',
                '[60-70)': 'middle_age',
                '[70-80)': 'senior',
                '[80-90)': 'elderly',
                '[90-100)': 'elderly'
            }
            return age_mapping.get(age_str, age_str)
        
        # Apply age mapping
        demo_df['age'] = demo_df['age'].apply(map_age_group)
        
        # Split features and target
        target_col = 'target' if 'target' in test_df.columns else 'readmitted'
        X_test = test_df.drop(columns=[target_col])
        
        # Merge demographics
        merged = pd.concat([X_test, demo_df], axis=1)
        
        # Model paths
        models_dir = Path(__file__).parent / "models"
        model_files = {
            "gradient_boosting": models_dir / "gradient_boosting_model.joblib",
            "random_forest": models_dir / "random_forest_model_rebuilt.joblib",
            "logistic_regression": models_dir / "logistic_regression_model.joblib"
        }
        
        result = {}
        
        for model_key, model_path in model_files.items():
            if not model_path.exists():
                result[model_key] = {"error": f"Model file not found: {model_path}"}
                continue
            
            # Load model and predict
            model = joblib.load(model_path)
            
            # Get predicted probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = model.predict(X_test)
            
            # Add probabilities to merged data
            merged['predicted_risk'] = probs
            
            # Categorize risk into Low, Medium, High
            def categorize_risk(prob):
                if prob < 0.33:
                    return 'Low'
                elif prob < 0.67:
                    return 'Medium'
                else:
                    return 'High'
            
            merged['risk_category'] = merged['predicted_risk'].apply(categorize_risk)
            
            # Calculate risk category distribution by group
            groups_data = []
            for group in sorted(merged[demographic].unique()):
                group_mask = merged[demographic] == group
                group_data = merged[group_mask]
                
                # Count by risk category
                category_counts = group_data['risk_category'].value_counts()
                total = len(group_data)
                
                groups_data.append({
                    "group": str(group),
                    "count": int(total),
                    "mean_risk": float(group_data['predicted_risk'].mean()),
                    "risk_categories": {
                        "Low": int(category_counts.get('Low', 0)),
                        "Medium": int(category_counts.get('Medium', 0)),
                        "High": int(category_counts.get('High', 0)),
                        "Low_pct": float(category_counts.get('Low', 0) / total * 100),
                        "Medium_pct": float(category_counts.get('Medium', 0) / total * 100),
                        "High_pct": float(category_counts.get('High', 0) / total * 100)
                    }
                })
            
            result[model_key] = {
                "model_name": MODEL_DISPLAY_NAMES[model_key],
                "demographic": demographic,
                "groups": groups_data
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating distributions: {str(e)}")
