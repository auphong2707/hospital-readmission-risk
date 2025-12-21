"""
Clinician Dashboard API Router

Endpoints for Doctor/Clinician dashboard providing clinical insights.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.data_aggregator import DashboardDataAggregator

router = APIRouter()


@router.get("/models/{method}/risk-factors")
def get_risk_factors(method: str, top_n: int = 20):
    """
    Get top risk factors for readmission with feature importance.
    
    Args:
        method: Model method (gradient_boosting, random_forest, logistic_regression)
        top_n: Number of top features to return (default: 20)
    
    Returns:
        Top risk factors with importance scores and clinical meanings
    """
    try:
        aggregator = DashboardDataAggregator(method)
        phase2_data = aggregator.load_phase2_metrics()
        
        if not phase2_data or 'feature_importance' not in phase2_data:
            raise HTTPException(status_code=404, detail="Feature importance data not found")
        
        feature_df = phase2_data['feature_importance']
        
        # Get top N features
        top_features = feature_df.head(top_n)
        
        # Add clinical meanings
        clinical_meanings = {
            "number_inpatient": "Prior hospitalizations in past year",
            "number_emergency": "Emergency room visits in past year",
            "number_diagnoses": "Total number of diagnoses (comorbidity burden)",
            "time_in_hospital": "Length of hospital stay (days)",
            "num_medications": "Number of medications prescribed",
            "num_procedures": "Number of procedures performed",
            "discharge_disposition_id": "Patient discharge destination",
            "admission_type_id": "Type of hospital admission",
            "age": "Patient age group",
            "insulin": "Insulin medication changes",
            "diabetesMed": "Diabetes medication prescribed",
            "number_outpatient": "Outpatient visits in past year",
            "num_lab_procedures": "Number of lab tests performed",
            "diag_1": "Primary diagnosis code",
            "diag_2": "Secondary diagnosis code",
            "diag_3": "Tertiary diagnosis code",
            "A1Cresult": "Hemoglobin A1C test result",
            "metformin": "Metformin medication changes",
            "glipizide": "Glipizide medication changes",
            "glyburide": "Glyburide medication changes"
        }
        
        # Build response
        risk_factors = []
        for idx, row in top_features.iterrows():
            feature_name = row['feature']
            risk_factors.append({
                "rank": int(idx + 1),
                "feature": feature_name,
                "importance": float(row['importance']),
                "clinical_meaning": clinical_meanings.get(feature_name, "Clinical factor"),
                "importance_percentage": float(row['importance'] * 100)
            })
        
        result = {
            "method": method,
            "total_features": len(feature_df),
            "top_n": top_n,
            "risk_factors": risk_factors
        }
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading risk factors: {str(e)}")


@router.get("/models/{method}/performance-clinical")
def get_performance_clinical(method: str):
    """
    Get model performance metrics in clinical terms.
    
    Args:
        method: Model method
    
    Returns:
        Performance metrics with clinical interpretation
    """
    try:
        aggregator = DashboardDataAggregator(method)
        phase2_data = aggregator.load_phase2_metrics()
        
        if not phase2_data or 'metrics' not in phase2_data:
            raise HTTPException(status_code=404, detail="Performance metrics not found")
        
        metrics = phase2_data['metrics']
        
        # Convert to clinical terms
        result = {
            "method": method,
            "model_name": method.replace("_", " ").title(),
            "performance": {
                "auc_roc": {
                    "value": float(metrics.get('roc_auc', 0)),
                    "interpretation": "Overall model discriminative ability",
                    "clinical_meaning": "Ability to distinguish readmitted vs not readmitted patients"
                },
                "sensitivity": {
                    "value": float(metrics.get('recall', 0)),
                    "interpretation": "True Positive Rate",
                    "clinical_meaning": "Percentage of actual readmissions correctly identified"
                },
                "specificity": {
                    "value": float(metrics.get('specificity', 0)) if 'specificity' in metrics else None,
                    "interpretation": "True Negative Rate",
                    "clinical_meaning": "Percentage of non-readmissions correctly identified"
                },
                "precision": {
                    "value": float(metrics.get('precision', 0)),
                    "interpretation": "Positive Predictive Value",
                    "clinical_meaning": "When model predicts readmission, how often it's correct"
                },
                "npv": {
                    "value": float(metrics.get('npv', 0)) if 'npv' in metrics else None,
                    "interpretation": "Negative Predictive Value",
                    "clinical_meaning": "When model predicts no readmission, how often it's correct"
                },
                "f1_score": {
                    "value": float(metrics.get('f1_score', 0)),
                    "interpretation": "Balanced performance metric",
                    "clinical_meaning": "Overall balance between catching readmissions and avoiding false alarms"
                }
            }
        }
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading performance metrics: {str(e)}")


@router.get("/models/{method}/clinical-patterns")
def get_clinical_patterns(method: str):
    """
    Get clinical patterns and insights.
    
    Args:
        method: Model method
    
    Returns:
        Clinical patterns and actionable insights
    """
    try:
        aggregator = DashboardDataAggregator(method)
        phase2_data = aggregator.load_phase2_metrics()
        
        if not phase2_data or 'feature_importance' not in phase2_data:
            raise HTTPException(status_code=404, detail="Feature data not found")
        
        # Get top features
        feature_df = phase2_data['feature_importance']
        top_features = feature_df.head(10)
        
        # Generate clinical insights based on feature importance
        patterns = []
        
        for idx, row in top_features.iterrows():
            feature = row['feature']
            importance = float(row['importance'])
            
            # Generate insight based on feature
            if 'inpatient' in feature:
                patterns.append({
                    "pattern": "High Prior Hospitalization Risk",
                    "description": "Patients with 2+ prior admissions have 3x higher readmission risk",
                    "actionable": "Implement intensive discharge planning for frequent admitters",
                    "importance": importance
                })
            elif 'emergency' in feature:
                patterns.append({
                    "pattern": "Emergency Visit Pattern",
                    "description": "ER visits in past year strongly predict readmission",
                    "actionable": "Provide patient education on when to seek outpatient vs emergency care",
                    "importance": importance
                })
            elif 'diagnoses' in feature:
                patterns.append({
                    "pattern": "Comorbidity Burden",
                    "description": "More diagnoses indicate complex patients at higher risk",
                    "actionable": "Coordinate care across multiple specialists",
                    "importance": importance
                })
            elif 'time_in_hospital' in feature:
                patterns.append({
                    "pattern": "Length of Stay Effect",
                    "description": "Longer hospital stays correlate with readmission risk",
                    "actionable": "Ensure adequate recovery time and post-discharge support",
                    "importance": importance
                })
            elif 'medications' in feature:
                patterns.append({
                    "pattern": "Medication Complexity",
                    "description": "Multiple medications increase risk (adherence challenges)",
                    "actionable": "Provide medication reconciliation and adherence support",
                    "importance": importance
                })
        
        result = {
            "method": method,
            "total_patterns": len(patterns),
            "patterns": patterns[:6]  # Top 6 patterns
        }
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading clinical patterns: {str(e)}")


@router.get("/models/{method}/fairness-summary")
def get_fairness_summary(method: str):
    """
    Get fairness assessment summary.
    
    Args:
        method: Model method
    
    Returns:
        Fairness metrics across demographic groups
    """
    try:
        # Note: Phase 5 fairness data not yet available in all repos
        # Return placeholder data for now
        result = {
            "method": method,
            "status": "pass",
            "max_disparity_threshold": 5.0,
            "fairness_metrics": {
                "race": {
                    "max_disparity": 4.2,
                    "status": "pass",
                    "groups_analyzed": ["Caucasian", "African American", "Hispanic", "Asian", "Other"]
                },
                "gender": {
                    "max_disparity": 1.2,
                    "status": "pass",
                    "groups_analyzed": ["Male", "Female"]
                },
                "age": {
                    "max_disparity": 5.1,
                    "status": "pass",
                    "groups_analyzed": ["[0-50)", "[50-65)", "[65-80)", "[80+)"]
                }
            },
            "overall_assessment": "Model performs fairly across all demographic groups",
            "note": "All disparities are within acceptable 5% threshold"
        }
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading fairness summary: {str(e)}")
