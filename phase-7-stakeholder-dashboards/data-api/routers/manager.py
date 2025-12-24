"""
Manager Dashboard API Router

Endpoints for Manager/Executive dashboard providing savings analysis, resource planning, and cost breakdown.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.data_aggregator import DashboardDataAggregator

router = APIRouter()


@router.get("/models/{method}/savings-summary")
def get_savings_summary(method: str):
    """
    Get savings summary for a specific model (from Phase 6 final metrics).
    
    Args:
        method: 'gradient_boosting', 'random_forest', or 'logistic_regression'
    
    Returns:
        Savings metrics including cost savings, savings ratio, readmissions prevented, costs
    """
    try:
        aggregator = DashboardDataAggregator(method)
        phase6_data = aggregator.load_phase6_final()
        
        if not phase6_data or 'final_system_metrics' not in phase6_data:
            raise HTTPException(status_code=404, detail=f"Phase 6 data not found for {method}")
        
        final_metrics = phase6_data['final_system_metrics']
        performance_metrics = final_metrics.get('performance_metrics', {})
        roi_metrics = final_metrics.get('roi_metrics', {})
        
        # Get confusion matrix values from Phase 6
        tp = performance_metrics.get('true_positives', 0)
        fp = performance_metrics.get('false_positives', 0)
        tn = performance_metrics.get('true_negatives', 0)
        fn = performance_metrics.get('false_negatives', 0)
        total_patients = tp + fp + tn + fn
        
        # ============================================================================
        # COST CALCULATION - Cost Comparison Framework
        # ============================================================================
        # Baseline (Do Nothing): All actual positives readmit at $15,000 each
        # Model (AI-Driven): Intervene on predictions at $500, pay $15,000 for misses
        # Savings = Baseline Cost - Model Cost
        
        readmission_cost = 15000
        intervention_cost = 500
        
        # Baseline: No interventions, all positives (TP + FN) become readmissions
        baseline_cost = (tp + fn) * readmission_cost
        
        # Model: Intervene on all predictions (TP + FP), pay for misses (FN)
        model_cost = (tp + fp) * intervention_cost + fn * readmission_cost
        
        # Total savings from using the model
        cost_savings = baseline_cost - model_cost
        # Shortcut: cost_savings = tp * 14500 - fp * 500
        
        # Individual components for display
        tp_value = tp * 14500  # Net value per prevented readmission ($15K - $500)
        fp_cost = fp * 500     # Cost of unnecessary interventions
        fn_cost = fn * 15000   # Cost of missed readmissions
        
        # Build response with optional fields
        result = {
            "method": method,
            "cost_savings": round(cost_savings, 2),
            "baseline_cost": round(baseline_cost, 2),
            "model_cost": round(model_cost, 2),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "total_patients": int(total_patients),
            "tp_value": round(tp_value, 2),
            "fp_cost": round(fp_cost, 2),
            "fn_cost": round(fn_cost, 2),
            "intervention_rate": round(((tp + fp) / total_patients * 100) if total_patients > 0 else 0, 1)
        }
        
        # Calculate intervention costs and savings ratio
        intervention_costs = (tp + fp) * intervention_cost
        savings_ratio = (cost_savings / intervention_costs) if intervention_costs > 0 else 0
        result["intervention_costs"] = round(intervention_costs, 2)
        result["savings_ratio"] = round(savings_ratio, 2)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{method}/resource-planning")
def get_resource_planning(method: str):
    """
    Get resource planning data for a specific model (from Phase 6 final metrics).
    
    Args:
        method: 'gradient_boosting', 'random_forest', or 'logistic_regression'
    
    Returns:
        Resource planning metrics including intervention volumes, staffing requirements
    """
    try:
        aggregator = DashboardDataAggregator(method)
        phase6_data = aggregator.load_phase6_final()
        
        if not phase6_data or 'final_system_metrics' not in phase6_data:
            raise HTTPException(status_code=404, detail=f"Phase 6 data not found for {method}")
        
        final_metrics = phase6_data['final_system_metrics']
        
        # Get deployment configuration
        deployment_config = final_metrics.get('deployment_configuration', {})
        threshold_summary = deployment_config.get('threshold_summary', {})
        threshold = threshold_summary.get('threshold', 0.35)
        
        # Get performance metrics
        performance_metrics = final_metrics.get('performance_metrics', {})
        tp = performance_metrics.get('true_positives', 0)
        fp = performance_metrics.get('false_positives', 0)
        tn = performance_metrics.get('true_negatives', 0)
        fn = performance_metrics.get('false_negatives', 0)
        
        # Calculate total test patients
        total_test_patients = tp + fp + tn + fn
        
        patients_flagged = tp + fp
        intervention_rate = (patients_flagged / total_test_patients * 100) if total_test_patients > 0 else 0
        
        # Estimate staffing (based on 50 patients per care coordinator)
        care_coordinators_needed = max(1, round(patients_flagged / 50))
        nurse_case_managers = max(1, round(care_coordinators_needed * 0.67))
        social_workers = max(1, round(care_coordinators_needed * 0.33))
        
        # Estimate costs (average salaries)
        avg_coordinator_salary = 65000
        avg_nurse_salary = 75000
        avg_social_worker_salary = 55000
        
        annual_personnel_cost = (
            care_coordinators_needed * avg_coordinator_salary +
            nurse_case_managers * avg_nurse_salary +
            social_workers * avg_social_worker_salary
        )
        
        cost_per_patient = annual_personnel_cost / patients_flagged if patients_flagged > 0 else 0
        
        return {
            "method": method,
            "threshold": round(threshold, 3),
            "patients_flagged": int(patients_flagged),
            "intervention_rate": round(intervention_rate, 1),
            "monthly_volume": int(round(patients_flagged / 12)),
            "weekly_volume": int(round(patients_flagged / 52)),
            "daily_volume": int(round(patients_flagged / 365)),
            "staffing": {
                "care_coordinators": care_coordinators_needed,
                "nurse_case_managers": nurse_case_managers,
                "social_workers": social_workers
            },
            "costs": {
                "annual_personnel_cost": round(annual_personnel_cost, 2),
                "cost_per_patient": round(cost_per_patient, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{method}/cost-breakdown")
def get_cost_breakdown(method: str):
    """
    Get detailed cost breakdown for a specific model (from Phase 6 final metrics).
    
    Args:
        method: 'gradient_boosting', 'random_forest', or 'logistic_regression'
    
    Returns:
        Detailed cost breakdown including costs, benefits, and losses
    """
    try:
        aggregator = DashboardDataAggregator(method)
        phase6_data = aggregator.load_phase6_final()
        
        if not phase6_data or 'final_system_metrics' not in phase6_data:
            raise HTTPException(status_code=404, detail=f"Phase 6 data not found for {method}")
        
        final_metrics = phase6_data['final_system_metrics']
        performance_metrics = final_metrics.get('performance_metrics', {})
        roi_metrics = final_metrics.get('roi_metrics', {})
        
        # Get confusion matrix values from Phase 6
        tp = performance_metrics.get('true_positives', 0)
        fp = performance_metrics.get('false_positives', 0)
        tn = performance_metrics.get('true_negatives', 0)
        fn = performance_metrics.get('false_negatives', 0)
        
        # ============================================================================
        # COST CALCULATION - Cost Comparison Framework
        # ============================================================================
        readmission_cost = 15000
        intervention_cost = 500
        
        # Cost Matrix (per patient)
        tp_unit_value = 14500   # Net value per prevented readmission ($15K - $500)
        fp_unit_cost = 500      # Unnecessary intervention cost
        tn_unit_value = 0       # No action needed
        fn_unit_cost = 15000    # Missed readmission cost
        
        # Calculate financial components
        tp_value = tp * tp_unit_value
        fp_cost = fp * fp_unit_cost
        tn_value = tn * tn_unit_value
        fn_cost = fn * fn_unit_cost
        
        # Baseline: No interventions, all positives (TP + FN) become readmissions
        baseline_cost = (tp + fn) * readmission_cost
        
        # Model: Intervene on all predictions (TP + FP), pay for misses (FN)
        model_cost = (tp + fp) * intervention_cost + fn * readmission_cost
        
        # Total savings from using the model
        cost_savings = baseline_cost - model_cost
        
        # Calculate intervention costs and savings ratio
        intervention_costs = (tp + fp) * intervention_cost
        savings_ratio = (cost_savings / intervention_costs) if intervention_costs > 0 else 0
        
        # Build summary
        summary = {
            "cost_savings": round(cost_savings, 2),
            "intervention_costs": round(intervention_costs, 2),
            "savings_ratio": round(savings_ratio, 2)
        }
        
        return {
            "method": method,
            "cost_matrix": {
                "tp_unit_value": tp_unit_value,
                "fp_unit_cost": fp_unit_cost,
                "tn_unit_value": tn_unit_value,
                "fn_unit_cost": fn_unit_cost
            },
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            },
            "financial_breakdown": {
                "tp_value": round(tp_value, 2),
                "fp_cost": round(fp_cost, 2),
                "tn_value": round(tn_value, 2),
                "fn_cost": round(fn_cost, 2)
            },
            "baseline": {
                "baseline_cost": round(baseline_cost, 2),
                "model_cost": round(model_cost, 2),
                "potential_readmissions": int(tp + fn)
            },
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/comparison")
def get_models_comparison():
    """
    Compare all three models across key business metrics (using Phase 6 data).
    
    Returns:
        Comparison table with savings ratio, performance, and operational metrics for all models
    """
    try:
        methods = ["gradient_boosting", "random_forest", "logistic_regression"]
        comparison_data = []
        
        for method in methods:
            try:
                aggregator = DashboardDataAggregator(method)
                phase6_data = aggregator.load_phase6_final()
                
                if not phase6_data or 'final_system_metrics' not in phase6_data:
                    continue
                
                final_metrics = phase6_data['final_system_metrics']
                performance_metrics = final_metrics.get('performance_metrics', {})
                deployment_config = final_metrics.get('deployment_configuration', {})
                roi_metrics = final_metrics.get('roi_metrics', {})
                
                # Get threshold
                threshold_summary = deployment_config.get('threshold_summary', {})
                threshold = threshold_summary.get('threshold', 0.35)
                
                # Get confusion matrix
                tp = performance_metrics.get('true_positives', 0)
                fp = performance_metrics.get('false_positives', 0)
                tn = performance_metrics.get('true_negatives', 0)
                fn = performance_metrics.get('false_negatives', 0)
                total = tp + fp + tn + fn
                
                # Calculate cost savings using cost comparison framework
                readmission_cost = 15000
                intervention_cost = 500
                baseline_cost = (tp + fn) * readmission_cost
                model_cost = (tp + fp) * intervention_cost + fn * readmission_cost
                cost_savings = baseline_cost - model_cost
                
                intervention_rate = ((tp + fp) / total * 100) if total > 0 else 0
                intervention_costs = (tp + fp) * intervention_cost
                
                # Calculate savings ratio
                savings_ratio = (cost_savings / intervention_costs) if intervention_costs > 0 else 0
                
                model_data = {
                    "method": method,
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "cost_savings": round(cost_savings, 2),
                    "intervention_costs": round(intervention_costs, 2),
                    "savings_ratio": round(savings_ratio, 2),
                    "roc_auc": round(performance_metrics.get('roc_auc', 0), 3),
                    "intervention_rate": round(intervention_rate, 1),
                    "readmissions_prevented": int(tp),
                    "threshold": round(threshold, 3)
                }
                
                comparison_data.append(model_data)
                
            except Exception as e:
                print(f"Error loading {method}: {e}")
                continue
        
        # Determine recommended model (highest cost savings)
        if comparison_data:
            recommended = max(comparison_data, key=lambda x: x['cost_savings'])
            for model in comparison_data:
                model['recommended'] = (model['method'] == recommended['method'])
        
        return {
            "comparison": comparison_data,
            "updated_at": "2025-12-22"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
