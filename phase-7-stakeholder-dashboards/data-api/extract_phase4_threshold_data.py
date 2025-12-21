"""
Extract threshold vs costs/benefits data from Phase 4 for all models.
Generates data with 100 threshold points for smooth curves.
"""

import sys
from pathlib import Path
import json
import numpy as np
from huggingface_hub import hf_hub_download

# Phase 4 HuggingFace repositories
PHASE4_REPOS = {
    "gradient_boosting": "auphong2707/hospital-readmission-phase4-lgbm-threshold",
    "random_forest": "auphong2707/hospital-readmission-phase4-rf-threshold",
    "logistic_regression": "auphong2707/hospital-readmission-phase4-lr-threshold"
}

def download_threshold_results(method):
    """Download threshold_results.csv from Phase 4 repo."""
    try:
        repo_id = PHASE4_REPOS[method]
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="outputs/threshold_results.csv",
            repo_type="model"
        )
        return file_path
    except Exception as e:
        print(f"Error downloading {method} threshold results: {e}")
        return None

def parse_threshold_data(csv_path):
    """Parse threshold_results.csv to extract threshold, costs, and benefits."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    # Assuming columns: threshold, total_costs, total_benefits, net_benefit, etc.
    data = {
        'threshold': df['threshold'].tolist() if 'threshold' in df.columns else [],
        'total_costs': df['total_costs'].tolist() if 'total_costs' in df.columns else [],
        'total_benefits': df['total_benefits'].tolist() if 'total_benefits' in df.columns else [],
        'net_benefit': df['net_benefit'].tolist() if 'net_benefit' in df.columns else []
    }
    
    return data

def extract_all_models_data():
    """Extract threshold data for all 3 models."""
    model_labels = {
        "gradient_boosting": "Gradient Boosting",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression"
    }
    
    all_data = {
        'costs': [],
        'benefits': []
    }
    
    for method, label in model_labels.items():
        print(f"\nProcessing {label}...")
        
        csv_path = download_threshold_results(method)
        if not csv_path:
            print(f"  ⚠️ Could not download data for {label}")
            continue
        
        data = parse_threshold_data(csv_path)
        
        if not data['threshold']:
            print(f"  ⚠️ No threshold data found for {label}")
            continue
        
        print(f"  ✅ Found {len(data['threshold'])} threshold points")
        
        # Add to costs data
        for threshold, cost in zip(data['threshold'], data['total_costs']):
            all_data['costs'].append({
                'threshold': threshold,
                'model': label,
                'value': cost
            })
        
        # Add to benefits data
        for threshold, benefit in zip(data['threshold'], data['total_benefits']):
            all_data['benefits'].append({
                'threshold': threshold,
                'model': label,
                'value': benefit
            })
    
    return all_data

def save_data():
    """Extract and save threshold data."""
    print("="*60)
    print("Extracting Phase 4 Threshold Data")
    print("="*60)
    
    all_data = extract_all_models_data()
    
    # Save to JSON
    output_path = Path(__file__).parent / "phase4_threshold_data.json"
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n✅ Saved threshold data to {output_path}")
    print(f"   Costs data points: {len(all_data['costs'])}")
    print(f"   Benefits data points: {len(all_data['benefits'])}")
    
    return all_data

if __name__ == "__main__":
    try:
        import pandas
    except ImportError:
        print("Installing pandas...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas
    
    save_data()
