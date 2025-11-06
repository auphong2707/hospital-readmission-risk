"""
Example: Upload training results to HuggingFace Hub

This script demonstrates how to upload your trained model and results
to HuggingFace Hub for sharing and version control.

Requirements:
    pip install huggingface_hub python-dotenv

Setup:
    1. Create a HuggingFace account at https://huggingface.co
    2. Get your API token from https://huggingface.co/settings/tokens
    3. Copy .env.example to .env and fill in your credentials:
       HF_TOKEN=your_token_here
       HF_USERNAME=your_username
    4. The repository name will be auto-generated as: username/hospital-readmission-lgbm

Usage:
    python example_upload_to_hf.py
    
Note:
    The training script (train_gradient_boosting.py) automatically uploads
    results after training if HF_TOKEN and HF_USERNAME are set in .env
"""

import json
from pathlib import Path
from utilities import upload_results_to_hf

def main():
    # Path to your model outputs
    output_dir = Path("../models")  # Adjust this path as needed
    
    # Load the training summary
    summary_path = output_dir / "training_summary.json"
    
    if not summary_path.exists():
        print(f"❌ Training summary not found at: {summary_path}")
        print("   Run train_gradient_boosting.py first to generate results.")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print("📊 Training Summary Loaded:")
    print(f"   - Model: {summary.get('model', 'N/A')}")
    print(f"   - Final Test ROC-AUC: {summary.get('final_test_metrics', {}).get('roc_auc', 'N/A'):.4f}")
    print(f"   - Training Time: {summary.get('total_time_seconds', 0)/60:.2f} minutes")
    
    # Upload to HuggingFace
    # Automatically uses HF_TOKEN and HF_USERNAME from .env file
    # Repository name auto-generated as: username/hospital-readmission-lgbm
    success = upload_results_to_hf(
        summary=summary,
        output_dir=output_dir,
        model_name="hospital-readmission-lgbm"
    )
    
    if success:
        print("\n✅ All done! Your model is now publicly available on HuggingFace! 🎉")
    else:
        print("\n⚠️  Upload failed. Check the error messages above.")
        print("   Make sure HF_TOKEN and HF_USERNAME are set in your .env file.")

if __name__ == "__main__":
    main()
