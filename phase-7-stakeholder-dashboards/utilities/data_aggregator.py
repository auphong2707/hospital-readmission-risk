"""
Data Aggregator for Phase 7 Dashboards

Downloads and aggregates data from HuggingFace Hub for dashboard consumption.
"""

from huggingface_hub import hf_hub_download
import json
import pandas as pd
from typing import Dict, Optional, List
import sys
from pathlib import Path

# Import the mapping file from project root
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from file_to_hf_repo_mapping import REPO_MAPPING, get_download_info
except ImportError:
    print("Warning: file_to_hf_repo_mapping.py not found in project root")
    REPO_MAPPING = {}
    def get_download_info(method, phase, filename):
        return None


class DashboardDataAggregator:
    """
    Aggregate data from Phases 1-6 via HuggingFace for dashboard consumption.
    
    Uses file_to_hf_repo_mapping.py as single source of truth for all repository links.
    """
    
    def __init__(self, method: str):
        """
        Initialize aggregator using the mapping file.
        
        Args:
            method: 'random_forest', 'gradient_boosting', 'logistic_regression'
        """
        if method not in REPO_MAPPING:
            raise ValueError(f"Method {method} not found in REPO_MAPPING")
        
        self.method = method
        self.mapping = REPO_MAPPING[method]
    
    def download_file(self, phase: str, filename: str) -> Optional[str]:
        """
        Download a file from HuggingFace using the mapping file.
        
        Args:
            phase: 'phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6'
            filename: The filename as defined in the mapping
        
        Returns:
            Local path to downloaded file (cached by HuggingFace Hub)
        """
        download_info = get_download_info(self.method, phase, filename)
        if not download_info:
            raise ValueError(f"File {filename} not found in mapping for {self.method}/{phase}")
        
        try:
            return hf_hub_download(
                repo_id=download_info["repo_id"],
                filename=download_info["file_path"],
                repo_type=download_info["repo_type"]
            )
        except Exception as e:
            print(f"Error downloading {filename} from {phase}: {e}")
            return None
    
    def load_phase1_data(self) -> Dict:
        """Load Phase 1 preprocessing data using mapping file."""
        try:
            split_info_path = self.download_file("phase1", "split_info.txt")
            if split_info_path:
                with open(split_info_path, 'r') as f:
                    split_info = f.read()
                return {'split_info': split_info}
        except Exception as e:
            print(f"Error loading Phase 1 data: {e}")
        return {}
    
    def load_phase2_metrics(self) -> Dict:
        """Load Phase 2 model metrics and visualizations using mapping file."""
        try:
            # Determine correct filename based on method
            if self.method == 'logistic_regression':
                metrics_file = "logistic_regression_metrics.json"
            else:
                metrics_file = f"{self.method}_metrics.json"
            
            metrics_path = self.download_file("phase2", metrics_file)
            if not metrics_path:
                return {}
            
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Load feature importance (same naming across all models)
            feat_imp_path = self.download_file("phase2", "feature_importance.csv")
            feature_importance = None
            if feat_imp_path:
                feature_importance = pd.read_csv(feat_imp_path)
            
            return {
                'metrics': metrics,
                'feature_importance': feature_importance
            }
        except Exception as e:
            print(f"Error loading Phase 2 metrics: {e}")
            return {}
    
    def load_phase3_calibration(self) -> Dict:
        """Load Phase 3 calibration metrics using mapping file."""
        try:
            # Determine calibrator prefix based on method (defined in mapping file)
            calibrator_prefix_map = {
                "gradient_boosting": "Gradient_Boosting_(LightGBM)",
                "random_forest": "Random_Forest",
                "logistic_regression": "Logistic_Regression"
            }
            calibrator_prefix = calibrator_prefix_map[self.method]
            metrics_file = f"{calibrator_prefix}_metrics.json"
            
            metrics_path = self.download_file("phase3", metrics_file)
            if not metrics_path:
                return {}
            
            with open(metrics_path, 'r') as f:
                calibration_metrics = json.load(f)
            
            return {'calibration_metrics': calibration_metrics}
        except Exception as e:
            print(f"Error loading Phase 3 calibration: {e}")
            return {}
    
    def load_phase4_roi(self) -> Dict:
        """Load Phase 4 ROI and threshold optimization using mapping file."""
        try:
            roi_path = self.download_file("phase4", "roi_metrics.json")
            thresholds_path = self.download_file("phase4", "optimal_thresholds.json")
            
            roi_metrics = {}
            thresholds = {}
            
            if roi_path:
                with open(roi_path, 'r') as f:
                    roi_metrics = json.load(f)
            
            if thresholds_path:
                with open(thresholds_path, 'r') as f:
                    thresholds = json.load(f)
            
            return {
                'roi_metrics': roi_metrics,
                'thresholds': thresholds
            }
        except Exception as e:
            print(f"Error loading Phase 4 ROI: {e}")
            return {}
    
    def load_phase5_fairness(self) -> Dict:
        """Load Phase 5 fairness assessment using mapping file."""
        try:
            # Note: Phase 5 not yet in mapping, will be added later
            # For now, return empty dict
            return {}
        except Exception as e:
            print(f"Error loading Phase 5 fairness: {e}")
            return {}
    
    def load_phase6_final(self) -> Dict:
        """Load Phase 6 final evaluation using mapping file."""
        try:
            # Note: Phase 6 not yet in mapping, will be added later
            # For now, return empty dict
            return {}
        except Exception as e:
            print(f"Error loading Phase 6 final evaluation: {e}")
            return {}
    
    def aggregate_all(self) -> Dict:
        """Load all phase data from HuggingFace into single dictionary."""
        print(f"Loading data for {self.method} from HuggingFace using mapping file...")
        return {
            'method': self.method,
            'phase1': self.load_phase1_data(),
            'phase2': self.load_phase2_metrics(),
            'phase3': self.load_phase3_calibration(),
            'phase4': self.load_phase4_roi(),
            'phase5': self.load_phase5_fairness(),
            'phase6': self.load_phase6_final(),
        }
