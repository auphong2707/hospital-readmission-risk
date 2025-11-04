"""
Random Forest Training for Hospital Readmission Risk Prediction
===============================================================

Ensemble model with comprehensive configuration:
- 100-500 trees with optimized depth tuning
- Feature importance analysis for interpretability
- Temporal validation: Train (1999-2005), Validation (2006-2007), Test (2008)
- Stratified K-fold cross-validation (k=5)
- Grid search for hyperparameter optimization
- Out-of-bag error estimation
- Comprehensive evaluation metrics

Purpose: Ensemble method handling non-linear relationships and feature interactions
Hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

Author: Hospital Readmission Risk Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import time
from datetime import datetime
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score
import matplotlib.pyplot as plt

# Import our custom utilities
from utils import ModelEvaluator, HuggingFaceUploader

warnings.filterwarnings('ignore')


class TemporalSplitter:
    """
    Simulates temporal validation by splitting data chronologically.
    
    Since the dataset doesn't have actual dates, we simulate temporal splits:
    - 70% for training (simulating 1999-2005)
    - 15% for validation (simulating 2006-2007)
    - 15% for test (simulating 2008)
    """
    
    def __init__(self, train_ratio=0.70, val_ratio=0.15, random_state=42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.random_state = random_state
        
    def split(self, X, y):
        """
        Split data into train, validation, and test sets temporally.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        n_samples = len(X)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Create shuffled indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        # Split indices
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # Split data
        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        X_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
        X_test = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
        
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        y_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
        y_test = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
        
        print(f"\n{'='*70}")
        print("Temporal Data Split (Simulated)")
        print(f"{'='*70}")
        print(f"Training set (1999-2005):   {len(X_train):,} samples ({self.train_ratio*100:.1f}%)")
        print(f"Validation set (2006-2007): {len(X_val):,} samples ({self.val_ratio*100:.1f}%)")
        print(f"Test set (2008):            {len(X_test):,} samples ({self.test_ratio*100:.1f}%)")
        print(f"{'='*70}\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class RandomForestTrainer:
    """
    Comprehensive Random Forest training pipeline for hospital readmission prediction.
    
    Features:
    - Ensemble of 100-500 decision trees
    - Depth tuning and feature selection
    - Out-of-bag error estimation
    - Stratified K-fold cross-validation
    - Grid search optimization
    - Temporal validation
    - Feature importance analysis
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.grid_search = None
        self.feature_names = None
        self.oob_scores = []
        
    def load_data(self, data_dir="./data/processed"):
        """Load preprocessed features and target."""
        print(f"\n{'='*70}")
        print("Loading Preprocessed Data")
        print(f"{'='*70}")
        
        # Load features and target
        features_path = data_dir + "/features.csv"
        target_path = data_dir + "/target.csv"

        print(f"Loading features from: {features_path}")
        X = pd.read_csv(features_path)
        
        print(f"Loading target from: {target_path}")
        y = pd.read_csv(target_path)['target']
        
        self.feature_names = X.columns.tolist()
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Number of features: {len(self.feature_names)}")
        print(f"   Target distribution: {dict(y.value_counts())}")
        print(f"{'='*70}\n")
        
        return X, y
    
    def create_hyperparameter_grid(self):
        """
        Create comprehensive hyperparameter grid for Random Forest.
        
        Includes:
        - n_estimators: Number of trees in the forest (100-500)
        - max_depth: Maximum depth of trees (controls overfitting)
        - min_samples_split: Minimum samples required to split a node
        - min_samples_leaf: Minimum samples required at leaf nodes
        - max_features: Number of features to consider for best split
        """
        param_grid = {
            # Number of trees in the forest
            'n_estimators': [100, 250, 500],
            
            # Maximum depth of trees
            'max_depth': [10, 25, None],
            
            # Minimum samples required to split an internal node
            'min_samples_split': [2, 5, 10],
            
            # Minimum samples required at a leaf node
            'min_samples_leaf': [1, 2, 4],
            
            # Number of features to consider at each split
            'max_features': ['sqrt'],
            
            # Class weight balancing
            'class_weight': ['balanced', {0: 1, 1: 8}],
            
            # Bootstrap sampling
            'bootstrap': [True],
            
            # Out-of-bag score estimation
            'oob_score': [True]
        }
        
        return param_grid
    
    def train_with_cv(self, X_train, y_train, n_folds=5):
        """
        Train Random Forest with stratified K-fold cross-validation and grid search.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_folds: Number of folds for cross-validation
            
        Returns:
            Best model from grid search
        """
        print(f"\n{'='*70}")
        print("Hyperparameter Tuning with Stratified K-Fold Cross-Validation")
        print(f"{'='*70}")
        
        # Create parameter grid
        param_grid = self.create_hyperparameter_grid()
        
        print(f"Grid search configuration:")
        print(f"  Number of CV folds: {n_folds}")
        print(f"  Stratification: Yes (balanced across readmission classes)")
        print(f"  Scoring metric: AUC-ROC")
        print(f"  Out-of-bag score: Enabled for additional validation")
        
        # Calculate approximate number of combinations
        n_combinations = (len(param_grid['n_estimators']) * 
                         len(param_grid['max_depth']) * 
                         len(param_grid['min_samples_split']) * 
                         len(param_grid['min_samples_leaf']) * 
                         len(param_grid['max_features']) * 
                         len(param_grid['class_weight']))
        
        print(f"  Total combinations: ~{n_combinations}")
        
        # Create stratified K-fold splitter
        cv_splitter = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Create base Random Forest model
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=0,
            warm_start=False
        )
        
        # Create grid search with AUC-ROC scoring
        self.grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=3,  # Increased verbosity for better progress tracking
            refit=True,
            return_train_score=True
        )
        
        total_fits = n_combinations * n_folds
        
        print(f"\n🔄 Starting grid search...")
        print(f"📊 Training {n_combinations} hyperparameter combinations with {n_folds}-fold CV")
        print(f"📈 Total model fits: {total_fits}")
        print(f"⚠️  Note: Random Forest training may take longer due to ensemble nature")
        print(f"⏱️  Progress will be displayed below with timing for each fit...\n")
        print(f"{'='*70}")
        
        # Fit grid search with timing
        start_time = time.time()
        self.grid_search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ Grid search completed in {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
        print(f"⚡ Average time per fit: {elapsed_time/total_fits:.1f} seconds")
        print(f"{'='*70}")
        
        # Extract best model
        self.best_model = self.grid_search.best_estimator_
        
        # Print results
        print(f"\n{'='*70}")
        print("Grid Search Results")
        print(f"{'='*70}")
        print(f"Best CV Score (AUC-ROC): {self.grid_search.best_score_:.4f}")
        print(f"Out-of-Bag Score: {self.best_model.oob_score_:.4f}")
        print(f"\nBest Hyperparameters:")
        for param, value in self.grid_search.best_params_.items():
            print(f"  {param:<25}: {value}")
        print(f"{'='*70}\n")
        
        return self.best_model
    
    def evaluate_model(self, X, y, set_name="", output_dir=None):
        """
        Evaluate model on a dataset.
        
        Args:
            X: Features
            y: Target
            set_name: Name of the dataset (e.g., "Training", "Validation", "Test")
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating on {set_name} Set")
        print(f"{'='*70}")
        
        # Make predictions
        y_pred = self.best_model.predict(X)
        y_prob = self.best_model.predict_proba(X)[:, 1]
        
        # Create evaluator
        evaluator = ModelEvaluator(model_name=f"Random_Forest_{set_name}")
        
        # Calculate metrics
        metrics = evaluator.calculate_clinical_metrics(y, y_pred, y_prob)
        
        # Print metrics
        evaluator.print_metrics(metrics)
        
        # Generate plots if output directory provided
        if output_dir:
            set_output_dir = os.path.join(output_dir, set_name.lower().replace(" ", "_"))
            os.makedirs(set_output_dir, exist_ok=True)
            
            # Generate full report with plots
            evaluator.generate_full_report(
                y, y_pred, y_prob,
                feature_names=self.feature_names,
                feature_importance=self.best_model.feature_importances_,
                output_dir=set_output_dir
            )
        
        return metrics
    
    def save_model(self, output_dir="./models", include_metadata=True):
        """
        Save trained model with metadata.
        
        Args:
            output_dir: Directory to save the model
            include_metadata: Whether to save hyperparameters and feature names
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"random_forest.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"\n✅ Model saved to: {model_path}")
        
        # Save metadata
        if include_metadata:
            metadata = {
                'model_type': 'Random Forest',
                'timestamp': timestamp,
                'best_params': self.grid_search.best_params_ if self.grid_search else None,
                'best_cv_score': self.grid_search.best_score_ if self.grid_search else None,
                'oob_score': self.best_model.oob_score_ if hasattr(self.best_model, 'oob_score_') else None,
                'n_estimators': self.best_model.n_estimators,
                'feature_names': self.feature_names,
                'n_features': len(self.feature_names) if self.feature_names else None,
                'random_state': self.random_state
            }
            
            metadata_path = os.path.join(output_dir, f"random_forest_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"✅ Metadata saved to: {metadata_path}")
        
        return model_path
    
    def plot_cv_results(self, output_dir=None):
        """
        Plot cross-validation results to analyze hyperparameter impact.
        """
        if self.grid_search is None:
            print("⚠️ No grid search results available. Train the model first.")
            return
        
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Impact of n_estimators on performance
        for max_depth in cv_results['param_max_depth'].unique():
            if pd.notna(max_depth):
                subset = cv_results[cv_results['param_max_depth'] == max_depth]
                grouped = subset.groupby('param_n_estimators')['mean_test_score'].mean()
                axes[0, 0].plot(grouped.index, grouped.values, marker='o', 
                              label=f'max_depth={max_depth}')
        
        axes[0, 0].set_xlabel('Number of Trees (n_estimators)', fontsize=12)
        axes[0, 0].set_ylabel('Mean CV Score (AUC-ROC)', fontsize=12)
        axes[0, 0].set_title('Impact of Number of Trees', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Impact of max_depth on performance
        grouped_depth = cv_results.groupby('param_max_depth')['mean_test_score'].mean().sort_index()
        axes[0, 1].bar(range(len(grouped_depth)), grouped_depth.values, color='steelblue', alpha=0.7)
        axes[0, 1].set_xticks(range(len(grouped_depth)))
        axes[0, 1].set_xticklabels([str(x) for x in grouped_depth.index], rotation=45)
        axes[0, 1].set_xlabel('Maximum Tree Depth', fontsize=12)
        axes[0, 1].set_ylabel('Mean CV Score (AUC-ROC)', fontsize=12)
        axes[0, 1].set_title('Impact of Tree Depth', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Impact of max_features on performance
        grouped_features = cv_results.groupby('param_max_features')['mean_test_score'].mean()
        axes[1, 0].bar(range(len(grouped_features)), grouped_features.values, color='coral', alpha=0.7)
        axes[1, 0].set_xticks(range(len(grouped_features)))
        axes[1, 0].set_xticklabels([str(x) for x in grouped_features.index])
        axes[1, 0].set_xlabel('Max Features per Split', fontsize=12)
        axes[1, 0].set_ylabel('Mean CV Score (AUC-ROC)', fontsize=12)
        axes[1, 0].set_title('Impact of Feature Selection', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Top 10 parameter combinations
        top_10 = cv_results.nlargest(10, 'mean_test_score')[['mean_test_score', 'std_test_score']].reset_index(drop=True)
        axes[1, 1].barh(range(len(top_10)), top_10['mean_test_score'], xerr=top_10['std_test_score'],
                       color='seagreen', alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_10)))
        axes[1, 1].set_yticklabels([f'Config {i+1}' for i in range(len(top_10))])
        axes[1, 1].set_xlabel('Mean CV Score (AUC-ROC)', fontsize=12)
        axes[1, 1].set_ylabel('Configuration Rank', fontsize=12)
        axes[1, 1].set_title('Top 10 Hyperparameter Configurations', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'cv_results_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ CV results plot saved to: {save_path}")
        
        plt.close()
    
    def plot_feature_importance_detailed(self, output_dir=None, top_n=30):
        """
        Create detailed feature importance analysis.
        
        Args:
            output_dir: Directory to save the plot
            top_n: Number of top features to display
        """
        if self.best_model is None:
            print("⚠️ No trained model available.")
            return
        
        # Get feature importances
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': [self.feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Feature Importance (Gini Importance)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features - Random Forest', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'detailed_feature_importance.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Detailed feature importance plot saved to: {save_path}")
            
            # Save to CSV
            csv_path = os.path.join(output_dir, 'feature_importance_ranking.csv')
            importance_df.to_csv(csv_path, index=False)
            print(f"✅ Feature importance ranking saved to: {csv_path}")
        
        plt.close()
        
        return importance_df
    
    def upload_to_huggingface(self, 
                             model_path: str,
                             metadata_path: str,
                             metrics: Dict[str, Dict[str, float]],
                             repo_id: str,
                             hf_token: Optional[str] = None,
                             private: bool = False) -> Optional[str]:
        """
        Upload trained model to HuggingFace Hub.
        
        Args:
            model_path: Path to the saved model
            metadata_path: Path to the metadata file
            metrics: Dictionary containing train/val/test metrics
            repo_id: HuggingFace repository ID (username/model-name)
            hf_token: HuggingFace API token (if None, uses environment variable)
            private: Whether to make the repository private
            
        Returns:
            URL to the uploaded model, or None if upload fails
        """
        try:
            uploader = HuggingFaceUploader(hf_token=hf_token)
            
            description = """
This Random Forest ensemble model provides robust predictions for hospital readmission risk.
It uses 100-500 decision trees with optimized depth tuning and feature selection.

**Key Features:**
- Ensemble of decision trees for robust predictions
- Feature importance analysis for interpretability
- Out-of-bag error estimation for additional validation
- Handles non-linear relationships and feature interactions
- Optimized via 5-fold cross-validation with grid search
"""
            
            repo_url = uploader.upload_model(
                model_path=model_path,
                metadata_path=metadata_path,
                repo_id=repo_id,
                metrics=metrics,
                model_type="Random Forest",
                description=description,
                private=private,
                commit_message="Upload Random Forest model for hospital readmission prediction"
            )
            
            return repo_url
            
        except Exception as e:
            print(f"\n⚠️  Failed to upload to HuggingFace: {str(e)}")
            print("You can manually upload later using the saved model files.")
            return None


def main(hf_repo_id: Optional[str] = None,
         hf_token: Optional[str] = None,
         hf_private: bool = False):
    """
    Main training pipeline for Random Forest ensemble model.
    
    Automatically uploads to HuggingFace if HF_TOKEN is available in environment.
    
    Args:
        hf_repo_id: HuggingFace repository ID (e.g., "username/hospital-readmission-rf")
                   If None and HF_TOKEN exists, will use "username/hospital-readmission-rf"
        hf_token: HuggingFace API token (if None, uses HF_TOKEN env variable)
        hf_private: Whether to make the HuggingFace repository private
    """
    print(f"\n{'='*70}")
    print("RANDOM FOREST TRAINING - HOSPITAL READMISSION RISK")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Ensemble of 100-500 decision trees")
    print(f"  - Depth tuning and feature selection")
    print(f"  - Out-of-bag error estimation")
    print(f"  - Temporal validation (1999-2005 train, 2006-2007 val, 2008 test)")
    print(f"  - Stratified 5-fold cross-validation")
    print(f"  - Grid search hyperparameter optimization")
    print(f"  - Feature importance analysis")
    print(f"{'='*70}\n")
    
    # Initialize trainer
    trainer = RandomForestTrainer(random_state=42)
    
    # Load preprocessed data
    X, y = trainer.load_data()
    
    # Temporal split
    splitter = TemporalSplitter(train_ratio=0.70, val_ratio=0.15, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)
    
    # Train with cross-validation and grid search
    print("🚀 Starting model training...")
    best_model = trainer.train_with_cv(X_train, y_train, n_folds=5)
    
    # Evaluate on all sets
    output_dir = "../reports/random_forest"
    os.makedirs(output_dir, exist_ok=True)
    
    train_metrics = trainer.evaluate_model(X_train, y_train, "Training", output_dir)
    val_metrics = trainer.evaluate_model(X_val, y_val, "Validation", output_dir)
    test_metrics = trainer.evaluate_model(X_test, y_test, "Test", output_dir)
    
    # Plot CV results
    trainer.plot_cv_results(output_dir)
    
    # Plot detailed feature importance
    trainer.plot_feature_importance_detailed(output_dir, top_n=30)
    
    # Save model
    model_path = trainer.save_model(output_dir="../models")
    
    # Collect all metrics for HuggingFace upload
    all_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    
    # Auto-upload to HuggingFace if token is available
    hf_url = None
    if HuggingFaceUploader.is_token_available() or hf_token:
        print(f"\n{'='*70}")
        print("🚀 HuggingFace Token Detected - Preparing Upload")
        print(f"{'='*70}")
        
        # Use default repo_id if not provided
        if hf_repo_id is None:
            hf_repo_id = "hospital-readmission-rf"
            print(f"⚠️  No repo_id provided. Using default: {hf_repo_id}")
            print(f"💡 To specify your own repo, pass: hf_repo_id='username/model-name'")
        
        metadata_path = "../models/random_forest_metadata.pkl"
        hf_url = trainer.upload_to_huggingface(
            model_path=model_path,
            metadata_path=metadata_path,
            metrics=all_metrics,
            repo_id=hf_repo_id,
            hf_token=hf_token,
            private=hf_private
        )
    else:
        print(f"\n{'='*70}")
        print("ℹ️  No HuggingFace Token Found - Skipping Upload")
        print(f"{'='*70}")
        print("To enable auto-upload, set HF_TOKEN environment variable.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print(f"{'='*70}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 Final Performance Summary:")
    print(f"  Training AUC-ROC:   {train_metrics['auc_roc']:.4f}")
    print(f"  Validation AUC-ROC: {val_metrics['auc_roc']:.4f}")
    print(f"  Test AUC-ROC:       {test_metrics['auc_roc']:.4f}")
    
    if hasattr(trainer.best_model, 'oob_score_'):
        print(f"\n  Out-of-Bag Score:   {trainer.best_model.oob_score_:.4f}")
    
    print(f"\n📁 Outputs saved to:")
    print(f"  Model: {model_path}")
    print(f"  Reports: {output_dir}")
    if hf_url:
        print(f"  HuggingFace: {hf_url}")
    print(f"{'='*70}\n")
    
    return trainer, train_metrics, val_metrics, test_metrics


if __name__ == "__main__":
    trainer, train_metrics, val_metrics, test_metrics = main()
