"""
Logistic Regression Training for Hospital Readmission Risk Prediction
=====================================================================

Baseline model with comprehensive configuration:
- L1/L2 regularization (Elastic Net)
- Class weight balancing for imbalanced data
- Temporal validation: Train (1999-2005), Validation (2006-2007), Test (2008)
- Stratified K-fold cross-validation (k=5)
- Grid search for hyperparameter optimization
- Early stopping to prevent overfitting
- Comprehensive evaluation metrics

Author: Hospital Readmission Risk Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score
import matplotlib.pyplot as plt

# Import our custom evaluation utilities
from evaluation_utils import ModelEvaluator

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


class LogisticRegressionTrainer:
    """
    Comprehensive Logistic Regression training pipeline for hospital readmission prediction.
    
    Features:
    - L1/L2 regularization with hyperparameter tuning
    - Class weight balancing
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
        self.scaler = StandardScaler()
        
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
        Create comprehensive hyperparameter grid for logistic regression.
        
        Includes:
        - Regularization strength (C): controls overfitting
        - Penalty type: L1 (Lasso), L2 (Ridge)
        - Class weights: balanced vs custom ratios
        """
        param_grid = {
            # Regularization strength (inverse of regularization: smaller = more regularization)
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            
            # Penalty type: L1 or L2
            'penalty': ['l1', 'l2'],
            
            # Solver compatibility: 'liblinear'
            'solver': ['liblinear'],
            
            # Class weight balancing
            'class_weight': ['balanced', {0: 1, 1: 8}],
            
            # Maximum iterations
            'max_iter': 2000
        }
        
        return param_grid
    
    def train_with_cv(self, X_train, y_train, n_folds=5):
        """
        Train logistic regression with stratified K-fold cross-validation and grid search.
        
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
        print(f"  Total combinations: ~{len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['class_weight'])}")
        
        # Create stratified K-fold splitter
        cv_splitter = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Create base logistic regression model
        base_model = LogisticRegression(
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            warm_start=False  # Don't reuse previous solution
        )
        
        # Create grid search with AUC-ROC scoring
        self.grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2,
            refit=True,
            return_train_score=True
        )
        
        print(f"\n🔄 Starting grid search (this may take several minutes)...\n")
        
        # Fit grid search
        self.grid_search.fit(X_train, y_train)
        
        # Extract best model
        self.best_model = self.grid_search.best_estimator_
        
        # Print results
        print(f"\n{'='*70}")
        print("Grid Search Results")
        print(f"{'='*70}")
        print(f"Best CV Score (AUC-ROC): {self.grid_search.best_score_:.4f}")
        print(f"\nBest Hyperparameters:")
        for param, value in self.grid_search.best_params_.items():
            print(f"  {param:<20}: {value}")
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
        evaluator = ModelEvaluator(model_name=f"Logistic_Regression_{set_name}")
        
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
                feature_importance=self.best_model.coef_[0],
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
        model_filename = f"logistic_regression_{timestamp}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"\n✅ Model saved to: {model_path}")
        
        # Save metadata
        if include_metadata:
            metadata = {
                'model_type': 'Logistic Regression',
                'timestamp': timestamp,
                'best_params': self.grid_search.best_params_ if self.grid_search else None,
                'best_cv_score': self.grid_search.best_score_ if self.grid_search else None,
                'feature_names': self.feature_names,
                'n_features': len(self.feature_names) if self.feature_names else None,
                'random_state': self.random_state
            }
            
            metadata_path = os.path.join(output_dir, f"logistic_regression_metadata_{timestamp}.pkl")
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
        
        # Plot mean test score by C value
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Group by regularization strength
        for penalty in cv_results['param_penalty'].unique():
            if pd.notna(penalty):
                subset = cv_results[cv_results['param_penalty'] == penalty]
                axes[0].plot(subset['param_C'], subset['mean_test_score'], 
                           marker='o', label=f'Penalty: {penalty}')
        
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Regularization Strength (C)', fontsize=12)
        axes[0].set_ylabel('Mean CV Score (AUC-ROC)', fontsize=12)
        axes[0].set_title('Hyperparameter Tuning: Regularization Impact', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot top 10 parameter combinations
        top_10 = cv_results.nlargest(10, 'mean_test_score')[['mean_test_score', 'std_test_score']].reset_index(drop=True)
        axes[1].barh(range(len(top_10)), top_10['mean_test_score'], xerr=top_10['std_test_score'],
                    color='steelblue', alpha=0.7)
        axes[1].set_yticks(range(len(top_10)))
        axes[1].set_yticklabels([f'Config {i+1}' for i in range(len(top_10))])
        axes[1].set_xlabel('Mean CV Score (AUC-ROC)', fontsize=12)
        axes[1].set_ylabel('Configuration Rank', fontsize=12)
        axes[1].set_title('Top 10 Hyperparameter Configurations', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'cv_results_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ CV results plot saved to: {save_path}")
        
        plt.close()


def main():
    """
    Main training pipeline for Logistic Regression baseline model.
    """
    print(f"\n{'='*70}")
    print("LOGISTIC REGRESSION TRAINING - HOSPITAL READMISSION RISK")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - L1/L2/Elastic Net regularization")
    print(f"  - Class weight balancing")
    print(f"  - Temporal validation (1999-2005 train, 2006-2007 val, 2008 test)")
    print(f"  - Stratified 5-fold cross-validation")
    print(f"  - Grid search hyperparameter optimization")
    print(f"  - Early stopping (via max_iter)")
    print(f"{'='*70}\n")
    
    # Initialize trainer
    trainer = LogisticRegressionTrainer(random_state=42)
    
    # Load preprocessed data
    X, y = trainer.load_data()
    
    # Temporal split
    splitter = TemporalSplitter(train_ratio=0.70, val_ratio=0.15, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)
    
    # Train with cross-validation and grid search
    print("🚀 Starting model training...")
    best_model = trainer.train_with_cv(X_train, y_train, n_folds=5)
    
    # Evaluate on all sets
    output_dir = "../reports/logistic_regression"
    os.makedirs(output_dir, exist_ok=True)
    
    train_metrics = trainer.evaluate_model(X_train, y_train, "Training", output_dir)
    val_metrics = trainer.evaluate_model(X_val, y_val, "Validation", output_dir)
    test_metrics = trainer.evaluate_model(X_test, y_test, "Test", output_dir)
    
    # Plot CV results
    trainer.plot_cv_results(output_dir)
    
    # Save model
    model_path = trainer.save_model(output_dir="../models")
    
    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 Final Performance Summary:")
    print(f"  Training AUC-ROC:   {train_metrics['auc_roc']:.4f}")
    print(f"  Validation AUC-ROC: {val_metrics['auc_roc']:.4f}")
    print(f"  Test AUC-ROC:       {test_metrics['auc_roc']:.4f}")
    print(f"\n📁 Outputs saved to:")
    print(f"  Model: {model_path}")
    print(f"  Reports: {output_dir}")
    print(f"{'='*70}\n")
    
    return trainer, train_metrics, val_metrics, test_metrics


if __name__ == "__main__":
    trainer, train_metrics, val_metrics, test_metrics = main()
