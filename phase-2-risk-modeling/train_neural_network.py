"""Complete pipeline to preprocess data and train Neural Network for 30-day readmission.

This script implements the neural network architecture from:
"Predicting all-cause risk of 30-day hospital readmission using artificial neural networks"
(Jamei et al., 2017, PLOS ONE)

Architecture:
- 2-layer neural network
- Hidden layer size: input_size // 2
- Dropout between all layers
- Adam optimizer
- Binary cross-entropy loss

Evaluation Pipeline (matches gradient boosting):
1. Final Holdout Split: development_set + final_test_set
2. Hyperparameter Search with K-fold CV
3. K-Fold Cross-Validation with best parameters
4. Train final model on full development_set
5. Final evaluation on untouched final_test_set

Usage (from project root):
    python ./phase-2-risk-modeling/train_neural_network.py
    
Kaggle usage:
    !python ./phase-2-risk-modeling/train_neural_network.py

Requirements:
    pip install torch scikit-learn pandas numpy matplotlib seaborn tqdm
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import warnings

import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import product

from utilities import (
    calculate_comprehensive_metrics,
    print_metrics_table,
    save_visualizations,
    save_learning_curves,
    save_validation_curves,
    save_metrics_comparison,
    calculate_permutation_importance,
    save_permutation_importance,
    detect_gpu,
    is_kaggle_environment,
    print_section,
    load_data,
    run_preprocessing,
    upload_results_to_hf
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Install with: pip install tqdm")

warnings.filterwarnings("ignore")


class ReadmissionNN(nn.Module):
    """Neural Network for hospital readmission prediction.
    
    Architecture from Jamei et al. (2017):
    - Input layer
    - Dropout
    - Hidden layer (size = input_size // 2)
    - Dropout
    - Output layer (sigmoid activation)
    """
    
    def __init__(self, input_size: int, dropout_rate: float = 0.5):
        """Initialize the neural network.
        
        Args:
            input_size: Number of input features
            dropout_rate: Dropout probability (default: 0.5)
        """
        super(ReadmissionNN, self).__init__()
        
        hidden_size = input_size // 2
        
        self.network = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class NNTrainer:
    """Trainer for the ReadmissionNN model.
    
    Responsibilities:
    - Hold model and train/val data
    - Fit with optional early stopping
    - Evaluate and return comprehensive metrics
    - Save model artifact
    """
    
    def __init__(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        output_dir: str = "models",
        device: str = "cpu",
        batch_size: int = 64,
        learning_rate: float = 0.001
    ):
        """Initialize the trainer.
        
        Args:
            model: PyTorch model instance
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            output_dir: Directory to save models
            device: Device to train on ('cpu' or 'cuda')
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Validation data
        self.X_val = None
        self.y_val = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.X_val = torch.FloatTensor(X_val_scaled).to(device)
            self.y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Track best model for early stopping
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def fit(
        self, 
        epochs: int = 5, 
        early_stopping_rounds: int | None = None,
        verbose: bool = True
    ):
        """Train the model with optional early stopping.
        
        Args:
            epochs: Maximum number of epochs
            early_stopping_rounds: Patience for early stopping (None to disable)
            verbose: Whether to print progress
        """
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(self.train_loader)
            
            # Validation
            if self.X_val is not None and early_stopping_rounds is not None:
                val_loss = self._validate()
                
                if verbose:
                    print(f"      Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= early_stopping_rounds:
                        if verbose:
                            print(f"      Early stopping at epoch {epoch+1}")
                        break
            else:
                if verbose:
                    print(f"      Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Restore best model if early stopping was used
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
    
    def _validate(self) -> float:
        """Validate the model and return loss."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_val)
            val_loss = self.criterion(outputs, self.y_val).item()
        self.model.train()
        return val_loss
    
    def predict_proba(self, X: np.ndarray):
        """Predict class probabilities (required for permutation_importance).
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Probabilities for each class [P(class=0), P(class=1)]
        """
        self.model.eval()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            proba_pos = self.model(X_tensor).cpu().numpy().flatten()
        
        # Return probabilities for both classes
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5):
        """Evaluate model with comprehensive metrics.
        
        Args:
            X: Features to evaluate on
            y: True labels
            threshold: Classification threshold
            
        Returns:
            tuple: (metrics_dict, probabilities, predictions)
        """
        self.model.eval()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy().flatten()
        
        pred = (proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(y, proba, threshold)
        
        return metrics, proba, pred
    
    def save(self, path: str | Path):
        """Save the trained model and scaler.
        
        Args:
            path: File path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both model and scaler
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'model_architecture': {
                'input_size': self.model.network[1].in_features,
                'dropout_rate': self.model.network[0].p
            }
        }
        torch.save(checkpoint, path)


def get_nn_param_grid():
    """Get hyperparameter grid for neural network.
    
    Returns:
        dict: Parameter grid for grid search
    """
    return {
        'dropout_rate': [0.3, 0.5, 0.7],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [32, 64, 128]
    }


def train_model(args: argparse.Namespace):
    """Main training function with robust evaluation pipeline."""
    start_time = time.time()
    
    # Set device
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Print configuration
    print_section("🚀 Neural Network Training - Hospital Readmission Risk", "=")
    print(f"⚙️  Configuration:")
    print(f"   - Data directory: {args.data_dir}")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - Test size: {args.test_size}")
    print(f"   - Validation size: {args.val_size}")
    print(f"   - K-fold splits: {args.n_splits}")
    print(f"   - Early stopping rounds: {args.early_stopping_rounds}")
    print(f"   - Max epochs: {args.max_epochs}")
    print(f"   - Random seed: {args.random_state}")
    print(f"   - Device: {device}")
    print(f"   - Environment: {'🏆 Kaggle' if args.environment == 'kaggle' else '💻 Local'}")
    
    # Set random seeds
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    
    # Load data
    X, y = load_data(args.data_dir)
    
    # Convert to numpy arrays
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y

    # STEP 1: Final holdout split
    print_section("🔀 Step 1: Final Holdout Split", "-")
    print(f"Splitting entire dataset into development_set ({1-args.test_size:.0%}) "
          f"and final_test_set ({args.test_size:.0%})...")
    X_development, X_final_test, y_development, y_final_test = train_test_split(
        X_array, y_array, 
        test_size=args.test_size, 
        random_state=args.random_state, 
        stratify=y_array
    )
    print(f"   ✅ Development set: {X_development.shape}")
    print(f"   ✅ Final test set (untouched): {X_final_test.shape}")

    # STEP 2: Hyperparameter Search
    print_section("🔍 Step 2: Hyperparameter Search with K-Fold CV", "-")
    
    param_grid = get_nn_param_grid()
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                          for v in product(*param_grid.values())]
    total_combinations = len(param_combinations)
    
    print(f"📊 Hyperparameter Search Space:")
    print(f"   Total parameter combinations: {total_combinations}")
    print(f"   K-fold splits: {args.n_splits}")
    print(f"   Total model fits: {total_combinations * args.n_splits}")
    print(f"   Scoring metric: ROC-AUC\n")
    
    best_score = -np.inf
    best_params = None
    all_search_results = []
    
    cv_search = StratifiedKFold(n_splits=args.n_splits, shuffle=True, 
                                random_state=args.random_state)
    
    search_start = time.time()
    
    for combo_idx, params in enumerate(param_combinations, 1):
        if combo_idx % 5 == 0 or combo_idx == 1 or combo_idx == total_combinations:
            print(f"🔎 Evaluating combination {combo_idx}/{total_combinations}")
            print(f"   Parameters: {params}")
        
        combo_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            cv_search.split(X_development, y_development), 1
        ):
            X_combo_train = X_development[train_idx]
            y_combo_train = y_development[train_idx]
            X_combo_val = X_development[val_idx]
            y_combo_val = y_development[val_idx]
            
            # Split training data for early stopping
            X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                X_combo_train, y_combo_train,
                test_size=args.val_size,
                random_state=args.random_state,
                stratify=y_combo_train
            )
            
            # Create model with current parameters
            model = ReadmissionNN(
                input_size=X_development.shape[1],
                dropout_rate=params['dropout_rate']
            )
            
            trainer = NNTrainer(
                model=model,
                X_train=X_inner_train,
                y_train=y_inner_train,
                X_val=X_inner_val,
                y_val=y_inner_val,
                device=str(device),
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate']
            )
            
            # Train with early stopping
            if args.early_stopping_rounds > 0:
                trainer.fit(
                    epochs=args.max_epochs,
                    early_stopping_rounds=args.early_stopping_rounds,
                    verbose=False
                )
            else:
                trainer.fit(epochs=args.max_epochs, verbose=False)
            
            # Evaluate on fold's validation set
            fold_metrics, _, _ = trainer.evaluate(X_combo_val, y_combo_val)
            combo_scores.append(fold_metrics['roc_auc'])
        
        mean_score = np.mean(combo_scores)
        std_score = np.std(combo_scores)
        
        all_search_results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': combo_scores
        })
        
        if combo_idx % 5 == 0 or combo_idx == 1 or combo_idx == total_combinations:
            print(f"   Mean ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            print(f"   >>> 🏆 New best score: {best_score:.4f}")
    
    search_time = time.time() - search_start
    
    print(f"\n{'='*60}")
    print("✅ Hyperparameter search completed")
    print(f"{'='*60}")
    print(f"⏱️  Search time: {search_time:.2f} seconds")
    print(f"🏆 Best CV ROC-AUC: {best_score:.4f}")
    print(f"📋 Best parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # STEP 3: K-Fold Training with best parameters
    print("\n" + "="*60)
    print("🎯 Step 3: Final K-Fold CV with Best Parameters")
    print("="*60)
    print(f"Re-training with best parameters across {args.n_splits} folds\n")
    
    cv_kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, 
                               random_state=args.random_state)
    
    fold_scores = []
    fold_details = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(
        cv_kfold.split(X_development, y_development), 1
    ):
        print(f"\n{'='*60}")
        print(f"📁 Fold {fold_idx}/{args.n_splits}")
        print(f"{'='*60}")
        
        X_fold_train = X_development[train_idx]
        y_fold_train = y_development[train_idx]
        X_fold_holdout = X_development[test_idx]
        y_fold_holdout = y_development[test_idx]
        
        print(f"   Fold train size: {len(X_fold_train)}")
        print(f"   Fold holdout size: {len(X_fold_holdout)}")
        
        # Nested split for early stopping
        print(f"\n   🔀 Nested split for early stopping (val_size={args.val_size})...")
        X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
            X_fold_train, y_fold_train,
            test_size=args.val_size,
            random_state=args.random_state,
            stratify=y_fold_train
        )
        print(f"      Inner train size: {len(X_inner_train)}")
        print(f"      Inner val size: {len(X_inner_val)}")
        
        # Train model
        print(f"\n   🏋️  Training neural network...")
        model = ReadmissionNN(
            input_size=X_development.shape[1],
            dropout_rate=best_params['dropout_rate']
        )
        
        trainer = NNTrainer(
            model=model,
            X_train=X_inner_train,
            y_train=y_inner_train,
            X_val=X_inner_val,
            y_val=y_inner_val,
            device=str(device),
            batch_size=best_params['batch_size'],
            learning_rate=best_params['learning_rate']
        )
        
        if args.early_stopping_rounds > 0:
            trainer.fit(
                epochs=args.max_epochs,
                early_stopping_rounds=args.early_stopping_rounds,
                verbose=True
            )
            print(f"      ✅ Training complete (with early stopping)")
        else:
            trainer.fit(epochs=args.max_epochs, verbose=True)
            print(f"      ✅ Training complete")
        
        # Evaluate on fold holdout
        print(f"\n   📊 Evaluating on fold holdout...")
        fold_metrics, fold_proba, fold_pred = trainer.evaluate(
            X_fold_holdout, y_fold_holdout
        )
        
        print(f"      ROC-AUC: {fold_metrics['roc_auc']:.4f}")
        print(f"      Precision: {fold_metrics['precision']:.4f}")
        print(f"      Recall: {fold_metrics['recall']:.4f}")
        print(f"      F1: {fold_metrics['f1']:.4f}")
        
        fold_scores.append(fold_metrics['roc_auc'])
        fold_details.append({
            'fold': fold_idx,
            'metrics': fold_metrics,
            'train_size': len(X_inner_train),
            'val_size': len(X_inner_val),
            'holdout_size': len(X_fold_holdout)
        })

    # STEP 4: Calculate CV statistics
    print_section("📊 Step 4: K-Fold Cross-Validation Results", "=")
    fold_scores_array = np.array(fold_scores)
    mean_score = fold_scores_array.mean()
    std_score = fold_scores_array.std()
    
    print(f"🎯 Cross-Validation ROC-AUC Scores:")
    for i, score in enumerate(fold_scores, 1):
        print(f"   Fold {i}: {score:.4f}")
    print(f"\n   {'─'*40}")
    print(f"   Mean ROC-AUC:   {mean_score:.4f}")
    print(f"   Std Dev:        {std_score:.4f}")
    print(f"   95% CI:         [{mean_score - 1.96*std_score:.4f}, "
          f"{mean_score + 1.96*std_score:.4f}]")
    print(f"   {'─'*40}")

    # STEP 5: Train final model
    print_section("🏗️  Step 5: Training Final Model on Development Set", "-")
    print("Training final model on full development set...")
    
    X_dev_train, X_dev_val, y_dev_train, y_dev_val = train_test_split(
        X_development, y_development,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_development
    )
    
    final_model = ReadmissionNN(
        input_size=X_development.shape[1],
        dropout_rate=best_params['dropout_rate']
    )
    
    final_trainer = NNTrainer(
        model=final_model,
        X_train=X_dev_train,
        y_train=y_dev_train,
        X_val=X_dev_val,
        y_val=y_dev_val,
        device=str(device),
        batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate']
    )
    
    if args.early_stopping_rounds > 0:
        final_trainer.fit(
            epochs=args.max_epochs,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=True
        )
    else:
        final_trainer.fit(epochs=args.max_epochs, verbose=True)
    
    print(f"✅ Final model trained on {len(X_dev_train)} samples")

    # STEP 6: Final evaluation
    print_section("🎯 Step 6: Final Evaluation on Untouched Test Set", "=")
    final_metrics, y_final_proba, y_final_pred = final_trainer.evaluate(
        X_final_test, y_final_test
    )
    
    print_metrics_table(final_metrics, "🎯 FINAL TEST SET RESULTS")
    
    print(f"\n📈 Model Performance Summary:")
    print(f"   Cross-Validation (Development Set):")
    print(f"      Mean ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
    print(f"   Final Test Set (Untouched Holdout):")
    print(f"      ROC-AUC: {final_metrics['roc_auc']:.4f}")

    # Create output directory
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save visualizations
    print_section("📊 Generating Visualizations", "-")
    save_visualizations(
        y_final_test, y_final_proba, y_final_pred, out_dir,
        model=None, X=None, feature_names=None  # NN doesn't have built-in feature importance
    )
    
    # Learning curves
    save_learning_curves(
        final_model, X_development, y_development, out_dir, cv=args.n_splits
    )
    
    # Validation curves
    save_validation_curves(all_search_results, out_dir)
    
    # Metrics comparison
    save_metrics_comparison(fold_details, out_dir)
    
    # Feature importance using permutation importance
    print_section("🔍 Calculating Feature Importance", "-")
    print("Using permutation importance method (model-agnostic)...")
    
    # Get feature names from original data
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f'Feature {i}' for i in range(X_array.shape[1])]
    
    # Calculate permutation importance on test set
    importance_dict = calculate_permutation_importance(
        model=final_trainer,
        X=X_final_test,
        y=y_final_test,
        feature_names=feature_names,
        n_repeats=10,
        random_state=args.random_state
    )
    
    # Save permutation importance plots and CSV
    if importance_dict is not None:
        save_permutation_importance(importance_dict, out_dir, top_n=20)

    # Save model and artifacts
    print_section("💾 Saving Results", "-")
    
    model_path = out_dir / "neural_network_model.pt"
    final_trainer.save(model_path)
    print(f"✅ Model saved: {model_path}")

    metrics_path = out_dir / "neural_network_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")
    
    fold_details_path = out_dir / "nn_cv_fold_details.json"
    with open(fold_details_path, "w") as f:
        json.dump(fold_details, f, indent=2)
    print(f"✅ Fold details saved: {fold_details_path}")

    # Create summary
    total_time = time.time() - start_time
    summary = {
        "model": "Neural Network (PyTorch)",
        "architecture": "2-layer NN (Jamei et al., 2017)",
        "task": "Hospital 30-Day Readmission Risk Prediction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": args.environment,
        "device": str(device),
        "evaluation_pipeline": {
            "description": "Robust nested CV with final holdout",
            "final_holdout_size": args.test_size,
            "k_folds": args.n_splits,
            "inner_val_size": args.val_size,
            "early_stopping_rounds": args.early_stopping_rounds,
            "max_epochs": args.max_epochs
        },
        "data": {
            "total_samples": len(X_array),
            "development_size": len(X_development),
            "final_test_size": len(X_final_test),
            "n_features": X_array.shape[1]
        },
        "best_params": best_params,
        "cross_validation": {
            "mean_roc_auc": float(mean_score),
            "std_roc_auc": float(std_score),
            "fold_scores": [float(s) for s in fold_scores],
            "n_folds": args.n_splits
        },
        "final_test_metrics": final_metrics,
        "hyperparameter_search_time_seconds": search_time,
        "total_time_seconds": total_time
    }
    
    summary_path = out_dir / "nn_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {summary_path}")

    # Upload to HuggingFace
    print_section("📤 Uploading to HuggingFace Hub", "-")
    upload_success = upload_results_to_hf(
        summary=summary,
        output_dir=out_dir,
        model_name="hospital-readmission-nn"
    )
    if not upload_success:
        print("⚠️  Upload skipped (set HF_TOKEN in .env to enable)")

    # Final summary
    print_section("✨ Training Complete!", "=")
    print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📁 All outputs saved to: {out_dir}")
    print(f"\n📊 Performance Summary:")
    print(f"   🔄 {args.n_splits}-Fold CV ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
    print(f"   🎯 Final Test ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print("\n🎉 Ready for deployment!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Train Neural Network for hospital readmission prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: Full hyperparameter search with 5-fold CV
    python train_neural_network.py
    
    # Custom K-fold splits and epochs
    python train_neural_network.py --n-splits 10 --max-epochs 10
    
    # Larger test set
    python train_neural_network.py --test-size 0.3
    
    # Disable early stopping
    python train_neural_network.py --early-stopping-rounds 0
        """
    )
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with features.csv and target.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save model and metrics")
    
    # Training arguments
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Final holdout test size (default: 0.2)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Inner validation size for early stopping (default: 0.1)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of K-fold CV splits (default: 5)")
    parser.add_argument("--early-stopping-rounds", type=int, default=3,
                        help="Early stopping patience (0 to disable, default: 3)")
    parser.add_argument("--max-epochs", type=int, default=5,
                        help="Maximum training epochs (default: 5)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()
    
    # Get repository root
    repo_root = Path(__file__).resolve().parents[1]
    
    # Auto-detect Kaggle environment
    on_kaggle = is_kaggle_environment()
    args.environment = "kaggle" if on_kaggle else "local"
    
    # Auto-detect GPU
    print("🔍 Auto-detecting optimal performance settings...")
    gpu_available = detect_gpu(verbose=True)
    args.use_gpu = gpu_available
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = "/kaggle/working/models" if on_kaggle else str(repo_root / "models")
    
    # Resolve paths
    data_dir = repo_root / args.data_dir
    preprocess_script = repo_root / "phase-1-data-explore-preprocessing" / "simple_preprocessing.py"
    
    features_file = data_dir / "features.csv"
    target_file = data_dir / "target.csv"
    
    # Check for processed data and run preprocessing if needed
    if not features_file.exists() or not target_file.exists():
        if not preprocess_script.exists():
            raise FileNotFoundError(f"Preprocessing script not found: {preprocess_script}")
        run_preprocessing(preprocess_script)
    else:
        print("✅ Processed data found, skipping preprocessing")
    
    # Update data_dir to absolute path
    args.data_dir = str(data_dir)
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()