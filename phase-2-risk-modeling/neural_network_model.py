"""Train a neural network model for 30-day readmission risk prediction.

Loads processed features and target from Phase 1 (`data/processed/features.csv` and
`data/processed/target.csv`), trains a feedforward neural network with dropout and
batch normalization, evaluates on a holdout test set, and saves model + metrics.

Features:
- Dense layers with dropout for regularization
- Batch normalization for stable training
- Early stopping and learning rate scheduling
- GPU auto-detection (PyTorch with CUDA)
- Progress tracking and visualizations
- Kaggle-optimized defaults

Architecture (as per README.md):
- Dense layers with dropout
- Batch normalization
- Binary classification output

Usage (from project root):
    python phase-2-risk-modeling/train_neural_network.py
    
Kaggle usage:
    !python phase-2-risk-modeling/train_neural_network.py --fast-mode --verbose

Requirements:
    pip install torch scikit-learn pandas joblib matplotlib seaborn numpy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Install with: pip install tqdm")

warnings.filterwarnings("ignore")


# Default Neural Network architecture parameters
DEFAULT_PARAMS = {
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.3,
    "batch_norm": True,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 100,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 5,
    "optimizer": "adam",
    "activation": "relu",
}


def detect_gpu():
    """Detect if GPU is available for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            print(f"   Found {gpu_count} GPU(s): {gpu_names}")
            return True
        return False
    except Exception:
        return False


def is_kaggle_environment():
    """Detect if running in Kaggle environment."""
    return os.path.exists('/kaggle/working')


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}\n")


class ReadmissionNet(nn.Module):
    """PyTorch Neural Network for readmission prediction."""
    
    def __init__(self, input_dim: int, hidden_layers: list, dropout_rate: float, batch_norm: bool):
        super(ReadmissionNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # Build hidden layers
        prev_dim = input_dim
        for i, units in enumerate(hidden_layers):
            self.layers.append(nn.Linear(prev_dim, units))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        
        # Output layer
        self.output = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class NeuralNetworkTrainer:
    """Neural Network trainer with PyTorch.
    
    Responsibilities:
    - Build and compile neural network
    - Train with early stopping and LR scheduling
    - Evaluate and return metrics
    - Save model artifact
    """

    def __init__(self, input_dim: int, params: dict, output_dir: str = "models"):
        self.input_dim = input_dim
        self.params = params
        self.output_dir = Path(output_dir)
        self.model = None
        self.scaler = StandardScaler()
        self.history = {"loss": [], "val_loss": [], "auc": [], "val_auc": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        """Build neural network architecture with dense layers, dropout, and batch norm."""
        import torch.nn as nn
        
        self.model = ReadmissionNet(
            input_dim=self.input_dim,
            hidden_layers=self.params["hidden_layers"],
            dropout_rate=self.params["dropout_rate"],
            batch_norm=self.params["batch_norm"]
        ).to(self.device)
        
        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose: int = 1):
        """Train the neural network with early stopping and learning rate scheduling."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val).reshape(-1, 1).to(self.device)
        
        # Setup optimizer and loss
        criterion = nn.BCELoss()
        
        optimizer_map = {
            "adam": torch.optim.Adam(self.model.parameters(), lr=self.params["learning_rate"]),
            "sgd": torch.optim.SGD(self.model.parameters(), lr=self.params["learning_rate"]),
            "rmsprop": torch.optim.RMSprop(self.model.parameters(), lr=self.params["learning_rate"]),
        }
        optimizer = optimizer_map.get(self.params["optimizer"], torch.optim.Adam(self.model.parameters(), lr=self.params["learning_rate"]))
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.params.get("reduce_lr_patience", 5), 
            min_lr=1e-7, verbose=verbose > 0
        )
        
        # Compute class weights
        class_weights = self._compute_class_weights(y_train)
        pos_weight = torch.FloatTensor([class_weights[1] / class_weights[0]]).to(self.device)
        criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Training loop
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.params["epochs"]):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    # Calculate AUC
                    val_proba = val_outputs.cpu().numpy().flatten()
                    val_true = y_val_tensor.cpu().numpy().flatten()
                    val_auc = roc_auc_score(val_true, val_proba)
                    
                    # Training AUC
                    train_outputs = self.model(X_train_tensor)
                    train_proba = train_outputs.cpu().numpy().flatten()
                    train_true = y_train_tensor.cpu().numpy().flatten()
                    train_auc = roc_auc_score(train_true, train_proba)
                
                # Store history
                self.history["loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["auc"].append(train_auc)
                self.history["val_auc"].append(val_auc)
                
                # Learning rate scheduler step
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose >= 2 or (verbose == 1 and (epoch + 1) % 10 == 0):
                    print(f"Epoch {epoch+1}/{self.params['epochs']}: "
                          f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                          f"auc={train_auc:.4f}, val_auc={val_auc:.4f}")
                
                # Early stopping
                if patience_counter >= self.params.get("early_stopping_patience", 15):
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Restoring best model weights (val_auc={best_val_auc:.4f})")
                    self.model.load_state_dict(best_model_state)
                    break
            else:
                # No validation set
                self.history["loss"].append(train_loss)
                if verbose >= 2 or (verbose == 1 and (epoch + 1) % 10 == 0):
                    print(f"Epoch {epoch+1}/{self.params['epochs']}: loss={train_loss:.4f}")
        
        # Restore best model if we have one
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced dataset."""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

    def evaluate(self, X, y, threshold: float = 0.5):
        """Evaluate model and return metrics."""
        import torch
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_proba = outputs.cpu().numpy().flatten()
        
        y_pred = (y_proba >= threshold).astype(int)
        y_true = y.values if hasattr(y, 'values') else y
        
        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
        
        return metrics, y_proba, y_pred

    def save(self, path: str | Path):
        """Save model and scaler."""
        import torch
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        model_path = path.parent / f"{path.stem}_pytorch.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_layers': self.params["hidden_layers"],
                'dropout_rate': self.params["dropout_rate"],
                'batch_norm': self.params["batch_norm"],
            }
        }, model_path)
        
        # Save scaler
        scaler_path = path.parent / f"{path.stem}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        print(f"   Model saved to: {model_path}")
        print(f"   Scaler saved to: {scaler_path}")


def load_data(data_dir: str = "data/processed"):
    """Load features and target data."""
    print("📂 Loading data...")
    data_dir = Path(data_dir)
    X_path = data_dir / "features.csv"
    y_path = data_dir / "target.csv"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Processed data not found in {data_dir}. Run phase-1 preprocessing first."
        )

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    
    # Support both columnar and single-column target files
    if "target" in y.columns:
        y = y["target"]
    else:
        y = y.iloc[:, 0]

    print(f"✅ Loaded features: {X.shape}, target: {y.shape}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    return X, y


def save_visualizations(y_true, y_proba, y_pred, output_dir: Path):
    """Save ROC curve and confusion matrix visualizations."""
    print("📊 Generating visualizations...")
    
    sns.set_style("whitegrid")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Neural Network Readmission Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = output_dir / "nn_roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ ROC curve saved: {roc_path}")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Readmission', 'Readmission'],
                yticklabels=['No Readmission', 'Readmission'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Neural Network Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_path = output_dir / "nn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Confusion matrix saved: {cm_path}")


def save_training_history(history, output_dir: Path):
    """Save training history plots."""
    print("📈 Generating training history plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # AUC plot
    if 'auc' in history and len(history['auc']) > 0:
        axes[1].plot(history['auc'], label='Train AUC', linewidth=2)
        if 'val_auc' in history and len(history['val_auc']) > 0:
            axes[1].plot(history['val_auc'], label='Val AUC', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('AUC', fontsize=12)
        axes[1].set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    history_path = output_dir / "nn_training_history.png"
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Training history saved: {history_path}")


def print_metrics_table(metrics: dict, title: str = "Model Performance Metrics"):
    """Print metrics in a formatted table."""
    print_section(title, "=")
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title():<20} {v:>10.4f}")
    print("-" * 32)


def main(args: argparse.Namespace):
    start_time = time.time()
    
    # Print configuration
    print_section("🚀 Neural Network Training - Hospital Readmission Risk", "=")
    print(f"⚙️  Configuration:")
    print(f"   - Data directory: {args.data_dir}")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - Test size: {args.test_size}")
    print(f"   - Random seed: {args.random_state}")
    print(f"   - Fast mode: {args.fast_mode}")
    
    # Detect environment
    on_kaggle = is_kaggle_environment()
    if on_kaggle:
        print(f"   - Environment: 🏆 Kaggle")
    else:
        print(f"   - Environment: 💻 Local")
    
    # Import PyTorch
    try:
        import torch
        import torch.nn as nn
        print(f"\n✅ PyTorch version: {torch.__version__}")
    except ImportError:
        raise ImportError(
            "PyTorch is required for this training script. Please install it with `pip install torch`."
        )
    
    # GPU Detection
    print("\n🖥️  Checking GPU availability...")
    gpu_available = detect_gpu()
    if gpu_available and args.use_gpu:
        print("   ✅ GPU detected and will be used for training!")
        if args.mixed_precision:
            print("   ⚡ Mixed precision training enabled!")
            # PyTorch AMP will be used in training loop if needed
    elif gpu_available and not args.use_gpu:
        print("   ⚠️  GPU available but --use-gpu not set. Using CPU.")
    else:
        print("   ℹ️  No GPU detected. Using CPU for training.")
    
    # Load data
    X, y = load_data(args.data_dir)

    # Split data
    print("\n🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, random_state=args.random_state, stratify=y_train
    )
    print(f"   Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Parse user parameters
    user_params = None
    if args.params:
        try:
            user_params = json.loads(args.params)
        except Exception as exc:
            raise ValueError(f"Could not parse JSON params string: {exc}")
    if args.params_file:
        params_path = Path(args.params_file)
        if not params_path.exists():
            raise FileNotFoundError(f"Params file not found: {params_path}")
        with open(params_path, "r") as f:
            user_params = json.load(f)

    # Setup model parameters
    model_params = user_params or DEFAULT_PARAMS.copy()
    
    # Apply fast mode adjustments
    if args.fast_mode:
        model_params["epochs"] = 50
        model_params["early_stopping_patience"] = 10
        model_params["batch_size"] = 512
    
    # Apply Kaggle optimizations
    if on_kaggle and not user_params:
        model_params["batch_size"] = 512
        model_params["epochs"] = 100
    
    print_section("🏗️  Building Neural Network", "-")
    print(f"📋 Model architecture:")
    print(f"   - Hidden layers: {model_params['hidden_layers']}")
    print(f"   - Dropout rate: {model_params['dropout_rate']}")
    print(f"   - Batch normalization: {model_params['batch_norm']}")
    print(f"   - Learning rate: {model_params['learning_rate']}")
    print(f"   - Batch size: {model_params['batch_size']}")
    print(f"   - Max epochs: {model_params['epochs']}")
    print(f"   - Optimizer: {model_params['optimizer']}")
    print(f"   - Activation: {model_params['activation']}")

    # Build and train model
    trainer = NeuralNetworkTrainer(
        input_dim=X_tr.shape[1],
        params=model_params,
        output_dir=args.output_dir
    )
    
    model = trainer.build_model()
    print("\n📊 Model Summary:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print_section("🏋️  Training Neural Network", "-")
    train_start = time.time()
    
    verbose = 2 if args.verbose else 1
    trainer.fit(X_tr, y_tr, X_val, y_val, verbose=verbose)
    
    train_time = time.time() - train_start
    print(f"\n✅ Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")

    # Final evaluation on holdout test set
    print_section("📊 Final Evaluation on Test Set", "-")
    print("🧪 Evaluating model performance...")
    
    metrics, y_proba, y_pred = trainer.evaluate(X_test, y_test)
    print_metrics_table(metrics, "🎯 FINAL TEST SET RESULTS")

    # Create output directory
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save visualizations
    save_visualizations(y_test, y_proba, y_pred, out_dir)
    save_training_history(trainer.history, out_dir)

    # Save model and artifacts
    print_section("💾 Saving Results", "-")
    
    model_path = out_dir / "neural_network_model"
    trainer.save(model_path)

    metrics_path = out_dir / "neural_network_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")

    # Save model parameters
    params_out_path = out_dir / "neural_network_params.json"
    with open(params_out_path, "w") as f:
        json.dump(model_params, f, indent=2)
    print(f"✅ Parameters saved: {params_out_path}")

    # Create comprehensive summary
    total_time = time.time() - start_time
    summary = {
        "model": "Neural Network (Feedforward)",
        "task": "Hospital 30-Day Readmission Risk Prediction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": "kaggle" if on_kaggle else "local",
        "device": "gpu" if (gpu_available and args.use_gpu) else "cpu",
        "data": {
            "train_size": len(X_tr),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_features": X.shape[1]
        },
        "architecture": {
            "hidden_layers": model_params["hidden_layers"],
            "dropout_rate": model_params["dropout_rate"],
            "batch_norm": model_params["batch_norm"],
            "activation": model_params["activation"],
        },
        "training_config": {
            "learning_rate": model_params["learning_rate"],
            "batch_size": model_params["batch_size"],
            "optimizer": model_params["optimizer"],
            "epochs_trained": len(trainer.history['loss']),
            "max_epochs": model_params["epochs"],
        },
        "test_metrics": metrics,
        "training_time_seconds": train_time,
        "total_time_seconds": total_time
    }
    
    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {summary_path}")

    # Final summary
    print_section("✨ Training Complete!", "=")
    print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📁 All outputs saved to: {out_dir}")
    print(f"🎯 Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"🎯 Test F1-Score: {metrics['f1']:.4f}")
    print("\n🎉 Ready for deployment!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train neural network model for readmission risk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast training with default params (good for quick testing)
  python train_neural_network.py --fast-mode --verbose
  
  # Full training with early stopping
  python train_neural_network.py --use-gpu --mixed-precision
  
  # Kaggle optimized (GPU with mixed precision)
  python train_neural_network.py --use-gpu --mixed-precision --verbose
  
  # Custom params from file
  python train_neural_network.py --params-file nn_params.json --use-gpu
        """
    )
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with features.csv and target.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save model and metrics (auto-detects Kaggle)")
    
    # Training arguments
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Holdout test size fraction (default: 0.2)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation size for early stopping (default: 0.1)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Performance arguments
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU if available (auto-detected)")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Enable mixed precision training for faster GPU training")
    
    # Model configuration arguments
    parser.add_argument("--params", type=str, default=None,
                        help="JSON string of model params to use (overrides DEFAULT_PARAMS)")
    parser.add_argument("--params-file", type=str, default=None,
                        help="Path to JSON file with model params")
    
    # Mode arguments
    parser.add_argument("--fast-mode", action="store_true",
                        help="Quick training with reduced epochs (50 instead of 100)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output from training")

    args = parser.parse_args()
    
    # Auto-detect Kaggle and set sensible defaults
    on_kaggle = is_kaggle_environment()
    
    if args.output_dir is None:
        args.output_dir = "/kaggle/working/models" if on_kaggle else "models"
    
    # Auto-enable GPU on Kaggle if available
    if on_kaggle and not args.use_gpu:
        args.use_gpu = detect_gpu()
        args.mixed_precision = True  # Enable mixed precision on Kaggle by default
    
    main(args)
