"""Orchestrator to run preprocessing (optional) and train the neural network model.

This script will:
- check for processed data in `data/processed/` (features.csv & target.csv)
- optionally run phase-1 preprocessing if processed files are missing
- invoke `neural_network_model.py` with sensible defaults or forwarded args

Usage (from repository root):
    python phase-2-risk-modeling/train_neural_network.py

Example:
    python phase-2-risk-modeling/train_neural_network.py --skip-preprocess --fast-mode --use-gpu
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_preprocessing(preprocess_script: Path) -> None:
    print(f"Running preprocessing: {preprocess_script}")
    subprocess.run([sys.executable, str(preprocess_script)], check=True)


def run_trainer(trainer_script: Path, data_dir: Path, output_dir: Path, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(trainer_script), "--data-dir", str(data_dir), "--output-dir", str(output_dir)]
    cmd += extra_args
    print("Running trainer with command:")
    print(" ".join(cmd))
    # stream to console
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Orchestrate preprocessing + Neural Network training")
    parser.add_argument("--skip-preprocess", action="store_true", help="Do not run preprocessing even if processed data missing")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--params-file", type=str, default=None, help="Optional JSON params file to pass to trainer")
    parser.add_argument("--fast-mode", action="store_true", help="Pass --fast-mode to trainer (50 epochs instead of 100)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--run-preprocessing-script", type=str, default="phase-1-data-explore-preprocessing/simple_preprocessing.py", help="Preprocessing script to run if data missing")
    parser.add_argument("--trainer-script", type=str, default="phase-2-risk-modeling/neural_network_model.py", help="Trainer script path")
    parser.add_argument("--extra", type=str, nargs="*", default=[], help="Extra args to forward to trainer script (e.g. --test-size 0.3)")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    data_dir = repo_root / args.data_dir
    output_dir = repo_root / args.output_dir
    preprocess_script = repo_root / args.run_preprocessing_script
    trainer_script = repo_root / args.trainer_script

    features_file = data_dir / "features.csv"
    target_file = data_dir / "target.csv"

    if not features_file.exists() or not target_file.exists():
        if args.skip_preprocess:
            raise FileNotFoundError(f"Processed data not found in {data_dir} and --skip-preprocess was set.")
        if not preprocess_script.exists():
            raise FileNotFoundError(f"Preprocessing script not found: {preprocess_script}")
        run_preprocessing(preprocess_script)

    # Build extra args for trainer
    trainer_args: list[str] = []
    if args.params_file:
        trainer_args += ["--params-file", str(Path(args.params_file).resolve())]
    if args.fast_mode:
        trainer_args += ["--fast-mode"]
    if args.use_gpu:
        trainer_args += ["--use-gpu"]
    if args.mixed_precision:
        trainer_args += ["--mixed-precision"]
    # append any free-form extras provided by user
    if args.extra:
        trainer_args += args.extra

    rc = run_trainer(trainer_script, data_dir, output_dir, trainer_args)
    if rc != 0:
        print(f"Trainer exited with code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
