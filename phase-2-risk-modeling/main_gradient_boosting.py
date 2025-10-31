"""Orchestrator to run preprocessing (optional) and train the gradient boosting model.

This script will:
- check for processed data in `data/processed/` (features.csv & target.csv)
- optionally run phase-1 preprocessing if processed files are missing
- invoke `train_gradient_boosting.py` with sensible defaults or forwarded args

Usage (from repository root):
    python phase-2-risk-modeling/main_gradient_boosting.py

Example:
    python phase-2-risk-modeling/main_gradient_boosting.py --skip-preprocess --skip-search --n-jobs 4
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
    parser = argparse.ArgumentParser(description="Orchestrate preprocessing + LightGBM training")
    parser.add_argument("--skip-preprocess", action="store_true", help="Do not run preprocessing even if processed data missing")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--params-file", type=str, default=None, help="Optional JSON params file to pass to trainer")
    parser.add_argument("--skip-search", action="store_true", help="Pass --skip-search to trainer")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of jobs to pass to trainer")
    parser.add_argument("--run-preprocessing-script", type=str, default="phase-1-data-explore-preprocessing/simple_preprocessing.py", help="Preprocessing script to run if data missing")
    parser.add_argument("--trainer-script", type=str, default="phase-2-risk-modeling/train_gradient_boosting.py", help="Trainer script path")
    parser.add_argument("--extra", type=str, nargs="*", default=[], help="Extra args to forward to trainer script (e.g. --n-iter 10)")

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
    if args.skip_search:
        trainer_args += ["--skip-search"]
    if args.n_jobs:
        trainer_args += ["--n-jobs", str(args.n_jobs)]
    # append any free-form extras provided by user
    if args.extra:
        trainer_args += args.extra

    rc = run_trainer(trainer_script, data_dir, output_dir, trainer_args)
    if rc != 0:
        print(f"Trainer exited with code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
