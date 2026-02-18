#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test ENGINE training modes.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--rows", type=int, default=1024)
    return p.parse_args()


def create_dummy_split(path: Path, rows: int, feat_dim: int = 31) -> None:
    x = np.random.randn(rows, feat_dim).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=(rows, 4), p=[0.1, 0.8, 0.1]).astype(np.float32)
    arr = np.concatenate([x, labels], axis=1)
    np.save(path, arr)


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def run_mode(py: str, data_path: Path, loss_name: str, use_sampler: bool) -> None:
    tag = "sampler" if use_sampler else "nosampler"
    dir_ckpt = f"SMOKE_h100_{loss_name}_{tag}"
    cmd = [
        py,
        "main.py",
        "+model=tlob",
        "+dataset=engine",
        "hydra.job.chdir=False",
        "experiment.is_wandb=False",
        "experiment.is_data_preprocessed=True",
        "experiment.is_debug=True",
        "experiment.max_epochs=1",
        "experiment.horizon=100",
        f"experiment.loss_name={loss_name}",
        f"experiment.dir_ckpt={dir_ckpt}",
        "experiment.min_event_precision=0.20",
        "experiment.seed=42",
        "dataset.batch_size=64",
        f"dataset.use_weighted_sampler={'true' if use_sampler else 'false'}",
        "dataset.weighted_sampler_pow=1.0",
        f"dataset.data_path={data_path}",
        "model.hyperparameters_fixed.seq_size=128",
        "model.hyperparameters_fixed.lr=0.0001",
        "model.hyperparameters_fixed.num_layers=4",
        "model.hyperparameters_fixed.num_heads=1",
        "model.hyperparameters_fixed.all_features=True",
    ]
    run_cmd(cmd)


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="engine_smoke_") as td:
        data_dir = Path(td)
        create_dummy_split(data_dir / "train.npy", args.rows)
        create_dummy_split(data_dir / "val.npy", args.rows // 4)
        create_dummy_split(data_dir / "test.npy", args.rows // 4)

        run_mode(args.python, data_dir, loss_name="ce", use_sampler=False)
        run_mode(args.python, data_dir, loss_name="weighted_ce", use_sampler=False)
        run_mode(args.python, data_dir, loss_name="weighted_ce", use_sampler=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
