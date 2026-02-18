#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hourly ENGINE rebuild + retrain loop.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--input", required=True, help="snapshot source path")
    parser.add_argument("--winner-json", default="artifacts/engine_h100_recall_matrix.json")
    parser.add_argument("--dataset-output", default="data/ENGINE_live")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--checkpoint-root", default="data/checkpoints/TLOB")
    parser.add_argument("--retention-prefix", default="ENGINE_live_h100")
    parser.add_argument("--keep-top", type=int, default=3)
    parser.add_argument("--loop-seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def load_winner(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    winner = payload.get("selection", {}).get("winner")
    if winner is None:
        raise ValueError(f"winner not found in `{path}`")
    return winner


def build_live_dataset(args: argparse.Namespace, alpha_mult: float) -> None:
    cmd = [
        args.python,
        "scripts/build_engine_dataset.py",
        "--input",
        args.input,
        "--output-dir",
        args.dataset_output,
        "--normalize",
        "--strict-quality",
        "--workers",
        str(args.workers),
        "--alpha-mult",
        f"{alpha_mult:.2f}",
    ]
    run_cmd(cmd)


def run_training(args: argparse.Namespace, winner: dict, dir_ckpt: str) -> Path:
    loss_name = winner.get("loss_name", "weighted_ce")
    use_sampler = bool(winner.get("use_weighted_sampler", False))
    sampler_pow = float(winner.get("weighted_sampler_pow", 1.0))
    cmd = [
        args.python,
        "main.py",
        "+model=tlob",
        "+dataset=engine",
        "hydra.job.chdir=False",
        "experiment.is_wandb=False",
        "experiment.is_data_preprocessed=True",
        "experiment.is_sweep=False",
        f"experiment.max_epochs={args.max_epochs}",
        f"experiment.seed={args.seed}",
        "experiment.horizon=100",
        f"experiment.loss_name={loss_name}",
        "experiment.min_event_precision=0.20",
        f"experiment.dir_ckpt={dir_ckpt}",
        f"dataset.batch_size={args.batch_size}",
        f"dataset.data_path={args.dataset_output}",
        f"dataset.use_weighted_sampler={'true' if use_sampler else 'false'}",
        f"dataset.weighted_sampler_pow={sampler_pow}",
        f"model.hyperparameters_fixed.seq_size={args.seq_size}",
        f"model.hyperparameters_fixed.lr={args.lr}",
        "model.hyperparameters_fixed.num_layers=4",
        "model.hyperparameters_fixed.num_heads=1",
        "model.hyperparameters_fixed.all_features=True",
    ]
    run_cmd(cmd)
    return Path(args.checkpoint_root) / dir_ckpt / "metrics_summary.json"


def prune_checkpoints(args: argparse.Namespace, latest_dir: str) -> None:
    ckpt_root = Path(args.checkpoint_root)
    metrics_files = list(ckpt_root.glob(f"{args.retention_prefix}*/metrics_summary.json"))
    rows = []
    for path in metrics_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            flat = payload.get("flat_metrics", {})
            rows.append(
                {
                    "dir": path.parent.name,
                    "path": path,
                    "val_event_recall": float(flat.get("val_event_recall", 0.0)),
                }
            )
        except Exception:
            continue
    rows = sorted(rows, key=lambda r: -r["val_event_recall"])
    keep = {latest_dir}
    keep.update([r["dir"] for r in rows[: max(args.keep_top, 0)]])

    for row in rows:
        d = ckpt_root / row["dir"]
        if row["dir"] not in keep and d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Pruned checkpoint dir: {d}")


def run_cycle(args: argparse.Namespace) -> None:
    winner = load_winner(args.winner_json)
    alpha_mult = float(winner.get("alpha_mult", 0.5))
    ts = time.strftime("%Y%m%d_%H%M%S")
    dir_ckpt = f"{args.retention_prefix}_{ts}"

    build_live_dataset(args, alpha_mult)
    summary_path = run_training(args, winner, dir_ckpt)
    print(f"Cycle summary: {summary_path}")
    prune_checkpoints(args, latest_dir=dir_ckpt)


def main() -> int:
    args = parse_args()
    if args.once:
        run_cycle(args)
        return 0

    while True:
        start = time.time()
        run_cycle(args)
        elapsed = time.time() - start
        sleep_for = max(0, args.loop_seconds - int(elapsed))
        print(f"Sleeping {sleep_for}s until next cycle")
        time.sleep(sleep_for)


if __name__ == "__main__":
    raise SystemExit(main())
