#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.engine_recall import select_best_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ENGINE h100 recall experiment matrix.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--input",
        required=True,
        help="snapshot input path (jsonl/jsonl.gz file or folder)",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--alpha-values", default="0.50,0.35,0.20")
    parser.add_argument("--sampler-pow", type=float, default=1.0)
    parser.add_argument("--min-event-precision", type=float, default=0.20)
    parser.add_argument("--fallback-min-event-precision", type=float, default=0.15)
    parser.add_argument("--results-json", default="artifacts/engine_h100_recall_matrix.json")
    parser.add_argument("--no-baseline", dest="run_baseline", action="store_false")
    parser.set_defaults(run_baseline=True)
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def read_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing metrics summary `{path}`")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    flat = payload.get("flat_metrics", {})
    return {
        "metrics_path": str(path),
        "dir_ckpt": payload.get("dir_ckpt"),
        "dataset": payload.get("dataset"),
        "data_path": payload.get("data_path"),
        "loss_name": payload.get("loss_name"),
        "use_weighted_sampler": payload.get("use_weighted_sampler"),
        "weighted_sampler_pow": payload.get("weighted_sampler_pow"),
        "val_event_precision": float(flat.get("val_event_precision", 0.0)),
        "val_event_recall": float(flat.get("val_event_recall", 0.0)),
        "val_event_f1": float(flat.get("val_event_f1", 0.0)),
        "val_loss": float(flat.get("val_loss", 1e9)),
        "test_event_precision": float(flat.get("test_event_precision", 0.0)),
        "test_event_recall": float(flat.get("test_event_recall", 0.0)),
        "test_event_f1": float(flat.get("test_event_f1", 0.0)),
    }


def build_dataset(args: argparse.Namespace, alpha: float, dataset_dir: Path) -> None:
    cmd = [
        args.python,
        "scripts/build_engine_dataset.py",
        "--input",
        args.input,
        "--output-dir",
        str(dataset_dir),
        "--normalize",
        "--strict-quality",
        "--workers",
        str(args.workers),
        "--alpha-mult",
        f"{alpha:.2f}",
    ]
    run_cmd(cmd)


def train_run(
    args: argparse.Namespace,
    dataset_dir: Path,
    dir_ckpt: str,
    loss_name: str,
    use_weighted_sampler: bool,
) -> dict:
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
        f"experiment.dir_ckpt={dir_ckpt}",
        f"experiment.min_event_precision={args.min_event_precision}",
        f"dataset.batch_size={args.batch_size}",
        f"dataset.use_weighted_sampler={'true' if use_weighted_sampler else 'false'}",
        f"dataset.weighted_sampler_pow={args.sampler_pow}",
        f"dataset.data_path={dataset_dir}",
        f"model.hyperparameters_fixed.seq_size={args.seq_size}",
        f"model.hyperparameters_fixed.lr={args.lr}",
        "model.hyperparameters_fixed.num_layers=4",
        "model.hyperparameters_fixed.num_heads=1",
        "model.hyperparameters_fixed.all_features=True",
    ]
    run_cmd(cmd)
    metrics_path = Path("data/checkpoints/TLOB") / dir_ckpt / "metrics_summary.json"
    return read_summary(metrics_path)


def main() -> int:
    args = parse_args()
    alphas = [float(x.strip()) for x in args.alpha_values.split(",") if x.strip()]
    if not alphas:
        raise ValueError("no alpha values configured")

    results: dict[str, object] = {
        "config": vars(args),
        "datasets": [],
        "runs": [],
    }
    run_rows: list[dict] = []

    for alpha in alphas:
        tag = f"a{int(round(alpha * 100)):03d}"
        dataset_dir = Path(args.output_root) / f"ENGINE_{tag}"
        build_dataset(args, alpha, dataset_dir)
        results["datasets"].append({"alpha_mult": alpha, "dataset_dir": str(dataset_dir)})

    if args.run_baseline:
        baseline_alpha = alphas[0]
        baseline_tag = f"a{int(round(baseline_alpha * 100)):03d}"
        baseline_dataset = Path(args.output_root) / f"ENGINE_{baseline_tag}"
        baseline_ckpt = f"ENGINE_h100_baseline_{baseline_tag}_seed{args.seed}"
        baseline_run = train_run(
            args=args,
            dataset_dir=baseline_dataset,
            dir_ckpt=baseline_ckpt,
            loss_name="ce",
            use_weighted_sampler=False,
        )
        baseline_run["alpha_mult"] = baseline_alpha
        baseline_run["variant"] = "baseline_ce_no_sampler"
        run_rows.append(baseline_run)

    for alpha in alphas:
        tag = f"a{int(round(alpha * 100)):03d}"
        dataset_dir = Path(args.output_root) / f"ENGINE_{tag}"
        variants = [
            ("weighted_ce_no_sampler", "weighted_ce", False),
            ("weighted_ce_sampler", "weighted_ce", True),
        ]
        for variant_name, loss_name, use_sampler in variants:
            ckpt = f"ENGINE_h100_{tag}_{variant_name}_seed{args.seed}"
            run_info = train_run(
                args=args,
                dataset_dir=dataset_dir,
                dir_ckpt=ckpt,
                loss_name=loss_name,
                use_weighted_sampler=use_sampler,
            )
            run_info["alpha_mult"] = alpha
            run_info["variant"] = variant_name
            run_rows.append(run_info)

    winner, threshold = select_best_run(
        run_rows,
        min_event_precision=args.min_event_precision,
        fallback_min_event_precision=args.fallback_min_event_precision,
    )
    ranked = sorted(
        run_rows,
        key=lambda r: (
            -float(r.get("val_event_recall", 0.0)),
            -float(r.get("val_event_f1", 0.0)),
            float(r.get("val_loss", 1e9)),
        ),
    )
    results["runs"] = run_rows
    results["selection"] = {
        "guard_threshold_used": threshold,
        "winner": winner,
        "ranked_runs": ranked,
    }

    out = Path(args.results_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["selection"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
