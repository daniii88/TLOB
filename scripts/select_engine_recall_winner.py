#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.engine_recall import select_best_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best ENGINE recall run.")
    parser.add_argument(
        "--metrics-glob",
        default="data/checkpoints/TLOB/*/metrics_summary.json",
        help="glob pattern for metrics_summary.json files",
    )
    parser.add_argument(
        "--output",
        default="artifacts/engine_h100_recall_winner.json",
        help="output winner json path",
    )
    parser.add_argument("--min-event-precision", type=float, default=0.20)
    parser.add_argument("--fallback-min-event-precision", type=float, default=0.15)
    return parser.parse_args()


def load_runs(metrics_glob: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(metrics_glob)):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        flat = payload.get("flat_metrics", {})
        runs.append(
            {
                "metrics_path": path,
                "dir_ckpt": payload.get("dir_ckpt"),
                "val_event_precision": float(flat.get("val_event_precision", 0.0)),
                "val_event_recall": float(flat.get("val_event_recall", 0.0)),
                "val_event_f1": float(flat.get("val_event_f1", 0.0)),
                "val_loss": float(flat.get("val_loss", 1e9)),
                "test_event_precision": float(flat.get("test_event_precision", 0.0)),
                "test_event_recall": float(flat.get("test_event_recall", 0.0)),
                "test_event_f1": float(flat.get("test_event_f1", 0.0)),
            }
        )
    return runs


def main() -> int:
    args = parse_args()
    runs = load_runs(args.metrics_glob)
    if not runs:
        raise FileNotFoundError(f"no metrics summaries found for `{args.metrics_glob}`")

    winner, threshold = select_best_run(
        runs,
        min_event_precision=args.min_event_precision,
        fallback_min_event_precision=args.fallback_min_event_precision,
    )
    ranked = sorted(
        runs,
        key=lambda r: (
            -float(r.get("val_event_recall", 0.0)),
            -float(r.get("val_event_f1", 0.0)),
            float(r.get("val_loss", 1e9)),
        ),
    )
    result = {
        "guard_threshold_used": threshold,
        "winner": winner,
        "ranked_runs": ranked,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
