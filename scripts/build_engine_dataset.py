#!/usr/bin/env python3
"""Build TLOB-compatible ENGINE dataset from shadow entry-signal JSONL export.

Focus: keep only usable samples via strict quality gates and output a health report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from collections import Counter
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Sequence, Tuple

import numpy as np


FEATURE_NAMES = [
    "ask1_price",
    "ask1_qty",
    "bid1_price",
    "bid1_qty",
    "ask2_price",
    "ask2_qty",
    "bid2_price",
    "bid2_qty",
    "ask3_price",
    "ask3_qty",
    "bid3_price",
    "bid3_qty",
    "ask4_price",
    "ask4_qty",
    "bid4_price",
    "bid4_qty",
    "ask5_price",
    "ask5_qty",
    "bid5_price",
    "bid5_qty",
    "qi_top1",
    "qi_top2",
    "qi_top3",
    "qi_top4",
    "qi_top5",
    "ofi_250ms",
    "ofi_1s",
    "ofi_3s",
    "microprice_minus_mid",
    "spread_ticks",
    "depth_ratio",
]

DEFAULT_STRICT_EXCLUDE_TYPED_REASONS = {
    "no_signal:missing_best_bid",
    "no_signal:missing_best_ask",
    "no_signal:crossed_book",
    "no_signal:invalid_top_of_book_liquidity",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        required=True,
        help="input source: JSONL file, JSONL.GZ file, or directory containing snapshot files",
    )
    p.add_argument("--output-dir", default="data/ENGINE", help="output directory for npy files")
    p.add_argument("--market", default="", help="optional market filter, e.g. BTC-USD")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--len-smooth", type=int, default=10)
    p.add_argument("--alpha-mult", type=float, default=0.5)
    p.add_argument("--normalize", action="store_true", help="z-score features from train split")
    p.add_argument(
        "--strict-quality",
        dest="strict_quality",
        action="store_true",
        default=True,
        help="enable strict usable-data filters (default: enabled)",
    )
    p.add_argument(
        "--no-strict-quality",
        dest="strict_quality",
        action="store_false",
        help="disable strict usable-data filters",
    )
    p.add_argument("--max-spread-ticks", type=float, default=80.0)
    p.add_argument("--min-depth-ratio", type=float, default=0.02)
    p.add_argument("--max-depth-ratio", type=float, default=50.0)
    p.add_argument("--require-signal-seen", action="store_true")
    p.add_argument(
        "--exclude-typed-reason",
        action="append",
        default=[],
        help="typed_reason to exclude; repeatable",
    )
    p.add_argument(
        "--quality-report-name",
        default="quality_report.json",
        help="quality report filename under output-dir",
    )
    p.add_argument("--min-rows", type=int, default=1000)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="number of CPU worker processes for scanning input files",
    )
    return p.parse_args()


def resolve_input_path(raw_input: str) -> Path:
    input_path = Path(raw_input)
    if input_path.exists():
        return input_path
    if raw_input.startswith("/code/"):
        fallback = Path("/home/daniii") / raw_input.lstrip("/")
        if fallback.exists():
            print(f"Input path `{raw_input}` not found, using `{fallback}`")
            return fallback
    raise FileNotFoundError(f"input path `{raw_input}` does not exist")


def collect_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    input_files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"))
    )
    if not input_files:
        raise FileNotFoundError(f"no .jsonl/.jsonl.gz files found under `{input_path}`")
    return input_files


def iter_json_lines(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield line


def row_order_key(row: Dict) -> Tuple[int, int]:
    try:
        timestamp_ms = int(row.get("timestamp_ms", 0))
    except (TypeError, ValueError):
        timestamp_ms = 0
    try:
        block = int(row.get("block", 0))
    except (TypeError, ValueError):
        block = 0
    return timestamp_ms, block


def scan_input_file(
    input_file: str,
    strict_quality: bool,
    market_filter: str,
    require_signal_seen: bool,
    excluded_reasons: set[str],
    max_spread_ticks: float,
    min_depth_ratio: float,
    max_depth_ratio: float,
) -> Dict:
    raw_total = 0
    filtered = Counter()
    typed_reason_total = Counter()
    kept_typed_reasons = Counter()
    usable_rows = 0
    ordering_violations = 0
    first_key: Tuple[int, int] | None = None
    last_key: Tuple[int, int] | None = None

    for line in iter_json_lines(Path(input_file)):
        line = line.strip()
        if not line:
            continue
        raw_total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            filtered["invalid_json"] += 1
            continue
        typed_reason_total[str(row.get("typed_reason", "n/a"))] += 1
        filter_reason = validate_row(
            row=row,
            strict_quality=strict_quality,
            market_filter=market_filter,
            require_signal_seen=require_signal_seen,
            excluded_reasons=excluded_reasons,
            max_spread_ticks=max_spread_ticks,
            min_depth_ratio=min_depth_ratio,
            max_depth_ratio=max_depth_ratio,
        )
        if filter_reason is not None:
            filtered[filter_reason] += 1
            continue
        feature, mid = row_to_feature(row)
        if not np.isfinite(feature).all() or not np.isfinite(mid) or mid <= 0:
            filtered["non_finite_feature_or_mid"] += 1
            continue
        current_key = row_order_key(row)
        if first_key is None:
            first_key = current_key
        if last_key is not None and current_key < last_key:
            ordering_violations += 1
        last_key = current_key
        kept_typed_reasons[str(row.get("typed_reason", "n/a"))] += 1
        usable_rows += 1

    return {
        "file": input_file,
        "raw_total": raw_total,
        "filtered": dict(filtered),
        "typed_reason_total": dict(typed_reason_total),
        "kept_typed_reasons": dict(kept_typed_reasons),
        "usable_rows": usable_rows,
        "ordering_violations": ordering_violations,
        "first_key": first_key,
        "last_key": last_key,
    }


def materialize_input_file(
    input_file: str,
    expected_rows: int,
    feature_dim: int,
    features_out_path: str,
    mid_out_path: str,
    strict_quality: bool,
    market_filter: str,
    require_signal_seen: bool,
    excluded_reasons: set[str],
    max_spread_ticks: float,
    min_depth_ratio: float,
    max_depth_ratio: float,
) -> Dict:
    features = np.memmap(
        features_out_path, dtype=np.float32, mode="w+", shape=(expected_rows, feature_dim)
    )
    mids = np.memmap(mid_out_path, dtype=np.float64, mode="w+", shape=(expected_rows,))

    idx = 0
    for line in iter_json_lines(Path(input_file)):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        filter_reason = validate_row(
            row=row,
            strict_quality=strict_quality,
            market_filter=market_filter,
            require_signal_seen=require_signal_seen,
            excluded_reasons=excluded_reasons,
            max_spread_ticks=max_spread_ticks,
            min_depth_ratio=min_depth_ratio,
            max_depth_ratio=max_depth_ratio,
        )
        if filter_reason is not None:
            continue
        feature, mid_value = row_to_feature(row)
        if not np.isfinite(feature).all() or not np.isfinite(mid_value) or mid_value <= 0:
            continue
        if idx >= expected_rows:
            raise RuntimeError(
                f"more rows than expected while materializing `{input_file}`: "
                f"{idx + 1} > {expected_rows}"
            )
        features[idx] = feature.astype(np.float32, copy=False)
        mids[idx] = float(mid_value)
        idx += 1

    if idx != expected_rows:
        raise RuntimeError(
            f"row count mismatch while materializing `{input_file}`: expected {expected_rows}, got {idx}"
        )

    del mids
    del features
    return {
        "file": input_file,
        "rows": expected_rows,
        "features_out_path": features_out_path,
        "mid_out_path": mid_out_path,
    }


def materialize_job_runner(params: Tuple) -> Dict:
    return materialize_input_file(*params)


def to_levels(levels: Sequence[Sequence[float]], max_levels: int = 5) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for i in range(max_levels):
        if i < len(levels) and len(levels[i]) >= 2:
            out.append((float(levels[i][0]), float(levels[i][1])))
        else:
            out.append((0.0, 0.0))
    return out


def row_to_feature(row: Dict) -> Tuple[np.ndarray, float]:
    asks = to_levels(row.get("asks_top5", []), 5)
    bids = to_levels(row.get("bids_top5", []), 5)
    snap = row.get("features_snapshot") or {}
    feature = np.array(
        [
            asks[0][0],
            asks[0][1],
            bids[0][0],
            bids[0][1],
            asks[1][0],
            asks[1][1],
            bids[1][0],
            bids[1][1],
            asks[2][0],
            asks[2][1],
            bids[2][0],
            bids[2][1],
            asks[3][0],
            asks[3][1],
            bids[3][0],
            bids[3][1],
            asks[4][0],
            asks[4][1],
            bids[4][0],
            bids[4][1],
            float(snap.get("qi_top1", 0.0)),
            float(snap.get("qi_top2", 0.0)),
            float(snap.get("qi_top3", 0.0)),
            float(snap.get("qi_top4", 0.0)),
            float(snap.get("qi_top5", 0.0)),
            float(snap.get("ofi_250ms", 0.0)),
            float(snap.get("ofi_1s", 0.0)),
            float(snap.get("ofi_3s", 0.0)),
            float(snap.get("microprice_minus_mid", 0.0)),
            float(snap.get("spread_ticks", 0.0)),
            float(snap.get("depth_ratio", 0.0)),
        ],
        dtype=np.float64,
    )
    mid = float(row.get("mid_price") or 0.0)
    return feature, mid


def compute_labels(
    mid: np.ndarray, horizon: int, len_smooth: int, alpha_mult: float
) -> Tuple[np.ndarray, float]:
    n = len(mid)
    labels = np.full(n, np.inf, dtype=np.float32)
    if n < len_smooth + horizon + 1:
        return labels, 0.0

    csum = np.concatenate(([0.0], np.cumsum(mid, dtype=np.float64)))
    smoothed = (csum[len_smooth:] - csum[:-len_smooth]) / float(len_smooth)
    prev = smoothed[:-horizon]
    fut = smoothed[horizon:]
    pct = (fut - prev) / np.maximum(prev, 1e-12)
    alpha = float(np.mean(np.abs(pct)) * alpha_mult)
    labels_core = np.where(pct < -alpha, 2, np.where(pct > alpha, 0, 1)).astype(np.float32)
    start = len_smooth - 1
    labels[start : start + len(labels_core)] = labels_core
    return labels, alpha


def label_distribution(labels: np.ndarray) -> Dict[str, int]:
    valid = labels[np.isfinite(labels)].astype(np.int64)
    if valid.size == 0:
        return {"up": 0, "flat": 0, "down": 0}
    return {
        "up": int((valid == 0).sum()),
        "flat": int((valid == 1).sum()),
        "down": int((valid == 2).sum()),
    }


def validate_row(
    row: Dict,
    strict_quality: bool,
    market_filter: str,
    require_signal_seen: bool,
    excluded_reasons: set[str],
    max_spread_ticks: float,
    min_depth_ratio: float,
    max_depth_ratio: float,
) -> str | None:
    if market_filter and row.get("market") != market_filter:
        return "market_filtered"
    if require_signal_seen and not bool(row.get("signal_seen", False)):
        return "signal_not_seen"
    if row.get("features_snapshot") is None:
        return "missing_features_snapshot"

    typed_reason = str(row.get("typed_reason", "")).strip()
    if typed_reason in excluded_reasons:
        return f"excluded_typed_reason:{typed_reason}"

    best_bid = row.get("best_bid")
    best_ask = row.get("best_ask")
    mid = row.get("mid_price")
    if best_bid is None or best_ask is None or mid is None:
        return "missing_top_of_book"

    try:
        best_bid = float(best_bid)
        best_ask = float(best_ask)
        mid = float(mid)
    except (TypeError, ValueError):
        return "invalid_top_of_book_numeric"

    if not np.isfinite(best_bid) or not np.isfinite(best_ask) or not np.isfinite(mid):
        return "non_finite_top_of_book"
    if best_bid <= 0.0 or best_ask <= 0.0:
        return "non_positive_top_of_book"
    if best_ask <= best_bid:
        return "crossed_or_inverted_book"
    if mid <= 0.0:
        return "non_positive_mid"

    spread_ticks = float((row.get("features_snapshot") or {}).get("spread_ticks", np.nan))
    depth_ratio = float((row.get("features_snapshot") or {}).get("depth_ratio", np.nan))
    if not np.isfinite(spread_ticks):
        return "invalid_spread_ticks"
    if not np.isfinite(depth_ratio):
        return "invalid_depth_ratio"

    if strict_quality:
        if spread_ticks <= 0.0 or spread_ticks > max_spread_ticks:
            return "strict_spread_out_of_range"
        if depth_ratio < min_depth_ratio or depth_ratio > max_depth_ratio:
            return "strict_depth_ratio_out_of_range"
    return None


def compute_train_scaler(features: np.memmap, train_end: int, chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
    feature_dim = features.shape[1]
    if train_end <= 0:
        return np.zeros(feature_dim, dtype=np.float64), np.ones(feature_dim, dtype=np.float64)
    sum_x = np.zeros(feature_dim, dtype=np.float64)
    sum_x2 = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    for start in range(0, train_end, chunk_size):
        end = min(start + chunk_size, train_end)
        chunk = np.asarray(features[start:end], dtype=np.float64)
        sum_x += chunk.sum(axis=0)
        sum_x2 += np.square(chunk, dtype=np.float64).sum(axis=0)
        count += chunk.shape[0]
    mean = sum_x / max(count, 1)
    var = np.maximum(sum_x2 / max(count, 1) - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.where(std <= 1e-9, 1.0, std)
    return mean, std


def normalize_features_in_place(features: np.memmap, mean: np.ndarray, std: np.ndarray, chunk_size: int):
    n = features.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = np.asarray(features[start:end], dtype=np.float32)
        chunk = (chunk - mean.astype(np.float32)) / std.astype(np.float32)
        features[start:end] = chunk


def write_split_dataset(
    out_path: Path,
    features: np.memmap,
    labels: np.memmap,
    start: int,
    end: int,
    chunk_size: int,
):
    rows = max(end - start, 0)
    feature_dim = features.shape[1]
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(rows, feature_dim + labels.shape[1])
    )
    offset = 0
    for src_start in range(start, end, chunk_size):
        src_end = min(src_start + chunk_size, end)
        dst_end = offset + (src_end - src_start)
        out[offset:dst_end, :feature_dim] = features[src_start:src_end]
        out[offset:dst_end, feature_dim:] = labels[src_start:src_end]
        offset = dst_end
    del out


def main() -> int:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    input_files = collect_input_files(input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_reasons = set(args.exclude_typed_reason)
    if args.strict_quality:
        excluded_reasons |= DEFAULT_STRICT_EXCLUDE_TYPED_REASONS

    raw_total = 0
    filtered = Counter()
    typed_reason_total = Counter()
    kept_typed_reasons = Counter()
    ordering_violations = 0
    usable_rows = 0

    workers = max(1, args.workers)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            scan_results = list(
                executor.map(
                    scan_input_file,
                    [str(path) for path in input_files],
                    [args.strict_quality] * len(input_files),
                    [args.market] * len(input_files),
                    [args.require_signal_seen] * len(input_files),
                    [excluded_reasons] * len(input_files),
                    [args.max_spread_ticks] * len(input_files),
                    [args.min_depth_ratio] * len(input_files),
                    [args.max_depth_ratio] * len(input_files),
                )
            )
    else:
        scan_results = [
            scan_input_file(
                input_file=str(path),
                strict_quality=args.strict_quality,
                market_filter=args.market,
                require_signal_seen=args.require_signal_seen,
                excluded_reasons=excluded_reasons,
                max_spread_ticks=args.max_spread_ticks,
                min_depth_ratio=args.min_depth_ratio,
                max_depth_ratio=args.max_depth_ratio,
            )
            for path in input_files
        ]

    previous_last_key: Tuple[int, int] | None = None
    for result in scan_results:
        raw_total += int(result["raw_total"])
        filtered.update(result["filtered"])
        typed_reason_total.update(result["typed_reason_total"])
        kept_typed_reasons.update(result["kept_typed_reasons"])
        usable_rows += int(result["usable_rows"])
        ordering_violations += int(result["ordering_violations"])
        first_key = result["first_key"]
        last_key = result["last_key"]
        if previous_last_key is not None and first_key is not None and first_key < previous_last_key:
            ordering_violations += 1
        if last_key is not None:
            previous_last_key = tuple(last_key)

    if usable_rows < args.min_rows:
        raise RuntimeError(
            f"not enough usable rows after numeric checks: {usable_rows} < min_rows={args.min_rows}"
        )

    feature_dim = len(FEATURE_NAMES)
    chunk_size = 200_000

    with tempfile.TemporaryDirectory(prefix="engine_build_", dir=str(output_dir)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        features_path = tmpdir_path / "features.mm"
        mid_path = tmpdir_path / "mid.mm"
        labels_path = tmpdir_path / "labels.mm"
        per_file_dir = tmpdir_path / "per_file"
        per_file_dir.mkdir(parents=True, exist_ok=True)

        features = np.memmap(
            features_path, dtype=np.float32, mode="w+", shape=(usable_rows, feature_dim)
        )
        mid = np.memmap(mid_path, dtype=np.float64, mode="w+", shape=(usable_rows,))

        # Pass 2: materialize each file in parallel, then merge in input order.
        materialize_jobs = []
        for file_idx, result in enumerate(scan_results):
            rows = int(result["usable_rows"])
            features_out_path = str(per_file_dir / f"{file_idx:06d}_features.mm")
            mid_out_path = str(per_file_dir / f"{file_idx:06d}_mid.mm")
            materialize_jobs.append(
                (
                    str(input_files[file_idx]),
                    rows,
                    feature_dim,
                    features_out_path,
                    mid_out_path,
                    args.strict_quality,
                    args.market,
                    args.require_signal_seen,
                    excluded_reasons,
                    args.max_spread_ticks,
                    args.min_depth_ratio,
                    args.max_depth_ratio,
                )
            )

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                materialized = list(executor.map(materialize_job_runner, materialize_jobs))
        else:
            materialized = [materialize_input_file(*params) for params in materialize_jobs]

        idx = 0
        for file_idx, item in enumerate(materialized):
            rows = int(item["rows"])
            if rows == 0:
                continue
            features_chunk = np.memmap(
                item["features_out_path"], dtype=np.float32, mode="r", shape=(rows, feature_dim)
            )
            mid_chunk = np.memmap(item["mid_out_path"], dtype=np.float64, mode="r", shape=(rows,))
            features[idx : idx + rows] = features_chunk
            mid[idx : idx + rows] = mid_chunk
            idx += rows
            del mid_chunk
            del features_chunk
        if idx != usable_rows:
            raise RuntimeError(f"row count mismatch across passes: expected {usable_rows}, got {idx}")

        label_h10, alpha_h10 = compute_labels(mid, 10, args.len_smooth, args.alpha_mult)
        label_h20, alpha_h20 = compute_labels(mid, 20, args.len_smooth, args.alpha_mult)
        label_h50, alpha_h50 = compute_labels(mid, 50, args.len_smooth, args.alpha_mult)
        label_h100, alpha_h100 = compute_labels(mid, 100, args.len_smooth, args.alpha_mult)

        labels = np.memmap(labels_path, dtype=np.float32, mode="w+", shape=(usable_rows, 4))
        labels[:, 0] = label_h10
        labels[:, 1] = label_h20
        labels[:, 2] = label_h50
        labels[:, 3] = label_h100

        n = usable_rows
        train_end = int(n * args.train_ratio)
        val_end = train_end + int(n * args.val_ratio)

        if args.normalize:
            mu, sigma = compute_train_scaler(features, train_end, chunk_size)
            normalize_features_in_place(features, mu, sigma, chunk_size)
            scaler = {"mean": mu.tolist(), "std": sigma.tolist()}
        else:
            scaler = None

        write_split_dataset(output_dir / "train.npy", features, labels, 0, train_end, chunk_size)
        write_split_dataset(output_dir / "val.npy", features, labels, train_end, val_end, chunk_size)
        write_split_dataset(output_dir / "test.npy", features, labels, val_end, n, chunk_size)

        # ensure memmaps are closed before tempdir cleanup
        del labels
        del mid
        del features

    n = usable_rows
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)

    quality_report = {
        "input_path": str(input_path),
        "input_files_count": len(input_files),
        "input_files_sample": [str(path) for path in input_files[:10]],
        "workers": workers,
        "ordering_violations": ordering_violations,
        "market_filter": args.market or None,
        "strict_quality": bool(args.strict_quality),
        "raw_rows_total": raw_total,
        "usable_rows": int(n),
        "usable_ratio_pct": round((n / max(raw_total, 1)) * 100.0, 4),
        "filtered_counts": dict(filtered),
        "typed_reason_distribution_raw_top20": dict(typed_reason_total.most_common(20)),
        "typed_reason_distribution_kept_top20": dict(kept_typed_reasons.most_common(20)),
        "feature_dim": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "label_layout": ["h10", "h20", "h50", "h100"],
        "label_distribution": {
            "h10": label_distribution(label_h10),
            "h20": label_distribution(label_h20),
            "h50": label_distribution(label_h50),
            "h100": label_distribution(label_h100),
        },
        "len_smooth": args.len_smooth,
        "alpha_mult": args.alpha_mult,
        "alphas": {
            "h10": alpha_h10,
            "h20": alpha_h20,
            "h50": alpha_h50,
            "h100": alpha_h100,
        },
        "split_sizes": {"train": train_end, "val": val_end - train_end, "test": n - val_end},
        "normalized": bool(args.normalize),
        "scaler": scaler,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    with (output_dir / args.quality_report_name).open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    print(json.dumps(quality_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
