from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch


def _as_long_tensor(values: Iterable[int] | torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().long().flatten()
    if isinstance(values, np.ndarray):
        return torch.from_numpy(values).long().flatten()
    return torch.tensor(list(values), dtype=torch.long).flatten()


def compute_class_weights(
    labels: Iterable[int] | torch.Tensor | np.ndarray,
    num_classes: int = 3,
    min_weight: float = 1.0,
    max_weight: float = 50.0,
) -> torch.Tensor:
    y = _as_long_tensor(labels)
    counts = torch.bincount(y, minlength=num_classes).float()
    n = counts.sum().clamp(min=1.0)
    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    weights = n / (float(num_classes) * safe_counts)
    weights = torch.clamp(weights, min=min_weight, max=max_weight)
    weights = weights / weights.mean().clamp(min=1e-12)
    return weights


def compute_sample_weights(
    labels: Iterable[int] | torch.Tensor | np.ndarray,
    num_classes: int = 3,
    sampler_pow: float = 1.0,
) -> torch.Tensor:
    y = _as_long_tensor(labels)
    counts = torch.bincount(y, minlength=num_classes).float()
    n = counts.sum().clamp(min=1.0)
    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    class_weights = (n / safe_counts).pow(float(sampler_pow))
    class_weights = class_weights / class_weights.mean().clamp(min=1e-12)
    return class_weights[y]


def compute_event_metrics(
    targets: Iterable[int] | torch.Tensor | np.ndarray,
    predictions: Iterable[int] | torch.Tensor | np.ndarray,
    neutral_class: int = 1,
) -> dict[str, float]:
    y_true = _as_long_tensor(targets).numpy()
    y_pred = _as_long_tensor(predictions).numpy()

    true_event = y_true != neutral_class
    pred_event = y_pred != neutral_class

    tp = int(np.logical_and(true_event, pred_event).sum())
    fp = int(np.logical_and(~true_event, pred_event).sum())
    fn = int(np.logical_and(true_event, ~pred_event).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_tp": float(tp),
        "event_fp": float(fp),
        "event_fn": float(fn),
        "event_support": float(true_event.sum()),
    }


def predicted_event_rate(
    predictions: Iterable[int] | torch.Tensor | np.ndarray,
    neutral_class: int = 1,
) -> float:
    y_pred = _as_long_tensor(predictions).numpy()
    return float((y_pred != neutral_class).mean()) if y_pred.size > 0 else 0.0


def precision_at_top_percent(
    targets: Iterable[int] | torch.Tensor | np.ndarray,
    event_scores: Iterable[float] | torch.Tensor | np.ndarray,
    top_percent: float,
    neutral_class: int = 1,
) -> float:
    y_true = _as_long_tensor(targets).numpy()
    if isinstance(event_scores, torch.Tensor):
        scores = event_scores.detach().cpu().numpy().astype(np.float64)
    elif isinstance(event_scores, np.ndarray):
        scores = event_scores.astype(np.float64)
    else:
        scores = np.asarray(list(event_scores), dtype=np.float64)
    n = y_true.shape[0]
    if n == 0:
        return 0.0
    k = max(1, int(round(n * top_percent)))
    idx = np.argpartition(scores, -k)[-k:]
    return float((y_true[idx] != neutral_class).mean())


def threshold_sweep_event_metrics(
    targets: Iterable[int] | torch.Tensor | np.ndarray,
    event_scores: Iterable[float] | torch.Tensor | np.ndarray,
    thresholds: Iterable[float],
    neutral_class: int = 1,
) -> list[dict[str, float]]:
    y_true = _as_long_tensor(targets).numpy()
    if isinstance(event_scores, torch.Tensor):
        scores = event_scores.detach().cpu().numpy().astype(np.float64)
    elif isinstance(event_scores, np.ndarray):
        scores = event_scores.astype(np.float64)
    else:
        scores = np.asarray(list(event_scores), dtype=np.float64)

    true_event = y_true != neutral_class
    rows = []
    for th in thresholds:
        pred_event = scores >= float(th)
        tp = int(np.logical_and(true_event, pred_event).sum())
        fp = int(np.logical_and(~true_event, pred_event).sum())
        fn = int(np.logical_and(true_event, ~pred_event).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "threshold": float(th),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_event_rate": float(pred_event.mean()),
            }
        )
    return rows


def pick_threshold_for_target_rate(
    sweep_rows: list[dict[str, float]],
    target_rate: float,
) -> Optional[dict[str, float]]:
    if not sweep_rows:
        return None
    eligible = [r for r in sweep_rows if r["predicted_event_rate"] <= target_rate]
    if not eligible:
        return min(sweep_rows, key=lambda r: abs(r["predicted_event_rate"] - target_rate))
    return sorted(
        eligible,
        key=lambda r: (-r["precision"], -r["recall"], r["predicted_event_rate"]),
    )[0]


def select_best_run(
    runs: list[dict],
    min_event_precision: float = 0.20,
    fallback_min_event_precision: float = 0.15,
) -> tuple[Optional[dict], float]:
    if not runs:
        return None, min_event_precision

    def _eligible(threshold: float) -> list[dict]:
        return [r for r in runs if float(r.get("val_event_precision", 0.0)) >= threshold]

    eligible = _eligible(min_event_precision)
    used_threshold = min_event_precision
    if not eligible:
        eligible = _eligible(fallback_min_event_precision)
        used_threshold = fallback_min_event_precision
    if not eligible:
        eligible = list(runs)

    winner = sorted(
        eligible,
        key=lambda r: (
            -float(r.get("val_event_recall", 0.0)),
            -float(r.get("val_event_f1", 0.0)),
            float(r.get("val_loss", 1e9)),
        ),
    )[0]
    return winner, used_threshold
