import unittest

import torch
from torch.utils.data import WeightedRandomSampler

from utils.engine_recall import (
    compute_class_weights,
    compute_event_metrics,
    compute_sample_weights,
    predicted_event_rate,
    precision_at_top_percent,
    threshold_sweep_event_metrics,
    pick_threshold_for_target_rate,
    select_best_run,
)


class EngineRecallUtilsTest(unittest.TestCase):
    def test_compute_class_weights_deterministic_and_normalized(self):
        labels = torch.tensor([1] * 100 + [0] * 10 + [2] * 5, dtype=torch.long)
        w1 = compute_class_weights(labels)
        w2 = compute_class_weights(labels)
        self.assertTrue(torch.allclose(w1, w2))
        self.assertAlmostEqual(float(w1.mean()), 1.0, places=6)
        self.assertGreater(float(w1[2]), float(w1[0]))
        self.assertGreater(float(w1[0]), float(w1[1]))

    def test_compute_class_weights_handles_missing_class(self):
        labels = torch.tensor([1] * 50 + [0] * 5, dtype=torch.long)
        weights = compute_class_weights(labels)
        self.assertEqual(tuple(weights.shape), (3,))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_weighted_sampler_boosts_minority_presence(self):
        labels = torch.tensor([1] * 1000 + [0] * 20 + [2] * 20, dtype=torch.long)
        sample_weights = compute_sample_weights(labels, sampler_pow=1.0)
        sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=600,
            replacement=True,
        )
        idx = torch.tensor(list(iter(sampler)), dtype=torch.long)
        sampled_labels = labels[idx]

        base_event_ratio = float((labels != 1).float().mean())
        sampled_event_ratio = float((sampled_labels != 1).float().mean())
        self.assertGreater(sampled_event_ratio, base_event_ratio)

    def test_event_metrics(self):
        targets = [1, 1, 0, 2, 0, 2]
        predictions = [1, 0, 0, 1, 2, 2]
        metrics = compute_event_metrics(targets, predictions)
        self.assertAlmostEqual(metrics["event_precision"], 0.75, places=6)
        self.assertAlmostEqual(metrics["event_recall"], 0.75, places=6)
        self.assertAlmostEqual(metrics["event_f1"], 0.75, places=6)

    def test_predicted_event_rate(self):
        preds = [1, 1, 0, 2, 1]
        self.assertAlmostEqual(predicted_event_rate(preds), 0.4, places=6)

    def test_precision_at_top_percent(self):
        targets = [1, 1, 0, 2, 1, 0]
        scores = [0.1, 0.2, 0.95, 0.9, 0.3, 0.85]
        p_top = precision_at_top_percent(targets, scores, top_percent=0.5)
        self.assertAlmostEqual(p_top, 1.0, places=6)

    def test_threshold_sweep_and_pick(self):
        targets = [1, 1, 0, 2, 1, 0, 2, 1]
        scores = [0.1, 0.2, 0.95, 0.9, 0.3, 0.85, 0.8, 0.05]
        rows = threshold_sweep_event_metrics(targets, scores, thresholds=[0.5, 0.8, 0.95])
        self.assertEqual(len(rows), 3)
        pick = pick_threshold_for_target_rate(rows, target_rate=0.4)
        self.assertIsNotNone(pick)
        self.assertLessEqual(pick["predicted_event_rate"], 0.4)

    def test_select_best_run_guard_and_fallback(self):
        runs = [
            {"id": "a", "val_event_precision": 0.10, "val_event_recall": 0.90, "val_event_f1": 0.40, "val_loss": 0.2},
            {"id": "b", "val_event_precision": 0.22, "val_event_recall": 0.60, "val_event_f1": 0.50, "val_loss": 0.3},
            {"id": "c", "val_event_precision": 0.25, "val_event_recall": 0.55, "val_event_f1": 0.55, "val_loss": 0.1},
        ]
        winner, threshold = select_best_run(runs, min_event_precision=0.20, fallback_min_event_precision=0.15)
        self.assertEqual(threshold, 0.20)
        self.assertEqual(winner["id"], "b")

        no_guard_runs = [
            {"id": "x", "val_event_precision": 0.10, "val_event_recall": 0.70, "val_event_f1": 0.30, "val_loss": 0.2},
            {"id": "y", "val_event_precision": 0.16, "val_event_recall": 0.60, "val_event_f1": 0.40, "val_loss": 0.1},
        ]
        winner2, threshold2 = select_best_run(no_guard_runs, min_event_precision=0.20, fallback_min_event_precision=0.15)
        self.assertEqual(threshold2, 0.15)
        self.assertEqual(winner2["id"], "y")


if __name__ == "__main__":
    unittest.main()
