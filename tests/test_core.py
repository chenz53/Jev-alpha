import copy
import math
import unittest

import torch

from jev_alpha.calibrate import fit_temperature, nll, probabilities
from jev_alpha.data import render, validate
from jev_alpha.datasets.hard_cases import records
from jev_alpha.datasets.prepare import split_data
from jev_alpha.eval import evaluate, metrics, order_drift
from jev_alpha.loss import option_loss


class CoreTests(unittest.TestCase):
    def test_render_preserves_identity_after_shuffle(self):
        record = next(records())
        before = copy.deepcopy(record)
        messages, order = render(record, "decision", order=["unknown", "today", "tomorrow"], answer=True)
        self.assertEqual(messages[-1]["content"], "A")
        self.assertEqual(order[0], record["questions"]["decision"]["target"])
        self.assertEqual(record, before)
        with self.assertRaises(ValueError):
            render(record, "decision", order=["unknown", "today", "today"])

    def test_types_and_score_levels(self):
        for record in records():
            validate(record, targets=True)
        record = list(records())[-1]
        record["questions"]["decision"]["criteria"] = {"nan": "Invalid", "1": "One"}
        with self.assertRaises(ValueError):
            validate(record)

    def test_loss_shift_mask_padding_and_gradients(self):
        logits = torch.randn(2, 5, 12, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, -100, -100], [-100, -100, -100, -100, 7]])
        loss = option_loss(logits, labels, [2, 3], [3, 5, 7])
        expected = (
            torch.logsumexp(logits[0, 1, [3, 5]], 0)
            - logits[0, 1, 3]
            + torch.logsumexp(logits[1, 3, [3, 5, 7]], 0)
            - logits[1, 3, 7]
        ) / 2
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertEqual(logits.grad[0, 1, 7].item(), 0)
        self.assertEqual(logits.grad[0, 2].abs().sum().item(), 0)
        self.assertTrue(torch.allclose(option_loss(logits, labels, [2, 3], [3, 5, 7], 4), loss / 2))

    def test_loss_rejects_missing_or_invalid_answers(self):
        logits = torch.randn(1, 3, 10)
        for labels in [[-100, -100, -100], [-100, 3, 5], [-100, -100, 7]]:
            with self.assertRaises(ValueError):
                option_loss(logits, torch.tensor([labels]), [2], [3, 5, 7])

    def test_metrics_known_values_and_endpoint(self):
        result = metrics([[0.8, 0.2], [0.6, 0.4]], [0, 1])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["brier"], 0.4)
        self.assertAlmostEqual(result["ece"], 0.4)
        self.assertEqual(metrics([[1, 0]], [0])["bins"][9]["n"], 1)

    def prediction(self, index, target, split="calibration"):
        return {
            "id": str(index),
            "dataset": "cal" if split == "calibration" else "test",
            "split": split,
            "question": "q",
            "options": ["yes", "no"],
            "logits": [4.0, 0.0],
            "target": target,
            "shuffled": {"options": ["no", "yes"], "logits": [0.0, 4.0]},
        }

    def test_temperature_fit_and_order_alignment(self):
        rows = [self.prediction(i, int(i >= 3)) for i in range(4)]
        temperature = fit_temperature(rows)
        self.assertGreater(temperature, 1)
        self.assertLess(nll(rows, temperature), nll(rows, 1))
        self.assertAlmostEqual(probabilities([4, 0], temperature)[0], 0.75, places=5)
        self.assertEqual(order_drift(rows, temperature)["max_probability_change"], 0)
        with self.assertRaises(ValueError):
            fit_temperature([self.prediction(0, 0, "test")])
        with self.assertRaises(ValueError):
            probabilities([0, math.nan])

    def test_per_dataset_report(self):
        row = self.prediction(0, 0, "test")
        report = evaluate([row], {"temperature": 2, "datasets": ["cal"]})
        self.assertIn("test", report["datasets"])
        with self.assertRaises(ValueError):
            evaluate([row], {"temperature": 2, "datasets": ["test"]})

    def test_whole_test_holdout_and_leak_removal(self):
        rows = list(records())[:3]
        rows[0].update(dataset="source", split="train")
        rows[1].update(dataset="source", split="calibration")
        leak = copy.deepcopy(rows[2])
        leak.update(dataset="source", split="train", id="leak")
        groups, audit = split_data(rows + [leak])
        self.assertEqual(len(groups["train"]), 1)
        self.assertEqual(audit["removed"][0]["id"], "leak")
        rows[2]["dataset"] = "source"
        with self.assertRaises(ValueError):
            split_data(rows)

    def test_near_match_removal(self):
        rows = list(records())[:3]
        rows[0].update(dataset="train", split="train")
        rows[1].update(dataset="cal", split="calibration")
        rows[2]["state"] = " ".join(f"word{i}" for i in range(100))
        leak = copy.deepcopy(rows[2])
        leak.update(dataset="train", split="train", id="near")
        leak["state"] += " extra"
        _, audit = split_data(rows + [leak])
        self.assertEqual(audit["removed"][0]["id"], "near")


if __name__ == "__main__":
    unittest.main()
