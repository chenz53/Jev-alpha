"""Fit one temperature using only the calibration split."""

import argparse
import math
from pathlib import Path

from jev_alpha.io import read_jsonl, write_json


def probabilities(logits, temperature=1.0):
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("Logits must be nonempty and finite")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    values = [x / temperature for x in logits]
    maximum = max(values)
    weights = [math.exp(x - maximum) for x in values]
    return [x / sum(weights) for x in weights]


def nll(rows, temperature):
    total = 0.0
    for row in rows:
        values = [x / temperature for x in row["logits"]]
        maximum = max(values)
        total += maximum + math.log(sum(math.exp(x - maximum) for x in values)) - values[row["target"]]
    return total / len(rows)


def validate_rows(rows, split):
    if not rows:
        raise ValueError("Predictions must not be empty")
    identities = set()
    for row in rows:
        if row.get("split") != split or not row.get("dataset"):
            raise ValueError(f"Each prediction needs a dataset and split={split}")
        identity = (row["dataset"], row["id"], row["question"])
        if identity in identities:
            raise ValueError("Duplicate prediction identity")
        identities.add(identity)
        logits = row["logits"]
        probabilities(logits)
        if len(logits) < 2 or not isinstance(row["target"], int) or not 0 <= row["target"] < len(logits):
            raise ValueError("Invalid target or option count")
        if len(row["options"]) != len(logits) or len(set(row["options"])) != len(logits):
            raise ValueError("Options must be unique and match logits")


def fit_temperature(rows):
    validate_rows(rows, "calibration")
    # Search inverse temperature; NLL is convex in this scalar.
    low, high = 0.01, 100.0
    for _ in range(120):
        left = low + (high - low) / 3
        right = high - (high - low) / 3
        if nll(rows, 1 / left) < nll(rows, 1 / right):
            high = right
        else:
            low = left
    candidates = [1.0, 0.01, 100.0, 1 / ((low + high) / 2)]
    return min(candidates, key=lambda temperature: nll(rows, temperature))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    temperature = fit_temperature(rows)
    path = Path(args.checkpoint)
    if not path.is_dir():
        raise ValueError("Checkpoint directory does not exist")
    write_json(
        path / "temperature.json",
        {
            "temperature": temperature,
            "questions": len(rows),
            "datasets": sorted({row["dataset"] for row in rows}),
            "nll_before": nll(rows, 1.0),
            "nll_after": nll(rows, temperature),
            "bounds": [0.01, 100.0],
        },
    )


if __name__ == "__main__":
    main()
