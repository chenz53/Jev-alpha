"""Report per-dataset accuracy, Brier score, calibration, and option-order drift."""

import argparse
import json
from collections import defaultdict

from jev_alpha.calibrate import probabilities, validate_rows
from jev_alpha.io import read_jsonl, write_json


def metrics(distributions, targets):
    if not distributions or len(distributions) != len(targets):
        raise ValueError("Distributions and targets must have equal nonzero lengths")
    bins = [[] for _ in range(10)]
    correct = brier = 0.0
    for distribution, target in zip(distributions, targets):
        if (
            not 0 <= target < len(distribution)
            or any(not 0 <= p <= 1 for p in distribution)
            or abs(sum(distribution) - 1) > 1e-6
        ):
            raise ValueError("Invalid probability distribution or target")
        winner = max(range(len(distribution)), key=distribution.__getitem__)
        hit = int(winner == target)
        confidence = distribution[winner]
        correct += hit
        # Multiclass Brier is the sum over options, without division by K.
        brier += sum((p - int(index == target)) ** 2 for index, p in enumerate(distribution))
        bins[min(int(confidence * 10), 9)].append((confidence, hit))
    count = len(targets)
    ece = sum(abs(sum(p - hit for p, hit in bucket)) for bucket in bins) / count
    return {
        "n": count,
        "accuracy": correct / count,
        "brier": brier / count,
        "ece": ece,
        "bins": [
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "n": len(bucket),
                "confidence": sum(p for p, _ in bucket) / len(bucket) if bucket else None,
                "accuracy": sum(hit for _, hit in bucket) / len(bucket) if bucket else None,
            }
            for i, bucket in enumerate(bins)
        ],
    }


def order_drift(rows, temperature):
    differences = []
    for row in rows:
        reordered = row["shuffled"]
        keys = reordered["options"]
        if len(keys) != len(row["options"]) or set(keys) != set(row["options"]):
            raise ValueError("Shuffled options do not match the original options")
        values = probabilities(reordered["logits"], temperature)
        if len(values) != len(keys):
            raise ValueError("Shuffled logits do not match the options")
        mapping = dict(zip(keys, values))
        original = probabilities(row["logits"], temperature)
        differences.append([abs(p - mapping[key]) for key, p in zip(row["options"], original)])
    return {
        "mean_total_variation": sum(sum(d) / 2 for d in differences) / len(differences),
        "max_probability_change": max(max(d) for d in differences),
    }


def evaluate(rows, calibration):
    validate_rows(rows, "test")
    groups = defaultdict(list)
    for row in rows:
        if row["dataset"] in calibration["datasets"]:
            raise ValueError("Test datasets must not be used for calibration")
        groups[row["dataset"]].append(row)
    report = {"temperature": calibration["temperature"], "datasets": {}}
    for name, group in sorted(groups.items()):
        targets = [row["target"] for row in group]
        result = {}
        for key, temperature in [("before", 1.0), ("after", calibration["temperature"])]:
            result[key] = metrics([probabilities(row["logits"], temperature) for row in group], targets)
            result[key]["option_order"] = order_drift(group, temperature)
        report["datasets"][name] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("temperature")
    parser.add_argument("output")
    args = parser.parse_args()
    with open(args.temperature, encoding="utf-8") as stream:
        calibration = json.load(stream)
    write_json(args.output, evaluate(read_jsonl(args.predictions), calibration), compact=True)


if __name__ == "__main__":
    main()
