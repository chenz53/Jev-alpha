"""Check whole-dataset splits and remove exact and near training matches."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from jev_alpha.data import validate
from jev_alpha.io import read_jsonl, write_json


def normalized_state(record):
    return " ".join(re.findall(r"\w+", record["state"].casefold()))


def shingles(record):
    words = normalized_state(record).split()
    size = min(5, len(words))
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def match_index(records, threshold):
    references = [shingles(record) for record in records]
    index = defaultdict(set)
    for position, tokens in enumerate(references):
        for token in tokens:
            index[token].add(position)

    def overlaps(record):
        tokens = shingles(record)
        candidates = set().union(*(index.get(token, set()) for token in tokens))
        return any(len(tokens & references[i]) / len(tokens | references[i]) >= threshold for i in candidates)

    return overlaps


def split_data(records, threshold=0.9):
    if not 0 < threshold <= 1:
        raise ValueError("Near-match threshold must be in (0, 1]")
    groups = {key: [] for key in ("train", "calibration", "test")}
    sources, identities = {}, set()
    for record in records:
        validate(record, targets=True)
        split, source = record.get("split"), record.get("dataset")
        if split not in groups or not isinstance(source, str) or not source:
            raise ValueError("Each record needs dataset and train/calibration/test split")
        if not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError("Each record needs a nonempty string id")
        identity = (source, record["id"])
        if identity in identities:
            raise ValueError("Duplicate dataset/id")
        identities.add(identity)
        sources.setdefault(source, set()).add(split)
        if "test" in sources[source] and len(sources[source]) > 1:
            raise ValueError("Hold out whole test datasets")
        groups[split].append(record)
    if any(not rows for rows in groups.values()):
        raise ValueError("Each split needs at least one record")

    overlaps_test = match_index(groups["test"], threshold)
    if any(overlaps_test(record) for record in groups["calibration"]):
        raise ValueError("Calibration overlaps test; choose separate calibration data")
    overlaps_held_out = match_index(groups["test"] + groups["calibration"], threshold)
    removed, kept = [], []
    for record in groups["train"]:
        (removed if overlaps_held_out(record) else kept).append(record)
    if not kept:
        raise ValueError("Deduplication removed all training records")
    groups["train"] = kept
    audit = {
        "threshold": threshold,
        "method": "normalized state five-word shingle Jaccard similarity",
        "removed": [{"dataset": r["dataset"], "id": r["id"]} for r in removed],
        "counts": {key: len(rows) for key, rows in groups.items()},
        "datasets": {key: sorted(value) for key, value in sources.items()},
    }
    return groups, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    records = [row for path in args.input for row in read_jsonl(path)]
    groups, audit = split_data(records, args.threshold)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in groups.items():
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                # The wrapper prevents Arrow from merging variable question schemas.
                stream.write(json.dumps({"record": json.dumps(row)}) + "\n")
    audit["inputs"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.input}
    write_json(output / "audit.json", audit)


if __name__ == "__main__":
    main()
