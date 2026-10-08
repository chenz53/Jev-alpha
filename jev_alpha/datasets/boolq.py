"""Convert google/boolq to the shared decision record format."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from jev_alpha.io import write_json


def convert(row, index, split):
    if not isinstance(row["answer"], bool):
        raise ValueError("BoolQ answer must be boolean")
    return {
        "id": f"{split}-{index}",
        "dataset": "google/boolq",
        "split": "train" if split == "train" else "calibration",
        "state": row["passage"],
        "questions": {
            "answer": {
                "type": "noul",
                "instructions": row["question"],
                "criteria": {},
                "target": "yes" if row["answer"] else "no",
            }
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/boolq")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    revision = HfApi().dataset_info("google/boolq", revision=args.revision).sha
    dataset = load_dataset("google/boolq", revision=revision)
    for split in ("train", "validation"):
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(dataset[split]):
                stream.write(json.dumps(convert(row, index, split), ensure_ascii=False) + "\n")
    write_json(output / "source.json", {"dataset": "google/boolq", "revision": revision, "license": "CC-BY-SA-3.0"})


if __name__ == "__main__":
    main()
