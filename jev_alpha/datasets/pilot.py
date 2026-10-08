"""Make fixed small subsets for a bounded integration and calibration run."""

import json
import random
from pathlib import Path

from jev_alpha.io import read_jsonl, write_json


def main():
    output = Path("data/pilot")
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(42)
    for split, count in [("train", 128), ("calibration", 128), ("test", 18)]:
        rows = read_jsonl(f"data/prepared/{split}.jsonl")
        rows = rng.sample(rows, min(count, len(rows)))
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    write_json(
        output / "selection.json",
        {
            "seed": 42,
            "train": 128,
            "calibration": 128,
            "test": 18,
            "source": "data/prepared; filtered before selection",
        },
    )


if __name__ == "__main__":
    main()
