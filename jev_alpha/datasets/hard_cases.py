"""Small, fixed synthetic test datasets. These are not benchmark results."""

import argparse
import json
from pathlib import Path


def records():
    cases = [
        (
            "The parcel is at the depot. No delivery date is recorded.",
            "When will the parcel arrive?",
            {"today": "Today", "tomorrow": "Tomorrow", "unknown": "The state does not give a date"},
            "unknown",
        ),
        (
            "The sensor reads green. The choices name only other colors.",
            "Which option matches the sensor?",
            {"red": "Red", "blue": "Blue", "none": "None of the above"},
            "none",
        ),
        (
            "One witness says the door was open. Another says it was closed. Neither account has priority.",
            "What door status does the evidence establish?",
            {"open": "Open", "closed": "Closed", "unknown": "The evidence conflicts"},
            "unknown",
        ),
        (
            "The account has a valid password. Its last payment failed.",
            "Which team handles the recorded problem?",
            {"login": "Login access", "payments": "Payments", "other": "Other problems"},
            "payments",
        ),
        (
            "The sample contains copper and no iron. Both options refer to the same sample.",
            "Which metal is present?",
            {"iron": "Iron", "copper": "Copper", "none": "Neither"},
            "copper",
        ),
        (
            "The log records event B before event A. Event C has no time.",
            "Which event definitely comes first among A and B?",
            {"a": "A", "b": "B", "unknown": "Cannot determine"},
            "b",
        ),
    ]
    for index, (state, question, criteria, target) in enumerate(cases):
        yield make_record("synthetic-choice", index, state, "choice", question, criteria, target)
    binary = [
        ("The laboratory confirms that vial K contains salt.", "Does vial K contain salt?", "yes"),
        ("All switches are off. Switch R is one of these switches.", "Is switch R on?", "no"),
        ("The report omits the engine temperature.", "Does the state establish that the engine is hot?", "no"),
        ("There are exactly three birds in the cage.", "Are there more than two birds in the cage?", "yes"),
        ("No visitor entered the building on Monday.", "Did a visitor enter the building on Monday?", "no"),
        ("The first test passed. The second test failed.", "Did at least one test fail?", "yes"),
    ]
    for index, (state, question, target) in enumerate(binary):
        yield make_record("synthetic-noul", index, state, "noul", question, {}, target)
    for index, (state, target) in enumerate(
        [
            ("The inspection found zero damaged panels.", "0"),
            ("Exactly one panel has a crack. The other panels are intact.", "1"),
            ("Two panels have cracks and three are intact.", "2"),
            ("The only damage is on panel Z.", "1"),
            ("Every panel passed inspection without damage.", "0"),
            ("Panels P and Q are damaged. All other panels are intact.", "2"),
        ]
    ):
        yield make_record(
            "synthetic-score",
            index,
            state,
            "score",
            "How many panels are damaged?",
            {"0": "No panels", "1": "One panel", "2": "Two panels"},
            target,
        )


def make_record(dataset, index, state, kind, instructions, criteria, target):
    return {
        "id": str(index),
        "dataset": dataset,
        "split": "test",
        "state": state,
        "questions": {
            "decision": {"type": kind, "instructions": instructions, "criteria": criteria, "target": target}
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/hard_cases.jsonl")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as stream:
        for record in records():
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
