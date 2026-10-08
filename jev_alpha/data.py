"""Validate records and render the same messages for training and serving."""

import json
import math
from string import ascii_uppercase


LABELS = ascii_uppercase


def options(question):
    kind = question.get("type")
    if kind not in {"noul", "choice", "score"}:
        raise ValueError("Question type must be noul, choice, or score")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError("Question instructions must be a nonempty string")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("Question criteria must be an object")
    if kind == "noul":
        if criteria and set(criteria) != {"no", "yes"}:
            raise ValueError("noul criteria must be empty or contain no and yes")
        criteria = criteria or {"no": "No", "yes": "Yes"}
    if not 2 <= len(criteria) <= len(LABELS):
        raise ValueError("Each question needs 2 to 26 options")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in criteria.items()):
        raise ValueError("Option keys and descriptions must be nonempty strings")
    keys = list(criteria)
    if kind == "score":
        try:
            levels = [float(k) for k in keys]
        except ValueError as error:
            raise ValueError("Score keys must be numeric levels") from error
        if not all(math.isfinite(x) for x in levels) or len(set(levels)) != len(levels):
            raise ValueError("Score levels must be finite and unique")
        keys.sort(key=float)
    return [(key, criteria[key]) for key in keys]


def validate(record, *, targets=False):
    if not isinstance(record.get("state"), str):
        raise ValueError("State must be a string")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Questions must be a nonempty object")
    for key, question in questions.items():
        if not isinstance(key, str) or not key or not isinstance(question, dict):
            raise ValueError("Question keys must be strings and questions must be objects")
        keys = dict(options(question))
        if targets and question.get("target") not in keys:
            raise ValueError(f"Question {key} needs a target option key")


def render(record, question_id, *, order=None, answer=False):
    validate(record, targets=answer)
    question = record["questions"][question_id]
    criteria = dict(options(question))
    order = list(criteria) if order is None else list(order)
    if len(order) != len(criteria) or set(order) != set(criteria):
        raise ValueError("Option order must contain each option exactly once")
    # JSON quoting keeps option descriptions and state boundaries visible.
    lines = [
        "Read the state and select one option.",
        f"State: {json.dumps(record['state'], ensure_ascii=False)}",
        f"Question: {json.dumps(question['instructions'], ensure_ascii=False)}",
        f"Type: {question['type']}",
        "Options:",
    ]
    for label, key in zip(LABELS, order):
        description = criteria[key]
        if question["type"] == "score":
            description = f"Level {key}: {description}"
        lines.append(f"{label}: {json.dumps(description, ensure_ascii=False)}")
    lines.append("Return one option letter.")
    messages = [{"role": "user", "content": "\n".join(lines)}]
    if answer:
        messages.append({"role": "assistant", "content": LABELS[order.index(question["target"])]})
    return messages, order


def label_ids(tokenizer):
    encoded = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError("The tokenizer must encode each option letter as one token")
    ids = [item[0] for item in encoded]
    if len(set(ids)) != len(ids):
        raise ValueError("Option letters must have distinct token IDs")
    return ids


def prompt_ids(tokenizer, messages):
    if not tokenizer.chat_template:
        raise ValueError("The tokenizer needs a chat template")
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
    )
