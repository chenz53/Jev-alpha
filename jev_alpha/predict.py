"""Read probabilities in a forward pass; never generate answer text."""

import argparse
import json
import random
from pathlib import Path

import torch

from jev_alpha.calibrate import probabilities
from jev_alpha.data import label_ids, prompt_ids, render, validate
from jev_alpha.io import read_jsonl


@torch.inference_mode()
def question_logits(model, tokenizer, record, question_id, order=None, max_length=4096):
    messages, order = render(record, question_id, order=order)
    ids = prompt_ids(tokenizer, messages)
    if len(ids) > max_length:
        raise ValueError("Input exceeds max_length; do not truncate decision evidence")
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([ids], device=device)
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    logits = outputs.logits[0, -1, label_ids(tokenizer)[: len(order)]].float().cpu().tolist()
    return logits, order


def predict(model, tokenizer, record, temperature=1.0, max_length=4096):
    validate(record)
    model.eval()
    result = {}
    for question_id, question in record["questions"].items():
        logits, order = question_logits(model, tokenizer, record, question_id, max_length=max_length)
        distribution = dict(zip(order, probabilities(logits, temperature)))
        result[question_id] = distribution["yes"] if question["type"] == "noul" else distribution
    return result


def prediction_rows(model, tokenizer, records, *, seed=42, max_length=4096):
    model.eval()
    rng = random.Random(seed)
    for record in records:
        validate(record, targets=True)
        for question_id, question in record["questions"].items():
            logits, order = question_logits(model, tokenizer, record, question_id, max_length=max_length)
            shuffled = list(order)
            while shuffled == order:
                rng.shuffle(shuffled)
            shuffled_logits, _ = question_logits(model, tokenizer, record, question_id, shuffled, max_length)
            yield {
                "id": record["id"],
                "dataset": record["dataset"],
                "split": record["split"],
                "question": question_id,
                "options": order,
                "logits": logits,
                "target": order.index(question["target"]),
                "shuffled": {"options": shuffled, "logits": shuffled_logits},
            }


def main():
    from swift.arguments import BaseArguments
    from swift.tuners import Swift

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    model_args = BaseArguments(model=args.model, use_hf=True, attn_impl="sdpa")
    model, processor = model_args.get_model_processor()
    if args.adapter:
        model = Swift.from_pretrained(model, args.adapter)
    tokenizer = getattr(processor, "tokenizer", processor)
    model.eval()
    records = [json.loads(row["record"]) if "record" in row else row for row in read_jsonl(args.input)]
    temperature = 1.0
    temperature_path = args.temperature
    if temperature_path is None:
        candidate = Path(args.adapter or args.model) / "temperature.json"
        if candidate.is_file():
            temperature_path = candidate
    if temperature_path:
        with open(temperature_path, encoding="utf-8") as stream:
            temperature = json.load(stream)["temperature"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as stream:
        if args.evaluate:
            outputs = prediction_rows(model, tokenizer, records, max_length=args.max_length)
        else:
            outputs = (predict(model, tokenizer, row, temperature, args.max_length) for row in records)
        for output in outputs:
            stream.write(json.dumps(output, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
