"""Public template and callback hooks for per-epoch option shuffling."""

import hashlib
import json
import math
import random
from pathlib import Path

from swift.callbacks.base import TrainerCallback
from swift.dataset import LazyLLMDataset
from swift.template import Template

from jev_alpha.data import label_ids, options, prompt_ids, render
from jev_alpha.io import write_json
from jev_alpha.regularization import check_reference, kl_settings, reference_targets


class DecisionTemplate(Template):
    epoch = 0
    seed = 42

    def encode(self, inputs, *, return_length=False, **kwargs):
        if self.packing or self.padding_free or self.sequence_parallel_size != 1:
            raise ValueError("Phase 1 requires unpacked, padded inputs and no sequence parallelism")
        record = json.loads(inputs["jev_record"])
        question_id = inputs["jev_question"]
        order = list(dict(options(record["questions"][question_id])))
        if self.is_training:
            identity = inputs["jev_record"] + question_id
            digest = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16)
            random.Random(self.seed + self.epoch + digest).shuffle(order)
        messages, _ = render(record, question_id, order=order, answer=True)
        ids = prompt_ids(self.tokenizer, messages[:-1])
        answer = label_ids(self.tokenizer)[ord(messages[-1]["content"]) - ord("A")]
        # Append the known label token to avoid tokenizer boundary changes.
        encoded = {"input_ids": ids + [answer], "labels": [-100] * len(ids) + [answer], "jev_option_count": len(order)}
        if self.max_length and len(encoded["input_ids"]) > self.max_length:
            raise ValueError("Input exceeds max_length; do not truncate decision evidence")
        if return_length:
            encoded["lengths"] = len(encoded["input_ids"])
        return encoded

    def data_collator(self, batch, *, padding_to=None):
        batch = [dict(row) for row in batch]
        counts = [row.pop("jev_option_count") for row in batch]
        result = super().data_collator(batch, padding_to=padding_to)
        result["jev_option_counts"] = counts
        result["jev_answer_positions"] = (result["labels"] != -100).long().argmax(dim=1)
        return result

    def compute_sft_loss(self, model, inputs, num_items_in_batch=None, trainer=None):
        inputs = dict(inputs)
        counts = inputs.pop("jev_option_counts")
        positions = inputs.pop("jev_answer_positions")
        reference = None
        if trainer is not None and trainer.args.loss_type == "jev_options_kl":
            base = trainer.accelerator.unwrap_model(model)
            reference = reference_targets(base, inputs, positions, counts, label_ids(self.tokenizer), kl_settings()[0])
        outputs = super().compute_sft_loss(model, inputs, num_items_in_batch, trainer)
        outputs.jev_option_counts = counts
        if reference is not None:
            outputs.jev_reference = reference
        return outputs


class OptionEpochCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if not isinstance(self.trainer.train_dataset, LazyLLMDataset):
            raise ValueError("Option shuffling requires --lazy_tokenize true")
        if args.dataloader_num_workers != 0:
            raise ValueError("Option shuffling requires --dataloader_num_workers 0")
        if args.loss_type not in {"jev_options", "jev_options_kl"} or args.use_logits_to_keep:
            raise ValueError("Use an option loss and disable use_logits_to_keep")
        if args.loss_type == "jev_options_kl":
            if self.trainer.accelerator.num_processes != 1:
                raise ValueError("The KL reference currently supports one training process")
            check_reference(self.trainer.accelerator.unwrap_model(self.trainer.model))
            kl_settings()
        self.trainer.template.seed = args.seed
        count, weight = kl_settings()
        self.loss_settings = {
            "loss_type": args.loss_type,
            "top_k": count if args.loss_type == "jev_options_kl" else None,
            "kl_weight": weight if args.loss_type == "jev_options_kl" else 0,
            "temperature": 1,
            "direction": "reference_to_student",
            "normalization": "selected_non_option_tokens",
            "position": "before_answer",
            "reference": "frozen_base_with_adapter_disabled",
        }
        write_json(Path(args.output_dir) / "decision_loss.json", self.loss_settings)

    def on_save(self, args, state, control, **kwargs):
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "decision_loss.json"
        if not path.exists():
            write_json(path, self.loss_settings)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.trainer.template.epoch = math.floor(state.epoch or 0)
