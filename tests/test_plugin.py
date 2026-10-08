import json
import unittest
from types import SimpleNamespace

import torch
from swift.arguments import BaseArguments
from swift.template import get_template

from jev_alpha.data import prompt_ids, render
from jev_alpha.datasets.hard_cases import records
from jev_alpha.plugin import DecisionPreprocessor
from jev_alpha.template import OptionEpochCallback


class PluginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = BaseArguments(model="Qwen/Qwen3.5-0.8B-Base", use_hf=True)
        _, processor = args.get_model_processor(load_model=False)
        cls.tokenizer = processor.tokenizer
        cls.template = get_template(processor, template_type="jev_decision", max_length=4096)
        cls.template.set_mode("train")

    def test_epoch_shuffle_and_single_answer(self):
        row = DecisionPreprocessor().preprocess(next(records()))[0]
        self.template.epoch = 0
        first = self.template.encode(row)
        seen = set()
        for epoch in range(8):
            self.template.epoch = epoch
            encoded = self.template.encode(row)
            seen.add(tuple(encoded["input_ids"]))
            self.assertEqual(sum(token != -100 for token in encoded["labels"]), 1)
        self.assertGreater(len(seen), 1)
        self.template.epoch = 0
        self.assertEqual(first, self.template.encode(row))

    def test_collator_and_public_loss_hook(self):
        rows = [DecisionPreprocessor().preprocess(record)[0] for record in list(records())[:2]]
        batch = self.template.data_collator([self.template.encode(row) for row in rows])
        counts = batch.pop("jev_option_counts")
        self.assertEqual(counts, [3, 3])
        self.assertTrue(torch.all((batch["labels"] != -100).sum(1) == 1))
        batch.pop("labels")
        batch["jev_option_counts"] = counts

        def model(**inputs):
            self.assertNotIn("jev_option_counts", inputs)
            return SimpleNamespace(logits=torch.zeros(2, 1, 26))

        outputs = self.template.compute_sft_loss(model, batch)
        self.assertEqual(outputs.jev_option_counts, counts)

    def test_serving_prefix_matches_training(self):
        record = next(records())
        row = DecisionPreprocessor().preprocess(record)[0]
        self.template.set_mode("transformers")
        try:
            encoded = self.template.encode(row)
            messages, _ = render(record, "decision")
            self.assertEqual(encoded["input_ids"][:-1], prompt_ids(self.tokenizer, messages))
        finally:
            self.template.set_mode("train")

    def test_callback_rejects_eager_dataset(self):
        trainer = SimpleNamespace(train_dataset=[])
        callback = OptionEpochCallback(None, trainer)
        with self.assertRaises(ValueError):
            callback.on_train_begin(None, None, None)

    def test_preprocessor_splits_questions(self):
        record = next(records())
        record["questions"]["second"] = dict(record["questions"]["decision"])
        rows = DecisionPreprocessor().preprocess({"record": json.dumps(record)})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(len(json.loads(row["jev_record"])["questions"]) == 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
