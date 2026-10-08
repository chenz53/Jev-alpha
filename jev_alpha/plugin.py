"""Load with --external_plugins jev_alpha/plugin.py."""

import json
import os

from swift.callbacks import callbacks_map
from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset
from swift.loss import loss_map
from swift.template import TemplateMeta, register_template

from jev_alpha.data import render, validate
from jev_alpha.loss import OptionKLLoss, OptionLoss
from jev_alpha.template import DecisionTemplate, OptionEpochCallback


class DecisionPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        if "record" in row:
            row = json.loads(row["record"])
        validate(row, targets=True)
        result = []
        for question_id in row["questions"]:
            # Store JSON to avoid Arrow adding null fields to varying questions.
            record = {"state": row["state"], "questions": {question_id: row["questions"][question_id]}}
            messages, _ = render(record, question_id, answer=True)
            result.append({"messages": messages, "jev_record": json.dumps(record), "jev_question": question_id})
        return result


register_template(
    TemplateMeta(
        "jev_decision", prefix=[], prompt=["{{QUERY}}"], chat_sep=None, suffix=[], template_cls=DecisionTemplate
    ),
    exist_ok=True,
)
loss_map["jev_options"] = OptionLoss
loss_map["jev_options_kl"] = OptionKLLoss
callbacks_map["jev_epoch"] = OptionEpochCallback
register_dataset(
    DatasetMeta(
        dataset_name="jev_train",
        dataset_path=os.path.abspath(os.environ.get("JEV_TRAIN", "data/train.jsonl")),
        preprocess_func=DecisionPreprocessor(),
    ),
    exist_ok=True,
)
