"""Cross-entropy over only the valid option letters."""

import torch
import torch.nn.functional as F
from swift.loss import BaseLoss

from jev_alpha.data import label_ids
from jev_alpha.regularization import conditional_kl, kl_settings


def option_loss(logits, labels, counts, token_ids, num_items_in_batch=None):
    if logits.shape[:2] != labels.shape:
        raise ValueError("Option loss needs full sequence logits; disable use_logits_to_keep")
    mask = labels != -100
    if not torch.all(mask.sum(1) == 1):
        raise ValueError("Each question must supervise exactly one answer token")
    rows, positions = mask.nonzero(as_tuple=True)
    if torch.any(positions == 0):
        raise ValueError("Each answer must have a preceding prompt token")
    ids = torch.as_tensor(token_ids, device=logits.device)
    counts = torch.as_tensor(counts, device=logits.device)
    if counts.shape != rows.shape or torch.any((counts < 2) | (counts > len(ids))):
        raise ValueError("Invalid option counts")
    scores = logits[rows, positions - 1].float().index_select(-1, ids)
    valid = torch.arange(len(ids), device=logits.device)[None, :] < counts[:, None]
    matches = labels[rows, positions, None].eq(ids[None, :]) & valid
    if not torch.all(matches.sum(1) == 1):
        raise ValueError("The answer token must be a valid option letter")
    scores = scores.masked_fill(~valid, -torch.inf)
    total = F.cross_entropy(scores, matches.long().argmax(1), reduction="sum")
    # One supervised token per question makes accumulation count questions.
    denominator = len(rows) if num_items_in_batch is None else num_items_in_batch
    return total / denominator


class OptionLoss(BaseLoss):
    def __call__(self, outputs, labels, *, num_items_in_batch=None, **kwargs):
        return option_loss(
            outputs.logits,
            labels,
            outputs.jev_option_counts,
            label_ids(self.trainer.template.tokenizer),
            num_items_in_batch,
        )


class OptionKLLoss(OptionLoss):
    def __call__(self, outputs, labels, *, num_items_in_batch=None, **kwargs):
        ce = super().__call__(outputs, labels, num_items_in_batch=num_items_in_batch)
        rows, positions = (labels != -100).nonzero(as_tuple=True)
        scores = outputs.logits[rows, positions - 1]
        kl = conditional_kl(scores, *outputs.jev_reference)
        denominator = len(rows) if num_items_in_batch is None else num_items_in_batch
        mode = "train" if self.trainer.model.training else "eval"
        self.trainer.custom_metrics[mode]["option_ce"].update(ce.detach() * denominator / len(rows))
        self.trainer.custom_metrics[mode]["non_option_kl"].update(kl.detach())
        return ce + kl_settings()[1] * kl.sum() / denominator
