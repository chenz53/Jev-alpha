"""Frozen-reference KL over top-K non-option tokens at the answer position."""

import math
import os

import torch
import torch.nn.functional as F


def kl_settings():
    count = int(os.environ.get("JEV_KL_TOP_K", "64"))
    weight = float(os.environ.get("JEV_KL_WEIGHT", "1"))
    if count < 2 or not math.isfinite(weight) or weight < 0:
        raise ValueError("KL needs K >= 2 and a finite nonnegative weight")
    return count, weight


def select_reference(scores, counts, token_ids, top_k):
    scores = scores.detach().float().clone()
    ids = torch.as_tensor(token_ids, device=scores.device)
    counts = torch.as_tensor(counts, device=scores.device)
    if top_k < 2 or torch.any(top_k > scores.shape[-1] - counts):
        raise ValueError("Top-K exceeds the available non-option vocabulary")
    valid = torch.arange(len(ids), device=scores.device)[None, :] < counts[:, None]
    rows, columns = valid.nonzero(as_tuple=True)
    scores[rows, ids[columns]] = -torch.inf
    values, selected = scores.topk(top_k, dim=-1)
    return selected, F.log_softmax(values, dim=-1)


def conditional_kl(scores, selected, reference_logp):
    student_logp = F.log_softmax(scores.float().gather(-1, selected), dim=-1)
    return F.kl_div(student_logp, reference_logp.detach(), log_target=True, reduction="none").sum(-1)


def check_reference(model):
    if not hasattr(model, "disable_adapter") or not getattr(model, "peft_config", None):
        raise ValueError("The reference requires a PEFT LoRA model")
    for config in model.peft_config.values():
        if (
            config.peft_type != "LORA"
            or config.bias != "none"
            or config.modules_to_save
            or config.init_lora_weights is not True
            or config.use_dora
        ):
            raise ValueError("Use ordinary LoRA with frozen base weights and default initialization")
    if any(parameter.requires_grad and ".lora_" not in name for name, parameter in model.named_parameters()):
        raise ValueError("The reference requires all base parameters to stay frozen")


@torch.no_grad()
def reference_targets(model, inputs, positions, counts, token_ids, top_k):
    # Restore each module's mode because vision modules can stay in evaluation mode.
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with model.disable_adapter():
            outputs = model(**inputs)
            rows = torch.arange(len(positions), device=outputs.logits.device)
            scores = outputs.logits[rows, positions.to(rows.device) - 1]
            return select_reference(scores, counts, token_ids, top_k)
    finally:
        for module, training in modes:
            module.training = training
