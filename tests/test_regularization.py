import copy
import unittest

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from jev_alpha.loss import option_loss
from jev_alpha.regularization import check_reference, conditional_kl, reference_targets, select_reference


class RegularizationTests(unittest.TestCase):
    def test_reference_selection_excludes_only_valid_options(self):
        scores = torch.tensor([[9.0, 8, 7, 6, 5], [9.0, 8, 7, 6, 5]], requires_grad=True)
        selected, logp = select_reference(scores, [2, 3], [0, 1, 2], 2)
        self.assertEqual(selected.tolist(), [[2, 3], [3, 4]])
        self.assertFalse(logp.requires_grad)
        torch.testing.assert_close(logp.exp().sum(-1), torch.ones(2))
        with self.assertRaises(ValueError):
            select_reference(scores, [2, 3], [0, 1, 2], 3)

    def test_kl_direction_and_gradient_support(self):
        teacher = torch.tensor([[4.0, 3, 2, 1, 0]], requires_grad=True)
        student = torch.tensor([[2.0, 1, 0, 1, 8]], requires_grad=True)
        selected, logq = select_reference(teacher, [2], [0, 1], 2)
        actual = conditional_kl(student, selected, logq)
        logp = F.log_softmax(student[:, [2, 3]], -1)
        expected = (logq.exp() * (logq - logp)).sum(-1)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.all(student.grad[:, [0, 1, 4]] == 0))
        self.assertTrue(torch.all(student.grad[:, [2, 3]] != 0))

    def test_conditional_kl_ignores_common_shift_and_omitted_tokens(self):
        teacher = torch.tensor([[4.0, 3, 2, 1, 0]])
        selected, logq = select_reference(teacher, [2], [0, 1], 2)
        student = teacher.clone()
        student[:, [2, 3]] -= 100
        student[:, [0, 1, 4]] += 100
        torch.testing.assert_close(conditional_kl(student, selected, logq), torch.zeros(1), atol=1e-7, rtol=0)

    def test_accumulation_matches_batch_gradient(self):
        torch.manual_seed(5)
        logits = torch.randn(2, 3, 8, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1], [-100, 2, -100]])
        teacher = torch.randn(2, 8)
        selected, logq = select_reference(teacher, [2, 3], [1, 2, 3], 3)
        loss = option_loss(logits, labels, [2, 3], [1, 2, 3])
        loss += conditional_kl(logits[torch.arange(2), [1, 0]], selected, logq).mean()
        expected = torch.autograd.grad(loss, logits)[0]
        separate = logits.detach().clone().requires_grad_()
        total = 0
        for i, position in enumerate([1, 0]):
            total += option_loss(separate[i : i + 1], labels[i : i + 1], [i + 2], [1, 2, 3], 2)
            total += conditional_kl(separate[i : i + 1, position], selected[i : i + 1], logq[i : i + 1]).sum() / 2
        torch.testing.assert_close(torch.autograd.grad(total, separate)[0], expected)

    def test_disabled_adapter_is_frozen_causal_reference(self):
        torch.manual_seed(7)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        base = LlamaForCausalLM(config).eval()
        frozen = copy.deepcopy(base)
        model = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj", "v_proj"], lora_dropout=0.1))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.2)
        check_reference(model)
        model.train()
        inputs = {"input_ids": torch.tensor([[6, 7, 8, 1]]), "use_cache": False}
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        modes = [module.training for module in model.modules()]
        selected, logq = reference_targets(model, inputs, torch.tensor([3]), [2], [1, 2], 4)
        with torch.no_grad():
            expected = select_reference(frozen(**inputs).logits[:, 2], [2], [1, 2], 4)
        torch.testing.assert_close(selected, expected[0])
        torch.testing.assert_close(logq, expected[1])
        inputs["input_ids"][0, 3] = 22
        changed = reference_targets(model, inputs, torch.tensor([3]), [2], [1, 2], 4)
        torch.testing.assert_close(logq, changed[1])
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertFalse(
            any(
                module.disable_adapters
                for module in model.modules()
                if hasattr(module, "disable_adapters") and not callable(module.disable_adapters)
            )
        )
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, before[name])
        model.get_input_embeddings().weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            check_reference(model)


if __name__ == "__main__":
    unittest.main()
