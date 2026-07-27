from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl.protocol import DataProto
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor, compute_patch_imitation_objective


def test_patch_objective_uses_only_weighted_expert_action_tokens():
    log_probs = torch.tensor([[-1.0, -2.0, -3.0, -4.0]], requires_grad=True)
    token_weights = torch.tensor([[0.0, 0.0, 0.25, 0.75]])

    objective = compute_patch_imitation_objective(log_probs, token_weights)

    assert torch.isclose(objective, torch.tensor(-3.75))
    objective.backward()
    assert torch.equal(log_probs.grad, token_weights)


def test_minimized_negative_total_objective_adds_grpo_and_patch_gradients():
    parameter = torch.tensor(2.0, requires_grad=True)
    grpo_objective = 3.0 * parameter
    patch_objective = 5.0 * parameter
    patch_lambda = 0.2

    negative_total_objective = -(grpo_objective + patch_lambda * patch_objective)
    negative_total_objective.backward()

    assert torch.isclose(parameter.grad, torch.tensor(-(3.0 + patch_lambda * 5.0)))


def test_patch_objective_rejects_invalid_weights():
    log_probs = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="same shape"):
        compute_patch_imitation_objective(log_probs, torch.zeros((1, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        compute_patch_imitation_objective(log_probs, torch.tensor([[0.0, -1.0]]))


def test_zero_weight_rank_still_has_a_differentiable_zero_objective():
    log_probs = torch.tensor([[-float("inf"), -2.0]], requires_grad=True)
    objective = compute_patch_imitation_objective(log_probs, torch.zeros_like(log_probs))

    assert objective.item() == 0.0
    objective.backward()
    assert torch.equal(log_probs.grad, torch.zeros_like(log_probs))


class _TinyPolicy(nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.zeros(vocab_size))
        self.forward_calls = 0

    def forward(self, input_ids, **kwargs):
        self.forward_calls += 1
        batch_size, sequence_length = input_ids.shape
        logits = self.logit_bias.view(1, 1, -1).expand(batch_size, sequence_length, -1).clone()
        return SimpleNamespace(logits=logits)


def _actor_batch(with_patch: bool) -> DataProto:
    tensors = {
        "input_ids": torch.tensor([[4, 2, 2]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "position_ids": torch.arange(3).unsqueeze(0),
        "responses": torch.tensor([[2, 2]]),
        "response_mask": torch.ones((1, 2)),
        "old_log_probs": torch.zeros((1, 2)),
        "advantages": torch.zeros((1, 2)),
    }
    meta_info = {"temperature": 1.0}
    if with_patch:
        tensors.update(
            {
                "patch_input_ids": torch.tensor([[4, 3, 3]]),
                "patch_attention_mask": torch.ones((1, 3), dtype=torch.long),
                "patch_position_ids": torch.arange(3).unsqueeze(0),
                "patch_responses": torch.tensor([[3, 3]]),
                "patch_token_weights": torch.tensor([[0.0, 1.0]]),
            }
        )
        meta_info["patch_imitation_lambda"] = 1.0
    return DataProto.from_dict(tensors=tensors, meta_info=meta_info)


def _tiny_actor():
    config = ActorConfig(
        global_batch_size=1,
        micro_batch_size_per_device_for_update=1,
        padding_free=False,
        dynamic_batching=False,
        ppo_epochs=1,
    )
    config.global_batch_size_per_device = 1
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    actor = DataParallelPPOActor(config, policy, optimizer)
    actor.world_size = 1
    actor.log_probs_from_logits = lambda logits, labels: torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    return actor, policy


def test_update_policy_skips_extra_forward_when_disabled(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: None)
    actor, policy = _tiny_actor()

    actor.update_policy(_actor_batch(with_patch=False))

    assert policy.forward_calls == 1
    assert torch.equal(policy.logit_bias.detach(), torch.zeros_like(policy.logit_bias))


def test_update_policy_combines_patch_backward_before_same_optimizer_step(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: None)
    actor, policy = _tiny_actor()
    optimizer_steps = 0
    original_step = actor.actor_optimizer.step

    def counted_step():
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step()

    monkeypatch.setattr(actor.actor_optimizer, "step", counted_step)
    actor.update_policy(_actor_batch(with_patch=True))

    assert policy.forward_calls == 2
    assert optimizer_steps == 1
    assert policy.logit_bias.detach()[3] > 0


def test_two_minibatches_scale_and_merge_patch_gradient_at_each_original_step(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: None)

    grpo_batch = _actor_batch(with_patch=False).repeat(repeat_times=2, interleave=True)
    grpo_batch.batch["advantages"].fill_(1.0)
    combined_batch = _actor_batch(with_patch=True).repeat(repeat_times=2, interleave=True)
    combined_batch.batch["advantages"].fill_(1.0)
    # The two aligned patch rows form one globally normalized expert target.
    combined_batch.batch["patch_token_weights"].mul_(0.5)

    def capture_step_gradients(actor, batch):
        actor.actor_optimizer.param_groups[0]["lr"] = 0.0
        actor.config.max_grad_norm = 1.0e9
        captured_gradients = []
        original_step = actor.actor_optimizer.step

        def recorded_step():
            captured_gradients.append(actor.actor_module.logit_bias.grad.detach().clone())
            return original_step()

        monkeypatch.setattr(actor.actor_optimizer, "step", recorded_step)
        actor.update_policy(batch)
        return captured_gradients

    grpo_actor, grpo_policy = _tiny_actor()
    combined_actor, combined_policy = _tiny_actor()
    grpo_step_gradients = capture_step_gradients(grpo_actor, grpo_batch)
    combined_step_gradients = capture_step_gradients(combined_actor, combined_batch)

    assert grpo_policy.forward_calls == 2
    assert combined_policy.forward_calls == 4
    assert len(grpo_step_gradients) == 2
    assert len(combined_step_gradients) == 2
    assert all(torch.count_nonzero(gradient) > 0 for gradient in grpo_step_gradients)

    patch_step_gradients = [
        combined_gradient - grpo_gradient
        for combined_gradient, grpo_gradient in zip(combined_step_gradients, grpo_step_gradients, strict=True)
    ]
    # Each row owns half of the normalized target. patch_minibatch_scale=2
    # restores one complete-target gradient at each original optimizer step.
    full_target_gradient = torch.full_like(combined_policy.logit_bias, 1.0 / combined_policy.logit_bias.numel())
    full_target_gradient[3] -= 1.0
    for patch_gradient in patch_step_gradients:
        assert torch.allclose(patch_gradient, full_target_gradient)

    # Across K=2 original mini-batches, the first-order patch contribution is
    # K times the gradient of the complete globally normalized expert target.
    assert torch.allclose(torch.stack(patch_step_gradients).sum(dim=0), 2.0 * full_target_gradient)
