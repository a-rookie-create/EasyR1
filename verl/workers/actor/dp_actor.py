# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


def compute_patch_imitation_objective(log_probs: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    """Return the weighted expert-action log-likelihood to maximize.

    ``token_weights`` is constructed by the UI-S1 driver so that thinking and
    padding tokens have zero weight.  Its non-zero entries average tokens
    within one expert action, patch samples within one task-step, and finally
    the distinct patched task-steps in the selected rollout batch.
    """
    if log_probs.shape != token_weights.shape:
        raise ValueError(
            "Patch log-probabilities and token weights must have the same shape, "
            f"got {tuple(log_probs.shape)} and {tuple(token_weights.shape)}."
        )
    if torch.any(token_weights < 0):
        raise ValueError("Patch imitation token weights must be non-negative.")
    supervised_tokens = token_weights > 0
    return torch.sum(log_probs.masked_select(supervised_tokens) * token_weights.masked_select(supervised_tokens))


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    @torch.no_grad()
    def _snapshot_trainable_grads(self) -> dict[int, torch.Tensor]:
        """Save the GRPO gradient before adding the patch-imitation term.

        UI-S1 trains LoRA adapters, so this snapshot is small relative to the
        model and lets us measure the *actual* gradient contributed by each
        objective without an additional forward/backward pass.
        """
        return {
            id(parameter): parameter.grad.detach().clone()
            for parameter in self.actor_module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        }

    @torch.no_grad()
    def _accumulate_patch_gradient(
        self, accumulated_patch_grads: dict[int, torch.Tensor], grads_before_patch: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Add this micro-batch's patch gradient to the running total."""
        for parameter in self.actor_module.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            previous_grad = grads_before_patch.get(id(parameter))
            patch_grad = parameter.grad.detach()
            if previous_grad is not None:
                patch_grad = patch_grad - previous_grad
            if id(parameter) in accumulated_patch_grads:
                accumulated_patch_grads[id(parameter)].add_(patch_grad)
            else:
                accumulated_patch_grads[id(parameter)] = patch_grad.clone()
        return accumulated_patch_grads

    @torch.no_grad()
    def _gradient_component_metrics(self, patch_grads: dict[int, torch.Tensor]) -> dict[str, float]:
        """Measure GRPO, patch, and combined gradients before clipping.

        FSDP stores parameter shards on each rank.  Summing squared shard
        norms across ranks therefore gives the true global norm, rather than
        a per-rank value that changes with the number of GPUs.
        """
        parameters = tuple(self.actor_module.parameters())
        first_gradient = next(
            (parameter.grad for parameter in parameters if parameter.grad is not None), None
        )
        if first_gradient is not None:
            metric_device = first_gradient.device
        elif parameters:
            metric_device = parameters[0].device
        else:
            # Actor modules always have parameters, but keep this utility
            # usable for a minimal unit-test module as well.
            metric_device = torch.device("cpu")

        sums = torch.zeros(3, device=metric_device, dtype=torch.float32)
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            patch_grad = patch_grads.get(id(parameter))
            combined_grad = parameter.grad
            if patch_grad is None and combined_grad is None:
                continue

            if combined_grad is None:
                # This should be unreachable after backward, but retain a
                # well-defined decomposition if an unusual module clears a
                # gradient between the two objective terms.
                grpo_grad = -patch_grad
                sums[0] += torch.sum(grpo_grad.float().square())
                sums[1] += torch.sum(patch_grad.float().square())
                continue

            combined_grad = combined_grad.detach().float()
            if patch_grad is None:
                patch_grad = torch.zeros_like(combined_grad)
            else:
                patch_grad = patch_grad.float()
            grpo_grad = combined_grad - patch_grad
            sums[0] += torch.sum(grpo_grad.square())
            sums[1] += torch.sum(patch_grad.square())
            sums[2] += torch.sum(combined_grad.square())

        if isinstance(self.actor_module, FSDP) and dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)

        grpo_norm, patch_norm, combined_norm = torch.sqrt(sums.clamp_min(0.0)).tolist()
        cosine_denominator = grpo_norm * patch_norm
        # ||g_rl + g_patch||^2 = ||g_rl||^2 + ||g_patch||^2 + 2 <g_rl, g_patch>.
        cosine = (
            0.0
            if cosine_denominator == 0.0
            else (combined_norm**2 - grpo_norm**2 - patch_norm**2) / (2.0 * cosine_denominator)
        )
        return {
            "actor/grpo_grad_norm": grpo_norm,
            "actor/patch_imitation_grad_norm": patch_norm,
            "actor/combined_grad_norm": combined_norm,
            "actor/grpo_patch_grad_cosine": max(-1.0, min(1.0, cosine)),
        }

    @staticmethod
    @torch.no_grad()
    def _patch_target_metrics(log_probs: torch.Tensor, token_weights: torch.Tensor) -> dict[str, float] | None:
        """Return target-fit metrics for expert action tokens in this update.

        Target NLL is the standard, directly comparable discrepancy between
        the policy distribution and a discrete expert target.  Its exponential
        (perplexity) and the mean probability assigned to target tokens make
        the same signal easier to interpret during training.
        """
        supervised = token_weights > 0
        weights = token_weights.masked_select(supervised).float()
        selected_log_probs = log_probs.masked_select(supervised).float()
        totals = torch.stack(
            (
                weights.sum(),
                torch.sum(weights * selected_log_probs),
                torch.sum(weights * selected_log_probs.exp()),
            )
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        weight_sum, weighted_log_prob, weighted_target_probability = totals.tolist()
        if weight_sum <= 0.0:
            return None
        target_nll = -weighted_log_prob / weight_sum
        return {
            "actor/patch_target_nll": target_nll,
            "actor/patch_target_perplexity": float(
                torch.exp(torch.tensor(min(target_nll, 20.0))).item()
            ),
            "actor/patch_target_probability": weighted_target_probability / weight_sum,
        }

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]
        patch_keys = [
            "patch_input_ids",
            "patch_attention_mask",
            "patch_position_ids",
            "patch_responses",
            "patch_token_weights",
        ]
        present_patch_keys = [key for key in patch_keys if key in data.batch]
        if present_patch_keys and len(present_patch_keys) != len(patch_keys):
            missing_patch_keys = sorted(set(patch_keys) - set(present_patch_keys))
            raise KeyError(f"Incomplete patch imitation actor batch; missing keys: {missing_patch_keys}")
        patch_imitation_enabled = len(present_patch_keys) == len(patch_keys)
        patch_imitation_lambda = float(data.meta_info.get("patch_imitation_lambda", 0.0))
        if patch_imitation_enabled:
            if patch_imitation_lambda <= 0.0:
                raise ValueError("Attached patch imitation data requires a positive patch_imitation_lambda.")
            select_keys.extend(patch_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        # EasyR1 takes one optimizer step per actor mini-batch.  Patch weights
        # describe an average over the complete rollout batch, so this factor
        # keeps its first-order epoch contribution on the same scale as the
        # sequence of equally sized GRPO mini-batch means.
        patch_minibatch_scale = len(mini_batches)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                accumulated_patch_grads: dict[int, torch.Tensor] = {}
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        tau_positive=self.config.tau_positive,
                        tau_negative=self.config.tau_negative,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                    else:
                        loss = pg_loss

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    # This is the complete GRPO update term: policy gradient
                    # plus the optional KL regularizer.
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        batch_metrics["actor/kl_divergence"] = kl_loss.detach().item()
                        batch_metrics["actor/kl_regularization_loss"] = (
                            kl_loss.detach().item() * self.config.kl_coef
                        )
                        batch_metrics["actor/kl_coef"] = self.config.kl_coef

                    if patch_imitation_enabled:
                        patch_model_inputs = {
                            "input_ids": model_inputs["patch_input_ids"],
                            "attention_mask": model_inputs["patch_attention_mask"],
                            "position_ids": model_inputs["patch_position_ids"],
                            "responses": model_inputs["patch_responses"],
                        }
                        if "multi_modal_inputs" in model_inputs:
                            patch_model_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]

                        patch_log_probs = self._forward_micro_batch(patch_model_inputs, temperature=temperature)
                        patch_objective = compute_patch_imitation_objective(
                            patch_log_probs, model_inputs["patch_token_weights"]
                        )
                        patch_target_metrics = self._patch_target_metrics(
                            patch_log_probs, model_inputs["patch_token_weights"]
                        )
                        if patch_target_metrics is not None:
                            batch_metrics.update(patch_target_metrics)
                        # FSDP averages gradients across data-parallel ranks.
                        # The driver weights sum to one globally, hence
                        # world_size restores the intended global objective.
                        patch_gradient_term = (
                            patch_imitation_lambda * patch_objective * self.world_size * patch_minibatch_scale
                        )
                        grads_before_patch = self._snapshot_trainable_grads()
                        (-patch_gradient_term).backward()
                        accumulated_patch_grads = self._accumulate_patch_gradient(
                            accumulated_patch_grads, grads_before_patch
                        )
                        batch_metrics["actor/patch_objective"] = patch_objective.detach().item()
                        batch_metrics["actor/patch_lambda"] = patch_imitation_lambda

                    append_to_dict(metrics, batch_metrics)

                if patch_imitation_enabled:
                    append_to_dict(metrics, self._gradient_component_metrics(accumulated_patch_grads))
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
