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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import re
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from itertools import combinations
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from jinja2 import Template
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.dataset import get_image_size, process_image, qwen_coordinate_transforms
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker, TrainingProgressLogger
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    get_length_metric_samples,
    reduce_length_metric_samples,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def _compact_ui_action(action: Any) -> Any:
    """Drop legacy null placeholders before using an expert action as reward GT."""
    if not isinstance(action, dict):
        return action
    return {key: value for key, value in action.items() if value is not None}


def _json_log_default(value: Any) -> Any:
    """Convert tensor/array values retained in rollout audit records."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _as_object_array(values: list[Any]) -> np.ndarray:
    """Keep one Python value per rollout row, even when nested lengths match.

    ``np.array(values, dtype=object)`` still creates a 2-D object array when
    every nested list in a rollout wave has the same length. In UI-S1 that
    length is the retained image-history size, so different trajectory steps
    later fail to concatenate. Preallocating a one-dimensional object array
    preserves each nested value as a single row item.
    """
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _select_most_diverse_rollout_subset(
    candidate_states: list[dict[str, Any]], selection_scores: dict[str, float], selected_count: int
) -> tuple[list[dict[str, Any]], float]:
    """Choose the fixed-size candidate subset with the greatest score spread.

    UI-S1's default candidate pool is eight and it retains four trajectories,
    so this evaluates only C(8, 4)=70 combinations. Iteration order makes ties
    deterministic and preserves the earlier candidates when their diversity is
    identical.
    """
    if len(candidate_states) < selected_count:
        raise ValueError("candidate pool must contain at least selected_count trajectories")

    from examples.ui_s1.advantage_ui_s1 import rollout_score_std

    best_subset: tuple[dict[str, Any], ...] | None = None
    best_std = float("-inf")
    for subset in combinations(candidate_states, selected_count):
        diversity_std = rollout_score_std([selection_scores[state["traj_uid"]] for state in subset])
        if diversity_std > best_std:
            best_subset = subset
            best_std = diversity_std
    assert best_subset is not None
    return list(best_subset), best_std


def _thinking_for_history(response: str) -> str | None:
    """Keep rollout reasoning but deliberately discard every predicted action."""
    match = re.search(r"<thinking>\s*(.*?)\s*</thinking>", response or "", flags=re.DOTALL)
    if not match or not match.group(1).strip():
        return None
    return f"<thinking>\n{match.group(1).strip()}\n</thinking>"


def _history_text(history: list[str]) -> str:
    if not history:
        return "None"
    return "\n".join(f"{idx}. {action}" for idx, action in enumerate(history, start=1))


def _build_semi_online_prompt(
    goal: str, history: list[str], image_count: int = 1, format_prompt: Optional[str] = None
) -> str:
    if image_count < 1:
        raise ValueError(f"semi-online prompts require at least one image, got {image_count}")
    screenshot_context = []
    for image_index in range(image_count):
        label = "Current screenshot (use this image to choose the next action)" if image_index == image_count - 1 else (
            f"Previous screenshot {image_index + 1} (chronological context only)"
        )
        screenshot_context.append(f"{label}:\n<image>")
    content = (
        "\n".join(screenshot_context)
        + f"\n\nGoal: {goal.strip()}\n\n"
        "Previous model reasoning (actions deliberately omitted):\n"
        f"{_history_text(history)}\n\n"
        "Predict the next Android GUI action. Return exactly <thinking>...</thinking> followed by "
        "<tool_call>{\"name\":\"mobile_use\",\"arguments\":{...}}</tool_call>."
    )
    if format_prompt:
        return Template(format_prompt.strip()).render(content=content)
    return content


def _attach_ui_s1_advantages(data: DataProto, config: PPOConfig) -> dict[str, float]:
    """Attach UI-S1 equations (9)-(12) without changing the generic GRPO path."""
    # Keep UI-S1 experiment code out of EasyR1's normal trainer import path.
    from examples.ui_s1.advantage_ui_s1 import compute_ui_s1_advantages

    result = compute_ui_s1_advantages(
        token_level_rewards=data.batch["token_level_scores"],
        response_mask=data.batch["response_mask"],
        task_ids=data.non_tensor_batch["task_id"],
        trajectory_ids=data.non_tensor_batch["traj_uid"],
        step_ids=data.non_tensor_batch["step_id"],
        extract_matches=data.non_tensor_batch["extract_match"],
        gamma=config.algorithm.semi_online_gamma,
        step_advantage_weight=config.algorithm.semi_online_step_advantage_weight,
        episode_advantage_weight=config.algorithm.semi_online_episode_advantage_weight,
        normalize_by_std=config.algorithm.semi_online_normalize_by_std,
    )
    data.batch["advantages"] = result.advantages
    data.batch["returns"] = result.returns
    data.batch["ui_s1_step_returns"] = result.step_returns
    data.batch["ui_s1_episode_returns"] = result.episode_returns
    data.batch["ui_s1_episode_advantages"] = result.episode_advantages
    data.batch["ui_s1_step_advantages"] = result.step_advantages

    valid_advantages = result.advantages.masked_select(data.batch["response_mask"].bool())
    advantage_std = torch.std(valid_advantages, unbiased=False).item() if valid_advantages.numel() > 1 else 0.0
    return {
        "uis1/advantage_std": advantage_std,
        "uis1/step_reward_mean": result.step_rewards.mean().item(),
        "uis1/step_return_mean": result.step_returns.mean().item(),
        "uis1/episode_return_mean": result.episode_returns.mean().item(),
    }


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
        progress_logger: Optional[TrainingProgressLogger] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.progress_logger = progress_logger
        self._semi_online_format_prompt = None
        if config.data.format_prompt:
            with open(config.data.format_prompt, encoding="utf-8") as f:
                self._semi_online_format_prompt = f.read()

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None
        self._semi_online_rollout_records: list[dict[str, Any]] = []

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
            self.steps_per_epoch = len(train_dataloader)
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.steps_per_epoch = num_examples // config.data.rollout_batch_size
            self.training_steps = self.steps_per_epoch * config.trainer.total_epochs
        else:
            self.steps_per_epoch = len(train_dataloader)
            self.training_steps = self.steps_per_epoch * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def _progress(self, phase: str, status: str, **fields: Any) -> None:
        """Emit a concise milestone without changing framework console logging."""
        progress_logger = getattr(self, "progress_logger", None)
        if progress_logger is not None:
            progress_logger.log(phase, status, step=getattr(self, "global_step", None), **fields)

    def _should_save_checkpoint(self) -> bool:
        # A time-based checkpoint is deliberately checked only after the actor
        # update has finished. This guarantees that every periodic checkpoint
        # contains the most recently updated policy instead of a half-finished
        # optimization step. The interval is measured between completed saves.
        interval_seconds = getattr(self.config.trainer, "save_interval_seconds", 0.0)
        last_checkpoint_at = getattr(self, "_last_checkpoint_monotonic", None)
        by_time = (
            interval_seconds > 0
            and last_checkpoint_at is not None
            and time.monotonic() - last_checkpoint_at >= interval_seconds
        )
        by_step = self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0
        epoch_interval = self.config.trainer.save_every_n_epochs
        by_epoch = (
            epoch_interval > 0
            and self.steps_per_epoch > 0
            and self.global_step % (self.steps_per_epoch * epoch_interval) == 0
        )
        checkpoint_already_saved = getattr(self, "_last_checkpoint_step", None) == self.global_step
        return not checkpoint_already_saved and (by_time or by_step or by_epoch)

    def _record_checkpoint_saved(self) -> None:
        """Record the completed save so time-based cadence starts after I/O."""
        self._last_checkpoint_step = self.global_step
        self._last_checkpoint_monotonic = time.monotonic()

    def _append_semi_online_rollout_log_entries(self, entries: list[dict[str, Any]]) -> None:
        """Append visible JSONL records while semi-online generation is still running."""
        if not entries:
            return
        log_path = self.config.trainer.rollout_log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, default=_json_log_default) + "\n")
            # Keep the records observable by `tail -f` without waiting for a
            # training step or the Python process to exit.
            handle.flush()

    def _write_semi_online_rollout_logs(
        self, records: list[dict[str, Any]], candidate_attempt: int, advantage_std: float
    ) -> None:
        self._append_semi_online_rollout_log_entries(
            [
                {
                    "record_type": "rollout",
                    "global_step": self.global_step,
                    "candidate_attempt": candidate_attempt,
                    "selected_batch": True,
                    "advantage_std": advantage_std,
                    **record,
                }
                for record in records
            ]
        )

    def _write_semi_online_rollout_progress(self, states: list[dict[str, Any]]) -> None:
        """Log completed candidate rollouts before diversity selection is known."""
        entries = []
        for state in states:
            termination_reason = (
                "expert_trajectory_exhausted"
                if state["step_pos"] >= len(state["steps"])
                else "patch_threshold_exhausted"
            )
            entries.append(
                {
                    "record_type": "rollout_progress",
                    "global_step": self.global_step,
                    "task_id": state["task_id"],
                    "trajectory_id": state["traj_uid"],
                    "rollout_id": state["rollout_id"],
                    "goal": state["goal"],
                    "expert_step_count": len(state["steps"]),
                    "generated_step_count": len(state["events"]),
                    "patch_count": state["patch_count"],
                    "patch_threshold": self.config.algorithm.patch_threshold,
                    "episode_return": sum(float(event["reward"].get("overall", 0.0)) for event in state["events"]),
                    "selection_status": "pending",
                    "reached_trajectory_end": state["step_pos"] >= len(state["steps"]),
                    "termination_reason": termination_reason,
                    "events": state["events"],
                }
            )
        self._append_semi_online_rollout_log_entries(entries)

    def _write_semi_online_update_result(self, records: list[dict[str, Any]]) -> None:
        """Append one explicit confirmation after a selected batch updates the actor."""
        if not records:
            return
        self._append_semi_online_rollout_log_entries(
            [
                {
                    "record_type": "actor_update",
                    "global_step": self.global_step,
                    "update_completed": True,
                    "trajectory_ids": [record["trajectory_id"] for record in records],
                }
            ]
        )

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            self._progress("CHECKPOINT_LOAD", "SKIP", reason="no_checkpoint_requested")
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        checkpoint_started = time.perf_counter()
        self._progress("CHECKPOINT_LOAD", "START", path=load_checkpoint_path)
        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")
        self._progress("CHECKPOINT_LOAD", "END", elapsed_s=time.perf_counter() - checkpoint_started)

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        validation_started = time.perf_counter()
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        total_batches = len(self.val_dataloader)
        repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
        progress_interval = self.config.trainer.progress_validation_interval
        if progress_interval < 1:
            raise ValueError("trainer.progress_validation_interval must be positive")
        completed_prompts = 0
        print("Start validation...")
        self._progress(
            "VALIDATION",
            "START",
            total_batches=total_batches,
            batch_size=self.config.data.val_batch_size,
            generation_n=repeat_times,
        )
        validation_sync_started = time.perf_counter()
        self._progress("VALIDATION_ENGINE_SYNC", "START", direction="actor_to_vllm")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        self._progress("VALIDATION_ENGINE_SYNC", "END", elapsed_s=time.perf_counter() - validation_sync_started)
        for batch_index, batch_dict in enumerate(self.val_dataloader, start=1):
            test_batch = DataProto.from_single_dict(batch_dict)
            completed_prompts += len(test_batch)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            # Collect individual lengths across the complete validation set.
            # Reducing batch-level max/min is wrong when val_batch_size=1.
            for key, values in get_length_metric_samples(test_batch).items():
                length_metrics_lst[key].extend(values.detach().cpu().tolist())

            if batch_index % progress_interval == 0 or batch_index == total_batches:
                overall_rewards = reward_metrics_lst.get("overall", [])
                self._progress(
                    "VALIDATION",
                    "PROGRESS",
                    completed_batches=batch_index,
                    total_batches=total_batches,
                    completed_prompts=completed_prompts,
                    overall_reward_mean=float(np.mean(overall_rewards)) if overall_rewards else None,
                    elapsed_s=time.perf_counter() - validation_started,
                )

        validation_release_started = time.perf_counter()
        self._progress("VALIDATION_ENGINE_RELEASE", "START")
        self.actor_rollout_ref_wg.release_rollout_engine()
        self._progress("VALIDATION_ENGINE_RELEASE", "END", elapsed_s=time.perf_counter() - validation_release_started)
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {
            f"val_{key}": value for key, value in reduce_length_metric_samples(length_metrics_lst).items()
        }
        print("Finish validation.")
        validation_result = {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}
        self._progress(
            "VALIDATION",
            "END",
            elapsed_s=time.perf_counter() - validation_started,
            completed_prompts=completed_prompts,
            overall_reward=validation_result.get("val/overall_reward"),
            accuracy_reward=validation_result.get("val/accuracy_reward"),
        )
        return validation_result

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _encode_semi_online_examples(self, examples: list[dict[str, Any]], meta_info: dict[str, Any]) -> DataProto:
        tensors = defaultdict(list)
        non_tensors = defaultdict(list)

        for example in examples:
            messages = [{"role": "user", "content": []}]
            prompt = example["prompt"]
            content = []
            for i, text in enumerate(prompt.split("<image>")):
                if i != 0:
                    content.append({"type": "image"})
                if text:
                    content.append({"type": "text", "text": text})
            messages[0]["content"] = content

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example["images"]
            original_sizes = [get_image_size(image) for image in images]
            processed_images = [process_image(image, meta_info["min_pixels"], meta_info["max_pixels"]) for image in images]
            model_inputs = self.processor(processed_images, [raw_prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from ..models.transformers.qwen3_vl import get_rope_index
                else:
                    from ..models.transformers.qwen2_vl import get_rope_index

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                    attention_mask=attention_mask,
                )
                text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

            input_ids, attention_mask, position_ids = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="error",
            )
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > self.config.data.max_prompt_length:
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]

            tensors["input_ids"].append(input_ids)
            tensors["attention_mask"].append(attention_mask)
            tensors["position_ids"].append(position_ids)
            non_tensors["raw_prompt_ids"].append(raw_prompt_ids)
            non_tensors["multi_modal_data"].append({"images": images})
            non_tensors["ground_truth"].append(example["answer"])
            non_tensors["uid"].append(example["uid"])
            non_tensors["traj_uid"].append(example["traj_uid"])
            non_tensors["step_id"].append(example["step_id"])
            non_tensors["task_id"].append(example["task_id"])
            non_tensors["rollout_id"].append(example["rollout_id"])
            if (
                original_sizes
                and self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                image_coordinate_transforms = qwen_coordinate_transforms(
                    original_sizes, model_inputs.get("image_grid_thw", None), self.processor.image_processor
                )
                coordinate_transform = image_coordinate_transforms[-1]
            else:
                coordinate_transform = None
                image_coordinate_transforms = None
            non_tensors["coordinate_transform"].append(coordinate_transform)
            non_tensors["image_coordinate_transforms"].append(image_coordinate_transforms)

        tensor_batch = {key: torch.stack(value, dim=0) for key, value in tensors.items()}
        non_tensor_batch = {key: _as_object_array(value) for key, value in non_tensors.items()}
        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

    def _filter_grpo_groups(self, batch: DataProto) -> DataProto:
        uid_counts = defaultdict(int)
        for uid in batch.non_tensor_batch["uid"]:
            uid_counts[uid] += 1
        kept = [idx for idx, uid in enumerate(batch.non_tensor_batch["uid"]) if uid_counts[uid] > 1]
        if not kept:
            raise RuntimeError("No semi-online rollout group has at least two samples for GRPO.")
        if len(kept) < len(batch):
            print(f"Filtered {len(batch) - len(kept)} semi-online samples with singleton GRPO groups.")
        return batch[kept]

    def _trim_to_world_size_by_grpo_group(self, batch: DataProto) -> DataProto:
        world_size = self.actor_rollout_ref_wg.world_size
        remainder = len(batch) % world_size
        if remainder == 0:
            return batch

        uid_to_indices = defaultdict(list)
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            uid_to_indices[uid].append(idx)

        dropped_uids = set()
        dropped_count = 0
        for uid, indices in sorted(uid_to_indices.items(), key=lambda item: len(item[1])):
            dropped_uids.add(uid)
            dropped_count += len(indices)
            if (len(batch) - dropped_count) % world_size == 0:
                break

        kept = [idx for idx, uid in enumerate(batch.non_tensor_batch["uid"]) if uid not in dropped_uids]
        if not kept:
            raise RuntimeError("No semi-online samples remain after trimming batch to world size.")
        print(f"Trimmed {len(batch) - len(kept)} semi-online samples to match world size {world_size}.")
        return batch[kept]

    def _make_semi_online_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        print("Start generating semi-online trajectory batch...")
        rollout_started = time.perf_counter()
        self._semi_online_rollout_records = []
        try:
            batch_dict = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_dataloader)
            batch_dict = next(self.data_iterator)

        source_batch = DataProto.from_single_dict(batch_dict)
        if "trajectory_steps" not in source_batch.non_tensor_batch:
            raise RuntimeError("algorithm.semi_online=true requires trajectory-level data with `trajectory_steps`.")

        states = []
        selection_groups: dict[str, dict[str, Any]] = {}
        rollout_n = self.config.worker.rollout.n
        image_limit = self.config.algorithm.semi_online_image_limit
        generation_micro_batch_size = self.config.algorithm.semi_online_generation_micro_batch_size
        max_rollouts_per_task = self.config.algorithm.semi_online_max_rollouts_per_task
        diversity_refill_batch_size = self.config.algorithm.semi_online_diversity_refill_batch_size
        if image_limit < 1:
            raise ValueError("algorithm.semi_online_image_limit must be at least 1")
        if generation_micro_batch_size < 0:
            raise ValueError("algorithm.semi_online_generation_micro_batch_size must be non-negative")
        if max_rollouts_per_task < rollout_n:
            raise ValueError("algorithm.semi_online_max_rollouts_per_task must be at least worker.rollout.n")
        if diversity_refill_batch_size < 1:
            raise ValueError("algorithm.semi_online_diversity_refill_batch_size must be positive")
        self._progress(
            "ROLLOUT",
            "START",
            tasks=len(source_batch),
            rollout_n=rollout_n,
            max_candidates=max_rollouts_per_task,
            diversity_refill_batch=diversity_refill_batch_size,
            generation_micro_batch=generation_micro_batch_size,
        )
        for row_idx in range(len(source_batch)):
            task_id = str(source_batch.non_tensor_batch.get("task_id", np.array([row_idx], dtype=object))[row_idx])
            goal = str(source_batch.non_tensor_batch["goal"][row_idx])
            steps = list(source_batch.non_tensor_batch["trajectory_steps"][row_idx])
            group_key = f"{row_idx}:{task_id}"
            group = {
                "task_id": task_id,
                "goal": goal,
                "steps": steps,
                "group_key": group_key,
                "candidates": [],
                "selected": [],
                "candidate_count": 0,
            }
            selection_groups[group_key] = group
            for rollout_id in range(rollout_n):
                state = {
                    "task_id": task_id,
                    "selection_group_key": group_key,
                    "traj_uid": str(uuid.uuid4()),
                    "rollout_id": rollout_id,
                    "goal": goal,
                    "steps": steps,
                    "step_pos": 0,
                    "history": [],
                    "patch_count": 0,
                    "finished": False,
                    "events": [],
                    "progress_logged": False,
                    "selected_for_update": True,
                    "selection_reason": "initial_rollout_retained",
                }
                states.append(state)
                group["candidates"].append(state)
                group["selected"].append(state)
                group["candidate_count"] += 1

        meta_info = {
            "min_pixels": self.config.data.min_pixels,
            "max_pixels": self.config.data.max_pixels,
            "video_fps": self.config.data.video_fps,
            "n": 1,
        }
        step_batches = []

        def group_raw_total_advantages(group_states: list[dict[str, Any]]) -> dict[str, float]:
            from examples.ui_s1.advantage_ui_s1 import compute_ui_s1_advantages

            trajectory_ids = {state["traj_uid"] for state in group_states}
            candidate_batch = DataProto.concat(step_batches)
            rows = [
                index
                for index, traj_uid in enumerate(candidate_batch.non_tensor_batch["traj_uid"])
                if traj_uid in trajectory_ids
            ]
            candidate_batch = candidate_batch[rows]
            result = compute_ui_s1_advantages(
                token_level_rewards=candidate_batch.batch["token_level_scores"],
                response_mask=candidate_batch.batch["response_mask"],
                task_ids=candidate_batch.non_tensor_batch["task_id"],
                trajectory_ids=candidate_batch.non_tensor_batch["traj_uid"],
                step_ids=candidate_batch.non_tensor_batch["step_id"],
                extract_matches=candidate_batch.non_tensor_batch["extract_match"],
                gamma=self.config.algorithm.semi_online_gamma,
                step_advantage_weight=self.config.algorithm.semi_online_step_advantage_weight,
                episode_advantage_weight=self.config.algorithm.semi_online_episode_advantage_weight,
                normalize_by_std=False,
            )

            totals = defaultdict(float)
            counts = defaultdict(int)
            response_mask = candidate_batch.batch["response_mask"].bool()
            for row_idx, traj_uid in enumerate(candidate_batch.non_tensor_batch["traj_uid"]):
                token_count = int(response_mask[row_idx].sum().item())
                if token_count == 0:
                    continue
                step_advantage = float(
                    result.advantages[row_idx].masked_select(response_mask[row_idx]).mean().item()
                )
                totals[str(traj_uid)] += step_advantage
                counts[str(traj_uid)] += 1
            return {traj_uid: totals[traj_uid] / counts[traj_uid] for traj_uid in totals if counts[traj_uid] > 0}

        wave_index = 0
        while True:
            active_indices = [
                idx for idx, state in enumerate(states) if not state["finished"] and state["step_pos"] < len(state["steps"])
            ]
            if not active_indices:
                added_candidates = False
                refill_counts = []
                evaluated_candidate_counts = []
                for group in selection_groups.values():
                    candidate_states = group["candidates"]
                    evaluated_candidate_counts.append(group["candidate_count"])
                    pool_scores = group_raw_total_advantages(candidate_states)
                    selected_states, diversity_std = _select_most_diverse_rollout_subset(
                        candidate_states, pool_scores, rollout_n
                    )
                    selected_ids = {state["traj_uid"] for state in selected_states}
                    for state in candidate_states:
                        state["selection_advantage"] = pool_scores[state["traj_uid"]]
                        state["selected_for_update"] = state["traj_uid"] in selected_ids
                        if state["selected_for_update"]:
                            state["selection_reason"] = (
                                "initial_rollout_retained"
                                if len(candidate_states) == rollout_n
                                else "selected_max_diversity_subset"
                            )
                        else:
                            state["selection_reason"] = "not_selected_max_diversity_subset"

                    group["selected"] = selected_states
                    group["selection_scores"] = {
                        state["traj_uid"]: pool_scores[state["traj_uid"]] for state in selected_states
                    }
                    group["diversity_std"] = diversity_std
                    group["diversity_threshold_met"] = (
                        self.config.algorithm.semi_online_advantage_std_threshold <= 0.0
                        or diversity_std > self.config.algorithm.semi_online_advantage_std_threshold
                    )
                    if group["diversity_threshold_met"] or group["candidate_count"] >= max_rollouts_per_task:
                        refill_counts.append(0)
                        continue

                    refill_count = min(
                        diversity_refill_batch_size, max_rollouts_per_task - group["candidate_count"]
                    )
                    refill_counts.append(refill_count)
                    for _ in range(refill_count):
                        rollout_id = group["candidate_count"]
                        candidate = {
                            "task_id": group["task_id"],
                            "selection_group_key": group["group_key"],
                            "traj_uid": str(uuid.uuid4()),
                            "rollout_id": rollout_id,
                            "goal": group["goal"],
                            "steps": group["steps"],
                            "step_pos": 0,
                            "history": [],
                            "patch_count": 0,
                            "finished": False,
                            "events": [],
                            "progress_logged": False,
                            "selected_for_update": False,
                            "selection_reason": "pending_diversity_candidate",
                        }
                        states.append(candidate)
                        group["candidates"].append(candidate)
                        group["candidate_count"] += 1
                        added_candidates = True

                # These arrays intentionally share selection-group insertion
                # order: task_ids[i] -> candidate_counts[i] -> diversity_std[i].
                selection_group_values = list(selection_groups.values())
                task_ids = [group["task_id"] for group in selection_group_values]
                next_candidate_counts = [group["candidate_count"] for group in selection_group_values]
                diversity_stds = [round(float(group["diversity_std"]), 4) for group in selection_group_values]
                self._progress(
                    "DIVERSITY",
                    "RETRY" if added_candidates else "READY",
                    task_ids=task_ids,
                    candidate_counts=evaluated_candidate_counts,
                    diversity_std=diversity_stds,
                    refill_counts=refill_counts,
                    next_candidate_counts=next_candidate_counts,
                    threshold=self.config.algorithm.semi_online_advantage_std_threshold,
                    threshold_met=sum(group["diversity_threshold_met"] for group in selection_group_values),
                )
                if added_candidates:
                    continue
                break

            all_examples = []
            for state_idx in active_indices:
                state = states[state_idx]
                step = state["steps"][state["step_pos"]]
                step_id = step.get("step_id", state["step_pos"])
                answer = json.dumps(_compact_ui_action(step["action"]), ensure_ascii=False, separators=(",", ":"))
                image_start = max(0, state["step_pos"] - image_limit + 1)
                images = [item["image"] for item in state["steps"][image_start : state["step_pos"] + 1]]
                all_examples.append(
                    {
                        "prompt": _build_semi_online_prompt(
                            state["goal"], state["history"], image_count=len(images), format_prompt=self._semi_online_format_prompt
                        ),
                        "answer": answer,
                        "images": images,
                        "uid": f"{state['task_id']}:{step_id}",
                        "traj_uid": state["traj_uid"],
                        "step_id": step_id,
                        "task_id": state["task_id"],
                        "rollout_id": state["rollout_id"],
                        "state_idx": state_idx,
                    }
                )

            chunk_size = generation_micro_batch_size or len(all_examples)
            wave_index += 1
            wave_started = time.perf_counter()
            wave_rewards: dict[str, list[float]] = defaultdict(list)
            self._progress(
                "ROLLOUT_WAVE",
                "START",
                wave=wave_index,
                active_rollout_steps=len(all_examples),
                active_tasks=len({example["task_id"] for example in all_examples}),
                vllm_calls=(len(all_examples) + chunk_size - 1) // chunk_size,
            )
            for chunk_start in range(0, len(all_examples), chunk_size):
                examples = all_examples[chunk_start : chunk_start + chunk_size]
                step_batch = self._encode_semi_online_examples(examples, meta_info=meta_info)
                gen_batch = step_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
                )
                gen_batch.meta_info["n"] = 1
                gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
                gen_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
                gen_output = unpad_dataproto(gen_output, pad_size=pad_size)
                step_batch = step_batch.union(gen_output)

                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(step_batch))
                step_batch.batch["token_level_scores"] = reward_tensor
                for key, values in reward_metrics.items():
                    step_batch.non_tensor_batch[key] = np.array(values, dtype=object)
                    for value in values:
                        try:
                            wave_rewards[key].append(float(value))
                        except (TypeError, ValueError):
                            pass

                extract_match = []
                response_lengths = torch.sum(step_batch.batch["response_mask"], dim=-1)
                for row_idx, example in enumerate(examples):
                    state = states[example["state_idx"]]
                    gt_action = example["answer"]
                    response_len = int(response_lengths[row_idx].item())
                    response_ids = step_batch.batch["responses"][row_idx][:response_len]
                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    coordinate_transform = step_batch.non_tensor_batch["coordinate_transform"][row_idx]
                    from examples.ui_s1.reward_ui_s1_step import extract_action

                    extracted_action = extract_action(response_text, coordinate_transform)
                    accuracy = reward_metrics.get("accuracy", [0.0] * len(examples))[row_idx]
                    matched = float(accuracy) >= 1.0
                    extract_match.append(matched)
                    patched = False
                    termination_reason = None

                    if matched:
                        thinking = _thinking_for_history(response_text)
                        if thinking is not None:
                            state["history"].append(thinking)
                        state["step_pos"] += 1
                    else:
                        patch_threshold = self.config.algorithm.patch_threshold
                        can_patch = patch_threshold == -1 or state["patch_count"] < patch_threshold
                        if can_patch:
                            thinking = _thinking_for_history(response_text)
                            if thinking is not None:
                                state["history"].append(thinking)
                            state["patch_count"] += 1
                            state["step_pos"] += 1
                            patched = True
                        else:
                            state["finished"] = True
                            termination_reason = "patch_threshold_exhausted"

                    state["events"].append(
                        {
                            "step_id": example["step_id"],
                            "images": example["images"],
                            "expert_action": json.loads(gt_action),
                            "model_response": response_text,
                            "extracted_action": extracted_action,
                            "coordinate_transform": coordinate_transform,
                            "image_coordinate_transforms": step_batch.non_tensor_batch["image_coordinate_transforms"][row_idx],
                            "reward": {
                                key: float(values[row_idx])
                                for key, values in reward_metrics.items()
                                if row_idx < len(values)
                            },
                            "action_match": matched,
                            "patch_applied": patched,
                            "patch_count_after_step": state["patch_count"],
                            "termination_reason": termination_reason,
                        }
                    )

                step_batch.non_tensor_batch["extract_match"] = np.array(extract_match, dtype=object)
                step_batches.append(step_batch)

                completed_states = []
                for example in examples:
                    state = states[example["state_idx"]]
                    rollout_complete = state["finished"] or state["step_pos"] >= len(state["steps"])
                    if rollout_complete and not state["progress_logged"]:
                        state["progress_logged"] = True
                        completed_states.append(state)
                self._write_semi_online_rollout_progress(completed_states)
            reward_summary = {f"{key}_mean": float(np.mean(values)) for key, values in wave_rewards.items() if values}
            self._progress("REWARD", "SUMMARY", wave=wave_index, **reward_summary)
            self._progress(
                "ROLLOUT_WAVE",
                "END",
                wave=wave_index,
                elapsed_s=time.perf_counter() - wave_started,
                active_rollout_steps=len(all_examples),
            )

        if not step_batches:
            raise RuntimeError("Semi-online rollout produced no training samples.")

        batch = DataProto.concat(step_batches)
        selected_trajectory_ids = {
            state["traj_uid"] for group in selection_groups.values() for state in group["selected"]
        }
        selected_rows = [
            index for index, traj_uid in enumerate(batch.non_tensor_batch["traj_uid"]) if traj_uid in selected_trajectory_ids
        ]
        batch = batch[selected_rows]
        ui_s1_metrics = _attach_ui_s1_advantages(batch, self.config)
        ui_s1_metrics.update(
            {
                "uis1/diversity_group_std_mean": float(
                    np.mean([group["diversity_std"] for group in selection_groups.values()])
                ),
                "uis1/diversity_groups_threshold_met": float(
                    sum(group["diversity_threshold_met"] for group in selection_groups.values())
                ),
                "uis1/diversity_groups_max_pool_reached": float(
                    sum(
                        group["candidate_count"] == max_rollouts_per_task and not group["diversity_threshold_met"]
                        for group in selection_groups.values()
                    )
                ),
            }
        )
        batch.meta_info["ui_s1_metrics"] = ui_s1_metrics
        selected_states = [state for group in selection_groups.values() for state in group["selected"]]
        self._semi_online_rollout_records = [
            {
                "task_id": state["task_id"],
                "trajectory_id": state["traj_uid"],
                "rollout_id": state["rollout_id"],
                "goal": state["goal"],
                "expert_step_count": len(state["steps"]),
                "generated_step_count": len(state["events"]),
                "patch_count": state["patch_count"],
                "patch_threshold": self.config.algorithm.patch_threshold,
                "episode_return": sum(float(event["reward"].get("overall", 0.0)) for event in state["events"]),
                "selection_advantage": state.get("selection_advantage"),
                "candidate_pool_size": selection_groups[state["selection_group_key"]]["candidate_count"],
                "diversity_std": selection_groups[state["selection_group_key"]]["diversity_std"],
                "diversity_threshold_met": selection_groups[state["selection_group_key"]]["diversity_threshold_met"],
                "selected_for_update": state["selected_for_update"],
                "selection_reason": state["selection_reason"],
                "reached_trajectory_end": state["step_pos"] >= len(state["steps"]),
                "termination_reason": "expert_trajectory_exhausted"
                if state["step_pos"] >= len(state["steps"])
                else "patch_threshold_exhausted",
                "events": state["events"],
            }
            for state in selected_states
        ]

        # Episode advantages remain meaningful for singleton later steps, so do
        # not discard them merely because a same-step GRPO group became smaller.
        batch = self._trim_to_world_size_by_grpo_group(batch)
        selected_reward_metrics = defaultdict(list)
        for state in selected_states:
            for event in state["events"]:
                for key, value in event["reward"].items():
                    selected_reward_metrics[key].append(value)
        metrics.update({f"reward/{key}": value for key, value in reduce_metrics(selected_reward_metrics).items()})
        # Task -> selected rollout.  Each entry is generated action steps over
        # the complete expert trajectory length, e.g. 2/6 means generation
        # stopped after two of six expert steps.
        selected_rollout_progress = "[" + ",".join(
            "["
            + ",".join(
                f"{len(state['events'])}/{len(state['steps'])}"
                for state in sorted(group["selected"], key=lambda state: state["rollout_id"])
            )
            + "]"
            for group in selection_groups.values()
        ) + "]"
        self._progress(
            "ROLLOUT",
            "END",
            elapsed_s=time.perf_counter() - rollout_started,
            selected_rollouts=len(selected_states),
            candidate_counts=[group["candidate_count"] for group in selection_groups.values()],
            selected_rollout_progress=selected_rollout_progress,
            overall_reward_mean=ui_s1_metrics["uis1/step_reward_mean"],
            advantage_std=ui_s1_metrics["uis1/advantage_std"],
        )
        return batch

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        self._progress("TRAINING_LOOP", "START", planned_steps=self.training_steps)
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)
        # Resuming begins a fresh wall-clock interval. Checkpoint I/O and model
        # restoration should not make the first resumed update save immediately.
        self._last_checkpoint_step = -1
        self._last_checkpoint_monotonic = time.monotonic()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        if self.config.algorithm.semi_online and not self.config.algorithm.use_kl_loss:
            raise ValueError("UI-S1 semi-online RL requires algorithm.use_kl_loss=true so KL remains separate from returns.")
        while self.global_step < self.training_steps:
            self.global_step += 1
            self._progress(
                "STEP",
                "START",
                tasks_per_update=self.config.data.rollout_batch_size,
                rollout_n=self.config.worker.rollout.n,
                generation_micro_batch=self.config.algorithm.semi_online_generation_micro_batch_size,
            )

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    sync_started = time.perf_counter()
                    self._progress("ROLLOUT_ENGINE_SYNC", "START", direction="actor_to_vllm")
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    self._progress("ROLLOUT_ENGINE_SYNC", "END", elapsed_s=time.perf_counter() - sync_started)
                    if self.config.algorithm.semi_online:
                        candidate_metrics: dict[str, Any] = {}
                        batch = self._make_semi_online_batch_data(metrics=candidate_metrics)
                        ui_s1_metrics = batch.meta_info.get("ui_s1_metrics", {})
                        advantage_std = float(ui_s1_metrics.get("uis1/advantage_std", 0.0))
                        self._write_semi_online_rollout_logs(
                            self._semi_online_rollout_records, 1, advantage_std
                        )
                        metrics.update(candidate_metrics)
                        metrics.update(ui_s1_metrics)
                    else:
                        batch = self._make_batch_data(metrics=metrics)
                    release_started = time.perf_counter()
                    self._progress("ROLLOUT_ENGINE_RELEASE", "START")
                    self.actor_rollout_ref_wg.release_rollout_engine()
                    self._progress("ROLLOUT_ENGINE_RELEASE", "END", elapsed_s=time.perf_counter() - release_started)

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                self._progress("OLD_LOG_PROBS", "START")
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)
                self._progress("OLD_LOG_PROBS", "END", elapsed_s=timing_raw["old"])

                # compute ref_log_probs
                if self.use_reference_policy:
                    self._progress("REF_LOG_PROBS", "START")
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)
                    self._progress("REF_LOG_PROBS", "END", elapsed_s=timing_raw["ref"])

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # Semi-online batches already carry trajectory-level UI-S1
                    # advantages. All other algorithms use EasyR1's native path.
                    if not self.config.algorithm.semi_online:
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                self._progress(
                    "ADVANTAGE",
                    "END",
                    elapsed_s=timing_raw["adv"],
                    advantage_std=batch.meta_info.get("ui_s1_metrics", {}).get("uis1/advantage_std"),
                )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    self._progress("ACTOR_UPDATE", "START")
                    with timer("update_actor", timing_raw):
                        actor_batch = batch
                        if self.config.algorithm.semi_online:
                            # UI-S1's reference trainer pads variable-length
                            # multi-turn step batches before the PPO update.
                            # The final rollout selection and advantages have
                            # already been computed on the unpadded batch.
                            actor_update_divisor = (
                                self.config.worker.actor.global_batch_size * self.config.worker.rollout.n
                            )
                            actor_batch, actor_padding_size = pad_dataproto_to_divisor(
                                batch, actor_update_divisor
                            )
                            metrics["uis1/actor_update_padding"] = float(actor_padding_size)
                        actor_output = self.actor_rollout_ref_wg.update_actor(actor_batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)
                    if self.config.algorithm.semi_online:
                        self._write_semi_online_update_result(self._semi_online_rollout_records)
                    self._progress(
                        "ACTOR_UPDATE",
                        "END",
                        elapsed_s=timing_raw["update_actor"],
                        padding=metrics.get("uis1/actor_update_padding"),
                        pg_loss=actor_metrics.get("actor/pg_loss"),
                        kl_loss=actor_metrics.get("actor/kl_loss"),
                        grad_norm=actor_metrics.get("actor/grad_norm"),
                    )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self._should_save_checkpoint():
                    self._progress("CHECKPOINT_SAVE", "START")
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
                    self._record_checkpoint_saved()
                    self._progress("CHECKPOINT_SAVE", "END", elapsed_s=timing_raw["save_checkpoint"])

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            self._progress(
                "STEP",
                "END",
                elapsed_s=timing_raw["step"],
                generation_s=timing_raw.get("gen"),
                old_log_probs_s=timing_raw.get("old"),
                ref_log_probs_s=timing_raw.get("ref"),
                actor_update_s=timing_raw.get("update_actor"),
                throughput=metrics.get("perf/throughput"),
            )
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if not self._should_save_checkpoint():
            final_checkpoint_started = time.perf_counter()
            self._progress("CHECKPOINT_SAVE", "START", final=True)
            self._save_checkpoint()
            self._record_checkpoint_saved()
            self._progress("CHECKPOINT_SAVE", "END", final=True, elapsed_s=time.perf_counter() - final_checkpoint_started)
        self._progress("TRAINING_LOOP", "END", completed_steps=self.global_step)
