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
PPO config
"""

import math
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class PatchImitationConfig:
    enabled: bool = False
    """enable the auxiliary expert-action imitation objective"""
    lambda_initial: float = 0.0
    """imitation objective weight at the first trainer update; must be positive when enabled"""
    lambda_decay: float = 1.0
    """multiplicative decay applied once per completed trainer update"""
    lambda_min: float = 0.0
    """lower bound for the imitation objective weight"""
    lambda_cutoff_step: int = 0
    """last one-based trainer update that uses imitation; 0 disables the cutoff"""
    target_mode: str = "action_only"
    """tokens supervised by imitation; currently only `action_only` is implemented"""
    history_mode: str = "keep_model_thinking"
    """rollout history policy; currently only `keep_model_thinking` is implemented"""

    def post_init(self):
        lambda_values = {
            "lambda_initial": self.lambda_initial,
            "lambda_decay": self.lambda_decay,
            "lambda_min": self.lambda_min,
        }
        for name, value in lambda_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"algorithm.patch_imitation.{name} must be a finite number.")

        if self.lambda_initial < 0:
            raise ValueError("algorithm.patch_imitation.lambda_initial must be non-negative.")
        if self.enabled and self.lambda_initial <= 0:
            raise ValueError(
                "algorithm.patch_imitation.lambda_initial must be positive when patch imitation is enabled."
            )
        if not 0 < self.lambda_decay <= 1:
            raise ValueError("algorithm.patch_imitation.lambda_decay must be in the interval (0, 1].")
        if self.lambda_min < 0:
            raise ValueError("algorithm.patch_imitation.lambda_min must be non-negative.")
        if self.lambda_min > self.lambda_initial:
            raise ValueError(
                "algorithm.patch_imitation.lambda_min must not exceed algorithm.patch_imitation.lambda_initial."
            )
        if (
            isinstance(self.lambda_cutoff_step, bool)
            or not isinstance(self.lambda_cutoff_step, int)
            or self.lambda_cutoff_step < 0
        ):
            raise ValueError(
                "algorithm.patch_imitation.lambda_cutoff_step must be a non-negative integer."
            )
        if self.target_mode != "action_only":
            raise ValueError(
                "algorithm.patch_imitation.target_mode currently supports only `action_only`; "
                f"got {self.target_mode!r}."
            )
        if self.history_mode != "keep_model_thinking":
            raise ValueError(
                "algorithm.patch_imitation.history_mode currently supports only `keep_model_thinking`; "
                f"got {self.history_mode!r}."
            )


def compute_patch_imitation_lambda(config: PatchImitationConfig, global_step: int) -> float:
    """Return the effective patch-imitation weight for a one-based trainer update."""
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 1:
        raise ValueError("global_step must be a positive integer.")
    if not config.enabled:
        return 0.0
    if config.lambda_cutoff_step and global_step > config.lambda_cutoff_step:
        return 0.0

    completed_updates_before_step = global_step - 1
    return max(config.lambda_min, config.lambda_initial * config.lambda_decay**completed_updates_before_step)


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    semi_online: bool = False
    """enable UI-S1-style semi-online trajectory rollout"""
    patch_threshold: int = 0
    """number of expert-action patches allowed after action mismatch; -1 means unlimited"""
    semi_online_gamma: float = 0.5
    """discount factor for UI-S1 natural-segment step returns"""
    semi_online_step_advantage_weight: float = 1.0
    """weight of UI-S1 step-level advantage"""
    semi_online_episode_advantage_weight: float = 1.0
    """weight of UI-S1 episode-level advantage"""
    semi_online_normalize_by_std: bool = True
    """normalize UI-S1 advantages by rollout-group standard deviation"""
    semi_online_advantage_std_threshold: float = 0.3
    """minimum UI-S1 advantage standard deviation before accepting a rollout batch; 0 disables it"""
    semi_online_image_limit: int = 1
    """number of most recent trajectory screenshots retained for each semi-online prompt, including current"""
    semi_online_generation_micro_batch_size: int = 0
    """maximum active trajectories generated together; 0 generates all active trajectories together"""
    semi_online_max_rollouts_per_task: int = 20
    """maximum candidates sampled per task before selecting rollout.n trajectories for one UI-S1 update"""
    semi_online_diversity_refill_batch_size: int = 4
    """candidates added at once for each task that fails the UI-S1 diversity threshold"""
    patch_imitation: PatchImitationConfig = field(default_factory=PatchImitationConfig)
    """auxiliary expert-action imitation configuration for semi-online patches"""

    def post_init(self):
        if self.patch_imitation.enabled and not self.semi_online:
            raise ValueError("algorithm.patch_imitation.enabled=true requires algorithm.semi_online=true.")


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """periodic validation frequency, -1 disables periodic validation"""
    val_before_train: bool = True
    """validate before training"""
    val_after_train: bool = True
    """validate once after training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_every_n_epochs: int = 0
    """save at each N completed epochs; 0 disables epoch-based checkpointing"""
    save_interval_seconds: float = 0.0
    """save after a completed update when this many seconds have elapsed; 0 disables time-based saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    rollout_log_path: Optional[str] = None
    """JSONL path for semi-online rollout traces; defaults under save_checkpoint_path"""
    progress_log_path: Optional[str] = None
    """human-readable, line-buffered training progress log; defaults under save_checkpoint_path"""
    progress_validation_interval: int = 25
    """emit one concise validation-progress record after this many validation batches"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    warm_start_checkpoint_path: Optional[str] = None
    """Source full checkpoint for LoRA-only warm start; never restores optimizer state."""
    warm_start_dataloader: str = "inherit"
    """Whether a warm start restores its source dataloader cursor: ``inherit`` or ``reset``."""
    warm_start_global_step: int = 0
    """Source global step used as the visible log/checkpoint offset for a warm start."""
    append_existing_history: bool = False
    """Append structured logs prepared before launch instead of truncating them."""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_interval_seconds < 0:
            raise ValueError("trainer.save_interval_seconds must be non-negative.")
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        if self.rollout_log_path is None:
            self.rollout_log_path = os.path.join(self.save_checkpoint_path, "semi_online_rollouts.jsonl")
        self.rollout_log_path = os.path.abspath(self.rollout_log_path)
        if self.progress_log_path is None:
            self.progress_log_path = os.path.join(self.save_checkpoint_path, "training_progress.log")
        self.progress_log_path = os.path.abspath(self.progress_log_path)
        requested_checkpoint_path = self.load_checkpoint_path
        self.load_checkpoint_path = get_abs_path(requested_checkpoint_path, prompt="Model checkpoint")
        if requested_checkpoint_path is not None and self.load_checkpoint_path is None:
            raise ValueError(f"trainer.load_checkpoint_path does not exist: {requested_checkpoint_path}")
        requested_warm_start_path = self.warm_start_checkpoint_path
        self.warm_start_checkpoint_path = get_abs_path(requested_warm_start_path, prompt="Warm-start checkpoint")
        if requested_warm_start_path is not None and self.warm_start_checkpoint_path is None:
            raise ValueError(f"trainer.warm_start_checkpoint_path does not exist: {requested_warm_start_path}")
        if self.load_checkpoint_path is not None and self.warm_start_checkpoint_path is not None:
            raise ValueError("trainer.load_checkpoint_path and trainer.warm_start_checkpoint_path are mutually exclusive.")
        if self.warm_start_dataloader not in {"inherit", "reset"}:
            raise ValueError("trainer.warm_start_dataloader must be 'inherit' or 'reset'.")
        if isinstance(self.warm_start_global_step, bool) or not isinstance(self.warm_start_global_step, int):
            raise ValueError("trainer.warm_start_global_step must be an integer.")
        if self.warm_start_checkpoint_path is None and self.warm_start_global_step != 0:
            raise ValueError("trainer.warm_start_global_step requires trainer.warm_start_checkpoint_path.")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
