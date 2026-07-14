"""UI-S1 trajectory-level advantage computation for EasyR1 semi-online RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass
class UIS1AdvantageResult:
    advantages: torch.Tensor
    returns: torch.Tensor
    step_rewards: torch.Tensor
    step_returns: torch.Tensor
    episode_returns: torch.Tensor
    episode_advantages: torch.Tensor
    step_advantages: torch.Tensor


def _key(value: Any) -> str:
    return str(value)


def _step_key(value: Any, position: int) -> tuple[int, float | str, int]:
    try:
        return (0, float(value), position)
    except (TypeError, ValueError):
        return (1, _key(value), position)


def _group_normalize(
    values: list[float], normalize_by_std: bool, epsilon: float
) -> list[float]:
    if len(values) <= 1:
        return [0.0] * len(values)

    value_tensor = torch.tensor(values, dtype=torch.float64)
    centered = value_tensor - value_tensor.mean()
    if normalize_by_std:
        std = value_tensor.std(unbiased=True)
        if std > epsilon:
            centered = centered / std
        else:
            centered = torch.zeros_like(centered)
    return centered.tolist()


def compute_ui_s1_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    task_ids: Sequence[Any],
    trajectory_ids: Sequence[Any],
    step_ids: Sequence[Any],
    extract_matches: Sequence[Any],
    gamma: float = 0.5,
    step_advantage_weight: float = 1.0,
    episode_advantage_weight: float = 1.0,
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
) -> UIS1AdvantageResult:
    """Compute the dual-level advantage from UI-S1 equations (9)-(12).

    The episode score is the sum of every step reward in a patched rollout.
    Step returns are discounted only within a natural segment: a mismatch ends
    propagation to earlier steps, while patched later steps form a new segment.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    batch_size = token_level_rewards.shape[0]
    if response_mask.shape[0] != batch_size:
        raise ValueError("response_mask and token_level_rewards must have the same batch size")
    for name, values in {
        "task_ids": task_ids,
        "trajectory_ids": trajectory_ids,
        "step_ids": step_ids,
        "extract_matches": extract_matches,
    }.items():
        if len(values) != batch_size:
            raise ValueError(f"{name} has length {len(values)}, expected {batch_size}")

    mask = response_mask.to(dtype=token_level_rewards.dtype)
    step_rewards_tensor = (token_level_rewards * mask).sum(dim=-1)
    step_rewards = [float(value) for value in step_rewards_tensor.detach().cpu()]

    trajectory_rows: dict[tuple[str, str], list[int]] = {}
    task_trajectories: dict[str, list[tuple[str, str]]] = {}
    for row_idx in range(batch_size):
        task_key = _key(task_ids[row_idx])
        trajectory_key = (task_key, _key(trajectory_ids[row_idx]))
        if trajectory_key not in trajectory_rows:
            trajectory_rows[trajectory_key] = []
            task_trajectories.setdefault(task_key, []).append(trajectory_key)
        trajectory_rows[trajectory_key].append(row_idx)

    step_returns = [0.0] * batch_size
    episode_return_by_trajectory: dict[tuple[str, str], float] = {}
    for trajectory_key, row_indices in trajectory_rows.items():
        ordered_rows = sorted(row_indices, key=lambda row_idx: _step_key(step_ids[row_idx], row_idx))
        episode_return_by_trajectory[trajectory_key] = sum(step_rewards[row_idx] for row_idx in ordered_rows)

        running_return = 0.0
        for row_idx in reversed(ordered_rows):
            reward = step_rewards[row_idx]
            if bool(extract_matches[row_idx]):
                running_return = reward + gamma * running_return
                step_returns[row_idx] = running_return
            else:
                # Do not propagate patched future rewards through this mismatch.
                step_returns[row_idx] = reward
                running_return = 0.0

    episode_advantages = [0.0] * batch_size
    for task_key, trajectory_keys in task_trajectories.items():
        del task_key
        returns = [episode_return_by_trajectory[key] for key in trajectory_keys]
        normalized = _group_normalize(returns, normalize_by_std, epsilon)
        for trajectory_key, advantage in zip(trajectory_keys, normalized):
            for row_idx in trajectory_rows[trajectory_key]:
                episode_advantages[row_idx] = advantage

    step_advantages = [0.0] * batch_size
    step_groups: dict[tuple[str, str], list[int]] = {}
    for row_idx in range(batch_size):
        group_key = (_key(task_ids[row_idx]), _key(step_ids[row_idx]))
        step_groups.setdefault(group_key, []).append(row_idx)
    for row_indices in step_groups.values():
        normalized = _group_normalize([step_returns[row_idx] for row_idx in row_indices], normalize_by_std, epsilon)
        for row_idx, advantage in zip(row_indices, normalized):
            step_advantages[row_idx] = advantage

    device = token_level_rewards.device
    dtype = token_level_rewards.dtype
    episode_return_rows = torch.tensor(
        [episode_return_by_trajectory[(_key(task_ids[row_idx]), _key(trajectory_ids[row_idx]))] for row_idx in range(batch_size)],
        device=device,
        dtype=dtype,
    )
    step_return_rows = torch.tensor(step_returns, device=device, dtype=dtype)
    episode_advantage_rows = torch.tensor(episode_advantages, device=device, dtype=dtype)
    step_advantage_rows = torch.tensor(step_advantages, device=device, dtype=dtype)
    combined = episode_advantage_weight * episode_advantage_rows + step_advantage_weight * step_advantage_rows
    token_advantages = combined.unsqueeze(-1) * mask

    return UIS1AdvantageResult(
        advantages=token_advantages,
        returns=token_advantages.clone(),
        step_rewards=step_rewards_tensor,
        step_returns=step_return_rows,
        episode_returns=episode_return_rows,
        episode_advantages=episode_advantage_rows,
        step_advantages=step_advantage_rows,
    )
