"""Action-only expert targets for UI-S1 patch imitation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Sequence

import torch

from examples.ui_s1.reward_ui_s1_step import (
    ACTION_ARGUMENTS,
    OPTIONAL_ACTION_ARGUMENTS,
    is_valid_action_schema,
    parse_action,
    transform_action_coordinates,
)


_THINKING_PREFIX_PATTERN = re.compile(
    r"(?P<prefix>\s*<thinking>(?P<thinking>.*?)</thinking>)",
    flags=re.DOTALL,
)


def extract_exact_thinking_prefix(response: str) -> str | None:
    """Return the model's leading thinking block without rewriting its text."""
    match = _THINKING_PREFIX_PATTERN.match(response or "")
    if match is None or not match.group("thinking").strip():
        return None
    return match.group("prefix")


def find_exact_thinking_prefix_ids(tokenizer: Any, response_ids: Sequence[int]) -> list[int] | None:
    """Locate the sampled token prefix ending at the model's ``</thinking>``.

    The incremental scan only accepts an exact prefix of the sampled ids.  If
    the closing tag shares a token with later output, the sample is skipped
    instead of silently replacing rollout tokens with a re-tokenized prefix.
    """
    response_ids = [int(token_id) for token_id in response_ids]
    full_response = tokenizer.decode(response_ids, skip_special_tokens=True)
    exact_prefix = extract_exact_thinking_prefix(full_response)
    if exact_prefix is None:
        return None

    encoded_prefix = list(tokenizer.encode(exact_prefix, add_special_tokens=False))
    estimated_end = len(encoded_prefix)
    nearby_ends = range(
        max(1, estimated_end - 2),
        min(len(response_ids), estimated_end + 2) + 1,
    )
    fallback_ends = range(1, len(response_ids) + 1)
    visited_ends = set()
    for end in (*nearby_ends, *fallback_ends):
        if end in visited_ends:
            continue
        visited_ends.add(end)
        decoded_prefix = tokenizer.decode(response_ids[:end], skip_special_tokens=True)
        match = _THINKING_PREFIX_PATTERN.match(decoded_prefix)
        if match is None or not match.group("thinking").strip():
            continue
        if not decoded_prefix[match.end() :].strip():
            return response_ids[:end]
        break

    return None


def _has_valid_coordinate_transform(coordinate_transform: Any) -> bool:
    if not isinstance(coordinate_transform, dict):
        return False
    try:
        dimensions = [
            float(coordinate_transform[key])
            for key in ("original_width", "original_height", "model_width", "model_height")
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) and value > 0.0 for value in dimensions)


def expert_action_in_model_coordinates(ground_truth: Any, coordinate_transform: Any) -> dict[str, Any] | None:
    """Convert one raw-image expert action into the exact current model grid."""
    action, valid = parse_action(ground_truth)
    if action is None or valid != 1.0:
        return None

    action_name = action.get("action")
    required_fields = ACTION_ARGUMENTS.get(action_name)
    if required_fields is None:
        return None
    allowed_fields = {"action", *required_fields, *OPTIONAL_ACTION_ARGUMENTS.get(action_name, ())}
    action = {key: value for key, value in action.items() if key in allowed_fields and value is not None}

    coordinate_fields = ("coordinate", "coordinate2")
    if any(key in action for key in coordinate_fields):
        # Coordinate actions must fail closed: using raw screenshot
        # coordinates as a target for Qwen's resized visual grid is wrong.
        if not _has_valid_coordinate_transform(coordinate_transform):
            return None
        action = transform_action_coordinates(action, coordinate_transform, inverse=False)
        for key in coordinate_fields:
            if key in action:
                action[key] = [round(float(value)) for value in action[key]]

    if not is_valid_action_schema(action):
        return None
    return action


def build_expert_tool_call(action: dict[str, Any]) -> str:
    """Serialize the canonical action-only target appended after model thinking."""
    payload = {"name": "mobile_use", "arguments": action}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</tool_call>"


def normalized_patch_sample_weights(
    group_keys: Sequence[Any], step_ids: Sequence[Any], valid_indices: Sequence[int]
) -> dict[int, float]:
    """Give every distinct task-step equal total expert weight.

    If ``M`` selected rollouts patch the same step, each receives ``1/M`` of
    that task-step's share.  This prevents rollout multiplicity from making an
    otherwise identical intervention stronger.
    """
    valid_indices = [int(index) for index in valid_indices]
    if not valid_indices:
        return {}

    row_group = {index: (str(group_keys[index]), str(step_ids[index])) for index in valid_indices}
    counts = Counter(row_group.values())
    distinct_group_count = len(counts)
    return {index: 1.0 / (distinct_group_count * counts[row_group[index]]) for index in valid_indices}


PATCH_IMITATION_BATCH_KEYS = (
    "patch_input_ids",
    "patch_attention_mask",
    "patch_position_ids",
    "patch_responses",
    "patch_token_weights",
)


def attach_patch_imitation_tensors(batch: Any, tokenizer: Any) -> dict[str, float]:
    """Attach aligned expert targets to the finally selected rollout rows.

    Prompts, visual inputs, and the model's sampled thinking prefix are reused
    from the rollout row.  Only the appended expert ``<tool_call>`` ids receive
    non-zero weights.  Rows without a usable patch remain valid zero-weight
    inputs so every distributed rank follows the same forward/backward path.
    """
    required_tensor_keys = {
        "prompts",
        "responses",
        "input_ids",
        "attention_mask",
        "response_mask",
        "position_ids",
    }
    missing_tensor_keys = required_tensor_keys - set(batch.batch.keys())
    if missing_tensor_keys:
        raise KeyError(f"Patch imitation batch is missing tensor fields: {sorted(missing_tensor_keys)}")
    required_non_tensor_keys = {"patch_applied", "ground_truth", "step_id", "coordinate_transform"}
    missing_non_tensor_keys = required_non_tensor_keys - set(batch.non_tensor_batch)
    if missing_non_tensor_keys:
        raise KeyError(f"Patch imitation batch is missing non-tensor fields: {sorted(missing_non_tensor_keys)}")

    attempted_indices = [
        index for index, patch_applied in enumerate(batch.non_tensor_batch["patch_applied"]) if bool(patch_applied)
    ]
    empty_metrics = {
        "patch_imitation/attempted_samples": float(len(attempted_indices)),
        "patch_imitation/valid_samples": 0.0,
        "patch_imitation/skipped_samples": float(len(attempted_indices)),
        "patch_imitation/skipped_missing_thinking": 0.0,
        "patch_imitation/skipped_invalid_expert_action": 0.0,
        "patch_imitation/skipped_overlong_target": 0.0,
        "patch_imitation/distinct_task_steps": 0.0,
        "patch_imitation/target_tokens": 0.0,
        "patch_imitation/sample_weight_sum": 0.0,
    }
    if not attempted_indices:
        return empty_metrics

    max_response_length = int(batch.batch["responses"].size(-1))
    target_rows: dict[int, tuple[list[int], list[int]]] = {}
    skipped_missing_thinking = 0
    skipped_invalid_expert_action = 0
    skipped_overlong_target = 0
    for index in attempted_indices:
        response_length = int(batch.batch["response_mask"][index].sum().item())
        sampled_response_ids = batch.batch["responses"][index, :response_length].tolist()
        thinking_ids = find_exact_thinking_prefix_ids(tokenizer, sampled_response_ids)
        if thinking_ids is None:
            skipped_missing_thinking += 1
            continue

        expert_action = expert_action_in_model_coordinates(
            batch.non_tensor_batch["ground_truth"][index],
            batch.non_tensor_batch["coordinate_transform"][index],
        )
        if expert_action is None:
            skipped_invalid_expert_action += 1
            continue
        tool_call_ids = list(tokenizer.encode(build_expert_tool_call(expert_action), add_special_tokens=False))
        if not tool_call_ids or len(thinking_ids) + len(tool_call_ids) > max_response_length:
            skipped_overlong_target += 1
            continue
        target_rows[index] = (thinking_ids, tool_call_ids)

    valid_indices = list(target_rows)
    if not valid_indices:
        empty_metrics["patch_imitation/skipped_missing_thinking"] = float(skipped_missing_thinking)
        empty_metrics["patch_imitation/skipped_invalid_expert_action"] = float(skipped_invalid_expert_action)
        empty_metrics["patch_imitation/skipped_overlong_target"] = float(skipped_overlong_target)
        return empty_metrics

    if "selection_group_key" in batch.non_tensor_batch:
        group_keys = batch.non_tensor_batch["selection_group_key"]
    else:
        group_keys = batch.non_tensor_batch["task_id"]
    sample_weights = normalized_patch_sample_weights(group_keys, batch.non_tensor_batch["step_id"], valid_indices)

    patch_responses = batch.batch["responses"].clone()
    patch_input_ids = batch.batch["input_ids"].clone()
    patch_attention_mask = batch.batch["attention_mask"].clone()
    patch_position_ids = batch.batch["position_ids"].clone()
    patch_token_weights = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
    prompt_length = int(batch.batch["prompts"].size(-1))
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Patch imitation requires a tokenizer pad_token_id or eos_token_id.")

    for index, (thinking_ids, tool_call_ids) in target_rows.items():
        target_ids = thinking_ids + tool_call_ids
        target_length = len(target_ids)
        target_tensor = torch.tensor(
            target_ids,
            dtype=patch_responses.dtype,
            device=patch_responses.device,
        )

        patch_responses[index].fill_(int(pad_token_id))
        patch_responses[index, :target_length] = target_tensor
        patch_input_ids[index, prompt_length:].fill_(int(pad_token_id))
        patch_input_ids[index, prompt_length : prompt_length + target_length] = target_tensor

        patch_attention_mask[index, prompt_length:] = 0
        patch_attention_mask[index, prompt_length : prompt_length + target_length] = 1

        prompt_position_ids = patch_position_ids[index, ..., :prompt_length]
        delta_position_ids = torch.arange(
            1,
            max_response_length + 1,
            dtype=prompt_position_ids.dtype,
            device=prompt_position_ids.device,
        )
        if prompt_position_ids.dim() == 2:
            delta_position_ids = delta_position_ids.unsqueeze(0).expand(prompt_position_ids.size(0), -1)
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_ids
        patch_position_ids[index, ..., prompt_length:] = response_position_ids

        action_start = len(thinking_ids)
        action_end = target_length
        per_token_weight = sample_weights[index] / len(tool_call_ids)
        patch_token_weights[index, action_start:action_end] = per_token_weight

    batch.batch["patch_input_ids"] = patch_input_ids
    batch.batch["patch_attention_mask"] = patch_attention_mask
    batch.batch["patch_position_ids"] = patch_position_ids
    batch.batch["patch_responses"] = patch_responses
    batch.batch["patch_token_weights"] = patch_token_weights

    distinct_groups = {
        (str(group_keys[index]), str(batch.non_tensor_batch["step_id"][index])) for index in valid_indices
    }
    return {
        "patch_imitation/attempted_samples": float(len(attempted_indices)),
        "patch_imitation/valid_samples": float(len(valid_indices)),
        "patch_imitation/skipped_samples": float(len(attempted_indices) - len(valid_indices)),
        "patch_imitation/skipped_missing_thinking": float(skipped_missing_thinking),
        "patch_imitation/skipped_invalid_expert_action": float(skipped_invalid_expert_action),
        "patch_imitation/skipped_overlong_target": float(skipped_overlong_target),
        "patch_imitation/distinct_task_steps": float(len(distinct_groups)),
        "patch_imitation/target_tokens": float(sum(len(tool_call_ids) for _, tool_call_ids in target_rows.values())),
        "patch_imitation/sample_weight_sum": float(sum(sample_weights.values())),
    }


def zero_patch_imitation_padding(batch: Any, pad_size: int) -> None:
    """Make rows copied solely for actor divisibility contribute no expert gradient."""
    if pad_size <= 0 or "patch_token_weights" not in batch.batch:
        return
    batch.batch["patch_token_weights"][-pad_size:] = 0
