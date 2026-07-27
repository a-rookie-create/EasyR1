import copy
from types import SimpleNamespace

import torch

from examples.ui_s1.patch_imitation import (
    PATCH_IMITATION_BATCH_KEYS,
    attach_patch_imitation_tensors,
    build_expert_tool_call,
    expert_action_in_model_coordinates,
    normalized_patch_sample_weights,
    zero_patch_imitation_padding,
)


class CharacterTokenizer:
    """A reversible tokenizer that makes token-boundary assertions explicit."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) + 2 for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        decoded = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            decoded.append(chr(token_id - 2))
        return "".join(decoded)


def _make_rollout_batch(
    response_texts,
    ground_truths,
    *,
    max_response_length=512,
    patch_applied=None,
    group_keys=None,
    step_ids=None,
    coordinate_transforms=None,
):
    tokenizer = CharacterTokenizer()
    batch_size = len(response_texts)
    prompt_length = 6
    prompts = torch.arange(
        10,
        10 + batch_size * prompt_length,
        dtype=torch.long,
    ).reshape(batch_size, prompt_length)
    prompt_attention_mask = torch.ones((batch_size, prompt_length), dtype=torch.long)
    prompt_attention_mask[:, 0] = 0

    responses = torch.full(
        (batch_size, max_response_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    response_mask = torch.zeros_like(responses)
    for row_index, response_text in enumerate(response_texts):
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        assert len(response_ids) <= max_response_length
        responses[row_index, : len(response_ids)] = torch.tensor(response_ids)
        response_mask[row_index, : len(response_ids)] = 1

    input_ids = torch.cat((prompts, responses), dim=-1)
    attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=-1)

    prompt_position_ids = torch.arange(prompt_length, dtype=torch.long).view(1, 1, -1)
    prompt_position_ids = prompt_position_ids.expand(batch_size, 4, -1).clone()
    prompt_position_ids += torch.arange(4, dtype=torch.long).view(1, 4, 1) * 100
    response_delta = torch.arange(1, max_response_length + 1, dtype=torch.long).view(1, 1, -1)
    response_position_ids = prompt_position_ids[..., -1:] + response_delta
    position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)

    if patch_applied is None:
        patch_applied = [True] * batch_size
    if group_keys is None:
        group_keys = [f"group-{index}" for index in range(batch_size)]
    if step_ids is None:
        step_ids = list(range(batch_size))
    if coordinate_transforms is None:
        coordinate_transforms = [None] * batch_size

    return SimpleNamespace(
        batch={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
        },
        non_tensor_batch={
            "patch_applied": patch_applied,
            "ground_truth": ground_truths,
            "selection_group_key": group_keys,
            "task_id": group_keys,
            "step_id": step_ids,
            "coordinate_transform": coordinate_transforms,
        },
    )


def test_patch_expert_swipe_scales_both_endpoints_and_strips_reward_metadata():
    raw_action = {
        "action": "swipe",
        "coordinate": [240, 1589],
        "coordinate2": [1200, 3000],
        "bbox": [200, 1500, 300, 1650],
        "device_dim": [1440, 3120],
    }
    raw_snapshot = copy.deepcopy(raw_action)
    coordinate_transform = {
        "original_width": 1440,
        "original_height": 3120,
        "model_width": 980,
        "model_height": 2128,
    }

    transformed = expert_action_in_model_coordinates(raw_action, coordinate_transform)

    assert transformed == {
        "action": "swipe",
        "coordinate": [163, 1084],
        "coordinate2": [817, 2046],
    }
    assert raw_action == raw_snapshot
    assert "bbox" not in transformed
    assert "device_dim" not in transformed


def test_patch_target_preserves_sampled_thinking_masks_only_action_and_keeps_prompt_tensors():
    tokenizer = CharacterTokenizer()
    thinking_prefix = "\n <thinking>\n  Keep  these spaces.\nAnd this punctuation!  \n</thinking>"
    model_action = '\n<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[9,9]}}</tool_call>'
    batch = _make_rollout_batch(
        [thinking_prefix + model_action],
        [{"action": "system_button", "button": "Back"}],
    )
    original_tensors = {key: value.clone() for key, value in batch.batch.items()}

    metrics = attach_patch_imitation_tensors(batch, tokenizer)

    expert_tool_call = build_expert_tool_call({"action": "system_button", "button": "Back"})
    expected_target_ids = tokenizer.encode(
        thinking_prefix + expert_tool_call,
        add_special_tokens=False,
    )
    thinking_ids = tokenizer.encode(thinking_prefix, add_special_tokens=False)
    tool_call_ids = tokenizer.encode(expert_tool_call, add_special_tokens=False)
    target_length = len(expected_target_ids)
    prompt_length = batch.batch["prompts"].size(-1)

    assert metrics["patch_imitation/valid_samples"] == 1.0
    assert batch.batch["patch_responses"][0, :target_length].tolist() == expected_target_ids
    # The no-loss prefix uses the exact sampled ids, rather than a synthetic or
    # expert thought.
    assert (
        batch.batch["patch_responses"][0, : len(thinking_ids)].tolist()
        == original_tensors["responses"][0, : len(thinking_ids)].tolist()
    )

    token_weights = batch.batch["patch_token_weights"][0]
    assert torch.count_nonzero(token_weights[: len(thinking_ids)]) == 0
    assert torch.all(token_weights[len(thinking_ids) : target_length] > 0)
    assert torch.count_nonzero(token_weights[target_length:]) == 0
    assert torch.allclose(
        token_weights[len(thinking_ids) : target_length],
        torch.full((len(tool_call_ids),), 1.0 / len(tool_call_ids)),
    )
    assert torch.isclose(token_weights.sum(), torch.tensor(1.0))

    # Thinking remains visible as causal context even though it carries no
    # imitation weight; only right padding is removed from attention.
    assert torch.all(batch.batch["patch_attention_mask"][0, prompt_length : prompt_length + target_length] == 1)
    assert torch.count_nonzero(batch.batch["patch_attention_mask"][0, prompt_length + target_length :]) == 0

    # Patch construction is side-effect free for the rollout tensors, and the
    # expert branch reuses their exact prompt ids/mask/mRoPE positions.
    for key, snapshot in original_tensors.items():
        assert torch.equal(batch.batch[key], snapshot)
    assert torch.equal(
        batch.batch["patch_input_ids"][:, :prompt_length],
        original_tensors["input_ids"][:, :prompt_length],
    )
    assert torch.equal(
        batch.batch["patch_attention_mask"][:, :prompt_length],
        original_tensors["attention_mask"][:, :prompt_length],
    )
    assert torch.equal(
        batch.batch["patch_position_ids"][..., :prompt_length],
        original_tensors["position_ids"][..., :prompt_length],
    )


def test_patch_target_without_complete_thinking_is_skipped():
    tokenizer = CharacterTokenizer()
    batch = _make_rollout_batch(
        ['<tool_call>{"name":"mobile_use","arguments":{"action":"wait","time":2}}</tool_call>'],
        [{"action": "wait", "time": 2}],
    )

    metrics = attach_patch_imitation_tensors(batch, tokenizer)

    assert metrics["patch_imitation/attempted_samples"] == 1.0
    assert metrics["patch_imitation/valid_samples"] == 0.0
    assert metrics["patch_imitation/skipped_samples"] == 1.0
    assert all(key not in batch.batch for key in PATCH_IMITATION_BATCH_KEYS)


def test_patch_target_that_would_truncate_expert_action_is_skipped_whole():
    tokenizer = CharacterTokenizer()
    batch = _make_rollout_batch(
        ["<thinking>x</thinking>"],
        [{"action": "type", "text": "x" * 200}],
        max_response_length=48,
    )

    metrics = attach_patch_imitation_tensors(batch, tokenizer)

    assert metrics["patch_imitation/attempted_samples"] == 1.0
    assert metrics["patch_imitation/valid_samples"] == 0.0
    assert metrics["patch_imitation/skipped_samples"] == 1.0
    assert all(key not in batch.batch for key in PATCH_IMITATION_BATCH_KEYS)


def test_same_task_step_patch_multiplicity_does_not_change_its_total_weight():
    single_weights = normalized_patch_sample_weights(
        ["task-a", "task-b"],
        [3, 1],
        [0, 1],
    )
    repeated_weights = normalized_patch_sample_weights(
        ["task-a", "task-a", "task-a", "task-a", "task-b"],
        [3, 3, 3, 3, 1],
        [0, 1, 2, 3, 4],
    )

    assert sum(single_weights.values()) == 1.0
    assert sum(repeated_weights.values()) == 1.0
    assert single_weights[0] == 0.5
    assert sum(repeated_weights[index] for index in range(4)) == 0.5
    assert repeated_weights[4] == 0.5


def test_actor_padding_rows_have_zero_patch_weight_without_changing_real_rows():
    real_weights = torch.tensor(
        [
            [0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.50],
        ],
        dtype=torch.float32,
    )
    padded_weights = torch.cat((real_weights, real_weights[:2]), dim=0)
    batch = SimpleNamespace(batch={"patch_token_weights": padded_weights.clone()})

    zero_patch_imitation_padding(batch, pad_size=2)

    assert torch.equal(batch.batch["patch_token_weights"][: len(real_weights)], real_weights)
    assert torch.count_nonzero(batch.batch["patch_token_weights"][-2:]) == 0
    assert torch.isclose(
        batch.batch["patch_token_weights"].sum(),
        real_weights.sum(),
    )
