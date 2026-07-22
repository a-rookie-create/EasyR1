import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from verl.trainer.ray_trainer import (
    RayPPOTrainer,
    _as_object_array,
    _build_semi_online_prompt,
    _compact_ui_action,
    _select_most_diverse_rollout_subset,
    _thinking_for_history,
)
from verl.utils.dataset import RLHFDataset, qwen_coordinate_transform, qwen_coordinate_transforms
from verl.utils.logger import TrainingProgressLogger


def test_semi_online_prompt_contains_one_placeholder_per_retained_image():
    prompt = _build_semi_online_prompt("Open settings", ["<tool_call>{}</tool_call>"], image_count=2)
    assert prompt.count("<image>") == 2
    assert "Open settings" in prompt
    assert "Previous screenshot 1" in prompt
    assert "Current screenshot" in prompt
    assert "Previous model reasoning" in prompt


def test_legacy_null_action_fields_are_removed_before_reward_ground_truth():
    action = {
        "action": "click",
        "coordinate": [100, 200],
        "text": None,
        "time": None,
        "button": None,
        "coordinate2": None,
    }
    assert _compact_ui_action(action) == {"action": "click", "coordinate": [100, 200]}


def test_rollout_history_keeps_only_thinking_and_discards_action():
    response = '<thinking>Open it.</thinking><tool_call>{"name":"mobile_use","arguments":{"action":"open","text":"Calendar"}}</tool_call>'
    assert _thinking_for_history(response) == "<thinking>\nOpen it.\n</thinking>"
    assert _thinking_for_history("<tool_call>{}</tool_call>") is None


def test_qwen_transform_uses_final_grid_dimensions_not_only_pixel_budget():
    image_processor = SimpleNamespace(patch_size=14)
    transform = qwen_coordinate_transform((1440, 3120), [1, 152, 70], image_processor)

    assert transform == {
        "original_width": 1440.0,
        "original_height": 3120.0,
        "model_width": 980.0,
        "model_height": 2128.0,
        "scale_x": 980 / 1440,
        "scale_y": 2128 / 3120,
    }


def test_multi_image_transforms_are_kept_in_input_order_and_may_differ():
    image_processor = SimpleNamespace(patch_size=14)
    transforms = qwen_coordinate_transforms(
        [(1440, 3120), (1080, 1920)], [[1, 152, 70], [1, 120, 70]], image_processor
    )

    assert transforms[0]["model_width"] == 980.0
    assert transforms[0]["model_height"] == 2128.0
    assert transforms[1]["model_width"] == 980.0
    assert transforms[1]["model_height"] == 1680.0
    assert transforms[0]["scale_y"] != transforms[1]["scale_y"]


def test_object_array_keeps_variable_image_history_as_one_value_per_row():
    single_image_history = _as_object_array([[{"scale_x": 0.7}]])
    two_image_history = _as_object_array([[{"scale_x": 0.7}, {"scale_x": 0.8}]])

    merged = np.concatenate([single_image_history, two_image_history], axis=0)

    assert merged.shape == (2,)
    assert len(merged[0]) == 1
    assert len(merged[1]) == 2


def test_dataset_item_records_original_image_sizes_for_coordinate_transform(tmp_path, monkeypatch):
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (1440, 3120)).save(image_path)

    class FakeProcessor:
        image_processor = type("Qwen2VLImageProcessor", (), {"patch_size": 14})()

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return "prompt"

        def __call__(self, images, prompts, add_special_tokens, return_tensors):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "image_grid_thw": torch.tensor([[1, 152, 70]]),
            }

    class FakeTokenizer:
        pad_token_id = 0

        def encode(self, prompt, add_special_tokens):
            return [1, 2]

    monkeypatch.setattr(
        "verl.models.transformers.qwen2_vl.get_rope_index",
        lambda processor, input_ids, **kwargs: torch.zeros((3, len(input_ids)), dtype=torch.long),
    )

    dataset = object.__new__(RLHFDataset)
    dataset.dataset = [{"prompt": "click", "answer": "{}", "images": [str(image_path)]}]
    dataset.prompt_key = "prompt"
    dataset.answer_key = "answer"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.image_dir = None
    dataset.format_prompt = None
    dataset.processor = FakeProcessor()
    dataset.tokenizer = FakeTokenizer()
    dataset.min_pixels = None
    dataset.max_pixels = None
    dataset.max_prompt_length = 128
    dataset.truncation = "error"

    item = dataset[0]

    assert item["coordinate_transform"]["original_width"] == 1440.0
    assert item["coordinate_transform"]["original_height"] == 3120.0
    assert item["coordinate_transform"]["model_width"] == 980.0
    assert item["coordinate_transform"]["model_height"] == 2128.0


def test_diversity_refill_selects_the_most_diverse_fixed_size_subset():
    candidates = [{"traj_uid": f"trajectory-{idx}"} for idx in range(8)]
    scores = {
        "trajectory-0": 0.0,
        "trajectory-1": 0.0,
        "trajectory-2": 0.0,
        "trajectory-3": 0.0,
        "trajectory-4": -10.0,
        "trajectory-5": 10.0,
        "trajectory-6": 0.0,
        "trajectory-7": 0.0,
    }

    selected, diversity_std = _select_most_diverse_rollout_subset(candidates, scores, selected_count=4)

    selected_ids = {state["traj_uid"] for state in selected}
    assert len(selected) == 4
    assert {"trajectory-4", "trajectory-5"}.issubset(selected_ids)
    assert diversity_std > 0.0


def test_training_progress_log_is_human_readable_and_flushed(tmp_path):
    path = tmp_path / "training_progress.log"
    progress = TrainingProgressLogger(str(path))
    progress.log(
        "DIVERSITY",
        "RETRY",
        step=3,
        task_ids=["task-a", "task-b"],
        candidate_counts=[5, 4],
        diversity_std=[0.1234, 0.5678],
        refill_counts=[4, 0],
        next_candidate_counts=[8, 4],
        selected_rollout_progress="[[1/6,2/6,2/6,3/6],[1/4,1/4,2/4,2/4]]",
        elapsed_s=0.25,
    )

    line = path.read_text(encoding="utf-8").strip()
    assert "STEP 3 | DIVERSITY | RETRY" in line
    assert 'task_ids=["task-a","task-b"]' in line
    assert "candidate_counts=[5,4]" in line
    assert "diversity_std=[0.1234,0.5678]" in line
    assert "refill_counts=[4,0]" in line
    assert "next_candidate_counts=[8,4]" in line
    assert "selected_rollout_progress=[[1/6,2/6,2/6,3/6],[1/4,1/4,2/4,2/4]]" in line
    assert "elapsed_s=0.2500" in line


def test_rollout_audit_log_serializes_numpy_coordinate_transforms(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    rollout_log_path = tmp_path / "semi_online_rollouts.jsonl"
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(rollout_log_path=str(rollout_log_path)))

    trainer._append_semi_online_rollout_log_entries(
        [
            {
                "record_type": "rollout_progress",
                "events": [
                    {
                        "coordinate_transform": np.array({"scale_x": 0.5}, dtype=object),
                        "image_coordinate_transforms": np.array([{"scale_y": 0.7}], dtype=object),
                    }
                ],
            }
        ]
    )

    record = json.loads(rollout_log_path.read_text(encoding="utf-8"))
    assert record["events"][0]["coordinate_transform"] == {"scale_x": 0.5}
    assert record["events"][0]["image_coordinate_transforms"] == [{"scale_y": 0.7}]


def test_epoch_checkpoint_cadence_and_rollout_trace(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(save_freq=-1, save_every_n_epochs=1, rollout_log_path=str(tmp_path / "rollouts.jsonl"))
    )
    trainer.steps_per_epoch = 5
    trainer.global_step = 4
    assert not trainer._should_save_checkpoint()
    trainer.global_step = 5
    assert trainer._should_save_checkpoint()

    trainer._write_semi_online_rollout_logs(
        [{"task_id": "task-1", "events": [{"step_id": 0, "patch_applied": True}]}],
        candidate_attempt=2,
        advantage_std=0.1,
    )
    record = json.loads((tmp_path / "rollouts.jsonl").read_text(encoding="utf-8"))
    assert record["record_type"] == "rollout"
    assert record["global_step"] == 5
    assert record["candidate_attempt"] == 2
    assert record["selected_batch"]
    assert record["events"][0]["patch_applied"]
