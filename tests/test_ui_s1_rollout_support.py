import json
from types import SimpleNamespace

from verl.trainer.ray_trainer import RayPPOTrainer, _build_semi_online_prompt, _compact_ui_action, _tool_call_from_action
from verl.utils.logger import TrainingProgressLogger


def test_semi_online_prompt_contains_one_placeholder_per_retained_image():
    prompt = _build_semi_online_prompt("Open settings", ["<tool_call>{}</tool_call>"], image_count=2)
    assert prompt.count("<image>") == 2
    assert "Open settings" in prompt
    assert "Previous model outputs" in prompt


def test_legacy_null_action_fields_are_removed_before_reward_and_patch_history():
    action = {
        "action": "click",
        "coordinate": [100, 200],
        "text": None,
        "time": None,
        "button": None,
        "coordinate2": None,
    }
    assert _compact_ui_action(action) == {"action": "click", "coordinate": [100, 200]}
    assert '"text"' not in _tool_call_from_action(json.dumps(action))


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
        elapsed_s=0.25,
    )

    line = path.read_text(encoding="utf-8").strip()
    assert "STEP 3 | DIVERSITY | RETRY" in line
    assert 'task_ids=["task-a","task-b"]' in line
    assert "candidate_counts=[5,4]" in line
    assert "diversity_std=[0.1234,0.5678]" in line
    assert "elapsed_s=0.2500" in line


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
