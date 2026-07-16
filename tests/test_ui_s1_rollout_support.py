import json
from types import SimpleNamespace

from verl.trainer.ray_trainer import RayPPOTrainer, _build_semi_online_prompt


def test_semi_online_prompt_contains_one_placeholder_per_retained_image():
    prompt = _build_semi_online_prompt("Open settings", ["<tool_call>{}</tool_call>"], image_count=2)
    assert prompt.count("<image>") == 2
    assert "Open settings" in prompt
    assert "Previous model outputs" in prompt


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
