from types import SimpleNamespace

import pytest

from verl.trainer.ray_trainer import (
    PATCH_IMITATION_STATE_VERSION,
    _patch_imitation_config_dict,
    _validate_actor_checkpoint_for_resume,
    _validate_patch_imitation_resume_state,
)


def _config(enabled: bool = True, **overrides):
    values = {
        "enabled": enabled,
        "lambda_initial": 0.5 if enabled else 0.0,
        "lambda_decay": 0.9,
        "lambda_min": 0.1 if enabled else 0.0,
        "target_mode": "action_only",
        "history_mode": "keep_model_thinking",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _state(config, global_step: int = 4):
    patch_lambda = (
        max(config.lambda_min, config.lambda_initial * config.lambda_decay ** (global_step - 1))
        if config.enabled
        else 0.0
    )
    return {
        "version": PATCH_IMITATION_STATE_VERSION,
        "global_step": global_step,
        "patch_imitation": _patch_imitation_config_dict(config),
        "patch_imitation_lambda": patch_lambda,
    }


def test_matching_patch_configuration_can_resume():
    config = _config()
    _validate_patch_imitation_resume_state(_state(config), config, checkpoint_global_step=4)


def test_legacy_checkpoint_is_allowed_only_when_patch_imitation_is_disabled():
    _validate_patch_imitation_resume_state(None, _config(enabled=False), checkpoint_global_step=4)
    with pytest.raises(RuntimeError, match="predates patch-imitation"):
        _validate_patch_imitation_resume_state(None, _config(enabled=True), checkpoint_global_step=4)


def test_resume_rejects_patch_ablation_or_schedule_changes():
    saved_config = _config()
    with pytest.raises(RuntimeError, match="enabled or disabled"):
        _validate_patch_imitation_resume_state(_state(saved_config), _config(enabled=False), checkpoint_global_step=4)
    with pytest.raises(RuntimeError, match="differs from the checkpoint"):
        _validate_patch_imitation_resume_state(
            _state(saved_config),
            _config(lambda_decay=0.8),
            checkpoint_global_step=4,
        )


def test_resume_rejects_corrupt_step_or_unknown_state_version():
    config = _config()
    with pytest.raises(RuntimeError, match="global_step does not match"):
        _validate_patch_imitation_resume_state(_state(config, global_step=3), config, checkpoint_global_step=4)

    state = _state(config)
    state["version"] = PATCH_IMITATION_STATE_VERSION + 1
    with pytest.raises(RuntimeError, match="Unsupported trainer_state"):
        _validate_patch_imitation_resume_state(state, config, checkpoint_global_step=4)


def test_resume_rejects_missing_or_inconsistent_saved_lambda():
    config = _config()
    state = _state(config)
    state.pop("patch_imitation_lambda")
    with pytest.raises(RuntimeError, match="missing or inconsistent"):
        _validate_patch_imitation_resume_state(state, config, checkpoint_global_step=4)


def test_warm_start_initial_checkpoint_uses_phase_zero_without_lambda():
    config = _config()
    state = _state(config, global_step=99)
    state["phase_step"] = 0
    state["global_step_offset"] = 99
    state["patch_imitation_lambda"] = None

    _validate_patch_imitation_resume_state(
        state,
        config,
        checkpoint_global_step=99,
        checkpoint_phase_step=0,
    )

    state = _state(config)
    state["patch_imitation_lambda"] = 0.123
    with pytest.raises(RuntimeError, match="missing or inconsistent"):
        _validate_patch_imitation_resume_state(state, config, checkpoint_global_step=4)


def _make_full_actor_checkpoint(path, world_size: int):
    actor_path = path / "actor"
    actor_path.mkdir(parents=True)
    for rank in range(world_size):
        for prefix in ("model", "optim", "extra_state"):
            (actor_path / f"{prefix}_world_size_{world_size}_rank_{rank}.pt").touch()
    (path / "dataloader.pt").touch()


def test_full_actor_checkpoint_requires_matching_world_size(tmp_path):
    checkpoint_path = tmp_path / "global_step_4"
    _make_full_actor_checkpoint(checkpoint_path, world_size=2)

    _validate_actor_checkpoint_for_resume(str(checkpoint_path), current_world_size=2)
    with pytest.raises(RuntimeError, match="different GPU count is unsupported"):
        _validate_actor_checkpoint_for_resume(str(checkpoint_path), current_world_size=4)


def test_full_actor_checkpoint_rejects_missing_training_state(tmp_path):
    checkpoint_path = tmp_path / "global_step_4"
    _make_full_actor_checkpoint(checkpoint_path, world_size=2)
    (checkpoint_path / "actor" / "optim_world_size_2_rank_1.pt").unlink()

    with pytest.raises(RuntimeError, match="missing actor shards"):
        _validate_actor_checkpoint_for_resume(str(checkpoint_path), current_world_size=2)
