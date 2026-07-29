import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "examples" / "ui_s1" / "run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_warm_start_prepares_history_before_any_gpu_or_train_log(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    _write_jsonl(train_file, [{"trajectory_steps": [{"image": str(tmp_path / "missing.png")}]}])
    _write_jsonl(val_file, [{"trajectory_steps": [{"image": str(tmp_path / "missing.png")}]}])

    output_root = tmp_path / "output"
    source_run = output_root / "source"
    source_checkpoint = source_run / "global_step_99"
    adapter = source_checkpoint / "actor" / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 64,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            }
        ),
        encoding="utf-8",
    )
    (source_checkpoint / "dataloader.pt").write_bytes(b"placeholder")
    (source_run / "experiment_config.json").write_text(
        json.dumps(
            {
                "data": {
                    "train_files": str(train_file),
                    "shuffle": True,
                    "seed": 1,
                    "rollout_batch_size": 4,
                    "mini_rollout_batch_size": None,
                }
            }
        ),
        encoding="utf-8",
    )
    (source_run / "training_progress.log").write_text("a | STEP 99 | END\na | STEP 100 | START\n", encoding="utf-8")
    _write_jsonl(source_run / "experiment_log.jsonl", [{"step": 99}, {"step": 100}])
    _write_jsonl(source_run / "semi_online_rollouts.jsonl", [{"global_step": 99}, {"global_step": 100}])

    runtime_env = tmp_path / "runtime.env"
    runtime_env.write_text(
        "\n".join(
            [
                f"MODEL_PATH={tmp_path / 'base'}",
                f"TOKENIZER_PATH={tmp_path / 'base'}",
                f"DATA_DIR={data_dir}",
                f"OUTPUT_ROOT={output_root}",
                "GPU_IDS=0,1",
                "RAY_DASHBOARD_HOST=127.0.0.1",
                "VLLM_GPU_MEMORY_UTILIZATION=0.6",
                "VLLM_ENFORCE_EAGER=true",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ | {
        "RUNTIME_ENV": str(runtime_env),
        "TRAIN_ENV": str(ROOT / "examples" / "ui_s1" / "train.env"),
        "DATASET": "amex",
        "TRAIN_FILE": str(train_file),
        "VAL_FILE": str(val_file),
        "RUN_NAME": "warm",
        "RESUME": "false",
        "WARM_START_CHECKPOINT_PATH": str(source_checkpoint),
        "WARM_START_DATALOADER": "inherit",
        # Deliberately fail after lineage preparation but before raw logging,
        # monitoring, Ray, or GPU access.
        "N_GPUS_PER_NODE": "invalid",
    }
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 2
    assert "N_GPUS_PER_NODE must be a positive integer" in result.stderr
    target = output_root / "warm"
    assert (target / "warm_start.json").is_file()
    assert not (target / "train.log").exists()
    assert [json.loads(line)["step"] for line in (target / "experiment_log.jsonl").read_text().splitlines()] == [99]
    assert "STEP 100" not in (target / "training_progress.log").read_text(encoding="utf-8")


def test_resume_defaults_to_tracker_checkpoint_and_rolls_back_future_history_before_training(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_file = data_dir / "amex_train.jsonl"
    val_file = data_dir / "amex_val.jsonl"
    missing_image = tmp_path / "missing.png"
    _write_jsonl(train_file, [{"trajectory_steps": [{"image": str(missing_image)}]}])
    _write_jsonl(val_file, [{"trajectory_steps": [{"image": str(missing_image)}]}])

    output_root = tmp_path / "output"
    run_dir = output_root / "resume"
    checkpoint = run_dir / "global_step_99"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    for rank in range(2):
        for prefix in ("model", "optim", "extra_state"):
            (actor / f"{prefix}_world_size_2_rank_{rank}.pt").write_bytes(b"state")
    (checkpoint / "dataloader.pt").write_bytes(b"dataloader")
    (run_dir / "global_step_100").mkdir()
    (run_dir / "global_step_104").mkdir()
    (run_dir / "checkpoint_tracker.json").write_text(
        json.dumps(
            {
                "last_global_step": 99,
                "last_actor_path": str(actor),
                "best_global_step": 104,
                "best_val_reward_score": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training_progress.log").write_text(
        "2026 | STEP 99 | CHECKPOINT_SAVE | END\n"
        "2026 | STEP 99 | STEP | END\n"
        "2026 | STEP 100 | STEP | END\n"
        "2026 | STEP 104 | STEP | END\n",
        encoding="utf-8",
    )
    _write_jsonl(run_dir / "experiment_log.jsonl", [{"step": 99}, {"step": 100}, {"step": 104}])
    _write_jsonl(
        run_dir / "semi_online_rollouts.jsonl",
        [{"global_step": 99}, {"global_step": 100}, {"global_step": 104}],
    )
    (run_dir / "train.log").write_text("old branch", encoding="utf-8")

    runtime_env = tmp_path / "runtime.env"
    runtime_env.write_text(
        "\n".join(
            [
                f"MODEL_PATH={tmp_path / 'base'}",
                f"TOKENIZER_PATH={tmp_path / 'base'}",
                f"DATA_DIR={data_dir}",
                f"OUTPUT_ROOT={output_root}",
                "GPU_IDS=0,1",
                "RAY_DASHBOARD_HOST=127.0.0.1",
                "VLLM_GPU_MEMORY_UTILIZATION=0.6",
                "VLLM_ENFORCE_EAGER=true",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ | {
        "RUNTIME_ENV": str(runtime_env),
        "TRAIN_ENV": str(ROOT / "examples" / "ui_s1" / "train.env"),
        "DATASET": "amex",
        "TRAIN_FILE": str(train_file),
        "VAL_FILE": str(val_file),
        "RUN_NAME": "resume",
        "RESUME": "true",
    }
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)

    # The deliberately missing image stops execution after resume rollback but
    # before Ray or the trainer is started.
    assert result.returncode != 0
    assert "missing first screenshot" in (result.stdout + result.stderr)
    assert "Resume rollback prepared at global_step_99" in result.stdout
    assert [json.loads(line)["step"] for line in (run_dir / "experiment_log.jsonl").read_text().splitlines()] == [99]
    assert [
        json.loads(line)["global_step"]
        for line in (run_dir / "semi_online_rollouts.jsonl").read_text().splitlines()
    ] == [99]
    assert (run_dir / "training_progress.log").read_text(encoding="utf-8").rstrip().endswith(
        "| STEP 99 | STEP | END"
    )
    assert not (run_dir / "global_step_100").exists()
    assert not (run_dir / "global_step_104").exists()
    tracker = json.loads((run_dir / "checkpoint_tracker.json").read_text(encoding="utf-8"))
    assert tracker["last_global_step"] == 99
    assert tracker["best_global_step"] == 0
    assert (run_dir / "rollback_archive" / "before_global_step_99" / "train.log").is_file()

    # The real file logger must preserve the truncated prefix and append the
    # first newly completed step instead of reopening the file in write mode.
    from verl.utils.logger.logger import FileLogger

    logger = FileLogger(
        {
            "trainer": {
                "save_checkpoint_path": str(run_dir),
                "load_checkpoint_path": None,
                "find_last_checkpoint": True,
                "append_existing_history": False,
            }
        }
    )
    logger.log({"reward/overall": 0.5}, step=100)
    assert [
        json.loads(line)["step"]
        for line in (run_dir / "experiment_log.jsonl").read_text(encoding="utf-8").splitlines()
    ] == [99, 100]
