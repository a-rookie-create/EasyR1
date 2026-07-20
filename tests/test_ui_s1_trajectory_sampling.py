from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


UI_S1_DIR = Path(__file__).parents[1] / "examples" / "ui_s1"
sys.path.insert(0, str(UI_S1_DIR))

import prepare_ui_s1_android_control_rl_data as android_converter  # noqa: E402
from prepare_ui_s1_amex_rl_data import convert_amex_dir  # noqa: E402
from trajectory_sampling import (  # noqa: E402
    sample_trajectories,
    validate_dataset_name,
    validate_sample_ratio,
)


def test_sample_trajectories_scales_each_split_and_keeps_whole_items() -> None:
    trajectories = [
        {"task_id": str(index), "trajectory_steps": list(range(index + 1))}
        for index in range(10)
    ]

    sampled = sample_trajectories(trajectories, sample_ratio=0.3, seed=42, split="train")

    assert len(sampled) == 3
    assert all(item in trajectories for item in sampled)
    assert all(item["trajectory_steps"] == trajectories[int(item["task_id"])]["trajectory_steps"] for item in sampled)


def test_sample_trajectories_is_reproducible_and_split_specific() -> None:
    trajectories = list(range(20))

    first = sample_trajectories(trajectories, 0.5, seed=7, split="train")
    second = sample_trajectories(trajectories, 0.5, seed=7, split="train")
    validation = sample_trajectories(trajectories, 0.5, seed=7, split="validation")

    assert first == second
    assert first != validation


def test_sample_trajectories_keeps_at_least_one_from_non_empty_split() -> None:
    assert len(sample_trajectories([1, 2], 0.01, seed=42, split="test")) == 1


def test_amex_conversion_scales_all_splits_without_splitting_trajectories(tmp_path: Path) -> None:
    amex_dir = tmp_path / "amex"
    instruction_dir = amex_dir / "instruction_anno" / "instruction_anno"
    screenshot_dir = amex_dir / "screenshot" / "screenshot"
    instruction_dir.mkdir(parents=True)
    screenshot_dir.mkdir(parents=True)

    expected_step_counts = {}
    for trajectory_id in range(20):
        steps = []
        for step_id in range(trajectory_id % 3 + 1):
            image_name = f"{trajectory_id}_{step_id}.png"
            (screenshot_dir / image_name).touch()
            steps.append({"step_id": step_id, "image_path": image_name, "action": "TAP"})
        task_id = f"trajectory_{trajectory_id}"
        expected_step_counts[task_id] = len(steps)
        (instruction_dir / f"{task_id}.json").write_text(
            json.dumps({"instruction": task_id, "steps": steps}), encoding="utf-8"
        )

    output_dir = tmp_path / "output"
    counts = convert_amex_dir(
        amex_dir,
        output_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        seed=42,
        limit_trajectories=None,
        overwrite=False,
        sample_ratio=0.5,
        dataset_name="my_amex_half",
    )

    assert counts == (6, 2, 2)
    for split, expected_count in zip(("train", "val", "test"), counts):
        rows = [
            json.loads(line)
            for line in (output_dir / f"my_amex_half_{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == expected_count
        assert all(len(row["trajectory_steps"]) == expected_step_counts[row["task_id"]] for row in rows)
    stats = json.loads((output_dir / "my_amex_half_stats.json").read_text(encoding="utf-8"))
    assert stats["dataset_name"] == "my_amex_half"


def test_android_conversion_scales_all_official_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_splits = ("train", "validation", "test")
    episodes = []
    official_splits = {}
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for split_index, split in enumerate(target_splits):
        episode_ids = list(range(split_index * 10, split_index * 10 + 10))
        official_splits[split] = set(episode_ids)
        for episode_id in episode_ids:
            episodes.append(
                {
                    "episode_id": episode_id,
                    "goal": str(episode_id),
                    "actions": [{"action_type": "click", "x": 1, "y": 2}],
                }
            )
            (image_dir / f"{episode_id}_000.png").touch()

    monkeypatch.setattr(android_converter, "load_official_splits", lambda _: official_splits)
    monkeypatch.setattr(android_converter, "iter_episodes", lambda _: iter(episodes))
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_ui_s1_android_control_rl_data.py",
            "--android-control-dir",
            str(tmp_path),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--sample-ratio",
            "0.5",
            "--dataset-name",
            "my_android_half",
            "--seed",
            "42",
        ],
    )

    android_converter.main()

    for split in ("train", "val", "test"):
        rows = (output_dir / f"my_android_half_{split}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 5
        assert all(len(json.loads(row)["trajectory_steps"]) == 1 for row in rows)
    stats = json.loads((output_dir / "my_android_half_stats.json").read_text(encoding="utf-8"))
    assert stats["dataset_name"] == "my_android_half"


@pytest.mark.parametrize("name", ["", "   ", " padded", ".", "..", "folder/name", "folder\\name", "bad\0name"])
def test_validate_dataset_name_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="dataset-name"):
        validate_dataset_name(name)


@pytest.mark.parametrize("ratio", [0, -0.1, 1.01])
def test_validate_sample_ratio_rejects_out_of_range_values(ratio: float) -> None:
    with pytest.raises(ValueError, match="sample-ratio"):
        validate_sample_ratio(ratio)
