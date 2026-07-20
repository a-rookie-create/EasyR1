#!/usr/bin/env python3
"""Convert AMEX expert trajectories into EasyR1 UI-S1 RL JSONL data.

Each output row contains one complete trajectory.  The semi-online trainer owns
the rollout history and reads ``trajectory_steps`` during training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

if __package__:
    from .trajectory_sampling import sample_trajectories, validate_dataset_name, validate_sample_ratio
else:
    from trajectory_sampling import sample_trajectories, validate_dataset_name, validate_sample_ratio


def action_to_json(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def build_first_step_prompt(goal: str) -> str:
    return (
        "<image>\n"
        f"Goal: {goal.strip()}\n\n"
        "Previous model outputs:\n"
        "None\n\n"
        "Predict the next Android GUI action. Return <thinking></thinking> followed by "
        "{\"name\":\"mobile_use\",\"arguments\":{...}}."
    )


def nonzero_region(region: Any) -> list[float] | None:
    if not isinstance(region, list) or len(region) != 2:
        return None
    try:
        (x1, y1), (x2, y2) = region
        bbox = [float(x1), float(y1), float(x2), float(y2)]
    except (TypeError, ValueError):
        return None
    if bbox == [0.0, 0.0, 0.0, 0.0]:
        return None
    return bbox


def amex_step_to_action(step: dict[str, Any]) -> dict[str, Any]:
    action_type = str(step.get("action", "")).upper()
    touch = step.get("touch_coord", [0, 0])
    lift = step.get("lift_coord", [0, 0])
    device_dim = step.get("device_dim")

    if action_type == "TAP":
        action = {"action": "click", "coordinate": touch}
    elif action_type == "SWIPE":
        action = {"action": "swipe", "coordinate": touch, "coordinate2": lift}
    elif action_type == "TYPE":
        action = {"action": "type", "text": step.get("type_text", "")}
    elif action_type == "PRESS_BACK":
        action = {"action": "system_button", "button": "Back"}
    elif action_type == "PRESS_HOME":
        action = {"action": "system_button", "button": "Home"}
    elif action_type == "PRESS_ENTER":
        action = {"action": "system_button", "button": "Enter"}
    elif action_type == "TASK_COMPLETE":
        action = {"action": "terminate", "status": "success"}
    elif action_type == "TASK_IMPOSSIBLE":
        action = {"action": "terminate", "status": "failure"}
    else:
        raise ValueError(f"Unsupported AMEX action: {action_type!r}; raw={step}")

    bbox = nonzero_region(step.get("interest_region"))
    if bbox is not None:
        action["bbox"] = bbox
    if device_dim:
        action["device_dim"] = device_dim
    return action


def amex_instruction_paths(amex_dir: Path) -> list[Path]:
    instruction_dir = amex_dir / "instruction_anno" / "instruction_anno"
    return sorted(instruction_dir.glob("*.json"))


def has_trajectory_steps(instruction_path: Path) -> bool:
    """Return whether an instruction can produce a non-empty trajectory."""
    with instruction_path.open("r", encoding="utf-8") as f:
        item = json.load(f)
    return any(step.get("image_path") for step in item.get("steps", []))


def split_trajectories(
    instruction_paths: list[Path], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[Path]]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("--train-ratio and --val-ratio must be positive and leave a non-empty test split")
    shuffled_paths = list(instruction_paths)
    random.Random(seed).shuffle(shuffled_paths)
    train_count = round(len(shuffled_paths) * train_ratio)
    val_count = round(len(shuffled_paths) * val_ratio)
    return {
        "train": shuffled_paths[:train_count],
        "val": shuffled_paths[train_count : train_count + val_count],
        "test": shuffled_paths[train_count + val_count :],
    }


def build_amex_trajectory(instruction_path: Path, source_screenshot_dir: Path) -> dict[str, Any] | None:
    with instruction_path.open("r", encoding="utf-8") as f:
        item = json.load(f)

    goal = item.get("instruction", "")
    steps = []
    for raw_step in item.get("steps", []):
        image_name = raw_step.get("image_path")
        if not image_name:
            continue
        source_image_path = source_screenshot_dir / image_name
        if not source_image_path.is_file():
            raise FileNotFoundError(f"Missing AMEX screenshot: {source_image_path}")
        action = amex_step_to_action(raw_step)
        steps.append(
            {
                "step_id": raw_step.get("step_id"),
                "image": str(source_image_path),
                "action": action,
            }
        )

    if not steps:
        return None

    return {
        "prompt": build_first_step_prompt(goal),
        "answer": action_to_json(steps[0]["action"]),
        "images": [steps[0]["image"]],
        "task_id": instruction_path.stem,
        "source": "amex",
        "goal": goal,
        "trajectory_steps": steps,
    }


def convert_amex_dir(
    amex_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    limit_trajectories: int | None,
    overwrite: bool,
    sample_ratio: float = 1.0,
    dataset_name: str = "ui_s1_amex_rl",
) -> tuple[int, int, int]:
    validate_sample_ratio(sample_ratio)
    validate_dataset_name(dataset_name)
    if limit_trajectories is not None and limit_trajectories < 1:
        raise ValueError("--limit-trajectories must be positive")
    if sample_ratio < 1 and limit_trajectories is not None:
        raise ValueError("--sample-ratio cannot be combined with --limit-trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / (
        "conversion_stats.json" if dataset_name == "ui_s1_amex_rl" else f"{dataset_name}_stats.json"
    )
    output_paths = [
        output_dir / f"{dataset_name}_train.jsonl",
        output_dir / f"{dataset_name}_val.jsonl",
        output_dir / f"{dataset_name}_test.jsonl",
        stats_path,
    ]
    if not overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError(f"Output already exists in {output_dir}; pass --overwrite to replace it")
    source_screenshot_dir = amex_dir / "screenshot" / "screenshot"
    if not source_screenshot_dir.is_dir():
        raise FileNotFoundError(f"AMEX screenshot directory not found: {source_screenshot_dir}")
    instruction_paths = amex_instruction_paths(amex_dir)
    split_paths = split_trajectories(instruction_paths, train_ratio, val_ratio, seed)
    eligible_paths = {
        split: [path for path in paths if has_trajectory_steps(path)]
        for split, paths in split_paths.items()
    }
    sampled_paths = {
        split: sample_trajectories(paths, sample_ratio, seed, split)
        for split, paths in eligible_paths.items()
    }
    train_paths = sampled_paths["train"]
    val_paths = sampled_paths["val"]
    test_paths = sampled_paths["test"]
    if limit_trajectories is not None:
        train_paths = train_paths[:limit_trajectories]

    counts = []
    action_counts: dict[str, int] = {}
    for split, paths in (("train", train_paths), ("val", val_paths), ("test", test_paths)):
        output_path = output_dir / f"{dataset_name}_{split}.jsonl"
        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for instruction_path in paths:
                example = build_amex_trajectory(instruction_path, source_screenshot_dir)
                if example is None:
                    continue
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                for step in example["trajectory_steps"]:
                    action_name = step["action"]["action"]
                    action_counts[action_name] = action_counts.get(action_name, 0) + 1
                count += 1
        counts.append(count)
    stats = {
        "source": "amex",
        "dataset_name": dataset_name,
        "split_strategy": "seeded_trajectory_shuffle",
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": round(1.0 - train_ratio - val_ratio, 10)},
        "seed": seed,
        "sample_ratio": sample_ratio,
        "available_episode_counts": {split: len(paths) for split, paths in eligible_paths.items()},
        "limit_trajectories": limit_trajectories,
        "episode_counts": {"train": counts[0], "val": counts[1], "test": counts[2]},
        "action_counts": dict(sorted(action_counts.items())),
        "source_screenshot_dir": str(source_screenshot_dir),
        "image_path_mode": "direct",
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return counts[0], counts[1], counts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amex-dir",
        type=Path,
        required=True,
        help="Directory containing the source AMEX dataset.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="AMEX trajectory-level train split ratio."
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="AMEX trajectory-level validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic episode-level AMEX split seed.")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Randomly retain this fraction of complete trajectories in every split (0 < ratio <= 1).",
    )
    parser.add_argument(
        "--dataset-name",
        default="ui_s1_amex_rl",
        help="Output dataset filename prefix (default: ui_s1_amex_rl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for AMEX trajectory-level JSONL files.",
    )
    parser.add_argument(
        "--limit-trajectories",
        type=int,
        default=None,
        help="Optional cap on training trajectories only; validation remains episode-disjoint.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing generated JSONL files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_count, val_count, test_count = convert_amex_dir(
        args.amex_dir,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_trajectories=args.limit_trajectories,
        overwrite=args.overwrite,
        sample_ratio=args.sample_ratio,
        dataset_name=args.dataset_name,
    )
    print(f"Wrote {train_count} AMEX train trajectories.")
    print(f"Wrote {val_count} AMEX validation trajectories.")
    print(f"Wrote {test_count} AMEX test trajectories.")


if __name__ == "__main__":
    main()
