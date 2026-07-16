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


def action_to_json(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def build_first_step_prompt(goal: str) -> str:
    return (
        "<image>\n"
        f"Goal: {goal.strip()}\n\n"
        "Previous model outputs:\n"
        "None\n\n"
        "Predict the next Android GUI action. Return <thinking></thinking> followed by a "
        "<tool_call> containing {\"name\":\"mobile_use\",\"arguments\":{...}}."
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


def ensure_image_link(output_dir: Path, source_dir: Path) -> Path:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"AMEX screenshot directory not found: {source_dir}")
    image_link = output_dir / "images"
    if image_link.exists() or image_link.is_symlink():
        if not image_link.is_symlink() or image_link.resolve() != source_dir.resolve():
            raise FileExistsError(f"Image link already exists with a different target: {image_link}")
        return image_link
    image_link.symlink_to(source_dir, target_is_directory=True)
    return image_link


def build_amex_trajectory(
    instruction_path: Path, source_screenshot_dir: Path, output_screenshot_dir: Path
) -> dict[str, Any] | None:
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
                "image": str(output_screenshot_dir / image_name),
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
) -> tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "ui_s1_amex_rl_train.jsonl",
        output_dir / "ui_s1_amex_rl_val.jsonl",
        output_dir / "ui_s1_amex_rl_test.jsonl",
        output_dir / "conversion_stats.json",
    ]
    if not overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError(f"Output already exists in {output_dir}; pass --overwrite to replace it")
    source_screenshot_dir = amex_dir / "screenshot" / "screenshot"
    output_screenshot_dir = ensure_image_link(output_dir, source_screenshot_dir)
    instruction_paths = amex_instruction_paths(amex_dir)
    split_paths = split_trajectories(instruction_paths, train_ratio, val_ratio, seed)
    train_paths = split_paths["train"]
    val_paths = split_paths["val"]
    test_paths = split_paths["test"]
    if limit_trajectories is not None:
        train_paths = train_paths[:limit_trajectories]

    counts = []
    action_counts: dict[str, int] = {}
    for split, paths in (("train", train_paths), ("val", val_paths), ("test", test_paths)):
        output_path = output_dir / f"ui_s1_amex_rl_{split}.jsonl"
        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for instruction_path in paths:
                example = build_amex_trajectory(instruction_path, source_screenshot_dir, output_screenshot_dir)
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
        "split_strategy": "seeded_episode_shuffle_8_1_1",
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": round(1.0 - train_ratio - val_ratio, 10)},
        "seed": seed,
        "limit_trajectories": limit_trajectories,
        "episode_counts": {"train": counts[0], "val": counts[1], "test": counts[2]},
        "action_counts": dict(sorted(action_counts.items())),
        "source_screenshot_dir": str(source_screenshot_dir),
        "image_link": str(output_screenshot_dir),
    }
    (output_dir / "conversion_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
    )
    print(f"Wrote {train_count} AMEX train trajectories.")
    print(f"Wrote {val_count} AMEX validation trajectories.")
    print(f"Wrote {test_count} AMEX test trajectories.")


if __name__ == "__main__":
    main()
