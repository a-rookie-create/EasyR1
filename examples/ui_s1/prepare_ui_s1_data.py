#!/usr/bin/env python3
"""Prepare UI-S1-style GUI data for EasyR1 experiments.

Supported inputs:
- AndroidControl-style trajectory jsonl.
- AMEX directory with instruction_anno, element_anno, and screenshot folders.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def action_to_json(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "None"

    lines = []
    for idx, action in enumerate(history, start=1):
        lines.append(f"{idx}. {action_to_json(action)}")
    return "\n".join(lines)


def build_prompt(goal: str, history: list[dict[str, Any]]) -> str:
    return (
        "<image>\n"
        f"Goal: {goal.strip()}\n\n"
        "Previous actions:\n"
        f"{format_history(history)}\n\n"
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
        action = {"action": action_type.lower()}

    bbox = nonzero_region(step.get("interest_region"))
    if bbox is not None:
        action["bbox"] = bbox
    if device_dim:
        action["device_dim"] = device_dim
    return action


def remap_path(path: str, image_prefix_from: str | None, image_prefix_to: str | None) -> str:
    if image_prefix_from and image_prefix_to and path.startswith(image_prefix_from):
        return image_prefix_to.rstrip("/") + path[len(image_prefix_from.rstrip("/")) :]
    return path


def iter_examples(
    input_path: Path,
    limit_trajectories: int | None = None,
    image_prefix_from: str | None = None,
    image_prefix_to: str | None = None,
):
    with input_path.open("r", encoding="utf-8") as f:
        for traj_id, line in enumerate(f):
            if limit_trajectories is not None and traj_id >= limit_trajectories:
                break

            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            goal = item.get("goal", "")
            steps = item.get("steps", [])
            history: list[dict[str, Any]] = []

            for step_id, step in enumerate(steps):
                action = step.get("action_content", {})
                screenshot = step.get("screenshot")
                if not action or not screenshot:
                    continue

                yield {
                    "prompt": build_prompt(goal, history),
                    "answer": action_to_json(action),
                    "images": [remap_path(screenshot, image_prefix_from, image_prefix_to)],
                    "task_id": str(traj_id),
                    "step_id": step_id,
                    "trajectory_success": bool(item.get("is_successful", False)),
                }
                history.append(action)


def amex_instruction_paths(amex_dir: Path) -> list[Path]:
    instruction_dir = amex_dir / "instruction_anno" / "instruction_anno"
    return sorted(instruction_dir.glob("*.json"))


def iter_amex_examples(
    amex_dir: Path,
    split: str,
    val_trajectories: int,
    limit_trajectories: int | None = None,
):
    instruction_paths = amex_instruction_paths(amex_dir)
    if split == "train":
        selected_paths = instruction_paths[:-val_trajectories] if val_trajectories > 0 else instruction_paths
    elif split == "val":
        selected_paths = instruction_paths[-val_trajectories:] if val_trajectories > 0 else []
    else:
        raise ValueError(f"Unknown split: {split}")

    if limit_trajectories is not None:
        selected_paths = selected_paths[:limit_trajectories]

    screenshot_dir = amex_dir / "screenshot" / "screenshot"
    for traj_id, instruction_path in enumerate(selected_paths):
        with instruction_path.open("r", encoding="utf-8") as f:
            item = json.load(f)

        goal = item.get("instruction", "")
        history: list[dict[str, Any]] = []
        for raw_step in item.get("steps", []):
            image_name = raw_step.get("image_path")
            if not image_name:
                continue

            image_path = screenshot_dir / image_name
            action = amex_step_to_action(raw_step)
            yield {
                "prompt": build_prompt(goal, history),
                "answer": action_to_json(action),
                "images": [str(image_path)],
                "task_id": instruction_path.stem,
                "step_id": raw_step.get("step_id"),
                "source": "amex",
            }
            history.append(action)


def build_amex_trajectory(instruction_path: Path, amex_dir: Path) -> dict[str, Any] | None:
    screenshot_dir = amex_dir / "screenshot" / "screenshot"
    with instruction_path.open("r", encoding="utf-8") as f:
        item = json.load(f)

    goal = item.get("instruction", "")
    steps = []
    for raw_step in item.get("steps", []):
        image_name = raw_step.get("image_path")
        if not image_name:
            continue
        image_path = screenshot_dir / image_name
        action = amex_step_to_action(raw_step)
        steps.append(
            {
                "step_id": raw_step.get("step_id"),
                "image": str(image_path),
                "action": action,
            }
        )

    if not steps:
        return None

    return {
        "prompt": build_prompt(goal, []),
        "answer": action_to_json(steps[0]["action"]),
        "images": [steps[0]["image"]],
        "task_id": instruction_path.stem,
        "source": "amex",
        "goal": goal,
        "trajectory_steps": steps,
    }


def convert_file(
    input_path: Path,
    output_path: Path,
    limit_trajectories: int | None,
    image_prefix_from: str | None,
    image_prefix_to: str | None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for example in iter_examples(
            input_path,
            limit_trajectories=limit_trajectories,
            image_prefix_from=image_prefix_from,
            image_prefix_to=image_prefix_to,
        ):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_amex_dir(
    amex_dir: Path,
    output_dir: Path,
    val_trajectories: int,
    limit_trajectories: int | None,
    output_mode: str,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_mode == "trajectories":
        instruction_paths = amex_instruction_paths(amex_dir)
        train_paths = instruction_paths[:-val_trajectories] if val_trajectories > 0 else instruction_paths
        val_paths = instruction_paths[-val_trajectories:] if val_trajectories > 0 else []
        if limit_trajectories is not None:
            train_paths = train_paths[:limit_trajectories]
            val_paths = val_paths[:limit_trajectories]

        counts = []
        for split, paths in (("train", train_paths), ("val", val_paths)):
            output_path = output_dir / f"amex_{split}_trajectories.jsonl"
            count = 0
            with output_path.open("w", encoding="utf-8") as f:
                for instruction_path in paths:
                    example = build_amex_trajectory(instruction_path, amex_dir)
                    if example is None:
                        continue
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
            counts.append(count)
        return counts[0], counts[1]

    if output_mode != "steps":
        raise ValueError(f"Unknown AMEX output mode: {output_mode}")

    counts = []
    for split in ("train", "val"):
        output_path = output_dir / f"amex_{split}_steps.jsonl"
        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for example in iter_amex_examples(
                amex_dir,
                split=split,
                val_trajectories=val_trajectories,
                limit_trajectories=limit_trajectories,
            ):
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        counts.append(count)
    return counts[0], counts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-input",
        default=None,
        help="Input trajectory jsonl for training.",
    )
    parser.add_argument(
        "--val-input",
        default=None,
        help="Input trajectory jsonl for validation.",
    )
    parser.add_argument(
        "--amex-dir",
        default=None,
        help="Optional AMEX directory. If set, train-input and val-input are ignored.",
    )
    parser.add_argument(
        "--amex-val-trajectories",
        type=int,
        default=20,
        help="Number of AMEX trajectories to reserve for validation.",
    )
    parser.add_argument(
        "--amex-output-mode",
        choices=["steps", "trajectories"],
        default="steps",
        help="Whether AMEX output should be step-level samples or full trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/zst/biye215/datasets/ui_s1_easy_r1",
        help="Output directory for EasyR1 step-level jsonl files.",
    )
    parser.add_argument(
        "--limit-trajectories",
        type=int,
        default=None,
        help="Optional trajectory limit for smoke data generation.",
    )
    parser.add_argument(
        "--image-prefix-from",
        default=None,
        help="Optional screenshot path prefix to replace, for example /datasets/AndroidControl/images.",
    )
    parser.add_argument(
        "--image-prefix-to",
        default=None,
        help="Optional replacement screenshot path prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.amex_dir:
        train_count, val_count = convert_amex_dir(
            Path(args.amex_dir),
            output_dir,
            val_trajectories=args.amex_val_trajectories,
            limit_trajectories=args.limit_trajectories,
            output_mode=args.amex_output_mode,
        )
        print(f"Wrote {train_count} AMEX train {args.amex_output_mode} examples.")
        print(f"Wrote {val_count} AMEX val {args.amex_output_mode} examples.")
        return

    if args.train_input is None or args.val_input is None:
        raise ValueError("Set --amex-dir or provide both --train-input and --val-input.")

    train_count = convert_file(
        Path(args.train_input),
        output_dir / "android_control_train_steps.jsonl",
        args.limit_trajectories,
        args.image_prefix_from,
        args.image_prefix_to,
    )
    val_count = convert_file(
        Path(args.val_input),
        output_dir / "android_control_val_steps.jsonl",
        args.limit_trajectories,
        args.image_prefix_from,
        args.image_prefix_to,
    )
    print(f"Wrote {train_count} train step examples.")
    print(f"Wrote {val_count} val step examples.")


if __name__ == "__main__":
    main()
