#!/usr/bin/env python3
"""Convert UI-S1 AndroidControl trajectory jsonl into EasyR1 step-level jsonl."""

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
        "Predict the next Android GUI action. Return only one compact JSON object."
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-input",
        default="/home/zst/biye215/MobileAgent/UI-S1/datasets/android_control_train_example.jsonl",
        help="Input trajectory jsonl for training.",
    )
    parser.add_argument(
        "--val-input",
        default="/home/zst/biye215/MobileAgent/UI-S1/datasets/android_control_evaluation_std.jsonl",
        help="Input trajectory jsonl for validation.",
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
