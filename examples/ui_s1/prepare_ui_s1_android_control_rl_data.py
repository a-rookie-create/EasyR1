#!/usr/bin/env python3
"""Convert AndroidControl episodes into EasyR1 UI-S1 semi-online RL data.

The input is the official gzip-compressed TFRecord release.  One JSONL row is
one full episode, never an individual step.  Screenshot paths point to the
already extracted AndroidControl SFT images so conversion does not duplicate
the 41 GB image directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from .trajectory_sampling import sample_trajectories, validate_dataset_name, validate_sample_ratio
else:
    from trajectory_sampling import sample_trajectories, validate_dataset_name, validate_sample_ratio


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def protobuf_fields(buf: bytes):
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, pos = read_varint(buf, pos)
        elif wire_type == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire_type == 2:
            size, pos = read_varint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
        elif wire_type == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type: {wire_type}")
        yield field_number, wire_type, value


def parse_feature(feature_buf: bytes) -> tuple[str, list[int] | list[bytes]]:
    for field_number, wire_type, value in protobuf_fields(feature_buf):
        if field_number == 1 and wire_type == 2:
            values = [item for number, item_type, item in protobuf_fields(value) if number == 1 and item_type == 2]
            return "bytes", values
        if field_number == 3 and wire_type == 2:
            values: list[int] = []
            for number, item_type, item in protobuf_fields(value):
                if number != 1:
                    continue
                if item_type == 0:
                    values.append(item)
                elif item_type == 2:
                    pos = 0
                    while pos < len(item):
                        number_value, pos = read_varint(item, pos)
                        values.append(number_value)
            return "int64", values
    return "unknown", []


def parse_example(example_buf: bytes) -> dict[str, tuple[str, list[int] | list[bytes]]]:
    features: dict[str, tuple[str, list[int] | list[bytes]]] = {}
    for field_number, wire_type, value in protobuf_fields(example_buf):
        if field_number != 1 or wire_type != 2:
            continue
        for entry_number, entry_wire_type, entry in protobuf_fields(value):
            if entry_number != 1 or entry_wire_type != 2:
                continue
            key: str | None = None
            feature_buf: bytes | None = None
            for number, item_type, item in protobuf_fields(entry):
                if number == 1 and item_type == 2:
                    key = item.decode("utf-8", errors="replace")
                elif number == 2 and item_type == 2:
                    feature_buf = item
            if key is not None and feature_buf is not None:
                features[key] = parse_feature(feature_buf)
    return features


def read_tfrecord_records(path: Path):
    with gzip.open(path, "rb") as handle:
        while length_bytes := handle.read(8):
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated TFRecord length in {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            handle.read(4)
            record = handle.read(length)
            handle.read(4)
            if len(record) != length:
                raise ValueError(f"Truncated TFRecord payload in {path}")
            yield record


def bytes_values(features: dict[str, tuple[str, list[int] | list[bytes]]], name: str) -> list[bytes]:
    return features.get(name, ("bytes", []))[1]  # type: ignore[return-value]


def int_values(features: dict[str, tuple[str, list[int] | list[bytes]]], name: str) -> list[int]:
    return features.get(name, ("int64", []))[1]  # type: ignore[return-value]


def normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("action_type", action.get("action", ""))).lower()
    if action_type == "open_app":
        return {"action": "open", "text": action.get("app_name", "")}
    if action_type == "click":
        return {"action": "click", "coordinate": [action.get("x", 0), action.get("y", 0)]}
    if action_type == "long_press":
        output: dict[str, Any] = {"action": "long_press", "coordinate": [action.get("x", 0), action.get("y", 0)]}
        if "duration" in action:
            output["time"] = action["duration"]
        return output
    if action_type in {"input_text", "type"}:
        return {"action": "type", "text": action.get("text", action.get("typed_text", ""))}
    if action_type in {"wait", "status"}:
        return {"action": "wait", "time": action.get("time", 2)}
    if action_type in {"navigate_back", "back"}:
        return {"action": "system_button", "button": "Back"}
    if action_type in {"navigate_home", "home"}:
        return {"action": "system_button", "button": "Home"}
    if action_type in {"complete", "task_complete"}:
        return {"action": "terminate", "status": "success"}
    if action_type in {"scroll", "swipe"}:
        if {"x", "y", "x2", "y2"} <= set(action):
            return {"action": "swipe", "coordinate": [action["x"], action["y"]], "coordinate2": [action["x2"], action["y2"]]}
        direction = str(action.get("direction", "down")).lower()
        x, y = action.get("x", 540.0), action.get("y", 1800.0)
        if direction == "up":
            return {"action": "swipe", "coordinate": [x, 600.0], "coordinate2": [x, 1800.0]}
        if direction == "left":
            return {"action": "swipe", "coordinate": [900.0, y], "coordinate2": [200.0, y]}
        if direction == "right":
            return {"action": "swipe", "coordinate": [200.0, y], "coordinate2": [900.0, y]}
        return {"action": "swipe", "coordinate": [x, 1800.0], "coordinate2": [x, 600.0]}
    raise ValueError(f"Unsupported AndroidControl action: {action_type!r}; raw={action}")


def build_first_step_prompt(goal: str) -> str:
    return (
        "<image>\n"
        f"Goal: {goal.strip()}\n\n"
        "Previous model outputs:\nNone\n\n"
        "Predict the next Android GUI action. Return <thinking></thinking> followed by "
        "<tool_call>{\"name\":\"mobile_use\",\"arguments\":{...}}</tool_call>."
    )


def iter_episodes(data_dir: Path) -> Iterable[dict[str, Any]]:
    shards = sorted(data_dir.glob("android_control-*-of-*"))
    if not shards:
        raise FileNotFoundError(f"No AndroidControl shards found in {data_dir}")
    for shard in shards:
        for record in read_tfrecord_records(shard):
            features = parse_example(record)
            ids = int_values(features, "episode_id")
            goals = bytes_values(features, "goal")
            if not ids or not goals:
                raise ValueError(f"Missing episode_id or goal in {shard}")
            yield {
                "episode_id": int(ids[0]),
                "goal": goals[0].decode("utf-8", errors="replace"),
                "actions": [json.loads(item.decode("utf-8", errors="replace")) for item in bytes_values(features, "actions")],
            }


def load_official_splits(data_dir: Path) -> dict[str, set[int]]:
    raw = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    return {name: {int(episode_id) for episode_id in ids} for name, ids in raw.items()}


def build_trajectory(episode: dict[str, Any], image_dir: Path) -> tuple[dict[str, Any] | None, Counter[str]]:
    steps = []
    action_counts: Counter[str] = Counter()
    for step_id, raw_action in enumerate(episode["actions"]):
        image_path = image_dir / f"{episode['episode_id']}_{step_id:03d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing extracted screenshot: {image_path}")
        action = normalize_action(raw_action)
        action_counts[action["action"]] += 1
        steps.append({"step_id": step_id, "image": str(image_path), "action": action})
    if not steps:
        # A few official episodes are empty and cannot contribute a rollout.
        return None, action_counts
    return (
        {
            "prompt": build_first_step_prompt(episode["goal"]),
            "answer": json.dumps(steps[0]["action"], ensure_ascii=False, separators=(",", ":")),
            "images": [steps[0]["image"]],
            "task_id": str(episode["episode_id"]),
            "source": "android_control",
            "goal": episode["goal"],
            "trajectory_steps": steps,
        },
        action_counts,
    )


def collect_eligible_episode_ids(
    data_dir: Path, official_splits: dict[str, set[int]], target_splits: list[str]
) -> dict[str, list[int]]:
    """Collect IDs that belong to a split and contain a complete, usable trajectory."""
    eligible_ids = {split: [] for split in target_splits}
    for episode in iter_episodes(data_dir):
        if not episode["actions"]:
            continue
        split = next(
            (name for name in target_splits if episode["episode_id"] in official_splits.get(name, set())),
            None,
        )
        if split is not None:
            eligible_ids[split].append(episode["episode_id"])
    return eligible_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--android-control-dir",
        type=Path,
        required=True,
        help="Directory containing AndroidControl episodes.",
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing prepared training images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for trajectory JSONL files.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Optional limit per split for a smoke conversion.")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Randomly retain this fraction of complete trajectories in every split (0 < ratio <= 1).",
    )
    parser.add_argument(
        "--dataset-name",
        default="ui_s1_android_control_rl",
        help="Output dataset filename prefix (default: ui_s1_android_control_rl).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic trajectory sampling seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing generated JSONL files.")
    args = parser.parse_args()

    validate_sample_ratio(args.sample_ratio)
    validate_dataset_name(args.dataset_name)
    if args.limit_episodes is not None and args.limit_episodes < 1:
        raise ValueError("--limit-episodes must be positive")
    if args.sample_ratio < 1 and args.limit_episodes is not None:
        raise ValueError("--sample-ratio cannot be combined with --limit-episodes")

    official_splits = load_official_splits(args.android_control_dir)
    target_splits = ["train", "validation", "test"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / (
        "conversion_stats.json"
        if args.dataset_name == "ui_s1_android_control_rl"
        else f"{args.dataset_name}_stats.json"
    )
    output_paths = [
        args.output_dir / f"{args.dataset_name}_train.jsonl",
        args.output_dir / f"{args.dataset_name}_val.jsonl",
        args.output_dir / f"{args.dataset_name}_test.jsonl",
        stats_path,
    ]
    if not args.overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError(f"Output already exists in {args.output_dir}; pass --overwrite to replace it")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Extracted AndroidControl image directory not found: {args.image_dir}")

    output_handles = {}
    counts = Counter()
    skipped_empty_episodes = Counter()
    action_counts: Counter[str] = Counter()
    available_episode_counts: dict[str, int] | None = None
    selected_episode_ids: dict[str, set[int]] | None = None
    if args.sample_ratio < 1:
        eligible_ids = collect_eligible_episode_ids(args.android_control_dir, official_splits, target_splits)
        available_episode_counts = {split: len(ids) for split, ids in eligible_ids.items()}
        selected_episode_ids = {
            split: set(sample_trajectories(ids, args.sample_ratio, args.seed, split))
            for split, ids in eligible_ids.items()
        }
    try:
        for split in target_splits:
            name = "val" if split == "validation" else split
            output_handles[split] = (args.output_dir / f"{args.dataset_name}_{name}.jsonl").open(
                "w", encoding="utf-8"
            )
        for episode in iter_episodes(args.android_control_dir):
            split = next((name for name in target_splits if episode["episode_id"] in official_splits.get(name, set())), None)
            if split is None:
                continue
            if selected_episode_ids is not None and episode["episode_id"] not in selected_episode_ids[split]:
                continue
            if args.limit_episodes is not None and counts[split] >= args.limit_episodes:
                continue
            trajectory, episode_actions = build_trajectory(episode, args.image_dir)
            if trajectory is None:
                skipped_empty_episodes[split] += 1
                continue
            output_handles[split].write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            counts[split] += 1
            action_counts.update(episode_actions)
    finally:
        for handle in output_handles.values():
            handle.close()

    stats = {
        "source": "android_control",
        "dataset_name": args.dataset_name,
        "split_strategy": "official_episode_split",
        "included_splits": target_splits,
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        "available_episode_counts": available_episode_counts or dict(counts),
        "episode_counts": dict(counts),
        "skipped_empty_action_episodes": dict(skipped_empty_episodes),
        "action_counts": dict(sorted(action_counts.items())),
        "source_image_dir": str(args.image_dir),
        "image_path_mode": "direct",
        "limit_episodes": args.limit_episodes,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote AndroidControl trajectories: {dict(counts)}")


if __name__ == "__main__":
    main()
