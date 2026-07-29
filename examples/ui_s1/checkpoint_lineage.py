#!/usr/bin/env python3
"""Prepare UI-S1 run lineage before Ray or any training writer starts.

This module deliberately contains no Torch/Ray imports.  The shell runner uses
it to make destructive rollback operations deterministic and testable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable


STEP_DIR_RE = re.compile(r"global_step_([1-9][0-9]*)$")
PROGRESS_STEP_RE = re.compile(r"\| STEP ([0-9]+) \|")


def checkpoint_step(path: Path) -> int:
    match = STEP_DIR_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"checkpoint must end with global_step_<positive integer>: {path}")
    return int(match.group(1))


def require_warm_start_checkpoint(path: Path, require_dataloader: bool) -> int:
    if not path.is_dir():
        raise ValueError(f"warm-start checkpoint directory does not exist: {path}")
    step = checkpoint_step(path)
    adapter = path / "actor" / "lora_adapter"
    if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
        raise ValueError(f"warm-start LoRA adapter is incomplete: {adapter}")
    if require_dataloader and not (path / "dataloader.pt").is_file():
        raise ValueError(f"warm-start dataloader state is missing: {path / 'dataloader.pt'}")
    return step


def _atomic_replace(path: Path, write: Callable[[Any], None]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        write(handle)
    temp.replace(path)


def truncate_jsonl(path: Path, step_key: str, max_step: int) -> tuple[int, int]:
    """Keep valid JSONL records whose integer step key is at most max_step.

    Return ``(records_before, records_after)`` so resume preflight can report
    exactly what it changed.
    """
    if not path.is_file():
        return 0, 0

    records_before = 0
    records_after = 0
    def write(handle: Any) -> None:
        nonlocal records_before, records_after
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL in {path}:{line_number}: {exc.msg}") from exc
                step = record.get(step_key)
                if isinstance(step, bool) or not isinstance(step, int):
                    raise ValueError(f"missing integer {step_key!r} in {path}:{line_number}")
                records_before += 1
                if step <= max_step:
                    handle.write(line if line.endswith("\n") else line + "\n")
                    records_after += 1

    _atomic_replace(path, write)
    return records_before, records_after


def truncate_progress_log(path: Path, max_step: int) -> tuple[int, int]:
    """Keep the chronological prefix ending when ``max_step`` was completed.

    Filtering lines independently by their numeric step is unsafe: a later
    failed resume appends unscoped setup messages and starts again at STEP 0.
    Those lines are chronologically newer even though their step number is
    smaller.  A checkpoint represents one point in the append-only progress
    stream, so preserve only the prefix through that point.
    """
    if not path.is_file():
        return 0, 0

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    step_end_marker = f"| STEP {max_step} | STEP | END"
    checkpoint_end_marker = f"| STEP {max_step} | CHECKPOINT_SAVE | END"

    def last_line_containing(marker: str) -> int | None:
        matches = [index for index, line in enumerate(lines) if marker in line]
        return matches[-1] if matches else None

    boundary = last_line_containing(step_end_marker)
    if boundary is None:
        # A time-based checkpoint can be durable immediately before the
        # enclosing STEP END record is flushed.
        boundary = last_line_containing(checkpoint_end_marker)
    if boundary is None:
        # Support older progress formats while still selecting the checkpoint's
        # exact step rather than a later resume's restarted STEP 0.
        exact_step_lines = [
            index
            for index, line in enumerate(lines)
            if (match := PROGRESS_STEP_RE.search(line)) is not None and int(match.group(1)) == max_step
        ]
        boundary = exact_step_lines[-1] if exact_step_lines else None
    if boundary is None:
        raise ValueError(f"cannot locate STEP {max_step} boundary in progress log: {path}")

    def write(handle: Any) -> None:
        handle.writelines(lines[: boundary + 1])

    _atomic_replace(path, write)
    return len(lines), boundary + 1


def _copy_truncated_jsonl(source: Path, destination: Path, step_key: str, max_step: int) -> None:
    if not source.is_file():
        return
    shutil.copy2(source, destination)
    truncate_jsonl(destination, step_key, max_step)


def _copy_truncated_progress(source: Path, destination: Path, max_step: int) -> None:
    if not source.is_file():
        return
    shutil.copy2(source, destination)
    truncate_progress_log(destination, max_step)


def _source_config(source_run_dir: Path) -> dict[str, Any] | None:
    config_path = source_run_dir / "experiment_config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid source experiment config: {config_path}: {exc.msg}") from exc


def _validate_inherited_dataloader(source_config: dict[str, Any] | None, expected_data: dict[str, Any] | None) -> None:
    """Validate only data properties required by StatefulDataLoader restoration."""
    if expected_data is None:
        return
    if source_config is None:
        raise ValueError("cannot validate inherited dataloader: source experiment_config.json is missing")
    source_data = source_config.get("data", {})
    keys = ("train_files", "shuffle", "seed", "rollout_batch_size", "mini_rollout_batch_size")
    mismatches = {
        key: (source_data.get(key), expected_data.get(key))
        for key in keys
        if source_data.get(key) != expected_data.get(key)
    }
    if mismatches:
        raise ValueError(
            "WARM_START_DATALOADER=inherit requires compatible data loader settings; "
            f"mismatches={mismatches}. Use WARM_START_DATALOADER=reset to restart data order."
        )


def prepare_warm_start(source_checkpoint: Path, run_dir: Path, dataloader_mode: str, expected_data: dict[str, Any] | None) -> None:
    if dataloader_mode not in {"inherit", "reset"}:
        raise ValueError("WARM_START_DATALOADER must be 'inherit' or 'reset'")
    if run_dir.exists():
        raise ValueError(f"warm-start RUN_NAME already exists: {run_dir}")
    source_checkpoint = source_checkpoint.resolve()
    step = require_warm_start_checkpoint(source_checkpoint, require_dataloader=dataloader_mode == "inherit")
    source_run_dir = source_checkpoint.parent
    source_config = _source_config(source_run_dir)
    if dataloader_mode == "inherit":
        _validate_inherited_dataloader(source_config, expected_data)

    run_dir.mkdir(parents=True)
    _copy_truncated_progress(source_run_dir / "training_progress.log", run_dir / "training_progress.log", step)
    _copy_truncated_jsonl(source_run_dir / "experiment_log.jsonl", run_dir / "experiment_log.jsonl", "step", step)
    _copy_truncated_jsonl(
        source_run_dir / "semi_online_rollouts.jsonl", run_dir / "semi_online_rollouts.jsonl", "global_step", step
    )

    lineage_dir = run_dir / "warm_start"
    lineage_dir.mkdir()
    for filename in ("experiment_config.json", "experiment_config.resume.json"):
        source = source_run_dir / filename
        if source.is_file():
            shutil.copy2(source, lineage_dir / f"source_{filename}")
    if dataloader_mode == "inherit":
        shutil.copy2(source_checkpoint / "dataloader.pt", lineage_dir / "source_dataloader.pt")

    manifest = {
        "mode": "warm_start",
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": str(source_checkpoint),
        "source_global_step": step,
        "dataloader_mode": dataloader_mode,
        "history_files": ["training_progress.log", "experiment_log.jsonl", "semi_online_rollouts.jsonl"],
    }
    (run_dir / "warm_start.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _remove_future_checkpoints(run_dir: Path, max_step: int) -> list[str]:
    removed = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = STEP_DIR_RE.fullmatch(child.name)
        if match is not None and int(match.group(1)) > max_step:
            shutil.rmtree(child)
            removed.append(child.name)
    return sorted(removed)


def prepare_resume_rollback(run_dir: Path, checkpoint: Path) -> None:
    run_dir = run_dir.resolve()
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != run_dir:
        raise ValueError(f"resume checkpoint must be directly under RUN_DIR: {checkpoint}")
    step = checkpoint_step(checkpoint)
    if not checkpoint.is_dir():
        raise ValueError(f"resume checkpoint directory does not exist: {checkpoint}")
    history_records = {
        "training_progress.log": truncate_progress_log(run_dir / "training_progress.log", step),
        "experiment_log.jsonl": truncate_jsonl(run_dir / "experiment_log.jsonl", "step", step),
        "semi_online_rollouts.jsonl": truncate_jsonl(
            run_dir / "semi_online_rollouts.jsonl", "global_step", step
        ),
    }
    removed = _remove_future_checkpoints(run_dir, step)

    tracker_path = run_dir / "checkpoint_tracker.json"
    tracker: dict[str, Any] = {}
    if tracker_path.is_file():
        try:
            tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid checkpoint tracker: {tracker_path}: {exc.msg}") from exc
    if tracker.get("best_global_step", 0) > step:
        tracker["best_global_step"] = 0
        tracker["best_val_reward_score"] = 0.0
    tracker.update(
        {
            "last_global_step": step,
            "last_actor_path": str((checkpoint / "actor").resolve()),
        }
    )
    tracker_path.write_text(json.dumps(tracker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    archive_dir = run_dir / "rollback_archive" / f"before_global_step_{step}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("train.log", "generations.log", "gpu_memory_peak.json"):
        source = run_dir / filename
        if source.is_file():
            shutil.move(str(source), str(archive_dir / filename))
    for source in run_dir.glob("rewards*.png"):
        if source.is_file():
            shutil.move(str(source), str(archive_dir / source.name))
    manifest = {
        "rollback_to_global_step": step,
        "selected_checkpoint": str(checkpoint),
        "history_records": {
            filename: {"before": before, "after": after, "removed": before - after}
            for filename, (before, after) in history_records.items()
        },
        "removed_checkpoints": removed,
    }
    (archive_dir / "rollback.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = ", ".join(
        f"{filename}: {before}->{after}"
        for filename, (before, after) in history_records.items()
    )
    print(
        f"Resume rollback prepared at global_step_{step}; {summary}; "
        f"removed_checkpoints={removed or 'none'}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    warm = subparsers.add_parser("prepare-warm-start")
    warm.add_argument("--source-checkpoint", type=Path, required=True)
    warm.add_argument("--run-dir", type=Path, required=True)
    warm.add_argument("--dataloader-mode", required=True, choices=("inherit", "reset"))
    warm.add_argument("--expected-data-json")
    resume = subparsers.add_parser("prepare-resume-rollback")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        if args.command == "prepare-warm-start":
            expected_data = json.loads(args.expected_data_json) if args.expected_data_json else None
            if expected_data is not None and not isinstance(expected_data, dict):
                raise ValueError("--expected-data-json must encode an object")
            prepare_warm_start(args.source_checkpoint, args.run_dir, args.dataloader_mode, expected_data)
        else:
            prepare_resume_rollback(args.run_dir, args.checkpoint)
    except ValueError as exc:
        print(f"checkpoint lineage error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
