#!/usr/bin/env python3
"""Create a small, independent JSONL subset for an end-to-end smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path


def copy_prefix(source: Path, destination: Path, count: int) -> int:
    written = 0
    with source.open("r", encoding="utf-8") as reader, destination.open("x", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            writer.write(line)
            written += 1
            if written == count:
                break
    if written != count:
        raise ValueError(f"{source} contains only {written} non-empty records; expected {count}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-split", type=int, default=4)
    args = parser.parse_args()

    if args.episodes_per_split < 1:
        raise ValueError("--episodes-per-split must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        source = args.source_dir / f"ui_s1_android_control_rl_{split}.jsonl"
        destination = args.output_dir / source.name
        count = copy_prefix(source, destination, args.episodes_per_split)
        print(f"{split}: {count} records -> {destination}")


if __name__ == "__main__":
    main()
