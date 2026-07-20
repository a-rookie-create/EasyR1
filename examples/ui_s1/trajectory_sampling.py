"""Deterministic trajectory-level sampling shared by UI-S1 data converters."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def validate_dataset_name(dataset_name: str) -> None:
    """Validate a dataset name before using it as an output filename prefix."""
    if (
        not dataset_name.strip()
        or dataset_name != dataset_name.strip()
        or dataset_name in {".", ".."}
        or "/" in dataset_name
        or "\\" in dataset_name
        or "\0" in dataset_name
    ):
        raise ValueError("--dataset-name must be a non-empty filename prefix without path separators")


def validate_sample_ratio(sample_ratio: float) -> None:
    """Validate the fraction of each split that should be retained."""
    if not 0 < sample_ratio <= 1:
        raise ValueError("--sample-ratio must be greater than 0 and at most 1")


def sample_trajectories(
    trajectories: Sequence[T], sample_ratio: float, seed: int, split: str
) -> list[T]:
    """Randomly retain ``sample_ratio`` complete items from one dataset split.

    Counts are rounded to the nearest integer, with at least one trajectory kept
    for every non-empty split when the ratio is positive.  A split-specific seed
    keeps one split's sample stable if another split changes size.
    """
    validate_sample_ratio(sample_ratio)
    items = list(trajectories)
    if sample_ratio == 1 or not items:
        return items

    sample_count = min(len(items), max(1, int(len(items) * sample_ratio + 0.5)))
    seed_bytes = hashlib.sha256(f"{seed}:{split}".encode("utf-8")).digest()[:8]
    split_seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    selected_indices = random.Random(split_seed).sample(range(len(items)), sample_count)
    return [items[index] for index in selected_indices]
