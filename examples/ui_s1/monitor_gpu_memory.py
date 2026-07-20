#!/usr/bin/env python3
"""Continuously write GPU memory high-water marks for one UI-S1 run.

The monitor uses nvidia-smi and has no Python package dependencies. It is safe
to run alongside training: the JSON output is atomically replaced after every
sample, so `cat output/<run>/gpu_memory_peak.json` always sees valid JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_nvidia_smi_rows(output: str) -> list[dict[str, int | None]]:
    """Parse `index,total,used,utilization` rows emitted with nounits CSV."""
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            continue
        try:
            index = int(values[0])
            total_mib = int(values[1])
            used_mib = int(values[2])
        except ValueError:
            continue
        try:
            utilization_percent: int | None = int(values[3])
        except ValueError:
            utilization_percent = None
        rows.append(
            {
                "index": index,
                "total_mib": total_mib,
                "used_mib": used_mib,
                "utilization_percent": utilization_percent,
            }
        )
    return rows


def query_gpu_memory(gpu_ids: str) -> list[dict[str, int | None]]:
    command = [
        "nvidia-smi",
        f"--id={gpu_ids}",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_nvidia_smi_rows(result.stdout)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = handle.name
    os.replace(temporary_path, path)
    # The monitor runs inside Docker, while users normally inspect this file
    # from the host through VS Code. Keep it host-readable after each atomic
    # replacement even when the container user has a different UID.
    os.chmod(path, 0o644)


def monitor(args: argparse.Namespace) -> None:
    output_path = Path(args.output).resolve()
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    state: dict[str, Any] = {
        "started_at": utc_now(),
        "updated_at": None,
        "finished_at": None,
        "gpu_ids": [int(gpu_id) for gpu_id in args.gpu_ids.split(",") if gpu_id.strip()],
        "sample_interval_seconds": args.interval_seconds,
        "sample_count": 0,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_memory_budget_mib_per_gpu": None,
        "vllm_memory_budget_note": (
            "This is the vLLM total per-GPU budget, not pure KV cache. "
            "Model weights and runtime buffers consume part of it."
        ),
        "gpus": {},
        "overall_peak": None,
        "last_error": None,
    }

    while not stop_event.is_set():
        timestamp = utc_now()
        try:
            rows = query_gpu_memory(args.gpu_ids)
            if not rows:
                raise RuntimeError("nvidia-smi returned no GPU rows")
            state["sample_count"] += 1
            state["updated_at"] = timestamp
            state["last_error"] = None
            for row in rows:
                gpu_key = str(row["index"])
                total_mib = int(row["total_mib"])
                used_mib = int(row["used_mib"])
                utilization_percent = row["utilization_percent"]
                if state["vllm_memory_budget_mib_per_gpu"] is None:
                    state["vllm_memory_budget_mib_per_gpu"] = int(
                        total_mib * args.vllm_gpu_memory_utilization
                    )
                gpu_state = state["gpus"].setdefault(
                    gpu_key,
                    {
                        "total_mib": total_mib,
                        "current_used_mib": 0,
                        "current_utilization_percent": None,
                        "peak_used_mib": 0,
                        "peak_used_percent": 0.0,
                        "peak_at": None,
                    },
                )
                gpu_state["current_used_mib"] = used_mib
                gpu_state["current_utilization_percent"] = utilization_percent
                if used_mib >= gpu_state["peak_used_mib"]:
                    gpu_state["peak_used_mib"] = used_mib
                    gpu_state["peak_used_percent"] = round(used_mib / total_mib * 100, 2)
                    gpu_state["peak_at"] = timestamp

                overall_peak = state["overall_peak"]
                if overall_peak is None or used_mib >= overall_peak["used_mib"]:
                    state["overall_peak"] = {
                        "gpu_id": int(row["index"]),
                        "used_mib": used_mib,
                        "used_percent": round(used_mib / total_mib * 100, 2),
                        "at": timestamp,
                    }
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            state["updated_at"] = timestamp
            state["last_error"] = str(error)

        atomic_write_json(output_path, state)
        stop_event.wait(args.interval_seconds)

    state["finished_at"] = utc_now()
    atomic_write_json(output_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated physical GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--output", required=True, help="Dynamically updated JSON output path")
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, required=True)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        parser.error("--vllm-gpu-memory-utilization must be in (0, 1]")
    monitor(args)


if __name__ == "__main__":
    main()
