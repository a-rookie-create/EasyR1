"""Runtime fixes that must also apply in vLLM's spawned EngineCore process.

Python imports this module during interpreter startup whenever the EasyR1 root
is on ``PYTHONPATH``. The UI-S1 launcher ensures that this is true for its Ray
and vLLM child processes as well as the trainer parent process.
"""

import os


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _use_pageable_vllm_sleep_backup() -> bool:
    return _is_true(os.getenv("VLLM_ENABLE_SLEEP_MODE", "false")) and not _is_true(
        os.getenv("VLLM_SLEEP_PIN_MEMORY", "false")
    )


if _use_pageable_vllm_sleep_backup():
    try:
        import vllm.device_allocator.cumem as _cumem
    except ImportError:
        # Non-vLLM utilities in this repository must remain runnable.
        pass
    else:
        # CuMemAllocator.sleep imports this function into its own module.
        # Replacing that module-local reference limits the override to the
        # weight backup used by vLLM sleep, not to all vLLM CPU tensors.
        _cumem.is_pin_memory_available = lambda: False
        _cumem._easy_r1_pageable_sleep_backup = True
