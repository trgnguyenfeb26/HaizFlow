"""Apple silicon GPU (Metal) support through MLX, without importing MLX.

On macOS the NVIDIA CUDA stack is unavailable; Whisper Turbo instead decodes
on the Apple GPU with mlx-whisper. Core and the desktop UI only need to know
whether that path exists, so this module checks for the package and the
architecture and never loads Metal itself.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from functools import lru_cache


def apple_gpu_asr_available() -> bool:
    """True on Apple silicon with mlx-whisper installed in this interpreter."""
    if os.environ.get("HAIZFLOW_DISABLE_APPLE_GPU") == "1":
        return False
    return _apple_gpu_asr_installed()


@lru_cache(maxsize=1)
def _apple_gpu_asr_installed() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        return importlib.util.find_spec("mlx_whisper") is not None
    except (ImportError, ValueError):
        return False
