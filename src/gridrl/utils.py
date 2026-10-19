"""Общие утилиты: выбор устройства и сиды (чтобы код одинаково работал на CPU и GPU)."""

from __future__ import annotations

import random

import numpy as np


def get_device(device: str = "auto"):
    """'auto' — CUDA, если доступна, иначе CPU; иначе — явно заданное устройство."""
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
