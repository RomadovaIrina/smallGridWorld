from __future__ import annotations

import random

import numpy as np


def get_device(device: str = "auto"):
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


def parse_config(config_cls):
    """Аргументы командной строки из полей dataclass-конфига: --total-steps 1000, --no-memory."""
    import argparse
    from dataclasses import fields

    parser = argparse.ArgumentParser()
    for f in fields(config_cls):
        name = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(name, type=type(f.default), default=f.default)
    return config_cls(**vars(parser.parse_args()))
