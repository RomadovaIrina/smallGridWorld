"""Обучение PPO.

Примеры:
    uv run python scripts/train_ppo.py --setup 1 --no-memory
    uv run python scripts/train_ppo.py --setup 2 --total-steps 500000
"""

import argparse
from dataclasses import fields

from gridrl.agents.ppo import PPOConfig, train_ppo


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser()
    for f in fields(PPOConfig):
        name = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(name, type=type(f.default), default=f.default)
    return PPOConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    out_dir = train_ppo(parse_args())
    print(f"saved to {out_dir}")
