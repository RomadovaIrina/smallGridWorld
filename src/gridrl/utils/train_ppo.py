"""Обучение PPO.

Примеры:
    uv run python -m gridrl.utils.train_ppo --setup 1 --no-memory
    uv run python -m gridrl.utils.train_ppo --setup 2 --total-steps 500000
"""

from gridrl.models.ppo import PPOConfig, train_ppo
from gridrl.utils import parse_config

if __name__ == "__main__":
    out_dir = train_ppo(parse_config(PPOConfig))
    print(f"saved to {out_dir}")
