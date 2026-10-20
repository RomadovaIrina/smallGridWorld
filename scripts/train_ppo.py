"""Обучение PPO.

Примеры:
    uv run python scripts/train_ppo.py --setup 1 --no-memory
    uv run python scripts/train_ppo.py --setup 2 --total-steps 500000
"""

from gridrl.agents.ppo import PPOConfig, train_ppo
from gridrl.utils import parse_config

if __name__ == "__main__":
    out_dir = train_ppo(parse_config(PPOConfig))
    print(f"saved to {out_dir}")
