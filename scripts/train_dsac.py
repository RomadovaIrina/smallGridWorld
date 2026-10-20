"""Обучение Discrete SAC.

Примеры:
    uv run python scripts/train_dsac.py --setup 1 --no-memory
    uv run python scripts/train_dsac.py --setup 2 --no-soft-q
"""

from gridrl.agents.dsac import DSACConfig, train_dsac
from gridrl.utils import parse_config

if __name__ == "__main__":
    out_dir = train_dsac(parse_config(DSACConfig))
    print(f"saved to {out_dir}")
