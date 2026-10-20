"""Картинки карт для всех сетапов: outputs/envs/setup_<N>.png и общая outputs/envs/setups.png.

    uv run python scripts/plot_envs.py --seed 0
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridrl.envs import SETUPS, make_env, optimal_stats


def describe(setup_id, env):
    h, w, n_colors, fraction = SETUPS[setup_id]
    colors = "уникальные" if n_colors is None else n_colors
    opt = optimal_stats(env)
    return (
        f"Сетап {setup_id}: {h}x{w}, цветов пола {colors}, препятствий {int(env.obstacle_mask.sum())}\n"
        f"оптимальная длина пути {opt['length']:.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/envs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(SETUPS), figsize=(5 * len(SETUPS), 5.6))
    for ax, setup_id in zip(axes, SETUPS):
        env = make_env(setup_id, seed=args.seed, render_mode="rgb_array")
        env.reset(seed=args.seed)
        img = env.render()

        single, single_ax = plt.subplots(figsize=(5, 5.6))
        for a in (ax, single_ax):
            a.imshow(img, interpolation="nearest")
            a.set_title(describe(setup_id, env), fontsize=10)
            a.axis("off")
        single.tight_layout()
        single.savefig(out / f"setup_{setup_id}.png", dpi=120)
        plt.close(single)

    fig.suptitle("Зелёная клетка — цель, тёмно-серые — препятствия, рамка — стена, красный круг — агент")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "setups.png", dpi=120)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
