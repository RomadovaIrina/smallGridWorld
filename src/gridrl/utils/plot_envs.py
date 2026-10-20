"""Картинки карт для всех сетапов: outputs/envs/setup_<N>.png и общая outputs/envs/setups.png.

    uv run python -m gridrl.utils.plot_envs --seed 0
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridrl.envs import SETUPS, make_env, optimal_stats

CELL_PX = 24  # размер клетки в картинке env.render()


def draw_map(ax, env, draw_agent=True):
    """Рисует карту среды на осях matplotlib с номерами строк и столбцов.

    Номера — координаты внутренней области (строка, столбец), как в info["pos"].
    """
    ax.imshow(env._render_rgb(CELL_PX, draw_agent), interpolation="nearest")
    # imshow ставит центр пикселя i в координату i, поэтому центр клетки сдвинут на -0.5
    centers = [(i + 1) * CELL_PX + CELL_PX / 2 - 0.5 for i in range(max(env.h, env.w))]
    ax.set_xticks(centers[:env.w], [str(i) for i in range(env.w)], fontsize=8)
    ax.set_yticks(centers[:env.h], [str(i) for i in range(env.h)], fontsize=8)
    ax.set_xlabel("столбец")
    ax.set_ylabel("строка")
    ax.tick_params(length=0)


def to_pixels(pos):
    """Центр клетки (строка, столбец) в координатах картинки (x, y) — для рисования поверх карты."""
    r, c = pos
    return (c + 1) * CELL_PX + CELL_PX / 2 - 0.5, (r + 1) * CELL_PX + CELL_PX / 2 - 0.5


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

    fig, axes = plt.subplots(1, len(SETUPS), figsize=(5 * len(SETUPS), 5.8))
    for ax, setup_id in zip(axes, SETUPS):
        env = make_env(setup_id, seed=args.seed, render_mode="rgb_array")
        env.reset(seed=args.seed)

        single, single_ax = plt.subplots(figsize=(5, 5.8))
        for a in (ax, single_ax):
            draw_map(a, env)
            a.set_title(describe(setup_id, env), fontsize=10)
        single.tight_layout()
        single.savefig(out / f"setup_{setup_id}.png", dpi=120, bbox_inches="tight")
        plt.close(single)

    fig.suptitle("Зелёная клетка — цель, тёмно-серые — препятствия, рамка — стена, красный круг — агент")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "setups.png", dpi=120, bbox_inches="tight")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
