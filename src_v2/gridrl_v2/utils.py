"""Сохранение результатов и картинки: карта среды, тепловая карта посещений, GIF эпизодов."""
import csv
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

FLOOR_COLORS = plt.get_cmap("Pastel1").colors + plt.get_cmap("Pastel2").colors  # 17 цветов для пола


def save_run(cfg, algo, optimal, history, weights):
    """Сохраняет прогон в runs/<run_name>/: config.json, metrics.csv, model.pt"""
    run_dir = Path("runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"algo": algo, **cfg, "optimal": optimal}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save(weights, run_dir / "model.pt")
    print(f"saved to {run_dir}")


def cell_rgb(env, color):
    if color == env.goal_color:
        return (0.3, 0.8, 0.3)
    if color == env.obstacle_color or color == env.wall_color:
        return (0.25, 0.25, 0.25)
    return FLOOR_COLORS[color % len(FLOOR_COLORS)]


def draw_env(ax, env, show_agent=True, show_numbers=True):
    """Карта: клетки-прямоугольники, в клетке номер её цвета, G — цель, красный круг — агент.
    По осям — номера строк и столбцов (как в info["pos"]).
    """
    for r in range(-1, env.h + 1):  # -1 и h — рамка стены
        for c in range(-1, env.w + 1):
            color = env.cell_color(r, c)
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=cell_rgb(env, color),
                                     edgecolor="gray", linewidth=0.5)
            ax.add_patch(rect)
            if color == env.goal_color:
                ax.text(c, r, "G", ha="center", va="center", fontsize=11, weight="bold", color="white")
            elif show_numbers and color < env.n_colors and env.w <= 10:
                ax.text(c, r, str(color), ha="center", va="center", fontsize=7, color="gray")
    if show_agent:
        r, c = env.agent_pos
        ax.add_patch(patches.Circle((c, r), 0.3, color="red"))
    ax.set_xlim(-1.5, env.w + 0.5)
    ax.set_ylim(env.h + 0.5, -1.5)  # строка 0 сверху
    ax.set_aspect("equal")
    ax.set_xticks(range(env.w))
    ax.set_yticks(range(env.h))
    ax.tick_params(length=0, labelsize=8)
    ax.set_xlabel("столбец")
    ax.set_ylabel("строка")


def plot_heatmap(ax, env, episodes):
    """Сколько раз в среднем за эпизод агент побывал в каждой клетке."""
    visits = np.zeros((env.h, env.w))
    for ep in episodes:
        for r, c in ep["positions"]:
            visits[r, c] += 1
    visits = visits / len(episodes)
    visits[env.pos_goal] = 0  # в цель агент попадает ровно раз — не мешаем шкале

    draw_env(ax, env, show_agent=False, show_numbers=False)
    shown = np.ma.masked_where(visits == 0, visits)  # где агент не был — видна карта
    image = ax.imshow(shown, cmap="Blues", vmin=0, extent=(-0.5, env.w - 0.5, env.h - 0.5, -0.5), zorder=2)
    for r in range(env.h):
        for c in range(env.w):
            if visits[r, c] >= 0.01 and env.w <= 10:
                ax.text(c, r, f"{visits[r, c]:.2f}", ha="center", va="center", fontsize=6, zorder=3)
    return image


def save_gif(env, episodes, path, title=""):
    """Анимация нескольких эпизодов подряд: агент, его путь и что он видит."""
    color_names = {env.goal_color: "цель", env.obstacle_color: "препятствие", env.wall_color: "стена"}
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    draw_env(ax, env, show_agent=False)
    path_line, = ax.plot([], [], color="black", linewidth=2, alpha=0.6, zorder=3)
    agent, = ax.plot([], [], "o", color="red", markersize=14, zorder=4)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=3)
    with writer.saving(fig, path, dpi=100):
        for k, ep in enumerate(episodes):
            n = len(ep["positions"])
            for t in list(range(n)) + [n - 1] * 3:  # в конце эпизода пауза в 3 кадра
                rows = [p[0] for p in ep["positions"][:t + 1]]
                cols = [p[1] for p in ep["positions"][:t + 1]]
                path_line.set_data(cols, rows)
                agent.set_data([cols[-1]], [rows[-1]])
                seen = color_names.get(ep["colors"][t], f"пол {ep['colors'][t]}")
                ending = ""
                if t == n - 1:
                    ending = " — дошёл!" if ep["success"] else " — время вышло"
                ax.set_title(f"{title}\nэпизод {k + 1}/{len(episodes)}, шаг {t}, видит: {seen}{ending}",
                             fontsize=9)
                writer.grab_frame()
    plt.close(fig)
    print(f"saved {path}")
