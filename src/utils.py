"""Сохранение результатов и картинки: карта среды, тепловая карта посещений, GIF эпизодов."""
import csv
import json
import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

FLOOR_COLORS = plt.get_cmap("Pastel1").colors + plt.get_cmap("Pastel2").colors  # 17 цветов для пола


# ---------- сохранение результатов ----------
# Файлы пишутся сначала во временный файл, потом переименовываются: если процесс упадёт посреди записи,
# старый файл останется целым.

def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def start_run(cfg, algo, optimal):
    """Создаёт папку прогона <out_dir>/<run_name>/ и пишет config.json и status.json (running)."""
    run_dir = Path(cfg.get("out_dir", "runs")) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {"algo": algo, **cfg, "optimal": optimal})
    write_json(run_dir / "status.json", {"status": "running", "episodes_done": 0})
    (run_dir / "train.log").write_text("", encoding="utf-8")
    return run_dir


def log(run_dir, text):
    """Печатает строку и дописывает её в train.log прогона."""
    print(text)
    with open(Path(run_dir) / "train.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def save_progress(run_dir, history, status, weights=None, error=None, extra=None):
    """metrics.csv (одна строка на эпизод), status.json и, если переданы, веса model.pt."""
    if len(history) > 0:
        write_csv(Path(run_dir) / "metrics.csv", history)
    info = {"status": status, "episodes_done": len(history)}
    if len(history) > 0:
        info["env_steps"] = int(history[-1]["env_steps"])
        info["time_sec"] = round(history[-1]["time"], 1)
    if error is not None:
        info["error"] = error
    if extra is not None:
        info.update(extra)
    write_json(Path(run_dir) / "status.json", info)
    if weights is not None:
        tmp = Path(run_dir) / "model.pt.tmp"
        torch.save(weights, tmp)
        os.replace(tmp, Path(run_dir) / "model.pt")


def has_converged(history, window, tol):
    """Плато: средняя отдача за окно из window эпизодов выросла меньше чем на tol
    по сравнению с предыдущим окном — и так два раза подряд (смотрим три последних окна)."""
    if len(history) < 3 * window:
        return False
    returns = [row["return"] for row in history]
    first = np.mean(returns[-3 * window:-2 * window])
    second = np.mean(returns[-2 * window:-window])
    third = np.mean(returns[-window:])
    return second - first < tol and third - second < tol


def save_eval(run_dir, results):
    write_json(Path(run_dir) / "eval.json", results)


def eval_summary(episodes, gamma, optimal):
    """Сводка по eval-эпизодам: доля успехов, отдача, длина (всех и успешных), сравнение с оптимумом."""
    lengths = [ep["length"] for ep in episodes]
    success_lengths = [ep["length"] for ep in episodes if ep["success"]]
    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([ep["success"] for ep in episodes])),
        "failures": int(sum(1 for ep in episodes if not ep["success"])),
        "mean_return": float(np.mean([gamma ** (ep["length"] - 1) * ep["success"] for ep in episodes])),
        "mean_length": float(np.mean(lengths)),
        "mean_success_length": float(np.mean(success_lengths)) if success_lengths else None,
        "optimal_length": optimal["length"],
        "optimal_return": optimal["return"],
    }


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


def save_eval_pictures(env, episodes, folder, name, title):
    """Тепловая карта посещений (<name>_heatmap.png) и GIF до 5 коротких эпизодов (<name>.gif)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    image = plot_heatmap(ax, env, episodes)
    fig.colorbar(image, ax=ax, fraction=0.046, label="посещений за эпизод")
    ax.set_title(title, fontsize=9)
    fig.savefig(folder / f"{name}_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    short = [ep for ep in episodes if ep["length"] <= 60][:5]  # длинные эпизоды дают тысячи кадров
    if len(short) == 0:
        short = episodes[:1]
    save_gif(env, short, folder / f"{name}.gif", name)


def save_gif(env, episodes, path, title="", max_steps=None):
    """Анимация нескольких эпизодов подряд: агент, его путь и что он видит.
    max_steps — показать только первые max_steps шагов каждого эпизода (у необученного агента эпизоды длинные)."""
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
            if max_steps is not None:
                n = min(n, max_steps + 1)
            for t in list(range(n)) + [n - 1] * 3:  # в конце эпизода пауза в 3 кадра
                rows = [p[0] for p in ep["positions"][:t + 1]]
                cols = [p[1] for p in ep["positions"][:t + 1]]
                path_line.set_data(cols, rows)
                agent.set_data([cols[-1]], [rows[-1]])
                seen = color_names.get(ep["colors"][t], f"пол {ep['colors'][t]}")
                ending = ""
                if t == len(ep["positions"]) - 1:
                    ending = " — дошёл!" if ep["success"] else " — время вышло"
                elif t == n - 1:
                    ending = f" — показаны первые {n - 1} шагов"
                ax.set_title(f"{title}\nэпизод {k + 1}/{len(episodes)}, шаг {t}, видит: {seen}{ending}",
                             fontsize=9)
                writer.grab_frame()
    plt.close(fig)
    print(f"saved {path}")
