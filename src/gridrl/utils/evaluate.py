"""Проверка обученного агента: eval-эпизоды, тепловая карта посещений и GIF одного эпизода.

    uv run python -m gridrl.utils.evaluate runs/<имя прогона> --episodes 100

Результат: outputs/eval/<имя>_heatmap.png и outputs/eval/<имя>_episode.gif.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.distributions import Categorical

from gridrl.envs import make_env, optimal_stats
from gridrl.models import dsac, ppo
from gridrl.utils.plot_envs import CELL_PX, draw_map, to_pixels


def load_policy(run_dir, config, obs_dim, n_actions):
    """Восстанавливает сеть из model.pt. Возвращает функцию шага: (obs, prev_a, starts, h) -> (логиты, h)."""
    weights = torch.load(Path(run_dir) / "model.pt", map_location="cpu")
    if config["algo"] == "ppo":
        model = ppo.make_model(obs_dim, n_actions, config["hidden"], config["memory"])
        model.load_state_dict(weights)
        return lambda obs, prev_a, starts, h: ppo.model_step(model, obs, prev_a, starts, h)[::2]
    actor = dsac.make_net(obs_dim, n_actions, config["hidden"], config["memory"])
    actor.load_state_dict(weights["actor"])
    return lambda obs, prev_a, starts, h: dsac.net_step(actor, obs, prev_a, starts, h)


def run_episode(env, policy, hidden, seed, greedy):
    """Один эпизод. Возвращает позиции (строка, столбец), увиденные цвета и дошёл ли агент до цели."""
    obs, info = env.reset(seed=seed)
    n_actions = env.action_space.n
    prev_a = torch.zeros(1, n_actions)
    starts = torch.ones(1)
    h = torch.zeros(1, hidden)
    positions = [tuple(info["pos"])]
    colors = [info["color"]]
    while True:
        with torch.no_grad():
            logits, h = policy(torch.as_tensor(obs)[None], prev_a, starts, h)
        action = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
        obs, reward, terminated, truncated, info = env.step(int(action))
        positions.append(tuple(info["pos"]))
        colors.append(info["color"])
        prev_a = F.one_hot(action, n_actions).float()
        starts = torch.zeros(1)
        if terminated or truncated:
            return positions, colors, terminated


def plot_heatmap(env, episodes, title, path):
    """Сколько раз в среднем за эпизод агент побывал в каждой клетке."""
    visits = np.zeros((env.h, env.w))
    for positions, _, _ in episodes:
        for r, c in positions:
            visits[r, c] += 1
    visits /= len(episodes)

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_map(ax, env, draw_agent=False)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # тепловая карта поверх внутренней области; где агент не был — видно саму карту
    extent = (CELL_PX - 0.5, (env.w + 1) * CELL_PX - 0.5, (env.h + 1) * CELL_PX - 0.5, CELL_PX - 0.5)
    # цель посещается ровно раз за успешный эпизод — не включаем её в шкалу
    masked = np.ma.masked_where(visits == 0, visits)
    masked[env.pos_goal] = np.ma.masked
    im = ax.imshow(masked, cmap="Blues", alpha=0.9, extent=extent, interpolation="nearest",
                   vmin=0, vmax=max(masked.max(), 1e-6))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for r in range(env.h):
        for c in range(env.w):
            x, y = to_pixels((r, c))
            if (r, c) == env.pos_goal:
                ax.text(x, y, "G", ha="center", va="center", fontsize=10, color="green", weight="bold")
            elif visits[r, c] >= 0.01 and env.h * env.w <= 100:
                ax.text(x, y, f"{visits[r, c]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="посещений за эпизод")
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def color_name(env, code):
    if code == env.goal_color:
        return "цель"
    if code == env.obstacle_color:
        return "препятствие"
    if code == env.wall_color:
        return "стена"
    return f"пол {code}"


def save_episodes_gif(env, episodes, title, path):
    """Анимация нескольких эпизодов подряд: агент (красный кружок), его путь и что он видит."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    draw_map(ax, env, draw_agent=False)
    (trail,) = ax.plot([], [], color="black", linewidth=2, alpha=0.6)
    (start,) = ax.plot([], [], "s", color="black", markersize=8, alpha=0.6)
    (agent,) = ax.plot([], [], "o", color="red", markersize=14)

    # кадр = (номер эпизода, шаг); после конца эпизода держим последний кадр
    frames = []
    for k, (positions, _, _) in enumerate(episodes):
        frames += [(k, t) for t in range(len(positions))] + [(k, len(positions) - 1)] * 3

    def frame(item):
        k, t = item
        positions, colors, success = episodes[k]
        xy = np.array([to_pixels(p) for p in positions])
        trail.set_data(xy[:t + 1, 0], xy[:t + 1, 1])
        start.set_data([xy[0, 0]], [xy[0, 1]])
        agent.set_data([xy[t, 0]], [xy[t, 1]])
        status = ""
        if t == len(positions) - 1:
            status = " — дошёл!" if success else " — время вышло"
        ax.set_title(
            f"{title}\nэпизод {k + 1}/{len(episodes)}, шаг {t}, видит: {color_name(env, colors[t])}{status}",
            fontsize=9,
        )
        return trail, start, agent

    anim = FuncAnimation(fig, frame, frames=frames)
    anim.save(path, writer=PillowWriter(fps=3))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--greedy", action="store_true", help="брать самое вероятное действие вместо сэмплирования")
    parser.add_argument("--out", default="outputs/eval")
    args = parser.parse_args()

    config = json.loads((Path(args.run) / "config.json").read_text())
    torch.manual_seed(0)
    env = make_env(
        config["setup"], seed=config["env_seed"], see_obstacle=config["see_obstacle"],
        obs_mode=config["obs_mode"], render_mode="rgb_array",
    )
    policy = load_policy(args.run, config, env.observation_space.shape[0], env.action_space.n)

    # сиды eval-эпизодов не пересекаются с обучением
    episodes = [
        run_episode(env, policy, config["hidden"], seed=10_000 + i, greedy=args.greedy)
        for i in range(args.episodes)
    ]
    success = np.mean([done for _, _, done in episodes])
    length = np.mean([len(p) - 1 for p, _, _ in episodes])
    optimal = optimal_stats(env, config["gamma"])["length"]

    name = Path(args.run).name
    mode = "жадно" if args.greedy else "сэмплирование"
    summary = (
        f"{name} ({mode}): success {success:.2f}, длина {length:.1f} (оптимум {optimal:.1f}), "
        f"{args.episodes} эпизодов"
    )
    print(summary)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_heatmap(env, episodes, summary, out / f"{name}_heatmap.png")
    # в GIF — первые эпизоды подряд, пока не наберётся ~60 шагов (но не больше 6 эпизодов)
    shown, total = [], 0
    for episode in episodes[:6]:
        if shown and total + len(episode[0]) > 60:
            break
        shown.append(episode)
        total += len(episode[0])
    save_episodes_gif(env, shown, name, out / f"{name}_episodes.gif")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
