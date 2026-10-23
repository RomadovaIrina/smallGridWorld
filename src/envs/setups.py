import numpy as np

from src.envs.gridworld import ACTION_DELTAS, GridWorldEnv
from src.envs.mnist_env import MnistGridWorldEnv


def random_floor_coloring(h, w, n_colors, rng):
    return rng.integers(0, n_colors, size=(h, w))


def unique_floor_coloring(h, w, rng):
    return rng.permutation(h * w).reshape(h, w)


def random_obstacle_mask(h, w, fraction, rng):
    """Случайно ставит препятствия так, чтобы все свободные клетки были связаны."""
    n_obstacles = round(fraction * h * w)
    while True:
        mask = np.zeros(h * w, dtype=bool)
        mask[rng.choice(h * w, size=n_obstacles, replace=False)] = True
        mask = mask.reshape(h, w)
        first_free = tuple(np.argwhere(~mask)[0])
        dist = bfs_distances(~mask, first_free)
        if (dist[~mask] >= 0).all():
            return mask


def make_env(h, w, n_colors, obstacle_fraction, seed=0, mnist=False, **kwargs):
    """Случайная карта по seed: покраска, препятствия, цель. n_colors=None — у каждой клетки свой цвет.
    mnist=True — среда, где цвет показывается картинкой цифры из MNIST.
    """
    rng = np.random.default_rng(seed)
    if n_colors is None:
        floor = unique_floor_coloring(h, w, rng)
        n_colors = h * w
    else:
        floor = random_floor_coloring(h, w, n_colors, rng)
    obstacles = random_obstacle_mask(h, w, obstacle_fraction, rng)
    free_cells = np.argwhere(~obstacles)
    goal = tuple(free_cells[rng.integers(len(free_cells))])
    if mnist:
        return MnistGridWorldEnv(floor, goal, n_colors=n_colors, obstacle_mask=obstacles, **kwargs)
    return GridWorldEnv(floor, goal, n_colors=n_colors, obstacle_mask=obstacles, **kwargs)


def bfs_distances(free, start):
    """Длина кратчайшего пути от start до каждой клетки (-1 — не дойти)."""
    h, w = free.shape
    dist = np.full((h, w), -1)
    dist[start] = 0
    queue = [start]
    while queue:
        r, c = queue.pop(0)
        for dr, dc in ACTION_DELTAS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))
    return dist


def optimal_stats(env, gamma=0.99):
    """Агент, который знает свои координаты и идёт кратчайшим путём:
    средняя длина пути и средняя отдача по всем стартам. Для POMDP — верхняя граница.
    """
    dist = bfs_distances(~env.obstacle_mask, env.pos_goal).ravel()
    p = env.start_probs
    starts = p > 0
    return {
        "length": float((p[starts] * dist[starts]).sum()),
        "return": float((p[starts] * gamma ** (dist[starts] - 1)).sum()),
    }
