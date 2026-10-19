from __future__ import annotations

import numpy as np

from gridrl.envs.gridworld import ACTION_DELTAS, GridWorldEnv


def random_floor_coloring(h: int, w: int, n_colors: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_colors, size=(h, w))


def unique_floor_coloring(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(h * w).reshape(h, w)


def random_obstacle_mask(
    h: int,
    w: int,
    fraction: float,
    rng: np.random.Generator,
    max_tries: int = 1000,
) -> np.ndarray:
    """Расставляет препятствия, сохраняя связность свободных клеток."""
    n_obstacles = int(round(fraction * h * w))
    for _ in range(max_tries):
        mask = np.zeros(h * w, dtype=bool)
        mask[rng.choice(h * w, size=n_obstacles, replace=False)] = True
        mask = mask.reshape(h, w)
        free = np.argwhere(~mask)
        dist = bfs_distances(~mask, tuple(free[0]))
        if (dist[~mask] >= 0).all():
            return mask
    raise RuntimeError("не удалось сгенерировать связную карту препятствий")


# Высота, ширина, число цветов пола (None — уникальные), доля препятствий.
SETUPS = {
    1: (5, 5, None, 0.0),
    2: (5, 5, 5, 0.0),
    3: (10, 10, 7, 0.1),
    4: (10, 10, 4, 0.1),
}


def build_setup(setup_id: int, seed: int = 0) -> dict:
    h, w, n_colors, obstacle_fraction = SETUPS[setup_id]
    rng = np.random.default_rng(seed)
    if n_colors is None:
        floor = unique_floor_coloring(h, w, rng)
        n_colors = h * w
    else:
        floor = random_floor_coloring(h, w, n_colors, rng)
    obstacles = random_obstacle_mask(h, w, obstacle_fraction, rng)
    free = np.argwhere(~obstacles)
    goal = tuple(int(x) for x in free[rng.integers(len(free))])
    return dict(
        floor_colors=floor,
        n_colors=n_colors,
        obstacle_mask=obstacles,
        pos_goal=goal,
        pos_agent=None,
    )


def make_env(setup_id: int, seed: int = 0, **overrides) -> GridWorldEnv:
    return GridWorldEnv(**{**build_setup(setup_id, seed), **overrides})


def optimal_stats(env: GridWorldEnv, gamma: float = 0.99) -> dict[str, float]:
    """Оптимум при известной позиции: средняя длина пути и отдача.
    Для частично наблюдаемой среды отдача служит верхней границей.
    """
    dist = bfs_distances(~env.obstacle_mask, env.pos_goal).ravel()
    p = env.start_probs
    support = p > 0
    d = dist[support].astype(np.float64)
    return {
        "length": float((p[support] * d).sum()),
        "return": float((p[support] * gamma ** (d - 1)).sum()),
    }


def bfs_distances(passable_inner: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Кратчайшие расстояния по свободным клеткам; -1 для недостижимых."""
    h, w = passable_inner.shape
    dist = np.full((h, w), -1, dtype=np.int64)
    dist[source] = 0
    frontier = [source]
    while frontier:
        nxt = []
        for r, c in frontier:
            for dr, dc in ACTION_DELTAS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and passable_inner[rr, cc] and dist[rr, cc] < 0:
                    dist[rr, cc] = dist[r, c] + 1
                    nxt.append((rr, cc))
        frontier = nxt
    return dist
