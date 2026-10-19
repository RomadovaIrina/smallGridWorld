"""Хелперы генерации карт и четыре тестовых сетапа из задания.

Допущения (в задании не указаны):
- цель — случайная свободная клетка, выбирается по сиду сетапа;
- старт — равномерно по всем свободным клеткам, кроме цели;
- лимит эпизода — 4 * h * w шагов;
- препятствия расставляются так, чтобы все свободные клетки были связны.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridrl.envs import grid_core as core
from gridrl.envs.gridworld import GridWorldEnv


def random_floor_coloring(h: int, w: int, n_colors: int, rng: np.random.Generator) -> np.ndarray:
    """Случайная покраска пола: каждая клетка — равновероятно один из n_colors цветов."""
    return rng.integers(0, n_colors, size=(h, w))


def unique_floor_coloring(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Каждая клетка своего цвета (n_colors = h * w) — полностью наблюдаемая среда."""
    return rng.permutation(h * w).reshape(h, w)


def random_obstacle_mask(
    h: int,
    w: int,
    fraction: float,
    rng: np.random.Generator,
    max_tries: int = 1000,
) -> np.ndarray:
    """Случайно ставит round(fraction * h * w) препятствий, сохраняя связность свободных клеток."""
    n_obstacles = int(round(fraction * h * w))
    for _ in range(max_tries):
        mask = np.zeros(h * w, dtype=bool)
        mask[rng.choice(h * w, size=n_obstacles, replace=False)] = True
        mask = mask.reshape(h, w)
        free = np.argwhere(~mask)
        dist = core.bfs_distances(~mask, tuple(free[0]))
        if (dist[~mask] >= 0).all():
            return mask
    raise RuntimeError("не удалось сгенерировать связную карту препятствий")


@dataclass(frozen=True)
class SetupSpec:
    h: int
    w: int
    n_colors: int | None  # None — уникальный цвет у каждой клетки
    obstacle_fraction: float


SETUPS: dict[int, SetupSpec] = {
    1: SetupSpec(5, 5, None, 0.0),  # MDP: 25 уникальных цветов
    2: SetupSpec(5, 5, 5, 0.0),
    3: SetupSpec(10, 10, 7, 0.1),
    4: SetupSpec(10, 10, 4, 0.1),
}


def build_setup(setup_id: int, seed: int = 0) -> dict:
    """Параметры инициализации GridWorldEnv для сетапа 1–4 (детерминированно по seed)."""
    spec = SETUPS[setup_id]
    rng = np.random.default_rng(seed)
    if spec.n_colors is None:
        floor = unique_floor_coloring(spec.h, spec.w, rng)
        n_colors = spec.h * spec.w
    else:
        floor = random_floor_coloring(spec.h, spec.w, spec.n_colors, rng)
        n_colors = spec.n_colors
    obstacles = random_obstacle_mask(spec.h, spec.w, spec.obstacle_fraction, rng)
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
    """Создаёт среду сетапа; overrides перекрывают параметры (see_obstacle, max_steps, ...)."""
    return GridWorldEnv(**{**build_setup(setup_id, seed), **overrides})


def optimal_stats(env: GridWorldEnv, gamma: float = 0.99) -> dict[str, float]:
    """Оракул с полным знанием позиции: ожидаемые длина эпизода и дисконтированная отдача.

    Для MDP (сетап 1) это достижимый оптимум, для POMDP — верхняя граница.
    """
    dist = core.bfs_distances(~env.obstacle_mask, env.pos_goal).ravel()
    p = env.start_probs
    support = p > 0
    d = dist[support].astype(np.float64)
    return {
        "length": float((p[support] * d).sum()),
        "return": float((p[support] * gamma ** (d - 1)).sum()),
    }
