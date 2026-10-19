"""Чистая numpy-логика клеточного мира, общая для обычной и векторизованной сред.

Соглашения:
- координаты (row, col), 0-индексация, (0, 0) — левый верхний угол внутренней области;
- внутри логика работает с картой, дополненной рамкой стены: (h + 2, w + 2),
  поэтому внутренние координаты сдвинуты на +1;
- цвета: 0..n_colors-1 — пол, далее цель, препятствие, стена.
"""

from __future__ import annotations

import numpy as np

# Действия пронумерованы по часовой стрелке, начиная с «вверх».
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
N_ACTIONS = 4
ACTION_DELTAS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)
ACTION_NAMES = ("up", "right", "down", "left")


def goal_color(n_colors: int) -> int:
    return n_colors


def obstacle_color(n_colors: int) -> int:
    return n_colors + 1


def wall_color(n_colors: int) -> int:
    return n_colors + 2


def n_obs_colors(n_colors: int) -> int:
    return n_colors + 3


def build_padded_maps(
    floor_colors: np.ndarray,
    obstacle_mask: np.ndarray,
    pos_goal: tuple[int, int],
    n_colors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Собирает карту цветов и маску проходимости с рамкой стены.

    Возвращает (color_map, passable), обе формы (h + 2, w + 2).
    """
    h, w = floor_colors.shape
    color_map = np.full((h + 2, w + 2), wall_color(n_colors), dtype=np.int64)
    inner = np.where(obstacle_mask, obstacle_color(n_colors), floor_colors)
    color_map[1:-1, 1:-1] = inner
    color_map[pos_goal[0] + 1, pos_goal[1] + 1] = goal_color(n_colors)

    passable = np.zeros((h + 2, w + 2), dtype=bool)
    passable[1:-1, 1:-1] = ~obstacle_mask
    return color_map, passable


def step_positions(
    passable: np.ndarray, pos: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Делает шаг для батча агентов в padded-координатах.

    pos: (N, 2), actions: (N,). Возвращает (new_pos, target, bumped), где
    target — клетка, в которую агент пытался шагнуть, а bumped — упёрся ли он
    в стену/препятствие (тогда new_pos == pos).
    """
    target = pos + ACTION_DELTAS[actions]
    bumped = ~passable[target[:, 0], target[:, 1]]
    new_pos = np.where(bumped[:, None], pos, target)
    return new_pos, target, bumped


def observed_colors(
    color_map: np.ndarray,
    new_pos: np.ndarray,
    target: np.ndarray,
    bumped: np.ndarray,
    see_obstacle: bool,
) -> np.ndarray:
    """Цвет, который видит агент после шага (с учётом флага see_obstacle)."""
    seen = np.where(bumped[:, None], target, new_pos) if see_obstacle else new_pos
    return color_map[seen[:, 0], seen[:, 1]]


def start_distribution(
    pos_agent,
    obstacle_mask: np.ndarray,
    pos_goal: tuple[int, int],
) -> np.ndarray:
    """Превращает параметр pos_agent в распределение стартов формы (h * w,).

    pos_agent:
      - None — равномерно по всем свободным клеткам, кроме цели;
      - пара (row, col) — фиксированный старт;
      - массив (h, w) неотрицательных весов — на него накладывается маска
        препятствий (и цели), затем L1-нормировка.
    """
    h, w = obstacle_mask.shape
    allowed = ~obstacle_mask
    allowed[pos_goal] = False

    if pos_agent is None:
        weights = allowed.astype(np.float64)
    else:
        arr = np.asarray(pos_agent)
        if arr.shape == (2,):
            r, c = int(arr[0]), int(arr[1])
            if not (0 <= r < h and 0 <= c < w):
                raise ValueError(f"pos_agent {(r, c)} вне области {(h, w)}")
            if not allowed[r, c]:
                raise ValueError(f"pos_agent {(r, c)} — препятствие или цель")
            weights = np.zeros((h, w))
            weights[r, c] = 1.0
        elif arr.shape == (h, w):
            if (arr < 0).any():
                raise ValueError("веса pos_agent должны быть неотрицательными")
            weights = np.where(allowed, arr.astype(np.float64), 0.0)
        else:
            raise ValueError(
                f"pos_agent: ожидалась пара координат или массив {(h, w)}, получено {arr.shape}"
            )

    total = weights.sum()
    if total <= 0:
        raise ValueError("pos_agent не оставляет ни одной допустимой стартовой клетки")
    return (weights / total).ravel()


def bfs_distances(passable_inner: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Кратчайшие расстояния (в шагах) от source до всех клеток; -1 — недостижима."""
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
