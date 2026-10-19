"""GridWorld с частично наблюдаемыми «цветными» клетками (gymnasium.Env)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gridrl.envs import grid_core as core
from gridrl.envs.render import render_ansi, render_rgb

OBS_MODES = ("color", "coords")


class GridWorldEnv(gym.Env):
    """Лабиринт (h, w), окружённый стеной; цель агента — дойти до pos_goal.

    Параметры:
        floor_colors: (h, w) int — покраска пола значениями 0..n_colors-1
            (значения в клетках препятствий и цели игнорируются).
        n_colors: число цветов пола; по умолчанию max(floor_colors) + 1.
        obstacle_mask: (h, w) bool — внутренние препятствия; по умолчанию нет.
        pos_goal: (row, col) цели, фиксирована на всё обучение.
        pos_agent: None | (row, col) | (h, w) веса — распределение стартов,
            см. grid_core.start_distribution.
        see_obstacle: при попытке шагнуть в стену/препятствие агент видит её
            цвет (True) или цвет текущей клетки (False).
        max_steps: лимит шагов эпизода (truncation); по умолчанию 4 * h * w.
        obs_mode: "color" — one-hot цвета клетки (n_colors + 3);
            "coords" — one-hot позиции агента (h * w), отладочный режим (MDP).

    Награда: 1 при попадании в цель (эпизод завершается), иначе 0.
    В info всегда есть позиция агента ("pos") для отладки и визуализации.
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(
        self,
        floor_colors: np.ndarray,
        pos_goal: tuple[int, int],
        n_colors: int | None = None,
        obstacle_mask: np.ndarray | None = None,
        pos_agent=None,
        see_obstacle: bool = True,
        max_steps: int | None = None,
        obs_mode: str = "color",
        render_mode: str | None = None,
    ):
        floor_colors = np.asarray(floor_colors, dtype=np.int64)
        if floor_colors.ndim != 2:
            raise ValueError("floor_colors должен быть 2д массивом (h, w)")
        self.h, self.w = floor_colors.shape

        if obstacle_mask is None:
            obstacle_mask = np.zeros((self.h, self.w), dtype=bool)
        obstacle_mask = np.asarray(obstacle_mask, dtype=bool)
        if obstacle_mask.shape != (self.h, self.w):
            raise ValueError(f"obstacle_mask: ожидалась форма {(self.h, self.w)}")

        pos_goal = (int(pos_goal[0]), int(pos_goal[1]))
        if not (0 <= pos_goal[0] < self.h and 0 <= pos_goal[1] < self.w):
            raise ValueError(f"pos_goal {pos_goal} вне области")
        if obstacle_mask[pos_goal]:
            raise ValueError(f"pos_goal {pos_goal} совпадает с препятствием")

        free_floor = floor_colors[~obstacle_mask]
        if n_colors is None:
            n_colors = int(free_floor.max()) + 1
        if free_floor.min() < 0 or free_floor.max() >= n_colors:
            raise ValueError(f"цвета пола должны лежать в [0, {n_colors})")
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode должен быть одним из {OBS_MODES}")

        self.n_colors = int(n_colors)
        self.floor_colors = floor_colors
        self.obstacle_mask = obstacle_mask
        self.pos_goal = pos_goal
        self.see_obstacle = bool(see_obstacle)
        self.max_steps = int(max_steps) if max_steps is not None else 4 * self.h * self.w
        self.obs_mode = obs_mode
        self.render_mode = render_mode

        self.color_map, self.passable = core.build_padded_maps(
            floor_colors, obstacle_mask, pos_goal, self.n_colors
        )
        self.start_probs = core.start_distribution(pos_agent, obstacle_mask, pos_goal)

        self.n_obs_colors = core.n_obs_colors(self.n_colors)
        obs_dim = self.n_obs_colors if obs_mode == "color" else self.h * self.w
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(core.N_ACTIONS)

        # Состояние эпизода (padded-координаты).
        self._pos = np.zeros(2, dtype=np.int64)
        self._color = 0
        self._t = 0
        self._last_action: int | None = None

    # --- gymnasium API -------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "pos" in options:  # принудительный старт — для отладки и тестов
            r, c = options["pos"]
        else:
            idx = self.np_random.choice(self.h * self.w, p=self.start_probs)
            r, c = divmod(int(idx), self.w)
        self._pos = np.array([r + 1, c + 1], dtype=np.int64)
        self._color = int(self.color_map[r + 1, c + 1])
        self._t = 0
        self._last_action = None
        return self._obs(), self._info(bumped=False)

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"недопустимое действие {action}")
        new_pos, target, bumped = core.step_positions(
            self.passable, self._pos[None], np.array([action])
        )
        self._pos = new_pos[0]
        self._color = int(
            core.observed_colors(self.color_map, new_pos, target, bumped, self.see_obstacle)[0]
        )
        self._t += 1
        self._last_action = action

        terminated = (self._pos[0] - 1, self._pos[1] - 1) == self.pos_goal
        truncated = (not terminated) and self._t >= self.max_steps
        reward = 1.0 if terminated else 0.0
        return self._obs(), reward, terminated, truncated, self._info(bumped=bool(bumped[0]))

    def render(self):
        if self.render_mode == "rgb_array":
            return render_rgb(self, agent_pos=self.agent_pos)
        if self.render_mode == "ansi":
            return render_ansi(self, agent_pos=self.agent_pos)
        return None

    # --- helpers -------------------------------------------------------------

    @property
    def agent_pos(self) -> tuple[int, int]:
        """Позиция агента во внутренних координатах (row, col)."""
        return int(self._pos[0]) - 1, int(self._pos[1]) - 1

    @property
    def inner_color_map(self) -> np.ndarray:
        """Карта цветов внутренней области (h, w), включая цель и препятствия."""
        return self.color_map[1:-1, 1:-1]

    def _obs(self) -> np.ndarray:
        if self.obs_mode == "color":
            obs = np.zeros(self.n_obs_colors, dtype=np.float32)
            obs[self._color] = 1.0
        else:
            obs = np.zeros(self.h * self.w, dtype=np.float32)
            r, c = self.agent_pos
            obs[r * self.w + c] = 1.0
        return obs

    def _info(self, bumped: bool) -> dict[str, Any]:
        return {
            "pos": np.array(self.agent_pos, dtype=np.int64),
            "color": self._color,
            "bumped": bumped,
        }
