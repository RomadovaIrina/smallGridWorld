import colorsys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class GridWorldEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, floor_colors, pos_goal, n_colors=None, obstacle_mask=None, pos_agent=None,
                 see_obstacle=True, max_steps=None, obs_mode="color", render_mode=None):
        self.floor_colors = np.array(floor_colors)
        self.h, self.w = self.floor_colors.shape
        if obstacle_mask is None:
            obstacle_mask = np.zeros((self.h, self.w), dtype=bool)
        self.obstacle_mask = np.array(obstacle_mask, dtype=bool)
        self.pos_goal = tuple(pos_goal)
        self.n_colors = n_colors if n_colors is not None else int(self.floor_colors.max()) + 1
        self.see_obstacle = see_obstacle
        self.max_steps = max_steps if max_steps is not None else 4 * self.h * self.w
        self.obs_mode = obs_mode
        self.render_mode = render_mode

        # коды цветов: 0..n_colors-1 — пол, дальше цель, препятствие, стена
        self.goal_color = self.n_colors
        self.obstacle_color = self.n_colors + 1
        self.wall_color = self.n_colors + 2
        self.n_obs_colors = self.n_colors + 3

        # карта с рамкой стены: шаг за край просто попадает в клетку-стену
        self.color_map = np.full((self.h + 2, self.w + 2), self.wall_color)
        self.color_map[1:-1, 1:-1] = np.where(self.obstacle_mask, self.obstacle_color, self.floor_colors)
        self.color_map[self.pos_goal[0] + 1, self.pos_goal[1] + 1] = self.goal_color
        self.passable = np.zeros((self.h + 2, self.w + 2), dtype=bool)
        self.passable[1:-1, 1:-1] = ~self.obstacle_mask

        self.start_probs = self._start_distribution(pos_agent)
        obs_dim = self.n_obs_colors if obs_mode == "color" else self.h * self.w
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self._pos = (0, 0)  # позиция в координатах карты с рамкой
        self._color = 0
        self._t = 0

    def _start_distribution(self, pos_agent):
        """Вероятность стартовать из каждой клетки, вектор длины h*w."""
        allowed = ~self.obstacle_mask
        allowed[self.pos_goal] = False
        if pos_agent is None:
            weights = allowed.astype(float)
        elif np.shape(pos_agent) == (2,):
            weights = np.zeros((self.h, self.w))
            weights[tuple(pos_agent)] = 1.0
        else:
            weights = np.where(allowed, pos_agent, 0.0)
        return (weights / weights.sum()).ravel()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "pos" in options:  # старт вручную — для тестов
            r, c = options["pos"]
        else:
            r, c = divmod(int(self.np_random.choice(self.h * self.w, p=self.start_probs)), self.w)
        self._pos = (r + 1, c + 1)
        self._color = int(self.color_map[self._pos])
        self._t = 0
        return self._obs(), self._info(bumped=False)

    def step(self, action):
        dr, dc = ACTION_DELTAS[int(action)]
        target = (self._pos[0] + dr, self._pos[1] + dc)
        bumped = not self.passable[target]
        if not bumped:
            self._pos = target
        seen = target if bumped and self.see_obstacle else self._pos
        self._color = int(self.color_map[seen])
        self._t += 1

        terminated = self.agent_pos == self.pos_goal
        truncated = not terminated and self._t >= self.max_steps
        reward = 1.0 if terminated else 0.0
        return self._obs(), reward, terminated, truncated, self._info(bumped)

    @property
    def agent_pos(self):
        """Позиция (строка, столбец) без учёта рамки."""
        return self._pos[0] - 1, self._pos[1] - 1

    @property
    def inner_color_map(self):
        return self.color_map[1:-1, 1:-1]

    def _obs(self):
        if self.obs_mode == "color":
            obs = np.zeros(self.n_obs_colors, dtype=np.float32)
            obs[self._color] = 1.0
        else:
            obs = np.zeros(self.h * self.w, dtype=np.float32)
            r, c = self.agent_pos
            obs[r * self.w + c] = 1.0
        return obs

    def _info(self, bumped):
        return {"pos": np.array(self.agent_pos), "color": self._color, "bumped": bumped}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "ansi":
            return self._render_ansi()

    def _render_rgb(self, cell_px=24, draw_agent=True):
        hues = np.linspace(0.0, 1.0, self.n_colors, endpoint=False)
        floor_rgb = [colorsys.hsv_to_rgb(h, 0.35, 0.95) for h in hues]
        palette = np.concatenate([
            (np.array(floor_rgb) * 255).astype(np.uint8),
            np.array([(40, 200, 60), (60, 60, 60), (20, 20, 20)], dtype=np.uint8),  # цель, препятствие, стена
        ])
        img = palette[self.color_map]
        img = np.repeat(np.repeat(img, cell_px, axis=0), cell_px, axis=1)
        img[::cell_px, :] = img[::cell_px, :] * 0.8  # тонкая сетка
        img[:, ::cell_px] = img[:, ::cell_px] * 0.8
        if draw_agent:
            r, c = self._pos
            yy, xx = np.mgrid[:cell_px, :cell_px]
            center = (cell_px - 1) / 2
            disk = (yy - center) ** 2 + (xx - center) ** 2 <= (cell_px * 0.35) ** 2
            img[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px][disk] = (230, 30, 30)
        return img

    def _render_ansi(self):
        """'#' стена, 'X' препятствие, 'G' цель, 'A' агент, цифры/буквы — цвет пола."""
        symbols = "0123456789abcdefghijklmnopqrstuvwxyz"
        special = {self.wall_color: "#", self.obstacle_color: "X", self.goal_color: "G"}
        lines = []
        for r in range(self.h + 2):
            row = []
            for c in range(self.w + 2):
                code = int(self.color_map[r, c])
                if (r, c) == self._pos:
                    row.append("A")
                else:
                    row.append(special.get(code, symbols[code % len(symbols)]))
            lines.append(" ".join(row))
        return "\n".join(lines)
