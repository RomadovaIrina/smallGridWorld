import colorsys

import gymnasium as gym
import numpy as np
from gymnasium import spaces


UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OBS_MODES = ("color", "coords")


class GridWorldEnv(gym.Env):
    """GridWorld с one-hot наблюдением цвета текущей клетки.

    floor_colors и obstacle_mask имеют форму (h, w), координаты — (row, col).
    pos_agent задаёт фиксированный старт или веса стартовых позиций.
    При see_obstacle=True столкновение даёт наблюдение цвета преграды.
    obs_mode="coords" включает one-hot позиции для отладки.
    Лимит эпизода по умолчанию — 4 * h * w шагов.
    Награда за достижение цели — 1, за остальные шаги — 0.
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(
        self,
        floor_colors,
        pos_goal,
        n_colors=None,
        obstacle_mask=None,
        pos_agent=None,
        see_obstacle=True,
        max_steps=None,
        obs_mode="color",
        render_mode=None,
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

        self.goal_color = self.n_colors
        self.obstacle_color = self.n_colors + 1
        self.wall_color = self.n_colors + 2
        self.n_obs_colors = self.n_colors + 3

        # Рамка стены позволяет проверять соседнюю клетку без выхода за массив.
        self.color_map = np.full((self.h + 2, self.w + 2), self.wall_color, dtype=np.int64)
        self.color_map[1:-1, 1:-1] = np.where(obstacle_mask, self.obstacle_color, floor_colors)
        self.color_map[pos_goal[0] + 1, pos_goal[1] + 1] = self.goal_color
        self.passable = np.zeros((self.h + 2, self.w + 2), dtype=bool)
        self.passable[1:-1, 1:-1] = ~obstacle_mask
        self.start_probs = self._start_distribution(pos_agent)
        obs_dim = self.n_obs_colors if obs_mode == "color" else self.h * self.w
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self._pos = (0, 0)
        self._color = 0
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "pos" in options:
            r, c = options["pos"]
        else:
            idx = self.np_random.choice(self.h * self.w, p=self.start_probs)
            r, c = divmod(int(idx), self.w)
        self._pos = (int(r) + 1, int(c) + 1)
        self._color = int(self.color_map[r + 1, c + 1])
        self._t = 0
        return self._obs(), self._info(bumped=False)

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"недопустимое действие {action}")
        r, c = self._pos
        dr, dc = ACTION_DELTAS[action]
        target = (r + dr, c + dc)
        bumped = not self.passable[target]
        if not bumped:
            self._pos = target

        seen = target if bumped and self.see_obstacle else self._pos
        self._color = int(self.color_map[seen])
        self._t += 1

        terminated = (self._pos[0] - 1, self._pos[1] - 1) == self.pos_goal
        truncated = (not terminated) and self._t >= self.max_steps
        reward = 1.0 if terminated else 0.0
        return self._obs(), reward, terminated, truncated, self._info(bumped=bumped)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    @property
    def agent_pos(self):
        """Позиция (row, col) без учёта рамки стены."""
        return int(self._pos[0]) - 1, int(self._pos[1]) - 1

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
        return {
            "pos": np.array(self.agent_pos, dtype=np.int64),
            "color": self._color,
            "bumped": bumped,
        }

    def _start_distribution(self, pos_agent):
        h, w = self.h, self.w
        allowed = ~self.obstacle_mask
        allowed[self.pos_goal] = False

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

    def _render_rgb(self, cell_px=24, draw_agent=True):
        hues = np.linspace(0.0, 1.0, self.n_colors, endpoint=False)
        floor_rgb = [colorsys.hsv_to_rgb(h, 0.35, 0.95) for h in hues]
        palette = np.concatenate([
            (np.array(floor_rgb) * 255).astype(np.uint8),
            np.array([(40, 200, 60), (60, 60, 60), (20, 20, 20)], dtype=np.uint8),
        ])
        img = palette[self.color_map]
        img = np.repeat(np.repeat(img, cell_px, axis=0), cell_px, axis=1)
        img[::cell_px, :] = img[::cell_px, :] * 0.8
        img[:, ::cell_px] = img[:, ::cell_px] * 0.8

        if not draw_agent:
            return img
        r, c = self._pos
        yy, xx = np.mgrid[:cell_px, :cell_px]
        center = (cell_px - 1) / 2
        disk = (yy - center) ** 2 + (xx - center) ** 2 <= (cell_px * 0.35) ** 2
        cell = img[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px]
        cell[disk] = (230, 30, 30)
        return img

    def _render_ansi(self):
        symbols = "0123456789abcdefghijklmnopqrstuvwxyz"
        lines = []
        for r in range(self.h + 2):
            row = []
            for c in range(self.w + 2):
                code = int(self.color_map[r, c])
                if (r, c) == self._pos:
                    ch = "A"
                elif code == self.wall_color:
                    ch = "#"
                elif code == self.obstacle_color:
                    ch = "X"
                elif code == self.goal_color:
                    ch = "G"
                else:
                    ch = symbols[code] if code < len(symbols) else "?"
                row.append(ch)
            lines.append(" ".join(row))
        return "\n".join(lines)
