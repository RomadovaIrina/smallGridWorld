import gymnasium as gym
import numpy as np
from gymnasium import spaces

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class GridWorldEnv(gym.Env):
    """Лабиринт h x w. Агент видит только цвет своей клетки (one-hot) и должен дойти до цели.

    Цвета: 0..n_colors-1 — пол, n_colors — цель, n_colors+1 — препятствие, n_colors+2 — стена.
    Награда 1 за цель, иначе 0. Картинки карты рисуются в utils.py.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, floor_colors, pos_goal, n_colors=None, obstacle_mask=None, pos_agent=None,
                 see_obstacle=True, max_steps=None, obs_mode="color", render_mode=None):
        self.floor_colors = np.array(floor_colors)
        if self.floor_colors.ndim != 2 or not np.issubdtype(self.floor_colors.dtype, np.integer):
            raise ValueError("floor_colors должен быть двумерным массивом целых чисел")
        self.h, self.w = self.floor_colors.shape
        if obstacle_mask is None:
            obstacle_mask = np.zeros((self.h, self.w), dtype=bool)
        self.obstacle_mask = np.array(obstacle_mask, dtype=bool)
        if self.obstacle_mask.shape != (self.h, self.w):
            raise ValueError("obstacle_mask должен быть той же формы, что floor_colors")
        if n_colors is None:
            n_colors = int(self.floor_colors.max()) + 1
        self.n_colors = n_colors
        # цвета пола должны быть 0..n_colors-1, иначе клетку пола спутаем с целью, препятствием или стеной
        if self.floor_colors.min() < 0 or self.floor_colors.max() >= n_colors:
            raise ValueError(f"цвета пола должны быть от 0 до {n_colors - 1}")
        self.pos_goal = self.check_cell(pos_goal, "pos_goal")
        self.see_obstacle = see_obstacle
        if max_steps is None:
            max_steps = 4 * self.h * self.w
        self.max_steps = max_steps
        if obs_mode not in ("color", "coords"):
            raise ValueError(f"obs_mode должен быть 'color' или 'coords', получено {obs_mode!r}")
        self.obs_mode = obs_mode
        self.render_mode = render_mode

        self.goal_color = n_colors
        self.obstacle_color = n_colors + 1
        self.wall_color = n_colors + 2
        self.n_obs_colors = n_colors + 3

        # карта на 2 клетки больше: по краю стена, тогда шаг за край — это шаг в стену
        self.color_map = np.full((self.h + 2, self.w + 2), self.wall_color)
        self.color_map[1:-1, 1:-1] = np.where(self.obstacle_mask, self.obstacle_color, self.floor_colors)
        self.color_map[self.pos_goal[0] + 1, self.pos_goal[1] + 1] = self.goal_color

        self.start_probs = self.make_start_probs(pos_agent)

        if obs_mode == "color":
            obs_size = self.n_obs_colors
        else:
            obs_size = self.h * self.w
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.agent_pos = (0, 0)
        self.color = 0
        self.steps = 0

    def check_cell(self, pos, name):
        """Проверяет, что pos — две целые координаты внутри карты и не на препятствии. Возвращает (r, c)."""
        if len(pos) != 2 or not all(isinstance(x, (int, np.integer)) for x in pos):
            raise ValueError(f"{name}: нужны две целые координаты (строка, столбец), получено {pos}")
        r, c = int(pos[0]), int(pos[1])
        if not (0 <= r < self.h and 0 <= c < self.w):
            raise ValueError(f"{name}: клетка {(r, c)} вне карты {self.h}x{self.w}")
        if self.obstacle_mask[r, c]:
            raise ValueError(f"{name}: клетка {(r, c)} — препятствие")
        return r, c

    def check_start(self, pos):
        """Стартовать можно только со свободной клетки, которая не цель."""
        r, c = self.check_cell(pos, "старт")
        if (r, c) == self.pos_goal:
            raise ValueError(f"старт: клетка {(r, c)} — цель")
        return r, c

    def make_start_probs(self, pos_agent):
        """Вероятность стартовать из каждой клетки (вектор длины h*w)."""
        allowed = ~self.obstacle_mask
        allowed[self.pos_goal] = False
        if pos_agent is None:
            weights = allowed.astype(float)
        elif np.shape(pos_agent) == (2,):
            r, c = self.check_start(tuple(pos_agent))
            weights = np.zeros((self.h, self.w))
            weights[r, c] = 1.0
        else:
            weights = np.array(pos_agent, dtype=float)
            if weights.shape != (self.h, self.w):
                raise ValueError(f"веса стартов должны быть массивом формы {(self.h, self.w)}")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("веса стартов должны быть конечными неотрицательными числами")
            weights = np.where(allowed, weights, 0.0)
            if weights.sum() <= 0:
                raise ValueError("после маски препятствий и цели у стартов не осталось положительных весов")
        return (weights / weights.sum()).ravel()

    def cell_color(self, r, c):
        """Цвет клетки (r, c); r и c могут быть -1 или h/w — это стена."""
        return int(self.color_map[r + 1, c + 1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "pos" in options:
            self.agent_pos = self.check_start(tuple(options["pos"]))
        else:
            index = int(self.np_random.choice(self.h * self.w, p=self.start_probs))
            self.agent_pos = (index // self.w, index % self.w)
        self.color = self.cell_color(*self.agent_pos)
        self.steps = 0
        return self.get_obs(), self.get_info(False)

    def step(self, action):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"недопустимое действие {action}, нужно 0..3")
        dr, dc = ACTION_DELTAS[int(action)]
        r, c = self.agent_pos
        new_r, new_c = r + dr, c + dc
        new_color = self.cell_color(new_r, new_c)
        bumped = new_color in (self.wall_color, self.obstacle_color)

        if bumped:
            if self.see_obstacle:
                self.color = new_color  # видим стену, но стоим на месте
            else:
                self.color = self.cell_color(r, c)
        else:
            self.agent_pos = (new_r, new_c)
            self.color = new_color
        self.steps += 1

        terminated = self.agent_pos == self.pos_goal
        truncated = (not terminated) and self.steps >= self.max_steps
        reward = 1.0 if terminated else 0.0
        return self.get_obs(), reward, terminated, truncated, self.get_info(bumped)

    def get_obs(self):
        if self.obs_mode == "color":
            obs = np.zeros(self.n_obs_colors, dtype=np.float32)
            obs[self.color] = 1.0
        else:
            obs = np.zeros(self.h * self.w, dtype=np.float32)
            obs[self.agent_pos[0] * self.w + self.agent_pos[1]] = 1.0
        return obs

    def get_info(self, bumped):
        return {"pos": np.array(self.agent_pos), "color": self.color, "bumped": bumped}

    def render(self):
        """Текстом: # стена, X препятствие, G цель, A агент, число — цвет пола."""
        lines = []
        for r in range(-1, self.h + 1):
            row = []
            for c in range(-1, self.w + 1):
                color = self.cell_color(r, c)
                if (r, c) == self.agent_pos:
                    row.append(" A")
                elif color == self.wall_color:
                    row.append(" #")
                elif color == self.obstacle_color:
                    row.append(" X")
                elif color == self.goal_color:
                    row.append(" G")
                else:
                    row.append(f"{color:2d}")
            lines.append(" ".join(row))
        return "\n".join(lines)
