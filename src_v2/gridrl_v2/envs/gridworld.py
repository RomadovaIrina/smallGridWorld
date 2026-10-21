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
        self.h, self.w = self.floor_colors.shape
        if obstacle_mask is None:
            obstacle_mask = np.zeros((self.h, self.w), dtype=bool)
        self.obstacle_mask = np.array(obstacle_mask, dtype=bool)
        self.pos_goal = tuple(pos_goal)
        if n_colors is None:
            n_colors = int(self.floor_colors.max()) + 1
        self.n_colors = n_colors
        self.see_obstacle = see_obstacle
        if max_steps is None:
            max_steps = 4 * self.h * self.w
        self.max_steps = max_steps
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

    def make_start_probs(self, pos_agent):
        """Вероятность стартовать из каждой клетки (вектор длины h*w)."""
        allowed = ~self.obstacle_mask
        allowed[self.pos_goal] = False
        if pos_agent is None:
            weights = allowed.astype(float)
        elif np.shape(pos_agent) == (2,):
            weights = np.zeros((self.h, self.w))
            # Проверим а не хотим ли мы случайно посадить агента за карту
            if not (0 <= r < self.h and 0 <= c < self.w):
                raise ValueError("Start position is out of bounds")
            # Или в препятствие или целевую точку :)
            if not allowed[r, c]:
                raise ValueError("Start position is blocked or is the goal")
                
            weights[pos_agent[0], pos_agent[1]] = 1.0
        else:
            weights = np.where(allowed, pos_agent, 0.0)
        return (weights / weights.sum()).ravel()

    def cell_color(self, r, c):
        """Цвет клетки (r, c); r и c могут быть -1 или h/w — это стена."""
        return int(self.color_map[r + 1, c + 1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "pos" in options:
            self.agent_pos = tuple(options["pos"])
        else:
            index = int(self.np_random.choice(self.h * self.w, p=self.start_probs))
            self.agent_pos = (index // self.w, index % self.w)
        self.color = self.cell_color(*self.agent_pos)
        self.steps = 0
        return self.get_obs(), self.get_info(False)

    def step(self, action):
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
