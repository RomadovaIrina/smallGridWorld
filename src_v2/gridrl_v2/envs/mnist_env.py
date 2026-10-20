"""Копия GridWorldEnv, где цвет клетки показывается случайной картинкой цифры из MNIST.

Цвет k -> случайная картинка цифры k. Цветов всего n_colors + 3 <= 10, поэтому пола не больше 7 цветов.
Картинки уменьшены с 28x28 до 14x14 и вытянуты в вектор из 196 чисел от 0 до 1.
"""
import gzip
import urllib.request
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # зеркало, которое использует torchvision


def load_mnist(folder="data/mnist"):
    """Картинки MNIST 14x14 по цифрам: images[k] — массив (число картинок, 196) для цифры k.
    При первом запуске скачивает два файла (~10 МБ) в folder.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]:
        if not (folder / name).exists():
            print("скачиваю", name)
            urllib.request.urlretrieve(MNIST_URL + name, folder / name)

    with gzip.open(folder / "train-images-idx3-ubyte.gz") as f:
        pixels = np.frombuffer(f.read(), dtype=np.uint8, offset=16)  # первые 16 байт — заголовок
    with gzip.open(folder / "train-labels-idx1-ubyte.gz") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)  # первые 8 байт — заголовок

    pictures = pixels.reshape(-1, 28, 28) / 255.0
    pictures = pictures.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))  # 28x28 -> 14x14: среднее по квадратам 2x2
    pictures = pictures.reshape(-1, 196).astype(np.float32)
    images = []
    for digit in range(10):
        images.append(pictures[labels == digit])
    return images


class MnistGridWorldEnv(gym.Env):
    """Тот же лабиринт, но наблюдение — картинка цифры вместо one-hot цвета.

    images — список из 10 массивов картинок (см. load_mnist). Если не передать — загрузится MNIST.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, floor_colors, pos_goal, n_colors=None, obstacle_mask=None, pos_agent=None,
                 see_obstacle=True, max_steps=None, images=None, render_mode=None):
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
        self.render_mode = render_mode

        self.goal_color = n_colors
        self.obstacle_color = n_colors + 1
        self.wall_color = n_colors + 2
        self.n_obs_colors = n_colors + 3
        if self.n_obs_colors > 10:
            raise ValueError("в MNIST всего 10 цифр: цветов пола может быть не больше 7")

        if images is None:
            images = load_mnist()
        self.images = images

        # карта на 2 клетки больше: по краю стена, тогда шаг за край — это шаг в стену
        self.color_map = np.full((self.h + 2, self.w + 2), self.wall_color)
        self.color_map[1:-1, 1:-1] = np.where(self.obstacle_mask, self.obstacle_color, self.floor_colors)
        self.color_map[self.pos_goal[0] + 1, self.pos_goal[1] + 1] = self.goal_color

        self.start_probs = self.make_start_probs(pos_agent)

        obs_size = len(self.images[0][0])  # 196
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
        # единственное отличие от GridWorldEnv: вместо one-hot — случайная картинка цифры self.color
        pictures = self.images[self.color]
        index = self.np_random.integers(len(pictures))
        return pictures[index].copy()

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
