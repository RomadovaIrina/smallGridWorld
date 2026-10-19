"""Отрисовка карты среды: RGB-картинка и текст."""

from __future__ import annotations

import colorsys

import numpy as np

from gridrl.envs import grid_core as core

GOAL_RGB = (40, 200, 60)
OBSTACLE_RGB = (60, 60, 60)
WALL_RGB = (20, 20, 20)
AGENT_RGB = (230, 30, 30)


def floor_palette(n_colors: int) -> np.ndarray:
    """Различимые пастельные цвета пола (n_colors, 3) uint8."""
    hues = np.linspace(0.0, 1.0, n_colors, endpoint=False)
    rgb = [colorsys.hsv_to_rgb(h, 0.35, 0.95) for h in hues]
    return (np.array(rgb) * 255).astype(np.uint8)


def color_map_rgb(color_map: np.ndarray, n_colors: int) -> np.ndarray:
    """Переводит карту кодов цветов в RGB (H, W, 3)."""
    palette = np.concatenate(
        [floor_palette(n_colors), np.array([GOAL_RGB, OBSTACLE_RGB, WALL_RGB], dtype=np.uint8)]
    )
    return palette[color_map]


def render_rgb(env, agent_pos: tuple[int, int] | None = None, cell_px: int = 24) -> np.ndarray:
    """Карта среды с рамкой стены и агентом (кружок) в виде RGB-массива."""
    img = color_map_rgb(env.color_map, env.n_colors)
    img = np.repeat(np.repeat(img, cell_px, axis=0), cell_px, axis=1)
    # тонкая сетка между клетками
    img[::cell_px, :] = img[::cell_px, :] * 0.8
    img[:, ::cell_px] = img[:, ::cell_px] * 0.8
    if agent_pos is not None:
        r, c = agent_pos[0] + 1, agent_pos[1] + 1
        yy, xx = np.mgrid[:cell_px, :cell_px]
        center = (cell_px - 1) / 2
        disk = (yy - center) ** 2 + (xx - center) ** 2 <= (cell_px * 0.35) ** 2
        cell = img[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px]
        cell[disk] = AGENT_RGB
    return img


def render_ansi(env, agent_pos: tuple[int, int] | None = None) -> str:
    """Текстовая карта: '#' стена, 'X' препятствие, 'G' цель, 'A' агент, цифры/буквы — цвет пола."""
    symbols = "0123456789abcdefghijklmnopqrstuvwxyz"
    lines = []
    for r in range(env.color_map.shape[0]):
        row = []
        for c in range(env.color_map.shape[1]):
            code = int(env.color_map[r, c])
            if agent_pos is not None and (r - 1, c - 1) == tuple(agent_pos):
                ch = "A"
            elif code == core.wall_color(env.n_colors):
                ch = "#"
            elif code == core.obstacle_color(env.n_colors):
                ch = "X"
            elif code == core.goal_color(env.n_colors):
                ch = "G"
            else:
                ch = symbols[code] if code < len(symbols) else "?"
            row.append(ch)
        lines.append(" ".join(row))
    return "\n".join(lines)
