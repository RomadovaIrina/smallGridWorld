from gymnasium.envs.registration import register

from gridrl.envs.gridworld import GridWorldEnv
from gridrl.envs.setups import (
    SETUPS,
    build_setup,
    make_env,
    make_vec_env,
    optimal_stats,
    random_floor_coloring,
    random_obstacle_mask,
    unique_floor_coloring,
)

register(id="GridWorld-v0", entry_point="gridrl.envs.gridworld:GridWorldEnv")

__all__ = [
    "GridWorldEnv",
    "SETUPS",
    "build_setup",
    "make_env",
    "make_vec_env",
    "optimal_stats",
    "random_floor_coloring",
    "random_obstacle_mask",
    "unique_floor_coloring",
]
