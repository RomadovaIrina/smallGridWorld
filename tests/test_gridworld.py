import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import gridrl.envs  # noqa: F401  (регистрирует GridWorld-v0)
from gridrl.envs import GridWorldEnv, build_setup, make_env, optimal_stats
from gridrl.envs import grid_core as core
from gridrl.envs.grid_core import DOWN, LEFT, RIGHT, UP

N_COLORS = 3


def small_env(**kw):
    """3x3, цвета пола по строкам 0/1/2, препятствие в (1, 1), цель в (2, 2)."""
    floor = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    obstacles = np.zeros((3, 3), dtype=bool)
    obstacles[1, 1] = True
    params = dict(floor_colors=floor, n_colors=N_COLORS, obstacle_mask=obstacles, pos_goal=(2, 2))
    return GridWorldEnv(**{**params, **kw})


def color_of(obs):
    assert obs.sum() == 1.0
    return int(np.argmax(obs))


@pytest.mark.parametrize("setup_id", [1, 2, 3, 4])
def test_check_env(setup_id):
    check_env(make_env(setup_id, seed=0), skip_render_check=True)


def test_gym_make_registered():
    env = gym.make("GridWorld-v0", **build_setup(2, seed=0))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5 + 3,)


def test_observation_is_one_hot_of_start_cell():
    env = small_env()
    obs, info = env.reset(seed=0, options={"pos": (0, 1)})
    assert obs.shape == (N_COLORS + 3,)
    assert color_of(obs) == 0
    assert tuple(info["pos"]) == (0, 1)


def test_wall_bump_see_obstacle_true():
    env = small_env(see_obstacle=True)
    env.reset(seed=0, options={"pos": (0, 0)})
    obs, reward, term, trunc, info = env.step(UP)
    assert tuple(info["pos"]) == (0, 0) and info["bumped"]
    assert color_of(obs) == core.wall_color(N_COLORS)
    assert reward == 0.0 and not term and not trunc


def test_wall_bump_see_obstacle_false():
    env = small_env(see_obstacle=False)
    env.reset(seed=0, options={"pos": (0, 0)})
    obs, *_, info = env.step(LEFT)
    assert tuple(info["pos"]) == (0, 0) and info["bumped"]
    assert color_of(obs) == 0  # видит свою клетку


def test_obstacle_bump():
    env = small_env()
    env.reset(seed=0, options={"pos": (0, 1)})
    obs, *_, info = env.step(DOWN)
    assert tuple(info["pos"]) == (0, 1) and info["bumped"]
    assert color_of(obs) == core.obstacle_color(N_COLORS)


def test_moves_clockwise():
    env = small_env(obstacle_mask=np.zeros((3, 3), dtype=bool))
    env.reset(seed=0, options={"pos": (1, 1)})
    expected = {UP: (0, 1), RIGHT: (1, 2), DOWN: (2, 1), LEFT: (1, 0)}
    for a, pos in expected.items():
        env.reset(seed=0, options={"pos": (1, 1)})
        *_, info = env.step(a)
        assert tuple(info["pos"]) == pos


def test_goal_reward_and_termination():
    env = small_env()
    env.reset(seed=0, options={"pos": (2, 1)})
    obs, reward, term, trunc, _ = env.step(RIGHT)
    assert reward == 1.0 and term and not trunc
    assert color_of(obs) == core.goal_color(N_COLORS)


def test_truncation():
    env = small_env(max_steps=5)
    env.reset(seed=0, options={"pos": (0, 0)})
    for t in range(5):
        _, _, term, trunc, _ = env.step(UP)
        assert not term
        assert trunc == (t == 4)


def test_start_fixed_position():
    env = small_env(pos_agent=(0, 2))
    for s in range(10):
        _, info = env.reset(seed=s)
        assert tuple(info["pos"]) == (0, 2)


def test_start_weights_masked_and_normalized():
    weights = np.ones((3, 3))
    weights[0, 0] = 5.0
    env = small_env(pos_agent=weights)
    probs = env.start_probs.reshape(3, 3)
    assert probs[1, 1] == 0.0  # препятствие
    assert probs[2, 2] == 0.0  # цель
    assert np.isclose(probs.sum(), 1.0)
    assert np.isclose(probs[0, 0], 5.0 / 11.0)


def test_start_uniform_excludes_obstacles_and_goal():
    env = small_env()
    seen = {tuple(env.reset(seed=s)[1]["pos"]) for s in range(300)}
    assert (1, 1) not in seen and (2, 2) not in seen
    assert len(seen) == 7


def test_invalid_start_raises():
    with pytest.raises(ValueError):
        small_env(pos_agent=(1, 1))
    with pytest.raises(ValueError):
        small_env(pos_agent=np.zeros((3, 3)))


def test_coords_obs_mode():
    env = small_env(obs_mode="coords")
    obs, _ = env.reset(seed=0, options={"pos": (2, 0)})
    assert obs.shape == (9,) and obs[6] == 1.0


def test_reset_is_deterministic_by_seed():
    env = make_env(4, seed=0)
    a = [tuple(env.reset(seed=s)[1]["pos"]) for s in range(20)]
    b = [tuple(env.reset(seed=s)[1]["pos"]) for s in range(20)]
    assert a == b


@pytest.mark.parametrize("setup_id,n_obs,n_obstacles", [(1, 28, 0), (2, 8, 0), (3, 10, 10), (4, 7, 10)])
def test_setups(setup_id, n_obs, n_obstacles):
    for seed in range(5):
        env = make_env(setup_id, seed=seed)
        assert env.observation_space.shape == (n_obs,)
        assert env.obstacle_mask.sum() == n_obstacles
        dist = core.bfs_distances(~env.obstacle_mask, env.pos_goal)
        assert (dist[~env.obstacle_mask] >= 0).all()  # цель достижима отовсюду


def test_setup_1_colors_unique():
    env = make_env(1, seed=0)
    inner = env.inner_color_map
    assert len(np.unique(inner)) == 25


def test_optimal_stats():
    env = small_env(pos_agent=(0, 0))
    stats = optimal_stats(env, gamma=0.9)
    assert stats["length"] == 4
    assert np.isclose(stats["return"], 0.9 ** 3)


def test_render():
    env = small_env(render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.shape == (5 * 24, 5 * 24, 3) and img.dtype == np.uint8
    env = small_env(render_mode="ansi")
    env.reset(seed=0, options={"pos": (0, 0)})
    text = env.render()
    assert text.splitlines()[1].split()[1] == "A"
