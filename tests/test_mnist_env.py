import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gridrl_v2.envs import MnistGridWorldEnv, make_env
from gridrl_v2.envs.gridworld import RIGHT, UP

# вместо MNIST — фейковые «картинки»: у цифры k все пиксели равны k / 10, по 3 картинки на цифру
FAKE_IMAGES = [np.full((3, 196), k / 10, dtype=np.float32) for k in range(10)]


def small_env(**kw):
    floor = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    return MnistGridWorldEnv(floor, (2, 2), n_colors=3, images=FAKE_IMAGES, **kw)


def digit_of(obs):
    return round(float(obs[0]) * 10)


def test_check_env():
    check_env(small_env(), skip_render_check=True)


def test_obs_is_picture_of_cell_color():
    env = small_env()
    obs, _ = env.reset(seed=0, options={"pos": (1, 0)})
    assert obs.shape == (196,)
    assert digit_of(obs) == 1


def test_wall_shows_wall_digit():
    env = small_env()
    env.reset(seed=0, options={"pos": (0, 0)})
    obs, *_ = env.step(UP)
    assert digit_of(obs) == 3 + 2  # стена = n_colors + 2


def test_goal():
    env = small_env()
    env.reset(seed=0, options={"pos": (2, 1)})
    obs, reward, terminated, truncated, _ = env.step(RIGHT)
    assert reward == 1.0 and terminated
    assert digit_of(obs) == 3  # цель = n_colors


def test_too_many_colors():
    with pytest.raises(ValueError):
        make_env(5, 5, 8, 0.0, mnist=True, images=FAKE_IMAGES)


def test_make_env_mnist():
    env = make_env(10, 10, 7, 0.1, seed=0, mnist=True, images=FAKE_IMAGES)
    assert isinstance(env, MnistGridWorldEnv)
    obs, info = env.reset(seed=0)
    assert digit_of(obs) == info["color"]
