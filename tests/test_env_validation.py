"""Тесты на исправленные ошибки валидации (см. docs/audit.md). Проверяем обе среды: обычную и MNIST."""
import numpy as np
import pytest

from src.envs import GridWorldEnv, MnistGridWorldEnv, make_env

FAKE_IMAGES = [np.full((3, 196), k / 10, dtype=np.float32) for k in range(10)]
FLOOR = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
OBSTACLES = np.zeros((3, 3), dtype=bool)
OBSTACLES[1, 1] = True


def make(env_class, **kw):
    params = dict(floor_colors=FLOOR, pos_goal=(2, 2), n_colors=3, obstacle_mask=OBSTACLES)
    params.update(kw)
    if env_class is MnistGridWorldEnv:
        params["images"] = FAKE_IMAGES
    return env_class(**params)


ENVS = [GridWorldEnv, MnistGridWorldEnv]


# ошибка 1: фиксированный старт на цели, препятствии, вне карты, нецелые координаты
@pytest.mark.parametrize("env_class", ENVS)
@pytest.mark.parametrize("bad", [(2, 2), (1, 1), (3, 0), (-1, 0), (0, -1), (0.5, 0)])
def test_bad_fixed_start(env_class, bad):
    with pytest.raises(ValueError):
        make(env_class, pos_agent=bad)


@pytest.mark.parametrize("env_class", ENVS)
def test_good_fixed_start(env_class):
    env = make(env_class, pos_agent=(0, 2))
    _, info = env.reset(seed=0)
    assert tuple(info["pos"]) == (0, 2)


# ошибка 2: обход проверки через reset(options={"pos": ...})
@pytest.mark.parametrize("env_class", ENVS)
@pytest.mark.parametrize("bad", [(2, 2), (1, 1), (-1, 0), (5, 5)])
def test_bad_reset_pos(env_class, bad):
    env = make(env_class)
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"pos": bad})


# ошибка 3: плохие веса стартов
@pytest.mark.parametrize("env_class", ENVS)
@pytest.mark.parametrize("weights", [
    np.zeros((3, 3)),                      # сумма 0
    np.ones((2, 3)),                       # не та форма
    np.full((3, 3), -1.0),                 # отрицательные
    np.full((3, 3), np.nan),               # не числа
    np.where(OBSTACLES, 1.0, 0.0),         # вес только на препятствии -> после маски 0
])
def test_bad_start_weights(env_class, weights):
    with pytest.raises(ValueError):
        make(env_class, pos_agent=weights)


# ошибка 4: цель на препятствии или вне карты
@pytest.mark.parametrize("env_class", ENVS)
@pytest.mark.parametrize("bad_goal", [(1, 1), (3, 3), (-1, 0)])
def test_bad_goal(env_class, bad_goal):
    with pytest.raises(ValueError):
        make(env_class, pos_goal=bad_goal)


# ошибка 5: цвет пола совпадает со служебным
@pytest.mark.parametrize("env_class", ENVS)
def test_floor_color_out_of_range(env_class):
    floor = FLOOR.copy()
    floor[0, 0] = 3  # при n_colors=3 это код цели
    with pytest.raises(ValueError):
        make(env_class, floor_colors=floor)


@pytest.mark.parametrize("env_class", ENVS)
def test_floor_not_integer(env_class):
    with pytest.raises(ValueError):
        make(env_class, floor_colors=FLOOR.astype(float))


# ошибка 6: опечатка в obs_mode
def test_bad_obs_mode():
    with pytest.raises(ValueError):
        make(GridWorldEnv, obs_mode="colour")


# доп. аудит: недопустимое действие
@pytest.mark.parametrize("env_class", ENVS)
@pytest.mark.parametrize("bad_action", [-1, 4])
def test_bad_action(env_class, bad_action):
    env = make(env_class)
    env.reset(seed=0, options={"pos": (0, 0)})
    with pytest.raises(ValueError):
        env.step(bad_action)


# доп. аудит: reset не меняет карту, наблюдение лежит в observation_space
@pytest.mark.parametrize("env_class", ENVS)
def test_reset_keeps_map_and_obs_in_space(env_class):
    env = make(env_class)
    before = env.color_map.copy()
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        obs, *_ = env.step(0)
        assert env.observation_space.contains(obs)
    assert np.array_equal(before, env.color_map)


# доп. аудит: после удара о стену и reset агент стартует заново, шаги обнулены
def test_reset_after_bump():
    env = make(GridWorldEnv)
    env.reset(seed=0, options={"pos": (0, 0)})
    env.step(0)  # вверх в стену
    _, info = env.reset(seed=0, options={"pos": (0, 2)})
    assert tuple(info["pos"]) == (0, 2) and env.steps == 0


# доп. аудит: воспроизводимость карты и стартов по seed
def test_reproducible():
    a = make_env(10, 10, 4, 0.1, seed=3)
    b = make_env(10, 10, 4, 0.1, seed=3)
    assert np.array_equal(a.color_map, b.color_map) and a.pos_goal == b.pos_goal
    starts_a = [tuple(a.reset(seed=s)[1]["pos"]) for s in range(10)]
    starts_b = [tuple(b.reset(seed=s)[1]["pos"]) for s in range(10)]
    assert starts_a == starts_b
