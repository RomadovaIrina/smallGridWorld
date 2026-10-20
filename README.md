# RL for GridWorld

Реализованы клеточная среда GridWorld с наблюдением цвета клетки, четыре конфигурации карт и два агента: PPO (on-policy) и Discrete SAC (off-policy), каждый в вариантах без памяти (MLP) и с памятью (GRU). Реализация наивная, без оптимизаций скорости.

Автор: _ФИО_

## Установка и тесты

Нужен [uv](https://docs.astral.sh/uv/). Версия Python указана в `.python-version`.

```bash
uv sync --extra cpu      # без NVIDIA GPU
uv sync --extra cu126    # с NVIDIA GPU
uv run pytest
```

## Пример

```python
from gridrl.envs import make_env

env = make_env(2, seed=42, render_mode="ansi")
obs, info = env.reset(seed=42)
print(env.render())

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(env.render())
    if terminated or truncated:
        break
```

Здесь действия случайные, обучения нет.

## Обучение и графики

```bash
uv run python -m gridrl.utils.train_ppo --setup 2 --memory
uv run python -m gridrl.utils.train_dsac --setup 2 --no-memory --no-soft-q
uv run python -m gridrl.utils.plot_envs
uv run python -m gridrl.utils.plot_training runs/<прогон1> runs/<прогон2> --labels MLP GRU --name setup2
uv run python -m gridrl.utils.evaluate runs/<прогон> --episodes 100
```

Логи обучения пишутся в `runs/<имя>/` (`metrics.csv`, `config.json`, `model.pt`), в git не попадают. Картинки для отчёта сохраняются в `outputs/`:
- `outputs/envs/` — карты всех сетапов;
- `outputs/plots/` — графики обучения: отдача, доля успехов, длина эпизода, лосс и его компоненты, энтропия, вероятность нежадного действия;
- `outputs/eval/` — работа обученного агента: тепловая карта посещений за eval-эпизоды и GIF одного эпизода.

Подробный разбор кода — в `docs/walkthrough.md`.

## Файлы

Активная версия кода — `src_v2/` (упрощённая). Первая версия лежит в `src/` для сравнения и не используется.


- `src_v2/gridrl/envs/gridworld.py` — среда: инициализация, старт эпизода, движение, наблюдения и отрисовка.
- `src_v2/gridrl/envs/setups.py` — генерация карт, четыре конфигурации из задания и расчёт кратчайших расстояний через BFS.
- `src_v2/gridrl/envs/__init__.py` — импорты и регистрация среды как `GridWorld-v0` в Gymnasium.
- `src_v2/gridrl/models/ppo.py`, `src_v2/gridrl/models/dsac.py` — алгоритмы обучения вместе с сетями (MLP или GRU).
- `src_v2/gridrl/utils/__init__.py` — выбор устройства PyTorch, seed, разбор аргументов командной строки.
- `src_v2/gridrl/utils/*.py` — скрипты: обучение, картинки сред, графики, eval.
- `tests/test_gridworld.py` — проверки правил среды и совместимости с Gymnasium.

## Среда

- Действия: 0 вверх, 1 вправо, 2 вниз, 3 влево.
- Цвета пола: `0..n_colors-1`, далее цель, препятствие и стена.
- Наблюдение — one-hot вектор длины `n_colors + 3`.
- Награда 1 за достижение цели, иначе 0. При достижении цели эпизод завершается.
- При `see_obstacle=True` столкновение даёт наблюдение цвета преграды, позиция не меняется. При `False` агент видит цвет своей клетки.
- `obs_mode="coords"` даёт one-hot позиции для отладки.
- Координаты в `info` предназначены для отладки и не должны подаваться агенту при обучении по цветам.

## Принятые допущения

- Лимит эпизода — `4 * h * w` шагов.
- Все входные карты, включая веса `pos_agent`, имеют форму `(h, w)`.
- При генерации конфигурации цель выбирается случайно и остаётся фиксированной в созданной среде.
- По умолчанию старт выбирается равномерно среди свободных клеток, кроме цели.
- Случайные препятствия генерируются так, чтобы свободные клетки были связны.
- `make_env(..., seed=...)` задаёт генерацию карты и цели, а `env.reset(seed=...)` — выбор старта.
