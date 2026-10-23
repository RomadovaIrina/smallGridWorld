"""Собирает ноутбуки tasks/task1.ipynb, task2.ipynb, task3.ipynb из ячеек ниже (чтобы не править JSON руками).

    uv run python tools/make_notebooks.py
"""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def save(cells, path):
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    Path(path).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", path)


# ---------- общие ячейки задач 2 и 3 ----------

IMPORTS = '''
import csv
import json
import os
import sys
from pathlib import Path

# работаем из корня репозитория: там лежат src/ и outputs/
if Path.cwd().name == "tasks":
    os.chdir(Path.cwd().parent)
sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import utils
from src.envs import load_mnist, make_env
from src.models import DSAC, PPO
'''

MODE = '''
# режим запуска: при обычном открытии ноутбука — полное обучение всех экспериментов и все графики
# на сервере скрипт задаёт переменные, чтобы запускать эксперименты параллельно
RUN_MODE = os.environ.get("RUN_MODE", "full")  # full или smoke (короткая проверка)
RUN_ONLY = os.environ.get("RUN_ONLY", "")  # имя одного эксперимента, пусто — все
RUN_STAGES = os.environ.get("RUN_STAGES", "train,plots").split(",")

torch.set_num_threads(1)  # прогоны идут параллельно, каждому хватит одного потока
CUDA = torch.cuda.is_available()
print("режим:", RUN_MODE, "| эксперимент:", RUN_ONLY or "все", "| этапы:", RUN_STAGES, "| CUDA:", CUDA)
'''

SETUPS = '''
SETUPS = {
    1: dict(h=5, w=5, n_colors=None, obstacle_fraction=0.0),  # 25 уникальных цветов — полностью наблюдаемая
    2: dict(h=5, w=5, n_colors=5, obstacle_fraction=0.0),
    3: dict(h=10, w=10, n_colors=7, obstacle_fraction=0.1),
    4: dict(h=10, w=10, n_colors=4, obstacle_fraction=0.1),
}
'''

PPO_CFG = '''
ppo_cfg = {
    "device": "cpu",  # PPO на видеокарте не быстрее (замер: 479 против 429 шагов в секунду)
    "env_seed": 0,  # от него зависит карта
    "seed": 0,  # от него зависят сеть и старты эпизодов
    "memory": True,  # True — GRU, False — MLP без памяти
    "episodes_per_update": 10,  # сколько эпизодов собрать перед обновлением
    "epochs": 4,
    "episodes_per_batch": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "lr": 3e-4,
    "clip": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "hidden": 128,
    "save_every": 50,  # как часто сохранять метрики и веса на диск
}
'''

DSAC_CFG = '''
dsac_cfg = {
    "device": "cuda" if CUDA else "cpu",  # DSAC на видеокарте в 3 раза быстрее (167 против 52 шагов в секунду)
    "env_seed": 0,
    "seed": 0,
    "memory": True,
    "soft_q": True,  # мягкая Q (энтропия в цели критика) или обычная
    "buffer_episodes": 5000,  # сколько последних эпизодов помнить
    "learning_starts": 20,  # сколько эпизодов только копим опыт, не учась
    "updates_per_episode": 4,  # сколько обновлений после каждого эпизода
    "batch_size": 8,  # сколько случайных эпизодов из буфера в одном обновлении
    "gamma": 0.99,
    "tau": 0.005,  # скорость, с которой target-критики догоняют обычных
    "lr": 3e-4,
    "alpha": 0.005,  # температура: вес энтропии
    "hidden": 128,
    "save_every": 25,
}
'''

BUDGET = '''
# сколько эпизодов учиться: на больших картах дольше; smoke — короткая проверка, что всё работает
EPISODES = {
    "full": {"ppo": {1: 1000, 2: 1000, 3: 1500, 4: 1500}, "dsac": {1: 500, 2: 500, 3: 800, 4: 800}},
    "smoke": {"ppo": {1: 20, 2: 20, 3: 20, 4: 20}, "dsac": {1: 8, 2: 8, 3: 4, 4: 4}},
}
EVAL_EPISODES = {"full": 100, "smoke": 5}
'''

RUN_ONE = '''
def run_experiment(exp):
    # прогон, который уже закончился, второй раз не учим
    status_file = Path(OUT) / "runs" / exp["name"] / "status.json"
    if status_file.exists() and json.loads(status_file.read_text())["status"] == "completed":
        print("уже готово:", exp["name"])
        return

    if exp["algo"] == "ppo":
        alg = PPO(exp["cfg"])
    else:
        alg = DSAC(exp["cfg"])
    alg.train()

    # оценка на одних и тех же стартах (сиды 10000, 10001, ...): жадно и с сэмплированием действий
    n = EVAL_EPISODES[RUN_MODE]
    greedy = alg.evaluate(n, greedy=True)
    sampled = alg.evaluate(n, greedy=False)
    result = {"budget": RUN_MODE,
              "greedy": utils.eval_summary(greedy, alg.cfg["gamma"], alg.optimal),
              "stochastic": utils.eval_summary(sampled, alg.cfg["gamma"], alg.optimal)}
    utils.save_eval(alg.run_dir, result)

    s = result["stochastic"]
    title = f"{exp['name']}: успех {s['success_rate']:.2f}, длина {s['mean_length']:.1f} (оптимум {s['optimal_length']:.1f})"
    utils.save_eval_pictures(alg.env, sampled, Path(OUT) / "eval", exp["name"], title)
'''

RUN_ALL = '''
if "train" in RUN_STAGES:
    for exp in EXPERIMENTS:
        if RUN_ONLY == "" or RUN_ONLY == exp["name"]:
            run_experiment(exp)
'''

LOAD = '''
plt.style.use("seaborn-v0_8-darkgrid")


def load_metrics(run_path):
    # metrics.csv прогона: словарь {колонка: массив чисел}; None, если прогона нет
    path = Path(run_path) / "metrics.csv"
    if not path.exists():
        print("нет данных:", run_path)
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = {}
    for key in rows[0]:
        values = [{"True": 1.0, "False": 0.0, "": np.nan}.get(row[key], row[key]) for row in rows]
        data[key] = np.array(values, dtype=float)
    return data
'''

CURVE = '''
def plot_curve(ax, values, label, color, window=50):
    # линия — скользящее среднее за window эпизодов, полоса — скользящее стандартное отклонение
    # внутри одного прогона (у нас один сид, это не доверительный интервал)
    means = []
    stds = []
    for i in range(len(values)):
        last = values[max(0, i - window + 1):i + 1]
        last = last[~np.isnan(last)]
        means.append(np.mean(last) if len(last) > 0 else np.nan)
        stds.append(np.std(last) if len(last) > 0 else np.nan)
    means = np.array(means)
    stds = np.array(stds)
    ax.plot(means, color=color, linewidth=2, label=label)
    ax.fill_between(np.arange(len(values)), means - stds, means + stds, color=color, alpha=0.25)
'''

COMPARE = '''
def plot_compare(runs, labels, key, ylabel, filename):
    # runs: {"Env2": [путь1, путь2], ...} — на каждую среду свой график, на нём несколько прогонов
    colors = ["tab:red", "tab:blue", "tab:green"]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.3), squeeze=False)
    for ax, env_name in zip(axes[0], runs):
        for path, label, color in zip(runs[env_name], labels, colors):
            data = load_metrics(path)
            if data is not None:
                plot_curve(ax, data[key], label, color)
        ax.set_title(env_name)
        ax.set_xlabel("эпизод")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    Path(OUT, "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(OUT) / "figures" / filename, dpi=150, bbox_inches="tight")
    plt.show()
'''

TABLE = '''
def eval_rows(run_paths):
    # строки итоговой таблицы из eval.json прогонов
    rows = []
    for path in run_paths:
        eval_file = Path(path) / "eval.json"
        if not eval_file.exists():
            rows.append({"run": Path(path).name, "eval": "pending"})
            continue
        result = json.loads(eval_file.read_text(encoding="utf-8"))
        for mode in ["greedy", "stochastic"]:
            r = result[mode]
            rows.append({"run": Path(path).name, "eval": mode, "success_rate": round(r["success_rate"], 3),
                         "mean_return": round(r["mean_return"], 3), "mean_length": round(r["mean_length"], 1),
                         "failures": r["failures"], "optimal_length": round(r["optimal_length"], 2)})
    return rows


def save_table(rows, filename):
    Path(OUT, "tables").mkdir(parents=True, exist_ok=True)
    fields = ["run", "eval", "success_rate", "mean_return", "mean_length", "failures", "optimal_length"]
    with open(Path(OUT) / "tables" / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
'''


# ---------- задача 1 ----------

task1 = [
    md("""
# Задача 1. Среда GridWorld

Лабиринт `h x w`, окружённый стеной. Агент видит только цвет клетки, на которой стоит: one-hot вектор длины
`n_colors + 3` (цвета пола, затем цель, препятствие и стена). Действия: 0 вверх, 1 вправо, 2 вниз, 3 влево.
Награда 1 за попадание в цель, тогда эпизод заканчивается (`terminated`), иначе 0. Лимит `4·h·w` шагов (`truncated`).

Код: `src/envs/gridworld.py` и `src/envs/setups.py`. Результаты сохраняются в `outputs/task1/`.
"""),
    code('''
import csv
import os
import sys
from pathlib import Path

# работаем из корня репозитория
if Path.cwd().name == "tasks":
    os.chdir(Path.cwd().parent)
sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np

from src import utils
from src.envs import GridWorldEnv, bfs_distances, make_env, optimal_stats
from src.envs.gridworld import DOWN, LEFT, RIGHT, UP

OUT = Path("outputs/task1")
'''),
    code(SETUPS),
    code('''
CHECKS = []  # результаты проверок, в конце сохраним в outputs/task1/validation/checks.csv


def check(name, ok):
    CHECKS.append({"check": name, "ok": bool(ok)})
    print("OK  " if ok else "ОШИБКА", name)
'''),
    md("## Карты четырёх сред"),
    code('''
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
for i in SETUPS:
    env = make_env(**SETUPS[i], seed=0)
    env.reset(seed=0)
    utils.draw_env(axes[i - 1], env)
    axes[i - 1].set_title(f"Env{i}: {env.h}x{env.w}, цветов пола {env.n_colors}, препятствий {env.obstacle_mask.sum()}")
(OUT / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "figures" / "setups.png", dpi=150, bbox_inches="tight")
plt.show()
'''),
    md("""
## Характеристики сред

Oracle знает свои координаты и идёт кратчайшим путём (BFS) — это ориентир сверху, а не честный бейзлайн: у агентов
в задаче 2 координат нет. Случайная политика — ориентир снизу.
"""),
    code('''
def random_policy(env, n_episodes=200):
    # случайные действия: доля успехов, средняя длина и средняя дисконтированная отдача
    rng = np.random.default_rng(0)
    successes, lengths, returns = [], [], []
    for k in range(n_episodes):
        env.reset(seed=k)
        terminated, truncated = False, False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(int(rng.integers(4)))
        successes.append(terminated)
        lengths.append(env.steps)
        returns.append(0.99 ** (env.steps - 1) * terminated)
    return np.mean(successes), np.mean(lengths), np.mean(returns)
'''),
    code('''
rows = []
for i in SETUPS:
    env = make_env(**SETUPS[i], seed=0)
    free = ~env.obstacle_mask
    dist = bfs_distances(free, env.pos_goal)
    check(f"Env{i}: все свободные клетки достижимы", (dist[free] >= 0).all())
    oracle = optimal_stats(env)
    rand_success, rand_length, rand_return = random_policy(env)
    rows.append({"env": f"Env{i}", "size": f"{env.h}x{env.w}", "floor_colors": env.n_colors,
                 "obstacles": int(env.obstacle_mask.sum()), "free_cells": int(free.sum()),
                 "cells_per_color": round(free.sum() / len(np.unique(env.floor_colors[free])), 2),
                 "max_steps": env.max_steps, "longest_path": int(dist.max()),
                 "oracle_length": round(oracle["length"], 2), "oracle_return": round(oracle["return"], 3),
                 "random_success": round(rand_success, 3), "random_length": round(rand_length, 1),
                 "random_return": round(rand_return, 3)})
    print(rows[-1])
'''),
    code('''
(OUT / "tables").mkdir(parents=True, exist_ok=True)
with open(OUT / "tables" / "env_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
'''),
    code('''
# длина кратчайшего пути до цели из каждой клетки
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for i in SETUPS:
    env = make_env(**SETUPS[i], seed=0)
    dist = bfs_distances(~env.obstacle_mask, env.pos_goal).astype(float)
    dist[env.obstacle_mask] = np.nan
    image = axes[i - 1].imshow(dist, cmap="viridis")
    axes[i - 1].set_title(f"Env{i}: шагов до цели по кратчайшему пути")
    fig.colorbar(image, ax=axes[i - 1], fraction=0.046)
fig.savefig(OUT / "figures" / "bfs_distances.png", dpi=150, bbox_inches="tight")
plt.show()
'''),
    md("## Правила среды на маленькой карте 3x3"),
    code('''
floor = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
obstacles = np.zeros((3, 3), dtype=bool)
obstacles[1, 1] = True
env = GridWorldEnv(floor, (2, 2), n_colors=3, obstacle_mask=obstacles)
print(env.render())
'''),
    code('''
obs, info = env.reset(seed=0, options={"pos": (0, 0)})
obs, reward, terminated, truncated, info = env.step(UP)
check("удар о стену: агент на месте", tuple(info["pos"]) == (0, 0) and info["bumped"])
check("see_obstacle=True: видит цвет стены", np.argmax(obs) == env.wall_color)

env.reset(seed=0, options={"pos": (0, 1)})
obs, reward, terminated, truncated, info = env.step(DOWN)
check("удар о препятствие: агент на месте, видит препятствие", tuple(info["pos"]) == (0, 1) and np.argmax(obs) == env.obstacle_color)

env.reset(seed=0, options={"pos": (2, 1)})
obs, reward, terminated, truncated, info = env.step(RIGHT)
check("вход в цель: награда 1 и terminated", reward == 1.0 and terminated)
'''),
    code('''
# see_obstacle=False: при ударе агент видит цвет своей клетки
env_blind = GridWorldEnv(floor, (2, 2), n_colors=3, obstacle_mask=obstacles, see_obstacle=False)
env_blind.reset(seed=0, options={"pos": (0, 0)})
obs, reward, terminated, truncated, info = env_blind.step(LEFT)
check("see_obstacle=False: видит свою клетку", np.argmax(obs) == 0)

# лимит шагов
env_short = GridWorldEnv(floor, (2, 2), n_colors=3, obstacle_mask=obstacles, max_steps=3)
env_short.reset(seed=0, options={"pos": (0, 0)})
flags = [env_short.step(UP)[3] for _ in range(3)]
check("truncated ровно на 3-м шаге", flags == [False, False, True])
'''),
    md("## Стартовые позиции\n\nСтарт не может быть на цели, препятствии или вне карты: в этом случае `ValueError`."),
    code('''
for bad in [(2, 2), (1, 1), (-1, 0), (3, 0)]:
    try:
        GridWorldEnv(floor, (2, 2), n_colors=3, obstacle_mask=obstacles, pos_agent=bad)
        check(f"старт {bad} запрещён", False)
    except ValueError as error:
        check(f"старт {bad} запрещён: {error}", True)
'''),
    code('''
starts = set()
for s in range(500):
    obs, info = env.reset(seed=s)
    starts.add(tuple(info["pos"]))
check("случайный старт только со свободных клеток и не с цели", (1, 1) not in starts and (2, 2) not in starts)
check("встречаются все 7 допустимых клеток", len(starts) == 7)

# веса стартов: у (0, 0) вес 10, у остальных 6 клеток по 1
weights = np.ones((3, 3))
weights[0, 0] = 10
env_w = GridWorldEnv(floor, (2, 2), n_colors=3, obstacle_mask=obstacles, pos_agent=weights)
share = np.mean([tuple(env_w.reset(seed=s)[1]["pos"]) == (0, 0) for s in range(2000)])
check(f"доля старта из (0, 0) = {share:.2f}, ожидается {10 / 16:.2f}", abs(share - 10 / 16) < 0.05)
'''),
    md("## Цветовое и координатное наблюдения\n\nКоординатный режим `obs_mode=\"coords\"` нужен только для отладки: в нём задача становится полностью наблюдаемой."),
    code('''
env_color = make_env(**SETUPS[2], seed=0)
env_coords = make_env(**SETUPS[2], seed=0, obs_mode="coords")
obs_color, info = env_color.reset(seed=1)
obs_coords, _ = env_coords.reset(seed=1)
print("позиция", tuple(info["pos"]), "цвет клетки", info["color"])
print("цветовое наблюдение:", obs_color)
print("координатное наблюдение: единица на месте", np.argmax(obs_coords))
check("цветовое наблюдение — one-hot цвета", obs_color.sum() == 1 and np.argmax(obs_color) == info["color"])
check("координатное наблюдение — one-hot позиции", np.argmax(obs_coords) == info["pos"][0] * 5 + info["pos"][1])
'''),
    md("## Воспроизводимость"),
    code('''
a = make_env(**SETUPS[4], seed=0)
b = make_env(**SETUPS[4], seed=0)
c = make_env(**SETUPS[4], seed=1)
check("одинаковый seed — одинаковая карта и цель", np.array_equal(a.color_map, b.color_map) and a.pos_goal == b.pos_goal)
check("разный seed — разные карты", not np.array_equal(a.color_map, c.color_map))
starts_a = [tuple(a.reset(seed=s)[1]["pos"]) for s in range(20)]
starts_b = [tuple(b.reset(seed=s)[1]["pos"]) for s in range(20)]
check("одинаковый seed в reset — одинаковые старты", starts_a == starts_b)
'''),
    code('''
(OUT / "validation").mkdir(parents=True, exist_ok=True)
with open(OUT / "validation" / "checks.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["check", "ok"])
    writer.writeheader()
    writer.writerows(CHECKS)
print("проверок:", len(CHECKS), "| с ошибкой:", sum(not c["ok"] for c in CHECKS))
'''),
]


# ---------- задача 2 ----------

task2 = [
    md("""
# Задача 2. PPO и Discrete SAC

Сравниваем PPO и Discrete SAC, каждый с памятью (GRU) и без неё (MLP), на четырёх средах.
Дополнительно — DSAC с обычной Q вместо мягкой (`soft_q=False`) на Env2 и Env4.
Все прогоны с одним сидом. Результаты: `outputs/task2/` (`runs/` — прогоны, `figures/`, `eval/`, `tables/`).
"""),
    code(IMPORTS),
    code(MODE),
    code(SETUPS),
    md("## Гиперпараметры"),
    code(PPO_CFG),
    code(DSAC_CFG),
    code(BUDGET + '\n# smoke-прогоны пишем отдельно, чтобы не смешать их с настоящими\nOUT = "outputs/task2" if RUN_MODE == "full" else "runs/smoke/task2"\n'),
    md("## Список экспериментов"),
    code('''
EXPERIMENTS = []
for i in SETUPS:
    for memory in [True, False]:
        tag = "gru" if memory else "mlp"
        for algo, base in [("ppo", ppo_cfg), ("dsac", dsac_cfg)]:
            name = f"{algo}_env{i}_{tag}_seed0"
            cfg = {**base, **SETUPS[i], "memory": memory, "episodes": EPISODES[RUN_MODE][algo][i],
                   "budget": RUN_MODE, "out_dir": f"{OUT}/runs", "run_name": name}
            EXPERIMENTS.append({"name": name, "algo": algo, "cfg": cfg})

# требование задания: DSAC с обычной Q вместо мягкой
for i in [2, 4]:
    name = f"dsac_env{i}_gru_hardq_seed0"
    cfg = {**dsac_cfg, **SETUPS[i], "soft_q": False, "episodes": EPISODES[RUN_MODE]["dsac"][i],
           "budget": RUN_MODE, "out_dir": f"{OUT}/runs", "run_name": name}
    EXPERIMENTS.append({"name": name, "algo": "dsac", "cfg": cfg})

for exp in EXPERIMENTS:
    print(exp["name"], "эпизодов:", exp["cfg"]["episodes"], exp["cfg"]["device"])
'''),
    md("## Обучение и оценка\n\nЗавершённые прогоны пропускаются, поэтому ячейку можно перезапускать."),
    code(RUN_ONE),
    code(RUN_ALL),
    md("## Графики обучения\n\nСтроятся по сохранённым `metrics.csv`, поэтому их можно перерисовать без обучения."),
    code(LOAD),
    code(CURVE),
    code(COMPARE),
    code('''
if "plots" in RUN_STAGES:
    metrics = [("return", "средняя дисконтированная отдача"), ("success", "доля успешных эпизодов"),
               ("length", "длина эпизода"), ("loss", "полный лосс"), ("entropy", "энтропия стратегии"),
               ("explore_prob", "вероятность нежадного действия")]
    for algo in ["ppo", "dsac"]:
        runs = {f"Env{i}": [f"{OUT}/runs/{algo}_env{i}_gru_seed0", f"{OUT}/runs/{algo}_env{i}_mlp_seed0"] for i in SETUPS}
        for key, ylabel in metrics:
            plot_compare(runs, ["GRU", "MLP"], key, ylabel, f"{algo}_{key}.png")
'''),
    code('''
if "plots" in RUN_STAGES:
    # компоненты лосса
    runs = {f"Env{i}": [f"{OUT}/runs/ppo_env{i}_gru_seed0", f"{OUT}/runs/ppo_env{i}_mlp_seed0"] for i in SETUPS}
    plot_compare(runs, ["GRU", "MLP"], "policy_loss", "лосс актора", "ppo_policy_loss.png")
    plot_compare(runs, ["GRU", "MLP"], "value_loss", "лосс критика", "ppo_value_loss.png")
    runs = {f"Env{i}": [f"{OUT}/runs/dsac_env{i}_gru_seed0", f"{OUT}/runs/dsac_env{i}_mlp_seed0"] for i in SETUPS}
    plot_compare(runs, ["GRU", "MLP"], "q_loss", "лосс критиков", "dsac_q_loss.png")
    plot_compare(runs, ["GRU", "MLP"], "actor_loss", "лосс актора", "dsac_actor_loss.png")
'''),
    code('''
if "plots" in RUN_STAGES:
    # PPO против DSAC, оба с памятью
    runs = {f"Env{i}": [f"{OUT}/runs/ppo_env{i}_gru_seed0", f"{OUT}/runs/dsac_env{i}_gru_seed0"] for i in SETUPS}
    plot_compare(runs, ["PPO", "DSAC"], "return", "средняя дисконтированная отдача", "ppo_vs_dsac_return.png")
    # мягкая Q против обычной
    runs = {f"Env{i}": [f"{OUT}/runs/dsac_env{i}_gru_seed0", f"{OUT}/runs/dsac_env{i}_gru_hardq_seed0"] for i in [2, 4]}
    plot_compare(runs, ["soft_q=True", "soft_q=False"], "return", "средняя дисконтированная отдача", "dsac_soft_q_return.png")
'''),
    md("""
## Итоговая оценка

100 эпизодов на одних и тех же стартах (сиды 10000–10099), жадно и с сэмплированием действий.
`optimal_length` — средняя длина пути у oracle с доступом к координатам.
"""),
    code(TABLE),
    code('''
if "plots" in RUN_STAGES:
    save_table(eval_rows([f"{OUT}/runs/{exp['name']}" for exp in EXPERIMENTS]), "eval_summary.csv")
'''),
]


# ---------- задача 3 ----------

task3 = [
    md("""
# Задача 3. Цвета клеток картинками из MNIST

Среда `MnistGridWorldEnv` (`src/envs/mnist_env.py`) — копия обычной, но вместо one-hot цвета агент видит случайную
картинку цифры этого цвета: цвет 2 — случайная двойка из MNIST. Картинка 14x14, на вход сети идёт вектор из 196 чисел.
Цифр всего 10, поэтому цветов пола не больше 7: Env1 с 25 цветами исключён, берём Env2–Env4.

Главное сравнение — one-hot против MNIST на тех же картах с теми же гиперпараметрами (one-hot прогоны из `outputs/task2/runs`).
"""),
    code(IMPORTS),
    code(MODE),
    code(SETUPS + 'del SETUPS[1]  # 25 цветов не помещаются в 10 цифр\n'),
    md("## Проверка наблюдений\n\nКартинка должна быть из класса, равного цвету клетки; размер 196, тип float32, значения от 0 до 1."),
    code('''
images = load_mnist()
env = make_env(**SETUPS[3], seed=0, mnist=True)
obs, info = env.reset(seed=0)
print("размер", obs.shape, "тип", obs.dtype, "значения от", obs.min(), "до", obs.max())

for t in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # наблюдение должно в точности совпадать с одной из картинок своей цифры
    assert np.abs(images[info["color"]] - obs).sum(axis=1).min() == 0
    if terminated or truncated:
        obs, info = env.reset()
print("200 шагов: все картинки из правильного класса")
'''),
    code('''
fig, axes = plt.subplots(10, 8, figsize=(8, 10))
for digit in range(10):
    for j in range(8):
        picture = images[digit][np.random.randint(len(images[digit]))]
        axes[digit, j].imshow(picture.reshape(14, 14), cmap="gray")
        axes[digit, j].axis("off")
    axes[digit, 0].set_title(f"цвет {digit}", fontsize=8, loc="left")
Path("outputs/task3/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/task3/figures/mnist_colors.png", dpi=150, bbox_inches="tight")
plt.show()
'''),
    md("## Гиперпараметры\n\nТе же, что в задаче 2, только `\"mnist\": True`."),
    code(PPO_CFG),
    code(DSAC_CFG),
    code(BUDGET + '\n# smoke-прогоны пишем отдельно, чтобы не смешать их с настоящими\nOUT = "outputs/task3" if RUN_MODE == "full" else "runs/smoke/task3"\nTASK2 = "outputs/task2" if RUN_MODE == "full" else "runs/smoke/task2"\n'),
    md("## Список экспериментов"),
    code('''
EXPERIMENTS = []
for i in SETUPS:
    for algo, base in [("ppo", ppo_cfg), ("dsac", dsac_cfg)]:
        name = f"{algo}_mnist_env{i}_gru_seed0"
        cfg = {**base, **SETUPS[i], "mnist": True, "episodes": EPISODES[RUN_MODE][algo][i],
               "budget": RUN_MODE, "out_dir": f"{OUT}/runs", "run_name": name}
        EXPERIMENTS.append({"name": name, "algo": algo, "cfg": cfg})

for exp in EXPERIMENTS:
    print(exp["name"], "эпизодов:", exp["cfg"]["episodes"], exp["cfg"]["device"])
'''),
    md("## Обучение и оценка"),
    code(RUN_ONE),
    code(RUN_ALL),
    md("## One-hot против MNIST"),
    code(LOAD),
    code(CURVE),
    code(COMPARE),
    code('''
if "plots" in RUN_STAGES:
    for algo in ["ppo", "dsac"]:
        runs = {f"Env{i}": [f"{TASK2}/runs/{algo}_env{i}_gru_seed0", f"{OUT}/runs/{algo}_mnist_env{i}_gru_seed0"] for i in SETUPS}
        plot_compare(runs, ["one-hot", "MNIST"], "return", "средняя дисконтированная отдача", f"{algo}_onehot_vs_mnist_return.png")
        plot_compare(runs, ["one-hot", "MNIST"], "length", "длина эпизода", f"{algo}_onehot_vs_mnist_length.png")
'''),
    md("## Итоговая оценка"),
    code(TABLE),
    code('''
if "plots" in RUN_STAGES:
    paths = [f"{OUT}/runs/{exp['name']}" for exp in EXPERIMENTS]
    paths += [f"{TASK2}/runs/{algo}_env{i}_gru_seed0" for algo in ["ppo", "dsac"] for i in SETUPS]
    save_table(eval_rows(paths), "eval_summary.csv")
'''),
    md("""
## Что проверяет замена one-hot на MNIST

Черновик ответов на вопросы задания: перепроверить по графикам и таблице и переписать своими словами.

Раньше один цвет всегда давал одно и то же наблюдение. Теперь одному цвету
соответствуют тысячи разных картинок, и на каждом шаге приходит новая. Агенту нужно сначала распознать цифру и только
потом ориентироваться. Это проверяет, справляется ли метод со сложными и зашумлёнными наблюдениями.

## Почему задача стала сложнее

Вход вырос с 8–10 чисел до 196, наблюдения шумные, а сигнал для обучения тот же —
редкая награда за цель. Распознавать цифры сеть учится только через RL-лосс, без меток классов.

## Нужен ли более сложный энкодер

Скорее нет: картинки 14x14, классов 10, с этим справляется даже линейный
классификатор. Поэтому архитектура не менялась: первый слой `Linear(196 + 4 -> 128)` и `Tanh`.

## Как изменение наблюдений повлияло на результаты

Заполнить по графикам `outputs/task3/figures/*onehot_vs_mnist*` и таблице.
"""),
]

if __name__ == "__main__":
    save(task1, "tasks/task1.ipynb")
    save(task2, "tasks/task2.ipynb")
    save(task3, "tasks/task3.ipynb")
