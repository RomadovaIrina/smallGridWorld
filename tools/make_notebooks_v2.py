"""Собирает ноутбуки второго и третьего этапа экспериментов:
    task2_v2, task3_v2 — обучение до выхода на плато (seed 0), чтобы узнать, сколько шагов нужно каждому алгоритму;
    task2_v3, task3_v3 — одинаковый бюджет шагов среды N для всех, 3 сида.

    uv run python tools/make_notebooks_v2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebooks import (DSAC_CFG, IMPORTS, LOAD, MODE, PPO_CFG, RUN_ALL, RUN_ONE, SETUPS, TABLE, code, md,
                            save)

EVAL = '''
EVAL_EPISODES = {"full": 100, "smoke": 5}
'''

ROLLING = '''
def rolling_mean(values, window=50):
    # среднее за последние window эпизодов (пропуски nan не учитываем)
    result = []
    for i in range(len(values)):
        last = values[max(0, i - window + 1):i + 1]
        last = last[~np.isnan(last)]
        result.append(np.mean(last) if len(last) > 0 else np.nan)
    return np.array(result)
'''

COMPARE_STEPS = '''
def plot_compare_steps(runs, labels, key, ylabel, filename):
    # как в задаче 2, но по оси X — шаги среды; линия — скользящее среднее за 50 эпизодов (1 сид)
    colors = ["tab:red", "tab:blue", "tab:green"]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.3), squeeze=False)
    for ax, env_name in zip(axes[0], runs):
        for path, label, color in zip(runs[env_name], labels, colors):
            data = load_metrics(path)
            if data is not None:
                ax.plot(data["env_steps"], rolling_mean(data[key]), color=color, linewidth=2, label=label)
        ax.set_title(env_name)
        ax.set_xlabel("шаги среды")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    Path(OUT, "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(OUT) / "figures" / filename, dpi=150, bbox_inches="tight")
    plt.show()
'''

CONVERGENCE_TABLE = '''
def convergence_rows(run_paths):
    # когда каждый прогон вышел на плато (или упёрся в потолок эпизодов) и как он потом сыграл в оценке
    rows = []
    for path in run_paths:
        status_file = Path(path) / "status.json"
        if not status_file.exists():
            rows.append({"run": Path(path).name, "converged": "pending"})
            continue
        status = json.loads(status_file.read_text(encoding="utf-8"))
        row = {"run": Path(path).name, "converged": status.get("converged"),
               "episodes": status["episodes_done"], "env_steps": status.get("env_steps"),
               "time_min": round(status.get("time_sec", 0) / 60, 1)}
        eval_file = Path(path) / "eval.json"
        if eval_file.exists():
            result = json.loads(eval_file.read_text(encoding="utf-8"))
            row["greedy_success"] = round(result["greedy"]["success_rate"], 3)
            row["greedy_length"] = round(result["greedy"]["mean_length"], 1)
            row["optimal_length"] = round(result["greedy"]["optimal_length"], 2)
        rows.append(row)
    return rows


def save_rows(rows, filename):
    Path(OUT, "tables").mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(Path(OUT) / "tables" / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
'''

SEED_PLOTS = '''
def seed_curve(path, key, grid):
    # сглаженная кривая одного прогона, пересчитанная на общую сетку шагов среды
    data = load_metrics(path)
    if data is None:
        return None
    return np.interp(grid, data["env_steps"], rolling_mean(data[key]), right=np.nan)


def plot_seeds(runs, key, ylabel, filename):
    # runs: {"Env2": {"PPO": [пути по сидам], "DSAC": [...]}, ...}; линия — среднее по сидам,
    # полоса — стандартное отклонение между сидами (это настоящий разброс между запусками)
    colors = ["tab:red", "tab:blue", "tab:green"]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.3), squeeze=False)
    for ax, env_name in zip(axes[0], runs):
        grid = np.linspace(0, BUDGET[int(env_name[-1])], 200)
        for (label, paths), color in zip(runs[env_name].items(), colors):
            curves = [seed_curve(path, key, grid) for path in paths]
            curves = [c for c in curves if c is not None]
            if len(curves) == 0:
                continue
            mean = np.nanmean(curves, axis=0)
            std = np.nanstd(curves, axis=0)
            ax.plot(grid, mean, color=color, linewidth=2, label=f"{label} ({len(curves)} сида)")
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.25)
        ax.set_title(env_name)
        ax.set_xlabel("шаги среды")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    Path(OUT, "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(OUT) / "figures" / filename, dpi=150, bbox_inches="tight")
    plt.show()
'''

SEED_TABLE = '''
def seeds_rows(groups):
    # groups: {"ppo_env2_gru": [пути по сидам], ...}; среднее и разброс итоговой оценки по сидам
    rows = []
    for name, paths in groups.items():
        results = []
        for path in paths:
            eval_file = Path(path) / "eval.json"
            if eval_file.exists():
                results.append(json.loads(eval_file.read_text(encoding="utf-8")))
        if len(results) == 0:
            rows.append({"config": name, "seeds": 0})
            continue
        row = {"config": name, "seeds": len(results)}
        for mode in ["greedy", "stochastic"]:
            success = [r[mode]["success_rate"] for r in results]
            length = [r[mode]["mean_length"] for r in results]
            row[f"{mode}_success_mean"] = round(np.mean(success), 3)
            row[f"{mode}_success_std"] = round(np.std(success), 3)
            row[f"{mode}_length_mean"] = round(np.mean(length), 1)
            row[f"{mode}_length_std"] = round(np.std(length), 1)
        row["optimal_length"] = round(results[0]["greedy"]["optimal_length"], 2)
        rows.append(row)
    return rows
'''


def experiments_cell(mnist, seeds_code, extra_cfg, hardq):
    # ячейка со списком экспериментов: одинаковая логика для v2 и v3, задачи 2 и 3
    memories = "[True]" if mnist else "[True, False]"
    prefix = "_mnist" if mnist else ""
    mnist_cfg = '"mnist": True, ' if mnist else ""
    text = f'''
EXPERIMENTS = []
for seed in {seeds_code}:
    for i in SETUPS:
        for memory in {memories}:
            tag = "gru" if memory else "mlp"
            for algo, base in [("ppo", ppo_cfg), ("dsac", dsac_cfg)]:
                name = f"{{algo}}{prefix}_env{{i}}_{{tag}}_seed{{seed}}"
                cfg = {{**base, **SETUPS[i], {mnist_cfg}"memory": memory, "seed": seed, {extra_cfg}
                       "budget": RUN_MODE, "out_dir": f"{{OUT}}/runs", "run_name": name}}
                EXPERIMENTS.append({{"name": name, "algo": algo, "cfg": cfg}})
'''
    if hardq:
        extra_cfg = extra_cfg.replace("BEST[algo]", 'BEST["dsac"]')  # в этой ячейке всегда DSAC
        text += f'''
    # требование задания: DSAC с обычной Q вместо мягкой
    for i in [2, 4]:
        name = f"dsac_env{{i}}_gru_hardq_seed{{seed}}"
        cfg = {{**dsac_cfg, **SETUPS[i], "soft_q": False, "seed": seed, {extra_cfg}
               "budget": RUN_MODE, "out_dir": f"{{OUT}}/runs", "run_name": name}}
        EXPERIMENTS.append({{"name": name, "algo": "dsac", "cfg": cfg}})
'''
    text += '''
for exp in EXPERIMENTS:
    print(exp["name"], exp["cfg"]["device"])
'''
    return text


V2_BUDGET = '''
# учимся до выхода на плато: каждые 100 эпизодов сравниваем среднюю отдачу за последние conv_window эпизодов
# с предыдущими; если прирост меньше conv_tol два раза подряд — остановка.
# Потолок эпизодов — на случай, если алгоритм так и не сойдётся (тогда в status.json будет converged = false)
MAX_EPISODES = {"full": {1: 4000, 2: 4000, 3: 8000, 4: 8000}, "smoke": {1: 30, 2: 30, 3: 20, 4: 20}}
# на картах 10x10 плато ищем только после 2000 эпизодов: DSAC сначала долго «стоит» около нуля,
# и без этой паузы критерий принимал застой за схождение
MIN_EPISODES = {"full": {1: 0, 2: 0, 3: 2000, 4: 2000}, "smoke": {1: 0, 2: 0, 3: 0, 4: 0}}
CONVERGENCE = {
    "full": {"stop_when_converged": True, "conv_window": 200, "conv_tol": 0.01},
    "smoke": {"stop_when_converged": True, "conv_window": 5, "conv_tol": 0.01},
}
'''

TUNED_VALUES = """
ALPHAS = [0.01, 0.02, 0.05]  # alpha DSAC в переборе (0.005 — значение из этапа 2)
ENT_COEFS = [0.001, 0.03, 0.1]  # ent_coef PPO в переборе (0.01 — значение из этапа 2)
"""

PICK_BEST = """
def pick_best(candidates):
    # candidates: {значение параметра: папка прогона}; лучшее — с наибольшей средней отдачей в жадной оценке
    best_value, best_path, best_return = None, None, -1.0
    for value, path in candidates.items():
        result = json.loads((Path(path) / "eval.json").read_text())
        if result["greedy"]["mean_return"] > best_return:
            best_value, best_path, best_return = value, path, result["greedy"]["mean_return"]
    return best_value, best_path


def steps_to_plateau(path):
    # сколько шагов среды прогону понадобилось до плато (если не сошёлся — все его шаги)
    status = json.loads((Path(path) / "status.json").read_text())
    if status["converged"]:
        return status["converged_at"]["env_steps"]
    return status["env_steps"]
"""

CHOICE = """
V2 = "outputs/V2NAME" if RUN_MODE == "full" else "runs/smoke/V2NAME"
V2_TASK2 = "outputs/task2_v2" if RUN_MODE == "full" else "runs/smoke/task2_v2"
TUNING = "outputs/task2_tuning" if RUN_MODE == "full" else "runs/smoke/task2_tuning"

# на 5x5 значения как в этапе 2; на 10x10 — лучшие по перебору (на обычных картах, без MNIST)
BEST = {"ppo": {}, "dsac": {}}
TUNED = {"ppo": {}, "dsac": {}}  # прогон с выбранным значением
for i in SETUPS:
    BEST["ppo"][i] = {"ent_coef": 0.01}
    BEST["dsac"][i] = {"alpha": 0.005}
    TUNED["ppo"][i] = f"{V2_TASK2}/runs/ppo_env{i}_gru_seed0"
    TUNED["dsac"][i] = f"{V2_TASK2}/runs/dsac_env{i}_gru_seed0"
for i in [3, 4]:
    candidates = {0.01: TUNED["ppo"][i]}
    for value in ENT_COEFS:
        candidates[value] = f"{TUNING}/runs/ppo_env{i}_gru_ent{value}"
    value, TUNED["ppo"][i] = pick_best(candidates)
    BEST["ppo"][i] = {"ent_coef": value}
    candidates = {0.005: TUNED["dsac"][i]}
    for value in ALPHAS:
        candidates[value] = f"{TUNING}/runs/dsac_env{i}_gru_alpha{value}"
    value, TUNED["dsac"][i] = pick_best(candidates)
    BEST["dsac"][i] = {"alpha": value}
print("выбранные значения:", BEST)
"""

BUDGET_MNIST = """
# бюджет N — больший из «шагов до плато» у PPO и DSAC с памятью на MNIST-картах из этапа 2
BUDGET = {}
for i in SETUPS:
    BUDGET[i] = max(steps_to_plateau(f"{V2}/runs/ppo_mnist_env{i}_gru_seed0"),
                    steps_to_plateau(f"{V2}/runs/dsac_mnist_env{i}_gru_seed0"))
print("бюджет шагов по средам:", BUDGET)

SEEDS = [0, 1, 2]
"""

BUDGET_ONEHOT = """
# бюджет N — больший из «шагов до плато» у PPO и DSAC с выбранными значениями
BUDGET = {}
for i in SETUPS:
    BUDGET[i] = max(steps_to_plateau(TUNED["ppo"][i]), steps_to_plateau(TUNED["dsac"][i]))
print("бюджет шагов по средам:", BUDGET)

SEEDS = [0, 1, 2]
"""


def v3_choice_cell(mnist):
    # выбор alpha / ent_coef по перебору task2_tuning и бюджета шагов N для этапа 3
    v2 = "task3_v2" if mnist else "task2_v2"
    return CHOICE.replace("V2NAME", v2) + (BUDGET_MNIST if mnist else BUDGET_ONEHOT)


TUNING_EXPERIMENTS = """
EXPERIMENTS = []
for i in [3, 4]:
    common = {**SETUPS[i], "episodes": MAX_EPISODES[RUN_MODE][i], **CONVERGENCE[RUN_MODE],
              "min_episodes": MIN_EPISODES[RUN_MODE][i], "budget": RUN_MODE, "out_dir": f"{OUT}/runs"}
    for value in ALPHAS:
        name = f"dsac_env{i}_gru_alpha{value}"
        EXPERIMENTS.append({"name": name, "algo": "dsac", "cfg": {**dsac_cfg, **common, "alpha": value, "run_name": name}})
    for value in ENT_COEFS:
        name = f"ppo_env{i}_gru_ent{value}"
        EXPERIMENTS.append({"name": name, "algo": "ppo", "cfg": {**ppo_cfg, **common, "ent_coef": value, "run_name": name}})

for exp in EXPERIMENTS:
    print(exp["name"], exp["cfg"]["device"])
"""

TUNING_TABLE = """
if "plots" in RUN_STAGES:
    paths = []
    for i in [3, 4]:
        paths.append(f"{V2}/runs/dsac_env{i}_gru_seed0")
        paths += [f"{OUT}/runs/dsac_env{i}_gru_alpha{value}" for value in ALPHAS]
        paths.append(f"{V2}/runs/ppo_env{i}_gru_seed0")
        paths += [f"{OUT}/runs/ppo_env{i}_gru_ent{value}" for value in ENT_COEFS]
    save_rows(convergence_rows(paths), "tuning.csv")
"""

TUNING_PLOT = """
if "plots" in RUN_STAGES:
    # отдача по шагам среды при разных значениях параметра
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for algo, values, base, key in [("dsac", ALPHAS, 0.005, "alpha"), ("ppo", ENT_COEFS, 0.01, "ent")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        for ax, i in zip(axes, [3, 4]):
            paths = {base: f"{V2}/runs/{algo}_env{i}_gru_seed0"}
            for value in values:
                paths[value] = f"{OUT}/runs/{algo}_env{i}_gru_{key}{value}"
            for (value, path), color in zip(paths.items(), colors):
                data = load_metrics(path)
                if data is not None:
                    ax.plot(data["env_steps"], rolling_mean(data["return"]), color=color, label=f"{key} = {value}")
            ax.set_title(f"{algo.upper()} + GRU, Env{i}")
            ax.set_xlabel("шаги среды")
            ax.set_ylabel("средняя дисконтированная отдача")
            ax.legend(fontsize=8)
        Path(OUT, "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(OUT) / "figures" / f"{algo}_{key}_sweep.png", dpi=150, bbox_inches="tight")
        plt.show()
"""


def tuning_notebook():
    cells = [
        md("""
# Задача 2: перебор параметра исследования на картах 10x10

В этапе 2 DSAC с температурой alpha = 0.005 на больших картах почти сразу становится детерминированным
(энтропия около 0) и редко находит цель. Перебираем параметр, который отвечает за исследование, у обоих алгоритмов:
- DSAC: температура `alpha` — 0.01, 0.02, 0.05 (0.005 уже есть в `task2_v2`);
- PPO: коэффициент бонуса энтропии `ent_coef` — 0.001, 0.03, 0.1 (0.01 уже есть в `task2_v2`).

Оба с памятью (GRU), Env3 и Env4, обучение до плато по тем же правилам, что в `task2_v2`. Лучшие значения
используются в `task2_v3.ipynb`. Результаты — `outputs/task2_tuning/`.
"""),
        code(IMPORTS),
        code(MODE),
        code(SETUPS),
        md("## Гиперпараметры"),
        code(PPO_CFG),
        code(DSAC_CFG),
        code(V2_BUDGET + TUNED_VALUES + EVAL
             + '\nOUT = "outputs/task2_tuning" if RUN_MODE == "full" else "runs/smoke/task2_tuning"\n'
             + 'V2 = "outputs/task2_v2" if RUN_MODE == "full" else "runs/smoke/task2_v2"\n'),
        md("## Список экспериментов"),
        code(TUNING_EXPERIMENTS),
        md("## Обучение и оценка"),
        code(RUN_ONE),
        code(RUN_ALL),
        md("## Результаты перебора"),
        code(LOAD),
        code(ROLLING),
        code(CONVERGENCE_TABLE),
        code(TUNING_TABLE),
        code(TUNING_PLOT),
    ]
    save(cells, "tasks/task2_tuning.ipynb")


def v2_notebook(task, mnist):
    title = "Задача 3" if mnist else "Задача 2"
    out = f"{task}_v2"
    setups = SETUPS + ('del SETUPS[1]  # 25 цветов не помещаются в 10 цифр\n' if mnist else "")
    cells = [
        md(f"""
# {title}, этап 2: обучение до схождения

В первой версии (`{task}.ipynb`) PPO и DSAC получили разное число эпизодов, и DSAC на картах 10x10 не доучился.
Здесь каждый прогон учится, пока не выйдет на плато (или не упрётся в потолок эпизодов), — так мы узнаём,
сколько шагов среды нужно каждому алгоритму. Гиперпараметры те же, seed 0. Результаты — `outputs/{out}/`.

По этим точкам схождения выбирается одинаковый бюджет шагов для честного сравнения в `{task}_v3.ipynb`.
"""),
        code(IMPORTS),
        code(MODE),
        code(setups),
        md("## Гиперпараметры"),
        code(PPO_CFG),
        code(DSAC_CFG),
        code(V2_BUDGET + EVAL + f'\nOUT = "outputs/{out}" if RUN_MODE == "full" else "runs/smoke/{out}"\n'),
        md("## Список экспериментов"),
        code(experiments_cell(mnist, "[0]",
                              '"episodes": MAX_EPISODES[RUN_MODE][i], **CONVERGENCE[RUN_MODE], "min_episodes": MIN_EPISODES[RUN_MODE][i],', hardq=not mnist)),
        md("## Обучение и оценка"),
        code(RUN_ONE),
        code(RUN_ALL),
        md("## Когда алгоритмы сошлись"),
        code(LOAD),
        code(CONVERGENCE_TABLE),
        code('''
if "plots" in RUN_STAGES:
    save_rows(convergence_rows([f"{OUT}/runs/{exp['name']}" for exp in EXPERIMENTS]), "convergence.csv")
'''),
        md("## Графики обучения по шагам среды"),
        code(ROLLING),
        code(COMPARE_STEPS),
    ]
    prefix = "_mnist" if mnist else ""
    cells.append(code(f'''
if "plots" in RUN_STAGES:
    runs = {{f"Env{{i}}": [f"{{OUT}}/runs/ppo{prefix}_env{{i}}_gru_seed0", f"{{OUT}}/runs/dsac{prefix}_env{{i}}_gru_seed0"] for i in SETUPS}}
    for key, ylabel in [("return", "средняя дисконтированная отдача"), ("success", "доля успешных эпизодов"),
                        ("length", "длина эпизода"), ("entropy", "энтропия стратегии"),
                        ("explore_prob", "вероятность нежадного действия"), ("loss", "полный лосс")]:
        plot_compare_steps(runs, ["PPO + GRU", "DSAC + GRU"], key, ylabel, f"ppo_vs_dsac_{{key}}.png")
'''))
    if not mnist:
        cells.append(code('''
if "plots" in RUN_STAGES:
    for algo in ["ppo", "dsac"]:
        runs = {f"Env{i}": [f"{OUT}/runs/{algo}_env{i}_gru_seed0", f"{OUT}/runs/{algo}_env{i}_mlp_seed0"] for i in SETUPS}
        plot_compare_steps(runs, ["GRU", "MLP"], "return", "средняя дисконтированная отдача", f"{algo}_memory_return.png")
    runs = {f"Env{i}": [f"{OUT}/runs/dsac_env{i}_gru_seed0", f"{OUT}/runs/dsac_env{i}_gru_hardq_seed0"] for i in [2, 4]}
    plot_compare_steps(runs, ["soft_q=True", "soft_q=False"], "return", "средняя дисконтированная отдача", "dsac_soft_q_return.png")
'''))
    cells += [
        md("## Итоговая оценка"),
        code(TABLE),
        code('''
if "plots" in RUN_STAGES:
    save_table(eval_rows([f"{OUT}/runs/{exp['name']}" for exp in EXPERIMENTS]), "eval_summary.csv")
'''),
    ]
    save(cells, f"tasks/{out}.ipynb")


def v3_notebook(task, mnist):
    title = "Задача 3" if mnist else "Задача 2"
    out = f"{task}_v3"
    prefix = "_mnist" if mnist else ""
    setups = SETUPS + ('del SETUPS[1]  # 25 цветов не помещаются в 10 цифр\n' if mnist else "")
    cells = [
        md(f"""
# {title}, этап 3: одинаковый бюджет шагов и несколько сидов

Параметр исследования (alpha у DSAC, ent_coef у PPO) на картах 10x10 — лучший по перебору `task2_tuning.ipynb`.
Бюджет шагов среды N для каждой среды — больший из «шагов до плато» у PPO и DSAC с памятью (с выбранными значениями).
Все прогоны учатся ровно N шагов, поэтому оба алгоритма доучены, а бюджет у них одинаковый.
Каждая конфигурация — 3 сида (0, 1, 2); на графиках среднее по сидам и разброс между ними. Результаты — `outputs/{out}/`.
"""),
        code(IMPORTS),
        code(MODE),
        code(setups),
        md("## Гиперпараметры"),
        code(PPO_CFG),
        code(DSAC_CFG),
        code(TUNED_VALUES + EVAL + f'\nOUT = "outputs/{out}" if RUN_MODE == "full" else "runs/smoke/{out}"\n'),
        md("## Параметр исследования и бюджет шагов"),
        code(PICK_BEST),
        code(v3_choice_cell(mnist)),
        md("## Список экспериментов"),
        code(experiments_cell(mnist, "SEEDS", '"episodes": 1_000_000, "max_env_steps": BUDGET[i], **BEST[algo][i],',
                              hardq=not mnist)),
        md("## Обучение и оценка"),
        code(RUN_ONE),
        code(RUN_ALL),
        md("## Графики: среднее по сидам"),
        code(LOAD),
        code(ROLLING),
        code(SEED_PLOTS),
        code(f'''
if "plots" in RUN_STAGES:
    runs = {{}}
    for i in SETUPS:
        runs[f"Env{{i}}"] = {{
            "PPO + GRU": [f"{{OUT}}/runs/ppo{prefix}_env{{i}}_gru_seed{{s}}" for s in SEEDS],
            "DSAC + GRU": [f"{{OUT}}/runs/dsac{prefix}_env{{i}}_gru_seed{{s}}" for s in SEEDS],
        }}
    for key, ylabel in [("return", "средняя дисконтированная отдача"), ("success", "доля успешных эпизодов"),
                        ("length", "длина эпизода"), ("entropy", "энтропия стратегии"),
                        ("explore_prob", "вероятность нежадного действия"), ("loss", "полный лосс")]:
        plot_seeds(runs, key, ylabel, f"ppo_vs_dsac_{{key}}.png")
'''),
    ]
    if not mnist:
        cells.append(code('''
if "plots" in RUN_STAGES:
    for algo in ["ppo", "dsac"]:
        runs = {}
        for i in SETUPS:
            runs[f"Env{i}"] = {"GRU": [f"{OUT}/runs/{algo}_env{i}_gru_seed{s}" for s in SEEDS],
                               "MLP": [f"{OUT}/runs/{algo}_env{i}_mlp_seed{s}" for s in SEEDS]}
        plot_seeds(runs, "return", "средняя дисконтированная отдача", f"{algo}_memory_return.png")
    runs = {}
    for i in [2, 4]:
        runs[f"Env{i}"] = {"soft_q=True": [f"{OUT}/runs/dsac_env{i}_gru_seed{s}" for s in SEEDS],
                           "soft_q=False": [f"{OUT}/runs/dsac_env{i}_gru_hardq_seed{s}" for s in SEEDS]}
    plot_seeds(runs, "return", "средняя дисконтированная отдача", "dsac_soft_q_return.png")
'''))
    cells += [
        md("## Итоговая оценка по сидам"),
        code(CONVERGENCE_TABLE),
        code(SEED_TABLE),
        code(f'''
if "plots" in RUN_STAGES:
    groups = {{}}
    for exp in EXPERIMENTS:
        config = exp["name"].rsplit("_seed", 1)[0]
        groups.setdefault(config, []).append(f"{{OUT}}/runs/{{exp['name']}}")
    save_rows(seeds_rows(groups), "eval_seeds.csv")
'''),
    ]
    save(cells, f"tasks/{out}.ipynb")


v2_notebook("task2", mnist=False)
v2_notebook("task3", mnist=True)
v3_notebook("task2", mnist=False)
v3_notebook("task3", mnist=True)
tuning_notebook()
