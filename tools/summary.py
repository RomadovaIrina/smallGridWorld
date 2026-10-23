"""Сводная таблица outputs/summary.csv по всем прогонам задач 2 и 3: статус, бюджет, эпизоды, шаги, время, итоговая оценка."""
import csv
import json
from pathlib import Path

rows = []
for task_dir in sorted(Path("outputs").glob("task*")):
    task = task_dir.name
    for run_dir in sorted((task_dir / "runs").glob("*")):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        row = {"task": task, "run": run_dir.name, "algo": config["algo"], "memory": config["memory"],
               "soft_q": config.get("soft_q", ""), "mnist": config.get("mnist", False),
               "env": f"{config['h']}x{config['w']}, {config['n_colors']} цв., {config['obstacle_fraction']}",
               "seed": config["seed"], "budget": config.get("budget", ""), "episodes_planned": config["episodes"],
               "status": status["status"], "episodes_done": status["episodes_done"],
               "env_steps": status.get("env_steps", ""), "time_min": round(status.get("time_sec", 0) / 60, 1),
               "device": config["device"], "optimal_length": round(config["optimal"]["length"], 2),
               "max_env_steps": config.get("max_env_steps", ""), "converged": status.get("converged", "")}
        eval_file = run_dir / "eval.json"
        if eval_file.exists():
            result = json.loads(eval_file.read_text(encoding="utf-8"))
            for mode in ["greedy", "stochastic"]:
                row[f"{mode}_success"] = round(result[mode]["success_rate"], 3)
                row[f"{mode}_length"] = round(result[mode]["mean_length"], 1)
                row[f"{mode}_return"] = round(result[mode]["mean_return"], 3)
        rows.append(row)

if rows:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open("outputs/summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"outputs/summary.csv: {len(rows)} прогонов")
