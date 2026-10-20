"""Графики обучения по runs/<имя>/metrics.csv. Несколько прогонов рисуются на одних осях.

    uv run python -m gridrl.utils.plot_training runs/ppo_s2_gru runs/ppo_s2_mlp --name ppo_s2
    uv run python -m gridrl.utils.plot_training runs/dsac_s2_gru --labels GRU

Картинка сохраняется в outputs/plots/<name>.png.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Компоненты лосса для каждого алгоритма (рисуются, если есть в metrics.csv).
LOSS_PARTS = ["policy_loss", "value_loss", "q_loss", "actor_loss", "alpha_loss"]


def read_run(run_dir):
    with open(Path(run_dir) / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    data = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    config = json.loads((Path(run_dir) / "config.json").read_text())
    return data, config


def smooth(y, k):
    """Скользящее среднее по последним k точкам (пропуски NaN игнорируются)."""
    if k <= 1:
        return y
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        window = y[max(0, i - k + 1):i + 1]
        window = window[~np.isnan(window)]
        if len(window):
            out[i] = window.mean()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--name", default=None)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--out", default="outputs/plots")
    args = parser.parse_args()

    labels = args.labels or [Path(r).name for r in args.runs]
    runs = [read_run(r) for r in args.runs]
    has_alpha = any("alpha" in data for data, _ in runs)

    panels = ["return", "success", "ep_length", "loss", "loss_parts", "entropy", "explore_prob"]
    if has_alpha:
        panels.append("alpha")
    titles = {
        "return": "Средняя дисконтированная отдача",
        "success": "Доля эпизодов, дошедших до цели",
        "ep_length": "Длина эпизода",
        "loss": "Полный лосс",
        "loss_parts": "Компоненты лосса",
        "entropy": "Энтропия стратегии",
        "explore_prob": "Вероятность нежадного действия",
        "alpha": "Температура alpha (DSAC)",
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")

    for i, ((data, config), label) in enumerate(zip(runs, labels)):
        color = f"C{i}"
        x = data["global_step"]
        for ax, panel in zip(axes, panels):
            if panel == "loss_parts":
                styles = iter(["-", "--", ":", "-."])
                for part in LOSS_PARTS:
                    if part in data:
                        ax.plot(x, smooth(data[part], args.smooth), color=color,
                                linestyle=next(styles), label=f"{label}: {part}")
            elif panel in data:
                ax.plot(x, smooth(data[panel], args.smooth), color=color, label=label)

        # ориентир: агент, знающий свои координаты и идущий кратчайшим путём
        optimal = config.get("optimal")
        if optimal:
            axes[panels.index("return")].axhline(optimal["return"], color=color, linestyle=":", alpha=0.7)
            axes[panels.index("ep_length")].axhline(optimal["length"], color=color, linestyle=":", alpha=0.7)

    for ax, panel in zip(axes, panels):
        ax.set_title(titles[panel])
        ax.set_xlabel("шаги среды (сумма по всем копиям)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[panels.index("ep_length")].set_yscale("log")
    axes[panels.index("return")].text(
        0.02, 0.02, "пунктир — оптимум", transform=axes[panels.index("return")].transAxes, fontsize=8
    )

    name = args.name or "_vs_".join(labels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=110)
    print(f"saved to {out / (name + '.png')}")


if __name__ == "__main__":
    main()
