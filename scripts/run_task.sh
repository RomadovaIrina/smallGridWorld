#!/usr/bin/env bash
# Запуск обучения из ноутбука без открытого Jupyter, каждый эксперимент — отдельным процессом (параллельно).
#   bash scripts/run_task.sh task2 smoke      # короткая проверка всех экспериментов
#   bash scripts/run_task.sh task2 full 12    # полное обучение, до 12 процессов одновременно
# Логи: outputs/logs/<task>_<эксперимент>.log, выполненные ноутбуки: outputs/notebooks/
set -euo pipefail
TASK=$1
MODE=${2:-smoke}
JOBS=${3:-8}
cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/notebooks

# список экспериментов берём из самого ноутбука, чтобы не дублировать конфиги
NAMES=$(uv run python - "$TASK" "$MODE" 2>/dev/null <<'EOF'
import contextlib, io, json, os, sys
os.environ["RUN_MODE"] = sys.argv[2]
os.environ["MPLBACKEND"] = "Agg"
cells = json.load(open(f"tasks/{sys.argv[1]}.ipynb", encoding="utf-8"))["cells"]
code = [("".join(c["source"])) for c in cells if c["cell_type"] == "code"]
scope = {}
# выполняем только ячейки до списка экспериментов включительно (без обучения и графиков)
with contextlib.redirect_stdout(io.StringIO()):  # печать из ячеек не должна попасть в список имён
    for src in code:
        exec(src, scope)
        if "EXPERIMENTS" in scope:
            break
print(" ".join(exp["name"] for exp in scope["EXPERIMENTS"]))
EOF
)
echo "эксперименты: $NAMES"

run_one() {
    name=$1
    RUN_MODE=$MODE RUN_ONLY=$name RUN_STAGES=train uv run jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 --output-dir outputs/notebooks --output "${TASK}_${name}" \
        "tasks/${TASK}.ipynb" > "outputs/logs/${TASK}_${name}.log" 2>&1 \
        && echo "готово: $name" || echo "ОШИБКА: $name (см. outputs/logs/${TASK}_${name}.log)"
}
export -f run_one
export TASK MODE
echo $NAMES | tr ' ' '\n' | xargs -P "$JOBS" -I{} bash -c 'run_one {}'

# графики и таблицы по всем сохранённым прогонам
RUN_MODE=$MODE RUN_STAGES=plots uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 --output-dir outputs/notebooks --output "$TASK" "tasks/${TASK}.ipynb" \
    > "outputs/logs/${TASK}_plots.log" 2>&1 && echo "графики готовы" || echo "ОШИБКА графиков (см. outputs/logs/${TASK}_plots.log)"
