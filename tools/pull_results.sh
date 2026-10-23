#!/usr/bin/env bash
# Забирает результаты с сервера и собирает чистую копию.
#   bash tools/pull_results.sh
# 1) полная копия серверной папки (код, runs/, outputs/, логи, веса) -> D:/repos/petr/gridrl_svalka/server_copy
# 2) outputs/task2, outputs/task3, outputs/logs, outputs/notebooks с сервера -> рабочая копия
# 3) чистая копия (src/, tasks/, outputs/ + логи и чекпоинты в outputs/logs) -> C:/repos/smallGridWorld-experiments_2
set -euo pipefail
cd "$(dirname "$0")/.."
SSH="ssh -o BatchMode=yes -o LogLevel=ERROR -p 51333 root@187.127.86.34"
FULL=/d/repos/petr/gridrl_svalka/server_copy
CLEAN=/c/repos/smallGridWorld-experiments_2

mkdir -p "$FULL"
$SSH 'cd ~/smallGridWorld-experiments && tar czf - --exclude=.venv --exclude=__pycache__ .' < /dev/null | tar xzf - -C "$FULL"
echo "полная копия: $FULL"

# всё, что посчитано на сервере (task1 считается локально и не трогается)
for dir in "$FULL"/outputs/*/; do
    part=$(basename "$dir")
    if [ "$part" = "task1" ]; then continue; fi
    rm -rf "outputs/$part"
    cp -r "$dir" "outputs/$part"
done
uv run python tools/summary.py
echo "outputs обновлены в рабочей копии"

mkdir -p "$CLEAN"
rm -rf "$CLEAN/src" "$CLEAN/tasks" "$CLEAN/outputs"
mkdir -p "$CLEAN/src" "$CLEAN/tasks"
cp -r src/envs src/models src/utils.py src/__init__.py "$CLEAN/src/"
find "$CLEAN/src" -name __pycache__ -prune -exec rm -rf {} +
cp tasks/*.ipynb "$CLEAN/tasks/"
cp README.md "$CLEAN/"
if [ -f report.md ]; then cp report.md "$CLEAN/"; fi
cp -r outputs "$CLEAN/outputs"
# структура из ТЗ: пустые папки тоже создаём
for d in task1/figures task1/tables task1/validation task2/runs task2/figures task2/eval task2/tables \
         task3/runs task3/figures task3/eval task3/tables logs notebooks; do
    mkdir -p "$CLEAN/outputs/$d"
done
# все логи и чекпоинты дополнительно собираем в outputs/logs
for taskdir in "$CLEAN"/outputs/task*/; do
    task=$(basename "$taskdir")
    for run in "$taskdir"runs/*/; do
        [ -d "$run" ] || continue
        name=$(basename "$run")
        mkdir -p "$CLEAN/outputs/logs/checkpoints/$task/$name"
        for f in model.pt train.log status.json; do
            if [ -f "$run/$f" ]; then cp "$run/$f" "$CLEAN/outputs/logs/checkpoints/$task/$name/"; fi
        done
    done
done
echo "чистая копия: $CLEAN"
