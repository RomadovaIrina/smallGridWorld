#!/usr/bin/env bash
# перебор alpha / ent_coef на 10x10, затем (когда закончатся и этап 2, и перебор) этап 3
cd "$(dirname "$0")/.."
bash scripts/run_task.sh task2_tuning full 12 > outputs/logs/run_task2_tuning_full.log 2>&1
# ждём, пока доработает этап 2 (его запустил run_v2_v3.sh)
while pgrep -f "[r]un_task.sh task[23]_v2 " > /dev/null; do sleep 60; done
bash scripts/run_task.sh task2_v3 full 20 > outputs/logs/run_task2_v3_full.log 2>&1 &
bash scripts/run_task.sh task3_v3 full 8 > outputs/logs/run_task3_v3_full.log 2>&1 &
wait
echo "перебор и этап 3 завершены" > outputs/logs/run_tuning_v3_done.txt
