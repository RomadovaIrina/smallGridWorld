#!/usr/bin/env bash
# этап 2 (до схождения) для задач 2 и 3 параллельно, затем этап 3 (одинаковый бюджет, 3 сида)
cd "$(dirname "$0")/.."
bash scripts/run_task.sh task2_v2 full 18 > outputs/logs/run_task2_v2_full.log 2>&1 &
bash scripts/run_task.sh task3_v2 full 6 > outputs/logs/run_task3_v2_full.log 2>&1 &
wait
bash scripts/run_task.sh task2_v3 full 20 > outputs/logs/run_task2_v3_full.log 2>&1 &
bash scripts/run_task.sh task3_v3 full 8 > outputs/logs/run_task3_v3_full.log 2>&1 &
wait
echo "этапы 2 и 3 завершены" > outputs/logs/run_v2_v3_done.txt
