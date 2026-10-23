# RL для GridWorld

Тестовое задание по обучению с подкреплением: клеточная среда GridWorld, в которой агент видит только цвет своей клетки,
агенты PPO и Discrete SAC (с памятью GRU и без неё) и версия среды, где цвет показывается картинкой цифры из MNIST.

Автор: _ФИО_

Код написан с помощью ИИ-ассистента (задание это разрешает) и разобран мной; подробности и ограничения — в `report.md`.

## Структура

```
src/
  envs/
    gridworld.py   среда GridWorldEnv: наблюдение — one-hot цвета клетки
    mnist_env.py   MnistGridWorldEnv: копия среды, цвет показывается картинкой цифры MNIST 14x14
    setups.py      make_env (случайная карта по seed), BFS, оптимум (oracle)
  models/
    ppo.py         PPO: актор-критик с GAE, сеть с памятью (GRU) или без (MLP)
    dsac.py        Discrete SAC: два критика, target-критики, мягкая или обычная Q
  utils.py         сохранение прогонов, картинки: карта, тепловая карта посещений, GIF
tasks/
  task1.ipynb         задача 1: среда, проверки, карты, характеристики сред
  task2.ipynb         задача 2, первая версия: фиксированное число эпизодов (DSAC на 10x10 не доучен)
  task3.ipynb         задача 3, первая версия: MNIST-наблюдения
  task2_v2.ipynb      задача 2, этап 2: обучение до схождения (seed 0)
  task3_v2.ipynb      задача 3, этап 2: то же для MNIST
  task2_tuning.ipynb  перебор alpha (DSAC) и ent_coef (PPO) на картах 10x10
  task2_v3.ipynb      задача 2, этап 3: одинаковый бюджет шагов, лучшие параметры, 3 сида — основное сравнение
  task3_v3.ipynb      задача 3, этап 3: то же для MNIST
outputs/           все результаты (см. ниже)
report.md          отчёт
```

## Установка

Нужен Python 3.11+.

```bash
pip install torch gymnasium numpy matplotlib jupyter
```

Для видеокарты NVIDIA — сборка torch с CUDA, например `pip install torch --index-url https://download.pytorch.org/whl/cu126`.
Ноутбуки сами используют CUDA, если она доступна (DSAC на ней в ~3 раза быстрее, PPO — нет).
MNIST скачивается сам при первом запуске в `data/mnist` (~10 МБ).

## Запуск

Ноутбуки запускаются из папки `tasks/` или из корня репозитория, по порядку:

1. `task1.ipynb` — без обучения, около минуты.
2. `task2.ipynb`, `task3.ipynb` — первая версия экспериментов (история).
3. `task2_v2.ipynb`, `task3_v2.ipynb` — обучение до схождения.
4. `task2_tuning.ipynb` — перебор параметра исследования; использует результаты `task2_v2`.
5. `task2_v3.ipynb`, `task3_v3.ipynb` — основное сравнение; берут параметры из перебора и бюджет из этапа 2.

Последовательно всё это — много часов (DSAC на картах 10x10 самый долгий); на сервере все прогоны шли параллельно, всего ~9 часов.

Уже завершённые прогоны (`status.json` = `completed`) ноутбук пропускает, поэтому его можно перезапускать.

### Без Jupyter и параллельно

Ноутбук управляется переменными окружения:

- `RUN_MODE=full|smoke` — полное обучение или короткая проверка (smoke пишет в `runs/smoke/`, не смешиваясь с полными результатами);
- `RUN_ONLY=<имя эксперимента>` — обучить только один эксперимент (имена печатает ячейка «Список экспериментов»);
- `RUN_STAGES=train,plots` — что делать: обучение, графики или оба.

Пример: один эксперимент, затем графики по всем сохранённым прогонам:

```bash
RUN_MODE=full RUN_ONLY=ppo_env3_gru_seed0 RUN_STAGES=train jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 --output-dir outputs/notebooks --output task2_ppo_env3 tasks/task2.ipynb
RUN_MODE=full RUN_STAGES=plots jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 --output-dir outputs/notebooks --output task2 tasks/task2.ipynb
```

Так эксперименты запускались на арендованном сервере (RTX 3090, 28 ядер CPU): каждый отдельным процессом, все параллельно.

### Графики заново без обучения

Все графики строятся по сохранённым `metrics.csv`: запустить ноутбук с `RUN_STAGES=plots`.

## Результаты

```
outputs/
  task1/        figures/ (карты, BFS), tables/env_summary.csv, validation/checks.csv
  task2/, task3/                 первая версия: runs/<прогон>/, figures/, eval/ (тепловые карты и GIF), tables/
  task2_v2/, task3_v2/           этап 2: до схождения; tables/convergence.csv
  task2_tuning/                  перебор alpha и ent_coef; tables/tuning.csv, figures/*_sweep.png
  task2_v3/, task3_v3/           этап 3: 3 сида; tables/eval_seeds.csv (среднее и разброс по сидам)
  logs/         логи запусков; logs/checkpoints/ — копии весов и логов всех прогонов
  notebooks/    выполненные копии ноутбуков
  summary.csv   сводка по всем прогонам: статус, бюджет, шаги, время, итоговая оценка
```

В папке прогона: `config.json` (все параметры, бюджет `full`/`smoke`, оптимум), `metrics.csv` (одна строка на эпизод,
записывается по ходу обучения), `model.pt` (веса), `eval.json` (итоговая оценка), `status.json` (`running`/`completed`/`failed`),
`train.log`.

## Что сделано

- Задача 1 — среда со всеми параметрами из задания, четыре сетапа, проверки (`tasks/task1.ipynb`).
- Задача 2 — PPO с GAE и Discrete SAC с переключателем мягкой Q; сравнения с памятью и без, PPO и DSAC, `soft_q`; графики из задания; тепловые карты и GIF. Основное сравнение — при одинаковом бюджете шагов, после обучения до схождения и подбора параметра исследования, на 3 сидах.
- Задача 3 — MNIST-версия среды, сравнение one-hot и MNIST, ответы на вопросы.
- Задача 4 (векторизация) — **не сделана**, отложена из-за нехватки времени.
- Динамическая температура τ(u) (расширение из задачи 2) — не сделана.
