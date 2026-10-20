# План разбора тестового

Цель — не писать новое, а **понять уже написанный код и уметь его объяснить**. Порядок важен: сначала идея,
потом сразу кусок нашего кода, где она живёт. Отмечай пройденное галочками.

Активный код: `src_v2/gridrl_v2/`. Ноутбуки: `src_v2/test_task.ipynb`, `src_v2/test_task_v2.ipynb`, `src_v2/mnist_task_v1.ipynb`.

---

## День 1. Основы и PPO

### Шаг 1. Базовые понятия (~2 часа)

Агент, среда, состояние, действие, награда, эпизод, отдача, дисконт γ, стратегия π, функции ценности V(s) и Q(s, a).

- [ ] Читать: [Spinning Up, Part 1: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — лучший короткий текст по основам
- [ ] Смотреть: [David Silver, Lecture 1: Introduction to RL](https://www.youtube.com/watch?v=2pWv7GOvuf0) — классический курс DeepMind, первая лекция ([весь плейлист](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ))
- [ ] Или курс с картинками: [Hugging Face Deep RL Course, Unit 1](https://huggingface.co/learn/deep-rl-course/unit1/introduction)
- [ ] Обзорно, за 20 минут: [Steve Brunton — Overview of Deep RL Methods](https://www.youtube.com/watch?v=wDVteayWWvU)
- [ ] **В коде:** `envs/gridworld.py` — `reset`, `step`, награда, `terminated` и `truncated`. Создай среду в ноутбуке (`make_env(5, 5, 5, 0.0)`) и походи по ней руками, печатая `env.render()`.
- [ ] **Проверь себя:** почему при γ = 0.99 агенту выгодно идти к цели кратчайшим путём, хотя награда всегда 1? Чем `terminated` отличается от `truncated`?

### Шаг 2. MDP и POMDP — зачем память (~30 минут)

- [ ] Читать: [Spinning Up, Part 1](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html), раздел «States and Observations»
- [ ] **В коде:** `SETUPS` в ноутбуке. Env1 — 25 уникальных цветов (по цвету понятно, где агент, это MDP), Env2 — 5 цветов (непонятно, это POMDP).
- [ ] **Проверь себя:** что видит агент на Env2? Почему по одному наблюдению не понять, где он, и как помогает история прошлых наблюдений и действий?

### Шаг 3. Policy gradient и актор-критик (~2–3 часа)

Это ключ ко всему остальному. Идея: повышать вероятность действий, которые оказались лучше ожидаемого.
«Лучше ожидаемого» — это **преимущество** A = (что получили) − V(s). **Критик** учит V(s), **актор** учит π.
Что называют A2C — это и есть простой актор-критик.

- [ ] Смотреть: [Arxiv Insights — An introduction to Policy Gradient methods](https://www.youtube.com/watch?v=5P7I-xPq8u8) — понятно и с картинками
- [ ] Читать: [Spinning Up, Part 3: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [ ] Курс: [Hugging Face, Unit 4 (Policy Gradient)](https://huggingface.co/learn/deep-rl-course/unit4/introduction) и [Unit 6 (Actor-Critic)](https://huggingface.co/learn/deep-rl-course/unit6/introduction)
- [ ] Глубже: [Pieter Abbeel — L3 Policy Gradients and Advantage Estimation](https://www.youtube.com/watch?v=AKbX1Zvo7r8)
- [ ] По желанию, с формулами: [Lilian Weng — Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- [ ] **Проверь себя:** зачем нужен критик, если можно просто считать отдачу? Что значит A > 0 и A < 0?

### Шаг 4. PPO (~2–3 часа)

PPO — актор-критик, у которого стратегия **не может сильно измениться за одно обновление**. Обрезается не градиент и не действие,
а **отношение вероятностей** `ratio = π_new(a|s) / π_old(a|s)`: если оно вышло за [0.8, 1.2], дальше тянуть не даём.

- [ ] Смотреть: [Johnny Code — Simply Explaining PPO](https://www.youtube.com/watch?v=5VHLd9eCZ-w)
- [ ] Смотреть: [Pieter Abbeel — L4 TRPO and PPO](https://www.youtube.com/watch?v=KjWF8VIMGiY)
- [ ] Читать статью: [Schulman et al., 2017 — Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), разделы 3–5
- [ ] Про GAE (как считаем преимущество): [Schulman et al., 2015 — GAE](https://arxiv.org/abs/1506.02438), достаточно идеи из начала
- [ ] Популярно: [Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html), [Hugging Face, Unit 8](https://huggingface.co/learn/deep-rl-course/unit8/introduction)
- [ ] По желанию — какие «фишки» бывают в реальных реализациях: [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [ ] **В коде** `models/ppo.py`, сверху вниз:
  - [ ] `ActorCritic` — одна сеть с двумя головами: `actor` (логиты 4 действий) и `critic` (одно число V)
  - [ ] `play_episode` — как собирается опыт, что запоминаем на каждом шаге
  - [ ] `compute_advantages` — GAE: `deltas`, цикл с конца, `returns = advantages + values`
  - [ ] `update` — `ratio`, `clamp`, `policy_loss`, `value_loss`, `entropy`: это формула из статьи
  - [ ] `train` — пачка эпизодов → обновление → метрики
- [ ] **Проверь себя:** что будет, если убрать `clamp`? Зачем бонус энтропии (`ent_coef`)? Почему PPO on-policy: можно ли учиться на старых эпизодах?

---

## День 2. DSAC, память, результаты

### Шаг 5. Q-функция и off-policy (~1–2 часа)

Q(s, a) — ценность конкретного действия. **Off-policy** — опыт копится в replay-буфер и используется много раз.
Главная формула (уравнение Беллмана): `Q(s, a) ≈ r + γ · V(s')`. Target-сеть — медленная копия сети, по которой считаем правую часть.

- [ ] Курс: [Hugging Face, Unit 3 (Deep Q-Learning)](https://huggingface.co/learn/deep-rl-course/unit3/introduction) — хватит первой половины, DQN знать глубоко не нужно
- [ ] Читать: [Spinning Up, Part 2: Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — где on-policy, где off-policy
- [ ] **Проверь себя:** чем PPO и DSAC отличаются в том, как используют опыт?

### Шаг 6. SAC и Discrete SAC (~2–3 часа)

SAC максимизирует награду **плюс энтропию** стратегии, то есть поощряет не становиться слишком уверенным раньше времени.
Температура α — насколько сильно поощряется случайность. В дискретной версии действий всего 4, поэтому средние
по действиям считаются точно, суммой по 4 вариантам.

- [ ] Смотреть: [Pieter Abbeel — L5 DDPG and SAC](https://www.youtube.com/watch?v=pg-lKy7JIRk)
- [ ] Читать: [Spinning Up — SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) (там непрерывные действия, но идея та же)
- [ ] Статья нашей версии: [Christodoulou, 2019 — Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207) — короткая, разделы 3–4
- [ ] По желанию — оригинальный SAC: [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)
- [ ] **В коде** `models/dsac.py`:
  - [ ] `Net` — одна сеть; из неё сделаны актор (логиты) и два критика (Q для 4 действий)
  - [ ] `play_episode` — играем эпизод и кладём в буфер `self.buffer`
  - [ ] `episode_losses` — цель `y = r + γ(1 − done)·V(s')`, флаг `soft_q`, лосс критиков, лосс актора
  - [ ] `update` — шаги оптимизаторов и `lerp_` (target-критики понемногу догоняют обычных)
  - [ ] `train` — после каждого эпизода `updates_per_episode` обновлений на случайных эпизодах из буфера
- [ ] **Проверь себя:** зачем два критика и минимум из них? Чем мягкая Q (`soft_q=True`) отличается от обычной? Почему α маленькая (0.005) — что будет при большой?

### Шаг 7. Память (GRU) (~1 час)

- [ ] Читать: [Colah — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (GRU — упрощённый LSTM, он в конце статьи)
- [ ] **В коде:** `forward` в `ActorCritic` (ppo.py) и `Net` (dsac.py) — `memory=True` против `False`, `prev_actions`, `hidden`
- [ ] **Проверь себя:** зачем на вход подаётся предыдущее действие? Почему в начале эпизода память пустая (`hidden=None`)?

### Шаг 8. Свои результаты и MNIST (~1–2 часа)

- [ ] Пройтись по графикам (`outputs/plots/`), тепловым картам и гифкам (`outputs/eval/`)
- [ ] Сформулировать своими словами: где агент учится, где нет, почему (размер карты, число цветов, память)
- [ ] MNIST: перечитать черновики ответов в конце `mnist_task_v1.ipynb` и переписать своими словами
- [ ] Это и будет основа отчёта

---

## Вопросы, которые почти наверняка зададут

- [ ] Чем on-policy отличается от off-policy, на примере PPO и DSAC?
- [ ] Что такое преимущество и GAE? Зачем нужен критик?
- [ ] Что именно обрезается в PPO и зачем?
- [ ] Зачем в SAC энтропия, два критика и target-сети?
- [ ] Почему нужна память и как это видно по результатам (MLP против GRU)?
- [ ] Что меняет MNIST-версия среды и нужен ли сложный энкодер?
- [ ] Почему не сделана задача 4 (векторизация)? Честный ответ про время — нормальный: в задании сказано, что частичное решение тоже решение.
- [ ] Какие отличия нашего кода от статей и почему (см. ниже)?

## Чем наш код отличается от статей

- **PPO:** память (GRU) и предыдущее действие на входе; нормировка преимуществ и обрезка градиента (приёмы из популярных реализаций); опыт собирается целыми эпизодами; при обрыве эпизода по времени добавляется оценка `V`.
- **DSAC:** память (GRU); фиксированная температура α (в статье подбирается автоматически); флаг `soft_q` (по требованию задания); буфер из целых эпизодов, сети прогоняются по эпизоду с начала.

## Если застряла

Кидай мне конкретную строку или кусок кода с вопросом «что это и зачем» — объясню именно его.
