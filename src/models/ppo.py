"""PPO — on-policy актор-критик с GAE. Можно с памятью (GRU) или без (MLP).

Использование:
    alg = PPO(ppo_cfg)  # ppo_cfg — словарь, все ключи см. в test_task.ipynb
    alg.train()
    alg.save()
"""
import random
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from src import utils
from src.envs import make_env, optimal_stats


class ActorCritic(nn.Module):
    """Наблюдение (+ предыдущее действие) -> GRU или MLP -> логиты действий и V(s)."""

    def __init__(self, obs_size, n_actions, hidden, memory):
        super().__init__()
        self.memory = memory
        self.n_actions = n_actions
        if memory:
            self.encoder = nn.Sequential(nn.Linear(obs_size + n_actions, hidden), nn.Tanh())
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        else:
            self.encoder = nn.Sequential(nn.Linear(obs_size, hidden), nn.Tanh(),
                                         nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs, prev_actions, hidden=None):
        """obs: (T, obs_size) — шаги одного эпизода по порядку.
        prev_actions: (T,) — предыдущее действие на каждом шаге, -1 в начале эпизода.
        hidden — память GRU с прошлого вызова (None — пустая память).
        """
        if self.memory:
            prev_onehot = F.one_hot(prev_actions.clamp(min=0), self.n_actions).float()
            prev_onehot[prev_actions < 0] = 0.0  # в начале эпизода предыдущего действия нет
            x = self.encoder(torch.cat([obs, prev_onehot], dim=-1))
            x, hidden = self.gru(x.unsqueeze(0), hidden)  # батч из одного эпизода
            x = x.squeeze(0)
        else:
            x = self.encoder(obs)
        return self.actor(x), self.critic(x).squeeze(-1), hidden


class PPO:
    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

        # "mnist": True — наблюдения картинками цифр (задача 3); если ключа нет — обычные one-hot цвета
        self.env = make_env(cfg["h"], cfg["w"], cfg["n_colors"], cfg["obstacle_fraction"], seed=cfg["env_seed"],
                            mnist=cfg.get("mnist", False))
        self.env.reset(seed=cfg["seed"])  # дальше старты берутся из этого же генератора
        self.optimal = optimal_stats(self.env, cfg["gamma"])

        obs_size = self.env.observation_space.shape[0]
        self.device = cfg["device"]  # "cpu" или "cuda" (видеокарта)
        self.net = ActorCritic(obs_size, 4, cfg["hidden"], cfg["memory"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        self.history = []  # одна строка метрик на каждый эпизод
        self.steps = 0  # сколько шагов среды сыграно за обучение
        self.run_dir = None  # папка, куда сохраняется прогон (создаётся в train)
        self.converged_at = None  # эпизод и шаг выхода на плато (если обучаем до схождения)

    def play_episode(self, greedy=False, seed=None, learn=True):
        """Играет один эпизод и запоминает всё, что нужно для обучения и картинок.
        seed — сид старта (для честной оценки); learn=False — эпизод не считается в шаги обучения.
        """
        obs, info = self.env.reset(seed=seed)
        ep = {"obs": [], "prev_actions": [], "actions": [], "logp": [], "values": [], "rewards": [],
              "greedy_prob": [], "entropy": [], "positions": [tuple(info["pos"])], "colors": [info["color"]]}
        prev_action = -1
        hidden = None
        while True:
            obs_t = torch.tensor(obs, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, value, hidden = self.net(obs_t, torch.tensor([prev_action], device=self.device), hidden)
            dist = Categorical(logits=logits[0])
            action = logits[0].argmax() if greedy else dist.sample()

            next_obs, reward, terminated, truncated, info = self.env.step(action.item())
            ep["obs"].append(obs)
            ep["prev_actions"].append(prev_action)
            ep["actions"].append(action.item())
            ep["logp"].append(dist.log_prob(action).item())
            ep["values"].append(value.item())
            ep["rewards"].append(reward)
            ep["greedy_prob"].append(dist.probs.max().item())
            ep["entropy"].append(dist.entropy().item())
            ep["positions"].append(tuple(info["pos"]))
            ep["colors"].append(info["color"])

            obs = next_obs
            prev_action = action.item()
            if terminated or truncated:
                break

        # если эпизод оборвали по времени, агент не провалился — оцениваем, сколько он ещё получил бы
        ep["last_value"] = 0.0
        if truncated:
            with torch.no_grad():
                _, value, _ = self.net(torch.tensor(obs, device=self.device).unsqueeze(0), torch.tensor([prev_action], device=self.device), hidden)
            ep["last_value"] = value.item()
        ep["success"] = terminated
        ep["length"] = len(ep["actions"])
        if learn:
            self.steps += ep["length"]
        return ep

    def compute_advantages(self, ep):
        """GAE: насколько каждое действие оказалось лучше, чем ожидал критик."""
        rewards = np.array(ep["rewards"])
        values = np.array(ep["values"])
        next_values = np.append(values[1:], ep["last_value"])
        deltas = rewards + self.cfg["gamma"] * next_values - values

        advantages = np.zeros(len(rewards))
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = deltas[t] + self.cfg["gamma"] * self.cfg["gae_lambda"] * running
            advantages[t] = running
        returns = advantages + values  # цель для критика
        return advantages, returns

    def update(self, episodes):
        """Несколько эпох градиентного спуска по собранным эпизодам."""
        for ep in episodes:
            ep["advantages"], ep["returns"] = self.compute_advantages(ep)
        all_adv = np.concatenate([ep["advantages"] for ep in episodes])
        adv_mean, adv_std = all_adv.mean(), all_adv.std() + 1e-8

        stats = {"loss": [], "policy_loss": [], "value_loss": [], "entropy": []}
        for _ in range(self.cfg["epochs"]):
            shuffled = episodes.copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled), self.cfg["episodes_per_batch"]):
                batch = shuffled[i:i + self.cfg["episodes_per_batch"]]

                # прогоняем каждый эпизод целиком (память идёт от начала эпизода), потом склеиваем шаги
                all_logits, all_values = [], []
                for ep in batch:
                    logits, values, _ = self.net(torch.tensor(np.array(ep["obs"]), device=self.device),
                                            torch.tensor(ep["prev_actions"], device=self.device))
                    all_logits.append(logits)
                    all_values.append(values)
                logits = torch.cat(all_logits)
                values = torch.cat(all_values)
                actions = torch.tensor(np.concatenate([ep["actions"] for ep in batch]), device=self.device)
                old_logp = torch.tensor(np.concatenate([ep["logp"] for ep in batch]), dtype=torch.float32, device=self.device)
                adv = torch.tensor(np.concatenate([ep["advantages"] for ep in batch]), dtype=torch.float32, device=self.device)
                adv = (adv - adv_mean) / adv_std
                returns = torch.tensor(np.concatenate([ep["returns"] for ep in batch]), dtype=torch.float32, device=self.device)

                dist = Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(actions) - old_logp)  # во сколько раз изменилась вероятность
                clipped = torch.clamp(ratio, 1 - self.cfg["clip"], 1 + self.cfg["clip"])
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                value_loss = ((values - returns) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.cfg["vf_coef"] * value_loss - self.cfg["ent_coef"] * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                stats["loss"].append(loss.item())
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
        return {k: np.mean(v) for k, v in stats.items()}

    def train(self):
        self.run_dir = utils.start_run(self.cfg, "ppo", self.optimal)
        try:
            self.train_loop()
        except Exception as error:
            utils.save_progress(self.run_dir, self.history, "failed", error=repr(error))
            raise
        result = {"converged": self.converged_at is not None, "converged_at": self.converged_at}
        utils.save_progress(self.run_dir, self.history, "completed", weights=self.weights(), extra=result)

    def should_stop(self, episode):
        """Конец обучения раньше лимита эпизодов: кончился бюджет шагов среды или вышли на плато."""
        if self.steps >= self.cfg.get("max_env_steps", float("inf")):
            utils.log(self.run_dir, f"бюджет шагов {self.cfg['max_env_steps']} исчерпан: {self.steps} шагов")
            return True
        # плато ищем только после min_episodes: на больших картах DSAC сначала долго не учится,
        # и ровная кривая около нуля — это ещё не схождение
        ready = len(self.history) >= self.cfg.get("min_episodes", 0)
        if self.cfg.get("stop_when_converged", False) and ready and (episode + 1) % 100 == 0:
            if utils.has_converged(self.history, self.cfg["conv_window"], self.cfg["conv_tol"]):
                self.converged_at = {"episode": len(self.history), "env_steps": self.steps}
                utils.log(self.run_dir, f"вышли на плато: эпизод {len(self.history)}, шагов {self.steps}")
                return True
        return False

    def train_loop(self):
        start_time = time.time()
        episodes = []
        n_updates = 0
        for episode in range(self.cfg["episodes"]):
            # 1. играем эпизод
            ep = self.play_episode()
            episodes.append(ep)
            if len(episodes) < self.cfg["episodes_per_update"]:
                continue

            # 2. набрали пачку эпизодов — учимся на ней
            stats = self.update(episodes)
            n_updates += 1

            # 3. метрики каждого эпизода. Лоссы и update_entropy — от обновления после пачки,
            #    поэтому у всех эпизодов пачки они одинаковые (номер обновления — в колонке update)
            for ep in episodes:
                self.history.append({
                    "episode": len(self.history),
                    "update": n_updates,
                    "env_steps": self.steps,
                    "return": self.cfg["gamma"] ** (ep["length"] - 1) * ep["success"],
                    "success": ep["success"],
                    "length": ep["length"],
                    "explore_prob": 1 - np.mean(ep["greedy_prob"]),
                    "entropy": np.mean(ep["entropy"]),  # энтропия стратегии во время эпизода
                    "loss": stats["loss"],
                    "policy_loss": stats["policy_loss"],
                    "value_loss": stats["value_loss"],
                    "update_entropy": stats["entropy"],  # энтропия на данных обновления
                    "time": time.time() - start_time,
                })
            episodes = []
            stop = self.should_stop(episode)

            # 4. сохраняем прогресс на диск, чтобы не потерять его при падении
            if (episode + 1) % self.cfg.get("save_every", 50) == 0:
                utils.save_progress(self.run_dir, self.history, "running", weights=self.weights())
            if (episode + 1) % 100 == 0:
                last = self.history[-100:]
                utils.log(self.run_dir,
                          f"episode {episode + 1} | steps {self.steps} | return {np.mean([r['return'] for r in last]):.3f} "
                          f"(opt {self.optimal['return']:.3f}) | length {np.mean([r['length'] for r in last]):.1f} "
                          f"| entropy {np.mean([r['entropy'] for r in last]):.2f}")
            if stop:
                break

    def evaluate(self, n_episodes=100, greedy=False, first_seed=10_000):
        """Оценка без обучения: старты по сидам first_seed, first_seed+1, ... (одинаковые для всех моделей)."""
        torch.manual_seed(first_seed)  # чтобы сэмплирование действий тоже повторялось
        return [self.play_episode(greedy, seed=first_seed + i, learn=False) for i in range(n_episodes)]

    def weights(self):
        return {"net": self.net.state_dict()}

    def save(self):
        """Сохранить текущее состояние вручную (train() и так сохраняет всё сам)."""
        utils.save_progress(self.run_dir, self.history, "completed", weights=self.weights())
