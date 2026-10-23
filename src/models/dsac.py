"""Discrete SAC — off-policy актор-критик. Можно с памятью (GRU) или без (MLP).

Использование:
    alg = DSAC(dsac_cfg)  # dsac_cfg — словарь, все ключи см. в test_task.ipynb
    alg.train()
    alg.save()
"""
import copy
import random
import time
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from src import utils
from src.envs import make_env, optimal_stats


class Net(nn.Module):
    """Наблюдение (+ предыдущее действие) -> GRU или MLP -> 4 числа.
    У актора это логиты действий, у критика — Q(s, a) для каждого действия.
    """

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
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs, prev_actions, hidden=None):
        """obs: (T, obs_size), prev_actions: (T,) с -1 в начале эпизода."""
        if self.memory:
            prev_onehot = F.one_hot(prev_actions.clamp(min=0), self.n_actions).float()
            prev_onehot[prev_actions < 0] = 0.0
            x = self.encoder(torch.cat([obs, prev_onehot], dim=-1))
            x, hidden = self.gru(x.unsqueeze(0), hidden)
            x = x.squeeze(0)
        else:
            x = self.encoder(obs)
        return self.head(x), hidden


class DSAC:
    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

        # "mnist": True — наблюдения картинками цифр (задача 3); если ключа нет — обычные one-hot цвета
        self.env = make_env(cfg["h"], cfg["w"], cfg["n_colors"], cfg["obstacle_fraction"], seed=cfg["env_seed"],
                            mnist=cfg.get("mnist", False))
        self.env.reset(seed=cfg["seed"])
        self.optimal = optimal_stats(self.env, cfg["gamma"])

        obs_size = self.env.observation_space.shape[0]
        self.device = cfg["device"]  # "cpu" или "cuda" (видеокарта)
        self.actor = Net(obs_size, 4, cfg["hidden"], cfg["memory"]).to(self.device)
        self.q1 = Net(obs_size, 4, cfg["hidden"], cfg["memory"]).to(self.device)
        self.q2 = Net(obs_size, 4, cfg["hidden"], cfg["memory"]).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)  # медленные копии критиков для цели
        self.q2_target = copy.deepcopy(self.q2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr"])
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg["lr"])

        self.buffer = deque(maxlen=cfg["buffer_episodes"])  # replay-буфер из целых эпизодов
        self.steps = 0  # сколько шагов сыграно за обучение
        self.history = []  # одна строка метрик на каждый эпизод
        self.run_dir = None  # папка, куда сохраняется прогон (создаётся в train)
        self.converged_at = None  # эпизод и шаг выхода на плато (если обучаем до схождения)

    def play_episode(self, greedy=False, learn=True, seed=None):
        """Играет один эпизод. Если learn=True — кладёт его в буфер. seed — сид старта (для честной оценки)."""
        obs, info = self.env.reset(seed=seed)
        ep = {"obs": [obs], "prev_actions": [-1], "actions": [], "rewards": [], "greedy_prob": [], "entropy": [],
              "positions": [tuple(info["pos"])], "colors": [info["color"]]}
        prev_action = -1
        hidden = None
        while True:
            obs_t = torch.tensor(obs, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, hidden = self.actor(obs_t, torch.tensor([prev_action], device=self.device), hidden)
            dist = Categorical(logits=logits[0])
            action = logits[0].argmax() if greedy else dist.sample()

            obs, reward, terminated, truncated, info = self.env.step(action.item())
            prev_action = action.item()
            ep["obs"].append(obs)  # obs на 1 длиннее actions: последнее — наблюдение после последнего шага
            ep["prev_actions"].append(prev_action)
            ep["actions"].append(prev_action)
            ep["rewards"].append(reward)
            ep["greedy_prob"].append(dist.probs.max().item())
            ep["entropy"].append(dist.entropy().item())
            ep["positions"].append(tuple(info["pos"]))
            ep["colors"].append(info["color"])
            if learn:
                self.steps += 1
            if terminated or truncated:
                break

        ep["terminated"] = terminated  # при обрыве по времени будущее ещё есть — цель считаем как обычно
        ep["success"] = terminated
        ep["length"] = len(ep["actions"])
        if learn:
            self.buffer.append(ep)
        return ep

    def episode_losses(self, ep):
        """Лосс критиков и актора на одном эпизоде из буфера. Сети прогоняются по эпизоду с самого начала."""
        alpha, gamma = self.cfg["alpha"], self.cfg["gamma"]
        obs = torch.tensor(np.array(ep["obs"]), device=self.device)  # все наблюдения, на 1 больше шагов
        prev_actions = torch.tensor(ep["prev_actions"], device=self.device)
        actions = torch.tensor(ep["actions"], device=self.device)
        rewards = torch.tensor(ep["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.zeros(ep["length"], device=self.device)
        if ep["terminated"]:
            dones[-1] = 1.0  # после цели будущих наград нет

        # цель для критиков: y = r + gamma * (1 - done) * V(s')
        with torch.no_grad():
            logits_next = self.actor(obs, prev_actions)[0][1:]
            pi_next = F.softmax(logits_next, dim=-1)
            logpi_next = F.log_softmax(logits_next, dim=-1)
            q_next = torch.min(self.q1_target(obs, prev_actions)[0], self.q2_target(obs, prev_actions)[0])[1:]
            if self.cfg["soft_q"]:
                v_next = (pi_next * (q_next - alpha * logpi_next)).sum(-1)
            else:
                v_next = (pi_next * q_next).sum(-1)
            y = rewards + gamma * (1 - dones) * v_next

        q1 = self.q1(obs, prev_actions)[0][:-1]  # последнее наблюдение нужно только для цели
        q2 = self.q2(obs, prev_actions)[0][:-1]
        q1_a = q1.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q2_a = q2.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_loss = ((q1_a - y) ** 2 + (q2_a - y) ** 2).mean()

        # актор: выбирать действия с большим Q, но не становиться слишком уверенным
        logits = self.actor(obs, prev_actions)[0][:-1]
        pi = F.softmax(logits, dim=-1)
        logpi = F.log_softmax(logits, dim=-1)
        q_min = torch.min(q1, q2).detach()
        actor_loss = (pi * (alpha * logpi - q_min)).sum(-1).mean()
        return q_loss, actor_loss

    def update(self):
        """Одно обновление на batch_size случайных эпизодах из буфера."""
        q_losses, actor_losses = [], []
        for _ in range(self.cfg["batch_size"]):
            q_loss, actor_loss = self.episode_losses(random.choice(self.buffer))
            q_losses.append(q_loss)
            actor_losses.append(actor_loss)
        q_loss = torch.stack(q_losses).mean()
        actor_loss = torch.stack(actor_losses).mean()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target-критики понемногу догоняют обычных
        with torch.no_grad():
            for target, net in [(self.q1_target, self.q1), (self.q2_target, self.q2)]:
                for tp, p in zip(target.parameters(), net.parameters()):
                    tp.lerp_(p, self.cfg["tau"])

        return q_loss.item(), actor_loss.item()

    def train(self):
        self.run_dir = utils.start_run(self.cfg, "dsac", self.optimal)
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
        for episode in range(self.cfg["episodes"]):
            # 1. играем эпизод и кладём его в буфер
            ep = self.play_episode()

            # 2. учимся на случайных эпизодах из буфера (первые learning_starts эпизодов только копим опыт)
            q_losses, actor_losses = [], []
            if len(self.buffer) >= self.cfg["learning_starts"]:
                for _ in range(self.cfg["updates_per_episode"]):
                    q_loss, actor_loss = self.update()
                    q_losses.append(q_loss)
                    actor_losses.append(actor_loss)
            if len(q_losses) > 0:
                q_loss = np.mean(q_losses)
                actor_loss = np.mean(actor_losses)
            else:
                q_loss = np.nan
                actor_loss = np.nan

            # 3. метрики эпизода; лоссы — среднее по обновлениям сразу после этого эпизода
            self.history.append({
                "episode": episode,
                "env_steps": self.steps,
                "return": self.cfg["gamma"] ** (ep["length"] - 1) * ep["success"],
                "success": ep["success"],
                "length": ep["length"],
                "explore_prob": 1 - np.mean(ep["greedy_prob"]),
                "entropy": np.mean(ep["entropy"]),
                "loss": q_loss + actor_loss,
                "q_loss": q_loss,
                "actor_loss": actor_loss,
                "time": time.time() - start_time,
            })

            stop = self.should_stop(episode)

            # 4. сохраняем прогресс на диск, чтобы не потерять его при падении
            if (episode + 1) % self.cfg.get("save_every", 50) == 0:
                utils.save_progress(self.run_dir, self.history, "running", weights=self.weights())
            if (episode + 1) % 50 == 0:
                last = self.history[-50:]
                utils.log(self.run_dir,
                          f"episode {episode + 1} | steps {self.steps} | return {np.mean([r['return'] for r in last]):.3f} "
                          f"(opt {self.optimal['return']:.3f}) | length {np.mean([r['length'] for r in last]):.1f} "
                          f"| entropy {np.mean([r['entropy'] for r in last]):.2f}")
            if stop:
                break

    def evaluate(self, n_episodes=100, greedy=False, first_seed=10_000):
        """Оценка без обучения: эпизоды не попадают в буфер, старты по сидам first_seed, first_seed+1, ..."""
        torch.manual_seed(first_seed)  # чтобы сэмплирование действий тоже повторялось
        return [self.play_episode(greedy, learn=False, seed=first_seed + i) for i in range(n_episodes)]

    def weights(self):
        return {"actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict()}

    def save(self):
        """Сохранить текущее состояние вручную (train() и так сохраняет всё сам)."""
        utils.save_progress(self.run_dir, self.history, "completed", weights=self.weights())
