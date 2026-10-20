"""Discrete SAC (off-policy актор-критик) с опциональной памятью (GRU).

Цикл: играем в num_envs средах, складываем опыт в replay-буфер;
раз в train_every шагов берём из буфера случайные окна последовательных шагов
и обновляем два Q-критика, актора и температуру alpha.
Окно = burn_in шагов «прогрева» памяти + seq_len шагов, на которых считается лосс.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from gridrl.agents.networks import Head
from gridrl.envs import make_env, make_vec_env, optimal_stats
from gridrl.utils import get_device, set_seed


@dataclass
class DSACConfig:
    setup: int = 2
    env_seed: int = 0  # карта (покраска, препятствия, цель)
    seed: int = 0  # инициализация сетей, стартовые позиции, сэмплирование из буфера
    memory: bool = True
    see_obstacle: bool = True
    obs_mode: str = "color"
    soft_q: bool = True  # мягкая Q (с энтропией в таргете) или обычная
    total_steps: int = 200_000
    num_envs: int = 8
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    train_every: int = 4  # одно обновление раз в столько шагов векторной среды
    batch_size: int = 32  # окон в батче
    burn_in: int = 8
    seq_len: int = 32
    gamma: float = 0.99
    tau: float = 0.005  # скорость обновления target-критиков
    q_lr: float = 3e-4
    actor_lr: float = 3e-4
    alpha: float = 0.005  # начальная температура; больше ~0.02 — политика почти случайна (см. docs)
    autotune: bool = True  # подбирать alpha под целевую энтропию
    target_entropy_scale: float = 0.1  # целевая энтропия = scale * log(n_actions)
    hidden: int = 128
    log_every: int = 2048
    device: str = "auto"
    run_dir: str = "runs"
    run_name: str = ""


class SequenceReplay:
    """Кольцевой буфер формы (время, среда): шаги каждой среды лежат подряд,
    поэтому из него можно доставать окна последовательных шагов для GRU."""

    def __init__(self, capacity: int, num_envs: int, obs_dim: int, n_actions: int, seed: int):
        self.cap = capacity // num_envs
        self.num_envs = num_envs
        self.obs = np.zeros((self.cap, num_envs, obs_dim), dtype=np.float32)
        self.prev_a = np.zeros((self.cap, num_envs, n_actions), dtype=np.float32)
        self.starts = np.zeros((self.cap, num_envs), dtype=np.float32)
        self.actions = np.zeros((self.cap, num_envs), dtype=np.int64)
        self.rewards = np.zeros((self.cap, num_envs), dtype=np.float32)
        self.terminated = np.zeros((self.cap, num_envs), dtype=np.float32)
        self.truncated = np.zeros((self.cap, num_envs), dtype=np.float32)
        self.ptr = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.cap if self.full else self.ptr

    def add(self, obs, prev_a, starts, actions, rewards, terminated, truncated):
        i = self.ptr
        self.obs[i], self.prev_a[i], self.starts[i] = obs, prev_a, starts
        self.actions[i], self.rewards[i] = actions, rewards
        self.terminated[i], self.truncated[i] = terminated, truncated
        self.ptr = (self.ptr + 1) % self.cap
        self.full = self.full or self.ptr == 0

    def sample(self, batch: int, length: int, device) -> dict[str, torch.Tensor]:
        """batch окон длины length из случайных сред. Возвращает тензоры (length, batch, ...)."""
        oldest = self.ptr if self.full else 0
        offsets = self.rng.integers(0, len(self) - length + 1, size=batch)
        t_idx = (oldest + offsets[None, :] + np.arange(length)[:, None]) % self.cap
        e_idx = self.rng.integers(0, self.num_envs, size=batch)[None, :]
        out = {
            name: torch.as_tensor(getattr(self, name)[t_idx, e_idx], device=device)
            for name in ("obs", "prev_a", "starts", "actions", "rewards", "terminated", "truncated")
        }
        out["starts"][0] = 1.0  # память в начале окна неизвестна — начинаем с нуля
        return out


def polyak_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.lerp_(sp, tau)


def train_dsac(cfg: DSACConfig) -> Path:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    envs = make_vec_env(
        cfg.setup, cfg.num_envs, seed=cfg.env_seed,
        see_obstacle=cfg.see_obstacle, obs_mode=cfg.obs_mode,
    )
    n_envs = cfg.num_envs
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    run_name = cfg.run_name or (
        f"dsac_s{cfg.setup}_{'gru' if cfg.memory else 'mlp'}_seed{cfg.seed}_{int(time.time())}"
    )
    out_dir = Path(cfg.run_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = optimal_stats(make_env(cfg.setup, seed=cfg.env_seed), cfg.gamma)
    (out_dir / "config.json").write_text(
        json.dumps({"algo": "dsac", **asdict(cfg), "optimal": reference}, indent=2)
    )

    def make_net(gain):
        return Head(obs_dim, n_actions, cfg.hidden, cfg.memory, gain=gain).to(device)

    actor = make_net(0.01)
    q1, q2 = make_net(1.0), make_net(1.0)
    q1_target, q2_target = make_net(1.0), make_net(1.0)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.q_lr)
    log_alpha = torch.tensor(np.log(cfg.alpha), dtype=torch.float32, device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.q_lr)
    target_entropy = cfg.target_entropy_scale * np.log(n_actions)

    buffer = SequenceReplay(cfg.buffer_size, n_envs, obs_dim, n_actions, cfg.seed)
    window = cfg.burn_in + cfg.seq_len + 1  # +1: для последнего шага нужен следующий

    def update() -> dict[str, float]:
        b = buffer.sample(cfg.batch_size, window, device)
        alpha = log_alpha.exp().detach()
        h0 = actor.initial_state(cfg.batch_size, device)
        cur = slice(cfg.burn_in, window - 1)  # шаги, на которых считается лосс
        nxt = slice(cfg.burn_in + 1, window)  # следующие за ними

        # --- критики ---
        with torch.no_grad():
            logits_next, _ = actor.unroll(b["obs"], b["prev_a"], b["starts"], h0)
            pi_next = F.softmax(logits_next[nxt], -1)
            logpi_next = F.log_softmax(logits_next[nxt], -1)
            q1t, _ = q1_target.unroll(b["obs"], b["prev_a"], b["starts"], h0)
            q2t, _ = q2_target.unroll(b["obs"], b["prev_a"], b["starts"], h0)
            min_q_next = torch.min(q1t[nxt], q2t[nxt])
            if cfg.soft_q:
                v_next = (pi_next * (min_q_next - alpha * logpi_next)).sum(-1)
            else:
                v_next = (pi_next * min_q_next).sum(-1)
            target = b["rewards"][cur] + cfg.gamma * (1.0 - b["terminated"][cur]) * v_next
            # на шаге обрыва по времени следующего наблюдения в буфере нет — не учимся на нём
            mask = 1.0 - b["truncated"][cur]

        q1_all, _ = q1.unroll(b["obs"], b["prev_a"], b["starts"], h0)
        q2_all, _ = q2.unroll(b["obs"], b["prev_a"], b["starts"], h0)
        a = b["actions"][cur].unsqueeze(-1)
        q1_a = q1_all[cur].gather(-1, a).squeeze(-1)
        q2_a = q2_all[cur].gather(-1, a).squeeze(-1)
        q_loss = (((q1_a - target) ** 2 + (q2_a - target) ** 2) * mask).sum() / mask.sum().clamp(min=1)

        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        # --- актор ---
        logits, _ = actor.unroll(b["obs"], b["prev_a"], b["starts"], h0)
        pi = F.softmax(logits[cur], -1)
        logpi = F.log_softmax(logits[cur], -1)
        min_q = torch.min(q1_all[cur], q2_all[cur]).detach()
        actor_loss = (pi * (alpha * logpi - min_q)).sum(-1).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # --- температура ---
        entropy = -(pi * logpi).sum(-1).detach()
        alpha_loss = torch.zeros((), device=device)
        if cfg.autotune:
            alpha_loss = (log_alpha.exp() * (entropy - target_entropy)).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

        polyak_update(q1_target, q1, cfg.tau)
        polyak_update(q2_target, q2, cfg.tau)

        return {
            "loss": q_loss.item() + actor_loss.item() + alpha_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item(),
            "entropy": entropy.mean().item(),
            "q_mean": q1_a.mean().item(),
        }

    obs_np, _ = envs.reset(seed=cfg.seed)
    obs = torch.as_tensor(obs_np, device=device)
    prev_a = torch.zeros(n_envs, n_actions, device=device)
    starts = torch.ones(n_envs, device=device)
    h = actor.initial_state(n_envs, device)
    ep_len = np.zeros(n_envs, dtype=np.int64)

    global_step, vec_step, next_log = 0, 0, cfg.log_every
    ep_success, ep_lengths, ep_returns, greedy_prob, upd_stats = [], [], [], [], []
    t_start = time.time()
    log_file = open(out_dir / "metrics.csv", "w", newline="")
    writer = None

    while global_step < cfg.total_steps:
        with torch.no_grad():
            logits, h = actor.step(obs, prev_a, starts, h)
        dist = Categorical(logits=logits)
        action = dist.sample()
        greedy_prob.append(dist.probs.max(-1).values.mean().item())

        obs_np, reward, term, trunc, _ = envs.step(action.cpu().numpy())
        buffer.add(
            obs.cpu().numpy(), prev_a.cpu().numpy(), starts.cpu().numpy(),
            action.cpu().numpy(), reward, term, trunc,
        )
        global_step += n_envs
        vec_step += 1
        ep_len += 1

        done = term | trunc
        for i in np.flatnonzero(done):
            ep_success.append(float(term[i]))
            ep_lengths.append(ep_len[i])
            ep_returns.append(cfg.gamma ** (ep_len[i] - 1) if term[i] else 0.0)
            ep_len[i] = 0

        obs = torch.as_tensor(obs_np, device=device)
        starts = torch.as_tensor(done, dtype=torch.float32, device=device)
        prev_a = F.one_hot(action, n_actions).float() * (1.0 - starts).unsqueeze(-1)

        if global_step >= cfg.learning_starts and vec_step % cfg.train_every == 0:
            upd_stats.append(update())

        if global_step >= next_log or global_step >= cfg.total_steps:
            next_log += cfg.log_every
            row = {
                "global_step": global_step,
                "time": round(time.time() - t_start, 2),
                "episodes": len(ep_success),
                "success": np.mean(ep_success) if ep_success else np.nan,
                "ep_length": np.mean(ep_lengths) if ep_lengths else np.nan,
                "return": np.mean(ep_returns) if ep_returns else np.nan,
                "explore_prob": 1.0 - np.mean(greedy_prob),
            }
            for k in ("loss", "q_loss", "actor_loss", "alpha_loss", "alpha", "entropy", "q_mean"):
                row[k] = np.mean([s[k] for s in upd_stats]) if upd_stats else np.nan
            if writer is None:
                writer = csv.DictWriter(log_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log_file.flush()
            if len(upd_stats) or global_step <= cfg.log_every:
                sps = int(global_step / (time.time() - t_start))
                print(
                    f"step {global_step:>8} | return {row['return']:.3f} (opt {reference['return']:.3f})"
                    f" | success {row['success']:.2f} | len {row['ep_length']:.1f}"
                    f" | entropy {row['entropy']:.2f} | alpha {row['alpha']:.3f} | {sps} sps"
                )
            ep_success, ep_lengths, ep_returns, greedy_prob, upd_stats = [], [], [], [], []

    log_file.close()
    envs.close()
    torch.save(
        {"actor": actor.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict()},
        out_dir / "model.pt",
    )
    return out_dir
