"""PPO (актор-критик) с GAE и опциональной памятью (GRU).

Цикл: собрать роллаут rollout_len шагов в num_envs средах -> посчитать
преимущества через GAE -> несколько эпох градиентного спуска по
clipped-лоссу PPO. Для GRU минибатчи режутся по средам, чтобы
последовательности оставались целыми.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from gridrl.agents.networks import ActorCritic
from gridrl.envs import make_env, optimal_stats
from gridrl.utils import get_device, set_seed


@dataclass
class PPOConfig:
    setup: int = 2
    env_seed: int = 0  # карта (покраска, препятствия, цель)
    seed: int = 0  # инициализация сети и стартовые позиции
    memory: bool = True
    see_obstacle: bool = True
    obs_mode: str = "color"
    total_steps: int = 300_000
    num_envs: int = 16
    rollout_len: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    device: str = "auto"
    run_dir: str = "runs"
    run_name: str = ""


def make_vec_env(cfg: PPOConfig) -> gym.vector.SyncVectorEnv:
    def factory():
        return make_env(
            cfg.setup, seed=cfg.env_seed, see_obstacle=cfg.see_obstacle, obs_mode=cfg.obs_mode
        )

    # SAME_STEP: после конца эпизода step сразу возвращает наблюдение нового,
    # а последнее наблюдение старого лежит в info["final_obs"].
    return gym.vector.SyncVectorEnv(
        [factory] * cfg.num_envs, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """Generalized Advantage Estimation. Все тензоры (T, N); dones[t] — эпизод кончился на шаге t."""
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        next_v = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * not_done - values[t]
        last = delta + gamma * lam * not_done * last
        advantages[t] = last
    return advantages, advantages + values


def train_ppo(cfg: PPOConfig) -> Path:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    envs = make_vec_env(cfg)
    n_envs, T = cfg.num_envs, cfg.rollout_len
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    run_name = cfg.run_name or (
        f"ppo_s{cfg.setup}_{'gru' if cfg.memory else 'mlp'}_seed{cfg.seed}_{int(time.time())}"
    )
    out_dir = Path(cfg.run_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = optimal_stats(make_env(cfg.setup, seed=cfg.env_seed), cfg.gamma)
    (out_dir / "config.json").write_text(
        json.dumps({"algo": "ppo", **asdict(cfg), "optimal": reference}, indent=2)
    )

    model = ActorCritic(obs_dim, n_actions, cfg.hidden, cfg.memory).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    # Буферы роллаута, форма (T, N, ...).
    buf_obs = torch.zeros(T, n_envs, obs_dim, device=device)
    buf_prev_a = torch.zeros(T, n_envs, n_actions, device=device)
    buf_starts = torch.zeros(T, n_envs, device=device)
    buf_actions = torch.zeros(T, n_envs, dtype=torch.long, device=device)
    buf_logp = torch.zeros(T, n_envs, device=device)
    buf_values = torch.zeros(T, n_envs, device=device)
    buf_rewards = torch.zeros(T, n_envs, device=device)
    buf_dones = torch.zeros(T, n_envs, device=device)

    obs_np, _ = envs.reset(seed=cfg.seed)
    obs = torch.as_tensor(obs_np, device=device)
    prev_a = torch.zeros(n_envs, n_actions, device=device)
    starts = torch.ones(n_envs, device=device)
    h = model.initial_state(n_envs, device)
    ep_len = np.zeros(n_envs, dtype=np.int64)

    num_updates = cfg.total_steps // (n_envs * T)
    mb_envs = n_envs // cfg.num_minibatches
    global_step = 0
    t_start = time.time()

    log_file = open(out_dir / "metrics.csv", "w", newline="")
    writer = None

    for update in range(1, num_updates + 1):
        h_init = h.clone()
        ep_success, ep_lengths, ep_returns = [], [], []
        greedy_prob = []

        # --- 1. сбор опыта ---------------------------------------------------
        for t in range(T):
            buf_obs[t], buf_prev_a[t], buf_starts[t] = obs, prev_a, starts
            with torch.no_grad():
                logits, value, h = model.step(obs, prev_a, starts, h)
            dist = Categorical(logits=logits)
            action = dist.sample()
            buf_actions[t], buf_logp[t], buf_values[t] = action, dist.log_prob(action), value
            greedy_prob.append(dist.probs.max(-1).values.mean().item())

            obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
            global_step += n_envs
            ep_len += 1
            reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
            action_1h = F.one_hot(action, n_actions).float()

            # Обрыв по времени — не провал: добавляем к награде оценку
            # ценности последнего наблюдения (bootstrap), будто эпизод продолжился.
            if trunc.any():
                idx = torch.as_tensor(np.flatnonzero(trunc), device=device)
                final_obs = torch.as_tensor(
                    np.stack(info["final_obs"][trunc]), dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    _, v_final, _ = model.step(
                        final_obs, action_1h[idx], torch.zeros(len(idx), device=device), h[idx]
                    )
                reward[idx] += cfg.gamma * v_final

            done = term | trunc
            buf_rewards[t] = reward
            buf_dones[t] = torch.as_tensor(done, dtype=torch.float32, device=device)

            for i in np.flatnonzero(done):
                ep_success.append(float(term[i]))
                ep_lengths.append(ep_len[i])
                ep_returns.append(cfg.gamma ** (ep_len[i] - 1) if term[i] else 0.0)
                ep_len[i] = 0

            obs = torch.as_tensor(obs_np, device=device)
            starts = buf_dones[t].clone()
            prev_a = action_1h * (1.0 - starts).unsqueeze(-1)

        # --- 2. преимущества ----------------------------------------------------
        with torch.no_grad():
            _, next_value, _ = model.step(obs, prev_a, starts, h)
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, next_value, cfg.gamma, cfg.gae_lambda
        )

        # --- 3. обновление ------------------------------------------------------
        stats = {k: [] for k in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")}
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(n_envs, device=device)
            for start in range(0, n_envs, mb_envs):
                mb = perm[start:start + mb_envs]
                logits, values, _ = model.unroll(
                    buf_obs[:, mb], buf_prev_a[:, mb], buf_starts[:, mb], h_init[mb]
                )
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(buf_actions[:, mb])
                entropy = dist.entropy().mean()

                log_ratio = new_logp - buf_logp[:, mb]
                ratio = log_ratio.exp()
                adv = advantages[:, mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = torch.max(
                    -adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                ).mean()
                value_loss = 0.5 * ((values - returns[:, mb]) ** 2).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    stats["loss"].append(loss.item())
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.item())
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())

        # --- 4. логирование -----------------------------------------------------
        row = {
            "update": update,
            "global_step": global_step,
            "time": round(time.time() - t_start, 2),
            "episodes": len(ep_success),
            "success": np.mean(ep_success) if ep_success else np.nan,
            "ep_length": np.mean(ep_lengths) if ep_lengths else np.nan,
            "return": np.mean(ep_returns) if ep_returns else np.nan,
            "explore_prob": 1.0 - np.mean(greedy_prob),
            **{k: np.mean(v) for k, v in stats.items()},
        }
        if writer is None:
            writer = csv.DictWriter(log_file, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        log_file.flush()
        if update % 10 == 0 or update == num_updates:
            sps = int(global_step / (time.time() - t_start))
            print(
                f"step {global_step:>8} | return {row['return']:.3f} (opt {reference['return']:.3f})"
                f" | success {row['success']:.2f} | len {row['ep_length']:.1f}"
                f" | entropy {row['entropy']:.2f} | {sps} sps"
            )

    log_file.close()
    envs.close()
    torch.save(model.state_dict(), out_dir / "model.pt")
    return out_dir
