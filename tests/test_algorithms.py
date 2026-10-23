"""Быстрые проверки PPO и DSAC: обучение идёт, лоссы конечные, градиенты там, где нужно, файлы сохраняются."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import DSAC, PPO

ENV2 = dict(h=5, w=5, n_colors=5, obstacle_fraction=0.0)

PPO_CFG = {
    "device": "cpu", "env_seed": 0, "seed": 0, "memory": True, "episodes": 6, "episodes_per_update": 3,
    "epochs": 2, "episodes_per_batch": 2, "gamma": 0.99, "gae_lambda": 0.95, "lr": 3e-4, "clip": 0.2,
    "ent_coef": 0.01, "vf_coef": 0.5, "hidden": 16, "run_name": "test_ppo", "save_every": 3, **ENV2,
}
DSAC_CFG = {
    "device": "cpu", "env_seed": 0, "seed": 0, "memory": True, "soft_q": True, "episodes": 4,
    "buffer_episodes": 100, "learning_starts": 1, "updates_per_episode": 1, "batch_size": 2,
    "gamma": 0.99, "tau": 0.005, "lr": 3e-4, "alpha": 0.005, "hidden": 16, "run_name": "test_dsac",
    "save_every": 2, **ENV2,
}


def params(net):
    return [p.detach().clone() for p in net.parameters()]


def changed(before, net):
    return any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))


@pytest.mark.parametrize("memory", [True, False])
def test_ppo_trains_and_saves(tmp_path, memory):
    alg = PPO({**PPO_CFG, "memory": memory, "out_dir": str(tmp_path)})
    before = params(alg.net)
    alg.train()
    assert changed(before, alg.net)
    assert len(alg.history) == 6
    for row in alg.history:
        assert np.isfinite(row["loss"]) and np.isfinite(row["policy_loss"]) and np.isfinite(row["value_loss"])
    run = tmp_path / "test_ppo"
    for name in ["config.json", "metrics.csv", "model.pt", "status.json", "train.log"]:
        assert (run / name).exists(), name
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    weights = torch.load(run / "model.pt")
    alg.net.load_state_dict(weights["net"])  # чекпоинт загружается обратно


def test_ppo_gradients_reach_all_parameters():
    alg = PPO({**PPO_CFG, "out_dir": "unused"})
    episodes = [alg.play_episode() for _ in range(3)]
    for ep in episodes:
        ep["advantages"], ep["returns"] = alg.compute_advantages(ep)
    ep = episodes[0]
    logits, values, _ = alg.net(torch.tensor(np.array(ep["obs"])), torch.tensor(ep["prev_actions"]))
    (logits.sum() + values.sum()).backward()
    for name, p in alg.net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_ppo_evaluate_does_not_learn():
    alg = PPO({**PPO_CFG, "out_dir": "unused"})
    before = params(alg.net)
    steps = alg.steps
    episodes = alg.evaluate(3)
    assert len(episodes) == 3 and alg.steps == steps and not changed(before, alg.net)
    # одинаковые сиды -> одинаковые старты
    again = alg.evaluate(3)
    assert [ep["positions"][0] for ep in episodes] == [ep["positions"][0] for ep in again]


def test_gae_terminal_and_truncated():
    alg = PPO({**PPO_CFG, "out_dir": "unused"})
    ep = {"rewards": [0.0, 1.0], "values": [0.5, 0.8], "last_value": 0.0}
    adv, ret = alg.compute_advantages(ep)
    g, lam = 0.99, 0.95
    d1 = 1.0 - 0.8  # последний шаг, после цели будущего нет
    d0 = 0.0 + g * 0.8 - 0.5
    assert np.allclose(adv, [d0 + g * lam * d1, d1])
    ep["last_value"] = 2.0  # обрыв по времени: добавляется gamma * V(последнее наблюдение)
    adv2, _ = alg.compute_advantages(ep)
    assert np.isclose(adv2[1], 1.0 + g * 2.0 - 0.8)


@pytest.mark.parametrize("memory", [True, False])
@pytest.mark.parametrize("soft_q", [True, False])
def test_dsac_trains_and_saves(tmp_path, memory, soft_q):
    alg = DSAC({**DSAC_CFG, "memory": memory, "soft_q": soft_q, "out_dir": str(tmp_path)})
    before_actor, before_q1 = params(alg.actor), params(alg.q1)
    alg.train()
    assert changed(before_actor, alg.actor) and changed(before_q1, alg.q1)
    finite = [row for row in alg.history if not np.isnan(row["q_loss"])]
    assert len(finite) > 0 and all(np.isfinite(r["q_loss"]) and np.isfinite(r["actor_loss"]) for r in finite)
    run = tmp_path / "test_dsac"
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    assert (run / "model.pt").exists() and (run / "metrics.csv").exists()


def test_dsac_target_critics_have_no_grad_and_move_slowly():
    alg = DSAC({**DSAC_CFG, "out_dir": "unused"})
    for _ in range(3):
        alg.play_episode()
    target_before = params(alg.q1_target)
    alg.update()
    assert all(p.grad is None for p in alg.q1_target.parameters())
    assert changed(target_before, alg.q1_target)  # чуть сдвинулись к q1
    assert not all(torch.equal(a, b) for a, b in zip(alg.q1_target.parameters(), alg.q1.parameters()))


def test_dsac_actor_loss_does_not_touch_critics():
    alg = DSAC({**DSAC_CFG, "out_dir": "unused"})
    alg.play_episode()
    q_loss, actor_loss = alg.episode_losses(alg.buffer[0])
    actor_loss.backward()
    assert all(p.grad is None for p in alg.q1.parameters())
    assert any(p.grad is not None for p in alg.actor.parameters())


def test_dsac_soft_q_changes_target():
    alg = DSAC({**DSAC_CFG, "alpha": 1.0, "out_dir": "unused"})
    alg.play_episode()
    ep = alg.buffer[0]
    torch.manual_seed(0)
    soft, _ = alg.episode_losses(ep)
    alg.cfg = {**alg.cfg, "soft_q": False}
    hard, _ = alg.episode_losses(ep)
    assert not torch.isclose(soft, hard)


def test_dsac_evaluate_does_not_learn():
    alg = DSAC({**DSAC_CFG, "out_dir": "unused"})
    alg.play_episode()
    buffer_len, steps = len(alg.buffer), alg.steps
    before = params(alg.actor)
    alg.evaluate(3)
    assert len(alg.buffer) == buffer_len and alg.steps == steps and not changed(before, alg.actor)


@pytest.mark.skipif(not Path("data/mnist/train-images-idx3-ubyte.gz").exists(), reason="нет MNIST в data/mnist")
def test_ppo_mnist_smoke(tmp_path):
    alg = PPO({**PPO_CFG, "mnist": True, "episodes": 3, "out_dir": str(tmp_path)})
    alg.train()
    assert alg.env.observation_space.shape == (196,)
    assert np.isfinite(alg.history[0]["loss"])


def test_has_converged():
    from src import utils
    rising = [{"return": i / 600} for i in range(600)]
    flat = [{"return": 0.9} for _ in range(600)]
    assert not utils.has_converged(rising, 200, 0.01)
    assert utils.has_converged(flat, 200, 0.01)
    assert not utils.has_converged(flat[:500], 200, 0.01)  # мало эпизодов


def test_max_env_steps_stops_training(tmp_path):
    alg = PPO({**PPO_CFG, "episodes": 1000, "max_env_steps": 200, "out_dir": str(tmp_path)})
    alg.train()
    assert alg.steps >= 200 and len(alg.history) < 1000
    alg = DSAC({**DSAC_CFG, "episodes": 1000, "max_env_steps": 150, "out_dir": str(tmp_path)})
    alg.train()
    assert alg.steps >= 150 and len(alg.history) < 1000


def test_stop_when_converged(tmp_path):
    # среда 1x2: старт рядом с целью, любой шаг вправо — успех; плато наступает сразу
    alg = PPO({**PPO_CFG, "h": 2, "w": 2, "n_colors": 4, "episodes": 2000, "stop_when_converged": True,
               "conv_window": 50, "conv_tol": 0.5, "out_dir": str(tmp_path)})
    alg.train()
    status = json.loads((tmp_path / "test_ppo" / "status.json").read_text())
    assert status["converged"] and status["converged_at"]["episode"] < 2000


def test_min_episodes_delays_convergence(tmp_path):
    alg = PPO({**PPO_CFG, "h": 2, "w": 2, "n_colors": 4, "episodes": 2000, "stop_when_converged": True,
               "conv_window": 50, "conv_tol": 0.5, "min_episodes": 400, "out_dir": str(tmp_path)})
    alg.train()
    assert alg.converged_at is not None and alg.converged_at["episode"] >= 400
