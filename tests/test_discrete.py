"""Discrete-action agents (src/rl/discrete.py) and train_discrete.py's setup:
targets, exploration, the temperature update, target networks, configs."""

import math
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.rl.discrete import DQN, DiscreteSAC, epsilon_greedy, linear_schedule

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
ROOT = str(Path(CONFIG_DIR).parent)


def _algo(name, **over):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_discrete", overrides=[f"algo={name}"])
    algo = cfg.algo
    for k, v in over.items():
        OmegaConf.update(algo, k, v)
    return algo


def _batch(n=8, d=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g), torch.randint(4, (n,), generator=g), torch.rand(n, generator=g),
            torch.full((n,), 0.9), torch.randn(n, d, generator=g))


@pytest.mark.parametrize("double", [True, False])
def test_dqn_target(double):
    torch.manual_seed(0)
    agent = DQN(5, 4, _algo("dqn", double=double), obs_norm=False)
    for p in agent.q_target.parameters():                       # make online and target differ
        p.data.add_(0.5 * torch.randn_like(p))
    feats, actions, rewards, discounts, nxt = _batch()
    _, m = agent.critic_loss(feats, actions, rewards, discounts, nxt)
    with torch.no_grad():
        chooser = agent.q if double else agent.q_target
        a = chooser(nxt).argmax(1)
        want = rewards + discounts * agent.q_target(nxt)[torch.arange(8), a]
    assert m["train/target_q_mean"] == pytest.approx(want.mean().item(), rel=1e-6)


def test_dqn_explores_by_epsilon():
    agent = DQN(5, 4, _algo("dqn", total_timesteps=200,
                            **{"epsilon.start": 1.0, "epsilon.end": 0.0, "epsilon.fraction": 0.5}), obs_norm=False)
    feats = torch.randn(4000, 5)
    greedy = agent.act(feats, 0, explore=False)
    assert agent.epsilon(0) == 1.0 and agent.epsilon(50) == pytest.approx(0.5) and agent.epsilon(500) == 0.0
    assert torch.equal(agent.act(feats, 500, explore=True), greedy)                     # epsilon 0
    agree = (agent.act(feats, 50, explore=True) == greedy).float().mean().item()
    assert agree == pytest.approx(0.5 + 0.5 / 4, abs=0.04)                              # half random, a quarter of those agree


def test_hard_target_update_waits_for_its_step():
    agent = DQN(5, 4, _algo("dqn", tau=1.0, target_network_frequency=10), obs_norm=False)
    for p in agent.q.parameters():
        p.data.add_(1.0)
    agent.update_targets(5)
    assert not torch.equal(agent.q[1].weight, agent.q_target[1].weight)
    agent.update_targets(10)
    assert torch.equal(agent.q[1].weight, agent.q_target[1].weight)


def test_discrete_sac_soft_value_is_an_exact_expectation():
    torch.manual_seed(0)
    agent = DiscreteSAC(5, 4, _algo("sac_discrete", autotune_alpha=False, alpha=0.3), obs_norm=False)
    feats, actions, rewards, discounts, nxt = _batch()
    _, m = agent.critic_loss(feats, actions, rewards, discounts, nxt)
    with torch.no_grad():
        p, logp = agent.policy(nxt)
        q = torch.min(agent.q1_target(nxt), agent.q2_target(nxt))
        want = rewards + discounts * (p * (q - 0.3 * logp)).sum(1)
    assert agent.alpha == pytest.approx(0.3) and m["train/target_q_mean"] == pytest.approx(want.mean().item(), rel=1e-5)


@pytest.mark.parametrize("scale, direction", [(0.01, -1), (0.999, +1)])
def test_temperature_moves_towards_the_target_entropy(scale, direction):
    # A fresh softmax policy is near uniform (entropy ~ log 4): a tiny target
    # entropy should lower alpha, one at the maximum should raise it.
    torch.manual_seed(0)
    agent = DiscreteSAC(5, 4, _algo("sac_discrete", target_entropy_scale=scale, alpha_lr=0.05), obs_norm=False)
    assert agent.target_entropy == pytest.approx(scale * math.log(4))
    before = agent.alpha
    feats = torch.randn(64, 5)
    for _ in range(5):
        agent.after_critic(feats, step=1)
    assert np.sign(agent.alpha - before) == direction


def test_discrete_sac_acts_greedily_in_evaluation():
    agent = DiscreteSAC(5, 4, _algo("sac_discrete"), obs_norm=False)
    feats = torch.randn(200, 5)
    probs, _ = agent.policy(feats)
    assert torch.equal(agent.act(feats, 0, explore=False), probs.argmax(1))
    assert agent.act(feats, 0, explore=True).shape == (200,)


def test_linear_schedule_clamps():
    assert linear_schedule(1.0, 0.1, 10, 0) == 1.0 and linear_schedule(1.0, 0.1, 10, 99) == pytest.approx(0.1)


@pytest.mark.parametrize("overrides, algo, env", [([], "dqn", "fourrooms_pixels"),
                                                  (["algo=sac_discrete"], "sac_discrete", "fourrooms_pixels"),
                                                  (["env=fourrooms_state", "embedding=none"], "dqn", "fourrooms_state")])
def test_config_discrete_composes(overrides, algo, env):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_discrete", overrides=overrides)
    assert cfg.algo.name == algo and cfg.env.name == env and "buffer_compression" in cfg.algo


def test_fourrooms_state_observation_includes_the_goal():
    pytest.importorskip("gym_gridworlds")
    from src.sac_utils import _make_state
    env = _make_state("Gym-Gridworlds/FourRooms-13x13-v0", 0,
                      wrappers=["gym_gridworlds.observation_wrappers.MatrixWithGoalWrapper"])
    obs, _ = env.reset(seed=0)
    assert obs.shape == (338,) and obs.sum() == 2                  # one agent cell, one goal cell


def test_evaluate_passes_a_discrete_action_as_an_int():
    from src.sac_utils import evaluate

    class Env(gym.Env):
        observation_space, action_space = gym.spaces.Box(-1, 1, (2,)), gym.spaces.Discrete(3)

        def reset(self, seed=None, options=None):
            self.t = 0
            return np.zeros(2, np.float32), {}

        def step(self, action):
            assert isinstance(action, int)
            self.t += 1
            return np.zeros(2, np.float32), 1.0, False, self.t >= 3, {}
    stats = evaluate(lambda o: torch.tensor([2]), Env(), 2, "cpu")
    assert stats["return_mean"] == 3.0


def test_evaluate_resets_to_the_given_seeds():
    pytest.importorskip("gym_gridworlds")
    from src.sac_utils import _make_state, evaluate
    env = _make_state("Gym-Gridworlds/FourRooms-13x13-v0", 0,
                      wrappers=["gym_gridworlds.observation_wrappers.MatrixWithGoalWrapper"])

    def observations(seeds):
        seen = []

        def act(o):
            seen.append(o.clone())
            return torch.tensor([len(seen) % 4])                             # cycles through the moves
        evaluate(act, env, len(seeds), "cpu", seeds=seeds)
        return torch.stack(seen)
    first = observations([3, 4, 5])
    assert torch.equal(observations([3, 4, 5]), first)                       # same episodes, step for step
    assert not torch.equal(observations([6, 7, 8])[0], first[0])             # other seeds, another start/goal
    act = lambda o: torch.tensor([0])  # noqa: E731
    with pytest.raises(ValueError, match="2 seeds for 3"):
        evaluate(act, env, 3, "cpu", seeds=[1, 2])


def test_epsilon_greedy_uses_its_own_generator():
    greedy = lambda f: torch.zeros(len(f), dtype=torch.long)  # noqa: E731
    feats = torch.randn(4000, 5)
    torch.manual_seed(0)
    want = torch.rand(3)
    torch.manual_seed(0)
    acts = epsilon_greedy(greedy, 4, 0.5, torch.Generator().manual_seed(1))(feats)
    assert torch.equal(torch.rand(3), want)                                  # global stream untouched
    assert (acts == 0).float().mean().item() == pytest.approx(0.5 + 0.5 / 4, abs=0.04)
    assert torch.equal(epsilon_greedy(greedy, 4, 0.0, torch.Generator())(feats), greedy(feats))


def test_epsilon_greedy_evaluation_logs_its_own_keys():
    from src.rl.common import log_eval

    class Wandb:
        def log(self, d, step):
            self.d = d
    w = Wandb()
    log_eval({"return_mean": 0.5, "return_std": 0.1, "success_rate": 0.5}, 10, w, prefix="eval_eps")
    assert set(w.d) == {"eval_eps/mean_reward", "eval_eps/std_reward", "eval_eps/success_rate", "global_step"}


def test_continuous_envs_are_refused():
    out = subprocess.run([sys.executable, "train_discrete.py", "env=pendulum", "embedding=none", "wandb.enabled=false",
                          "save.enabled=false", "device=cpu", "hydra.run.dir=/tmp/pvr_test_discrete_refuse"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "needs discrete actions" in out.stderr + out.stdout
