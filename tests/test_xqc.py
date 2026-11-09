"""XQC (src/rl/xqc.py): the categorical projection and loss, weight
normalization, BatchNorm placement and modes, the policy, reward
normalization, the discount heuristic, configs, and both scripts end to end."""

import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from src.rl.xqc import (
    CriticEnsemble,
    DrQv2XQCActor,
    NormedLinear,
    RewardNormalizer,
    TanhGaussianActor,
    actor_q,
    critic_loss,
    heuristic_discount,
    linear_lr,
    make_target,
    resolve_gamma,
    xqc_learning_rates,
    project_categorical,
    project_weights,
    xqc_mlp,
)

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
ROOT = os.path.dirname(CONFIG_DIR)


# ---- categorical projection ---------------------------------------------------

def test_projection_splits_mass_between_neighbouring_atoms():
    probs = torch.tensor([[1.0, 0.0, 0.0]])
    out = project_categorical(torch.tensor([[0.25, 0.0, 1.0]]), probs, -1.0, 1.0)
    assert torch.allclose(out, torch.tensor([[0.0, 0.75, 0.25]]))


def test_projection_on_an_atom_and_outside_the_support():
    probs = torch.tensor([[0.5, 0.5, 0.0]])
    on_atom = project_categorical(torch.tensor([[0.0, 7.0, -3.0]]), probs, -1.0, 1.0)
    assert torch.allclose(on_atom, torch.tensor([[0.0, 0.5, 0.5]]))   # 0 exact, 7 clipped to 1


def test_projection_keeps_mass_and_the_mean_inside_the_support():
    torch.manual_seed(0)
    atoms = torch.linspace(-5, 5, 101)
    probs = torch.randn(64, 101).softmax(-1)
    values = 0.1 + 0.97 * atoms                       # a reward and discount: stays in [-4.75, 4.95]
    out = project_categorical(values.expand(64, -1), probs, -5.0, 5.0)
    assert torch.allclose(out.sum(-1), torch.ones(64), atol=1e-6)
    assert torch.allclose((out * atoms).sum(-1), (probs * values).sum(-1), atol=1e-5)


# ---- weight normalization and BatchNorm ---------------------------------------

def test_projection_gives_unit_weight_rows_and_leaves_biases_alone():
    torch.manual_seed(0)
    critic = CriticEnsemble(4, 2, [16, 16])
    linears = [m for m in critic.modules() if isinstance(m, NormedLinear)]
    hidden, out = [m for m in linears if m.bias is None], [m for m in linears if m.bias is not None]
    assert len(hidden) == 4 and len(out) == 2          # hidden layers have no bias; outputs do
    for layer in linears:
        layer.weight.data.mul_(3.0)
    for layer in out:
        layer.bias.data.fill_(2.0)
    project_weights(critic)
    for layer in linears:
        assert torch.allclose(layer.weight.norm(dim=1), torch.ones(layer.out_features), atol=1e-6)
    for layer in out:
        assert torch.equal(layer.bias, torch.full((layer.out_features,), 2.0))


def test_layers_with_project_false_are_left_alone():
    layer = NormedLinear(5, 4, project=False)
    layer.weight.data.mul_(3.0)
    before = layer.weight.clone()
    project_weights(layer)
    assert torch.equal(layer.weight, before)


def test_hidden_linears_are_scale_invariant_through_their_batchnorm():
    torch.manual_seed(0)
    mlp = xqc_mlp(6, [16, 16]).train()
    assert mlp[1].bias is None and mlp[0].momentum == 0.01 and mlp[0].eps == 1e-3   # Flax's BN
    kinds = [type(m).__name__ for m in mlp]
    assert kinds == ["BatchNorm1d", "NormedLinear", "BatchNorm1d", "ReLU",
                     "NormedLinear", "BatchNorm1d", "ReLU"]
    for m in mlp:                                     # exact invariance up to BN's eps
        if isinstance(m, nn.BatchNorm1d):
            m.eps = 1e-8
    x = torch.randn(32, 6)
    before = mlp(x)
    mlp[1].weight.data.mul_(5.0)                      # first hidden Linear, before its BN
    assert torch.allclose(mlp(x), before, atol=1e-4)


def _critic(**kw):
    torch.manual_seed(0)
    c = CriticEnsemble(4, 2, [32, 32], **kw)
    project_weights(c)
    return c


def test_critic_batchnorm_sees_current_and_next_pairs_together():
    critic = _critic().eval()                         # as in the scripts
    target = make_target(critic)
    s, a, s2, a2 = torch.randn(8, 4), torch.rand(8, 2), torch.randn(8, 4) + 3.0, torch.rand(8, 2)
    critic_loss(critic, target, s, a, s2, a2, torch.zeros(8), torch.full((8,), 0.9))
    first_bn = critic.members[0][0][0]
    joint_mean = torch.cat([torch.cat([s, a], 1), torch.cat([s2, a2], 1)]).mean(0)
    assert torch.allclose(first_bn.running_mean, 0.01 * joint_mean, atol=1e-6)
    # The target critic's statistics never move (they're never used).
    assert torch.equal(target.members[0][0][0].running_mean, torch.zeros(6))
    assert not critic.training                        # mode restored


def test_categorical_loss_with_no_bootstrap_is_the_reward_atom_cross_entropy():
    critic = _critic(num_atoms=11, v_min=-5.0, v_max=5.0).eval()
    target = make_target(critic)
    s, a, s2, a2 = torch.randn(8, 4), torch.rand(8, 2), torch.randn(8, 4), torch.rand(8, 2)
    rewards = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])   # on the atoms -5..5
    loss, stats = critic_loss(critic, target, s, a, s2, a2, rewards, torch.zeros(8))
    per_critic, q = stats["per_critic"], stats["q"]
    assert torch.allclose(stats["target_mean"], rewards.mean())
    with torch.no_grad():                             # the same joint batch, batch statistics
        logp = critic.train()(torch.cat([s, s2]), torch.cat([a, a2]))[:, :8].log_softmax(-1)
    idx = (rewards + 5).long()
    expected = -logp[:, torch.arange(8), idx].mean(-1)
    assert torch.allclose(per_critic, expected, atol=1e-5) and torch.allclose(loss, expected.sum(), atol=1e-5)
    assert q.shape == (2, 8)


def test_categorical_target_comes_from_the_target_critic_with_the_smaller_q():
    critic = _critic(num_atoms=11)
    target = make_target(critic)
    with torch.no_grad():                             # member 1: all mass on +5; member 0: on -5
        target.members[0][-1].bias.copy_(torch.tensor([50.0] + [0.0] * 10))
        target.members[1][-1].bias.copy_(torch.tensor([0.0] * 10 + [50.0]))
    _, stats = critic_loss(critic, target, torch.randn(4, 4), torch.rand(4, 2),
                           torch.randn(4, 4), torch.rand(4, 2), torch.zeros(4), torch.ones(4))
    assert stats["target_mean"].item() == pytest.approx(-5.0, abs=1e-3)
    assert stats["mass_vmin"].item() == pytest.approx(1.0, abs=1e-3)   # all on the edge atom
    assert stats["mass_vmax"].item() == pytest.approx(0.0, abs=1e-3)


def test_edge_mass_shows_returns_beyond_the_support():
    critic = _critic(num_atoms=11)
    target = make_target(critic)
    rewards = torch.tensor([0.0, 0.0, 9.0, 9.0])       # half the targets above v_max
    _, stats = critic_loss(critic, target, torch.randn(4, 4), torch.rand(4, 2),
                           torch.randn(4, 4), torch.rand(4, 2), rewards, torch.zeros(4))
    assert stats["mass_vmax"].item() == pytest.approx(0.5) and stats["mass_vmin"].item() == 0.0
    # discount 0: every atom of a sample shifts to its reward, so half are clipped
    assert stats["clip_percentage"].item() == pytest.approx(0.5)


def test_mse_variant_matches_sac_target():
    critic = _critic(critic_loss="mse")
    target = make_target(critic)
    loss, stats = critic_loss(
        critic, target, torch.randn(8, 4), torch.rand(8, 2), torch.randn(8, 4), torch.rand(8, 2),
        torch.ones(8), torch.zeros(8), next_bonus=torch.full((8,), 7.0))
    assert stats["target_mean"].item() == pytest.approx(1.0)   # discount 0: the bonus is masked too
    assert stats["per_critic"].shape == (2,) and "mass_vmin" not in stats


def test_actor_q_uses_running_statistics_and_restores_the_mode():
    critic = _critic().train()
    first_bn = critic.members[0][0][0]
    before = first_bn.running_mean.clone()
    q = actor_q(critic, torch.randn(8, 4), torch.rand(8, 2))
    assert q.shape == (8,) and torch.equal(first_bn.running_mean, before) and critic.training


# ---- policies --------------------------------------------------------------------

def test_tanh_gaussian_log_prob_matches_torch_transformed_distribution():
    torch.manual_seed(0)
    actor = TanhGaussianActor(5, 3, [16, 16]).eval()
    feats = torch.randn(64, 5)
    torch.manual_seed(1)
    action, log_prob = actor.sample(feats)
    mean, log_std = actor(feats)
    dist = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mean, log_std.exp()),
        [torch.distributions.TanhTransform(cache_size=1)])
    torch.manual_seed(1)
    u = mean + log_std.exp() * torch.randn_like(mean)
    ref = dist.log_prob(torch.tanh(u)).sum(-1)
    keep = action.abs().max(-1).values < 0.999        # TanhTransform's inverse is inexact near +-1
    assert torch.allclose(log_prob[keep], ref[keep], atol=1e-3)
    assert action.abs().max() <= 1.0


def test_drqv2_xqc_actor_keeps_the_trunk_and_adds_batchnorm():
    actor = DrQv2XQCActor(64, 3, 50, [32, 32]).eval()
    dist = actor(torch.randn(5, 64), 0.5)
    assert dist.mean.shape == (5, 3) and dist.mean.abs().max() <= 1
    assert isinstance(actor.trunk[1], nn.LayerNorm) and isinstance(actor.body[0], nn.BatchNorm1d)


# ---- rewards, discount, learning rate -----------------------------------------------

def test_reward_normalizer_is_the_std_of_the_discounted_return():
    norm = RewardNormalizer(gamma=0.5)
    returns = []
    for episode in range(20):
        g = 0.0
        for t in range(10):
            r = float(episode % 3 + t)
            norm.update(r, done=t == 9)
            g = 0.5 * g + r
            returns.append(g)
    assert norm.G == 0.0                                    # reset at the episode end
    assert norm.scale == pytest.approx(np.std(returns), rel=1e-3)
    assert norm.normalize(torch.tensor([norm.scale])).item() == pytest.approx(1.0)


@pytest.mark.parametrize("length, repeat, gamma", [
    (200, 1, 0.975),       # Pendulum
    (1600, 1, 0.995),      # BipedalWalker (1 - 5/1600 = 0.9969, clipped)
    (1000, 2, 0.99),       # DMC at action repeat 2
    (50, 1, 0.95),         # clipped below
])
def test_discount_heuristic(length, repeat, gamma):
    assert heuristic_discount(length, repeat) == pytest.approx(gamma)


def test_learning_rates_follow_the_authors_per_optimizer_counts():
    # The paper's run: 500k agent steps, 5k warm-up, UTD 2, actor every 3rd update.
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        algo = compose(config_name="config_xqc").algo
    last = (algo.total_timesteps - algo.learning_starts) * algo.updates_per_step - 1
    last_actor = last - last % algo.policy_delay           # the last update the actor steps on
    critic_end, _ = xqc_learning_rates(last, algo)
    _, actor_end = xqc_learning_rates(last_actor, algo)
    # optax counts each optimizer's own steps over total_timesteps * UTD: the
    # critic stops ~1% short of lr_end, the (delayed) actor a third of the way.
    assert critic_end == pytest.approx(3e-4 - 0.99 * 2.7e-4, rel=1e-4)
    assert actor_end == pytest.approx(3e-4 - (last_actor // 3) / 1e6 * 2.7e-4, rel=1e-6)
    assert actor_end == pytest.approx(2.11e-4, rel=1e-3)
    assert xqc_learning_rates(0, algo) == (3e-4, 3e-4)
    assert xqc_learning_rates(3, algo)[1] == pytest.approx(linear_lr(3e-4, 3e-5, 1, 1_000_000))


def test_linear_lr_decays_then_holds():
    assert linear_lr(3e-4, 3e-5, 0, 100) == pytest.approx(3e-4)
    assert linear_lr(3e-4, 3e-5, 50, 100) == pytest.approx(1.65e-4)
    assert linear_lr(3e-4, 3e-5, 500, 100) == pytest.approx(3e-5)
    assert linear_lr(1e-4, None, 50, 100) == 1e-4


# ---- configs and scripts ------------------------------------------------------------

def test_configs_compose():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        x = compose(config_name="config_xqc", overrides=["env=pendulum", "embedding=none"])
        d = compose(config_name="config_drqv2", overrides=["algo=drqv2_xqc"])
    assert x.algo.name == "xqc" and x.algo.gamma == "auto" and x.algo.updates_per_step == 2
    # The authors' defaults (conf/config.yaml, conf/agent/{base_agent,xqc}.yaml).
    assert x.algo.policy_delay == 3 and list(x.algo.critic_hidden) == [512] * 4
    assert list(x.algo.actor_hidden) == [256] * 4 and x.algo.n_critics == 2
    assert x.algo.learning_starts == 5000 and x.algo.tau == 0.005 and x.algo.init_alpha == 0.01
    assert x.algo.batch_size == 256 and x.algo.buffer_size == 1_000_000 and x.algo.nstep == 1
    assert x.algo.critic_lr == x.algo.actor_lr == 3e-4 and x.algo.lr_end == 3e-5
    assert (x.algo.num_atoms, x.algo.v_min, x.algo.v_max) == (101, -5.0, 5.0)
    assert x.algo.target_entropy_scale == 0.5 and x.algo.reward_normalization
    assert d.algo.arch == "xqc" and d.algo.nstep == 3 and d.algo.lr == 1e-4 and d.algo.tau == 0.01
    assert d.algo.hidden_dim == 1024 and d.algo.num_blocks == 4 and d.algo.feature_dim == 50


@pytest.mark.parametrize("env_name, repeat, gamma", [
    ("pendulum", 1, 0.975),
    ("bipedal_walker_state", 1, 0.995),
    ("dm_control_state", 2, 0.99),        # dm_control's own 1000-step limit, in env steps
    ("dm_control_pixels", 2, 0.99),
])
def test_auto_gamma_on_the_real_wrapper_stack(env_name, repeat, gamma):
    from omegaconf import OmegaConf
    from src.sac_utils import make_env
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_xqc",
                      overrides=[f"env={env_name}", "embedding=none", f"algo.action_repeat={repeat}"])
    env = make_env(cfg, None, wrap_encoder=False)
    assert resolve_gamma("auto", env, repeat) == pytest.approx(gamma)
    env.close()


def _run(script, *overrides, tmp_path, env_name="pendulum"):
    out = subprocess.run(
        [sys.executable, script, f"env={env_name}", "embedding=none", "wandb.enabled=false",
         "save.enabled=true", "device=cpu", f"hydra.run.dir={tmp_path}", *overrides],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa")})
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    return out.stdout


def test_train_xqc_runs_end_to_end(tmp_path):
    text = _run("train_xqc.py", "algo.total_timesteps=300", "algo.learning_starts=200",
                "algo.log_frequency=100", "algo.eval_frequency=300", "algo.n_episodes_test=1",
                "algo.batch_size=32", tmp_path=tmp_path)
    assert "Discount: 0.9750" in text and "qf_loss=" in text and "edge_mass=" in text and "clip=" in text
    ckpt = torch.load(tmp_path / "final_checkpoint.pt", weights_only=False)
    rows = [m for k, m in ckpt["critic"].items() if k.endswith("weight") and m.dim() == 2]
    assert all(torch.allclose(w.norm(dim=1), torch.ones(w.shape[0]), atol=1e-4) for w in rows)


def test_train_drqv2_xqc_runs_end_to_end(tmp_path):
    text = _run("train_drqv2.py", "algo=drqv2_xqc", "algo.total_timesteps=300",
                "algo.learning_starts=200", "algo.log_frequency=100", "algo.eval_frequency=300",
                "algo.n_episodes_test=1", "algo.batch_size=32", "algo.hidden_dim=64",
                tmp_path=tmp_path)
    assert "qf_loss=" in text


def test_train_xqc_auto_gamma_on_dmc_at_action_repeat_2(tmp_path):
    text = _run("train_xqc.py", "algo.action_repeat=2", "algo.total_timesteps=5",
                "algo.learning_starts=5", "algo.eval_frequency=1000", tmp_path=tmp_path,
                env_name="dm_control_state")
    assert "Discount: 0.9900" in text
