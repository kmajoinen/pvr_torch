"""DrQ-v2 agent pieces (src/rl/drqv2.py) and its config."""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.rl.drqv2 import Actor, Critic, TruncatedNormal, stddev_schedule

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))


@pytest.mark.parametrize("schedule", [{"init": 1.0, "final": 0.1, "duration": 100},
                                      "linear(1.0,0.1,100)"])
def test_linear_stddev_schedule(schedule):
    assert stddev_schedule(schedule, 0) == pytest.approx(1.0)
    assert stddev_schedule(schedule, 50) == pytest.approx(0.55)
    assert stddev_schedule(schedule, 100) == pytest.approx(0.1)
    assert stddev_schedule(schedule, 10_000) == pytest.approx(0.1)


def test_constant_stddev_schedule():
    assert stddev_schedule(0.2, 123) == 0.2


def test_truncated_normal_clamps_with_straight_through_grad():
    torch.manual_seed(0)
    loc = torch.full((1000, 2), 0.95, requires_grad=True)
    x = TruncatedNormal(loc, torch.ones_like(loc)).sample()
    assert x.min() > -1 and x.max() < 1
    x.sum().backward()
    assert torch.allclose(loc.grad, torch.ones_like(loc))   # clamp passes gradient


def test_truncated_normal_noise_clip():
    loc = torch.zeros(1000, 2)
    x = TruncatedNormal(loc, torch.full_like(loc, 5.0)).sample(clip=0.3)
    assert x.abs().max() <= 0.3 + 1e-6


def test_actor_critic_shapes():
    actor, critic = Actor(64, 3, 50, 32), Critic(64, 3, 50, 32)
    feats = torch.randn(5, 64)
    dist = actor(feats, 0.5)
    assert dist.mean.shape == (5, 3) and dist.mean.abs().max() <= 1
    q1, q2 = critic(feats, dist.sample(clip=0.3))
    assert q1.shape == q2.shape == (5, 1)


def test_config_drqv2_composes_on_config_sac():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_drqv2", overrides=["embedding=resnet18"])
    assert cfg.algo.name == "drqv2" and cfg.algo.nstep == 3
    assert cfg.encoder.grads == "none" and cfg.embedding.name == "resnet18"
    assert cfg.algo.log_frequency % cfg.algo.update_every_steps == 0
    assert cfg.algo.eval_frequency % cfg.algo.update_every_steps == 0


def test_pieg_preset_matches_drqv2_optimizer_settings():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_drqv2",
                      overrides=["+experiment=pieg", "env=dmc_walker"])
    assert cfg.encoder.lr == cfg.algo.lr == 1e-4
    assert cfg.encoder.grad_clip is None and cfg.algo.grad_clip is None
    assert cfg.env.builder == "dmc" and cfg.env.name == "dmc_walker"


@pytest.mark.parametrize("config_name", ["config_drqv2", "config_sac"])
@pytest.mark.parametrize("overrides", [["+experiment=pieg", "env=dmc_walker"],
                                       ["env=dmc_walker", "+experiment=pieg"]])
def test_pieg_preset_pins_84px_frames(config_name, overrides):
    # PIE-G's preprocess doesn't resize: at the env's default 224px the
    # linear head would be ~411M parameters instead of ~63M.
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    assert cfg.env.image_size == 84
