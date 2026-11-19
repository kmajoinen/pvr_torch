"""Adroit's dense reward (door, hammer, relocate): gymnasium-robotics 1.2.1 to
1.4.2 flip the sign of the reach term (+0.1 * |palm - target| instead of
DAPG's -0.1; Gymnasium-Robotics PR #220). The env builders correct it."""

import os

import gymnasium as gym
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from src.sac_utils import _ADROIT_REACH_TERM, _adroit_reward_is_buggy, _make


def _reach_distance(env, kind, target_attr, palm_attr):
    u = env.unwrapped
    target = u.data.site_xpos[getattr(u, target_attr)] if kind == "site" else u.data.xpos[getattr(u, target_attr)]
    return float(np.linalg.norm(u.data.site_xpos[getattr(u, palm_attr)] - target))


@pytest.mark.skipif(not _adroit_reward_is_buggy(), reason="this gymnasium-robotics has the original sign")
@pytest.mark.parametrize("task", sorted(_ADROIT_REACH_TERM))
def test_the_reach_term_gets_dapgs_sign_back(task):
    raw, ours = gym.make(f"{task}-v1"), _make(f"{task}-v1")
    raw.reset(seed=0), ours.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-1, 1, raw.action_space.shape).astype(np.float32)
        _, r_raw, *_ = raw.step(a)
        _, r_ours, *_ = ours.step(a)
        d = _reach_distance(raw, *_ADROIT_REACH_TERM[task])
        assert r_ours == pytest.approx(r_raw - 0.2 * d, abs=1e-9)     # +0.1 d  ->  -0.1 d
        assert d > 0


def test_the_sparse_reward_is_left_alone():
    env = _make("AdroitHandDoor-v1", reward_type="sparse")
    assert type(env).__name__ != "_AdroitRewardFix"
    env.reset(seed=0)
    _, r, *_ = env.step(env.action_space.sample())
    assert r in (-0.1, 10.0)


def test_pen_has_no_reach_term_to_fix():
    assert type(_make("AdroitHandPen-v1")).__name__ != "_AdroitRewardFix"
