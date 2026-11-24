"""Adroit's dense reward (door, hammer, relocate): gymnasium-robotics 1.2.1 to
1.4.2 flip the sign of the reach term (+0.1 * |palm - target| instead of
DAPG's -0.1; Gymnasium-Robotics PR #220). The env builders correct it."""

import gymnasium as gym
import numpy as np
import pytest

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


# DAPG's dense rewards written out, with the reach term as Gymnasium-Robotics
# PR #220 restores it (-0.1 * |palm - target|): independent of the wrapper's
# "reward - 0.2 * d" correction.
def _door_reference(u):
    d = u.data
    door = float(d.qpos[u.door_hinge_addrs])
    r = -0.1 * np.linalg.norm(d.site_xpos[u.grasp_site_id] - d.site_xpos[u.handle_site_id])
    r += -0.1 * (door - 1.57) ** 2
    r += -1e-5 * np.sum(d.qvel ** 2)
    return r + 2 * (door > 0.2) + 8 * (door > 1.0) + 10 * (door > 1.35)


def _hammer_reference(u):
    d = u.data
    hammer, palm = d.xpos[u.obj_body_id], d.site_xpos[u.S_grasp_site_id]
    head, nail, goal = d.site_xpos[u.tool_site_id], d.site_xpos[u.target_obj_site_id], d.site_xpos[u.goal_site_id]
    goal_distance = np.linalg.norm(nail - goal)
    r = -0.1 * np.linalg.norm(palm - hammer)
    r -= np.linalg.norm(head - nail)
    r -= 10 * goal_distance
    r -= 1e-2 * np.linalg.norm(d.qvel)
    r += 2 * (hammer[2] > 0.04 and head[2] > 0.04)
    return r + 25 * (goal_distance < 0.02) + 75 * (goal_distance < 0.01)


def _relocate_reference(u):
    d = u.data
    obj, palm, target = d.xpos[u.obj_body_id], d.site_xpos[u.S_grasp_site_id], d.site_xpos[u.target_obj_site_id]
    goal_distance = np.linalg.norm(obj - target)
    r = -0.1 * np.linalg.norm(palm - obj)
    if obj[2] > 0.04:
        r += 1.0 - 0.5 * np.linalg.norm(palm - target) - 0.5 * goal_distance
    return r + 10 * (goal_distance < 0.1) + 20 * (goal_distance < 0.05)


_REFERENCE = {"AdroitHandDoor": _door_reference, "AdroitHandHammer": _hammer_reference,
              "AdroitHandRelocate": _relocate_reference}


@pytest.mark.parametrize("task", sorted(_REFERENCE))
def test_the_dense_reward_matches_dapgs_formula(task):
    env = _make(f"{task}-v1")
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        _, r, *_ = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
        assert r == pytest.approx(float(_REFERENCE[task](env.unwrapped)), abs=1e-9)
