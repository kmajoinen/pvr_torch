"""Meta-World through make_env (configs/env/metaworld_*.yaml, src/sac_utils.py):
frames the right way up, reproducible layouts for fixed evaluation seeds,
episodes recorded once, success tracked."""

import os

import numpy as np
import pytest

pytest.importorskip("metaworld")
os.environ.setdefault("MUJOCO_GL", "osmesa")

from hydra import compose, initialize_config_dir

from src.sac_utils import _success_fn_for, make_env

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))


def _env(name, *overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_xqc", overrides=[f"env={name}", "embedding=none", *overrides])
    return make_env(cfg, None, wrap_encoder=False)


def test_frames_are_rotated_back_upright_for_an_upside_down_camera():
    import mujoco
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_xqc", overrides=["env=metaworld_pixels", "embedding=none"])
    env = _env("metaworld_pixels", "env.image_size=64", "seed=1")
    obs, _ = env.reset(seed=3)
    m, d = env.unwrapped.model, env.unwrapped.data
    mujoco.mj_forward(m, d)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cfg.env.kwargs.camera_name)
    up_z = d.cam_xmat[cam].reshape(3, 3)[2, 1]         # the camera's up axis, world z
    # The configured camera is mounted upside down exactly when rotate_180 is on.
    assert (up_z < 0) == bool(cfg.env.rotate_180)
    raw = env.unwrapped.render().astype(int)           # Meta-World's own frame, rotated 180 degrees
    # A second render, so compared by distance, not equality (software
    # rendering isn't bit-reproducible): the frame is the 180-degree rotation,
    # not the vertical flip (mirrored) or the raw frame.
    dist = {k: np.abs(obs.astype(int) - v).mean()
            for k, v in {"rot180": raw[::-1, ::-1], "vflip": raw[::-1], "raw": raw}.items()}
    assert obs.shape == (64, 64, 3) and dist["rot180"] < 0.2 * min(dist["vflip"], dist["raw"]), dist


@pytest.mark.parametrize("name, overrides", [("metaworld_state", []), ("metaworld_pixels", ["env.image_size=64"])])
def test_the_same_seed_gives_the_same_layout_in_any_env_of_the_run(name, overrides):
    a, b = _env(name, "seed=1", *overrides), _env(name, "seed=1", *overrides)
    first = {s: a.reset(seed=s)[0] for s in (3, 7)}
    for _ in range(3):
        a.reset()                                      # unseeded (training) resets in between
    assert all(np.array_equal(a.reset(seed=s)[0], first[s]) for s in first)
    assert all(np.array_equal(b.reset(seed=s)[0], first[s]) for s in first)   # train and eval envs agree
    assert not np.array_equal(first[3], first[7])
    other_run = _env(name, "seed=2", *overrides).reset(seed=3)[0]
    assert not np.array_equal(other_run, first[3])     # another training seed, other layouts


def test_episodes_are_recorded_once_with_success():
    env = _env("metaworld_state", "seed=1")
    env.reset(seed=0)
    for t in range(500):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert truncated and t == 499 and info["episode"]["l"] == 500
    assert "success" in info
    success = _success_fn_for("Meta-World/MT1")
    assert success([{"success": 0.0}, {"success": 1.0}, {"success": 0.0}]) == 1.0
    assert success([{"success": 0.0}] * 3) == 0.0


def test_other_tasks_by_name():
    env = _env("metaworld_state", "seed=1", "env.kwargs.env_name=button-press-topdown-v3")
    assert env.reset(seed=0)[0].shape == (39,)
