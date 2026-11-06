"""The encoder state probe (probe_encoder.py) and the env plumbing it and the
simple benchmark rely on: env.kwargs, env-id registration, pixel frames, and
embedding.pretrained=false reaching the encoder."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from probe_encoder import fit_mlp_probe, fit_probe, probe_env, probe_lengths, split_standardize, r2_scores, ridge_fit_predict, state_names, true_state
from src.sac_utils import _make_gym, env_kwargs

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.mark.parametrize("dims", [8, 400])                       # more frames than features, and fewer
def test_ridge_recovers_a_linear_target(dims):
    rng = np.random.default_rng(0)
    z, w = rng.normal(size=(200, dims)), rng.normal(size=(dims, 2))
    y = z @ w
    z_test = rng.normal(size=(50, dims))
    pred = ridge_fit_predict(z, y, z_test, 1e-6)
    if dims < 200:
        assert np.allclose(pred, z_test @ w, atol=1e-3)
    assert ridge_fit_predict(z, y, z, 1e-6).shape == y.shape


def test_r2_is_one_when_exact_zero_for_the_mean_and_nan_when_constant():
    y = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
    assert r2_scores(y, y)[0] == 1 and np.isnan(r2_scores(y, y)[1])
    assert r2_scores(y, np.ones_like(y))[0] == pytest.approx(0)


def test_probe_holds_out_whole_episodes_and_finds_the_signal():
    rng = np.random.default_rng(0)
    state = rng.normal(size=(1000, 2))
    feats = np.hstack([state @ rng.normal(size=(2, 6)), rng.normal(size=(1000, 4))])   # signal + noise
    res = fit_probe(feats, state, np.repeat(np.arange(20), 50), 0.2)
    assert res["n_test"] == 200 and res["n_train"] == 800
    assert np.nanmean(res["r2"]) > 0.95
    noise = fit_probe(rng.normal(size=(1000, 6)), state, np.repeat(np.arange(20), 50), 0.2)
    assert np.nanmean(noise["r2"]) < 0.2


def test_fourrooms_frames_kwargs_and_true_state():
    pytest.importorskip("gym_gridworlds")
    env = _make_gym("Gym-Gridworlds/FourRooms-13x13-v0", 64, random_goals=False)
    frame, _ = env.reset(seed=0)
    assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8 and env.action_space.n == 4
    u = env.unwrapped
    state = true_state(env, None)
    assert tuple(state[:2]) == tuple(u.agent_pos) and u.grid[int(state[2]), int(state[3])] == 10   # the goal tile
    goal = state[2:].copy()
    env.reset(seed=1)
    assert np.array_equal(true_state(env, None)[2:], goal)                                  # random_goals=False held
    assert state_names(env.spec.id, 4)[0] == "agent_row"


def test_bipedal_frames_are_squashed_not_cropped():
    pytest.importorskip("Box2D")
    frame, _ = _make_gym("BipedalWalker-v3", 64).reset(seed=0)
    assert frame.shape == (64, 64, 3)


def test_builtin_envs_dont_need_mujoco():
    code = ("import sys; from src.sac_utils import _register_env_ids; sys.modules.pop('gymnasium_robotics', None);"
            "_register_env_ids('Pendulum-v1'); assert 'gymnasium_robotics' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(Path(CONFIG_DIR).parent))


def _cfg(name, *overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize("env", ["pendulum_pixels", "fourrooms_pixels", "bipedal_walker_pixels",
                                 "bipedal_walker_state"])
def test_env_configs_compose(env):
    cfg = _cfg("config_probe", f"env={env}")
    assert cfg.env.builder in ("gym", "state")


def test_env_kwargs_come_from_the_config():
    assert env_kwargs(_cfg("config_sac", "env=fourrooms_pixels", "env.kwargs.random_goals=false")) == {"random_goals": False}
    assert env_kwargs(_cfg("config_sac", "env=pendulum_pixels")) == {}


@pytest.mark.parametrize("value, expected", [("true", True), ("false", False)])
def test_rl_scripts_pass_embedding_pretrained_on(monkeypatch, value, expected):
    import src.embeddings
    from src.encoder_config import embedding_kwargs, encoder_settings
    from src.rl.common import build_encoder_stack
    seen = {}

    class Stop(Exception):
        pass

    def spy(name, **kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(src.embeddings, "EmbeddingNet", spy)
    cfg = _cfg("config_sac", "embedding=resnet18", f"embedding.pretrained={value}")
    enc = encoder_settings(cfg, valid_grads=("none", "critic", "actor", "both"))
    with pytest.raises(Stop):
        build_encoder_stack(cfg, enc, embedding_kwargs(cfg), "cpu", False, "test")
    assert seen["pretrained"] is expected


def test_the_mlp_reads_what_the_ridge_cant():
    # A product of features, like a speed from per-frame cos/sin: no linear readout.
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(1500, 4))
    state = (feats[:, 0] * feats[:, 1])[:, None]
    episodes = np.repeat(np.arange(30), 50)
    assert np.nanmean(fit_probe(feats, state, episodes, 0.2)["r2"]) < 0.2
    assert np.nanmean(fit_mlp_probe(feats, state, episodes, 0.2, "cpu")["r2"]) > 0.8


def test_probe_env_uses_the_runs_action_repeat():
    from src.sac_utils import ActionRepeat
    assert isinstance(probe_env(_cfg("config_probe", "env=pendulum_pixels", "algo.action_repeat=3")), ActionRepeat)
    assert not isinstance(probe_env(_cfg("config_probe", "env=pendulum_pixels")), ActionRepeat)
    env = probe_env(_cfg("config_probe", "env=pendulum_pixels", "algo.action_repeat=3"))
    obs, _ = env.reset(seed=0)
    assert true_state(env, obs).shape == (3,)                                               # state passes through


@pytest.mark.parametrize("overrides, tag", [
    (["embedding=resnet18"], "resnet18"),
    (["embedding=resnet18", "embedding.pretrained=false"], "resnet18-scratch"),
    (["embedding=custom_checkpoint", "embedding.name=x", "embedding.checkpoint=/a.pt", "embedding.pretrained=false"], "x"),
])
def test_random_weights_runs_are_named_apart(overrides, tag):
    from src.rl.common import embedding_tag
    assert embedding_tag(_cfg("config_sac", *overrides), None) == tag


def test_features_that_never_varied_cant_blow_up_held_out_frames():
    # A channel silent on every non-test frame, firing on the held-out ones.
    rng = np.random.default_rng(0)
    state = rng.normal(size=(1000, 2))
    feats = np.hstack([state @ rng.normal(size=(2, 6)), np.zeros((1000, 1))])
    feats[800:, 6] = 50.0
    episodes = np.repeat(np.arange(20), 50)
    *_, z, _, _, _ = split_standardize(feats, state, episodes, 0.2)
    assert z.shape[1] == 6 and np.abs(z).max() <= 10
    assert fit_probe(feats, state, episodes, 0.2)["features_kept"] == 6
    assert np.nanmean(fit_mlp_probe(feats, state, episodes, 0.2, "cpu")["r2"]) > 0.9


@pytest.mark.parametrize("overrides, expected", [
    (["env=fourrooms_pixels"], (50, 5000)),
    (["env=dm_control_pixels"], (300, 6000)),                     # 20 episodes that leave the start pose
    (["env=dm_control_pixels", "probe.episode_steps=100", "probe.frames=2000"], (100, 2000)),
])
def test_probe_lengths(overrides, expected):
    assert probe_lengths(_cfg("config_probe", *overrides)) == expected
