"""Fixed evaluation seeds only pair episodes across runs if reset(seed=s)
really fixes the start state, and a run's training and evaluation envs --
two envs alive in one process, rendering in alternation -- must show the
same frame for the same state. Meta-World ignored the seed silently
(tests/test_metaworld.py); Gymnasium's MuJoCo renderer let one env render
through the other's GL context (fixed by _OwnGLContext, src/sac_utils.py).

The pixel cases compare frames exactly, so they run only under
MUJOCO_GL=egl (a GPU node): a WSL laptop's software renderer (OSMesa) isn't
bit-reproducible -- anti-aliased edges of the same state differ by a few grey
levels between renders, even within a process. The GL-context fix itself is
checked on the context calls instead (test_each_mujoco_env_renders_through_
its_own_gl_context), which holds on any renderer; the state cases check the
seeding everywhere."""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from hydra import compose, initialize_config_dir

from src.sac_utils import make_env

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))

_EGL = pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "egl",
                          reason="exact pixel comparisons need EGL (run on a GPU node)")


def _env(name, *overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_xqc", overrides=[f"env={name}", "embedding=none", *overrides])
    return make_env(cfg, None, wrap_encoder=False)


@pytest.mark.parametrize("name, overrides", [
    pytest.param("dm_control_pixels", ["env.image_size=64"], marks=_EGL),
    pytest.param("dmc_walker", ["env.image_size=64"], marks=_EGL),
    pytest.param("adroit_door_pixels", ["env.image_size=64"], marks=_EGL),
    pytest.param("metaworld_pixels", ["env.image_size=64", "seed=1"], marks=_EGL),
    ("dm_control_state", []),
    ("adroit_door_state", []),
])
def test_reset_seed_fixes_the_start_state(name, overrides):
    if name.startswith("metaworld"):
        pytest.importorskip("metaworld")
    a, b = _env(name, *overrides), _env(name, *overrides)
    first = a.reset(seed=3)[0]
    assert np.array_equal(b.reset(seed=3)[0], first)     # the other env, same state
    b.reset(seed=4)                                      # renders in alternation, as in training
    a.reset()                                            # an unseeded (training) reset
    assert np.array_equal(a.reset(seed=3)[0], first)
    assert np.array_equal(b.reset(seed=3)[0], first)
    assert not np.array_equal(b.reset(seed=4)[0], first)


def _track_contexts(envs):
    """Record which env's viewer made its GL context current last, and fail
    a render that happens while another env's context is current."""
    current = {}
    for env in envs:
        viewer = env.unwrapped.mujoco_renderer._viewers["rgb_array"]
        make_current, render = viewer.make_context_current, viewer.render

        def tracked_make_current(viewer=viewer, make_current=make_current):
            current["viewer"] = viewer
            return make_current()

        def tracked_render(*args, viewer=viewer, render=render, **kwargs):
            assert current.get("viewer") is viewer, "rendered through another env's GL context"
            return render(*args, **kwargs)

        viewer.make_context_current, viewer.render = tracked_make_current, tracked_render


@pytest.mark.parametrize("name, overrides", [
    ("adroit_door_pixels", ["env.image_size=64"]),
    ("metaworld_pixels", ["env.image_size=64", "seed=1"]),
])
def test_each_mujoco_env_renders_through_its_own_gl_context(name, overrides):
    """Gymnasium's MujocoRenderer leaves the last-created context current;
    with a training and an evaluation env alive, one would render through
    the other's. Checked on the context calls, not on pixels, so it holds
    whatever the renderer's own noise."""
    if name.startswith("metaworld"):
        pytest.importorskip("metaworld")
    a, b = _env(name, *overrides), _env(name, *overrides)
    a.reset(seed=3), b.reset(seed=3)                 # both viewers exist now
    _track_contexts([a, b])
    for _ in range(2):                               # alternate, as training and evaluation do
        a.reset(seed=3)
        b.reset(seed=4)
        a.step(a.action_space.sample())
