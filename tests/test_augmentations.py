"""Contract tests for src/augmentations.py: every registered augmentation
keeps shape/dtype/range, draws per sample, is reproducible under
torch.manual_seed, and builds from its configs/augmentation/*.yaml file."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.augmentations import (
    available_augmentations,
    build_augmentation,
    make_augmentation,
    register_augmentation,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "augmentation"
REGISTERED = [n for n in available_augmentations() if n not in ("pipeline", "none")]


def _batch(n=16, c=3, size=32):
    torch.manual_seed(0)
    return torch.rand(n, c, size, size)


@pytest.fixture(scope="session")
def image_dir(tmp_path_factory):
    """A small folder of distinct images for random_overlay (stand-in for
    Places365)."""
    from torchvision.io import write_png
    root = tmp_path_factory.mktemp("overlay_images")
    (root / "sub").mkdir()
    g = torch.Generator().manual_seed(0)
    for i in range(8):
        img = torch.randint(0, 256, (3, 40 + i, 48), dtype=torch.uint8, generator=g)
        write_png(img, str(root / ("sub" if i % 2 else ".") / f"img{i}.png"))
    return root


def _required(name, image_dir):
    """Params an augmentation can't do without."""
    return {"image_dir": str(image_dir)} if name == "random_overlay" else {}


@pytest.mark.parametrize("name", REGISTERED)
def test_contract_shape_dtype_range(name, image_dir):
    x = _batch()
    y = make_augmentation(name, **_required(name, image_dir))(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.min() >= -1e-6 and y.max() <= 1 + 1e-6


@pytest.mark.parametrize("name", REGISTERED)
def test_draws_per_sample(name, image_dir):
    # A batch of identical images must come out different across samples
    # (p=1 forces every transform to fire, so differences come from the
    # per-sample parameters, not from per-sample on/off coin flips alone).
    params = {} if name in ("random_shift", "random_overlay", "random_resized_crop") else {"p": 1.0}
    aug = make_augmentation(name, **params, **_required(name, image_dir))
    img = torch.rand(1, 3, 32, 32)
    torch.manual_seed(0)
    y = aug(img.expand(64, -1, -1, -1).contiguous())
    if name in ("horizontal_flip", "random_grayscale"):
        # No continuous parameter: with p=1 every sample is transformed alike;
        # per-sample draws show as a mix at p=0.5.
        if name == "horizontal_flip":
            assert torch.allclose(y, img.flip(-1).expand_as(y))
        else:
            assert torch.allclose(y[:, 0], y[:, 1]) and torch.allclose(y[:, 1], y[:, 2])
        torch.manual_seed(0)
        y = make_augmentation(name, p=0.5)(img.expand(64, -1, -1, -1).contiguous())
    distinct = {tuple(row.flatten().tolist()[:50]) for row in y}
    assert len(distinct) > 1, f"{name} applied one draw to the whole batch"


@pytest.mark.parametrize("name", REGISTERED)
def test_reproducible_under_manual_seed(name, image_dir):
    aug = make_augmentation(name, **_required(name, image_dir))
    x = _batch()
    torch.manual_seed(123)
    a = aug(x)
    torch.manual_seed(123)
    b = aug(x)
    assert torch.equal(a, b)


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_config_file_builds(path, image_dir):
    cfg = OmegaConf.load(path)
    assert {"name", "stage", "params"} <= set(cfg.keys()) <= {"name", "stage", "params", "strong"}
    # The only setting a file may leave unset (???) is an overlay's image folder.
    for key in OmegaConf.missing_keys(cfg):
        assert key.endswith("image_dir"), key
        OmegaConf.update(cfg, key, str(image_dir))
    views = [cfg] + ([cfg.strong] if "strong" in cfg else [])
    for view in views:
        aug = build_augmentation(view)
        if view.name is None:
            assert aug is None
        else:
            x = _batch()
            assert aug(x).shape == x.shape


def test_random_overlay_pool_loads_on_first_call_at_about_1_5x(image_dir):
    aug = make_augmentation("random_overlay", image_dir=str(image_dir))
    assert aug.inner.pool is None
    aug(_batch(n=4, size=32))
    assert aug.inner.pool.shape == (8, 3, 48, 48)          # all 8 found (recursive); 1.5 x 32
    fixed = make_augmentation("random_overlay", image_dir=str(image_dir), resolution=40)
    fixed(_batch(n=4, size=32))
    assert fixed.inner.pool.shape[-1] == 40


def test_random_overlay_varies_even_from_a_single_image(image_dir):
    # Every draw gets its own random resized crop and flip (DMC-GB-style),
    # so one pool image still yields many different overlays.
    aug = make_augmentation("random_overlay", image_dir=str(image_dir), alpha=1.0, num_images=1)
    torch.manual_seed(0)
    y = aug(_batch(n=16, size=32))                         # alpha=1: output is the overlay
    assert len({tuple(row.flatten()[:64].tolist()) for row in y}) > 8


def test_random_overlay_same_overlay_on_every_stacked_frame(image_dir):
    aug = make_augmentation("random_overlay", image_dir=str(image_dir), alpha=1.0)
    y = aug(_batch(n=4, c=9))                              # alpha=1: output is the overlay
    assert torch.equal(y[:, 0:3], y[:, 3:6]) and torch.equal(y[:, 0:3], y[:, 6:9])


def test_random_overlay_pool_is_the_same_every_run(image_dir):
    a = make_augmentation("random_overlay", image_dir=str(image_dir), num_images=3)
    b = make_augmentation("random_overlay", image_dir=str(image_dir), num_images=3)
    a(_batch(n=2)), b(_batch(n=2))
    assert torch.equal(a.inner.pool, b.inner.pool)


def test_random_overlay_rejects_a_new_frame_size(image_dir):
    aug = make_augmentation("random_overlay", image_dir=str(image_dir))
    aug(_batch(n=2, size=32))
    with pytest.raises(ValueError, match="set up for"):
        aug(_batch(n=2, size=40))


@pytest.mark.parametrize("kwargs, exc", [
    ({}, ValueError),
    ({"image_dir": "???"}, ValueError),
    ({"image_dir": "/no/such/dir"}, FileNotFoundError),
])
def test_random_overlay_needs_an_image_folder(kwargs, exc):
    with pytest.raises(exc):
        make_augmentation("random_overlay", **kwargs)


def test_none_and_null_build_nothing():
    assert build_augmentation(None) is None
    assert build_augmentation({"name": None}) is None
    assert build_augmentation({"name": "none", "stage": "rl"}) is None


def test_make_none_is_a_checked_identity():
    aug = make_augmentation("none")
    x = _batch()
    assert torch.equal(aug(x), x)
    with pytest.raises(ValueError, match="float"):
        aug(torch.zeros(2, 3, 8, 8, dtype=torch.uint8))
    with pytest.raises(TypeError, match="no params"):
        make_augmentation("none", p=1.0)


def test_pipeline_accepts_bare_names():
    aug = make_augmentation("pipeline", transforms=["none", {"name": "random_shift", "params": {"pad": 0}}])
    x = _batch()
    assert torch.equal(aug(x), x)


def test_pipeline_threads_params():
    # pad=0 makes random_shift an identity, so an identity pipeline proves
    # the per-transform params reached the constructor.
    aug = make_augmentation("pipeline", transforms=[
        {"name": "random_shift", "params": {"pad": 0}},
        {"name": "horizontal_flip", "params": {"p": 0.0}},
    ])
    x = _batch()
    assert torch.equal(aug(x), x)


def test_random_shift_pad_frac_scales_with_resolution():
    aug = make_augmentation("random_shift", pad_frac=0.0)
    x = _batch(size=224)
    assert torch.equal(aug(x), x)
    # Default pad_frac=4/84: identical to DrQ-v2's pad=4 at 84px.
    torch.manual_seed(0)
    a = make_augmentation("random_shift")(_batch(size=84))
    torch.manual_seed(0)
    b = make_augmentation("random_shift", pad=4)(_batch(size=84))
    assert torch.equal(a, b)


@pytest.mark.parametrize("kwargs, ok", [
    ({"pad": 4}, True),
    ({"pad": 4.0}, True),
    ({"pad_frac": 0.1}, True),
    ({"pad": 4, "pad_frac": 0.1}, False),
    ({"pad": 4.5}, False),
    ({"pad": -1}, False),
    ({"pad": True}, False),
    ({"pad_frac": -0.1}, False),
])
def test_random_shift_pad_arguments(kwargs, ok):
    if ok:
        make_augmentation("random_shift", **kwargs)
    else:
        with pytest.raises(ValueError):
            make_augmentation("random_shift", **kwargs)


def test_random_shift_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        make_augmentation("random_shift")(torch.rand(2, 3, 32, 40))


def _exact_shift(x, pad, seed):
    """Integer-shift reference: the same randint draw, then a slice of the
    replicate-padded image."""
    torch.manual_seed(seed)
    n, _, h, w = x.shape
    s = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), dtype=torch.float32).long()
    xp = torch.nn.functional.pad(x.float(), (pad,) * 4, "replicate")
    return torch.stack([xp[i, :, s[i, 0, 0, 1]:s[i, 0, 0, 1] + h, s[i, 0, 0, 0]:s[i, 0, 0, 0] + w]
                        for i in range(n)])


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-4), (torch.bfloat16, 1e-2)])
@pytest.mark.parametrize("size", [84, 224])
def test_random_shift_is_an_exact_integer_shift(dtype, tol, size):
    # In bf16 a grid built in the input dtype samples the wrong pixels at 224px.
    pad = round(4 / 84 * size)
    x = torch.rand(4, 3, size, size)
    ref = _exact_shift(x, pad, seed=0)
    torch.manual_seed(0)
    out = make_augmentation("random_shift")(x.to(dtype))
    assert out.dtype == dtype
    assert (out.float() - ref).abs().max() < tol


@pytest.mark.parametrize("bad, hint", [
    (torch.rand(2, 3, 16, 16) * 255, "0-255"),
    ((torch.rand(2, 3, 16, 16) - 0.45) / 0.225, "normalisation"),
])
def test_range_check_on_first_call(bad, hint):
    with pytest.raises(ValueError, match=hint):
        make_augmentation("horizontal_flip")(bad)


def test_range_check_runs_once():
    aug = make_augmentation("horizontal_flip")
    aug(_batch())
    aug(torch.rand(2, 3, 16, 16) * 255)   # not re-checked: avoids a GPU sync per call


def test_random_shift_yaml_overrides():
    from hydra import compose, initialize_config_dir
    root = str(CONFIG_DIR.parent)
    with initialize_config_dir(config_dir=root, version_base=None):
        for override in ("augmentation.params.pad=4", "augmentation.params.pad_frac=0.1"):
            cfg = compose("config_sac", overrides=["augmentation=random_shift", override])
            build_augmentation(cfg.augmentation)
        cfg = compose("config_sac", overrides=["augmentation=random_shift",
                                               "augmentation.params.pad=4",
                                               "augmentation.params.pad_frac=0.1"])
        with pytest.raises(ValueError, match="not both"):
            build_augmentation(cfg.augmentation)


def test_rejects_uint8_and_hwc_input():
    aug = make_augmentation("random_shift")
    with pytest.raises(ValueError, match="float"):
        aug(torch.zeros(4, 3, 32, 32, dtype=torch.uint8))
    with pytest.raises(ValueError, match="float"):
        aug(torch.zeros(32, 32, 3))


def test_colour_transform_rejects_frame_stacked_input():
    with pytest.raises(ValueError, match="frame_stack"):
        make_augmentation("color_jitter")(torch.rand(2, 9, 32, 32))
    # Geometric transforms are channel-agnostic.
    assert make_augmentation("random_shift")(torch.rand(2, 9, 32, 32)).shape == (2, 9, 32, 32)


def test_unknown_name_and_bad_params():
    with pytest.raises(ValueError, match="Unknown augmentation"):
        make_augmentation("does_not_exist")
    with pytest.raises(TypeError, match="random_shift"):
        make_augmentation("random_shift", not_a_param=1)
    with pytest.raises(ValueError, match="pipeline entries"):
        make_augmentation("pipeline", transforms=[{"name": "random_shift", "padd": 1}])


def test_custom_registration():
    @register_augmentation("_test_invert")
    class Invert(nn.Module):
        def forward(self, x):
            return 1.0 - x

    x = _batch()
    assert torch.allclose(build_augmentation({"name": "_test_invert"})(x), 1.0 - x)
    with pytest.raises(ValueError, match="already taken"):
        register_augmentation("_test_invert")(Invert)
