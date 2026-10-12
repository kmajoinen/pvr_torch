"""
Image augmentations: a registry of nn.Modules, built by name from config.

Contract -- every registered augmentation, built-in or added later:
  input    float tensor (N, C, H, W), values in [0, 1]
  output   same shape, dtype, device and value range
  random   parameters drawn independently per sample, never one draw shared
           by the whole batch
  RNG      torch's global RNG, so runs are reproducible via torch.manual_seed
           (the training scripts seed it from cfg.seed). Kornia has no
           torch.Generator support, so per-module generators aren't offered.

Nothing in this file knows about encoders, replay buffers or training loops.
Where an augmentation is applied -- which batches, which losses -- is the
caller's decision (train_sac.py / train_bc.py, and `stage` in
configs/augmentation/*.yaml). That separation is what lets other pipelines
(e.g. encoder pretraining) reuse these transforms unchanged.

In the training scripts the input is EmbeddingNet.preprocess() output: the
frame after the encoder's own resize/crop, so "H, W" is the encoder's input
resolution (224 for most PVRs, 84 for rl3d_resnet18), not the env's render
size.

Two kinds of entries:
  - kornia-backed: standard transforms (colour jitter, blur, flip, erasing).
    Kornia draws parameters per sample on whatever device the batch is on.
  - bespoke: implemented in this file, for transforms kornia lacks or that
    are worth owning. DrQ-v2's random shift is the first.

Adding an augmentation:
    @register_augmentation("my_aug")
    class MyAug(nn.Module):
        def __init__(self, strength: float = 0.1): ...
        def forward(self, x): ...   # (N, C, H, W) float [0, 1] -> same

    # configs/augmentation/my_aug.yaml
    name: my_aug
    stage: rl
    params: {strength: 0.1}

Usage:
    build_augmentation(cfg.augmentation)              # from a config node
    make_augmentation("random_shift", pad_frac=0.05)  # programmatic
    make_augmentation("pipeline", transforms=[
        {"name": "random_shift"},
        {"name": "color_jitter", "params": {"brightness": 0.2}},
    ])
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_augmentation(name: str):
    """Decorator: register a class or factory function under `name`. Its
    keyword arguments are the `params:` accepted from config."""
    def decorator(factory):
        if name in _REGISTRY or name in ("pipeline", "none"):
            raise ValueError(f"augmentation name {name!r} is already taken")
        _REGISTRY[name] = factory
        return factory
    return decorator


def available_augmentations():
    return sorted(_REGISTRY) + ["pipeline"]


# ------------------------------------------------------------------------------
# Bespoke transforms
# ------------------------------------------------------------------------------

@register_augmentation("random_shift")
class RandomShiftsAug(nn.Module):
    """Random shift from DrQ-v2 (Yarats et al., 2021): replicate-pad by `pad`
    pixels, then crop back to the original size at a random offset,
    independently per sample. Channel-agnostic. Input must be square.

    The pad is set as a fraction of the image side (pad_frac) rather than in
    pixels, because this runs at the encoder's input resolution, which
    differs between encoders: a fixed pad=4 would be a ~1.8% shift at 224px
    but ~4.8% at 84px. The default 4/84 reproduces DrQ-v2's pad=4 at 84px.
    Pass `pad` instead to fix the pixel count regardless of resolution.
    """

    def __init__(self, pad_frac: float = 4 / 84, pad: Optional[int] = None):
        super().__init__()
        if pad is not None and pad < 0:
            raise ValueError(f"pad must be >= 0, got {pad}")
        if pad_frac < 0:
            raise ValueError(f"pad_frac must be >= 0, got {pad_frac}")
        self.pad_frac = pad_frac
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w, f"random_shift needs square input, got {h}x{w}"
        pad = self.pad if self.pad is not None else round(self.pad_frac * h)
        if pad == 0:
            return x
        x = F.pad(x, (pad,) * 4, 'replicate')
        eps = 1.0 / (h + 2 * pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


# ------------------------------------------------------------------------------
# Kornia-backed transforms
# ------------------------------------------------------------------------------
# Defaults match the previous torchvision-based versions of these transforms,
# except that parameters are now drawn per sample. Note kornia's own default
# p differs for blur (0.5); it's set to 1.0 here so the transform always fires,
# as before.

def _kornia():
    try:
        import kornia.augmentation as K
    except ImportError as e:
        raise ImportError(
            "This augmentation needs kornia: pip install kornia "
            "(it's listed in requirements.txt)."
        ) from e
    return K


class _RequireRGB(nn.Module):
    """Colour transforms are defined on 3-channel RGB. With a frame-stacked
    input (embedding=random, frame_stack > 1) the channel count is 3*k, which
    kornia would reject with a less helpful error."""

    def __init__(self, inner: nn.Module, name: str):
        super().__init__()
        self.inner = inner
        self.name = name

    def forward(self, x):
        if x.shape[1] != 3:
            raise ValueError(
                f"{self.name} needs 3-channel RGB input, got {x.shape[1]} "
                "channels (frame_stack > 1?). Geometric transforms "
                "(random_shift, horizontal_flip, random_erasing) work on any "
                "channel count."
            )
        return self.inner(x)


@register_augmentation("color_jitter")
def _color_jitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=1.0):
    K = _kornia()
    return _RequireRGB(
        K.ColorJitter(brightness=brightness, contrast=contrast,
                      saturation=saturation, hue=hue, p=p),
        "color_jitter",
    )


@register_augmentation("gaussian_blur")
def _gaussian_blur(kernel_size=5, sigma=(0.1, 2.0), p=1.0):
    K = _kornia()
    return K.RandomGaussianBlur(kernel_size=kernel_size, sigma=tuple(sigma), p=p)


@register_augmentation("horizontal_flip")
def _horizontal_flip(p=0.5):
    K = _kornia()
    return K.RandomHorizontalFlip(p=p)


@register_augmentation("random_erasing")
def _random_erasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0, p=0.5):
    K = _kornia()
    return K.RandomErasing(scale=tuple(scale), ratio=tuple(ratio), value=value, p=p)


# ------------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------------

class _CheckInput(nn.Module):
    """Fails fast on the most likely misuse: passing a raw uint8 HWC replay
    batch instead of a float NCHW one. Range isn't checked -- that would
    force a GPU sync on every call."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        if x.ndim != 4 or not x.is_floating_point():
            raise ValueError(
                "augmentations take float (N, C, H, W) input in [0, 1], got "
                f"dtype={x.dtype} shape={tuple(x.shape)}"
            )
        return self.inner(x)


def _make(name: str, params: dict) -> nn.Module:
    if name == "pipeline":
        transforms = params.pop("transforms", None)
        if params:
            raise TypeError(f"pipeline takes only `transforms`, got {sorted(params)}")
        if not transforms:
            raise ValueError("pipeline needs a non-empty `transforms` list")
        return nn.Sequential(*[_make_from_spec(t) for t in transforms])
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown augmentation {name!r}. Available: {available_augmentations()}"
        )
    try:
        return _REGISTRY[name](**params)
    except TypeError as e:
        raise TypeError(f"bad params for augmentation {name!r}: {e}") from e


def _make_from_spec(spec) -> nn.Module:
    spec = dict(spec)
    name = spec.pop("name", None)
    params = dict(spec.pop("params", None) or {})
    if name is None or spec:
        raise ValueError(
            "pipeline entries look like {name: ..., params: {...}}, got "
            f"name={name!r} plus extra keys {sorted(spec)}"
        )
    return _make(name, params)


def make_augmentation(name: str, **params) -> nn.Module:
    """Build a registered augmentation (or a `pipeline` of them) by name."""
    return _CheckInput(_make(name, dict(params)))


def _to_plain(node):
    """Config node -> plain Python containers, with lists turned into tuples
    (kornia wants tuples for ranges like sigma=(0.1, 2.0))."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(node):
            node = OmegaConf.to_container(node, resolve=True)
    except ImportError:
        pass
    if isinstance(node, dict):
        return {k: _to_plain(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return tuple(_to_plain(v) for v in node)
    return node


def build_augmentation(aug_cfg) -> Optional[nn.Module]:
    """Augmentation config -> nn.Module, or None for no augmentation.

    aug_cfg is a mapping (OmegaConf node or plain dict) with `name` and
    optional `params`. Any other keys, e.g. `stage`, are the caller's
    concern and are ignored here, so the same config works for any pipeline.
    """
    if aug_cfg is None:
        return None
    spec = _to_plain(aug_cfg)
    name = spec.get("name")
    if name in (None, "none"):
        return None
    return make_augmentation(name, **dict(spec.get("params") or {}))
