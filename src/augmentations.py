"""
Image augmentations: a registry of nn.Modules, built by name from config.

Contract -- every registered augmentation, built-in or added later:
  input    float tensor (N, C, H, W), values in [0, 1]
  output   same shape, dtype, device and value range
  random   parameters drawn independently per sample, never one draw shared
           by the whole batch. One known exception: kornia's ColorJitter
           draws its brightness/contrast/saturation/hue factors per sample
           but the order it applies them in once per batch.
  RNG      torch's global RNGs, so runs are reproducible via torch.manual_seed
           (the training scripts seed it from cfg.seed; it seeds CPU and CUDA
           alike). Kornia has no torch.Generator support, so per-module
           generators aren't offered.

The input contract is enforced by make_augmentation(): dtype and shape on
every call, value range on the first call only (see _CheckInput). The range
check exists because some encoders' own preprocessing produces 0-255 input
(R3M, VIP) or ImageNet-normalised input (CLIP, timm); EmbeddingNet.preprocess()
must stop before either step, or colour transforms clamp the frame to [0, 1]
and erasing fills with the wrong value, without any error.

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
    Kornia draws parameters per sample, on the CPU with the global CPU RNG,
    and copies them to the batch's device. (Calling .to(device) directly on
    a kornia module would move its sampling to that device; .to() on a
    parent module, including the wrappers below, does not.)
  - bespoke: implemented in this file, for transforms kornia lacks or that
    are worth owning. DrQ-v2's random shift is the first. It draws on the
    batch's device.

Adding an augmentation:
    @register_augmentation("my_aug")
    class MyAug(nn.Module):
        def __init__(self, strength: float = 0.1): ...
        def forward(self, x): ...   # (N, C, H, W) float [0, 1] -> same

    # configs/augmentation/my_aug.yaml
    name: my_aug
    stage: rl
    params: {strength: 0.1}

  Registration happens when the decorator runs, i.e. when its module is
  imported. Put new entries in this file, or make sure the module that
  defines them is imported before build_augmentation() is called; otherwise
  the name is reported as unknown.

Usage:
    build_augmentation(cfg.augmentation)              # from a config node
    make_augmentation("random_shift", pad_frac=0.05)  # programmatic
    make_augmentation("pipeline", transforms=[
        "random_shift",                               # bare name: default params
        {"name": "color_jitter", "params": {"brightness": 0.2}},
    ])
"""

import numbers
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}
_RESERVED = ("pipeline", "none")


def register_augmentation(name: str):
    """Decorator: register a class or factory function under `name`. Its
    keyword arguments are the `params:` accepted from config."""
    def decorator(factory):
        if name in _REGISTRY or name in _RESERVED:
            raise ValueError(f"augmentation name {name!r} is already taken")
        _REGISTRY[name] = factory
        return factory
    return decorator


def available_augmentations():
    return sorted(_REGISTRY) + list(_RESERVED)


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
    but ~4.8% at 84px. The default pad_frac=4/84 reproduces DrQ-v2's pad=4 at
    84px. Pass `pad` instead to fix the pixel count regardless of resolution;
    passing both is an error.

    The sampling grid is built in at least fp32 whatever the input dtype. In
    bf16 the representable spacing near +-1 (2^-7) is about one pixel at
    224px, which would turn the integer shift into a sub-pixel resample.
    """

    def __init__(self, pad_frac: Optional[float] = None, pad: Optional[int] = None):
        super().__init__()
        if pad is not None and pad_frac is not None:
            raise ValueError(
                f"random_shift takes pad_frac or pad, not both (got pad_frac="
                f"{pad_frac}, pad={pad}). From config, set the other to null."
            )
        if pad is not None:
            if (isinstance(pad, bool) or not isinstance(pad, numbers.Real)
                    or not float(pad).is_integer() or pad < 0):
                raise ValueError(
                    f"pad must be a non-negative whole number of pixels, got {pad!r}"
                )
            pad = int(pad)
        elif pad_frac is None:
            pad_frac = 4 / 84
        if pad_frac is not None and pad_frac < 0:
            raise ValueError(f"pad_frac must be >= 0, got {pad_frac}")
        self.pad_frac = pad_frac
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.shape
        if h != w:
            raise ValueError(f"random_shift needs square input, got {h}x{w}")
        pad = self.pad if self.pad is not None else round(self.pad_frac * h)
        if pad == 0:
            return x
        in_dtype = x.dtype
        dtype = torch.promote_types(in_dtype, torch.float32)
        size = h + 2 * pad
        x = F.pad(x.to(dtype), (pad,) * 4, 'replicate')
        eps = 1.0 / size
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                size,
                                device=x.device,
                                dtype=dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)

        shift = torch.randint(0,
                              2 * pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=dtype)
        shift *= 2.0 / size

        grid = base_grid.unsqueeze(0) + shift  # broadcasts to (n, h, h, 2)
        out = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        return out.to(in_dtype)


@register_augmentation("random_overlay")
class RandomOverlay(nn.Module):
    """Random overlay from SODA / SVEA (Hansen et al., 2021), PIE-G's strong
    augmentation: x' = (1 - alpha) * x + alpha * I, with I a natural image
    drawn independently per sample.

    An approximation of DMC-GB's loader, which SVEA and PIE-G use: that
    streams the full Places365 training split and gives every image a fresh
    RandomResizedCrop + RandomHorizontalFlip. Here, images come from
    `image_dir` (searched recursively for .jpg/.jpeg/.png), and a fixed pool
    of `num_images` of them (chosen with a fixed seed, so every run uses the
    same pool) is kept on the batch's device; every draw then gets its own
    random resized crop (scale `crop_scale`, DMC-GB's default) and flip, so
    overlays stay varied even from a finite pool.

    The pool is loaded on the first call, once the frame size is known: at
    `resolution` (default ~1.5x the frame side, so crops are taken from more
    pixels than they end up with) -- e.g. ~126px for 84px frames. GPU memory
    is num_images * 3 * resolution^2 bytes: ~240 MB for 5000 images at
    126px, ~1.7 GB at 336px (224px frames), so lower num_images there.
    With a frame-stacked input (3*k channels) the same overlay (same image,
    crop and flip) goes on every frame of a sample, as in SODA.
    """

    _EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(self, image_dir: Optional[str] = None, alpha: float = 0.5,
                 num_images: int = 5000, resolution: Optional[int] = None,
                 crop_scale=(0.08, 1.0)):
        super().__init__()
        import os
        if not image_dir or image_dir == "???":
            raise ValueError("random_overlay needs params.image_dir: a folder of natural images "
                             "(e.g. Places365's validation set)")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"random_overlay image_dir {image_dir!r} is not a directory")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"random_overlay alpha must be in [0, 1], got {alpha}")
        files = sorted(
            os.path.join(root, f)
            for root, _, names in os.walk(image_dir)
            for f in names if f.lower().endswith(self._EXTENSIONS)
        )
        if not files:
            raise FileNotFoundError(f"no .jpg/.jpeg/.png images under {image_dir!r}")
        order = torch.randperm(len(files), generator=torch.Generator().manual_seed(0))
        self._paths = [files[i] for i in order[:num_images].tolist()]
        self.alpha = alpha
        self.resolution = resolution
        self.crop_scale = tuple(crop_scale)
        self._frame_size = None
        self.register_buffer("pool", None, persistent=False)   # loaded on first call

    @staticmethod
    def _load(paths, resolution):
        import torchvision.transforms.functional as TF
        from torchvision.io import ImageReadMode, read_image
        images = []
        for p in paths:
            img = read_image(p, mode=ImageReadMode.RGB)                     # uint8 (3, H, W)
            img = TF.resize(img, resolution, antialias=True)
            images.append(TF.center_crop(img, resolution))
        return torch.stack(images)                                          # uint8 (M, 3, r, r)

    def _setup(self, h, w, device):
        side = self.resolution or int(round(1.5 * max(h, w)))
        print(f"random_overlay: loading {len(self._paths)} images at {side}px "
              f"({len(self._paths) * 3 * side * side / 1e6:.0f} MB)")
        self.pool = self._load(self._paths, side).to(device)
        K = _kornia()
        self._crop = K.RandomResizedCrop((h, w), scale=self.crop_scale)
        self._flip = K.RandomHorizontalFlip(p=0.5)
        self._frame_size = (h, w)

    def forward(self, x):
        n, c, h, w = x.shape
        if c % 3:
            raise ValueError(f"random_overlay needs 3*k channels (RGB frames), got {c}")
        if self.pool is None:
            self._setup(h, w, x.device)
        elif (h, w) != self._frame_size:
            raise ValueError(f"random_overlay was set up for {self._frame_size} frames, got {(h, w)}")
        idx = torch.randint(0, self.pool.shape[0], (n,), device=self.pool.device)
        images = self.pool[idx].to(device=x.device, dtype=x.dtype) / 255.0
        images = self._flip(self._crop(images))       # per sample: fresh crop and flip
        images = images.repeat(1, c // 3, 1, 1)
        return (1 - self.alpha) * x + self.alpha * images


@register_augmentation("random_resized_crop")
class RandomResizedCrop(nn.Module):
    """torchvision's RandomResizedCrop, drawn per sample: a crop covering
    `scale` of the image area with aspect ratio in `ratio`, resized back to
    the input size (bilinear). Crop boxes come from torchvision's own
    get_params (same rejection sampling and fallback), so the distribution
    is torchvision's -- R3M uses scale (0.2, 1.0). Channel-agnostic: on a
    stacked input (3*k channels: frame stacks, or pretrain.py's clips with
    clip_augmentation=shared) every frame of a sample gets the same crop.
    Draws on the CPU RNG; one resize per sample, so it's meant for batches
    of tens, not thousands."""

    def __init__(self, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)):
        super().__init__()
        scale, ratio = tuple(float(v) for v in scale), tuple(float(v) for v in ratio)
        if not 0 < scale[0] <= scale[1] <= 1:
            raise ValueError(f"random_resized_crop scale must satisfy 0 < min <= max <= 1, got {scale}")
        if not 0 < ratio[0] <= ratio[1]:
            raise ValueError(f"random_resized_crop ratio must satisfy 0 < min <= max, got {ratio}")
        self.scale, self.ratio = scale, ratio

    def forward(self, x):
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        h, w = x.shape[-2:]
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            top, left, ch, cw = T.RandomResizedCrop.get_params(x[i], self.scale, self.ratio)
            out[i] = TF.resized_crop(x[i], top, left, ch, cw, [h, w], antialias=True)
        return out


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
            "This augmentation needs kornia: pip install kornia (the version "
            "is pinned in requirements.txt; anything older than 0.6.10 treats "
            "gaussian_blur's sigma differently)."
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
    # sigma is a (min, max) range sampled per sample. Kornia only treats it
    # that way from 0.6.10; before that it was a fixed (sigma_x, sigma_y).
    # kernel_size is in pixels, so the same setting blurs less at 224px than
    # at 84px, and a 5-tap kernel truncates sigma=2 at 1 sigma.
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

# How far outside [0, 1] the first batch may stray before it's rejected. Bicubic
# resizing overshoots (up to ~0.42 on a worst-case 0/1 checkerboard, far less
# on real frames), so the tolerance is loose; it still catches 0-255 input and
# ImageNet-normalised input, which leave [0, 1] by more than 1.
_RANGE_TOL = 0.5


class _CheckInput(nn.Module):
    """Fails fast on the likely misuses: a raw uint8 HWC replay batch instead
    of a float NCHW one (checked every call, free), and frames that aren't in
    [0, 1] (checked on the first call only: reading the range back forces a
    GPU sync)."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._range_checked = False

    def forward(self, x):
        if x.ndim != 4 or not x.is_floating_point():
            raise ValueError(
                "augmentations take float (N, C, H, W) input in [0, 1], got "
                f"dtype={x.dtype} shape={tuple(x.shape)}"
            )
        if not self._range_checked:
            lo, hi = (v.item() for v in torch.aminmax(x.detach()))
            if lo < -_RANGE_TOL or hi > 1 + _RANGE_TOL:
                raise ValueError(
                    f"augmentations take input in [0, 1], got values in "
                    f"[{lo:.3g}, {hi:.3g}]. Max well above 1 suggests 0-255 "
                    "input; values well below 0 suggest mean/std "
                    "normalisation already applied. Augment before either "
                    "step (EmbeddingNet.preprocess() should stop at "
                    "resize/crop)."
                )
            self._range_checked = True
        return self.inner(x)


def _make(name: str, params: dict) -> nn.Module:
    if name == "none":
        if params:
            raise TypeError(f"augmentation 'none' takes no params, got {sorted(params)}")
        return nn.Identity()
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
    """A pipeline entry: a bare name, or {name: ..., params: {...}}."""
    if isinstance(spec, str):
        return _make(spec, {})
    try:
        spec = dict(spec)
    except (TypeError, ValueError):
        raise ValueError(
            "pipeline entries are a name or {name: ..., params: {...}}, "
            f"got {spec!r}"
        ) from None
    name = spec.pop("name", None)
    params = dict(spec.pop("params", None) or {})
    if name is None or spec:
        raise ValueError(
            "pipeline entries look like {name: ..., params: {...}}, got "
            f"name={name!r} plus extra keys {sorted(spec)}"
        )
    return _make(name, params)


def make_augmentation(name: str, **params) -> nn.Module:
    """Build a registered augmentation (or a `pipeline` of them) by name.
    "none" gives an identity that still enforces the input contract."""
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
