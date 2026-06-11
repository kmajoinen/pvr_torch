"""
Image augmentations for pixel-based RL and BC.

All augmentations expect float tensors of shape (N, C, H, W) in [0, 1] range,
applied between pixel normalisation and mean/std normalisation in EmbeddingNet.

Usage:
    from src.augmentations import make_augmentation
    aug = make_augmentation('random_shift', pad=4)          # single
    aug = make_augmentation('pipeline', names=['random_shift', 'color_jitter'])  # combined
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# ------------------------------------------------------------------------------
# Random shift (DrQ-v2)
# ------------------------------------------------------------------------------

class RandomShiftsAug(nn.Module):
    """Random shift augmentation from DrQ-v2 (Yarats et al., 2021).
    Pads by `pad` pixels with edge replication then takes a random crop,
    giving each sample in the batch an independent shift.
    Input: float (N, C, H, W) with H == W.
    """
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


# ------------------------------------------------------------------------------
# Color augmentations
# ------------------------------------------------------------------------------

class ColorJitter(nn.Module):
    """Wraps torchvision ColorJitter with RL-friendly defaults.
    Input: float (N, C, H, W) in [0, 1].
    """
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        super().__init__()
        self._jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue)

    def forward(self, x):
        return self._jitter(x)


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            return x.flip(-1)
        return x


class GaussianBlur(nn.Module):
    """Wraps torchvision GaussianBlur.
    Input: float (N, C, H, W) in [0, 1].
    """
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0)):
        super().__init__()
        self._blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, x):
        return self._blur(x)


class RandomErasing(nn.Module):
    """Wraps torchvision RandomErasing (Cutout-style).
    Input: float (N, C, H, W) in [0, 1]. Erased regions are set to 0.
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        super().__init__()
        self._erasing = T.RandomErasing(p=p, scale=scale, ratio=ratio, value=0)

    def forward(self, x):
        # RandomErasing expects (C, H, W) or (N, C, H, W)
        return torch.stack([self._erasing(xi) for xi in x])


# ------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------

_REGISTRY = {
    'random_shift':      lambda **kw: RandomShiftsAug(**kw),
    'color_jitter':      lambda **kw: ColorJitter(**kw),
    'horizontal_flip':   lambda **kw: RandomHorizontalFlip(**kw),
    'gaussian_blur':     lambda **kw: GaussianBlur(**kw),
    'random_erasing':    lambda **kw: RandomErasing(**kw),
}


def make_augmentation(name, **kwargs):
    """
    Build an augmentation module by name.

    Single augmentations:
        make_augmentation('random_shift', pad=4)
        make_augmentation('color_jitter', brightness=0.4)

    Pipeline of multiple augmentations (kwargs ignored):
        make_augmentation('pipeline', names=['random_shift', 'color_jitter'])
    """
    if name == 'pipeline':
        names = kwargs.get('names', ['random_shift'])
        return nn.Sequential(*[make_augmentation(n) for n in names])
    if name not in _REGISTRY:
        raise ValueError(f"Unknown augmentation '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
