"""
DrQ-v2's actor and critic (Yarats et al. 2021, facebookresearch/drqv2),
on top of this repo's encoders instead of DrQ-v2's own conv net.

Both networks start with their own trunk, Linear(feat_dim, feature_dim) +
LayerNorm + Tanh, that compresses the encoder's features before the MLPs.
The actor is deterministic with scheduled exploration noise: a tanh mean
plus a TruncatedNormal of fixed (scheduled) std, no entropy term. Actions
live in [-1, 1]; train_drqv2.py rescales them to the env's action space.
"""

import re

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


def weight_init(m):
    """DrQ-v2's init: orthogonal weights, zero biases."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def stddev_schedule(schedule, step: int) -> float:
    """A constant, or {init, final, duration}: linear from init to final over
    `duration` agent steps, then final. Also accepts DrQ-v2's string form
    'linear(init,final,duration)'."""
    if isinstance(schedule, (int, float)):
        return float(schedule)
    if isinstance(schedule, str):
        m = re.fullmatch(r"\s*linear\(([^,]+),([^,]+),([^,]+)\)\s*", schedule)
        if m is None:
            return float(schedule)
        init, final, duration = (float(g) for g in m.groups())
    else:
        init, final, duration = (float(schedule[k]) for k in ("init", "final", "duration"))
    mix = float(np.clip(step / duration, 0.0, 1.0))
    return (1.0 - mix) * init + mix * final


class TruncatedNormal(pyd.Normal):
    """Normal whose samples are clamped into (low, high) with a straight-
    through gradient; sample(clip) additionally clips the noise (DrQ-v2's
    target policy smoothing). log_prob/entropy are the untruncated
    Normal's, as in DrQ-v2."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low, self.high, self.eps = low, high, eps

    def _clamp(self, x):
        clamped = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped.detach()

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


def _trunk(feat_dim: int, feature_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(feat_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = _trunk(feat_dim, feature_dim)
        self.policy = _mlp(feature_dim, hidden_dim, action_dim)
        self.apply(weight_init)

    def forward(self, feats, std: float) -> TruncatedNormal:
        mu = torch.tanh(self.policy(self.trunk(feats)))
        return TruncatedNormal(mu, torch.ones_like(mu) * std)


class Critic(nn.Module):
    """Twin Q heads on one shared trunk."""

    def __init__(self, feat_dim: int, action_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = _trunk(feat_dim, feature_dim)
        self.q1 = _mlp(feature_dim + action_dim, hidden_dim, 1)
        self.q2 = _mlp(feature_dim + action_dim, hidden_dim, 1)
        self.apply(weight_init)

    def forward(self, feats, action):
        h = torch.cat([self.trunk(feats), action], dim=-1)
        return self.q1(h), self.q2(h)
