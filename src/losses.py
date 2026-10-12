"""
Auxiliary encoder losses: objectives that train the encoder alongside, or
instead of, the RL/BC losses. Selected from config as a list:

    encoder:
      aux:
        - {name: l2sp, weight: 1.0}
        - {name: my_loss, weight: 0.1, scale: 2.0}   # extra keys -> __init__

Adding a loss:
    @register_aux_loss("my_loss")
    class MyLoss(AuxLoss):
        def __init__(self, encoder, scale: float = 1.0):
            super().__init__(encoder)
            self.scale = scale
            self.head = nn.Linear(encoder.out_size, 64)   # optional learnable head

        def forward(self, batch: AuxBatch) -> torch.Tensor:
            ...   # return an UNWEIGHTED scalar

Contract:
  - __init__(encoder, **params) runs once, after the encoder is built and
    before training. `encoder` is the EmbeddingNet. Snapshot from it if needed
    (l2sp does), but don't store it as an attribute -- nn.Module would
    register it as a submodule. Use batch.encoder in forward() instead.
  - forward(batch) returns an unweighted scalar tensor. The training script
    multiplies it by `weight`, sums the terms and backpropagates into the
    encoder. It must not step optimizers or touch the RL/BC networks.
  - Parameters the loss owns (decoders, predictor heads) are optimised by the
    encoder's optimizer, at encoder.lr.
  - Declare what the loss reads from AuxBatch with the class attributes
    below, so the training scripts can reject configs they can't serve
    before anything expensive runs:
        needs_features  = True   # batch.obs / batch.feats
        needs_augmented = True   # batch.obs_aug / batch.feats_aug
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn


@dataclass
class AuxBatch:
    """What an aux loss can read. Fields a loss didn't declare a need for may
    be None."""
    encoder: nn.Module
    # Preprocessed observations: float (N, C, H, W) in [0, 1] at the encoder's
    # input resolution (EmbeddingNet.preprocess output).
    obs: Optional[torch.Tensor] = None
    obs_aug: Optional[torch.Tensor] = None
    # Encoder outputs for the above, carrying gradient when the encoder trains.
    feats: Optional[torch.Tensor] = None
    feats_aug: Optional[torch.Tensor] = None


class AuxLoss(nn.Module):
    needs_features: bool = False
    needs_augmented: bool = False

    def __init__(self, encoder: nn.Module):
        super().__init__()

    def forward(self, batch: AuxBatch) -> torch.Tensor:
        raise NotImplementedError


_REGISTRY: Dict[str, Type[AuxLoss]] = {}


def register_aux_loss(name: str):
    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, AuxLoss)):
            raise TypeError(f"{cls!r} must subclass AuxLoss")
        if name in _REGISTRY:
            raise ValueError(f"aux loss name {name!r} is already taken")
        _REGISTRY[name] = cls
        return cls
    return decorator


def available_aux_losses():
    return sorted(_REGISTRY)


# ------------------------------------------------------------------------------
# Built-in losses
# ------------------------------------------------------------------------------

@register_aux_loss("l2sp")
class L2SP(AuxLoss):
    """||theta - theta_0||^2 over all encoder parameters (Xuhong et al., 2018):
    adapt the encoder while penalising drift from its pretrained weights."""

    def __init__(self, encoder: nn.Module):
        super().__init__(encoder)
        self._anchor = {n: p.detach().clone() for n, p in encoder.named_parameters()}

    def forward(self, batch: AuxBatch) -> torch.Tensor:
        return sum(
            ((p - self._anchor[n]) ** 2).sum()
            for n, p in batch.encoder.named_parameters()
        )


# ------------------------------------------------------------------------------
# Config -> losses
# ------------------------------------------------------------------------------

@dataclass
class AuxTerm:
    name: str
    weight: float
    loss: AuxLoss


def _split_spec(spec) -> Tuple[str, float, dict]:
    if not isinstance(spec, dict):
        raise ValueError(f"aux entries look like {{name: l2sp, weight: 1.0}}, got {spec!r}")
    spec = dict(spec)
    name = spec.pop("name", None)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown aux loss {name!r}. Available: {available_aux_losses()} "
            "(add your own in src/losses.py)"
        )
    weight = spec.pop("weight", None)
    if not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError(f"aux loss {name!r} needs a non-negative numeric `weight`, got {weight!r}")
    return name, float(weight), spec


def aux_loss_classes(specs) -> List[Type[AuxLoss]]:
    """Validate a config list of aux specs without building anything, and
    return the loss classes (for checking their declared needs)."""
    names = [_split_spec(s)[0] for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"aux loss listed more than once: {dupes}")
    return [_REGISTRY[n] for n in names]


def build_aux_losses(specs, encoder: nn.Module) -> List[AuxTerm]:
    aux_loss_classes(specs)
    terms = []
    for spec in specs:
        name, weight, params = _split_spec(spec)
        try:
            loss = _REGISTRY[name](encoder, **params)
        except TypeError as e:
            raise TypeError(f"bad params for aux loss {name!r}: {e}") from e
        terms.append(AuxTerm(name, weight, loss))
    return terms


def total_aux_loss(terms: List[AuxTerm], batch: AuxBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted sum of all terms, plus each term's unweighted value for logging."""
    total, values = 0.0, {}
    for t in terms:
        value = t.loss(batch)
        values[t.name] = value.item()
        total = total + t.weight * value
    return total, values
