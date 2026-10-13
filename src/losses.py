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
    below. The training loop fills only what some loss declared (each costs
    an encoder pass), and configs that can't supply it are rejected before
    anything expensive runs:
        needs_features  = True   # batch.obs / batch.feats      (clean view)
        needs_augmented = True   # batch.obs_aug / batch.feats_aug
    needs_augmented also requires an augmentation with stage aux or both
    (see src/views.py).
  - Declare which of those views the loss backpropagates through with the
    grad_features / grad_augmented properties. They default to
    needs_features / needs_augmented. Override them when the loss reads a
    view only as a fixed target (aug_consistency does): that view is then
    encoded without gradient, which saves a full set of encoder activations,
    and it reaches the loss detached even when the task loss's encode of the
    same view carries gradient. The training scripts pass
    aux_grad_views(terms) to encode_views.
  - Param names are checked against __init__'s signature at startup, before
    the encoder loads. To check param values that early too, override the
    check_params classmethod (aug_consistency does, for `target`).
"""

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AuxBatch:
    """What an aux loss can read. Fields a loss didn't declare a need for may
    be None."""
    encoder: nn.Module
    # Preprocessed observations: float (N, C, H, W) in [0, 1] at the encoder's
    # input resolution (EmbeddingNet.preprocess output).
    obs: Optional[torch.Tensor] = None
    obs_aug: Optional[torch.Tensor] = None
    # Encoder outputs for the above. A view carries gradient only if some loss
    # declared it (grad_features / grad_augmented); otherwise it is detached.
    feats: Optional[torch.Tensor] = None
    feats_aug: Optional[torch.Tensor] = None


class AuxLoss(nn.Module):
    needs_features: bool = False
    needs_augmented: bool = False

    def __init__(self, encoder: nn.Module):
        super().__init__()

    @property
    def grad_features(self) -> bool:
        """Does this loss backpropagate through batch.feats?"""
        return self.needs_features

    @property
    def grad_augmented(self) -> bool:
        """Does this loss backpropagate through batch.feats_aug?"""
        return self.needs_augmented

    @classmethod
    def check_params(cls, **params) -> None:
        """Validate config param values without building the loss. Runs at
        startup, before the encoder loads. Raise ValueError on a bad value."""

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

def _backbone(encoder: nn.Module) -> nn.Module:
    """EmbeddingNet's pretrained backbone (.embedding), without a trainable
    head; any other module as-is."""
    return getattr(encoder, "embedding", encoder)


@register_aux_loss("l2sp")
class L2SP(AuxLoss):
    """||theta - theta_0||^2 over the backbone's parameters (Xuhong et al.,
    2018): adapt the encoder while penalising drift from its pretrained
    weights. A trainable head (embedding.head=linear) is excluded: it starts
    from a random init, which L2-SP has no reason to anchor to."""

    def __init__(self, encoder: nn.Module):
        super().__init__(encoder)
        self._anchor = {n: p.detach().clone() for n, p in _backbone(encoder).named_parameters()}

    def forward(self, batch: AuxBatch) -> torch.Tensor:
        return sum(
            ((p - self._anchor[n]) ** 2).sum()
            for n, p in _backbone(batch.encoder).named_parameters()
        )


@register_aux_loss("aug_consistency")
class AugConsistency(AuxLoss):
    """MSE between the embeddings of the clean and augmented views of the
    same observations, on the raw encoder output.

    target: which branch is the fixed target (detached); the other one
    receives the gradient. 'clean' (default) pulls augmented embeddings
    toward clean ones -- the usual invariance direction. The target branch
    is declared gradient-free (grad_features / grad_augmented), so it is
    encoded without building a graph.

    On its own (encoder.grads=none, no other aux loss) this objective is
    degenerate: mapping every input to the same embedding gives zero loss, so
    that's the optimum, and nothing in it preserves information. It's
    SimSiam without the predictor SimSiam showed is necessary. Pair it with
    l2sp or with RL gradients (grads != none), and watch
    train/embedding_std_ratio, which the training scripts warn on.
    """
    needs_features = True
    needs_augmented = True

    @classmethod
    def check_params(cls, target: str = "clean") -> None:
        if target not in ("clean", "augmented"):
            raise ValueError(f"aug_consistency target must be 'clean' or 'augmented', got {target!r}")

    def __init__(self, encoder: nn.Module, target: str = "clean"):
        super().__init__(encoder)
        self.check_params(target=target)
        self.target = target

    @property
    def grad_features(self) -> bool:
        return self.target == "augmented"

    @property
    def grad_augmented(self) -> bool:
        return self.target == "clean"

    def forward(self, batch: AuxBatch) -> torch.Tensor:
        clean, aug = batch.feats, batch.feats_aug
        # encode_views already hands the target branch over detached; detach
        # here too so direct callers get the same semantics.
        if self.target == "clean":
            clean = clean.detach()
        else:
            aug = aug.detach()
        return F.mse_loss(aug, clean)


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
    """Validate a config list of aux specs without building anything (names,
    weights, duplicates, param names against __init__, param values via
    check_params) and return the loss classes, for checking their declared
    needs. Runs at startup, before the encoder loads."""
    split = [_split_spec(s) for s in specs]
    names = [name for name, _, _ in split]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"aux loss listed more than once: {dupes}")
    classes = []
    for name, _, params in split:
        cls = _REGISTRY[name]
        try:
            inspect.signature(cls.__init__).bind(None, None, **params)  # self, encoder
        except TypeError as e:
            raise TypeError(f"bad params for aux loss {name!r}: {e}") from None
        cls.check_params(**params)
        classes.append(cls)
    return classes


def build_aux_losses(specs, encoder: nn.Module) -> List[AuxTerm]:
    aux_loss_classes(specs)
    terms = []
    for spec in specs:
        name, weight, params = _split_spec(spec)
        try:
            loss = _REGISTRY[name](encoder, **params)
        except TypeError as e:
            raise TypeError(f"bad params for aux loss {name!r}: {e}") from e
        # grad_* depend on params, so this is the earliest they can be checked.
        # encode_views would otherwise raise only at the first training update.
        if (loss.grad_features and not loss.needs_features) or \
                (loss.grad_augmented and not loss.needs_augmented):
            raise ValueError(
                f"aux loss {name!r} declares gradient through a view it doesn't "
                f"read: grad_features={loss.grad_features} with needs_features="
                f"{loss.needs_features}, grad_augmented={loss.grad_augmented} "
                f"with needs_augmented={loss.needs_augmented}"
            )
        terms.append(AuxTerm(name, weight, loss))
    return terms


def aux_grad_views(terms: List[AuxTerm]) -> Tuple[bool, bool]:
    """(clean, augmented): does any built loss backpropagate through that
    view? Pass to encode_views as aux_grad_clean / aux_grad_aug."""
    return (any(t.loss.grad_features for t in terms),
            any(t.loss.grad_augmented for t in terms))


def total_aux_loss(terms: List[AuxTerm], batch: AuxBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted sum of all terms, plus each term's unweighted value for logging."""
    total, values = 0.0, {}
    for t in terms:
        value = t.loss(batch)
        values[t.name] = value.item()
        total = total + t.weight * value
    return total, values
