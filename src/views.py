"""
Per-update observation views: which encodings the task loss and the aux
losses see, given augmentation.stage. Shared by train_sac.py and
train_bc.py so both algorithms route augmentation the same way.

    stage   task loss sees    aux losses can read
    rl      augmented         clean
    aux     clean             clean + augmented
    both    augmented         clean + augmented

The task loss is the SAC critic/actor or the BC policy loss. The
augmentation is drawn once per update; callers that encode the task view
again later (SAC's actor phase) reuse Views.x_task rather than re-drawing.

Cost: when an aux loss reads the same view as the task loss, the task's
encode is reused, so an update costs at most one encoder forward pass more
than the task loss alone. A view is encoded with gradient only if the task
loss or some aux loss backpropagates through it (aux_grad_clean /
aux_grad_aug), so a view read only as a detached target builds no graph.

Pure tensor plumbing: no optimizers, no losses, nothing RL-specific.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from src.losses import AuxBatch


@dataclass
class Views:
    # Preprocessed pixels the task loss sees (clean or augmented). Re-encode
    # these, not a fresh augmentation, when the task needs another pass.
    x_task: torch.Tensor
    # encoder(x_task). Carries gradient iff the task loss trains the encoder;
    # detached otherwise, even when the same encode feeds an aux loss.
    feats_task: torch.Tensor
    # What aux losses read; fields are filled only if some loss declared
    # needs_features / needs_augmented.
    aux: AuxBatch


def encode_views(
    featurize: Callable[[torch.Tensor, bool], torch.Tensor],
    encoder: torch.nn.Module,
    x_clean: torch.Tensor,
    augment: Optional[Callable[[torch.Tensor], torch.Tensor]],
    *,
    task_aug: bool,
    task_grad: bool,
    aux_needs_clean: bool,
    aux_needs_aug: bool,
    aux_grad_clean: Optional[bool] = None,
    aux_grad_aug: Optional[bool] = None,
) -> Views:
    """
    featurize(x, grad) -> features: the caller's encoder call on preprocessed
    pixels (autocast, grad context). x_clean: EmbeddingNet.preprocess output.
    augment: the augmentation (or None).

    aux_needs_clean / aux_needs_aug: does some aux loss read that view
    (needs_features / needs_augmented, from the loss classes).
    aux_grad_clean / aux_grad_aug: does some aux loss backpropagate through
    that view (losses.aux_grad_views(terms), from the built losses). A view
    an aux loss reads without gradient is encoded without a graph, or handed
    over detached when it shares the task's encode. None means "same as
    aux_needs_*": gradient wherever an aux loss reads, which is always
    correct but builds a graph for views a loss only uses as a target.
    """
    if aux_grad_clean is None:
        aux_grad_clean = aux_needs_clean
    if aux_grad_aug is None:
        aux_grad_aug = aux_needs_aug
    if (aux_grad_clean and not aux_needs_clean) or (aux_grad_aug and not aux_needs_aug):
        raise ValueError(
            "aux_grad_clean/aux_grad_aug asks for gradient through a view no "
            f"aux loss reads (aux_needs_clean={aux_needs_clean}, "
            f"aux_needs_aug={aux_needs_aug})"
        )
    if (task_aug or aux_needs_aug) and augment is None:
        raise ValueError("augmented view requested but no augmentation is configured")

    x_aug = augment(x_clean) if (task_aug or aux_needs_aug) else None
    x_task = x_aug if task_aug else x_clean

    # The task view doubles as the aux losses' view of the same kind.
    aux_grad_shared = aux_grad_aug if task_aug else aux_grad_clean
    f_task = featurize(x_task, task_grad or aux_grad_shared)
    # The shared encode may carry gradient for the aux loss; the task loss
    # must still not reach the encoder unless routed there.
    feats_task = f_task if task_grad else f_task.detach()
    # And vice versa: an aux loss that doesn't backpropagate through the
    # shared view gets it detached, even when the task's encode has a graph.
    f_shared_aux = f_task if aux_grad_shared else f_task.detach()

    feats_clean = feats_aug = None
    if aux_needs_clean:
        feats_clean = featurize(x_clean, aux_grad_clean) if task_aug else f_shared_aux
    if aux_needs_aug:
        feats_aug = f_shared_aux if task_aug else featurize(x_aug, aux_grad_aug)

    aux = AuxBatch(
        encoder=encoder,
        obs=x_clean if aux_needs_clean else None,
        obs_aug=x_aug if aux_needs_aug else None,
        feats=feats_clean,
        feats_aug=feats_aug,
    )
    return Views(x_task=x_task, feats_task=feats_task, aux=aux)


class CollapseMonitor:
    """Tracks the per-dimension std of the raw encoder output across a batch,
    relative to its first measured value, and warns once if it drops below
    warn_frac -- the signature of an encoder collapsing toward a constant
    (see aug_consistency in src/losses.py). Measure features before any
    LayerNorm, which would hide a collapse."""

    def __init__(self, warn_frac: float = 0.1):
        self.warn_frac = warn_frac
        self.initial = None
        self.current = float("nan")
        self._warned = False

    def update(self, feats: torch.Tensor, where: str = "") -> None:
        self.current = feats.detach().float().std(dim=0).mean().item()
        if self.initial is None:
            self.initial = self.current
        elif not self._warned and self.current < self.warn_frac * self.initial:
            print(
                f"WARNING {where}: embedding std fell to {self.ratio:.1%} of its "
                "initial value -- the encoder may be collapsing (see "
                "aug_consistency in src/losses.py)."
            )
            self._warned = True

    @property
    def ratio(self) -> float:
        return self.current / self.initial if self.initial else float("nan")
