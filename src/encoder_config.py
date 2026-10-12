"""
Resolution and validation of the encoder / augmentation / buffer settings
shared by train_sac.py and train_bc.py.

Everything here runs before any env, dataset or encoder is built, so a
contradictory config fails in seconds instead of after model loading, and a
setting that would be silently ignored raises instead.

The three independent axes:
  encoder.grads   which task losses backprop into the encoder
                  (SAC: none | critic | actor | both; BC: none | policy)
  encoder.aux     list of auxiliary losses (src/losses.py)
  augmentation    which transform (configs/augmentation/), and at which
                  stage it's applied

The encoder trains (gets an optimizer) iff grads != none or aux is non-empty.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.losses import aux_loss_classes

AUG_STAGES = ("rl", "aux", "both")
BUFFER_OBS = ("auto", "features", "pixels")

_REMOVED_FINETUNE = (
    "The `finetune.*` config block was replaced in config_version 2:\n"
    "  finetune.enabled            -> removed; the encoder trains iff "
    "encoder.grads != none or encoder.aux is non-empty\n"
    "  finetune.encoder_grads      -> encoder.grads\n"
    "  finetune.encoder_lr / encoder_betas / grad_clip -> encoder.lr / betas / grad_clip\n"
    "  finetune.aux_loss=l2sp, aux_weight=w -> 'encoder.aux=[{name:l2sp,weight:w}]'\n"
    "  slow-frozen (pixels in buffer, nothing trains) -> buffer_obs=pixels\n"
    "See RUNNING_EXPERIMENTS.md, 'Encoder training and augmentation'."
)


@dataclass(frozen=True)
class EncoderSettings:
    has_embedding: bool
    grads: str
    aux_specs: List[dict]
    trains: bool
    aug_name: Optional[str]
    aug_stage: str

    @property
    def aux_names(self) -> List[str]:
        return [s["name"] for s in self.aux_specs]


def _plain(node):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def encoder_settings(cfg, valid_grads: Tuple[str, ...]) -> EncoderSettings:
    if "finetune" in cfg:
        raise ValueError(_REMOVED_FINETUNE)

    has_embedding = cfg.embedding.get("name") is not None
    enc = cfg.encoder
    grads = str(enc.grads)
    if grads not in valid_grads:
        raise ValueError(f"encoder.grads={grads!r}; expected one of {valid_grads}")

    aux_specs = list(_plain(enc.get("aux")) or [])
    aux_classes = aux_loss_classes(aux_specs)
    # Routing of features / augmented observations into aux losses is not
    # wired yet, so a loss that needs them would silently get None.
    unserved = [s["name"] for s, c in zip(aux_specs, aux_classes)
                if c.needs_features or c.needs_augmented]
    if unserved:
        raise NotImplementedError(
            f"aux losses {unserved} need encoder features or augmented "
            "observations, which the training loop doesn't pass to aux losses yet."
        )

    aug = cfg.get("augmentation") or {}
    aug_name = aug.get("name")
    aug_name = None if aug_name in (None, "none") else str(aug_name)
    aug_stage = str(aug.get("stage", "rl"))
    if aug_stage not in AUG_STAGES:
        raise ValueError(f"augmentation.stage={aug_stage!r}; expected one of {AUG_STAGES}")
    if aug_name is not None and aug_stage != "rl":
        raise NotImplementedError(
            f"augmentation.stage={aug_stage!r} is not implemented yet; only "
            "stage=rl (the task loss sees augmented observations) is."
        )

    trains = grads != "none" or bool(aux_specs)
    if not has_embedding:
        used = [k for k, on in (("encoder.grads", grads != "none"),
                                ("encoder.aux", bool(aux_specs)),
                                ("augmentation", aug_name is not None)) if on]
        if used:
            raise ValueError(
                f"{', '.join(used)} set, but there is no encoder (embedding=none): "
                "these only apply to pixel observations through an embedding."
            )

    return EncoderSettings(has_embedding, grads, aux_specs, trains, aug_name, aug_stage)


def resolve_buffer_obs(cfg, s: EncoderSettings) -> Tuple[bool, str]:
    """SAC only: does the replay buffer store pixels (re-encoded every
    update) or features (encoded once, at env-step time)? Returns
    (pixels, human-readable reason)."""
    mode = str(cfg.get("buffer_obs", "auto"))
    if mode not in BUFFER_OBS:
        raise ValueError(f"buffer_obs={mode!r}; expected one of {BUFFER_OBS}")

    if not s.has_embedding:
        if mode == "pixels":
            raise ValueError("buffer_obs=pixels needs an embedding (embedding=none stores state vectors)")
        return False, "state vectors (no embedding)"

    needs = []
    if s.trains:
        needs.append(f"encoder trains: grads={s.grads}"
                     + (f", aux={s.aux_names}" if s.aux_specs else ""))
    if s.aug_name is not None:
        needs.append(f"augmentation {s.aug_name!r} is drawn per sample at every update")

    if mode == "features":
        if needs:
            raise ValueError(
                "buffer_obs=features stores encoded features, which can't be "
                f"re-encoded or augmented, but {'; '.join(needs)}. "
                "Use buffer_obs=auto or pixels."
            )
        return False, "features (buffer_obs=features)"
    if mode == "pixels":
        if needs:
            return True, f"pixels (buffer_obs=pixels; {'; '.join(needs)})"
        return True, ("pixels (buffer_obs=pixels, slow-frozen: same features as "
                      "the fast path, re-encoded every update)")
    if needs:
        return True, f"pixels (auto: {'; '.join(needs)})"
    return False, "features (auto: frozen encoder, no augmentation; each frame encoded once at env-step time)"


def warn_unused_encoder_overrides(s: EncoderSettings) -> None:
    """encoder.lr/betas/grad_clip have defaults, so the config alone can't tell
    a deliberate setting from the default. The command-line overrides can."""
    if s.trains:
        return
    try:
        from hydra.core.hydra_config import HydraConfig
        overrides = HydraConfig.get().overrides.task
    except Exception:
        return
    unused = [o for o in overrides
              if o.lstrip("+~").split("=")[0] in ("encoder.lr", "encoder.betas", "encoder.grad_clip")]
    if unused:
        print(f"WARNING: {unused} have no effect: the encoder is frozen "
              "(encoder.grads=none and encoder.aux is empty).")
