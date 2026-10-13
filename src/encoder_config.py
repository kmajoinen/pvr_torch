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
                  stage it's applied (src/views.py):
                    rl    the task loss sees augmented observations
                    aux   only aux losses see them; the task loss sees clean
                    both  both do

The encoder trains (gets an optimizer) iff grads != none or aux is non-empty.
"""

import os
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
    trainable: str          # 'all' | 'head' (encoder.trainable); see EmbeddingNet.set_trainable
    aug_name: Optional[str]
    aug_stage: str
    task_aug: bool          # the task loss sees augmented observations
    aux_needs_clean: bool   # some aux loss reads the clean view (needs_features)
    aux_needs_aug: bool     # some aux loss reads the augmented view

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
    aux_needs_clean = any(c.needs_features for c in aux_classes)
    aux_needs_aug = any(c.needs_augmented for c in aux_classes)
    aug_readers = [s["name"] for s, c in zip(aux_specs, aux_classes) if c.needs_augmented]

    aug = cfg.get("augmentation") or {}
    aug_name = aug.get("name")
    aug_name = None if aug_name in (None, "none") else str(aug_name)
    aug_stage = str(aug.get("stage", "rl"))
    if aug_stage not in AUG_STAGES:
        raise ValueError(f"augmentation.stage={aug_stage!r}; expected one of {AUG_STAGES}")
    if aug_name is None and aug_stage != "rl":
        raise ValueError(f"augmentation.stage={aug_stage} has no effect without an augmentation")
    if aug_name is not None and aug_stage in ("aux", "both") and not aux_needs_aug:
        raise ValueError(
            f"augmentation.stage={aug_stage} feeds augmented observations to aux "
            f"losses, but none of encoder.aux={[s['name'] for s in aux_specs]} "
            "reads them (needs_augmented). Add one, e.g. aug_consistency, or use stage=rl."
        )
    if aug_readers and (aug_name is None or aug_stage == "rl"):
        raise ValueError(
            f"aux losses {aug_readers} read augmented observations, which need an "
            "augmentation with stage=aux or both "
            f"(got augmentation={aug_name}, stage={aug_stage})."
        )
    task_aug = aug_name is not None and aug_stage in ("rl", "both")

    trains = grads != "none" or bool(aux_specs)

    trainable = str(enc.get("trainable", "all"))
    if trainable not in ("all", "head"):
        raise ValueError(f"encoder.trainable={trainable!r}; expected all or head")
    emb_head = cfg.embedding.get("head") if has_embedding else None
    if emb_head == "linear" and not trains:
        raise ValueError(
            "embedding.head=linear is trainable, but the encoder is frozen "
            "(encoder.grads=none, encoder.aux empty), so it would stay a random "
            "projection. Set encoder.grads (e.g. critic) or encoder.aux."
        )
    if trainable == "head" and emb_head != "linear":
        raise ValueError(
            f"encoder.trainable=head trains only the head, but embedding.head={emb_head} "
            "has no parameters. Use embedding.head=linear, or encoder.trainable=all."
        )

    if not has_embedding:
        used = [k for k, on in (("encoder.grads", grads != "none"),
                                ("encoder.aux", bool(aux_specs)),
                                ("augmentation", aug_name is not None)) if on]
        if used:
            raise ValueError(
                f"{', '.join(used)} set, but there is no encoder (embedding=none): "
                "these only apply to pixel observations through an embedding."
            )

    # Without RL gradients, only aux losses move the encoder. One that reads
    # no observations (l2sp) only pulls weights back toward where they start,
    # so its gradient is zero throughout and the run equals a frozen one.
    # Keyed off needs_*, so a custom loss that reads observations counts.
    if trains and grads == "none" and not (aux_needs_clean or aux_needs_aug):
        raise ValueError(
            f"encoder.aux={[s['name'] for s in aux_specs]} with encoder.grads=none: no aux "
            "loss reads observations, so nothing moves the encoder (l2sp's gradient is "
            "zero at the pretrained weights) and the run equals a frozen one. Set "
            "encoder.grads, or add an aux loss that reads observations (e.g. aug_consistency)."
        )

    return EncoderSettings(
        has_embedding, grads, aux_specs, trains, trainable, aug_name, aug_stage,
        task_aug, aux_needs_clean, aux_needs_aug,
    )


def embedding_kwargs(cfg) -> dict:
    """The optional encoder fields in cfg.embedding (arch, checkpoint,
    preprocess; see EmbeddingNet), validated and ready to pass to
    EmbeddingNet. Call before the encoder loads, so a missing checkpoint or a
    bad preprocess spec fails in seconds.

    A relative checkpoint path is resolved against the directory the run was
    launched from, not Hydra's run directory (train_sac.py runs with
    hydra.job.chdir=true)."""
    from omegaconf import OmegaConf

    emb = cfg.embedding
    if emb.get("name") is None:
        return {}
    # .get() returns None for a `???` value instead of raising, which would
    # silently fall back to the architecture's default weights. (`k in emb`
    # is False for `???` too, hence keys().)
    missing = [k for k in ("arch", "checkpoint", "preprocess", "layer", "head", "head_dim")
               if k in emb.keys() and OmegaConf.is_missing(emb, k)]
    if missing:
        raise ValueError(
            f"embedding.{missing[0]} must be set for embedding={emb.name} "
            f"(e.g. embedding.{missing[0]}=...)"
        )
    kwargs = {}
    if emb.get("arch") is not None:
        kwargs["arch"] = str(emb.arch)
    if emb.get("checkpoint") is not None:
        from hydra.utils import to_absolute_path
        from src.embeddings import CHECKPOINT_ARCHS
        arch = str(emb.get("arch") or emb.name)
        if arch not in CHECKPOINT_ARCHS:
            raise ValueError(
                f"embedding.checkpoint can't be loaded into arch={arch!r}: its builder "
                "always loads or downloads its own default weights first. "
                f"Supported: {sorted(CHECKPOINT_ARCHS)}"
            )
        path = to_absolute_path(str(emb.checkpoint))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"embedding.checkpoint={emb.checkpoint!r}: no file at {path}")
        kwargs["checkpoint"] = path
    if emb.get("preprocess") is not None:
        from src.embeddings import preprocess_transforms
        spec = _plain(emb.preprocess)
        preprocess_transforms(spec)   # raises on a bad spec
        kwargs["preprocess"] = spec

    # Truncation and head (src/truncation.py). Which module a layer name
    # resolves to, and whether the head fits that layer's output, can only
    # be checked once the encoder is built; types and names are checked here.
    from src.truncation import HEADS
    layer = emb.get("layer")
    if layer is not None:
        if isinstance(layer, bool) or not isinstance(layer, (int, str)) or \
                (isinstance(layer, int) and layer < 1) or layer == "":
            raise ValueError(f"embedding.layer={layer!r}: expected a stage name (layer2), "
                             "a block count >= 1, or a module path")
        kwargs["layer"] = layer
    head = emb.get("head")
    if head is not None:
        if head not in HEADS:
            raise ValueError(f"embedding.head={head!r}; expected one of {HEADS}")
        kwargs["head"] = str(head)
    head_dim = emb.get("head_dim")
    if head_dim is not None:
        if head != "linear":
            raise ValueError("embedding.head_dim only applies to embedding.head=linear")
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim < 1:
            raise ValueError(f"embedding.head_dim={head_dim!r}: expected a positive integer")
        kwargs["head_dim"] = head_dim
    return kwargs


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
              if o.lstrip("+~").split("=")[0] in ("encoder.lr", "encoder.betas",
                                                  "encoder.grad_clip", "encoder.trainable")]
    if unused:
        print(f"WARNING: {unused} have no effect: the encoder is frozen "
              "(encoder.grads=none and encoder.aux is empty).")
