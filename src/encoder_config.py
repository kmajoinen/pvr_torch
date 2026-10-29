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
    bn_adapt: bool = False              # encoder.bn=adapt
    strong_name: Optional[str] = None   # augmentation.strong: the SVEA-style second view
    strong_weight: float = 0.0

    @property
    def aux_names(self) -> List[str]:
        return [s["name"] for s in self.aux_specs]


def _plain(node):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def check_head_is_trained(net, trains: bool) -> None:
    """A trainable head (embedding.head=linear or fc) on an encoder that
    doesn't train has to come trained from embedding.checkpoint, or it stays
    a random projection. Config validation (encoder_settings) only sees that
    a checkpoint is set; this checks, once the encoder is built, that the
    file held head weights -- a file saved by this repo with a trained head.
    Other files (a bare backbone, e.g. an official R3M snapshot) load into
    the backbone only."""
    from src.truncation import TRAINABLE_HEADS
    head = net.build_spec.get("head")
    if head in TRAINABLE_HEADS and not trains and not net.head_loaded:
        raise ValueError(
            f"embedding.head={head} is trainable, but the encoder is frozen and "
            f"{net.checkpoint!r} has no weights for it, so it would stay a random "
            "projection. Load an encoder saved with this head (e.g. a pretrain.py VIP "
            "snapshot), or set encoder.grads / encoder.aux so it trains.")


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
    pretrain_only = [s["name"] for s, c in zip(aux_specs, aux_classes)
                     if c.needs_clips or c.needs_text or c.needs_views or c.needs_labels]
    if pretrain_only:
        raise ValueError(
            f"encoder.aux {pretrain_only}: these objectives read video clips, text, multi-view "
            "images or class labels, which replay buffers and BC datasets don't provide. Use "
            "them in pretrain.py."
        )
    aux_needs_clean = any(c.needs_features for c in aux_classes)
    aux_needs_aug = any(c.needs_augmented for c in aux_classes)
    aug_readers = [s["name"] for s, c in zip(aux_specs, aux_classes) if c.needs_augmented]

    aug = cfg.get("augmentation") or {}
    from omegaconf import OmegaConf
    if OmegaConf.is_config(aug):
        unset = sorted(OmegaConf.missing_keys(aug))
        if unset:
            raise ValueError(f"augmentation settings must be set: {['augmentation.' + k for k in unset]}")
    aug_name = aug.get("name")
    aug_name = None if aug_name in (None, "none") else str(aug_name)
    aug_stage = str(aug.get("stage", "rl"))

    # augmentation.strong: an optional second, stronger view of the task
    # observations (applied on top of the task view), which the critic loss
    # (BC: the policy loss) also fits: (1 - weight) * L(task view) +
    # weight * L(strong view) -- SVEA (Hansen et al., 2021), PIE-G.
    strong = aug.get("strong")
    strong_name, strong_weight = None, 0.0
    if strong is not None:
        strong_name = strong.get("name")
        strong_name = None if strong_name in (None, "none") else str(strong_name)
        if strong_name is not None:
            strong_weight = strong.get("weight", 0.5)
            if isinstance(strong_weight, bool) or not isinstance(strong_weight, (int, float)) \
                    or not 0 < strong_weight <= 1:
                raise ValueError(f"augmentation.strong.weight={strong_weight!r}: expected a number in (0, 1]")
            strong_weight = float(strong_weight)

    bn = str(enc.get("bn", "frozen"))
    if bn not in ("frozen", "adapt"):
        raise ValueError(f"encoder.bn={bn!r}; expected frozen or adapt")
    bn_adapt = bn == "adapt"
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
    from src.truncation import TRAINABLE_HEADS
    emb_head = cfg.embedding.get("head") if has_embedding else None
    if emb_head in TRAINABLE_HEADS and not trains and not cfg.embedding.get("checkpoint"):
        raise ValueError(
            f"embedding.head={emb_head} is trainable, but the encoder is frozen "
            "(encoder.grads=none, encoder.aux empty), so it would stay a random "
            "projection. Set encoder.grads (e.g. critic) or encoder.aux."
        )
    if trainable == "head" and emb_head not in TRAINABLE_HEADS:
        raise ValueError(
            f"encoder.trainable=head trains only the head, but embedding.head={emb_head} "
            "has no parameters. Use embedding.head=linear (or fc), or encoder.trainable=all."
        )

    if not has_embedding:
        used = [k for k, on in (("encoder.grads", grads != "none"),
                                ("encoder.aux", bool(aux_specs)),
                                ("augmentation", aug_name is not None),
                                ("augmentation.strong", strong_name is not None),
                                ("encoder.bn=adapt", bn_adapt)) if on]
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
        bn_adapt=bn_adapt, strong_name=strong_name, strong_weight=strong_weight,
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
    missing = [k for k in ("arch", "checkpoint", "preprocess", "layer", "head", "head_dim",
                           "frame_stack", "frame_fusion")
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
        from src.truncation import TRAINABLE_HEADS
        if head not in TRAINABLE_HEADS:
            raise ValueError("embedding.head_dim only applies to embedding.head=linear or fc")
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim < 1:
            raise ValueError(f"embedding.head_dim={head_dim!r}: expected a positive integer")
        kwargs["head_dim"] = head_dim

    # Frame stacking (EmbeddingNet): k frames per observation, fused by
    # channels (into the backbone's input; `random` only), concat or diff
    # (PIE-G) of the per-frame outputs.
    from src.truncation import FRAME_FUSIONS
    frame_stack = emb.get("frame_stack", 1)
    if isinstance(frame_stack, bool) or not isinstance(frame_stack, int) or frame_stack < 1:
        raise ValueError(f"embedding.frame_stack={frame_stack!r}: expected an integer >= 1")
    fusion = emb.get("frame_fusion")
    if fusion is not None:
        if fusion not in FRAME_FUSIONS:
            raise ValueError(f"embedding.frame_fusion={fusion!r}; expected one of {FRAME_FUSIONS}")
        if frame_stack < 2:
            raise ValueError("embedding.frame_fusion needs embedding.frame_stack > 1")
        if fusion == "channels" and str(emb.get("arch") or emb.name) != "random":
            raise ValueError("embedding.frame_fusion=channels only works with the `random` encoder "
                             "(others have a fixed 3-channel first layer); use concat or diff")
        kwargs["frame_fusion"] = str(fusion)
    if frame_stack > 1:
        kwargs["frame_stack"] = frame_stack
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
    if s.strong_name is not None:
        needs.append(f"strong view {s.strong_name!r} is drawn per sample at every update")
    if s.bn_adapt:
        needs.append("encoder.bn=adapt: BatchNorm statistics keep changing, so features can't be cached")

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
