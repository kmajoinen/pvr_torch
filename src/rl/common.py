"""
Setup shared by the off-policy scripts (train_sac.py, train_drqv2.py), so
each script only holds its own algorithm: the encoder and its augmentations
(EncoderStack), the encode helpers (Encoding), the encoder's optimizer and
aux losses (EncoderOptim), the wandb run (init_wandb) and saved encoders
(encoder_record).

What each setting means is documented where it's validated
(src/encoder_config.py) and in RUNNING_EXPERIMENTS.md.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from src.encoder_config import check_head_is_trained
from src.losses import aux_grad_views, build_aux_losses


# ------------------------------------------------------------------------------
# Encoder and augmentations
# ------------------------------------------------------------------------------

@dataclass
class EncoderStack:
    net: Optional[nn.Module]           # EmbeddingNet, or None for state observations
    aug: Optional[nn.Module]           # the task view's augmentation
    strong: Optional[nn.Module]        # augmentation.strong
    pixels_in_buffer: bool             # final decision, after the memory guard
    buffer_reason: str
    trainable: str                     # 'none' | 'all' | 'head'
    bn_ref: Optional[dict]             # BatchNorm statistics as loaded (bn=adapt)


def build_encoder_stack(cfg, enc, emb_kwargs, device, pixels_in_buffer: bool,
                        buffer_reason: str) -> EncoderStack:
    emb_name = cfg.embedding.get("name", None)
    if emb_name is None:
        return EncoderStack(None, None, None, pixels_in_buffer, buffer_reason, "none", None)

    from src.augmentations import build_augmentation
    from src.embeddings import EmbeddingNet

    # Only ever non-None on the pixel path (resolve_buffer_obs guarantees
    # augmentation => pixels), so this banner means it really runs.
    aug = build_augmentation(cfg.augmentation)
    if aug is not None:
        aug = aug.to(device)
        print(
            f"Augmentation: {enc.aug_name}  stage={enc.aug_stage}"
            f"  params={dict(cfg.augmentation.get('params') or {})}"
            "  (replay-batch encodes only; rollouts/eval stay clean)"
        )
    strong = None
    if enc.strong_name is not None:
        strong = build_augmentation(cfg.augmentation.strong).to(device)
        print(
            f"Strong view: {enc.strong_name}  weight={enc.strong_weight}"
            f"  params={dict(cfg.augmentation.strong.get('params') or {})}"
            "  (on top of the task view; critic loss only)"
        )

    net = EmbeddingNet(
        emb_name,
        pretrained=True,
        train=enc.trains,
        disable_cuda=(str(device) == "cpu"),
        model_dir=cfg.model_dir,
        input_size=cfg.env.get("image_size"),
        **emb_kwargs,
    )
    # eval() ALWAYS, even while weights train: freeze BatchNorm running
    # stats (encoder.bn=adapt switches only the BatchNorm layers, during
    # updates). Trainability is controlled purely via requires_grad.
    net.eval()
    check_head_is_trained(net, enc.trains)
    trainable = enc.trainable if enc.trains else "none"
    net.set_trainable(trainable)
    bn_ref = None
    if enc.bn_adapt:
        if not net.has_batchnorm():
            raise ValueError(f"encoder.bn=adapt, but {emb_name!r} has no BatchNorm layers")
        bn_ref = net.bn_stats()   # as loaded; for train/bn_stat_drift
    print(
        f"\nEmbedding: {emb_name}  →  obs_size: {net.out_size}"
        f"  (trainable={trainable})"
        + "".join(f"  {k}={v}" for k, v in emb_kwargs.items())
    )
    head_params = sum(p.numel() for p in net.head.parameters())
    if head_params:
        print(f"Encoder head: {head_params / 1e6:.1f}M parameters")

    # The fast path stores features; when a wide output (e.g. head=flatten
    # on an early layer) makes each feature vector bigger than the frame it
    # came from, pixels are the cheaper thing to store.
    frame_side = cfg.env.get("image_size")
    if (not pixels_in_buffer and str(cfg.get("buffer_obs", "auto")) == "auto"
            and frame_side):
        frame_bytes = 3 * net.frame_stack * frame_side ** 2
        feat_bytes = 4 * net.out_size
        if feat_bytes > frame_bytes:
            pixels_in_buffer = True
            buffer_reason = (
                f"pixels (auto: a {net.out_size}-dim float32 feature "
                f"takes {feat_bytes / frame_bytes:.1f}x the memory of a "
                f"{frame_side}px frame; every update re-encodes the batch)"
            )
    return EncoderStack(net, aug, strong, pixels_in_buffer, buffer_reason, trainable, bn_ref)


class Encoding:
    """Encode helpers for the pixel path, in three steps so augmentation sits
    between preprocessing and the encoder (at the encoder's input resolution)
    and a view can be encoded more than once without re-drawing it (see
    src/views.py)."""

    def __init__(self, stack: EncoderStack, device, amp_bf16: bool):
        self.net, self.aug, self.strong = stack.net, stack.aug, stack.strong
        self.device = device
        self.amp = amp_bf16 and device.type == "cuda"
        self._grad_contract_checked = False

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp)

    def preprocess(self, obs_batch) -> torch.Tensor:
        """(B, H, W, C) uint8 (tensor or ndarray) -> float (B, C, h, w) in
        [0, 1] at the encoder's input resolution. Same layout/dtype contract
        as FrozenEncoderWrapper."""
        x = (
            torch.as_tensor(np.ascontiguousarray(obs_batch))
            if not torch.is_tensor(obs_batch)
            else obs_batch
        )
        x = x.to(self.device).permute(0, 3, 1, 2)
        with torch.no_grad(), self._autocast():
            return self.net.preprocess(x)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Replay-batch views only -- never rollout-time action selection or
        eval(), where the encoder should see the real observation. Runs in
        fp32, outside autocast: it's cheap next to the encoder, and under
        autocast kornia's convolutions (e.g. blur) would compute the
        augmented view in bf16 -- the output is still fp32, but rounded by
        up to ~2 uint8 levels while the clean view stays exact."""
        with torch.no_grad():
            return self.aug(x)

    def strong_augment(self, x: torch.Tensor) -> torch.Tensor:
        """The strong view (augmentation.strong), applied on top of the task
        view; same conventions as augment()."""
        with torch.no_grad():
            return self.strong(x)

    def featurize(self, x: torch.Tensor, grad: bool) -> torch.Tensor:
        """preprocess() output (optionally augmented) -> (B, feat_dim) float32."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, self._autocast():
            feats = self.net.encode_preprocessed(x)
        if grad and not self._grad_contract_checked:
            # If this trips, EmbeddingNet.encode breaks the graph internally
            # (no_grad/inference_mode, a numpy round-trip, or a
            # non-differentiable jit path, e.g. some CLIP torch.jit archives).
            assert (
                torch.is_tensor(feats) and feats.requires_grad
            ), "EmbeddingNet.encode returned a graph-less result under enable_grad"
            self._grad_contract_checked = True
        if not torch.is_tensor(feats):
            feats = torch.as_tensor(feats, device=self.device)
        return feats.float()

    def encode_single(self, obs_np) -> torch.Tensor:
        return self.featurize(self.preprocess(np.asarray(obs_np)[None]), grad=False)

    def check_augmentations(self, obs) -> None:
        """One real frame through preprocess -> augmentation(s) at startup,
        so an input contract violation (value range, channel count,
        non-square frame) fails now rather than at the first update after
        learning_starts. This is also where the one-time range check (and
        random_overlay's pool load) runs."""
        x = None
        for module in (self.aug, self.strong):
            if module is not None:
                x = self.preprocess(np.asarray(obs)[None]) if x is None else x
                with torch.no_grad():
                    module(x)


# ------------------------------------------------------------------------------
# Encoder optimizer and aux losses
# ------------------------------------------------------------------------------

class EncoderOptim:
    """The encoder's own optimizer (encoder.lr, much lower than the heads'),
    its aux losses, and the drift metric. Inert when the encoder doesn't
    train."""

    def __init__(self, cfg, enc, net, device):
        self.cfg = cfg
        self.net = net
        self.optimizer, self.theta0, self.params, self.aux_terms = None, None, [], []
        if enc.trains:
            self.aux_terms = build_aux_losses(enc.aux_specs, net)
            for t in self.aux_terms:
                t.loss.to(device)
            # Learnable heads owned by aux losses train alongside the encoder.
            self.params = list(net.parameters()) + [
                p for t in self.aux_terms for p in t.loss.parameters()
            ]
            # Separate, much lower LR: at the head LRs a resnet18's weight sums
            # drift ~3% in 50 gradient steps (measured) -- far too fast for a
            # pretrained representation you want to adapt, not destroy.
            self.optimizer = optim.Adam(
                self.params, lr=cfg.encoder.lr, betas=tuple(cfg.encoder.betas)
            )
            # Kept on CPU: only read for the drift metric at log time, and a
            # GPU copy would double the encoder's memory (l2sp keeps its own
            # anchor). Backbone only: a trainable head's drift from its random
            # init would dominate the metric and make it incomparable across
            # routings.
            self.theta0 = {
                n: p.detach().cpu().clone() for n, p in net.embedding.named_parameters()
            }
            print(f"Encoder trains: grads={enc.grads}  aux={enc.aux_names}  lr={cfg.encoder.lr}")
        self.aux_grad_clean, self.aux_grad_aug = aux_grad_views(self.aux_terms)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> float:
        """Clip (encoder.grad_clip; null = no clipping) and step; returns the
        pre-clip grad norm."""
        clip = self.cfg.encoder.grad_clip
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            self.params, max_norm=float("inf") if clip is None else clip))
        self.optimizer.step()
        return gnorm

    def drift(self) -> float:
        with torch.no_grad():
            return float(
                sum(
                    ((p.detach().cpu() - self.theta0[n]) ** 2).sum()
                    for n, p in self.net.embedding.named_parameters()
                ).sqrt()
            )


def encode_replay_batch(E: Encoding, stack: EncoderStack, enc, eo: "EncoderOptim",
                        b_obs, b_next_obs, task_grad: bool):
    """Pixel path: one augmentation draw per update, routed by
    augmentation.stage (src/views.py). Returns (views, next_feats,
    feats_strong):
      views         task view features (views.feats_task, with gradient iff
                    task_grad) and what the aux losses read
      next_feats    next_obs in the task's view, with its own independent
                    draw (DrQ-v2 augments obs and next_obs separately); no
                    target encoder: the live encoder under no_grad
      feats_strong  augmentation.strong on top of the task view (critic loss
                    only), or None
    """
    from src.views import encode_views
    views = encode_views(
        E.featurize,
        stack.net,
        E.preprocess(b_obs),
        E.augment if stack.aug is not None else None,
        task_aug=enc.task_aug,
        task_grad=task_grad,
        aux_needs_clean=enc.aux_needs_clean,
        aux_needs_aug=enc.aux_needs_aug,
        aux_grad_clean=eo.aux_grad_clean,
        aux_grad_aug=eo.aux_grad_aug,
    )
    x_next = E.preprocess(b_next_obs)
    if enc.task_aug:
        x_next = E.augment(x_next)
    next_feats = E.featurize(x_next, grad=False)
    feats_strong = None
    if stack.strong is not None:
        feats_strong = E.featurize(E.strong_augment(views.x_task), grad=task_grad)
    return views, next_feats, feats_strong


# ------------------------------------------------------------------------------
# Logging, wandb and checkpoints
# ------------------------------------------------------------------------------

def log_episode(info, step: int, wandb) -> None:
    ep = info.get("episode")
    if ep is None:
        return
    print(f"  step {step:>8,}  episodic_return={float(ep['r']):8.1f}")
    if wandb is not None:
        wandb.log(
            {
                "charts/episodic_return": float(ep["r"]),
                "charts/episodic_length": int(ep["l"]),
                "global_step": step,
            },
            step=step,
        )


def log_eval(stats: dict, step: int, wandb) -> None:
    print(
        f"  step {step:>8,}  EVAL return={stats['return_mean']:.1f}"
        f"±{stats['return_std']:.1f}"
        + (f"  success={stats['success_rate']:.2%}" if "success_rate" in stats else "")
    )
    if wandb is not None:
        log = {
            "eval/mean_reward": stats["return_mean"],
            "eval/std_reward": stats["return_std"],
            "global_step": step,
        }
        if "success_rate" in stats:
            log["eval/success_rate"] = stats["success_rate"]
        wandb.log(log, step=step)


def encoder_log(enc, stack: EncoderStack, eo: "EncoderOptim", collapse, enc_gnorm_critic: float,
                enc_gnorm_actor: float, aux_total, aux_values) -> dict:
    """The encoder's wandb metrics for one log step (same keys in every
    script)."""
    log = {}
    if enc.bn_adapt:
        log["train/bn_stat_drift"] = stack.net.bn_stat_drift(stack.bn_ref)
    if enc.trains:
        log["train/encoder_grad_norm_critic"] = enc_gnorm_critic
        log["train/encoder_grad_norm_actor"] = enc_gnorm_actor
        log["train/encoder_drift_l2"] = eo.drift()
        log["train/embedding_std"] = collapse.current
        log["train/embedding_std_ratio"] = collapse.ratio
        if aux_total is not None:
            log["train/aux_loss"] = aux_total.item()
            for name, value in aux_values.items():
                log[f"train/aux/{name}"] = value
    return log

def init_wandb(cfg, algo: str, enc, stack: EncoderStack):
    """Start the wandb run (after the encoder is built: the buffer decision
    can still change there, and the run name and logged config should
    describe the run that actually happens). Returns the wandb module, or
    None when wandb is off. Run names are labels only; charts are keyed by
    the logged metric names, which stay fixed for continuity."""
    if not cfg.wandb.enabled:
        return None
    import wandb

    net = stack.net
    emb_tag = cfg.embedding.get("name") or "state"
    layer = cfg.embedding.get("layer")
    if layer is not None:
        emb_tag += f"-L{layer}" if isinstance(layer, int) else f"-{layer}"
    if cfg.embedding.get("head") is not None:
        emb_tag += f"-{cfg.embedding.head}"
    if net is not None and net.frame_stack > 1:
        emb_tag += f"-fs{net.frame_stack}{net.frame_fusion}"
    tags = ""
    if enc.trains:
        tags += "_ft-" + "-".join([enc.grads] + enc.aux_names)
    if enc.bn_adapt:
        tags += "_bnadapt"
    if enc.aug_name is not None:
        tags += f"_aug-{enc.aug_name}" + (f"-{enc.aug_stage}" if enc.aug_stage != "rl" else "")
    if enc.strong_name is not None:
        tags += f"_strong-{enc.strong_name}"
    if (stack.pixels_in_buffer and not enc.trains and enc.aug_name is None
            and enc.strong_name is None and not enc.bn_adapt):
        tags += "_slowfrozen"
    run_name = cfg.wandb.run_name or f"{algo}_{cfg.env.name}_{emb_tag}{tags}_s{cfg.seed}"

    config = OmegaConf.to_container(cfg, resolve=True)
    config["buffer_obs_resolved"] = "pixels" if stack.pixels_in_buffer else "features"
    config["buffer_obs_reason"] = stack.buffer_reason
    if net is not None and net.build_warnings:
        # Printed during construction, before wandb started; log them so a
        # reloaded encoder built differently is visible in the run too.
        config["encoder_build_warnings"] = net.build_warnings
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        mode=cfg.wandb.get("mode", "online"),
        config=config,
        save_code=True,
    )
    print(f"wandb run: {wandb.run.url}")
    return wandb


def encoder_record(cfg, net) -> dict:
    """An encoder as saved: weights plus how it was built, so a reload with a
    different layer/head/preprocess/frame size is flagged
    (EmbeddingNet._load_checkpoint). Used as best_encoder.pt, and merged
    into final_checkpoint.pt."""
    return {
        "encoder": net.state_dict(),
        "embedding": OmegaConf.to_container(cfg.embedding, resolve=True),
        "input_size": net.input_size,
    }


def buffer_options(algo_cfg, pixels_in_buffer: bool) -> dict:
    """ReplayBuffer storage options from algo.buffer_* (compression applies
    to pixel buffers only: feature vectors barely compress)."""
    compression = str(algo_cfg.get("buffer_compression") or "none")
    if compression != "none" and not pixels_in_buffer:
        print(f"Note: algo.buffer_compression={compression} ignored: the buffer stores features, not pixels")
        compression = "none"
    if compression == "none":
        return {}
    return {"compression": compression, "compression_level": algo_cfg.get("buffer_compression_level"),
            "decode_threads": int(algo_cfg.get("buffer_decode_threads") or 8),
            "max_gb": algo_cfg.get("buffer_max_gb")}


def buffer_status(rb):
    """(stdout fragment, wandb metrics) for a compressed buffer's size; empty otherwise."""
    if getattr(rb, "compression", "none") == "none":
        return "", {}
    gb, ratio = rb.stored_bytes / 1e9, rb.compression_ratio()
    return (f"  buffer={gb:.1f}GB({ratio:.1f}x)",
            {"charts/buffer_gb": gb, "charts/buffer_compression_ratio": ratio, "charts/buffer_size": len(rb)})

