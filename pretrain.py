"""
Encoder pretraining on video: train any encoder this repo builds
(configs/embedding) with registered objectives (src/losses.py) on a clip
dataset (src/data/clips.py), then use the result in the RL/BC scripts.

    python pretrain.py +experiment=r3m pretrain_data.root=/path/to/ssv2_frames
    python pretrain.py +experiment=r3m embedding=resnet18 batch_size=64 lr=4e-4

Each step: batch_size videos, the frames the objective's FrameSampler picks
from each (R3M: 5) -> scaled to [0, 1] at their stored size
(pretrain_data.frames_as_stored; otherwise through the embedding's own
resize/crop) -> augmented (augmentation; clip_augmentation=shared gives every frame of a
video the same draw) -> encoded -> objectives -> one Adam step over the
encoder and the objectives' own heads. The encoder is in train() mode while
training (BatchNorm uses and updates batch statistics) and eval() mode for
validation batches.

Outputs, in the run directory:
  checkpoint.pt         latest state, saved after every validation batch:
                        the encoder in the format embedding.checkpoint loads
                        (weights + how it was built), plus objectives,
                        optimizer and step, for resuming
  encoder_step<N>.pt    encoder-only snapshots every save.keep_every steps
                        and at the end; they never change once written
  embedding.yaml        an embedding config pointing at the latest snapshot;
                        copy it to configs/embedding/<name>.yaml and run
                        `python train_sac.py embedding=<name>`

The saved encoder keeps the embedding's default input pipeline (e.g.
ResNets: resize 256, center crop 224, ImageNet normalisation), the same one
the released R3M weights get here, so the two compare under identical RL
preprocessing.

Resuming: rerun with the same hydra.run.dir; checkpoint.pt is picked up
(resume=true), every RNG is reseeded from (seed, step) so the data stream
continues instead of repeating, and wandb continues the same run.

Logged (wandb and stdout), averaged over each log_frequency interval:
train/loss, train/<objective> (unweighted), train/<objective>/<metric>
(R3M: the official metric names), train/grad_norm, charts/steps_per_sec,
charts/data_wait_frac (share of wall time spent waiting for the data
loader: near 1 means the loader is the bottleneck); eval/* for validation
batches.
"""

import os
import random
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.augmentations import build_augmentation
from src.data.clips import ClipDataset, ClipManifest
from src.encoder_config import embedding_kwargs
from src.losses import AuxBatch, aux_loss_classes, build_aux_losses, objective_terms
from src.rl.common import encoder_record


def _plain(node):
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def _check_objectives(cfg):
    specs = list(_plain(cfg.objectives) or [])
    if not specs:
        raise ValueError("objectives is empty: pick a method preset (e.g. +experiment=r3m) or set "
                         "objectives='[{name: r3m, weight: 1.0}]'")
    classes = aux_loss_classes(specs)
    not_clip = [s["name"] for s, c in zip(specs, classes) if not c.needs_clips]
    if not_clip:
        raise ValueError(f"objectives {not_clip} don't read video clips; pretrain.py trains clip "
                         "objectives (needs_clips, see src/losses.py)")
    return specs


def _shared_sampler(terms, train: bool):
    """All objectives read the same frames of each video, so they must agree
    on which ones."""
    samplers = [t.loss.clip_sampler(train) for t in terms]
    first = samplers[0]
    for t, s in zip(terms[1:], samplers[1:]):
        if type(s) is not type(first) or vars(s) != vars(first):
            raise ValueError(f"objectives {terms[0].name!r} and {t.name!r} sample different frames "
                             f"({type(first).__name__}{vars(first)} vs {type(s).__name__}{vars(s)})")
    return first


def reseed(seed: int, step: int) -> int:
    """Seed every RNG from (seed, step). Called after resuming, before the data
    loaders start: their workers' seeds come from torch's RNG, so reseeding
    with the run seed alone would replay the opening batches (same videos,
    frames, crops and negatives) after every resume."""
    s = int(np.random.SeedSequence([seed, step]).generate_state(1)[0])
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    return s


def _loader(cfg, data, manifest_file, sampler, num_workers, device, seed):
    from hydra.utils import to_absolute_path
    root = to_absolute_path(str(data.root))
    manifest = ClipManifest(root, manifest_file, min_length=sampler.min_length,
                            text_strip=int(data.get("text_strip") or 0))
    print(f"  {manifest.source}: {len(manifest):,} videos"
          + (f" ({manifest.skipped:,} shorter than {sampler.min_length} frames skipped)"
             if manifest.skipped else ""))
    loader = torch.utils.data.DataLoader(
        ClipDataset(manifest, sampler, seed=seed), batch_size=cfg.batch_size,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
    )
    return iter(loader)


def as_stored(frames, size: int):
    """uint8 frames -> float [0, 1], as ConvertImageDtype would, after checking
    they really are pretrain_data.image_size (the startup check only compares
    the config with the encoder; a ResNet would silently train at any size)."""
    if tuple(frames.shape[-2:]) != (size, size):
        raise ValueError(f"frames are {tuple(frames.shape[-2:])}px on disk, but "
                         f"pretrain_data.image_size is {size}: set it to the --size they were prepared at")
    return frames.float().div_(255)


def _save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)       # a job killed mid-save keeps the previous file


@hydra.main(config_path="configs", config_name="config_pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Validate before anything expensive ────────────────────────────────────
    specs = _check_objectives(cfg)
    if cfg.embedding.get("name") is None:
        raise ValueError("pretrain.py needs an encoder: set embedding=<config>")
    emb_kwargs = embedding_kwargs(cfg)
    if emb_kwargs.get("frame_stack", 1) > 1:
        raise ValueError("embedding.frame_stack applies to RL observations; pretraining reads "
                         "single frames (objectives pick several per video themselves)")
    if cfg.clip_augmentation not in ("shared", "per_frame"):
        raise ValueError(f"clip_augmentation={cfg.clip_augmentation!r}; expected shared or per_frame")
    data = cfg.pretrain_data
    if OmegaConf.is_missing(data, "root"):
        raise ValueError("pretrain_data.root must be set (the prepared dataset's folder)")
    from src.embeddings import CHECKPOINT_ARCHS
    arch = emb_kwargs.get("arch") or str(cfg.embedding.name)
    if arch not in CHECKPOINT_ARCHS:
        raise ValueError(
            f"pretrain.py can't train arch={arch!r}: the encoder it saves has to load back through "
            f"embedding.checkpoint, which supports {sorted(CHECKPOINT_ARCHS)}. To continue from "
            "released weights, name the architecture and the file, e.g. embedding.arch=resnet50 "
            "embedding.checkpoint=~/.r3m/r3m_50/model.pt")
    keep = cfg.save.get("keep_every")
    if keep and keep % cfg.eval_frequency:
        raise ValueError(f"save.keep_every ({keep}) must be a multiple of eval_frequency "
                         f"({cfg.eval_frequency}): snapshots are taken when checkpoint.pt is saved")
    frames_as_stored = bool(data.get("frames_as_stored", False))

    # ── Encoder, objectives, optimizer ────────────────────────────────────────
    from src.embeddings import EmbeddingNet
    encoder = EmbeddingNet(
        cfg.embedding.name, pretrained=bool(cfg.embedding.get("pretrained", True)), train=True,
        disable_cuda=device.type == "cpu", model_dir=cfg.model_dir,
        input_size=int(data.image_size), **emb_kwargs,
    )
    encoder.set_trainable(cfg.trainable)
    if frames_as_stored and int(data.image_size) != encoder.in_shape[-1]:
        raise ValueError(
            f"pretrain_data.frames_as_stored: the frames are {data.image_size}px, but "
            f"{cfg.embedding.name!r} takes {encoder.in_shape[-1]}px input. Prepare the data at that "
            "size, or set frames_as_stored=false to run the encoder's own resize/crop")
    terms = build_aux_losses(specs, encoder)
    for t in terms:
        t.loss.to(device)
    params = [p for p in encoder.parameters() if p.requires_grad] + \
             [p for t in terms for p in t.loss.parameters()]
    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_obj = sum(p.numel() for t in terms for p in t.loss.parameters())
    print(f"\nEncoder: {cfg.embedding.name} (arch={encoder.arch}, pretrained="
          f"{cfg.embedding.get('pretrained', True)}) -> {encoder.out_size} dims, "
          f"{n_enc / 1e6:.1f}M trainable  |  objectives: {[t.name for t in terms]}, "
          f"{n_obj / 1e6:.1f}M head parameters")
    optimizer = torch.optim.Adam(params, lr=cfg.lr, betas=tuple(cfg.betas),
                                 weight_decay=cfg.weight_decay)
    grad_clip = float("inf") if cfg.grad_clip is None else float(cfg.grad_clip)

    aug = build_augmentation(cfg.augmentation)
    if aug is not None:
        aug = aug.to(device)
        print(f"Augmentation: {cfg.augmentation.name} params={dict(cfg.augmentation.get('params') or {})} "
              f"({cfg.clip_augmentation})")

    # ── Run directory: resume, wandb, embedding.yaml ──────────────────────────
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    step, wandb_id = 0, None
    if cfg.resume and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        for t in terms:
            t.loss.load_state_dict(ckpt["objectives"][t.name])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, wandb_id = int(ckpt["step"]), ckpt.get("wandb_id")
        print(f"Resumed from {ckpt_path} at step {step:,}")

    # ── Data (after resuming: seeded from the step) ───────────────────────────
    seed = reseed(cfg.seed, step)
    print("Data:")
    train_iter = _loader(cfg, data, data.train_manifest, _shared_sampler(terms, True),
                         int(cfg.num_workers), device, seed)
    val_iter = _loader(cfg, data, data.get("val_manifest") or data.train_manifest,
                       _shared_sampler(terms, False), min(2, int(cfg.num_workers)), device, seed + 1)

    names = "+".join(t.name for t in terms)
    label = f"{names}_{data.name}_{cfg.embedding.name}"
    wandb = None
    if cfg.wandb.enabled:
        import wandb
        aug_tag = f"_aug-{cfg.augmentation.name}" if aug is not None else ""
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                   name=cfg.wandb.run_name or f"{label}{aug_tag}_s{cfg.seed}",
                   config=OmegaConf.to_container(cfg, resolve=True),
                   mode=cfg.wandb.get("mode", "online"), id=wandb_id, resume="allow",
                   save_code=True)
        wandb_id = wandb.run.id
        print(f"wandb run: {wandb.run.url}")

    def write_embedding_yaml(snapshot):
        emb_cfg = OmegaConf.to_container(cfg.embedding, resolve=True)
        for key in ("pretrained", "train"):      # about building from scratch; the checkpoint replaces it
            emb_cfg.pop(key, None)
        emb_cfg.update(name=label, arch=encoder.arch, checkpoint=snapshot)
        with open(os.path.join(run_dir, "embedding.yaml"), "w") as f:
            f.write(f"# Encoder from pretrain.py ({run_dir}). Copy to configs/embedding/{label}.yaml,\n"
                    f"# then: python train_sac.py embedding={label}\n")
            f.write(OmegaConf.to_yaml(OmegaConf.create(emb_cfg)))

    def save(step, final=False):
        if not cfg.save.enabled:
            return
        record = encoder_record(cfg, encoder)
        record["step"] = step
        _save({**record, "objectives": {t.name: t.loss.state_dict() for t in terms},
               "optimizer": optimizer.state_dict(), "wandb_id": wandb_id,
               "config": OmegaConf.to_container(cfg, resolve=True)}, ckpt_path)
        keep = cfg.save.get("keep_every")
        if final or (keep and step % keep == 0):
            snapshot = os.path.join(run_dir, f"encoder_step{step:07d}.pt")
            _save(record, snapshot)
            write_embedding_yaml(snapshot)

    # ── Training loop ─────────────────────────────────────────────────────────
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(cfg.perf.amp_bf16))

    def encode(frames, augment):
        """uint8 (B, T, 3, H, W) -> features (B, T, D)."""
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:]).to(device, non_blocking=True)
        # As stored: only the encoder's normalisation applies (in
        # encode_preprocessed); /255 is what its ConvertImageDtype step does.
        x = as_stored(x, int(data.image_size)) if frames_as_stored else encoder.preprocess(x)
        if augment and aug is not None:
            if cfg.clip_augmentation == "shared":
                x = aug(x.reshape(b, t * x.shape[1], *x.shape[2:])).reshape(x.shape)
            else:
                x = aug(x)
        with autocast:
            feats = encoder.encode_preprocessed(x)
        return feats.float().reshape(b, t, -1)

    def run_objectives(frames, texts, augment):
        feats = encode(frames, augment)
        return objective_terms(terms, AuxBatch(encoder=encoder, clip_feats=feats, text=list(texts)))

    def fmt(values, keys):
        return "  ".join(f"{k.split('/')[-1]}={values[k]:.4f}" for k in keys if k in values)

    print(f"\nTraining {cfg.train_steps:,} steps of {cfg.batch_size} videos"
          + (f" (from step {step:,})" if step else ""))
    sums, n_sum, data_time, t_log = {}, 0, 0.0, time.time()
    while step < cfg.train_steps:
        encoder.train()
        for t in terms:
            t.loss.train()
        t0 = time.time()
        frames, texts = next(train_iter)
        data_time += time.time() - t0

        total, values = run_objectives(frames, texts, augment=True)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        values["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()
        values["loss"] = total.detach()
        step += 1
        for k, v in values.items():
            sums[k] = sums.get(k, 0.0) + v
        n_sum += 1

        if step % cfg.log_frequency == 0 or step == cfg.train_steps:
            elapsed = time.time() - t_log
            avg = {k: float(v) / n_sum for k, v in sums.items()}
            sps, wait = n_sum / elapsed, data_time / elapsed
            print(f"  step {step:>9,}  loss={avg['loss']:.4f}  {fmt(avg, sorted(k for k in avg if '/' in k))}"
                  f"  steps/s={sps:.1f}  data_wait={wait:.0%}")
            if wandb is not None:
                log = {f"train/{k}": v for k, v in avg.items()}
                log.update({"charts/steps_per_sec": sps, "charts/data_wait_frac": wait,
                            "global_step": step})
                wandb.log(log, step=step)
            sums, n_sum, data_time, t_log = {}, 0, 0.0, time.time()

        if step % cfg.eval_frequency == 0 or step == cfg.train_steps:
            t_eval = time.time()
            encoder.eval()
            for t in terms:
                t.loss.eval()
            frames, texts = next(val_iter)
            with torch.no_grad():
                total, values = run_objectives(frames, texts, augment=False)
            values["loss"] = total
            ev = {k: float(v) for k, v in values.items()}
            print(f"  step {step:>9,}  EVAL loss={ev['loss']:.4f}  {fmt(ev, sorted(k for k in ev if '/' in k))}")
            if wandb is not None:
                wandb.log({f"eval/{k}": v for k, v in ev.items()}, step=step)
            save(step, final=step == cfg.train_steps)
            t_log += time.time() - t_eval      # keep steps/s about training

    if cfg.save.enabled:
        print(f"\nSaved to: {run_dir}  (embedding.yaml: how to use the encoder)")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
