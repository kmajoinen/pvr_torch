"""
Encoder pretraining: train any encoder this repo builds (configs/embedding)
with registered objectives (src/losses.py) on a video-clip or image dataset,
then use the result in the RL/BC scripts.

    python pretrain.py +experiment=r3m pretrain_data.root=/path/to/ssv2
    python pretrain.py +experiment=imagenet_supervised pretrain_data.root=/path/to/imagenet
    python pretrain.py +experiment=moco_v2 embedding=resnet18 pretrain_data.root=...

Two kinds of data (pretrain_data.type), each with its own objectives:
  clips   (R3M) batch_size videos, the frames the objective's FrameSampler
          picks from each -> [0, 1] at their stored size
          (pretrain_data.frames_as_stored; otherwise the embedding's own
          resize/crop) -> augmented (clip_augmentation=shared: the same draw
          on every frame of a video) -> encoded -> objectives
  images  (supervised, MoCo) batch_size images, each cropped to
          pretrain_data.image_size in the loader workers (one random
          resized crop per view the objective needs) -> [0, 1] ->
          augmented, each view with its own draws -> the objectives, which
          encode the views themselves
Then one optimizer step (optimizer: adam | sgd, lr_schedule) over the
encoder and the objectives' own heads. The encoder is in train() mode while
training (BatchNorm uses and updates batch statistics) and eval() mode for
validation. Image runs can be sized in epochs (epochs, eval_every_epochs,
save.keep_every_epochs); validation is then a full pass over the validation
split (pretrain_data.eval_batches limits it).

Outputs, in the run directory:
  checkpoint.pt         latest state, saved after every validation:
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
the released weights get here, so they compare under identical RL
preprocessing.

Resuming: rerun with the same hydra.run.dir; checkpoint.pt is picked up
(resume=true). Clip data is reseeded from (seed, step) so the stream
continues instead of repeating; image data continues at the exact position
in its epoch (same batch_size and num_workers). wandb continues the run.

Logged (wandb and stdout), averaged over each log_frequency interval:
train/loss, train/<objective> (unweighted), train/<objective>/<metric>
(R3M: the official metric names; supervised: top1/top5; MoCo: acc1),
train/grad_norm, train/lr, train/epoch (images), charts/steps_per_sec,
charts/data_wait_frac (share of wall time spent waiting for the data
loader: near 1 means the loader is the bottleneck); eval/* for validation.
"""

import math
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


def _check_objectives(cfg, data_type: str):
    """The objective specs, after checking they all read the kind of data
    pretrain_data provides."""
    specs = list(_plain(cfg.objectives) or [])
    if not specs:
        raise ValueError("objectives is empty: pick a method preset (e.g. +experiment=r3m) or set "
                         "objectives='[{name: r3m, weight: 1.0}]'")
    classes = aux_loss_classes(specs)
    reads = {"clips": lambda c: c.needs_clips, "images": lambda c: c.needs_views > 0}
    if data_type not in reads:
        raise ValueError(f"pretrain_data.type={data_type!r}; expected clips or images")
    wrong = [s["name"] for s, c in zip(specs, classes) if not reads[data_type](c)]
    if wrong:
        raise ValueError(f"objectives {wrong} don't read {data_type} (pretrain_data.type), "
                         "see their needs_* in src/losses.py")
    return specs


def _shared_sampler(terms, train: bool):
    """All clip objectives read the same frames of each video, so they must
    agree on which ones."""
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


def _data_loader(dataset, cfg, num_workers, device, persistent=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=persistent and num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def _loader(cfg, data, manifest_file, sampler, num_workers, device, seed):
    from hydra.utils import to_absolute_path
    root = to_absolute_path(str(data.root))
    manifest = ClipManifest(root, manifest_file, min_length=sampler.min_length,
                            text_strip=int(data.get("text_strip") or 0))
    print(f"  {manifest.source}: {len(manifest):,} videos"
          + (f" ({manifest.skipped:,} shorter than {sampler.min_length} frames skipped)"
             if manifest.skipped else ""))
    return iter(_data_loader(ClipDataset(manifest, sampler, seed=seed), cfg, num_workers, device))


def _image_dataset(cfg, data, split, train: bool, views: int, seed: int, start_batch: int = 0):
    from hydra.utils import to_absolute_path
    from src.data.images import ImageDataset
    crop = data.get("train_crop") or {}
    every = data.get("every") or 1
    if not train:
        every = data.get("val_every") or every
    return ImageDataset(
        to_absolute_path(str(data.root)), split, train=train, size=int(data.image_size), views=views,
        scale=tuple(crop.get("scale", (0.08, 1.0))), ratio=tuple(crop.get("ratio", (3 / 4, 4 / 3))),
        resize=int(data.get("eval_resize") or 256), seed=seed, batch_size=int(cfg.batch_size),
        start_batch=start_batch, every=int(every), interpolation=str(data.get("interpolation") or "bilinear"))


def scale_to_images(epochs: float, sched: dict, n_images: int, train_images: float):
    """cfg.train_images: stretch an epoch-sized run over n_images images to
    train_images images in total. Epochs and the lr schedule's lengths in
    epochs (warmup, milestones, total) scale by the same factor, so the
    schedule keeps its shape (MAE's warmup stays 2.5% of a 1600-epoch run);
    validation and snapshot intervals stay as they are. Returns (epochs,
    schedule)."""
    factor = float(train_images) / (n_images * float(epochs))
    sched = dict(sched)
    for key in ("warmup", "total"):
        if sched.get(key):
            sched[key] = float(sched[key]) * factor
    if sched.get("milestones"):
        sched["milestones"] = [float(m) * factor for m in sched["milestones"]]
    return float(epochs) * factor, sched


def build_optimizer(cfg, named_params, lr: float):
    """cfg.optimizer over (name, parameter) pairs: adam | adamw | sgd.
    no_weight_decay_1d: biases and other 1-D parameters (norm weights) get no
    weight decay, as in MAE (timm's add_weight_decay)."""
    named_params = list(named_params)
    wd = float(cfg.weight_decay)
    if cfg.get("no_weight_decay_1d") and wd:
        decay = [p for n, p in named_params if p.ndim > 1 and not n.endswith(".bias")]
        no_decay = [p for n, p in named_params if p.ndim <= 1 or n.endswith(".bias")]
        groups = [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": wd}]
    else:
        groups = [{"params": [p for _, p in named_params], "weight_decay": wd}]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=float(cfg.momentum),
                               nesterov=bool(cfg.get("nesterov", False)))
    opt = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    return opt(groups, lr=lr, betas=tuple(cfg.betas))


def final_crop(x, spec):
    """cfg.final_crop on a float (N, C, H, W) batch: resize the shorter side to
    spec.resize, then center-crop spec.crop (either may be null)."""
    from torchvision.transforms import v2
    if spec.get("resize") is not None:
        x = v2.functional.resize(x, int(spec.resize), antialias=True)
    if spec.get("crop") is not None:
        x = v2.functional.center_crop(x, int(spec.crop))
    return x


def lr_factor(sched, step: int, steps_per_epoch=None) -> float:
    """Multiplier on the base learning rate. With steps_per_epoch (an epoch-
    sized image run), milestones and warmup are in epochs and cosine/step
    decay change once per epoch, as in the torchvision and MoCo recipes;
    otherwise everything is in steps."""
    name = str(sched.get("name", "constant"))
    unit = steps_per_epoch or 1
    t = step // unit                                         # epoch (or step) index
    warmup = float(sched.get("warmup", 0) or 0)
    if name == "warmup_cosine":
        # MAE's (util/lr_sched.py): updated every step, in fractional epochs;
        # linear warmup from 0, then a half cosine over the rest of the run.
        e, total = step / unit, float(sched.total)
        if e < warmup:
            return e / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(e - warmup, total - warmup) / (total - warmup)))
    if warmup and step < warmup * unit:
        return (step + 1) / (warmup * unit)
    if name == "constant":
        return 1.0
    if name == "cosine":
        total = float(sched.total)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, total) / total))
    if name == "step":
        gamma = float(sched.get("gamma", 0.1))
        return gamma ** sum(t >= m for m in sched.milestones)
    raise ValueError(f"lr_schedule.name={name!r}; expected constant, cosine, step or warmup_cosine")


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
    data = cfg.pretrain_data
    data_type = str(data.get("type", "clips"))
    images = data_type == "images"
    specs = _check_objectives(cfg, data_type)
    if cfg.embedding.get("name") is None:
        raise ValueError("pretrain.py needs an encoder: set embedding=<config>")
    emb_kwargs = embedding_kwargs(cfg)
    if emb_kwargs.get("frame_stack", 1) > 1:
        raise ValueError("embedding.frame_stack applies to RL observations; pretraining reads "
                         "single frames (objectives pick several per video themselves)")
    if cfg.clip_augmentation not in ("shared", "per_frame"):
        raise ValueError(f"clip_augmentation={cfg.clip_augmentation!r}; expected shared or per_frame")
    if OmegaConf.is_missing(data, "root"):
        raise ValueError("pretrain_data.root must be set (the prepared dataset's folder)")
    if cfg.get("epochs") is not None and not images:
        raise ValueError("epochs applies to image datasets; size clip runs with train_steps")
    if cfg.get("train_images") is not None and not (images and cfg.get("epochs")):
        raise ValueError("train_images stretches an epoch-sized image run: it needs image data "
                         "and epochs (clip runs: size them with train_steps)")
    from src.embeddings import CHECKPOINT_ARCHS
    arch = emb_kwargs.get("arch") or str(cfg.embedding.name)
    if arch not in CHECKPOINT_ARCHS:
        raise ValueError(
            f"pretrain.py can't train arch={arch!r}: the encoder it saves has to load back through "
            f"embedding.checkpoint, which supports {sorted(CHECKPOINT_ARCHS)}. To continue from "
            "released weights, name the architecture and the file, e.g. embedding.arch=resnet50 "
            "embedding.checkpoint=~/.r3m/r3m_50/model.pt")
    if cfg.optimizer not in ("adam", "adamw", "sgd"):
        raise ValueError(f"optimizer={cfg.optimizer!r}; expected adam, adamw or sgd")
    if cfg.get("final_crop") is not None and images:
        raise ValueError("final_crop applies to clip data; image crops are set by pretrain_data "
                         "(train_crop, eval_resize, image_size)")
    if images:
        from hydra.utils import to_absolute_path
        from src.data.images import load_classes
        classes = load_classes(to_absolute_path(str(data.root)))
        for s in specs:
            n = s.get("num_classes", 1000)
            if s["name"] == "supervised" and classes and n != len(classes):
                raise ValueError(f"supervised num_classes={n}, but the dataset at {data.root} has "
                                 f"{len(classes)} classes: set objectives.0.num_classes={len(classes)}")
    # Image crops come out of the loader at image_size, so they're always used as stored.
    frames_as_stored = images or bool(data.get("frames_as_stored", False))

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
            f"pretrain_data: the {'images are cropped to' if images else 'frames are'} "
            f"{data.image_size}px, but {cfg.embedding.name!r} takes {encoder.in_shape[-1]}px input. "
            + ("Set pretrain_data.image_size to that." if images else
               "Prepare the data at that size, or set frames_as_stored=false to run the encoder's "
               "own resize/crop"))
    terms = build_aux_losses(specs, encoder)
    for t in terms:
        t.loss.to(device)
    # Trainable parameters only: objectives may hold frozen copies (MoCo's key encoder).
    named = [(n, p) for n, p in encoder.named_parameters() if p.requires_grad] + \
            [(n, p) for t in terms for n, p in t.loss.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_obj = sum(p.numel() for t in terms for p in t.loss.parameters() if p.requires_grad)
    print(f"\nEncoder: {cfg.embedding.name} (arch={encoder.arch}, pretrained="
          f"{cfg.embedding.get('pretrained', True)}) -> {encoder.out_size} dims, "
          f"{n_enc / 1e6:.1f}M trainable  |  objectives: {[t.name for t in terms]}, "
          f"{n_obj / 1e6:.1f}M head parameters")
    base_lr = float(cfg.lr)
    if cfg.get("lr_reference_batch"):        # linear scaling rule
        base_lr *= cfg.batch_size / float(cfg.lr_reference_batch)
    optimizer = build_optimizer(cfg, named, base_lr)
    grad_clip = float("inf") if cfg.grad_clip is None else float(cfg.grad_clip)

    aug = build_augmentation(cfg.augmentation)
    if aug is not None:
        aug = aug.to(device)
        print(f"Augmentation: {cfg.augmentation.name} params={dict(cfg.augmentation.get('params') or {})} "
              + ("(each view its own draws)" if images else f"({cfg.clip_augmentation})"))

    # ── Run directory: resume, then data, wandb ───────────────────────────────
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
        if images:
            old = ckpt.get("config") or {}
            changed = {k: (old.get(k), cfg[k]) for k in ("batch_size", "num_workers") if old.get(k) != cfg[k]}
            if changed:
                print(f"WARNING: {changed} (checkpoint, now) differ: the image stream won't continue "
                      "exactly where it stopped")

    # Data after resuming: clip streams are seeded from the step; image
    # streams skip to the step's position in their epoch.
    seed = reseed(cfg.seed, step)
    print("Data:")
    steps_per_epoch = None
    views = max(t.loss.needs_views for t in terms) if images else 0
    if images:
        train_set = _image_dataset(cfg, data, data.train_split, True, views, cfg.seed, start_batch=step)
        if any(t.loss.needs_labels for t in terms) and not train_set.labelled:
            raise ValueError(f"{[t.name for t in terms if t.loss.needs_labels]} need class labels, but "
                             f"{train_set.dir} has unlabelled images (e.g. video frames)")
        steps_per_epoch = max(1, len(train_set) // cfg.batch_size)
        print(f"  {train_set.dir}: {len(train_set):,} images ({steps_per_epoch:,} steps per epoch), "
              f"{views} view(s), crop scale {train_set.scale}"
              + (f", {train_set.thinning}" if train_set.thinning else ""))
        train_iter = iter(_data_loader(train_set, cfg, int(cfg.num_workers), device))
    else:
        train_iter = _loader(cfg, data, data.train_manifest, _shared_sampler(terms, True),
                             int(cfg.num_workers), device, seed)
        val_iter = _loader(cfg, data, data.get("val_manifest") or data.train_manifest,
                           _shared_sampler(terms, False), min(2, int(cfg.num_workers)), device, seed + 1)

    def in_steps(epochs_key, steps_key):
        """A frequency given in epochs (image runs) or steps."""
        epochs = OmegaConf.select(cfg, epochs_key)
        return round(epochs * steps_per_epoch) if epochs and steps_per_epoch else OmegaConf.select(cfg, steps_key)

    epochs = cfg.get("epochs")
    sched = OmegaConf.to_container(cfg.lr_schedule, resolve=True)
    if cfg.get("train_images"):
        epochs, sched = scale_to_images(epochs, sched, len(train_set), cfg.train_images)
        print(f"  train_images={float(cfg.train_images):,.0f}: {epochs:,.1f} epochs over "
              f"{len(train_set):,} images (lr schedule stretched with it: {sched})")
    train_steps = int(epochs * steps_per_epoch) if epochs else int(cfg.train_steps)
    eval_frequency = int(in_steps("eval_every_epochs", "eval_frequency"))
    keep = in_steps("save.keep_every_epochs", "save.keep_every")
    if keep and keep % eval_frequency:
        raise ValueError(f"snapshots every {keep} steps (save.keep_every / keep_every_epochs) must be a "
                         f"multiple of the validation interval ({eval_frequency} steps): they're taken "
                         "when checkpoint.pt is saved")
    # The schedule counts epochs only in epoch-sized runs; image runs sized
    # with train_steps count steps, like clip runs.
    sched_epoch = steps_per_epoch if epochs else None
    if sched.get("name") in ("cosine", "warmup_cosine") and sched.get("total") is None:
        sched["total"] = epochs if sched_epoch else train_steps
    if sched.get("name") == "warmup_cosine" and float(sched.get("warmup") or 0) >= float(sched["total"]):
        raise ValueError(f"lr_schedule: warmup ({sched.get('warmup')}) must be shorter than the "
                         f"schedule ({sched['total']} {'epochs' if sched_epoch else 'steps'})")
    sched = OmegaConf.create(sched)

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
        if final or (keep and step % keep == 0):
            snapshot = os.path.join(run_dir, f"encoder_step{step:07d}.pt")
            _save(record, snapshot)
            write_embedding_yaml(snapshot)

    # ── Training loop ─────────────────────────────────────────────────────────
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(cfg.perf.amp_bf16))
    size = int(data.image_size)

    def to_input(x, augment):
        """uint8 (N, 3, H, W) on the CPU -> float [0, 1] encoder input, augmented."""
        x = x.to(device, non_blocking=True)
        x = as_stored(x, size) if frames_as_stored else encoder.preprocess(x)
        return aug(x) if augment and aug is not None else x

    def clip_batch(raw, augment):
        frames, texts = raw
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:]).to(device, non_blocking=True)
        # As stored: only the encoder's normalisation applies (in
        # encode_preprocessed); /255 is what its ConvertImageDtype step does.
        x = as_stored(x, size) if frames_as_stored else encoder.preprocess(x)
        if augment and aug is not None:
            if cfg.clip_augmentation == "shared":
                x = aug(x.reshape(b, t * x.shape[1], *x.shape[2:])).reshape(x.shape)
            else:
                x = aug(x)
        if cfg.get("final_crop") is not None:
            x = final_crop(x, cfg.final_crop)
        with autocast:
            feats = encoder.encode_preprocessed(x)
        return AuxBatch(encoder=encoder, clip_feats=feats.float().reshape(b, t, -1), text=list(texts))

    def image_batch(raw, augment):
        imgs, labels = raw                                  # (B, V, 3, S, S), (B,)
        views_in = [to_input(imgs[:, v], augment) for v in range(imgs.shape[1])]
        return AuxBatch(encoder=encoder, views=views_in, labels=labels.to(device, non_blocking=True))

    make_batch = image_batch if images else clip_batch

    def run_objectives(raw, augment):
        batch = make_batch(raw, augment)
        if not images:                 # clip features were encoded under autocast already
            return objective_terms(terms, batch)
        with autocast:                 # image objectives encode their views themselves
            return objective_terms(terms, batch)

    def set_mode(train):
        encoder.train(train)
        for t in terms:
            t.loss.train(train)

    def evaluate():
        """Clips: one validation batch. Images: the validation split (or
        eval_batches of it), metrics weighted by batch size."""
        set_mode(False)
        if not images:
            with torch.no_grad():
                total, values = run_objectives(next(val_iter), augment=False)
            values["loss"] = total
            return {k: float(v) for k, v in values.items()}
        val_set = _image_dataset(cfg, data, data.val_split, False, views, cfg.seed)
        loader = _data_loader(val_set, cfg, int(cfg.num_workers), device, persistent=False)
        sums, n = {}, 0
        limit = data.get("eval_batches")
        with torch.no_grad():
            for i, raw in enumerate(loader):
                if limit is not None and i >= limit:
                    break
                total, values = run_objectives(raw, augment=False)
                values["loss"] = total
                b = len(raw[1])
                for k, v in values.items():
                    sums[k] = sums.get(k, 0.0) + float(v) * b
                n += b
        return {k: v / n for k, v in sums.items()}

    def fmt(values, keys):
        return "  ".join(f"{k.split('/')[-1]}={values[k]:.4f}" for k in keys if k in values)

    unit = "images" if images else "videos"
    print(f"\nTraining {train_steps:,} steps of {cfg.batch_size} {unit}"
          + (f" ({train_steps / steps_per_epoch:.1f} epochs)" if steps_per_epoch else "")
          + (f" (from step {step:,})" if step else "")
          + f", validating every {eval_frequency:,} steps")
    sums, n_sum, data_time, t_log = {}, 0, 0.0, time.time()
    while step < train_steps:
        lr = base_lr * lr_factor(sched, step, sched_epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr
        set_mode(True)
        t0 = time.time()
        raw = next(train_iter)
        data_time += time.time() - t0

        total, values = run_objectives(raw, augment=True)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        values["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()
        values["loss"] = total.detach()
        step += 1
        for k, v in values.items():
            sums[k] = sums.get(k, 0.0) + v
        n_sum += 1

        if step % cfg.log_frequency == 0 or step == train_steps:
            elapsed = time.time() - t_log
            avg = {k: float(v) / n_sum for k, v in sums.items()}
            sps, wait = n_sum / elapsed, data_time / elapsed
            epoch = f"  epoch={step / steps_per_epoch:.2f}" if steps_per_epoch else ""
            print(f"  step {step:>9,}{epoch}  loss={avg['loss']:.4f}  "
                  f"{fmt(avg, sorted(k for k in avg if '/' in k))}  lr={lr:.3g}"
                  f"  steps/s={sps:.1f}  data_wait={wait:.0%}")
            if wandb is not None:
                log = {f"train/{k}": v for k, v in avg.items()}
                log.update({"train/lr": lr, "charts/steps_per_sec": sps, "charts/data_wait_frac": wait,
                            "global_step": step})
                if steps_per_epoch:
                    log["train/epoch"] = step / steps_per_epoch
                wandb.log(log, step=step)
            sums, n_sum, data_time, t_log = {}, 0, 0.0, time.time()

        if step % eval_frequency == 0 or step == train_steps:
            t_eval = time.time()
            ev = evaluate()
            print(f"  step {step:>9,}  EVAL loss={ev['loss']:.4f}  {fmt(ev, sorted(k for k in ev if '/' in k))}")
            if wandb is not None:
                wandb.log({f"eval/{k}": v for k, v in ev.items()}, step=step)
            save(step, final=step == train_steps)
            t_log += time.time() - t_eval      # keep steps/s about training

    if cfg.save.enabled:
        print(f"\nSaved to: {run_dir}  (embedding.yaml: how to use the encoder)")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
