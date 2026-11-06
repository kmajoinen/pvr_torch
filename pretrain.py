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
Then schedule.updates_per_batch optimizer steps on that batch (optimizer:
adam | adamw | sgd, lr_schedule) over the encoder and the objectives' own
heads, each with fresh augmentation draws unless schedule.reaugment is
false. The encoder is in train() mode while training (BatchNorm uses and
updates batch statistics) and eval() mode for validation.

Schedule (schedule.*, src/data/order.py), for either data type:
  sampling  sweep: each item once per epoch, fresh order (clips: each video
            clips_per_video times) | random: uniform with replacement, an
            epoch being epoch_batches batches. Defaults: clips random (R3M),
            images sweep (ImageNet training)
  length    epochs, or train_steps updates, or train_images samples drawn
            (with epochs: epochs and lr schedule stretched to that count)
Frequencies can be given in epochs (eval_every_epochs,
save.keep_every_epochs); image validation is a full pass over the
validation split (pretrain_data.eval_batches limits it).

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

Multi-GPU (image objectives that declare multi_gpu, src/distributed.py): one
process per GPU on one node, batch_size being the total across GPUs:
    python -m torch.distributed.run --standalone --nproc_per_node=4 pretrain.py \
        hydra.run.dir=<a fixed folder> batch_size=1024 ...
Rank 0 validates, saves and logs. num_workers is per GPU.

Resuming: rerun with the same hydra.run.dir; checkpoint.pt is picked up
(resume=true). Checkpoints fall between batches (validation intervals must
be multiples of updates_per_batch). Random clip streams are reseeded from
(seed, step) so they continue instead of repeating; sweep streams and random
image streams continue at the exact item (same batch_size and num_workers).
What's drawn from each item (a clip's frames, crops, augmentation, negatives)
comes from RNGs reseeded on resume, so a resumed run is statistically
equivalent to an uninterrupted one, not bit-identical, and never replays the
same draws. wandb continues the run.

Logged (wandb and stdout), averaged over each log_frequency interval:
train/loss, train/<objective> (unweighted), train/<objective>/<metric>
(R3M: the official metric names; supervised: top1/top5; MoCo: acc1),
train/grad_norm, train/lr, train/epoch (images), charts/steps_per_sec,
charts/data_wait_frac (share of wall time spent waiting for the data
loader: near 1 means the loader is the bottleneck); eval/* for validation.
"""

import time as _time
# Before the heavy imports, counted in startup_seconds. A list, taken once:
# Hydra's -m runs later jobs in this same process, which start their clock in main().
_T_START = [_time.time()]

import math
import os
import random
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src import distributed
from src.augmentations import build_augmentation
from src.data.clips import ClipDataset, ClipManifest
from src.encoder_config import embedding_kwargs
from src.losses import AuxBatch, aux_loss_classes, build_aux_losses, objective_terms
from src.rl.common import encoder_record
from src.timing import timer_from_config


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


def reseed(seed: int, step: int, rank: int = 0) -> int:
    """Seed every RNG from (seed, step). Called after resuming, before the data
    loaders start: their workers' seeds come from torch's RNG, so reseeding
    with the run seed alone would replay the opening batches (same videos,
    frames, crops and negatives) after every resume. With several GPUs each
    process's rank joins in, so they draw different crops and MAE masks (rank
    0's draws are a single-GPU run's)."""
    s = int(np.random.SeedSequence([seed, step] + ([rank] if rank else [])).generate_state(1)[0])
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    return s


def _data_loader(dataset, cfg, num_workers, device, persistent=True, batch_size=None):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size or cfg.batch_size, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=persistent and num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def _clip_dataset(cfg, data, manifest_file, sampler, seed, **order):
    """A ClipDataset over a manifest; `order`: ClipDataset's sampling options."""
    from hydra.utils import to_absolute_path
    root = to_absolute_path(str(data.root))
    manifest = ClipManifest(root, manifest_file, min_length=sampler.min_length,
                            text_strip=int(data.get("text_strip") or 0))
    print(f"  {manifest.source}: {len(manifest):,} videos"
          + (f" ({manifest.skipped:,} shorter than {sampler.min_length} frames skipped)"
             if manifest.skipped else ""))
    return ClipDataset(manifest, sampler, seed=seed, **order)


def _loader(cfg, data, manifest_file, sampler, num_workers, device, seed, **order):
    return iter(_data_loader(_clip_dataset(cfg, data, manifest_file, sampler, seed, **order),
                             cfg, num_workers, device))


def schedule_settings(cfg, images: bool) -> dict:
    """cfg.schedule, validated, with the data type's default sampling (clips:
    random, as R3M; images: sweep, as ImageNet training)."""
    sch = cfg.get("schedule") or {}
    sampling = sch.get("sampling") or ("sweep" if images else "random")
    if sampling not in ("sweep", "random"):
        raise ValueError(f"schedule.sampling={sampling!r}; expected sweep or random")
    def given(key):                                   # 0 is a value (and rejected), unset is 1
        v = sch.get(key)
        return 1 if v is None else v
    out = {"sampling": sampling, "epoch_batches": sch.get("epoch_batches"),
           "clips_per_video": given("clips_per_video"), "updates_per_batch": given("updates_per_batch"),
           "reaugment": sch.get("reaugment", True)}
    checked = ("clips_per_video", "updates_per_batch") + (("epoch_batches",) if out["epoch_batches"] is not None else ())
    for key in checked:
        v = out[key]
        if isinstance(v, bool) or int(v) != v or v < 1:
            raise ValueError(f"schedule.{key} must be a whole number >= 1, got {v!r}")
        out[key] = int(v)
    if out["clips_per_video"] != 1 and (images or sampling != "sweep"):
        raise ValueError("schedule.clips_per_video applies to clip data with sampling=sweep (draws per "
                         "video per epoch)")
    if out["epoch_batches"] is not None and sampling != "random":
        raise ValueError("schedule.epoch_batches sets the length of a random-sampling epoch; with "
                         "sampling=sweep an epoch is the whole dataset")
    if not isinstance(out["reaugment"], bool):
        raise ValueError(f"schedule.reaugment must be true or false, got {out['reaugment']!r}")
    return out


def _image_dataset(cfg, data, split, train: bool, views: int, seed: int, start_batch: int = 0,
                   order: str = "sweep", batch_size=None, rank: int = 0, world: int = 1):
    from hydra.utils import to_absolute_path
    from src.data.images import ImageDataset
    crop = data.get("train_crop") or {}
    every = data.get("every") or 1
    if not train:
        every = data.get("val_every") or every
    return ImageDataset(
        to_absolute_path(str(data.root)), split, train=train, size=int(data.image_size), views=views,
        scale=tuple(crop.get("scale", (0.08, 1.0))), ratio=tuple(crop.get("ratio", (3 / 4, 4 / 3))),
        resize=int(data.get("eval_resize") or 256), seed=seed, batch_size=int(batch_size or cfg.batch_size),
        start_batch=start_batch, every=int(every), interpolation=str(data.get("interpolation") or "bilinear"),
        order=order, rank=rank, world=world)


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


def resume_changes(old_config, cfg, sch: dict, images: bool) -> dict:
    """What differs from a checkpoint's config in a way that moves the data
    stream, as {name: (checkpoint, now)}. Compares resolved schedules, so a
    checkpoint from before schedule.* existed, or `sampling: sweep` written
    out where it was the default, doesn't count as a change. A random clip
    stream has no position to lose: it only cares if the sampling changed."""
    if not old_config:
        return {}
    old_sch = schedule_settings(old_config, images)
    # Only a sweep splits its order by worker; a random batch's items don't depend on the worker count.
    keys = ("batch_size",) if sch["sampling"] == old_sch["sampling"] == "random" else ("batch_size", "num_workers")
    changed = {k: (old_config.get(k), cfg[k]) for k in keys if old_config.get(k) != cfg[k]}
    changed.update({f"schedule.{k}": (old_sch[k], v) for k, v in sch.items() if old_sch[k] != v})
    keeps_position = images or sch["sampling"] == "sweep" or "schedule.sampling" in changed
    return changed if keeps_position else {}


def epochs_to_steps(epochs: float, steps_per_epoch: int, updates_per_batch: int) -> int:
    """A frequency in epochs as steps: the nearest whole number of batches, so
    it falls between batches however fractional the epoch count."""
    return max(updates_per_batch, round(epochs * steps_per_epoch / updates_per_batch) * updates_per_batch)


def plan_run(cfg, sch: dict, n_items: int) -> dict:
    """The run's length from cfg and the schedule (schedule_settings), for a
    training set of n_items samples per sweep (images, or clips: videos x
    clips_per_video):
      batches_per_epoch  a sweep's whole dataset, or schedule.epoch_batches
                         random batches (default: the same number of samples)
      steps_per_epoch    batches_per_epoch x updates_per_batch
      train_steps        epochs x steps_per_epoch; else train_images samples
                         drawn (ceil(train_images / batch_size) batches x
                         updates_per_batch); else train_steps
      epochs, lr_schedule  with train_images and epochs, both stretched to
                         the image count (scale_to_images)
      samples_per_epoch  what train_images counts against"""
    B, K = int(cfg.batch_size), sch["updates_per_batch"]
    batches_per_epoch = sch["epoch_batches"] if sch["epoch_batches"] else max(1, n_items // B)
    samples_per_epoch = batches_per_epoch * B if sch["epoch_batches"] else n_items
    epochs = cfg.get("epochs")
    sched = OmegaConf.to_container(cfg.lr_schedule, resolve=True)
    if cfg.get("train_images") and epochs:
        epochs, sched = scale_to_images(epochs, sched, samples_per_epoch, cfg.train_images)
    if epochs:
        train_steps = int(epochs * batches_per_epoch * K)
    elif cfg.get("train_images"):
        train_steps = math.ceil(float(cfg.train_images) / B) * K
    else:
        train_steps = int(cfg.train_steps)
    return {"batches_per_epoch": batches_per_epoch, "steps_per_epoch": batches_per_epoch * K,
            "samples_per_epoch": samples_per_epoch, "epochs": epochs, "train_steps": train_steps,
            "lr_schedule": sched}


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
    t_script = _T_START.pop() if _T_START else time.time()
    procs = distributed.setup(str(cfg.device))
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark
    device = distributed.device_of(procs, str(cfg.device))
    print(f"\nDevice: {device}" + (f" (one of {procs.world} processes)" if procs.active else ""))

    # ── Validate before anything expensive ────────────────────────────────────
    data = cfg.pretrain_data
    data_type = str(data.get("type", "clips"))
    images = data_type == "images"
    specs = _check_objectives(cfg, data_type)
    local_batch = int(cfg.batch_size)
    if procs.active:
        single_gpu = [s["name"] for s, c in zip(specs, aux_loss_classes(specs)) if not c.multi_gpu]
        if single_gpu:
            raise ValueError(f"objectives {single_gpu} run on one GPU for now: they use the whole batch "
                             "together (negatives, queues), which needs gathering across GPUs. "
                             "Multi-GPU works for: supervised, mae")
        if local_batch % procs.world:
            raise ValueError(f"batch_size={local_batch} is the total across GPUs, and doesn't divide "
                             f"evenly over {procs.world}")
        local_batch //= procs.world
        distributed.check_same(procs, "the run directory (set hydra.run.dir to a fixed folder)",
                               hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        print(f"Multi-GPU: {procs.world} processes, batch_size {cfg.batch_size} = {local_batch} per GPU, "
              f"num_workers {cfg.num_workers} per GPU")
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
    sch = schedule_settings(cfg, images)
    K = sch["updates_per_batch"]
    if cfg.get("train_images") is not None and not float(cfg.train_images) > 0:
        raise ValueError(f"train_images must be > 0, got {cfg.train_images!r}")
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
        changed = resume_changes(ckpt.get("config"), cfg, sch, images)
        if ckpt.get("world_size", 1) != procs.world:
            changed["GPUs"] = (ckpt.get("world_size", 1), procs.world)
        if changed:
            print(f"WARNING: {changed} (checkpoint, now) differ: the run won't continue at the item "
                  "it stopped at")

    # Data after resuming. A checkpoint lies on a batch boundary (validation
    # runs only there), so step // K batches have been drawn. Random clip
    # streams are reseeded from the step; sweep streams and random image
    # streams skip to the exact item (per-item draws are reseeded either way).
    seed = reseed(cfg.seed, step, procs.rank)
    start_batch = step // K
    print("Data:")
    views = max(t.loss.needs_views for t in terms) if images else 0
    if images:
        train_set = _image_dataset(cfg, data, data.train_split, True, views, cfg.seed,
                                   start_batch=start_batch, order=sch["sampling"],
                                   batch_size=local_batch, rank=procs.rank, world=procs.world)
        if any(t.loss.needs_labels for t in terms) and not train_set.labelled:
            raise ValueError(f"{[t.name for t in terms if t.loss.needs_labels]} need class labels, but "
                             f"{train_set.dir} has unlabelled images (e.g. video frames)")
        n_items = len(train_set)
        print(f"  {train_set.dir}: {len(train_set):,} images, {views} view(s), crop scale {train_set.scale}"
              + (f", {train_set.thinning}" if train_set.thinning else ""))
        train_iter = iter(_data_loader(train_set, cfg, int(cfg.num_workers), device, batch_size=local_batch))
    else:
        clips = _clip_dataset(cfg, data, data.train_manifest, _shared_sampler(terms, True), seed,
                              order=sch["sampling"], clips_per_video=sch["clips_per_video"],
                              order_seed=int(cfg.seed), batch_size=int(cfg.batch_size), start_batch=start_batch)
        n_items = clips.items_per_epoch
        train_iter = iter(_data_loader(clips, cfg, int(cfg.num_workers), device))
        val_iter = _loader(cfg, data, data.get("val_manifest") or data.train_manifest,
                           _shared_sampler(terms, False), min(2, int(cfg.num_workers)), device, seed + 1)

    plan = plan_run(cfg, sch, n_items)
    batches_per_epoch, steps_per_epoch = plan["batches_per_epoch"], plan["steps_per_epoch"]
    epochs, train_steps, sched = plan["epochs"], plan["train_steps"], plan["lr_schedule"]
    item = "images" if images else ("clips" if sch["sampling"] == "sweep" else "videos")
    print(f"  schedule: {sch['sampling']} sampling, an epoch = {batches_per_epoch:,} batches of "
          f"{cfg.batch_size} {item}, {K} update{'s' if K > 1 else ''} per batch"
          + (" (fresh augmentation each)" if K > 1 and sch["reaugment"] else
             " (the same augmented views)" if K > 1 else ""))
    if cfg.get("train_images") and cfg.get("epochs"):
        print(f"  train_images={float(cfg.train_images):,.0f}: {epochs:,.1f} epochs of "
              f"{plan['samples_per_epoch']:,} {item} (lr schedule stretched with it: {sched})")
    if K > 1:
        reused = [t.name for t in terms if getattr(t.loss, "assumes_fresh_batches", False)]
        if reused:
            print(f"WARNING: {reused} assume each batch is used once (schedule.updates_per_batch={K}); see "
                  "their docstrings")

    def in_steps(epochs_key, steps_key):
        """A frequency given in epochs or steps."""
        epochs = OmegaConf.select(cfg, epochs_key)
        return epochs_to_steps(epochs, steps_per_epoch, K) if epochs else OmegaConf.select(cfg, steps_key)

    eval_frequency = int(in_steps("eval_every_epochs", "eval_frequency"))
    keep = in_steps("save.keep_every_epochs", "save.keep_every")
    if keep and keep % eval_frequency:
        raise ValueError(f"snapshots every {keep} steps (save.keep_every / keep_every_epochs) must be a "
                         f"multiple of the validation interval ({eval_frequency} steps): they're taken "
                         "when checkpoint.pt is saved")
    if eval_frequency % K:
        raise ValueError(f"validation every {eval_frequency} steps isn't a multiple of "
                         f"schedule.updates_per_batch={K}: checkpoints are saved at validation and must "
                         "fall between batches, so a resumed run continues at the exact item")
    # The schedule counts epochs only in epoch-sized runs; runs sized in
    # steps or images (without epochs) count steps.
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
    if cfg.wandb.enabled and (cfg.get("timing") or {}).get("benchmark_steps"):
        print("wandb: off in benchmark mode (the measurements go to stdout and timing_benchmark.json)")
    elif cfg.wandb.enabled and procs.is_main:
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

    def checkpoint(step):
        record = encoder_record(cfg, encoder)
        record["step"] = step
        return record, {**record, "objectives": {t.name: t.loss.state_dict() for t in terms},
                        "optimizer": optimizer.state_dict(), "wandb_id": wandb_id,
                        "world_size": procs.world, "config": OmegaConf.to_container(cfg, resolve=True)}

    def save(step, final=False):
        if not cfg.save.enabled:
            return
        record, full = checkpoint(step)
        _save(full, ckpt_path)
        if final or (keep and step % keep == 0):
            snapshot = os.path.join(run_dir, f"encoder_step{step:07d}.pt")
            _save(record, snapshot)
            write_embedding_yaml(snapshot)

    # ── Training loop ─────────────────────────────────────────────────────────
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(cfg.perf.amp_bf16))
    size = int(data.image_size)

    def stage(raw):
        """A loader batch on the device as [0, 1] encoder input, before
        augmentation: the part an update reusing the batch doesn't redo."""
        if images:
            imgs, labels = raw                                  # (B, V, 3, S, S), (B,)
            return [as_stored(imgs[:, v].to(device, non_blocking=True), size) for v in range(imgs.shape[1])], \
                labels.to(device, non_blocking=True)
        frames, texts = raw
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:]).to(device, non_blocking=True)
        # As stored: only the encoder's normalisation applies (in
        # encode_preprocessed); /255 is what its ConvertImageDtype step does.
        x = as_stored(x, size) if frames_as_stored else encoder.preprocess(x)
        return x, (b, t, list(texts))

    def augmented(staged, augment):
        """Images: each view its own draws. Clips: one draw per video
        (clip_augmentation=shared) or per frame, then final_crop."""
        if images:
            views_in, labels = staged
            return [aug(v) if augment and aug is not None else v for v in views_in], labels
        x, meta = staged
        b, t, _ = meta
        if augment and aug is not None:
            if cfg.clip_augmentation == "shared":
                x = aug(x.reshape(b, t * x.shape[1], *x.shape[2:])).reshape(x.shape)
            else:
                x = aug(x)
        if cfg.get("final_crop") is not None:
            x = final_crop(x, cfg.final_crop)
        return x, meta

    def objectives_on(inputs):
        if images:                     # image objectives encode their views themselves
            views_in, labels = inputs
            with autocast:
                return objective_terms(terms, AuxBatch(encoder=encoder, views=views_in, labels=labels))
        x, (b, t, texts) = inputs
        with autocast:
            feats = encoder.encode_preprocessed(x)
        return objective_terms(terms, AuxBatch(encoder=encoder, clip_feats=feats.float().reshape(b, t, -1),
                                               text=texts))

    def run_objectives(raw, augment):
        return objectives_on(augmented(stage(raw), augment))

    # The training forward pass, through DDP with several GPUs; validation
    # runs on rank 0 and uses objectives_on directly.
    train_forward = distributed.wrap(procs, device, encoder, [t.loss for t in terms], objectives_on,
                                     find_unused_parameters=bool(cfg.perf.get("ddp_find_unused", False)))

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
        # Rank 0 validates alone, at the per-GPU batch size it trains at (the
        # total could run out of memory), so eval_batches scales up to cover
        # the same number of images.
        val_set = _image_dataset(cfg, data, data.val_split, False, views, cfg.seed)
        loader = _data_loader(val_set, cfg, int(cfg.num_workers), device, persistent=False,
                              batch_size=local_batch)
        sums, n = {}, 0
        limit = data.get("eval_batches")
        limit = None if limit is None else int(limit) * procs.world
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

    print(f"\nTraining {train_steps:,} steps of {cfg.batch_size} {item}"
          + (f" ({train_steps / steps_per_epoch:.1f} epochs)" if steps_per_epoch else "")
          + (f" (from step {step:,})" if step else "")
          + f", validating every {eval_frequency:,} steps")
    # Run-time estimate (src/timing.py): eta in the logs, timing.json, benchmark mode.
    timer, bench_steps = timer_from_config(
        cfg, train_steps, eval_frequency, 0, run_dir,
        {"script": "pretrain", "objectives": names, "data": data.name, "embedding": cfg.embedding.name,
         "batch_size": cfg.batch_size, "image_size": int(data.image_size),
         "num_workers": int(cfg.num_workers), "amp_bf16": bool(cfg.perf.amp_bf16), "gpus": procs.world},
        t_script, start_step=step, final_eval=True, save_per_eval=bool(cfg.save.enabled),
        log_frequency=int(cfg.log_frequency), main_process=procs.is_main)
    bench_from = step
    sums, n_sum, data_time, t_log = {}, 0, 0.0, time.time()
    # The loop: draw a batch, then K updates on it (schedule.updates_per_batch),
    # each with fresh augmentation draws unless schedule.reaugment is false.
    stop = False
    while step < train_steps and not stop:
        set_mode(True)
        t0 = time.time()
        raw = next(train_iter)
        data_time += time.time() - t0
        staged, inputs = stage(raw), None
        for _ in range(K):
            if step >= train_steps:
                break
            lr = base_lr * lr_factor(sched, step, sched_epoch)
            for g in optimizer.param_groups:
                g["lr"] = lr
            set_mode(True)
            if inputs is None or sch["reaugment"]:
                inputs = augmented(staged, augment=True)
            total, values = train_forward(inputs)
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
                avg = distributed.mean_over_ranks(procs, {k: float(v) / n_sum for k, v in sums.items()}, device)
                sps, wait = n_sum / elapsed, data_time / elapsed
                timer.mark(step)
                timing_text, timing_metrics = timer.status(step)
                timer.write(step)
                epoch = f"  epoch={step / steps_per_epoch:.2f}" if steps_per_epoch else ""
                print(f"  step {step:>9,}{epoch}  loss={avg['loss']:.4f}  "
                      f"{fmt(avg, sorted(k for k in avg if '/' in k))}  lr={lr:.3g}"
                      f"  steps/s={sps:.1f}  data_wait={wait:.0%}{timing_text}")
                if wandb is not None:
                    log = {f"train/{k}": v for k, v in avg.items()}
                    log.update({"train/lr": lr, "charts/steps_per_sec": sps, "charts/data_wait_frac": wait,
                                **timing_metrics, "global_step": step})
                    if steps_per_epoch:
                        log["train/epoch"] = step / steps_per_epoch
                    wandb.log(log, step=step)
                sums, n_sum, data_time, t_log = {}, 0, 0.0, time.time()

            bench_now = False
            if bench_steps and step % cfg.log_frequency == 0:      # rank 0's clock decides
                bench_now = distributed.from_main(
                    procs, timer.rate() is not None and step - bench_from >= bench_steps, device)
            scheduled = step % eval_frequency == 0 or step == train_steps
            eval_now = scheduled
            if bench_now:
                eval_now = distributed.from_main(procs, scheduled or timer.mean_excluded("eval") is None, device)
            if eval_now:
                t_eval = time.time()
                if procs.is_main:
                    with timer.exclude("eval", timing_only=not scheduled):
                        ev = evaluate()
                        print(f"  step {step:>9,}  EVAL loss={ev['loss']:.4f}  {fmt(ev, sorted(k for k in ev if '/' in k))}")
                        if wandb is not None:
                            wandb.log({f"eval/{k}": v for k, v in ev.items()}, step=step)
                    if cfg.save.enabled and not bench_steps:
                        with timer.exclude("save"):
                            save(step, final=step == train_steps)
                    elif cfg.save.enabled and (scheduled or timer.mean_excluded("save") is None):
                        # Benchmark: time one checkpoint write (to a scratch name -- a
                        # checkpoint.pt here would be resumed by the next run).
                        test_path = os.path.join(run_dir, ".timing_save_test.pt")
                        with timer.exclude("save", timing_only=not scheduled):
                            _save(checkpoint(step)[1], test_path)
                        os.remove(test_path)
                distributed.barrier(procs)         # the others wait for rank 0
                t_log += time.time() - t_eval      # keep steps/s about training

            if bench_now:                          # benchmark mode: measured; predict and stop
                timer.write(step)
                if procs.is_main:
                    print("\n" + timer.benchmark_report(step))
                stop = True
                break

    if cfg.save.enabled and not bench_steps:
        print(f"\nSaved to: {run_dir}  (embedding.yaml: how to use the encoder)")
    if wandb is not None:
        wandb.finish()
    if not bench_steps and step > timer.start_step:
        # The final record: the job's whole time, final validation and save
        # included. Not for a job resumed after the run had already finished
        # (chained jobs): it trained nothing, and would replace the record.
        timer.write(step)
    distributed.close(procs)


if __name__ == "__main__":
    main()
