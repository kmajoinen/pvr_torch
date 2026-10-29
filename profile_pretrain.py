"""
profile_pretrain.py: where a pretrain.py training step spends its time.
Builds the encoder, objectives, optimizer, augmentation and data loader
pretrain.py builds from the same config, runs its training step (the clip
branch: clip objectives such as R3M and VIP only), and saves nothing but the trace.

    python profile_pretrain.py +experiment=r3m embedding=resnet50 pretrain_data.root=/path/to/ssv2
    python profile_pretrain.py +experiment=r3m ... profile.synthetic=true      # no data needed
    python profile_pretrain.py +experiment=r3m ... +objectives.0.text_dropout=false   # a variant

After profile.warmup steps, four passes:
  1. timing    profile.steps steps run as pretrain.py runs them: steps/s and
               data-loader wait (compare wandb's charts/steps_per_sec)
  2. phases    the same steps with a GPU sync after every phase: ms per phase
               (data, input, encoder forward, objective -- with R3M's text
               encoder and reward head split out -- backward, optimizer).
               Serialised, so the phases add up to more than pass 1's step
               time: what pass 1 overlaps (CPU launching ahead of the GPU)
  3. profiler  torch.profiler over profile.profile_steps steps: top ops by GPU
               and by CPU time; kernel launches, host-to-device copies and
               syncs per step; GPU busy fraction = kernel time / pass 1's step
               time (well below 1: the GPU waits for Python, kernel launches
               or syncs, and a larger batch costs little extra time); and
               trace.json in the run directory (open in ui.perfetto.dev)
  4. syncs     one step under torch.cuda.set_sync_debug_mode: the lines that
               make the CPU wait for the GPU (not every sync is reported)

The training step below mirrors pretrain.py's clip_batch / training loop;
keep them in step.
"""

import collections
import contextlib
import linecache
import os
import random
import time
import traceback
import warnings

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from pretrain import _check_objectives, _loader, _shared_sampler, as_stored, build_optimizer, final_crop, reseed
from src.augmentations import build_augmentation
from src.encoder_config import embedding_kwargs
from src.losses import AuxBatch, build_aux_losses, objective_terms

REPO = os.path.dirname(os.path.abspath(__file__))

# profile.synthetic: SSv2-like descriptions (DistilBERT's cost depends on their length)
SYNTHETIC_TEXTS = [
    "putting a pen on a table", "pushing a cup from left to right",
    "taking one of many coins from the table", "moving a phone closer to a book",
    "pretending to open a drawer without actually opening it", "tearing a sheet of paper into two pieces",
    "lifting a bottle up completely, then letting it drop down", "turning the camera left while filming a chair",
]


class Phases:
    """record_function labels for the profiler; while `timing`, also a GPU
    sync and wall-clock total per label."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.timing = False
        self.totals = collections.defaultdict(float)

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def __call__(self, name):
        with torch.profiler.record_function(name):
            if not self.timing:
                yield
                return
            self.sync()
            t0 = time.perf_counter()
            yield
            self.sync()
            self.totals[name] += time.perf_counter() - t0


def _label_r3m(phase):
    """Phase labels inside R3M's loss, without touching src/losses.py."""
    from src.losses import _LanguageReward, _SentenceEncoder

    def wrap(cls, attr, name):
        f = getattr(cls, attr)

        def labelled(*args, **kwargs):
            with phase(name):
                return f(*args, **kwargs)
        setattr(cls, attr, labelled)
    wrap(_SentenceEncoder, "tokenize", "r3m/text_tokenize")
    wrap(_SentenceEncoder, "embed", "r3m/text_encoder")
    wrap(_LanguageReward, "forward", "r3m/reward_head")


def _synthetic(batch_size, frames_per_clip, size, device, seed):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, 256, (batch_size, frames_per_clip, 3, size, size), dtype=torch.uint8, generator=g)
    if device.type == "cuda":
        frames = frames.pin_memory()                  # as the loader's pin_memory
    texts = [SYNTHETIC_TEXTS[i % len(SYNTHETIC_TEXTS)] for i in range(batch_size)]
    while True:
        yield frames, texts


def _device_us(e):
    return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)


def _sync_sites(run_step):
    """Run one step with sync warnings on; group them by the innermost repo
    line on the stack."""
    sites = collections.Counter()
    show = warnings.showwarning

    def record(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" not in str(message):
            return show(message, category, filename, lineno, file, line)
        frames = [f for f in traceback.extract_stack()[:-1]
                  if f.filename.startswith(REPO) and not f.filename.endswith("profile_pretrain.py")]
        f = frames[-1] if frames else None
        where = (f"{os.path.relpath(f.filename, REPO)}:{f.lineno}  {(linecache.getline(f.filename, f.lineno)).strip()}"
                 if f else f"{filename}:{lineno}")
        sites[where] += 1

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = record
        torch.cuda.set_sync_debug_mode("warn")
        try:
            run_step()
            torch.cuda.synchronize()
        finally:
            torch.cuda.set_sync_debug_mode("default")
            warnings.showwarning = show
    return sites


@hydra.main(config_path="configs", config_name="config_profile_pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    prof_cfg = cfg.profile
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ── As pretrain.py builds them (clip data only) ───────────────────────────
    data = cfg.pretrain_data
    if str(data.get("type", "clips")) != "clips":
        raise ValueError("profile_pretrain.py profiles clip objectives (pretrain_data.type=clips)")
    specs = _check_objectives(cfg, "clips")
    if not prof_cfg.synthetic and OmegaConf.is_missing(data, "root"):
        raise ValueError("set pretrain_data.root, or profile.synthetic=true")
    frames_as_stored = bool(data.get("frames_as_stored", False))
    from src.embeddings import EmbeddingNet
    encoder = EmbeddingNet(
        cfg.embedding.name, pretrained=bool(cfg.embedding.get("pretrained", True)), train=True,
        disable_cuda=device.type == "cpu", model_dir=cfg.model_dir,
        input_size=int(data.image_size), **embedding_kwargs(cfg),
    )
    encoder.set_trainable(cfg.trainable)
    terms = build_aux_losses(specs, encoder)
    for t in terms:
        t.loss.to(device)
    named = [(n, p) for n, p in encoder.named_parameters() if p.requires_grad] + \
            [(n, p) for t in terms for n, p in t.loss.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    optimizer = build_optimizer(cfg, named, float(cfg.lr))
    grad_clip = float("inf") if cfg.grad_clip is None else float(cfg.grad_clip)
    aug = build_augmentation(cfg.augmentation)
    if aug is not None:
        aug = aug.to(device)
    encoder.train(True)
    for t in terms:
        t.loss.train(True)

    sampler = _shared_sampler(terms, True)
    size = int(data.image_size)
    if prof_cfg.synthetic:
        train_iter = _synthetic(int(cfg.batch_size), sampler.frames_per_clip, size, device, cfg.seed)
    else:
        train_iter = _loader(cfg, data, data.train_manifest, sampler, int(cfg.num_workers), device,
                             reseed(cfg.seed, 0))
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(cfg.perf.amp_bf16))
    phase = Phases(device)
    _label_r3m(phase)

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "none"
    obj = {k: v for k, v in (specs[0].items() if len(specs) == 1 else ())
           if k in ("text_dropout", "lang_weight", "tcn_weight", "num_negatives")}
    print(f"Device: {device} ({gpu}), {os.cpu_count()} CPUs visible, torch threads {torch.get_num_threads()}\n"
          f"Encoder: {cfg.embedding.name} -> {encoder.out_size} dims | objectives: {[t.name for t in terms]} {obj}\n"
          f"Batch: {cfg.batch_size} videos x {sampler.frames_per_clip} frames at {size}px | "
          f"amp_bf16={cfg.perf.amp_bf16} tf32={cfg.perf.tf32} | "
          f"data: {'synthetic' if prof_cfg.synthetic else f'{data.name}, {cfg.num_workers} workers'}")

    # ── The training step (pretrain.py's clip_batch + loop body) ──────────────
    data_time = 0.0

    def run_step():
        nonlocal data_time
        t0 = time.perf_counter()
        with phase("data"):
            frames, texts = next(train_iter)
        data_time += time.perf_counter() - t0
        with phase("input"):
            b, t = frames.shape[:2]
            x = frames.reshape(b * t, *frames.shape[2:]).to(device, non_blocking=True)
            x = as_stored(x, size) if frames_as_stored else encoder.preprocess(x)
            if aug is not None:
                if cfg.clip_augmentation == "shared":
                    x = aug(x.reshape(b, t * x.shape[1], *x.shape[2:])).reshape(x.shape)
                else:
                    x = aug(x)
            if cfg.get("final_crop") is not None:
                x = final_crop(x, cfg.final_crop)
        with phase("encoder_forward"):
            with autocast:
                feats = encoder.encode_preprocessed(x)
        with phase("objective"):
            batch = AuxBatch(encoder=encoder, clip_feats=feats.float().reshape(b, t, -1), text=list(texts))
            total, values = objective_terms(terms, batch)
        with phase("backward"):
            optimizer.zero_grad(set_to_none=True)
            total.backward()
        with phase("optimizer"):
            values["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
        return values

    print(f"\nWarm-up: {prof_cfg.warmup} steps")
    for _ in range(int(prof_cfg.warmup)):
        run_step()
    phase.sync()

    # ── 1. timing ─────────────────────────────────────────────────────────────
    n = int(prof_cfg.steps)
    data_time = 0.0
    t0 = time.perf_counter()
    for _ in range(n):
        run_step()
    phase.sync()
    elapsed = time.perf_counter() - t0
    step_ms = 1000 * elapsed / n
    print(f"\n[1] timing, {n} steps as pretrain.py runs them\n"
          f"    {n / elapsed:.2f} steps/s = {step_ms:.1f} ms/step, data wait {data_time / elapsed:.1%}"
          f"  -> 1M steps in {1e6 * step_ms / 3.6e6:.1f} h")

    # ── 2. phases ─────────────────────────────────────────────────────────────
    phase.timing = True
    for _ in range(n):
        run_step()
    phase.timing = False
    tot = phase.totals
    serial_ms = sum(tot[k] for k in ("data", "input", "encoder_forward", "objective", "backward", "optimizer"))
    serial_ms *= 1000 / n
    print(f"\n[2] phases, synchronised after each (ms/step; sum {serial_ms:.1f} vs {step_ms:.1f} unsynchronised)")
    for k in ("data", "input", "encoder_forward", "objective", "r3m/text_tokenize", "r3m/text_encoder",
              "r3m/reward_head", "backward", "optimizer"):
        if k not in tot:
            continue
        indent = "      " if k.startswith("r3m/") else "    "
        print(f"{indent}{k:<20} {1000 * tot[k] / n:8.2f}")
    if "objective" in tot:
        rest = tot["objective"] - sum(tot.get(k, 0.0) for k in ("r3m/text_tokenize", "r3m/text_encoder",
                                                                 "r3m/reward_head"))
        print(f"      {'(rest: tcn, norms, losses)':<20} {1000 * rest / n:8.2f}")

    # ── 3. profiler ───────────────────────────────────────────────────────────
    k = int(prof_cfg.profile_steps)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        for _ in range(k):
            with torch.profiler.record_function("train_step"):
                run_step()
        phase.sync()
    events = prof.key_averages()
    from torch.autograd import DeviceType
    kernel_us = sum(_device_us(e) for e in events
                    if e.device_type == DeviceType.CUDA and not e.is_user_annotation)
    counts = {e.key: e.count for e in events}
    print(f"\n[3] torch.profiler, {k} steps")
    if device.type == "cuda":
        busy = kernel_us / 1000 / k / step_ms
        print(f"    GPU kernel time {kernel_us / 1000 / k:.1f} ms/step of {step_ms:.1f} ms -> GPU busy {busy:.0%}")
    for name in ("cudaLaunchKernel", "cudaMemcpyAsync", "cudaStreamSynchronize", "cudaDeviceSynchronize",
                 "cudaEventSynchronize", "aten::item", "aten::_local_scalar_dense", "aten::randperm"):
        if name in counts:
            print(f"    {name:<28} {counts[name] / k:8.1f} per step")
    limit = int(prof_cfg.row_limit)
    if device.type == "cuda":
        print("\n    top ops by GPU time:")
        print(events.table(sort_by="self_cuda_time_total", row_limit=limit))
    print("    top ops by CPU time:")
    print(events.table(sort_by="self_cpu_time_total", row_limit=limit))
    if prof_cfg.trace:
        path = os.path.join(os.getcwd(), "trace.json")
        prof.export_chrome_trace(path)
        print(f"    trace: {path}  (ui.perfetto.dev)")

    # ── 4. syncs ──────────────────────────────────────────────────────────────
    if device.type == "cuda":
        sites = _sync_sites(run_step)
        print("\n[4] synchronising calls in one step" + ("" if sites else ": none reported"))
        for where, c in sites.most_common():
            print(f"    {c:4d}x  {where}")


if __name__ == "__main__":
    main()
