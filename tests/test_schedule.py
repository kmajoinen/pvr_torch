"""Pretraining schedules (pretrain.py's schedule.*, src/data/order.py): sweep
and random sampling for images and clips, exact resumes, the run length
from epochs / steps / images, and the option checks."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision.io import write_jpeg

from pretrain import epochs_to_steps, plan_run, resume_changes, schedule_settings
from src.data.clips import ClipDataset, ClipManifest
from src.data.images import ImageDataset
from src.data.order import epoch_stream, random_stream
from src.losses import R3MFrameSampler

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


class _Stream(IterableDataset):
    """The order helpers through a real DataLoader (worker split, batching)."""

    def __init__(self, fn, n, seed=3, batch_size=2, start_batch=0):
        self.fn, self.n, self.seed, self.batch_size, self.start_batch = fn, n, seed, batch_size, start_batch

    def __iter__(self):
        info = get_worker_info()
        wid, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        for _, i in self.fn(self.n, self.seed, wid, workers, self.start_batch, self.batch_size):
            yield i


def _batches(ds, n, workers):
    it = iter(DataLoader(ds, batch_size=ds.batch_size, num_workers=workers))
    return [next(it).tolist() for _ in range(n)]


@pytest.mark.parametrize("workers", [0, 2])
def test_sweep_covers_every_item_once_per_epoch(workers):
    got = sum(_batches(_Stream(epoch_stream, 12, batch_size=3), 8, workers), [])   # 2 epochs
    assert sorted(got[:12]) == list(range(12)) and sorted(got[12:]) == list(range(12))
    assert got[:12] != list(range(12))                                             # shuffled


def test_random_draws_with_replacement_independent_of_workers():
    one = _batches(_Stream(random_stream, 12, batch_size=4), 9, 0)
    assert _batches(_Stream(random_stream, 12, batch_size=4), 9, 3) == one
    flat = sum(one, [])
    assert len(set(flat)) < len(flat) and all(0 <= i < 12 for i in flat)        # repeats: with replacement


@pytest.mark.parametrize("fn", [epoch_stream, random_stream])
@pytest.mark.parametrize("workers, start", [(0, 5), (2, 5), (3, 7)])
def test_both_orders_resume_exactly(fn, workers, start):
    fresh = _batches(_Stream(fn, 10, batch_size=2), start + 6, workers)
    assert _batches(_Stream(fn, 10, batch_size=2, start_batch=start), 6, workers) == fresh[start:]


class _ImageIds(ImageDataset):
    def _item(self, i):
        return i


def _image_folder(root, n=10):
    for i in range(n):
        d = root / "train" / f"c{i % 2}"
        d.mkdir(parents=True, exist_ok=True)
        write_jpeg(torch.zeros((3, 8, 8), dtype=torch.uint8), str(d / f"{i}.jpg"))


@pytest.mark.parametrize("order", ["sweep", "random"])
def test_image_dataset_orders_resume_exactly(tmp_path, order):
    _image_folder(tmp_path)
    make = lambda start: _ImageIds(str(tmp_path), "train", train=True, seed=1, batch_size=3,      # noqa: E731
                                   start_batch=start, order=order)
    fresh = _batches(make(0), 9, 2)
    assert _batches(make(4), 5, 2) == fresh[4:]


class _ClipIds(ClipDataset):
    def sample(self, rng, video=None):
        return int(rng.integers(0, len(self.manifest))) if video is None else video


def _videos(root, n=5, length=6):
    rows = ["path,len,txt"]
    for v in range(n):
        d = root / "frames" / f"v{v}"
        d.mkdir(parents=True)
        for i in range(1, length + 1):
            write_jpeg(torch.zeros((3, 8, 8), dtype=torch.uint8), str(d / f"{i:06d}.jpg"))
        rows.append(f"frames/v{v},{length},t")
    (root / "manifest.csv").write_text("\n".join(rows) + "\n")
    return ClipManifest(str(root), "manifest.csv", min_length=3)


def test_clip_sweep_draws_each_video_clips_per_video_times_and_resumes(tmp_path):
    m = _videos(tmp_path)
    make = lambda start: _ClipIds(m, R3MFrameSampler(0.2), order="sweep", clips_per_video=2,     # noqa: E731
                                  order_seed=7, batch_size=2, start_batch=start)
    assert make(0).items_per_epoch == 10
    fresh = _batches(make(0), 10, 2)                                    # 2 epochs of 5 batches
    epoch = sorted(sum(fresh[:5], []))
    assert epoch == sorted(list(range(5)) * 2)
    assert _batches(make(3), 7, 2) == fresh[3:]


def test_clip_random_is_unchanged_by_default(tmp_path):
    m = _videos(tmp_path)
    ds = ClipDataset(m, R3MFrameSampler(0.2))
    assert ds.order == "random" and ds.items_per_epoch == 5


def _cfg(*overrides, data="ssv2"):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config_pretrain",
                       overrides=[f"pretrain_data={data}", "pretrain_data.root=/x", *overrides])


def test_defaults_per_data_type():
    assert schedule_settings(_cfg(), images=False)["sampling"] == "random"
    s = schedule_settings(_cfg(data="imagenet"), images=True)
    assert s["sampling"] == "sweep" and s["updates_per_batch"] == 1 and s["reaugment"] is True


@pytest.mark.parametrize("overrides, images, match", [
    (["schedule.sampling=shuffle"], False, "expected sweep or random"),
    (["schedule.clips_per_video=2"], False, "sampling=sweep"),                  # random clips
    (["schedule.clips_per_video=2", "schedule.sampling=sweep"], True, "clip data"),
    (["schedule.epoch_batches=10", "schedule.sampling=sweep"], False, "random-sampling epoch"),
    (["schedule.updates_per_batch=0"], False, "whole number"),
    (["schedule.epoch_batches=0", "schedule.sampling=random"], False, "whole number"),
    (["schedule.clips_per_video=0", "schedule.sampling=sweep"], False, "whole number"),
    (["schedule.reaugment=maybe"], False, "true or false"),
])
def test_bad_schedules(overrides, images, match):
    with pytest.raises(ValueError, match=match):
        schedule_settings(_cfg(*overrides, data="imagenet" if images else "ssv2"), images)


def _plan(*overrides, n_items=1000, images=True):
    cfg = _cfg("batch_size=10", *overrides, data="imagenet" if images else "ssv2")
    return plan_run(cfg, schedule_settings(cfg, images), n_items)


def test_run_length_in_steps_epochs_and_images():
    p = _plan("train_steps=500")
    assert p["train_steps"] == 500 and p["batches_per_epoch"] == 100 and p["epochs"] is None
    assert _plan("epochs=3")["train_steps"] == 300
    p = _plan("epochs=3", "schedule.updates_per_batch=4")
    assert p["steps_per_epoch"] == 400 and p["train_steps"] == 1200
    p = _plan("epochs=2", "schedule.sampling=random", "schedule.epoch_batches=25")
    assert p["batches_per_epoch"] == 25 and p["train_steps"] == 50 and p["samples_per_epoch"] == 250
    # train_images counts samples drawn: K repeats of a batch count once.
    assert _plan("train_images=995", "schedule.updates_per_batch=3")["train_steps"] == 100 * 3
    p = _plan("epochs=4", "train_images=8000", "lr_schedule.name=warmup_cosine", "lr_schedule.warmup=1")
    assert p["epochs"] == pytest.approx(8) and p["train_steps"] == 800 and p["lr_schedule"]["warmup"] == 2


def test_clip_runs_can_have_epochs():
    p = _plan("epochs=2", "schedule.sampling=sweep", "schedule.clips_per_video=3", n_items=5 * 3 * 40,
              images=False)
    assert p["batches_per_epoch"] == 60 and p["train_steps"] == 120


def test_presets_keep_their_numbers():
    # R3M: random clips, a step budget; ImageNet: sweep epochs -- as before.
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        r3m = compose(config_name="config_pretrain", overrides=["+experiment=r3m", "pretrain_data.root=/x"])
        sup = compose(config_name="config_pretrain",
                      overrides=["+experiment=imagenet_supervised", "pretrain_data.root=/x"])
    p = plan_run(r3m, schedule_settings(r3m, False), 168_913)
    assert schedule_settings(r3m, False)["sampling"] == "random" and p["train_steps"] == 1_000_000
    p = plan_run(sup, schedule_settings(sup, True), 1_281_167)
    assert p["steps_per_epoch"] == 5004 and p["train_steps"] == 90 * 5004
    assert OmegaConf.select(sup, "schedule.updates_per_batch") == 1


def test_a_frequency_in_epochs_falls_between_batches():
    assert epochs_to_steps(0.33, 15012, 3) % 3 == 0                     # round(0.33 * 15012) = 4954 isn't
    assert epochs_to_steps(0.33, 15012, 3) == pytest.approx(0.33 * 15012, abs=3)
    assert epochs_to_steps(1, 5004, 1) == 5004                          # K = 1: plain rounding
    assert epochs_to_steps(1e-6, 100, 4) == 4                           # never 0


def _resume(old, *overrides, images=True):
    cfg = _cfg(*overrides, data="imagenet" if images else "ssv2")
    old = None if old is None else OmegaConf.to_container(old, resolve=True)
    return resume_changes(old, cfg, schedule_settings(cfg, images), images)


def _old_config(*overrides, images=True, drop_schedule=False):
    cfg = OmegaConf.to_container(_cfg(*overrides, data="imagenet" if images else "ssv2"), resolve=True)
    if drop_schedule:
        del cfg["schedule"]                                             # from before schedule.* existed
    return OmegaConf.create(cfg)


@pytest.mark.parametrize("images", [True, False])
def test_resuming_a_checkpoint_from_before_schedules_warns_of_nothing(images):
    assert _resume(_old_config(images=images, drop_schedule=True), images=images) == {}


def test_writing_out_a_default_sampling_is_not_a_change():
    assert _resume(_old_config(), "schedule.sampling=sweep") == {}


def test_real_changes_are_reported():
    assert _resume(_old_config(), "schedule.updates_per_batch=2") == {"schedule.updates_per_batch": (1, 2)}
    assert set(_resume(_old_config(), "batch_size=8", "num_workers=3")) == {"batch_size", "num_workers"}
    assert _resume(None) == {}


def test_a_random_clip_stream_only_minds_a_change_of_sampling():
    assert _resume(_old_config(images=False), "batch_size=8", images=False) == {}
    old = _old_config("schedule.sampling=sweep", images=False)
    assert "schedule.sampling" in _resume(old, images=False)
    assert "batch_size" in _resume(_old_config("schedule.sampling=sweep", images=False), "schedule.sampling=sweep",
                                   "batch_size=8", images=False)


def test_a_random_image_stream_doesnt_mind_the_worker_count():
    old = _old_config("schedule.sampling=random")
    assert _resume(old, "schedule.sampling=random", "num_workers=3") == {}
    assert set(_resume(old, "schedule.sampling=random", "batch_size=8")) == {"batch_size"}
    assert "num_workers" in _resume(_old_config(), "num_workers=3")           # sweep does
