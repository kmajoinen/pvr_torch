"""Image pretraining data and objectives: shards (src/data/shards.py), the
ImageNet converter (src/data/prepare_imagenet.py), the image dataset
(src/data/images.py), the supervised and MoCo objectives, and pretrain.py's
learning-rate schedules. Uses a tiny synthetic ImageNet in Kaggle's and
torchvision's layouts."""

import io
import os
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from src.data.images import ImageDataset, decode, load_classes
from src.data.prepare_imagenet import main as prepare
from src.data.shards import ShardReader, ShardWriter, load_index
from src.losses import AuxBatch, MoCo, Supervised

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
WNIDS = ["n01440764", "n01443537", "n02102040"]


def _jpeg(color, size=(80, 60), mode="RGB", fmt="JPEG"):
    b = io.BytesIO()
    Image.new(mode, size, color).save(b, fmt)
    return b.getvalue()


def _fake_images():
    """{split: [(relative path, wnid, bytes)]}; class c's images are colour c*80."""
    train, val = [], []
    for c, w in enumerate(WNIDS):
        for i in range(6):
            data = _jpeg((c * 80, 0, 0), size=(80 + 10 * i, 60))
            if (c, i) == (0, 1):
                data = _jpeg((0, 0, 0, 0), mode="CMYK")
            if (c, i) == (1, 2):
                data = _jpeg(128, mode="L")
            if (c, i) == (2, 3):
                data = _jpeg((160, 0, 0), fmt="PNG")       # a PNG with a .JPEG name, as in ImageNet
            train.append((f"{w}/{w}_{i}.JPEG", w, data))
        for i in range(2):
            val.append((f"ILSVRC2012_val_{c * 2 + i:08d}.JPEG", w, _jpeg((c * 80, 0, 0))))
    return {"train": train, "val": val}


def _fake_kaggle_zip(tmp):
    imgs = _fake_images()
    path = tmp / "imagenet-object-localization-challenge.zip"
    with zipfile.ZipFile(path, "w") as z:
        for rel, _, data in imgs["train"]:
            z.writestr(f"ILSVRC/Data/CLS-LOC/train/{rel}", data)
        for rel, _, data in imgs["val"]:
            z.writestr(f"ILSVRC/Data/CLS-LOC/val/{rel}", data)
        z.writestr("LOC_val_solution.csv", "ImageId,PredictionString\n" + "".join(
            f"{os.path.splitext(rel)[0]},{w} 1 2 3 4 {w} 5 6 7 8\n" for rel, w, _ in imgs["val"]))
        z.writestr("LOC_synset_mapping.txt", "".join(f"{w} class {w}\n" for w in WNIDS))
    return path


def _fake_folders(tmp):
    imgs = _fake_images()
    for rel, _, data in imgs["train"]:
        (tmp / "tv" / "train" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / "tv" / "train" / rel).write_bytes(data)
    for rel, w, data in imgs["val"]:
        (tmp / "tv" / "val" / w).mkdir(parents=True, exist_ok=True)
        (tmp / "tv" / "val" / w / rel).write_bytes(data)
    return tmp / "tv"


@pytest.fixture(scope="module")
def imagenet(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("imagenet")
    out = tmp / "out"
    prepare(["--source", str(_fake_kaggle_zip(tmp)), "--out", str(out), "--workers", "2",
             "--shard-gb", str(2e-6)])                     # ~2 KB shards: several per split
    return out


# ---- shards ------------------------------------------------------------------

def test_shards_round_trip_and_stay_plain_tar(tmp_path):
    rng = np.random.default_rng(0)
    blobs = [rng.bytes(int(rng.integers(1, 3000))) for _ in range(40)]
    with ShardWriter(str(tmp_path), max_bytes=10_000) as w:
        for i, b in enumerate(blobs):
            w.add(f"item{i}.bin", b, label=i % 3)
    index, reader = load_index(str(tmp_path)), ShardReader(str(tmp_path))
    assert index["shard"].max() >= 2
    assert all(reader.read(index, i) == b for i, b in enumerate(blobs))
    assert list(index["label"][:4]) == [0, 1, 2, 0]
    with tarfile.open(tmp_path / "shard_000000.tar") as t:            # inspectable with tar
        assert t.getnames()[0] == "item0.bin"


# ---- converter ---------------------------------------------------------------

def test_kaggle_zip_conversion(imagenet):
    assert load_classes(str(imagenet)) == WNIDS
    for split, n in (("train", 18), ("val", 6)):
        index = load_index(str(imagenet / split))
        assert len(index["key"]) == n and index["shard"].max() >= 1
        reader = ShardReader(str(imagenet / split))
        for i in range(n):
            img = decode(reader.read(index, i))
            assert img.shape[0] == 3                                  # CMYK, gray, PNG -> RGB
    val = load_index(str(imagenet / "val"))
    assert list(val["label"]) == [0, 0, 1, 1, 2, 2]                   # from LOC_val_solution.csv


def test_folder_layout_matches_kaggle(imagenet, tmp_path):
    out = tmp_path / "out"
    prepare(["--source", str(_fake_folders(tmp_path)), "--out", str(out), "--workers", "1"])
    for split in ("train", "val"):
        a, b = load_index(str(imagenet / split)), load_index(str(out / split))
        assert list(a["key"]) == list(b["key"]) and list(a["label"]) == list(b["label"])


def test_subsets_and_downscaling(tmp_path):
    zp = _fake_kaggle_zip(tmp_path)
    (tmp_path / "classes.txt").write_text(f"{WNIDS[2]}\n{WNIDS[0]}\n")
    out = tmp_path / "sub"
    prepare(["--source", str(zp), "--out", str(out), "--classes", str(tmp_path / "classes.txt"),
             "--per-class", "2", "--short-side", "65", "--workers", "1"])
    assert load_classes(str(out)) == [WNIDS[0], WNIDS[2]]             # relabelled 0, 1
    index = load_index(str(out / "train"))
    assert list(index["label"]) == [0, 0, 1, 1]
    reader = ShardReader(str(out / "train"))
    sides = [min(decode(reader.read(index, i)).shape[1:]) for i in range(4)]
    assert all(s <= 65 for s in sides)


# ---- dataset -----------------------------------------------------------------

def _stream(ds, n_batches, workers, batch):
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers)
    it = iter(loader)
    return [next(it) for _ in range(n_batches)]


@pytest.mark.parametrize("workers", [0, 2])
def test_each_epoch_sees_every_image_once(imagenet, workers):
    ds = ImageDataset(str(imagenet), "train", train=True, size=32, views=2, seed=0, batch_size=3)
    batches = _stream(ds, 6, workers, 3)                              # 18 images = one epoch
    imgs, labels = torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])
    assert imgs.shape == (18, 2, 3, 32, 32) and imgs.dtype == torch.uint8
    assert sorted(labels.tolist()) == [0] * 6 + [1] * 6 + [2] * 6


class _IndexOnly(ImageDataset):
    """Yields sample indices instead of images (the order is what's tested)."""

    def _item(self, i):
        return i


@pytest.mark.parametrize("workers, start", [(0, 5), (2, 5), (2, 4), (3, 7)])
def test_resume_continues_the_stream_exactly(imagenet, workers, start):
    # 18 images, batches of 2: 9 per epoch, so the resumed stream crosses an epoch.
    def batches(start_batch, n):
        ds = _IndexOnly(str(imagenet), "train", train=True, seed=3, batch_size=2, start_batch=start_batch)
        it = iter(torch.utils.data.DataLoader(ds, batch_size=2, num_workers=workers))
        return [next(it).tolist() for _ in range(n)]
    assert batches(start, 8) == batches(0, start + 8)[start:]


def test_eval_is_one_ordered_center_cropped_pass(imagenet):
    ds = ImageDataset(str(imagenet), "val", train=False, size=48, resize=56)
    items = list(ds)
    assert len(items) == 6 and [l for _, l in items] == [0, 0, 1, 1, 2, 2]
    assert items[0][0].shape == (1, 3, 48, 48)
    assert torch.equal(list(ds)[3][0], items[3][0])                   # deterministic


# ---- objectives ----------------------------------------------------------------

class _Enc(nn.Module):
    out_size = 8

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 4, stride=4), nn.BatchNorm2d(4), nn.Flatten(),
                                 nn.Linear(4 * 4 * 4, 8))

    def encode_preprocessed(self, x):
        return self.net(x)


def test_supervised_loss_and_accuracy():
    torch.manual_seed(0)
    enc = _Enc()
    loss_fn = Supervised(enc, num_classes=10)
    x, y = torch.rand(16, 3, 16, 16), torch.randint(0, 10, (16,))
    loss, m = loss_fn(AuxBatch(encoder=enc, views=[x], labels=y))
    assert loss.item() == pytest.approx(torch.nn.functional.cross_entropy(loss_fn.fc(enc.net(x)), y).item())
    assert 0 <= m["top1"] <= m["top5"] <= 1


def test_moco_queue_momentum_and_key_order():
    torch.manual_seed(0)
    enc = _Enc()
    moco = MoCo(enc, dim=4, queue_size=10, momentum=0.9, temperature=0.5, bn_splits=2)
    x = torch.rand(6, 3, 16, 16)
    # Eval mode: no momentum update, no enqueue; identical views and encoders
    # give every query its own key back (the un-shuffle restores the order).
    moco.eval(), enc.eval()
    queue = moco.queue.clone()
    loss, m = moco(AuxBatch(encoder=enc, views=[x, x]))
    assert torch.equal(moco.queue, queue) and m["acc1"] == 1.0
    # Training: keys enqueued (with wrap-around), key encoder moves toward the query encoder.
    moco.train(), enc.train()
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(1.0)
    before = [p.clone() for p in moco.key_encoder.parameters()]
    for _ in range(2):
        loss, _ = moco(AuxBatch(encoder=enc, views=[x, x.flip(-1)]))
    assert int(moco.queue_ptr) == 12 % 10
    after = list(moco.key_encoder.parameters())
    target = list(enc.parameters())
    for b, a, q in zip(before, after, target):
        expected = 0.9 * (0.9 * b + 0.1 * q) + 0.1 * q
        assert torch.allclose(a, expected, atol=1e-6)
    loss.backward()
    assert all(p.grad is None for p in moco.key_encoder.parameters())
    assert all(p.grad is not None for p in enc.parameters())


# ---- pretrain.py ---------------------------------------------------------------

def test_lr_schedules():
    from pretrain import lr_factor
    step = OmegaConf.create({"name": "step", "milestones": [30, 60], "gamma": 0.1})
    assert lr_factor(step, 29 * 100, steps_per_epoch=100) == 1.0
    assert lr_factor(step, 30 * 100, steps_per_epoch=100) == pytest.approx(0.1)
    assert lr_factor(step, 61 * 100 + 5, steps_per_epoch=100) == pytest.approx(0.01)
    cos = OmegaConf.create({"name": "cosine", "total": 200})
    assert lr_factor(cos, 0, 100) == 1.0
    assert lr_factor(cos, 100 * 100 + 99, 100) == pytest.approx(0.5)  # changes once per epoch
    warm = OmegaConf.create({"name": "constant", "warmup": 10})
    assert lr_factor(warm, 4) == pytest.approx(0.5) and lr_factor(warm, 10) == 1.0


@pytest.mark.parametrize("preset", ["imagenet_supervised", "moco_v2"])
def test_presets_compose(preset):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_pretrain",
                      overrides=[f"+experiment={preset}", "embedding=resnet18", "pretrain_data.root=/x",
                                 "objectives.0.weight=2.0"])     # list entries override by index
    assert cfg.pretrain_data.type == "images" and cfg.embedding.name == "resnet18"
    assert cfg.embedding.pretrained is False and cfg.optimizer == "sgd" and cfg.batch_size == 256
    assert cfg.objectives[0].weight == 2.0


def test_rl_scripts_reject_labelled_objectives():
    from src.encoder_config import encoder_settings
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_sac", overrides=["encoder.aux=[{name:supervised,weight:1.0}]"])
    with pytest.raises(ValueError, match="pretrain.py"):
        encoder_settings(cfg, valid_grads=("none", "critic", "actor", "both"))


@pytest.mark.parametrize("aug, ok", [([], False), (["augmentation=random_shift"], False),
                                     (["augmentation=random_shift", "augmentation.stage=aux"], True),
                                     (["augmentation=random_shift", "augmentation.stage=both"], True)])
def test_rl_moco_needs_augmented_views(aug, ok):
    # Two views of a replay frame differ only through an augmentation the aux
    # losses see (stage aux or both).
    from src.encoder_config import encoder_settings
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_sac",
                      overrides=["encoder.aux=[{name:moco,weight:1.0}]"] + aug)
    grads = ("none", "critic", "actor", "both")
    if not ok:
        with pytest.raises(ValueError, match="two views"):
            encoder_settings(cfg, valid_grads=grads)
        return
    s = encoder_settings(cfg, valid_grads=grads)
    assert s.trains and s.aux_views == 2 and s.views_aug


@pytest.mark.parametrize("preset", ["imagenet_supervised", "moco_v2", "r3m"])
@pytest.mark.parametrize("first", [True, False])
def test_preset_augmentation_can_be_replaced(preset, first):
    # Presets select their augmentation as a config group, so augmentation=<x>
    # on the command line replaces it (a preset that set it in its body would
    # win over the command line's group choice).
    ov = [f"+experiment={preset}", "augmentation=none"]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_pretrain",
                      overrides=(ov if first else ov[::-1]) + ["pretrain_data.root=/x"])
    assert cfg.augmentation.name is None and not cfg.augmentation.get("params")


def test_schedule_counts_steps_when_the_run_is_sized_in_steps():
    # pretrain.py passes steps_per_epoch only for epoch-sized runs; a
    # train_steps-sized image run decays over its steps.
    from pretrain import lr_factor
    cos = OmegaConf.create({"name": "cosine", "total": 1000})
    assert lr_factor(cos, 999, None) == pytest.approx(0.0, abs=1e-4)
    assert lr_factor(cos, 500, None) == pytest.approx(0.5)


def test_bounded_loading_keeps_order(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from src.data import prepare_imagenet as m
    zp = _fake_kaggle_zip(tmp_path)
    src = m.Source(str(zp))
    refs = [e[2] for e in src.entries("train")]
    with ProcessPoolExecutor(2, initializer=m._init, initargs=(str(zp), None, 90, True)) as pool:
        got = list(m._ordered_load(pool, refs, chunksize=4, in_flight=2))
    assert got == [src.read(r) for r in refs]


def test_conversion_rejects_undecodable_images(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"ILSVRC/Data/CLS-LOC/train/{WNIDS[0]}/{WNIDS[0]}_0.JPEG", b"not an image")
        z.writestr("ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG", _jpeg((0, 0, 0)))
        z.writestr("LOC_val_solution.csv", f"ImageId,PredictionString\nILSVRC2012_val_00000001,{WNIDS[0]} 1 2 3 4\n")
    with pytest.raises(ValueError, match="can't be decoded"):
        prepare(["--source", str(path), "--out", str(tmp_path / "out"), "--workers", "1"])


def test_supervised_class_count_must_match_the_data(imagenet):
    import subprocess, sys
    out = subprocess.run([sys.executable, "pretrain.py", "+experiment=imagenet_supervised", "embedding=resnet18",
                          f"pretrain_data.root={imagenet}", "device=cpu", "hydra.run.dir=" + str(imagenet.parent / "run"),
                          "num_workers=0"],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert "num_classes=1000, but the dataset" in out.stderr + out.stdout


def test_default_conversion_needs_no_opencv(tmp_path, monkeypatch):
    # OpenCV is needed only for --short-side; forked workers inherit the patch.
    import builtins
    real_import = builtins.__import__

    def no_cv2(name, *a, **k):
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_cv2)
    prepare(["--source", str(_fake_kaggle_zip(tmp_path)), "--out", str(tmp_path / "o"), "--workers", "1"])
    assert len(load_index(str(tmp_path / "o" / "train"))["key"]) == 18


def test_imagefolder_tar_matches_the_folder_layout(imagenet, tmp_path):
    # A cluster's packed copy: the torchvision layout inside one tar, under a prefix.
    folders = _fake_folders(tmp_path)
    tar = tmp_path / "ILSVRC2012_imagefolder.tar"
    with tarfile.open(tar, "w") as t:
        t.add(folders / "train", arcname="ILSVRC2012/train")
        t.add(folders / "val", arcname="ILSVRC2012/val")
    out = tmp_path / "from_tar"
    prepare(["--source", str(tar), "--out", str(out), "--workers", "2"])
    assert load_classes(str(out)) == WNIDS
    for split in ("train", "val"):
        a, b = load_index(str(imagenet / split)), load_index(str(out / split))
        assert sorted(zip(a["key"], a["label"])) == sorted(zip(b["key"], b["label"]))
        reader = ShardReader(str(out / split))
        assert all(decode(reader.read(b, i)).shape[0] == 3 for i in range(len(b["key"])))
    sub = tmp_path / "sub"
    prepare(["--source", str(tar), "--out", str(sub), "--per-class", "2", "--workers", "1"])
    assert sorted(load_index(str(sub / "train"))["label"]) == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("workers", [0, 2])
def test_extracted_image_folders_read_like_shards(imagenet, tmp_path, workers):
    # An extracted ImageFolder tree (e.g. ILSVRC2012_imagefolder.tar untarred
    # to a node's /tmp) is read directly, with the same labels as the shards.
    root = _fake_folders(tmp_path)
    assert load_classes(str(root)) == WNIDS
    ds = ImageDataset(str(root), "train", train=True, size=32, views=2, seed=0, batch_size=3)
    assert not ds.sharded and len(ds) == 18
    labels = torch.cat([b[1] for b in _stream(ds, 6, workers, 3)])
    assert sorted(labels.tolist()) == [0] * 6 + [1] * 6 + [2] * 6
    val = ImageDataset(str(root), "val", train=False, size=48)
    assert [l for _, l in val] == list(load_index(str(imagenet / "val"))["label"])
