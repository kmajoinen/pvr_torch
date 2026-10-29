"""
Image datasets for pretraining (pretrain.py, pretrain_data.type=images), in
either of two layouts under <root>, detected per split:

  shards  (src/data/shards.py, written by src/data/prepare_imagenet.py)
            <root>/classes.txt          one class per line (label = line number)
            <root>/<split>/shard_*.tar  JPEG bytes
            <root>/<split>/index.npz    key, shard, offset, size, label per image
  folders (torchvision's ImageFolder layout, e.g. an extracted
          ILSVRC2012_imagefolder.tar on a node's local disk)
            <root>/<split>/<class>/<image>
          Classes are the sorted folder names of <root>/train (or classes.txt
          if present), so labels match torchvision's and the shards'. The
          file list is read once, at startup -- fast on a local disk, slow on
          a shared filesystem with millions of files.

Images vary in size, so cropping to a fixed size happens per image in the
data-loader workers, before batching -- as in torchvision's ImageNet
pipeline. Training: `views` independent RandomResizedCrops per image (MoCo
needs 2); the remaining augmentations run on the GPU in pretrain.py.
Evaluation: resize the short side to `resize`, center-crop to `size`
(torchvision's 256 -> 224), or random crops when an objective needs several
views.

Training order is by epoch: each epoch is a fresh permutation of all images
(without replacement), split across workers. Each item is
(uint8 tensor (views, 3, size, size), label); label -1 for unlabelled data.

Video frames as images: a clip dataset's frame shards
(src/data/prepare_ssv2.py, `<root>/<split>/index.npz`) are image shards
without labels, so they load here as they are. `every` thins a split out:
video frames (keys <video>/<frame number>.<ext>, as prepare_ssv2.py writes
them) keep frames 1, 1 + every, 1 + 2 * every, ... of each video, so every
video keeps at least its first frame (consecutive frames are
near-duplicates); other images keep every n-th in stored order.
"""

import math
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.data.shards import ShardReader, load_index


_IMAGE = (".jpeg", ".jpg", ".png")


def load_classes(root: str):
    """classes.txt, else the sorted class folders of <root>/train, else None."""
    path = os.path.join(root, "classes.txt")
    if os.path.isfile(path):
        with open(path) as f:
            return [line.split()[0] for line in f if line.strip()]
    train = os.path.join(root, "train")
    if os.path.isdir(train):
        return sorted(e.name for e in os.scandir(train) if e.is_dir())
    return None


def _folder_index(root: str, split: str) -> dict:
    """The ImageFolder layout as an index: relative paths (bytes, compact for
    forked workers) and labels, sorted by path."""
    classes = load_classes(root)
    label = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in sorted(e.name for e in os.scandir(os.path.join(root, split)) if e.is_dir()):
        if c not in label:
            raise ValueError(f"{root}/{split}/{c}: a class that {root}/train doesn't have")
        files = sorted(e.name for e in os.scandir(os.path.join(root, split, c))
                       if e.is_file() and e.name.lower().endswith(_IMAGE))
        paths += [f"{c}/{f}" for f in files]
        labels += [label[c]] * len(files)
    if not paths:
        raise ValueError(f"no images in {root}/{split}/<class>/")
    return {"key": np.array(paths, dtype=np.bytes_), "label": np.array(labels, dtype=np.int32)}


def _frame_numbers(keys) -> Optional[np.ndarray]:
    """Each key's frame number if every key is <video>/<number>.<ext> (video
    frames), else None."""
    keys = np.asarray(keys, dtype=np.bytes_)
    _, slash, name = np.char.rpartition(keys, b"/").T
    stem = np.char.partition(name, b".")[:, 0]
    if not len(keys) or not (slash == b"/").all() or not np.char.isdigit(stem).all():
        return None
    return stem.astype(np.int64)


def decode(data: bytes) -> torch.Tensor:
    """JPEG/PNG bytes -> uint8 (3, H, W) RGB (grayscale and CMYK converted)."""
    from torchvision.io import ImageReadMode, decode_image
    return decode_image(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB)


class ImageDataset(IterableDataset):
    """train=True: endless, epoch by epoch; train=False: one ordered pass.

    start_batch (training): batches already consumed (a resumed run). With
    the same batch size and worker count, the stream continues exactly where
    it stopped: data-loader workers hand out whole batches in turn, starting
    with worker 0, so the batch after start_batch belongs to role
    start_batch % workers -- a fresh loader's worker i takes role
    (i + start_batch) % workers, and role r had produced
    ceil((start_batch - r) / workers) batches."""

    def __init__(self, root: str, split: str, train: bool, size: int = 224, views: int = 1,
                 scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), resize: int = 256, seed: int = 0,
                 batch_size: int = 1, start_batch: int = 0, limit: Optional[int] = None,
                 every: int = 1, interpolation: str = "bilinear"):
        self.dir = os.path.join(root, split)
        self.sharded = os.path.isfile(os.path.join(self.dir, "index.npz"))
        if int(every) != every or every < 1:
            raise ValueError(f"every must be a whole number >= 1, got {every}")
        every = int(every)
        if self.sharded:
            # Keys aren't needed to read shards (a frame index has millions);
            # `every` reads them once, to find frame numbers.
            fields = ("shard", "offset", "size", "label") + (("key",) if every > 1 else ())
            self.index = load_index(self.dir, fields=fields)
            self.reader = ShardReader(self.dir)
        elif os.path.isdir(self.dir):
            self.index, self.reader = _folder_index(root, split), None
        else:
            raise FileNotFoundError(f"{self.dir!r}: neither shards (index.npz) nor class folders")
        # How `every` thinned the split, for the startup printout: a change to
        # the key format would otherwise fall back to stored order silently.
        self.thinning = None
        if every > 1:
            frames = _frame_numbers(self.index["key"])
            keep = (frames - 1) % every == 0 if frames is not None else np.arange(len(self.index["key"])) % every == 0
            self.index = {k: v[keep] for k, v in self.index.items()}
            self.thinning = f"every={every}: " + ("per video" if frames is not None else "stored order")
        if self.sharded:
            self.index.pop("key", None)
        if limit is not None:
            self.index = {k: v[:limit] for k, v in self.index.items()}
        if str(interpolation).upper() not in ("BILINEAR", "BICUBIC"):
            raise ValueError(f"interpolation={interpolation!r}; expected bilinear or bicubic")
        self.train, self.size, self.views = train, size, views
        self.scale, self.ratio, self.resize = tuple(scale), tuple(ratio), resize
        self.interpolation = str(interpolation).upper()
        self.seed, self.batch_size, self.start_batch = seed, batch_size, start_batch

    def __len__(self):
        return len(self.index["label"])

    @property
    def labelled(self) -> bool:
        return bool(len(self)) and bool((self.index["label"] >= 0).all())

    def _crops(self, img):
        from torchvision.transforms import InterpolationMode, v2
        mode = InterpolationMode[self.interpolation]
        if self.train or self.views > 1:
            crop = v2.RandomResizedCrop(self.size, scale=self.scale, ratio=self.ratio,
                                        interpolation=mode, antialias=True)
            return torch.stack([crop(img) for _ in range(self.views)])
        img = v2.functional.resize(img, self.resize, interpolation=mode, antialias=True)
        return v2.functional.center_crop(img, self.size).unsqueeze(0)

    def _read(self, i) -> bytes:
        if self.sharded:
            return self.reader.read(self.index, i)
        with open(os.path.join(self.dir, self.index["key"][i].decode()), "rb") as f:
            return f.read()

    def _item(self, i):
        return self._crops(decode(self._read(i))), int(self.index["label"][i])

    def _order(self, epoch):
        return np.random.default_rng([self.seed, epoch]).permutation(len(self))

    def __iter__(self):
        info = get_worker_info()
        wid, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        if not self.train:
            for i in range(wid, len(self), workers):
                yield self._item(i)
            return
        # This worker's stream: its share of every epoch's permutation, in order.
        wid = (wid + self.start_batch) % workers
        skip = math.ceil(max(self.start_batch - wid, 0) / workers) * self.batch_size
        n = len(self)
        epoch = 0
        while True:
            share = len(range(wid, n, workers))
            if skip >= share:
                skip -= share
                epoch += 1
                continue
            for i in self._order(epoch)[wid::workers][skip:]:
                yield self._item(int(i))
            skip = 0
            epoch += 1
