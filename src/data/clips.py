"""
Video clip datasets for pretraining (pretrain.py).

On disk, a manifest plus the frames, in either of two layouts:

    <root>/<manifest>.csv     one row per video: path, len, txt[, start]

  folders  <root>/<path>/000001.jpg ...: the video's frames, numbered 1..len
           (R3M's frame layout; its official loader reads the same folders,
           though running that code would still need its own manifest.csv,
           with absolute paths and its 2-character text prefix)
  shards   (a `start` column) <root>/<path>/ is a shard folder
           (src/data/shards.py) holding the frames of many videos; frame i
           of a video is index row start + i - 1

`path` is relative to <root> (or absolute); `txt` is the video's description
(may be empty); all frames have the same size. src/data/prepare_ssv2.py
writes either for Something-Something v2; any other video source only needs
a script that writes the same.

ClipDataset samples forever. Each item is one video and the frames its
FrameSampler picks (src/losses.py), as (uint8 tensor (T, 3, H, W), text).
Which video (`order`, pretrain.py's schedule.sampling):
  random  uniformly, with replacement -- R3M's R3MBuffer sampling
  sweep   epoch by epoch, each video clips_per_video times per epoch, in a
          fresh shuffled order: a video's clips spread over the epoch, so
          rarely in one batch; resumes at the exact video (src/data/order.py;
          which frames each clip uses comes from the worker's RNG, reseeded
          on resume)
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class ClipManifest:
    """A manifest's videos with at least `min_length` frames."""

    def __init__(self, root: str, manifest: str, min_length: int = 1, text_strip: int = 0):
        import pandas as pd
        path = manifest if os.path.isabs(manifest) else os.path.join(root, manifest)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"clip manifest {path!r} not found")
        # Strings: SSv2-style numeric folder names would otherwise parse as ints.
        df = pd.read_csv(path, keep_default_na=False, dtype={"path": str, "txt": str})
        sharded = "start" in df.columns
        missing = {"path", "len", "txt"} - set(df.columns)
        if missing:
            raise ValueError(f"{path!r} lacks columns {sorted(missing)} (expected path, len, txt)")
        keep = df["len"] >= min_length
        self.skipped = int((~keep).sum())
        df = df[keep]
        if not len(df):
            raise ValueError(f"{path!r} has no videos with at least {min_length} frames")
        self.paths: List[str] = [p if os.path.isabs(p) else os.path.join(root, p) for p in df["path"]]
        self.lengths = df["len"].to_numpy(dtype=np.int64)
        # text_strip: characters cut from the start of every text (R3M's
        # Ego4D loader cuts the narrations' "C " prefix this way).
        self.texts: List[str] = [str(t)[text_strip:] for t in df["txt"]]
        self.starts = df["start"].to_numpy(dtype=np.int64) if sharded else None
        self.source = path

    def __len__(self):
        return len(self.paths)


def read_frame(video_path: str, index: int) -> torch.Tensor:
    from torchvision.io import ImageReadMode, read_image
    return read_image(os.path.join(video_path, f"{index:06d}.jpg"), mode=ImageReadMode.RGB)


class ClipDataset(IterableDataset):
    """Endless (frames, text) samples. Each data-loader worker draws from its
    own numpy Generator, seeded from the worker's torch seed, so workers
    don't repeat each other and a run is reproducible from torch.manual_seed."""

    def __init__(self, manifest: ClipManifest, sampler, seed: Optional[int] = None, order: str = "random",
                 clips_per_video: int = 1, order_seed: int = 0, batch_size: int = 1, start_batch: int = 0):
        if order not in ("random", "sweep"):
            raise ValueError(f"order={order!r}; expected random or sweep")
        self.manifest = manifest
        self.sampler = sampler
        self.seed = seed
        self.order, self.clips_per_video = order, int(clips_per_video)
        # sweep: the epoch permutations come from order_seed (the run's seed),
        # so they're the same after a resume; frame draws use the worker's RNG.
        self.order_seed, self.batch_size, self.start_batch = order_seed, batch_size, start_batch
        self._shards = {}             # shard folder -> (index, reader), loaded on first use

    def _frame(self, i: int, idx: int) -> torch.Tensor:
        m = self.manifest
        if m.starts is None:
            return read_frame(m.paths[i], idx)
        from src.data.images import decode
        from src.data.shards import ShardReader, load_index
        d = m.paths[i]
        if d not in self._shards:
            self._shards[d] = (load_index(d, fields=("shard", "offset", "size")), ShardReader(d))
        index, reader = self._shards[d]
        return decode(reader.read(index, int(m.starts[i]) + idx - 1))

    def _rng(self):
        info = get_worker_info()
        seed = info.seed if info is not None else (self.seed if self.seed is not None else torch.initial_seed())
        return np.random.default_rng(seed)

    @property
    def items_per_epoch(self) -> int:
        """Samples in one sweep epoch: every video, clips_per_video times."""
        return len(self.manifest) * self.clips_per_video

    def sample(self, rng, video: Optional[int] = None):
        m = self.manifest
        i = int(rng.integers(0, len(m))) if video is None else video
        frames = [self._frame(i, idx) for idx in self.sampler(int(m.lengths[i]), rng)]
        return torch.stack(frames), m.texts[i]

    def __iter__(self):
        rng = self._rng()
        if self.order == "random":
            while True:
                yield self.sample(rng)
        from src.data.order import epoch_stream
        info = get_worker_info()
        wid, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        n = len(self.manifest)
        for _, j in epoch_stream(self.items_per_epoch, self.order_seed, wid, workers,
                                 self.start_batch, self.batch_size):
            yield self.sample(rng, video=j % n)
