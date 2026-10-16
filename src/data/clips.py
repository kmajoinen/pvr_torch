"""
Video clip datasets for pretraining (pretrain.py).

On disk -- R3M's frame layout (its official loader reads the same folders;
running that code on this data would still need its own manifest.csv, with
absolute paths and its 2-character text prefix):

    <root>/<manifest>.csv     one row per video: path, len, txt
    <root>/<path>/000001.jpg  its frames, numbered 1..len, all the same size

`path` is relative to <root> (or absolute); `txt` is the video's description
(may be empty). src/data/prepare_ssv2.py writes this for Something-Something
v2; any other video source only needs a script that writes the same.

ClipDataset samples forever, with replacement: each item is one uniformly
drawn video and the frames its FrameSampler picks (src/losses.py), as
(uint8 tensor (T, 3, H, W), text) -- the same sampling as R3M's R3MBuffer.
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

    def __init__(self, manifest: ClipManifest, sampler, seed: Optional[int] = None):
        self.manifest = manifest
        self.sampler = sampler
        self.seed = seed

    def _rng(self):
        info = get_worker_info()
        seed = info.seed if info is not None else (self.seed if self.seed is not None else torch.initial_seed())
        return np.random.default_rng(seed)

    def sample(self, rng):
        m = self.manifest
        i = int(rng.integers(0, len(m)))
        frames = [read_frame(m.paths[i], idx) for idx in self.sampler(int(m.lengths[i]), rng)]
        return torch.stack(frames), m.texts[i]

    def __iter__(self):
        rng = self._rng()
        while True:
            yield self.sample(rng)
