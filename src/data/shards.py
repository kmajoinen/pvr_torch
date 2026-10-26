"""
Sharded storage: many small files (images, video frames) packed into a few
large tar files plus an index, so a dataset of millions of samples is a few
hundred files on disk -- within scratch file quotas, and far easier on a
parallel filesystem than millions of small reads.

    <dir>/shard_000000.tar ...    plain tar files (`tar tf` lists them)
    <dir>/index.npz               per sample: key, shard, offset, size, label

Every sample is one tar member stored as-is (e.g. its JPEG bytes). The index
records where its bytes start in which shard, so a reader fetches any sample
with a single positioned read: random access, no tar parsing at training
time.

    with ShardWriter(dir, max_bytes=1 << 30) as w:
        w.add("n01440764_10026.jpg", jpeg_bytes, label=0)
    index = load_index(dir)
    data = ShardReader(dir).read(index, i)    # bytes of sample i
"""

import io
import os
import tarfile

import numpy as np


class ShardWriter:
    """Appends samples to shard_<n>.tar files of up to about `max_bytes`, and
    writes index.npz on close. Each member's data offset is taken from the
    header tarfile actually writes, and checked against a re-read of every
    finished shard."""

    def __init__(self, directory: str, max_bytes: int = 1 << 30):
        os.makedirs(directory, exist_ok=True)
        self.dir, self.max_bytes = directory, max_bytes
        self.keys, self.shards, self.offsets, self.sizes, self.labels = [], [], [], [], []
        self._tar, self._n = None, -1

    def _open_next(self):
        self._close_shard()
        self._n += 1
        self._tar = tarfile.open(self.path(self._n), "w", format=tarfile.USTAR_FORMAT)

    @property
    def num_shards(self) -> int:
        return self._n + 1

    def path(self, n: int) -> str:
        return os.path.join(self.dir, f"shard_{n:06d}.tar")

    def add(self, key: str, data: bytes, label: int = -1):
        if self._tar is None or self._tar.offset >= self.max_bytes:
            self._open_next()
        info = tarfile.TarInfo(key)
        info.size = len(data)
        header = info.tobuf(self._tar.format, self._tar.encoding, self._tar.errors)
        offset = self._tar.offset + len(header)
        self._tar.addfile(info, io.BytesIO(data))
        self.keys.append(key)
        self.shards.append(self._n)
        self.offsets.append(offset)
        self.sizes.append(len(data))
        self.labels.append(label)

    def _close_shard(self):
        if self._tar is None:
            return
        self._tar.close()
        self._tar = None
        # Verify the recorded offsets against tarfile's own reading of the shard.
        mine = {k: o for k, s, o in zip(self.keys, self.shards, self.offsets) if s == self._n}
        with tarfile.open(self.path(self._n)) as t:
            for m in t:
                if mine.get(m.name) != m.offset_data:
                    raise RuntimeError(f"shard {self._n}: offset of {m.name!r} recorded as "
                                       f"{mine.get(m.name)}, but the tar has it at {m.offset_data}")

    def close(self):
        self._close_shard()
        np.savez(os.path.join(self.dir, "index.npz"),
                 key=np.array(self.keys, dtype=str), shard=np.array(self.shards, dtype=np.int32),
                 offset=np.array(self.offsets, dtype=np.int64), size=np.array(self.sizes, dtype=np.int64),
                 label=np.array(self.labels, dtype=np.int32))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        elif self._tar is not None:
            self._tar.close()


def load_index(directory: str, fields=None) -> dict:
    """The index's arrays; `fields` loads only those (a video-frame index has
    millions of rows, and reading needs only shard, offset and size)."""
    path = os.path.join(directory, "index.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no shard index at {path!r} (expected a ShardWriter output folder)")
    with np.load(path) as z:
        return {k: z[k] for k in (fields or z.files)}


def ordered_map(pool, fn, items, chunksize=64, in_flight=64):
    """fn over chunks of `items` (any iterable, consumed lazily) in a process
    pool; results in order, one per item, with at most `in_flight` chunks
    pending. Executor.map submits everything up front and holds every result
    until it is consumed, so workers that outrun the shard writer would
    buffer the whole dataset in RAM."""
    from collections import deque
    pending, chunk = deque(), []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunksize:
            pending.append(pool.submit(fn, chunk))
            chunk = []
            if len(pending) >= in_flight:
                yield from pending.popleft().result()
    if chunk:
        pending.append(pool.submit(fn, chunk))
    while pending:
        yield from pending.popleft().result()


class ShardReader:
    """Positioned reads from a shard folder. File handles open lazily, per
    process (so each data-loader worker gets its own) and stay open."""

    def __init__(self, directory: str):
        self.dir = directory
        self._fds, self._pid = {}, None

    def read(self, index: dict, i: int) -> bytes:
        if self._pid != os.getpid():                 # forked into a worker: fresh handles
            self._fds, self._pid = {}, os.getpid()
        shard = int(index["shard"][i])
        fd = self._fds.get(shard)
        if fd is None:
            fd = self._fds[shard] = os.open(os.path.join(self.dir, f"shard_{shard:06d}.tar"), os.O_RDONLY)
        size = int(index["size"][i])
        data = os.pread(fd, size, int(index["offset"][i]))
        if len(data) != size:
            raise IOError(f"short read from shard {shard} ({len(data)} of {size} bytes)")
        return data

    def __getstate__(self):                          # don't pickle open descriptors
        return {"dir": self.dir, "_fds": {}, "_pid": None}
