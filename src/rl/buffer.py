"""
Replay buffer for the off-policy scripts: one circular array of
observations (features, state vectors or uint8 frames), n-step returns, and
correct episode boundaries.

Each observation is stored once: a transition's next_obs is the slot after
it (as in SB3's optimize_memory_usage). When an episode ends -- terminated
or truncated -- its final observation goes into one extra slot right after
the last transition, so the last transition's next_obs is the real final
state, not the next episode's first frame. That matters for truncation
(time limits), where the target still bootstraps from next_obs.

sample() returns (obs, next_obs, actions, rewards, discounts) with
    rewards    sum_{i<k} gamma^i r_{t+i}  over k = nstep steps, or fewer if
               the episode ends first (never crossing into the next episode)
    discounts  gamma^k, or 0 if the episode terminated within those steps
so every agent forms its target as  rewards + discounts * V(next_obs).
nstep=1 is the plain one-step target.

compression (pixel buffers): each observation is compressed losslessly when
added (zstd, lz4, or zlib from the standard library) and copied into one
preallocated byte block (the arena), used as a ring: frames are appended in
order, and when the write position wraps, the oldest frames in the way are
dropped from the buffer first. Once frames are dropped (the buffer is at
capacity), writing restarts at the front as soon as there's room there, so
the arena pages ever written -- the memory it uses -- stay at about the
stored bytes plus one frame, independent of how the allocator would handle a
million separate objects; max_gb (the arena size) is a hard cap.
Sampled batches are decompressed in a thread pool (these codecs release the
GIL). Batches are then prefetched: sample() hands out the batch prepared at
the previous call -- its slots and their bytes chosen then, on the calling
thread, so nothing races with add() -- and starts decoding the next one,
overlapping it with the caller's update. Rendered frames compress ~3-5x at
224px (measured: DMC cheetah ~5x, Adroit hammer ~3.7x); a 1M-frame 224px
buffer takes ~30-40 GB instead of 151 GB. Without max_gb the arena is the
uncompressed size plus codec overhead (address space only: pages count as
memory once written), so `capacity` is the limit, as without compression;
max_gb makes the arena smaller, and then the buffer holds the newest
observations that fit in it.

storage="disk": the observations (raw frames, or the compressed arena) live
in a memory-mapped file in storage_dir -- meant for a node-local SSD --
instead of process memory. The file is unlinked as soon as it's mapped, so
its space is freed when the process ends, even after a crash; it's sparse,
so disk blocks are taken only as frames are written. Pages read or written
recently stay in the page cache, which the kernel reclaims under the job's
memory limit, so a small --mem holds only the hot part of the buffer.
Sampled batches are read (and decompressed) in the thread pool and
prefetched as above, so page faults happen off the training thread; add()
waits for those reads before it writes, so a slot can't be overwritten
while it's being read.

Timing: sample() and add() keep their wall time (what the caller waits;
sample() includes the copy to the device); timings() reports the mean per
call since the last report.
"""

import os
import time

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

COMPRESSIONS = ("none", "zstd", "lz4", "zlib")
STORAGES = ("ram", "disk")


def rss_gb() -> float:
    """This process's resident memory (Linux), in GB; NaN elsewhere."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def process_memory_gb() -> dict:
    """This process's resident memory split by kind (Linux; NaN elsewhere):
    anon (heap, arrays, a RAM buffer) and file (memory-mapped files, e.g. a
    disk buffer's cached pages), in GB."""
    out = {"rss_anon": float("nan"), "rss_file": float("nan")}
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    out["rss_anon"] = int(line.split()[1]) / 1e6
                elif line.startswith("RssFile:"):
                    out["rss_file"] = int(line.split()[1]) / 1e6
    except OSError:
        pass
    return out


def _cgroup_dir() -> Optional[str]:
    """This process's memory cgroup folder (v2, or v1's memory controller)."""
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":                                  # v2
            d = "/sys/fs/cgroup" + path
            if os.path.isfile(os.path.join(d, "memory.current")):
                return d
        elif "memory" in controllers.split(","):               # v1
            d = "/sys/fs/cgroup/memory" + path
            if os.path.isdir(d):
                return d
    return None


def cgroup_memory_gb() -> float:
    """The job's memory use as the OOM killer counts it (its cgroup: anon
    memory plus page cache), in GB; NaN if unavailable."""
    d = _cgroup_dir()
    for name in ("memory.current", "memory.usage_in_bytes"):
        try:
            with open(os.path.join(d, name)) as f:
                return int(f.read()) / 1e9
        except (OSError, TypeError, ValueError):
            continue
    return float("nan")


def memory_limit_bytes() -> Optional[int]:
    """The job's memory limit: the smallest limit on its cgroup or any parent
    (Slurm may set it on the job's cgroup while the step's reports none),
    else Slurm's --mem / --mem-per-cpu; None if unknown."""
    limits, d = [], _cgroup_dir()
    root = "/sys/fs/cgroup"
    while d and d.startswith(root):
        for name in ("memory.max", "memory.limit_in_bytes"):
            try:
                with open(os.path.join(d, name)) as f:
                    v = f.read().strip()
                if v != "max" and int(v) < 1 << 60:              # v1 reports "no limit" as a huge number
                    limits.append(int(v))
            except (OSError, ValueError):
                continue
        if d.rstrip("/") in (root, root + "/memory"):
            break
        d = os.path.dirname(d.rstrip("/"))
    if limits:
        return min(limits)
    mb = os.environ.get("SLURM_MEM_PER_NODE")
    if mb and mb.isdigit():
        return int(mb) << 20
    per_cpu, cpus = os.environ.get("SLURM_MEM_PER_CPU"), os.environ.get("SLURM_CPUS_ON_NODE")
    if per_cpu and per_cpu.isdigit() and cpus and cpus.isdigit():
        return int(per_cpu) * int(cpus) << 20
    return None


def fs_type(path: str, mounts: str = "/proc/mounts") -> str:
    """Filesystem type of the mount holding `path` (Linux; '' if unknown)."""
    path, best = os.path.realpath(path), ("", "")
    try:
        with open(mounts) as f:
            for line in f:
                fields = line.split()
                if len(fields) < 3:
                    continue
                mnt, fstype = fields[1], fields[2]
                if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best[0]):
                    best = (mnt, fstype)
    except OSError:
        pass
    return best[1]


def default_storage_dir() -> str:
    """Where a disk buffer goes without algo.buffer_dir: the job's folder on
    the node's local disk (/tmp/$SLURM_JOB_ID, as the job scripts use), or
    the system temp dir outside Slurm."""
    import tempfile
    job = os.environ.get("SLURM_JOB_ID")
    return os.path.join("/tmp", job) if job else tempfile.gettempdir()


def make_codec(name: str, level: Optional[int] = None):
    """(compress, decompress) for bytes. compress runs on the adding thread;
    decompress may run on several threads at once."""
    if name == "zlib":
        import zlib
        lvl = 1 if level is None else level
        return (lambda b: zlib.compress(b, lvl)), zlib.decompress
    if name == "lz4":
        try:
            import lz4.frame as lz4f
        except ImportError as e:
            raise ImportError("buffer compression lz4 needs: pip install lz4") from e
        return (lambda b: lz4f.compress(b, compression_level=level or 0)), lz4f.decompress
    if name == "zstd":
        try:
            import zstandard
        except ImportError as e:
            raise ImportError("buffer compression zstd needs: pip install zstandard") from e
        cctx = zstandard.ZstdCompressor(level=3 if level is None else level)
        local = threading.local()          # decompressor objects aren't shareable across threads

        def decompress(b):
            d = getattr(local, "d", None)
            if d is None:
                d = local.d = zstandard.ZstdDecompressor()
            return d.decompress(b)
        return cctx.compress, decompress
    raise ValueError(f"buffer compression {name!r}; expected one of {COMPRESSIONS}")


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_dim: int,
        device,
        obs_dtype=np.float32,
        nstep: int = 1,
        gamma: float = 0.99,
        compression: str = "none",
        compression_level: Optional[int] = None,
        decode_threads: int = 8,
        max_gb: Optional[float] = None,
        storage: str = "ram",
        storage_dir: Optional[str] = None,
    ):
        if nstep < 1:
            raise ValueError(f"nstep must be >= 1, got {nstep}")
        if compression not in COMPRESSIONS:
            raise ValueError(f"buffer compression {compression!r}; expected one of {COMPRESSIONS}")
        if storage not in STORAGES:
            raise ValueError(f"buffer storage {storage!r}; expected one of {STORAGES}")
        self.storage = storage
        self.storage_dir = (storage_dir or default_storage_dir()) if storage == "disk" else None
        self.disk_bytes = 0                         # size of the backing file(s), storage=disk
        self._t_sample = self._t_add = 0.0
        self._n_sample = self._n_add = 0
        self.capacity = capacity
        self.device = device
        self.nstep = nstep
        self.gamma = gamma
        self.compression = compression
        self.obs_shape, self.obs_dtype = tuple(obs_shape), np.dtype(obs_dtype)
        self.stored_bytes = 0                       # compressed observation bytes held
        self._raw_frame_bytes = int(np.prod(obs_shape)) * self.obs_dtype.itemsize
        self.arena_bytes = (int(max_gb * 1e9) if max_gb is not None
                            else capacity * (self._raw_frame_bytes + 64))
        # Pre-flight estimate BEFORE allocating, so an oversized
        # buffer_size/image_size combination announces itself instead of
        # dying inside np.zeros.
        gb = capacity * self._raw_frame_bytes / 1e9
        print(
            f"Replay buffer: {capacity:,} x {obs_shape} {self.obs_dtype.name}"
            + (f"  ({gb:.1f} GB for obs; next_obs derived, not stored separately)" if compression == "none"
               else f"  ({gb:.1f} GB uncompressed; {compression}, lossless, into a "
                    f"{self.arena_bytes / 1e9:.1f} GB arena{' (buffer_max_gb)' if max_gb else ''}; "
                    "stored size logged as it fills)")
            + (f"  nstep={nstep}" if nstep > 1 else "")
            + (f"  on disk in {self.storage_dir}" if storage == "disk" else "")
        )
        self._check_fits(capacity, max_gb)
        self._pending = None
        if compression == "none":
            self.obs = self._alloc("obs", (capacity, *obs_shape), obs_dtype)
        else:
            # The arena's pages are only backed by memory (or disk) once written.
            self._arena = self._alloc("arena", (self.arena_bytes,), np.uint8)
            self._off = np.zeros(capacity, dtype=np.int64)     # each slot's bytes in the arena
            self._len = np.zeros(capacity, dtype=np.int64)
            self._write_off = 0
            self.high_water = 0
            self._compress, self._decompress = make_codec(compression, compression_level)
        if compression != "none" or storage == "disk":
            self._pool = ThreadPoolExecutor(max(1, decode_threads))
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.done = np.zeros(capacity, dtype=bool)      # episode ended after this transition
        self.is_final = np.zeros(capacity, dtype=bool)  # slot holds only a final observation
        self.pos, self.count = 0, 0        # next slot to write; valid slots (ending at pos)

    def _check_fits(self, capacity: int, max_gb: Optional[float]):
        """In RAM, the buffer must fit in the job's memory with room to spare
        for everything else (CUDA, the encoder, the renderer). Fails at
        startup when it certainly won't fit, instead of an OOM kill hours in;
        warns when it leaves under 8 GB. A disk buffer's cached pages are
        reclaimable, so only its disk space is checked (in _alloc).
        PVR_SKIP_BUFFER_CHECK=1 skips this."""
        limit = memory_limit_bytes()
        if self.storage != "ram" or limit is None or os.environ.get("PVR_SKIP_BUFFER_CHECK") == "1":
            return
        raw = capacity * self._raw_frame_bytes
        if self.compression == "none":
            need, what = raw, "the uncompressed buffer"
        elif max_gb is not None:
            need, what = self.arena_bytes, "algo.buffer_max_gb"
        else:
            if raw > limit - (8 << 30):
                print(f"Note: compressed, the buffer's size depends on the frames; it's capped only by "
                      f"its capacity ({raw / 1e9:.0f} GB uncompressed) and the job has {limit / 1e9:.0f} GB. "
                      f"algo.buffer_max_gb={max(1, (limit - (10 << 30)) // 10**9)} would bound it.")
            return
        if need >= limit:
            raise ValueError(
                f"{what} needs {need / 1e9:.1f} GB, but the job's memory limit is {limit / 1e9:.1f} GB: it "
                "would be OOM-killed once the buffer fills. Request more memory, shrink the buffer "
                "(algo.buffer_size, env.image_size, algo.buffer_max_gb), compress it, or use "
                "algo.buffer_storage=disk. (PVR_SKIP_BUFFER_CHECK=1 skips this check.)")
        if need > limit - (8 << 30):
            print(f"WARNING: {what} needs {need / 1e9:.1f} GB of the job's {limit / 1e9:.1f} GB, leaving "
                  "under 8 GB for everything else; an OOM kill is likely once the buffer fills.")

    def _alloc(self, name: str, shape: tuple, dtype):
        """RAM: an ordinary array (pages backed once written). Disk: a
        memory-mapped file in storage_dir, unlinked once mapped (its space is
        freed when the process ends, even after a crash) and sparse (disk
        blocks are taken as frames are written)."""
        if self.storage == "ram":
            return np.zeros(shape, dtype=dtype) if name == "obs" else np.empty(shape, dtype=dtype)
        import shutil
        import tempfile
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        os.makedirs(self.storage_dir, exist_ok=True)
        fstype = fs_type(self.storage_dir)
        if fstype in ("tmpfs", "ramfs"):
            raise ValueError(
                f"buffer storage=disk: {self.storage_dir} is on {fstype}, which is RAM: the buffer would count "
                "against the job's memory and couldn't be reclaimed. Point algo.buffer_dir at a node-local disk "
                "(on Slurm: --tmp=<size> --constraint=localdisk, and check `findmnt -T <dir>`)")
        free = shutil.disk_usage(self.storage_dir).free
        if nbytes > free:
            # A sparse file only fills as frames arrive, but running out of disk
            # mid-run kills the process (SIGBUS on the mapped write), so the
            # whole file must fit. A compressed arena is sized uncompressed
            # unless buffer_max_gb caps it.
            hint = (f"; or cap the compressed arena: algo.buffer_max_gb={max(1, free // 10**9 - 1)}"
                    if name == "arena" else "")
            raise ValueError(
                f"buffer storage=disk needs up to {nbytes / 1e9:.1f} GB in {self.storage_dir}, which has "
                f"{free / 1e9:.1f} GB free: request more local disk (--tmp), shrink the buffer, or compress it"
                + hint)
        fd, path = tempfile.mkstemp(prefix=f"replay_{name}_", suffix=".bin", dir=self.storage_dir)
        try:
            os.ftruncate(fd, nbytes)
            arr = np.memmap(path, dtype=dtype, mode="r+", shape=shape)
        finally:
            os.close(fd)
            os.unlink(path)
        self.disk_bytes += nbytes
        return arr

    def timings(self, reset: bool = True) -> dict:
        """Mean wall time per sample() and add() call, in ms, since the last
        reset (NaN if none)."""
        out = {"sample_ms": 1000 * self._t_sample / self._n_sample if self._n_sample else float("nan"),
               "add_ms": 1000 * self._t_add / self._n_add if self._n_add else float("nan")}
        if reset:
            self._t_sample = self._t_add = 0.0
            self._n_sample = self._n_add = 0
        return out

    @property
    def full(self) -> bool:
        return self.count == self.capacity

    @property
    def oldest(self) -> int:
        return (self.pos - self.count) % self.capacity

    def _drop_oldest(self):
        if self.compression != "none":
            self.stored_bytes -= int(self._len[self.oldest])
        self.count -= 1

    def _store_compressed(self, i, obs):
        blob = self._compress(np.ascontiguousarray(obs, dtype=self.obs_dtype).tobytes())
        n = len(blob)
        if n > self.arena_bytes:
            raise ValueError(f"one compressed observation ({n:,} bytes) exceeds the buffer arena "
                             f"({self.arena_bytes:,} bytes): raise algo.buffer_max_gb")
        if self.count == self.capacity:    # slot i is the oldest one: it's replaced
            self._drop_oldest()
        if self.count == 0:
            self._write_off = 0
        elif self._write_off > self._off[self.oldest] >= n:
            # The live frames are one run, [off[oldest], write_off), with room
            # for this frame in front of it (left by dropped frames): restart
            # there, so writing cycles through about the stored bytes instead
            # of moving on into pages never used before.
            self._write_off = 0
        if self._write_off + n > self.arena_bytes:
            # Wrap: the frames still beyond the old write position are the
            # oldest ones left from the previous lap; drop them first.
            while self.count and self._off[self.oldest] >= self._write_off:
                self._drop_oldest()
            self._write_off = 0
        w = self._write_off
        # Drop the oldest frames this write overlaps (the arena is FIFO, so
        # they're always the oldest ones).
        while self.count and self._off[self.oldest] < w + n and w < self._off[self.oldest] + self._len[self.oldest]:
            self._drop_oldest()
        self._arena[w:w + n] = np.frombuffer(blob, dtype=np.uint8)
        self._off[i], self._len[i] = w, n
        self._write_off = w + n
        self.high_water = max(self.high_water, self._write_off)   # arena bytes ever written
        self.stored_bytes += n

    def _write(self, obs, action, reward, terminated, done, is_final):
        i = self.pos
        if self.compression == "none":
            self.obs[i] = obs
            if self.count == self.capacity:
                self.count -= 1
        else:
            self._store_compressed(i, obs)
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = terminated
        self.done[i] = done
        self.is_final[i] = is_final
        self.pos = (self.pos + 1) % self.capacity
        self.count += 1

    def compression_ratio(self) -> float:
        n = len(self)
        return n * self._raw_frame_bytes / self.stored_bytes if self.stored_bytes else 1.0

    def add(self, obs, next_obs, action, reward, terminated, truncated=False):
        t0 = time.perf_counter()
        if self.storage == "disk":
            self._wait_reads()          # the prefetched batch's reads finish before any slot changes
        done = bool(terminated or truncated)
        self._write(obs, action, reward, bool(terminated), done, False)
        if done:
            # The episode's final observation, which no later add() supplies.
            self._write(next_obs, 0.0, 0.0, False, True, True)
        self._t_add += time.perf_counter() - t0
        self._n_add += 1

    def __len__(self):
        return self.count

    def _returns(self, idx):
        """n-step reward sums, discounts and next indices for start slots idx."""
        cap = self.capacity
        rewards = np.zeros(len(idx), dtype=np.float32)
        steps = np.zeros(len(idx), dtype=np.int64)
        terminal = np.zeros(len(idx), dtype=bool)
        alive = np.ones(len(idx), dtype=bool)
        for i in range(self.nstep):
            cur = (idx + i) % cap
            rewards += alive * (self.gamma ** i) * self.rewards[cur]
            steps += alive
            terminal |= alive & self.terminated[cur]
            alive &= ~self.done[cur]
        discounts = (self.gamma ** steps) * ~terminal
        return rewards, discounts.astype(np.float32), (idx + steps) % cap, steps

    def _indices(self, batch_size: int):
        size = len(self)
        oldest = self.oldest
        idx = np.empty(batch_size, dtype=np.int64)
        missing = np.arange(batch_size)
        for _ in range(100):
            cand = (oldest + np.random.randint(0, size, size=len(missing))) % self.capacity
            _, _, _, steps = self._returns(cand)
            # A start needs to be a transition (not a final-observation slot)
            # whose next observation has already been written, without the
            # window wrapping past the newest slot.
            logical = (cand - oldest) % self.capacity
            ok = ~self.is_final[cand] & (logical + steps <= size - 1)
            idx[missing[ok]] = cand[ok]
            missing = missing[~ok]
            if not len(missing):
                break
        else:
            raise RuntimeError("replay buffer has too few complete transitions to sample from")
        rewards, discounts, next_idx, _ = self._returns(idx)
        return idx, next_idx, rewards, discounts

    def _to(self, x):
        return torch.as_tensor(x, device=self.device)

    def sample(self, batch_size: int):
        t0 = time.perf_counter()
        try:
            if self.compression != "none" or self.storage == "disk":
                return self._sample_prefetched(batch_size)
            idx, next_idx, rewards, discounts = self._indices(batch_size)
            return (self._to(self.obs[idx]), self._to(self.obs[next_idx]), self._to(self.actions[idx]),
                    self._to(rewards), self._to(discounts))
        finally:
            self._t_sample += time.perf_counter() - t0
            self._n_sample += 1

    # ---- prefetched sampling (compressed and/or disk storage) ---------------------

    def _decode_into(self, out, blobs, start):
        for j, b in enumerate(blobs):
            out[start + j] = np.frombuffer(self._decompress(b), dtype=self.obs_dtype).reshape(self.obs_shape)

    def _read_into(self, out, slots, start):
        """Disk, uncompressed: copy frames out of the memory-mapped file."""
        for j, i in enumerate(slots):
            out[start + j] = self.obs[i]

    def _read_decode_into(self, out, spans, start):
        """Disk, compressed: read each frame's bytes from the mapped arena, then decode."""
        for j, (o, n) in enumerate(spans):
            out[start + j] = np.frombuffer(self._decompress(self._arena[o:o + n].tobytes()),
                                           dtype=self.obs_dtype).reshape(self.obs_shape)

    def _wait_reads(self):
        if self._pending is not None:
            for f in self._pending[2]:
                f.result()

    def _prepare(self, batch_size: int):
        """Choose a batch now (on this thread) and start reading/decoding its
        frames in the pool."""
        idx, next_idx, rewards, discounts = self._indices(batch_size)
        slots = np.concatenate([idx, next_idx])
        if self.compression == "none":                 # disk: the threads read the frames
            work, fn = slots, self._read_into
        elif self.storage == "ram":
            # Copied out now, on this thread: a wrap can't overwrite them mid-decode.
            work = [self._arena[self._off[i]:self._off[i] + self._len[i]].tobytes() for i in slots]
            fn = self._decode_into
        else:
            # Disk: the threads read the bytes too, so page faults stay off this
            # thread; add() waits for them before writing (no overwrite mid-read).
            work, fn = list(zip(self._off[slots].tolist(), self._len[slots].tolist())), self._read_decode_into
        out = np.empty((len(slots), *self.obs_shape), dtype=self.obs_dtype)
        chunk = 16
        futures = [self._pool.submit(fn, out, work[s:s + chunk], s) for s in range(0, len(work), chunk)]
        return batch_size, out, futures, self.actions[idx].copy(), rewards, discounts

    def _sample_prefetched(self, batch_size: int):
        if self._pending is None or self._pending[0] != batch_size:
            self._pending = self._prepare(batch_size)
        n, out, futures, actions, rewards, discounts = self._pending
        self._pending = self._prepare(batch_size)        # decodes while the caller trains
        for f in futures:
            f.result()
        return (self._to(out[:n]), self._to(out[n:]), self._to(actions),
                self._to(rewards), self._to(discounts))
