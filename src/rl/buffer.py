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
added (zstd, lz4, or zlib from the standard library) and kept as bytes, and
sampled batches are decompressed in a thread pool (these codecs release the
GIL). Batches are then prefetched: sample() hands out the batch prepared at
the previous call -- its slots and their bytes chosen then, on the calling
thread, so nothing races with add() -- and starts decoding the next one,
overlapping it with the caller's update. Rendered frames compress ~3-5x at
224px (measured: DMC cheetah ~5x, Adroit hammer ~3.7x); a 1M-frame 224px
buffer takes ~30-40 GB instead of 151 GB. max_gb caps the compressed
storage: once reached, the buffer stops growing and wraps from there.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

COMPRESSIONS = ("none", "zstd", "lz4", "zlib")


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
    ):
        if nstep < 1:
            raise ValueError(f"nstep must be >= 1, got {nstep}")
        if compression not in COMPRESSIONS:
            raise ValueError(f"buffer compression {compression!r}; expected one of {COMPRESSIONS}")
        self.capacity = capacity
        self.device = device
        self.nstep = nstep
        self.gamma = gamma
        self.compression = compression
        self.obs_shape, self.obs_dtype = tuple(obs_shape), np.dtype(obs_dtype)
        self.max_bytes = None if max_gb is None else int(max_gb * 1e9)
        self.stored_bytes = 0                       # compressed observation bytes held
        self._raw_frame_bytes = int(np.prod(obs_shape)) * self.obs_dtype.itemsize
        # Pre-flight estimate BEFORE allocating, so an oversized
        # buffer_size/image_size combination announces itself instead of
        # dying inside np.zeros.
        gb = capacity * self._raw_frame_bytes / 1e9
        print(
            f"Replay buffer: {capacity:,} x {obs_shape} {self.obs_dtype.name}"
            + (f"  ({gb:.1f} GB for obs; next_obs derived, not stored separately)" if compression == "none"
               else f"  ({gb:.1f} GB uncompressed; {compression}, lossless, stored size logged as it fills"
                    + (f", capped at {max_gb} GB" if max_gb else "") + ")")
            + (f"  nstep={nstep}" if nstep > 1 else "")
        )
        if compression == "none":
            self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        else:
            self.obs = [b""] * capacity             # compressed bytes per slot
            self._compress, self._decompress = make_codec(compression, compression_level)
            self._pool = ThreadPoolExecutor(max(1, decode_threads))
            self._pending = None
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.done = np.zeros(capacity, dtype=bool)      # episode ended after this transition
        self.is_final = np.zeros(capacity, dtype=bool)  # slot holds only a final observation
        self.pos, self.full = 0, False

    def _write(self, obs, action, reward, terminated, done, is_final):
        i = self.pos
        if self.compression == "none":
            self.obs[i] = obs
        else:
            blob = self._compress(np.ascontiguousarray(obs, dtype=self.obs_dtype).tobytes())
            self.stored_bytes += len(blob) - len(self.obs[i])
            self.obs[i] = blob
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = terminated
        self.done[i] = done
        self.is_final[i] = is_final
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0
        if self.max_bytes is not None and not self.full and self.stored_bytes >= self.max_bytes:
            # Storage cap reached: stop growing, wrap around from here.
            print(f"Replay buffer: {self.stored_bytes / 1e9:.1f} GB reached (buffer max_gb) at "
                  f"{self.pos:,} observations; capacity fixed there, oldest overwritten from now on")
            self.capacity, self.pos, self.full = self.pos, 0, True

    def compression_ratio(self) -> float:
        n = len(self)
        return n * self._raw_frame_bytes / self.stored_bytes if self.stored_bytes else 1.0

    def add(self, obs, next_obs, action, reward, terminated, truncated=False):
        done = bool(terminated or truncated)
        self._write(obs, action, reward, bool(terminated), done, False)
        if done:
            # The episode's final observation, which no later add() supplies.
            self._write(next_obs, 0.0, 0.0, False, True, True)

    def __len__(self):
        return self.capacity if self.full else self.pos

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
        oldest = self.pos if self.full else 0
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
        if self.compression != "none":
            return self._sample_compressed(batch_size)
        idx, next_idx, rewards, discounts = self._indices(batch_size)
        return (self._to(self.obs[idx]), self._to(self.obs[next_idx]), self._to(self.actions[idx]),
                self._to(rewards), self._to(discounts))

    # ---- compressed storage -----------------------------------------------------

    def _decode_into(self, out, blobs, start):
        for j, b in enumerate(blobs):
            out[start + j] = np.frombuffer(self._decompress(b), dtype=self.obs_dtype).reshape(self.obs_shape)

    def _prepare(self, batch_size: int):
        """Choose a batch now (on this thread) and start decoding its frames."""
        idx, next_idx, rewards, discounts = self._indices(batch_size)
        blobs = [self.obs[i] for i in idx] + [self.obs[i] for i in next_idx]
        out = np.empty((len(blobs), *self.obs_shape), dtype=self.obs_dtype)
        chunk = 16
        futures = [self._pool.submit(self._decode_into, out, blobs[s:s + chunk], s)
                   for s in range(0, len(blobs), chunk)]
        return batch_size, out, futures, self.actions[idx].copy(), rewards, discounts

    def _sample_compressed(self, batch_size: int):
        if self._pending is None or self._pending[0] != batch_size:
            self._pending = self._prepare(batch_size)
        n, out, futures, actions, rewards, discounts = self._pending
        self._pending = self._prepare(batch_size)        # decodes while the caller trains
        for f in futures:
            f.result()
        return (self._to(out[:n]), self._to(out[n:]), self._to(actions),
                self._to(rewards), self._to(discounts))
