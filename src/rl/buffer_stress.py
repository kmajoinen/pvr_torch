"""
Replay-buffer memory check, without training: fill a ReplayBuffer with real
rendered frames while sampling batches as training does, and print the
process's resident memory next to the bytes the buffer stores. If memory
grows faster than the stored bytes, the buffer is the cause; if it doesn't,
look elsewhere in the training run.

    MUJOCO_GL=egl python -m src.rl.buffer_stress --env adroit_hammer_pixels \
        --image-size 224 --frames 300000 --compression zstd

--env is a configs/env/*.yaml name (pixel builders). A pool of frames is
rendered once with random actions; every added frame is one of them with a
few pixels changed, so each compresses separately at a realistic ratio.
"""

import argparse
import os
import time

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="adroit_hammer_pixels")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--frames", type=int, default=300_000)
    p.add_argument("--capacity", type=int, default=1_000_000)
    p.add_argument("--compression", default="zstd")
    p.add_argument("--max-gb", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sample-every", type=int, default=1, help="sample a batch every N adds (SAC: 1)")
    p.add_argument("--decode-threads", type=int, default=8)
    p.add_argument("--pool", type=int, default=200, help="distinct rendered frames")
    p.add_argument("--report-every", type=int, default=25_000)
    a = p.parse_args(argv)

    from omegaconf import OmegaConf
    from src.rl.buffer import ReplayBuffer, rss_gb
    from src.sac_utils import ENV_BUILDERS

    env_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "env", f"{a.env}.yaml"))
    env = ENV_BUILDERS[env_cfg.builder](env_cfg.id, a.image_size)
    obs, _ = env.reset(seed=0)
    pool = []
    for _ in range(a.pool):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        pool.append(np.asarray(obs))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    shape = pool[0].shape
    print(f"{a.env}: {a.pool} rendered frames {shape}")

    rng = np.random.default_rng(0)
    rb = ReplayBuffer(a.capacity, shape, 1, "cpu", obs_dtype=np.uint8, compression=a.compression,
                      decode_threads=a.decode_threads, max_gb=a.max_gb)
    base = rss_gb()
    print(f"start: rss {base:.2f} GB")
    t0, n_samples = time.time(), 0
    for t in range(1, a.frames + 1):
        f = pool[t % a.pool].copy()
        ys, xs = rng.integers(0, shape[0], 20), rng.integers(0, shape[1], 20)
        f[ys, xs] = rng.integers(0, 255, (20, shape[2]), dtype=np.uint8)
        rb.add(f, f, np.zeros(1), 0.0, False, truncated=t % 500 == 0)
        if t > 2 * a.batch_size and t % a.sample_every == 0:
            rb.sample(a.batch_size)
            n_samples += 1
        if t % a.report_every == 0 or t == a.frames:
            rss, stored = rss_gb(), rb.stored_bytes / 1e9
            print(f"{t:9,} frames  stored {stored:6.2f} GB ({rb.compression_ratio():.1f}x)  "
                  f"rss {rss:6.2f} GB  rss - start - stored {rss - base - stored:6.2f} GB  "
                  f"{t / (time.time() - t0):.0f} adds/s  {n_samples / (time.time() - t0):.1f} batches/s",
                  flush=True)


if __name__ == "__main__":
    main()
