"""Replay buffer (src/rl/buffer.py): episode boundaries, n-step returns,
wrap-around. Observations are their own index (obs = [t]) so sampled
transitions can be checked exactly."""

import numpy as np
import pytest
import torch

import src.rl.buffer as buffer
from src.rl.buffer import ReplayBuffer as _ReplayBuffer
from src.rl.buffer import fs_type as _fs_type

GAMMA = 0.5


@pytest.fixture(autouse=True, params=[("none", "ram"), ("zlib", "ram"), ("none", "disk"), ("zlib", "disk")],
                ids=lambda p: "-".join(p))
def compression(request, monkeypatch, tmp_path):
    """Every test runs on raw and compressed storage (zlib: always available),
    each in RAM and on disk. pytest's folders count as a real disk: on
    diskless cluster nodes /tmp is RAM (tmpfs), which disk storage refuses
    (test_disk_storage_refuses_a_ram_backed_folder checks that)."""
    import functools
    monkeypatch.setattr(buffer, "fs_type", lambda path, mounts="/proc/mounts": "ext4")
    comp, storage = request.param
    monkeypatch.setitem(globals(), "ReplayBuffer",
                        functools.partial(_ReplayBuffer, compression=comp, decode_threads=2,
                                          storage=storage, storage_dir=str(tmp_path / "buf")))
    return comp


def _fill(rb, episodes):
    """episodes: list of (length, ends_with) with ends_with in
    {'terminated', 'truncated', None (still running)}. obs t of episode e is
    100*e + t; reward of step t is 1 + t."""
    for e, (length, ends) in enumerate(episodes):
        for t in range(length):
            last = t == length - 1
            rb.add(np.array([100 * e + t], dtype=np.float32), np.array([100 * e + t + 1], dtype=np.float32),
                   np.zeros(1), 1.0 + t, terminated=last and ends == "terminated",
                   truncated=last and ends == "truncated")


def _all(rb, n=4000):
    np.random.seed(0)
    obs, nxt, _, r, d = rb.sample(n)
    return obs.numpy()[:, 0], nxt.numpy()[:, 0], r.numpy(), d.numpy()


def test_truncated_episode_bootstraps_from_its_real_final_obs():
    rb = ReplayBuffer(100, (1,), 1, "cpu", gamma=GAMMA)
    _fill(rb, [(3, "truncated"), (3, None)])
    obs, nxt, _, d = _all(rb)
    last = obs == 2                                  # last transition of episode 0
    assert last.any()
    assert (nxt[last] == 3).all()                     # its final obs, not 100 (next episode)
    assert np.allclose(d[last], GAMMA)                # truncation still bootstraps
    assert not np.isin(obs, [3]).any()                # final-observation slots are never starts


def test_terminated_episode_gets_zero_discount():
    rb = ReplayBuffer(100, (1,), 1, "cpu", gamma=GAMMA)
    _fill(rb, [(3, "terminated"), (3, None)])
    obs, nxt, _, d = _all(rb)
    assert np.allclose(d[obs == 2], 0.0) and (nxt[obs == 2] == 3).all()
    assert np.allclose(d[obs == 1], GAMMA)


def test_nstep_sums_and_stops_at_episode_end():
    rb = ReplayBuffer(100, (1,), 1, "cpu", nstep=3, gamma=GAMMA)
    _fill(rb, [(5, "terminated"), (6, None)])
    obs, nxt, r, d = _all(rb)
    # start t=0: full 3 steps, rewards 1, 2, 3
    m = obs == 0
    assert np.allclose(r[m], 1 + GAMMA * 2 + GAMMA ** 2 * 3) and (nxt[m] == 3).all()
    assert np.allclose(d[m], GAMMA ** 3)
    # start t=3: steps 3, 4 (terminated) -> 2 steps, zero discount, final obs 5
    m = obs == 3
    assert np.allclose(r[m], 4 + GAMMA * 5) and (nxt[m] == 5).all() and np.allclose(d[m], 0)
    # never reaches into episode 1 from episode 0
    assert not ((obs < 100) & (nxt >= 100)).any()
    # the running episode's last nstep-1 transitions have no complete window yet
    assert not np.isin(obs, [103, 104, 105]).any()


def test_wraparound_keeps_windows_valid():
    rb = ReplayBuffer(10, (1,), 1, "cpu", nstep=2, gamma=GAMMA)
    _fill(rb, [(4, "truncated"), (4, "truncated"), (4, None)])   # 4+1 + 4+1 + 4 = 14 slots > 10
    assert len(rb) == 10
    obs, nxt, r, d = _all(rb)
    # every sampled transition is a real 2-step (or episode-ending) window
    for o, n in zip(obs, nxt):
        assert n == o + 2 or (n == o + 1 and o % 100 == 3)
    assert not np.isin(obs, [0, 1, 2, 3]).any()      # episode 0 was overwritten


def test_too_few_transitions_raises():
    rb = ReplayBuffer(10, (1,), 1, "cpu", nstep=3)
    _fill(rb, [(2, None)])
    with pytest.raises(RuntimeError, match="too few"):
        rb.sample(4)


def test_compressed_storage_returns_the_same_frames():
    # uint8 frames whose content is their slot number: decoded batches must
    # match raw storage exactly (lossless), whichever slots are drawn.
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (40, 8, 8, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(40)
    rb = _ReplayBuffer(100, (8, 8, 3), 1, "cpu", obs_dtype=np.uint8, compression="zlib")
    for t in range(39):
        rb.add(frames[t], frames[t + 1], np.zeros(1), 1.0, False, truncated=t == 38)
    assert rb.stored_bytes == int(rb._len[:len(rb)].sum())
    for _ in range(3):
        obs, nxt, _, _, _ = rb.sample(16)
        ids = obs[:, 0, 0, 0].long()
        assert torch.equal(obs, torch.as_tensor(frames[ids]))
        assert torch.equal(nxt, torch.as_tensor(frames[ids + 1]))


@pytest.mark.parametrize("capacity", [1000, 37])
def test_arena_wraps_and_drops_the_oldest(capacity):
    # Incompressible frames into an arena of ~40 frames: the buffer keeps the
    # newest that fit (fewer than capacity when the arena is the limit), and
    # every sampled frame decodes to what was stored in that slot.
    rng = np.random.default_rng(0)
    shape, frame_bytes = (16, 16, 3), 16 * 16 * 3
    stored = {}
    rb = _ReplayBuffer(capacity, shape, 1, "cpu", obs_dtype=np.uint8, compression="zlib",
                       max_gb=40 * frame_bytes / 1e9)
    for t in range(500):
        f = rng.integers(0, 255, shape, dtype=np.uint8)
        f[0, 0, 0], f[0, 0, 1] = t % 256, t // 256          # frame id
        stored[t] = f
        rb.add(f, f, np.zeros(1), 1.0, False, truncated=t % 50 == 49)
        assert rb.stored_bytes <= rb.arena_bytes and len(rb) <= capacity
        if t > 60 and t % 7 == 0:
            obs, nxt, _, _, _ = rb.sample(8)
            for o in list(obs) + list(nxt):
                tid = int(o[0, 0, 0]) + 256 * int(o[0, 0, 1])
                assert torch.equal(o, torch.as_tensor(stored[tid]))
    assert len(rb) < 45 and len(rb) <= capacity
    assert rb.stored_bytes == int(rb._len[[(rb.oldest + k) % rb.capacity for k in range(len(rb))]].sum())


@pytest.mark.parametrize("max_gb", [None, 2e-4])
def test_arena_memory_stays_near_stored_bytes_after_capacity_wraps(max_gb):
    # Compressible frames: stored bytes are far below the arena. After the
    # buffer reaches capacity, the highest offset ever written (the arena
    # memory in use) must stay near the stored bytes, not creep to the arena
    # end -- and every sampled frame still decodes to its own slot's frame.
    rng = np.random.default_rng(0)
    shape = (32, 32, 3)
    stored = {}
    rb = _ReplayBuffer(200, shape, 1, "cpu", obs_dtype=np.uint8, compression="zlib", max_gb=max_gb)
    biggest = 0
    for t in range(3000):
        f = np.zeros(shape, np.uint8)
        rows = 2 + t % 4                                   # varying compressed sizes
        f[:rows] = rng.integers(0, 255, (rows, 32, 3))
        f[-1, 0, 0], f[-1, 0, 1] = t % 256, t // 256       # frame id
        stored[t] = f
        rb.add(f, f, np.zeros(1), 0.0, False, truncated=t % 50 == 49)
        biggest = max(biggest, int(rb._len[(rb.pos - 1) % rb.capacity]))
        if t > 300 and t % 11 == 0:
            obs, nxt, _, _, _ = rb.sample(8)
            for o in list(obs) + list(nxt):
                assert torch.equal(o, torch.as_tensor(stored[int(o[-1, 0, 0]) + 256 * int(o[-1, 0, 1])]))
    assert rb.high_water <= rb.stored_bytes + 2 * biggest
    assert len(rb) >= 190                                  # a few fewer than capacity at most
    live = sorted((int(rb._off[(rb.oldest + k) % rb.capacity]), int(rb._len[(rb.oldest + k) % rb.capacity]))
                  for k in range(len(rb)))
    assert all(a + la <= b for (a, la), (b, _) in zip(live, live[1:]))   # no two live frames overlap



# ── disk storage, timings, memory check ───────────────────────────────────────

def test_disk_file_is_unlinked_and_sparse(tmp_path):
    rb = _ReplayBuffer(1000, (64, 64, 3), 1, "cpu", obs_dtype=np.uint8, storage="disk",
                       storage_dir=str(tmp_path))
    assert rb.disk_bytes == 1000 * 64 * 64 * 3 and list(tmp_path.iterdir()) == []   # unlinked once mapped
    rb.add(np.ones((64, 64, 3), np.uint8), np.ones((64, 64, 3), np.uint8), np.zeros(1), 0.0, False)
    assert isinstance(rb.obs, np.memmap)


@pytest.mark.parametrize("comp", ["none", "zlib"])
def test_disk_prefetch_keeps_frames_actions_and_next_obs_aligned_through_wraps(tmp_path, comp):
    # obs t = t, next_obs = t + 1, action = t: any frame read from a slot that
    # changed under the prefetching reads would break the alignment.
    rb = _ReplayBuffer(300, (4, 4), 1, "cpu", obs_dtype=np.float32, compression=comp, decode_threads=4,
                       storage="disk", storage_dir=str(tmp_path))
    np.random.seed(0)
    for t in range(3000):
        rb.add(np.full((4, 4), t, np.float32), np.full((4, 4), t + 1, np.float32), np.array([t], np.float32),
               0.0, False, truncated=(t % 97 == 96))
        if t > 100:
            obs, nxt, act, _, _ = rb.sample(32)
            o, n = obs.numpy()[:, 0, 0], nxt.numpy()[:, 0, 0]
            assert np.array_equal(n, o + 1) and np.array_equal(act.numpy()[:, 0], o)
            assert (obs.numpy() == o[:, None, None]).all()                 # whole frames, not torn


def test_disk_needs_the_space(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    monkeypatch.setattr(shutil, "disk_usage", lambda d: namedtuple("u", "total used free")(10, 0, 10))
    with pytest.raises(ValueError, match="GB free"):
        _ReplayBuffer(1000, (64, 64, 3), 1, "cpu", obs_dtype=np.uint8, storage="disk", storage_dir=str(tmp_path))


def test_default_disk_dir_is_the_jobs_tmp(monkeypatch):
    from src.rl.buffer import default_storage_dir
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert default_storage_dir() == "/tmp/12345"


def test_timings_report_and_reset():
    rb = _ReplayBuffer(100, (1,), 1, "cpu")
    _fill(rb, [(50, None)])
    rb.sample(8)
    t = rb.timings()
    assert t["sample_ms"] > 0 and t["add_ms"] > 0
    assert np.isnan(rb.timings()["sample_ms"])                              # reset


@pytest.mark.parametrize("kw, outcome", [
    (dict(), "raises"),                                                     # 1000 x 3 MB raw = 3 GB >= 2 GB
    (dict(compression="zlib", max_gb=1.0), "ok"),
    (dict(compression="zlib", max_gb=2.5), "raises"),
    (dict(compression="zlib"), "note"),                                     # uncapped: only a note
    (dict(storage="disk"), "ok"),                                           # page cache: not counted
])
def test_a_ram_buffer_must_fit_the_jobs_memory(monkeypatch, tmp_path, capsys, kw, outcome):
    import src.rl.buffer as buffer
    monkeypatch.setattr(buffer, "memory_limit_bytes", lambda: 2 * 10**9)
    if kw.get("storage") == "disk":
        kw["storage_dir"] = str(tmp_path)
    make = lambda: _ReplayBuffer(1000, (1000, 1000, 3), 1, "cpu", obs_dtype=np.uint8, **kw)   # noqa: E731
    if outcome == "raises":
        with pytest.raises(ValueError, match="OOM"):
            make()
        monkeypatch.setenv("PVR_SKIP_BUFFER_CHECK", "1")
        make()
    else:
        make()
        assert ("buffer_max_gb=" in capsys.readouterr().out) == (outcome == "note")


def test_fs_type_picks_the_longest_mount_prefix(tmp_path):
    fs_type = _fs_type                                           # the real one, not the fixture's
    mounts = tmp_path / "mounts"
    mounts.write_text("/dev/sda1 / ext4 rw 0 0\n"
                      "tmpfs /tmp tmpfs rw 0 0\n"
                      "/dev/nvme0n1 /tmp/local xfs rw 0 0\n"
                      "tmpfs /tmpfoo tmpfs rw 0 0\n")
    assert fs_type("/tmp/123", str(mounts)) == "tmpfs"
    assert fs_type("/tmp/local/123", str(mounts)) == "xfs"
    assert fs_type("/tmpfoobar", str(mounts)) == "ext4"          # /tmpfoo is not a prefix of /tmpfoobar
    assert fs_type("/home/x", str(mounts)) == "ext4"
    assert fs_type("/x", str(tmp_path / "missing")) == ""


@pytest.mark.parametrize("fstype", ["tmpfs", "ramfs"])
def test_disk_storage_refuses_a_ram_backed_folder(tmp_path, monkeypatch, fstype):
    monkeypatch.setattr(buffer, "fs_type", lambda path, mounts="/proc/mounts": fstype)
    with pytest.raises(ValueError, match=f"on {fstype}, which is RAM"):
        _ReplayBuffer(100, (8, 8, 3), 1, "cpu", obs_dtype=np.uint8, storage="disk", storage_dir=str(tmp_path))


def test_memory_limit_falls_back_to_slurm_per_cpu(monkeypatch):
    import src.rl.buffer as buffer
    monkeypatch.setattr(buffer, "_cgroup_dir", lambda: None)
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4000")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    assert buffer.memory_limit_bytes() == 4000 * 8 << 20
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "60000")
    assert buffer.memory_limit_bytes() == 60000 << 20


def test_memory_limit_takes_the_smallest_up_the_cgroup_tree(monkeypatch, tmp_path):
    import os.path as osp
    import src.rl.buffer as buffer
    root = tmp_path / "sys" / "fs" / "cgroup"
    job = root / "slurm" / "job_1"
    step = job / "step_0"
    step.mkdir(parents=True)
    (step / "memory.max").write_text("max\n")
    (job / "memory.max").write_text(str(24 << 30) + "\n")
    (root / "slurm" / "memory.max").write_text(str(512 << 30) + "\n")
    real_open = open
    to_real = lambda p: str(p).replace("/sys/fs/cgroup", str(root), 1)            # noqa: E731
    monkeypatch.setattr(buffer, "_cgroup_dir", lambda: "/sys/fs/cgroup/slurm/job_1/step_0")
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: real_open(to_real(p), *a, **k))
    assert buffer.memory_limit_bytes() == 24 << 30


def test_disk_space_hint_for_an_uncapped_compressed_arena(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    monkeypatch.setattr(shutil, "disk_usage", lambda d: namedtuple("u", "total used free")(0, 0, 5 * 10**9))
    with pytest.raises(ValueError, match="buffer_max_gb=4"):
        _ReplayBuffer(50_000, (224, 224, 3), 1, "cpu", obs_dtype=np.uint8, compression="zlib",   # 7.5 GB arena
                      storage="disk", storage_dir=str(tmp_path))
