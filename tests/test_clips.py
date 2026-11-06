"""Clip datasets (src/data/clips.py) and the SSv2 prep script."""

import json
import os

import numpy as np
import pytest
import torch
from torchvision.io import write_jpeg

from src.data.clips import ClipDataset, ClipManifest
from src.losses import R3MFrameSampler


def _dataset(root, lengths, texts=None):
    """Frame folders whose frame i is a constant image of value i."""
    rows = ["path,len,txt"]
    for v, n in enumerate(lengths):
        d = root / "frames" / f"v{v}"
        d.mkdir(parents=True)
        for i in range(1, n + 1):
            write_jpeg(torch.full((3, 16, 16), i * 5, dtype=torch.uint8), str(d / f"{i:06d}.jpg"), quality=100)
        rows.append(f"frames/v{v},{n},{(texts or [''] * len(lengths))[v]}")
    (root / "manifest.csv").write_text("\n".join(rows) + "\n")


def test_manifest_paths_lengths_text(tmp_path):
    _dataset(tmp_path, [2, 6, 9], texts=["C walks", "", "C sits"])
    m = ClipManifest(str(tmp_path), "manifest.csv", min_length=3, text_strip=2)
    assert len(m) == 2 and m.skipped == 1
    assert m.paths[0] == str(tmp_path / "frames" / "v1") and list(m.lengths) == [6, 9]
    assert m.texts == ["", "sits"]          # empty text stays empty (R3M masks it)


def test_numeric_paths_and_texts_stay_strings(tmp_path):
    (tmp_path / "123").mkdir()
    (tmp_path / "manifest.csv").write_text("path,len,txt\n123,5,42\n")
    m = ClipManifest(str(tmp_path), "manifest.csv")
    assert m.paths == [str(tmp_path / "123")] and m.texts == ["42"]


def test_resumed_runs_draw_new_batches(tmp_path):
    # pretrain.py reseeds from (seed, step) before starting its loaders: a run
    # resumed at step 3 must not replay the batches a fresh run starts with,
    # and train/validation loaders must not mirror each other.
    from omegaconf import OmegaConf
    from pretrain import _loader, reseed
    _dataset(tmp_path, [30] * 8)
    cfg = OmegaConf.create({"batch_size": 4})
    data = OmegaConf.create({"root": str(tmp_path), "text_strip": 0})

    def first_batches(step):
        seed = reseed(1, step)
        train = _loader(cfg, data, "manifest.csv", R3MFrameSampler(0.2), 0, torch.device("cpu"), seed)
        val = _loader(cfg, data, "manifest.csv", R3MFrameSampler(0.2), 0, torch.device("cpu"), seed + 1)
        return next(train)[0], next(val)[0]

    fresh_train, fresh_val = first_batches(0)
    assert not torch.equal(fresh_train, fresh_val)
    assert torch.equal(first_batches(0)[0], fresh_train)          # reproducible
    assert not torch.equal(first_batches(3)[0], fresh_train)      # resumed: continues, no replay


def test_dataset_returns_the_sampled_frames_in_order(tmp_path):
    _dataset(tmp_path, [10, 12], texts=["a", "b"])
    ds = ClipDataset(ClipManifest(str(tmp_path), "manifest.csv", min_length=3), R3MFrameSampler(0.2))
    rng = np.random.default_rng(0)
    for _ in range(20):
        frames, text = ds.sample(rng)
        assert frames.shape == (5, 3, 16, 16) and frames.dtype == torch.uint8 and text in ("a", "b")
        idx = (frames[:, 0, 0, 0].float() / 5).round().long().tolist()   # frame number per slot
        start, goal, s0, s1, s2 = idx
        assert s0 < s1 <= s2 and start <= 3


def test_loader_workers_draw_different_samples(tmp_path):
    _dataset(tmp_path, [30] * 6)
    ds = ClipDataset(ClipManifest(str(tmp_path), "manifest.csv"), R3MFrameSampler(0.2))
    torch.manual_seed(0)
    it = iter(torch.utils.data.DataLoader(ds, batch_size=8, num_workers=2))
    a, b = next(it)[0], next(it)[0]          # one batch from each worker
    assert not torch.equal(a, b)


def _fake_videos(root, n, codec="libvpx-vp9"):
    av = pytest.importorskip("av")
    if codec not in av.codecs_available:
        pytest.skip(f"PyAV lacks {codec}")
    (root / "videos").mkdir()
    entries = []
    for v in range(n):
        with av.open(str(root / "videos" / f"{v}.webm"), "w") as c:
            s = c.add_stream(codec, rate=12)
            s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
            for t in range(5 + v):
                img = np.full((48, 64, 3), 20 * t, np.uint8)
                for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
        entries.append({"id": str(v), "label": f"pushing thing {v}", "template": "Pushing [something]"})
    (root / "train.json").write_text(json.dumps(entries))
    return entries


def test_pretrain_rejects_frames_of_another_size():
    from pretrain import as_stored
    frames = torch.full((2, 3, 32, 32), 255, dtype=torch.uint8)
    assert torch.equal(as_stored(frames, 32), torch.ones(2, 3, 32, 32))
    with pytest.raises(ValueError, match="image_size is 224"):
        as_stored(frames, 224)


@pytest.mark.parametrize("old_marker", ["7", ""])
def test_prepare_ssv2_redoes_videos_with_unreadable_markers(tmp_path, old_marker):
    # A bare frame count (the first version's marker) or an empty file must
    # mean "convert again", not a permanent failure.
    pytest.importorskip("cv2")
    from src.data.prepare_ssv2 import main
    _fake_videos(tmp_path, 1)
    out = tmp_path / "out"
    args = ["--videos", str(tmp_path / "videos"), "--labels", str(tmp_path / "train.json"),
            "--out", str(out), "--split", "train", "--size", "32", "--format", "folders", "--workers", "1"]
    (out / "frames" / "0").mkdir(parents=True)
    (out / "frames" / "0" / ".done").write_text(old_marker)
    main(args)
    m = ClipManifest(str(out), "manifest_train.csv")
    assert len(m) == 1 and list(m.lengths) == [5]


@pytest.mark.parametrize("resize", ["squash", "center_crop"])
def test_prepare_ssv2_writes_frames_and_manifest(tmp_path, resize):
    pytest.importorskip("cv2")
    from src.data.prepare_ssv2 import main
    from torchvision.io import read_image
    _fake_videos(tmp_path, 3)
    out = tmp_path / "out"
    args = ["--videos", str(tmp_path / "videos"), "--labels", str(tmp_path / "train.json"),
            "--out", str(out), "--split", "train", "--size", "32", "--resize", resize, "--format", "folders",
            "--workers", "1"]
    main(args)
    m = ClipManifest(str(out), "manifest_train.csv")
    assert list(m.lengths) == [5, 6, 7] and m.texts[2] == "pushing thing 2"
    assert read_image(f"{m.paths[1]}/000006.jpg").shape == (3, 32, 32)
    main(args + ["--text", "template"])                 # rerun: videos skipped, manifest rewritten
    assert ClipManifest(str(out), "manifest_train.csv").texts[0] == "pushing something"
    main(args + ["--size", "24"])                        # other settings: frames reconverted
    assert read_image(f"{m.paths[1]}/000006.jpg").shape == (3, 24, 24)


def test_prepare_ssv2_shards_hold_the_same_frames(tmp_path):
    # The default output: the frames of all videos in a few shard files; the
    # dataset reads them exactly as it reads frame folders.
    pytest.importorskip("cv2")
    from src.data.prepare_ssv2 import main
    _fake_videos(tmp_path, 3)
    common = ["--videos", str(tmp_path / "videos"), "--labels", str(tmp_path / "train.json"),
              "--split", "train", "--size", "32", "--workers", "2"]
    main(common + ["--out", str(tmp_path / "f"), "--format", "folders"])
    main(common + ["--out", str(tmp_path / "s"), "--shard-gb", "1e-6"])        # ~1 KB: several shards
    folders = ClipManifest(str(tmp_path / "f"), "manifest_train.csv")
    shards = ClipManifest(str(tmp_path / "s"), "manifest_train.csv")
    assert shards.starts is not None and list(shards.starts) == [0, 5, 11]
    assert list(shards.lengths) == list(folders.lengths) and shards.texts == folders.texts
    assert len(os.listdir(tmp_path / "s" / "train")) > 2                       # several shards + index
    a, b = ClipDataset(folders, R3MFrameSampler(0.2)), ClipDataset(shards, R3MFrameSampler(0.2))
    for seed in range(5):
        fa, ta = a.sample(np.random.default_rng(seed))
        fb, tb = b.sample(np.random.default_rng(seed))
        assert torch.equal(fa, fb) and ta == tb
