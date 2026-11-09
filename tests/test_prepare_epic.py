"""EPIC-KITCHENS converter (src/data/prepare_epic.py): resampling, segment
ranges, the manifests' agreement with the shards, and the checker's verdict."""

import numpy as np
import pandas as pd
import pytest

from src.data.prepare_epic import main

RATE = 25          # source fps; frame i (from 0) is gray level 5 * i, at time i / 25


def _video(path, frames, rate=RATE, codec="libx264"):
    av = pytest.importorskip("av")
    if codec not in av.codecs_available:
        pytest.skip(f"PyAV lacks {codec}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mp4") as c:
        s = c.add_stream(codec, rate=rate)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.options = {"crf": "0"} if codec == "libx264" else {}
        for i in range(frames):
            img = np.full((48, 64, 3), 5 * i, np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def _ts(s):
    return f"00:00:{s:05.2f}"


def _dataset(tmp_path, videos, segments):
    """videos: {id: frame count}; segments: (video id, start s, stop s, narration)."""
    for vid, n in videos.items():
        _video(tmp_path / "EPIC" / vid.split("_")[0] / "videos" / f"{vid}.MP4", n)
    rows = [{"narration_id": f"{v}_{k}", "participant_id": v.split("_")[0], "video_id": v,
             "start_timestamp": _ts(a), "stop_timestamp": _ts(b), "narration": t}
            for k, (v, a, b, t) in enumerate(segments)]
    pd.DataFrame(rows).to_csv(tmp_path / "EPIC_100_train.csv", index=False)


def _run(tmp_path, *extra):
    main(["--videos", str(tmp_path / "EPIC"), "--annotations", str(tmp_path), "--out", str(tmp_path / "out"),
          "--split", "train", "--size", "16", "--workers", "2", *extra])
    return pd.read_csv(tmp_path / "out" / "manifest_train.csv", keep_default_na=False)


def _stored_levels(out, split="train"):
    from src.data.images import decode
    from src.data.shards import ShardReader, load_index
    d = str(out / split)
    index, reader = load_index(d, fields=("key", "shard", "offset", "size")), ShardReader(d)
    keys = [k.decode() if isinstance(k, bytes) else str(k) for k in index["key"]]
    return keys, [float(decode(reader.read(index, i)).float().mean()) for i in range(len(keys))]


def test_frames_resampled_on_the_fps_grid(tmp_path):
    _dataset(tmp_path, {"P01_01": 50}, [("P01_01", 0.3, 0.9, "open door")])
    _run(tmp_path, "--fps", "5")
    keys, levels = _stored_levels(tmp_path / "out")
    # 2 s at 5 fps: ticks 0, 0.2, ..., 1.8 -> source frames 0, 5, ..., 45
    assert keys == [f"P01_01/{n:06d}.jpg" for n in range(1, 11)]
    np.testing.assert_allclose(levels, [25 * k for k in range(10)], atol=3)
    times = np.load(tmp_path / "out" / "frame_times_train.npz")["time"]
    np.testing.assert_allclose(times, np.arange(10) * 0.2, atol=1e-6)


def test_segments_index_their_frames(tmp_path):
    _dataset(tmp_path, {"P01_01": 50, "P02_03": 30},
             [("P01_01", 0.3, 0.9, "open door"), ("P02_03", 0.0, 1.1, "take cup"),
              ("P01_01", 0.5, 1.5, "close door"), ("P01_01", 0.41, 0.59, "too short")])
    m = _run(tmp_path, "--fps", "5")
    keys, levels = _stored_levels(tmp_path / "out")
    assert list(m["txt"]) == ["open door", "close door", "take cup"]       # grouped by video; empty one dropped
    assert list(m["len"]) == [3, 5, 6]                                     # ticks in [0.3,0.9], [0.5,1.5], [0,1.1]
    for _, r in m.iterrows():
        first, last = keys[r["start"]], keys[r["start"] + r["len"] - 1]
        assert first.split("/")[0] == last.split("/")[0] == r["video_id"]
    # open door's first frame is tick 0.4 s = source frame 10
    assert abs(levels[m["start"][0]] - 50) < 3
    videos = pd.read_csv(tmp_path / "out" / "manifest_train_videos.csv", keep_default_na=False)
    assert list(videos["video_id"]) == ["P01_01", "P02_03"] and list(videos["len"]) == [10, 6]
    assert set(videos["txt"]) == {""}


def test_fps_zero_keeps_every_frame(tmp_path):
    _dataset(tmp_path, {"P01_01": 20}, [("P01_01", 0.0, 0.4, "wash")])
    m = _run(tmp_path, "--fps", "0")
    assert list(m["len"]) == [11]                                          # frames at 0, 0.04, ..., 0.4


def test_missing_video_leaves_out_its_segments(tmp_path, capsys):
    _dataset(tmp_path, {"P01_01": 25}, [("P01_01", 0.0, 0.9, "stir"), ("P09_01", 0.0, 1.0, "gone")])
    m = _run(tmp_path, "--fps", "5")
    assert list(m["video_id"]) == ["P01_01"]
    out = capsys.readouterr().out
    assert "1 of 2 videos missing" in out and "1 segments belong to videos that weren't stored" in out


def test_manifests_only_rebuilds_the_same_rows(tmp_path):
    _dataset(tmp_path, {"P01_01": 50}, [("P01_01", 0.3, 0.9, "open door"), ("P01_01", 1.0, 1.9, "pour")])
    first = _run(tmp_path, "--fps", "5")
    (tmp_path / "out" / "manifest_train.csv").unlink()
    again = _run(tmp_path, "--fps", "5", "--manifests-only")
    pd.testing.assert_frame_equal(first, again)
    none = _run(tmp_path, "--fps", "5", "--manifests-only", "--text", "none")
    assert set(none["txt"]) == {""}


def test_check_and_loader_accept_the_output(tmp_path):
    from src.data.check import check_dataset
    from src.data.clips import ClipDataset, ClipManifest
    from src.losses import R3MFrameSampler
    _dataset(tmp_path, {"P01_01": 50, "P01_02": 40},
             [("P01_01", 0.0, 1.0, "open door"), ("P01_02", 0.2, 1.4, "take cup")])
    _run(tmp_path, "--fps", "5")
    rep = check_dataset(str(tmp_path / "out"), image_size=16, full=True)
    assert not rep.errors, rep.errors
    for manifest in ("manifest_train.csv", "manifest_train_videos.csv"):
        ds = ClipDataset(ClipManifest(str(tmp_path / "out"), manifest, min_length=3), R3MFrameSampler(0.2), seed=0)
        frames, _ = ds.sample(np.random.default_rng(0))
        assert frames.shape == (5, 3, 16, 16)
        start, goal, s0, s1, s2 = frames.float().mean(dim=(1, 2, 3)).numpy()
        assert start < goal and s0 <= s1 + 3 and s1 <= s2 + 3              # time order within the clip


def test_windows_split_a_video_into_equal_parts():
    from src.data.prepare_epic import windows
    times = np.arange(70) * 0.1                                            # 6.9 s at 10 fps
    parts = windows(70, times, 3.0)                                        # 3 parts of at most 3 s
    assert [n for _, n in parts] == [23, 24, 23] and [a for a, _ in parts] == [0, 23, 47]
    assert all(times[a + n - 1] - times[a] <= 3.0 for a, n in parts)
    assert windows(5, times[:5], 60.0) == [(0, 5)]                         # shorter than a window: whole


def test_window_manifest_from_frame_times(tmp_path):
    _dataset(tmp_path, {"P01_01": 50, "P02_01": 20}, [("P01_01", 0.3, 0.9, "open door"), ("P02_01", 0.0, 0.5, "x")])
    _run(tmp_path, "--fps", "5")
    _run(tmp_path, "--fps", "5", "--manifests-only", "--window", "0.8")
    w = pd.read_csv(tmp_path / "out" / "manifest_train_win0.8s.csv", keep_default_na=False)
    keys, _ = _stored_levels(tmp_path / "out")
    # P01_01: 10 frames over 1.8 s -> 3 parts; P02_01: 4 frames over 0.6 s -> 1
    assert list(w["video_id"]) == ["P01_01"] * 3 + ["P02_01"] and list(w["len"]) == [3, 4, 3, 4]
    assert set(w["txt"]) == {""}
    for _, r in w.iterrows():
        assert keys[r["start"]].split("/")[0] == keys[r["start"] + r["len"] - 1].split("/")[0] == r["video_id"]
    from src.data.check import check_dataset
    assert not check_dataset(str(tmp_path / "out"), image_size=16).errors


def test_a_limited_run_extrapolates_by_size(tmp_path, capsys):
    _dataset(tmp_path, {"P01_01": 25, "P01_02": 25}, [("P01_01", 0.0, 0.9, "a"), ("P01_02", 0.0, 0.9, "b")])
    _run(tmp_path, "--fps", "5", "--limit", "1")
    out = capsys.readouterr().out
    assert "min/GB" in out and "the whole split is" in out
