"""RLDS/TFDS converter (src/data/prepare_rlds.py): the TensorFlow-free
TFRecord/Example reader, schema-driven field choice, splits, and the
checker's verdict on the output. Datasets are written here in TFDS's on-disk
format (features.json, dataset_info.json, <name>-<split>.tfrecord-*)."""

import json
import struct

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from src.data.prepare_rlds import main, parse_example, read_records  # noqa: E402

PKG = "tensorflow_datasets.core.features."


# ── Writing TFDS datasets ─────────────────────────────────────────────────────

def _vint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    return _vint(field << 3 | 2) + _vint(len(payload)) + payload


def _example(features: dict) -> bytes:
    """{name: list of bytes | float array | ("int", list)} -> tf.train.Example."""
    entries = b""
    for key, v in features.items():
        if isinstance(v, tuple):
            feat = _ld(3, _ld(1, b"".join(_vint(int(x)) for x in v[1])))
        elif isinstance(v, np.ndarray):
            feat = _ld(2, _ld(1, v.astype("<f4").tobytes()))
        else:
            feat = _ld(1, b"".join(_ld(1, x) for x in v))
        entries += _ld(1, _ld(1, key.encode()) + _ld(2, feat))
    return _ld(1, entries)


def _tfrecord(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(struct.pack("<Q", len(r)) + b"\0" * 4 + r + b"\0" * 4)


def _image_feature(h, w, fmt="png"):
    return {"pythonClassName": PKG + "image_feature.Image",
            "image": {"shape": {"dimensions": [str(h), str(w), "3"]}, "dtype": "uint8", "encodingFormat": fmt}}


def _tensor(dtype, dims=()):
    return {"pythonClassName": PKG + "tensor_feature.Tensor",
            "tensor": {"shape": {"dimensions": [str(d) for d in dims]} if dims else {}, "dtype": dtype}}


def _dict(features):
    return {"pythonClassName": PKG + "features_dict.FeaturesDict", "featuresDict": {"features": features}}


def _frame(level, fmt=".png"):
    return cv2.imencode(fmt, np.full((24, 32, 3), level, np.uint8))[1].tobytes()


def _dataset(d, name="toy", splits=None, cameras=("image", "wrist_image"), text="natural_language_instruction",
             codepoints=False):
    """splits: {split: [[episode steps, ...] per file]}; camera c, step t of
    episode e (global count) is gray level 10 * t + 100 * c, instruction f"task {e}"."""
    splits = splits or {"train": [[3, 4], [5]]}
    d.mkdir(parents=True, exist_ok=True)
    obs = {c: _image_feature(24, 32) for c in cameras}
    obs["state"] = _tensor("float32", (7,))
    steps = {"observation": _dict(obs), "is_first": _tensor("bool")}     # obs is filled in below
    if text and codepoints:
        obs[text] = _tensor("int32", (16,))
    elif text:
        obs[text] = _tensor("string")
    feats = _dict({"steps": {"pythonClassName": PKG + "dataset_feature.Dataset",
                             "sequence": {"feature": _dict(steps)}}})
    (d / "features.json").write_text(json.dumps(feats))
    info = {"name": name, "splits": []}
    e = 0
    for split, files in splits.items():
        info["splits"].append({"name": split, "shardLengths": [str(len(f)) for f in files]})
        for i, episodes in enumerate(files):
            records = []
            for n in episodes:
                ex = {f"steps/observation/{c}": [_frame(10 * t + 100 * k) for t in range(n)]
                      for k, c in enumerate(cameras)}
                ex["steps/observation/state"] = np.zeros(7 * n, np.float32)
                if text and codepoints:
                    row = [ord(ch) for ch in f"task {e}"] + [0] * (16 - len(f"task {e}"))
                    ex[f"steps/observation/{text}"] = ("int", row * n)
                elif text:
                    ex[f"steps/observation/{text}"] = [f"task {e}".encode()] * n
                records.append(_example(ex))
                e += 1
            _tfrecord(d / f"{name}-{split}.tfrecord-{i:05d}-of-{len(files):05d}", records)
    (d / "dataset_info.json").write_text(json.dumps(info))
    return d


def _run(src, out, *extra):
    main(["--source", str(src), "--out", str(out), "--size", "16", "--workers", "2", *extra])
    return {s: pd.read_csv(out / f"manifest_{s}.csv", keep_default_na=False)
            for s in ("train", "validation") if (out / f"manifest_{s}.csv").exists()}


def _levels(out, split):
    from src.data.images import decode
    from src.data.shards import ShardReader, load_index
    d = str(out / split)
    index, reader = load_index(d, fields=("key", "shard", "offset", "size")), ShardReader(d)
    keys = [str(k) for k in index["key"]]
    return keys, [float(decode(reader.read(index, i)).float().mean()) for i in range(len(keys))]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_reader_round_trips_examples(tmp_path):
    ex = {"a": [b"x", b"", b"\xff" * 300], "f": np.array([1.5, -2.0, 3.25], np.float32),
          "i": ("int", [0, 1, 300, 2**40, -1])}
    _tfrecord(tmp_path / "r.tfrecord", [_example(ex), _example({"a": [b"y"]})])
    recs = list(read_records(tmp_path / "r.tfrecord"))
    got = parse_example(recs[0])
    assert got["a"] == ex["a"]
    np.testing.assert_array_equal(got["f"], ex["f"])
    assert list(got["i"]) == [0, 1, 300, 2**40, -1]
    assert parse_example(recs[0], keys={"f"}).keys() == {"f"} and parse_example(recs[1])["a"] == [b"y"]


def test_episodes_become_clips_of_the_main_camera(tmp_path, capsys):
    src = _dataset(tmp_path / "toy" / "1.0.0", splits={"train": [[3, 4], [5]], "test": [[6]]})
    m = _run(tmp_path, tmp_path / "out")                 # --source two levels above the dataset
    assert list(m["train"]["len"]) == [3, 4, 5] and list(m["validation"]["len"]) == [6]
    assert list(m["train"]["txt"]) == ["task 0", "task 1", "task 2"] and list(m["validation"]["txt"]) == ["task 3"]
    assert list(m["train"]["episode"]) == ["00000_0000", "00000_0001", "00001_0000"]
    assert set(m["train"]["camera"]) == {"image"}       # wrist_image left out by default
    keys, levels = _levels(tmp_path / "out", "train")
    for _, r in m["train"].iterrows():                  # each row: its episode's frames, in step order
        assert keys[r["start"]] == f"{r['episode']}/000001.jpg"
        np.testing.assert_allclose(levels[r["start"]:r["start"] + r["len"]], [10 * t for t in range(r["len"])],
                                   atol=2)
    out = capsys.readouterr().out
    assert "cameras: image, wrist_image -> storing image" in out and src.name in out


def test_check_auto_and_loader_accept_the_output(tmp_path):
    from src.data.check import check_dataset, resolve_config
    from src.data.clips import ClipDataset, ClipManifest
    from src.losses import R3MFrameSampler
    _dataset(tmp_path / "toy", splits={"train": [[4, 5], [6]], "test": [[5]]})
    _run(tmp_path / "toy", tmp_path / "out")
    rep = check_dataset(str(tmp_path / "out"), image_size=16, full=True)
    assert not rep.errors, rep.errors
    cfg = resolve_config({"type": "auto", "root": str(tmp_path / "out")}, "clips", image_size=16)
    assert cfg["train_manifest"] == "manifest_train.csv" and cfg["val_manifest"] == "manifest_validation.csv"
    images = resolve_config({"type": "auto", "root": str(tmp_path / "out")}, "images", image_size=16)
    assert images["train_split"] == "train" and images["val_split"] == "validation"
    ds = ClipDataset(ClipManifest(str(tmp_path / "out"), "manifest_train.csv", min_length=3),
                     R3MFrameSampler(0.2), seed=0)
    frames, text = ds.sample(np.random.default_rng(0))
    assert frames.shape == (5, 3, 16, 16) and text.startswith("task ")


def test_train_only_datasets_hold_out_files(tmp_path):
    _dataset(tmp_path / "toy", splits={"train": [[3]] * 10})
    m = _run(tmp_path / "toy", tmp_path / "out", "--holdout", "0.2")
    assert len(m["train"]) == 8 and list(m["validation"]["episode"]) == ["00008_0000", "00009_0000"]
    m = _run(tmp_path / "toy", tmp_path / "out0", "--holdout", "0")
    assert "validation" not in m


def test_camera_choice(tmp_path, capsys):
    _dataset(tmp_path / "toy", cameras=("wrist_rgb", "agentview_rgb"), splits={"train": [[3]]})
    assert set(_run(tmp_path / "toy", tmp_path / "a")["train"]["camera"]) == {"agentview_rgb"}
    m = _run(tmp_path / "toy", tmp_path / "b", "--camera", "all")["train"]
    assert list(m["camera"]) == ["wrist_rgb", "agentview_rgb"] and list(m["episode"]) == ["00000_0000"] * 2
    keys, levels = _levels(tmp_path / "b", "train")
    assert keys[m["start"][1]] == "00000_0000_agentview_rgb/000001.jpg" and abs(levels[m["start"][1]] - 100) < 2
    with pytest.raises(SystemExit, match="no camera 'top'.*wrist_rgb, agentview_rgb"):
        _run(tmp_path / "toy", tmp_path / "c", "--camera", "top")


def test_text_fields(tmp_path):
    _dataset(tmp_path / "lt", text="instruction", codepoints=True, splits={"train": [[3, 3]]})
    assert list(_run(tmp_path / "lt", tmp_path / "a")["train"]["txt"]) == ["task 0", "task 1"]
    assert set(_run(tmp_path / "lt", tmp_path / "b", "--text", "none")["train"]["txt"]) == {""}
    _dataset(tmp_path / "plain", text=None, splits={"train": [[3]]})
    assert set(_run(tmp_path / "plain", tmp_path / "c")["train"]["txt"]) == {""}


def test_a_truncated_file_keeps_its_whole_episodes(tmp_path, capsys):
    src = _dataset(tmp_path / "toy", splits={"train": [[3, 4, 5]]})
    f = next(src.glob("*.tfrecord*"))
    f.write_bytes(f.read_bytes()[:-100])
    m = _run(src, tmp_path / "out", "--holdout", "0")
    assert list(m["train"]["len"]) == [3, 4]
    assert "stopped after 2 episodes (EOFError" in capsys.readouterr().out


def test_source_with_several_datasets_names_them(tmp_path):
    _dataset(tmp_path / "oxe" / "bridge" / "0.1.0", name="bridge")
    _dataset(tmp_path / "oxe" / "kuka" / "0.1.0", name="kuka")
    with pytest.raises(SystemExit, match=r"2 datasets under .*\(bridge, kuka\): point --source at one"):
        _run(tmp_path / "oxe", tmp_path / "out")
    _dataset(tmp_path / "oxe" / "kuka" / "0.2.0", name="kuka", splits={"train": [[7]]})
    assert list(_run(tmp_path / "oxe" / "kuka", tmp_path / "out", "--holdout", "0")["train"]["len"]) == [7]


def test_limit_extrapolates_by_size(tmp_path, capsys):
    _dataset(tmp_path / "toy", splits={"train": [[3]] * 4})
    m = _run(tmp_path / "toy", tmp_path / "out", "--limit", "1")
    assert len(m["train"]) == 1 and len(m["validation"]) == 1
    out = capsys.readouterr().out
    assert "min/GB" in out and "the whole dataset is" in out and "output about" in out


def test_every_keeps_every_nth_step(tmp_path):
    _dataset(tmp_path / "toy", splits={"train": [[7, 2]]})
    m = _run(tmp_path / "toy", tmp_path / "out", "--every", "3", "--holdout", "0")["train"]
    assert list(m["len"]) == [3, 1]                     # steps 0, 3, 6 and step 0
    _, levels = _levels(tmp_path / "out", "train")
    np.testing.assert_allclose(levels[:3], [0, 30, 60], atol=2)


def test_checker_names_the_converter(tmp_path):
    from src.data.check import check_dataset
    _dataset(tmp_path / "oxe" / "bridge" / "0.1.0", name="bridge")
    _dataset(tmp_path / "oxe" / "kuka" / "0.1.0", name="kuka")
    one = " ".join(check_dataset(str(tmp_path / "oxe" / "bridge")).notes)
    assert "prepare_rlds --source" in one and "bridge" in one
    many = " ".join(check_dataset(str(tmp_path / "oxe")).notes)
    assert "2 TFDS/RLDS datasets (e.g. bridge, kuka)" in many and "<name>" in many
    _dataset(tmp_path / "oxe" / "kuka" / "0.2.0", name="kuka")          # versions of one dataset: still one
    versions = " ".join(check_dataset(str(tmp_path / "oxe" / "kuka")).notes)
    assert f"--source {tmp_path / 'oxe' / 'kuka'} " in versions and "datasets" not in versions
