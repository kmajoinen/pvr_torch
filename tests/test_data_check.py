"""The dataset checker (src/data/check.py): clean datasets pass and give a
config pretrain.py can use; each kind of format problem is reported."""

import os

import pandas as pd
import pytest
import torch
from torchvision.io import encode_jpeg, write_jpeg

from src.data.check import check_dataset, main
from src.data.shards import ShardWriter

S = 32     # frame side used throughout (checked with image_size=S)


def _jpeg(value, size=(S, S)):
    return encode_jpeg(torch.full((3, *size), value, dtype=torch.uint8), quality=95).numpy().tobytes()


def _clip_shards(root, splits=(("train", [5, 8, 12]), ("validation", [6, 7])), size=(S, S), text="do a thing"):
    for split, lengths in splits:
        rows = []
        with ShardWriter(os.path.join(root, split)) as w:
            for v, n in enumerate(lengths):
                start = len(w.keys)
                for i in range(1, n + 1):
                    w.add(f"v{v}/{i:06d}.jpg", _jpeg(10 * i, size))
                rows.append({"path": split, "len": n, "txt": text, "start": start})
        pd.DataFrame(rows).to_csv(os.path.join(root, f"manifest_{split}.csv"), index=False)


def _clip_folders(root, lengths=(4, 9)):
    rows = []
    for v, n in enumerate(lengths):
        d = os.path.join(root, "frames", f"v{v}")
        os.makedirs(d)
        for i in range(1, n + 1):
            write_jpeg(torch.full((3, S, S), 10 * i, dtype=torch.uint8), os.path.join(d, f"{i:06d}.jpg"))
        rows.append({"path": f"frames/v{v}", "len": n, "txt": ""})
    pd.DataFrame(rows).to_csv(os.path.join(root, "manifest_train.csv"), index=False)


def _image_folders(root, splits=("train", "val")):
    for split in splits:
        for c in ("cat", "dog"):
            d = os.path.join(root, split, c)
            os.makedirs(d)
            for i in range(3):
                write_jpeg(torch.full((3, 40, 50), 50 * i, dtype=torch.uint8), os.path.join(d, f"{c}{i}.jpg"))


def _check(root, **kw):
    return check_dataset(str(root), image_size=S, **kw)


def test_clip_shards_pass_and_also_check_as_frames(tmp_path):
    _clip_shards(tmp_path)
    rep = _check(tmp_path, full=True)
    assert rep.ok and not rep.warnings, rep.format()
    clips, frames = rep.configs[tmp_path.name], rep.configs[f"{tmp_path.name}_frames"]
    assert clips["type"] == "clips" and clips["val_manifest"] == "manifest_validation.csv"
    assert clips["image_size"] == S and clips["frames_as_stored"] is True
    assert frames["type"] == "images" and frames["train_split"] == "train" and frames["val_split"] == "validation"
    assert any("thins per video" in n for n in rep.notes) and any("unlabelled" in n for n in rep.notes)


def test_suggested_clip_config_loads(tmp_path):
    from src.data.clips import ClipManifest
    _clip_shards(tmp_path)
    cfg = _check(tmp_path).configs[tmp_path.name]
    m = ClipManifest(cfg["root"], cfg["train_manifest"], min_length=3)
    assert len(m) == 3


def test_clip_folders_pass_and_are_not_mistaken_for_class_folders(tmp_path):
    _clip_folders(tmp_path)
    rep = _check(tmp_path, full=True)
    assert rep.ok, rep.format()
    assert list(rep.configs) == [tmp_path.name]             # no image config from frames/<video>/
    assert any("no text" in n for n in rep.notes)
    assert any("no validation manifest" in n for n in rep.notes)


def test_missing_manifest_columns(tmp_path):
    pd.DataFrame([{"video": "a", "n": 3}]).to_csv(tmp_path / "manifest.csv", index=False)
    rep = _check(tmp_path)
    assert not rep.ok and "missing columns" in rep.errors[0] and "txt" in rep.errors[0]


def test_missing_frames_in_a_folder(tmp_path):
    _clip_folders(tmp_path)
    os.remove(tmp_path / "frames" / "v1" / "000009.jpg")
    rep = _check(tmp_path, full=True)
    assert any("frames missing" in e for e in rep.errors)


def test_video_running_past_the_shard_index(tmp_path):
    _clip_shards(tmp_path)
    df = pd.read_csv(tmp_path / "manifest_train.csv")
    df.loc[2, "len"] = 99
    df.to_csv(tmp_path / "manifest_train.csv", index=False)
    rep = _check(tmp_path)
    assert any("run past the index" in e for e in rep.errors)


def test_truncated_shard(tmp_path):
    _clip_shards(tmp_path, splits=(("train", [5, 8]),))
    shard = sorted((tmp_path / "train").glob("shard_*.tar"))[-1]
    shard.write_bytes(shard.read_bytes()[:100])
    rep = _check(tmp_path)
    assert any("past their shard's end" in e for e in rep.errors)


def test_undecodable_frame(tmp_path):
    _clip_folders(tmp_path)
    (tmp_path / "frames" / "v0" / "000001.jpg").write_bytes(b"not a jpeg")
    rep = _check(tmp_path, full=True)
    assert any("don't decode" in e for e in rep.errors)


def test_frames_of_another_size_or_shape(tmp_path):
    _clip_shards(tmp_path / "a")
    assert any("frames_as_stored needs them equal" in w for w in check_dataset(str(tmp_path / "a"), image_size=224).warnings)
    _clip_shards(tmp_path / "b", size=(S, 2 * S))
    rep = _check(tmp_path / "b")
    assert any("not square" in w for w in rep.warnings)
    assert rep.configs["b"]["frames_as_stored"] is False


def test_short_videos_are_flagged(tmp_path):
    _clip_shards(tmp_path, splits=(("train", [2, 8]),))
    assert any("fewer than 3 frames" in w for w in _check(tmp_path).warnings)


def test_image_folders_pass_with_labels(tmp_path):
    _image_folders(tmp_path)
    rep = _check(tmp_path, full=True)
    assert rep.ok, rep.format()
    cfg = rep.configs[tmp_path.name]
    assert cfg["type"] == "images" and cfg["train_split"] == "train" and cfg["val_split"] == "val"
    assert any("2 classes" in n for n in rep.notes) and not rep.warnings


def test_small_images_are_flagged_against_the_encoder_size(tmp_path):
    _image_folders(tmp_path)
    rep = check_dataset(str(tmp_path), image_size=224)
    assert any("upsample" in w for w in rep.warnings)


def test_image_folders_without_a_training_split(tmp_path):
    _image_folders(tmp_path, splits=("val",))
    rep = _check(tmp_path)
    assert any("define the classes" in e for e in rep.errors)


def test_unknown_layouts_get_hints(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "clip1.mp4").write_bytes(b"")
    (tmp_path / "loose.jpg").write_bytes(_jpeg(5))
    rep = _check(tmp_path)
    assert not rep.ok and "no dataset in a format" in rep.errors[0]
    assert any("video files" in n for n in rep.notes) and any("loose in the root" in n for n in rep.notes)


def test_cli_exit_status(tmp_path, capsys):
    _clip_shards(tmp_path / "good")
    assert main([str(tmp_path / "good"), "--image-size", str(S)]) == 0
    assert "OK: no errors" in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    assert main([str(tmp_path / "empty")]) == 1


def test_an_off_by_one_start_is_caught_from_the_keys(tmp_path):
    _clip_shards(tmp_path, splits=(("train", [5, 8, 12]),))
    df = pd.read_csv(tmp_path / "manifest_train.csv")
    df["start"] += 1                               # start taken after each video's first frame was added
    df.loc[2, "len"] -= 1                          # keep the last row inside the index (it's then a
    df.to_csv(tmp_path / "manifest_train.csv", index=False)   # valid segment: frames 2-12 of one video)
    rep = _check(tmp_path)
    errs = [e for e in rep.errors if "consecutive frames" in e]
    assert errs and "2 rows" in errs[0] and "video 1 of the manifest" in errs[0]
    assert "'v0/000002.jpg' to 'v1/000001.jpg'" in errs[0]


def test_segments_of_a_longer_video_pass(tmp_path):
    _clip_shards(tmp_path, splits=(("train", [12]),))
    seg = pd.DataFrame([{"path": "train", "len": 5, "txt": "a", "start": 0},      # frames 1-5
                        {"path": "train", "len": 6, "txt": "b", "start": 4}])     # frames 5-10, overlapping
    seg.to_csv(tmp_path / "manifest_train.csv", index=False)
    assert _check(tmp_path).ok


def test_validation_prefers_val_over_test(tmp_path):
    _clip_shards(tmp_path / "clips", splits=(("train", [5]), ("test", [5]), ("validation", [5])))
    assert _check(tmp_path / "clips").configs["clips"]["val_manifest"] == "manifest_validation.csv"
    _image_folders(tmp_path / "imgs", splits=("train", "test", "val"))
    assert _check(tmp_path / "imgs").configs["imgs"]["val_split"] == "val"


def test_frames_as_stored_only_at_the_encoder_size(tmp_path):
    _clip_shards(tmp_path)
    assert check_dataset(str(tmp_path), image_size=224).configs[tmp_path.name]["frames_as_stored"] is False
    assert _check(tmp_path).configs[tmp_path.name]["frames_as_stored"] is True


def test_frame_sizes_must_agree_between_manifests(tmp_path):
    _clip_shards(tmp_path, splits=(("train", [5, 6]),))
    _clip_shards(tmp_path, splits=(("validation", [5]),), size=(S + 8, S + 8))
    assert any("between manifests" in e for e in _check(tmp_path).errors)


def test_unknown_layout_scan_stops_early(tmp_path, monkeypatch):
    import src.data.check as check
    seen = []

    def endless(*a, **k):
        i = 0
        while True:
            seen.append(i)
            yield os.path.join(str(tmp_path), f"f{i}.txt")
            i += 1
    monkeypatch.setattr(check.glob, "iglob", endless)
    check._guess_unknown(str(tmp_path), check.Report())
    assert len(seen) == 20000


def test_a_non_square_manifest_is_compared_too(tmp_path):
    # Training frames square at the encoder size, validation frames not square:
    # the sizes disagree, and the frames can't all be used as stored.
    _clip_shards(tmp_path, splits=(("train", [5, 6]),))
    _clip_shards(tmp_path, splits=(("validation", [5]),), size=(S - 8, S + 8))
    rep = _check(tmp_path)
    assert any("between manifests" in e and f"{S - 8}x{S + 8}" in e for e in rep.errors)
    assert rep.configs[tmp_path.name]["frames_as_stored"] is False


# ── Raw sources, quick mode, pretrain.py's startup check ─────────────────────

@pytest.mark.parametrize("files, expect", [
    (["P01/videos/P01_01.MP4"], "prepare_epic"),
    (["videos/12345.webm"], "prepare_ssv2"),
    (["ILSVRC/Data/CLS-LOC/train/x.JPEG", "LOC_val_solution.csv"], "prepare_imagenet"),
    (["bridge/0.1.0/dataset_info.json", "bridge/0.1.0/bridge-train.tfrecord-00000-of-00001"], "RLDS"),
    (["wds/imagenet1k-train-000000.tar", "wds/imagenet1k-train-000001.tar"], "WebDataset"),
])
def test_known_raw_sources_name_their_converter(tmp_path, files, expect):
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_bytes(b"")
    rep = _check(tmp_path)
    assert not rep.ok and any(expect in n for n in rep.notes), rep.notes


def test_a_known_archive_names_its_converter(tmp_path):
    (tmp_path / "something-something-v2.tar").write_bytes(b"")
    rep = _check(tmp_path / "something-something-v2.tar")
    assert any("not a folder" in e for e in rep.errors) and any("prepare_ssv2" in n for n in rep.notes)
    assert any("doesn't exist" in e for e in _check(tmp_path / "nope").errors)


def test_an_unrecognised_folder_is_described(tmp_path):
    for f in ("apple_000.zip", "apple_001.zip", "ball_000.zip", "category_list.json"):
        (tmp_path / f).write_bytes(b"")
    note = " ".join(_check(tmp_path).notes)
    assert "Top level (4)" in note and "3 .zip" in note and "1 .json" in note and "unpack" in note


def test_a_prepared_dataset_one_level_down_is_pointed_out(tmp_path):
    _clip_shards(tmp_path / "ssv2")
    rep = _check(tmp_path)
    assert not rep.ok and any("one level down: ssv2/" in n for n in rep.notes)


def test_quick_skips_the_whole_index_scan(tmp_path):
    _clip_shards(tmp_path, splits=(("train", [5, 8, 12]),))
    df = pd.read_csv(tmp_path / "manifest_train.csv")
    df["start"] += 1
    df.loc[2, "len"] -= 1
    df.to_csv(tmp_path / "manifest_train.csv", index=False)
    assert any("consecutive frames" in e for e in _check(tmp_path).errors)
    assert not any("consecutive frames" in e for e in _check(tmp_path, quick=True).errors)


def _auto(root, **kw):
    return {"name": None, "type": "auto", "root": str(root), **kw}


def test_auto_picks_the_kind_the_objectives_read(tmp_path):
    from src.data.check import resolve_config
    _clip_shards(tmp_path)
    clips = resolve_config(_auto(tmp_path), "clips", image_size=S)
    assert clips["type"] == "clips" and clips["train_manifest"] == "manifest_train.csv"
    assert clips["val_manifest"] == "manifest_validation.csv" and clips["image_size"] == S
    frames = resolve_config(_auto(tmp_path), "images", image_size=S)     # the clip shards as images
    assert frames["type"] == "images" and frames["train_split"] == "train" and frames["val_split"] == "validation"
    assert frames["every"] == 10


def test_auto_keeps_values_set_by_hand(tmp_path):
    from src.data.check import resolve_config
    _clip_shards(tmp_path)
    out = resolve_config(_auto(tmp_path, every=3, name="mine", eval_batches=None), "images", image_size=S)
    assert out["every"] == 3 and out["name"] == "mine" and out["eval_batches"] == 20


def test_auto_on_class_folders(tmp_path):
    from src.data.check import resolve_config
    _image_folders(tmp_path)
    out = resolve_config(_auto(tmp_path), "images")
    assert out["train_split"] == "train" and out["val_split"] == "val" and out["every"] == 1


def test_a_named_config_passes_through_when_its_files_exist(tmp_path):
    from src.data.check import resolve_config
    _clip_shards(tmp_path)
    cfg = {"name": "x", "type": "clips", "root": str(tmp_path), "train_manifest": "manifest_train.csv",
           "val_manifest": None, "image_size": 7}
    assert resolve_config(cfg, "clips") is cfg


def test_a_named_config_on_the_wrong_layout_says_what_is_there(tmp_path):
    from src.data.check import resolve_config
    _clip_shards(tmp_path)
    cfg = {"name": "imagenet", "type": "images", "root": str(tmp_path), "train_split": "train",
           "val_split": "val"}
    with pytest.raises(ValueError) as e:
        resolve_config(cfg, "images")
    msg = str(e.value)
    assert "no val_split 'val'" in msg and "an image dataset: train_split=train, val_split=validation" in msg
    assert "pretrain_data=auto" in msg


def test_auto_without_a_matching_dataset(tmp_path):
    from src.data.check import resolve_config
    _image_folders(tmp_path)
    with pytest.raises(ValueError, match="Image datasets can't feed clip objectives"):
        resolve_config(_auto(tmp_path), "clips")
    (tmp_path / "rlds").mkdir()
    (tmp_path / "rlds" / "x.tfrecord-00000-of-00001").write_bytes(b"")
    with pytest.raises(ValueError, match="(?s)no clips dataset.*RLDS"):
        resolve_config(_auto(tmp_path / "rlds"), "clips")
    with pytest.raises(ValueError, match="doesn't exist"):
        resolve_config(_auto(tmp_path / "nope"), "images")


def test_pretrain_resolves_auto_before_building_anything(tmp_path):
    from hydra import compose, initialize_config_dir
    from pretrain import resolve_data
    _clip_shards(tmp_path)
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config_pretrain",
                      overrides=["+experiment=mae", "pretrain_data=auto", f"pretrain_data.root={tmp_path}",
                                 f"pretrain_data.image_size={S}"])
        bad = compose(config_name="config_pretrain",
                      overrides=["+experiment=r3m", f"pretrain_data.root={tmp_path / 'missing'}"])
    resolve_data(cfg)
    assert cfg.pretrain_data.type == "images" and cfg.pretrain_data.train_split == "train"
    assert cfg.pretrain_data.name == os.path.basename(str(tmp_path)) + "_frames"
    with pytest.raises(ValueError, match="doesn't exist"):
        resolve_data(bad)
