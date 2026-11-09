"""
Checks a pretraining dataset against the formats pretrain.py reads
(DATASETS.md), reports what's wrong, and prints a pretrain_data config for
what it found.

    python -m src.data.check /path/to/data                  # find and check every layout under it
    python -m src.data.check /path/to/data --image-size 224 --sample 500
    python -m src.data.check /path/to/data --full           # decode every frame/image (slow)

What it looks for under the root:
  clips   manifest*.csv (path, len, txt[, start]) plus the frames, as
          per-video folders (<path>/000001.jpg ...) or shards (`start`
          column; <path>/index.npz + shard_*.tar)
  images  <split>/index.npz + shards, or <split>/<class>/<images>
A clip dataset's frame shards are also an image dataset (video frames, as
pretrain_data=ssv2_frames reads them); both views are checked.

By default `sample` videos/images are decoded, plus checks over the whole
index that need no decoding (lengths, ranges, shard bounds). Exit status 1
if anything is wrong (errors); warnings are things pretrain.py accepts but
that are probably not what you want.

Library use (e.g. a check at load time): check_dataset(root, ...) returns a
Report with .errors, .warnings, .notes and .configs.
"""

import argparse
import glob
import itertools
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Frame counts below this are skipped by R3M's and VIP's samplers (src/losses.py).
MIN_CLIP_FRAMES = 3
_VIDEO_EXT = (".mp4", ".webm", ".avi", ".mkv", ".mov")
_IMAGE_EXT = (".jpg", ".jpeg", ".png")
_VAL_NAMES = ("val", "validation", "valid", "test")


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    configs: Dict[str, dict] = field(default_factory=dict)   # suggested pretrain_data configs

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = []
        for title, items in (("ERRORS", self.errors), ("WARNINGS", self.warnings), ("NOTES", self.notes)):
            if items:
                lines.append(f"{title} ({len(items)})")
                lines += [f"  - {m}" for m in items]
        if self.configs:
            from omegaconf import OmegaConf
            for name, cfg in self.configs.items():
                lines.append(f"\nSuggested configs/pretrain_data/{name}.yaml:")
                lines.append(OmegaConf.to_yaml(OmegaConf.create(cfg)).rstrip())
        lines.append("\n" + ("OK: no errors" if self.ok else f"FAILED: {len(self.errors)} error(s)"))
        return "\n".join(lines)


def _decode(data: bytes):
    from src.data.images import decode
    return decode(data)


def _shard_bounds(shard_dir: str, index: dict, rep: Report, where: str) -> bool:
    """Every shard the index names exists, and every sample lies inside its shard."""
    bad_files, out_of_bounds = [], 0
    for shard in np.unique(index["shard"]):
        path = os.path.join(shard_dir, f"shard_{int(shard):06d}.tar")
        if not os.path.isfile(path):
            bad_files.append(os.path.basename(path))
            continue
        rows = index["shard"] == shard
        out_of_bounds += int(((index["offset"][rows] + index["size"][rows]) > os.path.getsize(path)).sum())
    if bad_files:
        rep.errors.append(f"{where}: shard files missing: {bad_files[:5]}{'...' if len(bad_files) > 5 else ''}")
    if out_of_bounds:
        rep.errors.append(f"{where}: {out_of_bounds:,} samples end past their shard's end "
                          "(truncated or overwritten shards; convert again)")
    return not bad_files and not out_of_bounds


def _sizes_summary(sizes) -> str:
    if not sizes:
        return "none decoded"
    uniq = sorted(set(sizes))
    if len(uniq) == 1:
        h, w = uniq[0]
        return f"all {h}x{w}"
    sides = [min(s) for s in sizes]
    return f"{len(uniq)} different sizes, shorter side {min(sides)}-{max(sides)}"


# ── Clips ─────────────────────────────────────────────────────────────────────

def _check_starts(shard_dir: str, starts, lengths, row_ids, rep: Report, where: str):
    """Each sharded row covers one video's consecutive frames: its first and
    last frames' keys (<video>/<number>) name the same video, numbered len - 1
    apart. Catches an off-by-one `start` (every clip then takes a frame from
    the neighbouring video), over the whole manifest, without decoding. Rows
    that are segments of a longer video pass."""
    from src.data.images import _frame_numbers
    from src.data.shards import load_index
    keys = np.asarray(load_index(shard_dir, fields=("key",))["key"], dtype=np.bytes_)
    num = _frame_numbers(keys)
    if num is None:
        rep.notes.append(f"{where}: frame keys aren't <video>/<number>.jpg, so `start` values can't be "
                         "checked against them (and `every` thins in stored order)")
        return
    vid = np.char.rpartition(keys, b"/")[:, 0]
    first, last = starts, starts + lengths - 1
    bad = (vid[first] != vid[last]) | (num[last] - num[first] != lengths - 1)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        rep.errors.append(
            f"{where}: {int(bad.sum()):,} rows don't cover one video's consecutive frames, e.g. video "
            f"{int(row_ids[i]) + 1} of the manifest (start {int(starts[i])}, len {int(lengths[i])}) runs from "
            f"{keys[first[i]].decode()!r} to {keys[last[i]].decode()!r}: an off-by-one `start`? (take it as "
            "the index row of the video's first frame, before adding its frames)")

def check_manifest(root: str, manifest: str, rep: Report, image_size: int, sample: int, full: bool,
                   rng, quick: bool = False) -> Optional[dict]:
    """One clip manifest. Returns facts for the suggested config (None if
    unreadable). quick: skip the whole-manifest `start` check (it reads every
    key in the index)."""
    import pandas as pd
    name = os.path.basename(manifest)
    try:
        df = pd.read_csv(manifest, keep_default_na=False, dtype={"path": str, "txt": str})
    except Exception as e:                                   # noqa: BLE001 -- report any parse error
        rep.errors.append(f"{name}: can't be read as CSV ({e})")
        return None
    missing = {"path", "len", "txt"} - set(df.columns)
    if missing:
        rep.errors.append(f"{name}: missing columns {sorted(missing)} (expected path, len, txt[, start]; "
                          f"has {list(df.columns)})")
        return None
    if not len(df):
        rep.errors.append(f"{name}: no rows")
        return None
    lengths = pd.to_numeric(df["len"], errors="coerce")
    if lengths.isna().any() or (lengths % 1 != 0).any() or (lengths < 1).any():
        rep.errors.append(f"{name}: `len` must be a whole number >= 1 for every row "
                          f"({int((lengths.isna() | (lengths < 1)).sum())} bad rows)")
        return None
    lengths = lengths.to_numpy(dtype=np.int64)
    short = int((lengths < MIN_CLIP_FRAMES).sum())
    rep.notes.append(f"{name}: {len(df):,} videos, {int(lengths.sum()):,} frames, length "
                     f"{lengths.min()}-{lengths.max()} (median {int(np.median(lengths))})")
    if short:
        rep.warnings.append(f"{name}: {short:,} videos have fewer than {MIN_CLIP_FRAMES} frames; R3M and VIP "
                            "skip them")
    texts = df["txt"].astype(str)
    empty = int((texts.str.strip() == "").sum())
    if empty == len(df):
        rep.notes.append(f"{name}: no text in `txt`: R3M's language term needs it (lang_weight=0 works "
                         "without); VIP and MAE don't use text")
    elif empty:
        rep.notes.append(f"{name}: {empty:,} videos have empty text (R3M masks them out of its language term)")

    sharded = "start" in df.columns
    paths = [p if os.path.isabs(p) else os.path.join(root, p) for p in df["path"]]
    sizes, decode_errors = [], []
    picks = range(len(df)) if full else rng.choice(len(df), size=min(sample, len(df)), replace=False)

    if sharded:
        starts = pd.to_numeric(df["start"], errors="coerce")
        if starts.isna().any() or (starts < 0).any():
            rep.errors.append(f"{name}: `start` must be a whole number >= 0 for every row")
            return None
        starts = starts.to_numpy(dtype=np.int64)
        readers = {}
        for d in sorted(set(paths)):
            rows = np.array([p == d for p in paths])
            where = f"{name} -> {os.path.relpath(d, root)}"
            if not os.path.isfile(os.path.join(d, "index.npz")):
                rep.errors.append(f"{where}: no index.npz (rows with a `start` column point at a shard "
                                  "folder written by src/data/shards.py's ShardWriter)")
                continue
            from src.data.shards import ShardReader, load_index
            index = load_index(d, fields=("shard", "offset", "size"))
            n = len(index["offset"])
            past = int((starts[rows] + lengths[rows] > n).sum())
            if past:
                rep.errors.append(f"{where}: {past:,} videos run past the index's {n:,} frames "
                                  "(start + len > frames in the index)")
                continue
            if _shard_bounds(d, index, rep, where):
                readers[d] = (index, ShardReader(d))
            if not quick:
                _check_starts(d, starts[rows], lengths[rows], np.flatnonzero(rows), rep, where)

        def read(i, frame):
            index, reader = readers[paths[i]]
            return reader.read(index, int(starts[i]) + frame - 1)
    else:
        missing_dirs = [p for p in sorted(set(paths)) if not os.path.isdir(p)]
        if missing_dirs:
            rep.errors.append(f"{name}: {len(missing_dirs):,} video folders don't exist, e.g. "
                              f"{os.path.relpath(missing_dirs[0], root)!r} (paths are relative to the root)")
        readers = {p: None for p in set(paths) if os.path.isdir(p)}

        def read(i, frame):
            with open(os.path.join(paths[i], f"{frame:06d}.jpg"), "rb") as f:
                return f.read()

    missing_frames = 0
    for i in picks:
        i = int(i)
        if paths[i] not in readers:
            continue
        length = int(lengths[i])
        frames = range(1, length + 1) if full else sorted({1, length, int(rng.integers(1, length + 1))})
        for frame in frames:
            try:
                data = read(i, frame)
            except FileNotFoundError:
                missing_frames += 1
                continue
            try:
                img = _decode(data)
                sizes.append(tuple(img.shape[1:]))
            except Exception as e:                           # noqa: BLE001
                decode_errors.append(f"{df['path'][i]} frame {frame} ({type(e).__name__})")
    if missing_frames:
        rep.errors.append(f"{name}: {missing_frames:,} frames missing among the checked videos (frame folders "
                          "need 000001.jpg ... <len>.jpg, numbered from 1)")
    if decode_errors:
        rep.errors.append(f"{name}: {len(decode_errors):,} frames don't decode, e.g. {decode_errors[:3]}")

    scope = "all frames" if full else f"{len(picks):,} video{'s' if len(picks) != 1 else ''} sampled"
    rep.notes.append(f"{name}: frames {_sizes_summary(sizes)} ({scope}, {'shards' if sharded else 'frame folders'})")
    shape = None
    if sizes:
        uniq = set(sizes)
        if len(uniq) > 1:
            rep.errors.append(f"{name}: frames differ in size ({_sizes_summary(sizes)}); a clip dataset's "
                              "frames must all be the same size: convert them at one size")
        else:
            (h, w), = uniq
            shape = (h, w)
            if h != w:
                rep.warnings.append(f"{name}: frames are {h}x{w}, not square: pretrain_data.frames_as_stored "
                                    "needs square frames at the encoder's input size; use "
                                    "frames_as_stored=false (the encoder's own resize/crop) or convert square")
            elif h != image_size:
                rep.warnings.append(f"{name}: frames are {h}x{w}, the encoder input checked against is "
                                    f"{image_size} (--image-size): frames_as_stored needs them equal; use "
                                    "frames_as_stored=false or convert at that size")
    return {"shape": shape, "sharded": sharded, "dirs": sorted(set(paths))}


# ── Images ────────────────────────────────────────────────────────────────────

def check_image_split(root: str, split: str, rep: Report, image_size: int, sample: int, full: bool, rng,
                      frames_of: Optional[str] = None) -> Optional[dict]:
    from src.data.images import _folder_index, _frame_numbers, load_classes
    d = os.path.join(root, split)
    where = f"{split}/" + (f" (frames of {frames_of})" if frames_of else "")
    if os.path.isfile(os.path.join(d, "index.npz")):
        from src.data.shards import ShardReader, load_index
        index = load_index(d)
        missing = {"key", "shard", "offset", "size", "label"} - set(index)
        if missing:
            rep.errors.append(f"{where}: index.npz lacks {sorted(missing)} (write it with ShardWriter)")
            return None
        if not _shard_bounds(d, index, rep, where):
            return None
        reader = ShardReader(d)
        read = lambda i: reader.read(index, i)              # noqa: E731
        layout = "shards"
    else:
        if load_classes(root) is None:
            rep.errors.append(f"{where}: class folders, but neither {root}/train nor {root}/classes.txt "
                              "to define the classes (labels follow the training split's sorted folders)")
            return None
        try:
            index = _folder_index(root, split)
        except (ValueError, OSError, TypeError) as e:
            rep.errors.append(f"{where}: {e}")
            return None
        read = lambda i: open(os.path.join(d, index["key"][i].decode()), "rb").read()   # noqa: E731
        layout = "class folders"
    n = len(index["key"])
    if not n:
        rep.errors.append(f"{where}: no images")
        return None
    labels = index["label"]
    labelled = bool((labels >= 0).all())
    if labelled:
        classes = load_classes(root)
        k = int(labels.max()) + 1
        rep.notes.append(f"{where}: {n:,} images, {len(np.unique(labels)):,} classes ({layout})")
        if classes and k > len(classes):
            rep.errors.append(f"{where}: labels go up to {k - 1}, but classes.txt lists {len(classes)} classes")
    else:
        rep.notes.append(f"{where}: {n:,} images, unlabelled ({layout}): fine for MAE and MoCo; supervised "
                         "training needs labels")
    if _frame_numbers(index["key"]) is not None:
        rep.notes.append(f"{where}: keys are <video>/<frame number>: pretrain_data.every thins per video")

    picks = range(n) if full else rng.choice(n, size=min(sample, n), replace=False)
    sizes, bad = [], []
    for i in picks:
        try:
            sizes.append(tuple(_decode(read(int(i))).shape[1:]))
        except Exception as e:                               # noqa: BLE001
            bad.append(f"{index['key'][int(i)]!r} ({type(e).__name__})")
    if bad:
        rep.errors.append(f"{where}: {len(bad):,} images don't decode, e.g. {bad[:3]}")
    rep.notes.append(f"{where}: {_sizes_summary(sizes)} ({'all' if full else f'{len(picks):,} sampled'})")
    small = sum(min(s) < image_size for s in sizes)
    if small:
        rep.warnings.append(f"{where}: {small:,} of {len(sizes):,} checked images have a side shorter than "
                            f"{image_size}: the random crops upsample them")
    return {"labelled": labelled}


# ── Whole dataset ─────────────────────────────────────────────────────────────

def _pick_val(candidates, matches) -> Optional[str]:
    """The validation candidate whose best match ranks highest in _VAL_NAMES
    (val/validation before test, which is often unlabelled)."""
    ranked = [(min(i for i, v in enumerate(_VAL_NAMES) if matches(c, v)), c)
              for c in candidates if any(matches(c, v) for v in _VAL_NAMES)]
    return min(ranked)[1] if ranked else None


def _image_split_dirs(root: str, exclude=()) -> List[str]:
    out = []
    for e in sorted(os.scandir(root), key=lambda e: e.name):
        if not e.is_dir() or e.name in exclude:
            continue
        if os.path.isfile(os.path.join(e.path, "index.npz")):
            out.append(e.name)
            continue
        sub = [c for c in os.scandir(e.path) if c.is_dir()]
        if sub and any(f.name.lower().endswith(_IMAGE_EXT) for c in sub[:5] for f in os.scandir(c.path)):
            out.append(e.name)
    return out


def _prepared_subfolders(root: str) -> List[str]:
    """Subfolders that are prepared datasets themselves (root pointed one level too high)."""
    out = []
    for e in sorted(os.scandir(root), key=lambda e: e.name):
        if e.is_dir() and (glob.glob(os.path.join(e.path, "manifest*.csv"))
                           or glob.glob(os.path.join(e.path, "*", "index.npz"))):
            out.append(e.name)
    return out


def _known_source(root: str, files: List[str]) -> Optional[str]:
    """A raw dataset this repo has a converter (or a plan) for, recognised by its
    file names: what it is and the command that prepares it."""
    import re
    norm = [f.replace(os.sep, "/") for f in files]
    if any(re.search(r"(^|/)P\d\d/videos/P\d\d_\d+\.mp4$", f, re.I) for f in norm):
        return ("EPIC-KITCHENS videos (<P##>/videos/<id>.MP4): convert with python -m src.data.prepare_epic "
                f"--videos {root} --annotations <clone of github.com/epic-kitchens/epic-kitchens-100-annotations> "
                "--out <dataset folder> --split train (then --split validation); run_prepare_epic.sh does both")
    if any(re.search(r"(^|/)\d+\.webm$", f) for f in norm) or any("something-something" in f.lower() for f in norm):
        return ("Something-Something v2 (numbered .webm videos / its labels): convert with python -m "
                "src.data.prepare_ssv2 (run_prepare_ssv2.sh extracts Triton's archive and converts both splits)")
    if any(re.search(r"\.tar$", f) for f in norm) and any(re.search(r"\d{4,}\.tar$", f) for f in norm):
        return ("numbered .tar files, probably WebDataset shards (e.g. imagenet-1k-wds): not read directly; "
                "no converter yet (DATASETS.md section 5)")
    if any(re.search(r"(^|/)(ILSVRC|LOC_val_solution\.csv|n\d{8}/)", f) for f in norm) or \
            any(re.search(r"imagenet.*\.(zip|tar)$", f, re.I) for f in norm):
        return ("ImageNet (ILSVRC files or <wnid> folders): convert with python -m src.data.prepare_imagenet "
                f"--source {root} --out <dataset folder> (it also reads Kaggle's zip and the imagefolder tar)")
    if any("tfrecord" in f for f in norm) or any(f.endswith(("dataset_info.json", "features.json")) for f in norm):
        return ("TFDS/RLDS TFRecords (e.g. Open X-Embodiment, BridgeData's RLDS release): no converter yet; "
                "DATASETS.md section 5 shows how to write one")
    return None


def _guess_unknown(root: str, rep: Report):
    """Nothing in a known layout: say what the folder looks like instead."""
    rep.errors.append("no dataset in a format pretrain.py reads: expected manifest*.csv (video clips), or "
                      "<split>/index.npz or <split>/<class>/<images> (images); see DATASETS.md")
    nested = _prepared_subfolders(root)
    if nested:
        rep.notes.append(f"prepared datasets one level down: {', '.join(n + '/' for n in nested)} -- point the "
                         f"root at one of them (e.g. {os.path.join(root, nested[0])})")
    # islice: a raw dataset can hold millions of files; look at the first 20k only.
    files = [os.path.relpath(p, root)
             for p in itertools.islice(glob.iglob(os.path.join(root, "**", "*"), recursive=True), 20000)]
    known = _known_source(root, files)
    if known:
        rep.notes.append(f"looks like {known}")
        return
    ext = lambda e: [f for f in files if f.lower().endswith(e)]          # noqa: E731
    if not files:
        rep.notes.append("the folder is empty")
    if ext(_VIDEO_EXT):
        rep.notes.append(f"found video files (e.g. {ext(_VIDEO_EXT)[0]!r}): decode them into frames + a "
                         "manifest (src/data/prepare_ssv2.py does this for SSv2; DATASETS.md, 'Video clips')")
    if ext(_IMAGE_EXT):
        top = [f for f in ext(_IMAGE_EXT) if os.sep not in f]
        rep.notes.append(f"found images (e.g. {ext(_IMAGE_EXT)[0]!r})" + (
            ": loose in the root -- an image dataset needs them under a split folder (<root>/train/...)"
            if top else ": as frames of videos, add a manifest; as an image set, put them in "
                        "<split>/<class>/ folders or convert with src/data/prepare_imagenet.py"))
    if ext(".tar"):
        rep.notes.append(f"found tar files (e.g. {ext('.tar')[0]!r}): WebDataset/other tars aren't read "
                         "directly; convert them (DATASETS.md)")


def check_dataset(root: str, image_size: int = 224, sample: int = 200, full: bool = False,
                  seed: int = 0, quick: bool = False) -> Report:
    """Everything under root, checked; quick (pretrain.py's startup check):
    decode only `sample` items and skip the scans over a whole index."""
    rep = Report()
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        rep.errors.append(f"{root!r} is not a folder")
        return rep
    rng = np.random.default_rng(seed)
    name = os.path.basename(root.rstrip(os.sep)) or "data"

    manifests = sorted(glob.glob(os.path.join(root, "manifest*.csv")))
    clip_dirs, clip_sizes, frame_folders = {}, {}, set()
    for m in manifests:
        facts = check_manifest(root, m, rep, image_size, sample, full, rng, quick=quick)
        if facts:
            if facts["shape"] is not None:
                clip_sizes[os.path.basename(m)] = facts["shape"]
            for d in facts["dirs"]:
                rel = os.path.relpath(d, root)
                if facts["sharded"]:
                    clip_dirs.setdefault(rel, os.path.basename(m))
                else:   # per-video frame folders (e.g. frames/<video>/): not class folders
                    frame_folders.add(rel.split(os.sep)[0])
    if len(set(clip_sizes.values())) > 1:
        shapes = {m: f"{h}x{w}" for m, (h, w) in clip_sizes.items()}
        rep.errors.append(f"frames differ in size between manifests ({shapes}): one run reads them "
                          "all at one pretrain_data.image_size, so it would fail at its first validation; "
                          "convert them at one size")
    shapes = set(clip_sizes.values())
    h, w = next(iter(shapes)) if len(shapes) == 1 else (None, None)
    clip_size = h if h is not None and h == w else None      # one square size for every manifest
    if manifests:
        base = [os.path.basename(m) for m in manifests]
        train = next((b for b in base if "train" in b), base[0])
        val = _pick_val([b for b in base if b != train], lambda b, v: v in b)
        if val is None:
            rep.notes.append("no validation manifest: pretrain.py validates on the training manifest "
                             "(val_manifest: null)")
        rep.configs[name] = {
            "name": name, "type": "clips", "root": root, "train_manifest": train, "val_manifest": val,
            "image_size": int(clip_size or image_size),
            # as stored only if every manifest's frames are image_size x image_size
            # (pretrain.py rejects anything else)
            "frames_as_stored": bool(clip_sizes) and shapes == {(image_size, image_size)},
            "text_strip": 0}

    splits = _image_split_dirs(root, exclude=frame_folders)
    facts = {s: check_image_split(root, s, rep, image_size, sample, full, rng, frames_of=clip_dirs.get(s))
             for s in splits}
    good = [s for s in splits if facts[s]]
    if good:
        train = "train" if "train" in good else good[0]
        val = _pick_val([s for s in good if s != train], lambda s, v: s.lower() == v)
        if val is None:
            rep.warnings.append("no validation split (val/validation/valid/test) next to the training split: "
                                f"set val_split to one, or to {train!r} to validate on training images")
        frames = train in clip_dirs
        key = f"{name}_frames" if frames else (f"{name}_images" if name in rep.configs else name)
        rep.configs[key] = {
            "name": key, "type": "images", "root": root,
            "train_split": train, "val_split": val or train, "image_size": image_size,
            "train_crop": {"scale": [0.2, 1.0] if frames else [0.08, 1.0], "ratio": [0.75, 4 / 3]},
            "interpolation": "bilinear", "every": 10 if frames else 1, "val_every": 50 if frames else None,
            "eval_resize": 256, "eval_batches": 20 if frames else None}

    if not manifests and not splits:
        _guess_unknown(root, rep)
    return rep


# ── pretrain.py's startup check ───────────────────────────────────────────────

def _missing(cfg: dict, root: str) -> List[str]:
    """What the config names that isn't under root (existence only: cheap)."""
    if not os.path.isdir(root):
        return [f"{root} doesn't exist" if not os.path.exists(root) else f"{root} isn't a folder"]
    out = []
    if cfg.get("type") == "clips":
        for key in ("train_manifest", "val_manifest"):
            m = cfg.get(key)
            if m and not os.path.isfile(m if os.path.isabs(m) else os.path.join(root, m)):
                out.append(f"no {key} {m!r}")
    else:
        for key in ("train_split", "val_split"):
            s = cfg.get(key)
            if not s:
                continue
            d = os.path.join(root, s)
            if os.path.isfile(os.path.join(d, "index.npz")):
                continue
            if not (os.path.isdir(d) and any(e.is_dir() for e in os.scandir(d))):
                out.append(f"no {key} {s!r} (a shard folder with index.npz, or class folders)")
    return out


_KIND = {"clips": "a video-clip", "images": "an image"}


def _wanted(cfg: dict) -> str:
    if cfg.get("type") == "clips":
        return "video clips (manifest*.csv + frames)"
    return "images (<split>/index.npz shards or <split>/<class>/ folders)"


def resolve_config(cfg: dict, want: str, image_size: int = 224, sample: int = 4) -> dict:
    """pretrain.py's check of pretrain_data before anything expensive. cfg: the
    pretrain_data config as a plain dict; want: clips | images, what the
    objectives read. type auto: returns the config detected under root (as
    check_dataset suggests it), with cfg's non-null values on top. Otherwise
    returns cfg once what it names exists. Raises ValueError saying what the
    folder holds instead and how to get from there to a run."""
    root = os.path.abspath(str(cfg["root"]))
    name, auto = cfg.get("name") or "auto", cfg.get("type") in (None, "auto")
    if not auto:
        missing = _missing(cfg, root)
        if not missing:
            return cfg
    if not os.path.isdir(root):
        raise ValueError(f"pretrain_data.root={root} doesn't exist" if not os.path.exists(root)
                         else f"pretrain_data.root={root} isn't a folder")
    rep = check_dataset(root, image_size=int(cfg.get("image_size") or image_size), sample=sample, quick=True)
    found = [c for c in rep.configs.values() if c["type"] == want]
    if auto and found:
        out = dict(found[0])
        out.update({k: v for k, v in cfg.items() if v is not None and k != "type"})
        out["root"] = root
        return out

    if auto:
        head = f"pretrain_data=auto: no {want} dataset in {root}"
    else:
        head = (f"pretrain_data={name} reads {_wanted(cfg)} from {root}, but: " + "; ".join(missing))
    lines = [head, "", f"What {root} holds (python -m src.data.check {root} for the full check):"]
    if rep.configs:
        for c in rep.configs.values():
            parts = ([f"train_manifest={c['train_manifest']}", f"val_manifest={c['val_manifest']}"]
                     if c["type"] == "clips" else [f"train_split={c['train_split']}", f"val_split={c['val_split']}"])
            lines.append(f"  - {_KIND[c['type']]} dataset: {', '.join(parts)}")
        lines += [f"  - {m}" for m in rep.errors]
    else:
        lines += [f"  - {m}" for m in rep.errors + rep.notes]
    other = [c for c in rep.configs.values() if c["type"] != want]
    if found:
        lines += ["", f"It is {_KIND[want]} dataset pretrain.py reads, in another layout than pretrain_data={name} "
                      f"expects: use pretrain_data=auto, or the config python -m src.data.check prints."]
    elif other:
        reads = "videos" if want == "clips" else "images"
        lines += ["", f"It holds {_KIND[other[0]['type']]} dataset, but the objectives read {want} ({reads}). "
                      + ("A clip dataset's frame shards also work as images (pretrain_data=auto does that)."
                         if want == "images" else
                         "Image datasets can't feed clip objectives (R3M, VIP): they need videos.")]
    raise ValueError("\n".join(lines))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="the dataset folder (the value pretrain_data.root would get)")
    p.add_argument("--image-size", type=int, default=224, help="encoder input side to check frames against")
    p.add_argument("--sample", type=int, default=200, help="videos/images decoded per manifest/split")
    p.add_argument("--full", action="store_true", help="decode every frame and image (slow)")
    p.add_argument("--quick", action="store_true",
                   help="what pretrain.py checks at startup: a few samples, no whole-index scans")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rep = check_dataset(a.root, a.image_size, 8 if a.quick else a.sample, a.full, a.seed, quick=a.quick)
    print(rep.format())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
