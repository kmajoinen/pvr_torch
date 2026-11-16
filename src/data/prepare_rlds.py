"""
RLDS / TFDS robot datasets (Open X-Embodiment, BridgeData's RLDS release,
...) -> clip dataset (src/data/clips.py, shards layout): one clip per
episode, one camera's frames as size x size JPEGs, text = the episode's
language instruction. No TensorFlow needed: TFRecords and tf.train.Example
are read directly.

    python -m src.data.prepare_rlds \
        --source /scratch/shareddata/dldata/open_x_embodiment/tensorflow_datasets/bridge \
        --out $WRKDIR/data/oxe/bridge --workers 16

--source is a TFDS dataset folder: the one holding features.json,
dataset_info.json and <name>-<split>.tfrecord-* files, or a folder up to
three levels above it (<name>/<version>/; the highest version is taken).
Fields are found from features.json, so no per-dataset code:

  camera   an Image feature under steps/observation. Default: one named
           `image` or `rgb`, else the first that isn't a wrist/gripper/hand
           camera. --camera <name> picks another, --camera all stores every
           camera (one clip per camera per episode). The run lists them.
  text     the first non-empty step of language_instruction,
           natural_language_instruction or instruction (language_table's
           codepoint arrays too). --text <field> picks another, --text none
           leaves txt empty.

Splits: train -> train; the dataset's val/validation/test split ->
validation. Datasets with only train (most of OXE) hold out the last
--holdout of its files (default 2%) as validation. Writes, in --out:

  train/, validation/      shards (src/data/shards.py), keys
          <episode>/000001.jpg..., episode = <file number>_<index in file>
  manifest_train.csv, manifest_validation.csv   one row per episode
          (and camera): path, len, txt, start, episode, camera

The shard folders are also image datasets of the frames (MAE/MVP), so
pretrain_data=auto finds both a clips and a frames config in --out.

--every K keeps every Kth step (from the first): for datasets recorded
faster than the others, e.g. DROID at 15 Hz -> 5 Hz with --every 3, Bridge's
rate. R3M and VIP sample frames by position in the clip, so the step rate
sets how far apart in time their frames are.

--limit N converts the first N files of each split and prints the rate in
minutes per GB read, and, scaled by GB read the same way, the hours and the
output size the whole dataset would take. Compare that size with `quota`
before the full run. Not resumable: an interrupted run is converted again. A file that
ends early (truncated) keeps the episodes before the break (reported).
Each worker holds one file's converted JPEGs (~25 KB per frame at 224px,
quality 95, measured on Bridge).

Needs OpenCV (PNG and JPEG frames are both read).
"""

import argparse
import glob
import json
import os
import re
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import numpy as np

_VAL_SPLITS = ("val", "validation", "test")
_TEXT_FIELDS = ("steps/language_instruction", "steps/observation/natural_language_instruction",
                "steps/observation/instruction")
_SIDE_CAMERAS = ("wrist", "gripper", "hand", "eye", "depth")


# ── TFRecord + tf.train.Example, without TensorFlow ──────────────────────────

def read_records(path: str):
    """Each record's bytes. TFRecord framing: uint64 length, uint32 CRC,
    data, uint32 CRC (CRCs not checked: a broken record fails to parse)."""
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) < 12:
                raise EOFError("file ends inside a record header (truncated)")
            n = struct.unpack("<Q", head[:8])[0]
            data = f.read(n)
            if len(data) < n or len(f.read(4)) < 4:
                raise EOFError("file ends inside a record (truncated)")
            yield data


def _varint(b, i: int):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _fields(b):
    """(field number, wire type, value) of one protobuf message; values of
    length-delimited fields are slices of b."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        num, wire = key >> 3, key & 7
        if wire == 2:
            size, i = _varint(b, i)
            yield num, wire, b[i:i + size]
            i += size
        elif wire == 0:
            v, i = _varint(b, i)
            yield num, wire, v
        elif wire == 5:
            yield num, wire, b[i:i + 4]
            i += 4
        elif wire == 1:
            yield num, wire, b[i:i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _feature(b):
    """tf.train.Feature: list of bytes, or a float32 / int64 array."""
    for kind, _, lst in _fields(b):
        if kind == 1:                                            # bytes_list
            return [bytes(v) for _, _, v in _fields(lst)]
        if kind == 2:                                            # float_list (packed or not)
            parts = [np.frombuffer(bytes(v), "<f4") for _, _, v in _fields(lst)]
            return np.concatenate(parts) if parts else np.zeros(0, np.float32)
        if kind == 3:                                            # int64_list (packed or not)
            vals = []
            for _, wire, v in _fields(lst):
                if wire == 2:
                    j = 0
                    while j < len(v):
                        x, j = _varint(v, j)
                        vals.append(x)
                else:
                    vals.append(v)
            return np.array(vals, dtype=np.uint64).view(np.int64)
    return []


def parse_example(record: bytes, keys=None) -> dict:
    """A serialized tf.train.Example as {feature name: value}; `keys` parses
    only those (an episode's record holds every step of every field)."""
    out = {}
    for num, _, feats in _fields(memoryview(record)):
        if num != 1:
            continue
        for _, _, entry in _fields(feats):
            key = value = None
            for f, _, v in _fields(entry):
                if f == 1:
                    key = bytes(v).decode()
                elif f == 2:
                    value = v
            if key is not None and (keys is None or key in keys):
                out[key] = _feature(value) if value is not None else []
    return out


# ── The dataset's schema (features.json, dataset_info.json) ───────────────────

def _leaves(node: dict, prefix: str = ""):
    """(field name as in the records, feature class, shape, dtype) of every leaf."""
    sub = node.get("featuresDict", {}).get("features")
    if sub is not None:
        for k, v in sub.items():
            yield from _leaves(v, f"{prefix}/{k}" if prefix else k)
        return
    if "sequence" in node:
        yield from _leaves(node["sequence"]["feature"], prefix)
        return
    info = node.get("image") or node.get("tensor") or node.get("text") or {}
    dims = info.get("shape", {}).get("dimensions")
    yield (prefix, node.get("pythonClassName", "").rsplit(".", 1)[-1],
           tuple(int(d) for d in dims) if dims else (), info.get("dtype", "string" if "text" in node else None))


def find_datasets(source: str, depth: int = 3) -> List[str]:
    """Folders under source (itself included) holding a TFDS dataset."""
    found = []
    for level in range(depth + 1):
        pattern = os.path.join(source, *["*"] * level, "features.json")
        found += [os.path.dirname(p) for p in sorted(glob.glob(pattern))
                  if os.path.isfile(os.path.join(os.path.dirname(p), "dataset_info.json"))]
    return found


def _version_key(path: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"[._-]", os.path.basename(path))]


def find_dataset(source: str) -> str:
    """The one TFDS dataset folder at or under source: its highest version if
    there are several; an error naming them if they're different datasets."""
    if not os.path.isdir(source):
        sys.exit(f"--source {source!r} isn't a folder")
    found = find_datasets(source)
    if not found:
        sys.exit(f"no TFDS dataset (features.json + dataset_info.json) in {source} or 3 levels below")
    names = {}
    for d in found:
        with open(os.path.join(d, "dataset_info.json")) as f:
            names.setdefault(json.load(f).get("name", d), []).append(d)
    if len(names) > 1:
        listing = ", ".join(sorted(names)[:15]) + (" ..." if len(names) > 15 else "")
        sys.exit(f"{len(names)} datasets under {source} ({listing}): point --source at one of them")
    return max(next(iter(names.values())), key=_version_key)


def read_schema(d: str) -> dict:
    with open(os.path.join(d, "features.json")) as f:
        leaves = list(_leaves(json.load(f)))
    with open(os.path.join(d, "dataset_info.json")) as f:
        info = json.load(f)
    images = [(k, shape) for k, cls, shape, _ in leaves
              if cls == "Image" and k.startswith("steps/") and (not shape or shape[-1] == 3)]
    texts = [(k, dtype, shape) for k, cls, shape, dtype in leaves
             if k.startswith("steps/") and "embedding" not in k
             and (cls == "Text" or (dtype == "string" and shape == ()) or
                  (k.endswith("instruction") and dtype in ("int32", "int64") and len(shape) == 1))]
    files = {}
    for s in info.get("splits", []):
        paths = sorted(glob.glob(os.path.join(d, f"*-{s['name']}.tfrecord*")))
        if paths:
            files[s["name"]] = paths
    return {"name": info.get("name", os.path.basename(d)), "dir": d, "images": images, "texts": texts,
            "files": files}


def pick_cameras(images, wanted: Optional[str]) -> List[str]:
    keys = [k for k, _ in images]
    short = {k.rsplit("/", 1)[-1]: k for k in keys}
    if not keys:
        sys.exit("the dataset has no RGB Image feature under steps/")
    if wanted == "all":
        return keys
    if wanted:
        if wanted in keys:
            return [wanted]
        if wanted in short:
            return [short[wanted]]
        sys.exit(f"no camera {wanted!r}; the dataset has: {', '.join(short)}")
    for name in ("image", "rgb"):
        if name in short:
            return [short[name]]
    main = [k for k in keys if not any(s in k.rsplit("/", 1)[-1] for s in _SIDE_CAMERAS)]
    return [(main or keys)[0]]


def pick_text(texts, wanted: Optional[str]):
    """(field, its shape) or None for no text."""
    fields = {k: shape for k, _, shape in texts}
    if wanted == "none":
        return None
    if wanted:
        for k in (wanted, f"steps/{wanted}", f"steps/observation/{wanted}"):
            if k in fields:
                return k, fields[k]
        sys.exit(f"no text field {wanted!r}; the dataset has: {', '.join(fields) or 'none'}")
    for k in _TEXT_FIELDS:
        if k in fields:
            return k, fields[k]
    return next(iter(fields.items()), None)


def episode_text(value, shape) -> str:
    """The first non-empty step's instruction: strings, or (language_table)
    fixed-length arrays of Unicode codepoints, zero-padded."""
    if isinstance(value, list):
        steps = [v.decode("utf-8", "replace").strip() for v in value]
    else:
        n = shape[0] if shape else len(value)
        steps = ["".join(map(chr, row[row > 0])).strip() for row in np.asarray(value).reshape(-1, n)]
    return next((s for s in steps if s), "")


# ── Conversion ────────────────────────────────────────────────────────────────

def _file_number(path: str, default: int) -> int:
    m = re.search(r"tfrecord-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else default


def _convert_file(jobs):
    """Per (path, number, cameras, text field, text shape, every, size,
    resize, quality): (episodes, error). An episode is (name, {camera:
    [JPEG]}, text)."""
    import cv2
    from src.data.prepare_ssv2 import _resize
    cv2.setNumThreads(1)
    out = []
    for path, number, cams, text_key, text_shape, every, size, resize, quality in jobs:
        episodes, error = [], None
        keys = set(cams) | ({text_key} if text_key else set())
        try:
            for k, record in enumerate(read_records(path)):
                ex = parse_example(record, keys)
                frames = {}
                for cam in cams:
                    jpgs = []
                    for data in ex.get(cam, [])[::every]:
                        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            raise ValueError(f"episode {k}: undecodable {cam} frame")
                        jpgs.append(cv2.imencode(".jpg", _resize(img, size, resize),
                                                 [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
                    frames[cam] = jpgs
                text = episode_text(ex[text_key], text_shape) if text_key and text_key in ex else ""
                episodes.append((f"{number:05d}_{k:04d}", frames, text))
        except Exception as e:      # noqa: BLE001 -- keep the episodes before a broken one
            error = f"{type(e).__name__}: {e}"
        out.append((path, episodes, error))
    return out


def plan_splits(files: dict, holdout: float):
    """{output split: files}: train, and validation from the dataset's own
    val/test split or the last `holdout` of train's files."""
    train_name = "train" if "train" in files else next(iter(files), None)
    if train_name is None:
        return {}
    plan = {"train": list(files[train_name])}
    val_name = next((s for s in _VAL_SPLITS if s in files and s != train_name), None)
    if val_name:
        plan["validation"] = list(files[val_name])
    elif holdout > 0 and len(plan["train"]) >= 2:
        k = max(1, round(len(plan["train"]) * holdout))
        plan["train"], plan["validation"] = plan["train"][:-k], plan["train"][-k:]
    return plan


def convert_split(a, split: str, paths: List[str], cams, text, pool) -> dict:
    import pandas as pd
    from src.data.shards import ShardWriter, ordered_map
    multi = len(cams) > 1
    rows, broken, n_frames = [], [], 0
    jobs = [(p, _file_number(p, i), cams, text[0] if text else None, text[1] if text else (),
             a.every, a.size, a.resize, a.quality) for i, p in enumerate(paths)]
    with ShardWriter(os.path.join(a.out, split), max_bytes=int(a.shard_gb * (1 << 30))) as w:
        for i, (path, episodes, err) in enumerate(
                ordered_map(pool, _convert_file, jobs, chunksize=1, in_flight=a.workers + 2), 1):
            if err:
                broken.append((os.path.basename(path), len(episodes), err))
            for name, frames, txt in episodes:
                for cam, jpgs in frames.items():
                    if not jpgs:
                        continue
                    cam_name = cam.rsplit("/", 1)[-1]
                    video = f"{name}_{cam_name}" if multi else name
                    rows.append({"path": split, "len": len(jpgs), "txt": txt, "start": len(w.keys),
                                 "episode": name, "camera": cam_name})
                    for n, data in enumerate(jpgs, 1):
                        w.add(f"{video}/{n:06d}.jpg", data)
                    n_frames += len(jpgs)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {split}: {i:,}/{len(jobs):,} files  {len(rows):,} clips  {n_frames:,} frames", flush=True)
    out_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(a.out, split, "*")))
    m = pd.DataFrame(rows, columns=["path", "len", "txt", "start", "episode", "camera"])
    m.to_csv(os.path.join(a.out, f"manifest_{split}.csv"), index=False)
    if len(m):
        print(f"Wrote manifest_{split}.csv: {len(m):,} clips, {int(m['len'].sum()):,} frames, median "
              f"{int(m['len'].median())} per clip, {int((m['len'] < 3).sum()):,} under 3 frames "
              f"(R3M/VIP skip those), {int((m['txt'] == '').sum()):,} without text")
    else:
        print(f"Wrote manifest_{split}.csv: no clips")
    for name, kept, err in broken[:10]:
        print(f"  {name}: stopped after {kept} episodes ({err})")
    if len(broken) > 10:
        print(f"  ... {len(broken) - 10} more files stopped early")
    return {"clips": len(m), "broken": len(broken), "bytes": out_bytes}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="TFDS dataset folder (or up to 3 levels above it)")
    p.add_argument("--out", required=True, help="dataset root (shards and manifests go here)")
    p.add_argument("--camera", default=None, help="camera (Image feature) name, or `all` (default: main camera)")
    p.add_argument("--text", default=None, help="text field, or `none` (default: the language instruction)")
    p.add_argument("--holdout", type=float, default=0.02,
                   help="fraction of train's files held out as validation when the dataset has no val/test split")
    p.add_argument("--every", type=int, default=1, help="keep every Kth step (e.g. 3: 15 Hz -> 5 Hz)")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--resize", choices=("squash", "center_crop"), default="squash")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality")
    p.add_argument("--shard-gb", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None, help="only the first N files of each split (timing run)")
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    a = p.parse_args(argv)
    if a.every < 1:
        sys.exit("--every must be at least 1")

    d = find_dataset(a.source)
    schema = read_schema(d)
    cams = pick_cameras(schema["images"], a.camera)
    text = pick_text(schema["texts"], a.text)
    plan = plan_splits(schema["files"], a.holdout)
    if not plan:
        sys.exit(f"no <name>-<split>.tfrecord* files in {d} for the splits dataset_info.json lists")
    full = {s: list(ps) for s, ps in plan.items()}
    if a.limit is not None:
        plan = {s: ps[:a.limit] for s, ps in plan.items()}
    size = lambda ps: sum(os.path.getsize(q) for q in ps) / 2**30       # noqa: E731
    cam_names = [k.rsplit("/", 1)[-1] for k, _ in schema["images"]]
    print(f"{schema['name']} ({d})")
    print(f"  cameras: {', '.join(cam_names)} -> storing {', '.join(c.rsplit('/', 1)[-1] for c in cams)}")
    print(f"  text: {text[0] if text else 'none'}" + (
        f" (others: {', '.join(k for k, _, _ in schema['texts'] if not text or k != text[0])})"
        if len(schema["texts"]) > 1 else ""))
    print("  " + ", ".join(f"{s}: {len(ps):,} files, {size(ps):.1f} GB" for s, ps in plan.items())
          + f" -> {a.out} ({a.size}px {a.resize}, {f'every {a.every} steps, ' if a.every > 1 else ''}"
            f"{a.workers} workers)")

    os.makedirs(a.out, exist_ok=True)
    t0, out_gb = time.time(), 0.0
    with ProcessPoolExecutor(a.workers) as pool:
        for split, paths in plan.items():
            out_gb += convert_split(a, split, paths, cams, text, pool)["bytes"] / 2**30
    gb, minutes = sum(size(ps) for ps in plan.values()), (time.time() - t0) / 60
    print(f"Converted {gb:.1f} GB in {minutes:.1f} min: {minutes / max(gb, 1e-9):.2f} min/GB "
          f"with {a.workers} workers; wrote {out_gb:.2f} GB ({out_gb / max(gb, 1e-9):.3f} GB per GB read)")
    if a.limit is not None:
        full_gb = sum(size(ps) for ps in full.values())
        scale = full_gb / max(gb, 1e-9)
        print(f"  the whole dataset is {full_gb:.1f} GB: about {minutes * scale / 60:.1f} h at this rate "
              "(not resumable: give the job a time limit well above that), output about "
              f"{out_gb * scale:.1f} GB (check `quota`: a run that fills it fails partway)")


def detect(root: str, files: List[str]) -> Optional[str]:
    """check.py's hint for a raw RLDS/TFDS folder (relative paths under root)."""
    if not (any("tfrecord" in f for f in files) or
            any(f.endswith(("dataset_info.json", "features.json")) for f in files)):
        return None
    # globbed, not taken from `files`: those are the first 20k entries, a few
    # datasets' worth in a folder of many (OXE: ~1000 files each)
    # grouped by dataset name, as find_dataset does: versions of one dataset are one
    found = find_datasets(root) if os.path.isdir(root) else []
    tops = {}
    for d in found:
        try:
            with open(os.path.join(d, "dataset_info.json")) as f:
                name = json.load(f).get("name")
        except (OSError, ValueError):
            name = None
        tops.setdefault(name or os.path.relpath(d, root).split(os.sep)[0],
                        os.path.relpath(d, root).split(os.sep)[0])
    tops = sorted(set(tops.values())) if len(tops) > 1 else []
    cmd = "python -m src.data.prepare_rlds --source {} --out <dataset folder>"
    if len(tops) > 1:
        return (f"{len(tops)} TFDS/RLDS datasets (e.g. {', '.join(tops[:5])}): convert one at a time with "
                + cmd.format(os.path.join(root, "<name>")) + " (run_prepare_oxe.sh)")
    return "a TFDS/RLDS dataset (TFRecords): convert with " + cmd.format(root) + " (run_prepare_oxe.sh)"


if __name__ == "__main__":
    main()
