"""
ImageNet-1k (ILSVRC2012) -> shards (src/data/shards.py) for pretrain.py.

    python -m src.data.prepare_imagenet --source imagenet-object-localization-challenge.zip \
        --out /scratch/.../imagenet --workers 16

--source is any of:
  - Kaggle's zip (kaggle competitions download -c imagenet-object-localization-challenge),
    read directly: its 1.28M files are never extracted
  - that zip, extracted (the folder holding ILSVRC/ and LOC_val_solution.csv)
  - torchvision's layout: <dir>/train/<wnid>/*.JPEG, <dir>/val/<wnid>/*.JPEG
  - that layout packed in one tar (e.g. a cluster's ILSVRC2012_imagefolder.tar;
    train/ and val/ may sit under any prefix inside it), streamed once from
    start to end -- sequential reads, never extracted. --per-class keeps the
    first N per class in archive order.

Memory: with the zip, each worker parses its index (~1.5M entries) --
several hundred MB per worker, so ~32 GB for 16 workers.

Writes <out>/classes.txt (wnid and name; label = line number, the standard
sorted-wnid order, as torchvision's ImageNet models use) and <out>/<split>/
(shards + index.npz). Images are stored byte-for-byte by default;
--short-side N re-encodes those larger than N on the short side (faster to
decode in training, at a small loss of detail in tight crops). --classes
(a file of wnids, e.g. ImageNet-100's) and --per-class N make subsets.
"""

import argparse
import csv
import io
import json
import os
import re
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

KAGGLE_ROOT = "ILSVRC/Data/CLS-LOC"
_WNID = re.compile(r"n\d{8}")
_IMAGE = (".jpeg", ".jpg", ".png")


def is_tar_source(path: str) -> bool:
    return os.path.isfile(path) and not zipfile.is_zipfile(path) and tarfile.is_tarfile(path)


def _tar_stream(path: str, splits):
    """(split, key, wnid, bytes) for every <...>/<split>/<wnid>/<file> image in
    a tar, in archive order, reading it front to back once."""
    with tarfile.open(path, "r|*") as t:
        for m in t:
            parts = m.name.split("/")
            if (m.isfile() and len(parts) >= 3 and parts[-3] in splits
                    and _WNID.fullmatch(parts[-2]) and parts[-1].lower().endswith(_IMAGE)):
                yield parts[-3], parts[-1], parts[-2], t.extractfile(m).read()


class Source:
    """ImageNet as a list of (key, wnid, ref) per split, plus read(ref) -> bytes."""

    def __init__(self, path: str):
        self.path = path
        self.zip = None
        if zipfile.is_zipfile(path):
            self.kind = "kaggle_zip"
        elif os.path.isfile(os.path.join(path, "LOC_val_solution.csv")):
            self.kind = "kaggle_dir"
        elif os.path.isdir(os.path.join(path, "train")):
            self.kind = "folders"
        else:
            raise ValueError(f"{path!r} is neither Kaggle's ImageNet zip, its extracted folder, "
                             "a folder with train/<wnid>/ subfolders, nor a tar of that layout")

    def _zip(self):
        if self.zip is None:
            self.zip = zipfile.ZipFile(self.path)
        return self.zip

    def _names(self):
        if self.kind == "kaggle_zip":
            return self._zip().namelist()
        out = []
        for d, _, files in os.walk(self.path):
            out += [os.path.relpath(os.path.join(d, f), self.path) for f in files]
        return out

    def read(self, ref: str) -> bytes:
        if self.kind == "kaggle_zip":
            return self._zip().read(ref)
        with open(os.path.join(self.path, ref), "rb") as f:
            return f.read()

    def _text(self, name):
        try:
            return self.read(name).decode()
        except (KeyError, FileNotFoundError):
            return None

    def synset_names(self) -> dict:
        text = self._text("LOC_synset_mapping.txt")
        if text is None:
            return {}
        return {line.split(" ", 1)[0]: line.split(" ", 1)[1].strip() for line in text.splitlines() if line}

    def entries(self, split: str):
        names = self._names()
        image = lambda n: n.lower().endswith((".jpeg", ".jpg", ".png"))
        if self.kind == "folders":
            pre = f"{split}/"
            return sorted((os.path.basename(n), n.split("/")[1], n) for n in names
                          if n.startswith(pre) and n.count("/") == 2 and image(n))
        pre = f"{KAGGLE_ROOT}/{split}/"
        if split == "train":
            return sorted((os.path.basename(n), n.split("/")[-2], n) for n in names
                          if n.startswith(pre) and image(n))
        # Kaggle's validation images sit in one folder; labels are in a csv.
        rows = csv.DictReader(io.StringIO(self._text(f"LOC_{split}_solution.csv")))
        wnid = {r["ImageId"]: r["PredictionString"].split()[0] for r in rows}
        return sorted((os.path.basename(n), wnid[os.path.splitext(os.path.basename(n))[0]], n)
                      for n in names if n.startswith(pre) and image(n))


_source, _short_side, _quality, _verify = None, None, None, True


def _init(path, short_side, quality, verify):
    """path None: workers only process bytes handed to them (tar streaming)."""
    global _source, _short_side, _quality, _verify
    _source = Source(path) if path is not None else None
    _short_side, _quality, _verify = short_side, quality, verify
    import torch
    torch.set_num_threads(1)    # one process per CPU already
    if short_side is not None:  # OpenCV is needed only to re-encode
        import cv2
        cv2.setNumThreads(1)


def _checked(ref, data):
    """The bytes to store, after decoding them as training will (unless
    --no-verify), so a file the training loader can't read fails here."""
    if _verify:
        from src.data.images import decode
        try:
            decode(data)
        except Exception as e:
            raise ValueError(f"{ref}: can't be decoded ({type(e).__name__}: {e})") from None
    return data


def _load(ref):
    return _process(ref, _source.read(ref))


def _process(ref, data):
    """Bytes to store for one image: as they are, or re-encoded smaller."""
    if _short_side is None:
        return _checked(ref, data)
    import cv2
    import numpy as np
    # Ignore EXIF orientation, as torchvision's decode_image (the stored-as-is
    # path) and the usual PIL pipeline do; OpenCV would otherwise rotate.
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"{ref}: can't be decoded")
    h, w = img.shape[:2]
    if min(h, w) <= _short_side:
        return _checked(ref, data)      # kept as is: still check it with the training decoder
    s = _short_side / min(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return _checked(ref, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _quality])[1].tobytes())


def _load_chunk(refs):
    return [_load(r) for r in refs]


def _process_chunk(items):
    return [(meta, _process(meta[1], data)) for meta, data in items]


def _ordered_map(pool, fn, items, chunksize=64, in_flight=64):
    from src.data.shards import ordered_map
    return ordered_map(pool, fn, items, chunksize, in_flight)


def _ordered_load(pool, refs, chunksize=64, in_flight=64):
    """pool.map(_load, refs) in order, bounded (see _ordered_map)."""
    return _ordered_map(pool, _load_chunk, refs, chunksize, in_flight)


def _write_meta(a, out, images, classes, shards):
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"source": os.path.abspath(a.source), "images": images, "classes": classes,
                   "short_side": a.short_side, "quality": a.quality if a.short_side else None,
                   "shards": shards}, f, indent=1)


def _convert_tar(a):
    """One pass over a tar holding train/<wnid>/ and val/<wnid>/: every split
    written at once. Labels need all classes, which are known only at the end,
    so they're assigned then (sorted wnids, as from folders)."""
    from src.data.shards import ShardWriter
    keep = None
    if a.classes:
        with open(a.classes) as f:
            keep = {line.split()[0] for line in f if line.strip()}
    os.makedirs(a.out, exist_ok=True)
    writers = {s: ShardWriter(os.path.join(a.out, s), max_bytes=int(a.shard_gb * (1 << 30))) for s in a.splits}
    wnids_of = {s: [] for s in a.splits}
    count = {}

    def items():
        for split, key, wnid, data in _tar_stream(a.source, set(a.splits)):
            if keep is not None and wnid not in keep:
                continue
            if a.per_class:
                if count.get((split, wnid), 0) >= a.per_class:
                    continue
                count[(split, wnid)] = count.get((split, wnid), 0) + 1
            yield (split, key, wnid), data

    t0, n = time.time(), 0
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(None, a.short_side, a.quality, not a.no_verify)) as pool:
        for (split, key, wnid), data in _ordered_map(pool, _process_chunk, items(), in_flight=4 * a.workers):
            writers[split].add(key, data)
            wnids_of[split].append(wnid)
            n += 1
            if n % 50_000 == 0:
                print(f"  {n:,} images  {n / (time.time() - t0):.0f} images/s")
    found = {s: sorted(set(w)) for s, w in wnids_of.items()}
    wnids = sorted(keep) if keep is not None else found.get("train") or next(iter(found.values()))
    if keep is not None and set(wnids) - set().union(*found.values()):
        raise SystemExit(f"--classes lists wnids not in the source: {sorted(set(wnids) - set().union(*found.values()))[:5]}")
    label = {w: i for i, w in enumerate(wnids)}
    for split, w in writers.items():
        stray = set(found[split]) - set(label)
        if stray:
            raise SystemExit(f"{split} has classes that train lacks: {sorted(stray)[:5]}")
        if not wnids_of[split]:
            raise SystemExit(f"no {split}/<wnid>/ images in {a.source}")
        w.labels = [label[x] for x in wnids_of[split]]
        w.close()
        _write_meta(a, os.path.join(a.out, split), len(w.labels), len(wnids), w.num_shards)
        print(f"{split}: {len(w.labels):,} images, {len(found[split])} classes -> {w.num_shards} shards")
    with open(os.path.join(a.out, "classes.txt"), "w") as f:
        f.writelines(f"{w}\n" for w in wnids)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--classes", help="file with one wnid per line: only these classes")
    p.add_argument("--per-class", type=int, help="at most N images per class (first N by name)")
    p.add_argument("--short-side", type=int, help="re-encode images larger than this on the short side")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality for --short-side")
    p.add_argument("--shard-gb", type=float, default=1.0)
    p.add_argument("--no-verify", action="store_true",
                   help="skip decoding every stored image once (on by default: a few minutes)")
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    a = p.parse_args(argv)

    from src.data.shards import ShardWriter
    if is_tar_source(a.source):
        print(f"Source: {a.source} (tar, streamed)")
        return _convert_tar(a)
    src = Source(a.source)
    print(f"Source: {a.source} ({src.kind})")
    splits = {s: src.entries(s) for s in a.splits}
    wnids = sorted({w for _, w, _ in splits.get("train") or next(iter(splits.values()))})
    if a.classes:
        with open(a.classes) as f:
            keep = {line.split()[0] for line in f if line.strip()}
        unknown = keep - set(wnids)
        if unknown:
            raise SystemExit(f"--classes lists wnids not in the source: {sorted(unknown)[:5]}")
        wnids = sorted(keep)
    label = {w: i for i, w in enumerate(wnids)}
    os.makedirs(a.out, exist_ok=True)
    names = src.synset_names()
    with open(os.path.join(a.out, "classes.txt"), "w") as f:
        f.writelines(f"{w} {names.get(w, '')}".rstrip() + "\n" for w in wnids)

    for split, entries in splits.items():
        entries = [e for e in entries if e[1] in label]
        if a.per_class:
            kept, count = [], {}
            for e in entries:                      # sorted by name, so the first N per class
                if count.get(e[1], 0) < a.per_class:
                    kept.append(e)
                    count[e[1]] = count.get(e[1], 0) + 1
            entries = kept
        print(f"{split}: {len(entries):,} images, {len({e[1] for e in entries})} classes")
        t0, out = time.time(), os.path.join(a.out, split)
        with ShardWriter(out, max_bytes=int(a.shard_gb * (1 << 30))) as w, \
                ProcessPoolExecutor(a.workers, initializer=_init,
                                    initargs=(a.source, a.short_side, a.quality, not a.no_verify)) as pool:
            for i, ((key, wnid, _), data) in enumerate(
                    zip(entries, _ordered_load(pool, [e[2] for e in entries], in_flight=4 * a.workers)), 1):
                w.add(key, data, label[wnid])
                if i % 50_000 == 0 or i == len(entries):
                    print(f"  {i:,}/{len(entries):,}  {i / (time.time() - t0):.0f} images/s")
        _write_meta(a, out, len(entries), len(wnids), w.num_shards)
        print(f"  -> {out}: {w.num_shards} shards")


if __name__ == "__main__":
    main()
