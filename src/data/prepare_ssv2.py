"""
Something-Something v2 -> clip dataset (src/data/clips.py layout): every frame
of every video as a size x size JPEG, plus a manifest with each video's
description.

    python -m src.data.prepare_ssv2 \
        --videos /scratch/.../20bn-something-something-v2 \
        --labels /scratch/.../labels/something-something-v2-train.json \
        --out /scratch/.../ssv2_frames --split train --workers 32

Run once per split (train, validation). Writes <out>/frames/<id>/000001.jpg...
and <out>/manifest_<split>.csv. Re-running skips videos already done with the
same --size/--resize/--quality (a .done file in each folder records them;
different settings reconvert), so an interrupted job can just be restarted.
--limit N converts only the first N videos of the split (a quick subset).

Text: SSv2's per-video `label` ("putting a pen on a table") by default, the
closest match to Ego4D's narrations that R3M used; `template` gives the
class template instead ("putting something on something").

Resize: `squash` (default) resizes the whole frame to size x size, which is
what R3M's README describes for its Ego4D frames; `center_crop` resizes the
short side and crops the middle instead, keeping the aspect ratio.

Needs PyAV (pip install av) and OpenCV.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def _resize(frame, size: int, mode: str):
    import cv2
    h, w = frame.shape[:2]
    if mode == "squash":
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (nh - size) // 2, (nw - size) // 2
    return frame[top:top + size, left:left + size]


def convert_video(video_path: str, out_dir: str, size: int, resize: str, quality: int) -> int:
    """Decode every frame, write 000001.jpg..., return the frame count.
    Skipped (count read back) if out_dir's .done marker records the same
    settings; otherwise the folder's frames are replaced."""
    import av
    import cv2
    settings = {"size": size, "resize": resize, "quality": quality}
    done = os.path.join(out_dir, ".done")
    try:
        with open(done) as f:
            record = json.load(f)
    except (OSError, ValueError):      # missing, empty, or not JSON: redo the video
        record = None
    if isinstance(record, dict) and {k: record.get(k) for k in settings} == settings:
        return int(record["frames"])
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, name))
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            rgb = _resize(frame.to_ndarray(format="rgb24"), size, resize)
            n += 1
            cv2.imwrite(os.path.join(out_dir, f"{n:06d}.jpg"), rgb[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
    with open(done + ".tmp", "w") as f:
        json.dump({"frames": n, **settings}, f)
    os.replace(done + ".tmp", done)
    return n


def _job(args):
    import cv2
    cv2.setNumThreads(1)        # one process per CPU already
    vid, video_path, out_dir, size, resize, quality = args
    try:
        return vid, convert_video(video_path, out_dir, size, resize, quality), None
    except Exception as e:      # a corrupt video shouldn't stop the whole split
        return vid, 0, f"{type(e).__name__}: {e}"


def _text(entry: dict, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "template":
        return entry["template"].replace("[", "").replace("]", "").lower()
    return entry["label"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, help="folder with <id>.webm files")
    p.add_argument("--labels", required=True, help="the split's json, e.g. something-something-v2-train.json")
    p.add_argument("--out", required=True, help="dataset root (frames/ and manifests go here)")
    p.add_argument("--split", required=True, help="names the manifest: manifest_<split>.csv")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--resize", choices=("squash", "center_crop"), default="squash")
    p.add_argument("--text", choices=("label", "template", "none"), default="label")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality")
    p.add_argument("--ext", default="webm")
    p.add_argument("--limit", type=int, default=None, help="only the first N videos")
    # The CPUs this process may use (under Slurm: the allocation, not the node).
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    a = p.parse_args(argv)

    with open(a.labels) as f:
        entries = json.load(f)
    if a.limit is not None:
        entries = entries[:a.limit]
    if a.text != "none" and entries and "label" not in entries[0]:
        sys.exit(f"{a.labels} has no labels (the test split?); use --text none")
    jobs = [(e["id"], os.path.join(a.videos, f"{e['id']}.{a.ext}"),
             os.path.join(a.out, "frames", e["id"]), a.size, a.resize, a.quality) for e in entries]
    print(f"{len(jobs):,} videos -> {a.out} ({a.size}px {a.resize}, {a.workers} workers)")

    lengths, failed, t0 = {}, [], time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for i, (vid, n, err) in enumerate(pool.map(_job, jobs, chunksize=16), 1):
            if err is not None or n == 0:
                failed.append((vid, err or "no frames"))
            else:
                lengths[vid] = n
            if i % 1000 == 0 or i == len(jobs):
                print(f"  {i:,}/{len(jobs):,}  {i / (time.time() - t0):.0f} videos/s  failed={len(failed)}")

    import pandas as pd
    rows = [{"path": os.path.join("frames", e["id"]), "len": lengths[e["id"]], "txt": _text(e, a.text)}
            for e in entries if e["id"] in lengths]
    manifest = os.path.join(a.out, f"manifest_{a.split}.csv")
    pd.DataFrame(rows, columns=["path", "len", "txt"]).to_csv(manifest, index=False)
    total = sum(lengths.values())
    print(f"Wrote {manifest}: {len(rows):,} videos, {total:,} frames"
          + (f", mean {total / len(rows):.1f} per video" if rows else ""))
    if failed:
        print(f"{len(failed)} videos failed (left out of the manifest), e.g.:")
        for vid, err in failed[:5]:
            print(f"  {vid}: {err}")


if __name__ == "__main__":
    main()
