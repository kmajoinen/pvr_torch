"""
EPIC-KITCHENS-100 -> clip dataset (src/data/clips.py, shards layout): each
video's frames resampled to --fps, as size x size JPEGs, stored once; one
manifest row per narrated action segment, pointing into its video's frames.

    python -m src.data.prepare_epic \
        --videos /scratch/shareddata/dldata/epic-kitchens/EPIC-KITCHENS \
        --annotations epic-kitchens-100-annotations \
        --out /scratch/.../epic --split train --workers 32

Run once per split (train, validation). --videos holds <P##>/videos/<video>.MP4
(the download scripts' layout); --annotations is a clone of
github.com/epic-kitchens/epic-kitchens-100-annotations (EPIC_100_<split>.csv)
or one CSV in that format. Writes, in --out:

  <split>/shard_*.tar + index.npz   (src/data/shards.py) each video once,
          keys <video>/000001.jpg... in time order
  frame_times_<split>.npz   each stored frame's time in its video (seconds):
          the manifests are computed from it, so --manifests-only rebuilds
          them (other annotations, --text) without decoding again
  manifest_<split>.csv   one row per narrated segment: the stored frames
          whose times fall in [start_timestamp, stop_timestamp]; txt is the
          narration ("open door"), as Ego4D's narrations are for R3M
  manifest_<split>_videos.csv   one row per whole video, no text: hour-scale
          horizons, on purpose (EPIC videos run up to 62 min)
  manifest_<split>_win<N>s.csv   with --window N: each video cut into equal
          parts of at most N seconds, no text (VIP over longer horizons than
          a segment; --manifests-only adds it without decoding)

For VIP, the segment manifest is the one comparable to the official setup
(R3M's Ego4D clips, at most a few minutes): VIP draws its goal anywhere after
the start, so whole videos put start and goal up to an hour apart. VIP
ignores the text.

Rows share frames (segments overlap and both manifests read the same
shards), so nothing is stored twice. The shard folder <split> is also an
image dataset of the frames (MAE/MVP; configs/pretrain_data/epic_frames.yaml).

Frames: the first decoded frame at or after each 1/fps tick (videos are
29.97-90 fps; --fps 0 keeps every frame). Decoding is the bulk of the cost
(every frame of 1080p H.264 is decoded either way); --fps sets what's kept.
Resize as prepare_ssv2.py: `squash` (default) or `center_crop`.

The run ends with its rate in minutes per GB of video; with --limit, also
the whole split's GB and the time it would take at that rate (videos range
from a minute to an hour, so scale by GB, not by count). Videos are
processed longest first. A video that stops decoding partway
keeps the frames before the error (reported); a missing or unreadable one is
left out with its segments. Not resumable: an interrupted split is converted
again. Each worker holds one video's JPEGs (~20 KB per frame at 224px:
~0.7 GB for EPIC's longest, 62 min at 10 fps).

Needs PyAV (pip install av) and OpenCV.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

EXTS = (".MP4", ".mp4")


def find_video(root: str, video_id: str):
    """<root>/<participant>/videos/<id>.MP4 (the download scripts' layout),
    or <root>/<id>.MP4; None if absent."""
    pid = video_id.split("_")[0]
    for d in (os.path.join(root, pid, "videos"), os.path.join(root, pid), root):
        for ext in EXTS:
            p = os.path.join(d, video_id + ext)
            if os.path.isfile(p):
                return p
    return None


def encode_resampled(video_path: str, fps: float, size: int, resize: str, quality: int):
    """(JPEG bytes, time in seconds) of the first frame at or after each
    1/fps tick (every frame if fps is 0), plus an error message if decoding
    stopped early (None otherwise)."""
    import av
    import cv2
    from src.data.prepare_ssv2 import _resize
    frames, times, error = [], [], None
    step = 1.0 / fps if fps else 0.0
    eps = 1e-4                       # float slack: a frame exactly on a tick counts
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        t0 = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        next_tick = 0.0
        try:
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                t = frame.time - t0
                if t < next_tick - eps:
                    continue
                rgb = _resize(frame.to_ndarray(format="rgb24"), size, resize)
                frames.append(cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
                times.append(t)
                if step:
                    next_tick = (np.floor(t / step + eps) + 1) * step
        except av.error.FFmpegError as e:
            if not frames:
                raise
            error = f"decoding stopped at {times[-1]:.1f}s ({type(e).__name__}: {e})"
    return frames, times, error


def _encode_chunk(jobs):
    """(vid, frames or None, times, error) per (vid, path, fps, size, resize, quality)."""
    import cv2
    cv2.setNumThreads(1)            # one process per CPU already
    out = []
    for vid, path, fps, size, resize, quality in jobs:
        try:
            frames, times, err = encode_resampled(path, fps, size, resize, quality)
            out.append((vid, frames, times, err))
        except Exception as e:      # a corrupt video shouldn't stop the whole split
            out.append((vid, None, [], f"{type(e).__name__}: {e}"))
    return out


def read_annotations(path: str, split: str):
    import pandas as pd
    csv = path if os.path.isfile(path) else os.path.join(path, f"EPIC_100_{split}.csv")
    if not os.path.isfile(csv):
        sys.exit(f"no {csv}: --annotations takes a clone of "
                 "github.com/epic-kitchens/epic-kitchens-100-annotations or one EPIC_100_<split>.csv")
    df = pd.read_csv(csv, keep_default_na=False, dtype={"narration": str})
    missing = {"video_id", "start_timestamp", "stop_timestamp", "narration"} - set(df.columns)
    if missing:
        sys.exit(f"{csv} lacks columns {sorted(missing)}")
    df["start_s"] = pd.to_timedelta(df["start_timestamp"]).dt.total_seconds()
    df["stop_s"] = pd.to_timedelta(df["stop_timestamp"]).dt.total_seconds()
    return df, csv


def save_frame_times(path: str, record: dict):
    """record: video id -> (start in the index, times array)."""
    vids = list(record)
    np.savez(path + ".tmp.npz", video_id=np.array(vids, dtype=str),
             start=np.array([record[v][0] for v in vids], dtype=np.int64),
             len=np.array([len(record[v][1]) for v in vids], dtype=np.int64),
             time=np.concatenate([np.asarray(record[v][1], dtype=np.float64) for v in vids]) if vids
             else np.zeros(0))
    os.replace(path + ".tmp.npz", path)


def load_frame_times(path: str) -> dict:
    z = np.load(path)
    bounds = np.concatenate([[0], np.cumsum(z["len"])])
    return {str(v): (int(s), z["time"][bounds[i]:bounds[i + 1]])
            for i, (v, s) in enumerate(zip(z["video_id"], z["start"]))}


def windows(n: int, times, seconds: float):
    """(first frame index, length) of equal parts of a video's n stored frames,
    each spanning at most `seconds` (by the frames' times)."""
    span = float(times[-1] - times[0]) if n else 0.0
    parts = max(1, int(np.ceil(span / seconds - 1e-9)))
    bounds = np.linspace(0, n, parts + 1).round().astype(int)
    return [(int(a), int(b - a)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def write_manifests(out: str, split: str, ann, record: dict, text: str, window: float = None):
    """manifest_<split>.csv (segments) and manifest_<split>_videos.csv from
    the stored frames' times. Returns (segment rows, segments left out as
    empty, segments of videos not stored)."""
    import pandas as pd
    rows, empty, absent = [], 0, 0
    for vid, seg in ann.groupby("video_id", sort=False):
        if vid not in record:
            absent += len(seg)
            continue
        start, times = record[vid]
        lo = np.searchsorted(times, seg["start_s"].to_numpy(), side="left")
        hi = np.searchsorted(times, seg["stop_s"].to_numpy(), side="right")
        for (_, r), a, b in zip(seg.iterrows(), lo, hi):
            if b <= a:
                empty += 1
                continue
            rows.append({"path": split, "len": int(b - a), "txt": r["narration"] if text == "narration" else "",
                         "start": start + int(a), "video_id": vid,
                         "narration_id": r.get("narration_id", "")})
    seg_csv = os.path.join(out, f"manifest_{split}.csv")
    pd.DataFrame(rows, columns=["path", "len", "txt", "start", "video_id", "narration_id"]).to_csv(seg_csv, index=False)
    vids = [v for v in dict.fromkeys(ann["video_id"]) if v in record]
    pd.DataFrame([{"path": split, "len": len(record[v][1]), "txt": "", "start": record[v][0], "video_id": v}
                  for v in vids], columns=["path", "len", "txt", "start", "video_id"]).to_csv(
        os.path.join(out, f"manifest_{split}_videos.csv"), index=False)
    lengths = np.array([r["len"] for r in rows])
    print(f"Wrote {seg_csv}: {len(rows):,} segments"
          + (f", {lengths.sum():,} frames, median {int(np.median(lengths))} per segment, "
             f"{int((lengths < 3).sum()):,} under 3 frames (R3M/VIP skip those)" if rows else ""))
    print(f"Wrote manifest_{split}_videos.csv: {len(vids):,} videos")
    if window:
        rows_w = [{"path": split, "len": length, "txt": "", "start": record[v][0] + a, "video_id": v}
                  for v in vids for a, length in windows(len(record[v][1]), record[v][1], window)]
        name = f"manifest_{split}_win{window:g}s.csv"
        pd.DataFrame(rows_w, columns=["path", "len", "txt", "start", "video_id"]).to_csv(
            os.path.join(out, name), index=False)
        print(f"Wrote {name}: {len(rows_w):,} windows of at most {window:g} s")
    if empty:
        print(f"  {empty:,} segments hold no stored frame (shorter than 1/fps) and are left out")
    if absent:
        print(f"  {absent:,} segments belong to videos that weren't stored and are left out")
    return len(rows), empty, absent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, help="folder with <P##>/videos/<video>.MP4")
    p.add_argument("--annotations", required=True,
                   help="epic-kitchens-100-annotations folder (EPIC_100_<split>.csv), or one CSV")
    p.add_argument("--out", required=True, help="dataset root (shards, frame times and manifests go here)")
    p.add_argument("--split", required=True, help="train or validation: picks EPIC_100_<split>.csv, names the outputs")
    p.add_argument("--fps", type=float, default=10.0, help="frames kept per second (0: every frame)")
    p.add_argument("--shard-gb", type=float, default=1.0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--resize", choices=("squash", "center_crop"), default="squash")
    p.add_argument("--text", choices=("narration", "none"), default="narration")
    p.add_argument("--window", type=float, default=None,
                   help="also write manifest_<split>_win<N>s.csv: videos cut into parts of at most N seconds")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality")
    p.add_argument("--limit", type=int, default=None, help="only the first N videos (in annotation order)")
    p.add_argument("--manifests-only", action="store_true",
                   help="rebuild the manifests from frame_times_<split>.npz; no decoding")
    # The CPUs this process may use (under Slurm: the allocation, not the node).
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    a = p.parse_args(argv)

    ann, csv = read_annotations(a.annotations, a.split)
    vids = list(dict.fromkeys(ann["video_id"]))
    all_vids = vids
    if a.limit is not None:
        vids = vids[:a.limit]
        ann = ann[ann["video_id"].isin(vids)]
    times_path = os.path.join(a.out, f"frame_times_{a.split}.npz")
    if a.manifests_only:
        if not os.path.isfile(times_path):
            sys.exit(f"no {times_path}: convert the split first")
        write_manifests(a.out, a.split, ann, load_frame_times(times_path), a.text, a.window)
        return

    paths = {v: find_video(a.videos, v) for v in vids}
    missing = [v for v in vids if paths[v] is None]
    if len(missing) == len(vids):
        sys.exit(f"none of the {len(vids)} videos of {csv} are under {a.videos} "
                 f"(expected e.g. {os.path.join(a.videos, vids[0].split('_')[0], 'videos', vids[0] + '.MP4')})")
    if missing:
        print(f"{len(missing)} of {len(vids)} videos missing, their segments left out: {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))
    todo = sorted((v for v in vids if paths[v] is not None), key=lambda v: -os.path.getsize(paths[v]))
    gb = sum(os.path.getsize(paths[v]) for v in todo) / 2**30
    print(f"{len(todo):,} videos ({gb:.1f} GB) -> {a.out} "
          f"({a.fps:g} fps, {a.size}px {a.resize}, {a.workers} workers)")

    from src.data.shards import ShardWriter, ordered_map
    os.makedirs(a.out, exist_ok=True)
    record, failed, truncated, t0, kept = {}, [], [], time.time(), 0
    jobs = [(v, paths[v], a.fps, a.size, a.resize, a.quality) for v in todo]
    with ShardWriter(os.path.join(a.out, a.split), max_bytes=int(a.shard_gb * (1 << 30))) as w, \
            ProcessPoolExecutor(a.workers) as pool:
        # One video per task, few pending: a video's frames are held in memory until written.
        for i, (vid, frames, times, err) in enumerate(
                ordered_map(pool, _encode_chunk, jobs, chunksize=1, in_flight=a.workers + 2), 1):
            if not frames:
                failed.append((vid, err or "no frames"))
            else:
                if err:
                    truncated.append((vid, err))
                record[vid] = (len(w.keys), np.asarray(times))
                for n, data in enumerate(frames, 1):
                    w.add(f"{vid}/{n:06d}.jpg", data)
                kept += len(frames)
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i:,}/{len(jobs):,} videos  {kept:,} frames  {el / 60:.0f} min  failed={len(failed)}",
                      flush=True)

    minutes = (time.time() - t0) / 60
    print(f"Converted {gb:.1f} GB of video in {minutes:.1f} min: {minutes / max(gb, 1e-9):.2f} min/GB "
          f"with {a.workers} workers")
    if a.limit is not None:
        found = [find_video(a.videos, v) for v in all_vids]
        full_gb = sum(os.path.getsize(f) for f in found if f) / 2**30
        print(f"  the whole split is {full_gb:.0f} GB: about {full_gb * minutes / max(gb, 1e-9) / 60:.1f} h at "
              "this rate (not resumable: give the job a time limit well above that)")
    save_frame_times(times_path, record)
    write_manifests(a.out, a.split, ann, record, a.text, a.window)
    for title, items in (("stopped decoding partway (frames before kept)", truncated),
                         ("failed (left out)", failed)):
        if items:
            print(f"{len(items)} videos {title}:")
            for vid, err in items[:10]:
                print(f"  {vid}: {err}")


if __name__ == "__main__":
    main()
