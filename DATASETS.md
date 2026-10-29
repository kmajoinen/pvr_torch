# Datasets for pretraining

What `pretrain.py` reads, how to get a dataset into that form, and how to
check it. Behavioural-cloning datasets are a different thing (observations
plus actions); see `GUIDE.md`, section 3.

**In short:** a pretraining dataset is either **video clips** (a manifest CSV
plus frames) or **images** (class folders, or shards). Convert your data
into one of the two, write a small config, and check it:
```bash
python -m src.data.check /path/to/data        # reports problems, prints a config
python pretrain.py +experiment=r3m pretrain_data=my_data pretrain_data.root=/path/to/data
```

## 1. Which kind of data each method needs

| method | data | also needs |
|---|---|---|
| R3M (`+experiment=r3m`) | video clips | a text description per video (or `objectives.0.lang_weight=0`) |
| VIP (`+experiment=vip`) | video clips | — |
| MAE / MVP (`+experiment=mae`, `mvp`) | images (video frames work) | — |
| MoCo v2 (`+experiment=moco_v2`) | images (video frames work) | — |
| supervised (`+experiment=imagenet_supervised`) | images | a class label per image |

R3M and VIP skip videos shorter than 3 frames.

## 2. Video clips

### The manifest

One CSV per split, in the dataset's root (e.g. `manifest_train.csv`,
`manifest_validation.csv`), one row per video:

| column | meaning |
|---|---|
| `path` | where the frames are, relative to the root (or absolute): the video's frame folder, or (with `start`) the shard folder |
| `len` | number of frames, a whole number >= 1 |
| `txt` | the video's description; may be empty (R3M's language term masks empty ones out) |
| `start` | shards only: the index row of the video's first frame |

### The frames, in one of two layouts

**Frame folders:** `<root>/<path>/000001.jpg`, `000002.jpg`, ... up to `len`,
numbered from 1 with six digits. Simple, but one file per frame: millions
of files for a large dataset, more than a typical scratch quota allows.

**Shards (recommended):** a folder of `shard_000000.tar`, ... (~1 GB each,
plain tar files) plus `index.npz`, written by `src/data/shards.py`'s
`ShardWriter`. The frames of all videos go into one shard folder, one after
another; a video's frame *i* is index row `start + i - 1`. Name each frame
`<video id>/<6-digit frame number>.jpg`: then the same shards also work as
an image dataset of video frames, thinned per video (`pretrain_data.every`).

Rows may also point into the middle of a longer video (`start` at any
row, `len` the segment's length), e.g. one row per narrated segment, as
Ego4D's and EPIC-Kitchens' narrations are given.

### What the frames must be

- **One size for the whole dataset**, JPEG (PNG also decodes).
- **Square, at the encoder's input size** (224 for the ResNets and ViTs
  here), to use them as stored (`frames_as_stored: true`, as R3M does).
  Otherwise set `frames_as_stored: false`: the encoder's own resize and
  center crop runs on every frame.
- `src/data/prepare_ssv2.py` stores 224x224 frames, squashed (`--resize
  squash`, as R3M describes its Ego4D frames) or center-cropped
  (`--resize center_crop`).

### How they're read

`src/data/clips.py` streams videos at random, with replacement; each sample
is one video and the frames the method picks from it (R3M: 5, VIP: 4). The
run length is a number of steps (`train_steps`), not epochs. Validation is
one batch from the validation manifest.

## 3. Images

### Two layouts, per split folder (`<root>/train`, `<root>/val`, ...)

**Class folders** (torchvision's ImageFolder): `<root>/<split>/<class>/<image>`.
Labels come from the sorted folder names of `<root>/train`, or from
`<root>/classes.txt` (one class per line; label = line number). Fine on a
local disk; slow on a shared filesystem with millions of files.

**Shards:** `<root>/<split>/shard_*.tar` + `index.npz` (fields `key`,
`shard`, `offset`, `size`, `label`), written with `ShardWriter`; label -1
means unlabelled. `python -m src.data.prepare_imagenet --source <folder,
tar or zip>` converts a class-folder layout into shards.

Images can be any size: each is cropped in the loader
(`RandomResizedCrop` to `image_size` while training; resize + center crop
for validation).

### Video frames as images

A clip dataset's frame shards are already an image dataset: point an
image config at the shard folder (`train_split: train`), as
`configs/pretrain_data/ssv2_frames.yaml` does. Frames are unlabelled, so
supervised training rejects them. With keys `<video>/<number>.jpg`,
`every: n` keeps frames 1, 1+n, 1+2n, ... of each video (so every video
keeps at least one); the run prints `every=n: per video` (or `stored order`
for other keys).

### How they're read

`src/data/images.py`: true epochs (every image once per epoch, a fresh
order each epoch), so runs are sized in `epochs` (or a total image count,
`train_images`). Validation is a pass over the validation split, or its
first `eval_batches` batches.

## 4. The config

One file per dataset in `configs/pretrain_data/`, chosen with
`pretrain_data=<name>`; `root` is usually set on the command line.

**Video clips** (`configs/pretrain_data/ssv2.yaml` is the reference):
```yaml
name: my_videos               # used in run names
type: clips
root: ???                     # the dataset folder
train_manifest: manifest_train.csv
val_manifest: manifest_validation.csv    # null: validate on the training manifest
image_size: 224               # the side the frames are stored at
frames_as_stored: true        # use frames whole (needs square frames at the encoder's input size)
text_strip: 0                 # characters cut from the start of every text (Ego4D: 2, its "C ")
```

**Images** (`configs/pretrain_data/imagenet.yaml`, `ssv2_frames.yaml`):
```yaml
name: my_images
type: images
root: ???
train_split: train
val_split: val
image_size: 224               # crop size fed to the encoder
train_crop: {scale: [0.08, 1.0], ratio: [0.75, 1.3333333333333333]}
interpolation: bilinear       # crops and validation resize: bilinear | bicubic
every: 1                      # use every n-th image (video frames: per video)
val_every: null               # the same for validation (null: as `every`)
eval_resize: 256              # validation: resize the shorter side, then center-crop
eval_batches: null            # validation batches (null: the whole split)
```
A method preset may override some of these (e.g. `+experiment=mae` sets
`interpolation: bicubic` and its crop scale).

## 5. Converting a dataset

**Existing converters:**
- `python -m src.data.prepare_ssv2`: video files + a labels JSON -> clips
  (shards or frame folders). A starting point for other video datasets.
- `python -m src.data.prepare_imagenet`: class folders (as a folder, a tar
  or Kaggle's zip) -> image shards.

**Your own:** anything that writes the layouts above works. With shards,
the conversion is short:
```python
import os

import pandas as pd
from torchvision.io import decode_image, encode_jpeg
from torchvision.transforms.v2.functional import resize

from src.data.shards import ShardWriter


def jpeg_224(path):
    """An image file -> 224x224 JPEG bytes (squashed, as prepare_ssv2.py's default)."""
    img = decode_image(path, mode="RGB")
    return encode_jpeg(resize(img, [224, 224], antialias=True), quality=95).numpy().tobytes()


def write_clips(videos, out, split):
    """videos: (video id, list of frame image paths in order, description) per video."""
    rows = []
    with ShardWriter(os.path.join(out, split)) as w:
        for vid, frames, text in videos:
            start = len(w.keys)
            for i, f in enumerate(frames, 1):
                w.add(f"{vid}/{i:06d}.jpg", jpeg_224(f))
            rows.append({"path": split, "len": len(frames), "txt": text, "start": start})
    pd.DataFrame(rows, columns=["path", "len", "txt", "start"]).to_csv(
        os.path.join(out, f"manifest_{split}.csv"), index=False)
```
For an image dataset, the same with `w.add(key, jpeg_bytes, label=class_index)`
(-1 if unlabelled) and a `classes.txt` in the root.

**On a cluster:**
- Keep datasets as shards: a few hundred files instead of millions, within
  file quotas.
- Convert once, on a CPU node, with the source extracted to the node's
  local disk (`run_prepare_ssv2.sh` does this).
- For training, copy the shards to the GPU node's local disk at job start
  (`run_pretrain.sh`): random reads from a shared filesystem are slow.

## 6. Checking a dataset

```bash
python -m src.data.check /path/to/data                  # sample-based, quick
python -m src.data.check /path/to/data --full           # decode every frame/image
python -m src.data.check /path/to/data --image-size 224 --sample 500
```
It finds every manifest and image split under the root and checks:
- manifests: the columns, `len`/`start` values, videos shorter than 3 frames, how many have text;
- frames: folders or shard ranges present, shards not truncated, each sharded row covering one video's consecutive frames (from the keys, over the whole manifest: catches an off-by-one `start`), frames decode, one square size matching `--image-size` across all manifests;
- images: labels and `classes.txt`, decoding, images smaller than the crop size;
- what kind of thinning `every` would do.

It ends with a suggested `configs/pretrain_data/*.yaml` for each layout it
found, and exits with status 1 if there are errors. For a folder in no
known layout, it says what it found instead (video files, loose images,
tar files, TFRecords) and which section of this file applies.

## 7. Datasets worth converting next

Candidates on Triton for pretraining, most relevant first (none prepared yet):

| dataset | kind | why |
|---|---|---|
| EPIC-Kitchens | egocentric video + narrations | part of MVP's training data; narrations work like Ego4D's for R3M |
| Open X-Embodiment, BridgeData | robot video + language instructions | in-domain data for R3M/VIP; frames for MAE |
| CO3Dv2, MVImgNet, OmniObject3D | object-centric multi-view video | object views from many angles |
| Open Images, LAION-400M (if stored as images), Hypersim | images | large image sets for MAE/MoCo |

Planned: a `prepare_data.py` that converts common source layouts (video
files, frame folders, image folders, WebDataset tars, RLDS) automatically,
and a check when `pretrain.py` loads a dataset, with guidance for known
datasets.
