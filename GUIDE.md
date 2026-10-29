# Using and extending the repo

How to work with this repo day to day, and how to add your own datasets,
augmentations, objectives and encoders. What exists today is in
`CAPABILITIES.md`; every option in detail is in `RUNNING_EXPERIMENTS.md`.

## 1. The shape of the repo

The repo trains visual encoders (`pretrain.py`) and measures them on
control (`train_sac.py`, `train_drqv2.py`, `train_bc.py`). Every choice is a
config option, so an experiment is one command line.

| where | what |
|---|---|
| `pretrain.py` | encoder pretraining: objective x data x architecture x augmentation |
| `train_sac.py`, `train_drqv2.py`, `train_bc.py` | downstream: SAC, DrQ-v2, behavioural cloning on a (frozen or fine-tuned) encoder |
| `profile_pretrain.py` | where a pretraining step spends its time (clip objectives) |
| `src/losses.py` | one registry of objectives: pretraining objectives and RL/BC auxiliary losses |
| `src/augmentations.py` | registry of image augmentations |
| `src/embeddings.py` | every encoder (`EmbeddingNet`), its input pipeline, checkpoint loading |
| `src/truncation.py` | stopping an encoder early (`embedding.layer`) and output heads (`embedding.head`) |
| `src/data/` | pretraining data (`clips.py`, `images.py`, `shards.py`, prep scripts) and BC datasets |
| `src/rl/` | replay buffer, shared RL setup, DrQ-v2 agent |
| `src/vision_models/` | model code the repo vendors (MAE/ViT, MoCo, RL3D, ...) |
| `configs/` | Hydra configs, one folder per choice (below) |
| `tests/` | pytest suite, including parity tests against official implementations |

Config folders (`configs/<group>/<name>.yaml`, chosen with `<group>=<name>`):

| group | chooses | used by |
|---|---|---|
| `embedding` | the encoder | all scripts |
| `augmentation` | the augmentation | all scripts |
| `pretrain_data` | the pretraining dataset | `pretrain.py` |
| `experiment` | a method's full recipe (`+experiment=r3m`) | all scripts |
| `env` | the environment | `train_sac.py`, `train_drqv2.py` |
| `algo` | the RL/BC algorithm's settings | `train_sac.py`, `train_drqv2.py`, `train_bc.py` |
| `dataset` | the BC dataset | `train_bc.py` |

## 2. Running things

**Command lines.** Choose files by group, then override any value:
```bash
python pretrain.py +experiment=vip embedding=resnet18 batch_size=16 pretrain_data.root=$WRKDIR/data/ssv2
```
- `+experiment=<name>` loads a recipe; anything after it overrides the recipe.
- `key=value` changes a value that exists in the config; `+key=value` adds
  one that doesn't (e.g. `+embedding.layer=layer2` on embedding files that
  don't list `layer`).
- `--cfg job` prints the final config without running: the quickest way to
  check what an override did.
- `-m` sweeps: `python pretrain.py -m +experiment=r3m embedding=resnet18,resnet50 seed=1,2,3`.

**From pretraining to control.** A pretraining run directory holds
`checkpoint.pt` (for resuming), encoder snapshots `encoder_step<N>.pt`, and
`embedding.yaml` (every field needed to load the encoder). Load a snapshot
in any downstream script:
```bash
python train_sac.py env=dm_control_pixels embedding=custom_checkpoint \
    embedding.name=my_r3m embedding.arch=resnet50 embedding.checkpoint=$RUN/encoder_step1000000.pt
```
(or copy `embedding.yaml` to `configs/embedding/<name>.yaml` and use
`embedding=<name>`). Compare against the released models
(`embedding=r3m_resnet50`, `vip_resnet50`, `mvp_vits_mae_hoi`, ...) under
the same settings.

**Resuming.** Rerun `pretrain.py` with the same `hydra.run.dir`; it picks up
`checkpoint.pt`, and wandb continues the same run.

**On the cluster.** Job scripts are templates to copy: `run_pretrain.sh`
(clip pretraining, copies the data shards to the node's local disk),
`run_imagenet.sh` (image pretraining), `run_sac.sh` / `run_ft_sac.sh`
(SAC), `run_ft_bc.sh` (BC), `debug_*.sh` (short smoke tests). `*.sh` files
are gitignored: `git add -f` a script to keep it in the repo.

**Locally, before a cluster run.**
- Run the tests: `env/bin/python -m pytest tests -q`.
- Check the config composes: add `--cfg job`.
- Do a tiny run on CPU: `device=cpu`, a small `batch_size` and a few
  `train_steps` (or `epochs=1`), on a small dataset (e.g. `prepare_ssv2.py
  --limit 20`).
- Size the real run: read `steps/s` and `data_wait` from a few hundred
  steps; for clip objectives `profile_pretrain.py` breaks a step down.

**Multi-GPU.**
- Not implemented yet: one GPU per run. Planned: DDP via `torchrun`
  (`TODO.md`).

## 3. Adding a dataset

The pretraining formats in full, with a conversion example, are in
`DATASETS.md`; `python -m src.data.check <root>` checks a dataset and
prints a config for it. In brief:

### Pretraining: video clips

What `pretrain.py` reads (`src/data/clips.py`):
- a manifest CSV, one row per video: `path, len, txt[, start]`
  (`txt` may be empty; R3M needs it, VIP doesn't);
- the frames, either as **frame folders** (`<root>/<path>/000001.jpg` ...,
  frames numbered from 1) or as **shards** (`<root>/<path>/shard_*.tar` +
  `index.npz`, written with `src/data/shards.py`'s `ShardWriter`; `start`
  is the video's first frame's row in the index). Shards keep a dataset of
  millions of frames to a few hundred files.

Write a prep script that produces this (copy `src/data/prepare_ssv2.py`;
store frames at the encoder's input size and name each frame
`<video>/<6-digit frame number>.jpg`, which is what lets the frames also
serve as an image dataset, thinned per video), then a config:
```yaml
# configs/pretrain_data/my_videos.yaml
name: my_videos
type: clips
root: ???                       # set on the command line
train_manifest: manifest_train.csv
val_manifest: manifest_validation.csv   # null: validate on the training manifest
image_size: 224                 # the side the frames were stored at
frames_as_stored: true          # use the frames whole (only the encoder's normalisation)
text_strip: 0                   # characters cut from the start of every text
```
Use it: `python pretrain.py +experiment=r3m pretrain_data=my_videos pretrain_data.root=/path`.

### Pretraining: images

What `pretrain.py` reads (`src/data/images.py`), per split:
- an **ImageFolder** tree (`<root>/<split>/<class>/<image>`; labels from
  the sorted class folders or `<root>/classes.txt`), or
- **shards** (`<root>/<split>/shard_*.tar` + `index.npz`, with a `label`
  per image; -1 for unlabelled data). `python -m src.data.prepare_imagenet
  --source <folder, tar or zip>` converts an ImageFolder layout.

```yaml
# configs/pretrain_data/my_images.yaml
name: my_images
type: images
root: ???
train_split: train
val_split: val
image_size: 224                 # crop size fed to the encoder
train_crop: {scale: [0.08, 1.0], ratio: [0.75, 1.3333333333333333]}
interpolation: bilinear         # or bicubic
every: 1                        # thin out: every n-th image (video frames: per video)
val_every: null
eval_resize: 256                # validation: resize, then center-crop image_size
eval_batches: null              # null: the whole validation split
```
Objectives that need labels (`supervised`) reject unlabelled data at startup.
A clip dataset's frame shards are already an image dataset: point
`train_split` at the clip shard folder (as `pretrain_data=ssv2_frames` does).

### Behavioural cloning datasets

`train_bc.py` picks a loader by `dataset.format` (`minari`, `vd4rl`,
`pickle`). To add one:
1. A `torch.utils.data.Dataset` whose items are `(obs, action)`: pixel
   observations as uint8 `(C, H, W)`, states as float vectors; with
   attributes `obs_shape` and `action_dim` (see `src/data/pickle_dataset.py`).
2. A branch for your `format` in `train_bc.py`'s dataset section.
3. `configs/dataset/<name>.yaml` with `format`, `id`, `max_episodes`,
   `eval_env_id` (the env the policy is rolled out in) and
   `train_from_pixels`.

### Environments

`train_sac.py` / `train_drqv2.py` build envs through `ENV_BUILDERS` in
`src/sac_utils.py` (`dmc`, `gym`, `carracing`, `state`). An env any builder
already covers needs only `configs/env/<name>.yaml` (`name`, `builder`,
`id`, `image_size`); a new kind of env needs a builder function there.

## 4. Adding an augmentation

Augmentations are `nn.Module`s in a registry (`src/augmentations.py`).

**The contract:** input and output are float `(N, C, H, W)` in [0, 1], same
shape, dtype and device; random parameters are drawn per sample, from
torch's global RNG (so `seed` makes runs reproducible). It runs at the
encoder's input resolution (after its resize/crop), on the GPU, on training
batches only.

```python
# src/augmentations.py
@register_augmentation("my_aug")
class MyAug(nn.Module):
    def __init__(self, strength: float = 0.1):
        super().__init__()
        self.strength = strength

    def forward(self, x):                  # (N, C, H, W) in [0, 1]
        noise = torch.randn_like(x) * self.strength
        return (x + noise).clamp(0, 1)
```
```yaml
# configs/augmentation/my_aug.yaml
name: my_aug
stage: rl          # RL/BC: who sees the augmented view (rl | aux | both); ignored by pretrain.py
params:
  strength: 0.1
```
- **Combine** existing ones without code: `name: pipeline` with a list of
  `transforms` (see `configs/augmentation/moco_v2.yaml`).
- **Where it applies** is the caller's decision. RL/BC: `stage` decides
  whether the task loss, the aux losses or both see the augmented view.
  Pretraining: clips get one draw per video (`clip_augmentation=shared`) or
  per frame; images get one draw per view. Images are cropped in the
  loader (`pretrain_data.train_crop`), not by an augmentation.
- **Test it** against the contract: `tests/test_augmentations.py` runs
  every registered augmentation through shape/dtype/range, per-sample and
  reproducibility checks, and builds every file in `configs/augmentation/`.

## 5. Adding a loss or pretraining objective

One registry serves both uses (`src/losses.py`): pretraining objectives
(`pretrain.py`'s `objectives`) and auxiliary losses during RL/BC
(`encoder.aux`). Its module docstring is the full contract.

```python
# src/losses.py
@register_aux_loss("my_loss")
class MyLoss(AuxLoss):
    needs_features = True            # what it reads (see below)

    @classmethod
    def check_params(cls, scale=1.0):          # runs at startup, before anything loads
        if scale <= 0:
            raise ValueError(f"my_loss scale must be > 0, got {scale}")

    def __init__(self, encoder, scale: float = 1.0):
        super().__init__(encoder)
        self.scale = scale
        self.head = nn.Linear(encoder.out_size, 64)    # owned parameters train with the encoder

    def forward(self, batch: AuxBatch):
        z = self.head(batch.feats)
        loss = self.scale * z.pow(2).mean()
        return loss, {"z_norm": z.norm(dim=1).mean().detach()}   # (loss, metrics) or just loss
```
```bash
python train_sac.py 'encoder.aux=[{name:my_loss,weight:0.1,scale:2.0}]' encoder.grads=critic
```
This one reads RL/BC features (`needs_features`), so it's an auxiliary loss.
A pretraining objective declares `needs_clips` or `needs_views` instead and
is used as `python pretrain.py 'objectives=[{name:my_objective,weight:1.0}]' ...`.

**What a loss reads** is declared with class attributes, and only that is
computed (each costs an encoder pass). Configs that can't supply it are
rejected at startup.

| attribute | the loss gets | supplied by |
|---|---|---|
| `needs_features` | `batch.obs`, `batch.feats`: the clean view and its features | RL/BC |
| `needs_augmented` | `batch.obs_aug`, `batch.feats_aug` (needs an augmentation with `stage: aux` or `both`) | RL/BC |
| `needs_clips` | `batch.clip_feats` (B, T, D) and `batch.text`; also define `clip_sampler()` returning a `FrameSampler` (which frames of each video) | `pretrain.py` (clip data) |
| `needs_views = n` | `batch.views`: n augmented views per image, encoded by the loss itself (`batch.encoder.encode_preprocessed`) | `pretrain.py` (image data) |
| `needs_labels` | `batch.labels` | `pretrain.py` (labelled images) |

Rules of thumb:
- Return the **unweighted** loss; the config's `weight` scales it.
- Metrics are logged as `train/<name>/<key>` (and `eval/<name>/<key>`) in
  pretraining, `train/aux/<name>/<key>` in RL/BC. Once a metric name is in
  use, keep it: renaming breaks wandb charts.
- Don't store the live encoder as an attribute (it would become a second
  copy of its parameters inside the loss); use `batch.encoder` in
  `forward`. A **frozen copy** of it is fine, e.g. an EMA target:
  `self.target = copy.deepcopy(encoder).requires_grad_(False)`, updated in
  `forward` while `self.training` (MoCo does exactly this).
- An objective that needs the backbone's internals (MAE's masking) can call
  `batch.encoder.normalize(view)` and then `batch.encoder.embedding`
  directly.
- For a method recipe, add `configs/experiment/<method>.yaml` (below).
- When porting a published method, add a parity test that runs the
  official code on the same inputs and compares losses and weights after an
  optimizer step (`tests/test_r3m.py`, `tests/test_vip.py`), skipped when
  the official package isn't installed.

## 6. Adding an encoder

Encoders are branches of `_get_embedding()` in `src/embeddings.py`, each
returning `(model, transforms, forward_fn)`:
- `model`: an `nn.Module`;
- `transforms`: the input pipeline on uint8 frames, ending in
  `T.ConvertImageDtype(torch.float)` followed only by value rescaling
  (`T.Normalize`); augmentation runs between the two;
- `forward_fn(model, x)`: how to get features (default `model(x)`).

Then `configs/embedding/<name>.yaml` (`name`, `pretrained`, `train`).
To make it **trainable in `pretrain.py`** and loadable from your own
checkpoints, it needs a branch that builds the model without downloading
weights when `pretrained=false`, and an entry in `CHECKPOINT_ARCHS`.
`embedding.layer` and `embedding.head` then work on it as on every encoder
(ViTs: their blocks must be in a `blocks` list).

## 7. Adding a method preset

A preset is a recipe: a config that sets everything a method needs.
```yaml
# configs/experiment/my_method.yaml
# @package _global_
# Where the numbers come from (paper section, official config file).
defaults:
  - override /embedding: resnet50
  - override /pretrain_data: ssv2
  - override /augmentation: r3m

embedding:
  pretrained: false
objectives:
  - {name: my_loss, weight: 1.0}
batch_size: 32
train_steps: 1_000_000
lr: 1.0e-4
```
`# @package _global_` is required (the file sets top-level keys). Record
the source of every value in its comments, and add a compose test (see
`test_presets` in `tests/test_mae.py`).

## 8. Before calling something done

- Tests pass (`env/bin/python -m pytest tests -q`), with new tests for the new piece.
- A tiny end-to-end run works (CPU is fine).
- `RUNNING_EXPERIMENTS.md` documents the new options, and `CAPABILITIES.md`
  lists the new piece with its status (done / ready / experimental).
- Logged metric names that already exist are unchanged.
