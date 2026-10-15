# Running experiments

Practical reference for launching training runs: which script, which envs,
which embeddings (and where their weights come from), and how encoder
training and augmentation work. This complements `README.md` rather than replacing it.

## Which script

- **`train_sac.py`** — primary script. Explicit CleanRL-style training loop.
  Supports frozen-fast PVR, slow-frozen PVR, PVR finetune (with gradient
  routing), aux losses, augmentation, and state-only baselines. Config:
  `configs/config_sac.yaml`.
- **`train_drqv2.py`** — DrQ-v2's agent with the same encoder, augmentation,
  buffer and wandb setup as `train_sac.py` (every `encoder.*`/`augmentation`
  option works the same). Config: `configs/config_drqv2.yaml` (=
  `config_sac.yaml` with `algo: drqv2`). See [DrQ-v2](#drq-v2-train_drqv2py).
- **`train_sac_sb3.py`** — kept for posterity, Stable-Baselines3-backed.
  Supports state / CNN-from-scratch (SB3's NatureCNN) / frozen PVR (fast and
  slow) / PVR finetune (actor-only gradient routing, no `encoder.grads`
  choice). No augmentation or aux losses. Config:
  `configs/config_sac_sb3.yaml`. Use this specifically if you need SB3's
  `CnnPolicy` baseline or don't care about gradient-routing control.

Everything below is about `train_sac.py` unless stated otherwise; it all
applies to `train_drqv2.py` as well.

## Environments

`train_sac.py` dispatches on `cfg.env.builder`, one of four values:

| builder     | function       | notes |
|-------------|----------------|-------|
| `dmc`       | `_make_dmc`    | dm_control via shimmy; renders pixels at `cfg.env.image_size` via dm_control's own `render_kwargs` (height/width/camera_id) |
| `carracing` | `_make_carracing` | Gymnasium's `CarRacing-v3`, native pixel obs |
| `gym`       | `_make_gym`    | generic pixel builder for envs with no native pixel obs and no dm_control-style render_kwargs (Adroit, FrankaKitchen) — `AddRenderObservation` at the env's default camera/resolution; `image_size` isn't used to control render resolution here, only `EmbeddingNet`'s own resize downstream |
| `state`     | `_make_state`  | raw/flattened proprioceptive state (any Gym/dm_control env; `Dict` observations get auto-flattened) |

Only `configs/env/*.yaml` files that have a `builder:` field work with
`train_sac.py`. Currently that's:

- `dm_control_pixels.yaml` / `dm_control_state.yaml` (`dm_control/cheetah-run-v0`, pixels/state) — also the generic template for any other dm_control domain/task, see below
- `dmc_walker.yaml` (`dm_control/walker-walk-v0`, pixels)
- `carracing_pixels.yaml` (`builder: carracing`)
- `pendulum.yaml` (`builder: state`, `Pendulum-v1`)
- `adroit_door_pixels.yaml` / `adroit_door_state.yaml`, and the same `_pixels`/`_state` pair for `adroit_hammer`, `adroit_pen`, `adroit_relocate` (`builder: gym`/`state`)
- `franka_kitchen_pixels.yaml` / `franka_kitchen_state.yaml` (`builder: gym`/`state`)

The old `adroit_*.yaml`/`franka_kitchen.yaml` files (only `id:`, no
`builder:`) are still there, untouched, for `train_sac_sb3.py`'s universal
env-id dispatch (`src/gym_wrappers.py::make_gym_env`) — the new
`_pixels`/`_state` variants above are `train_sac.py`-specific siblings, not
replacements.

`atari_breakout.yaml` and `minigrid_doorkey.yaml` still only work with
`train_sac_sb3.py` — `ENV_BUILDERS` has no branch replicating
`make_gym_env`'s `AtariPreprocessing`/`FrameStackObservation`/
`ImgObsWrapper` handling, so those need actual code added to
`src/sac_utils.py`, not just a config file.

**Other dm_control domain/task combos** ("since dmc is installed, shouldn't
other combos work?" — yes): `_make_dmc` is fully generic, no cheetah-specific
logic at all — any `dm_control/<domain>-<task>-v0` id shimmy registers works
by just pointing `id:` at it. `dm_control/<domain>-<task>-v0` is a genuine
two-axis template (any domain × any task), so there's no need for a
dedicated file per combo — same pattern as `embedding.name=` overriding a
base embedding config. Just override on the CLI:
```bash
python train_sac.py env=dm_control_pixels env.id=dm_control/finger-spin-v0 env.name=dmc_finger_spin
python train_sac.py env=dm_control_pixels env.id=dm_control/reacher-easy-v0 env.name=dmc_reacher_easy
```
(Adroit/FrankaKitchen got dedicated files instead, unlike dm_control, because each hand-manipulation task is a genuinely distinct env id with no shared domain/task template to override into — `AdroitHandDoor-v1`, `AdroitHandHammer-v1`, etc. aren't parameterizations of one base config the way dm_control ids are.)

Example:
```bash
python train_sac.py env=dm_control_pixels embedding=resnet50
python train_sac.py env=pendulum embedding=none          # state, no encoder
python train_sac.py env=dm_control_state embedding=none  # state, dm_control
python train_sac.py env=adroit_door_pixels embedding=resnet50
python train_sac.py env=franka_kitchen_state embedding=none
```

## Embeddings

Set via `embedding=<name>` (selects `configs/embedding/<name>.yaml`) or
override the name directly: `embedding.name=<name>`. Currently defined
config files:

```
resnet18  resnet34  resnet50  random  none
clip_vit
openclip_vit_b32  openclip_vit_l14  openclip_rn50
r3m_resnet18  r3m_resnet34  r3m_resnet50
vip_resnet50
liv_resnet50
vc1_vitb  vc1_vitl
rl3d_resnet18
mvp_vits_mae_hoi  mvp_vits_mae_in  mvp_vits_sup_in
mvp_vitb_mae_egosoup  mvp_vitl256_mae_egosoup
mae_base
moco_aug
dinov2_vits14  dinov2_vitb14  dinov2_vitl14  dinov2_vitg14
custom_checkpoint
```

DINOv2 loads through `torch.hub`, which downloads the code and weights on
first use and caches them under `$TORCH_HOME` (default `~/.cache/torch`).
On the cluster, set `TORCH_HOME` to scratch, and load each model once from a
login node if compute nodes have no internet.

### Your own weights and input preprocessing

Three optional fields on any embedding config (see `EmbeddingNet` in
`src/embeddings.py`):

- **`checkpoint`** — load weights from a file into the architecture named by
  **`arch`** (default: the embedding name): an encoder saved by this repo
  (`best_encoder.pt`, or `final_checkpoint.pt`, whose `encoder` entry is
  used), including a trained head, or a bare backbone state_dict (a
  pretraining run, or a stock torchvision one). Keys must match exactly; a
  partial or wrong-architecture file is an error, not a warning. Supported
  for `random`, `resnet18/34/50`, `dinov2_*` and `openclip_*`; the other
  encoders always load their own default weights first, so they're rejected
  at startup. Relative paths resolve against the directory
  you launch from. `embedding=custom_checkpoint` is a ready-made template
  that requires `checkpoint`:
  ```bash
  python train_sac.py embedding=custom_checkpoint embedding.name=my_run \
      embedding.arch=resnet18 embedding.checkpoint=/path/to/weights.pt
  ```
- **`preprocess`** — replace the encoder's built-in input pipeline:
  `resize` (shorter side; null = keep), `crop` (center, square; null = none),
  `interpolation` (bilinear | bicubic | nearest), `mean`/`std` (null = no
  normalisation). E.g. PIE-G's native-resolution input, `x/255 - 0.5`:
  ```bash
  python train_sac.py embedding=resnet18 env.image_size=84 \
      '+embedding.preprocess={resize:null,crop:null,mean:[0.5,0.5,0.5],std:[1.0,1.0,1.0]}'
  ```
  The `+` is needed because most embedding configs don't list `preprocess`
  (`custom_checkpoint` does). Without a resize, the encoder sees frames at
  `env.image_size`, which the scripts pass to the encoder automatically.

### Intermediate layers (truncation) and output heads

Two more optional embedding fields (see `src/truncation.py`):

- **`layer`** — stop the backbone early; later layers never run. `layer1`..
  `layer4` for ResNets (torchvision, MoCo, R3M/VIP, CLIP/OpenCLIP/LIV RN50,
  RL3D), a block count for ViTs (`6` = output of the first 6 blocks: MAE,
  MVP, VC-1, CLIP ViT, DINOv2), or an exact module path. Outputs are raw (no
  final norm layer).
- **`head`** — turn that output into a vector: `avgpool` (default with
  `layer`; spatial mean, or mean over patch tokens), `cls` (ViT CLS token),
  `flatten` (the whole map, no parameters; the RL heads' first layers do the
  projecting), or `linear` (PIE-G's encoder projection: flatten → trainable
  Linear(`head_dim`, default 1024) → LayerNorm). A `linear` head needs the
  encoder to train; with `encoder.trainable=head` it trains on a frozen
  backbone. PIE-G's actor and critic additionally have their own
  Linear-LayerNorm-Tanh trunks, which this SAC's networks don't.
- **Size check**: `flatten` and `linear` scale with the map. ResNet18 layer2
  is 128×11×11 ≈ 15.5k values at 84px but ≈ 100k at 224px, which makes a
  `linear` head ~16M vs ~100M parameters (printed at startup). When a frozen
  encoder's features would take more memory than the frames themselves,
  `buffer_obs=auto` stores pixels instead and says so.

```bash
# Frozen ResNet18 layer2, average-pooled (fast path, no trainable parts)
python train_sac.py embedding=resnet18 +embedding.layer=layer2

# DINOv2 ViT-S after 6 blocks, CLS token
python train_sac.py embedding=dinov2_vits14 +embedding.layer=6 +embedding.head=cls

# PIE-G-style: layer2 at native 84px, trainable linear head on a frozen backbone
python train_sac.py embedding=resnet18 env.image_size=84 \
    '+embedding.preprocess={resize:null,crop:null,mean:[0.5,0.5,0.5],std:[1.0,1.0,1.0]}' \
    +embedding.layer=layer2 +embedding.head=linear \
    encoder.grads=critic encoder.trainable=head encoder.lr=1e-4 augmentation=random_shift
```

`encoder.lr` (default `1e-5`) also sets the head's learning rate; PIE-G
trains its head at the actor/critic rate. PVR's compressed intermediate
layers remain available as their own encoder names (`resnet50_l3`,
`resnet50_l4`, `moco_*_l3`, ...).

### PIE-G options: BatchNorm adaptation, frame stacking, strong views

- **`encoder.bn: frozen | adapt`** — with `adapt`, BatchNorm layers (only
  those) normalise each replay/training batch with its own statistics and
  update their running ones; acting and evaluation use the running ones, as
  in PIE-G. It changes features over time, so the buffer stores pixels, and
  the encoder is saved even when no weights train. Encoders without
  BatchNorm (ViTs) are rejected. `train/bn_stat_drift` logs how far the
  running statistics have moved from their loaded values.
- **`embedding.frame_stack: k`** + **`embedding.frame_fusion`** — the env
  stacks k frames (the buffer holds all of them). `channels` feeds the stack
  to the backbone as one wide input (`random` only, its default); `concat`
  (the default otherwise) encodes each frame and joins the outputs; `diff`
  is PIE-G's: frames 2..k plus each one's difference from the previous
  frame. Fusion happens before the head, so a `linear` head sees all
  frames. Augmentations treat a stack alike (same shift, same overlay
  image); colour transforms need 3 channels, so they don't work with
  stacks. SAC only: BC datasets hold single frames.
- **`random_overlay`** — blends each frame with a natural image (SODA/SVEA).
  Needs `params.image_dir`, a folder of images such as Places365's
  validation set. This approximates DMC-GB's loader (used by SVEA and
  PIE-G), which streams the full Places365 training split with a fresh
  random resized crop and flip per image: here a fixed pool of `num_images`
  (default 5000, the same every run) is loaded onto the GPU on first use,
  at ~1.5× the frame side, and every draw gets its own random resized crop
  (`crop_scale`) and flip. GPU memory is `num_images × 3 × side²` bytes:
  ~240 MB at 126px (84px frames), ~1.7 GB at 336px (224px frames), so lower
  `num_images` for large frames.
- **`augmentation.strong`** — an optional second, stronger view applied on
  top of the task view, which the critic (BC: the policy) also fits:
  `(1 - weight) * L(task view) + weight * L(strong view)`, as in SVEA and
  PIE-G. Targets and the actor stay on the task view. Logged as
  `train/qf_loss_strong` (BC: `train/bc_loss_strong`). The `pieg` preset is
  random shift plus an overlay strong view at weight 0.5.

PIE-G's full recipe is the preset `configs/experiment/pieg.yaml` (all of
the above plus layer2 at native 84px with a trainable linear head), on
DrQ-v2 as in the paper, or on SAC:
```bash
python train_drqv2.py +experiment=pieg env=dmc_walker \
    augmentation.strong.params.image_dir=/path/to/places365/val_256
python train_sac.py +experiment=pieg augmentation.strong.params.image_dir=...
```
Presets in `configs/experiment/` set a method's whole recipe; anything can
still be overridden after them on the command line.

`src/embeddings.py::_get_embedding()` supports quite a few more names than
have a dedicated yaml file (e.g. `resnet50_l3`, `resnet50_l4`,
`resnet50_places`, `moco_croponly*`, `demy`, `maskrcnn_l3`, and MoCo
"uber"-ensemble names) — for those, start from an existing embedding config
and override the name on the CLI, e.g.:
```bash
python train_sac.py embedding=resnet50 embedding.name=resnet50_l3
```

### Where the weights come from

**Auto-download, nothing to provide:**
- `resnet18` / `resnet34` / `resnet50` — torchvision, `ImageNet1K` weights.
- `clip_vit` / `clip_rn50` — via `clip.load(...)`, OpenAI's own cache.
- `openclip_*` — via `open_clip.create_model_and_transforms(...)`, HuggingFace hub cache.
- `r3m_*` — via `load_r3m(...)`, downloads from Google Drive (`gdown`) to R3M's own cache dir on first use. **No outbound internet on compute nodes** on most clusters — run once on the login node first to warm the cache, or the job will fail trying to download mid-run.
- `vip_resnet50` — via `load_vip(...)`, same lab/codebase as R3M, same download-cache behavior and caveat.
- `liv_resnet50` — via `load_liv(...)`, own cache dir/download on first use. Needs a two-step install (`pip install` the repo, then a nested `cd liv/models/clip && pip install -e .`) — not in `requirements.txt`, see the comment there.
- `mvp_*` (5 variants) — via `mvp.load(...)`, own cache dir/download on first use (Berkeley Box-hosted). Genuinely pip-installable with no version conflicts, unlike VC-1.
- `random` — freshly initialized, no weights at all.
- `none` — no embedding, raw state passed straight through.

**Need a local checkpoint file** (not provided in this repo, download/train
separately and place in `models/`, or point at a different location via
`model_dir=`): `mae_base` / `mae_large` / `mae_huge`, `vc1_vitb` / `vc1_vitl`,
`rl3d_resnet18`, every `moco_*` name, `resnet50_l3` / `resnet50_l4` /
`resnet50_places*`, `demy`, `maskrcnn_l3`. Expected filenames (must match
exactly, in `models/` by default):
```
mae_pretrain_vit_base.pth / _large.pth / _huge.pth
vc1_vitb.pth / vc1_vitl.pth   (from https://dl.fbaipublicfiles.com/eai-vc/ --
  loaded directly via the mae_* architecture classes, not the vc_models
  package, see src/embeddings.py's VC-1 branch for why)
rl3d_resnet18.tar   (rename from videoae_co3d.tar -- download at
  https://github.com/YanjieZe/rl3d/tree/main/checkpoints; architecture is
  vendored in src/vision_models/rl3d.py, not pip-installed -- see that
  file's docstring for why)
moco_aug.pth.tar, moco_aug_l3.pth, moco_aug_l4.pth, moco_aug_places.pth.tar, ...
  (one file per moco_* embedding name -- see src/embeddings.py's MOCO section
  for the exact filename each name expects)
resnet50_l3.tar, resnet50_l4.pth.tar, resnet50_places.pth.tar, ...
demy.pth
maskrcnn_l3.pth  (also requires detectron2 installed separately)
```

### `model_dir` — pointing at checkpoints that live elsewhere

Since you haven't provided these checkpoint files separately yet, and on
the cluster they'll likely live on scratch storage rather than next to the
code, `config_sac.yaml` has:
```yaml
model_dir: null   # if set, look here first for <embedding_name>.<any extension>
```
Set it via `model_dir=/path/to/checkpoints` on the CLI. When set, the
loader looks for a file named `<embedding_name>.*` in that directory
instead of the default `models/` location — e.g. with
`embedding=resnet50 embedding.name=resnet50_l3 model_dir=/scratch/.../ckpts`,
it looks for `/scratch/.../ckpts/resnet50_l3.*`. **Crashes loudly** if
`model_dir` is set but no matching file is found there (no silent fallback).

Only checkpoint-file-based embeddings honor `model_dir` — `resnet18/34/50`
(plain, non-`_l3`/`_l4`/`_places` versions), `clip_*`, `openclip_*`, and
`r3m_*` never consult it; they always use their own package's loader/cache.

## Encoder training and augmentation

Three independent settings, shared by `train_sac.py` and `train_bc.py` and
validated before anything is loaded (`src/encoder_config.py`):

```yaml
encoder:
  grads: none        # which task losses backprop into the encoder
                     #   SAC: none | critic | actor | both    BC: none | policy
  lr: 1.0e-5         # separate optimizer/LR from the policy/critic
  betas: [0.9, 0.999]
  grad_clip: 10.0    # SAC default; BC defaults to 1.0
  aux: []            # auxiliary losses, e.g. [{name: l2sp, weight: 1.0}]
augmentation: none   # Hydra group: configs/augmentation/*.yaml
buffer_obs: auto     # SAC only: auto | features | pixels
```

- **The encoder trains** (gets an optimizer) iff `encoder.grads != none` or
  `encoder.aux` is non-empty. Otherwise it's frozen.
- **`encoder.grads`** — which RL loss trains the encoder: `critic` (DrQ-v2's
  default), `actor`, `both` (sequential Adam steps, not one combined
  gradient — see the module docstring in `train_sac.py`), or `none`. For BC,
  `policy` lets the BC loss backprop into the encoder.
- **`encoder.aux`** — a list of auxiliary losses from `src/losses.py`, each
  `{name, weight, ...params}`:
  - `l2sp` anchors the backbone to its pretrained values (`||θ - θ₀||²`).
    It's a regulariser, not a training signal: its gradient is zero at the
    pretrained weights, so it needs another loss to move the encoder. With
    `grads: none` that has to be an aux loss that reads observations (e.g.
    `aug_consistency`); `l2sp` alone is rejected at startup, since the run
    would equal a frozen one.
  - `aug_consistency` (`target: clean | augmented`, default `clean`) is the
    MSE between the embeddings of the clean and augmented views; the
    `target` branch is detached. Needs an augmentation with `stage: aux` or
    `both`. On its own it has a degenerate optimum (every input mapped to
    the same embedding), so pair it with `l2sp` or `grads != none`.

  To add your own, subclass `AuxLoss` in `src/losses.py` and decorate it
  with `@register_aux_loss("name")`; the file's docstring has the contract.
- **`augmentation`** — picks a file from `configs/augmentation/` (`none`,
  `random_shift`, `color_jitter`, `gaussian_blur`, `horizontal_flip`,
  `random_erasing`, `random_overlay`, `shift_jitter` as a pipeline example,
  and the `pieg` preset with a strong view, see below). Each has a
  `name`, a `stage` and `params`. `stage` decides who sees the augmented
  view (`src/views.py`):

  | stage | task loss (SAC actor/critic, BC policy loss) | aux losses |
  |---|---|---|
  | `rl` | augmented | clean |
  | `aux` | clean | clean + augmented |
  | `both` | augmented | clean + augmented |

  `aux` and `both` need an aux loss that reads the augmented view (e.g.
  `aug_consistency`); contradictory combinations raise at startup. The
  augmentation is drawn once per update and shared by SAC's critic and
  actor phases, and an update costs at most one encoder pass more than the
  task loss alone. It applies to sampled training batches only, never to
  rollouts or evaluation. Augmentations run at the encoder's input
  resolution (after its resize/crop) and draw parameters per sample; see
  `src/augmentations.py` for the contract and how to add one.
- **Collapse monitor** — whenever the encoder trains, the per-dimension std
  of its raw output (before the heads' LayerNorm) is logged as
  `train/embedding_std` and `train/embedding_std_ratio` (also printed as
  `emb_std_ratio`), with a one-time warning if it drops below 10%. The
  ratio's baseline is the first reading after 1000 updates (BC: batches),
  not the first update, because a freshly initialised `linear` head changes
  scale quickly at first; until then the ratio is NaN.
- **`buffer_obs`** (SAC) — what the replay buffer stores. `auto` stores
  pixels if the encoder trains or augmentation is on (both need every
  sampled batch re-encoded), and features otherwise: the fast path, where
  the encoder runs once per env step and the buffer holds feature vectors.
  `pixels` forces pixels; with a frozen encoder and no augmentation that's
  "slow-frozen", the same features re-encoded every update, useful as a
  control. `features` raises if pixels are needed. The choice and the
  reason are printed at startup, next to the buffer's memory estimate.
- **Memory**: pixels in the replay buffer cost `buffer_size × H × W × 3`
  bytes (next_obs is derived from the following slot, not stored). At 224px
  that's ~15GB per 100k transitions — override `env.image_size=112` (or
  `84`) and/or shrink `algo.buffer_size` for pixel-buffer runs.
- **BatchNorm**: the encoder stays in `eval()` even while it trains, so
  BatchNorm running statistics never update (trainable BN affine params
  only), unless `encoder.bn=adapt` (see PIE-G options above).
- **Migrating from `finetune.*`** (`config_version` 1): `finetune.enabled`
  is gone and any `finetune.*` key raises with a translation table.
  `finetune.encoder_grads` → `encoder.grads`, `encoder_lr`/`encoder_betas`/
  `grad_clip` → `encoder.lr`/`betas`/`grad_clip`, `aux_loss: l2sp` +
  `aux_weight: w` → `encoder.aux: [{name: l2sp, weight: w}]`, and the old
  slow-frozen mode → `buffer_obs=pixels`.

### Example commands
```bash
# DrQ-v2-style critic-only finetuning
python train_sac.py encoder.grads=critic env.image_size=112 embedding=resnet18

# DrQ-v2-style finetuning with its random shift
python train_sac.py encoder.grads=critic augmentation=random_shift \
    env.image_size=112 embedding=resnet18

# Frozen encoder + augmentation (pixels in the buffer, picked automatically)
python train_sac.py augmentation=color_jitter env.image_size=112

# Sweep augmentation and encoder independently
python train_sac.py -m augmentation=none,random_shift embedding=r3m_resnet50,vc1_vitb \
    env.image_size=112

# Slow-frozen control (protocol-compatible with the fast path)
python train_sac.py buffer_obs=pixels env.image_size=112


# Augmentation only through a consistency loss: actor/critic see clean
# observations, the encoder learns invariance (anchored by l2sp)
python train_sac.py augmentation=random_shift augmentation.stage=aux \
    'encoder.aux=[{name:aug_consistency,weight:0.1},{name:l2sp,weight:1.0}]' \
    env.image_size=112

# DrQ-style augmented RL plus the consistency loss, critic+actor gradients
python train_sac.py augmentation=random_shift augmentation.stage=both encoder.grads=both \
    'encoder.aux=[{name:aug_consistency,weight:0.1}]' env.image_size=112

# BC with the BC loss training the encoder, plus augmentation
python train_bc.py embedding=resnet18 encoder.grads=policy augmentation=random_shift
```

## DrQ-v2 (`train_drqv2.py`)

DrQ-v2's agent (Yarats et al. 2021; `src/rl/drqv2.py`,
`configs/algo/drqv2.yaml`, defaults from its easy-task config): a
deterministic tanh actor with scheduled exploration noise
(`algo.stddev_schedule`, linear from `init` to `final` over `duration`
agent steps; no entropy term), twin critics on n-step targets
(`algo.nstep: 3`) with a clipped-noise target action (`algo.stddev_clip`),
one update per `algo.update_every_steps` agent steps, and a
Linear-LayerNorm-Tanh trunk (`algo.feature_dim`) in front of each MLP.
Uniform random actions until `algo.learning_starts`.

```bash
# DrQ-v2 as published: critic-trained encoder, random shift
python train_drqv2.py env=dmc_walker embedding=resnet18 env.image_size=84 encoder.grads=critic augmentation=random_shift

# Frozen PVR on DrQ-v2 (the default encoder settings)
python train_drqv2.py embedding=r3m_resnet50 env.image_size=112
```

- **Steps**: `algo.action_repeat: 2` env frames per agent step (rewards
  summed). `total_timesteps`, `learning_starts`, the schedule and all
  frequencies count agent steps; `charts/frames` logs env frames, the x-axis
  DrQ-v2 papers use. `log_frequency`/`eval_frequency` must be multiples of
  `update_every_steps`.
- **Encoder**: DrQ-v2's own conv encoder isn't included; any embedding
  works. DrQ-v2 trains its encoder from the critic only
  (`encoder.grads=critic`); `encoder.lr` stays separate from `algo.lr`.
- **Metrics** reuse `train_sac.py`'s keys where they mean the same thing
  (`train/qf*_loss`, `train/qf*_values`, `train/target_q_mean`,
  `train/actor_loss`, grad norms, encoder metrics, `eval/*`), plus
  `train/actor_stddev`. Runs are named `drqv2_...`.
- **`algo.grad_clip: null`** (DrQ-v2 doesn't clip); grad norms are still
  logged. The encoder clips separately at `encoder.grad_clip` (null: no
  clipping, as in the `pieg` preset).
- **Deviation from the reference**: DrQ-v2's buffer only samples starts
  with a full n-step window, so the last `nstep - 1` transitions of each
  episode are never trained on. Here they are, with the shorter window that
  remains (~0.4% of transitions at 500 agent steps per episode).

`algo.nstep` and `algo.action_repeat` also exist for `train_sac.py`
(default 1). SAC's n-step target, `r_t + γ r_{t+1} + ... + γ^n (Q - α log π)(s_{t+n})`,
leaves out the entropy bonuses of the intermediate steps -- the usual
simplification, so `nstep > 1` isn't exactly soft Q-learning. The replay buffer (`src/rl/buffer.py`, shared by both) keeps
each episode's real final observation, so a truncated episode (time limit)
bootstraps from its own last state; before, its last transition's next_obs
was the next episode's first frame. n-step returns never cross an episode
end, and a terminated step gets discount 0.

## Cluster job scripts

`run_sac.sh` (full run) and `debug_sac.sh` (quick smoke test) are the
current SLURM templates for `train_sac.py` — copy and adjust
`embedding=`/`env=`/`encoder.*`/`augmentation=` overrides as needed. `sac_train.sh` /
`sac_debug.sh` are the equivalents for `train_sac_sb3.py`.
