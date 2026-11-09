# Running experiments

Practical reference for launching training runs: which script, which envs,
which embeddings (and where their weights come from), and how encoder
training and augmentation work. What exists is summarised in `CAPABILITIES.md`;
how to extend the repo is in `GUIDE.md`.

## Which script

- **`train_sac.py`** — primary script. Explicit CleanRL-style training loop.
  Supports frozen-fast PVR, slow-frozen PVR, PVR finetune (with gradient
  routing), aux losses, augmentation, and state-only baselines. Config:
  `configs/config_sac.yaml`.
- **`train_drqv2.py`** — DrQ-v2's agent with the same encoder, augmentation,
  buffer and wandb setup as `train_sac.py` (every `encoder.*`/`augmentation`
  option works the same). Config: `configs/config_drqv2.yaml` (=
  `config_sac.yaml` with `algo: drqv2`). See [DrQ-v2](#drq-v2-train_drqv2py).
- **`pretrain.py`** — trains an encoder with registered objectives, on video
  clips (R3M, VIP) or images (ImageNet: supervised, MoCo v2, MAE; video
  frames: MAE as in MVP), producing a checkpoint the scripts above load as
  an embedding. Config: `configs/config_pretrain.yaml`. See
  [Pretraining](#pretraining-pretrainpy).
- **`train_xqc.py`** — XQC (Palenicek et al., ICLR 2026): SAC with
  BatchNorm + weight-normalised MLPs and a categorical critic, with the same
  encoder, augmentation, buffer and wandb setup as `train_sac.py`. Config:
  `configs/config_xqc.yaml`. Its pixel recipe (DrQ-v2 with XQC's networks)
  is `train_drqv2.py algo=drqv2_xqc`. See [XQC](#xqc-train_xqcpy-algodrqv2_xqc).
- **`train_discrete.py`** — discrete actions (FourRooms): DQN (`algo=dqn`,
  default) or discrete SAC (`algo=sac_discrete`), with the same encoder,
  augmentation, buffer and wandb setup as `train_sac.py`. Config:
  `configs/config_discrete.yaml`. See [Discrete actions](#discrete-actions-train_discretepy).
- **`train_bc.py`** — behavioral cloning from offline datasets (V-D4RL,
  Minari, pickled trajectories), with the same `encoder.*`/`augmentation`
  options (BC: `encoder.grads` is `none | policy`). Config:
  `configs/config_bc.yaml`. See [Behavioral cloning](#behavioral-cloning-train_bcpy).
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
| `gym`       | `_make_gym`    | generic pixel builder for envs with no native pixel obs and no dm_control-style render_kwargs (Adroit, FrankaKitchen) — `AddRenderObservation` at the env's default camera and native resolution (480×480 for Adroit), then resized to `image_size` before the replay buffer |
| `state`     | `_make_state`  | raw/flattened proprioceptive state (any Gym/dm_control env; `Dict` observations get auto-flattened) |

Only `configs/env/*.yaml` files that have a `builder:` field work with
`train_sac.py`. Currently that's:

- `dm_control_pixels.yaml` / `dm_control_state.yaml` (`dm_control/cheetah-run-v0`, pixels/state) — also the generic template for any other dm_control domain/task, see below
- `dmc_walker.yaml` (`dm_control/walker-walk-v0`, pixels)
- `carracing_pixels.yaml` (`builder: carracing`)
- `pendulum.yaml` (`builder: state`, `Pendulum-v1`)
- `adroit_door_pixels.yaml` / `adroit_door_state.yaml`, and the same `_pixels`/`_state` pair for `adroit_hammer`, `adroit_pen`, `adroit_relocate` (`builder: gym`/`state`)
- `franka_kitchen_pixels.yaml` / `franka_kitchen_state.yaml` (`builder: gym`/`state`)
- `pendulum_pixels.yaml` (`builder: gym`; use `+embedding.frame_stack=3`, one frame shows no velocity)
- `bipedal_walker_pixels.yaml` / `bipedal_walker_state.yaml` (`builder: gym`/`state`; needs Box2D:
  `pip install --only-binary :all: Box2D`; `box2d-py`'s source build fails on Python 3.11). Frames are 600x400, squashed
  to square, since the encoders' center crop would otherwise cut off the walker
- `fourrooms_pixels.yaml` / `fourrooms_state.yaml` (`builder: gym`/`state`,
  `Gym-Gridworlds/FourRooms-13x13-v0`; needs `pip install git+https://github.com/sparisi/gym_gridworlds`).
  Discrete actions: `train_discrete.py`. The state version is the agent's and the goal's cells
  (338 numbers), through `env.wrappers` (observation wrappers by import path). Random start,
  fixed goal (bottom-right corner); `env.kwargs.random_goals=true` moves the goal at every
  reset, which from state took DQN past 200k steps (still ~35% greedy success there)

`env.kwargs` passes extra arguments to `gym.make`, e.g. `env.kwargs.random_goals=true` for FourRooms.

The old `adroit_*.yaml`/`franka_kitchen.yaml`/`dmc_cheetah.yaml` files (only `id:`, no
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
vit_s16  vit_b16  vit_l16      (random init; weights via `checkpoint`)
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
  for `random`, `resnet18/34/50`, `dinov2_*`, `openclip_*` and `vit_s16/b16/l16`
  (which also take MAE-format files: MAE, MVP, VC-1 encoders); the other
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
  Linear(`head_dim`, default 1024) → LayerNorm), or `fc` (a plain trainable
  Linear(`head_dim`), PyTorch's default init, no norm: VIP's representation).
  A `linear`/`fc` head needs the encoder to train, or trained weights from
  `checkpoint` (a file saved by this repo with that head: with a frozen
  encoder, a checkpoint without head weights, e.g. an official R3M
  snapshot, is rejected at startup rather than leaving the head random);
  with `encoder.trainable=head` it trains on a frozen backbone. PIE-G's actor and critic additionally have their own
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
- `r3m_*` — via `load_r3m(...)`, downloads from Google Drive (`gdown`) to R3M's own cache dir on first use. On clusters whose compute nodes have no outbound internet (Triton's do have it), run once on the login node first to warm the cache, or the job will fail trying to download mid-run.
- `vip_resnet50` — via `load_vip(...)`, same lab/codebase as R3M, same download-cache behavior and caveat. Install it without its dependencies (`pip install gdown && pip install --no-deps git+https://github.com/facebookresearch/vip.git`): its `setup.py` pins `hydra-core==1.1.1`, `omegaconf==2.1.1`, `pillow==9.0.1` and `gdown==4.4.0`, and Hydra 1.1 has no `version_base`, which every script here uses -- a plain install (or reinstall) stops them all from starting. R3M's `setup.py` only sets minimum versions, so a plain install is fine there.
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

## Checking an encoder sees the task: `probe_encoder.py`

Before spending RL time: does an encoder's output carry an environment's true
state? It collects frames from a random policy, encodes them through the
training input pipeline, fits a ridge regression and a small MLP to the true
state, and reports R² on held-out episodes (FourRooms: also tiles off and how
often the exact cell comes out). The MLP matters for speeds, which aren't a
linear function of stacked per-frame features. Pass the run's
`algo.action_repeat`, so stacked frames are as far apart as in training. Same `env=` and `embedding=` options as `train_sac.py`; takes
minutes; results print and go to `probe.json`.
```bash
python probe_encoder.py env=fourrooms_pixels embedding=resnet18
python probe_encoder.py env=dm_control_pixels embedding=custom_checkpoint embedding.arch=resnet50 \
    embedding.checkpoint=$RUN/encoder_step1000000.pt embedding.name=r3m50_ours +embedding.frame_stack=3
python probe_encoder.py env=pendulum_pixels embedding=resnet18 embedding.pretrained=false   # random weights, a control
```
A random policy doesn't reach everything a trained one does, so this shows the
information is there; it doesn't predict returns, and it can't show pretraining
helps (random weights score well on it). `embedding.pretrained=false`
gives frozen random weights in the RL scripts too, tagged `-scratch` in run names.

## Encoder training and augmentation

Three independent settings, shared by `train_sac.py` and `train_bc.py` and
validated before anything is loaded (`src/encoder_config.py`):

```yaml
encoder:
  grads: none        # which task losses backprop into the encoder
                     #   SAC: none | critic | actor | both    BC: none | policy
  lr: 1.0e-5         # separate optimizer/LR from the policy/critic
  aux_lr: null       # aux losses' own parameters (decoders, heads); null: lr
  betas: [0.9, 0.999]
  grad_clip: 10.0    # SAC default; BC defaults to 1.0. Aux losses' own parameters
                     #   are clipped separately, and left out of the logged norm
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
- **`encoder.trainable`** — which parts of a training encoder update: `all`
  (default), `head` (only a trainable `embedding.head`), or a list of
  modules, everything else frozen: `'encoder.trainable=[layer3]'` (one
  ResNet stage), `[layer3, layer4]`, `[layer3.1.conv2]` (an exact module),
  `[blocks.6]` (one ViT block), `[layer4, head]`. Names resolve as for
  `embedding.layer`; `head` is the output head. Each listed module must have
  parameters and get gradient from the encoder's output (checked at startup
  with one dummy pass, so e.g. a module after an `embedding.layer` cut is
  rejected), and a trainable `linear`/`fc` head left out of the list must
  come trained from `embedding.checkpoint`. Entries may not overlap (a
  stage and one of its blocks). The startup log prints the selected modules
  and their sizes, wandb's config gets `encoder_trainable_params`, and runs
  get a `_train-<modules>` tag. Training a middle layer still
  backpropagates through the layers after it (that's how its gradient
  arrives), but not through the ones before.
  - **ViTs:** timm-style ViTs apply a final LayerNorm (`norm`) after the
    last block, so "the last two blocks" as usually fine-tuned is
    `[blocks.10, blocks.11, norm]` (ViT-S/B, 12 blocks).
  - **`encoder.bn=adapt` with a list** adapts only the listed modules'
    BatchNorm statistics, so frozen modules stay fixed entirely; frozen
    weights with every statistic adapting is `encoder.grads=none` with
    `encoder.bn=adapt`. A selection without BatchNorm (e.g.
    `[layer3.1.conv2]`) is rejected with `bn=adapt`.
  - Not yet: layer-wise learning rates (LLRD). The natural extension is a
    mapping, `{layer3: 0.1, layer4: 1.0}`, one optimizer group per entry at
    `encoder.lr` × its multiplier (the list being the all-1.0 case).
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
  - `ema_consistency` (`momentum`, default 0.99; `distance: mse | cosine`) is
    the distance between the encoder's features of the clean view and those
    of a target encoder whose weights are an EMA of the encoder's, updated
    every update. It starts at zero and only pulls toward the encoder's own
    recent past, so like `l2sp` it can't train alone. It keeps a second copy
    of the encoder in GPU memory and costs one more encoder pass per update.
    `encoder.bn=adapt` is rejected: the encoder would normalise with batch
    statistics and the target with running ones, and the loss would measure
    that difference.
    Logs `train/aux/ema_consistency/cos`.
  - `reconstruction` (`size`, default 64, a power of 2; `grad`; `latent_l2`,
    default 1e-6; `channels`, default 256) is SAC-AE's: a deconvolutional
    decoder rebuilds the clean frames, downsampled to `size`, from the
    features (MSE; all frames of a stack). `grad: true` trains the encoder.
    `grad: false` detaches the features, so only the decoder learns: a probe
    of what a frozen or RL-trained encoder keeps, which leaves training
    unchanged. Logs `train/aux/reconstruction/psnr`. Set `encoder.aux_lr`
    (SAC-AE: 1e-3); at the default `encoder.lr` the decoder barely learns.
    Reconstructed images aren't logged yet.
  - `moco` and `mae` run on replay or dataset frames as well as in
    `pretrain.py`. Their views are independent draws of the run's augmentation
    with `stage: aux` or `both`, otherwise the clean frames. `moco` compares
    two views, so it needs such an augmentation (rejected otherwise), and
    `encoder.bn=adapt` is rejected with it. Replayed frames can appear as
    their own negatives in its queue, so use a smaller `queue_size` (e.g.
    4096), and set `encoder.aux_lr` for its head. `mae` needs a whole `vit_*`
    encoder (no `layer`, `head` or `frame_stack`); `decoder_depth=2` or so
    makes it cheaper per update. `supervised`, `r3m` and `vip` stay
    `pretrain.py`-only: replay buffers have no labels or clips.

  To add your own, subclass `AuxLoss` in `src/losses.py` and decorate it
  with `@register_aux_loss("name")`; the file's docstring has the contract.
- **`augmentation`** — picks a file from `configs/augmentation/` (`none`,
  `random_shift`, `random_resized_crop`, `color_jitter`, `random_grayscale`,
  `gaussian_blur`, `horizontal_flip`, `random_erasing`, `random_overlay`,
  `shift_jitter` as a pipeline example, the pretraining presets `r3m` and
  `moco_v2`, and the `pieg` preset with a strong view, see below). Each has a
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
  `84`), shrink `algo.buffer_size`, or compress the buffer:
  `algo.buffer_compression=zstd` (lossless; `pip install zstandard`; `lz4`
  and the built-in but slower `zlib` also work). Rendered frames compress
  ~3–5× at 224px (measured: DMC cheetah ~5×, Adroit hammer ~3.7×), so 1M
  frames take ~30–40 GB instead of 151 GB. Sampled batches are decompressed
  in background threads (`algo.buffer_decode_threads`) while the previous
  update runs, so each update trains on a batch drawn one update earlier.
  Compressed frames live in one preallocated byte block used as a ring, so
  the buffer's memory is the bytes it stores; `algo.buffer_max_gb` sizes
  that block (a hard cap: the buffer then keeps the newest frames that fit).
  `charts/buffer_gb` / `charts/buffer_compression_ratio` log the stored
  size, and every run logs the process's memory (`charts/rss_gb`, `rss=` on
  stdout) next to it -- memory growing faster than the buffer points
  elsewhere. Leave headroom in `--mem` beyond the buffer.
  `python -m src.rl.buffer_stress --env adroit_hammer_pixels --image-size 224
  --compression zstd` fills a buffer with real frames while sampling, without
  training, and prints memory against stored bytes plus decode throughput
  (batches/s, which should stay above the run's SPS); `--storage disk` tests
  a disk buffer.
- **Buffer on a local disk**: `algo.buffer_storage=disk` keeps the frames
  (or, with compression, the compressed arena) in a memory-mapped file in
  `algo.buffer_dir` (default `/tmp/$SLURM_JOB_ID`: request a node-local
  disk with `--tmp=<size>` and `--constraint=localdisk`). The file is
  unlinked as soon as it's mapped (freed when the run ends, even after a
  crash) and sparse (disk space is used as frames arrive). At startup the
  folder must not be RAM-backed (tmpfs/ramfs: that would count against the
  job's memory, unreclaimable) and must have room for the whole file, since
  a full disk mid-run kills the process; a compressed arena is sized
  uncompressed unless `algo.buffer_max_gb` caps it (a 1M-frame 224px
  buffer: ~150 GB of `--tmp` without a cap). Recently used pages stay in the page cache, which the
  kernel reclaims under the job's limit, so a small `--mem` holds only the
  hot part. Batches are read (and decompressed) in background threads and
  prefetched; `add()` waits for those reads before it writes. Works with or
  without compression, for SAC and DrQ-v2.
- **Memory check at startup**: a RAM buffer that can't fit the job's memory
  (its cgroup limit, else `--mem`) stops the run before training: the
  uncompressed size, or `algo.buffer_max_gb` with compression. Under 8 GB
  left for everything else gives a warning; an uncapped compressed buffer
  bigger than the job gets a note suggesting a `buffer_max_gb`.
  `PVR_SKIP_BUFFER_CHECK=1` skips it. Disk buffers aren't checked against
  memory.
- **Buffer metrics** (stdout and wandb, every log): `charts/sample_ms` (mean
  wait per batch, incl. the copy to the GPU) and `charts/add_ms`;
  `charts/rss_anon_gb` / `charts/rss_file_gb` (heap and arrays vs mapped
  files, e.g. a disk buffer's cached pages) next to `charts/rss_gb`;
  `charts/cgroup_mem_gb` (what the OOM killer counts, page cache included);
  `charts/buffer_disk_gb` for disk buffers. In disk mode `add()` waits for
  the prefetched reads, so a slow disk shows in `add_ms`: the buffer's cost
  per step is `sample_ms + add_ms`. sacct's MaxRSS includes a disk buffer's
  cached pages (reclaimable); `rss_anon_gb` and `cgroup_mem_gb` are the
  honest numbers. `run_buffer_ram.sh` (array: none / zstd / lz4, 24G) and
  `run_buffer_ssd.sh` (array: none / zstd on the local disk, 8G) compare the
  storage options on one SAC setup; compare `charts/steps_per_sec_recent`
  and `sample_ms + add_ms` once the buffer is full.
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

## XQC (`train_xqc.py`, `algo=drqv2_xqc`)

XQC (Palenicek et al., "XQC: Well-conditioned Optimization Accelerates
Deep Reinforcement Learning", ICLR 2026, arXiv:2509.25174), ported to
PyTorch from the authors' JAX code (`danielpalenicek/xqc`, MIT) in
`src/rl/xqc.py`. It changes only the actor's and critic's MLPs and the
critic loss, in three parts the paper's ablations find each matter (BN
most, then CE, then WN):

- **BatchNorm** on the input and after every hidden Linear (Linear without
  bias → BN → ReLU). The critic sees (s, a) and (s', a') as one batch, so
  its statistics cover both; the target critic normalises with batch
  statistics, the actor loss evaluates the critic with running ones.
- **Weight normalisation**: after every optimizer step, each Linear's
  weight rows are projected to unit norm (the BN after it makes that
  harmless), which keeps the effective learning rate constant.
- **Categorical critic**: 101 atoms on [-5, 5], cross-entropy against the
  C51-projected target of the target critic with the smaller expected
  value. Rewards are divided by the running std of the discounted return
  (`algo.reward_normalization`), so values fit the support.

The paper has two recipes, and the repo has both:

| | `train_xqc.py` (state recipe, Table 1) | `train_drqv2.py algo=drqv2_xqc` (pixel recipe, Table 2) |
|---|---|---|
| agent | SAC: tanh-Gaussian policy, temperature from 0.01, target entropy −A/2 | DrQ-v2: deterministic actor, scheduled noise, no entropy term |
| networks | input BN, critic 4 × 512 (×2), actor 4 × 256 | DrQ-v2's trunk (Linear + LayerNorm + Tanh, 50-d), then 4 × 1024 |
| updates | 2 per agent step, actor every 3rd | 1 per 2 agent steps |
| lr | 3e-4 → 3e-5 linear decay, Adam | 1e-4 constant, Adam |
| targets | 1-step, γ from episode length (`gamma: auto`), τ 0.005 | 3-step, γ 0.99, τ 0.01 |

```bash
# State
python train_xqc.py env=pendulum embedding=none
# Frozen PVR features (the state recipe on encoder features)
python train_xqc.py env=pendulum_pixels embedding=resnet50 +embedding.frame_stack=3
# The paper's pixel recipe: encoder trained from the critic, random shift
python train_drqv2.py algo=drqv2_xqc embedding=random env.image_size=84 \
    embedding.frame_stack=3 encoder.grads=critic augmentation=random_shift
```

- **`gamma: auto`** is the authors' heuristic, `1 − 5/T` clipped to
  [0.95, 0.995], with T the episode length in agent steps (Pendulum 0.975,
  BipedalWalker 0.995, DMC at action repeat 2: 0.99). The startup log
  prints it. A number overrides it.
- **Learning-rate decay** copies the authors' code (optax's
  `linear_schedule` inside `conditionally_mask`, checked against optax
  0.2.3): each optimizer counts its own steps over
  `total_timesteps × updates_per_step`, which includes the warm-up. So the
  critic ends ~1% short of `lr_end` (500k steps, 5k warm-up), and the actor
  and temperature, which step on every 3rd update, only get a third of the
  way (3e-4 → ~2.1e-4). Probably an accident on their side, kept to match
  their curves; `lr_end: null` keeps all three constant.
- **Paper vs repo**: the paper's pixel experiments trained DrQ-v2's conv
  encoder from scratch, and its state recipe used raw state. Using either
  recipe on frozen pretrained features is untested in the paper. Table 2
  doesn't list reward normalisation; it's on here, since DMC returns at γ
  0.99 don't fit [-5, 5] without it.
- **Ablations** (the paper's Fig. 10): `algo.critic_loss=mse`,
  `algo.weight_norm=false`. Both scripts support every `encoder.*` option;
  `augmentation.strong` isn't supported. `obs_norm` doesn't apply to
  `train_xqc.py` (the networks start with their own BatchNorm).
- **α is in normalised-reward units**: the entropy bonus is added inside
  the normalised target, so `train/alpha` drifts with the reward scale.
  Compare `train/alpha_reward_units` (α × reward scale) across tasks or
  with SAC's `train/alpha`.
- **Support saturation**: the normaliser divides by the return's std
  without subtracting its mean, so on dense, same-sign rewards normalised
  values can reach the [-5, 5] edges, where the critic can't tell actions
  apart. `train/target_mass_vmin`/`vmax` log the target's probability on
  the two edge atoms (`edge_mass=` in the console); values well above 0
  mean the support is too narrow.
- **Metrics** reuse `train_sac.py`'s keys, plus `train/critic_lr`,
  `train/actor_lr`, `train/reward_scale` (the std rewards are divided by),
  `train/alpha_reward_units`, `train/target_mass_vmin`/`vmax` and
  `charts/updates` (gradient steps: XQC makes 2 per agent step, SAC 1,
  DrQ-v2 ½, so compare against both). Runs are named `xqc_...`.
- **Comparing with SAC**: besides the method, γ (0.975/0.99 vs 0.99), UTD
  (2 vs 1), network size and input normalisation differ. SAC's networks
  already LayerNorm encoder features (`obs_norm` auto), but per sample,
  where XQC's input BatchNorm standardises each feature over the batch.
  Set `algo.gamma` to the same number for both unless you want XQC as
  published.

## Discrete actions (`train_discrete.py`)

DQN or discrete SAC (`src/rl/discrete.py`) for environments with discrete
actions, on the same encoder, augmentation, replay buffer, wandb and timing
setup as `train_sac.py`:

```bash
python train_discrete.py env=fourrooms_pixels embedding=resnet18                  # DQN, frozen encoder
python train_discrete.py algo=sac_discrete env=fourrooms_pixels embedding=resnet18
python train_discrete.py env=fourrooms_state embedding=none                        # from state
python train_discrete.py env=fourrooms_pixels embedding=resnet18 encoder.grads=critic
```

- **DQN** (`configs/algo/dqn.yaml`): double DQN on n-step targets
  (`algo.nstep: 3`), Huber loss (`algo.loss`), epsilon-greedy exploration
  annealed linearly over the first `algo.epsilon.fraction` of the run
  (`algo.epsilon.{start,end,fraction}`), a hard target copy
  every `algo.target_network_frequency` steps (`algo.tau < 1` for Polyak),
  `algo.double=false` for plain DQN.
- **Discrete SAC** (`configs/algo/sac_discrete.yaml`; Christodoulou 2019): a
  softmax policy and twin Q-networks over all actions, so the soft value and
  policy loss are exact expectations over actions; the temperature is tuned
  towards `target_entropy_scale × log(n_actions)` (0.5; CleanRL's `sac_atari`
  uses 0.89, which with 4 actions keeps the training policy taking its best
  action only ~50% of the time). Evaluation takes the most likely action.
  Where reaching the goal ends the episode (FourRooms), the entropy bonus is
  a reward for staying alive: about `alpha × entropy / (1 − gamma)` when the
  time limit is bootstrapped, forfeited at the goal. Keep
  `algo.alpha × log(n_actions) / (1 − gamma)` well below the goal reward, or
  the agent first learns to avoid the goal until the temperature has decayed
  (at `alpha: 0.2`, `gamma: 0.99`, 4 actions that's ~28 against +1, and
  FourRooms from state scored 0% until ~18k steps on every seed). The default
  `alpha: 0.001` (~0.14) reaches 100% by 12.5k on every seed; the autotuned
  temperature then settles near 0.007 whatever it starts from.
- **Encoder**: `encoder.grads` is `none | critic` -- the critic loss (DQN's
  TD loss, discrete SAC's twin-Q loss) can train the encoder; discrete SAC's
  actor reads detached features. `encoder.aux`, augmentation, `buffer_obs`,
  `embedding.pretrained` and frame stacks work as in `train_sac.py`;
  `augmentation.strong` isn't supported.
- **Updates**: one every `algo.train_frequency` env steps, `algo.gradient_steps`
  each time.
- **Metrics** reuse `train_sac.py`'s keys where they mean the same thing; DQN
  adds `train/epsilon` and `train/q_max_mean`.
- **Evaluation**: greedy (`eval/*`, which picks `best_agent.pt`), plus, with
  `algo.eval_epsilon > 0` (default 0.05, as Nature DQN), an epsilon-greedy
  run logged as `eval_eps/*`: a greedy policy that walks into one wall stays
  there until the time limit, so the greedy number alone can read as total
  failure. Both passes, and every evaluation of the run, reset to the same
  `algo.n_episodes_test` (50) episodes, from fixed seeds drawn from `seed`, so
  differences between them are the policy's, not the episodes'; the
  epsilon coins come from their own generator, so turning
  `algo.eval_epsilon` on doesn't change training. FourRooms' only reward is +1 at
  the goal, so `eval/mean_reward` is its success rate and
  `charts/episodic_length` its steps to the goal (a random policy: 25%
  success, ~171 steps with the fixed goal; 32%, ~158 with random goals).

## Behavioral cloning (`train_bc.py`)

An MLP policy (`hidden_size`, tanh-bounded output) regresses the dataset's
actions with MSE, on embedding features for pixel datasets or raw state
otherwise. 5% of the dataset is held out for `val/loss`; the policy is also
rolled out in `dataset.eval_env_id` every `algo.eval_frequency` epochs for
`algo.n_episodes_test` episodes (`eval/mean_reward`). `dataset=` picks the
data: `vd4rl_cheetah_expert` (or `vd4rl` with `dataset.task`/`dataset.quality`,
downloaded from the HF hub on first use), `door_expert` (Minari),
`halfcheetah_expert`, or a pickle from
`behavioral_cloning/save_opt_trajectories.py` (`dmc_cheetah_random`).
Training counts epochs, not env steps: the data is fixed (cheetah_run expert
is ~50k transitions, ~1.5k batches per epoch at `algo.batch_size: 32`), and
wandb logs with `step=epoch`. Every batch is re-encoded from pixels, so
frozen and finetuned encoders cost about the same per epoch. Datasets hold
single frames, so `embedding.frame_stack > 1` is rejected.

```bash
# Frozen ResNet-18 on V-D4RL cheetah_run expert
python train_bc.py dataset=vd4rl_cheetah_expert embedding=resnet18

# Same, with the BC loss finetuning the encoder
python train_bc.py dataset=vd4rl_cheetah_expert embedding=resnet18 encoder.grads=policy
```

## Pretraining (`pretrain.py`)

Trains an encoder (`embedding=`, from scratch with
`embedding.pretrained=false`; any architecture `embedding.checkpoint` can
load: ResNets, the `random` CNN, DINOv2, OpenCLIP, `vit_*`) on a video or image dataset
(`pretrain_data.type: clips | images`) with objectives from the same
registry as the RL aux losses (`src/losses.py`). A method is a preset
in `configs/experiment/`; everything else (architecture, data, augmentation,
batch size, objective weights) stays an ordinary override, so a reduced or
modified recreation is one command.

```bash
# R3M, released-model settings, ResNet-50 on Something-Something v2
python pretrain.py +experiment=r3m pretrain_data.root=/path/to/ssv2
# smaller encoder / ablations
python pretrain.py +experiment=r3m embedding=resnet18 pretrain_data.root=...
python pretrain.py +experiment=r3m objectives.0.lang_weight=0 pretrain_data.root=...
```

**Data.** A clip dataset is a manifest (`path,len,txt[,start]`, one row per
video) plus its frames, stored either as frame shards (the default: the
frames of all videos in ~1 GB tar files with an index, a few hundred files
for all of SSv2) or as per-video frame folders (`000001.jpg`..., R3M's
layout; one file per frame, so millions for SSv2 -- more than a typical
scratch file quota). `src/data/clips.py` reads either; the official R3M code
would additionally need its own `manifest.csv` with absolute paths and its
2-character text prefix. Something-Something v2 (about 169k training videos,
~20 GB of webm; on Triton a shared archive, which `run_prepare_ssv2.sh`
extracts to a node's local disk and converts there) is converted once per
split, in a CPU job:
```bash
python -m src.data.prepare_ssv2 --videos $SSV2/20bn-something-something-v2 \
    --labels $SSV2/something-something-v2-labels/labels/train.json \
    --out $WRKDIR/data/ssv2 --split train --workers 32   # then --split validation (labels/validation.json)
```
Frames are 224×224 JPEGs (`--resize squash`, as R3M describes its Ego4D
frames; `center_crop` keeps the aspect ratio), texts are SSv2's per-video
labels ("putting a pen on a table"; `--text template` for the class
template). Expect on the order of 100 GB for the training split.
`--limit N` converts a subset. Shards aren't resumable (an interrupted split
is converted again); `--format folders` is, skipping videos finished with
the same settings. `run_pretrain.sh` copies the shards to the node's local
disk at job start. Other datasets need only a script that writes the same
layout (`DATASETS.md` has the formats and an example; `python -m
src.data.check <root>` checks the result and prints a config), plus a
`configs/pretrain_data/<name>.yaml` (`text_strip: 2`
reproduces R3M's cut of Ego4D's "C " narration prefix).

**R3M** (`configs/experiment/r3m.yaml`, objective `r3m`): time-contrastive,
video-language and L1/L2 terms, a term-for-term port of the official
`r3m/trainer.py`. `tests/test_r3m.py` checks it against the official code
(identical metrics and identical weights after an Adam step; runs when the
`r3m` package is installed). With real DistilBERT as well (needs
`transformers` and a download): `PVR_TEST_NETWORK=1 python -m pytest
tests/test_r3m.py -k real_distilbert` -- if only that fails, check the
transformers version first and pin it. Settings are the released models': 16 videos ×
5 frames per step, Adam 1e-4, weights 1/1/1e-5/1e-5, 3 negatives, start/goal
frames from the first/last 20%, one `random_resized_crop` (scale 0.2–1) per
video, ResNet trained from scratch with BatchNorm in train mode, 224px frames
used whole with ImageNet normalisation (`pretrain_data.frames_as_stored`),
fp32, and DistilBERT's dropout on while training with the text re-encoded at
every reward call, as in the official code (`text_dropout: false`: encode
once per step without dropout -- cheaper, but a deviation). 1M steps by default (the paper's experiments; the
released models ran 1.5M). The language term needs `transformers` and
DistilBERT (`distilbert-base-uncased`), downloaded on first use into
`HF_HOME` (the job scripts set `$WRKDIR/hf`; on a cluster without internet on
its compute nodes, load it once on a login node). Differences from the
official code:
- validation batches come from the SSv2 validation split; the official code
  draws them from the training manifest (`pretrain_data.val_manifest=null`
  does that too). Validation doesn't affect training.
- the crop runs on the GPU per batch instead of in loader workers (same crop
  distribution: torchvision's own sampling);
- one GPU, no DataParallel; logged values are interval averages.

**R3M on a ViT** (`configs/experiment/r3m_vit.yaml`): the `r3m` preset
(objective, weights, frame sampling, per-video crop, SSv2 clips) on MVP's
ViT-S/16 from scratch, with ViT training settings instead of R3M's ResNet
ones: 64 videos per step, AdamW 3e-4 (`lr_reference_batch: 64`, so the lr
scales with `batch_size`), weight decay 0.05 without decay on biases and
norms, `warmup_cosine` with 10k warmup steps over `train_steps`,
`grad_clip: 1.0`, bf16. Starting values to tune (lr, weight decay, the
L1/L2 weights for a 384-dim output, the warmup), not a published recipe:
R3M released no ViT. With `+experiment=mvp` on the same `vit_s16`, R3M and
MAE differ only in the objective. `run_r3m_vit.sh` is its job script.
```bash
python pretrain.py +experiment=r3m_vit pretrain_data.root=/path/to/ssv2
```

**VIP** (`configs/experiment/vip.yaml`, objective `vip`): the embedding
distance to a goal frame is trained as a goal-conditioned value function
(VIP's dual offline-RL objective, reward -1 per step) on a random
sub-trajectory start..goal and one transition inside it, plus L1/L2 terms.
A term-for-term port of the official `vip/trainer.py`, frame sampling of its
`VIPBuffer`; `tests/test_vip.py` compares it with the official code
(metrics and weights after an Adam step, exactly; runs only when the `vip`
package is installed -- with `--no-deps`, see `vip_resnet50` above). Same clip data as R3M, no text needed:
```bash
python pretrain.py +experiment=vip pretrain_data.root=/path/to/ssv2
python pretrain.py +experiment=vip embedding=resnet18 train_steps=500_000 ...
```
Settings are the released model's (`config_vip_ego4d.yaml`, which the VIP
README names as its config, and the paper's Appendix D.1): 32 videos × 4
frames per step, Adam 1e-4, 2M steps, gamma 0.98, L1/L2 weights 1e-3, 3
negatives (the objective's own default is 0, as in the official model
class). The
representation is the ResNet's fc resized to 1024 (`embedding.head=fc`,
`head_dim: 1024`), saved with the encoder. Frames: one random resized crop
per video (scale 0.2–1, the same on all four), then the official loader's
resize to 256 and center crop to 224 (`final_crop`, also at validation);
validation batches sample like training batches, without the random crop. The
official loader's reward tensor is float64, so its TD term runs in double
precision; here it stays float32. Fed float32 rewards, the official trainer
gives exactly this loss and update; with float64 the losses agree to ~1e-7,
but weights after one Adam step can differ by up to ~lr (its first step is
about lr × sign(grad), and rounding flips near-zero gradients). 2M steps at
batch 32 is long on one GPU -- size runs with `profile_pretrain.py` (below).

**Output and use.** The run directory gets `checkpoint.pt` (latest state for
resuming; saved with every validation batch, every `eval_frequency` steps),
encoder-only `encoder_step<N>.pt` every `save.keep_every` steps and at the
end (never overwritten), and `embedding.yaml`, pointing at the latest
snapshot: copy it to `configs/embedding/<name>.yaml` and use
`embedding=<name>` in `train_sac.py`/`train_drqv2.py`/`train_bc.py`. The
encoder keeps the architecture's default input pipeline (ResNets: resize
256, center crop 224), the same one the released R3M weights get here, so
the two compare under identical RL preprocessing. Interrupted runs resume
from `checkpoint.pt` when rerun with the same `hydra.run.dir`; the data
stream is reseeded from the step, so it continues rather than repeats. The official R3M checkpoints load the same way:
`embedding.arch=resnet50 embedding.checkpoint=~/.r3m/r3m_50/model.pt` (or
the `snapshot.pt` of an official training run).

**Throughput.** stdout/wandb report `charts/steps_per_sec` and
`charts/data_wait_frac` (share of time waiting for the loader). To size a
full run, train a few hundred steps on a converted subset (`--limit 2000`)
per architecture and read both; a high `data_wait` means more
`num_workers`/CPUs, not a bigger GPU. `profile_pretrain.py` (same config as
`pretrain.py`, clip objectives) breaks a step down: steps/s, ms per phase
(data, encoder, text encoder, reward head, backward, optimizer), GPU busy
fraction, host-device syncs, and a Chrome trace; `run_profile_pretrain.sh`
runs it for R3M and a few variants (`profile.synthetic=true` needs no data).

### Schedules: how batches are drawn and used

The training loop is up to three nested levels, each optional:
```
for epoch:                                  # schedule.sampling: sweep or random
    for batch in the epoch's batches:       # sweep: each item once; random: epoch_batches draws
        for _ in range(updates_per_batch):  # gradient steps on the same batch
            update(augment(batch))          # fresh draws unless reaugment=false
```
```yaml
schedule:
  sampling: null          # sweep | random (null: clips random, images sweep)
  epoch_batches: null     # random: batches per epoch (null: dataset size / batch_size)
  clips_per_video: 1      # clips + sweep: draws per video per epoch
  updates_per_batch: 1
  reaugment: true
epochs / train_steps / train_images   # run length: epochs, updates, or samples drawn
```

| setup | settings |
|---|---|
| R3M / VIP as published | the default: random clips, `train_steps` |
| ImageNet supervised / MoCo / MAE | the default: sweep, `epochs` |
| R3M in epochs over videos | `schedule.sampling=sweep schedule.clips_per_video=4 epochs=50` |
| an epoch of N random batches | `schedule.sampling=random schedule.epoch_batches=N epochs=...` |
| each batch used for K updates | `schedule.updates_per_batch=K` (fresh augmentation; `reaugment=false` for the same views) |

- **Sweep over clips** shuffles every video's `clips_per_video` draws over
  the epoch, so one video's clips rarely share a batch (R3M and VIP take the
  batch's other samples as negatives; their own random negatives pair a
  sample with itself far more often than this happens). Every video counts
  once per epoch, whatever its length, as in R3M's own sampling.
- **Counting:** `train_steps`, `log_frequency` and `eval_frequency` are in
  updates; `train_images` counts samples drawn (a batch reused K times
  counts once). `eval_frequency` must be a multiple of `updates_per_batch`,
  so checkpoints fall between batches (`eval_every_epochs` rounds to one).
- **Resuming** continues at the exact item in sweep streams and random image
  streams (same `batch_size` and `num_workers`). What's drawn from each item
  (a clip's frames, crops, augmentation, negatives) is reseeded, so a resumed
  run is statistically equivalent to an uninterrupted one, not bit-identical,
  and never replays the same draws. Random clip streams are reseeded too.
- **Reused batches and crops:** image crops are cut in the loader, so with
  `reaugment` they stay fixed for the batch; GPU augmentations are redrawn.
  MoCo assumes each batch is used once (its queue would get the same keys K
  times); `pretrain.py` warns.
- The startup log prints the schedule: sampling, batches per epoch, updates
  per batch.

### Multi-GPU pretraining

One process per GPU on one node (DDP, `src/distributed.py`), started by
PyTorch's launcher:
```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 pretrain.py \
    +experiment=mae pretrain_data.root=... hydra.run.dir=$WRKDIR/pretrain_runs/mae_4gpu \
    batch_size=1024 num_workers=15
```
In a job: `--gpus=4`, about 16 CPUs per GPU, and `srun` in front of the same
line.
- **Which objectives:** `supervised` and `mae` (so `+experiment=imagenet_supervised`,
  `mae`, `mvp`). MoCo, R3M and VIP compare samples within a batch
  (negatives, a queue), which would need gathering across GPUs, so they're
  refused with more than one process.
- **`batch_size` is the total** across GPUs, so the same config gives the same
  recipe on any number of GPUs; each GPU takes `batch_size / GPUs`, which
  must divide evenly. Epochs, `train_images` and `lr_reference_batch` all
  use the total, so scaling up is deliberate: e.g. `batch_size=1024` on 4
  GPUs gives MAE/MVP 4x the learning rate (linear scaling) and a quarter of
  the steps. `num_workers` is per GPU.
- **`hydra.run.dir` must be a fixed folder:** every process runs Hydra, and a
  timestamped default could differ between them (checked at startup).
- **Data:** each GPU reads its own share of every batch; a sweep still sees
  every image once per epoch, and resuming continues exactly with the same
  GPU count (a different count warns, as a changed `num_workers` does).
- **What rank 0 alone does:** printing, wandb, validation, checkpoints,
  `timing.json`; the others wait (up to 30 minutes, the limit on any wait
  between GPUs, after which a hung job fails). Validation runs at the per-GPU
  batch size, with `pretrain_data.eval_batches` scaled by the GPU count, so
  it covers the same number of images as on one GPU. Logged training values
  are averaged over GPUs. Augmentation and MAE masks are drawn independently
  per GPU (rank 0's draws are those of a single-GPU run).
- **BatchNorm** (ResNets) normalises per GPU, over `batch_size / GPUs`
  images, as in torchvision's own multi-GPU training; rank 0's running
  statistics are copied to the others at each step. No synced BatchNorm yet.
- **`perf.ddp_find_unused=true`** if DDP stops on parameters that get no
  gradient (it names them); MAE on `vit_*` and supervised ResNets don't need
  it.
- **Status:** tested on CPU (several processes with gloo): the gradient
  equals the single-process one on the full batch, data split and resume
  are exact (`tests/test_distributed.py`). Not yet run on GPUs:
  `debug_ddp.sh` runs the same 1000 steps on 1 and 2 GPUs to compare loss and
  `steps/s`. Both runs train on the whole total batch per step, so `steps/s`
  (per log interval) compares directly. Losses should be close, not equal:
  BatchNorm sees half the batch per GPU and rank 1 draws other crops. If the
  2-GPU run's `data_wait` is clearly higher, the loader limits it: more CPUs
  per GPU, not code.

### ImageNet (supervised, MoCo v2)

```bash
# torchvision's reference recipe: 90 epochs, SGD 0.1 per 256 images, /10 every 30 epochs
python pretrain.py +experiment=imagenet_supervised pretrain_data.root=/path/to/imagenet
# MoCo v2, 200 epochs (PVR's MoCo ResNets); crop-only variant: augmentation=none
python pretrain.py +experiment=moco_v2 pretrain_data.root=/path/to/imagenet
python pretrain.py +experiment=moco_v2 embedding=resnet18 augmentation=none ...
```

**Getting ImageNet** (Kaggle): accept the rules of the
`imagenet-object-localization-challenge` competition on kaggle.com, put an
API token in `~/.kaggle/kaggle.json`, then
`pip install kaggle && kaggle competitions download -c imagenet-object-localization-challenge -p <dir>`
(~160 GB, one zip). Don't unzip it: 1.33M files, more than a typical
scratch file quota. Convert the zip directly into ~150 shard files:
```bash
python -m src.data.prepare_imagenet --source <dir>/imagenet-object-localization-challenge.zip \
    --out /scratch/.../imagenet --workers 16
```
The converter also reads the extracted Kaggle folder, torchvision's
`train/<wnid>/` + `val/<wnid>/` layout, or that layout packed in one tar (a
cluster's shared `ILSVRC2012_imagefolder.tar`, streamed once from start to
end, never extracted). Labels follow the sorted-wnid order
torchvision's ImageNet models use. Images are stored byte-for-byte;
`--short-side 256` re-encodes larger ones, which makes decoding (usually the
bottleneck) several times cheaper at a small cost in detail. `--classes
<file of wnids>` (e.g. ImageNet-100's list) and `--per-class N` make
subsets. Every stored image is decoded once during conversion, so a
file the training loader can't read fails there instead of mid-run
(`--no-verify` skips it). Each worker holds the zip's index in memory:
request ~32 GB for 16 workers.

**Without converting:** `pretrain_data.root` can also point at an extracted
ImageFolder tree (`<root>/train/<wnid>/`, `<root>/val/<wnid>/`; labels in the
same sorted-wnid order). On Triton, `run_imagenet.sh` does this by default:
it extracts the shared `ILSVRC2012_imagefolder.tar` to the node's local
`/tmp/$SLURM_JOB_ID` at job start (scicomp's local-storage pattern), trains
from there, and removes it at the end. Don't point it at millions of files
on a shared filesystem: listing and reading them is slow there.

**Storage** (`src/data/shards.py`): plain tar files of ~1 GB plus an
`index.npz` with each image's shard, byte offset and label; a loader worker
reads an image with one positioned read. `tar tf` still lists a shard.

**Data pipeline** (`src/data/images.py`): every epoch is a fresh
permutation without replacement. Loader workers decode each image and cut
one RandomResizedCrop per view (`pretrain_data.train_crop`; MoCo: 2 views,
scale 0.2–1) to `image_size`; the GPU then applies `augmentation` to each
view with its own draws. Validation: resize 256, center crop 224, a full
pass over the split (`pretrain_data.eval_batches` limits it; MoCo validates
its contrastive loss on random crops). Runs are sized in epochs (`epochs`,
`eval_every_epochs`, `save.keep_every_epochs`), or by a total image count
(`train_images`: epochs and the lr schedule's warmup/milestones stretch to
it, keeping the schedule's shape; for datasets smaller or larger than the
recipe's), learning rates scale with
`batch_size / lr_reference_batch`, and `lr_schedule`'s step and cosine
decay change once per epoch, as in the reference recipes (a linear warmup,
and MAE's `warmup_cosine`, change every step). Resuming continues at
the exact position in the epoch (same `batch_size` and `num_workers`).

**MoCo v2** (objective `moco`): query view through the encoder and an MLP
head, key view through momentum copies of both (0.999, updated at each
training step), InfoNCE against a 65536-key queue at temperature 0.2. The
paper's 8-GPU BatchNorm shuffling is emulated on one GPU (`bn_splits: 8`:
both views encoded in chunks of 32, the keys after a random permutation).
Implemented from the papers, not the official code (its license is
non-commercial).

**Compute.** One GPU, or several (see "Multi-GPU pretraining"). The
supervised ResNet-18 run (90 epochs, 450k steps, bf16, `run_imagenet.sh`)
took 10 h 20 min; ResNet-50 is several times that -- watch `data_wait`; `perf.amp_bf16=true` and
`--short-side 256` help. ImageNet-100 or fewer epochs make quick runs
practical. The supervised classifier and MoCo's heads belong to the
objective, so the saved encoder is the feature extractor, used in RL with
the standard ResNet input pipeline.

### MAE and MVP (ViTs on images or video frames)

```bash
# MAE's official ImageNet recipe: ViT-B/16, 800 epochs
python pretrain.py +experiment=mae pretrain_data.root=/path/to/imagenet
# MVP: MAE on video frames -- MVP's ViT-S/16 on SSv2 frames at 0.3 fps
python pretrain.py +experiment=mvp pretrain_data.root=/path/to/ssv2
python pretrain.py +experiment=mvp pretrain_data.every=10 epochs=400 ...
```

**MAE** (objective `mae`): 75% of each image's 16px patches are masked; the
ViT encodes the rest, a light decoder (512 wide, 8 blocks) reconstructs the
masked patches' pixels, and the loss is the MSE on those, against per-patch
normalised targets (`norm_pix_loss`). The encoder, decoder and loss are the
official `models_mae.py`'s own code, split into `MAEEncoder` / `MAEDecoder`
(`src/vision_models/mae.py`); `tests/test_mae.py` checks the split gives
the official model's loss and gradients exactly. The decoder belongs to the
objective, so the saved encoder is the ViT alone (features: the CLS token
after the final norm, as MAE/MVP/VC-1 use). The `mae` preset is the official
ImageNet recipe: AdamW (betas 0.9/0.95), weight decay 0.05 without decay on
biases and norms (`no_weight_decay_1d`), lr 1.5e-4 per 256 images
(`lr_reference_batch`), 40 warmup epochs then a half cosine updated every
step (`lr_schedule.name=warmup_cosine`), 800 epochs, random resized crop
(scale 0.2–1, bicubic: `pretrain_data.interpolation`) + horizontal flip,
bf16. The official batch is 4096 across GPUs; here `batch_size` is one
GPU's (256), with the lr scaled to it.

**Architectures**: `vit_s16`, `vit_b16`, `vit_l16` (`configs/embedding/`),
MAE's layout as MVP defines it -- MVP's ViT-S has 12 heads of width 32, not
DeiT's 6. Random init only; weights come from `embedding.checkpoint`: a
pretraining snapshot, or a released MAE-format file (a whole MAE model's
decoder keys, or a timm classifier, are dropped). They're built for 224px
input: 256px models (MVP's `vitl-256-mae-egosoup`, 257 position embeddings)
don't load this way (use `mvp_vitl256_mae_egosoup`). Truncation works with
block counts (`embedding.layer=6 embedding.head=cls`). Input pipeline: MAE's
evaluation transform (bicubic resize 256, center crop 224, ImageNet
normalisation).

**MVP** (`configs/experiment/mvp.yaml`): MAE on video frames, with the
MVP paper's recipe (Section 4). Data: `pretrain_data=ssv2_frames` reads the
SSv2 frame shards `prepare_ssv2.py` wrote for the clip dataset as
unlabelled images -- no second conversion. `every: 40` keeps frames 1, 41,
81, ... of each video (so every video contributes at least one), 0.3 fps, the rate MVP sampled SSv2 at (consecutive
12 fps frames are near-duplicates); `val_every: 50` and `eval_batches: 20`
keep validation short. Frames are images to the image pipeline (per-view
crops in the workers, resume at the exact image); objectives that need labels reject
them. Crop scale [0.1, 0.75], MVP's for its wide HOI frames (ours are
squashed square, so the reason doesn't carry over directly). Budget: "MVP's schedule" has two readings, and
`train_images` toggles between them. By default the preset runs MVP's 1600
epochs (MAE's 40 warmup epochs), so each frame is seen as often as in MVP;
SSv2 alone at 0.3 fps (~0.28M frames, estimate: each video keeps
ceil(length / 40)) is smaller than MVP's ~700K, so that's ~40% of MVP's
compute (~1.75M steps at batch 256), and it scales with `every`.
`train_images=1.12e9` instead trains on MVP's total image count (1600 ×
700K), whatever the frame count: ~4,000 epochs here, ~4.4M steps. The run
prints the real frame count and epochs at startup. Read steps/s from a few
hundred `pretrain.py` steps first (`profile_pretrain.py` handles clip
objectives only): either way this takes days on one GPU, and much longer
for the `mae` preset's ViT-B. ViT-S has 12 heads as in MVP's
code and released weights (the paper says 6); the decoder is MAE's
default (the paper says only "a light decoder"). Any image objective can
use `ssv2_frames` (e.g. MoCo on video frames:
`+experiment=moco_v2 pretrain_data=ssv2_frames`).

**Adding an objective**: register it in `src/losses.py`. Clip objectives:
`needs_clips = True` and a `clip_sampler()` that returns a `FrameSampler`
(which frames of each video it reads); `forward()` gets `batch.clip_feats`
(B, T, D) and `batch.text`. Image objectives: `needs_views = <n>` (and
`needs_labels`); `forward()` gets `batch.views` (n augmented views,
preprocessed but not normalised) and encodes them itself with
`batch.encoder.encode_preprocessed`. Objectives that read clips, text or
labels are rejected in `encoder.aux`, since replay buffers hold none of them;
image objectives without labels also work there (`moco`, `mae`), on views
made from the frames.

## Estimating run time

`train_sac.py`, `train_drqv2.py` and `pretrain.py` estimate their remaining
time as they run (`src/timing.py`; `timing.enabled`, on by default, costs
nothing per step):

- **In the logs:** `elapsed=` and `eta=` on every log line; in wandb
  `charts/eta_hours`, `charts/elapsed_hours` and
  `charts/steps_per_sec_recent`. The estimate uses the training rate over
  the last few log intervals (not since the start, so it follows the current
  speed), with evaluation and checkpoint time measured separately and added
  for the evaluations still to come (counted as each script runs them: none
  during SAC/DrQ-v2's warm-up; `pretrain.py` also validates at the last step
  and saves `checkpoint.pt` with every validation; the RL scripts save once,
  at the end). The warm-up (random actions, no updates) is a separate,
  faster phase; the estimate appears once training proper has a measured
  rate. In a resumed run `elapsed=` includes the earlier jobs.
  `charts/SPS` is unchanged (steps since the start / total time, so it's
  inflated by the warm-up early on).
- **`timing.json`** in the run directory: the measured rates, evaluation
  cost, GPU and the settings that drive the cost, rewritten at every log.
- **Before a long run: benchmark mode.** `timing.benchmark_steps=N` trains N
  steps past the warm-up, times one evaluation and one checkpoint write (to a
  scratch file it deletes, never `checkpoint.pt`), prints the predicted time
  for the job and a suggested `--time` in sbatch's format, and stops without
  saving. It runs with wandb off and writes `timing_benchmark.json`, not
  `timing.json` (so a later real run in the same folder doesn't count it as
  earlier time); N must end before the run's total steps. The prediction is the time measured so far -- which contains the
  one-time costs (start-up, data-loader workers, the first CUDA steps) --
  plus the estimated rest, minus the evaluation and save done only for
  timing:
  ```bash
  python train_sac.py ...your settings... timing.benchmark_steps=2000
  python pretrain.py +experiment=r3m_vit ... timing.benchmark_steps=500
  ```
  Run it on the node type the real job will use: rendering, data loading
  and the GPU all set the rate.
- **From earlier runs:** `python -m src.timing summary <folders>` lists the
  measured runs; `python -m src.timing predict <run dir> --steps N
  [--eval-every M]` predicts a run with the same setup but another length,
  including that run's one-time costs (`one_time_seconds` in timing.json;
  a benchmark's folder works too, through its `timing_benchmark.json`).

The rate depends on the whole setup -- encoder, image size, batch size,
environment, GPU -- so a measurement only carries over to runs that match.

## Cluster job scripts

Templates to copy and adjust (Triton partitions, `$WRKDIR` paths). The
pretraining ones copy (or extract) their data to the node's local disk at
job start and resume from `checkpoint.pt` when resubmitted with the same
`RUN_DIR` (chain jobs with `sbatch --dependency=afterany:<jobid>`).

- **Data:** `run_prepare_ssv2.sh`, the one-time SSv2 conversion to shards
  (CPU job).
- **Pretraining:** `run_pretrain.sh` (R3M, ResNet-50 on SSv2),
  `run_vip.sh` (VIP), `run_r3m_vit.sh` (R3M on ViT-S/16),
  `run_mvp_epochs.sh` / `run_mvp_images.sh` (MVP, the two budgets),
  `run_imagenet.sh` (ImageNet: supervised or MoCo v2, set `METHOD`/`ARCH`),
  and `run_pretrain_generic.sh`, any network x data x method x augmentation
  from four variables (`ARCH`, `DATA`, `METHOD`, `AUG`; the combination is
  checked before anything runs, `DRY_RUN=1` prints the command).
  `debug_pretrain.sh` is a smoke test (including a resume), `debug_ddp.sh`
  the same short run on 1 and 2 GPUs (`METHOD`, `ARCH`) to compare loss and
  steps/s,
  `run_profile_pretrain.sh` runs `profile_pretrain.py` on R3M variants.
- **Downstream:** `run_sac.sh` (full run), `run_ft_sac.sh` (fine-tuning)
  and `debug_sac.sh` (quick smoke test) for `train_sac.py` -- adjust
  `embedding=`/`env=`/`encoder.*`/`augmentation=`; `run_ft_r3m.sh` is SAC on
  your own R3M encoder (`embedding=custom_checkpoint`). `debug_pieg.sh` /
  `arrayrun_pieg.sh` run the PIE-G preset on `train_drqv2.py`.
  `run_buffer_ram.sh` / `run_buffer_ssd.sh` (job arrays) compare
  replay-buffer storage options on one SAC setup (see "Buffer metrics"
  above), and `run_buffer_stress_1m.sh` runs `src.rl.buffer_stress` on a
  1M-frame compressed disk buffer, without training. `run_ft_bc.sh` is the template for `train_bc.py`
  (ResNet-18 finetuning on cheetah_run; `run_bc.sh` is a legacy script for
  a different cluster).

`*.sh` files are gitignored, so scripts not yet tracked (e.g.
`run_ft_bc.sh`) need `git add -f`.
