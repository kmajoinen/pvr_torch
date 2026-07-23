# Running experiments

Practical reference for launching training runs: which script, which envs,
which embeddings (and where their weights come from), and how finetuning
works. This complements `README.md` rather than replacing it.

## Which script

- **`train_sac.py`** — primary script. Explicit CleanRL-style training loop.
  Supports frozen-fast PVR, slow-frozen PVR, PVR finetune (with gradient
  routing), and state-only baselines. Config: `configs/config_sac.yaml`.
- **`train_sac_sb3.py`** — kept for posterity, Stable-Baselines3-backed.
  Supports state / CNN-from-scratch (SB3's NatureCNN) / frozen PVR (fast and
  slow) / PVR finetune (actor-only gradient routing, no `encoder_grads`
  choice). Config: `configs/config_sac_sb3.yaml`. Use this specifically if
  you need SB3's `CnnPolicy` baseline or don't care about gradient-routing
  control.

Everything below is about `train_sac.py` unless stated otherwise.

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
python train_sac.py env=dm_control_pixels env.id=dm_control/walker-walk-v0 env.name=dmc_walker
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
mae_base
moco_aug
```

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

## Finetuning

Controlled by the `finetune:` block in `config_sac.yaml`:
```yaml
finetune:
  enabled: false          # false -> frozen-fast path (FrozenEncoderWrapper,
                           #   features precomputed once per env step)
  encoder_grads: critic   # none | critic | actor | both
  encoder_lr: 1.0e-5       # separate optimizer/LR from the policy/critic
  grad_clip: 10.0
  aux_loss: none           # none | l2sp
  aux_weight: 1.0
```

- **`enabled: false`** — the fast path. Encoder runs once per env step
  outside the training loop; replay buffer stores feature vectors, not
  pixels. This is what you want for a pure frozen-PVR comparison.
- **`enabled: true, encoder_grads: none, aux_loss: none`** — "slow-frozen":
  pixels go in the buffer and get re-encoded every gradient step under
  `no_grad`, but nothing trains. Slower than the fast path for the same
  result; only useful as a structural stepping stone toward finetuning or
  augmentation experiments (which need pixels in the buffer).
- **`encoder_grads`** — which RL loss trains the encoder: `critic` (DrQ-v2's
  default), `actor`, `both` (sequential Adam steps, not one combined
  gradient — see the module docstring in `train_sac.py`), or `none`.
- **`aux_loss: l2sp`** — independent of `encoder_grads`; anchors weights to
  their pretrained values (`||θ - θ₀||²`). Combine with `encoder_grads=none`
  for pure regularized adaptation with zero RL gradient into the encoder.
- **Memory**: finetuning stores raw pixels in the replay buffer
  (`2 × buffer_size × H × W × 3` bytes for obs + next_obs). At the frozen
  path's native 224px render that's ~30GB per 100k transitions — override
  `env.image_size=112` (or `84`) and/or shrink `algo.buffer_size` for
  finetune runs. The estimate is printed at startup either way.

Augmentation (currently `random_shift` only — DrQ-v2-style pad+shift, not
literal DrQ-v1 random-crop; see `src/augmentations.py` for the full
registry) plugs into the same finetune path — it only applies to sampled
training batches, never to rollout/eval, and only fires when pixels are
actually in the buffer:
```yaml
# configs/embedding/<name>.yaml
augmentation: random_shift   # null by default
```

### Example finetune commands
```bash
# DrQ-v2-style critic-only finetuning
python train_sac.py finetune.enabled=true finetune.encoder_grads=critic \
    env.image_size=112 embedding=resnet18

# Slow-frozen (protocol-compatible with the fast path, needed for augmentation)
python train_sac.py finetune.enabled=true finetune.encoder_grads=none \
    env.image_size=112

# Aux-only adaptation, zero RL gradient into the encoder
python train_sac.py finetune.enabled=true finetune.aux_loss=l2sp \
    finetune.encoder_grads=none env.image_size=112
```

## Cluster job scripts

`run_sac.sh` (full run) and `debug_sac.sh` (quick smoke test) are the
current SLURM templates for `train_sac.py` — copy and adjust
`embedding=`/`env=`/`finetune.*` overrides as needed. `sac_train.sh` /
`sac_debug.sh` are the equivalents for `train_sac_sb3.py`.
