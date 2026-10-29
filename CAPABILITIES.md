# What the repo can do: pretraining

State as of 2026-10-29. This covers the pretraining side (`pretrain.py`);
the RL/BC side that uses the trained encoders is summarised at the end.
Commands and options in detail: `RUNNING_EXPERIMENTS.md`.

**Status labels** used throughout:

| label | meaning |
|---|---|
| **done** | implemented, and a full run has completed on the cluster |
| **ready** | implemented and tested (unit tests and a short end-to-end run), no full run yet |
| **experimental** | builds and trains a step, but untested at scale and without a matching recipe |
| **planned** | not implemented yet |

## 1. The idea in one command

"Pretrain with algorithm *x*, on data *y*, with architecture *z* and
augmentation *k*" is one command, with each choice a separate option:

```bash
python pretrain.py +experiment=<x> pretrain_data=<y> embedding=<z> augmentation=<k> \
    pretrain_data.root=/path/to/data [any other option=value ...]
```

- `+experiment=<x>` loads the method's full recipe (objective, optimizer,
  schedule, batch size, default architecture/data/augmentation).
  Anything after it overrides the recipe.
- The result is an encoder checkpoint that the RL/BC scripts load as an
  embedding, so the pretraining choices become one downstream variable.
- `-m` sweeps any of these (e.g. `embedding=resnet18,resnet50 seed=1,2,3`).

## 2. The four choices at a glance

| choice | option | available now | planned |
|---|---|---|---|
| **algorithm** (*x*) | `+experiment=` preset, or `objectives=[...]` | R3M, VIP, supervised classification, MoCo v2, MAE (MVP-style on video frames) | DINO, DynaMo, LIV |
| **data** (*y*) | `pretrain_data=` | Something-Something v2 as video clips + text, SSv2 frames as images, ImageNet-1k, any ImageFolder-layout image set | Ego4D-style data (needs a prep script) |
| **architecture** (*z*) | `embedding=` | ResNet-18/34/50; small 5-layer CNN; ViT-S/16, ViT-B/16, ViT-L/16 (MAE layout); DINOv2 and OpenCLIP ViTs experimental | — |
| **augmentation** (*k*) | `augmentation=` | 9 transforms, combinable into pipelines; presets for R3M/VIP and MoCo v2 | — |
| supporting options | optimizer, schedule, precision | Adam, AdamW, SGD; constant / cosine / step with warmup, MAE's warmup-cosine; bf16; run length in epochs, steps or total images (`train_images`) | multi-GPU (DDP) |

## 3. Which algorithm accepts which data

Each algorithm reads one kind of data, and `pretrain.py` checks this at
startup. This coupling is the main restriction on free combination.

| algorithm | video clips + text (SSv2) | video frames as images (SSv2 frames) | labelled images (ImageNet) |
|---|---|---|---|
| R3M | **yes** (needs text unless `lang_weight=0`) | no | no |
| VIP | **yes** (no text needed) | no | no |
| supervised | no | no (frames have no labels) | **yes** |
| MoCo v2 | no | **yes** | **yes** (labels ignored) |
| MAE / MVP | no | **yes** (MVP) | **yes** (MAE) |

Several objectives can be trained together as a weighted sum
(`objectives=[{...}, {...}]`), as long as they read the same kind of data
(clip objectives must also agree on which frames they sample).

## 4. Algorithms

| | R3M | VIP | supervised | MoCo v2 | MAE / MVP |
|---|---|---|---|---|---|
| **status** | done | ready | done | ready | ready |
| **kind** | self-supervised, video + language | self-supervised, video (goal-conditioned value) | supervised | self-supervised, contrastive | self-supervised, masked reconstruction |
| **learns from** | which frame is later in a video; which frame pairs match the video's description | embedding distance to a goal frame behaves like a value function along the video | ImageNet class labels | two augmented crops of one image should match, other images shouldn't | reconstruct the pixels of 75% masked image patches |
| **data** | clips + text | clips | labelled images | images / frames | images / frames |
| **preset** | `+experiment=r3m` | `+experiment=vip` | `+experiment=imagenet_supervised` | `+experiment=moco_v2` | `+experiment=mae` (ImageNet), `+experiment=mvp` (SSv2 frames) |
| **recipe** | released R3M models: batch 16 videos x 5 frames, Adam 1e-4, 1M steps | released VIP model: batch 32 videos x 4 frames, Adam 1e-4, 2M steps, gamma 0.98, 3 negatives | torchvision reference: 90 epochs, SGD 0.1 per 256 images, lr /10 at epochs 30 and 60 | MoCo v2 paper: 200 epochs, SGD 0.03, cosine, queue 65536, temperature 0.2 | official MAE: AdamW 1.5e-4 per 256 images, wd 0.05, 40 warmup epochs + cosine, 800 epochs; MVP paper: same, 1600 epochs, frames at 0.3 fps, crop scale 0.1-0.75 |
| **fidelity** | term-for-term port of the official code; test checks identical loss and weights after one step | term-for-term port of the official code; checked against the official trainer in review (exact loss and update); the repo's parity test needs the `vip` package | standard recipe; check against torchvision's weights proposed | written from the papers; 8-GPU BatchNorm shuffle emulated on one GPU | runs the official MAE model code, split into encoder and decoder; test checks identical loss and gradients |
| **default architecture** | ResNet-50 | ResNet-50 + 1024-wide fc | ResNet-50 | ResNet-50 | ViT-B/16 (MAE), ViT-S/16 (MVP) |
| **ablations built in** | `lang_weight=0` (time-contrastive only), `tcn_weight=0` (language only), negatives, frame window | `num_negatives`, gamma, L1/L2 weights | label smoothing | crop only (`augmentation=none`), queue size, temperature | mask ratio, decoder size, `norm_pix_loss`, frame thinning (`pretrain_data.every`) |
| **completed runs** | ResNet-50, SSv2, 1M steps (~23 h at ~12 steps/s) | none yet | ResNet-18, ImageNet, 90 epochs (450k steps, 10 h 20 min) | none yet | none yet |

Not pretraining objectives, but in the same registry: `l2sp` and
`aug_consistency`, auxiliary losses for fine-tuning an encoder during RL/BC.

## 5. Architectures

| architecture | `embedding=` | output | pretrain from scratch | downstream with own weights | released weights (load-only) |
|---|---|---|---|---|---|
| ResNet-18 / 34 / 50 | `resnet18`, `resnet34`, `resnet50` | 512 / 512 / 2048 (or `head=fc` to any width) | **done** (RN18 supervised, RN50 R3M) | yes | torchvision ImageNet; R3M (18/34/50); VIP, LIV (50); PVR's MoCo variants |
| small 5-layer CNN | `random` | conv map | ready (untested in pretraining) | yes | — |
| ViT-S/16, ViT-B/16, ViT-L/16 (MAE layout; ViT-S as in MVP, 12 heads) | `vit_s16`, `vit_b16`, `vit_l16` | CLS token, 384 / 768 / 1024 | ready (MAE / MVP) | yes, also from released 224px MAE-format files (MAE, MVP, VC-1) | — |
| DINOv2 ViT-S/B/L/g, 14px patches | `dinov2_vits14` ... | CLS token, 384 / 768 / 1024 / 1536 | **experimental**: one supervised and one MoCo step run on ViT-S; no recipe | yes | DINOv2 |
| OpenCLIP ViT-B/32, ViT-L/14, RN50 | `openclip_*` | CLIP embedding | experimental (not tested) | yes | OpenCLIP (LAION), OpenAI |
| released-only models | `r3m_*`, `vip_resnet50`, `liv_resnet50`, `mvp_*`, `vc1_*`, `mae_base`, `clip_vit`, `rl3d_resnet18`, ... | various | no: these builders always load their own weights | — | yes (the baselines to compare against) |

On any architecture: `embedding.layer` stops the network early (e.g.
`layer2`, or the first 6 ViT blocks), and `embedding.head` picks how the
output becomes a vector (`avgpool`, `cls`, `flatten`, or a trainable
`linear` / `fc`).

## 6. Data

| dataset | `pretrain_data=` | kind | size | storage | status |
|---|---|---|---|---|---|
| Something-Something v2 | `ssv2` | video clips + one description per video | 168,913 train / 24,777 val videos; frames 224x224 (squashed), JPEG q95 | ~1 GB tar shards (`prepare_ssv2.py`) | **done** (R3M) |
| SSv2 frames | `ssv2_frames` | the same shards read as unlabelled images, every 40th frame of each video (0.3 fps, as MVP sampled SSv2) | ~0.28M frames (estimate) | no second conversion | ready (MVP, MoCo) |
| ImageNet-1k | `imagenet` | labelled images | 1.28M train / 50k val, 1000 classes | shards (`prepare_imagenet.py`, from Kaggle's zip or a tar) or an extracted ImageFolder | **done** (supervised); subsets with `--classes` (e.g. ImageNet-100) and `--per-class` |
| other image sets | new yaml | images (+ labels) | — | ImageFolder layout, or shards via `prepare_imagenet.py` | ready |
| other video sets | new yaml | clips (+ text) | — | needs a prep script writing the same manifest + shards | ready (format), none prepared |

How the data is consumed:
- **Clips:** an endless stream, each sample a random video and the frames
  the algorithm asks for. The budget is a number of updates (`train_steps`).
- **Images and frames:** true epochs: every image once per epoch, in a
  fresh random order. The budget is `epochs` (or `train_steps`).

## 7. Augmentation

| transform | `augmentation=` | notes |
|---|---|---|
| random resized crop | `random_resized_crop` | R3M's and VIP's crop; for images, the crop happens in the loader instead (`pretrain_data.train_crop`, bilinear or bicubic) |
| colour jitter | `color_jitter` | |
| greyscale | `random_grayscale` | |
| Gaussian blur | `gaussian_blur` | |
| horizontal flip | `horizontal_flip` | supervised and MAE presets |
| random erasing | `random_erasing` | |
| random shift (DrQ) | `random_shift` | RL-style pad-and-shift |
| random overlay | `random_overlay` | blends in natural images; needs an image folder (e.g. Places365) |
| none | `none` | |
| **presets** | `r3m`, `moco_v2`, `pieg`, `shift_jitter` | `moco_v2` = jitter + greyscale + blur + flip as one pipeline |

Any list of transforms can be combined with `name: pipeline` (see
`configs/augmentation/moco_v2.yaml`). Video clips: one draw per video,
applied identically to all its frames (`clip_augmentation=shared`, as in R3M
and VIP), or a separate draw per frame (`per_frame`); `final_crop` adds a
fixed resize + center crop afterwards (VIP's loader). Images: each view gets
its own draw.

## 8. Examples

Each line reads as "pretrain with *x*, on *y*, with *z*, and *k*". Data paths
(`pretrain_data.root=...`) are left out.

| x / y / z / k | command | status |
|---|---|---|
| R3M / SSv2 / ResNet-50 / R3M crop (the completed run) | `+experiment=r3m` | done |
| R3M without language (time-contrastive only) / SSv2 / ResNet-18 / R3M crop | `+experiment=r3m embedding=resnet18 objectives.0.lang_weight=0` | ready |
| VIP / SSv2 / ResNet-50 + fc / VIP crop | `+experiment=vip` | ready |
| VIP with 3 negatives / SSv2 / ResNet-18 / VIP crop | `+experiment=vip embedding=resnet18 objectives.0.num_negatives=3` | ready |
| supervised / ImageNet / ResNet-18 / flip (the completed run) | `+experiment=imagenet_supervised embedding=resnet18 perf.amp_bf16=true` | done |
| supervised / ImageNet-100 / ResNet-50 / flip | `+experiment=imagenet_supervised objectives.0.num_classes=100` (data prepared with `--classes`) | ready |
| MoCo v2 / ImageNet / ResNet-50 / MoCo v2 augmentation | `+experiment=moco_v2` | ready |
| MoCo v2 / SSv2 frames / ResNet-18 / crop only | `+experiment=moco_v2 pretrain_data=ssv2_frames embedding=resnet18 augmentation=none` | ready |
| MAE / ImageNet / ViT-B/16 / crop + flip (official recipe) | `+experiment=mae` | ready |
| MVP: MAE / SSv2 frames / ViT-S/16 / crop + flip | `+experiment=mvp` | ready |
| MVP, shorter run on more frames / SSv2 frames / ViT-S/16 | `+experiment=mvp pretrain_data.every=10 epochs=400` | ready |
| MVP with MVP's total image count instead of its epochs | `+experiment=mvp train_images=1.12e9` | ready |
| architecture x seed sweep | `-m +experiment=r3m embedding=resnet18,resnet50 seed=1,2,3` | ready |

Using the result downstream (one command per encoder, same RL settings):
```bash
python train_sac.py embedding=custom_checkpoint embedding.arch=resnet50 \
    embedding.checkpoint=$RUN_DIR/encoder_step1000000.pt embedding.name=r3m50_ssv2_ours
python train_sac.py embedding=custom_checkpoint embedding.arch=vit_s16 \
    embedding.checkpoint=$RUN_DIR/encoder_step<N>.pt embedding.name=mvp_s16_ssv2_ours
python train_sac.py embedding=r3m_resnet50     # the released R3M, for comparison
```
(A VIP encoder also needs `embedding.head=fc embedding.head_dim=1024`; the
run's `embedding.yaml` lists every field.)

## 9. Limits

**Combinations**
- Each algorithm accepts one kind of data (table in section 3). Notably,
  R3M and VIP need clips, and supervised training needs labelled images
  (not video frames).
- A preset's recipe is tuned for its default architecture. Swapping the
  architecture doesn't change the optimizer or learning rate (e.g. MoCo's
  SGD recipe on a ViT composes, but isn't a sensible ViT recipe).
- Only ResNets, the small CNN, the `vit_*` ViTs, DINOv2 and OpenCLIP can be
  trained. The other models in the repo (R3M, VIP, LIV, MVP, VC-1, MAE,
  OpenAI CLIP) are load-only baselines (MAE-format files also load into
  `vit_*`).

**Methods and data**
- VIP, MoCo and MAE/MVP have no completed full run yet.
- No DINO, DynaMo or LIV yet (see `TODO.md`, Roadmap).
- Only SSv2 and ImageNet are prepared. R3M's and VIP's own data (Ego4D) and
  MVP's (hand-object-interaction frames) aren't available, so these runs are
  the methods on SSv2, not reproductions of the released models.
- **Epochs vs images:** a recipe's epochs only mean something relative to
  its dataset's size. MVP trained 1600 epochs over ~700K frames (~1.1B
  images); SSv2 alone at 0.3 fps is ~0.28M frames (estimate). The MVP preset
  keeps MVP's 1600 epochs (same repetition per frame, ~40% of MVP's compute,
  ~1.75M steps); `train_images=1.12e9` toggles to MVP's image count (same
  compute, ~4,000 epochs, ~4.4M steps). Either takes days on one
  GPU even for a ViT-S.
- A ViT-S has 12 heads here, as in MVP's code and released weights (the
  paper says 6). 256px ViTs (MVP's ViT-L/256) don't load into `vit_l16`.
- A frozen encoder with a trainable head (`linear`/`fc`) must load that
  head from its checkpoint; otherwise the run stops at startup.

**Compute and evaluation**
- One GPU per run (no multi-GPU/DDP). This caps the batch size (MAE's
  official batch is 4096; here 256 with the learning rate scaled), makes
  long runs slow (R3M ResNet-50: ~23 h for 1M steps; VIP's 2M steps at
  batch 32 longer), and MoCo's multi-GPU BatchNorm shuffle is emulated.
- Validation during clip pretraining is one batch, so the curves are noisy.
  Image runs validate on the validation split (or a fixed number of batches).
- The quality of an encoder is only measured downstream (RL/BC). No
  linear probe or reconstruction probe yet (proposed in `TODO.md`).
- Loading your own encoder downstream needs `custom_checkpoint` plus `arch`
  and `checkpoint` (simplification tabled in `TODO.md`).

## 10. Downstream side (for context)

| | options |
|---|---|
| agents | SAC (`train_sac.py`), DrQ-v2 (`train_drqv2.py`), behavioural cloning (`train_bc.py`) |
| environments | DeepMind Control (any domain/task), Adroit (door, hammer, pen, relocate), FrankaKitchen, Pendulum; pixels or state; CarRacing configured but missing its dependency |
| BC datasets | V-D4RL, Minari, pickled trajectories |
| encoder use | frozen (features cached in the buffer), or fine-tuned from the critic, actor, both, or the BC loss; auxiliary losses; BatchNorm adaptation; frame stacking |
| augmentation | the same transforms, plus a SVEA/PIE-G "strong view" loss; `+experiment=pieg` preset |
| embeddings | every architecture above plus about 30 released models (section 5) |
