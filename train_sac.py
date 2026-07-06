"""
Smoke-test script for the PVR + SAC + Stable-Baselines3 pipeline.

Three MODES, meant to be run in order so you isolate failures instead of
debugging SAC + pixels + a custom encoder all at once:

  1. state   Vanilla SAC on a state-based env (Pendulum-v1).
             Confirms SAC itself learns -- no vision involved.

  2. pixels  SAC on a pixel env using SB3's default CNN (NatureCNN).
             Confirms the pixel plumbing works: VecTransposeImage,
             image dtype/normalization, replay buffer with image obs.

  3. pvr     Same pixel env, but the CNN is replaced by a frozen ResNet18
             via PVRFeaturesExtractor. This is the config you'll actually
             use to benchmark PVRs against each other.

Two ENVS are supported for the pixel modes:
  --env carracing   gymnasium's CarRacing-v3. Only needs `gymnasium[box2d]`.
                     Good for a same-laptop wiring check.
  --env dmc_cheetah  DeepMind Control's cheetah-run, via `dm_control` +
                     `shimmy` (the Gymnasium-compatibility shim). This is
                     closer to what PVR papers actually benchmark on, but
                     has a heavier dependency stack (mujoco, dm_control).

Usage
-----
    pip install "stable-baselines3[extra]" "gymnasium[box2d]" torch torchvision
    # for dmc_cheetah: pip install dm_control shimmy opencv-python

    # 1) wiring check on your laptop -- tiny numbers, just confirm no crash
    python train_sac.py --mode state  --debug
    python train_sac.py --mode pixels --env carracing --debug
    python train_sac.py --mode pvr    --env carracing --debug

    # 2) real runs -- move to the cluster, esp. for --mode pvr / dmc_cheetah
    python train_sac.py --mode pvr --env dmc_cheetah \
        --total-timesteps 500000 --buffer-size 100000

TODO
----
  - Swap torchvision resnet18 for your pvr_torch encoder / R3M / VC-1
    checkpoint -- PVRFeaturesExtractor only needs forward(x)->(B,embed_dim).
  - Add EvalCallback + tensorboard logging (tensorboard_log=... on SAC)
    for anything beyond a smoke test.
  - Once a policy trains, the `imitation` library (built on top of SB3)
    can both roll out an expert policy into trajectories and train BC on
    them -- likely less work than hand-rolling the trajectory collection.
"""
import argparse

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
# torchvision is imported lazily inside run_pvr() -- state/pixels modes
# don't need it, and its compiled extensions are a separate, more fragile
# dependency (torch/torchvision ABI mismatch) than anything else here.

from feature_extractor import PVRFeaturesExtractor


def make_state_env():
    return make_vec_env("Pendulum-v1", n_envs=1)


def _make_carracing():
    # Ships with gymnasium (pip install gymnasium[box2d]); needs no extra
    # simulator, good for a same-laptop wiring check.
    return gym.make("CarRacing-v3", render_mode="rgb_array")


def _make_dmc_cheetah(image_size: int = 84):
    # dm_control's tasks return proprioceptive state by default -- getting
    # pixels requires explicitly rendering and swapping it in as the obs.
    #
    # MuJoCo/dm_control default to a windowed (GLFW) rendering backend,
    # which fails on a headless cluster node / SSH session with no X11
    # display. `osmesa` is a safe, universal, CPU-only fallback; `egl` is
    # faster (hardware-accelerated offscreen rendering) but needs a GPU
    # node with EGL configured -- try that once osmesa is confirmed working.
    # Must be set before dm_control's C extension loads, hence setdefault()
    # here rather than relying on it being exported in the shell.
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")

    import shimmy  # noqa: F401  (import registers "dm_control/*-v0" ids)
    from gymnasium.wrappers import AddRenderObservation, ResizeObservation

    env = gym.make("dm_control/cheetah-run-v0", render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)  # obs <- rendered frame
    env = ResizeObservation(env, (image_size, image_size))
    return env


ENV_BUILDERS = {
    "carracing": _make_carracing,
    "dmc_cheetah": _make_dmc_cheetah,
}


def make_pixel_env(env_name: str):
    env = make_vec_env(ENV_BUILDERS[env_name], n_envs=1)
    env = VecTransposeImage(env)  # HWC -> CHW, what SB3's CNN policies expect
    return env


def maybe_init_wandb(args, run_name: str):
    """
    Returns a wandb run if --wandb was passed, else None. Uses
    sync_tensorboard=True, which piggybacks on the tensorboard_log you
    already pass to SAC() -- wandb patches the SummaryWriter, so every
    scalar SB3 (and EvalCallback) writes to tensorboard is mirrored to
    wandb automatically, no separate logging code needed. Must be called
    BEFORE the SAC(...) model is constructed so the patch is in place
    before SB3 creates its writer.

    If your cluster's compute nodes have no outbound internet (plausible,
    given you needed to pre-fetch torchvision weights on the login node
    earlier), set `WANDB_MODE=offline` in the job script -- wandb then
    writes locally with no network calls, and you `wandb sync <run_dir>`
    from the login node afterward.
    """
    if not args.wandb:
        return None
    import wandb
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        sync_tensorboard=True,
        save_code=False,
    )


def build_callbacks(args, eval_env, run_name: str, wandb_run=None):
    """
    EvalCallback: periodically runs the current policy on a held-out env
    and logs mean reward -- your actual "is it learning anything" signal
    while the job is running, rather than waiting for it to finish.
    Its `log_path` writes small .npz reward arrays regardless of --no-save;
    that's the number you actually want for a "did it learn" check.

    CheckpointCallback: saves the full model every `checkpoint_freq` steps.
    Skipped entirely under --no-save. Useful for a real run you might need
    to resume after a job timeout; unnecessary disk writes for a throwaway
    "does it learn" check.

    WandbCallback (only if --wandb): also uploads gradients/model info to
    wandb directly. Its own model_save_path is likewise skipped under
    --no-save -- the tensorboard-mirrored scalars (returns, losses, eval
    reward) still show up in the wandb dashboard either way.
    """
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None if args.no_save else f"./checkpoints/{run_name}/best",
        log_path=f"./checkpoints/{run_name}/eval",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    callbacks = [eval_callback]
    if not args.no_save:
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=f"./checkpoints/{run_name}",
            name_prefix=run_name,
        ))
    if wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(
            model_save_path=None if args.no_save else f"./checkpoints/{run_name}/wandb_models",
            verbose=2,
        ))
    return callbacks


def run_state(args):
    env = make_state_env()
    eval_env = make_state_env()
    run_name = "sac_pendulum_state"
    wandb_run = maybe_init_wandb(args, run_name)
    model = SAC(
        "MlpPolicy", env, verbose=1,
        tensorboard_log=args.tensorboard_log,
        learning_starts=min(500, args.total_timesteps),
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=build_callbacks(args, eval_env, run_name, wandb_run),
    )
    if not args.no_save:
        model.save(run_name)
    if wandb_run is not None:
        wandb_run.finish()


def run_pixels(args):
    env = make_pixel_env(args.env)
    eval_env = make_pixel_env(args.env)
    run_name = f"sac_{args.env}_defaultcnn"
    wandb_run = maybe_init_wandb(args, run_name)
    model = SAC(
        "CnnPolicy",  # SB3's default NatureCNN
        env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        buffer_size=args.buffer_size,
        learning_starts=min(1_000, args.total_timesteps),
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=build_callbacks(args, eval_env, run_name, wandb_run),
    )
    if not args.no_save:
        model.save(run_name)
    if wandb_run is not None:
        wandb_run.finish()


RESNET_EMBED_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


def run_pvr(args):
    import torch.nn as nn
    import torchvision.models as tv_models

    env = make_pixel_env(args.env)
    eval_env = make_pixel_env(args.env)
    run_name = f"sac_{args.env}_{args.backbone}"
    wandb_run = maybe_init_wandb(args, run_name)

    encoder = getattr(tv_models, args.backbone)(weights="IMAGENET1K_V1")
    encoder.fc = nn.Identity()  # strip classifier -> pooled feature vector
    embed_dim = RESNET_EMBED_DIMS[args.backbone]

    policy_kwargs = dict(
        features_extractor_class=PVRFeaturesExtractor,
        features_extractor_kwargs=dict(
            encoder=encoder,
            embed_dim=embed_dim,
            input_size=224,
            freeze=True,  # standard PVR-eval protocol: probe, don't finetune
        ),
        net_arch=[256, 256],
        normalize_images=False,  # PVRFeaturesExtractor does its own /255 + ImageNet norm
    )

    model = SAC(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        buffer_size=args.buffer_size,
        learning_starts=min(1_000, args.total_timesteps),
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=build_callbacks(args, eval_env, run_name, wandb_run),
    )
    if not args.no_save:
        model.save(run_name)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["state", "pixels", "pvr"], required=True)
    parser.add_argument("--env", choices=list(ENV_BUILDERS.keys()), default="carracing",
                         help="Only used for --mode pixels/pvr.")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--tensorboard-log", type=str, default="./tb_logs",
                         help="View with: tensorboard --logdir ./tb_logs")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                         help="Run eval episodes every N training steps.")
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000,
                         help="Save a checkpoint every N training steps.")
    parser.add_argument("--backbone", choices=list(RESNET_EMBED_DIMS.keys()),
                         default="resnet18", help="Only used for --mode pvr.")
    parser.add_argument("--no-save", action="store_true",
                         help="Skip all model checkpointing (final save, "
                              "best-model save, periodic checkpoints). Eval "
                              "reward numbers are still logged (small .npz "
                              "files, not full model weights).")
    parser.add_argument("--wandb", action="store_true",
                         help="Mirror tensorboard logs (returns, losses, "
                              "eval reward) to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="pvr-sac")
    parser.add_argument("--wandb-entity", type=str, default=None,
                         help="Defaults to your wandb account's default entity.")
    parser.add_argument("--debug", action="store_true",
                         help="Override to tiny numbers -- just check nothing crashes.")
    args = parser.parse_args()

    if args.debug:
        args.total_timesteps = 300
        args.buffer_size = 1_000
        args.eval_freq = 100
        args.checkpoint_freq = 100

    {"state": run_state, "pixels": run_pixels, "pvr": run_pvr}[args.mode](args)
