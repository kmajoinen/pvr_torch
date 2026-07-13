"""
SAC on state / pixels / pretrained-visual-representation (PVR) features.

Modes (cfg.mode):
  state   Vanilla SAC on Pendulum-v1 -- sanity check that SAC learns at all.
  pixels  SAC + SB3's NatureCNN on a pixel env -- checks the pixel plumbing.
          (override env.image_size=84 for this one; NatureCNN wants small frames)
  pvr     SAC on top of a pretrained encoder. With freeze_encoder=true
          (standard PVR-eval protocol) the encoder runs once per env step in
          a VecEnv wrapper and SAC trains an MlpPolicy on features -- the
          fast path. With freeze_encoder=false the encoder sits inside the
          policy as a features extractor and finetunes jointly.

Usage
-----
    python train_sac.py                                    # defaults from configs/
    python train_sac.py embedding=resnet18 algo.total_timesteps=500000
    python train_sac.py mode=pixels env.image_size=84
    python train_sac.py freeze_encoder=false wandb.enabled=true
    python train_sac.py --multirun embedding=resnet18,resnet50,mae_base seed=1,2,3

Hydra chdirs into hydra.run.dir per run, so checkpoints/, tb_logs/ and eval
logs land in a self-contained run directory; anything that must resolve
against the repo root (models_dir) goes through to_absolute_path().
"""
import os

# Must be set before dm_control's C extension loads; EGL = GPU offscreen
# rendering, the fast option on a GPU node. Override in the shell if needed.
os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import gymnasium as gym
import numpy as np
import torch as th
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

from feature_extractor_fable5 import (
    LayerNormExtractor,
    PVREncoderWrapper,
    PVRFeaturesExtractor,
    build_encoder,
)


# ---------------------------------------------------------------------------
# Envs
# ---------------------------------------------------------------------------

def _make_dmc(env_id: str, image_size: int):
    import shimmy  # noqa: F401  (registers "dm_control/*-v0" ids)
    from gymnasium.wrappers import AddRenderObservation

    # Render natively at the encoder's input size instead of rendering small
    # and upsampling 84->224 on GPU: better image, no interpolation pass.
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        render_kwargs=dict(height=image_size, width=image_size, camera_id=0),
    )
    return AddRenderObservation(env, render_only=True)


def _make_carracing(env_id: str, image_size: int):
    # CarRacing renders 96x96 natively; any resize to the encoder input size
    # happens on GPU in the wrapper/extractor.
    return gym.make(env_id, render_mode="rgb_array")


ENV_BUILDERS = {"dmc": _make_dmc, "carracing": _make_carracing}


def make_pixel_env(cfg: DictConfig):
    builder = ENV_BUILDERS[cfg.env.builder]
    return make_vec_env(
        lambda: builder(cfg.env.id, cfg.env.image_size),
        n_envs=cfg.env.n_envs,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv if cfg.env.n_envs > 1 else DummyVecEnv,
    )


# ---------------------------------------------------------------------------
# Logging / callbacks
# ---------------------------------------------------------------------------

def maybe_init_wandb(cfg: DictConfig, run_name: str):
    """sync_tensorboard=True mirrors every scalar SB3 writes to tensorboard
    into wandb; must run BEFORE SAC() so the SummaryWriter patch is in place.
    On nodes without outbound internet: WANDB_MODE=offline + `wandb sync`."""
    if not cfg.wandb.enabled:
        return None
    import wandb
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name or run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,
        save_code=False,
    )


def build_callbacks(cfg: DictConfig, eval_env, run_name: str, wandb_run=None):
    callbacks = [EvalCallback(
        eval_env,
        best_model_save_path=f"checkpoints/{run_name}/best" if cfg.save.enabled else None,
        log_path=f"checkpoints/{run_name}/eval",
        eval_freq=max(cfg.algo.eval_frequency // cfg.env.n_envs, 1),
        n_eval_episodes=cfg.algo.n_episodes_test,
        deterministic=True,
    )]
    if cfg.save.enabled:
        callbacks.append(CheckpointCallback(
            save_freq=max(cfg.save.checkpoint_frequency // cfg.env.n_envs, 1),
            save_path=f"checkpoints/{run_name}",
            name_prefix=run_name,
        ))
    if wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(
            model_save_path=f"checkpoints/{run_name}/wandb_models" if cfg.save.enabled else None,
            verbose=2,
        ))
    return callbacks


def sac_kwargs(cfg: DictConfig):
    return dict(
        verbose=1,
        seed=cfg.seed,
        device=cfg.device,
        tensorboard_log="tb_logs",
        buffer_size=cfg.algo.buffer_size,
        batch_size=cfg.algo.batch_size,
        learning_starts=min(cfg.algo.learning_starts, cfg.algo.total_timesteps),
        train_freq=cfg.algo.train_freq,
        gradient_steps=cfg.algo.gradient_steps,
        # Previously omitted entirely -- SAC() silently ran on SB3's hardcoded
        # defaults (3e-4/0.99/0.005/etc) regardless of what configs/algo said.
        # Wired through here for a fair comparison against the other scripts.
        learning_rate=cfg.algo.learning_rate,
        gamma=cfg.algo.gamma,
        tau=cfg.algo.tau,
        target_update_interval=cfg.algo.target_update_interval,
        ent_coef=cfg.algo.ent_coef,
    )


def fit(cfg: DictConfig, model: SAC, eval_env, run_name: str, wandb_run):
    model.learn(
        total_timesteps=cfg.algo.total_timesteps,
        callback=build_callbacks(cfg, eval_env, run_name, wandb_run),
    )
    if cfg.save.enabled:
        model.save(run_name)
    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_state(cfg: DictConfig):
    run_name = "sac_pendulum_state"
    wandb_run = maybe_init_wandb(cfg, run_name)
    env = make_vec_env("Pendulum-v1", n_envs=1, seed=cfg.seed)
    eval_env = make_vec_env("Pendulum-v1", n_envs=1, seed=cfg.seed + 1)
    model = SAC("MlpPolicy", env, **sac_kwargs(cfg))
    fit(cfg, model, eval_env, run_name, wandb_run)


def run_pixels(cfg: DictConfig):
    run_name = f"sac_{cfg.env.name}_naturecnn"
    wandb_run = maybe_init_wandb(cfg, run_name)
    env = VecTransposeImage(make_pixel_env(cfg))
    eval_env = VecTransposeImage(make_pixel_env(cfg))
    model = SAC("CnnPolicy", env, **sac_kwargs(cfg))
    fit(cfg, model, eval_env, run_name, wandb_run)


def run_pvr(cfg: DictConfig):
    run_name = f"sac_{cfg.env.name}_{cfg.embedding.name}" \
               f"{'' if cfg.freeze_encoder else '_ft'}"
    wandb_run = maybe_init_wandb(cfg, run_name)

    encoder, embed_dim, input_size = build_encoder(
        cfg.embedding.name, models_dir=to_absolute_path(cfg.embedding.models_dir)
    )

    if cfg.freeze_encoder:
        # Fast path: encode once per env step, train MlpPolicy on features.
        if cfg.perf.compile_encoder:
            encoder = th.compile(encoder)
        def wrap(venv):
            # Sharing one encoder instance between train and eval envs is
            # safe here: it lives outside the policy, pure inference only.
            return PVREncoderWrapper(
                venv, encoder, embed_dim, input_size,
                device=cfg.device, amp_bf16=cfg.perf.amp_bf16,
            )
        env = wrap(make_pixel_env(cfg))
        eval_env = wrap(make_pixel_env(cfg))
        policy = "MlpPolicy"
        policy_kwargs = dict(
            features_extractor_class=LayerNormExtractor,
            net_arch=list(cfg.algo.net_arch),
        )
    else:
        # Finetune path: encoder inside the policy, trained jointly.
        env = VecTransposeImage(make_pixel_env(cfg))
        eval_env = VecTransposeImage(make_pixel_env(cfg))
        policy = "CnnPolicy"
        policy_kwargs = dict(
            features_extractor_class=PVRFeaturesExtractor,
            features_extractor_kwargs=dict(
                encoder=encoder, embed_dim=embed_dim,
                input_size=input_size, freeze=False,
            ),
            net_arch=list(cfg.algo.net_arch),
            normalize_images=False,  # extractor does /255 + ImageNet norm itself
        )

    model = SAC(policy, env, policy_kwargs=policy_kwargs, **sac_kwargs(cfg))
    fit(cfg, model, eval_env, run_name, wandb_run)


# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config_sac_fable5_sb3", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_random_seed(cfg.seed)

    # TF32 + cudnn autotuning back ON: the resnet50 NaN turned out to be the
    # aliased-encoder polyak corruption (fixed structurally), not a numerics
    # issue -- so take the Hopper throughput. Toggle off via perf.* if ever
    # bisecting numerics again.
    th.backends.cuda.matmul.allow_tf32 = cfg.perf.tf32
    th.backends.cudnn.allow_tf32 = cfg.perf.tf32
    th.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark

    {"state": run_state, "pixels": run_pixels, "pvr": run_pvr}[cfg.mode](cfg)


if __name__ == "__main__":
    main()
