"""
SAC (via Stable-Baselines3) training script, mirroring train_bc.py's
hydra-configured structure.

Three MODES, meant to be run in order so you isolate failures instead of
debugging SAC + pixels + a custom encoder all at once:

  state   Vanilla SAC on a state-based env (Pendulum-v1).
          Confirms SAC itself learns -- no vision involved.

  pixels  SAC on cfg.env using SB3's default CNN (NatureCNN).
          Confirms the pixel plumbing works: image dtype/normalization,
          replay buffer with image obs.

  pvr     Same env, but the CNN is replaced by cfg.embedding via
          PVRFeaturesExtractor. This is the config you'll actually use to
          benchmark PVRs against each other.

Usage
-----
    python train_sac.py mode=state
    python train_sac.py mode=pixels env=dmc_cheetah
    python train_sac.py mode=pvr env=dmc_cheetah embedding=resnet50
    python train_sac.py mode=pvr embedding=mae_base algo.total_timesteps=500000
    python train_sac.py mode=pvr algo.total_timesteps=300 algo.buffer_size=1000  # quick smoke test
"""

import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from feature_extractor import PVRFeaturesExtractor
from src.gym_wrappers import make_gym_env


# ------------------------------------------------------------------------------
# Envs
# ------------------------------------------------------------------------------

def make_state_env(env_id: str):
    # train_from_pixels=False -> make_gym_env's non-pixel branch, which
    # FlattenObservations whatever the env natively returns (proprioceptive
    # sensors for dm_control/FrankaKitchen/etc, not pixels).
    return make_vec_env(lambda: make_gym_env(train_from_pixels=False, id=env_id), n_envs=1)


def make_pixel_env(env_id: str):
    # make_gym_env's train_from_pixels path already applies
    # AddRenderObservation + ResizeObservation(84,84) + ChannelsFirstWrapper,
    # so the resulting obs are already (C, H, W) -- no VecTransposeImage needed.
    return make_vec_env(lambda: make_gym_env(train_from_pixels=True, id=env_id), n_envs=1)


# ------------------------------------------------------------------------------
# Safety nets (cheap, generically useful -- not backbone-specific)
# ------------------------------------------------------------------------------

class NaNGuardCallback(BaseCallback):
    """
    Raises immediately, with the actual cause, the first time a reward goes
    non-finite -- instead of letting the run continue until SB3's Normal(...)
    constructor rejects a NaN action mean several steps later deep in
    torch.distributions, with no context on whether the NaN originated from
    env/reward divergence or from the optimizer.

    Deliberately does NOT check policy parameters here: SB3's off-policy loop
    runs predict() -> env.step() -> callback.on_step() for a step, and only
    calls train() *after* that -- immediately followed by the next step's
    predict(), with no callback in between. See guard_train() below.
    """

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None and not np.isfinite(rewards).all():
            raise RuntimeError(
                f"Non-finite reward at step {self.num_timesteps}: {rewards} "
                "-- env/physics likely diverged, not a policy/encoder issue."
            )
        return True


def guard_train(model):
    """
    Wraps model.train() so a non-finite parameter is caught the instant a
    gradient update produces it, with the gradient-step count and parameter
    name -- the one window NaNGuardCallback structurally can't cover.
    """
    original_train = model.train

    def guarded_train(*args, **kwargs):
        original_train(*args, **kwargs)
        for name, param in model.policy.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(
                    f"Non-finite parameter '{name}' immediately after train() "
                    f"at step {model.num_timesteps} -- gradient update diverged "
                    "(rewards were finite, so this isn't an env/physics issue)."
                )

    model.train = guarded_train


# ------------------------------------------------------------------------------
# Wandb / callbacks
# ------------------------------------------------------------------------------

def maybe_init_wandb(cfg: DictConfig, run_name: str):
    """
    Returns a wandb run if cfg.wandb.enabled, else None. Uses
    sync_tensorboard=True, which piggybacks on the tensorboard_log SAC
    writes to -- wandb patches the SummaryWriter, so every scalar SB3 (and
    EvalCallback) writes to tensorboard is mirrored to wandb automatically.
    Must be called BEFORE the SAC(...) model is constructed so the patch is
    in place before SB3 creates its writer.

    If cluster compute nodes have no outbound internet, set
    WANDB_MODE=offline in the job script -- wandb then writes locally with
    no network calls, and you `wandb sync <run_dir>` from the login node
    afterward.
    """
    if not cfg.wandb.enabled:
        return None
    import wandb
    from omegaconf import OmegaConf
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name or run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,
        save_code=False,
    )


def build_callbacks(cfg: DictConfig, eval_env, save_dir: str, wandb_run=None):
    """
    EvalCallback: periodically runs the current policy on a held-out env and
    logs mean reward -- the "is it learning anything" signal while the job
    is running. Its log_path writes small .npz reward arrays regardless of
    cfg.no_save.

    CheckpointCallback: saves the full model every checkpoint_freq steps.
    Skipped entirely under cfg.no_save.

    WandbCallback: also uploads gradients/model info to wandb directly, if
    enabled. Its own model_save_path is likewise skipped under cfg.no_save.
    """
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None if cfg.no_save else os.path.join(save_dir, "best"),
        log_path=os.path.join(save_dir, "eval"),
        eval_freq=cfg.algo.eval_freq,
        n_eval_episodes=cfg.algo.n_eval_episodes,
        deterministic=True,
    )
    callbacks = [eval_callback, NaNGuardCallback()]
    if not cfg.no_save:
        callbacks.append(CheckpointCallback(
            save_freq=cfg.algo.checkpoint_freq,
            save_path=save_dir,
            name_prefix="sac",
        ))
    if wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(
            model_save_path=None if cfg.no_save else os.path.join(save_dir, "wandb_models"),
            verbose=2,
        ))
    return callbacks


def sac_kwargs(cfg: DictConfig, save_dir: str) -> dict:
    """Hyperparameters shared by all three modes, read from cfg.algo."""
    return dict(
        verbose=1,
        device=cfg.device,
        tensorboard_log=save_dir,
        learning_rate=cfg.algo.learning_rate,
        batch_size=cfg.algo.batch_size,
        gamma=cfg.algo.gamma,
        tau=cfg.algo.tau,
        train_freq=cfg.algo.train_freq,
        gradient_steps=cfg.algo.gradient_steps,
        target_update_interval=cfg.algo.target_update_interval,
        ent_coef=cfg.algo.ent_coef,
        # Clamp so small smoke-test runs (algo.total_timesteps overridden low)
        # still exceed learning_starts and actually exercise training.
        learning_starts=min(cfg.algo.learning_starts, cfg.algo.total_timesteps),
    )


# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

def run_state(cfg: DictConfig, save_dir: str):
    env = make_state_env(cfg.env.id)
    eval_env = make_state_env(cfg.env.id)
    wandb_run = maybe_init_wandb(cfg, f"sac_{cfg.env.id}_state")

    model = SAC("MlpPolicy", env, **sac_kwargs(cfg, save_dir))
    guard_train(model)
    model.learn(
        total_timesteps=cfg.algo.total_timesteps,
        callback=build_callbacks(cfg, eval_env, save_dir, wandb_run),
    )
    if not cfg.no_save:
        model.save(os.path.join(save_dir, "final_model"))
    if wandb_run is not None:
        wandb_run.finish()


def run_pixels(cfg: DictConfig, save_dir: str):
    env = make_pixel_env(cfg.env.id)
    eval_env = make_pixel_env(cfg.env.id)
    wandb_run = maybe_init_wandb(cfg, f"sac_{cfg.env.id}_defaultcnn")

    model = SAC(
        "CnnPolicy",  # SB3's default NatureCNN
        env,
        buffer_size=cfg.algo.buffer_size,
        **sac_kwargs(cfg, save_dir),
    )
    guard_train(model)
    model.learn(
        total_timesteps=cfg.algo.total_timesteps,
        callback=build_callbacks(cfg, eval_env, save_dir, wandb_run),
    )
    if not cfg.no_save:
        model.save(os.path.join(save_dir, "final_model"))
    if wandb_run is not None:
        wandb_run.finish()


def run_pvr(cfg: DictConfig, save_dir: str):
    env = make_pixel_env(cfg.env.id)
    eval_env = make_pixel_env(cfg.env.id)
    wandb_run = maybe_init_wandb(cfg, f"sac_{cfg.env.id}_{cfg.embedding.name}")

    policy_kwargs = dict(
        features_extractor_class=PVRFeaturesExtractor,
        features_extractor_kwargs=dict(
            embedding_name=cfg.embedding.name,
            freeze=True,  # standard PVR-eval protocol: probe, don't finetune
            disable_cuda=(cfg.device == "cpu"),
        ),
        net_arch=[256, 256],
        normalize_images=False,  # PVRFeaturesExtractor does its own /255 + ImageNet norm
    )

    model = SAC(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=cfg.algo.buffer_size,
        **sac_kwargs(cfg, save_dir),
    )
    guard_train(model)
    model.learn(
        total_timesteps=cfg.algo.total_timesteps,
        callback=build_callbacks(cfg, eval_env, save_dir, wandb_run),
    )
    if not cfg.no_save:
        model.save(os.path.join(save_dir, "final_model"))
    if wandb_run is not None:
        wandb_run.finish()


MODES = {"state": run_state, "pixels": run_pixels, "pvr": run_pvr}


@hydra.main(config_path="configs", config_name="config_sac", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # The resnet50 NaN (see project memory) turned out to be an aliased-
    # encoder polyak-update corruption, fixed structurally in
    # PVRFeaturesExtractor -- not a numerics issue. So these default to
    # PyTorch's Hopper-friendly settings; toggle off via perf.* only if ever
    # bisecting a numerics issue again.
    torch.backends.cuda.matmul.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark

    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"\nMode: {cfg.mode}  Device: {cfg.device}  Output dir: {save_dir}")

    MODES[cfg.mode](cfg, save_dir)


if __name__ == "__main__":
    main()
