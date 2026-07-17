"""
SAC (via Stable-Baselines3) training script, mirroring train_bc.py's
hydra-configured structure.

Five MODES, meant to be run in order so you isolate failures instead of
debugging SAC + pixels + a custom encoder all at once:

  state     Vanilla SAC on a state-based env (Pendulum-v1).
            Confirms SAC itself learns -- no vision involved.

  pixels    SAC on cfg.env using SB3's default CNN (NatureCNN), trained
            end-to-end from scratch. Confirms the pixel plumbing works:
            image dtype/normalization, replay buffer with image obs. This
            is also the "CNN trained from scratch" baseline for comparison
            against the pvr* modes below.

  pvr       Frozen cfg.embedding backbone, SLOW: the encoder lives inside
            PVRFeaturesExtractor as part of the SB3 policy, so it gets
            re-run on every gradient step's sampled minibatch even though
            its weights never change. Simple, but wasteful.

  pvr_fast  Frozen cfg.embedding backbone, FAST: the encoder lives outside
            the policy entirely (FrozenPVRVecWrapper), encoding each frame
            exactly once per env step; the replay buffer stores feature
            vectors, not pixels, and gradient steps never touch the
            encoder. Same result as pvr, much less compute -- this is the
            config you'll actually use to benchmark PVRs against each
            other. Cannot finetune (see FrozenPVRVecWrapper's docstring
            for why not, by construction, not as a missing feature).

  pvr_ft    Same as pvr, but the encoder trains jointly with the RL head
            (freeze=False) -- necessarily the "slow", in-policy mechanism,
            since finetuning requires gradients to flow through the
            encoder on every step.

Usage
-----
    python train_sac.py mode=state
    python train_sac.py mode=pixels env=dmc_cheetah
    python train_sac.py mode=pvr_fast env=dmc_cheetah embedding=resnet50
    python train_sac.py mode=pvr_fast embedding=mae_base algo.total_timesteps=500000
    python train_sac.py mode=pvr_ft embedding=resnet18
    python train_sac.py mode=pvr_fast algo.total_timesteps=300 algo.buffer_size=1000  # quick smoke test
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

from feature_extractor import FrozenPVRVecWrapper, LayerNormExtractor, PVRFeaturesExtractor
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

def maybe_init_wandb(cfg: DictConfig, run_name: str, save_dir: str):
    """
    Returns a wandb run if cfg.wandb.enabled, else None. Uses
    sync_tensorboard=True, which piggybacks on the tensorboard_log SAC
    writes to -- wandb patches the SummaryWriter, so every scalar SB3 (and
    EvalCallback) writes to tensorboard is mirrored to wandb automatically.
    Must be called BEFORE the SAC(...) model is constructed so the patch is
    in place before SB3 creates its writer.

    wandb.tensorboard.patch(root_logdir=save_dir) is called explicitly,
    before wandb.init(), so wandb.init(sync_tensorboard=True) sees
    wandb.patched["tensorboard"] already populated and skips its own
    auto-patch (verified against wandb 0.28.0 source -- no double-patch).
    Without this, wandb's patch defaults to root_logdir="" -> resolved as
    os.getcwd(), which never contains save_dir's absolute Hydra output
    path -- every SB3-created event-file writer then logs "Found log
    directory outside of given root_logdir, dropping given root_logdir".

    If cluster compute nodes have no outbound internet, set
    WANDB_MODE=offline in the job script -- wandb then writes locally with
    no network calls, and you `wandb sync <run_dir>` from the login node
    afterward.
    """
    if not cfg.wandb.enabled:
        return None
    import wandb
    from omegaconf import OmegaConf
    wandb.tensorboard.patch(root_logdir=save_dir)
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
    wandb_run = maybe_init_wandb(cfg, f"sac_{cfg.env.id}_state", save_dir)

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
    wandb_run = maybe_init_wandb(cfg, f"sac_{cfg.env.id}_defaultcnn", save_dir)

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
    """Handles mode in {pvr, pvr_fast, pvr_ft} -- see module docstring for
    what each one means. pvr and pvr_ft share the same in-policy plumbing
    (differing only in freeze=True/False); pvr_fast is the only genuinely
    different code path, since it can't be used for finetuning."""
    if cfg.mode == "pvr_fast":
        env = FrozenPVRVecWrapper(
            make_pixel_env(cfg.env.id), cfg.embedding.name,
            disable_cuda=(cfg.device == "cpu"), model_dir=cfg.model_dir,
            amp_bf16=cfg.perf.amp_bf16,
        )
        eval_env = FrozenPVRVecWrapper(
            make_pixel_env(cfg.env.id), cfg.embedding.name,
            disable_cuda=(cfg.device == "cpu"), model_dir=cfg.model_dir,
            amp_bf16=cfg.perf.amp_bf16,
        )
        wandb_run = maybe_init_wandb(cfg, f"sac_{cfg.env.id}_{cfg.embedding.name}_fast", save_dir)
        policy_kwargs = dict(
            features_extractor_class=LayerNormExtractor,
            net_arch=[256, 256],
        )
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            buffer_size=cfg.algo.buffer_size,
            **sac_kwargs(cfg, save_dir),
        )
    else:
        freeze = (cfg.mode == "pvr")  # "pvr" = frozen/slow, "pvr_ft" = finetune
        env = make_pixel_env(cfg.env.id)
        eval_env = make_pixel_env(cfg.env.id)
        wandb_run = maybe_init_wandb(
            cfg, f"sac_{cfg.env.id}_{cfg.embedding.name}{'' if freeze else '_ft'}", save_dir
        )
        policy_kwargs = dict(
            features_extractor_class=PVRFeaturesExtractor,
            features_extractor_kwargs=dict(
                embedding_name=cfg.embedding.name,
                freeze=freeze,
                disable_cuda=(cfg.device == "cpu"),
                model_dir=cfg.model_dir,
            ),
            net_arch=[256, 256],
            normalize_images=False,  # PVRFeaturesExtractor does its own /255 + ImageNet norm
            # SB3 defaults share_features_extractor to False: actor, critic,
            # and critic_target would each get their OWN independently
            # constructed PVRFeaturesExtractor (verified against SB3 2.9.0
            # source). Harmless when frozen (three copies of the same fixed
            # function), but for pvr_ft it means three SEPARATE encoders
            # drifting apart under their own network's gradients only --
            # not "one PVR finetuned by the agent". True either way: forces
            # actor+critic to share one instance (critic_target still gets
            # its own, hard-synced then polyak-updated toward it -- the
            # normal, correct SAC target-network mechanism, safe here since
            # PVRFeaturesExtractor always builds a fresh EmbeddingNet from a
            # string, never aliasing a live module).
            share_features_extractor=True,
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


MODES = {
    "state": run_state,
    "pixels": run_pixels,
    "pvr": run_pvr,
    "pvr_fast": run_pvr,
    "pvr_ft": run_pvr,
}


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
