"""
SAC training with an explicit, train_bc.py-style loop instead of SB3's
black-box model.learn(). Manually drives the exact two methods learn()
calls internally -- collect_rollouts() and train() -- chunked into
eval_freq-sized "epochs" with explicit eval/logging/checkpointing between
chunks, mirroring train_bc.py's for-epoch loop shape.

This is NOT the documented public SB3 API -- model.learn() + Callbacks is.
collect_rollouts()/train()/_setup_learn() are internal methods (verified
against SB3 2.9.0 source) whose signatures could change across versions.
Kept as a separate file specifically to compare loop structure against
train_bc.py and against train_sac.py's Callback-based version; train_sac.py
remains the primary script. Everything except the loop itself (env
builders, safety nets, hyperparameter wiring) is imported from train_sac.py
to keep the comparison isolated to just "how is training driven".

Usage: same Hydra overrides as train_sac.py.
    python train_sac_loop.py mode=pvr env=franka_kitchen embedding=resnet18
"""

import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from feature_extractor import PVRFeaturesExtractor
from train_sac import (
    NaNGuardCallback,
    guard_train,
    make_pixel_env,
    make_state_env,
    sac_kwargs,
)


def run_manual_loop(cfg: DictConfig, model: SAC, eval_env, save_dir: str, use_wandb: bool):
    """
    Reimplementation of OffPolicyAlgorithm.learn()'s inner loop (SAC
    inherits it unchanged), chunked by eval_freq instead of running to
    total_timesteps in one opaque call. One "epoch" here = one eval_freq
    -sized chunk of env steps, echoing train_bc.py's one-epoch-per-dataset
    -pass structure even though SAC has no dataset to epoch over.
    """
    guard_train(model)

    total_timesteps = cfg.algo.total_timesteps
    eval_freq = cfg.algo.eval_freq
    n_epochs = max(1, -(-total_timesteps // eval_freq))  # ceil division

    # _setup_learn() initializes replay_buffer/num_timesteps bookkeeping and
    # wraps our callback the same way learn() does -- collect_rollouts()
    # requires this to have run first.
    _, callback = model._setup_learn(
        total_timesteps, NaNGuardCallback(), reset_num_timesteps=True, tb_log_name="run"
    )
    callback.on_training_start(locals(), globals())

    print(f"\nTraining for {total_timesteps:,} steps  "
          f"(eval every {eval_freq:,} steps, {n_epochs} eval points)")
    best_eval_reward = -float("inf")
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        target_timesteps = min(epoch * eval_freq, total_timesteps)

        # Train: collect rollouts + gradient steps until this epoch's target.
        # This while-loop body is exactly what learn() runs internally.
        while model.num_timesteps < target_timesteps:
            rollout = model.collect_rollouts(
                model.env,
                train_freq=model.train_freq,
                action_noise=model.action_noise,
                callback=callback,
                learning_starts=model.learning_starts,
                replay_buffer=model.replay_buffer,
                log_interval=4,
            )
            if not rollout.continue_training:
                break
            if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
                gradient_steps = (
                    model.gradient_steps if model.gradient_steps >= 0 else rollout.episode_timesteps
                )
                if gradient_steps > 0:
                    model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)

        # Validate: SB3's own evaluate_policy() -- the same utility
        # EvalCallback uses internally, so eval semantics stay identical to
        # train_sac.py's version even though we're calling it by hand.
        eval_reward, eval_std = evaluate_policy(
            model, eval_env, n_eval_episodes=cfg.algo.n_eval_episodes, deterministic=True,
        )

        elapsed = time.time() - t0
        print(f"  step {model.num_timesteps:>8,}/{total_timesteps:,}"
              f"  eval_reward={eval_reward:8.1f} ± {eval_std:.1f}"
              f"  ({elapsed:.0f}s)", flush=True)

        if use_wandb:
            import wandb
            wandb.log(
                {"eval/mean_reward": eval_reward, "eval/std_reward": eval_std},
                step=model.num_timesteps,
            )

        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            if not cfg.no_save:
                model.save(os.path.join(save_dir, "best_model"))

        checkpoint_every_n_epochs = max(1, cfg.algo.checkpoint_freq // eval_freq)
        if not cfg.no_save and epoch % checkpoint_every_n_epochs == 0:
            model.save(os.path.join(save_dir, f"checkpoint_{model.num_timesteps}"))

    callback.on_training_end()

    if not cfg.no_save:
        model.save(os.path.join(save_dir, "final_model"))

    print(f"\nBest eval reward: {best_eval_reward:.1f}")
    print(f"Model saved to: {save_dir}")


def build_model(cfg: DictConfig, save_dir: str) -> tuple[SAC, "VecEnv"]:
    """Same model construction as train_sac.py's run_state/run_pixels/run_pvr,
    just returning the (unlearned) model + eval_env instead of calling
    model.learn() itself."""
    if cfg.mode == "state":
        env = make_state_env(cfg.env.id)
        eval_env = make_state_env(cfg.env.id)
        model = SAC("MlpPolicy", env, **sac_kwargs(cfg, save_dir))

    elif cfg.mode == "pixels":
        env = make_pixel_env(cfg.env.id)
        eval_env = make_pixel_env(cfg.env.id)
        model = SAC(
            "CnnPolicy", env, buffer_size=cfg.algo.buffer_size, **sac_kwargs(cfg, save_dir)
        )

    else:  # pvr
        env = make_pixel_env(cfg.env.id)
        eval_env = make_pixel_env(cfg.env.id)
        policy_kwargs = dict(
            features_extractor_class=PVRFeaturesExtractor,
            features_extractor_kwargs=dict(
                embedding_name=cfg.embedding.name,
                freeze=True,
                disable_cuda=(cfg.device == "cpu"),
                model_dir=cfg.model_dir,
            ),
            net_arch=[256, 256],
            normalize_images=False,
        )
        model = SAC(
            "CnnPolicy", env, policy_kwargs=policy_kwargs,
            buffer_size=cfg.algo.buffer_size, **sac_kwargs(cfg, save_dir),
        )

    return model, eval_env


@hydra.main(config_path="configs", config_name="config_sac", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark

    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"\nMode: {cfg.mode}  Device: {cfg.device}  Output dir: {save_dir}")

    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        from omegaconf import OmegaConf
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            # SAC.train() records actor/critic loss, ent_coef, etc. into SB3's
            # own logger regardless of manual vs learn()-driven calls, and
            # collect_rollouts() dumps them to tensorboard_log=save_dir on its
            # own internal schedule (episode-count-based, not synced to our
            # eval_freq epochs) -- sync_tensorboard mirrors those into wandb
            # the moment they're written, which is more robust than trying to
            # read model.logger.name_to_value ourselves (it gets cleared by
            # Logger.dump() on that same internal schedule, so polling it from
            # our loop would race and sometimes see stale/empty values). The
            # explicit wandb.log() call in run_manual_loop() below only needs
            # to cover eval/mean_reward and eval/std_reward -- values that
            # only our loop computes, with no SB3-internal equivalent.
            sync_tensorboard=True,
        )

    model, eval_env = build_model(cfg, save_dir)
    run_manual_loop(cfg, model, eval_env, save_dir, use_wandb)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
