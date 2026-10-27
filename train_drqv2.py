"""
DrQ-v2 (Yarats et al. 2021) on this repo's encoders and environments.

    python train_drqv2.py env=dmc_walker embedding=resnet18 env.image_size=84 \
        encoder.grads=critic augmentation=random_shift
    python train_drqv2.py +experiment=pieg env=dmc_walker \
        augmentation.strong.params.image_dir=...

Same setup as train_sac.py -- encoder, augmentation, buffer_obs, encoder.*
options, wandb, checkpoints (src/rl/common.py) -- with DrQ-v2's agent
(src/rl/drqv2.py, configs/algo/drqv2.yaml):

  - deterministic actor with scheduled exploration noise
    (algo.stddev_schedule) instead of SAC's entropy-tuned policy; uniform
    random actions for the first algo.learning_starts steps
  - critic target from n-step returns (algo.nstep) with a smoothed target
    action, noise clipped at algo.stddev_clip
  - one update every algo.update_every_steps agent steps; target critic
    soft-updated after each (algo.tau)
  - algo.action_repeat env frames per agent step. total_timesteps,
    learning_starts, the schedule and all frequencies count agent steps;
    charts/frames is env frames, the x-axis DrQ-v2 papers report.

DrQ-v2 trains its encoder from the critic loss only and augments with
random_shift: that's encoder.grads=critic augmentation=random_shift here.
Defaults (encoder.grads=none, augmentation=none) give a frozen encoder,
like train_sac.py. DrQ-v2's own conv encoder isn't included; any
embedding works, including a trainable head (embedding.head=linear).

Logged metrics reuse train_sac.py's keys where they mean the same thing
(train/qf*_loss, train/qf*_values, train/target_q_mean, train/actor_loss,
grad norms, encoder metrics, eval/*); DrQ-v2 adds train/actor_stddev.
"""

import os

# Must be set before dm_control's C extension loads; EGL = GPU offscreen
# rendering, the fast option on a GPU node. Override in the shell if needed.
os.environ.setdefault("MUJOCO_GL", "egl")

import copy
import random
import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

from src.encoder_config import (
    embedding_kwargs,
    encoder_settings,
    resolve_buffer_obs,
    warn_unused_encoder_overrides,
)
from src.losses import total_aux_loss
from src.rl.buffer import ReplayBuffer
from src.rl.common import (
    EncoderOptim,
    Encoding,
    buffer_options,
    buffer_status,
    build_encoder_stack,
    encode_replay_batch,
    encoder_log,
    encoder_record,
    init_wandb,
    log_episode,
    log_eval,
)
from src.rl.drqv2 import Actor, Critic, stddev_schedule
from src.sac_utils import _success_fn_for, evaluate, make_env
from src.views import CollapseMonitor


@hydra.main(config_path="configs", config_name="config_drqv2", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ── Reproducibility / perf ────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Encoder / augmentation / buffer settings ──────────────────────────────
    enc = encoder_settings(cfg, valid_grads=("none", "critic", "actor", "both"))
    pixels_in_buffer, buffer_reason = resolve_buffer_obs(cfg, enc)
    warn_unused_encoder_overrides(enc)
    emb_kwargs = embedding_kwargs(cfg)
    routing = enc.grads
    encoder_trains = enc.trains
    save_encoder = encoder_trains or enc.bn_adapt

    # ── Encoder, wandb, envs (src/rl/common.py) ───────────────────────────────
    stack = build_encoder_stack(cfg, enc, emb_kwargs, device, pixels_in_buffer, buffer_reason)
    embedding_net, pixels_in_buffer = stack.net, stack.pixels_in_buffer
    wandb = init_wandb(cfg, "drqv2", enc, stack)

    wrap = not pixels_in_buffer
    env = make_env(cfg, embedding_net, wrap_encoder=wrap)
    eval_env = make_env(cfg, embedding_net, wrap_encoder=wrap)
    assert isinstance(env.action_space, gym.spaces.Box), "continuous actions only"
    env.action_space.seed(cfg.seed)

    action_dim = int(np.prod(env.action_space.shape))
    obs_shape = env.observation_space.shape
    if pixels_in_buffer:
        feat_dim = int(embedding_net.out_size)
        buffer_dtype = np.uint8
    else:
        feat_dim = int(np.prod(obs_shape))
        buffer_dtype = np.float32
    E = Encoding(stack, device, cfg.perf.amp_bf16)

    # The agent acts in [-1, 1]; the env gets the rescaled action. The
    # buffer stores the agent's action.
    low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
    to_env = lambda a: low + (a + 1.0) * 0.5 * (high - low)

    # ── Networks / optimizers ─────────────────────────────────────────────────
    a = cfg.algo
    for key in ("log_frequency", "eval_frequency"):
        if a[key] % a.update_every_steps:
            raise ValueError(f"algo.{key} ({a[key]}) must be a multiple of "
                             f"algo.update_every_steps ({a.update_every_steps}): "
                             "logging and evaluation run on update steps")
    actor = Actor(feat_dim, action_dim, a.feature_dim, a.hidden_dim).to(device)
    critic = Critic(feat_dim, action_dim, a.feature_dim, a.hidden_dim).to(device)
    critic_target = copy.deepcopy(critic).requires_grad_(False)
    actor_optimizer = optim.Adam(actor.parameters(), lr=a.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=a.lr)
    eo = EncoderOptim(cfg, enc, embedding_net, device)
    # null: no clipping (DrQ-v2's default); the norm is still logged.
    grad_clip = float("inf") if a.get("grad_clip") is None else a.grad_clip

    print(f"\nBuffer stores: {stack.buffer_reason}")
    rb = ReplayBuffer(
        a.buffer_size, obs_shape, action_dim, device, obs_dtype=buffer_dtype,
        nstep=int(a.nstep), gamma=a.gamma, **buffer_options(a, pixels_in_buffer),
    )
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    best_eval_return = -float("inf")
    action_repeat = int(a.get("action_repeat", 1))

    # ── Training loop ─────────────────────────────────────────────────────────
    print(
        f"\nTraining for {a.total_timesteps:,} agent steps "
        f"({a.total_timesteps * action_repeat:,} frames; "
        f"learning starts at {a.learning_starts:,})"
    )
    obs, _ = env.reset(seed=cfg.seed)
    E.check_augmentations(obs)
    t0 = time.time()
    enc_gnorm_critic = float("nan")
    enc_gnorm_actor = float("nan")
    collapse = CollapseMonitor()

    for global_step in range(1, a.total_timesteps + 1):
        std = stddev_schedule(a.stddev_schedule, global_step)

        # ---- act --------------------------------------------------------------
        if global_step <= a.learning_starts:
            action = np.random.uniform(-1.0, 1.0, action_dim).astype(np.float32)
        else:
            if pixels_in_buffer:
                obs_t = E.encode_single(obs)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(obs_t, std).sample(clip=None).squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(
            to_env(torch.as_tensor(action, device=device)).cpu().numpy()
        )
        rb.add(obs, next_obs, action, reward, terminated, truncated)

        if terminated or truncated:
            log_episode(info, global_step, wandb)
            obs, _ = env.reset()
        else:
            obs = next_obs

        if global_step <= a.learning_starts or global_step % a.update_every_steps != 0:
            continue

        # ---- sample + featurize -----------------------------------------------
        b_obs, b_next_obs, b_actions, b_rewards, b_discounts = rb.sample(a.batch_size)

        route_critic = routing in ("critic", "both")
        route_actor = routing in ("actor", "both")

        if enc.bn_adapt:
            embedding_net.set_bn_train(True)

        feats_strong = None
        if pixels_in_buffer:
            views, next_feats, feats_strong = encode_replay_batch(
                E, stack, enc, eo, b_obs, b_next_obs, task_grad=route_critic
            )
            feats = views.feats_task
            if encoder_trains and global_step % a.log_frequency == 0:
                # Counted in updates, like train_sac.py (one per step there).
                collapse.update(feats, step=(global_step - a.learning_starts) // a.update_every_steps,
                                where=f"step {global_step:,}")
        else:
            feats, next_feats = b_obs, b_next_obs

        # ---- critic update ------------------------------------------------------
        with torch.no_grad():
            next_actions = actor(next_feats, std).sample(clip=a.stddev_clip)
            tq1, tq2 = critic_target(next_feats, next_actions)
            next_q_value = b_rewards + b_discounts * torch.min(tq1, tq2).view(-1)

        q1, q2 = critic(feats, b_actions)
        qf1_a_values, qf2_a_values = q1.view(-1), q2.view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        critic_loss = qf_loss
        qf_loss_strong = None
        if feats_strong is not None:
            qs1, qs2 = critic(feats_strong, b_actions)
            qf_loss_strong = (F.mse_loss(qs1.view(-1), next_q_value)
                              + F.mse_loss(qs2.view(-1), next_q_value))
            w = enc.strong_weight
            critic_loss = (1 - w) * qf_loss + w * qf_loss_strong

        aux_total, aux_values = None, {}
        if eo.aux_terms:
            aux_total, aux_values = total_aux_loss(eo.aux_terms, views.aux)

        enc_step_now = route_critic or bool(eo.aux_terms)
        enc_gnorm_critic = float("nan")
        enc_gnorm_actor = float("nan")
        if enc_step_now:
            eo.zero_grad()
        critic_optimizer.zero_grad(set_to_none=True)
        (critic_loss if aux_total is None else critic_loss + aux_total).backward()
        q_gnorm = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip))
        critic_optimizer.step()
        if enc_step_now:
            enc_gnorm_critic = eo.step()

        # ---- actor update -------------------------------------------------------
        if pixels_in_buffer and route_actor:
            # Re-encode with grad: the critic phase's graph is consumed and
            # the encoder may have just stepped. Same pixels and draw.
            feats_pi = E.featurize(views.x_task, grad=True)
        else:
            feats_pi = feats.detach()
        dist = actor(feats_pi, std)
        pi = dist.sample(clip=a.stddev_clip)
        aq1, aq2 = critic(feats_pi, pi)
        actor_loss = -torch.min(aq1, aq2).mean()

        if route_actor:
            eo.zero_grad()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gnorm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=grad_clip))
        actor_optimizer.step()
        if route_actor:
            enc_gnorm_actor = eo.step()
        # actor_loss.backward() also leaves grads on the critic; its
        # zero_grad clears them before the next critic update.

        if enc.bn_adapt:
            embedding_net.set_bn_train(False)

        # ---- target update ------------------------------------------------------
        with torch.no_grad():
            for p, tp in zip(critic.parameters(), critic_target.parameters()):
                tp.lerp_(p, a.tau)

        # ---- logging --------------------------------------------------------------
        # Logging and evaluation run on update steps (frequencies checked
        # against update_every_steps above).
        if global_step % a.log_frequency == 0:
            sps = int(global_step / (time.time() - t0))
            print(
                f"  step {global_step:>8,}  qf_loss={qf_loss.item():8.3f}"
                f"  q1={qf1_a_values.mean().item():7.2f}"
                f"  actor_loss={actor_loss.item():8.3f}"
                f"  std={std:.3f}  SPS={sps}"
                f"  gnorm(q/actor)={q_gnorm:.2f}/{actor_gnorm:.2f}"
                + buffer_status(rb)[0]
                + (
                    f"  enc_gnorm(c/a)={enc_gnorm_critic:.2f}/{enc_gnorm_actor:.2f}"
                    f"  emb_std_ratio={collapse.ratio:.2f}"
                    if encoder_trains
                    else ""
                )
            )
            if wandb is not None:
                log = {
                    "train/qf1_loss": qf1_loss.item(),
                    "train/qf2_loss": qf2_loss.item(),
                    "train/qf_loss": qf_loss.item() / 2.0,
                    "train/qf1_values": qf1_a_values.mean().item(),
                    "train/qf2_values": qf2_a_values.mean().item(),
                    "train/target_q_mean": next_q_value.mean().item(),
                    "train/actor_loss": actor_loss.item(),
                    "train/actor_stddev": std,
                    "train/q_grad_norm": q_gnorm,
                    "train/actor_grad_norm": actor_gnorm,
                    "charts/SPS": sps,
                    **buffer_status(rb)[1],
                    "charts/frames": global_step * action_repeat,
                    "global_step": global_step,
                }
                if qf_loss_strong is not None:
                    log["train/qf_loss_strong"] = qf_loss_strong.item() / 2.0
                log.update(encoder_log(enc, stack, eo, collapse, enc_gnorm_critic,
                                       enc_gnorm_actor, aux_total, aux_values))
                wandb.log(log, step=global_step)

        # ---- periodic evaluation ----------------------------------------------------
        if global_step % a.eval_frequency == 0:
            stats = evaluate(
                lambda o: to_env(actor(o, std).mean),
                eval_env,
                a.n_episodes_test,
                device,
                encode_fn=E.encode_single if pixels_in_buffer else None,
                success_fn=_success_fn_for(cfg.env.id),
            )
            log_eval(stats, global_step, wandb)
            if stats["return_mean"] > best_eval_return:
                best_eval_return = stats["return_mean"]
                if cfg.save.enabled:
                    torch.save(actor.state_dict(), os.path.join(save_dir, "best_actor.pt"))
                    if save_encoder:
                        torch.save(encoder_record(cfg, embedding_net),
                                   os.path.join(save_dir, "best_encoder.pt"))

    # ── Final save ────────────────────────────────────────────────────────────
    if cfg.save.enabled:
        ckpt = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if save_encoder:
            ckpt.update(encoder_record(cfg, embedding_net))
        torch.save(ckpt, os.path.join(save_dir, "final_checkpoint.pt"))
        print(f"\nSaved to: {save_dir}")
    print(f"Best eval return: {best_eval_return:.1f}")

    env.close()
    eval_env.close()
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
