"""
Soft Actor-Critic on pretrained visual representations (PVRs), written as an
explicit training loop in the style of train_bc.py.

The SAC update math is adapted from CleanRL's sac_continuous_action.py
(https://github.com/vwxyzjn/cleanrl, MIT license), which is benchmarked
against reference results -- the tanh-Gaussian log-prob correction, twin-Q
target, and automatic entropy tuning are kept verbatim. Nothing from CleanRL
needs to be installed; this file is self-contained.

Three independent settings decide what happens to the encoder
(validated up front by src/encoder_config.py):

  encoder.grads     which RL losses backprop into the encoder (below)
  encoder.aux       auxiliary encoder losses (src/losses.py)
  augmentation      configs/augmentation/*.yaml; applied to replay-batch
                    encodes only, never to rollouts or evaluation. Its
                    `stage` decides who sees the augmented view
                    (src/views.py): rl = actor/critic, aux = aux losses
                    only, both = both. One draw per update, shared by the
                    critic and actor phases.

The encoder trains iff grads != none or aux is non-empty. What the replay
buffer stores follows from these (buffer_obs=auto):

  FEATURES (frozen encoder, no augmentation -- the standard PVR-eval protocol)
      The frozen encoder lives in an env wrapper (FrozenEncoderWrapper): each
      frame is encoded exactly once at env-step time, the replay buffer
      stores feature vectors, and gradient updates never touch the encoder.

  PIXELS (encoder trains, or augmentation is on)
      Pixels are stored in the replay buffer and every sampled minibatch is
      re-encoded -- necessarily slower (that cost is the definition of
      finetuning, not a defect). buffer_obs=pixels forces this path with a
      frozen encoder and no augmentation ("slow-frozen"): the same features
      as the fast path, useful as a control. Gradient routing into the
      encoder is one visible enum:

        encoder.grads:
          none    No RL gradient reaches the encoder (it may still train on
                  encoder.aux).
          critic  DrQ-v2 recipe: critic loss trains the encoder, actor takes
                  detached features. The literature's default choice.
          actor   Actor loss trains the encoder (both gradient paths of the
                  actor loss -- through the policy input AND through the Q
                  evaluation). NOTE: this is the FULL actor-loss gradient;
                  SB3's share_features_extractor=True routes only the
                  policy-input path (verified empirically on SB3 2.9.0).
                  TODO: to replicate SB3's variant exactly, detach the Q-eval
                  input (see the marked line in the actor update).
          both    critic + actor gradients. NOTE: applied as SEQUENTIAL
                  Adam steps (one critic-phase step every update, plus
                  policy_frequency actor-phase steps on firing steps, each
                  re-encoding), not as one summed-gradient step -- summing
                  would require evaluating the actor loss against
                  pre-critic-update Q networks, changing SAC's ordering.
                  Same for encoder.aux, which always rides the critic-phase
                  cadence regardless of routing (so actor+l2sp alternates
                  pure-aux and pure-actor steps). With Adam these dynamics
                  differ from a combined step; compare routings via
                  train/encoder_drift_l2, which is routing-agnostic.

      Independently of routing, auxiliary losses can train the encoder
      (encoder.aux): 'l2sp' anchors weights to their pretrained values
      (||theta - theta_0||^2, Xuhong et al. 2018) -- adapt while penalizing
      drift. aux with grads=none = adaptation without RL gradients at all,
      which needs an aux loss that reads observations (e.g. aug_consistency):
      l2sp alone has zero gradient at the pretrained weights and is rejected.

Design notes:
  - Actor and critics each apply their own LayerNorm to input features --
    only when the input IS encoder features (embedding set). Raw state
    vectors skip it: per-sample LN on a ~17-dim state aliases states that
    differ by uniform shift/scale and destroys magnitude information.
  - Bootstrapping is masked on `terminated` only, so truncation-only envs
    (dm_control, 1000-step episodes) bootstrap through the time limit, from
    the episode's real final observation (src/rl/buffer.py). algo.nstep
    gives n-step targets; algo.action_repeat repeats each action.
  - No target encoder: target-Q features come from the live encoder under
    no_grad (DrQ-v2 convention -- target networks exist for the Q heads only).
  - Encoder stays in eval() mode even while its weights train: pretrained
    ResNets carry BatchNorm, and letting BN running stats update on RL frames
    is a known silent-corruption source (and the failure mode behind this
    project's original resnet50 NaN). Frozen BN statistics, trainable BN
    affine params -- the standard finetuning choice. encoder.bn=adapt
    opts into updating them (PIE-G).
  - Pixel buffer memory: uint8 pixels cost buffer_size * H * W * 3 bytes
    (next_obs is derived from the following slot, not stored). At 224px
    that is ~15 GB per 100k -- override env.image_size (e.g. 112 or 84) or
    shrink algo.buffer_size; EmbeddingNet's transforms resize internally
    either way. The script prints the estimate at startup.

Usage:
    python train_sac.py                                  # frozen fast path
    python train_sac.py encoder.grads=critic env.image_size=112 embedding=resnet18
    python train_sac.py augmentation=random_shift env.image_size=112   # frozen + DrQ shift
    python train_sac.py buffer_obs=pixels env.image_size=112           # slow-frozen control
    python train_sac.py augmentation=random_shift augmentation.stage=aux \
        'encoder.aux=[{name:aug_consistency,weight:0.1},{name:l2sp,weight:1.0}]' \
        env.image_size=112                                      # aux-only adaptation
    python train_sac.py env=pendulum embedding=none      # state-based sanity check
"""


import os

# Must be set before dm_control's C extension loads; EGL = GPU offscreen
# rendering, the fast option on a GPU node. Override in the shell if needed.
os.environ.setdefault("MUJOCO_GL", "egl")

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
from src.sac_utils import Actor, SoftQNetwork, _success_fn_for, evaluate, make_env
from src.views import CollapseMonitor

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


@hydra.main(config_path="configs", config_name="config_sac", version_base=None)
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
    # Validated before anything expensive is built (src/encoder_config.py).
    enc = encoder_settings(cfg, valid_grads=("none", "critic", "actor", "both"))
    pixels_in_buffer, buffer_reason = resolve_buffer_obs(cfg, enc)
    warn_unused_encoder_overrides(enc)
    emb_kwargs = embedding_kwargs(cfg)
    routing = enc.grads
    encoder_trains = enc.trains
    # Adapting BatchNorm statistics change the encoder without training it,
    # so it's saved then too.
    save_encoder = encoder_trains or enc.bn_adapt

    # ── Encoder, wandb, envs (src/rl/common.py) ───────────────────────────────
    stack = build_encoder_stack(cfg, enc, emb_kwargs, device, pixels_in_buffer, buffer_reason)
    embedding_net, pixels_in_buffer = stack.net, stack.pixels_in_buffer
    emb_name = cfg.embedding.get("name", None)
    wandb = init_wandb(cfg, "sac", enc, stack)

    # Features path: env wrapper encodes, buffer stores features.
    # Pixels path: env emits pixels, encoding happens in the loop.
    wrap = not pixels_in_buffer
    env = make_env(cfg, embedding_net, wrap_encoder=wrap)
    eval_env = make_env(cfg, embedding_net, wrap_encoder=wrap)
    assert isinstance(env.action_space, gym.spaces.Box), "continuous actions only"
    env.action_space.seed(cfg.seed)

    action_dim = int(np.prod(env.action_space.shape))
    obs_shape = env.observation_space.shape   # (H, W, C) uint8, or (feat_dim,) / state
    if pixels_in_buffer:
        feat_dim = int(embedding_net.out_size)
        buffer_dtype = np.uint8
    else:
        feat_dim = int(np.prod(obs_shape))
        buffer_dtype = np.float32
    E = Encoding(stack, device, cfg.perf.amp_bf16)

    # ── Networks / optimizers ─────────────────────────────────────────────────
    hidden = list(cfg.algo.net_arch)
    # Input LayerNorm only when obs are encoder features; raw state vectors
    # go in unnormalized (matches SB3's MlpPolicy -- see SoftQNetwork docs).
    # cfg.obs_norm overrides the auto default (null) -- set obs_norm=false
    # on a pixel/PVR run to ablate whether LayerNorm helps there too.
    obs_norm = cfg.get("obs_norm", None)
    if obs_norm is None:
        obs_norm = emb_name is not None
    actor = Actor(feat_dim, action_dim, hidden, env.action_space, obs_norm).to(device)
    qf1 = SoftQNetwork(feat_dim, action_dim, hidden, obs_norm).to(device)
    qf2 = SoftQNetwork(feat_dim, action_dim, hidden, obs_norm).to(device)
    qf1_target = SoftQNetwork(feat_dim, action_dim, hidden, obs_norm).to(device)
    qf2_target = SoftQNetwork(feat_dim, action_dim, hidden, obs_norm).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=cfg.algo.q_lr, betas=tuple(cfg.algo.q_betas),
    )
    actor_optimizer = optim.Adam(
        actor.parameters(), lr=cfg.algo.policy_lr, betas=tuple(cfg.algo.policy_betas)
    )
    eo = EncoderOptim(cfg, enc, embedding_net, device)

    # automatic entropy tuning
    if cfg.algo.autotune_alpha:
        target_entropy = -float(action_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam(
            [log_alpha], lr=cfg.algo.alpha_lr, betas=tuple(cfg.algo.alpha_betas)
        )
    else:
        alpha = cfg.algo.alpha

    print(f"\nBuffer stores: {stack.buffer_reason}")
    rb = ReplayBuffer(
        cfg.algo.buffer_size, obs_shape, action_dim, device, obs_dtype=buffer_dtype,
        nstep=int(cfg.algo.get("nstep", 1)), gamma=cfg.algo.gamma, **buffer_options(cfg.algo, pixels_in_buffer),
    )
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    best_eval_return = -float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(
        f"\nTraining for {cfg.algo.total_timesteps:,} steps "
        f"(learning starts at {cfg.algo.learning_starts:,})"
    )
    obs, _ = env.reset(seed=cfg.seed)
    E.check_augmentations(obs)
    t0 = time.time()
    # Per-phase encoder grad norms: each written ONLY by its own phase and
    # reset when that phase doesn't step this global_step -- so `both` mode
    # logs both norms instead of the actor norm shadowing the critic one,
    # and `actor` mode logs NaN (not a stale value) on non-firing steps.
    enc_gnorm_critic = float("nan")
    enc_gnorm_actor = float("nan")
    collapse = CollapseMonitor()   # training encoders only

    for global_step in range(1, cfg.algo.total_timesteps + 1):
        # ---- act --------------------------------------------------------------
        if global_step <= cfg.algo.learning_starts:
            action = env.action_space.sample()
        else:
            if pixels_in_buffer:
                obs_t = E.encode_single(obs)
            else:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
            with torch.no_grad():
                action_t, _, _ = actor.get_action(obs_t)
            action = action_t.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)

        # The buffer keeps the episode's real final observation, and its
        # discounts mask only termination: on truncation (time limit) the
        # value of next_obs is still real and is bootstrapped.
        rb.add(obs, next_obs, action, reward, terminated, truncated)

        if terminated or truncated:
            log_episode(info, global_step, wandb)
            obs, _ = env.reset()
        else:
            obs = next_obs

        if global_step <= cfg.algo.learning_starts:
            continue

        # ---- sample + featurize -----------------------------------------------
        b_obs, b_next_obs, b_actions, b_rewards, b_discounts = rb.sample(
            cfg.algo.batch_size
        )

        route_critic = routing in ("critic", "both")
        route_actor = routing in ("actor", "both")

        if enc.bn_adapt:
            # BatchNorm normalises with each replay batch's own statistics
            # and updates its running ones for every encode of this update;
            # acting and evaluation (after the update) use the running ones.
            embedding_net.set_bn_train(True)

        feats_strong = None
        if pixels_in_buffer:
            views, next_feats, feats_strong = encode_replay_batch(
                E, stack, enc, eo, b_obs, b_next_obs, task_grad=route_critic
            )
            feats = views.feats_task
            # Every log step (each measurement is a GPU sync); the baseline is
            # set after CollapseMonitor's warmup, counted in updates.
            if encoder_trains and global_step % cfg.algo.log_frequency == 0:
                collapse.update(feats, step=global_step - cfg.algo.learning_starts,
                                where=f"step {global_step:,}")
        else:
            feats, next_feats = b_obs, b_next_obs

        # ---- critic update ------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_pi, _ = actor.get_action(next_feats)
            qf1_next = qf1_target(next_feats, next_actions)
            qf2_next = qf2_target(next_feats, next_actions)
            min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
            next_q_value = b_rewards + b_discounts * min_qf_next.view(-1)

        qf1_a_values = qf1(feats, b_actions).view(-1)
        qf2_a_values = qf2(feats, b_actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        # The loss the critic steps on. qf1/qf2/qf_loss above stay the task
        # view's, so their logged metrics keep their meaning.
        critic_loss = qf_loss
        qf_loss_strong = None
        if feats_strong is not None:
            qf_loss_strong = (
                F.mse_loss(qf1(feats_strong, b_actions).view(-1), next_q_value)
                + F.mse_loss(qf2(feats_strong, b_actions).view(-1), next_q_value)
            )
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
        q_optimizer.zero_grad()
        # One backward for both: the critic and an aux loss can share an
        # encoder graph (src/views.py), which a second backward couldn't
        # traverse. Aux losses don't touch the Q networks, so the critic's
        # gradients are unchanged; the encoder's are the sum, as before.
        (critic_loss if aux_total is None else critic_loss + aux_total).backward()
        q_gnorm = float(
            torch.nn.utils.clip_grad_norm_(
                list(qf1.parameters()) + list(qf2.parameters()),
                max_norm=cfg.algo.grad_clip,
            )
        )
        q_optimizer.step()
        if enc_step_now:
            enc_gnorm_critic = eo.step()

        # ---- actor (+ alpha) update, delayed ------------------------------------
        if global_step % cfg.algo.policy_frequency == 0:
            for _ in range(cfg.algo.policy_frequency):
                if pixels_in_buffer:
                    if route_actor:
                        # Re-encode with grad each inner iteration: the
                        # encoder may have just been updated, and the critic
                        # phase's graph is already consumed. Same pixels (and
                        # augmentation draw) the critic saw.
                        feats_pi = E.featurize(views.x_task, grad=True)
                    else:
                        # Reuse critic-phase features (detached). If the
                        # encoder stepped this iteration these are stale by
                        # exactly one encoder update -- negligible at
                        # encoder.lr, and it saves a full encoder forward.
                        feats_pi = feats.detach()
                else:
                    feats_pi = feats

                pi, log_pi, _ = actor.get_action(feats_pi)
                # FULL actor-loss gradient: encoder grads (when route_actor)
                # flow through BOTH the policy input and the Q evaluation.
                # TODO(SB3-mirror): to replicate SB3's share=True variant
                # (policy-input path only), evaluate the Qs on
                # feats_pi.detach() instead of feats_pi.
                qf1_pi = qf1(feats_pi, pi)
                qf2_pi = qf2(feats_pi, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = (alpha * log_pi - min_qf_pi).mean()

                if route_actor:
                    eo.zero_grad()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_gnorm = float(
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), max_norm=cfg.algo.grad_clip
                    )
                )
                actor_optimizer.step()
                if route_actor:
                    enc_gnorm_actor = eo.step()
                # Note: actor_loss.backward() also deposits gradients on
                # qf1/qf2 params; q_optimizer.zero_grad() clears them before
                # the next critic update (same ordering CleanRL relies on).

                if cfg.algo.autotune_alpha:
                    with torch.no_grad():
                        _, log_pi_a, _ = actor.get_action(
                            feats_pi.detach() if feats_pi.requires_grad else feats_pi
                        )
                    alpha_loss = (-log_alpha.exp() * (log_pi_a + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        if enc.bn_adapt:
            embedding_net.set_bn_train(False)   # acting/eval: running statistics

        # ---- polyak target update -----------------------------------------------
        if global_step % cfg.algo.target_network_frequency == 0:
            tau = cfg.algo.tau
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

        # ---- logging --------------------------------------------------------------
        if global_step % cfg.algo.log_frequency == 0:
            sps = int(global_step / (time.time() - t0))
            print(
                f"  step {global_step:>8,}  qf_loss={qf_loss.item():8.3f}"
                f"  q1={qf1_a_values.mean().item():7.2f}"
                f"  actor_loss={actor_loss.item():8.3f}"
                f"  alpha={alpha:.3f}  SPS={sps}"
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
                    "train/alpha": alpha,
                    "train/entropy": -log_pi.mean().item(),
                    "train/q_grad_norm": q_gnorm,
                    "train/actor_grad_norm": actor_gnorm,
                    "charts/SPS": sps,
                    **buffer_status(rb)[1],
                    "global_step": global_step,
                }
                if cfg.algo.autotune_alpha:
                    log["train/alpha_loss"] = alpha_loss.item()
                if qf_loss_strong is not None:
                    log["train/qf_loss_strong"] = qf_loss_strong.item() / 2.0
                log.update(encoder_log(enc, stack, eo, collapse, enc_gnorm_critic,
                                       enc_gnorm_actor, aux_total, aux_values))
                wandb.log(log, step=global_step)

        # ---- periodic evaluation ----------------------------------------------------
        if global_step % cfg.algo.eval_frequency == 0:
            stats = evaluate(
                lambda o: actor.get_action(o)[2],          # mean action
                eval_env,
                cfg.algo.n_episodes_test,
                device,
                encode_fn=E.encode_single if pixels_in_buffer else None,
                success_fn=_success_fn_for(cfg.env.id),
            )
            log_eval(stats, global_step, wandb)
            if stats["return_mean"] > best_eval_return:
                best_eval_return = stats["return_mean"]
                if cfg.save.enabled:
                    torch.save(
                        actor.state_dict(), os.path.join(save_dir, "best_actor.pt")
                    )
                    if save_encoder:
                        torch.save(encoder_record(cfg, embedding_net),
                                   os.path.join(save_dir, "best_encoder.pt"))

    # ── Final save ────────────────────────────────────────────────────────────
    if cfg.save.enabled:
        ckpt = {
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
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
