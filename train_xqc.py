"""
XQC (Palenicek et al., ICLR 2026) on this repo's encoders and environments:
SAC with XQC's BatchNorm + weight-normalised MLPs and categorical critic
(src/rl/xqc.py, configs/algo/xqc.yaml).

    python train_xqc.py env=pendulum embedding=none                  # state
    python train_xqc.py env=pendulum_pixels embedding=resnet50 +embedding.frame_stack=3
    python train_xqc.py env=dm_control_state embedding=none algo.action_repeat=2

Same setup as train_sac.py -- encoder, augmentation, buffer_obs, encoder.*
options (grads none | critic | actor | both, aux losses, bn), wandb,
checkpoints, timing (src/rl/common.py) -- with XQC's agent, following the
authors' code:

  - algo.updates_per_step updates per agent step (UTD 2), each on its own
    batch: critic step -> weight projection -> Polyak target update ->
    actor + temperature step on every algo.policy_delay-th update
  - critic: (s, a) and (s', a') in one BatchNorm batch; categorical
    cross-entropy against the projected target of the target critic with
    the smaller expected value; rewards divided by the running std of the
    discounted return (algo.reward_normalization)
  - discount from the episode length (algo.gamma=auto)
  - Adam, learning rates decayed linearly to algo.lr_end; no grad clipping
  - temperature from 0.01, target entropy -action_dim / 2

augmentation.strong isn't supported. obs_norm doesn't apply (the networks
start with their own BatchNorm). Logged metrics reuse train_sac.py's keys;
XQC adds train/critic_lr, train/actor_lr, train/reward_scale,
train/alpha_reward_units (alpha weighs entropy against normalised rewards:
compare this one across tasks or with SAC's train/alpha),
train/target_mass_vmin/vmax (target probability on the edge atoms: values
piling up there mean [v_min, v_max] is too narrow) and charts/updates
(gradient steps, for plots against update count).
"""

import time as _time
# Before the heavy imports, counted in startup_seconds. A list, taken once:
# Hydra's -m runs later jobs in this same process, which start their clock in main().
_T_START = [_time.time()]

import os

# Must be set before dm_control's C extension loads; EGL = GPU offscreen
# rendering, the fast option on a GPU node. Override in the shell if needed.
os.environ.setdefault("MUJOCO_GL", "egl")

import math
import random
import time

import hydra
import numpy as np
import torch
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
from src.rl.xqc import (
    CriticEnsemble,
    RewardNormalizer,
    TanhGaussianActor,
    actor_q,
    bn_mode,
    critic_loss,
    make_target,
    polyak_,
    project_weights,
    resolve_gamma,
    set_lr,
    xqc_learning_rates,
)
from src.sac_utils import _success_fn_for, evaluate, make_env
from src.timing import timer_from_config
from src.views import CollapseMonitor


@hydra.main(config_path="configs", config_name="config_xqc", version_base=None)
def main(cfg: DictConfig) -> None:
    t_script = _T_START.pop() if _T_START else time.time()
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
    if enc.strong_name is not None:
        raise ValueError("augmentation.strong isn't supported by train_xqc.py")
    pixels_in_buffer, buffer_reason = resolve_buffer_obs(cfg, enc)
    warn_unused_encoder_overrides(enc)
    emb_kwargs = embedding_kwargs(cfg)
    routing = enc.grads
    encoder_trains = enc.trains
    save_encoder = encoder_trains or enc.bn_adapt

    # ── Encoder, wandb, envs (src/rl/common.py) ───────────────────────────────
    stack = build_encoder_stack(cfg, enc, emb_kwargs, device, pixels_in_buffer, buffer_reason)
    embedding_net, pixels_in_buffer = stack.net, stack.pixels_in_buffer
    wandb = init_wandb(cfg, "xqc", enc, stack)

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

    # The agent acts in [-1, 1] (XQC rescales the env's actions); the env gets
    # the rescaled action. The buffer stores the agent's action.
    low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
    to_env = lambda act: low + (act + 1.0) * 0.5 * (high - low)

    a = cfg.algo
    action_repeat = int(a.get("action_repeat", 1))
    gamma = resolve_gamma(a.gamma, env, action_repeat)
    print(f"\nDiscount: {gamma:.4f}" + (" (auto: 1 - 5/T)" if a.gamma == "auto" else ""))

    # ── Networks / optimizers ─────────────────────────────────────────────────
    actor = TanhGaussianActor(feat_dim, action_dim, list(a.actor_hidden),
                              normalize_last_layer=a.normalize_last_layer).to(device)
    critic = CriticEnsemble(
        feat_dim, action_dim, list(a.critic_hidden), n_critics=a.n_critics,
        critic_loss=a.critic_loss, num_atoms=a.num_atoms, v_min=a.v_min, v_max=a.v_max,
        normalize_last_layer=a.normalize_last_layer,
    ).to(device)
    if a.weight_norm:
        project_weights(actor)
        project_weights(critic)
    critic_target = make_target(critic)
    actor.eval()        # acting, evaluation and next-state actions: running statistics
    critic.eval()
    n_params = sum(p.numel() for p in critic.parameters()) + sum(p.numel() for p in actor.parameters())
    print(f"XQC networks: {n_params / 1e6:.2f}M parameters "
          f"(critic {list(a.critic_hidden)} x{a.n_critics}, actor {list(a.actor_hidden)}, "
          f"{a.critic_loss} loss)")

    critic_optimizer = optim.Adam(critic.parameters(), lr=a.critic_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=a.actor_lr)
    log_alpha = torch.tensor(math.log(a.init_alpha), device=device, requires_grad=True)
    alpha_optimizer = optim.Adam([log_alpha], lr=a.actor_lr)
    target_entropy = -float(a.target_entropy_scale) * action_dim
    eo = EncoderOptim(cfg, enc, embedding_net, device)
    grad_clip = float("inf") if a.get("grad_clip") is None else a.grad_clip
    reward_norm = RewardNormalizer(gamma) if a.reward_normalization else None

    print(f"\nBuffer stores: {stack.buffer_reason}")
    rb = ReplayBuffer(
        a.buffer_size, obs_shape, action_dim, device, obs_dtype=buffer_dtype,
        nstep=int(a.get("nstep", 1)), gamma=gamma, **buffer_options(a, pixels_in_buffer),
    )
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    best_eval_return = -float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(
        f"\nTraining for {a.total_timesteps:,} agent steps "
        f"({a.total_timesteps * action_repeat:,} frames; learning starts at "
        f"{a.learning_starts:,}; {a.updates_per_step} updates per step)"
    )
    obs, _ = env.reset(seed=cfg.seed)
    E.check_augmentations(obs)
    t0 = time.time()
    enc_gnorm_critic = float("nan")
    enc_gnorm_actor = float("nan")
    collapse = CollapseMonitor()
    timer, bench_steps = timer_from_config(
        cfg, a.total_timesteps, a.eval_frequency, a.learning_starts, save_dir,
        {"script": "train_xqc", "env": cfg.env.id, "image_size": cfg.env.get("image_size"),
         "embedding": cfg.embedding.get("name"), "encoder_grads": enc.grads,
         "pixels_in_buffer": pixels_in_buffer, "batch_size": a.batch_size,
         "updates_per_step": a.updates_per_step, "action_repeat": action_repeat,
         "buffer_compression": a.get("buffer_compression", "none")},
        t_script, log_frequency=a.log_frequency)
    benchmark_done = False
    n_updates = 0           # critic steps

    def final_checkpoint():
        ckpt = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "critic_target": critic_target.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "reward_normalizer": reward_norm.state_dict() if reward_norm is not None else None,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if save_encoder:
            ckpt.update(encoder_record(cfg, embedding_net))
        return ckpt

    def run_eval(global_step, timing_only=False):
        nonlocal best_eval_return
        with timer.exclude("eval", timing_only=timing_only):
            stats = evaluate(
                lambda o: to_env(actor.mean_action(o)),
                eval_env,
                a.n_episodes_test,
                device,
                encode_fn=E.encode_single if pixels_in_buffer else None,
                success_fn=_success_fn_for(cfg.env.id),
            )
            log_eval(stats, global_step, wandb)
            if stats["return_mean"] > best_eval_return and not bench_steps:
                best_eval_return = stats["return_mean"]
                if cfg.save.enabled:
                    torch.save(actor.state_dict(), os.path.join(save_dir, "best_actor.pt"))
                    if save_encoder:
                        torch.save(encoder_record(cfg, embedding_net),
                                   os.path.join(save_dir, "best_encoder.pt"))

    def update():
        """One XQC update on a fresh batch. Returns this update's metrics."""
        nonlocal n_updates, enc_gnorm_critic, enc_gnorm_actor
        b_obs, b_next_obs, b_actions, b_rewards, b_discounts = rb.sample(a.batch_size)
        if reward_norm is not None:
            b_rewards = reward_norm.normalize(b_rewards)

        route_critic = routing in ("critic", "both")
        route_actor = routing in ("actor", "both")
        if enc.bn_adapt:
            embedding_net.set_bn_train(True)

        views = None
        if pixels_in_buffer:
            views, next_feats, _ = encode_replay_batch(
                E, stack, enc, eo, b_obs, b_next_obs, task_grad=route_critic
            )
            feats = views.feats_task
        else:
            feats, next_feats = b_obs, b_next_obs
        alpha = log_alpha.exp().item()

        # ---- critic -------------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_pi = actor.sample(next_feats)      # eval mode
        loss, stats = critic_loss(
            critic, critic_target, feats, b_actions, next_feats, next_actions,
            b_rewards, b_discounts, next_bonus=-alpha * next_log_pi,
        )
        aux_total, aux_values = None, {}
        if eo.aux_terms:
            aux_total, aux_values = total_aux_loss(eo.aux_terms, views.aux)

        enc_step_now = route_critic or bool(eo.aux_terms)
        enc_gnorm_critic = float("nan")
        enc_gnorm_actor = float("nan")
        if enc_step_now:
            eo.zero_grad()
        critic_optimizer.zero_grad(set_to_none=True)
        (loss if aux_total is None else loss + aux_total).backward()
        q_gnorm = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip))
        critic_lr, actor_lr = xqc_learning_rates(n_updates, a)
        set_lr(critic_optimizer, critic_lr)
        critic_optimizer.step()
        if a.weight_norm:
            project_weights(critic)
        if enc_step_now:
            enc_gnorm_critic = eo.step()
        polyak_(critic, critic_target, a.tau)

        # ---- actor + temperature, delayed --------------------------------------
        metrics = None
        if n_updates % a.policy_delay == 0:
            if pixels_in_buffer and route_actor:
                # Re-encode with grad: the critic's graph is consumed and the
                # encoder may have just stepped. Same pixels and draw.
                feats_pi = E.featurize(views.x_task, grad=True)
            else:
                feats_pi = feats.detach()
            with bn_mode(True, actor):
                pi, log_pi = actor.sample(feats_pi)
            # Encoder grads (route_actor) flow through the policy input and
            # the Q evaluation, as in train_sac.py.
            actor_loss = (alpha * log_pi - actor_q(critic, feats_pi, pi)).mean()
            if route_actor:
                eo.zero_grad()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gnorm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=grad_clip))
            set_lr(actor_optimizer, actor_lr)
            actor_optimizer.step()
            if a.weight_norm:
                project_weights(actor)
            if route_actor:
                enc_gnorm_actor = eo.step()

            entropy = -log_pi.detach().mean()
            alpha_loss = log_alpha.exp() * (entropy - target_entropy)
            alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            set_lr(alpha_optimizer, actor_lr)
            alpha_optimizer.step()
            metrics = {
                "train/actor_loss": actor_loss.item(),
                "train/entropy": entropy.item(),
                "train/alpha_loss": alpha_loss.item(),
                "train/actor_grad_norm": actor_gnorm,
                "train/actor_lr": actor_lr,
            }
        else:
            # The actor's BatchNorm statistics update on every update, as in
            # the authors' code (only its parameter step is delayed).
            with torch.no_grad(), bn_mode(True, actor):
                actor(feats.detach())
        n_updates += 1

        if enc.bn_adapt:
            embedding_net.set_bn_train(False)

        out = {
            **stats, "q_gnorm": q_gnorm,
            "alpha": log_alpha.exp().item(), "feats": feats, "aux_total": aux_total,
            "aux_values": aux_values,
        }
        return out, metrics

    last_actor_metrics = {}
    for global_step in range(1, a.total_timesteps + 1):
        # ---- act --------------------------------------------------------------
        if global_step <= a.learning_starts:
            action = np.random.uniform(-1.0, 1.0, action_dim).astype(np.float32)
        else:
            if pixels_in_buffer:
                obs_t = E.encode_single(obs)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor.sample(obs_t)[0].squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(
            to_env(torch.as_tensor(action, device=device)).cpu().numpy()
        )
        # The buffer keeps the episode's real final observation and masks
        # only termination: truncation (time limits) still bootstraps.
        rb.add(obs, next_obs, action, reward, terminated, truncated)
        if reward_norm is not None:
            reward_norm.update(reward, terminated or truncated)

        if terminated or truncated:
            log_episode(info, global_step, wandb)
            obs, _ = env.reset()
        else:
            obs = next_obs

        if global_step <= a.learning_starts:
            if global_step % a.log_frequency == 0 or global_step == a.learning_starts:
                timer.mark(global_step)        # the warm-up's own rate
            continue

        # ---- updates ------------------------------------------------------------
        for _ in range(a.updates_per_step):
            last, actor_metrics = update()
            if actor_metrics is not None:
                last_actor_metrics = actor_metrics
        if encoder_trains and pixels_in_buffer and global_step % a.log_frequency == 0:
            collapse.update(last["feats"], step=global_step - a.learning_starts,
                            where=f"step {global_step:,}")

        # ---- logging --------------------------------------------------------------
        if global_step % a.log_frequency == 0:
            sps = int(global_step / (time.time() - t0))
            timer.mark(global_step)
            timing_text, timing_metrics = timer.status(global_step)
            timer.write(global_step)
            buffer_text, buffer_metrics = buffer_status(rb)     # once: it resets the buffer's timings
            per_critic, q = last["per_critic"], last["q"]
            critic_lr = critic_optimizer.param_groups[0]["lr"]
            reward_scale = reward_norm.scale if reward_norm is not None else 1.0
            edge_text = (f"  edge_mass={last['mass_vmin'].item():.3f}/{last['mass_vmax'].item():.3f}"
                         if "mass_vmin" in last else "")
            print(
                f"  step {global_step:>8,}  qf_loss={per_critic.mean().item():8.3f}"
                f"  q1={q[0].mean().item():7.2f}"
                f"  actor_loss={last_actor_metrics.get('train/actor_loss', float('nan')):8.3f}"
                f"  alpha={last['alpha']:.4f}  SPS={sps}"
                f"  gnorm(q/actor)={last['q_gnorm']:.2f}/"
                f"{last_actor_metrics.get('train/actor_grad_norm', float('nan')):.2f}"
                f"  r_scale={reward_scale:.3f}" + edge_text
                + buffer_text
                + (
                    f"  enc_gnorm(c/a)={enc_gnorm_critic:.2f}/{enc_gnorm_actor:.2f}"
                    f"  emb_std_ratio={collapse.ratio:.2f}"
                    if encoder_trains
                    else ""
                )
                + timing_text
            )
            if wandb is not None:
                log = {
                    "train/qf_loss": per_critic.mean().item(),
                    "train/target_q_mean": last["target_mean"].item(),
                    "train/alpha": last["alpha"],
                    # alpha weighs the entropy against normalised rewards;
                    # this is the same weight against the env's rewards.
                    "train/alpha_reward_units": last["alpha"] * reward_scale,
                    "train/q_grad_norm": last["q_gnorm"],
                    "train/critic_lr": critic_lr,
                    "train/reward_scale": reward_scale,
                    "charts/SPS": sps,
                    "charts/frames": global_step * action_repeat,
                    "charts/updates": n_updates,
                    **last_actor_metrics,
                    **buffer_metrics,
                    **timing_metrics,
                    "global_step": global_step,
                }
                if "mass_vmin" in last:
                    log["train/target_mass_vmin"] = last["mass_vmin"].item()
                    log["train/target_mass_vmax"] = last["mass_vmax"].item()
                for i in range(per_critic.shape[0]):
                    log[f"train/qf{i + 1}_loss"] = per_critic[i].item()
                    log[f"train/qf{i + 1}_values"] = q[i].mean().item()
                log.update(encoder_log(enc, stack, eo, collapse, enc_gnorm_critic,
                                       enc_gnorm_actor, last["aux_total"], last["aux_values"]))
                wandb.log(log, step=global_step)

        # ---- periodic evaluation ----------------------------------------------------
        if global_step % a.eval_frequency == 0:
            run_eval(global_step)

        # ---- benchmark mode: measure, predict, stop ------------------------------------
        if (bench_steps and global_step % a.log_frequency == 0 and timer.rate() is not None
                and global_step - a.learning_starts >= bench_steps):
            if timer.mean_excluded("eval") is None:
                run_eval(global_step, timing_only=True)   # one evaluation, to time it
            if cfg.save.enabled:
                test_path = os.path.join(save_dir, ".timing_save_test.pt")
                with timer.exclude("save", timing_only=True):
                    torch.save(final_checkpoint(), test_path)
                os.remove(test_path)
            timer.write(global_step)
            print("\n" + timer.benchmark_report(global_step))
            benchmark_done = True
            break

    # ── Final save ────────────────────────────────────────────────────────────
    if cfg.save.enabled and not benchmark_done:
        with timer.exclude("save"):
            torch.save(final_checkpoint(), os.path.join(save_dir, "final_checkpoint.pt"))
        print(f"\nSaved to: {save_dir}")
    if not benchmark_done:
        print(f"Best eval return: {best_eval_return:.1f}")

    env.close()
    eval_env.close()
    if wandb is not None:
        wandb.finish()
    if not benchmark_done:
        timer.write(global_step)       # the final record: the job's whole time, final save included


if __name__ == "__main__":
    main()
