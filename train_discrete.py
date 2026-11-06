"""
Discrete-action RL on pretrained visual representations: DQN or discrete
SAC (src/rl/discrete.py), with the same encoder, augmentation, replay
buffer, wandb and checkpoint setup as train_sac.py (src/rl/common.py).

    python train_discrete.py env=fourrooms_pixels embedding=resnet18              # DQN, frozen encoder
    python train_discrete.py algo=sac_discrete env=fourrooms_pixels embedding=resnet18
    python train_discrete.py env=fourrooms_state embedding=none                    # from state
    python train_discrete.py env=fourrooms_pixels embedding=resnet18 encoder.grads=critic

Config: configs/config_discrete.yaml (= config_sac.yaml with algo: dqn and
env: fourrooms_pixels); configs/algo/dqn.yaml, configs/algo/sac_discrete.yaml.
encoder.grads is none | critic: the critic loss (DQN's TD loss, discrete
SAC's twin-Q loss) is the one that can train the encoder, as in DrQ-v2;
discrete SAC's actor reads detached features. Everything else about the
encoder (encoder.*, augmentation, buffer_obs, embedding.pretrained,
frame stacks) works as in train_sac.py; augmentation.strong isn't supported.

Logged metrics reuse train_sac.py's keys where they mean the same thing
(train/qf*_loss, train/qf*_values, train/target_q_mean, train/q_grad_norm,
train/actor_loss, train/alpha, train/entropy, charts/*, eval/*); DQN adds
train/epsilon and train/q_max_mean. With algo.eval_epsilon > 0, each
evaluation runs twice: greedy (eval/*; picks best_agent.pt) and
epsilon-greedy (eval_eps/*). FourRooms' only reward is +1 at the goal, so
eval/mean_reward is its success rate, and charts/episodic_length its steps
to the goal.
"""

import time as _time
# Before the heavy imports, counted in startup_seconds. A list, taken once:
# Hydra's -m runs later jobs in this same process, which start their clock in main().
_T_START = [_time.time()]

import os

# Must be set before dm_control's C extension loads; EGL = GPU offscreen
# rendering, the fast option on a GPU node. Override in the shell if needed.
os.environ.setdefault("MUJOCO_GL", "egl")

import random
import time

import hydra
import numpy as np
import torch
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
from src.rl.discrete import AGENTS, epsilon_greedy
from src.sac_utils import _success_fn_for, evaluate, make_env
from src.timing import timer_from_config
from src.views import CollapseMonitor


@hydra.main(config_path="configs", config_name="config_discrete", version_base=None)
def main(cfg: DictConfig) -> None:
    t_script = _T_START.pop() if _T_START else time.time()
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.allow_tf32 = cfg.perf.tf32
    torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    a = cfg.algo
    if a.name not in AGENTS:
        raise ValueError(f"train_discrete.py: algo={a.name!r}; expected one of {sorted(AGENTS)}")
    enc = encoder_settings(cfg, valid_grads=("none", "critic"))
    if enc.strong_name is not None:
        raise ValueError("augmentation.strong isn't supported by train_discrete.py")
    pixels_in_buffer, buffer_reason = resolve_buffer_obs(cfg, enc)
    warn_unused_encoder_overrides(enc)
    emb_kwargs = embedding_kwargs(cfg)
    save_encoder = enc.trains or enc.bn_adapt

    stack = build_encoder_stack(cfg, enc, emb_kwargs, device, pixels_in_buffer, buffer_reason)
    embedding_net, pixels_in_buffer = stack.net, stack.pixels_in_buffer
    emb_name = cfg.embedding.get("name", None)

    wrap = not pixels_in_buffer
    env = make_env(cfg, embedding_net, wrap_encoder=wrap)
    eval_env = make_env(cfg, embedding_net, wrap_encoder=wrap)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(f"train_discrete.py needs discrete actions; {cfg.env.id} has {env.action_space} "
                         "(continuous: train_sac.py or train_drqv2.py)")
    env.action_space.seed(cfg.seed)
    n_actions = int(env.action_space.n)
    obs_shape = env.observation_space.shape
    feat_dim = int(embedding_net.out_size) if pixels_in_buffer else int(np.prod(obs_shape))
    E = Encoding(stack, device, cfg.perf.amp_bf16)

    obs_norm = cfg.get("obs_norm", None)
    if obs_norm is None:
        obs_norm = emb_name is not None
    agent = AGENTS[a.name](feat_dim, n_actions, a, obs_norm).to(device)
    wandb = init_wandb(cfg, agent.name, enc, stack)
    eo = EncoderOptim(cfg, enc, embedding_net, device)

    print(f"\nAgent: {a.name}, {n_actions} actions, features {feat_dim}  |  buffer stores: {stack.buffer_reason}")
    rb = ReplayBuffer(
        a.buffer_size, obs_shape, 1, device, obs_dtype=np.uint8 if pixels_in_buffer else np.float32,
        nstep=int(a.get("nstep", 1)), gamma=a.gamma, **buffer_options(a, pixels_in_buffer),
    )
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    best_eval_return = -float("inf")

    def features(o):
        """One observation (as the env emits it) -> (1, feat_dim) on the device."""
        if pixels_in_buffer:
            return E.encode_single(o)
        return torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)

    print(f"\nTraining for {a.total_timesteps:,} steps (learning starts at {a.learning_starts:,}, "
          f"an update every {a.train_frequency} step(s))")
    obs, _ = env.reset(seed=cfg.seed)
    E.check_augmentations(obs)
    t0 = time.time()
    collapse = CollapseMonitor()
    timer, bench_steps = timer_from_config(
        cfg, a.total_timesteps, a.eval_frequency, a.learning_starts, save_dir,
        {"script": "train_discrete", "algo": a.name, "env": cfg.env.id, "image_size": cfg.env.get("image_size"),
         "embedding": emb_name, "encoder_grads": enc.grads, "pixels_in_buffer": pixels_in_buffer,
         "batch_size": a.batch_size},
        t_script, log_frequency=a.log_frequency)
    benchmark_done = False
    metrics, enc_gnorm, aux_total, aux_values, q_gnorm = {}, float("nan"), None, {}, float("nan")

    def final_checkpoint():
        ckpt = {"agent": agent.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)}
        if save_encoder:
            ckpt.update(encoder_record(cfg, embedding_net))
        return ckpt

    # Every evaluation, greedy and epsilon-greedy, resets to the same episodes
    # (FourRooms: the same start/goal pairs), so their difference and the
    # learning curve show the policy, not which episodes were drawn.
    eval_seeds = [int(s) for s in np.random.default_rng([cfg.seed, 1]).integers(2**31, size=a.n_episodes_test)]
    eval_eps = float(a.get("eval_epsilon", 0.0))
    eval_gen = torch.Generator(device=device).manual_seed(cfg.seed)

    def run_eval(step, timing_only=False):
        nonlocal best_eval_return
        greedy = lambda o: agent.act(o, step, explore=False)  # noqa: E731
        with timer.exclude("eval", timing_only=timing_only):
            eval_kw = dict(encode_fn=E.encode_single if pixels_in_buffer else None,
                           success_fn=_success_fn_for(cfg.env.id), seeds=eval_seeds)
            stats = evaluate(greedy, eval_env, a.n_episodes_test, device, **eval_kw)
            log_eval(stats, step, wandb)
            # A greedy policy that bumps into one wall stays there until the time
            # limit; a little randomness shows whether the rest of it works.
            if eval_eps > 0:
                eps_stats = evaluate(epsilon_greedy(greedy, n_actions, eval_eps, eval_gen), eval_env,
                                     a.n_episodes_test, device, **eval_kw)
                log_eval(eps_stats, step, wandb, prefix="eval_eps", label=f"EVAL eps={eval_eps:g}")
            if stats["return_mean"] > best_eval_return and not bench_steps:
                best_eval_return = stats["return_mean"]
                if cfg.save.enabled:
                    torch.save(agent.state_dict(), os.path.join(save_dir, "best_agent.pt"))
                    if save_encoder:
                        torch.save(encoder_record(cfg, embedding_net), os.path.join(save_dir, "best_encoder.pt"))

    for step in range(1, a.total_timesteps + 1):
        # ---- act ------------------------------------------------------------------
        if step <= a.learning_starts:
            action = int(env.action_space.sample())
        else:
            action = int(agent.act(features(obs), step, explore=True).item())
        next_obs, reward, terminated, truncated, info = env.step(action)
        rb.add(obs, next_obs, np.array([action], dtype=np.float32), reward, terminated, truncated)
        if terminated or truncated:
            log_episode(info, step, wandb)
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step <= a.learning_starts:
            if step % a.log_frequency == 0 or step == a.learning_starts:
                timer.mark(step)
            continue

        # ---- update ---------------------------------------------------------------
        if step % int(a.train_frequency) == 0:
            for _ in range(int(a.get("gradient_steps", 1))):
                b_obs, b_next_obs, b_actions, b_rewards, b_discounts = rb.sample(a.batch_size)
                b_actions = b_actions.view(-1).long()
                route_critic = enc.grads == "critic"
                if enc.bn_adapt:
                    embedding_net.set_bn_train(True)
                if pixels_in_buffer:
                    views, next_feats, _ = encode_replay_batch(E, stack, enc, eo, b_obs, b_next_obs,
                                                               task_grad=route_critic)
                    feats = views.feats_task
                    if enc.trains and step % a.log_frequency == 0:
                        collapse.update(feats, step=step - a.learning_starts, where=f"step {step:,}")
                else:
                    feats, next_feats = b_obs, b_next_obs

                loss, metrics = agent.critic_loss(feats, b_actions, b_rewards, b_discounts, next_feats)
                aux_total, aux_values = (total_aux_loss(eo.aux_terms, views.aux) if eo.aux_terms
                                         else (None, {}))
                enc_step = route_critic or bool(eo.aux_terms)
                if enc_step:
                    eo.zero_grad()
                agent.critic_opt.zero_grad(set_to_none=True)
                (loss if aux_total is None else loss + aux_total).backward()
                q_gnorm = float(torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=a.grad_clip))
                agent.critic_opt.step()
                enc_gnorm = eo.step() if enc_step else float("nan")
                metrics.update(agent.after_critic(feats.detach(), step))
                if enc.bn_adapt:
                    embedding_net.set_bn_train(False)
        agent.update_targets(step)

        # ---- logging --------------------------------------------------------------
        if step % a.log_frequency == 0:
            sps = int(step / (time.time() - t0))
            timer.mark(step)
            timing_text, timing_metrics = timer.status(step)
            timer.write(step)
            buffer_text, buffer_metrics = buffer_status(rb)     # once: it resets the buffer's timings
            shown = {k.split("/")[-1]: v for k, v in metrics.items()
                     if k.split("/")[-1] in ("qf_loss", "qf1_values", "epsilon", "alpha", "entropy")}
            print(f"  step {step:>8,}  " + "  ".join(f"{k}={v:.3f}" for k, v in shown.items())
                  + f"  q_gnorm={q_gnorm:.2f}  SPS={sps}" + buffer_text
                  + (f"  enc_gnorm={enc_gnorm:.2f}  emb_std_ratio={collapse.ratio:.2f}" if enc.trains else "")
                  + timing_text)
            if wandb is not None:
                log = {**metrics, "train/q_grad_norm": q_gnorm, "charts/SPS": sps, **buffer_metrics,
                       **timing_metrics, "global_step": step}
                log.update(encoder_log(enc, stack, eo, collapse, enc_gnorm, float("nan"), aux_total, aux_values))
                wandb.log(log, step=step)

        if step % a.eval_frequency == 0:
            run_eval(step)

        if (bench_steps and step % a.log_frequency == 0 and timer.rate() is not None
                and step - a.learning_starts >= bench_steps):
            if timer.mean_excluded("eval") is None:
                run_eval(step, timing_only=True)
            if cfg.save.enabled:
                test_path = os.path.join(save_dir, ".timing_save_test.pt")
                with timer.exclude("save", timing_only=True):
                    torch.save(final_checkpoint(), test_path)
                os.remove(test_path)
            timer.write(step)
            print("\n" + timer.benchmark_report(step))
            benchmark_done = True
            break

    if cfg.save.enabled and not benchmark_done:
        with timer.exclude("save"):
            torch.save(final_checkpoint(), os.path.join(save_dir, "final_checkpoint.pt"))
        print(f"\nSaved to: {save_dir}")
    if not benchmark_done:
        print(f"Best eval return: {best_eval_return:.3f}")
    env.close()
    eval_env.close()
    if wandb is not None:
        wandb.finish()
    if not benchmark_done:
        timer.write(step)


if __name__ == "__main__":
    main()
