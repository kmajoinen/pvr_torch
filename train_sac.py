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
      drift. aux with grads=none = adaptation without RL gradients at all.

Design notes:
  - Actor and critics each apply their own LayerNorm to input features --
    only when the input IS encoder features (embedding set). Raw state
    vectors skip it: per-sample LN on a ~17-dim state aliases states that
    differ by uniform shift/scale and destroys magnitude information.
  - Bootstrapping is masked on `terminated` only, so truncation-only envs
    (dm_control, 1000-step episodes) bootstrap through the time limit.
  - No target encoder: target-Q features come from the live encoder under
    no_grad (DrQ-v2 convention -- target networks exist for the Q heads only).
  - Encoder stays in eval() mode even while its weights train: pretrained
    ResNets carry BatchNorm, and letting BN running stats update on RL frames
    is a known silent-corruption source (and the failure mode behind this
    project's original resnet50 NaN). Frozen BN statistics, trainable BN
    affine params -- the standard finetuning choice.
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
    python train_sac.py 'encoder.aux=[{name:l2sp,weight:1.0}]' \
        env.image_size=112                                      # aux-only adaptation
    python train_sac.py augmentation=random_shift augmentation.stage=aux \
        'encoder.aux=[{name:aug_consistency,weight:0.1},{name:l2sp,weight:1.0}]' \
        env.image_size=112                                      # consistency-only augmentation
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
    encoder_settings,
    resolve_buffer_obs,
    warn_unused_encoder_overrides,
)
from src.losses import aux_grad_views, build_aux_losses, total_aux_loss
from src.views import CollapseMonitor, encode_views
from src.sac_utils import (
    Actor,
    ReplayBuffer,
    SoftQNetwork,
    _success_fn_for,
    evaluate,
    make_env,
)

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
    routing = enc.grads
    encoder_trains = enc.trains
    if encoder_trains:
        print(
            f"Encoder trains: grads={routing}  aux={enc.aux_names}"
            f"  lr={cfg.encoder.lr}"
        )

    # ── Weights & Biases ──────────────────────────────────────────────────────
    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb

        emb_tag = cfg.embedding.get("name") or "state"
        # Run names are labels only; charts are keyed by the logged metric
        # names, which stay fixed for continuity.
        tags = ""
        if encoder_trains:
            tags += "_ft-" + "-".join([routing] + enc.aux_names)
        if enc.aug_name is not None:
            tags += f"_aug-{enc.aug_name}" + (
                f"-{enc.aug_stage}" if enc.aug_stage != "rl" else ""
            )
        if pixels_in_buffer and not encoder_trains and enc.aug_name is None:
            tags += "_slowfrozen"
        run_name = (
            cfg.wandb.run_name or f"sac_{cfg.env.name}_{emb_tag}{tags}_s{cfg.seed}"
        )
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            mode=cfg.wandb.get("mode", "online"),
            config=OmegaConf.to_container(cfg, resolve=True),
            save_code=True,
        )
        print(f"wandb run: {wandb.run.url}")

    # ── Encoder ───────────────────────────────────────────────────────────────
    emb_name = cfg.embedding.get("name", None)
    embedding_net = None
    aug_module = None
    if emb_name is not None:
        from src.augmentations import build_augmentation
        from src.embeddings import EmbeddingNet

        # Only ever non-None on the pixel path (resolve_buffer_obs guarantees
        # augmentation => pixels), so this banner means it really runs.
        aug_module = build_augmentation(cfg.augmentation)
        if aug_module is not None:
            aug_module = aug_module.to(device)
            print(
                f"Augmentation: {enc.aug_name}  stage={enc.aug_stage}"
                f"  params={dict(cfg.augmentation.get('params') or {})}"
                "  (replay-batch encodes only; rollouts/eval stay clean)"
            )

        frame_stack = cfg.embedding.get("frame_stack", 1)
        embedding_net = EmbeddingNet(
            emb_name,
            in_channels=3 * frame_stack if emb_name == "random" else 3,
            pretrained=True,
            train=encoder_trains,
            disable_cuda=(str(device) == "cpu"),
            model_dir=cfg.model_dir,
        )
        # eval() ALWAYS, even while weights train: freeze BatchNorm running
        # stats (see module docstring). Trainability is controlled purely via
        # requires_grad below, never via train()-mode.
        embedding_net.eval()
        for p in embedding_net.parameters():
            p.requires_grad_(encoder_trains)
        print(
            f"\nEmbedding: {emb_name}  →  obs_size: {embedding_net.out_size}"
            f"  (trainable={encoder_trains})"
        )

    # ── Envs ──────────────────────────────────────────────────────────────────
    # Features path: env wrapper encodes, buffer stores features.
    # Pixels path: env emits pixels, encoding happens in the loop.
    wrap = not pixels_in_buffer
    env = make_env(cfg, embedding_net, wrap_encoder=wrap)
    eval_env = make_env(cfg, embedding_net, wrap_encoder=wrap)
    assert isinstance(env.action_space, gym.spaces.Box), "continuous actions only"
    env.action_space.seed(cfg.seed)

    action_dim = int(np.prod(env.action_space.shape))
    if pixels_in_buffer:
        obs_shape = env.observation_space.shape  # (H, W, C) uint8
        feat_dim = int(embedding_net.out_size)
        buffer_dtype = np.uint8
    else:
        obs_shape = env.observation_space.shape  # (feat_dim,) or state
        feat_dim = int(np.prod(obs_shape))
        buffer_dtype = np.float32

    # ── Encoding helper (pixels path) ─────────────────────────────────────────
    amp_on = cfg.perf.amp_bf16 and device.type == "cuda"
    _grad_contract_checked = False

    # Three steps, so augmentation sits between preprocessing and the
    # encoder (at the encoder's input resolution) and a view can be encoded
    # more than once without re-drawing it -- see src/views.py.
    def preprocess(obs_batch) -> torch.Tensor:
        """(B, H, W, C) uint8 (tensor or ndarray) -> float (B, C, h, w) in
        [0, 1] at the encoder's input resolution. Same layout/dtype contract
        as FrozenEncoderWrapper."""
        x = (
            torch.as_tensor(np.ascontiguousarray(obs_batch))
            if not torch.is_tensor(obs_batch)
            else obs_batch
        )
        x = x.to(device).permute(0, 3, 1, 2)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            return embedding_net.preprocess(x)

    def augment(x: torch.Tensor) -> torch.Tensor:
        """Replay-batch views only -- never rollout-time action selection or
        eval(), where the encoder should see the real observation. Runs in
        fp32, outside autocast: it's cheap next to the encoder, and under
        autocast kornia's convolutions (e.g. blur) would compute the
        augmented view in bf16 -- the output is still fp32, but rounded by
        up to ~2 uint8 levels while the clean view stays exact."""
        with torch.no_grad():
            return aug_module(x)

    def featurize(x: torch.Tensor, grad: bool) -> torch.Tensor:
        """preprocess() output (optionally augmented) -> (B, feat_dim) float32."""
        nonlocal _grad_contract_checked
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            feats = embedding_net.encode_preprocessed(x)
        if grad and not _grad_contract_checked:
            # TODO(EmbeddingNet contract): if this trips, EmbeddingNet.encode
            # breaks the graph internally (no_grad/inference_mode, a numpy
            # round-trip, or a non-differentiable jit path, e.g. some CLIP
            # torch.jit archives). Fix inside src/embeddings.py for
            # train=True, or add a grad-capable encode_train() there.
            assert (
                torch.is_tensor(feats) and feats.requires_grad
            ), "EmbeddingNet.encode returned a graph-less result under enable_grad"
            _grad_contract_checked = True
        if not torch.is_tensor(feats):
            feats = torch.as_tensor(feats, device=device)
        return feats.float()

    def encode_single(obs_np) -> torch.Tensor:
        return featurize(preprocess(np.asarray(obs_np)[None]), grad=False)

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
    assert isinstance(actor.obs_norm, torch.nn.Identity if emb_name is None else torch.nn.LayerNorm), f"obs_norm wiring wrong for emb_name={emb_name}"
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

    enc_optimizer, theta0, enc_params, aux_terms = None, None, [], []
    if encoder_trains:
        aux_terms = build_aux_losses(enc.aux_specs, embedding_net)
        for t in aux_terms:
            t.loss.to(device)
        # Learnable heads owned by aux losses train alongside the encoder.
        enc_params = list(embedding_net.parameters()) + [
            p for t in aux_terms for p in t.loss.parameters()
        ]
        # Separate, much lower LR: at the head LRs a resnet18's weight sums
        # drift ~3% in 50 gradient steps (measured) -- far too fast for a
        # pretrained representation you want to adapt, not destroy.
        enc_optimizer = optim.Adam(
            enc_params, lr=cfg.encoder.lr, betas=tuple(cfg.encoder.betas)
        )
        # Kept on CPU: only read for the drift metric at log time, and a GPU
        # copy would double the encoder's memory (l2sp keeps its own anchor).
        theta0 = {
            n: p.detach().cpu().clone() for n, p in embedding_net.named_parameters()
        }
    aux_grad_clean, aux_grad_aug = aux_grad_views(aux_terms)

    def encoder_drift() -> float:
        with torch.no_grad():
            return float(
                sum(
                    ((p.detach().cpu() - theta0[n]) ** 2).sum()
                    for n, p in embedding_net.named_parameters()
                ).sqrt()
            )

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

    print(f"\nBuffer stores: {buffer_reason}")
    rb = ReplayBuffer(
        cfg.algo.buffer_size, obs_shape, action_dim, device, obs_dtype=buffer_dtype
    )
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    best_eval_return = -float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(
        f"\nTraining for {cfg.algo.total_timesteps:,} steps "
        f"(learning starts at {cfg.algo.learning_starts:,})"
    )
    obs, _ = env.reset(seed=cfg.seed)
    if aug_module is not None:
        # One real frame through preprocess -> augmentation now, so an input
        # contract violation (value range, channel count, non-square frame)
        # fails at startup rather than at the first update after
        # learning_starts. This is also where the one-time range check runs.
        augment(preprocess(np.asarray(obs)[None]))
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
                obs_t = encode_single(obs)
            else:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
            with torch.no_grad():
                action_t, _, _ = actor.get_action(obs_t)
            action = action_t.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)

        # bootstrap mask uses `terminated` only: on truncation (time limit)
        # the value of next_obs is still real and should be bootstrapped.
        rb.add(obs, next_obs, action, reward, terminated)

        if terminated or truncated:
            ep = info.get("episode")
            if ep is not None:
                print(
                    f"  step {global_step:>8,}  episodic_return={float(ep['r']):8.1f}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "charts/episodic_return": float(ep["r"]),
                            "charts/episodic_length": int(ep["l"]),
                            "global_step": global_step,
                        },
                        step=global_step,
                    )
            obs, _ = env.reset()
        else:
            obs = next_obs

        if global_step <= cfg.algo.learning_starts:
            continue

        # ---- sample + featurize -----------------------------------------------
        b_obs, b_next_obs, b_actions, b_rewards, b_terms = rb.sample(
            cfg.algo.batch_size
        )

        route_critic = routing in ("critic", "both")
        route_actor = routing in ("actor", "both")

        if pixels_in_buffer:
            # One augmentation draw per update, routed by augmentation.stage.
            views = encode_views(
                featurize,
                embedding_net,
                preprocess(b_obs),
                augment if aug_module is not None else None,
                task_aug=enc.task_aug,
                task_grad=route_critic,
                aux_needs_clean=enc.aux_needs_clean,
                aux_needs_aug=enc.aux_needs_aug,
                aux_grad_clean=aux_grad_clean,
                aux_grad_aug=aux_grad_aug,
            )
            feats = views.feats_task
            # next_obs gets the task's view, with its own independent draw
            # (DrQ-v2 augments obs and next_obs separately). No target
            # encoder: live encoder under no_grad (DrQ-v2 convention).
            x_next = preprocess(b_next_obs)
            if enc.task_aug:
                x_next = augment(x_next)
            next_feats = featurize(x_next, grad=False)

            # First update and every log step; each measurement is a GPU sync.
            if encoder_trains and (
                collapse.initial is None or global_step % cfg.algo.log_frequency == 0
            ):
                collapse.update(feats, where=f"step {global_step:,}")
        else:
            feats, next_feats = b_obs, b_next_obs

        # ---- critic update ------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_pi, _ = actor.get_action(next_feats)
            qf1_next = qf1_target(next_feats, next_actions)
            qf2_next = qf2_target(next_feats, next_actions)
            min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
            next_q_value = b_rewards + (
                1 - b_terms
            ) * cfg.algo.gamma * min_qf_next.view(-1)

        qf1_a_values = qf1(feats, b_actions).view(-1)
        qf2_a_values = qf2(feats, b_actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        aux_total, aux_values = None, {}
        if aux_terms:
            aux_total, aux_values = total_aux_loss(aux_terms, views.aux)

        enc_step_now = route_critic or bool(aux_terms)
        enc_gnorm_critic = float("nan")
        enc_gnorm_actor = float("nan")
        if enc_step_now:
            enc_optimizer.zero_grad(set_to_none=True)
        q_optimizer.zero_grad()
        # One backward for both: the critic and an aux loss can share an
        # encoder graph (src/views.py), which a second backward couldn't
        # traverse. Aux losses don't touch the Q networks, so the critic's
        # gradients are unchanged; the encoder's are the sum, as before.
        (qf_loss if aux_total is None else qf_loss + aux_total).backward()
        q_gnorm = float(
            torch.nn.utils.clip_grad_norm_(
                list(qf1.parameters()) + list(qf2.parameters()),
                max_norm=cfg.algo.grad_clip,
            )
        )
        q_optimizer.step()

        if enc_step_now:
            enc_gnorm_critic = float(
                torch.nn.utils.clip_grad_norm_(
                    enc_params, max_norm=cfg.encoder.grad_clip
                )
            )
            enc_optimizer.step()

        # ---- actor (+ alpha) update, delayed ------------------------------------
        if global_step % cfg.algo.policy_frequency == 0:
            for _ in range(cfg.algo.policy_frequency):
                if pixels_in_buffer:
                    if route_actor:
                        # Re-encode with grad each inner iteration: the
                        # encoder may have just been updated, and the critic
                        # phase's graph is already consumed. Same pixels (and
                        # augmentation draw) the critic saw.
                        feats_pi = featurize(views.x_task, grad=True)
                    else:
                        # Reuse critic-phase features (detached). If the
                        # encoder stepped this iteration these are stale by
                        # exactly one enc_optimizer update -- negligible at
                        # encoder_lr, and it saves a full encoder forward.
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
                    enc_optimizer.zero_grad(set_to_none=True)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_gnorm = float(
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), max_norm=cfg.algo.grad_clip
                    )
                )
                actor_optimizer.step()
                if route_actor:
                    enc_gnorm_actor = float(
                        torch.nn.utils.clip_grad_norm_(
                            enc_params, max_norm=cfg.encoder.grad_clip
                        )
                    )
                    enc_optimizer.step()
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
                + (
                    f"  enc_gnorm(c/a)={enc_gnorm_critic:.2f}/{enc_gnorm_actor:.2f}"
                    f"  emb_std_ratio={collapse.ratio:.2f}"
                    if encoder_trains
                    else ""
                )
            )
            if use_wandb:
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
                    "global_step": global_step,
                }
                if cfg.algo.autotune_alpha:
                    log["train/alpha_loss"] = alpha_loss.item()
                if encoder_trains:
                    log["train/encoder_grad_norm_critic"] = enc_gnorm_critic
                    log["train/encoder_grad_norm_actor"] = enc_gnorm_actor
                    log["train/encoder_drift_l2"] = encoder_drift()
                    log["train/embedding_std"] = collapse.current
                    log["train/embedding_std_ratio"] = collapse.ratio
                    if aux_total is not None:
                        log["train/aux_loss"] = aux_total.item()
                        for name, value in aux_values.items():
                            log[f"train/aux/{name}"] = value
                wandb.log(log, step=global_step)

        # ---- periodic evaluation ----------------------------------------------------
        if global_step % cfg.algo.eval_frequency == 0:
            stats = evaluate(
                actor,
                eval_env,
                cfg.algo.n_episodes_test,
                device,
                encode_fn=encode_single if pixels_in_buffer else None,
                success_fn=_success_fn_for(cfg.env.id),
            )
            print(
                f"  step {global_step:>8,}  EVAL return={stats['return_mean']:.1f}"
                f"±{stats['return_std']:.1f}"
                + (
                    f"  success={stats['success_rate']:.2%}"
                    if "success_rate" in stats
                    else ""
                )
            )
            if use_wandb:
                log = {
                    "eval/mean_reward": stats["return_mean"],
                    "eval/std_reward": stats["return_std"],
                    "global_step": global_step,
                }
                if "success_rate" in stats:
                    log["eval/success_rate"] = stats["success_rate"]
                wandb.log(
                    log,
                    step=global_step,
                )
            if stats["return_mean"] > best_eval_return:
                best_eval_return = stats["return_mean"]
                if cfg.save.enabled:
                    torch.save(
                        actor.state_dict(), os.path.join(save_dir, "best_actor.pt")
                    )
                    if encoder_trains:
                        torch.save(
                            embedding_net.state_dict(),
                            os.path.join(save_dir, "best_encoder.pt"),
                        )

    # ── Final save ────────────────────────────────────────────────────────────
    if cfg.save.enabled:
        ckpt = {
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if encoder_trains:
            ckpt["encoder"] = embedding_net.state_dict()
        torch.save(ckpt, os.path.join(save_dir, "final_checkpoint.pt"))
        print(f"\nSaved to: {save_dir}")
    print(f"Best eval return: {best_eval_return:.1f}")

    env.close()
    eval_env.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
