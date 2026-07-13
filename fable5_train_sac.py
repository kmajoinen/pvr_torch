"""
Soft Actor-Critic on frozen pretrained visual features (PVRs), written as an
explicit training loop in the style of train_bc.py.

The SAC update math is adapted from CleanRL's sac_continuous_action.py
(https://github.com/vwxyzjn/cleanrl, MIT license), which is benchmarked
against reference results -- the tanh-Gaussian log-prob correction, twin-Q
target, and automatic entropy tuning are kept verbatim. Nothing from CleanRL
needs to be installed; this file is self-contained.

Every intermediate quantity (Q losses, Q values, actor loss, alpha, entropy)
is computed in plain sight in the loop below -- add a key to the wandb.log
dict at the point where the tensor exists.

Design notes:
  - The frozen encoder lives in an env wrapper (FrozenEncoderWrapper), so the
    replay buffer stores compact feature vectors and gradient updates never
    touch the encoder. Encoder finetuning is deliberately out of scope here
    (it would need the encoder inside the critic with DrQ-style gradient
    routing); use the SB3 script's finetune path for that.
  - Actor and critics each apply their own LayerNorm to the input features:
    raw frozen-encoder features have large arbitrary scale, which otherwise
    blows up mean/log_std within the first gradient steps.
  - Time limits are handled correctly: bootstrapping is masked on
    `terminated` only, so truncation-only envs (dm_control, 1000-step
    episodes) bootstrap through the time limit as they should.

Usage:
    python train_sac.py                                    # defaults from configs/
    python train_sac.py embedding=resnet18 algo.total_timesteps=500000
    python train_sac.py env=pendulum embedding=none        # state-based sanity check
    python train_sac.py algo.batch_size=128 wandb.enabled=true
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

from fable5_feature_extractor import FrozenEncoderWrapper

LOG_STD_MAX = 2
LOG_STD_MIN = -5


# ------------------------------------------------------------------------------
# Environments
# ------------------------------------------------------------------------------

def _make_dmc(env_id: str, image_size: int):
    import shimmy  # noqa: F401  (registers "dm_control/*-v0" ids)
    from gymnasium.wrappers import AddRenderObservation

    # Render natively at the encoder's input size: no 84->224 upsampling pass.
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        render_kwargs=dict(height=image_size, width=image_size, camera_id=0),
    )
    return AddRenderObservation(env, render_only=True)


def _make_carracing(env_id: str, image_size: int):
    return gym.make(env_id, render_mode="rgb_array")


def _make_state(env_id: str, image_size: int):
    return gym.make(env_id)


ENV_BUILDERS = {"dmc": _make_dmc, "carracing": _make_carracing, "state": _make_state}


def make_env(cfg: DictConfig, embedding_net=None):
    """embedding_net: src.embeddings.EmbeddingNet (shared instance) or None for state envs."""
    env = ENV_BUILDERS[cfg.env.builder](cfg.env.id, cfg.env.get("image_size", 84))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if embedding_net is not None:
        env = FrozenEncoderWrapper(env, embedding_net, amp_bf16=cfg.perf.amp_bf16)
    return env


# ------------------------------------------------------------------------------
# Replay buffer (flat float32 observations -- features or state vectors)
# ------------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminations = np.zeros(capacity, dtype=np.float32)
        self.pos, self.full = 0, False

    def add(self, obs, next_obs, action, reward, terminated):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminations[self.pos] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self), size=batch_size)
        to = lambda x: torch.as_tensor(x, device=self.device)
        return (to(self.obs[idx]), to(self.next_obs[idx]), to(self.actions[idx]),
                to(self.rewards[idx]), to(self.terminations[idx]))


# ------------------------------------------------------------------------------
# Networks (CleanRL SAC, plus input LayerNorm for large-scale PVR features)
# ------------------------------------------------------------------------------

class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: list):
        super().__init__()
        self.obs_norm = nn.LayerNorm(obs_dim)
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)

    def forward(self, obs, action):
        x = torch.cat([self.obs_norm(obs), action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: list, action_space):
        super().__init__()
        self.obs_norm = nn.LayerNorm(obs_dim)
        self.fc1 = nn.Linear(obs_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc_mean = nn.Linear(hidden[1], action_dim)
        self.fc_logstd = nn.Linear(hidden[1], action_dim)
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        x = self.obs_norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()                       # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # tanh-squash correction to the Gaussian log-prob
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def evaluate(actor: Actor, env, n_episodes: int, device) -> dict:
    actor.eval()
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return, done = 0.0, False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, mean_action = actor.get_action(obs_t)   # deterministic
            obs, reward, term, trunc, _ = env.step(mean_action.squeeze(0).cpu().numpy())
            ep_return += float(reward)
            done = term or trunc
        returns.append(ep_return)
    actor.train()
    return {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns))}


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config_sac_fable5", version_base=None)
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

    # ── Weights & Biases ──────────────────────────────────────────────────────
    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        run_name = cfg.wandb.run_name or \
            f"sac_{cfg.env.name}_{cfg.embedding.get('name') or 'state'}_s{cfg.seed}"
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            mode=cfg.wandb.get("mode", "online"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"wandb run: {wandb.run.url}")

    # ── Envs (+ optional frozen encoder) ──────────────────────────────────────
    emb_name = cfg.embedding.get("name", None)
    embedding_net = None
    if emb_name is not None:
        from src.embeddings import EmbeddingNet
        # train=False => frozen; this loop only supports the frozen protocol
        # (finetuning belongs in the SB3 PVRFeaturesExtractor path).
        embedding_net = EmbeddingNet(
            emb_name, pretrained=True, train=False,
            disable_cuda=(str(device) == "cpu"),
        )
        embedding_net.eval()
        print(f"\nEmbedding: {emb_name}  →  obs_size: {embedding_net.out_size}")

    env = make_env(cfg, embedding_net)       # sharing one EmbeddingNet instance
    eval_env = make_env(cfg, embedding_net)  # is safe: pure inference, no state
    assert isinstance(env.action_space, gym.spaces.Box), "continuous actions only"
    env.action_space.seed(cfg.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # ── Networks / optimizers ─────────────────────────────────────────────────
    hidden = list(cfg.algo.net_arch)
    actor = Actor(obs_dim, action_dim, hidden, env.action_space).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim, hidden).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim, hidden).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim, hidden).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim, hidden).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.algo.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.algo.policy_lr)

    # automatic entropy tuning
    if cfg.algo.autotune_alpha:
        target_entropy = -float(action_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=cfg.algo.q_lr)
    else:
        alpha = cfg.algo.alpha

    n_params = sum(p.numel() for p in list(actor.parameters()) + list(qf1.parameters()))
    print(f"Actor: obs({obs_dim}) → action({action_dim})  |  actor+critic ≈ {n_params:,} params")

    rb = ReplayBuffer(cfg.algo.buffer_size, obs_dim, action_dim, device)
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    best_eval_return = -float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {cfg.algo.total_timesteps:,} steps "
          f"(learning starts at {cfg.algo.learning_starts:,})")
    obs, _ = env.reset(seed=cfg.seed)
    t0 = time.time()

    for global_step in range(1, cfg.algo.total_timesteps + 1):
        # ---- act --------------------------------------------------------------
        if global_step <= cfg.algo.learning_starts:
            action = env.action_space.sample()
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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
                print(f"  step {global_step:>8,}  episodic_return={float(ep['r']):8.1f}")
                if use_wandb:
                    wandb.log({"charts/episodic_return": float(ep["r"]),
                               "charts/episodic_length": int(ep["l"])}, step=global_step)
            obs, _ = env.reset()
        else:
            obs = next_obs

        if global_step <= cfg.algo.learning_starts:
            continue

        # ---- critic update ------------------------------------------------------
        b_obs, b_next_obs, b_actions, b_rewards, b_terms = rb.sample(cfg.algo.batch_size)

        with torch.no_grad():
            next_actions, next_log_pi, _ = actor.get_action(b_next_obs)
            qf1_next = qf1_target(b_next_obs, next_actions)
            qf2_next = qf2_target(b_next_obs, next_actions)
            min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
            next_q_value = b_rewards + (1 - b_terms) * cfg.algo.gamma * min_qf_next.view(-1)

        qf1_a_values = qf1(b_obs, b_actions).view(-1)
        qf2_a_values = qf2(b_obs, b_actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # ---- actor (+ alpha) update, delayed ------------------------------------
        if global_step % cfg.algo.policy_frequency == 0:
            # compensate for the delay by doing policy_frequency updates
            for _ in range(cfg.algo.policy_frequency):
                pi, log_pi, _ = actor.get_action(b_obs)
                qf1_pi = qf1(b_obs, pi)
                qf2_pi = qf2(b_obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = (alpha * log_pi - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if cfg.algo.autotune_alpha:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(b_obs)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # ---- polyak target update -----------------------------------------------
        if global_step % cfg.algo.target_network_frequency == 0:
            tau = cfg.algo.tau
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # ---- logging --------------------------------------------------------------
        if global_step % cfg.algo.log_frequency == 0:
            sps = int(global_step / (time.time() - t0))
            print(f"  step {global_step:>8,}  qf_loss={qf_loss.item():8.3f}"
                  f"  q1={qf1_a_values.mean().item():7.2f}"
                  f"  actor_loss={actor_loss.item():8.3f}"
                  f"  alpha={alpha:.3f}  SPS={sps}")
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
                    "charts/SPS": sps,
                }
                if cfg.algo.autotune_alpha:
                    log["train/alpha_loss"] = alpha_loss.item()
                wandb.log(log, step=global_step)

        # ---- periodic evaluation ----------------------------------------------------
        if global_step % cfg.algo.eval_frequency == 0:
            stats = evaluate(actor, eval_env, cfg.algo.n_episodes_test, device)
            print(f"  step {global_step:>8,}  EVAL return={stats['return_mean']:.1f}"
                  f"±{stats['return_std']:.1f}")
            if use_wandb:
                wandb.log({"eval/return_mean": stats["return_mean"],
                           "eval/return_std": stats["return_std"]}, step=global_step)
            if cfg.save.enabled and stats["return_mean"] > best_eval_return:
                best_eval_return = stats["return_mean"]
                torch.save(actor.state_dict(), os.path.join(save_dir, "best_actor.pt"))

    # ── Final save ────────────────────────────────────────────────────────────
    if cfg.save.enabled:
        torch.save({
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, os.path.join(save_dir, "final_checkpoint.pt"))
        print(f"\nSaved to: {save_dir}")
    print(f"Best eval return: {best_eval_return:.1f}")

    env.close()
    eval_env.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
