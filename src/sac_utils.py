"""
Env builders, replay buffer, and networks for train_sac.py's explicit SAC
loop. Split out of train_sac.py to keep that file to the training loop
itself -- these pieces are self-contained (no closures over loop-local
training state), so moving them here is a pure reorganization.

The SAC network/update math is adapted from CleanRL's
sac_continuous_action.py (https://github.com/vwxyzjn/cleanrl, MIT license).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from omegaconf import DictConfig

from feature_extractor import FrozenEncoderWrapper

LOG_STD_MAX = 2
LOG_STD_MIN = -5


# ------------------------------------------------------------------------------
# Environments
# ------------------------------------------------------------------------------


def _make_dmc(env_id: str, image_size: int):
    import shimmy  # noqa: F401  (registers "dm_control/*-v0" ids)
    from gymnasium.wrappers import AddRenderObservation

    env = gym.make(
        env_id,
        render_mode="rgb_array",
        render_kwargs=dict(height=image_size, width=image_size, camera_id=0),
    )
    return AddRenderObservation(env, render_only=True)


def _make_carracing(env_id: str, image_size: int):
    return gym.make(env_id, render_mode="rgb_array")


def _make_state(env_id: str, image_size: int):
    # FlattenObservation: dm_control/FrankaKitchen/etc emit Dict observations
    # (separate proprioceptive components) natively -- ReplayBuffer needs a
    # flat obs_shape, and a Dict space has no .shape at all. Box-observation
    # envs (e.g. Pendulum-v1) pass through this wrapper as a no-op.
    return gym.wrappers.FlattenObservation(gym.make(env_id))


ENV_BUILDERS = {"dmc": _make_dmc, "carracing": _make_carracing, "state": _make_state}


def make_env(cfg: DictConfig, embedding_net=None, wrap_encoder: bool = True):
    """
    wrap_encoder=True  -> frozen fast path: env emits feature vectors.
    wrap_encoder=False -> finetune path: env emits raw uint8 HWC pixels;
                          encoding happens inside the training loop.
    """
    env = ENV_BUILDERS[cfg.env.builder](cfg.env.id, cfg.env.get("image_size", 84))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if embedding_net is not None and wrap_encoder:
        env = FrozenEncoderWrapper(env, embedding_net, amp_bf16=cfg.perf.amp_bf16)
    return env


# ------------------------------------------------------------------------------
# Replay buffer (float32 feature/state vectors OR uint8 image frames)
# ------------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_dim: int,
        device,
        obs_dtype=np.float32,
    ):
        self.capacity = capacity
        self.device = device
        # Pre-flight estimate BEFORE allocating, so an oversized
        # buffer_size/image_size combination announces itself instead of
        # dying inside np.zeros.
        itemsize = np.dtype(obs_dtype).itemsize
        gb = 2 * capacity * int(np.prod(obs_shape)) * itemsize / 1e9
        print(
            f"Replay buffer: {capacity:,} x {obs_shape} {np.dtype(obs_dtype).name}"
            f"  ({gb:.1f} GB for obs+next_obs)"
        )
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
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
        # TODO(augmentation): per-sample random shift/crop (PIE-G / DrQ style)
        # goes here, applied to to(self.obs[idx]) / to(self.next_obs[idx])
        # before returning -- only meaningful when the buffer stores pixels.
        return (
            to(self.obs[idx]),
            to(self.next_obs[idx]),
            to(self.actions[idx]),
            to(self.rewards[idx]),
            to(self.terminations[idx]),
        )


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
            torch.tensor(
                (action_space.high - action_space.low) / 2.0, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0, dtype=torch.float32
            ),
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
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------


def evaluate(actor: Actor, env, n_episodes: int, device, encode_fn=None) -> dict:
    """encode_fn: obs -> feature tensor (finetune path, where env emits
    pixels); None when the env already emits features/state vectors."""
    actor.eval()
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return, done = 0.0, False
        while not done:
            if encode_fn is not None:
                obs_t = encode_fn(obs)
            else:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
            with torch.no_grad():
                _, _, mean_action = actor.get_action(obs_t)  # deterministic
            obs, reward, term, trunc, _ = env.step(mean_action.squeeze(0).cpu().numpy())
            ep_return += float(reward)
            done = term or trunc
        returns.append(ep_return)
    actor.train()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }
