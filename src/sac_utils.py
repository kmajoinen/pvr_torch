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


def _make_gym(env_id: str, image_size: int):
    """Generic pixel builder for envs that don't natively emit pixel
    observations and don't support dm_control's render_kwargs convention
    (Adroit hand envs, FrankaKitchen): render_mode="rgb_array" + swap the
    observation for the rendered frame via AddRenderObservation. image_size
    isn't used to control render resolution here -- unlike dm_control,
    these envs don't expose a height/width render kwarg, so this renders at
    whatever the env's default camera/resolution is; EmbeddingNet's own
    transform pipeline resizes to the encoder's input size regardless of
    native render size.
    """
    import gymnasium_robotics  # noqa: F401  (registers AdroitHand*/FrankaKitchen ids)
    from gymnasium.wrappers import AddRenderObservation

    env = gym.make(env_id, render_mode="rgb_array")
    return AddRenderObservation(env, render_only=True)


def _make_state(env_id: str, image_size: int):
    if "dm_control" in env_id:
        import shimmy  # noqa: F401  (registers "dm_control/*-v0" ids)
    elif "Adroit" in env_id or "FrankaKitchen" in env_id:
        import gymnasium_robotics  # noqa: F401  (registers AdroitHand*/FrankaKitchen ids)
    # FlattenObservation: dm_control/FrankaKitchen/etc emit Dict observations
    # (separate proprioceptive components) natively -- ReplayBuffer needs a
    # flat obs_shape, and a Dict space has no .shape at all. Box-observation
    # envs (e.g. Pendulum-v1) pass through this wrapper as a no-op.
    return gym.wrappers.FlattenObservation(gym.make(env_id))


ENV_BUILDERS = {
    "dmc": _make_dmc,
    "carracing": _make_carracing,
    "gym": _make_gym,
    "state": _make_state,
}


class _StackToChannels(gym.ObservationWrapper):
    """
    gymnasium.wrappers.FrameStackObservation stacks along a NEW leading
    axis -- (num_stack, H, W, C), verified against Farama's docs -- not
    concatenated into the channel axis. Every pixel builder in this file
    (and FrozenEncoderWrapper/the raw-pixel buffer path) expects a plain
    (H, W, C) frame, so this merges the stack axis into the channel axis
    right after stacking: (N, H, W, C) -> (H, W, N*C). Everything
    downstream then sees a wider-than-usual but otherwise ordinary frame.
    """

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        n, h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, n * c), dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        n, h, w, c = obs.shape
        return np.asarray(obs).transpose(1, 2, 0, 3).reshape(h, w, n * c)


def make_env(cfg: DictConfig, embedding_net=None, wrap_encoder: bool = True):
    """
    wrap_encoder=True  -> frozen fast path: env emits feature vectors.
    wrap_encoder=False -> finetune path: env emits raw uint8 HWC pixels;
                          encoding happens inside the training loop.

    frame_stack (cfg.embedding.get("frame_stack", 1)): stacks N consecutive
    frames into the channel axis before anything else sees them. Only
    meaningful for embeddings trained from scratch alongside the policy
    (embedding=random) -- pretrained PVR backbones have a fixed 3-channel
    first layer and can't accept a wider input at all, so this is never
    applied for any other embedding regardless of what's set here.
    """
    env = ENV_BUILDERS[cfg.env.builder](cfg.env.id, cfg.env.get("image_size", 84))
    frame_stack = cfg.embedding.get("frame_stack", 1) if cfg.env.builder != "state" else 1
    if frame_stack > 1 and cfg.embedding.get("name") == "random":
        env = gym.wrappers.FrameStackObservation(env, frame_stack)
        env = _StackToChannels(env, frame_stack)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if embedding_net is not None and wrap_encoder:
        env = FrozenEncoderWrapper(env, embedding_net, amp_bf16=cfg.perf.amp_bf16)
    return env


# ------------------------------------------------------------------------------
# Replay buffer (float32 feature/state vectors OR uint8 image frames)
# ------------------------------------------------------------------------------


class ReplayBuffer:
    """
    Stores obs once, not obs+next_obs separately -- next_obs is derived at
    sample time as the following slot in the same circular array (same
    scheme as SB3's ReplayBuffer(optimize_memory_usage=True), verified
    against their actual sampling logic). Halves memory for pixel
    observations, where next_obs is the same data as the following
    transition's obs for every non-episode-boundary transition anyway.

    Never samples index `pos` itself -- its "next" slot is either not
    written yet (buffer not full) or belongs to whatever just overwrote it
    (buffer full), not a valid successor. At true episode boundaries the
    "next_obs" this derives is technically the start of a different
    episode, not a real successor state -- same accepted imprecision as
    SB3's implementation, harmless since terminated transitions never
    bootstrap through next_obs in the loss (masked by 1 - terminated).
    """

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
        gb = capacity * int(np.prod(obs_shape)) * itemsize / 1e9
        print(
            f"Replay buffer: {capacity:,} x {obs_shape} {np.dtype(obs_dtype).name}"
            f"  ({gb:.1f} GB for obs; next_obs derived, not stored separately)"
        )
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminations = np.zeros(capacity, dtype=np.float32)
        self.pos, self.full = 0, False

    def add(self, obs, next_obs, action, reward, terminated):
        # next_obs isn't stored -- the following add() call's obs becomes
        # this transition's next_obs automatically once written to
        # self.pos+1. Still taken as a parameter to keep this a drop-in
        # call-site replacement (the caller always has both handy).
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminations[self.pos] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int):
        if self.full:
            # Offset uniformly in [1, capacity) from pos, so idx is never
            # pos itself and idx+1 is always a genuinely-written slot.
            idx = (self.pos + np.random.randint(1, self.capacity, size=batch_size)) % self.capacity
        else:
            idx = np.random.randint(0, self.pos, size=batch_size)
        next_idx = (idx + 1) % self.capacity
        to = lambda x: torch.as_tensor(x, device=self.device)
        # TODO(augmentation): per-sample random shift/crop (PIE-G / DrQ style)
        # goes here, applied to to(self.obs[idx]) / to(self.obs[next_idx])
        # before returning -- only meaningful when the buffer stores pixels.
        return (
            to(self.obs[idx]),
            to(self.obs[next_idx]),
            to(self.actions[idx]),
            to(self.rewards[idx]),
            to(self.terminations[idx]),
        )


# ------------------------------------------------------------------------------
# Networks (CleanRL SAC, plus input LayerNorm for large-scale PVR features)
# ------------------------------------------------------------------------------


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: list, obs_norm: bool = True):
        super().__init__()
        # Input LayerNorm is for high-dim PVR features only. On a low-dim
        # state vector, per-sample normalization aliases states (LN(x) is
        # invariant to x -> a*x + b*1) and couples every dim through noisy
        # 17-sample statistics -- pass obs_norm=False there.
        self.obs_norm = nn.LayerNorm(obs_dim) if obs_norm else nn.Identity()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)

    def forward(self, obs, action):
        x = torch.cat([self.obs_norm(obs), action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: list, action_space, obs_norm: bool = True):
        super().__init__()
        # See SoftQNetwork: LayerNorm for PVR features, Identity for state.
        self.obs_norm = nn.LayerNorm(obs_dim) if obs_norm else nn.Identity()
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


def _success_fn_for(env_id: str):
    """Per-episode success extractor for envs with a meaningful task-
    completion signal in info, or None for envs without one (return is the
    only universal metric). Dispatches on env_id substring, same convention
    as _make_dmc/_make_state/gym_wrappers.py.

    Takes the list of info dicts collected across one episode's steps,
    returns a single 0..1 success value for that episode.
    """
    if "Adroit" in env_id:
        # info["success"] is a per-step boolean (D4RL convention) -- treat
        # the episode as successful if it was ever True at any point, not
        # just at the final step (matches train_bc.py::evaluate()'s
        # max(ep_success, ...) convention for the same envs).
        return lambda infos: float(any(i.get("success", False) for i in infos))
    if "FrankaKitchen" in env_id:
        # No boolean success field at all -- FrankaKitchen is a composite
        # multi-subtask env. info carries tasks_to_complete (remaining) and
        # episode_task_completions (done so far), both lists. Total subtask
        # count = remaining + done, read at the first step (before anything
        # completes); fraction complete = done at the last step / total.
        def _franka_success(infos):
            total = len(infos[0]["tasks_to_complete"]) + len(infos[0]["episode_task_completions"])
            if total == 0:
                return 0.0
            return len(infos[-1]["episode_task_completions"]) / total

        return _franka_success
    return None


def evaluate(actor: Actor, env, n_episodes: int, device, encode_fn=None, success_fn=None) -> dict:
    """encode_fn: obs -> feature tensor (finetune path, where env emits
    pixels); None when the env already emits features/state vectors.
    success_fn: see _success_fn_for(); None means no success-rate tracking
    (that key is omitted from the returned dict entirely)."""
    actor.eval()
    returns = []
    successes = [] if success_fn is not None else None
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return, done = 0.0, False
        ep_infos = [] if success_fn is not None else None
        while not done:
            if encode_fn is not None:
                obs_t = encode_fn(obs)
            else:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
            with torch.no_grad():
                _, _, mean_action = actor.get_action(obs_t)  # deterministic
            obs, reward, term, trunc, info = env.step(mean_action.squeeze(0).cpu().numpy())
            ep_return += float(reward)
            if ep_infos is not None:
                ep_infos.append(info)
            done = term or trunc
        returns.append(ep_return)
        if successes is not None:
            successes.append(success_fn(ep_infos))
    actor.train()
    stats = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }
    if successes is not None:
        stats["success_rate"] = float(np.mean(successes))
    return stats
