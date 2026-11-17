"""
Env builders, evaluation, and SAC networks for the explicit training loops
(train_sac.py; the env builders and evaluate() are shared with
train_drqv2.py). The replay buffer lives in src/rl/buffer.py. Split out of train_sac.py to keep that file to the training loop
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


class ActionRepeat(gym.Wrapper):
    """Repeat each action `repeat` times, summing the rewards and stopping
    early if the episode ends (DrQ-v2 uses 2 on DMC). The builders apply it
    to the base env, below any rendering wrapper, so only the frame the agent
    actually sees is rendered."""

    def __init__(self, env, repeat: int):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info


def _repeat(env, action_repeat: int):
    return ActionRepeat(env, action_repeat) if action_repeat > 1 else env


def _register_env_ids(env_id: str) -> None:
    """Import the package that registers env_id, if Gymnasium doesn't already
    know it (Pendulum, BipedalWalker: built in, so no MuJoCo/GL import)."""
    if "dm_control" in env_id:
        import shimmy  # noqa: F401  (registers "dm_control/*-v0" ids)
    elif "Gym-Gridworlds" in env_id:
        import gym_gridworlds  # noqa: F401
    elif "Meta-World" in env_id:
        import mujoco
        if int(mujoco.__version__.split(".")[0]) >= 4:
            raise ImportError(
                f"Meta-World 3.0.0 needs MuJoCo 3 (found {mujoco.__version__}): under MuJoCo 4 "
                "building its envs fails in reset_mocap_welds. Install the pinned version: "
                "pip install mujoco==3.8.1 (requirements.txt)")
        import metaworld  # noqa: F401  (registers "Meta-World/MT1" etc.)
    elif env_id not in gym.registry:
        import gymnasium_robotics  # noqa: F401  (registers AdroitHand*/FrankaKitchen ids)


class _MetaWorldAdapter(gym.Wrapper):
    """Two fixes for Meta-World (3.0.0):
      - it ignores reset's seed (its envs are seeded with an old-style seed()
        method), and its RandomTaskSelectWrapper draws the next task -- the
        object/goal layout -- from that unseeded generator, so reset(seed=s)
        doesn't determine the layout. A seeded reset here calls seed(s) first:
        the same seed gives the same layout, as fixed evaluation seeds need.
        Unseeded resets (training) are unchanged.
      - MT1 records episode statistics itself (info["episode"]), which collides
        with make_env's RecordEpisodeStatistics; its entry is dropped, so
        episodes are recorded once, like every other env's (after action
        repeat)."""

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.unwrapped.seed(int(seed))
        return self.env.reset(options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "episode" in info:
            info = {k: v for k, v in info.items() if k != "episode"}
        return obs, reward, terminated, truncated, info


class _OwnGLContext(gym.Wrapper):
    """Gymnasium's MujocoRenderer (Adroit, Meta-World, FrankaKitchen) makes its
    GL context current only when one env holds several viewers, and its
    OffScreenViewer.render never does. With a training and an evaluation env
    in one process, the context created last stays current and the other env
    renders through it -- wrong frames, without any error. Make this env's own
    context current before every render. Does nothing for envs without that
    renderer (dm_control makes its context current itself; Pendulum,
    BipedalWalker render without MuJoCo)."""

    def render(self):
        renderer = getattr(self.env.unwrapped, "mujoco_renderer", None)
        viewer = renderer._viewers.get("rgb_array") if renderer is not None else None
        if viewer is not None:
            viewer.make_context_current()
        return self.env.render()


def _make(env_id: str, **kwargs):
    env = gym.make(env_id, **kwargs)
    return _MetaWorldAdapter(env) if "Meta-World" in env_id else env


def _wrap(env, wrappers):
    """cfg.env.wrappers: observation wrappers by import path, applied right
    after gym.make (e.g. gym_gridworlds' MatrixWithGoalWrapper, so a state
    observation includes FourRooms' random goal)."""
    import importlib
    for path in wrappers or ():
        module, name = str(path).rsplit(".", 1)
        env = getattr(importlib.import_module(module), name)(env)
    return env


def _make_dmc(env_id: str, image_size: int, action_repeat: int = 1, wrappers=(), **env_kwargs):
    from gymnasium.wrappers import AddRenderObservation

    _register_env_ids(env_id)
    env = _wrap(gym.make(
        env_id,
        render_mode="rgb_array",
        render_kwargs=dict(height=image_size, width=image_size, camera_id=0),
        **env_kwargs,
    ), wrappers)
    return AddRenderObservation(_repeat(env, action_repeat), render_only=True)


def _make_carracing(env_id: str, image_size: int, action_repeat: int = 1, wrappers=(), **env_kwargs):
    return _repeat(_wrap(_make(env_id, render_mode="rgb_array", **env_kwargs), wrappers), action_repeat)


def _make_gym(env_id: str, image_size: int, action_repeat: int = 1, wrappers=(), **env_kwargs):
    """Generic pixel builder for envs that don't natively emit pixel
    observations and don't support dm_control's render_kwargs convention
    (Adroit hand envs, FrankaKitchen, Pendulum, BipedalWalker, FourRooms):
    render_mode="rgb_array" + swap the observation for the rendered frame via
    AddRenderObservation, then resize down to image_size. Unlike dm_control,
    these envs don't expose a height/width render kwarg, so rendering itself
    always happens at the env's native camera resolution (480x480 for Adroit)
    -- ResizeObservation shrinks the frame post-render, before it reaches the
    replay buffer. Without this, finetune mode (raw pixels in the buffer)
    stores every frame at native resolution regardless of image_size, which
    is both wasted memory (EmbeddingNet resizes to the encoder's input size
    anyway) and, at buffer_size=1_000_000, enough to OOM (480x480x3 x 1M
    = 691 GB). Frames that aren't square (BipedalWalker's 600x400) are
    squashed: the encoders' center crop would otherwise cut off the walker,
    which sits left of center.
    """
    from gymnasium.wrappers import AddRenderObservation, ResizeObservation

    _register_env_ids(env_id)
    env = _repeat(_wrap(_make(env_id, render_mode="rgb_array", **env_kwargs), wrappers), action_repeat)
    env = AddRenderObservation(_OwnGLContext(env), render_only=True)
    return ResizeObservation(env, (image_size, image_size))


def _make_state(env_id: str, image_size: int, action_repeat: int = 1, wrappers=(), **env_kwargs):
    _register_env_ids(env_id)
    # FlattenObservation: dm_control/FrankaKitchen/etc emit Dict observations
    # (separate proprioceptive components) natively -- ReplayBuffer needs a
    # flat obs_shape, and a Dict space has no .shape at all. Box-observation
    # envs (e.g. Pendulum-v1) pass through this wrapper as a no-op.
    return gym.wrappers.FlattenObservation(_repeat(_wrap(_make(env_id, **env_kwargs), wrappers), action_repeat))


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


def env_kwargs(cfg) -> dict:
    """cfg.env.kwargs: extra arguments for gym.make (e.g. FourRooms' random_goals)."""
    from omegaconf import OmegaConf
    kw = cfg.env.get("kwargs")
    return OmegaConf.to_container(kw, resolve=True) if OmegaConf.is_config(kw) else dict(kw or {})


def make_env(cfg: DictConfig, embedding_net=None, wrap_encoder: bool = True):
    """
    wrap_encoder=True  -> frozen fast path: env emits feature vectors.
    wrap_encoder=False -> finetune path: env emits raw uint8 HWC pixels;
                          encoding happens inside the training loop.

    frame_stack (cfg.embedding.get("frame_stack", 1)): stacks N consecutive
    frames into the channel axis, (H, W, 3N), before anything else sees
    them. How the encoder uses the stack is its own business
    (embedding.frame_fusion, see EmbeddingNet): `random` can take it as one
    wide input; pretrained backbones encode each frame and fuse the outputs.
    """
    env = ENV_BUILDERS[cfg.env.builder](
        cfg.env.id, cfg.env.get("image_size", 84),
        action_repeat=int(cfg.algo.get("action_repeat", 1)),
        wrappers=list(cfg.env.get("wrappers") or ()),
        **env_kwargs(cfg),
    )
    if cfg.env.get("rotate_180", False):
        # Cameras mounted upside down (Meta-World's corner cameras: their up
        # axis points down) render frames rotated 180 degrees. Undoing that
        # takes a rotation: a vertical flip alone leaves them mirrored.
        env = gym.wrappers.TransformObservation(
            env, lambda o: np.ascontiguousarray(o[::-1, ::-1]), env.observation_space)
    frame_stack = cfg.embedding.get("frame_stack", 1) if cfg.env.builder != "state" else 1
    if frame_stack > 1 and cfg.embedding.get("name") is not None:
        env = gym.wrappers.FrameStackObservation(env, frame_stack)
        env = _StackToChannels(env, frame_stack)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if embedding_net is not None and wrap_encoder:
        env = FrozenEncoderWrapper(env, embedding_net, amp_bf16=cfg.perf.amp_bf16)
    return env


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
    if "Meta-World" in env_id:
        # info["success"] is per step (float 0/1); an episode succeeds if the
        # task was ever completed, Meta-World's own evaluation convention.
        return lambda infos: float(any(i.get("success", 0.0) > 0 for i in infos))
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


def evaluate(act_fn, env, n_episodes: int, device, encode_fn=None, success_fn=None, seeds=None) -> dict:
    """act_fn: (1, feat_dim) tensor -> (1, action_dim) deterministic action
    (SAC: the mean action; DrQ-v2: the policy mean).
    encode_fn: obs -> feature tensor (pixel path, where the env emits
    pixels); None when the env already emits features/state vectors.
    success_fn: see _success_fn_for(); None means no success-rate tracking
    (that key is omitted from the returned dict entirely).
    seeds: one reset seed per episode, to evaluate on fixed episodes; None
    continues the env's own random stream."""
    if seeds is not None and len(seeds) != n_episodes:
        raise ValueError(f"evaluate: {len(seeds)} seeds for {n_episodes} episodes")
    returns = []
    successes = [] if success_fn is not None else None
    for i in range(n_episodes):
        obs, _ = env.reset(seed=None if seeds is None else seeds[i])
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
                action = act_fn(obs_t)
            action = action.squeeze(0).cpu().numpy()
            # A discrete action comes out 0-d: pass a plain int (gym_gridworlds
            # looks actions up in a dict, which an ndarray can't key).
            obs, reward, term, trunc, info = env.step(action.item() if action.ndim == 0 else action)
            ep_return += float(reward)
            if ep_infos is not None:
                ep_infos.append(info)
            done = term or trunc
        returns.append(ep_return)
        if successes is not None:
            successes.append(success_fn(ep_infos))
    stats = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }
    if successes is not None:
        stats["success_rate"] = float(np.mean(successes))
    return stats
