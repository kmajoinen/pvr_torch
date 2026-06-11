import numpy as np
import gymnasium
from gymnasium import logger
from gymnasium.wrappers import (
    AtariPreprocessing,
    FlattenObservation,
    AddRenderObservation,
    ResizeObservation,
    TimeLimit,
    TransformObservation,
    DiscretizeAction,
    FilterObservation,
)

# Register environments that need explicit imports
try:
    import gymnasium_robotics  # FrankaKitchen-v1, Adroit, Fetch, PointMaze, ...
except ImportError:
    pass
try:
    import minigrid  # MiniGrid-* envs
except ImportError:
    pass


def get_wrapper_names(env):
    """Return the names of all wrappers applied to an environment."""
    names = []
    current = env
    while isinstance(current, gymnasium.Wrapper):
        names.append(type(current).__name__)
        current = current.env
    return names


# ------------------------------------------------------------------------------
# Custom wrappers
# ------------------------------------------------------------------------------

class ChannelsFirstWrapper(gymnasium.ObservationWrapper):
    """Convert (H, W, C) observations to (C, H, W)."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape  # (H, W, C)
        new_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        self.observation_space = gymnasium.spaces.Box(
            low=env.observation_space.low.transpose((2, 0, 1)),
            high=env.observation_space.high.transpose((2, 0, 1)),
            shape=new_shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


class NormalizeObservationWrapper(gymnasium.ObservationWrapper):
    """Normalize observations by their bounded range."""

    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            logger.warn("Observation space is unbounded; observations will not be normalised.")
            self.obs_bound = np.ones_like(low)
            return
        self.obs_bound = np.maximum(np.abs(low), np.abs(high))
        self.observation_space = gymnasium.spaces.Box(low / self.obs_bound,
                                                      high / self.obs_bound)

    def observation(self, observation):
        return observation / self.obs_bound


# ------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------

def make_gym_env(train_from_pixels: bool, num_actions: int = None, **kwargs):
    """
    Create and wrap a gymnasium environment.

    Args:
        train_from_pixels: if True, pixel observations are returned (C, H, W)
            resized to 84×84; requires render_mode='rgb_array'.
        num_actions: if set and the action space is continuous (Box), discretise
            it into this many bins via DiscretizeAction. Pass None to keep
            continuous actions (required for SAC).
        **kwargs: passed to gymnasium.make() — must include 'id'.
    """
    if train_from_pixels:
        kwargs["render_mode"] = "rgb_array"

    env_name = kwargs["id"]

    # ── Atari ──────────────────────────────────────────────────────────────────
    if "ALE" in env_name:  # e.g. ALE/Breakout-v5
        import ale_py
        gymnasium.register_envs(ale_py)
        env = gymnasium.make(
            **kwargs,
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
            max_num_frames_per_episode=18_000,
            obs_type="rgb",
        )
        env = AtariPreprocessing(env)
        env = gymnasium.wrappers.FrameStackObservation(env, 4)

    # ── MiniGrid ───────────────────────────────────────────────────────────────
    elif "MiniGrid" in env_name:  # e.g. MiniGrid-DoorKey-8x8-v0
        from minigrid import wrappers as minigrid_wrappers
        env = gymnasium.make(**kwargs)
        env = minigrid_wrappers.ImgObsWrapper(env)  # removes 'mission' field

    # ── Everything else ────────────────────────────────────────────────────────
    else:
        # Lazy imports triggered by env name
        if "dm_control" in env_name:  # e.g. dm_control/cheetah-run-v0
            import shimmy
        if "Vizdoom" in env_name:
            from vizdoom import gymnasium_wrapper
        if "MiniWorld" in env_name:
            import miniworld
        if "PyFlyt" in env_name:
            import PyFlyt.gym_envs
        if "Gym-MinAtar" in env_name:
            import gym_minatar
        if "Gym-Gridworlds" in env_name:
            import gym_gridworlds
            from gym_gridworlds.observation_wrappers import (
                MatrixWrapper,
                MatrixWithGoalWrapper,
            )
        if "OGBench" in env_name:
            import ogbench
            env_name_stripped = env_name[len("OGBench/"):]
            kwargs.pop("id")
            env = ogbench.make_env_and_datasets(env_name_stripped, env_only=True, **kwargs)
        else:
            env = gymnasium.make(**kwargs)

        # Discretise continuous actions when requested (e.g. for DQN/BC with discrete policy)
        # Pass num_actions=None to keep continuous actions for SAC etc.
        if isinstance(env.action_space, gymnasium.spaces.Box) and num_actions is not None:
            env = DiscretizeAction(env, bins=num_actions)

        if "PointMaze" in env_name:
            env = FilterObservation(env, ["observation"])

        if train_from_pixels:
            env = AddRenderObservation(env, render_only=True)
            env = ResizeObservation(env, (84, 84))
            env = ChannelsFirstWrapper(env)
        else:
            if "Gym-Gridworlds" in env_name:
                if "Taxi" in env_name or "CleanDirt" in env_name:
                    env = MatrixWithGoalWrapper(env)
                else:
                    env = MatrixWrapper(env)
                env = FlattenObservation(env)
            elif "Gym-MinAtar" in env_name:
                env = ChannelsFirstWrapper(env)
            else:
                env = FlattenObservation(env)

    return env
