import torch
import numpy as np
from src.gym_wrappers import make_gym_env


def make_environment(cfg):
    """
    Build an environment from a hydra config node.
    cfg must have an 'env' sub-config with at least 'id',
    and top-level 'train_from_pixels' and optionally 'num_actions'.
    """
    env_kwargs = dict(cfg.env)
    return make_gym_env(
        train_from_pixels=cfg.get("train_from_pixels", False),
        num_actions=cfg.get("num_actions", None),
        **env_kwargs,
    )


class Environment:
    """
    Thin stateful wrapper around a gymnasium env for evaluation loops.
    Tracks per-episode return and step count; handles auto-reset on done.
    The embedding (if any) is applied externally by the caller.
    """

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = 0.0
        self.episode_step = 0

    def reset(self, seed=None):
        obs, info = self.gym_env.reset(seed=seed)
        self.episode_return = 0.0
        self.episode_step = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.gym_env.step(action)
        done = terminated or truncated
        self.episode_return += float(reward)
        self.episode_step += 1

        if done:
            obs, _ = self.gym_env.reset()
            self.episode_return = 0.0
            self.episode_step = 0

        return obs, reward, done, info

    def close(self):
        self.gym_env.close()

    @property
    def observation_space(self):
        return self.gym_env.observation_space

    @property
    def action_space(self):
        return self.gym_env.action_space
