"""
PyTorch Dataset for V-D4RL (Visual D4RL) — pixel-based offline RL datasets.
HuggingFace repo: conglu/vd4rl

Files are NPZ shards: vd4rl/main/{task}/{quality}/{resolution}px/{id}-501.npz
Each shard is one episode (~501 steps). Downloaded via snapshot_download and cached.

Available tasks:    cheetah_run, walker_walk, reacher_easy, cartpole_swingup
Available quality:  expert, medium, medium_replay, medium_expert
Available res:      64  (84px exists only in the distracting variant)

Usage:
    ds = VD4RLDataset('cheetah_run', 'expert')
    obs, action = ds[0]   # (3, 64, 64) uint8, (6,) float32
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

_TASK_TO_ENV = {
    'cheetah_run':      'dm_control/cheetah-run-v0',
    'walker_walk':      'dm_control/walker-walk-v0',
    'reacher_easy':     'dm_control/reacher-easy-v0',
    'cartpole_swingup': 'dm_control/cartpole-swingup-v0',
}


def task_to_env_id(task: str) -> str:
    if task not in _TASK_TO_ENV:
        raise ValueError(f"Unknown V-D4RL task '{task}'. Known: {list(_TASK_TO_ENV)}")
    return _TASK_TO_ENV[task]


class VD4RLDataset(Dataset):
    REPO = "conglu/vd4rl"

    def __init__(self, task: str, quality: str, resolution: int = 64,
                 max_episodes: int = None, cache_dir: str = None):
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("pip install huggingface_hub")

        prefix = f"vd4rl/main/{task}/{quality}/{resolution}px"
        print(f"  Downloading {self.REPO}/{prefix} ...")

        local_dir = snapshot_download(
            repo_id=self.REPO,
            repo_type="dataset",
            allow_patterns=[f"{prefix}/*.npz"],
            cache_dir=cache_dir,
        )

        shards = sorted(glob.glob(os.path.join(local_dir, prefix, "*.npz")))
        if not shards:
            raise FileNotFoundError(
                f"No NPZ files found under {prefix}. "
                f"Check task/quality — available resolutions may differ per split."
            )

        if max_episodes is not None:
            shards = shards[:max_episodes]

        print(f"  Loading {len(shards)} episode shards ...")
        obs_list, act_list = [], []

        # Peek at the first shard to learn key names
        sample = np.load(shards[0])
        obs_key = self._find_key(sample, ('observation', 'obs', 'observations', 'image'))
        act_key = self._find_key(sample, ('action', 'actions'))
        print(f"  NPZ keys — obs: '{obs_key}', action: '{act_key}'")

        for path in shards:
            data = np.load(path)
            obs_list.append(data[obs_key])
            act_list.append(data[act_key])

        obs_all = np.concatenate(obs_list, axis=0)
        act_all = np.concatenate(act_list, axis=0)

        # V-D4RL stores images as (N, H, W, C) — convert to (N, C, H, W)
        if obs_all.ndim == 4 and obs_all.shape[-1] in (1, 3):
            obs_all = obs_all.transpose(0, 3, 1, 2)

        self._obs     = obs_all.astype(np.uint8)
        self._actions = act_all.astype(np.float32)
        self.obs_shape  = self._obs.shape[1:]
        self.action_dim = self._actions.shape[1]

        print(f"  {len(self._obs):,} steps | obs {self.obs_shape} → action ({self.action_dim},)")

    @staticmethod
    def _find_key(data, candidates):
        for key in candidates:
            if key in data:
                return key
        raise KeyError(f"None of {candidates} found. Keys: {list(data.keys())}")

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, idx):
        return (torch.from_numpy(np.array(self._obs[idx])),
                torch.from_numpy(self._actions[idx]))
