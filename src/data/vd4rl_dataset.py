"""
PyTorch Dataset for V-D4RL (Visual D4RL) — pixel-based offline RL datasets.
HuggingFace repo: conglu/vd4rl

Observations are (C, H, W) uint8 — same format as PickleDataset / gym_wrappers.
Actions are continuous float32.

Available tasks:    cheetah_run, walker_walk, reacher_easy, cartpole_swingup
Available quality:  expert, medium, medium_replay, medium_expert
Available res:      64, 84  (pixels)

Usage:
    ds = VD4RLDataset('cheetah_run', 'expert', resolution=84)
    obs, action = ds[0]   # (3, 84, 84) uint8, (6,) float32
"""

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

    def __init__(self, task: str, quality: str, resolution: int = 84,
                 seed: int = 1, max_episodes: int = None, cache_dir: str = None):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("pip install huggingface_hub")
        try:
            import zarr
        except ImportError:
            raise ImportError("pip install zarr")

        filename = f"{task}/{quality}/{resolution}px/{seed}.zarr"
        print(f"  Downloading {self.REPO}/{filename} ...")
        path = hf_hub_download(
            repo_id=self.REPO,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )

        # zarr files on HuggingFace may be a zip store or a directory store
        try:
            store = zarr.ZipStore(path, mode='r')
            data  = zarr.open(store, mode='r')
        except Exception:
            data = zarr.open(path, mode='r')

        # Print keys on first open to help debug structure mismatches
        print(f"  zarr keys: {list(data.keys())}")

        obs  = self._load_key(data, ('observation', 'obs', 'observations'))
        acts = self._load_key(data, ('action', 'actions'))

        # V-D4RL stores images as (N, H, W, C) — convert to (N, C, H, W)
        if obs.ndim == 4 and obs.shape[-1] in (1, 3):
            obs = obs.transpose(0, 3, 1, 2)

        if max_episodes is not None:
            # Approximate episode boundary by terminal/done flags
            try:
                done = self._load_key(data, ('terminal', 'done', 'dones')).astype(bool)
                ends = np.where(done)[0][:max_episodes]
                cut  = int(ends[-1]) + 1 if len(ends) else len(obs)
                obs, acts = obs[:cut], acts[:cut]
            except KeyError:
                obs  = obs[:max_episodes * 1000]
                acts = acts[:max_episodes * 1000]

        self._obs     = obs.astype(np.uint8)
        self._actions = acts.astype(np.float32)
        self.obs_shape  = self._obs.shape[1:]       # (C, H, W)
        self.action_dim = self._actions.shape[1]

        print(f"  {len(self._obs):,} steps | obs {self.obs_shape} → action ({self.action_dim},)")

    @staticmethod
    def _load_key(data, candidates):
        for key in candidates:
            if key in data:
                return np.array(data[key])
        raise KeyError(f"None of {candidates} found in zarr. Keys: {list(data.keys())}")

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, idx):
        return (torch.from_numpy(np.array(self._obs[idx])),
                torch.from_numpy(self._actions[idx]))
