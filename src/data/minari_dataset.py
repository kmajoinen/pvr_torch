"""
PyTorch Dataset wrapper around Minari offline RL datasets.

Supports state observations (returned as-is) and pixel observations
(returned as (C, H, W) uint8 for downstream embedding).

Usage:
    from src.data.minari_dataset import MinariDataset
    ds = MinariDataset('D4RL/door/expert-v2')
    obs, action = ds[0]          # (39,) float32, (28,) float32
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MinariDataset(Dataset):
    """
    Flattens all episodes into a single (obs, action) dataset.

    Args:
        dataset_id:    Minari dataset identifier, e.g. 'D4RL/door/expert-v2'.
        max_episodes:  Cap the number of episodes loaded (None = all).
        obs_key:       Key into episode.observations if it is a dict.
                       Leave None for array observations.
    """

    def __init__(self, dataset_id: str, max_episodes: int = None, obs_key: str = None):
        import minari
        minari_ds = minari.load_dataset(dataset_id)

        obs_list, act_list = [], []
        for i, ep in enumerate(minari_ds.iterate_episodes()):
            if max_episodes is not None and i >= max_episodes:
                break

            obs = ep.observations  # (T+1, *obs_shape) or dict
            if isinstance(obs, dict):
                assert obs_key is not None, \
                    f"obs is a dict with keys {list(obs.keys())}; pass obs_key="
                obs = obs[obs_key]

            # observations has one more entry than actions (initial obs included)
            obs_list.append(obs[:-1].astype(np.float32))
            act_list.append(ep.actions.astype(np.float32))

        self._obs = np.concatenate(obs_list, axis=0)      # (N, *obs_shape)
        self._actions = np.concatenate(act_list, axis=0)  # (N, action_dim)

        self.obs_shape = self._obs.shape[1:]
        self.action_dim = self._actions.shape[1]
        print(f"  loaded {len(obs_list)} episodes | "
              f"{len(self._obs):,} steps | "
              f"obs {self.obs_shape} → action ({self.action_dim},)")

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, idx):
        obs    = torch.from_numpy(self._obs[idx])
        action = torch.from_numpy(self._actions[idx])
        return obs, action
