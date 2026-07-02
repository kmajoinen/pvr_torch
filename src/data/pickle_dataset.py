"""
PyTorch Dataset wrapper around trajectory pickles produced by
behavioral_cloning/save_opt_trajectories.py.

Each pickle has keys: obs, action, reward, done
where each value is a list of per-episode numpy arrays.
Pixel obs are (T, C, H, W) uint8; state obs are (T, obs_dim) float64.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class PickleDataset(Dataset):
    def __init__(self, path: str, max_episodes: int = None):
        data = pickle.load(open(path, 'rb'))
        obs_list = data['obs']
        act_list = data['action']

        if max_episodes is not None:
            obs_list = obs_list[:max_episodes]
            act_list = act_list[:max_episodes]

        self._obs = np.concatenate(obs_list, axis=0)
        acts = np.concatenate(act_list, axis=0)
        self._actions = acts.astype(np.float32) if acts.ndim > 1 else acts[:, None].astype(np.float32)

        self.obs_shape = self._obs.shape[1:]
        self.action_dim = self._actions.shape[1]

        print(f"  loaded {len(obs_list)} episodes | "
              f"{len(self._obs):,} steps | "
              f"obs {self.obs_shape} → action ({self.action_dim},)")

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self._obs[idx])), torch.from_numpy(self._actions[idx])
