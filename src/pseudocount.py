"""
Count-based exploration utilities used by ReplayMemory.

`_close_mask` returns a boolean mask over stored observations that fall
within a given L2 radius of a query observation. Used to maintain
approximate visit counts in embedding space.
"""

import numpy as np


def _close_mask(obs, stored_obs, radius: float) -> np.ndarray:
    """
    Args:
        obs:        query observation, any shape — will be flattened.
        stored_obs: array of stored observations, shape (N, ...).
        radius:     L2 distance threshold.

    Returns:
        Boolean array of shape (N,), True where distance <= radius.
    """
    obs_flat = np.asarray(obs).flatten()
    stored_flat = np.asarray(stored_obs).reshape(len(stored_obs), -1)
    dists = np.linalg.norm(stored_flat - obs_flat, axis=1)
    return dists <= radius
