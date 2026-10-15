"""
Replay buffer for the off-policy scripts: one circular array of
observations (features, state vectors or uint8 frames), n-step returns, and
correct episode boundaries.

Each observation is stored once: a transition's next_obs is the slot after
it (as in SB3's optimize_memory_usage). When an episode ends -- terminated
or truncated -- its final observation goes into one extra slot right after
the last transition, so the last transition's next_obs is the real final
state, not the next episode's first frame. That matters for truncation
(time limits), where the target still bootstraps from next_obs.

sample() returns (obs, next_obs, actions, rewards, discounts) with
    rewards    sum_{i<k} gamma^i r_{t+i}  over k = nstep steps, or fewer if
               the episode ends first (never crossing into the next episode)
    discounts  gamma^k, or 0 if the episode terminated within those steps
so every agent forms its target as  rewards + discounts * V(next_obs).
nstep=1 is the plain one-step target.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_dim: int,
        device,
        obs_dtype=np.float32,
        nstep: int = 1,
        gamma: float = 0.99,
    ):
        if nstep < 1:
            raise ValueError(f"nstep must be >= 1, got {nstep}")
        self.capacity = capacity
        self.device = device
        self.nstep = nstep
        self.gamma = gamma
        # Pre-flight estimate BEFORE allocating, so an oversized
        # buffer_size/image_size combination announces itself instead of
        # dying inside np.zeros.
        itemsize = np.dtype(obs_dtype).itemsize
        gb = capacity * int(np.prod(obs_shape)) * itemsize / 1e9
        print(
            f"Replay buffer: {capacity:,} x {obs_shape} {np.dtype(obs_dtype).name}"
            f"  ({gb:.1f} GB for obs; next_obs derived, not stored separately)"
            + (f"  nstep={nstep}" if nstep > 1 else "")
        )
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.done = np.zeros(capacity, dtype=bool)      # episode ended after this transition
        self.is_final = np.zeros(capacity, dtype=bool)  # slot holds only a final observation
        self.pos, self.full = 0, False

    def _write(self, obs, action, reward, terminated, done, is_final):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = terminated
        self.done[i] = done
        self.is_final[i] = is_final
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def add(self, obs, next_obs, action, reward, terminated, truncated=False):
        done = bool(terminated or truncated)
        self._write(obs, action, reward, bool(terminated), done, False)
        if done:
            # The episode's final observation, which no later add() supplies.
            self._write(next_obs, 0.0, 0.0, False, True, True)

    def __len__(self):
        return self.capacity if self.full else self.pos

    def _returns(self, idx):
        """n-step reward sums, discounts and next indices for start slots idx."""
        cap = self.capacity
        rewards = np.zeros(len(idx), dtype=np.float32)
        steps = np.zeros(len(idx), dtype=np.int64)
        terminal = np.zeros(len(idx), dtype=bool)
        alive = np.ones(len(idx), dtype=bool)
        for i in range(self.nstep):
            cur = (idx + i) % cap
            rewards += alive * (self.gamma ** i) * self.rewards[cur]
            steps += alive
            terminal |= alive & self.terminated[cur]
            alive &= ~self.done[cur]
        discounts = (self.gamma ** steps) * ~terminal
        return rewards, discounts.astype(np.float32), (idx + steps) % cap, steps

    def sample(self, batch_size: int):
        size = len(self)
        oldest = self.pos if self.full else 0
        idx = np.empty(batch_size, dtype=np.int64)
        missing = np.arange(batch_size)
        for _ in range(100):
            cand = (oldest + np.random.randint(0, size, size=len(missing))) % self.capacity
            _, _, _, steps = self._returns(cand)
            # A start needs to be a transition (not a final-observation slot)
            # whose next observation has already been written, without the
            # window wrapping past the newest slot.
            logical = (cand - oldest) % self.capacity
            ok = ~self.is_final[cand] & (logical + steps <= size - 1)
            idx[missing[ok]] = cand[ok]
            missing = missing[~ok]
            if not len(missing):
                break
        else:
            raise RuntimeError("replay buffer has too few complete transitions to sample from")
        rewards, discounts, next_idx, _ = self._returns(idx)
        to = lambda x: torch.as_tensor(x, device=self.device)
        return (
            to(self.obs[idx]),
            to(self.obs[next_idx]),
            to(self.actions[idx]),
            to(rewards),
            to(discounts),
        )
