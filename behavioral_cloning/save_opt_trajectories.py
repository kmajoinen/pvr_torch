"""
Collect demonstration trajectories from any environment and save to pickle.

The saved pickle has keys: obs, action, reward, done
where each value is a list of per-episode numpy arrays.
This format is consumed by save_embedded_obs.py and main_bc_1.py.

Usage:
    python behavioral_cloning/save_opt_trajectories.py --env dm_control/cheetah-run-v0 \
        --train_from_pixels --n_trajectories 1000

Policy options:
    random     (default) random actions — useful for pipeline testing
    checkpoint load a trained policy from --policy_path
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gym_wrappers import make_gym_env


parser = argparse.ArgumentParser()
parser.add_argument('--env',              type=str, default='dm_control/cheetah-run-v0')
parser.add_argument('--n_trajectories',   type=int, default=1000)
parser.add_argument('--max_steps',        type=int, default=1000,
                    help='Maximum steps per trajectory.')
parser.add_argument('--train_from_pixels', action='store_true',
                    help='Store pixel observations (84x84 CHW uint8).')
parser.add_argument('--num_actions',      type=int, default=None,
                    help='Discretise continuous actions into N bins. None = keep continuous.')
parser.add_argument('--policy',           type=str, default='random',
                    choices=['random', 'checkpoint'])
parser.add_argument('--policy_path',      type=str, default=None,
                    help='Path to policy checkpoint when --policy=checkpoint.')
parser.add_argument('--save_path',        type=str, default='behavioral_cloning')
parser.add_argument('--seed',             type=int, default=0)


def make_policy(policy_type, policy_path, env):
    if policy_type == 'random':
        return lambda obs: env.action_space.sample()
    elif policy_type == 'checkpoint':
        assert policy_path is not None, '--policy_path required for checkpoint policy'
        import torch
        from src.models import ContinuousPolicyNet
        checkpoint = torch.load(policy_path, map_location='cpu')
        obs_size = int(np.prod(env.observation_space.shape))
        import gymnasium
        if isinstance(env.action_space, gymnasium.spaces.Box):
            action_dim = int(np.prod(env.action_space.shape))
        else:
            action_dim = env.action_space.n
        policy = ContinuousPolicyNet(obs_size, action_dim)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
        def fn(obs):
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                return policy(obs_t).squeeze(0).numpy()
        return fn
    else:
        raise ValueError(f'Unknown policy type: {policy_type}')


def collect_trajectory(env, policy_fn, max_steps):
    obs, _ = env.reset()
    obs_list, act_list, rew_list, done_list = [], [], [], []

    for _ in range(max_steps):
        action = policy_fn(obs)
        obs_list.append(obs.copy() if hasattr(obs, 'copy') else obs)
        act_list.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rew_list.append(float(reward))
        done_list.append(bool(done))
        if done:
            break

    return (np.array(obs_list), np.array(act_list),
            np.array(rew_list),  np.array(done_list))


def run(flags):
    env = make_gym_env(
        train_from_pixels=flags.train_from_pixels,
        num_actions=flags.num_actions,
        id=flags.env,
    )
    env.reset(seed=flags.seed)

    policy_fn = make_policy(flags.policy, flags.policy_path, env)

    obs_all, act_all, rew_all, done_all = [], [], [], []
    for _ in tqdm(range(flags.n_trajectories), desc='collecting'):
        o, a, r, d = collect_trajectory(env, policy_fn, flags.max_steps)
        obs_all.append(o)
        act_all.append(a)
        rew_all.append(r)
        done_all.append(d)

    env.close()

    data = dict(obs=obs_all, action=act_all, reward=rew_all, done=done_all)

    os.makedirs(flags.save_path, exist_ok=True)
    save_name = os.path.join(flags.save_path, flags.env.replace('/', '_') + '.pickle')
    with open(save_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {flags.n_trajectories} trajectories to {save_name}')


if __name__ == '__main__':
    flags = parser.parse_args()
    run(flags)
