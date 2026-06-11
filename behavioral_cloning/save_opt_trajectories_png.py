"""
Collect pixel trajectories and save each frame as a PNG file.
Non-pixel data (actions, rewards, done) is saved alongside as a pickle per trajectory.

This format is useful when trajectories are too large to keep in a single pickle
but you still want to access individual frames efficiently.

Usage:
    python behavioral_cloning/save_opt_trajectories_png.py \
        --env dm_control/cheetah-run-v0 --n_trajectories 100
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
from src.gym_wrappers import make_gym_env
from behavioral_cloning.save_opt_trajectories import make_policy, collect_trajectory


parser = argparse.ArgumentParser()
parser.add_argument('--env',             type=str,  default='dm_control/cheetah-run-v0')
parser.add_argument('--n_trajectories',  type=int,  default=1000)
parser.add_argument('--max_steps',       type=int,  default=1000)
parser.add_argument('--num_actions',     type=int,  default=None)
parser.add_argument('--policy',          type=str,  default='random',
                    choices=['random', 'checkpoint'])
parser.add_argument('--policy_path',     type=str,  default=None)
parser.add_argument('--save_path',       type=str,  default='behavioral_cloning')
parser.add_argument('--seed',            type=int,  default=0)


def run(flags):
    env = make_gym_env(
        train_from_pixels=True,
        num_actions=flags.num_actions,
        id=flags.env,
    )
    env.reset(seed=flags.seed)

    policy_fn = make_policy(flags.policy, flags.policy_path, env)

    save_dir = os.path.join(flags.save_path, flags.env.replace('/', '_'))
    os.makedirs(save_dir, exist_ok=True)

    for t in tqdm(range(flags.n_trajectories), desc='collecting'):
        obs_arr, act_arr, rew_arr, done_arr = collect_trajectory(
            env, policy_fn, flags.max_steps
        )

        # Save frames as PNG (obs is (T, C, H, W) uint8 — convert to HWC for cv2)
        for i, frame in enumerate(obs_arr):
            img = np.transpose(frame, (1, 2, 0))  # CHW → HWC
            cv2.imwrite(os.path.join(save_dir, f'{t}_{i}.png'), img)

        # Save non-pixel data
        with open(os.path.join(save_dir, f'{t}.pickle'), 'wb') as f:
            pickle.dump(dict(action=act_arr, reward=rew_arr, done=done_arr), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    env.close()
    print(f'Saved {flags.n_trajectories} trajectories to {save_dir}')


if __name__ == '__main__':
    flags = parser.parse_args()
    run(flags)
