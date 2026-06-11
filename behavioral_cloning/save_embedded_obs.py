"""
Pass raw pixel observations through an embedding model and save the result.

Input:  trajectory pickle produced by save_opt_trajectories.py
        (dict with keys: obs [list of (T, C, H, W) uint8], action, reward, done)
Output: pickle with the same structure but obs replaced by embedding vectors.

This pre-computation speeds up main_bc_2.py, which loads the embedded obs
directly and skips the embedding forward pass during training.

Usage:
    python behavioral_cloning/save_embedded_obs.py \
        --env dm_control/cheetah-run-v0 \
        --embedding_name resnet50
"""

import os
import argparse
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings import EmbeddingNet


parser = argparse.ArgumentParser()
parser.add_argument('--env',            type=str, default='dm_control/cheetah-run-v0')
parser.add_argument('--embedding_name', type=str, default='resnet50')
parser.add_argument('--data_path',      type=str, default='behavioral_cloning')
parser.add_argument('--batch_size',     type=int, default=64)
parser.add_argument('--disable_cuda',   action='store_true')
parser.add_argument('--run_id',         type=int, default=1)


def load_demo_pickle(data_path):
    """Load trajectory pickle and flatten per-episode lists into arrays."""
    print('loading %s ...' % data_path)
    data = pickle.load(open(data_path, 'rb'))
    n_trajectories = len(data['reward'])
    for k in ('obs', 'action', 'reward', 'done'):
        if isinstance(data[k], list):
            data[k] = np.concatenate(data[k])
    n_samples = len(data['reward'])
    print(f'  {n_trajectories} trajectories, {n_samples} samples')
    return data


def run(flags):
    env_key = flags.env.replace('/', '_')
    save_name = os.path.join(flags.data_path, f'{env_key}_{flags.embedding_name}.pickle')
    if os.path.isfile(save_name):
        print(f'Already exists, skipping: {save_name}')
        return

    torch.manual_seed(flags.run_id)
    np.random.seed(flags.run_id)
    random.seed(flags.run_id)

    embedding_model = EmbeddingNet(
        flags.embedding_name,
        pretrained=True,
        train=False,
        disable_cuda=flags.disable_cuda,
    )

    # Save embedding weights (needed to reproduce random embedding experiments)
    emb_path = os.path.join(flags.data_path, flags.embedding_name)
    if flags.embedding_name == 'random':
        emb_path += '_' + str(flags.run_id)
    torch.save({'embedding_model_state_dict': embedding_model.state_dict()},
               emb_path + '.tar')

    data_path = os.path.join(flags.data_path, env_key + '.pickle')
    data = load_demo_pickle(data_path)

    print('Embedding observations ...')
    n_samples = data['obs'].shape[0]

    # obs shape: (N, C, H, W) — channels-first from new wrappers
    obs_raw = data['obs']
    obs_embedded = []
    for i in tqdm(range(0, n_samples, flags.batch_size)):
        batch = torch.from_numpy(obs_raw[i:i + flags.batch_size])
        emb = embedding_model(batch)          # returns numpy in eval mode
        obs_embedded.append(emb)
    obs_embedded = np.concatenate(obs_embedded)[:n_samples]

    out = dict(
        obs=obs_embedded,
        action=data['action'][:n_samples],
        reward=data['reward'][:n_samples],
        done=data['done'][:n_samples],
    )

    with open(save_name, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved to {save_name}')


if __name__ == '__main__':
    flags = parser.parse_args()
    run(flags)
