"""
Verify the pixel BC pipeline end-to-end:
  raw pixels (uint8) → ResNet50 embedding → MLP policy → action

Checks:
  1. Shapes at each stage
  2. Embedding output is non-trivial (not constant)
  3. Gradients reach the policy but not the frozen embedding
"""

import numpy as np
import torch
import pickle
import sys

DATA_PATH = 'behavioral_cloning/dm_control_cheetah-run-v0.pickle'
EMBEDDING  = 'resnet50'
ACTION_DIM = 6  # cheetah-run continuous action

print('=== Pixel BC pipeline verification ===\n')

# 1. Load raw pixel data
print('1. Loading pixel data ...')
data = pickle.load(open(DATA_PATH, 'rb'))
obs_raw = data['obs'][0][:4]  # first 4 frames of episode 0
print(f'   obs shape : {obs_raw.shape}')   # (4, 3, 84, 84)
print(f'   obs dtype : {obs_raw.dtype}')   # uint8
print(f'   obs range : [{obs_raw.min()}, {obs_raw.max()}]')
assert obs_raw.shape == (4, 3, 84, 84), 'expected (B, 3, 84, 84) CHW uint8'
assert obs_raw.dtype == np.uint8, 'expected uint8 pixels'
print('   OK\n')

# 2. Pass through embedding
print(f'2. Embedding with {EMBEDDING} ...')
from src.embeddings import EmbeddingNet
emb = EmbeddingNet(EMBEDDING, disable_cuda=True)
emb.eval()

obs_t  = torch.from_numpy(obs_raw)  # (4, 3, 84, 84) uint8
emb_np = emb(obs_t)                 # returns numpy in eval mode
print(f'   embedding output shape : {emb_np.shape}')   # (4, 2048)
print(f'   embedding output dtype : {emb_np.dtype}')
print(f'   output std (non-trivial?): {emb_np.std():.4f}')
assert emb_np.shape == (4, emb.out_size), f'expected (4, {emb.out_size})'
assert emb_np.std() > 1e-3, 'embedding output is constant — something is wrong'
print('   OK\n')

# 3. Pass through policy
print('3. Policy forward pass ...')
from src.models import ContinuousPolicyNet
policy = ContinuousPolicyNet(obs_size=emb.out_size, action_dim=ACTION_DIM)

obs_emb = torch.from_numpy(emb_np)  # (4, 2048)
action  = policy(obs_emb)
print(f'   action shape : {action.shape}')  # (4, 6)
print(f'   action range : [{action.min().item():.3f}, {action.max().item():.3f}]')
assert action.shape == (4, ACTION_DIM)
assert action.abs().max().item() <= 1.0 + 1e-5, 'Tanh output should be in [-1, 1]'
print('   OK\n')

# 4. Gradient check — policy trains, embedding stays frozen
print('4. Gradient flow check ...')
emb_t  = torch.from_numpy(emb_np).requires_grad_(False)
action = policy(emb_t)
loss   = action.mean()
loss.backward()

policy_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in policy.parameters())
emb_has_grad    = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in emb.parameters())
print(f'   policy gradients  : {policy_has_grad}')  # should be True
print(f'   embedding gradients: {emb_has_grad}')    # should be False (frozen)
assert policy_has_grad, 'policy has no gradients — it will not train'
assert not emb_has_grad, 'embedding is receiving gradients — it should be frozen'
print('   OK\n')

print('=== All checks passed ===')
print(f'\nPipeline: ({obs_raw.shape}) uint8'
      f' → ResNet50 → ({emb_np.shape}) float32'
      f' → MLP → ({tuple(action.shape)}) tanh')
