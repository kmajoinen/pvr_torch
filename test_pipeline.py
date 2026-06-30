"""
Pipeline verification script.

Tests the full chain:  env → obs → embedding → policy → gradient update

Three checks are run in sequence:
  1. Frozen embedding  — gradients flow through policy only
  2. End-to-end        — gradients flow through embedding + policy
  3. Augmentation      — random-shift applied in training mode

Run:
    python test_pipeline.py                          # DMC cheetah, random embedding
    python test_pipeline.py --embedding moco_aug     # pretrained MoCo
    python test_pipeline.py --augmentation           # enable random-shift aug
    python test_pipeline.py --env FrankaKitchen-v1   # swap env (slow)
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
import gymnasium
from src.gym_wrappers import make_gym_env
from src.embeddings import EmbeddingNet
from src.augmentations import make_augmentation


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def build_policy(in_size: int, action_space) -> nn.Sequential:
    if isinstance(action_space, gymnasium.spaces.Box):
        out_size = int(np.prod(action_space.shape))
    else:
        out_size = action_space.n
    return nn.Sequential(
        nn.Linear(in_size, 256),
        nn.ReLU(),
        nn.Linear(256, out_size),
    )


def collect_obs(env, n: int, seed: int = 0) -> np.ndarray:
    """Roll out random actions, return (n, C, H, W) uint8 array."""
    obs, _ = env.reset(seed=seed)
    buf = []
    while len(buf) < n:
        buf.append(obs.copy())
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
    return np.stack(buf[:n])


def weight_hash(module: nn.Module) -> float:
    """Scalar fingerprint of all parameters — changes if any weight changes."""
    return sum(p.sum().item() for p in module.parameters())


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def run(args):
    # ── 1. Environment ────────────────────────────────────────────────────────
    section("Environment")
    env = make_gym_env(train_from_pixels=True, num_actions=None, id=args.env)
    print(f"  id:           {args.env}")
    print(f"  obs space:    {env.observation_space}")
    print(f"  action space: {env.action_space}")

    t0 = time.time()
    obs_np = collect_obs(env, n=args.batch_size, seed=42)
    sps = args.batch_size / (time.time() - t0)
    print(f"  collected {args.batch_size} obs in {1/sps*args.batch_size:.2f}s  ({sps:.1f} steps/sec)")
    env.close()

    obs_t = torch.from_numpy(obs_np)  # (B, C, H, W) uint8

    # ── 2. Embedding ──────────────────────────────────────────────────────────
    section("Embedding")
    aug = make_augmentation("random_shift", pad=4) if args.augmentation else None
    embedding = EmbeddingNet(args.embedding, disable_cuda=args.disable_cuda,
                             augmentation=aug)
    device = embedding.device
    print(f"  name:         {args.embedding}")
    print(f"  output size:  {embedding.out_size}")
    print(f"  device:       {device}")
    print(f"  augmentation: {aug is not None}")

    # ── 3. Policy ─────────────────────────────────────────────────────────────
    section("Policy")
    policy = build_policy(embedding.out_size, env.action_space).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  architecture: {policy}")
    print(f"  parameters:   {n_params:,}")

    # Fixed random targets (same shape as policy output) for supervised updates
    if isinstance(env.action_space, gymnasium.spaces.Box):
        target_dim = int(np.prod(env.action_space.shape))
    else:
        target_dim = env.action_space.n
    targets = torch.randn(args.batch_size, target_dim, device=device)

    # ── Check 1: Frozen embedding ─────────────────────────────────────────────
    section("Check 1 — frozen embedding, policy gradients only")
    embedding.eval()

    t0 = time.time()
    with torch.no_grad():
        embedded = embedding(obs_t.to(device))  # numpy in eval mode
    embedded = torch.as_tensor(embedded, device=device, dtype=torch.float32)
    print(f"  embed time:   {time.time()-t0:.3f}s  shape: {embedded.shape}")

    # Verify no NaN/Inf in embedding output
    assert torch.isfinite(embedded).all(), "Embedding output contains NaN/Inf"
    print(f"  embedding output finite: ✓")

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    h_before = weight_hash(policy)
    losses = []

    for _ in range(args.n_updates):
        pred = policy(embedded)
        loss = F.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    h_after = weight_hash(policy)
    check("policy weights updated",  h_before != h_after)
    check("loss decreased",          losses[-1] < losses[0])
    print(f"  loss:  {losses[0]:.4f} → {losses[-1]:.4f}")

    # ── Check 2: End-to-end gradients ─────────────────────────────────────────
    section("Check 2 — end-to-end gradients (embedding + policy trainable)")
    embedding.train()
    for p in embedding.embedding.parameters():
        p.requires_grad_(True)

    optimizer_e2e = torch.optim.Adam(
        list(embedding.embedding.parameters()) + list(policy.parameters()), lr=1e-4
    )
    h_emb_before = weight_hash(embedding.embedding)
    e2e_losses = []

    for _ in range(args.n_updates):
        emb = embedding(obs_t.to(device))           # tensor in train mode
        emb = emb.to(device=device, dtype=torch.float32)
        pred = policy(emb)
        loss = F.mse_loss(pred, targets)
        optimizer_e2e.zero_grad()
        loss.backward()
        optimizer_e2e.step()
        e2e_losses.append(loss.item())

    h_emb_after = weight_hash(embedding.embedding)
    check("embedding weights updated", h_emb_before != h_emb_after)
    check("loss decreased",            e2e_losses[-1] < e2e_losses[0])
    print(f"  loss:  {e2e_losses[0]:.4f} → {e2e_losses[-1]:.4f}")

    # ── Check 3: Augmentation ─────────────────────────────────────────────────
    if args.augmentation:
        section("Check 3 — augmentation (random-shift active in train mode)")
        embedding.train()
        out_a = embedding(obs_t.to(device))
        out_b = embedding(obs_t.to(device))
        check("aug produces different outputs each forward pass",
              not torch.allclose(out_a, out_b))

        embedding.eval()
        out_c = embedding(obs_t.to(device))
        out_d = embedding(obs_t.to(device))
        out_c_t = torch.as_tensor(out_c)
        out_d_t = torch.as_tensor(out_d)
        check("aug disabled in eval mode (deterministic)",
              torch.allclose(out_c_t, out_d_t))

    # ── Summary ───────────────────────────────────────────────────────────────
    section("All checks passed ✓")


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def check(label: str, condition: bool):
    icon = "✓" if condition else "✗"
    print(f"  [{icon}] {label}")
    if not condition:
        raise AssertionError(f"Check failed: {label}")


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",          default="dm_control/cheetah-run-v0")
    parser.add_argument("--embedding",    default="random",
                        help="Embedding name. 'random' needs no checkpoint.")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--n_updates",    type=int, default=30)
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--disable_cuda", action="store_true")
    args = parser.parse_args()
    run(args)
