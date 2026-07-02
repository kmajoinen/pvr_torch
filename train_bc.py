"""
Behavioral Cloning training script.

Supports:
  - State observations (no embedding needed)
  - Pixel observations (via EmbeddingNet, set cfg.embedding to a model name)
  - Minari offline RL datasets (cfg.dataset.format: minari)
  - Pickle trajectory files from save_opt_trajectories.py (cfg.dataset.format: pickle)

Usage:
    python train_bc.py                                     # door expert, default config
    python train_bc.py dataset=dmc_cheetah_random embedding=resnet50
    python train_bc.py algo.batch_size=128                 # override any param
    python train_bc.py device=cpu                          # force CPU
    python train_bc.py dataset.max_episodes=10             # quick smoke test
"""

import os
import random
import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models import ContinuousPolicyNet


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def evaluate(policy, env_id: str, n_episodes: int, device,
             embedding=None, train_from_pixels: bool = False) -> dict:
    from src.gym_wrappers import make_gym_env

    env = make_gym_env(train_from_pixels=train_from_pixels, id=env_id)
    policy.eval()

    returns, successes = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return, ep_success = 0.0, 0.0
        done = False
        while not done:
            if train_from_pixels:
                obs_t = torch.from_numpy(np.asarray(obs)).unsqueeze(0).to(device)  # uint8
            else:
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)

            if embedding is not None:
                with torch.no_grad():
                    obs_t = torch.as_tensor(
                        embedding(obs_t), device=device, dtype=torch.float32
                    )  # (1, embed_dim)
            with torch.no_grad():
                action = policy(obs_t).squeeze(0).cpu().numpy()
            obs, reward, term, trunc, info = env.step(action)
            ep_return += reward
            ep_success = max(ep_success, float(info.get('success', 0)))
            done = term or trunc
        returns.append(ep_return)
        successes.append(ep_success)

    env.close()
    policy.train()
    return {
        'return_mean':  np.mean(returns),
        'return_std':   np.std(returns),
        'success_rate': np.mean(successes),
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config_bc", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Weights & Biases ──────────────────────────────────────────────────────
    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        from omegaconf import OmegaConf
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            mode=cfg.wandb.get('mode', 'online'),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"wandb run: {wandb.run.url}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    fmt = cfg.dataset.get('format', 'minari')
    train_from_pixels = bool(cfg.dataset.get('train_from_pixels', False))

    print(f"\nLoading dataset [{fmt}]: {cfg.dataset.id}")
    if fmt == 'pickle':
        from src.data.pickle_dataset import PickleDataset
        dataset = PickleDataset(
            cfg.dataset.id,
            max_episodes=cfg.dataset.max_episodes,
        )
        eval_env_id_override = cfg.dataset.eval_env_id
    elif fmt == 'vd4rl':
        from src.data.vd4rl_dataset import VD4RLDataset, task_to_env_id
        dataset = VD4RLDataset(
            task=cfg.dataset.task,
            quality=cfg.dataset.quality,
            resolution=cfg.dataset.resolution,
            max_episodes=cfg.dataset.max_episodes,
        )
        # Derive eval env from task name if not set explicitly in config
        if cfg.dataset.eval_env_id is None:
            eval_env_id_override = task_to_env_id(cfg.dataset.task)
        else:
            eval_env_id_override = cfg.dataset.eval_env_id
    else:
        from src.data.minari_dataset import MinariDataset
        dataset = MinariDataset(
            cfg.dataset.id,
            max_episodes=cfg.dataset.max_episodes,
            obs_key=cfg.dataset.obs_key,
        )
        eval_env_id_override = cfg.dataset.eval_env_id

    n_val = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.algo.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.algo.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    print(f"  train: {n_train:,}  val: {n_val:,}")

    # ── Optional embedding ────────────────────────────────────────────────────
    embedding = None
    obs_size = int(np.prod(dataset.obs_shape))
    emb_name = cfg.embedding.get('name', None)
    if emb_name is not None:
        from src.embeddings import EmbeddingNet
        embedding = EmbeddingNet(emb_name, disable_cuda=(str(device) == 'cpu'))
        embedding.eval()
        obs_size = embedding.out_size
        print(f"\nEmbedding: {emb_name}  →  obs_size: {obs_size}")

    # ── Policy ────────────────────────────────────────────────────────────────
    policy = ContinuousPolicyNet(
        obs_size=obs_size,
        action_dim=dataset.action_dim,
        hidden_size=cfg.hidden_size,
        batch_norm=cfg.batch_norm,
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\nPolicy: obs({obs_size}) → action({dataset.action_dim})  |  {n_params:,} params")

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.algo.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.algo.max_epochs
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    n_batches = len(train_loader)
    log_every = max(100, n_batches // 10)

    print(f"\nTraining for {cfg.algo.max_epochs} epochs  "
          f"(batch={cfg.algo.batch_size}, {n_batches:,} batches/epoch)")
    best_val_loss = float('inf')
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for epoch in range(1, cfg.algo.max_epochs + 1):
        # Train
        policy.train()
        t0 = time.time()
        train_loss = 0.0
        for i, (obs_b, act_b) in enumerate(train_loader):
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            if embedding is not None:
                with torch.no_grad():
                    obs_b = torch.as_tensor(
                        embedding(obs_b), device=device, dtype=torch.float32
                    )
            pred = policy(obs_b)
            loss = F.mse_loss(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            if (i + 1) % log_every == 0:
                elapsed = time.time() - t0
                avg = train_loss / (i + 1)
                pct = 100 * (i + 1) / n_batches
                print(f"  epoch {epoch}/{cfg.algo.max_epochs}"
                      f"  [{pct:4.0f}%]  batch {i+1:,}/{n_batches:,}"
                      f"  loss={avg:.4f}  ({elapsed:.0f}s)", flush=True)

        train_loss /= n_batches
        scheduler.step()

        # Validate
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                obs_b, act_b = obs_b.to(device), act_b.to(device)
                if embedding is not None:
                    obs_b = torch.as_tensor(
                        embedding(obs_b), device=device, dtype=torch.float32
                    )
                val_loss += F.mse_loss(policy(obs_b), act_b).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(policy.state_dict(), os.path.join(save_dir, 'best_policy.pt'))

        lr = scheduler.get_last_lr()[0]
        if use_wandb:
            wandb.log({'train/loss': train_loss, 'val/loss': val_loss,
                       'train/lr': lr}, step=epoch)

        if epoch % cfg.algo.eval_frequency == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:4d}/{cfg.algo.max_epochs}"
                  f"  train={train_loss:.4f}  val={val_loss:.4f}"
                  f"  lr={lr:.2e}  ({elapsed:.1f}s)")

            if eval_env_id_override is not None:
                stats = evaluate(
                    policy, eval_env_id_override,
                    n_episodes=cfg.algo.n_episodes_test,
                    device=device,
                    embedding=embedding,
                    train_from_pixels=train_from_pixels,
                )
                print(f"         eval  return={stats['return_mean']:.1f}"
                      f"±{stats['return_std']:.1f}"
                      f"  success={stats['success_rate']:.2%}")
                if use_wandb:
                    wandb.log({'eval/return_mean': stats['return_mean'],
                               'eval/return_std':  stats['return_std'],
                               'eval/success_rate': stats['success_rate']}, step=epoch)

    if use_wandb:
        wandb.finish()
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}/best_policy.pt")


if __name__ == "__main__":
    main()
