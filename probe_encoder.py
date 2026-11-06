"""
probe_encoder.py: how much of an environment's true state an encoder's
features carry. Collects frames from a random policy, encodes them through
the same input pipeline training uses, fits a ridge regression from features
and a small MLP to the true state on the first episodes, and reports R^2 on
the held-out ones. Takes minutes, and separates "the encoder doesn't see it"
from "the RL is broken" before any RL run. The MLP matters for speeds: a
speed from stacked frames isn't a linear function of per-frame features, so
the ridge understates them, and the RL heads are MLPs.

    python probe_encoder.py env=fourrooms_pixels embedding=resnet18
    python probe_encoder.py env=dm_control_pixels embedding=custom_checkpoint \
        embedding.arch=resnet50 embedding.checkpoint=/path/to/encoder_step1000000.pt \
        embedding.name=r3m50_ours +embedding.frame_stack=3
    python probe_encoder.py env=pendulum_pixels embedding=resnet18 embedding.pretrained=false   # a control
    ... algo.action_repeat=2      # as the run being checked (DrQ-v2: 2): stacked frames are that far apart

The state is the env's own observation (Pendulum: cos, sin, angular speed;
BipedalWalker: including lidar and speeds a frame doesn't show; cheetah:
positions and velocities), or for FourRooms the agent's and goal's cells, with
the mean error in tiles and how often the exact cell comes out. A random
policy doesn't reach everything a trained one does, so this is a check that
the information is there, not a prediction of returns; it can show an encoder
is missing something, not that pretraining helps (random weights score well on
it). Results print and go to probe.json in the run directory.
"""

import copy
import json
import os
from collections import deque

import cv2
import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.encoder_config import embedding_kwargs
from src.sac_utils import _register_env_ids, _repeat, env_kwargs

GRIDWORLD_GOAL = 10            # gym_gridworlds' GOOD tile


def state_names(env_id: str, n: int):
    known = {"Pendulum": ["cos", "sin", "angular_speed"],
             "Gym-Gridworlds": ["agent_row", "agent_col", "goal_row", "goal_col"]}
    for key, names in known.items():
        if key in env_id:
            return names
    return [f"dim{i}" for i in range(n)]


def true_state(env, obs) -> np.ndarray:
    u = env.unwrapped
    if type(u).__module__.startswith("gym_gridworlds"):
        goal = np.argwhere(u.grid == GRIDWORLD_GOAL)[0]
        return np.array([*u.agent_pos, *goal], dtype=np.float64)
    return np.asarray(gym.spaces.utils.flatten(env.observation_space, obs), dtype=np.float64)


def ridge_fit_predict(z_train, y_train, z_test, lam):
    """Ridge in whichever form is smaller (features can far outnumber frames)."""
    if z_train.shape[1] <= z_train.shape[0]:
        w = np.linalg.solve(z_train.T @ z_train + lam * np.eye(z_train.shape[1]), z_train.T @ y_train)
        return z_test @ w
    alpha = np.linalg.solve(z_train @ z_train.T + lam * np.eye(z_train.shape[0]), y_train)
    return z_test @ (z_train.T @ alpha)


def r2_scores(y, pred):
    """R^2 per dimension; NaN where the held-out targets don't vary."""
    ss = ((y - y.mean(0)) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss > 1e-12, 1 - ((y - pred) ** 2).sum(0) / ss, np.nan)


def split_standardize(feats, states, episodes, test_fraction: float):
    """Whole held-out episodes, so neighbouring frames don't leak into the
    test set; the last training episodes choose the ridge penalty and the
    MLP's stopping point. Standardized on the non-test frames."""
    ids = np.unique(episodes)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    test = np.isin(episodes, ids[-n_test:])
    train_ids = ids[:-n_test]
    val = np.isin(episodes, train_ids[-max(1, len(train_ids) // 5):])
    train = ~test & ~val
    # Features that hardly vary on these frames (a ReLU channel that never
    # fires on a simple scene) are dropped, and the rest clipped: one firing
    # only on a held-out frame would otherwise standardize to something huge,
    # and an MLP's weights for it never trained.
    mu, sd = feats[~test].mean(0), feats[~test].std(0)
    keep = sd > 1e-5
    z = np.clip((feats[:, keep] - mu[keep]) / sd[keep], -10, 10)
    y_mu, y_sd = states[~test].mean(0), states[~test].std(0) + 1e-9
    return train, val, test, z, (states - y_mu) / y_sd, y_mu, y_sd


def fit_probe(feats, states, episodes, test_fraction: float, lams=(1e-2, 1e0, 1e2, 1e4)):
    train, val, test, z, y, y_mu, y_sd = split_standardize(feats, states, episodes, test_fraction)

    def score(lam):
        return np.nanmean(r2_scores(y[val], ridge_fit_predict(z[train], y[train], z[val], lam)))
    lam = max(lams, key=score)
    pred = ridge_fit_predict(z[~test], y[~test], z[test], lam) * y_sd + y_mu
    return {"r2": r2_scores(states[test], pred), "lam": lam, "pred": pred, "target": states[test],
            "n_train": int((~test).sum()), "n_test": int(test.sum()), "features_kept": int(z.shape[1])}


def fit_mlp_probe(feats, states, episodes, test_fraction: float, device, hidden: int = 256, batch: int = 256,
                  max_epochs: int = 300, patience: int = 30, seed: int = 0):
    """As fit_probe, with a two-hidden-layer MLP in place of the ridge, stopped
    at its best validation loss."""
    train, val, test, z, y, y_mu, y_sd = split_standardize(feats, states, episodes, test_fraction)
    torch.manual_seed(seed)
    z = torch.as_tensor(z, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    tr, va = torch.as_tensor(np.flatnonzero(train), device=device), torch.as_tensor(np.flatnonzero(val), device=device)
    mlp = nn.Sequential(nn.Linear(z.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, y.shape[1])).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, since = float("inf"), None, 0
    for _ in range(max_epochs):
        for i in tr[torch.randperm(len(tr), device=device)].split(batch):
            opt.zero_grad(set_to_none=True)
            nn.functional.mse_loss(mlp(z[i]), y_t[i]).backward()
            opt.step()
        with torch.no_grad():
            loss = nn.functional.mse_loss(mlp(z[va]), y_t[va]).item()
        if loss < best:
            best, best_state, since = loss, copy.deepcopy(mlp.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    mlp.load_state_dict(best_state)
    with torch.no_grad():
        pred = mlp(z[torch.as_tensor(np.flatnonzero(test), device=device)]).cpu().numpy() * y_sd + y_mu
    return {"r2": r2_scores(states[test], pred), "pred": pred, "target": states[test]}


def probe_lengths(cfg):
    """(steps per episode, frames): probe.episode_steps, else the env config's
    probe_episode_steps (physics tasks that start from one pose need long
    episodes to leave it), else 50; probe.frames, else enough for 20 episodes
    and at least 5000."""
    p = cfg.probe
    steps = int(p.episode_steps or cfg.env.get("probe_episode_steps") or 50)
    return steps, int(p.frames or max(5000, 20 * steps))


def probe_env(cfg):
    """The env with its state observation, not frames, and the run's action
    repeat: training stacks frames action_repeat env steps apart."""
    _register_env_ids(cfg.env.id)
    size = int(cfg.env.get("image_size", 224))
    make = dict(render_mode="rgb_array", **env_kwargs(cfg))
    if cfg.env.builder == "dmc":
        make["render_kwargs"] = dict(height=size, width=size, camera_id=0)
    return _repeat(gym.make(cfg.env.id, **make), int(cfg.algo.get("action_repeat", 1)))


def collect(cfg, net, device):
    """Random-policy features and true states, in batches (frames aren't kept)."""
    size = int(cfg.env.get("image_size", 224))
    env = probe_env(cfg)
    env.action_space.seed(int(cfg.seed))
    k = int(cfg.embedding.get("frame_stack") or 1)
    p = cfg.probe
    episode_steps, frames = probe_lengths(cfg)

    def frame():
        f = np.asarray(env.render())
        # As _make_gym does (squashing if not square); dmc renders at this size.
        return f if f.shape[:2] == (size, size) else cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)

    feats, states, episodes, batch = [], [], [], []

    def flush():
        if batch:
            x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device)
            with torch.no_grad():
                feats.append(net.encode(x).float().cpu().numpy())
            batch.clear()

    n, ep = 0, 0
    while n < frames:
        obs, _ = env.reset(seed=int(cfg.seed) + ep)
        stack = deque([frame()] * k, maxlen=k)         # padded with the first frame, as the training stack
        for t in range(episode_steps):
            batch.append(np.concatenate(stack, axis=-1))
            states.append(true_state(env, obs))
            episodes.append(ep)
            n += 1
            if len(batch) == int(p.batch_size):
                flush()
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc or n >= frames:
                break
            stack.append(frame())
        ep += 1
    flush()
    env.close()
    return np.concatenate(feats), np.stack(states), np.array(episodes)


@hydra.main(config_path="configs", config_name="config_probe", version_base=None)
def main(cfg: DictConfig) -> None:
    from src.embeddings import EmbeddingNet
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    net = EmbeddingNet(cfg.embedding.name, pretrained=bool(cfg.embedding.get("pretrained", True)),
                       disable_cuda=device.type == "cpu", model_dir=cfg.model_dir,
                       input_size=int(cfg.env.get("image_size", 224)), **embedding_kwargs(cfg))
    net.eval()
    feats, states, episodes = collect(cfg, net, device)
    res = fit_probe(feats, states, episodes, float(cfg.probe.test_fraction))
    names = state_names(cfg.env.id, states.shape[1])
    r2 = res["r2"]
    out = {"env": cfg.env.id, "embedding": str(cfg.embedding.name), "pretrained": cfg.embedding.get("pretrained", True),
           "frame_stack": int(cfg.embedding.get("frame_stack") or 1), "feature_dims": int(feats.shape[1]),
           "features_kept": res["features_kept"],
           "episodes": int(episodes.max()) + 1, "n_train": res["n_train"], "n_test": res["n_test"], "ridge_penalty": res["lam"],
           "mean_r2": float(np.nanmean(r2)), "r2": {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, r2)}}
    if out["episodes"] < 15:
        print(f"WARNING: only {out['episodes']} episodes: raise probe.frames, or the held-out set is a few episodes")
    print(f"\n{out['embedding']} (frame_stack={out['frame_stack']}, {out['feature_dims']} dims, {out['features_kept']} vary) on {out['env']}: "
          f"{out['n_train']} train / {out['n_test']} held-out frames")
    print(f"ridge: mean R^2 {out['mean_r2']:.3f}  " + "  ".join(f"{n}={v:.2f}" for n, v in zip(names, r2) if not np.isnan(v)))
    mlp = fit_mlp_probe(feats, states, episodes, float(cfg.probe.test_fraction), device, seed=int(cfg.seed))
    out["mlp_mean_r2"] = float(np.nanmean(mlp["r2"]))
    out["mlp_r2"] = {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, mlp["r2"])}
    print(f"MLP:  mean R^2 {out['mlp_mean_r2']:.3f}  "
          + "  ".join(f"{n}={v:.2f}" for n, v in zip(names, mlp["r2"]) if not np.isnan(v)))
    if "Gym-Gridworlds" in cfg.env.id:
        # A fixed goal (random_goals=false) is constant: predicting it is trivial, so it's left out.
        goal_varies = bool(np.ptp(states[:, 2:], axis=0).max() > 0)
        for key, r in (("", res), ("mlp_", mlp)):
            err = np.abs(r["pred"] - r["target"])
            exact = np.round(r["pred"]) == r["target"]
            out.update({f"{key}agent_error_tiles": float(err[:, :2].mean()),
                        f"{key}agent_exact_cell": float(exact[:, :2].all(1).mean())})
            line = (f"{key.rstrip('_') or 'ridge'}: agent {out[f'{key}agent_error_tiles']:.2f} tiles off, "
                    f"exact cell {out[f'{key}agent_exact_cell']:.0%}")
            if goal_varies:
                out.update({f"{key}goal_error_tiles": float(err[:, 2:].mean()),
                            f"{key}goal_exact_cell": float(exact[:, 2:].all(1).mean())})
                line += (f"   goal {out[f'{key}goal_error_tiles']:.2f} tiles off, "
                         f"exact cell {out[f'{key}goal_exact_cell']:.0%}")
            print(line)
    with open(os.path.join(HydraConfig.get().runtime.output_dir, "probe.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
