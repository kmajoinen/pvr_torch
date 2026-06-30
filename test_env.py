"""
Sanity-check script. Verifies that:
  1. All required packages import correctly.
  2. The FrankaKitchen environment runs with the gym wrappers.
  3. A vision model (embedding) processes observations end-to-end.
  4. A random policy can step through the environment using embedded observations.

Usage:
  python test_env.py                          # random CNN embedding
  python test_env.py --embedding_name resnet50
  python test_env.py --n_steps 50 --disable_cuda
"""

import argparse
import time
import sys
import warnings
import os

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # suppress gymnasium-robotics deprecation noise


# ------------------------------------------------------------------------------
# 1. Import checks
# ------------------------------------------------------------------------------

def check_imports():
    required = [
        ("torch",               "torch"),
        ("torchvision",         "torchvision"),
        ("gymnasium",           "gymnasium"),
        ("gymnasium_robotics",  "gymnasium_robotics"),
        ("minigrid",            "minigrid"),
        ("timm",                "timm"),
        ("numpy",               "numpy"),
        ("PIL",                 "Pillow"),
    ]
    ok = True
    for module, package in required:
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "?")
            print(f"  [ok] {package} {version}")
        except ImportError:
            print(f"  [MISSING] {package}  ->  pip install {package}")
            ok = False

    try:
        import clip
        print("  [ok] CLIP (openai)")
    except ImportError:
        print("  [MISSING] CLIP  ->  pip install git+https://github.com/openai/CLIP.git")
        ok = False

    return ok


# ------------------------------------------------------------------------------
# 2. Main test
# ------------------------------------------------------------------------------

def run(args):
    import torch
    import numpy as np
    import types

    sys.path.insert(0, ".")
    from src.embeddings import EmbeddingNet
    from src.gym_wrappers import make_gym_env

    # Build a flags namespace compatible with make_gym_env
    flags = types.SimpleNamespace(
        env=args.env,
        embedding_name=args.embedding_name,
        num_input_frames=1,
        run_id=0,
    )

    print(f"\n--- Environment: {args.env} ---")
    print(f"--- Embedding:   {args.embedding_name} ---\n")

    # Load vision model
    print("Loading embedding model...")
    t0 = time.time()
    embedding = EmbeddingNet(
        args.embedding_name,
        disable_cuda=args.disable_cuda,
    )
    print(f"  Embedding output size: {embedding.out_size}")
    print(f"  Device: {embedding.device}")
    print(f"  Loaded in {time.time() - t0:.2f}s")

    # Build environment with embedding wrapper
    print("\nBuilding environment...")
    env = make_gym_env(flags, embedding_model=embedding, seed=args.seed)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space:      {env.action_space}")

    # Run random policy
    print(f"\nRunning {args.n_steps} random steps...")
    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    successes = 0
    episode = 0
    t0 = time.time()

    for step in range(args.n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        successes += info.get("success", 0.0)

        if terminated or truncated:
            episode += 1
            obs, info = env.reset()

    elapsed = time.time() - t0
    env.close()

    print(f"  Steps/sec:     {args.n_steps / elapsed:.1f}")
    print(f"  Total reward:  {total_reward:.4f}")
    print(f"  Successes:     {successes}")
    print(f"  Episodes done: {episode}")
    print(f"  Obs shape:     {obs.shape}")
    print("\nAll checks passed.")


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",             default="FrankaKitchen-v1")
    parser.add_argument("--embedding_name",  default="random",
                        help="Embedding to use. 'random' requires no checkpoint.")
    parser.add_argument("--n_steps",         type=int, default=20)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--disable_cuda",    action="store_true")
    parser.add_argument("--skip_imports",    action="store_true")
    args = parser.parse_args()

    if not args.skip_imports:
        print("=== Import checks ===")
        if not check_imports():
            sys.exit(1)

    run(args)
