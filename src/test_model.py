import numpy as np
import torch
from tqdm import tqdm


def test(model, env, stat_keys, n_episodes=100):
    """
    Evaluate a PolicyNet (LSTM) in env for n_episodes.
    Uses the Environment wrapper API (reset/step).
    Returns a dict mapping each stat_key to a list of per-episode values.
    """
    stats = {k: [] for k in stat_keys}

    for _ in tqdm(range(n_episodes), desc='testing episode'):
        obs, _ = env.reset()
        agent_state = model.initial_state(batch_size=1)
        agent_state = tuple(s.to(model.device) for s in agent_state)

        ep_return = 0.
        ep_success = 0.
        ep_step = 0
        done = False

        while not done:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            obs_t = obs_t.view(1, 1, -1).to(model.device)   # (T=1, B=1, obs_dim)
            done_t = torch.tensor([[done]], dtype=torch.bool).to(model.device)

            with torch.no_grad():
                output, agent_state = model({'obs': obs_t, 'done': done_t}, agent_state)

            action = output['action'].item()
            obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get('success', 0.)))
            ep_step += 1

        if 'episode_return'  in stat_keys: stats['episode_return'].append(ep_return)
        if 'episode_success' in stat_keys: stats['episode_success'].append(ep_success)
        if 'episode_step'    in stat_keys: stats['episode_step'].append(ep_step)

    return stats
