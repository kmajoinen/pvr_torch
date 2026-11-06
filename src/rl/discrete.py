"""
Discrete-action agents for train_discrete.py, behind one interface so the
training loop is the same for both:

  act(feats, step, explore)   (B,) int64 actions; explore=False is the
                              greedy/most likely action (evaluation)
  critic_loss(feats, actions, rewards, discounts, next_feats)
                              (loss, metrics): the loss the loop backprops,
                              together with any aux losses, into the critic
                              and, with encoder.grads=critic, the encoder
  critic                      the networks critic_loss trains (its optimizer
                              is critic_opt; the loop clips and steps it)
  after_critic(feats, step)   metrics: anything else this update does, on
                              detached features (discrete SAC: actor, alpha)
  update_targets(step)        soft (tau < 1) or hard (tau = 1) target update

rewards and discounts come from the replay buffer's n-step sums (discounts
= gamma^k, or 0 if the episode terminated), so every target is
rewards + discounts * V(next_feats).

DQN: double DQN (the online network picks the next action, the target
network values it; algo.double=false for plain DQN), Huber or MSE loss,
epsilon-greedy exploration annealed linearly over the first
epsilon.fraction of the run. Discrete SAC (Christodoulou,
2019): a softmax policy and twin Q-networks over all actions, so the
soft value and the policy loss are exact expectations over actions instead
of samples; temperature tuned towards target_entropy_scale x log(n_actions)
(CleanRL's sac_atari uses 0.89).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, out_dim: int, hidden, obs_norm: bool) -> nn.Sequential:
    """LayerNorm on encoder features only (see sac_utils.SoftQNetwork), then
    ReLU layers."""
    layers = [nn.LayerNorm(in_dim) if obs_norm else nn.Identity()]
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    return nn.Sequential(*layers, nn.Linear(in_dim, out_dim))


def soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.mul_(1 - tau).add_(p, alpha=tau)


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    return end + (start - end) * max(0.0, 1.0 - step / max(1, duration))


def epsilon_greedy(act_fn, n_actions: int, eps: float, generator: torch.Generator):
    """act_fn's actions, each replaced by a random one with probability eps.
    The coins come from `generator` (on the actions' device), not torch's
    global RNG, so turning this on doesn't change a seeded run's training."""
    def act(feats):
        greedy = act_fn(feats)
        coin = torch.rand(greedy.shape, device=greedy.device, generator=generator) < eps
        random = torch.randint(n_actions, greedy.shape, device=greedy.device, generator=generator)
        return torch.where(coin, random, greedy)
    return act


class DQN(nn.Module):
    name = "dqn"

    def __init__(self, feat_dim: int, n_actions: int, algo, obs_norm: bool):
        super().__init__()
        hidden = list(algo.net_arch)
        self.q = mlp(feat_dim, n_actions, hidden, obs_norm)
        self.q_target = mlp(feat_dim, n_actions, hidden, obs_norm)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.requires_grad_(False)
        self.n_actions = n_actions
        self.algo = algo
        self.double = bool(algo.get("double", True))
        self.loss = str(algo.get("loss", "huber"))
        if self.loss not in ("huber", "mse"):
            raise ValueError(f"algo.loss={self.loss!r}; expected huber or mse")
        self.critic = self.q
        self.critic_opt = torch.optim.Adam(self.q.parameters(), lr=algo.lr, betas=tuple(algo.betas))

    def epsilon(self, step: int) -> float:
        """Annealed over the first epsilon.fraction of the run, so it always ends inside it."""
        e = self.algo.epsilon
        return linear_schedule(e.start, e.end, int(e.fraction * self.algo.total_timesteps), step)

    def act(self, feats, step: int, explore: bool):
        with torch.no_grad():
            greedy = self.q(feats).argmax(1)
        if not explore:
            return greedy
        random = torch.randint(self.n_actions, greedy.shape, device=greedy.device)
        coin = torch.rand(greedy.shape, device=greedy.device) < self.epsilon(step)
        return torch.where(coin, random, greedy)

    def critic_loss(self, feats, actions, rewards, discounts, next_feats):
        q_all = self.q(feats)
        q = q_all.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            chooser = self.q if self.double else self.q_target
            next_a = chooser(next_feats).argmax(1, keepdim=True)
            target = rewards + discounts * self.q_target(next_feats).gather(1, next_a).squeeze(1)
        loss = F.smooth_l1_loss(q, target) if self.loss == "huber" else F.mse_loss(q, target)
        return loss, {"train/qf_loss": loss.item(), "train/qf1_values": q.mean().item(),
                      "train/q_max_mean": q_all.max(1).values.mean().item(),
                      "train/target_q_mean": target.mean().item()}

    def after_critic(self, feats, step: int) -> dict:
        return {"train/epsilon": self.epsilon(step)}

    def update_targets(self, step: int) -> None:
        if step % int(self.algo.target_network_frequency) == 0:
            soft_update(self.q, self.q_target, float(self.algo.tau))


class DiscreteSAC(nn.Module):
    name = "sacd"

    def __init__(self, feat_dim: int, n_actions: int, algo, obs_norm: bool):
        super().__init__()
        hidden = list(algo.net_arch)
        self.actor = mlp(feat_dim, n_actions, hidden, obs_norm)
        self.q1, self.q2 = mlp(feat_dim, n_actions, hidden, obs_norm), mlp(feat_dim, n_actions, hidden, obs_norm)
        self.q1_target, self.q2_target = (mlp(feat_dim, n_actions, hidden, obs_norm) for _ in range(2))
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)
        self.algo = algo
        self.critic = nn.ModuleList([self.q1, self.q2])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=algo.q_lr, betas=tuple(algo.q_betas))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=algo.policy_lr, betas=tuple(algo.policy_betas))
        self.autotune = bool(algo.autotune_alpha)
        self.target_entropy = float(algo.target_entropy_scale) * math.log(n_actions)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(float(algo.alpha))), requires_grad=self.autotune)
        self.alpha_opt = (torch.optim.Adam([self.log_alpha], lr=algo.alpha_lr, betas=tuple(algo.alpha_betas))
                          if self.autotune else None)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def policy(self, feats):
        logits = self.actor(feats)
        return F.softmax(logits, -1), F.log_softmax(logits, -1)

    def act(self, feats, step: int, explore: bool):
        with torch.no_grad():
            probs, _ = self.policy(feats)
        return torch.multinomial(probs, 1).squeeze(1) if explore else probs.argmax(1)

    def critic_loss(self, feats, actions, rewards, discounts, next_feats):
        with torch.no_grad():
            probs, logp = self.policy(next_feats)
            q_next = torch.min(self.q1_target(next_feats), self.q2_target(next_feats))
            v_next = (probs * (q_next - self.alpha * logp)).sum(1)
            target = rewards + discounts * v_next
        q1 = self.q1(feats).gather(1, actions[:, None]).squeeze(1)
        q2 = self.q2(feats).gather(1, actions[:, None]).squeeze(1)
        l1, l2 = F.mse_loss(q1, target), F.mse_loss(q2, target)
        return l1 + l2, {"train/qf1_loss": l1.item(), "train/qf2_loss": l2.item(),
                         "train/qf_loss": (l1 + l2).item() / 2, "train/qf1_values": q1.mean().item(),
                         "train/qf2_values": q2.mean().item(), "train/target_q_mean": target.mean().item()}

    def after_critic(self, feats, step: int) -> dict:
        metrics = {"train/alpha": self.alpha}
        if step % int(self.algo.policy_frequency):
            return metrics
        for _ in range(int(self.algo.policy_frequency)):
            probs, logp = self.policy(feats)
            with torch.no_grad():
                q = torch.min(self.q1(feats), self.q2(feats))
            actor_loss = (probs * (self.alpha * logp - q)).sum(1).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            gnorm = float(nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.algo.grad_clip)))
            self.actor_opt.step()
            entropy = -(probs * logp).sum(1).mean().detach()
            metrics.update({"train/actor_loss": actor_loss.item(), "train/actor_grad_norm": gnorm,
                            "train/entropy": entropy.item()})
            if self.autotune:
                # E_a[-alpha (log pi(a) + target)] = -alpha (target - entropy)
                alpha_loss = -(self.log_alpha.exp() * (self.target_entropy - entropy))
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()
                metrics["train/alpha_loss"] = alpha_loss.item()
            metrics["train/alpha"] = self.alpha
        return metrics

    def update_targets(self, step: int) -> None:
        if step % int(self.algo.target_network_frequency) == 0:
            soft_update(self.q1, self.q1_target, float(self.algo.tau))
            soft_update(self.q2, self.q2_target, float(self.algo.tau))


AGENTS = {"dqn": DQN, "sac_discrete": DiscreteSAC}
