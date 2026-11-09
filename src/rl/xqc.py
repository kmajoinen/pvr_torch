"""
XQC (Palenicek et al., "XQC: Well-conditioned Optimization Accelerates Deep
Reinforcement Learning", ICLR 2026, arXiv:2509.25174), ported from the
authors' JAX code (github.com/danielpalenicek/xqc, MIT license). Nothing
from it needs to be installed.

XQC changes only the actor's and critic's MLPs and the critic loss, in three
parts the paper's ablations find each matter (BN most, then CE, then WN):

  BN   BatchNorm on the network input and after every hidden Linear, in the
       order Linear (no bias) -> BN -> ReLU. The critic sees (s, a) and
       (s', a') as one joint batch, so its statistics cover both.
  WN   after every optimizer step, each Linear's weight rows are projected
       to unit L2 norm. Hidden Linears have no bias; the output layers'
       biases are left alone (the authors' code would normalise a hidden
       bias jointly with the weights, but its hidden layers have none). The
       BN after a hidden Linear makes it scale-invariant, so this keeps the
       effective learning rate constant.
  CE   a C51-style categorical critic: 101 atoms on [-5, 5], cross-entropy
       against the projected target distribution. Rewards are divided by a
       running std of the discounted return so values fit the support.

Used by train_xqc.py (the paper's state recipe: SAC, configs/algo/xqc.yaml)
and train_drqv2.py with algo=drqv2_xqc (the paper's pixel recipe: DrQ-v2
with these MLPs and loss, configs/algo/drqv2_xqc.yaml).

BatchNorm modes follow the authors' code: the critic loss runs the online
and target critics in train mode (batch statistics) on the joint batch; the
actor loss evaluates the critic in eval mode (running statistics); acting,
evaluation and the next-state actions use the actor in eval mode. The
target critic's running statistics are never used (its BN momentum is 0).
"""

import contextlib
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Flax's BatchNorm(momentum=0.99, epsilon=1e-3): running = 0.99 * running +
# 0.01 * batch. PyTorch's momentum is the weight of the new batch.
BN_MOMENTUM = 0.01
BN_EPS = 1e-3


# ------------------------------------------------------------------------------
# Layers and weight normalization
# ------------------------------------------------------------------------------

class NormedLinear(nn.Linear):
    """Linear with orthogonal init and zero bias (Flax's Dense in XQC), whose
    weight rows are projected to unit L2 norm by project_weights() if
    `project`. The bias, if any, is never rescaled."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gain: float = math.sqrt(2), project: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.orthogonal_(self.weight, gain=gain)
        if bias:
            nn.init.zeros_(self.bias)
        self.project = project

    @torch.no_grad()
    def project_(self) -> None:
        self.weight.div_(self.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))


@torch.no_grad()
def project_weights(module: nn.Module) -> None:
    """XQC's weight normalization: every NormedLinear with project=True."""
    for m in module.modules():
        if isinstance(m, NormedLinear) and m.project:
            m.project_()


def _bn(dim: int) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(dim, momentum=BN_MOMENTUM, eps=BN_EPS)


def xqc_mlp(in_dim: int, hidden) -> nn.Sequential:
    """BN on the input, then per width: Linear (no bias) -> BN -> ReLU."""
    layers = [_bn(in_dim)]
    for width in hidden:
        layers += [NormedLinear(in_dim, width, bias=False), _bn(width), nn.ReLU()]
        in_dim = width
    return nn.Sequential(*layers)


def freeze_bn_statistics(module: nn.Module) -> None:
    """BN layers keep their running statistics fixed (momentum 0); for the
    target critic, which always normalises with batch statistics."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0


@contextlib.contextmanager
def bn_mode(train: bool, *modules: nn.Module):
    """Temporarily put modules in train/eval mode (BatchNorm: batch vs
    running statistics), restoring each one's mode afterwards."""
    before = [m.training for m in modules]
    for m in modules:
        m.train(train)
    try:
        yield
    finally:
        for m, was in zip(modules, before):
            m.train(was)


# ------------------------------------------------------------------------------
# Critic
# ------------------------------------------------------------------------------

class CriticEnsemble(nn.Module):
    """n_critics independent XQC MLPs on concat(features, action), each with
    a categorical (num_atoms logits) or scalar (critic_loss=mse) output.
    `trunk` (DrQ-v2 recipe): a module applied to the features first, shared
    by all members."""

    def __init__(self, feat_dim: int, action_dim: int, hidden, n_critics: int = 2,
                 critic_loss: str = "categorical", num_atoms: int = 101,
                 v_min: float = -5.0, v_max: float = 5.0,
                 normalize_last_layer: bool = True, trunk: nn.Module = None, trunk_dim: int = None):
        super().__init__()
        if critic_loss not in ("categorical", "mse"):
            raise ValueError(f"critic_loss={critic_loss!r}; expected categorical or mse")
        self.categorical = critic_loss == "categorical"
        self.trunk = trunk
        in_dim = (trunk_dim if trunk is not None else feat_dim) + action_dim
        n_out = num_atoms if self.categorical else 1
        self.members = nn.ModuleList([
            nn.Sequential(xqc_mlp(in_dim, hidden),
                          NormedLinear(hidden[-1], n_out, project=normalize_last_layer))
            for _ in range(n_critics)
        ])
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.register_buffer("atoms", torch.linspace(v_min, v_max, num_atoms))

    def forward(self, feats, action) -> torch.Tensor:
        """-> (n_critics, B, num_atoms) logits, or (n_critics, B, 1) values."""
        h = feats if self.trunk is None else self.trunk(feats)
        x = torch.cat([h, action], dim=-1)
        return torch.stack([m(x) for m in self.members])

    def q_values(self, out: torch.Tensor) -> torch.Tensor:
        """forward() output -> (n_critics, B) expected Q."""
        if self.categorical:
            return (out.softmax(-1) * self.atoms).sum(-1)
        return out.squeeze(-1)


def project_categorical(values: torch.Tensor, probs: torch.Tensor,
                        v_min: float, v_max: float) -> torch.Tensor:
    """C51 projection: the distribution putting probs[b, j] on values[b, j]
    (clipped to [v_min, v_max]), split linearly between the two nearest of
    the evenly spaced atoms. (B, n) -> (B, n), rows summing to 1."""
    n = probs.shape[-1]
    delta = (v_max - v_min) / (n - 1)
    b = (values.clamp(v_min, v_max) - v_min) / delta
    lower, upper = b.floor(), b.ceil()
    m_lower = probs * (upper + (lower == upper).to(probs.dtype) - b)
    m_upper = probs * (b - lower)
    out = torch.zeros_like(probs)
    out.scatter_add_(-1, lower.long().clamp(0, n - 1), m_lower)
    out.scatter_add_(-1, upper.long().clamp(0, n - 1), m_upper)
    return out


def critic_loss(critic: CriticEnsemble, critic_target: CriticEnsemble, feats, actions,
                next_feats, next_actions, rewards, discounts, next_bonus=None):
    """XQC's critic loss on one joint batch: (feats, actions) and
    (next_feats, next_actions) go through each critic together, in train
    mode, so BatchNorm normalises over both. The target is
        rewards + discounts * (Z'(s', a') + next_bonus)
    where Z' is the distribution of the target critic with the smallest
    expected value for that sample (categorical), or the smallest value
    (mse). next_bonus: SAC's -alpha * log pi(a'|s'), or None (DrQ-v2).
    rewards are the (n-step, normalised) sums; discounts gamma^k or 0.

    Returns (loss, stats): stats holds per_critic (C,) losses, q (C, B)
    expected values, target_mean, and (categorical) mass_vmin / mass_vmax,
    the target's mean probability on the two edge atoms -- values piling up
    there mean the support is too narrow for the (normalised) returns.
    All detached."""
    B = feats.shape[0]
    x = torch.cat([feats, next_feats.detach()], dim=0)
    a = torch.cat([actions, next_actions.detach()], dim=0)
    with bn_mode(True, critic):
        out = critic(x, a)[:, :B]
    bonus = 0.0 if next_bonus is None else next_bonus.view(-1, 1)
    with torch.no_grad(), bn_mode(True, critic_target):
        t_out = critic_target(x, a)[:, B:]
        t_q = critic_target.q_values(t_out)                       # (C, B)
        if critic.categorical:
            pick = t_q.argmin(dim=0)                              # per sample
            probs = t_out.softmax(-1)[pick, torch.arange(B, device=pick.device)]
            target_values = rewards.view(-1, 1) + discounts.view(-1, 1) * (critic.atoms + bonus)
            target = project_categorical(target_values, probs, critic.v_min, critic.v_max)
            target_mean = (target * critic.atoms).sum(-1).mean()
            edges = target[:, [0, -1]].mean(0)
        else:
            next_v = t_q.min(dim=0).values
            if next_bonus is not None:
                next_v = next_v + next_bonus.view(-1)
            target = rewards + discounts * next_v
            target_mean = target.mean()
            edges = None
    if critic.categorical:
        per_critic = -(target * out.log_softmax(-1)).sum(-1).mean(-1)
    else:
        per_critic = (0.5 * (out.squeeze(-1) - target) ** 2).mean(-1)
    stats = {"per_critic": per_critic.detach(), "q": critic.q_values(out.detach()),
             "target_mean": target_mean}
    if edges is not None:
        stats["mass_vmin"], stats["mass_vmax"] = edges[0], edges[1]
    return per_critic.sum(), stats


def actor_q(critic: CriticEnsemble, feats, actions) -> torch.Tensor:
    """min over the critics of Q(feats, actions), critic in eval mode
    (running statistics), as in XQC's actor loss. (B,)"""
    with bn_mode(False, critic):
        return critic.q_values(critic(feats, actions)).min(dim=0).values


def make_target(critic: nn.Module) -> nn.Module:
    """A frozen copy for Polyak averaging (parameters only), always in train
    mode with fixed BN statistics: it normalises with batch statistics."""
    target = copy.deepcopy(critic).requires_grad_(False)
    freeze_bn_statistics(target)
    return target.train()


@torch.no_grad()
def polyak_(net: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.lerp_(p, tau)


# ------------------------------------------------------------------------------
# Actors
# ------------------------------------------------------------------------------

class TanhGaussianActor(nn.Module):
    """XQC's SAC policy: XQC MLP, then mean and log-std heads (log-std
    clipped to [log_std_min, log_std_max]); actions are tanh of a Gaussian
    sample, in [-1, 1] (the script rescales them to the env's bounds)."""

    def __init__(self, feat_dim: int, action_dim: int, hidden, normalize_last_layer: bool = True,
                 log_std_min: float = -10.0, log_std_max: float = 2.0):
        super().__init__()
        self.body = xqc_mlp(feat_dim, hidden)
        self.mean = NormedLinear(hidden[-1], action_dim, project=normalize_last_layer)
        self.log_std = NormedLinear(hidden[-1], action_dim, gain=1.0, project=normalize_last_layer)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def forward(self, feats):
        h = self.body(feats)
        return self.mean(h), self.log_std(h).clamp(self.log_std_min, self.log_std_max)

    def sample(self, feats):
        """-> (action (B, A) in [-1, 1], log_prob (B,)), reparameterised."""
        mean, log_std = self(feats)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        log_prob = torch.distributions.Normal(mean, std).log_prob(u)
        # log |d tanh(u)/du| = log(1 - tanh(u)^2), in its stable form.
        log_prob = log_prob - 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        return torch.tanh(u), log_prob.sum(-1)

    def mean_action(self, feats):
        return torch.tanh(self(feats)[0])


class DrQv2XQCActor(nn.Module):
    """DrQ-v2's actor (src/rl/drqv2.py) with XQC's MLP: DrQ-v2's trunk
    (Linear + LayerNorm + Tanh, unchanged), then the XQC MLP and a tanh mean;
    exploration noise from the std schedule, as in DrQ-v2."""

    def __init__(self, feat_dim: int, action_dim: int, feature_dim: int, hidden,
                 normalize_last_layer: bool = True):
        super().__init__()
        from src.rl.drqv2 import _trunk, weight_init
        self.trunk = _trunk(feat_dim, feature_dim)
        self.trunk.apply(weight_init)
        self.body = xqc_mlp(feature_dim, hidden)
        self.mean = NormedLinear(hidden[-1], action_dim, project=normalize_last_layer)

    def forward(self, feats, std: float):
        from src.rl.drqv2 import TruncatedNormal
        mu = torch.tanh(self.mean(self.body(self.trunk(feats))))
        return TruncatedNormal(mu, torch.ones_like(mu) * std)


def drqv2_xqc_critic(feat_dim: int, action_dim: int, algo) -> CriticEnsemble:
    """DrQ-v2's critic with XQC's members: DrQ-v2's shared trunk, then
    algo.num_blocks XQC blocks of algo.hidden_dim per critic."""
    from src.rl.drqv2 import _trunk, weight_init
    trunk = _trunk(feat_dim, algo.feature_dim)
    trunk.apply(weight_init)
    return CriticEnsemble(
        feat_dim, action_dim, [algo.hidden_dim] * algo.num_blocks, n_critics=algo.get("n_critics", 2),
        critic_loss=algo.critic_loss, num_atoms=algo.num_atoms, v_min=algo.v_min, v_max=algo.v_max,
        normalize_last_layer=algo.normalize_last_layer, trunk=trunk, trunk_dim=algo.feature_dim,
    )


# ------------------------------------------------------------------------------
# Rewards, discount, learning rate
# ------------------------------------------------------------------------------

class RewardNormalizer:
    """Rewards divided by the running std of the discounted return
    G <- gamma * G + r (reset at episode ends), with no mean subtraction
    (XQC's RewardNormalizer; Engstrom et al. 2020)."""

    def __init__(self, gamma: float, eps: float = 1e-8):
        self.gamma, self.eps = float(gamma), eps
        self.G = 0.0
        # RunningMeanStd as in OpenAI baselines: a tiny prior count.
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, reward: float, done: bool) -> None:
        self.G = self.gamma * self.G + float(reward)
        delta = self.G - self.mean
        total = self.count + 1
        self.mean += delta / total
        self.var = (self.var * self.count + delta ** 2 * self.count / total) / total
        self.count = total
        if done:
            self.G = 0.0

    @property
    def scale(self) -> float:
        return math.sqrt(self.var) + self.eps

    def normalize(self, rewards):
        return rewards / self.scale

    def state_dict(self) -> dict:
        return {"G": self.G, "mean": self.mean, "var": self.var, "count": self.count}


def episode_steps(env) -> int:
    """The env's episode length in env steps: the TimeLimit's, or
    dm_control's own step limit (shimmy registers no TimeLimit)."""
    spec = getattr(env, "spec", None)
    if spec is not None and spec.max_episode_steps:
        return int(spec.max_episode_steps)
    inner = getattr(env.unwrapped, "_env", None)
    limit = getattr(inner, "_step_limit", None)
    if limit is not None and math.isfinite(limit):
        return int(limit)
    raise ValueError("algo.gamma=auto needs the episode length, which this env doesn't "
                     "report: set algo.gamma to a number")


def heuristic_discount(episode_len: int, action_repeat: int = 1) -> float:
    """XQC's discount: 1 - 5/T clipped to [0.95, 0.995], T = the episode's
    length in agent steps."""
    t = episode_len / action_repeat
    return float(np.clip(((t / 5) - 1) / (t / 5), 0.95, 0.995))


def resolve_gamma(gamma, env, action_repeat: int = 1) -> float:
    if isinstance(gamma, str):
        if gamma != "auto":
            raise ValueError(f"algo.gamma={gamma!r}; expected a number or 'auto'")
        return heuristic_discount(episode_steps(env), action_repeat)
    return float(gamma)


def linear_lr(lr: float, lr_end, step: int, total: int) -> float:
    """lr -> lr_end linearly over `total` optimizer steps (lr_end None:
    constant)."""
    if lr_end is None:
        return float(lr)
    frac = min(step / max(total, 1), 1.0)
    return float(lr + frac * (lr_end - lr))


def xqc_learning_rates(update: int, algo):
    """The learning rates of XQC update number `update` (0-based):
    (critic_lr, actor_lr), the actor's also the temperature's.

    As in the authors' code (optax.linear_schedule inside
    optax.conditionally_mask), every optimizer counts its own steps over
    transition_steps = total_timesteps * updates_per_step, which includes
    the warm-up that has no updates: the critic ends ~1% short of lr_end
    (500k steps, 5k warm-up), and the actor and temperature, which step on
    every policy_delay-th update only, get about a third of the way
    (3e-4 -> ~2.1e-4). Deliberate, to match their curves."""
    total = algo.total_timesteps * algo.updates_per_step
    critic = linear_lr(algo.critic_lr, algo.lr_end, update, total)
    actor_steps = -(-update // algo.policy_delay)        # actor steps before this update
    actor = linear_lr(algo.actor_lr, algo.lr_end, actor_steps, total)
    return critic, actor


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
