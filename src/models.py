import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions, batch_norm=False):
        super(PolicyNet, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        # Linear layers
        self.fc = nn.Sequential(
            init_(nn.Linear(observation_shape[0], 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        # Add batch norm
        if batch_norm:
            self.fc = nn.Sequential(
                nn.BatchNorm1d(observation_shape[0]),
                *list(self.fc)
            )

        # LSTM
        self.core = nn.LSTM(1024, 1024, 2)

        # Outputs
        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.policy = init_(nn.Linear(1024, num_actions))
        self.baseline = init_(nn.Linear(1024, 1))


    @property
    def device(self):
        return next(self.parameters()).device


    def initial_state(self, batch_size):
        return tuple(torch.zeros(self.core.num_layers, batch_size,
                                self.core.hidden_size) for _ in range(2))


    def forward(self, inputs, core_state=()):
        x = inputs['obs'] # Original shape -> (unroll_length, batch_size, obs_size)
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1).float() # Merge time and batch -> (unroll_length * batch_size, obs_size)
        x = x.to(device=self.device)

        core_input = self.fc(x)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (1 - inputs['done'].float()).abs().to(device=self.device)
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s.to(device=self.device) for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline,
                    action=action), core_state




# ==============================================================================
# Continuous policy for BC / SAC / IQL
# ==============================================================================

class ContinuousPolicyNet(nn.Module):
    """
    MLP policy with continuous (tanh-bounded) output.
    Used for BC on manipulation tasks and as the actor in SAC/IQL.

    Args:
        obs_size:    dimensionality of the (flat) input observation or embedding.
        action_dim:  dimensionality of the continuous action.
        hidden_size: width of each hidden layer.
        batch_norm:  prepend a BatchNorm1d layer (helps when inputs are not normalised).
    """

    def __init__(self, obs_size: int, action_dim: int,
                 hidden_size: int = 1024, batch_norm: bool = False):
        super().__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'),
        )

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(obs_size))
        layers += [
            init_(nn.Linear(obs_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())
