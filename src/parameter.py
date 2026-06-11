"""
Hyperparameter schedulers used by PrioritizedReplayMemory (alpha, beta)
and anywhere else a value needs to anneal over training steps.

Each scheduler exposes:
  .value  — current value (float property)
  .step() — advance one step

Usage in omegaconf / hydra configs:
  alpha:
    id: Linear
    start: 0.6
    end: 0.6
    steps: 1
  beta:
    id: Linear
    start: 0.4
    end: 1.0
    steps: 1000000
"""


class Constant:
    def __init__(self, value, **kwargs):
        self._value = float(value)

    @property
    def value(self):
        return self._value

    def step(self):
        pass


class Linear:
    """Linearly interpolates from `start` to `end` over `steps` steps."""

    def __init__(self, start, end, steps, **kwargs):
        self._start = float(start)
        self._end = float(end)
        self._steps = int(steps)
        self._t = 0

    @property
    def value(self):
        if self._steps == 0:
            return self._end
        frac = min(self._t / self._steps, 1.0)
        return self._start + frac * (self._end - self._start)

    def step(self):
        self._t += 1


class Exponential:
    """Exponentially decays from `start` toward `end` with rate `decay`."""

    def __init__(self, start, end, decay, **kwargs):
        self._start = float(start)
        self._end = float(end)
        self._decay = float(decay)
        self._t = 0

    @property
    def value(self):
        import math
        return self._end + (self._start - self._end) * math.exp(-self._decay * self._t)

    def step(self):
        self._t += 1
