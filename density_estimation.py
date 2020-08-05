import collections
import numpy as np

import jax
from jax import numpy as jnp, random, jit, lax

import flax
from flax import nn, optim


# class DensityEstimator:
#     def log_p(self, states, actions):
#         raise NotImplementedError

#     def update(self, states, actions):
#         raise NotImplementedError


class TabularDensity():
    def __init__(self):
        self.observations = collections.defaultdict(lambda: 0)
        self.eps = 1e-8
        self.total = 0

    def _flatten(self, s, a):
        return np.concatenate([s.flatten(), a.flatten()])

    def update(self, states, actions):
        for s, a in zip(states, actions):
            self.observations[self._flatten(s, a)] += 1
            self.total += 1

    def log_p(self, states, actions):
        counts = np.zeros(states.shape[0])
        for i, (s, a) in enumerate(zip(states, actions)):
            obs = self._flatten(s, a)
            counts[i] = self.observations[obs]
        return np.log(counts / self.total + self.eps)

    def count(self, states, actions):
        return self.log_p(states, actions).exp() * self.total
