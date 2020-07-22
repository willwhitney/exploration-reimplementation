import numpy as np
from typing import Any
import matplotlib.pyplot as plt

import jax
from jax import numpy as jnp, random, lax, profiler

from flax import struct


@struct.dataclass
class ReplayState():
    states: jnp.ndarray
    actions: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    max_size: int
    next_slot: int
    length: int

    def __len__(self):
        return self.length


def new_replay(state_shape, action_shape, max_size=int(1e5)):
    return ReplayState(
        states=jnp.zeros((max_size, *state_shape)),
        actions=jnp.zeros((max_size, *action_shape)),
        next_states=jnp.zeros((max_size, *state_shape)),
        rewards=jnp.zeros((max_size, 1)),
        max_size=max_size,
        next_slot=0,
        length=0)


def append(replay: ReplayState, s, a, sp, r):
    index = replay.next_slot
    if index < replay.length:
        replay = _invalidate(replay, index)




def _invalidate(replay: ReplayState, index):
    return replay

class Replay:
    def __init__(self, state_shape, action_shape, max_size=int(1e5)):
        # state s, next state sp (s prime), action a, reward r, done d
        self.s = np.zeros((max_size, *state_shape))
        self.a = np.zeros((max_size, *action_shape))
        self.sp = np.zeros((max_size, *state_shape))
        self.r = np.zeros((max_size, 1))
        self.max_size = max_size
        self.next_slot = 0
        self.length = 0

    def append(self, s, a, sp, r):
        index = self.next_slot
        if index < self.length:
            self._invalidate(index)
        self.s[index] = s
        self.a[index] = a
        self.sp[index] = sp
        self.r[index] = r
        self.next_slot = (self.next_slot + 1) % self.max_size
        self.length = max(self.length, self.next_slot)
        return index

    def sample(self, n):
        indices = np.random.randint(0, self.length, size=(n,))
        return (self.s[indices], self.a[indices],
                self.sp[indices], self.r[indices])

    def _invalidate(self, index):
        pass

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return (self.s[i], self.a[i], self.sp[i], self.r[i])


# def sample(replay, rng):
#     index = random.randint(rng, (1,), minval=0, maxval=len(replay))
#     return replay[index]
# sample_batch = jax.vmap(sample, in_axes=(None, 0))  # noqa: E305


class TracingReplay(Replay):
    def predecessors(self, state):
        raise NotImplementedError


class LowPrecisionTracingReplay(TracingReplay):
    def __init__(self, *args, **kwargs):
        self.mins = kwargs.pop('mins')
        self.maxs = kwargs.pop('maxs')
        self.n_bins = kwargs.pop('n_bins')
        super().__init__(*args, **kwargs)

        # self.forward_trace = collections.defaultdict(list)
        self.trace = collections.defaultdict(list)

    def discretize(self, s):
        normalized_s = (s - self.mins) / self.maxs
        normalized_s = normalized_s.clip(0, 1)
        return (normalized_s * self.n_bins).astype(np.uint8)

    def append(self, s, a, sp, r):
        index = super().append(s, a, sp, r)
        discrete_sp = self.discretize(sp)
        self.trace[discrete_sp].append(index)
        # self.forward_trace[index] = discrete_sp
        return index

    def _invalidate(self, index):
        transition = self[index]
        discrete_sp = self.discretize(transition[2])
        self.trace[discrete_sp].remove(index)
        return super()._invalidate(index)

    def predecessors(self, state):
        return self.trace[self.discretize(state)]


if __name__ == "__main__":
    replay = Replay((10,), (1,))
    replay.append(np.ones((10,)), np.zeros((1,)), np.zeros((10,)), 1)
    replay.append(np.ones((10,)) / 2, np.zeros((1,)) + 0.1,
                  np.zeros((10,)) + 0.3, 0)

    transitions = replay.sample(3)
    print(transitions)
