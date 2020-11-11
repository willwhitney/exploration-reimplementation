import collections

import numpy as np
import matplotlib.pyplot as plt

import utils


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
        return self.get_transitions(indices)

    def get_transitions(self, indices):
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
        self.min_s = kwargs.pop('min_s')
        self.max_s = kwargs.pop('max_s')
        self.n_bins = kwargs.pop('n_bins')
        super().__init__(*args, **kwargs)

        # self.forward_trace = collections.defaultdict(list)
        self.trace = collections.defaultdict(list)

    def discretize(self, s):
        normalized_s = (s - self.min_s) / self.max_s
        normalized_s = normalized_s.clip(0, 1)
        discrete_s = (normalized_s * self.n_bins).astype(np.uint8)
        return tuple(discrete_s.flatten())

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


# ----- Visualizations for gridworld ---------------------------------
def render_trajectory(replay, n, ospec, bins, vis_dims=(0, 1)):
    x_dim, y_dim = vis_dims
    end = replay.next_slot
    start = max(0, end - n)
    flat_spec = utils.flatten_observation_spec(ospec)

    transitions = replay[start:end]
    states = transitions[0]
    render = np.zeros((bins, bins))
    discrete_states = utils.discretize_observation(states, flat_spec, bins,
                                                   preserve_batch=True)
    for state in discrete_states:
        # loc = state.argmax(axis=1)
        # discrete_state = utils.discretize_observation(state, flat_spec, bins)
        render[state[x_dim], state[y_dim]] += 1
    return render


def display_trajectory(*args):
    trajectory = render_trajectory(*args)
    fig, ax = plt.subplots()
    img = ax.imshow(trajectory)
    fig.colorbar(img, ax=ax)
    ax.set_title("Last trajectory")
    fig.show()
    plt.close(fig)
# -------------------------------------------------------------------


if __name__ == "__main__":
    replay = Replay((10,), (1,))
    replay.append(np.ones((10,)), np.zeros((1,)), np.zeros((10,)), 1)
    replay.append(np.ones((10,)) / 2, np.zeros((1,)) + 0.1,
                  np.zeros((10,)) + 0.3, 0)

    transitions = replay.sample(3)
    print(transitions)
