import typing
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import numpy as jnp, jit, lax
from flax import struct

import gridworld

@struct.dataclass
class DensityState():
    observations: jnp.ndarray
    eps: float = 1e-8
    total: int = 1


def new(state_values_per_dim, action_values_per_dim):
    """Creates a new DensityState.

    Arguments:
    - state_values_per_dim: a list containing the number of discrete values that
        each dimension of the state vector can take on. For example, in a 2D
        gridworld of size 10x5, state_values_per_dim = [10, 5].
    - action_values_per_dim: defined the same way as state_values_per_dim, but
        for the vector of discrete actions.
    """
    state_values_per_dim = list(state_values_per_dim)
    action_values_per_dim = list(action_values_per_dim)
    total_dimension_list = state_values_per_dim + action_values_per_dim
    storage = jnp.zeros(tuple(total_dimension_list), dtype=jnp.uint32)
    return DensityState(observations=storage)


@jax.jit
def update_batch(density_state: DensityState, states, actions):
    keys = _make_key_gridworld_batch(states, actions)
    new_observations = jax.ops.index_add(density_state.observations, keys, 1)
    new_total = density_state.total + states.shape[0]
    density_state = density_state.replace(observations=new_observations,
                                          total=new_total)
    return density_state


def get_count(density_state: DensityState, state, action):
    key = _make_key_gridworld(state, action)
    return density_state.observations[key]
# get_count_batch = jax.vmap(get_count, in_axes=(None, 0, 0))  # noqa: E305


@jax.jit
def get_count_batch(density_state: DensityState, states, actions):
    keys = _make_key_gridworld_batch(states, actions)
    return density_state.observations[keys]


def log_p(density_state: DensityState, state, action):
    count = get_count(density_state, state, action)
    return jnp.log(count / density_state.total + density_state.eps)
log_p_batch = jax.vmap(log_p, in_axes=(None, 0, 0))  # noqa: E305


def _process_gridworld_state(s):
    return jnp.argmax(s, axis=-1).astype(jnp.int16)


def _process_gridworld_action(a):
    return a.astype(jnp.int16)


def _make_key_gridworld(s, a):
    s = _process_gridworld_state(s)
    a = _process_gridworld_action(a)
    return tuple(jnp.concatenate([s.flatten(), a.flatten()]))
_make_key_gridworld_batch = jax.vmap(_make_key_gridworld)  # noqa: E305


def _flatten(s, a):
    return jnp.concatenate([s.flatten(), a.flatten()])


# -------- Visualization tools for gridworld -------------------------------
def render_density_map(density_state: DensityState,
                       env: gridworld.GridWorld):
    count_map = gridworld.render_function(
        jax.partial(get_count_batch, density_state),
        env, reduction=jnp.min)
    return count_map


def display_density_map(density_state: DensityState,
                        env: gridworld.GridWorld):
    density_map = render_density_map(density_state, env)
    fig, ax = plt.subplots()
    img = ax.imshow(density_map)
    fig.colorbar(img, ax=ax)
    ax.set_title("State visitation counts (min)")
    fig.show()
    plt.close(fig)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    import random
    density_state = new([10, 10], [4])
    gw = gridworld.new(10)
    for i in range(10):
        a = random.randint(0, 3)
        a_vec = jnp.array(a, dtype=jnp.int16)
        gw, obs, r = gridworld.step(gw, a)
        print(obs)
        density_state = update_batch(density_state,
                                     jnp.expand_dims(obs, axis=0),
                                     jnp.expand_dims(a, axis=0))
    display_density_map(density_state, gw)
