from flax import struct
from jax import numpy as jnp


@struct.dataclass
class DensityState:
    total: int = 0


def new(*args, **kwargs):
    return DensityState()


def update_batch(density_state, states, actions):
    return density_state


def get_count(density_state, state, action):
    return jnp.ones(())


def get_count_batch(density_state, states, actions):
    return jnp.ones((states.shape[0]))
