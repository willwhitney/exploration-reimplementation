import jax
from jax import numpy as jnp

from flax import struct

import utils


@struct.dataclass
class DensityState:
    kernel_cov: jnp.ndarray
    observations: jnp.ndarray
    total: int = 1


def new(observation_spec, action_spec, max_obs=100000):
    kernel_std_scale = 1 / 4.
    flat_ospec = utils.flatten_observation_spec(observation_spec)
    concat_min = jnp.concatenate([flat_ospec.minimum, action_spec.minimum],
                                 axis=0)
    concat_max = jnp.concatenate([flat_ospec.maximum, action_spec.maximum],
                                 axis=0)
    # TODO: switch to covariance
    kernel_std = kernel_std_scale * (concat_max - concat_min)
    observations = jnp.ndarray((max_obs, *concat_min.shape))
    return DensityState(kernel_std, observation)


def log_p(density_state: DensityState, state, action):
    key = _make_key(state, action)



def _make_key(s, a):
    flat_s = utils.flatten_observation(s)
    return jnp.concatenate([flat_s, a], axis=0)
make_key_batch = jax.vmap(make_key)
