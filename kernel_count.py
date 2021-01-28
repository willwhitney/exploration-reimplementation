import numpy as np

import jax
from jax import numpy as jnp, random, lax

from flax import struct

import utils


@struct.dataclass
class DensityState:
    kernel_cov: jnp.ndarray
    observations: jnp.ndarray
    total: int = 0
    next_slot: int = 0
    max_obs: int = 100000
    scale_factor: float = 1.0


def new(observation_spec, action_spec, max_obs=100000,
        state_std_scale=1, action_std_scale=1, **kwargs):
    flat_ospec = utils.flatten_observation_spec(observation_spec)
    state_std = flat_ospec.maximum - flat_ospec.minimum
    action_std = np.array(action_spec.maximum - action_spec.minimum)
    action_std = action_std.reshape((-1,))

    state_std = state_std_scale * state_std
    action_std = action_std_scale * action_std
    concat_std = jnp.concatenate([state_std, action_std], axis=0)
    kernel_cov = jnp.diag(concat_std ** 2)

    # calculate a scalar to shift the max to 1
    max_pdf = _max_pdf_value(kernel_cov)
    scale_factor = 1 / max_pdf

    # initialize this to some reasonable size
    observations = jnp.zeros((1024, *concat_std.shape))
    return DensityState(kernel_cov, observations, max_obs=max_obs,
                        scale_factor=scale_factor)


def _max_pdf_value(cov):
    k = cov.shape[0]
    return (2 * jnp.pi) ** (-k / 2) * jnp.linalg.det(cov) ** (- 1/2)


def get_count(density_state: DensityState, state, action):
    """Put one unit of count on the space at every observation.
    Fall-off from that is specified by the covariance."""
    key = _make_key(state, action)
    observations = density_state.observations
    obs_size = observations.shape[0]
    mask = jnp.linspace(0, obs_size - 1, obs_size) < density_state.total
    mask = mask.astype(jnp.int32)

    probs_per_obs = _normal_pdf_batchedmean(
        observations, density_state.kernel_cov, key)
    masked_obs = mask * probs_per_obs
    count = masked_obs.sum()
    return count * density_state.scale_factor
get_count_batch = jax.vmap(get_count, in_axes=(None, 0, 0))


def update_batch(density_state: DensityState, states, actions):
    obs_size = density_state.observations.shape[0]

    # double the size of observations if needed
    # print(density_state.next_slot, states.shape[0], obs_size)
    while density_state.next_slot + states.shape[0] >= obs_size:
        density_state = _grow_observations(density_state)
        obs_size = density_state.observations.shape[0]

    return _update_batch(density_state, states, actions)


# @jax.jit
def _update_batch(density_state: DensityState, states, actions):
    bsize = states.shape[0]
    next_slot = density_state.next_slot
    observations = density_state.observations
    obs_size = observations.shape[0]

    keys = _make_key_batch(states, actions)
    indices = jnp.arange(next_slot, next_slot + bsize)

    observations = jax.ops.index_update(observations, indices, keys)
    total = jnp.minimum(density_state.total + bsize, obs_size)
    next_slot = (next_slot + bsize) % obs_size
    return density_state.replace(observations=observations,
                                 total=total, next_slot=next_slot)


def _grow_observations(density_state: DensityState):
    """An un-jittable function which grows the size of the observations array"""
    observations = density_state.observations
    obs_size = observations.shape[0]
    new_obs_size = jnp.minimum(density_state.max_obs, 2 * obs_size)
    new_shape = (new_obs_size, *observations.shape[1:])
    new_observations = jnp.zeros(new_shape)
    print(f"Growing KDE observations from {obs_size} to {new_obs_size}.")
    observations = jax.ops.index_update(new_observations,
                                        jax.ops.index[:obs_size],
                                        observations)
    return density_state.replace(observations=observations)


def _make_key(s, a):
    flat_s = utils.flatten_observation(s)
    flat_a = jnp.array(a).reshape((-1,))
    return jnp.concatenate([flat_s, flat_a], axis=0)
_make_key_batch = jax.vmap(_make_key)


def _normal_pdf(mean, cov, x):
    return jax.scipy.stats.multivariate_normal.pdf(x, mean, cov)
# for evaluating one x at many means:
_normal_pdf_batchedmean = jax.vmap(_normal_pdf, in_axes=(0, None, None))
# for evaluating many xs at a single mean:
_normal_pdf_batchedx = jax.vmap(_normal_pdf, in_axes=(None, None, 0))
_normal_pdf_batchedxmean = jax.vmap(_normal_pdf, in_axes=(0, None, 0))


if __name__ == "__main__":
    from dm_control import suite
    from observation_domains import DOMAINS
    import jax_specs
    import point

    # env = suite.load('point_mass', 'easy')
    # ospec = DOMAINS['point_mass']['easy']
    env = suite.load('point', 'velocity')
    ospec = DOMAINS['point']['velocity']

    aspec = env.action_spec()
    j_aspec = jax_specs.convert_dm_spec(aspec)
    density_state = new(ospec, aspec, state_std_scale=1e-2, action_std_scale=1)

    timestep = env.reset()
    state = utils.flatten_observation(timestep.observation)
    actions = utils.sample_uniform_actions(j_aspec, jax.random.PRNGKey(0), 1)
    action = actions[0]

    states = jnp.expand_dims(state, axis=0)
    density_state = update_batch(density_state, states, actions)

    timestep2 = env.reset()
    state2 = utils.flatten_observation(timestep2.observation)

    # print("S1 pdf:", pdf(density_state, state, action))
    print("S1 count:", get_count(density_state, state, action))
    density_state_updated = update_batch(density_state,
                                         jnp.expand_dims(state, axis=0),
                                         jnp.expand_dims(action, axis=0))
    # print("S1 pdf after self update:", pdf(density_state_updated, state, action))

    # print("S2 pdf:", pdf(density_state, state2, action))
    print("S2 count:", get_count(density_state, state2, action))
    density_state_updated = update_batch(density_state,
                                         jnp.expand_dims(state2, axis=0),
                                         jnp.expand_dims(action, axis=0))
    # print("S2 pdf after self update:", pdf(density_state_updated, state2, action))
    print("S2 count after self update:", get_count(density_state_updated,
                                                   state2, action))

    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np

    n_points = 200
    x = np.linspace(-0.3, 0.3, n_points)
    y = np.linspace(-0.3, 0.3, n_points)

    X, Y = np.meshgrid(x, y)
    states = np.stack([X, Y]).transpose().reshape((-1, 2))
    actions = jnp.repeat(jnp.expand_dims(action, axis=0), n_points**2, axis=0)
    Z = get_count_batch(density_state, states, actions)
    Z = np.array(Z).reshape((n_points, n_points))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.savefig('kernel_3d.png')






