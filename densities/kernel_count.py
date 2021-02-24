import numpy as np

import jax
from jax import numpy as jnp, random, lax

from flax import struct

import utils


DTYPE = jnp.float16


@struct.dataclass
class DensityState:
    kernel_cov: jnp.ndarray
    observations: jnp.ndarray
    weights: jnp.ndarray
    total: int = 0
    next_slot: int = 0
    max_obs: int = 100000
    scale_factor: float = 1.0
    tolerance: float = 0.95


def new(observation_spec, action_spec, max_obs=100000,
        state_std_scale=1, action_std_scale=1, tolerance=0.95, **kwargs):
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
    # starting_size = 65536
    starting_size = 1024
    observations = jnp.zeros((starting_size, *concat_std.shape), dtype=DTYPE)
    weights = jnp.zeros((starting_size,))
    return DensityState(kernel_cov, observations, weights,
                        max_obs=int(max_obs), scale_factor=scale_factor,
                        tolerance=tolerance)


def _max_pdf_value(cov):
    k = cov.shape[0]
    return (2 * jnp.pi) ** (-k / 2) * jnp.linalg.det(cov) ** (- 1/2)


@jax.profiler.trace_function
@jax.jit
def _similarity_per_obs(density_state: DensityState, state, action):
    """Put one unit of count on the space at every observation.
    Fall-off from that is specified by the covariance."""
    key = _make_key(state, action)
    observations = density_state.observations
    obs_size = observations.shape[0]

    mask = jnp.linspace(0, obs_size - 1, obs_size) < density_state.total
    mask = mask.astype(jnp.int32)

    diag_probs_per_obs = _normal_diag_pdf_batchedmean(
        observations, density_state.kernel_cov, key)
    # probs_per_obs = _normal_pdf_batchedmean(
    #     observations, density_state.kernel_cov, key)
    weighted_obs = mask * diag_probs_per_obs * density_state.scale_factor
    return weighted_obs
_similarity_per_obs_batch = jax.vmap(_similarity_per_obs, in_axes=(None, 0, 0))


@jax.profiler.trace_function
@jax.jit
def get_count(density_state: DensityState, state, action):
    """Put one unit of count on the space at every observation.
    Fall-off from that is specified by the covariance."""
    sim_per_obs = _similarity_per_obs(density_state, state, action)
    count_per_obs = density_state.weights * sim_per_obs
    count = count_per_obs.sum()
    return count
get_count_batch = jax.vmap(get_count, in_axes=(None, 0, 0))


@jax.profiler.trace_function
def update_batch(density_state: DensityState, states, actions):
    obs_size = density_state.observations.shape[0]

    # increase the size of observations if needed
    while ((density_state.next_slot + states.shape[0] >= obs_size) and
           (obs_size < density_state.max_obs)):
        density_state = _grow_observations(density_state)
        obs_size = density_state.observations.shape[0]

    with jax.profiler.TraceContext("kernel update similarity"):
        # compute which states/actions to add as observations vs as weights
        sims_per_obs = _similarity_per_obs_batch(density_state, states, actions)
        sims_per_obs = np.array(sims_per_obs)

    with jax.profiler.TraceContext("kernel update for"):
        new_states, new_actions = [], []
        new_weights = np.zeros((obs_size,))

        for (state, action, sims) in zip(states, actions, sims_per_obs):
            similar_obs = sims > density_state.tolerance
            n_similar_obs = similar_obs.sum()
            if n_similar_obs >= 1:
                new_weights += np.array(similar_obs) / n_similar_obs
            else:
                new_states.append(state)
                new_actions.append(action)

    if len(new_states) > 0:
        with jax.profiler.TraceContext("kernel update new states"):
            new_states = np.stack(new_states)
            new_actions = np.stack(new_actions)
            # next_slot = density_state.next_slot

            # if density_state.total < density_state.max_obs:
            #     indices = jnp.arange(next_slot, next_slot + len(new_states))
            #     indices = indices % density_state.max_obs
            # else:
            #     # use random indices to avoid overwriting whole blocks
            #     # TODO: pass an RNG key in here
            #     rng = random.PRNGKey(density_state.next_slot)
            #     indices = random.randint(rng, shape=(len(new_states),),
            #                             minval=0, maxval=density_state.max_obs)
            density_state = _add_observations(density_state,
                                              new_states, new_actions)

            # if density_state.next_slot < next_slot:
            #     print(f"Density wrapped next_slot from {next_slot} to "
            #         f"{density_state.next_slot}.")

    if new_weights.sum() > 0:
        with jax.profiler.TraceContext("kernel update add weights"):
            density_state = _add_weights(density_state, new_weights)
    return density_state


@jax.profiler.trace_function
@jax.jit
def _add_weights(density_state: DensityState, new_weights):
    new_weights = density_state.weights + new_weights
    return density_state.replace(weights=new_weights)


@jax.profiler.trace_function
@jax.jit
def _add_observations(density_state: DensityState, states, actions):
    bsize = states.shape[0]
    next_slot = density_state.next_slot
    observations = density_state.observations
    weights = density_state.weights
    obs_size = observations.shape[0]

    use_linear_indices = density_state.total < density_state.max_obs
    use_linear_indices = use_linear_indices.astype(int)

    linear_indices = jnp.linspace(next_slot, next_slot + bsize - 1, bsize)
    linear_indices = linear_indices.round().astype(int)
    linear_indices = linear_indices % density_state.max_obs

    # use random indices to avoid overwriting whole blocks
    # TODO: pass an RNG key in here
    rng = random.PRNGKey(density_state.next_slot)
    random_indices = random.randint(rng, shape=(bsize,),
                                    minval=0, maxval=density_state.max_obs)

    indices = (use_linear_indices * linear_indices
               + (1 - use_linear_indices) * random_indices)

    keys = _make_key_batch(states, actions)
    observations = jax.ops.index_update(observations, indices, keys)
    weights = jax.ops.index_update(weights, indices, 1)
    total = jnp.minimum(density_state.total + bsize, obs_size)
    next_slot = (next_slot + bsize) % obs_size
    return density_state.replace(observations=observations, weights=weights,
                                 total=total, next_slot=next_slot)


@jax.profiler.trace_function
def _grow_observations(density_state: DensityState):
    """An un-jittable function which grows the size of the observations array"""
    observations = density_state.observations
    weights = density_state.weights
    obs_size = observations.shape[0]
    new_obs_size = jnp.minimum(density_state.max_obs, 2 * obs_size)
    print(f"Growing KDE observations from {obs_size} to {new_obs_size}.")

    # growing observations
    new_shape = (new_obs_size, *observations.shape[1:])
    new_observations = jnp.zeros(new_shape, dtype=DTYPE)
    observations = jax.ops.index_update(new_observations,
                                        jax.ops.index[:obs_size],
                                        observations)

    # growing weights
    new_weight_shape = (new_obs_size,)
    new_weights = jnp.zeros(new_weight_shape, dtype=DTYPE)
    weights = jax.ops.index_update(new_weights,
                                   jax.ops.index[:obs_size],
                                   weights)
    return density_state.replace(observations=observations, weights=weights)


@jax.profiler.trace_function
def _make_key(s, a):
    flat_s = utils.flatten_observation(s)
    flat_a = jnp.array(a).reshape((-1,))
    return jnp.concatenate([flat_s, flat_a], axis=0).astype(DTYPE)
_make_key_batch = jax.vmap(_make_key)


@jax.jit
def _normal_pdf(mean, cov, x):
    return jax.scipy.stats.multivariate_normal.pdf(x, mean, cov)
# for evaluating one x at many means:
_normal_pdf_batchedmean = jax.vmap(_normal_pdf, in_axes=(0, None, None))
# for evaluating many xs at a single mean:
_normal_pdf_batchedx = jax.vmap(_normal_pdf, in_axes=(None, None, 0))
_normal_pdf_batchedxmean = jax.vmap(_normal_pdf, in_axes=(0, None, 0))


_batched_norm_pdf = jax.vmap(jax.scipy.stats.norm.pdf)


@jax.profiler.trace_function
@jax.jit
def _normal_diag_pdf(mean, cov, x):
    diag_indices = jnp.diag_indices(cov.shape[0])
    variances = cov[diag_indices]
    axis_probs = _batched_norm_pdf(x, mean, variances**0.5)
    return axis_probs.prod()
# for evaluating one x at many means:
_normal_diag_pdf_batchedmean = jax.vmap(_normal_diag_pdf, in_axes=(0, None, None))
# for evaluating many xs at a single mean:
_normal_diag_pdf_batchedx = jax.vmap(_normal_diag_pdf, in_axes=(None, None, 0))
_normal_diag_pdf_batchedxmean = jax.vmap(_normal_diag_pdf, in_axes=(0, None, 0))



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
    j_ospec = jax_specs.convert_dm_spec(ospec)
    density_state = new(ospec, aspec, state_std_scale=1e-1, action_std_scale=1)
    # import ipdb; ipdb.set_trace()

    timestep = env.reset()
    state = utils.flatten_observation(timestep.observation)
    actions = utils.sample_uniform_actions(j_aspec, jax.random.PRNGKey(0), 1)
    action = actions[0]

    # states = jnp.expand_dims(state, axis=0)
    # density_state = update_batch(density_state, states, actions)

    # visited_states = []
    # visited_actions = []
    # for i in range(128):
    #     timestep = env.step(action)
    #     state = utils.flatten_observation(timestep.observation)
    #     action = utils.sample_uniform_actions(j_aspec, jax.random.PRNGKey(i), 1)[0]
    #     visited_states.append(state)
    #     visited_actions.append(action)

    # density_state = update_batch(density_state,
    #                              jnp.stack(visited_states),
    #                              jnp.stack(visited_actions))
    # env.reset()



    # ---------- sanity checking counts --------------------
    timestep2 = env.step(jnp.ones(aspec.shape))
    state2 = utils.flatten_observation(timestep2.observation)

    # print("S1 pdf:", pdf(density_state, state, action))
    print("S1 count:", get_count(density_state, state, action))

    # print("S2 pdf:", pdf(density_state, state2, action))
    print("S2 count:", get_count(density_state, state2, action))
    density_state_updated = update_batch(density_state,
                                         jnp.expand_dims(state2, axis=0),
                                         jnp.expand_dims(action, axis=0))
    # print("S2 pdf after self update:", pdf(density_state_updated, state2, action))
    print("S2 count after self update:", get_count(density_state_updated,
                                                   state2, action))

    print("Batch of counts:", get_count_batch(density_state_updated,
                                              jnp.stack([state, state2]),
                                              jnp.stack([action, action])))

    # --------- plotting the kernel -----------------
    # from mpl_toolkits import mplot3d
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import numpy as np

    # n_points = 200
    # x = np.linspace(-0.3, 0.3, n_points)
    # y = np.linspace(-0.3, 0.3, n_points)

    # X, Y = np.meshgrid(x, y)
    # states = np.stack([X, Y]).transpose().reshape((-1, 2))
    # actions = jnp.repeat(jnp.expand_dims(action, axis=0), n_points**2, axis=0)
    # Z = get_count_batch(density_state, states, actions)
    # Z = np.array(Z).reshape((n_points, n_points))
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # fig.savefig('kernel_3d.png')







